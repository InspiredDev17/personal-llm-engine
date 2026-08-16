//! Trusted, immutable execution-placement planning and bounded layer staging.
//!
//! This crate consumes only an already verified directory package. It does not
//! parse client paths, choose a cognitive model, authorize eviction, or accept
//! package-defined execution logic.

use personalai_engine_loader::{
    DirectorySecurityEventSink, LoadedDirectoryTensors, VerifiedDirectoryPackage,
};
use personalai_engine_memory::MemoryPool;
use personalai_engine_model_format::{
    DIRECTORY_MANIFEST_VERSION, DirectoryTensorRecord, Quantization,
};
use personalai_engine_types::{
    ArchitectureId, BackendId, DType, EngineError, MAX_RESOURCE_MEMORY_BYTES, ModelId,
    ModelPackageHash, ModelPackageRejection, SecurityEvent,
};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;

/// Largest supported trusted decoder-layer population in one placement plan.
pub const MAX_PLACEMENT_LAYERS: u32 = 4_096;
/// Hard maximum asynchronous prefetch depth.
pub const MAX_PREFETCH_DEPTH: u8 = 1;

/// Closed weight-residency modes understood by the trusted runtime.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WeightResidencyMode {
    /// Complete F32 execution weights remain resident.
    FullyResident,
    /// Trusted engine-selected host/device partitioning.
    HybridResident,
    /// One architecture stage is resident while the next may be prefetched.
    LayerStreamed,
}

impl WeightResidencyMode {
    /// Returns the stable content-free diagnostic name.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::FullyResident => "fully-resident",
            Self::HybridResident => "hybrid-resident",
            Self::LayerStreamed => "layer-streamed",
        }
    }
}

/// Physical memory topology relevant to trusted placement decisions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemoryTopology {
    /// CPU host memory only.
    HostOnly,
    /// Logically distinct CPU/GPU ownership over one unified memory system.
    Unified,
    /// Separate host and device memory with an explicit transfer boundary.
    DiscreteDevice,
}

/// Trusted backend limits used only to validate an internal plan.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BackendPlacementCapabilities {
    backend_id: BackendId,
    topology: MemoryTopology,
    maximum_fast_bytes: u64,
    maximum_staging_bytes: u64,
    layer_streaming: bool,
}

impl BackendPlacementCapabilities {
    /// Constructs bounded, owner-selected backend placement capabilities.
    pub fn new(
        backend_id: BackendId,
        topology: MemoryTopology,
        maximum_fast_bytes: u64,
        maximum_staging_bytes: u64,
        layer_streaming: bool,
    ) -> Result<Self, PlacementError> {
        if maximum_fast_bytes == 0 || maximum_staging_bytes == 0 {
            return Err(PlacementError::InvalidPlan(EngineError::invalid_input(
                "backend placement limits must be positive",
            )));
        }
        if maximum_fast_bytes > MAX_RESOURCE_MEMORY_BYTES
            || maximum_staging_bytes > MAX_RESOURCE_MEMORY_BYTES
        {
            return Err(PlacementError::InvalidPlan(EngineError::invalid_input(
                "backend placement limits exceed the engine memory ceiling",
            )));
        }
        Ok(Self {
            backend_id,
            topology,
            maximum_fast_bytes,
            maximum_staging_bytes,
            layer_streaming,
        })
    }

    /// Returns conservative CPU-reference capabilities.
    pub fn cpu_reference(maximum_bytes: u64) -> Result<Self, PlacementError> {
        Self::new(
            BackendId::new("cpu-reference").map_err(PlacementError::InvalidPlan)?,
            MemoryTopology::HostOnly,
            maximum_bytes,
            maximum_bytes,
            true,
        )
    }

    /// Returns the exact trusted backend identity.
    pub const fn backend_id(&self) -> &BackendId {
        &self.backend_id
    }

    /// Returns the declared physical memory topology.
    pub const fn topology(&self) -> MemoryTopology {
        self.topology
    }
}

/// Engine-owned hard memory envelope for one placement decision.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PlacementResourceEnvelope {
    total_bytes: u64,
    mapped_bytes: u64,
    staging_bytes: u64,
    safety_reserve_bytes: u64,
}

impl PlacementResourceEnvelope {
    /// Constructs a checked resource envelope.
    pub fn new(
        total_bytes: u64,
        mapped_bytes: u64,
        staging_bytes: u64,
        safety_reserve_bytes: u64,
    ) -> Result<Self, PlacementError> {
        if total_bytes == 0 || mapped_bytes == 0 || staging_bytes == 0 {
            return Err(PlacementError::InvalidPlan(EngineError::invalid_input(
                "placement resource limits must be positive",
            )));
        }
        if total_bytes > MAX_RESOURCE_MEMORY_BYTES
            || mapped_bytes > MAX_RESOURCE_MEMORY_BYTES
            || staging_bytes > MAX_RESOURCE_MEMORY_BYTES
            || safety_reserve_bytes > MAX_RESOURCE_MEMORY_BYTES
        {
            return Err(PlacementError::InvalidPlan(EngineError::invalid_input(
                "placement resource limit exceeds the engine memory ceiling",
            )));
        }
        if safety_reserve_bytes > total_bytes {
            return Err(PlacementError::InvalidPlan(EngineError::invalid_input(
                "placement safety reserve exceeds the total limit",
            )));
        }
        Ok(Self { total_bytes, mapped_bytes, staging_bytes, safety_reserve_bytes })
    }

    /// Returns the total logical memory ceiling.
    pub const fn total_bytes(self) -> u64 {
        self.total_bytes
    }
}

/// Exact trusted policy for one plan derivation; there is no implicit fallback.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PlacementPolicy {
    mode: WeightResidencyMode,
    prefetch_depth: u8,
}

impl PlacementPolicy {
    /// Constructs a policy with a bounded prefetch depth.
    pub fn new(mode: WeightResidencyMode, prefetch_depth: u8) -> Result<Self, PlacementError> {
        if prefetch_depth > MAX_PREFETCH_DEPTH
            || (mode != WeightResidencyMode::LayerStreamed && prefetch_depth != 0)
        {
            return Err(PlacementError::InvalidPlan(EngineError::invalid_input(
                "placement prefetch depth is invalid for the selected mode",
            )));
        }
        Ok(Self { mode, prefetch_depth })
    }

    /// Requires the existing complete-residency fast path.
    pub const fn fully_resident() -> Self {
        Self { mode: WeightResidencyMode::FullyResident, prefetch_depth: 0 }
    }

    /// Selects bounded layer streaming with depth-one prefetch.
    pub const fn layer_streamed() -> Self {
        Self { mode: WeightResidencyMode::LayerStreamed, prefetch_depth: 1 }
    }

    /// Returns the exact owner-selected residency mode.
    pub const fn mode(self) -> WeightResidencyMode {
        self.mode
    }
}

/// Stable architecture stage kind.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PlacementStageKind {
    /// Token embedding input stage.
    Embedding,
    /// One exact zero-based decoder layer.
    DecoderLayer(u32),
    /// Final normalization and output projection.
    Output,
}

/// One trusted immutable tensor range in a placement stage.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlacementTensorRange {
    logical_name: String,
    source_name: String,
    dtype: DType,
    quantization: Quantization,
    shard_index: u32,
    byte_offset: u64,
    byte_length: u64,
    required_alignment: u32,
    content_hash: ModelPackageHash,
    execution_bytes: u64,
}

impl PlacementTensorRange {
    /// Returns the architecture-visible tensor name.
    pub fn logical_name(&self) -> &str {
        &self.logical_name
    }

    /// Returns the verified manifest tensor name supplying the bytes.
    pub fn source_name(&self) -> &str {
        &self.source_name
    }

    /// Returns the signed storage data type.
    pub const fn dtype(&self) -> DType {
        self.dtype
    }

    /// Returns the signed storage quantization.
    pub const fn quantization(&self) -> Quantization {
        self.quantization
    }

    /// Returns the trusted shard index; this is never accepted from a client.
    pub const fn shard_index(&self) -> u32 {
        self.shard_index
    }

    /// Returns the trusted byte offset; this is never accepted from a client.
    pub const fn byte_offset(&self) -> u64 {
        self.byte_offset
    }

    /// Returns the exact signed storage length.
    pub const fn byte_length(&self) -> u64 {
        self.byte_length
    }

    /// Returns the required signed alignment.
    pub const fn required_alignment(&self) -> u32 {
        self.required_alignment
    }

    /// Returns the exact signed tensor hash.
    pub const fn content_hash(&self) -> ModelPackageHash {
        self.content_hash
    }

    /// Returns the exact F32 execution-materialization length.
    pub const fn execution_bytes(&self) -> u64 {
        self.execution_bytes
    }
}

/// One unequal-size architecture stage derived from signed tensor records.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlacementStage {
    ordinal: u32,
    kind: PlacementStageKind,
    tensors: Vec<PlacementTensorRange>,
    storage_bytes: u64,
    execution_bytes: u64,
    metadata_bytes: u64,
    accounted_bytes: u64,
}

impl PlacementStage {
    /// Returns the deterministic stage ordinal.
    pub const fn ordinal(&self) -> u32 {
        self.ordinal
    }

    /// Returns the architecture stage kind.
    pub const fn kind(&self) -> PlacementStageKind {
        self.kind
    }

    /// Returns immutable trusted tensor ranges.
    pub fn tensors(&self) -> &[PlacementTensorRange] {
        &self.tensors
    }

    /// Returns the exact signed bytes referenced by this stage.
    pub const fn storage_bytes(&self) -> u64 {
        self.storage_bytes
    }

    /// Returns the exact F32 materialization bytes for this stage.
    pub const fn execution_bytes(&self) -> u64 {
        self.execution_bytes
    }

    /// Returns the conservative tensor-map metadata reservation.
    pub const fn metadata_bytes(&self) -> u64 {
        self.metadata_bytes
    }

    /// Returns source snapshots, F32 tensor bytes, and materialized-map metadata.
    pub const fn accounted_bytes(&self) -> u64 {
        self.accounted_bytes
    }

    fn source_names(&self) -> Result<Vec<String>, PlacementError> {
        let mut names = Vec::new();
        names.try_reserve_exact(self.tensors.len()).map_err(|_| {
            PlacementError::ResourceExhausted(EngineError::resource_exhaustion(
                "placement source-name allocation failed",
            ))
        })?;
        for tensor in &self.tensors {
            if !names.iter().any(|name| name == tensor.source_name()) {
                names.push(tensor.source_name.clone());
            }
        }
        Ok(names)
    }
}

/// Immutable execution placement bound to one exact verified artifact/backend.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExecutionPlacementPlan {
    model_id: ModelId,
    artifact_id: Option<String>,
    package_digest: ModelPackageHash,
    package_format_version: (u16, u16),
    architecture_id: ArchitectureId,
    backend_id: BackendId,
    quantization: Quantization,
    model_generation: u64,
    plan_generation: u64,
    mode: WeightResidencyMode,
    topology: MemoryTopology,
    stages: Vec<PlacementStage>,
    mapped_bytes: u64,
    immutable_overhead_bytes: u64,
    complete_execution_bytes: u64,
    complete_materialized_bytes: u64,
    guaranteed_staging_bytes: u64,
    required_envelope_bytes: u64,
    prefetch_depth: u8,
}

struct PlacementMeasurements {
    stages: Vec<PlacementStage>,
    mapped_bytes: u64,
    immutable_overhead_bytes: u64,
    complete_execution_bytes: u64,
    complete_materialized_bytes: u64,
}

impl ExecutionPlacementPlan {
    /// Derives one plan exclusively from a verified package and trusted limits.
    #[allow(clippy::too_many_arguments)]
    pub fn derive(
        package: &VerifiedDirectoryPackage,
        capabilities: &BackendPlacementCapabilities,
        envelope: PlacementResourceEnvelope,
        policy: PlacementPolicy,
        currently_reserved_bytes: u64,
        plan_generation: u64,
    ) -> Result<Self, PlacementError> {
        if plan_generation == 0 {
            return Err(PlacementError::InvalidPlan(EngineError::invalid_input(
                "placement plan generation must be nonzero",
            )));
        }
        let manifest = package.manifest();
        if manifest.layer_count() == 0 || manifest.layer_count() > MAX_PLACEMENT_LAYERS {
            return Err(PlacementError::InvalidPlan(EngineError::corrupt_model_package(
                "placement layer count is outside hard bounds",
            )));
        }
        let measurements = measure_verified_package(package)?;
        if measurements.mapped_bytes > envelope.mapped_bytes {
            return Err(resource("mapped package exceeds placement envelope"));
        }
        let guaranteed_staging_bytes =
            select_staging_requirement(&measurements, capabilities, envelope, policy)?;
        let execution_commitment = if policy.mode == WeightResidencyMode::FullyResident {
            measurements.complete_materialized_bytes
        } else {
            guaranteed_staging_bytes
        };
        let required_envelope_bytes = [
            currently_reserved_bytes,
            measurements.mapped_bytes,
            measurements.immutable_overhead_bytes,
            execution_commitment,
            envelope.safety_reserve_bytes,
        ]
        .into_iter()
        .try_fold(0_u64, |total, value| checked_add(total, value, "placement resource envelope"))?;
        if required_envelope_bytes > envelope.total_bytes {
            return Err(resource("placement cannot preserve the configured safety reserve"));
        }
        Ok(Self {
            model_id: package.registration().model_id().clone(),
            artifact_id: package.registration().registry_artifact_id().map(str::to_owned),
            package_digest: *package.registration().expected_hash(),
            package_format_version: DIRECTORY_MANIFEST_VERSION,
            architecture_id: manifest.architecture_id().clone(),
            backend_id: capabilities.backend_id.clone(),
            quantization: manifest.quantization(),
            model_generation: package.registration().registry_sequence().unwrap_or(1),
            plan_generation,
            mode: policy.mode,
            topology: capabilities.topology,
            stages: measurements.stages,
            mapped_bytes: measurements.mapped_bytes,
            immutable_overhead_bytes: measurements.immutable_overhead_bytes,
            complete_execution_bytes: measurements.complete_execution_bytes,
            complete_materialized_bytes: measurements.complete_materialized_bytes,
            guaranteed_staging_bytes,
            required_envelope_bytes,
            prefetch_depth: policy.prefetch_depth,
        })
    }

    /// Returns the exact registered model identity.
    pub const fn model_id(&self) -> &ModelId {
        &self.model_id
    }

    /// Returns the optional signed Model Foundry artifact identity.
    pub fn artifact_id(&self) -> Option<&str> {
        self.artifact_id.as_deref()
    }

    /// Returns the exact configured package digest.
    pub const fn package_digest(&self) -> ModelPackageHash {
        self.package_digest
    }

    /// Returns the first-party package format revision.
    pub const fn package_format_version(&self) -> (u16, u16) {
        self.package_format_version
    }

    /// Returns the exact allowlisted architecture identity.
    pub const fn architecture_id(&self) -> &ArchitectureId {
        &self.architecture_id
    }

    /// Returns the actual trusted backend identity.
    pub const fn backend_id(&self) -> &BackendId {
        &self.backend_id
    }

    /// Returns the signed package quantization identity.
    pub const fn quantization(&self) -> Quantization {
        self.quantization
    }

    /// Returns the signed registry sequence or owner-pinned generation one.
    pub const fn model_generation(&self) -> u64 {
        self.model_generation
    }

    /// Returns the nonzero engine-owned plan generation.
    pub const fn plan_generation(&self) -> u64 {
        self.plan_generation
    }

    /// Returns the exact selected residency mode.
    pub const fn mode(&self) -> WeightResidencyMode {
        self.mode
    }

    /// Returns the physical memory topology used for the decision.
    pub const fn topology(&self) -> MemoryTopology {
        self.topology
    }

    /// Returns the immutable ordered stages.
    pub fn stages(&self) -> &[PlacementStage] {
        &self.stages
    }

    /// Returns complete mapped tensor-shard bytes.
    pub const fn mapped_bytes(&self) -> u64 {
        self.mapped_bytes
    }

    /// Returns tokenizer, provenance, and tensor-metadata bytes.
    pub const fn immutable_overhead_bytes(&self) -> u64 {
        self.immutable_overhead_bytes
    }

    /// Returns complete F32 execution-weight bytes.
    pub const fn complete_execution_bytes(&self) -> u64 {
        self.complete_execution_bytes
    }

    /// Returns the peak complete-residency commitment.
    ///
    /// This includes every F32 execution tensor, the complete materialized
    /// tensor map, and one largest source snapshot used for checked conversion.
    pub const fn complete_materialized_bytes(&self) -> u64 {
        self.complete_materialized_bytes
    }

    /// Returns the maximum simultaneously staged F32 bytes.
    pub const fn guaranteed_staging_bytes(&self) -> u64 {
        self.guaranteed_staging_bytes
    }

    /// Returns the conservative total envelope commitment.
    pub const fn required_envelope_bytes(&self) -> u64 {
        self.required_envelope_bytes
    }

    /// Returns the bounded asynchronous prefetch depth.
    pub const fn prefetch_depth(&self) -> u8 {
        self.prefetch_depth
    }

    /// Revalidates that a live verified package still has this plan's exact
    /// registry, digest, architecture, quantization, and generation identity.
    pub fn verify_package_binding(
        &self,
        package: &VerifiedDirectoryPackage,
    ) -> Result<(), PlacementError> {
        if package.registration().model_id() != &self.model_id
            || package.registration().expected_hash() != &self.package_digest
            || package.manifest().architecture_id() != &self.architecture_id
            || package.manifest().quantization() != self.quantization
            || package.registration().registry_sequence().unwrap_or(1) != self.model_generation
        {
            return Err(PlacementError::ArtifactIdentityMismatch);
        }
        Ok(())
    }
}

fn measure_verified_package(
    package: &VerifiedDirectoryPackage,
) -> Result<PlacementMeasurements, PlacementError> {
    let manifest = package.manifest();
    let stages = derive_stages(manifest.tensors(), manifest.layer_count(), manifest.tied_output())?;
    let mapped_bytes = manifest.shards().iter().try_fold(0_u64, |total, shard| {
        checked_add(total, shard.byte_length(), "mapped shard bytes")
    })?;
    let manifest_metadata_bytes = u64::try_from(manifest.tensors().len())
        .ok()
        .and_then(|count| count.checked_mul(192))
        .ok_or_else(|| resource("placement manifest metadata bytes overflow"))?;
    let tokenizer_bytes = u64::try_from(package.tokenizer_bytes().len())
        .map_err(|_| resource("tokenizer bytes are not representable"))?;
    let provenance_bytes = u64::try_from(package.provenance_bytes().len())
        .map_err(|_| resource("provenance bytes are not representable"))?;
    let immutable_overhead_bytes = checked_add(
        checked_add(tokenizer_bytes, provenance_bytes, "immutable package overhead")?,
        manifest_metadata_bytes,
        "immutable package overhead",
    )?;
    let complete_execution_bytes = stages.iter().try_fold(0_u64, |total, stage| {
        checked_add(total, stage.execution_bytes, "complete execution weights")
    })?;
    let largest_source_snapshot = manifest
        .tensors()
        .iter()
        .map(DirectoryTensorRecord::byte_length)
        .max()
        .ok_or_else(|| resource("placement manifest contains no tensors"))?;
    let complete_materialized_bytes = checked_add(
        checked_add(
            complete_execution_bytes,
            manifest_metadata_bytes,
            "complete materialized weights",
        )?,
        largest_source_snapshot,
        "complete materialized weights",
    )?;
    Ok(PlacementMeasurements {
        stages,
        mapped_bytes,
        immutable_overhead_bytes,
        complete_execution_bytes,
        complete_materialized_bytes,
    })
}

fn select_staging_requirement(
    measurements: &PlacementMeasurements,
    capabilities: &BackendPlacementCapabilities,
    envelope: PlacementResourceEnvelope,
    policy: PlacementPolicy,
) -> Result<u64, PlacementError> {
    match policy.mode {
        WeightResidencyMode::FullyResident => {
            if measurements.complete_execution_bytes > capabilities.maximum_fast_bytes {
                return Err(resource("complete execution weights exceed backend capacity"));
            }
            Ok(0)
        }
        WeightResidencyMode::LayerStreamed => {
            if !capabilities.layer_streaming {
                return Err(PlacementError::UnsupportedPlacement {
                    mode: policy.mode,
                    backend: capabilities.backend_id.clone(),
                });
            }
            let required = required_streaming_staging(&measurements.stages, policy.prefetch_depth)?;
            if required > envelope.staging_bytes || required > capabilities.maximum_staging_bytes {
                return Err(resource("streaming stages exceed the staging ceiling"));
            }
            Ok(required)
        }
        WeightResidencyMode::HybridResident => Err(PlacementError::UnsupportedPlacement {
            mode: policy.mode,
            backend: capabilities.backend_id.clone(),
        }),
    }
}

/// Request-independent cancellation state for bounded placement preparation.
#[derive(Clone, Debug, Default)]
pub struct PlacementCancellation(Arc<AtomicBool>);

/// Read-only cancellation probe shared by trusted placement execution.
///
/// The local service implements this through its existing request cancellation
/// flag, so a cancellation watcher can stop an in-flight prefetch without a
/// second polling thread or client-visible placement state.
pub trait PlacementCancellationProbe: Sync {
    /// Returns whether placement work must stop.
    fn is_cancelled(&self) -> bool;
}

impl PlacementCancellation {
    /// Creates a live cancellation state.
    pub fn new() -> Self {
        Self::default()
    }

    /// Requests cancellation.
    pub fn cancel(&self) {
        self.0.store(true, Ordering::Release);
    }

    /// Returns whether cancellation has been requested.
    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::Acquire)
    }
}

impl PlacementCancellationProbe for PlacementCancellation {
    fn is_cancelled(&self) -> bool {
        self.is_cancelled()
    }
}

impl PlacementCancellationProbe for AtomicBool {
    fn is_cancelled(&self) -> bool {
        self.load(Ordering::Acquire)
    }
}

/// Deterministic test/operation hook for placement failure injection.
pub trait PlacementFaultInjector: Send + Sync {
    /// Runs before one stage is materialized.
    fn before_materialize(&self, _stage: &PlacementStage) -> Result<(), EngineError> {
        Ok(())
    }

    /// Runs before one materialized stage is handed to trusted execution.
    fn before_execute(&self, _stage: &PlacementStage) -> Result<(), EngineError> {
        Ok(())
    }
}

/// Production no-fault implementation.
#[derive(Debug, Default)]
pub struct NoPlacementFaults;

impl PlacementFaultInjector for NoPlacementFaults {}

/// Content-free metrics from one bounded streamed-stage traversal.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct StreamedPlacementMetrics {
    /// Stages successfully handed to execution.
    pub stages_executed: u32,
    /// Prefetch workers successfully started.
    pub prefetch_started: u32,
    /// Prefetched stages successfully materialized.
    pub prefetch_completed: u32,
    /// Observed prefetch failures.
    pub prefetch_failed: u32,
    /// Maximum planned simultaneous staging bytes.
    pub peak_staging_bytes: u64,
}

/// Runs a trusted layer-streamed plan with at most one prefetch worker.
pub fn run_layer_streamed(
    plan: &ExecutionPlacementPlan,
    package: &VerifiedDirectoryPackage,
    pool: &MemoryPool,
    cancellation: &dyn PlacementCancellationProbe,
    faults: &dyn PlacementFaultInjector,
    events: &mut impl DirectorySecurityEventSink,
    mut execute: impl FnMut(&PlacementStage, &LoadedDirectoryTensors) -> Result<(), EngineError>,
) -> Result<StreamedPlacementMetrics, PlacementError> {
    if plan.mode != WeightResidencyMode::LayerStreamed {
        return Err(PlacementError::InvalidPlan(EngineError::invalid_input(
            "streamed runner requires a layer-streamed plan",
        )));
    }
    plan.verify_package_binding(package)?;
    let materialize =
        |stage: &PlacementStage| materialize_stage(package, pool, cancellation, faults, stage);
    let mut trusted_execute = |stage: &PlacementStage, tensors: &LoadedDirectoryTensors| {
        faults.before_execute(stage)?;
        execute(stage, tensors)
    };
    let result = run_bounded_pipeline(
        &plan.stages,
        plan.guaranteed_staging_bytes,
        cancellation,
        &materialize,
        &mut trusted_execute,
    );
    if matches!(result, Err(PlacementError::ArtifactIdentityMismatch)) {
        events.emit(SecurityEvent::ModelPackageRejected {
            model_id: package.registration().model_id().clone(),
            reason: ModelPackageRejection::SignatureFailure,
        });
    }
    result
}

/// Placement-specific fail-closed error vocabulary.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PlacementError {
    /// The trusted backend has no implementation for the exact requested mode.
    UnsupportedPlacement {
        /// Requested mode.
        mode: WeightResidencyMode,
        /// Exact selected backend.
        backend: BackendId,
    },
    /// The verified artifact no longer matches the immutable plan binding.
    ArtifactIdentityMismatch,
    /// A plan or policy violated trusted invariants.
    InvalidPlan(EngineError),
    /// A finite placement resource was unavailable.
    ResourceExhausted(EngineError),
    /// Placement or prefetch was cancelled.
    Cancelled,
    /// Trusted backend preparation/execution failed.
    BackendFailure(EngineError),
}

impl PlacementError {
    /// Converts into the shared bounded engine error vocabulary.
    pub fn into_engine_error(self) -> EngineError {
        match self {
            Self::UnsupportedPlacement { .. } => {
                EngineError::not_implemented("selected backend does not implement placement mode")
            }
            Self::ArtifactIdentityMismatch => {
                EngineError::authentication_failure("placement artifact identity changed")
            }
            Self::InvalidPlan(error)
            | Self::ResourceExhausted(error)
            | Self::BackendFailure(error) => error,
            Self::Cancelled => EngineError::cancellation(None),
        }
    }
}

fn derive_stages(
    records: &[DirectoryTensorRecord],
    layer_count: u32,
    tied_output: bool,
) -> Result<Vec<PlacementStage>, PlacementError> {
    let stage_count = usize::try_from(layer_count)
        .ok()
        .and_then(|count| count.checked_add(2))
        .ok_or_else(|| resource("placement stage count overflows"))?;
    let mut grouped = Vec::new();
    grouped
        .try_reserve_exact(stage_count)
        .map_err(|_| resource("placement stage allocation failed"))?;
    grouped.push((PlacementStageKind::Embedding, Vec::new()));
    for layer in 0..layer_count {
        grouped.push((PlacementStageKind::DecoderLayer(layer), Vec::new()));
    }
    grouped.push((PlacementStageKind::Output, Vec::new()));

    for record in records {
        let stage_index = if record.name() == "token_embedding.weight" {
            0
        } else if matches!(record.name(), "final_norm.weight" | "output.weight") {
            stage_count - 1
        } else {
            let layer = parse_layer_index(record.name(), layer_count)?;
            usize::try_from(layer)
                .ok()
                .and_then(|index| index.checked_add(1))
                .ok_or_else(|| resource("placement layer index overflows"))?
        };
        grouped[stage_index].1.push(range_from_record(record, record.name())?);
    }
    if tied_output {
        let embedding = records
            .iter()
            .find(|record| record.name() == "token_embedding.weight")
            .ok_or_else(|| {
                PlacementError::InvalidPlan(EngineError::corrupt_model_package(
                    "tied output is missing its embedding source",
                ))
            })?;
        grouped[stage_count - 1].1.push(range_from_record(embedding, "output.weight")?);
    }

    let mut stages = Vec::new();
    stages
        .try_reserve_exact(stage_count)
        .map_err(|_| resource("placement stage allocation failed"))?;
    for (ordinal, (kind, tensors)) in grouped.into_iter().enumerate() {
        if tensors.is_empty() {
            return Err(PlacementError::InvalidPlan(EngineError::corrupt_model_package(
                "placement architecture stage has no tensors",
            )));
        }
        let storage_bytes = tensors.iter().try_fold(0_u64, |total, tensor| {
            checked_add(total, tensor.byte_length, "stage storage bytes")
        })?;
        let execution_bytes = tensors.iter().try_fold(0_u64, |total, tensor| {
            checked_add(total, tensor.execution_bytes, "stage execution bytes")
        })?;
        let unique_sources = tensors.iter().fold(Vec::<&str>::new(), |mut names, tensor| {
            if !names.contains(&tensor.source_name()) {
                names.push(tensor.source_name());
            }
            names
        });
        let metadata_bytes = u64::try_from(unique_sources.len())
            .ok()
            .and_then(|count| count.checked_mul(192))
            .ok_or_else(|| resource("stage metadata bytes overflow"))?;
        let accounted_bytes = checked_add(
            checked_add(execution_bytes, storage_bytes, "stage accounted bytes")?,
            metadata_bytes,
            "stage accounted bytes",
        )?;
        stages.push(PlacementStage {
            ordinal: u32::try_from(ordinal).map_err(|_| resource("stage ordinal overflows"))?,
            kind,
            tensors,
            storage_bytes,
            execution_bytes,
            metadata_bytes,
            accounted_bytes,
        });
    }
    Ok(stages)
}

fn parse_layer_index(name: &str, layer_count: u32) -> Result<u32, PlacementError> {
    let suffix = name.strip_prefix("layers.").ok_or_else(|| {
        PlacementError::InvalidPlan(EngineError::corrupt_model_package(
            "placement found an unknown architecture tensor",
        ))
    })?;
    let (index, remainder) = suffix.split_once('.').ok_or_else(|| {
        PlacementError::InvalidPlan(EngineError::corrupt_model_package(
            "placement layer tensor name is malformed",
        ))
    })?;
    if index.len() != 3 || !index.bytes().all(|byte| byte.is_ascii_digit()) || remainder.is_empty()
    {
        return Err(PlacementError::InvalidPlan(EngineError::corrupt_model_package(
            "placement layer tensor name is not canonical",
        )));
    }
    let layer = index.parse::<u32>().map_err(|_| {
        PlacementError::InvalidPlan(EngineError::corrupt_model_package(
            "placement layer index is invalid",
        ))
    })?;
    if layer >= layer_count {
        return Err(PlacementError::InvalidPlan(EngineError::corrupt_model_package(
            "placement layer index exceeds the manifest",
        )));
    }
    Ok(layer)
}

fn range_from_record(
    record: &DirectoryTensorRecord,
    logical_name: &str,
) -> Result<PlacementTensorRange, PlacementError> {
    let execution_bytes = record
        .shape()
        .element_count()
        .checked_mul(4)
        .ok_or_else(|| resource("placement execution tensor bytes overflow"))?;
    Ok(PlacementTensorRange {
        logical_name: logical_name.to_owned(),
        source_name: record.name().to_owned(),
        dtype: record.dtype(),
        quantization: record.quantization(),
        shard_index: record.shard_index(),
        byte_offset: record.byte_offset(),
        byte_length: record.byte_length(),
        required_alignment: record.required_alignment(),
        content_hash: record.content_hash(),
        execution_bytes,
    })
}

fn required_streaming_staging(
    stages: &[PlacementStage],
    prefetch_depth: u8,
) -> Result<u64, PlacementError> {
    if prefetch_depth == 0 {
        return stages
            .iter()
            .map(PlacementStage::accounted_bytes)
            .max()
            .ok_or_else(|| resource("placement plan has no stages"));
    }
    let mut maximum = 0_u64;
    for index in 0..stages.len() {
        let current = stages[index].accounted_bytes;
        let next = stages.get(index + 1).map_or(0, |stage| stage.accounted_bytes);
        maximum = maximum.max(checked_add(current, next, "adjacent staging bytes")?);
    }
    Ok(maximum)
}

fn materialize_stage(
    package: &VerifiedDirectoryPackage,
    pool: &MemoryPool,
    cancellation: &dyn PlacementCancellationProbe,
    faults: &dyn PlacementFaultInjector,
    stage: &PlacementStage,
) -> Result<LoadedDirectoryTensors, PlacementError> {
    check_cancelled(cancellation)?;
    faults.before_materialize(stage).map_err(PlacementError::BackendFailure)?;
    let names = stage.source_names()?;
    let loaded = package.materialize_named_tensors(&names, pool).map_err(classify_engine_error)?;
    check_cancelled(cancellation)?;
    Ok(loaded)
}

fn run_bounded_pipeline<T, M, E>(
    stages: &[PlacementStage],
    peak_staging_bytes: u64,
    cancellation: &dyn PlacementCancellationProbe,
    materialize: &M,
    execute: &mut E,
) -> Result<StreamedPlacementMetrics, PlacementError>
where
    T: Send,
    M: Fn(&PlacementStage) -> Result<T, PlacementError> + Sync,
    E: FnMut(&PlacementStage, &T) -> Result<(), EngineError>,
{
    check_cancelled(cancellation)?;
    let first = stages.first().ok_or_else(|| {
        PlacementError::InvalidPlan(EngineError::internal_invariant("placement plan has no stages"))
    })?;
    let mut current = materialize(first)?;
    let mut metrics =
        StreamedPlacementMetrics { peak_staging_bytes, ..StreamedPlacementMetrics::default() };
    for index in 0..stages.len() {
        let stage = &stages[index];
        check_cancelled(cancellation)?;
        let next_stage = stages.get(index + 1);
        let (execution_result, next_result, started) = thread::scope(|scope| {
            let handle = next_stage.map(|next| {
                thread::Builder::new()
                    .name("placement-prefetch".to_owned())
                    .spawn_scoped(scope, move || materialize(next))
            });
            let execution_result = execute(stage, &current);
            match handle {
                None => (execution_result, None, false),
                Some(Err(_error)) => (
                    execution_result,
                    Some(Err(PlacementError::BackendFailure(EngineError::backend_failure(
                        "placement prefetch thread creation failed",
                    )))),
                    false,
                ),
                Some(Ok(handle)) => {
                    let result = handle.join().unwrap_or_else(|_| {
                        Err(PlacementError::BackendFailure(EngineError::backend_failure(
                            "placement prefetch worker panicked",
                        )))
                    });
                    (execution_result, Some(result), true)
                }
            }
        });
        if started {
            metrics.prefetch_started = checked_increment(metrics.prefetch_started)?;
        }
        if let Err(error) = execution_result {
            drop(next_result);
            return Err(PlacementError::BackendFailure(error));
        }
        metrics.stages_executed = checked_increment(metrics.stages_executed)?;
        let Some(next_result) = next_result else {
            break;
        };
        match next_result {
            Ok(next) => {
                metrics.prefetch_completed = checked_increment(metrics.prefetch_completed)?;
                current = next;
            }
            Err(error) => {
                metrics.prefetch_failed = checked_increment(metrics.prefetch_failed)?;
                return Err(error);
            }
        }
        check_cancelled(cancellation)?;
    }
    Ok(metrics)
}

fn classify_engine_error(error: EngineError) -> PlacementError {
    match error {
        EngineError::ResourceExhaustion(_) => PlacementError::ResourceExhausted(error),
        EngineError::Cancellation(_) => PlacementError::Cancelled,
        EngineError::SignatureFailure(_) | EngineError::AuthenticationFailure(_) => {
            PlacementError::ArtifactIdentityMismatch
        }
        _ => PlacementError::BackendFailure(error),
    }
}

fn check_cancelled(cancellation: &dyn PlacementCancellationProbe) -> Result<(), PlacementError> {
    if cancellation.is_cancelled() { Err(PlacementError::Cancelled) } else { Ok(()) }
}

fn checked_add(left: u64, right: u64, label: &'static str) -> Result<u64, PlacementError> {
    left.checked_add(right).ok_or_else(|| resource(label))
}

fn checked_increment(value: u32) -> Result<u32, PlacementError> {
    value.checked_add(1).ok_or_else(|| resource("placement metric counter overflows"))
}

fn resource(reason: &'static str) -> PlacementError {
    PlacementError::ResourceExhausted(EngineError::resource_exhaustion(reason))
}

#[cfg(test)]
mod tests {
    use super::*;
    use personalai_engine_types::TensorShape;
    use std::sync::Mutex;

    fn hash(byte: u8) -> ModelPackageHash {
        ModelPackageHash::from_sha384_bytes([byte; ModelPackageHash::BYTE_LEN])
    }

    fn record(name: &str, elements: u64, offset: u64) -> DirectoryTensorRecord {
        DirectoryTensorRecord::new(
            name,
            DType::F32,
            TensorShape::new([elements]).unwrap(),
            0,
            offset,
            elements.checked_mul(4).unwrap(),
            4,
            Quantization::None,
            hash(u8::try_from(elements % 255).unwrap()),
        )
        .unwrap()
    }

    fn stage(ordinal: u32, bytes: u64) -> PlacementStage {
        PlacementStage {
            ordinal,
            kind: if ordinal == 0 {
                PlacementStageKind::Embedding
            } else {
                PlacementStageKind::DecoderLayer(ordinal - 1)
            },
            tensors: Vec::new(),
            storage_bytes: bytes,
            execution_bytes: bytes,
            metadata_bytes: 0,
            accounted_bytes: bytes,
        }
    }

    struct CountedMaterialization(Arc<Mutex<u32>>);

    impl Drop for CountedMaterialization {
        fn drop(&mut self) {
            let mut live = self.0.lock().unwrap();
            *live -= 1;
        }
    }

    #[test]
    fn unequal_layers_and_tied_output_produce_exact_ordered_stages() {
        let records = vec![
            record("final_norm.weight", 2, 0),
            record("layers.000.q_proj.weight", 7, 8),
            record("layers.001.q_proj.weight", 13, 36),
            record("token_embedding.weight", 5, 88),
        ];
        let stages = derive_stages(&records, 2, true).unwrap();
        assert_eq!(stages.len(), 4);
        assert_eq!(stages[0].kind(), PlacementStageKind::Embedding);
        assert_eq!(stages[1].kind(), PlacementStageKind::DecoderLayer(0));
        assert_eq!(stages[2].kind(), PlacementStageKind::DecoderLayer(1));
        assert_eq!(stages[3].kind(), PlacementStageKind::Output);
        assert_eq!(stages[0].execution_bytes(), 20);
        assert_eq!(stages[1].execution_bytes(), 28);
        assert_eq!(stages[2].execution_bytes(), 52);
        assert_eq!(stages[3].execution_bytes(), 28);
        assert_eq!(stages[3].tensors()[1].logical_name(), "output.weight");
        assert_eq!(stages[3].tensors()[1].source_name(), "token_embedding.weight");
        assert_eq!(required_streaming_staging(&stages, 1).unwrap(), 736);
        assert_eq!(required_streaming_staging(&stages, 0).unwrap(), 440);
    }

    #[test]
    fn unknown_or_out_of_range_layer_names_fail_closed() {
        let unknown = vec![record("package.script", 1, 0)];
        assert!(matches!(derive_stages(&unknown, 1, false), Err(PlacementError::InvalidPlan(_))));
        let outside = vec![
            record("token_embedding.weight", 1, 0),
            record("layers.001.q_proj.weight", 1, 4),
            record("final_norm.weight", 1, 8),
            record("output.weight", 1, 12),
        ];
        assert!(matches!(derive_stages(&outside, 1, false), Err(PlacementError::InvalidPlan(_))));
    }

    #[test]
    fn placement_policy_and_resource_envelopes_are_bounded() {
        assert!(PlacementPolicy::new(WeightResidencyMode::FullyResident, 1).is_err());
        assert!(PlacementPolicy::new(WeightResidencyMode::LayerStreamed, 2).is_err());
        assert!(PlacementResourceEnvelope::new(100, 100, 100, 100).is_ok());
        assert!(PlacementResourceEnvelope::new(100, 100, 100, 101).is_err());
        assert!(PlacementResourceEnvelope::new(MAX_RESOURCE_MEMORY_BYTES + 1, 1, 1, 0).is_err());
        assert!(
            BackendPlacementCapabilities::cpu_reference(MAX_RESOURCE_MEMORY_BYTES + 1).is_err()
        );
        assert!(BackendPlacementCapabilities::cpu_reference(0).is_err());
        assert_eq!(PlacementPolicy::layer_streamed().prefetch_depth, 1);
    }

    #[test]
    fn bounded_pipeline_prefetches_one_stage_and_preserves_order() {
        let stages = vec![stage(0, 10), stage(1, 30), stage(2, 20)];
        let cancellation = PlacementCancellation::new();
        let materialized = Mutex::new(Vec::new());
        let materialize = |stage: &PlacementStage| {
            materialized.lock().unwrap().push(stage.ordinal());
            Ok(stage.ordinal())
        };
        let mut executed = Vec::new();
        let metrics =
            run_bounded_pipeline(&stages, 50, &cancellation, &materialize, &mut |stage, value| {
                assert_eq!(stage.ordinal(), *value);
                executed.push(stage.ordinal());
                Ok(())
            })
            .unwrap();
        assert_eq!(executed, vec![0, 1, 2]);
        assert_eq!(*materialized.lock().unwrap(), vec![0, 1, 2]);
        assert_eq!(metrics.stages_executed, 3);
        assert_eq!(metrics.prefetch_started, 2);
        assert_eq!(metrics.prefetch_completed, 2);
        assert_eq!(metrics.peak_staging_bytes, 50);
    }

    #[test]
    fn cancellation_stops_after_joining_prefetch_and_drops_stage_state() {
        let stages = vec![stage(0, 10), stage(1, 20), stage(2, 30)];
        let cancellation = PlacementCancellation::new();
        let live = Arc::new(Mutex::new(0_u32));
        let materialize = {
            let live = Arc::clone(&live);
            move |_stage: &PlacementStage| {
                *live.lock().unwrap() += 1;
                Ok(CountedMaterialization(Arc::clone(&live)))
            }
        };
        let result = run_bounded_pipeline(
            &stages,
            50,
            &cancellation,
            &materialize,
            &mut |_stage, _value| {
                cancellation.cancel();
                Ok(())
            },
        );
        assert_eq!(result, Err(PlacementError::Cancelled));
        assert_eq!(*live.lock().unwrap(), 0);
    }

    #[test]
    fn failed_prefetch_is_observed_and_materialized_state_is_released() {
        let stages = vec![stage(0, 10), stage(1, 20)];
        let cancellation = PlacementCancellation::new();
        let materialize = |stage: &PlacementStage| {
            if stage.ordinal() == 1 {
                Err(PlacementError::BackendFailure(EngineError::backend_failure(
                    "injected prefetch failure",
                )))
            } else {
                Ok(stage.ordinal())
            }
        };
        let result = run_bounded_pipeline(
            &stages,
            30,
            &cancellation,
            &materialize,
            &mut |_stage, _value| Ok(()),
        );
        assert!(matches!(result, Err(PlacementError::BackendFailure(_))));
    }

    #[test]
    fn checked_stage_math_rejects_overflow_for_many_deterministic_inputs() {
        for left in [0, 1, u64::MAX / 2, u64::MAX - 1, u64::MAX] {
            for right in [0, 1, u64::MAX / 2, u64::MAX] {
                assert_eq!(
                    checked_add(left, right, "test").is_ok(),
                    left.checked_add(right).is_some()
                );
            }
        }
    }

    #[test]
    fn verified_package_is_safe_to_share_with_one_bounded_prefetch_worker() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<VerifiedDirectoryPackage>();
    }
}
