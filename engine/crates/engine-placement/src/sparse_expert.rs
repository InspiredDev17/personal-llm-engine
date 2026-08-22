//! Bounded sparse-expert planning and immutable host-cache lifecycle.
//!
//! This module is deliberately engine-internal infrastructure. It does not add
//! a model architecture, package format revision, protocol message, or client
//! placement authority. A future reviewed architecture implementation must
//! derive descriptors from an already verified package before using this code.

use super::{
    ExecutionPlacementPlan, PlacementCancellation, PlacementCancellationProbe, PlacementError,
    check_cancelled, classify_engine_error, resource,
};
use personalai_engine_crypto::PackageHashAccumulator;
use personalai_engine_memory::{HostBuffer, MemoryOwner, MemoryPool, MemoryReservation};
use personalai_engine_model_format::{Quantization, encode_dtype};
use personalai_engine_types::{
    ArchitectureId, BackendId, DType, EngineError, MAX_RESOURCE_MEMORY_BYTES, ModelId,
    ModelPackageHash,
};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread;
use std::time::Duration;

/// Largest reviewed sparse-expert layer population in one internal plan.
pub const MAX_SPARSE_EXPERT_LAYERS: u32 = 4_096;
/// Largest reviewed expert population in one layer.
pub const MAX_EXPERTS_PER_LAYER: u32 = 4_096;
/// Aggregate descriptor ceiling across one model.
pub const MAX_TOTAL_EXPERTS: u32 = 262_144;
/// Largest selected expert count accepted for one token and layer.
pub const MAX_ACTIVE_EXPERTS_PER_TOKEN: u16 = 64;
/// Bounded expert lookahead. There is never more than one prefetch worker per cache.
pub const MAX_EXPERT_PREFETCH_DEPTH: u8 = 1;
/// Process-wide ceiling for sparse-expert prefetch workers across all model caches.
pub const MAX_GLOBAL_EXPERT_PREFETCH_WORKERS: usize = 4;
/// Explicit stack requested for each short-lived sparse-expert prefetch worker.
pub const SPARSE_EXPERT_PREFETCH_STACK_BYTES: usize = 256 * 1024;
/// Cancellation polling granularity while verifying one expert payload.
pub const SPARSE_EXPERT_HASH_CHUNK_BYTES: usize = 64 * 1024;
/// Conservative aggregate-accounting charge for one preallocated cache slot.
///
/// The complete slot table is reserved before the cache becomes usable, so a
/// later expert insertion performs no map or table allocation.
pub const SPARSE_EXPERT_CACHE_SLOT_BYTES: u64 = 256;

const SPARSE_EXPERT_PLAN_DIGEST_DOMAIN_V1: &[u8] = b"PersonalAI.SparseExpertPlan.v1\0";

/// One architecture-owned expert coordinate.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SparseExpertId {
    layer: u32,
    expert: u32,
}

impl SparseExpertId {
    /// Creates a coordinate inside hard engine bounds.
    pub fn new(layer: u32, expert: u32) -> Result<Self, PlacementError> {
        if layer >= MAX_SPARSE_EXPERT_LAYERS || expert >= MAX_EXPERTS_PER_LAYER {
            return Err(invalid("sparse expert coordinate is outside hard bounds"));
        }
        Ok(Self { layer, expert })
    }

    /// Zero-based layer ordinal.
    pub const fn layer(self) -> u32 {
        self.layer
    }

    /// Zero-based expert ordinal inside the layer.
    pub const fn expert(self) -> u32 {
        self.expert
    }
}

/// Exact immutable identity shared by one plan, source, cache, and lease.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SparseExpertPlanIdentity {
    model_id: ModelId,
    artifact_id: Option<String>,
    package_digest: ModelPackageHash,
    package_format_version: (u16, u16),
    architecture_id: ArchitectureId,
    backend_id: BackendId,
    quantization: Quantization,
    model_generation: u64,
    plan_generation: u64,
}

impl SparseExpertPlanIdentity {
    /// Derives an exact sparse-expert identity from an already verified dense plan.
    pub fn from_execution_plan(plan: &ExecutionPlacementPlan) -> Self {
        Self {
            model_id: plan.model_id().clone(),
            artifact_id: plan.artifact_id().map(str::to_owned),
            package_digest: plan.package_digest(),
            package_format_version: plan.package_format_version(),
            architecture_id: plan.architecture_id().clone(),
            backend_id: plan.backend_id().clone(),
            quantization: plan.quantization(),
            model_generation: plan.model_generation(),
            plan_generation: plan.plan_generation(),
        }
    }

    /// Constructs trusted synthetic identity state for crate-local tests only.
    #[cfg(test)]
    fn synthetic(
        model_id: ModelId,
        package_digest: ModelPackageHash,
        architecture_id: ArchitectureId,
        backend_id: BackendId,
        model_generation: u64,
        plan_generation: u64,
    ) -> Result<Self, PlacementError> {
        if model_generation == 0 || plan_generation == 0 {
            return Err(invalid("sparse expert identity generations must be nonzero"));
        }
        Ok(Self {
            model_id,
            artifact_id: None,
            package_digest,
            package_format_version: (0, 0),
            architecture_id,
            backend_id,
            quantization: Quantization::None,
            model_generation,
            plan_generation,
        })
    }

    /// Exact registered model identity.
    pub const fn model_id(&self) -> &ModelId {
        &self.model_id
    }

    /// Optional exact Model Foundry artifact identity inherited from placement.
    pub fn artifact_id(&self) -> Option<&str> {
        self.artifact_id.as_deref()
    }

    /// Exact verified package digest.
    pub const fn package_digest(&self) -> ModelPackageHash {
        self.package_digest
    }

    /// Exact first-party package format revision.
    pub const fn package_format_version(&self) -> (u16, u16) {
        self.package_format_version
    }

    /// Exact trusted architecture implementation.
    pub const fn architecture_id(&self) -> &ArchitectureId {
        &self.architecture_id
    }

    /// Actual trusted execution backend.
    pub const fn backend_id(&self) -> &BackendId {
        &self.backend_id
    }

    /// Exact signed package quantization identity.
    pub const fn quantization(&self) -> Quantization {
        self.quantization
    }

    /// Exact loaded-model generation.
    pub const fn model_generation(&self) -> u64 {
        self.model_generation
    }

    /// Exact placement-plan generation.
    pub const fn plan_generation(&self) -> u64 {
        self.plan_generation
    }
}

/// One signed expert payload selected by trusted architecture code.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SparseExpertDescriptor {
    id: SparseExpertId,
    storage_bytes: u64,
    execution_bytes: u64,
    content_hash: ModelPackageHash,
    storage_dtype: DType,
    quantization: Quantization,
}

impl SparseExpertDescriptor {
    /// Constructs a checked, data-only descriptor.
    pub fn new(
        id: SparseExpertId,
        storage_bytes: u64,
        execution_bytes: u64,
        content_hash: ModelPackageHash,
        storage_dtype: DType,
        quantization: Quantization,
    ) -> Result<Self, PlacementError> {
        if storage_bytes == 0
            || execution_bytes == 0
            || storage_bytes > MAX_RESOURCE_MEMORY_BYTES
            || execution_bytes > MAX_RESOURCE_MEMORY_BYTES
        {
            return Err(invalid("sparse expert byte length is outside hard bounds"));
        }
        Ok(Self { id, storage_bytes, execution_bytes, content_hash, storage_dtype, quantization })
    }

    /// Architecture-owned coordinate.
    pub const fn id(&self) -> SparseExpertId {
        self.id
    }

    /// Exact signed storage bytes copied into the immutable cache.
    pub const fn storage_bytes(&self) -> u64 {
        self.storage_bytes
    }

    /// Conservative execution-memory requirement for this expert.
    pub const fn execution_bytes(&self) -> u64 {
        self.execution_bytes
    }

    /// Signed payload digest.
    pub const fn content_hash(&self) -> ModelPackageHash {
        self.content_hash
    }

    /// Exact signed tensor storage data type.
    pub const fn storage_dtype(&self) -> DType {
        self.storage_dtype
    }

    /// Signed storage encoding.
    pub const fn quantization(&self) -> Quantization {
        self.quantization
    }
}

/// One ordered sparse layer inventory and its exact active-expert count.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SparseExpertLayerPlan {
    layer: u32,
    active_experts: u16,
    experts: Vec<SparseExpertDescriptor>,
    slot_offset: u32,
    selected_cache_upper_bound: u64,
    selected_execution_upper_bound: u64,
}

impl SparseExpertLayerPlan {
    /// Constructs a canonical contiguous expert inventory.
    pub fn new(
        layer: u32,
        active_experts: u16,
        experts: Vec<SparseExpertDescriptor>,
    ) -> Result<Self, PlacementError> {
        if layer >= MAX_SPARSE_EXPERT_LAYERS {
            return Err(invalid("sparse expert layer is outside hard bounds"));
        }
        let expert_count = u32::try_from(experts.len())
            .map_err(|_| resource("sparse expert count is not representable"))?;
        if expert_count == 0 || expert_count > MAX_EXPERTS_PER_LAYER {
            return Err(invalid("sparse expert count is outside hard bounds"));
        }
        let active_count = u32::from(active_experts);
        if active_experts == 0
            || active_experts > MAX_ACTIVE_EXPERTS_PER_TOKEN
            || active_count > expert_count
        {
            return Err(invalid("active sparse expert count is invalid"));
        }
        for (ordinal, descriptor) in experts.iter().enumerate() {
            let expected = SparseExpertId::new(
                layer,
                u32::try_from(ordinal).map_err(|_| resource("sparse expert ordinal overflows"))?,
            )?;
            if descriptor.id != expected {
                return Err(invalid("sparse expert inventory is not canonical"));
            }
        }
        let selected_cache_upper_bound = largest_sum(
            experts.iter().map(SparseExpertDescriptor::storage_bytes),
            usize::from(active_experts),
            "selected sparse expert storage bytes overflow",
        )?;
        let selected_execution_upper_bound = largest_sum(
            experts.iter().map(SparseExpertDescriptor::execution_bytes),
            usize::from(active_experts),
            "selected sparse expert execution bytes overflow",
        )?;
        Ok(Self {
            layer,
            active_experts,
            experts,
            slot_offset: 0,
            selected_cache_upper_bound,
            selected_execution_upper_bound,
        })
    }

    /// Zero-based layer ordinal.
    pub const fn layer(&self) -> u32 {
        self.layer
    }

    /// Exact selected-expert count for this reviewed architecture.
    pub const fn active_experts(&self) -> u16 {
        self.active_experts
    }

    /// Canonical expert inventory.
    pub fn experts(&self) -> &[SparseExpertDescriptor] {
        &self.experts
    }

    fn descriptor(&self, id: SparseExpertId) -> Result<&SparseExpertDescriptor, PlacementError> {
        if id.layer != self.layer {
            return Err(invalid("selected expert belongs to another layer"));
        }
        let descriptor = self
            .experts
            .get(usize::try_from(id.expert).map_err(|_| resource("expert index overflows"))?)
            .ok_or_else(|| invalid("selected expert is absent from the plan"))?;
        if descriptor.id != id {
            return Err(invalid("selected expert identity is not canonical"));
        }
        Ok(descriptor)
    }
}

/// Sparse expert residency is orthogonal to the dense-weight placement mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SparseExpertResidencyMode {
    /// All expert payloads remain cached.
    FullyResident,
    /// Only selected experts and an explicitly bounded hot set remain cached.
    ExpertStreamed,
}

/// Hard engine memory ceilings for one expert plan.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SparseExpertResourceEnvelope {
    total_bytes: u64,
    cache_bytes: u64,
    execution_staging_bytes: u64,
    safety_reserve_bytes: u64,
}

impl SparseExpertResourceEnvelope {
    /// Constructs a checked envelope. No caller can exceed the aggregate limit.
    pub fn new(
        total_bytes: u64,
        cache_bytes: u64,
        execution_staging_bytes: u64,
        safety_reserve_bytes: u64,
    ) -> Result<Self, PlacementError> {
        if total_bytes == 0 || cache_bytes == 0 || execution_staging_bytes == 0 {
            return Err(invalid("sparse expert resource ceilings must be positive"));
        }
        if [total_bytes, cache_bytes, execution_staging_bytes, safety_reserve_bytes]
            .into_iter()
            .any(|value| value > MAX_RESOURCE_MEMORY_BYTES)
        {
            return Err(invalid("sparse expert resource ceiling exceeds the engine maximum"));
        }
        let committed = checked_add(
            checked_add(cache_bytes, execution_staging_bytes, "expert envelope overflows")?,
            safety_reserve_bytes,
            "expert envelope overflows",
        )?;
        if committed > total_bytes {
            return Err(resource("sparse expert envelope cannot preserve its safety reserve"));
        }
        Ok(Self { total_bytes, cache_bytes, execution_staging_bytes, safety_reserve_bytes })
    }

    /// Aggregate limit represented by this plan.
    pub const fn total_bytes(self) -> u64 {
        self.total_bytes
    }

    /// Maximum immutable expert-cache bytes.
    pub const fn cache_bytes(self) -> u64 {
        self.cache_bytes
    }

    /// Maximum simultaneous expert execution/staging bytes.
    pub const fn execution_staging_bytes(self) -> u64 {
        self.execution_staging_bytes
    }

    /// Emergency/OS margin deliberately withheld from cache and execution use.
    ///
    /// This is not consumable scratch. Active execution uses the separately
    /// admitted staging quota exposed through [`SparseExpertStagingBuffer`].
    pub const fn safety_reserve_bytes(self) -> u64 {
        self.safety_reserve_bytes
    }
}

fn sparse_expert_plan_digest(
    identity: &SparseExpertPlanIdentity,
    layers: &[SparseExpertLayerPlan],
    mode: SparseExpertResidencyMode,
    envelope: SparseExpertResourceEnvelope,
    prefetch_depth: u8,
) -> Result<ModelPackageHash, PlacementError> {
    let mut digest = PackageHashAccumulator::new();
    digest.update(SPARSE_EXPERT_PLAN_DIGEST_DOMAIN_V1);
    digest_len_prefixed(&mut digest, identity.model_id.as_str().as_bytes())?;
    match identity.artifact_id.as_deref() {
        Some(artifact_id) => {
            digest.update(&[1]);
            digest_len_prefixed(&mut digest, artifact_id.as_bytes())?;
        }
        None => digest.update(&[0]),
    }
    digest.update(identity.package_digest.as_bytes());
    digest.update(&identity.package_format_version.0.to_le_bytes());
    digest.update(&identity.package_format_version.1.to_le_bytes());
    digest_len_prefixed(&mut digest, identity.architecture_id.as_str().as_bytes())?;
    digest_len_prefixed(&mut digest, identity.backend_id.as_str().as_bytes())?;
    digest.update(&[identity.quantization.id()]);
    digest.update(&identity.model_generation.to_le_bytes());
    digest.update(&identity.plan_generation.to_le_bytes());
    digest.update(&[match mode {
        SparseExpertResidencyMode::FullyResident => 1,
        SparseExpertResidencyMode::ExpertStreamed => 2,
    }]);
    digest.update(&envelope.total_bytes.to_le_bytes());
    digest.update(&envelope.cache_bytes.to_le_bytes());
    digest.update(&envelope.execution_staging_bytes.to_le_bytes());
    digest.update(&envelope.safety_reserve_bytes.to_le_bytes());
    digest.update(&[prefetch_depth]);
    digest.update(
        &u32::try_from(layers.len())
            .map_err(|_| resource("sparse expert digest layer count overflows"))?
            .to_le_bytes(),
    );
    for layer in layers {
        digest.update(&layer.layer.to_le_bytes());
        digest.update(&layer.active_experts.to_le_bytes());
        digest.update(
            &u32::try_from(layer.experts.len())
                .map_err(|_| resource("sparse expert digest inventory count overflows"))?
                .to_le_bytes(),
        );
        for descriptor in &layer.experts {
            digest.update(&descriptor.id.layer.to_le_bytes());
            digest.update(&descriptor.id.expert.to_le_bytes());
            digest.update(&descriptor.storage_bytes.to_le_bytes());
            digest.update(&descriptor.execution_bytes.to_le_bytes());
            digest.update(descriptor.content_hash.as_bytes());
            let dtype = encode_dtype(descriptor.storage_dtype)
                .map_err(|_| invalid("sparse expert dtype has no canonical encoding"))?;
            digest.update(&[dtype, descriptor.quantization.id()]);
        }
    }
    Ok(digest.finalize())
}

fn digest_len_prefixed(
    digest: &mut PackageHashAccumulator,
    bytes: &[u8],
) -> Result<(), PlacementError> {
    let length = u64::try_from(bytes.len())
        .map_err(|_| resource("sparse expert digest field length overflows"))?;
    digest.update(&length.to_le_bytes());
    digest.update(bytes);
    Ok(())
}

/// Immutable engine-derived sparse-expert execution plan.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SparseExpertExecutionPlan {
    identity: Arc<SparseExpertPlanIdentity>,
    plan_digest: ModelPackageHash,
    layers: Vec<SparseExpertLayerPlan>,
    mode: SparseExpertResidencyMode,
    envelope: SparseExpertResourceEnvelope,
    prefetch_depth: u8,
    total_experts: u32,
    slot_table_bytes: u64,
    total_storage_bytes: u64,
    total_cache_bytes: u64,
    required_cache_bytes: u64,
    required_execution_staging_bytes: u64,
}

impl SparseExpertExecutionPlan {
    /// Derives a complete architecture-owned inventory from a verified base plan.
    pub fn derive(
        base_plan: &ExecutionPlacementPlan,
        layers: Vec<SparseExpertLayerPlan>,
        mode: SparseExpertResidencyMode,
        envelope: SparseExpertResourceEnvelope,
        prefetch_depth: u8,
    ) -> Result<Self, PlacementError> {
        Self::new_internal(
            SparseExpertPlanIdentity::from_execution_plan(base_plan),
            layers,
            mode,
            envelope,
            prefetch_depth,
        )
    }

    #[cfg(test)]
    fn new_synthetic(
        identity: SparseExpertPlanIdentity,
        layers: Vec<SparseExpertLayerPlan>,
        mode: SparseExpertResidencyMode,
        envelope: SparseExpertResourceEnvelope,
        prefetch_depth: u8,
    ) -> Result<Self, PlacementError> {
        Self::new_internal(identity, layers, mode, envelope, prefetch_depth)
    }

    fn new_internal(
        identity: SparseExpertPlanIdentity,
        mut layers: Vec<SparseExpertLayerPlan>,
        mode: SparseExpertResidencyMode,
        envelope: SparseExpertResourceEnvelope,
        prefetch_depth: u8,
    ) -> Result<Self, PlacementError> {
        let layer_count = u32::try_from(layers.len())
            .map_err(|_| resource("sparse expert layer count is not representable"))?;
        if layer_count == 0 || layer_count > MAX_SPARSE_EXPERT_LAYERS {
            return Err(invalid("sparse expert layer count is outside hard bounds"));
        }
        if prefetch_depth > MAX_EXPERT_PREFETCH_DEPTH
            || (mode == SparseExpertResidencyMode::FullyResident && prefetch_depth != 0)
        {
            return Err(invalid("sparse expert prefetch depth is invalid for the mode"));
        }
        let mut total_experts = 0_u32;
        let mut total_storage_bytes = 0_u64;
        let mut previous_layer = None;
        for layer in &mut layers {
            if previous_layer.is_some_and(|previous| previous >= layer.layer) {
                return Err(invalid("sparse expert layers are not strictly ordered"));
            }
            previous_layer = Some(layer.layer);
            if layer.experts.iter().any(|descriptor| {
                descriptor.quantization != Quantization::None
                    && descriptor.quantization != identity.quantization
            }) {
                return Err(invalid(
                    "sparse expert quantization does not match the verified package",
                ));
            }
            layer.slot_offset = total_experts;
            total_experts = total_experts
                .checked_add(
                    u32::try_from(layer.experts.len())
                        .map_err(|_| resource("sparse expert count overflows"))?,
                )
                .ok_or_else(|| resource("total sparse expert count overflows"))?;
            if total_experts > MAX_TOTAL_EXPERTS {
                return Err(invalid("total sparse expert count exceeds the hard bound"));
            }
            total_storage_bytes =
                layer.experts.iter().try_fold(total_storage_bytes, |total, descriptor| {
                    checked_add(total, descriptor.storage_bytes, "expert storage bytes overflow")
                })?;
        }

        let slot_table_bytes = u64::from(total_experts)
            .checked_mul(SPARSE_EXPERT_CACHE_SLOT_BYTES)
            .ok_or_else(|| resource("expert cache slot-table bytes overflow"))?;
        let total_cache_bytes = checked_add(
            slot_table_bytes,
            total_storage_bytes,
            "expert total cache bytes overflow",
        )?;

        let (streaming_cache, streaming_execution) = maximum_active_pair(&layers, prefetch_depth)?;
        let required_cache_bytes = match mode {
            SparseExpertResidencyMode::FullyResident => total_cache_bytes,
            SparseExpertResidencyMode::ExpertStreamed => checked_add(
                slot_table_bytes,
                streaming_cache,
                "streamed expert cache bytes overflow",
            )?,
        };
        if required_cache_bytes > envelope.cache_bytes {
            return Err(resource("expert cache ceiling cannot satisfy the selected mode"));
        }
        if streaming_execution > envelope.execution_staging_bytes {
            return Err(resource("expert staging ceiling cannot satisfy the selected mode"));
        }
        let plan_digest =
            sparse_expert_plan_digest(&identity, &layers, mode, envelope, prefetch_depth)?;
        Ok(Self {
            identity: Arc::new(identity),
            plan_digest,
            layers,
            mode,
            envelope,
            prefetch_depth,
            total_experts,
            slot_table_bytes,
            total_storage_bytes,
            total_cache_bytes,
            required_cache_bytes,
            required_execution_staging_bytes: streaming_execution,
        })
    }

    /// Exact model/package/backend/generation binding.
    pub fn identity(&self) -> &SparseExpertPlanIdentity {
        self.identity.as_ref()
    }

    /// Canonical SHA-384 binding of the complete sparse inventory and policy.
    pub const fn plan_digest(&self) -> ModelPackageHash {
        self.plan_digest
    }

    /// Canonical layer inventory.
    pub fn layers(&self) -> &[SparseExpertLayerPlan] {
        &self.layers
    }

    /// Selected expert residency behavior.
    pub const fn mode(&self) -> SparseExpertResidencyMode {
        self.mode
    }

    /// Hard memory envelope.
    pub const fn envelope(&self) -> SparseExpertResourceEnvelope {
        self.envelope
    }

    /// Bounded prefetch depth.
    pub const fn prefetch_depth(&self) -> u8 {
        self.prefetch_depth
    }

    /// Total expert descriptors.
    pub const fn total_experts(&self) -> u32 {
        self.total_experts
    }

    /// Total signed expert storage bytes.
    pub const fn total_storage_bytes(&self) -> u64 {
        self.total_storage_bytes
    }

    /// Total payload plus conservative live cache-map metadata bytes.
    pub const fn total_cache_bytes(&self) -> u64 {
        self.total_cache_bytes
    }

    /// Minimum cache required by the selected mode.
    pub const fn required_cache_bytes(&self) -> u64 {
        self.required_cache_bytes
    }

    /// Conservative selected-expert execution/staging requirement.
    pub const fn required_execution_staging_bytes(&self) -> u64 {
        self.required_execution_staging_bytes
    }

    fn layer(&self, layer: u32) -> Result<&SparseExpertLayerPlan, PlacementError> {
        let index = self
            .layers
            .binary_search_by_key(&layer, SparseExpertLayerPlan::layer)
            .map_err(|_| invalid("selected sparse expert layer is absent"))?;
        self.layers.get(index).ok_or_else(|| invalid("selected sparse expert layer is absent"))
    }

    fn slot_index(&self, id: SparseExpertId) -> Result<usize, PlacementError> {
        let layer = self.layer(id.layer)?;
        layer.descriptor(id)?;
        let flat = layer
            .slot_offset
            .checked_add(id.expert)
            .ok_or_else(|| resource("expert cache slot index overflows"))?;
        let index =
            usize::try_from(flat).map_err(|_| resource("expert cache slot is not addressable"))?;
        if index
            >= usize::try_from(self.total_experts)
                .map_err(|_| resource("expert count is not addressable"))?
        {
            return Err(invalid("expert cache slot exceeds the immutable plan"));
        }
        Ok(index)
    }

    const fn slot_table_bytes(&self) -> u64 {
        self.slot_table_bytes
    }
}

/// Exact router result for one layer. Scores never become cache authority.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SparseExpertSelection {
    identity: Arc<SparseExpertPlanIdentity>,
    plan_digest: ModelPackageHash,
    layer: u32,
    experts: Vec<SparseExpertId>,
}

impl SparseExpertSelection {
    /// Validates exact count, layer binding, uniqueness, and plan membership.
    pub fn new(
        plan: &SparseExpertExecutionPlan,
        layer: u32,
        experts: Vec<SparseExpertId>,
    ) -> Result<Self, PlacementError> {
        let layer_plan = plan.layer(layer)?;
        if experts.len() != usize::from(layer_plan.active_experts) {
            return Err(invalid("router selection has the wrong expert count"));
        }
        let mut previous = None;
        for id in &experts {
            layer_plan.descriptor(*id)?;
            if previous.is_some_and(|value| value >= *id) {
                return Err(invalid("router selection must be unique canonical expert order"));
            }
            previous = Some(*id);
        }
        Ok(Self { identity: plan.identity.clone(), plan_digest: plan.plan_digest, layer, experts })
    }

    /// Exact plan identity that produced this router selection.
    pub fn identity(&self) -> &SparseExpertPlanIdentity {
        self.identity.as_ref()
    }

    /// Canonical sparse-plan digest that produced this selection.
    pub const fn plan_digest(&self) -> ModelPackageHash {
        self.plan_digest
    }

    /// Layer owning the selection.
    pub const fn layer(&self) -> u32 {
        self.layer
    }

    /// Selected experts in deterministic expert-ID order.
    pub fn experts(&self) -> &[SparseExpertId] {
        &self.experts
    }
}

/// Pinned, path-free source for exact signed expert bytes.
///
/// Implementations may copy from an already verified mapping or descriptor.
/// They receive no caller path or byte offset. The cache independently checks
/// exact length, digest, identity, cancellation, and stale generation. This is
/// a trusted no-panic boundary: release builds retain workspace panic-abort.
pub trait ExpertByteSource: Sync {
    /// Exact verified artifact identity served by this source.
    fn identity(&self) -> &SparseExpertPlanIdentity;

    /// Fills the complete destination and returns the number of initialized bytes.
    fn read_exact_expert(
        &self,
        descriptor: &SparseExpertDescriptor,
        destination: &mut [u8],
        cancellation: &dyn PlacementCancellationProbe,
    ) -> Result<usize, PlacementError>;
}

#[derive(Debug)]
struct ExpertCacheEntry {
    payload: Arc<HostBuffer>,
    storage_bytes: u64,
    last_access: u64,
    pins: u32,
    cache_generation: u64,
}

#[derive(Debug)]
struct ExpertCacheState {
    valid: bool,
    resident_ready: bool,
    populating: bool,
    cache_generation: u64,
    access_clock: u64,
    resident_bytes: u64,
    peak_resident_bytes: u64,
    entry_count: u32,
    staging_allocation_count: u32,
    slots: Vec<Option<ExpertCacheEntry>>,
    slot_reservation: Option<MemoryReservation>,
    cache_capacity_guard: Option<MemoryReservation>,
    execution_staging_guard: Option<MemoryReservation>,
    safety_guard: Option<MemoryReservation>,
    metrics: SparseExpertCacheMetrics,
}

#[derive(Debug)]
struct ExpertCacheInner {
    plan: Arc<SparseExpertExecutionPlan>,
    pool: MemoryPool,
    staging_pool: MemoryPool,
    state: Mutex<ExpertCacheState>,
    load_active: Mutex<bool>,
    load_in_flight: AtomicBool,
    load_ready: Condvar,
    prefetch_active: Mutex<bool>,
    prefetch_in_flight: AtomicBool,
    prefetch_ready: Condvar,
    lifecycle_cancellation: PlacementCancellation,
    #[cfg(test)]
    fail_prefetch_spawn: std::sync::atomic::AtomicBool,
    #[cfg(test)]
    load_token_hook: Mutex<Option<TokenAcquisitionHook>>,
    #[cfg(test)]
    prefetch_token_hook: Mutex<Option<TokenAcquisitionHook>>,
}

#[cfg(test)]
#[derive(Debug)]
struct TokenAcquisitionHook {
    entered: Arc<std::sync::Barrier>,
    release: Arc<std::sync::Barrier>,
}

/// Bounded immutable expert cache. One serialized miss prevents duplicate I/O.
#[derive(Clone, Debug)]
pub struct SparseExpertCache {
    inner: Arc<ExpertCacheInner>,
}

impl SparseExpertCache {
    /// Creates an empty cache bound to one immutable plan and aggregate pool.
    pub fn new(
        plan: Arc<SparseExpertExecutionPlan>,
        pool: &MemoryPool,
    ) -> Result<Self, PlacementError> {
        if plan.envelope.total_bytes > pool.limit_bytes() {
            return Err(resource("expert plan exceeds the aggregate memory pool"));
        }
        let total_experts = usize::try_from(plan.total_experts)
            .map_err(|_| resource("expert cache slot count is not addressable"))?;
        let staging_owner = MemoryOwner::SparseExpertStaging(plan.identity.model_id.clone());
        let execution_staging_guard = pool
            .reserve_for(plan.envelope.execution_staging_bytes, staging_owner.clone())
            .map_err(classify_engine_error)?;
        let staging_pool =
            MemoryPool::new_scoped(plan.envelope.execution_staging_bytes, staging_owner.clone())
                .map_err(classify_engine_error)?;
        let safety_guard = if plan.envelope.safety_reserve_bytes == 0 {
            None
        } else {
            Some(
                pool.reserve_for(plan.envelope.safety_reserve_bytes, staging_owner)
                    .map_err(classify_engine_error)?,
            )
        };
        let cache_owner = MemoryOwner::SparseExpertCache(plan.identity.model_id.clone());
        let cache_capacity_guard = pool
            .reserve_for(plan.envelope.cache_bytes, cache_owner.clone())
            .map_err(classify_engine_error)?;
        let cache_pool = MemoryPool::new_scoped(plan.envelope.cache_bytes, cache_owner.clone())
            .map_err(classify_engine_error)?;
        let slot_reservation = cache_pool
            .reserve_for(plan.slot_table_bytes(), cache_owner)
            .map_err(classify_engine_error)?;
        let mut slots = Vec::new();
        slots
            .try_reserve_exact(total_experts)
            .map_err(|_| resource("expert cache slot-table allocation failed"))?;
        let actual_slot_bytes = slots
            .capacity()
            .checked_mul(size_of::<Option<ExpertCacheEntry>>())
            .and_then(|bytes| u64::try_from(bytes).ok())
            .ok_or_else(|| resource("expert cache slot-table allocation overflows"))?;
        if actual_slot_bytes > plan.slot_table_bytes() {
            return Err(resource("expert cache slot-table allocation exceeds its reservation"));
        }
        slots.resize_with(total_experts, || None);
        let resident_ready = plan.mode != SparseExpertResidencyMode::FullyResident;
        let slot_table_bytes = plan.slot_table_bytes();
        Ok(Self {
            inner: Arc::new(ExpertCacheInner {
                plan,
                pool: cache_pool,
                staging_pool,
                state: Mutex::new(ExpertCacheState {
                    valid: true,
                    resident_ready,
                    cache_generation: 1,
                    access_clock: 0,
                    resident_bytes: slot_table_bytes,
                    peak_resident_bytes: slot_table_bytes,
                    entry_count: 0,
                    staging_allocation_count: 0,
                    slots,
                    slot_reservation: Some(slot_reservation),
                    cache_capacity_guard: Some(cache_capacity_guard),
                    execution_staging_guard: Some(execution_staging_guard),
                    safety_guard,
                    metrics: SparseExpertCacheMetrics::default(),
                    populating: false,
                }),
                load_active: Mutex::new(false),
                load_in_flight: AtomicBool::new(false),
                load_ready: Condvar::new(),
                prefetch_active: Mutex::new(false),
                prefetch_in_flight: AtomicBool::new(false),
                prefetch_ready: Condvar::new(),
                lifecycle_cancellation: PlacementCancellation::new(),
                #[cfg(test)]
                fail_prefetch_spawn: std::sync::atomic::AtomicBool::new(false),
                #[cfg(test)]
                load_token_hook: Mutex::new(None),
                #[cfg(test)]
                prefetch_token_hook: Mutex::new(None),
            }),
        })
    }

    /// Exact immutable plan.
    pub fn plan(&self) -> &SparseExpertExecutionPlan {
        &self.inner.plan
    }

    /// Returns one immutable lease, loading and verifying a miss exactly once.
    pub fn acquire(
        &self,
        id: SparseExpertId,
        source: &dyn ExpertByteSource,
        cancellation: &dyn PlacementCancellationProbe,
    ) -> Result<SparseExpertLease, PlacementError> {
        check_cancelled(cancellation)?;
        self.verify_source(source)?;
        self.acquire_internal(id, source, cancellation, false).map(|(lease, _loaded)| lease)
    }

    fn acquire_internal(
        &self,
        id: SparseExpertId,
        source: &dyn ExpertByteSource,
        cancellation: &dyn PlacementCancellationProbe,
        allow_population: bool,
    ) -> Result<(SparseExpertLease, bool), PlacementError> {
        let combined_cancellation = CombinedCancellation {
            first: cancellation,
            second: &self.inner.lifecycle_cancellation,
        };
        check_cancelled(&combined_cancellation)?;
        self.record_lookup(allow_population)?;
        if let Some(lease) = self.acquire_existing(id, true, allow_population)? {
            return Ok((lease, false));
        }
        self.record_miss(allow_population)?;

        let _load_token = self.acquire_load_token(&combined_cancellation)?;
        check_cancelled(&combined_cancellation)?;
        self.verify_source(source)?;
        if let Some(lease) = self.acquire_existing(id, false, allow_population)? {
            return Ok((lease, false));
        }

        let descriptor = self.descriptor(id)?.clone();
        let load_generation = self.prepare_capacity(descriptor.storage_bytes)?;
        let length = usize::try_from(descriptor.storage_bytes)
            .map_err(|_| resource("expert storage length is not addressable"))?;
        let mut payload = HostBuffer::allocate_preserving(
            &self.inner.pool,
            length,
            MemoryOwner::SparseExpertCache(self.inner.plan.identity.model_id.clone()),
            0,
        )
        .map_err(classify_engine_error)?;
        let read = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            source.read_exact_expert(&descriptor, payload.as_mut_slice(), &combined_cancellation)
        }))
        .unwrap_or_else(|_panic| {
            Err(PlacementError::BackendFailure(EngineError::backend_failure(
                "expert source panicked",
            )))
        })?;
        if read != length {
            return Err(PlacementError::ArtifactIdentityMismatch);
        }
        if hash_expert_payload(payload.as_slice(), &combined_cancellation)?
            != descriptor.content_hash
        {
            return Err(PlacementError::ArtifactIdentityMismatch);
        }
        check_cancelled(&combined_cancellation)?;
        self.insert_loaded(id, load_generation, Arc::new(payload)).map(|lease| (lease, true))
    }

    /// Acquires a complete canonical selection. Partial failure releases leases.
    pub fn acquire_selection(
        &self,
        selection: &SparseExpertSelection,
        source: &dyn ExpertByteSource,
        cancellation: &dyn PlacementCancellationProbe,
    ) -> Result<Vec<SparseExpertLease>, PlacementError> {
        self.acquire_selection_transaction(selection, source, cancellation)
            .map(|(leases, _newly_loaded)| leases)
    }

    fn acquire_selection_transaction(
        &self,
        selection: &SparseExpertSelection,
        source: &dyn ExpertByteSource,
        cancellation: &dyn PlacementCancellationProbe,
    ) -> Result<(Vec<SparseExpertLease>, Vec<SparseExpertId>), PlacementError> {
        check_cancelled(cancellation)?;
        self.verify_source(source)?;
        let layer = self.inner.plan.layer(selection.layer)?;
        if selection.identity != self.inner.plan.identity
            || selection.plan_digest != self.inner.plan.plan_digest
        {
            return Err(PlacementError::ArtifactIdentityMismatch);
        }
        if selection.experts.len() != usize::from(layer.active_experts) {
            return Err(invalid("expert selection no longer matches the immutable plan"));
        }
        let mut leases = Vec::new();
        leases
            .try_reserve_exact(selection.experts.len())
            .map_err(|_| resource("expert lease table allocation failed"))?;
        let mut newly_loaded = Vec::new();
        newly_loaded
            .try_reserve_exact(selection.experts.len())
            .map_err(|_| resource("expert rollback table allocation failed"))?;
        for id in &selection.experts {
            match self.acquire_internal(*id, source, cancellation, false) {
                Ok((lease, loaded)) => {
                    if loaded {
                        newly_loaded.push(*id);
                    }
                    leases.push(lease);
                }
                Err(error) => {
                    drop(leases);
                    self.rollback_new_entries(&newly_loaded)?;
                    return Err(error);
                }
            }
        }
        Ok((leases, newly_loaded))
    }

    /// Loads every expert for a fully resident plan without retaining leases.
    pub fn populate_fully_resident(
        &self,
        source: &dyn ExpertByteSource,
        cancellation: &dyn PlacementCancellationProbe,
    ) -> Result<(), PlacementError> {
        check_cancelled(cancellation)?;
        self.verify_source(source)?;
        if self.inner.plan.mode != SparseExpertResidencyMode::FullyResident {
            return Err(invalid("full expert population requires a fully resident plan"));
        }
        {
            let mut state = self.lock_state()?;
            ensure_valid(&state)?;
            if state.resident_ready {
                return Ok(());
            }
            if state.populating {
                return Err(invalid("fully resident expert population is already active"));
            }
            state.populating = true;
        }
        let result = (|| {
            for layer in &self.inner.plan.layers {
                for descriptor in &layer.experts {
                    drop(self.acquire_internal(descriptor.id, source, cancellation, true)?.0);
                }
            }
            let mut state = self.lock_state()?;
            ensure_valid(&state)?;
            if state.entry_count != self.inner.plan.total_experts
                || state.resident_bytes != self.inner.plan.total_cache_bytes
            {
                return Err(invalid("fully resident expert population is incomplete"));
            }
            state.populating = false;
            state.resident_ready = true;
            Ok(())
        })();
        if result.is_err() {
            self.invalidate()?;
        }
        result
    }

    /// Permanently invalidates this cache instance and rejects stale completions.
    pub fn invalidate(&self) -> Result<(), PlacementError> {
        let slot_count = {
            let mut state = self.lock_state()?;
            if state.valid {
                let cache_generation = state
                    .cache_generation
                    .checked_add(1)
                    .ok_or_else(|| resource("expert cache generation overflows"))?;
                let invalidations = checked_counter(state.metrics.invalidations)?;
                state.valid = false;
                state.resident_ready = false;
                state.populating = false;
                state.cache_generation = cache_generation;
                state.metrics.invalidations = invalidations;
            }
            state.slots.len()
        };
        self.inner.lifecycle_cancellation.cancel();
        self.inner.load_ready.notify_all();
        self.inner.prefetch_ready.notify_all();

        // Release one payload at a time outside the state lock. This path is
        // allocation-free, remains retryable after a repeated invalidation,
        // and is safe when memory pressure caused the terminal transition.
        for slot in 0..slot_count {
            let (removed, released_cache) = {
                let mut state = self.lock_state()?;
                let removed = take_unpinned_at(&self.inner.plan, &mut state, slot)?;
                let released_cache = maybe_release_invalid_cache(
                    &mut state,
                    self.inner.load_in_flight.load(Ordering::Acquire)
                        || self.inner.prefetch_in_flight.load(Ordering::Acquire),
                );
                (removed, released_cache)
            };
            drop(removed);
            drop(released_cache);
        }
        let released_cache = {
            let mut state = self.lock_state()?;
            maybe_release_invalid_cache(
                &mut state,
                self.inner.load_in_flight.load(Ordering::Acquire)
                    || self.inner.prefetch_in_flight.load(Ordering::Acquire),
            )
        };
        drop(released_cache);
        Ok(())
    }

    /// Content-free cache metrics.
    pub fn metrics(&self) -> Result<SparseExpertCacheMetrics, PlacementError> {
        let (mut metrics, resident_bytes, peak_resident_bytes, entry_count, staging_count) = {
            let state = self.lock_state()?;
            (
                state.metrics,
                state.resident_bytes,
                state.peak_resident_bytes,
                state.entry_count,
                state.staging_allocation_count,
            )
        };
        let staging = self.inner.staging_pool.statistics().map_err(classify_engine_error)?;
        metrics.resident_bytes = resident_bytes;
        metrics.peak_resident_bytes = peak_resident_bytes;
        metrics.entry_count = entry_count;
        metrics.staging_bytes = staging.reserved_bytes();
        metrics.peak_staging_bytes = staging.peak_bytes();
        metrics.staging_allocation_count = staging_count;
        Ok(metrics)
    }

    fn verify_source(&self, source: &dyn ExpertByteSource) -> Result<(), PlacementError> {
        if source.identity() != self.inner.plan.identity.as_ref() {
            return Err(PlacementError::ArtifactIdentityMismatch);
        }
        Ok(())
    }

    fn descriptor(&self, id: SparseExpertId) -> Result<&SparseExpertDescriptor, PlacementError> {
        self.inner.plan.layer(id.layer)?.descriptor(id)
    }

    fn record_lookup(&self, allow_population: bool) -> Result<(), PlacementError> {
        let mut state = self.lock_state()?;
        ensure_usable(&state, allow_population)?;
        state.metrics.lookups = checked_counter(state.metrics.lookups)?;
        Ok(())
    }

    fn record_miss(&self, allow_population: bool) -> Result<(), PlacementError> {
        let mut state = self.lock_state()?;
        ensure_usable(&state, allow_population)?;
        state.metrics.misses = checked_counter(state.metrics.misses)?;
        Ok(())
    }

    fn acquire_existing(
        &self,
        id: SparseExpertId,
        count_hit: bool,
        allow_population: bool,
    ) -> Result<Option<SparseExpertLease>, PlacementError> {
        let mut state = self.lock_state()?;
        ensure_usable(&state, allow_population)?;
        let access = checked_counter(state.access_clock)?;
        let next_hits = if count_hit { Some(checked_counter(state.metrics.hits)?) } else { None };
        let cache_generation = state.cache_generation;
        let slot = self.inner.plan.slot_index(id)?;
        let Some(entry) = state
            .slots
            .get_mut(slot)
            .ok_or_else(|| invalid("expert cache slot is absent"))?
            .as_mut()
        else {
            return Ok(None);
        };
        if entry.cache_generation != cache_generation {
            return Err(PlacementError::ArtifactIdentityMismatch);
        }
        let pins =
            entry.pins.checked_add(1).ok_or_else(|| resource("expert lease count overflows"))?;
        entry.pins = pins;
        entry.last_access = access;
        let payload = Arc::clone(&entry.payload);
        state.access_clock = access;
        if let Some(next_hits) = next_hits {
            state.metrics.hits = next_hits;
        }
        Ok(Some(SparseExpertLease {
            inner: Arc::clone(&self.inner),
            id,
            cache_generation,
            storage_dtype: self.descriptor(id)?.storage_dtype,
            payload,
        }))
    }

    fn prepare_capacity(&self, bytes: u64) -> Result<u64, PlacementError> {
        if bytes > self.inner.plan.envelope.cache_bytes {
            return Err(resource("one expert exceeds the cache ceiling"));
        }
        let removed = {
            let mut state = self.lock_state()?;
            ensure_valid(&state)?;
            let generation = state.cache_generation;
            let target = checked_add(state.resident_bytes, bytes, "expert cache bytes overflow")?;
            let mut removed = Vec::new();
            if target > self.inner.plan.envelope.cache_bytes {
                removed = evict_until(
                    &self.inner.plan,
                    &mut state,
                    self.inner.plan.envelope.cache_bytes - bytes,
                )?;
            }
            (generation, removed)
        };
        drop(removed.1);
        Ok(removed.0)
    }

    fn insert_loaded(
        &self,
        id: SparseExpertId,
        load_generation: u64,
        payload: Arc<HostBuffer>,
    ) -> Result<SparseExpertLease, PlacementError> {
        let payload_bytes = u64::try_from(payload.as_slice().len())
            .map_err(|_| resource("expert payload length overflows"))?;
        let mut state = self.lock_state()?;
        ensure_valid(&state)?;
        if state.cache_generation != load_generation {
            return Err(PlacementError::ArtifactIdentityMismatch);
        }
        let slot = self.inner.plan.slot_index(id)?;
        if state.slots.get(slot).ok_or_else(|| invalid("expert cache slot is absent"))?.is_some() {
            return Err(invalid("serialized expert load produced a duplicate entry"));
        }
        let resident_bytes =
            checked_add(state.resident_bytes, payload_bytes, "expert cache bytes overflow")?;
        if resident_bytes > self.inner.plan.envelope.cache_bytes {
            return Err(resource("expert cache insertion exceeds its ceiling"));
        }
        let access = checked_counter(state.access_clock)?;
        let entry_count = state
            .entry_count
            .checked_add(1)
            .ok_or_else(|| resource("expert cache entry count overflows"))?;
        let loads = checked_counter(state.metrics.loads)?;
        let bytes_loaded = checked_add(
            state.metrics.bytes_loaded,
            payload_bytes,
            "expert loaded-byte metric overflows",
        )?;
        let entry_slot =
            state.slots.get_mut(slot).ok_or_else(|| invalid("expert cache slot is absent"))?;
        *entry_slot = Some(ExpertCacheEntry {
            payload: Arc::clone(&payload),
            storage_bytes: payload_bytes,
            last_access: access,
            pins: 1,
            cache_generation: load_generation,
        });
        state.access_clock = access;
        state.entry_count = entry_count;
        state.resident_bytes = resident_bytes;
        state.peak_resident_bytes = state.peak_resident_bytes.max(resident_bytes);
        state.metrics.loads = loads;
        state.metrics.bytes_loaded = bytes_loaded;
        Ok(SparseExpertLease {
            inner: Arc::clone(&self.inner),
            id,
            cache_generation: load_generation,
            storage_dtype: self.descriptor(id)?.storage_dtype,
            payload,
        })
    }

    fn rollback_new_entries(&self, ids: &[SparseExpertId]) -> Result<(), PlacementError> {
        for id in ids {
            let (removed, released_cache) = {
                let mut state = self.lock_state()?;
                let slot = self.inner.plan.slot_index(*id)?;
                let removed = match state.slots.get_mut(slot) {
                    Some(entry_slot)
                        if entry_slot.as_ref().is_some_and(|entry| entry.pins == 0) =>
                    {
                        let entry = entry_slot
                            .take()
                            .ok_or_else(|| invalid("expert rollback entry disappeared"))?;
                        state.resident_bytes = state
                            .resident_bytes
                            .checked_sub(entry.storage_bytes)
                            .ok_or_else(|| invalid("expert rollback accounting underflows"))?;
                        state.entry_count = state
                            .entry_count
                            .checked_sub(1)
                            .ok_or_else(|| invalid("expert rollback entry count underflows"))?;
                        Some(entry)
                    }
                    Some(_) | None => None,
                };
                let released_cache = maybe_release_invalid_cache(
                    &mut state,
                    self.inner.load_in_flight.load(Ordering::Acquire)
                        || self.inner.prefetch_in_flight.load(Ordering::Acquire),
                );
                (removed, released_cache)
            };
            drop(removed);
            drop(released_cache);
        }
        Ok(())
    }

    fn lock_state(&self) -> Result<std::sync::MutexGuard<'_, ExpertCacheState>, PlacementError> {
        self.inner.state.lock().map_err(|_| invalid("expert cache state lock is poisoned"))
    }

    fn acquire_load_token(
        &self,
        cancellation: &dyn PlacementCancellationProbe,
    ) -> Result<ExpertLoadToken, PlacementError> {
        let mut active = self
            .inner
            .load_active
            .lock()
            .map_err(|_| invalid("expert cache load state is poisoned"))?;
        while *active {
            {
                let state = self.lock_state()?;
                ensure_valid(&state)?;
            }
            check_cancelled(cancellation)?;
            let (next, _timeout) = self
                .inner
                .load_ready
                .wait_timeout(active, Duration::from_millis(10))
                .map_err(|_| invalid("expert cache load wait is poisoned"))?;
            active = next;
        }
        {
            let state = self.lock_state()?;
            ensure_valid(&state)?;
        }
        check_cancelled(cancellation)?;
        *active = true;
        self.inner.load_in_flight.store(true, Ordering::Release);
        let token = ExpertLoadToken { inner: Arc::clone(&self.inner) };
        drop(active);
        #[cfg(test)]
        wait_for_token_acquisition_hook(&self.inner.load_token_hook);
        {
            let state = self.lock_state()?;
            ensure_valid(&state)?;
        }
        Ok(token)
    }

    fn acquire_prefetch_token(
        &self,
        cancellation: &dyn PlacementCancellationProbe,
    ) -> Result<ExpertPrefetchToken, PlacementError> {
        let mut active = self
            .inner
            .prefetch_active
            .lock()
            .map_err(|_| invalid("expert prefetch state is poisoned"))?;
        while *active {
            {
                let state = self.lock_state()?;
                ensure_valid(&state)?;
            }
            check_cancelled(cancellation)?;
            let (next, _timeout) = self
                .inner
                .prefetch_ready
                .wait_timeout(active, Duration::from_millis(10))
                .map_err(|_| invalid("expert prefetch wait is poisoned"))?;
            active = next;
        }
        {
            let state = self.lock_state()?;
            ensure_valid(&state)?;
        }
        check_cancelled(cancellation)?;
        *active = true;
        self.inner.prefetch_in_flight.store(true, Ordering::Release);
        let token = ExpertPrefetchToken { inner: Arc::clone(&self.inner) };
        drop(active);
        #[cfg(test)]
        wait_for_token_acquisition_hook(&self.inner.prefetch_token_hook);
        {
            let state = self.lock_state()?;
            ensure_valid(&state)?;
        }
        Ok(token)
    }
}

#[cfg(test)]
fn wait_for_token_acquisition_hook(hook: &Mutex<Option<TokenAcquisitionHook>>) {
    let hook = hook.lock().unwrap_or_else(std::sync::PoisonError::into_inner).take();
    if let Some(hook) = hook {
        hook.entered.wait();
        hook.release.wait();
    }
}

struct ExpertLoadToken {
    inner: Arc<ExpertCacheInner>,
}

impl Drop for ExpertLoadToken {
    fn drop(&mut self) {
        let mut active = match self.inner.load_active.lock() {
            Ok(active) => active,
            Err(poisoned) => poisoned.into_inner(),
        };
        *active = false;
        self.inner.load_in_flight.store(false, Ordering::Release);
        drop(active);
        self.inner.load_ready.notify_all();
        let released_cache = match self.inner.state.lock() {
            Ok(mut state) => maybe_release_invalid_cache(
                &mut state,
                self.inner.prefetch_in_flight.load(Ordering::Acquire),
            ),
            Err(_poisoned) => None,
        };
        drop(released_cache);
    }
}

struct ExpertPrefetchToken {
    inner: Arc<ExpertCacheInner>,
}

impl Drop for ExpertPrefetchToken {
    fn drop(&mut self) {
        let mut active = match self.inner.prefetch_active.lock() {
            Ok(active) => active,
            Err(poisoned) => poisoned.into_inner(),
        };
        *active = false;
        self.inner.prefetch_in_flight.store(false, Ordering::Release);
        drop(active);
        self.inner.prefetch_ready.notify_all();
        let released_cache = match self.inner.state.lock() {
            Ok(mut state) => maybe_release_invalid_cache(
                &mut state,
                self.inner.load_in_flight.load(Ordering::Acquire),
            ),
            Err(_poisoned) => None,
        };
        drop(released_cache);
    }
}

/// Immutable pinned expert payload. Drop releases only this request's pin.
#[derive(Debug)]
pub struct SparseExpertLease {
    inner: Arc<ExpertCacheInner>,
    id: SparseExpertId,
    cache_generation: u64,
    storage_dtype: DType,
    payload: Arc<HostBuffer>,
}

impl SparseExpertLease {
    /// Exact expert coordinate.
    pub const fn id(&self) -> SparseExpertId {
        self.id
    }

    /// Exact model/package/backend/plan identity owning this payload.
    pub fn identity(&self) -> &SparseExpertPlanIdentity {
        self.inner.plan.identity.as_ref()
    }

    /// Canonical sparse-plan digest owning this payload.
    pub fn plan_digest(&self) -> ModelPackageHash {
        self.inner.plan.plan_digest
    }

    /// Exact signed storage data type for this expert payload.
    pub const fn storage_dtype(&self) -> DType {
        self.storage_dtype
    }

    /// Immutable verified bytes. No mutable cache view exists.
    pub fn bytes(&self) -> &[u8] {
        self.payload.as_slice()
    }

    /// Returns whether cache invalidation/unload cancelled this execution lease.
    pub fn is_cancelled(&self) -> bool {
        self.inner.lifecycle_cancellation.is_cancelled()
    }

    /// Returns whether another lease belongs to this exact cache instance.
    pub fn shares_cache_instance_with(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }

    /// Allocates initialized scratch from this cache's admitted staging quota.
    pub fn allocate_execution_staging(
        &self,
        bytes: usize,
    ) -> Result<SparseExpertStagingBuffer, PlacementError> {
        if bytes == 0 {
            return Err(invalid("sparse expert staging length must be positive"));
        }
        {
            let mut state = self
                .inner
                .state
                .lock()
                .map_err(|_| invalid("expert cache state lock is poisoned"))?;
            ensure_valid(&state)?;
            if state.cache_generation != self.cache_generation {
                return Err(PlacementError::ArtifactIdentityMismatch);
            }
            let count = state
                .staging_allocation_count
                .checked_add(1)
                .ok_or_else(|| resource("expert staging allocation count overflows"))?;
            state.staging_allocation_count = count;
        }
        let pending = PendingSparseExpertStaging { inner: Arc::clone(&self.inner), armed: true };
        let buffer = HostBuffer::allocate_preserving(
            &self.inner.staging_pool,
            bytes,
            MemoryOwner::SparseExpertStaging(self.inner.plan.identity.model_id.clone()),
            0,
        )
        .map_err(classify_engine_error)?;
        {
            let state = self
                .inner
                .state
                .lock()
                .map_err(|_| invalid("expert cache state lock is poisoned"))?;
            ensure_valid(&state)?;
            if state.cache_generation != self.cache_generation {
                return Err(PlacementError::ArtifactIdentityMismatch);
            }
        }
        pending.commit();
        Ok(SparseExpertStagingBuffer { inner: Arc::clone(&self.inner), buffer: Some(buffer) })
    }
}

struct PendingSparseExpertStaging {
    inner: Arc<ExpertCacheInner>,
    armed: bool,
}

impl PendingSparseExpertStaging {
    fn commit(mut self) {
        self.armed = false;
    }
}

impl Drop for PendingSparseExpertStaging {
    fn drop(&mut self) {
        if self.armed {
            drop_one_staging_allocation(&self.inner);
        }
    }
}

/// Initialized execution scratch backed by the cache's admitted staging quota.
#[derive(Debug)]
pub struct SparseExpertStagingBuffer {
    inner: Arc<ExpertCacheInner>,
    buffer: Option<HostBuffer>,
}

impl SparseExpertStagingBuffer {
    /// Immutable initialized scratch bytes.
    pub fn as_slice(&self) -> &[u8] {
        self.buffer.as_ref().map_or(&[], HostBuffer::as_slice)
    }

    /// Mutable initialized scratch bytes for trusted execution code.
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        self.buffer.as_mut().map_or(&mut [], HostBuffer::as_mut_slice)
    }

    /// Accounted staging bytes.
    pub fn reserved_bytes(&self) -> u64 {
        self.buffer.as_ref().map_or(0, HostBuffer::reserved_bytes)
    }
}

impl Drop for SparseExpertStagingBuffer {
    fn drop(&mut self) {
        drop(self.buffer.take());
        drop_one_staging_allocation(&self.inner);
    }
}

fn drop_one_staging_allocation(inner: &ExpertCacheInner) {
    let released_cache = match inner.state.lock() {
        Ok(mut state) => {
            if state.staging_allocation_count == 0 {
                return;
            }
            state.staging_allocation_count -= 1;
            maybe_release_invalid_cache(
                &mut state,
                inner.load_in_flight.load(Ordering::Acquire)
                    || inner.prefetch_in_flight.load(Ordering::Acquire),
            )
        }
        Err(_poisoned) => None,
    };
    drop(released_cache);
}

impl Drop for SparseExpertLease {
    fn drop(&mut self) {
        let (removed, released_cache) = {
            let Ok(mut state) = self.inner.state.lock() else {
                return;
            };
            let mut remove = false;
            let cache_valid = state.valid;
            let Ok(slot) = self.inner.plan.slot_index(self.id) else {
                return;
            };
            if let Some(entry) = state.slots.get_mut(slot).and_then(Option::as_mut) {
                if entry.cache_generation == self.cache_generation && entry.pins != 0 {
                    entry.pins -= 1;
                    remove = !cache_valid && entry.pins == 0;
                }
            }
            if remove {
                let entry = state.slots.get_mut(slot).and_then(Option::take);
                if let Some(entry) = &entry {
                    let Some(resident_bytes) =
                        state.resident_bytes.checked_sub(entry.storage_bytes)
                    else {
                        return;
                    };
                    state.resident_bytes = resident_bytes;
                    let Some(entry_count) = state.entry_count.checked_sub(1) else {
                        return;
                    };
                    state.entry_count = entry_count;
                }
                let released_cache = maybe_release_invalid_cache(
                    &mut state,
                    self.inner.load_in_flight.load(Ordering::Acquire)
                        || self.inner.prefetch_in_flight.load(Ordering::Acquire),
                );
                (entry, released_cache)
            } else {
                (None, None)
            }
        };
        drop(removed);
        drop(released_cache);
    }
}

/// Content-free cache counters. No model bytes, paths, prompts, or outputs exist here.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SparseExpertCacheMetrics {
    /// Cache acquire calls.
    pub lookups: u64,
    /// Entries already present at first lookup.
    pub hits: u64,
    /// Entries absent at first lookup.
    pub misses: u64,
    /// Successfully verified cache insertions.
    pub loads: u64,
    /// Deterministic unpinned evictions.
    pub evictions: u64,
    /// Cache invalidation operations.
    pub invalidations: u64,
    /// Exact verified bytes loaded since creation.
    pub bytes_loaded: u64,
    /// Current immutable resident bytes.
    pub resident_bytes: u64,
    /// Peak immutable resident bytes.
    pub peak_resident_bytes: u64,
    /// Current entry count.
    pub entry_count: u32,
    /// Current initialized execution scratch bytes.
    pub staging_bytes: u64,
    /// Peak initialized execution scratch bytes.
    pub peak_staging_bytes: u64,
    /// Current staging allocation count.
    pub staging_allocation_count: u32,
}

/// Result of one bounded overlap between trusted work and one expert prefetch.
#[derive(Debug)]
pub struct SparseExpertPrefetchResult<T> {
    /// Result returned by the caller's independent work.
    pub work: T,
    /// Verified immutable selected-expert leases.
    pub leases: Vec<SparseExpertLease>,
}

struct CombinedCancellation<'a> {
    first: &'a dyn PlacementCancellationProbe,
    second: &'a dyn PlacementCancellationProbe,
}

fn hash_expert_payload(
    payload: &[u8],
    cancellation: &dyn PlacementCancellationProbe,
) -> Result<ModelPackageHash, PlacementError> {
    let mut digest = PackageHashAccumulator::new();
    for chunk in payload.chunks(SPARSE_EXPERT_HASH_CHUNK_BYTES) {
        check_cancelled(cancellation)?;
        digest.update(chunk);
    }
    check_cancelled(cancellation)?;
    Ok(digest.finalize())
}

struct PrefetchPermitGate {
    active: Mutex<usize>,
    ready: Condvar,
    limit: usize,
}

impl PrefetchPermitGate {
    const fn new(limit: usize) -> Self {
        Self { active: Mutex::new(0), ready: Condvar::new(), limit }
    }

    fn acquire(
        &self,
        cancellation: &dyn PlacementCancellationProbe,
    ) -> Result<PrefetchWorkerPermit<'_>, PlacementError> {
        let mut active = self
            .active
            .lock()
            .map_err(|_| invalid("global expert prefetch permit state is poisoned"))?;
        while *active >= self.limit {
            check_cancelled(cancellation)?;
            let (next, _timeout) = self
                .ready
                .wait_timeout(active, Duration::from_millis(10))
                .map_err(|_| invalid("global expert prefetch permit wait is poisoned"))?;
            active = next;
        }
        check_cancelled(cancellation)?;
        *active = active
            .checked_add(1)
            .ok_or_else(|| resource("global expert prefetch permit count overflows"))?;
        Ok(PrefetchWorkerPermit { gate: self })
    }
}

struct PrefetchWorkerPermit<'a> {
    gate: &'a PrefetchPermitGate,
}

impl Drop for PrefetchWorkerPermit<'_> {
    fn drop(&mut self) {
        let mut active = match self.gate.active.lock() {
            Ok(active) => active,
            Err(poisoned) => poisoned.into_inner(),
        };
        if let Some(next) = active.checked_sub(1) {
            *active = next;
        }
        drop(active);
        self.gate.ready.notify_one();
    }
}

static GLOBAL_EXPERT_PREFETCH_GATE: PrefetchPermitGate =
    PrefetchPermitGate::new(MAX_GLOBAL_EXPERT_PREFETCH_WORKERS);

impl PlacementCancellationProbe for CombinedCancellation<'_> {
    fn is_cancelled(&self) -> bool {
        self.first.is_cancelled() || self.second.is_cancelled()
    }
}

/// Runs exactly one scoped prefetch worker while trusted independent work executes.
///
/// The worker is always joined. One process-wide permit and one per-cache permit
/// bound concurrency, and the worker uses an explicit small stack. The work
/// closure receives the combined request, lifecycle, and dependent-failure
/// cancellation probe. It must remain transactional: callers commit KV or
/// activation side effects only after this function returns success. Failure
/// and cancellation release every partial lease before returning, so no task or
/// buffer can outlive cache ownership. Source and work callbacks are trusted
/// no-panic boundaries; `catch_unwind` converts failures only in builds whose
/// panic strategy actually unwinds, while release retains workspace panic-abort.
pub fn run_bounded_expert_prefetch<T: Send>(
    cache: &SparseExpertCache,
    selection: &SparseExpertSelection,
    source: &dyn ExpertByteSource,
    cancellation: &dyn PlacementCancellationProbe,
    work: impl FnOnce(&dyn PlacementCancellationProbe) -> Result<T, EngineError>,
) -> Result<SparseExpertPrefetchResult<T>, PlacementError> {
    if cache.plan().prefetch_depth == 0 {
        return Err(invalid("expert prefetch is disabled by the immutable plan"));
    }
    check_cancelled(cancellation)?;
    let _prefetch_token = cache.acquire_prefetch_token(cancellation)?;
    let lifecycle_cancellation =
        CombinedCancellation { first: cancellation, second: &cache.inner.lifecycle_cancellation };
    let _global_worker_permit = GLOBAL_EXPERT_PREFETCH_GATE.acquire(&lifecycle_cancellation)?;
    #[cfg(test)]
    if cache.inner.fail_prefetch_spawn.load(std::sync::atomic::Ordering::Acquire) {
        return Err(PlacementError::BackendFailure(EngineError::backend_failure(
            "expert prefetch worker creation failed",
        )));
    }
    let dependent_cancellation = PlacementCancellation::new();
    let combined_cancellation =
        CombinedCancellation { first: &lifecycle_cancellation, second: &dependent_cancellation };
    let (work_result, prefetch_result) = thread::scope(|scope| {
        let handle = thread::Builder::new()
            .name("expert-prefetch".to_owned())
            .stack_size(SPARSE_EXPERT_PREFETCH_STACK_BYTES)
            .spawn_scoped(scope, || {
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    cache.acquire_selection_transaction(selection, source, &combined_cancellation)
                }))
                .unwrap_or_else(|_panic| {
                    Err(PlacementError::BackendFailure(EngineError::backend_failure(
                        "expert prefetch worker panicked",
                    )))
                });
                if result.is_err() {
                    dependent_cancellation.cancel();
                }
                result
            })
            .map_err(|_error| {
                PlacementError::BackendFailure(EngineError::backend_failure(
                    "expert prefetch worker creation failed",
                ))
            })?;
        let work_result =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| work(&combined_cancellation)))
                .unwrap_or_else(|_panic| {
                    Err(EngineError::backend_failure("expert independent work panicked"))
                });
        if work_result.is_err() {
            dependent_cancellation.cancel();
        }
        let prefetch_result = handle.join().unwrap_or_else(|_| {
            dependent_cancellation.cancel();
            Err(PlacementError::BackendFailure(EngineError::backend_failure(
                "expert prefetch worker panicked",
            )))
        });
        Ok::<_, PlacementError>((work_result, prefetch_result))
    })?;
    match (work_result, prefetch_result) {
        (Ok(work), Ok((leases, newly_loaded))) => {
            if let Err(error) = check_cancelled(&combined_cancellation) {
                drop(leases);
                cache.rollback_new_entries(&newly_loaded)?;
                return Err(error);
            }
            Ok(SparseExpertPrefetchResult { work, leases })
        }
        (Err(work_error), Ok((leases, newly_loaded))) => {
            drop(leases);
            cache.rollback_new_entries(&newly_loaded)?;
            Err(classify_engine_error(work_error))
        }
        (Ok(_work), Err(prefetch_error)) => Err(prefetch_error),
        (Err(work_error), Err(prefetch_error)) => {
            if matches!(work_error, EngineError::Cancellation(_)) {
                Err(prefetch_error)
            } else if matches!(prefetch_error, PlacementError::Cancelled) {
                Err(classify_engine_error(work_error))
            } else {
                Err(prefetch_error)
            }
        }
    }
}

fn ensure_valid(state: &ExpertCacheState) -> Result<(), PlacementError> {
    if state.valid { Ok(()) } else { Err(PlacementError::ArtifactIdentityMismatch) }
}

fn ensure_usable(state: &ExpertCacheState, allow_population: bool) -> Result<(), PlacementError> {
    ensure_valid(state)?;
    if state.resident_ready || (allow_population && state.populating) {
        Ok(())
    } else {
        Err(invalid("fully resident expert cache is not ready"))
    }
}

fn evict_until(
    plan: &SparseExpertExecutionPlan,
    state: &mut ExpertCacheState,
    target_bytes: u64,
) -> Result<Vec<ExpertCacheEntry>, PlacementError> {
    let mut removed = Vec::new();
    removed
        .try_reserve_exact(
            usize::try_from(state.entry_count)
                .map_err(|_| resource("expert entry count is not addressable"))?,
        )
        .map_err(|_| resource("expert eviction release table allocation failed"))?;
    while state.resident_bytes > target_bytes {
        let candidate = state
            .slots
            .iter()
            .enumerate()
            .filter_map(|(slot, entry)| entry.as_ref().map(|entry| (slot, entry)))
            .filter(|(_slot, entry)| entry.pins == 0)
            .min_by_key(|(slot, entry)| (entry.last_access, *slot))
            .map(|(slot, _entry)| slot)
            .ok_or_else(|| resource("expert cache has no evictable entry"))?;
        let evictions = checked_counter(state.metrics.evictions)?;
        let entry = state
            .slots
            .get_mut(candidate)
            .and_then(Option::take)
            .ok_or_else(|| invalid("expert eviction candidate disappeared"))?;
        state.resident_bytes = state
            .resident_bytes
            .checked_sub(entry.storage_bytes)
            .ok_or_else(|| invalid("expert cache accounting underflows"))?;
        state.entry_count = state
            .entry_count
            .checked_sub(1)
            .ok_or_else(|| invalid("expert cache entry count underflows"))?;
        state.metrics.evictions = evictions;
        removed.push(entry);
    }
    if state.resident_bytes < plan.slot_table_bytes() {
        return Err(invalid("expert cache evicted its fixed slot reservation"));
    }
    Ok(removed)
}

fn take_unpinned_at(
    plan: &SparseExpertExecutionPlan,
    state: &mut ExpertCacheState,
    slot: usize,
) -> Result<Option<ExpertCacheEntry>, PlacementError> {
    if state.valid {
        return Err(invalid("expert invalidation cleanup requires a terminal cache"));
    }
    let Some(entry) = state.slots.get_mut(slot).and_then(|entry| {
        if entry.as_ref().is_some_and(|entry| entry.pins == 0) { entry.take() } else { None }
    }) else {
        return Ok(None);
    };
    state.resident_bytes = state
        .resident_bytes
        .checked_sub(entry.storage_bytes)
        .ok_or_else(|| invalid("expert cache invalidation accounting underflows"))?;
    state.entry_count = state
        .entry_count
        .checked_sub(1)
        .ok_or_else(|| invalid("expert cache entry count underflows"))?;
    if state.resident_bytes < plan.slot_table_bytes() {
        return Err(invalid("expert invalidation released fixed slot accounting early"));
    }
    Ok(Some(entry))
}

struct ReleasedInvalidCache {
    _slot_reservation: MemoryReservation,
    _slots: Vec<Option<ExpertCacheEntry>>,
    _cache_capacity_guard: MemoryReservation,
    _execution_staging_guard: MemoryReservation,
    _safety_guard: Option<MemoryReservation>,
}

fn maybe_release_invalid_cache(
    state: &mut ExpertCacheState,
    load_in_flight: bool,
) -> Option<ReleasedInvalidCache> {
    if state.valid
        || state.entry_count != 0
        || state.staging_allocation_count != 0
        || load_in_flight
    {
        return None;
    }
    if state.slot_reservation.is_none()
        || state.cache_capacity_guard.is_none()
        || state.execution_staging_guard.is_none()
    {
        return None;
    }
    let slot_reservation = state.slot_reservation.take()?;
    let Some(resident_bytes) = state.resident_bytes.checked_sub(slot_reservation.bytes()) else {
        state.slot_reservation = Some(slot_reservation);
        return None;
    };
    let slots = std::mem::take(&mut state.slots);
    let cache_capacity_guard = state.cache_capacity_guard.take()?;
    let execution_staging_guard = state.execution_staging_guard.take()?;
    let safety_guard = state.safety_guard.take();
    state.resident_bytes = resident_bytes;
    Some(ReleasedInvalidCache {
        _slot_reservation: slot_reservation,
        _slots: slots,
        _cache_capacity_guard: cache_capacity_guard,
        _execution_staging_guard: execution_staging_guard,
        _safety_guard: safety_guard,
    })
}

fn maximum_active_pair(
    layers: &[SparseExpertLayerPlan],
    prefetch_depth: u8,
) -> Result<(u64, u64), PlacementError> {
    let mut storage = 0_u64;
    let mut execution = 0_u64;
    for (index, current) in layers.iter().enumerate() {
        let next = if prefetch_depth == 0 { None } else { layers.get(index + 1) };
        storage = storage.max(checked_add(
            current.selected_cache_upper_bound,
            next.map_or(0, |layer| layer.selected_cache_upper_bound),
            "adjacent selected expert storage bytes overflow",
        )?);
        execution = execution.max(checked_add(
            current.selected_execution_upper_bound,
            next.map_or(0, |layer| layer.selected_execution_upper_bound),
            "adjacent selected expert execution bytes overflow",
        )?);
    }
    Ok((storage, execution))
}

fn largest_sum(
    values: impl Iterator<Item = u64>,
    count: usize,
    overflow: &'static str,
) -> Result<u64, PlacementError> {
    let values = values;
    let (minimum, maximum) = values.size_hint();
    let capacity = maximum.unwrap_or(minimum);
    let mut collected = Vec::new();
    collected
        .try_reserve_exact(capacity)
        .map_err(|_| resource("expert size table allocation failed"))?;
    collected.extend(values);
    collected.sort_unstable_by(|left, right| right.cmp(left));
    collected
        .into_iter()
        .take(count)
        .try_fold(0_u64, |total, value| checked_add(total, value, overflow))
}

fn checked_add(left: u64, right: u64, reason: &'static str) -> Result<u64, PlacementError> {
    left.checked_add(right).ok_or_else(|| resource(reason))
}

fn checked_counter(value: u64) -> Result<u64, PlacementError> {
    value.checked_add(1).ok_or_else(|| resource("expert metric counter overflows"))
}

fn invalid(reason: &'static str) -> PlacementError {
    PlacementError::InvalidPlan(EngineError::invalid_input(reason))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PlacementCancellation;
    use personalai_engine_crypto::hash_package;
    use std::collections::BTreeMap;
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
    use std::sync::{Barrier, Mutex, mpsc};
    use std::time::Instant;

    fn hash(byte: u8) -> ModelPackageHash {
        ModelPackageHash::from_sha384_bytes([byte; ModelPackageHash::BYTE_LEN])
    }

    fn identity(byte: u8) -> SparseExpertPlanIdentity {
        SparseExpertPlanIdentity::synthetic(
            ModelId::new(format!("model-{byte}")).unwrap(),
            hash(byte),
            ArchitectureId::new("sparse-moe-test-only").unwrap(),
            BackendId::new("cpu-reference").unwrap(),
            1,
            1,
        )
        .unwrap()
    }

    fn descriptor(layer: u32, expert: u32, bytes: &[u8]) -> SparseExpertDescriptor {
        SparseExpertDescriptor::new(
            SparseExpertId::new(layer, expert).unwrap(),
            u64::try_from(bytes.len()).unwrap(),
            u64::try_from(bytes.len()).unwrap(),
            hash_package(bytes),
            DType::F32,
            Quantization::None,
        )
        .unwrap()
    }

    fn plan(
        identity: SparseExpertPlanIdentity,
        payloads: &[Vec<Vec<u8>>],
        mode: SparseExpertResidencyMode,
        cache_bytes: u64,
        prefetch_depth: u8,
    ) -> Arc<SparseExpertExecutionPlan> {
        plan_with_active(identity, payloads, mode, cache_bytes, prefetch_depth, 1)
    }

    fn plan_with_active(
        identity: SparseExpertPlanIdentity,
        payloads: &[Vec<Vec<u8>>],
        mode: SparseExpertResidencyMode,
        cache_bytes: u64,
        prefetch_depth: u8,
        active_experts: u16,
    ) -> Arc<SparseExpertExecutionPlan> {
        let layers = payloads
            .iter()
            .enumerate()
            .map(|(layer, experts)| {
                SparseExpertLayerPlan::new(
                    u32::try_from(layer).unwrap(),
                    active_experts,
                    experts
                        .iter()
                        .enumerate()
                        .map(|(expert, bytes)| {
                            descriptor(
                                u32::try_from(layer).unwrap(),
                                u32::try_from(expert).unwrap(),
                                bytes,
                            )
                        })
                        .collect(),
                )
                .unwrap()
            })
            .collect();
        Arc::new(
            SparseExpertExecutionPlan::new_synthetic(
                identity,
                layers,
                mode,
                SparseExpertResourceEnvelope::new(4_096, cache_bytes, 1_024, 512).unwrap(),
                prefetch_depth,
            )
            .unwrap(),
        )
    }

    struct BytesSource {
        identity: SparseExpertPlanIdentity,
        payloads: BTreeMap<SparseExpertId, Vec<u8>>,
    }

    struct FailOnExpertSource {
        inner: BytesSource,
        failed: SparseExpertId,
    }

    struct PanickingSource {
        identity: SparseExpertPlanIdentity,
    }

    impl ExpertByteSource for PanickingSource {
        fn identity(&self) -> &SparseExpertPlanIdentity {
            &self.identity
        }

        fn read_exact_expert(
            &self,
            _descriptor: &SparseExpertDescriptor,
            _destination: &mut [u8],
            _cancellation: &dyn PlacementCancellationProbe,
        ) -> Result<usize, PlacementError> {
            panic!("injected expert source panic")
        }
    }

    impl ExpertByteSource for FailOnExpertSource {
        fn identity(&self) -> &SparseExpertPlanIdentity {
            self.inner.identity()
        }

        fn read_exact_expert(
            &self,
            descriptor: &SparseExpertDescriptor,
            destination: &mut [u8],
            cancellation: &dyn PlacementCancellationProbe,
        ) -> Result<usize, PlacementError> {
            if descriptor.id() == self.failed {
                return Err(PlacementError::BackendFailure(EngineError::backend_failure(
                    "injected expert source failure",
                )));
            }
            self.inner.read_exact_expert(descriptor, destination, cancellation)
        }
    }

    impl ExpertByteSource for BytesSource {
        fn identity(&self) -> &SparseExpertPlanIdentity {
            &self.identity
        }

        fn read_exact_expert(
            &self,
            descriptor: &SparseExpertDescriptor,
            destination: &mut [u8],
            cancellation: &dyn PlacementCancellationProbe,
        ) -> Result<usize, PlacementError> {
            check_cancelled(cancellation)?;
            let bytes = self
                .payloads
                .get(&descriptor.id)
                .ok_or(PlacementError::ArtifactIdentityMismatch)?;
            if bytes.len() != destination.len() {
                return Ok(bytes.len());
            }
            destination.copy_from_slice(bytes);
            Ok(bytes.len())
        }
    }

    fn source(identity: SparseExpertPlanIdentity, payloads: &[Vec<Vec<u8>>]) -> BytesSource {
        let payloads = payloads
            .iter()
            .enumerate()
            .flat_map(|(layer, experts)| {
                experts.iter().enumerate().map(move |(expert, bytes)| {
                    (
                        SparseExpertId::new(
                            u32::try_from(layer).unwrap(),
                            u32::try_from(expert).unwrap(),
                        )
                        .unwrap(),
                        bytes.clone(),
                    )
                })
            })
            .collect();
        BytesSource { identity, payloads }
    }

    #[test]
    fn plan_rejects_noncanonical_inventory_and_insufficient_capacity() {
        let first = descriptor(0, 0, &[1; 8]);
        let wrong = descriptor(0, 2, &[2; 8]);
        assert!(SparseExpertLayerPlan::new(0, 1, vec![first.clone(), wrong]).is_err());
        let layer = SparseExpertLayerPlan::new(0, 1, vec![first]).unwrap();
        let result = SparseExpertExecutionPlan::new_synthetic(
            identity(1),
            vec![layer],
            SparseExpertResidencyMode::FullyResident,
            SparseExpertResourceEnvelope::new(1_024, 4, 128, 64).unwrap(),
            0,
        );
        assert!(matches!(result, Err(PlacementError::ResourceExhausted(_))));

        let quantized = SparseExpertDescriptor::new(
            SparseExpertId::new(0, 0).unwrap(),
            8,
            8,
            hash(9),
            DType::I8,
            Quantization::AipQ8V1,
        )
        .unwrap();
        let mixed = SparseExpertExecutionPlan::new_synthetic(
            identity(1),
            vec![SparseExpertLayerPlan::new(0, 1, vec![quantized]).unwrap()],
            SparseExpertResidencyMode::ExpertStreamed,
            SparseExpertResourceEnvelope::new(4_096, 512, 1_024, 512).unwrap(),
            0,
        );
        assert!(matches!(mixed, Err(PlacementError::InvalidPlan(_))));

        let layer_one = SparseExpertLayerPlan::new(1, 1, vec![descriptor(1, 0, &[1; 8])]).unwrap();
        let layer_three =
            SparseExpertLayerPlan::new(3, 1, vec![descriptor(3, 0, &[3; 8])]).unwrap();
        let noncontiguous = SparseExpertExecutionPlan::new_synthetic(
            identity(1),
            vec![layer_one.clone(), layer_three.clone()],
            SparseExpertResidencyMode::ExpertStreamed,
            SparseExpertResourceEnvelope::new(
                4_096,
                2 * SPARSE_EXPERT_CACHE_SLOT_BYTES + 16,
                1_024,
                512,
            )
            .unwrap(),
            0,
        )
        .unwrap();
        assert_eq!(
            noncontiguous.layers().iter().map(SparseExpertLayerPlan::layer).collect::<Vec<_>>(),
            [1, 3]
        );
        assert!(matches!(
            SparseExpertExecutionPlan::new_synthetic(
                identity(1),
                vec![layer_three, layer_one],
                SparseExpertResidencyMode::ExpertStreamed,
                SparseExpertResourceEnvelope::new(
                    4_096,
                    2 * SPARSE_EXPERT_CACHE_SLOT_BYTES + 16,
                    1_024,
                    512,
                )
                .unwrap(),
                0,
            ),
            Err(PlacementError::InvalidPlan(_))
        ));
    }

    #[test]
    fn sparse_plan_digest_rejects_same_base_identity_with_divergent_inventory() {
        let exact = identity(24);
        let first_payloads = vec![vec![vec![1; 8]]];
        let second_payloads = vec![vec![vec![2; 8]]];
        let first_plan = plan(
            exact.clone(),
            &first_payloads,
            SparseExpertResidencyMode::ExpertStreamed,
            SPARSE_EXPERT_CACHE_SLOT_BYTES + 8,
            0,
        );
        let second_plan = plan(
            exact.clone(),
            &second_payloads,
            SparseExpertResidencyMode::ExpertStreamed,
            SPARSE_EXPERT_CACHE_SLOT_BYTES + 8,
            0,
        );
        assert_eq!(first_plan.identity(), second_plan.identity());
        assert_ne!(first_plan.plan_digest(), second_plan.plan_digest());

        let selection =
            SparseExpertSelection::new(&first_plan, 0, vec![SparseExpertId::new(0, 0).unwrap()])
                .unwrap();
        let pool = MemoryPool::new(4_096).unwrap();
        let second_cache = SparseExpertCache::new(Arc::clone(&second_plan), &pool).unwrap();
        assert!(matches!(
            second_cache.acquire_selection(
                &selection,
                &source(exact, &second_payloads),
                &PlacementCancellation::new(),
            ),
            Err(PlacementError::ArtifactIdentityMismatch)
        ));
    }

    struct CancelAfterChecks {
        checks: AtomicU64,
        cancel_after: u64,
    }

    impl PlacementCancellationProbe for CancelAfterChecks {
        fn is_cancelled(&self) -> bool {
            self.checks.fetch_add(1, Ordering::AcqRel) >= self.cancel_after
        }
    }

    #[test]
    fn expert_hashing_polls_cancellation_between_bounded_chunks() {
        let payload = vec![7_u8; SPARSE_EXPERT_HASH_CHUNK_BYTES * 3];
        let cancellation = CancelAfterChecks { checks: AtomicU64::new(0), cancel_after: 2 };
        assert!(matches!(
            hash_expert_payload(&payload, &cancellation),
            Err(PlacementError::Cancelled)
        ));
        assert!(cancellation.checks.load(Ordering::Acquire) >= 3);
    }

    #[test]
    fn global_prefetch_gate_is_bounded_and_wait_is_cancellable() {
        let gate = PrefetchPermitGate::new(1);
        let first_cancellation = PlacementCancellation::new();
        let first = gate.acquire(&first_cancellation).unwrap();
        let waiting_cancellation = PlacementCancellation::new();
        thread::scope(|scope| {
            let waiting = waiting_cancellation.clone();
            let gate_ref = &gate;
            let handle = scope.spawn(move || gate_ref.acquire(&waiting));
            thread::sleep(Duration::from_millis(20));
            waiting_cancellation.cancel();
            assert!(matches!(handle.join().unwrap(), Err(PlacementError::Cancelled)));
        });
        drop(first);
        let available = gate.acquire(&PlacementCancellation::new()).unwrap();
        drop(available);
    }

    #[test]
    fn cache_admission_holds_durable_staging_and_safety_capacity() {
        let payloads = vec![vec![vec![1; 8]]];
        let exact = identity(10);
        let plan = plan(
            exact.clone(),
            &payloads,
            SparseExpertResidencyMode::ExpertStreamed,
            SPARSE_EXPERT_CACHE_SLOT_BYTES + 8,
            0,
        );
        let pool = MemoryPool::new(4_096).unwrap();
        let existing = pool.reserve(2_000).unwrap();
        let cache = SparseExpertCache::new(Arc::clone(&plan), &pool).unwrap();
        assert_eq!(pool.used_bytes().unwrap(), 3_800);
        assert!(pool.reserve(297).is_err());
        let lease = cache
            .acquire(
                SparseExpertId::new(0, 0).unwrap(),
                &source(exact, &payloads),
                &PlacementCancellation::new(),
            )
            .unwrap();
        assert_eq!(lease.bytes(), &[1; 8]);
        assert_eq!(pool.used_bytes().unwrap(), 3_800);
        drop(lease);
        cache.invalidate().unwrap();
        assert_eq!(pool.used_bytes().unwrap(), 2_000);
        drop(existing);
        assert_eq!(pool.used_bytes().unwrap(), 0);

        let blocked_pool = MemoryPool::new(4_096).unwrap();
        let existing = blocked_pool.reserve(2_305).unwrap();
        assert!(matches!(
            SparseExpertCache::new(plan, &blocked_pool),
            Err(PlacementError::ResourceExhausted(_))
        ));
        assert_eq!(blocked_pool.used_bytes().unwrap(), 2_305);
        drop(existing);
    }

    #[test]
    fn selection_and_lease_are_bound_to_one_exact_plan_identity() {
        let payloads = vec![vec![vec![1; 8]]];
        let first_identity = identity(11);
        let second_identity = identity(12);
        let first_plan = plan(
            first_identity.clone(),
            &payloads,
            SparseExpertResidencyMode::ExpertStreamed,
            SPARSE_EXPERT_CACHE_SLOT_BYTES + 8,
            0,
        );
        let second_plan = plan(
            second_identity.clone(),
            &payloads,
            SparseExpertResidencyMode::ExpertStreamed,
            SPARSE_EXPERT_CACHE_SLOT_BYTES + 8,
            0,
        );
        let selection =
            SparseExpertSelection::new(&first_plan, 0, vec![SparseExpertId::new(0, 0).unwrap()])
                .unwrap();
        let pool = MemoryPool::new(4_096).unwrap();
        let cache = SparseExpertCache::new(second_plan, &pool).unwrap();
        assert!(matches!(
            cache.acquire_selection(
                &selection,
                &source(second_identity, &payloads),
                &PlacementCancellation::new(),
            ),
            Err(PlacementError::ArtifactIdentityMismatch)
        ));

        let first_pool = MemoryPool::new(4_096).unwrap();
        let first_cache = SparseExpertCache::new(first_plan, &first_pool).unwrap();
        let lease = first_cache
            .acquire(
                SparseExpertId::new(0, 0).unwrap(),
                &source(first_identity.clone(), &payloads),
                &PlacementCancellation::new(),
            )
            .unwrap();
        assert_eq!(lease.identity(), &first_identity);
    }

    #[test]
    fn cache_is_bounded_deterministic_and_memory_accounted() {
        let payloads = vec![vec![vec![1; 8], vec![2; 8], vec![3; 8]]];
        let identity = identity(1);
        let plan = plan(
            identity.clone(),
            &payloads,
            SparseExpertResidencyMode::ExpertStreamed,
            3 * SPARSE_EXPERT_CACHE_SLOT_BYTES + 2 * 8,
            0,
        );
        let pool = MemoryPool::new(4_096).unwrap();
        let cache = SparseExpertCache::new(plan, &pool).unwrap();
        let source = source(identity, &payloads);
        let cancellation = PlacementCancellation::new();

        drop(cache.acquire(SparseExpertId::new(0, 0).unwrap(), &source, &cancellation).unwrap());
        drop(cache.acquire(SparseExpertId::new(0, 1).unwrap(), &source, &cancellation).unwrap());
        drop(cache.acquire(SparseExpertId::new(0, 0).unwrap(), &source, &cancellation).unwrap());
        drop(cache.acquire(SparseExpertId::new(0, 2).unwrap(), &source, &cancellation).unwrap());

        let metrics = cache.metrics().unwrap();
        assert_eq!(metrics.hits, 1);
        assert_eq!(metrics.misses, 3);
        assert_eq!(metrics.evictions, 1);
        assert_eq!(metrics.resident_bytes, 3 * SPARSE_EXPERT_CACHE_SLOT_BYTES + 2 * 8);
        assert_eq!(
            pool.statistics()
                .unwrap()
                .bytes_for(&MemoryOwner::SparseExpertCache(ModelId::new("model-1").unwrap())),
            3 * SPARSE_EXPERT_CACHE_SLOT_BYTES + 2 * 8
        );
    }

    #[test]
    fn pinned_entry_blocks_unsafe_eviction() {
        let payloads = vec![vec![vec![1; 8], vec![2; 8]]];
        let identity = identity(2);
        let plan = plan(
            identity.clone(),
            &payloads,
            SparseExpertResidencyMode::ExpertStreamed,
            2 * SPARSE_EXPERT_CACHE_SLOT_BYTES + 8,
            0,
        );
        let pool = MemoryPool::new(4_096).unwrap();
        let cache = SparseExpertCache::new(plan, &pool).unwrap();
        let source = source(identity, &payloads);
        let cancellation = PlacementCancellation::new();
        let lease =
            cache.acquire(SparseExpertId::new(0, 0).unwrap(), &source, &cancellation).unwrap();
        assert!(matches!(
            cache.acquire(SparseExpertId::new(0, 1).unwrap(), &source, &cancellation),
            Err(PlacementError::ResourceExhausted(_))
        ));
        assert_eq!(lease.bytes(), &[1; 8]);
    }

    #[test]
    fn identity_digest_and_cancellation_fail_closed() {
        let payloads = vec![vec![vec![1; 8]]];
        let exact = identity(3);
        let plan = plan(
            exact.clone(),
            &payloads,
            SparseExpertResidencyMode::ExpertStreamed,
            SPARSE_EXPERT_CACHE_SLOT_BYTES + 8,
            0,
        );
        let pool = MemoryPool::new(4_096).unwrap();
        let cache = SparseExpertCache::new(plan, &pool).unwrap();
        let wrong = source(identity(4), &payloads);
        let cancellation = PlacementCancellation::new();
        assert!(matches!(
            cache.acquire(SparseExpertId::new(0, 0).unwrap(), &wrong, &cancellation),
            Err(PlacementError::ArtifactIdentityMismatch)
        ));

        let mut corrupt = source(exact, &payloads);
        corrupt.payloads.insert(SparseExpertId::new(0, 0).unwrap(), vec![9; 8]);
        assert!(matches!(
            cache.acquire(SparseExpertId::new(0, 0).unwrap(), &corrupt, &cancellation),
            Err(PlacementError::ArtifactIdentityMismatch)
        ));
        cancellation.cancel();
        assert!(matches!(
            cache.acquire(SparseExpertId::new(0, 0).unwrap(), &corrupt, &cancellation),
            Err(PlacementError::Cancelled)
        ));
    }

    #[test]
    fn truncated_source_is_rejected_without_a_partial_entry() {
        let payloads = vec![vec![vec![1; 8]]];
        let exact = identity(13);
        let plan = plan(
            exact.clone(),
            &payloads,
            SparseExpertResidencyMode::ExpertStreamed,
            SPARSE_EXPERT_CACHE_SLOT_BYTES + 8,
            0,
        );
        let pool = MemoryPool::new(4_096).unwrap();
        let cache = SparseExpertCache::new(plan, &pool).unwrap();
        let mut truncated = source(exact, &payloads);
        truncated.payloads.insert(SparseExpertId::new(0, 0).unwrap(), vec![1; 7]);
        assert!(matches!(
            cache.acquire(
                SparseExpertId::new(0, 0).unwrap(),
                &truncated,
                &PlacementCancellation::new(),
            ),
            Err(PlacementError::ArtifactIdentityMismatch)
        ));
        let metrics = cache.metrics().unwrap();
        assert_eq!(metrics.entry_count, 0);
        assert_eq!(metrics.resident_bytes, SPARSE_EXPERT_CACHE_SLOT_BYTES);
        cache.invalidate().unwrap();
        assert_eq!(pool.used_bytes().unwrap(), 0);
    }

    #[test]
    fn fully_resident_partial_failure_invalidates_all_loaded_experts() {
        let payloads = vec![vec![vec![1; 8], vec![2; 8]]];
        let exact = identity(14);
        let plan = plan(
            exact.clone(),
            &payloads,
            SparseExpertResidencyMode::FullyResident,
            2 * SPARSE_EXPERT_CACHE_SLOT_BYTES + 2 * 8,
            0,
        );
        let pool = MemoryPool::new(4_096).unwrap();
        let cache = SparseExpertCache::new(plan, &pool).unwrap();
        let source = FailOnExpertSource {
            inner: source(exact, &payloads),
            failed: SparseExpertId::new(0, 1).unwrap(),
        };
        assert!(matches!(
            cache.populate_fully_resident(&source, &PlacementCancellation::new()),
            Err(PlacementError::BackendFailure(_))
        ));
        assert_eq!(cache.metrics().unwrap().entry_count, 0);
        assert_eq!(cache.metrics().unwrap().resident_bytes, 0);
        assert_eq!(pool.used_bytes().unwrap(), 0);
        assert!(matches!(
            cache.acquire(
                SparseExpertId::new(0, 0).unwrap(),
                &source,
                &PlacementCancellation::new(),
            ),
            Err(PlacementError::Cancelled)
        ));
    }

    #[test]
    fn multi_expert_failure_rolls_back_only_new_unpinned_entries() {
        let payloads = vec![vec![vec![1; 8], vec![2; 8]]];
        let exact = identity(15);
        let plan = plan_with_active(
            exact.clone(),
            &payloads,
            SparseExpertResidencyMode::ExpertStreamed,
            2 * SPARSE_EXPERT_CACHE_SLOT_BYTES + 2 * 8,
            0,
            2,
        );
        let selection = SparseExpertSelection::new(
            &plan,
            0,
            vec![SparseExpertId::new(0, 0).unwrap(), SparseExpertId::new(0, 1).unwrap()],
        )
        .unwrap();
        let pool = MemoryPool::new(4_096).unwrap();
        let cache = SparseExpertCache::new(plan, &pool).unwrap();
        let source = FailOnExpertSource {
            inner: source(exact, &payloads),
            failed: SparseExpertId::new(0, 1).unwrap(),
        };
        assert!(matches!(
            cache.acquire_selection(&selection, &source, &PlacementCancellation::new()),
            Err(PlacementError::BackendFailure(_))
        ));
        let metrics = cache.metrics().unwrap();
        assert_eq!(metrics.entry_count, 0);
        assert_eq!(metrics.resident_bytes, 2 * SPARSE_EXPERT_CACHE_SLOT_BYTES);
        cache.invalidate().unwrap();
        assert_eq!(pool.used_bytes().unwrap(), 0);
    }

    struct BlockingSource {
        inner: BytesSource,
        entered: Arc<Barrier>,
        release: Arc<Barrier>,
    }

    impl ExpertByteSource for BlockingSource {
        fn identity(&self) -> &SparseExpertPlanIdentity {
            &self.inner.identity
        }

        fn read_exact_expert(
            &self,
            descriptor: &SparseExpertDescriptor,
            destination: &mut [u8],
            cancellation: &dyn PlacementCancellationProbe,
        ) -> Result<usize, PlacementError> {
            self.entered.wait();
            self.release.wait();
            self.inner.read_exact_expert(descriptor, destination, cancellation)
        }
    }

    struct CooperativeBlockingSource {
        inner: BytesSource,
        entered: mpsc::SyncSender<()>,
        exited_on_cancellation: AtomicBool,
    }

    impl ExpertByteSource for CooperativeBlockingSource {
        fn identity(&self) -> &SparseExpertPlanIdentity {
            &self.inner.identity
        }

        fn read_exact_expert(
            &self,
            _descriptor: &SparseExpertDescriptor,
            _destination: &mut [u8],
            cancellation: &dyn PlacementCancellationProbe,
        ) -> Result<usize, PlacementError> {
            self.entered.send(()).map_err(|_| invalid("cooperative source signal failed"))?;
            let deadline = Instant::now() + Duration::from_secs(1);
            while !cancellation.is_cancelled() {
                if Instant::now() >= deadline {
                    return Err(PlacementError::BackendFailure(EngineError::backend_failure(
                        "cooperative source cancellation timed out",
                    )));
                }
                thread::yield_now();
            }
            self.exited_on_cancellation.store(true, Ordering::Release);
            Err(PlacementError::Cancelled)
        }
    }

    #[test]
    fn invalidation_rejects_stale_completion_and_releases_bytes() {
        let payloads = vec![vec![vec![1; 8]]];
        let exact = identity(5);
        let plan = plan(
            exact.clone(),
            &payloads,
            SparseExpertResidencyMode::ExpertStreamed,
            SPARSE_EXPERT_CACHE_SLOT_BYTES + 8,
            0,
        );
        let pool = MemoryPool::new(4_096).unwrap();
        let cache = SparseExpertCache::new(plan, &pool).unwrap();
        let entered = Arc::new(Barrier::new(2));
        let release = Arc::new(Barrier::new(2));
        let source = BlockingSource {
            inner: source(exact, &payloads),
            entered: Arc::clone(&entered),
            release: Arc::clone(&release),
        };
        let cancellation = PlacementCancellation::new();
        thread::scope(|scope| {
            let handle = scope.spawn(|| {
                cache.acquire(SparseExpertId::new(0, 0).unwrap(), &source, &cancellation)
            });
            entered.wait();
            cache.invalidate().unwrap();
            assert_eq!(pool.used_bytes().unwrap(), 1_800);
            release.wait();
            assert!(matches!(handle.join().unwrap(), Err(PlacementError::Cancelled)));
        });
        assert_eq!(cache.metrics().unwrap().resident_bytes, 0);
        assert_eq!(pool.used_bytes().unwrap(), 0);
    }

    #[test]
    fn invalidation_during_load_token_establishment_releases_all_reservations() {
        let payloads = vec![vec![vec![1; 8]]];
        let plan = plan(
            identity(27),
            &payloads,
            SparseExpertResidencyMode::ExpertStreamed,
            SPARSE_EXPERT_CACHE_SLOT_BYTES + 8,
            0,
        );
        let pool = MemoryPool::new(4_096).unwrap();
        let cache = SparseExpertCache::new(plan, &pool).unwrap();
        let entered = Arc::new(Barrier::new(2));
        let release = Arc::new(Barrier::new(2));
        *cache.inner.load_token_hook.lock().unwrap() = Some(TokenAcquisitionHook {
            entered: Arc::clone(&entered),
            release: Arc::clone(&release),
        });
        let cancellation = PlacementCancellation::new();

        thread::scope(|scope| {
            let handle = scope.spawn(|| cache.acquire_load_token(&cancellation).map(drop));
            entered.wait();
            cache.invalidate().unwrap();
            assert!(pool.used_bytes().unwrap() > 0);
            release.wait();
            assert!(matches!(
                handle.join().unwrap(),
                Err(PlacementError::ArtifactIdentityMismatch)
            ));
        });

        assert_eq!(cache.metrics().unwrap().resident_bytes, 0);
        assert_eq!(pool.used_bytes().unwrap(), 0);
    }

    #[test]
    fn invalidation_during_prefetch_token_establishment_releases_all_reservations() {
        let payloads = vec![vec![vec![1; 8]]];
        let plan = plan(
            identity(28),
            &payloads,
            SparseExpertResidencyMode::ExpertStreamed,
            SPARSE_EXPERT_CACHE_SLOT_BYTES + 8,
            1,
        );
        let pool = MemoryPool::new(4_096).unwrap();
        let cache = SparseExpertCache::new(plan, &pool).unwrap();
        let entered = Arc::new(Barrier::new(2));
        let release = Arc::new(Barrier::new(2));
        *cache.inner.prefetch_token_hook.lock().unwrap() = Some(TokenAcquisitionHook {
            entered: Arc::clone(&entered),
            release: Arc::clone(&release),
        });
        let cancellation = PlacementCancellation::new();

        thread::scope(|scope| {
            let handle = scope.spawn(|| cache.acquire_prefetch_token(&cancellation).map(drop));
            entered.wait();
            cache.invalidate().unwrap();
            assert!(pool.used_bytes().unwrap() > 0);
            release.wait();
            assert!(matches!(
                handle.join().unwrap(),
                Err(PlacementError::ArtifactIdentityMismatch)
            ));
        });

        assert_eq!(cache.metrics().unwrap().resident_bytes, 0);
        assert_eq!(pool.used_bytes().unwrap(), 0);
    }

    #[test]
    fn invalidation_wakes_a_serialized_load_waiter() {
        let payloads = vec![vec![vec![1; 8]]];
        let exact = identity(19);
        let plan = plan(
            exact.clone(),
            &payloads,
            SparseExpertResidencyMode::ExpertStreamed,
            SPARSE_EXPERT_CACHE_SLOT_BYTES + 8,
            0,
        );
        let pool = MemoryPool::new(4_096).unwrap();
        let cache = SparseExpertCache::new(plan, &pool).unwrap();
        let entered = Arc::new(Barrier::new(2));
        let release = Arc::new(Barrier::new(2));
        let source = BlockingSource {
            inner: source(exact, &payloads),
            entered: Arc::clone(&entered),
            release: Arc::clone(&release),
        };
        let cancellation = PlacementCancellation::new();
        let (sender, receiver) = mpsc::sync_channel(1);
        thread::scope(|scope| {
            let first = scope.spawn(|| {
                cache.acquire(SparseExpertId::new(0, 0).unwrap(), &source, &cancellation)
            });
            entered.wait();
            scope.spawn(|| {
                let result =
                    cache.acquire(SparseExpertId::new(0, 0).unwrap(), &source, &cancellation);
                sender.send(result.map(|lease| lease.bytes().to_vec())).unwrap();
            });
            thread::sleep(Duration::from_millis(20));
            cache.invalidate().unwrap();
            assert!(matches!(
                receiver.recv_timeout(Duration::from_millis(200)).unwrap(),
                Err(PlacementError::ArtifactIdentityMismatch)
            ));
            release.wait();
            assert!(matches!(first.join().unwrap(), Err(PlacementError::Cancelled)));
        });
    }

    #[test]
    fn invalidation_cooperatively_cancels_an_inflight_source() {
        let payloads = vec![vec![vec![1; 8]]];
        let exact = identity(20);
        let plan = plan(
            exact.clone(),
            &payloads,
            SparseExpertResidencyMode::ExpertStreamed,
            SPARSE_EXPERT_CACHE_SLOT_BYTES + 8,
            0,
        );
        let pool = MemoryPool::new(4_096).unwrap();
        let cache = SparseExpertCache::new(plan, &pool).unwrap();
        let (entered, observed) = mpsc::sync_channel(1);
        let source = CooperativeBlockingSource {
            inner: source(exact, &payloads),
            entered,
            exited_on_cancellation: AtomicBool::new(false),
        };
        thread::scope(|scope| {
            let handle = scope.spawn(|| {
                cache.acquire(
                    SparseExpertId::new(0, 0).unwrap(),
                    &source,
                    &PlacementCancellation::new(),
                )
            });
            observed.recv_timeout(Duration::from_secs(1)).unwrap();
            cache.invalidate().unwrap();
            assert!(matches!(handle.join().unwrap(), Err(PlacementError::Cancelled)));
        });
        assert!(source.exited_on_cancellation.load(Ordering::Acquire));
        assert_eq!(pool.used_bytes().unwrap(), 0);
    }

    #[test]
    fn repeated_concurrent_invalidation_is_idempotent_without_allocating() {
        let payloads = vec![vec![vec![1; 8]]];
        let exact = identity(21);
        let plan = plan(
            exact.clone(),
            &payloads,
            SparseExpertResidencyMode::ExpertStreamed,
            SPARSE_EXPERT_CACHE_SLOT_BYTES + 8,
            0,
        );
        let pool = MemoryPool::new(4_096).unwrap();
        let cache = SparseExpertCache::new(plan, &pool).unwrap();
        drop(
            cache
                .acquire(
                    SparseExpertId::new(0, 0).unwrap(),
                    &source(exact, &payloads),
                    &PlacementCancellation::new(),
                )
                .unwrap(),
        );
        let pressure = pool.reserve(pool.available_bytes().unwrap()).unwrap();
        thread::scope(|scope| {
            let first = scope.spawn(|| cache.invalidate());
            let second = scope.spawn(|| cache.invalidate());
            first.join().unwrap().unwrap();
            second.join().unwrap().unwrap();
        });
        cache.invalidate().unwrap();
        let metrics = cache.metrics().unwrap();
        assert_eq!(metrics.invalidations, 1);
        assert_eq!(metrics.resident_bytes, 0);
        drop(pressure);
        assert_eq!(pool.used_bytes().unwrap(), 0);
    }

    #[test]
    fn scoped_prefetch_is_joined_and_returns_verified_leases() {
        let payloads = vec![vec![vec![1; 8]], vec![vec![2; 8]]];
        let exact = identity(6);
        let plan = plan(
            exact.clone(),
            &payloads,
            SparseExpertResidencyMode::ExpertStreamed,
            2 * SPARSE_EXPERT_CACHE_SLOT_BYTES + 2 * 8,
            1,
        );
        let selection =
            SparseExpertSelection::new(&plan, 1, vec![SparseExpertId::new(1, 0).unwrap()]).unwrap();
        let pool = MemoryPool::new(4_096).unwrap();
        let cache = SparseExpertCache::new(plan, &pool).unwrap();
        let source = source(exact, &payloads);
        let cancellation = PlacementCancellation::new();
        let result =
            run_bounded_expert_prefetch(&cache, &selection, &source, &cancellation, |_probe| {
                Ok::<_, EngineError>(41_u32)
            })
            .unwrap();
        assert_eq!(result.work, 41);
        assert_eq!(result.leases[0].bytes(), &[2; 8]);
    }

    #[test]
    fn prefetch_spawn_failure_does_not_execute_independent_work() {
        let payloads = vec![vec![vec![1; 8]]];
        let exact = identity(16);
        let plan = plan(
            exact.clone(),
            &payloads,
            SparseExpertResidencyMode::ExpertStreamed,
            SPARSE_EXPERT_CACHE_SLOT_BYTES + 8,
            1,
        );
        let selection =
            SparseExpertSelection::new(&plan, 0, vec![SparseExpertId::new(0, 0).unwrap()]).unwrap();
        let pool = MemoryPool::new(4_096).unwrap();
        let cache = SparseExpertCache::new(plan, &pool).unwrap();
        cache.inner.fail_prefetch_spawn.store(true, Ordering::Release);
        let work_called = AtomicBool::new(false);
        let result = run_bounded_expert_prefetch(
            &cache,
            &selection,
            &source(exact, &payloads),
            &PlacementCancellation::new(),
            |_probe| {
                work_called.store(true, Ordering::Release);
                Ok::<_, EngineError>(())
            },
        );
        assert!(matches!(result, Err(PlacementError::BackendFailure(_))));
        assert!(!work_called.load(Ordering::Acquire));
        assert_eq!(cache.metrics().unwrap().entry_count, 0);
    }

    #[test]
    fn source_and_independent_work_panics_become_typed_failures() {
        let payloads = vec![vec![vec![1; 8]]];
        let exact = identity(25);
        let plan = plan(
            exact.clone(),
            &payloads,
            SparseExpertResidencyMode::ExpertStreamed,
            SPARSE_EXPERT_CACHE_SLOT_BYTES + 8,
            1,
        );
        let selection =
            SparseExpertSelection::new(&plan, 0, vec![SparseExpertId::new(0, 0).unwrap()]).unwrap();
        let pool = MemoryPool::new(4_096).unwrap();
        let cache = SparseExpertCache::new(Arc::clone(&plan), &pool).unwrap();
        assert!(matches!(
            cache.acquire(
                SparseExpertId::new(0, 0).unwrap(),
                &PanickingSource { identity: exact.clone() },
                &PlacementCancellation::new(),
            ),
            Err(PlacementError::BackendFailure(_))
        ));
        assert_eq!(cache.metrics().unwrap().entry_count, 0);

        let result = run_bounded_expert_prefetch(
            &cache,
            &selection,
            &source(exact, &payloads),
            &PlacementCancellation::new(),
            |_probe| -> Result<(), EngineError> { panic!("injected independent work panic") },
        );
        assert!(matches!(result, Err(PlacementError::BackendFailure(_))));
        assert_eq!(cache.metrics().unwrap().entry_count, 0);
    }

    #[test]
    fn failed_independent_work_cancels_and_rolls_back_prefetch() {
        let payloads = vec![vec![vec![1; 8]]];
        let exact = identity(17);
        let plan = plan(
            exact.clone(),
            &payloads,
            SparseExpertResidencyMode::ExpertStreamed,
            SPARSE_EXPERT_CACHE_SLOT_BYTES + 8,
            1,
        );
        let selection =
            SparseExpertSelection::new(&plan, 0, vec![SparseExpertId::new(0, 0).unwrap()]).unwrap();
        let pool = MemoryPool::new(4_096).unwrap();
        let cache = SparseExpertCache::new(plan, &pool).unwrap();
        let result = run_bounded_expert_prefetch(
            &cache,
            &selection,
            &source(exact, &payloads),
            &PlacementCancellation::new(),
            |_probe| {
                Err::<(), _>(EngineError::backend_failure("injected independent work failure"))
            },
        );
        assert!(matches!(result, Err(PlacementError::BackendFailure(_))));
        let metrics = cache.metrics().unwrap();
        assert_eq!(metrics.entry_count, 0);
        assert_eq!(metrics.resident_bytes, SPARSE_EXPERT_CACHE_SLOT_BYTES);
    }

    #[test]
    fn source_failure_cancels_cooperative_work_and_remains_the_root_error() {
        let payloads = vec![vec![vec![1; 8]]];
        let exact = identity(23);
        let plan = plan(
            exact.clone(),
            &payloads,
            SparseExpertResidencyMode::ExpertStreamed,
            SPARSE_EXPERT_CACHE_SLOT_BYTES + 8,
            1,
        );
        let selection =
            SparseExpertSelection::new(&plan, 0, vec![SparseExpertId::new(0, 0).unwrap()]).unwrap();
        let pool = MemoryPool::new(4_096).unwrap();
        let cache = SparseExpertCache::new(plan, &pool).unwrap();
        let source = FailOnExpertSource {
            inner: source(exact, &payloads),
            failed: SparseExpertId::new(0, 0).unwrap(),
        };
        let observed = AtomicBool::new(false);
        let result = run_bounded_expert_prefetch(
            &cache,
            &selection,
            &source,
            &PlacementCancellation::new(),
            |probe| {
                let deadline = Instant::now() + Duration::from_secs(1);
                while !probe.is_cancelled() {
                    if Instant::now() >= deadline {
                        return Err(EngineError::backend_failure(
                            "dependent work cancellation timed out",
                        ));
                    }
                    thread::yield_now();
                }
                observed.store(true, Ordering::Release);
                Err::<(), _>(EngineError::cancellation(None))
            },
        );
        assert!(matches!(result, Err(PlacementError::BackendFailure(_))));
        assert!(observed.load(Ordering::Acquire));
        assert_eq!(cache.metrics().unwrap().entry_count, 0);
    }

    #[test]
    fn invalidation_after_fast_prefetch_prevents_successful_return() {
        let payloads = vec![vec![vec![1; 8]]];
        let exact = identity(24);
        let plan = plan(
            exact.clone(),
            &payloads,
            SparseExpertResidencyMode::ExpertStreamed,
            SPARSE_EXPERT_CACHE_SLOT_BYTES + 8,
            1,
        );
        let selection =
            SparseExpertSelection::new(&plan, 0, vec![SparseExpertId::new(0, 0).unwrap()]).unwrap();
        let pool = MemoryPool::new(4_096).unwrap();
        let cache = SparseExpertCache::new(plan, &pool).unwrap();
        let source = CountingSource { inner: source(exact, &payloads), reads: AtomicU64::new(0) };
        let (work_started, work_observed) = mpsc::sync_channel(1);
        let (release_work, release_observed) = mpsc::sync_channel(1);
        thread::scope(|scope| {
            let cache_ref = &cache;
            let selection_ref = &selection;
            let source_ref = &source;
            let handle = scope.spawn(move || {
                run_bounded_expert_prefetch(
                    cache_ref,
                    selection_ref,
                    source_ref,
                    &PlacementCancellation::new(),
                    move |_probe| {
                        work_started.send(()).unwrap();
                        release_observed.recv_timeout(Duration::from_secs(1)).unwrap();
                        Ok::<_, EngineError>(())
                    },
                )
            });
            work_observed.recv_timeout(Duration::from_secs(1)).unwrap();
            let deadline = Instant::now() + Duration::from_secs(1);
            while source.reads.load(Ordering::Acquire) == 0 {
                assert!(Instant::now() < deadline, "prefetch did not finish its source read");
                thread::yield_now();
            }
            cache.invalidate().unwrap();
            release_work.send(()).unwrap();
            assert!(matches!(handle.join().unwrap(), Err(PlacementError::Cancelled)));
        });
        assert_eq!(pool.used_bytes().unwrap(), 0);
    }

    struct CountingSource {
        inner: BytesSource,
        reads: AtomicU64,
    }

    impl ExpertByteSource for CountingSource {
        fn identity(&self) -> &SparseExpertPlanIdentity {
            &self.inner.identity
        }

        fn read_exact_expert(
            &self,
            descriptor: &SparseExpertDescriptor,
            destination: &mut [u8],
            cancellation: &dyn PlacementCancellationProbe,
        ) -> Result<usize, PlacementError> {
            self.reads.fetch_add(1, Ordering::AcqRel);
            self.inner.read_exact_expert(descriptor, destination, cancellation)
        }
    }

    #[test]
    fn simultaneous_readers_share_one_verified_immutable_load() {
        let payloads = vec![vec![vec![7; 8]]];
        let exact = identity(7);
        let plan = plan(
            exact.clone(),
            &payloads,
            SparseExpertResidencyMode::ExpertStreamed,
            SPARSE_EXPERT_CACHE_SLOT_BYTES + 8,
            0,
        );
        let pool = MemoryPool::new(4_096).unwrap();
        let cache = SparseExpertCache::new(plan, &pool).unwrap();
        let source = CountingSource { inner: source(exact, &payloads), reads: AtomicU64::new(0) };
        let cancellation = PlacementCancellation::new();
        let observed = Arc::new(Mutex::new(Vec::new()));
        thread::scope(|scope| {
            for _ in 0..2 {
                let observed = Arc::clone(&observed);
                let cache = cache.clone();
                let source = &source;
                let cancellation = &cancellation;
                scope.spawn(move || {
                    let lease = cache
                        .acquire(SparseExpertId::new(0, 0).unwrap(), source, cancellation)
                        .unwrap();
                    observed.lock().unwrap().push(lease.bytes().to_vec());
                });
            }
        });
        assert_eq!(source.reads.load(Ordering::Acquire), 1);
        assert_eq!(*observed.lock().unwrap(), vec![vec![7; 8], vec![7; 8]]);
        assert_eq!(cache.metrics().unwrap().loads, 1);
    }

    #[test]
    fn concurrent_prefetch_calls_share_one_cache_owned_worker_slot() {
        let payloads = vec![vec![vec![7; 8]]];
        let exact = identity(18);
        let plan = plan(
            exact.clone(),
            &payloads,
            SparseExpertResidencyMode::ExpertStreamed,
            SPARSE_EXPERT_CACHE_SLOT_BYTES + 8,
            1,
        );
        let selection =
            SparseExpertSelection::new(&plan, 0, vec![SparseExpertId::new(0, 0).unwrap()]).unwrap();
        let pool = MemoryPool::new(4_096).unwrap();
        let cache = SparseExpertCache::new(plan, &pool).unwrap();
        let entered = Arc::new(Barrier::new(2));
        let release = Arc::new(Barrier::new(2));
        let source = BlockingSource {
            inner: source(exact, &payloads),
            entered: Arc::clone(&entered),
            release: Arc::clone(&release),
        };
        let cancellation = PlacementCancellation::new();
        let work_calls = AtomicU64::new(0);
        thread::scope(|scope| {
            let first = scope.spawn(|| {
                run_bounded_expert_prefetch(&cache, &selection, &source, &cancellation, |_probe| {
                    work_calls.fetch_add(1, Ordering::AcqRel);
                    Ok::<_, EngineError>(())
                })
            });
            entered.wait();
            let second = scope.spawn(|| {
                run_bounded_expert_prefetch(&cache, &selection, &source, &cancellation, |_probe| {
                    work_calls.fetch_add(1, Ordering::AcqRel);
                    Ok::<_, EngineError>(())
                })
            });
            thread::sleep(Duration::from_millis(20));
            assert_eq!(work_calls.load(Ordering::Acquire), 1);
            release.wait();
            assert!(first.join().unwrap().is_ok());
            assert!(second.join().unwrap().is_ok());
        });
        assert_eq!(work_calls.load(Ordering::Acquire), 2);
        assert_eq!(cache.metrics().unwrap().loads, 1);
    }

    #[test]
    fn staging_uses_its_admitted_child_quota_and_survives_invalidation_safely() {
        let payloads = vec![vec![vec![8; 8]]];
        let exact = identity(22);
        let plan = plan(
            exact.clone(),
            &payloads,
            SparseExpertResidencyMode::ExpertStreamed,
            SPARSE_EXPERT_CACHE_SLOT_BYTES + 8,
            0,
        );
        let pool = MemoryPool::new(4_096).unwrap();
        let cache = SparseExpertCache::new(plan, &pool).unwrap();
        let lease = cache
            .acquire(
                SparseExpertId::new(0, 0).unwrap(),
                &source(exact, &payloads),
                &PlacementCancellation::new(),
            )
            .unwrap();
        let aggregate_before = pool.used_bytes().unwrap();
        let mut staging = lease.allocate_execution_staging(32).unwrap();
        assert_eq!(staging.as_slice(), &[0; 32]);
        staging.as_mut_slice()[0] = 7;
        assert_eq!(staging.reserved_bytes(), 32);
        assert_eq!(pool.used_bytes().unwrap(), aggregate_before);
        let metrics = cache.metrics().unwrap();
        assert_eq!(metrics.staging_bytes, 32);
        assert_eq!(metrics.peak_staging_bytes, 32);
        assert_eq!(metrics.staging_allocation_count, 1);
        assert!(matches!(
            lease.allocate_execution_staging(1_024),
            Err(PlacementError::ResourceExhausted(_))
        ));

        cache.invalidate().unwrap();
        assert!(matches!(
            lease.allocate_execution_staging(8),
            Err(PlacementError::ArtifactIdentityMismatch)
        ));
        drop(lease);
        assert_eq!(pool.used_bytes().unwrap(), aggregate_before);
        drop(staging);
        assert_eq!(pool.used_bytes().unwrap(), 0);
        let metrics = cache.metrics().unwrap();
        assert_eq!(metrics.staging_bytes, 0);
        assert_eq!(metrics.staging_allocation_count, 0);
        assert_eq!(metrics.peak_staging_bytes, 32);
    }

    #[test]
    fn invalidated_pinned_entry_is_released_after_its_last_lease() {
        let payloads = vec![vec![vec![8; 8]]];
        let exact = identity(8);
        let plan = plan(
            exact.clone(),
            &payloads,
            SparseExpertResidencyMode::ExpertStreamed,
            SPARSE_EXPERT_CACHE_SLOT_BYTES + 8,
            0,
        );
        let pool = MemoryPool::new(4_096).unwrap();
        let cache = SparseExpertCache::new(plan, &pool).unwrap();
        let source = source(exact, &payloads);
        let cancellation = PlacementCancellation::new();
        let lease =
            cache.acquire(SparseExpertId::new(0, 0).unwrap(), &source, &cancellation).unwrap();
        cache.invalidate().unwrap();
        assert_eq!(cache.metrics().unwrap().resident_bytes, SPARSE_EXPERT_CACHE_SLOT_BYTES + 8);
        drop(lease);
        assert_eq!(cache.metrics().unwrap().resident_bytes, 0);
        assert_eq!(pool.used_bytes().unwrap(), 0);
    }

    #[test]
    fn cancellation_during_prefetch_releases_partial_state() {
        let payloads = vec![vec![vec![9; 8]], vec![vec![10; 8]]];
        let exact = identity(9);
        let plan = plan(
            exact.clone(),
            &payloads,
            SparseExpertResidencyMode::ExpertStreamed,
            2 * SPARSE_EXPERT_CACHE_SLOT_BYTES + 2 * 8,
            1,
        );
        let selection =
            SparseExpertSelection::new(&plan, 1, vec![SparseExpertId::new(1, 0).unwrap()]).unwrap();
        let pool = MemoryPool::new(4_096).unwrap();
        let cache = SparseExpertCache::new(plan, &pool).unwrap();
        let entered = Arc::new(Barrier::new(2));
        let release = Arc::new(Barrier::new(2));
        let source = BlockingSource {
            inner: source(exact, &payloads),
            entered: Arc::clone(&entered),
            release: Arc::clone(&release),
        };
        let cancellation = PlacementCancellation::new();
        thread::scope(|scope| {
            let handle = scope.spawn(|| {
                run_bounded_expert_prefetch(&cache, &selection, &source, &cancellation, |_probe| {
                    Ok::<_, EngineError>(())
                })
            });
            entered.wait();
            cancellation.cancel();
            release.wait();
            assert!(matches!(handle.join().unwrap(), Err(PlacementError::Cancelled)));
        });
        let metrics = cache.metrics().unwrap();
        assert_eq!(metrics.entry_count, 0);
        assert_eq!(metrics.resident_bytes, 2 * SPARSE_EXPERT_CACHE_SLOT_BYTES);
        assert_eq!(pool.used_bytes().unwrap(), 2_064);
        cache.invalidate().unwrap();
        assert_eq!(pool.used_bytes().unwrap(), 0);
    }
}
