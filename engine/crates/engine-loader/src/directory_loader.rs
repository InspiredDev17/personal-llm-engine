//! Production-only admission for signed directory-backed `.paimodel` packages.

use crate::{
    ModelRegistry, RegisteredModel, RegistrationMode,
    qwen_registry_v3::{descriptor_member, sha384_hex},
    read_only_map::ReadOnlyFileMap,
};
use personalai_engine_crypto::{
    PackageHashAccumulator, hash_package, hash_signed_content, verify_directory_package_signature,
};
use personalai_engine_memory::{MemoryOwner, MemoryPool, MemoryReservation};
use personalai_engine_model_format::{
    DIRECTORY_DECODER_ARCHITECTURE_ID, DIRECTORY_QWEN2_DECODER_ARCHITECTURE_ID, DirectoryManifest,
    DirectoryTensorRecord, HashCatalog, PackageFileKind, Quantization, SignatureEnvelope,
};
use personalai_engine_quantization::QuantizedTensor;
use personalai_engine_tensor::Tensor;
use personalai_engine_types::{
    DType, EngineError, MAX_MODEL_PACKAGE_BYTES, ModelId, ModelPackageRejection, ResourceBudget,
    SecurityEvent, TensorShape,
};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File, OpenOptions};
use std::io::Read;
use std::os::unix::fs::{MetadataExt, OpenOptionsExt, PermissionsExt};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};

const HASH_CATALOG_FILE: &str = "hashes.aiph";
const SIGNATURE_FILE: &str = "signature.ed25519";
const MAX_HASH_CATALOG_BYTES: u64 = 256 * 1024;
const MAX_PROVENANCE_BYTES: u64 = 1024 * 1024;
const DIRECTORY_HASH_CHUNK_BYTES: usize = 1024 * 1024;
const DIRECTORY_COPY_CHUNK_BYTES: usize = 1024 * 1024;
/// Runtime version enforced by ordinary directory-package admission.
///
/// Metadata readiness inspection reports this value but cannot override it.
/// In particular, inspecting a future package never makes that package
/// executable by the current service.
pub const DIRECTORY_LOADER_ENGINE_VERSION: (u16, u16, u16) = (0, 2, 0);
const QWEN25_15B_VOCABULARY: u32 = 151_936;
const QWEN25_15B_HIDDEN: u32 = 1_536;
const QWEN25_15B_LAYERS: u32 = 28;
const QWEN25_15B_ATTENTION_HEADS: u32 = 12;
const QWEN25_15B_KV_HEADS: u32 = 2;
const QWEN25_15B_FEED_FORWARD: u32 = 8_960;
const QWEN25_RMS_EPSILON: f32 = 1.0e-6;
const QWEN25_STANDARD_CONTEXT: u32 = 32_768;
const QWEN25_STANDARD_ROPE_THETA: f32 = 1_000_000.0;
const QWEN25_MATH_CONTEXT: u32 = 4_096;
const QWEN25_MATH_ROPE_THETA: f32 = 10_000.0;

/// Mandatory typed event destination for production package validation.
pub trait DirectorySecurityEventSink {
    /// Receives one sanitized model-package rejection event.
    fn emit(&mut self, event: SecurityEvent);
}

/// Read-only cooperative cancellation observed during package loading.
///
/// This loader-owned trait avoids importing placement or service authority
/// into package verification. The ordinary synchronous API uses an internal
/// never-cancelled probe; owner composition can bind an existing request flag.
pub trait DirectoryLoadCancellationProbe: Sync {
    /// Returns whether the caller has cancelled the in-progress load.
    fn is_cancelled(&self) -> bool;
}

impl DirectoryLoadCancellationProbe for AtomicBool {
    fn is_cancelled(&self) -> bool {
        self.load(Ordering::Acquire)
    }
}

struct NeverCancelled;

impl DirectoryLoadCancellationProbe for NeverCancelled {
    fn is_cancelled(&self) -> bool {
        false
    }
}

impl<F> DirectorySecurityEventSink for F
where
    F: FnMut(SecurityEvent),
{
    fn emit(&mut self, event: SecurityEvent) {
        self(event);
    }
}

/// One immutable mapped shard admitted through every package gate.
#[derive(Debug)]
struct VerifiedShard {
    map: ReadOnlyFileMap,
    expected_hash: personalai_engine_types::ModelPackageHash,
}

/// Signed, hash-verified, architecture-validated directory package.
#[derive(Debug)]
pub struct VerifiedDirectoryPackage {
    registration: RegisteredModel,
    manifest: DirectoryManifest,
    tokenizer: Vec<u8>,
    provenance: Vec<u8>,
    shards: Vec<VerifiedShard>,
}

/// Cryptographically verified, metadata-only directory-package evidence.
///
/// The signed catalog, manifest, tokenizer, provenance, closed file set, and
/// shard identities have been checked. Tensor-shard contents have deliberately
/// not been mapped or hashed, so this type is not package admission,
/// qualification, or execution authority. It exists only to make a future
/// package's remaining blockers inspectable without weakening ordinary load.
#[derive(Debug)]
pub struct VerifiedDirectoryReadinessMetadata {
    registration: RegisteredModel,
    manifest: DirectoryManifest,
    tokenizer: Vec<u8>,
    provenance_hash: personalai_engine_types::ModelPackageHash,
}

impl VerifiedDirectoryReadinessMetadata {
    /// Returns the exact trusted local registration used by inspection.
    pub const fn registration(&self) -> &RegisteredModel {
        &self.registration
    }

    /// Returns the canonical, signed package manifest.
    pub const fn manifest(&self) -> &DirectoryManifest {
        &self.manifest
    }

    /// Returns the exact tokenizer bytes authenticated by manifest and catalog.
    ///
    /// The readiness consumer may parse these bytes under its own bounded
    /// memory pool. Ordinary loader admission remains unchanged.
    pub fn tokenizer_bytes(&self) -> &[u8] {
        &self.tokenizer
    }

    /// Returns the exact verified provenance digest without exposing content.
    pub const fn provenance_hash(&self) -> personalai_engine_types::ModelPackageHash {
        self.provenance_hash
    }
}

impl VerifiedDirectoryPackage {
    /// Returns the trusted local registration.
    pub const fn registration(&self) -> &RegisteredModel {
        &self.registration
    }

    /// Returns the parsed canonical manifest.
    pub const fn manifest(&self) -> &DirectoryManifest {
        &self.manifest
    }

    /// Returns immutable verified tokenizer bytes.
    pub fn tokenizer_bytes(&self) -> &[u8] {
        &self.tokenizer
    }

    /// Returns immutable verified provenance bytes.
    pub fn provenance_bytes(&self) -> &[u8] {
        &self.provenance
    }

    /// Revalidates every live mapping and emits a typed event on change.
    pub fn revalidate_mappings(
        &self,
        events: &mut impl DirectorySecurityEventSink,
    ) -> Result<(), EngineError> {
        self.revalidate_mappings_cancellable(&NeverCancelled, events)
    }

    /// Revalidates every live mapping with bounded cancellation checkpoints.
    pub fn revalidate_mappings_cancellable(
        &self,
        cancellation: &dyn DirectoryLoadCancellationProbe,
        events: &mut impl DirectorySecurityEventSink,
    ) -> Result<(), EngineError> {
        let result = self.validate_mapping_hashes_cancellable(cancellation);
        if let Err(error) = &result {
            if !matches!(error, EngineError::Cancellation(_)) {
                events.emit(SecurityEvent::ModelPackageRejected {
                    model_id: self.registration.model_id().clone(),
                    reason: rejection_reason(error),
                });
            }
        }
        result
    }

    /// Returns one immutable verified mapped tensor range.
    pub fn tensor_bytes(&self, tensor: &DirectoryTensorRecord) -> Result<&[u8], EngineError> {
        self.tensor_bytes_revalidated(tensor, &NeverCancelled)
    }

    /// Consumes package ownership, immediately dropping every live mapping.
    pub fn unload(self) {
        drop(self);
    }

    fn validate_mapping_hashes_cancellable(
        &self,
        cancellation: &dyn DirectoryLoadCancellationProbe,
    ) -> Result<(), EngineError> {
        self.shards.iter().try_for_each(|shard| {
            if shard.map.identity_is_unchanged()?
                && hash_package_cancellable(shard.map.bytes(), cancellation)? == shard.expected_hash
            {
                Ok(())
            } else {
                Err(EngineError::signature_failure(
                    "mapped tensor shard changed after package admission",
                ))
            }
        })
    }

    fn tensor_bytes_revalidated(
        &self,
        tensor: &DirectoryTensorRecord,
        cancellation: &dyn DirectoryLoadCancellationProbe,
    ) -> Result<&[u8], EngineError> {
        check_cancelled(cancellation)?;
        let shard = self
            .shards
            .get(usize::try_from(tensor.shard_index()).map_err(|_| {
                EngineError::resource_exhaustion("tensor shard index is not addressable")
            })?)
            .ok_or_else(|| EngineError::internal_invariant("validated tensor shard is missing"))?;
        if !shard.map.identity_is_unchanged()? {
            return Err(EngineError::signature_failure(
                "mapped tensor shard identity changed after package admission",
            ));
        }
        let bytes = Self::tensor_bytes_from_verified_shard(shard, tensor)?;
        if hash_package_cancellable(bytes, cancellation)? != tensor.content_hash() {
            return Err(EngineError::signature_failure(
                "mapped tensor content changed after package admission",
            ));
        }
        Ok(bytes)
    }

    fn tensor_bytes_from_verified_shard<'a>(
        shard: &'a VerifiedShard,
        tensor: &DirectoryTensorRecord,
    ) -> Result<&'a [u8], EngineError> {
        let start = usize::try_from(tensor.byte_offset())
            .map_err(|_| EngineError::resource_exhaustion("tensor offset is not addressable"))?;
        let length = usize::try_from(tensor.byte_length())
            .map_err(|_| EngineError::resource_exhaustion("tensor length is not addressable"))?;
        let end = start
            .checked_add(length)
            .ok_or_else(|| EngineError::resource_exhaustion("tensor range overflows"))?;
        shard
            .map
            .bytes()
            .get(start..end)
            .ok_or_else(|| EngineError::internal_invariant("validated tensor range is unavailable"))
    }

    /// Copies mapped tensor bytes into aggregate-accounted CPU storage.
    pub fn materialize_tensors(
        &self,
        pool: &MemoryPool,
    ) -> Result<LoadedDirectoryTensors, EngineError> {
        self.materialize_tensors_cancellable(pool, &NeverCancelled)
    }

    /// Copies all mapped tensors while polling bounded verification and conversion work.
    pub fn materialize_tensors_cancellable(
        &self,
        pool: &MemoryPool,
        cancellation: &dyn DirectoryLoadCancellationProbe,
    ) -> Result<LoadedDirectoryTensors, EngineError> {
        self.validate_mapping_hashes_cancellable(cancellation)?;
        check_cancelled(cancellation)?;
        let metadata_bytes = u64::try_from(self.manifest.tensors().len())
            .ok()
            .and_then(|count| count.checked_mul(192))
            .ok_or_else(|| EngineError::resource_exhaustion("tensor metadata bound overflows"))?;
        let metadata_pool =
            pool.scoped(MemoryOwner::ModelMetadata(self.registration.model_id().clone()));
        let weight_pool =
            pool.scoped(MemoryOwner::ModelWeights(self.registration.model_id().clone()));
        let metadata_reservation = metadata_pool.reserve(metadata_bytes)?;
        let mut tensors = BTreeMap::new();
        for record in self.manifest.tensors() {
            check_cancelled(cancellation)?;
            let tensor = self.materialize_tensor(record, &weight_pool, cancellation)?;
            if tensors.insert(record.name().to_owned(), tensor).is_some() {
                return Err(EngineError::internal_invariant(
                    "validated directory manifest contains duplicate tensor name",
                ));
            }
        }
        if self.manifest.tied_output()
            && self.manifest.architecture_id().as_str() == DIRECTORY_DECODER_ARCHITECTURE_ID
        {
            let embedding = self
                .manifest
                .tensors()
                .iter()
                .find(|record| record.name() == "token_embedding.weight")
                .ok_or_else(|| EngineError::internal_invariant("tied embedding is missing"))?;
            let output = self.materialize_tensor(embedding, &weight_pool, cancellation)?;
            if tensors.insert("output.weight".to_owned(), output).is_some() {
                return Err(EngineError::internal_invariant(
                    "tied output conflicts with explicit output tensor",
                ));
            }
        }
        Ok(LoadedDirectoryTensors { tensors, _metadata_reservation: metadata_reservation })
    }

    /// Materializes one trusted architecture-selected tensor subset.
    ///
    /// Names must originate from a validated placement plan. Protocol and
    /// request schemas do not expose this API, offsets, or package paths.
    pub fn materialize_named_tensors(
        &self,
        names: &[String],
        pool: &MemoryPool,
    ) -> Result<LoadedDirectoryTensors, EngineError> {
        self.materialize_named_tensors_cancellable(names, pool, &NeverCancelled)
    }

    /// Materializes one placement-selected subset with cooperative cancellation.
    pub fn materialize_named_tensors_cancellable(
        &self,
        names: &[String],
        pool: &MemoryPool,
        cancellation: &dyn DirectoryLoadCancellationProbe,
    ) -> Result<LoadedDirectoryTensors, EngineError> {
        check_cancelled(cancellation)?;
        if names.is_empty() || names.len() > self.manifest.tensors().len() {
            return Err(EngineError::invalid_input(
                "placement tensor selection is empty or exceeds the manifest",
            ));
        }
        let mut selected = BTreeSet::new();
        for name in names {
            if !selected.insert(name.as_str()) {
                return Err(EngineError::invalid_input(
                    "placement tensor selection contains a duplicate",
                ));
            }
        }
        let metadata_bytes =
            u64::try_from(names.len()).ok().and_then(|count| count.checked_mul(192)).ok_or_else(
                || EngineError::resource_exhaustion("placement metadata bound overflows"),
            )?;
        let metadata_pool =
            pool.scoped(MemoryOwner::ModelMetadata(self.registration.model_id().clone()));
        let tensor_pool =
            pool.scoped(MemoryOwner::PlacementStaging(self.registration.model_id().clone()));
        let metadata_reservation = metadata_pool.reserve(metadata_bytes)?;
        let mut tensors = BTreeMap::new();
        for name in names {
            check_cancelled(cancellation)?;
            let record =
                self.manifest.tensors().iter().find(|record| record.name() == name).ok_or_else(
                    || {
                        EngineError::internal_invariant(
                            "placement plan names a tensor absent from the verified manifest",
                        )
                    },
                )?;
            let tensor = self.materialize_tensor(record, &tensor_pool, cancellation)?;
            if tensors.insert(name.clone(), tensor).is_some() {
                return Err(EngineError::internal_invariant(
                    "placement materialization produced a duplicate tensor",
                ));
            }
        }
        Ok(LoadedDirectoryTensors { tensors, _metadata_reservation: metadata_reservation })
    }

    fn materialize_tensor(
        &self,
        record: &DirectoryTensorRecord,
        pool: &MemoryPool,
        cancellation: &dyn DirectoryLoadCancellationProbe,
    ) -> Result<Tensor, EngineError> {
        let live_bytes = self.tensor_bytes_revalidated(record, cancellation)?;
        let mut snapshot_reservation = pool.reserve(record.byte_length())?;
        let mut snapshot = Vec::new();
        snapshot
            .try_reserve_exact(live_bytes.len())
            .map_err(|_| EngineError::resource_exhaustion("tensor snapshot allocation failed"))?;
        let allocated_bytes = u64::try_from(snapshot.capacity()).map_err(|_| {
            EngineError::resource_exhaustion("tensor snapshot capacity is not representable")
        })?;
        if allocated_bytes > record.byte_length() {
            pool.grow_reservation(
                &mut snapshot_reservation,
                allocated_bytes - record.byte_length(),
            )?;
        }
        for chunk in live_bytes.chunks(DIRECTORY_COPY_CHUNK_BYTES) {
            check_cancelled(cancellation)?;
            snapshot.extend_from_slice(chunk);
        }
        check_cancelled(cancellation)?;
        if hash_package_cancellable(&snapshot, cancellation)? != record.content_hash() {
            return Err(EngineError::signature_failure(
                "snapshotted tensor content differs from the signed manifest",
            ));
        }
        match record.quantization() {
            Quantization::None => Tensor::convert_storage_to_f32_cancellable(
                record.dtype(),
                record.shape().clone(),
                &snapshot,
                pool,
                || cancellation.is_cancelled(),
            ),
            Quantization::AipQ8V1 | Quantization::AipQ4V1 => {
                QuantizedTensor::parse(record.quantization(), record.shape(), &snapshot)?
                    .dequantize_cancellable(pool, || cancellation.is_cancelled())
            }
        }
    }
}

/// Aggregate-accounted CPU tensors copied from verified read-only mappings.
#[derive(Debug)]
pub struct LoadedDirectoryTensors {
    tensors: BTreeMap<String, Tensor>,
    _metadata_reservation: MemoryReservation,
}

impl LoadedDirectoryTensors {
    /// Borrows the immutable architecture-selected tensor map.
    pub const fn tensors(&self) -> &BTreeMap<String, Tensor> {
        &self.tensors
    }

    /// Transfers the complete tensor map to the architecture implementation.
    pub fn into_tensors(self) -> BTreeMap<String, Tensor> {
        self.tensors
    }
}

pub(crate) fn load(
    registry: &ModelRegistry,
    model_id: &ModelId,
    budget: &ResourceBudget,
    events: &mut impl DirectorySecurityEventSink,
) -> Result<VerifiedDirectoryPackage, EngineError> {
    load_cancellable(registry, model_id, budget, &NeverCancelled, events)
}

pub(crate) fn load_cancellable(
    registry: &ModelRegistry,
    model_id: &ModelId,
    budget: &ResourceBudget,
    cancellation: &dyn DirectoryLoadCancellationProbe,
    events: &mut impl DirectorySecurityEventSink,
) -> Result<VerifiedDirectoryPackage, EngineError> {
    let result = load_inner(registry, model_id, budget, cancellation);
    if let Err(error) = &result {
        if !matches!(error, EngineError::Cancellation(_)) {
            events.emit(SecurityEvent::ModelPackageRejected {
                model_id: model_id.clone(),
                reason: rejection_reason(error),
            });
        }
    }
    result
}

/// Verifies signed package metadata without bypassing runtime admission.
pub(crate) fn inspect_readiness_metadata(
    registry: &ModelRegistry,
    model_id: &ModelId,
    budget: &ResourceBudget,
    events: &mut impl DirectorySecurityEventSink,
) -> Result<VerifiedDirectoryReadinessMetadata, EngineError> {
    let result = inspect_readiness_metadata_inner(registry, model_id, budget);
    if let Err(error) = &result {
        events.emit(SecurityEvent::ModelPackageRejected {
            model_id: model_id.clone(),
            reason: rejection_reason(error),
        });
    }
    result
}

#[allow(
    clippy::too_many_lines,
    reason = "metadata readiness repeats the signed envelope gates while deliberately omitting tensor content admission"
)]
fn inspect_readiness_metadata_inner(
    registry: &ModelRegistry,
    model_id: &ModelId,
    budget: &ResourceBudget,
) -> Result<VerifiedDirectoryReadinessMetadata, EngineError> {
    let registration = registry.resolve(model_id)?.clone();
    validate_root(registry.model_root())?;
    let artifact_metadata = validate_registry_artifact_parent(registry, &registration)?;
    let package_metadata = fs::symlink_metadata(registration.package_path())
        .map_err(|_| EngineError::invalid_input("registered directory package is unavailable"))?;
    if !package_metadata.is_dir()
        || package_metadata.file_type().is_symlink()
        || package_metadata.permissions().mode() & 0o222 != 0
    {
        return Err(EngineError::authentication_failure(
            "registered package must be a protected non-symlink directory",
        ));
    }

    let catalog_bytes = read_direct_file(
        registration.package_path(),
        HASH_CATALOG_FILE,
        MAX_HASH_CATALOG_BYTES.min(budget.max_memory_bytes()),
    )?;
    if hash_package(&catalog_bytes) != *registration.expected_hash() {
        return Err(EngineError::signature_failure(
            "directory package identity does not match local configuration",
        ));
    }
    validate_models_v2_member(
        &registration,
        "catalog",
        hash_package(&catalog_bytes),
        catalog_bytes.len(),
    )?;
    let catalog = HashCatalog::parse(&catalog_bytes)?;
    let catalog_length = usize_to_u64(catalog_bytes.len())?;
    ensure_entry_budget(0, catalog_length, budget)?;
    let signature_maximum = budget.max_memory_bytes().saturating_sub(catalog_length).min(96);
    let signature_bytes =
        read_direct_file(registration.package_path(), SIGNATURE_FILE, signature_maximum)?;
    let signature = SignatureEnvelope::parse(&signature_bytes)?;
    validate_models_v2_member(
        &registration,
        "signature",
        hash_package(&signature_bytes),
        signature_bytes.len(),
    )?;
    verify_directory_package_signature(
        registration.trusted_signer(),
        signature.key_id(),
        &hash_signed_content(&catalog_bytes),
        signature.signature(),
    )?;
    validate_exact_file_set(registration.package_path(), &catalog)?;

    let mut total_bytes = catalog_length
        .checked_add(usize_to_u64(signature_bytes.len())?)
        .ok_or_else(|| EngineError::resource_exhaustion("directory package size overflows"))?;
    let manifest_entry = required_entry(&catalog, "manifest.aipm")?;
    ensure_entry_budget(total_bytes, manifest_entry.byte_length(), budget)?;
    let manifest_bytes = read_catalog_file(registration.package_path(), manifest_entry)?;
    let manifest = DirectoryManifest::parse(&manifest_bytes)?;
    if manifest.model_id() != model_id {
        return Err(EngineError::authentication_failure(
            "directory manifest model ID differs from local registration",
        ));
    }
    validate_architecture(&manifest)?;
    validate_models_v2_manifest(&registration, &manifest, &manifest_bytes)?;
    if let Some(binding) = registration.registry_binding() {
        if manifest.context_limit().get() != binding.context_capacity {
            return Err(EngineError::authentication_failure(
                "directory manifest context differs from the signed registry decision",
            ));
        }
    }
    total_bytes = checked_total(total_bytes, manifest_bytes.len(), budget)?;

    let tokenizer_entry = required_entry(&catalog, "tokenizer.aiptok")?;
    ensure_entry_budget(total_bytes, tokenizer_entry.byte_length(), budget)?;
    let tokenizer = read_catalog_file(registration.package_path(), tokenizer_entry)?;
    if hash_package(&tokenizer) != manifest.tokenizer_hash() {
        return Err(corrupt("directory tokenizer hash differs from manifest"));
    }
    validate_models_v2_member(
        &registration,
        "tokenizer",
        hash_package(&tokenizer),
        tokenizer.len(),
    )?;
    if let Some(binding) = registration.registry_binding() {
        if manifest.tokenizer_hash() != binding.tokenizer_hash {
            return Err(EngineError::authentication_failure(
                "directory tokenizer identity differs from the signed registry decision",
            ));
        }
    }
    total_bytes = checked_total(total_bytes, tokenizer.len(), budget)?;

    let provenance_entry = required_entry(&catalog, "provenance.json")?;
    ensure_entry_budget(total_bytes, provenance_entry.byte_length(), budget)?;
    let provenance = read_catalog_file(registration.package_path(), provenance_entry)?;
    validate_provenance(&provenance, &registration)?;
    let provenance_hash = hash_package(&provenance);
    validate_models_v2_member(&registration, "provenance", provenance_hash, provenance.len())?;
    if let Some(binding) = registration.registry_binding() {
        if provenance_hash != binding.provenance_hash {
            return Err(EngineError::authentication_failure(
                "directory provenance identity differs from the signed registry decision",
            ));
        }
    }
    total_bytes = checked_total(total_bytes, provenance.len(), budget)?;

    if manifest.shards().len()
        != catalog
            .entries()
            .iter()
            .filter(|entry| entry.kind() == PackageFileKind::TensorShard)
            .count()
    {
        return Err(corrupt("manifest and hash catalog shard counts differ"));
    }
    for shard in manifest.shards() {
        let catalog_entry = required_entry(&catalog, &shard.file_name())?;
        if catalog_entry.kind() != PackageFileKind::TensorShard
            || catalog_entry.index() != shard.index()
            || catalog_entry.byte_length() != shard.byte_length()
            || catalog_entry.content_hash() != shard.content_hash()
        {
            return Err(corrupt("manifest and hash catalog shard identities differ"));
        }
        ensure_entry_budget(total_bytes, shard.byte_length(), budget)?;
        total_bytes = total_bytes
            .checked_add(shard.byte_length())
            .ok_or_else(|| EngineError::resource_exhaustion("directory package size overflows"))?;
        let (file, length) = open_direct_file(
            registration.package_path(),
            catalog_entry.name(),
            catalog_entry.byte_length(),
        )?;
        if usize_to_u64(length)? != catalog_entry.byte_length() {
            return Err(corrupt("tensor shard length differs from signed metadata"));
        }
        drop(file);
        validate_models_v2_member(
            &registration,
            &format!("tensor_shard.{:04}", shard.index()),
            shard.content_hash(),
            usize::try_from(shard.byte_length()).map_err(|_| {
                EngineError::resource_exhaustion("tensor shard length is not addressable")
            })?,
        )?;
    }
    validate_models_v2_total(&registration, total_bytes)?;
    if let Some(binding) = registration.registry_binding() {
        let mut observed = manifest
            .shards()
            .iter()
            .map(personalai_engine_model_format::DirectoryShardRecord::content_hash)
            .collect::<Vec<_>>();
        observed.sort_unstable();
        if observed != binding.tensor_shard_hashes {
            return Err(EngineError::authentication_failure(
                "directory tensor-shard identities differ from the signed registry decision",
            ));
        }
    }

    let after = fs::symlink_metadata(registration.package_path())
        .map_err(|_| EngineError::authentication_failure("package changed during inspection"))?;
    if after.dev() != package_metadata.dev() || after.ino() != package_metadata.ino() {
        return Err(EngineError::authentication_failure(
            "package directory changed during inspection",
        ));
    }
    if let Some(before) = artifact_metadata {
        let artifact = registration
            .package_path()
            .parent()
            .ok_or_else(|| EngineError::internal_invariant("registry artifact parent is absent"))?;
        let after = fs::symlink_metadata(artifact)
            .map_err(|_| EngineError::authentication_failure("artifact directory changed"))?;
        if after.dev() != before.dev() || after.ino() != before.ino() {
            return Err(EngineError::authentication_failure(
                "artifact directory changed during metadata inspection",
            ));
        }
    }
    let _ = total_bytes;
    Ok(VerifiedDirectoryReadinessMetadata { registration, manifest, tokenizer, provenance_hash })
}

#[allow(clippy::too_many_lines)]
fn load_inner(
    registry: &ModelRegistry,
    model_id: &ModelId,
    budget: &ResourceBudget,
    cancellation: &dyn DirectoryLoadCancellationProbe,
) -> Result<VerifiedDirectoryPackage, EngineError> {
    check_cancelled(cancellation)?;
    let registration = registry.resolve(model_id)?.clone();
    validate_root(registry.model_root())?;
    let artifact_metadata = validate_registry_artifact_parent(registry, &registration)?;
    let package_metadata = fs::symlink_metadata(registration.package_path())
        .map_err(|_| EngineError::invalid_input("registered directory package is unavailable"))?;
    if !package_metadata.is_dir()
        || package_metadata.file_type().is_symlink()
        || package_metadata.permissions().mode() & 0o222 != 0
    {
        return Err(EngineError::authentication_failure(
            "registered package must be a protected non-symlink directory",
        ));
    }

    let catalog_bytes = read_direct_file(
        registration.package_path(),
        HASH_CATALOG_FILE,
        MAX_HASH_CATALOG_BYTES.min(budget.max_memory_bytes()),
    )?;
    if hash_package(&catalog_bytes) != *registration.expected_hash() {
        return Err(EngineError::signature_failure(
            "directory package identity does not match local configuration",
        ));
    }
    validate_models_v2_member(
        &registration,
        "catalog",
        hash_package(&catalog_bytes),
        catalog_bytes.len(),
    )?;
    let catalog = HashCatalog::parse(&catalog_bytes)?;
    let catalog_length = usize_to_u64(catalog_bytes.len())?;
    ensure_entry_budget(0, catalog_length, budget)?;
    let signature_maximum = budget.max_memory_bytes().saturating_sub(catalog_length).min(96);
    let signature_bytes =
        read_direct_file(registration.package_path(), SIGNATURE_FILE, signature_maximum)?;
    let signature = SignatureEnvelope::parse(&signature_bytes)?;
    validate_models_v2_member(
        &registration,
        "signature",
        hash_package(&signature_bytes),
        signature_bytes.len(),
    )?;
    verify_directory_package_signature(
        registration.trusted_signer(),
        signature.key_id(),
        &hash_signed_content(&catalog_bytes),
        signature.signature(),
    )?;
    check_cancelled(cancellation)?;
    validate_exact_file_set(registration.package_path(), &catalog)?;

    let mut total_bytes = catalog_length
        .checked_add(usize_to_u64(signature_bytes.len())?)
        .ok_or_else(|| EngineError::resource_exhaustion("directory package size overflows"))?;
    let manifest_entry = required_entry(&catalog, "manifest.aipm")?;
    ensure_entry_budget(total_bytes, manifest_entry.byte_length(), budget)?;
    let manifest_bytes = read_catalog_file(registration.package_path(), manifest_entry)?;
    let manifest = DirectoryManifest::parse(&manifest_bytes)?;
    if manifest.model_id() != model_id {
        return Err(EngineError::authentication_failure(
            "directory manifest model ID differs from local registration",
        ));
    }
    if manifest.required_engine_version() > DIRECTORY_LOADER_ENGINE_VERSION {
        return Err(EngineError::not_implemented(
            "directory package requires a newer engine version",
        ));
    }
    check_cancelled(cancellation)?;
    validate_architecture(&manifest)?;
    validate_models_v2_manifest(&registration, &manifest, &manifest_bytes)?;
    if let Some(binding) = registration.registry_binding() {
        if manifest.context_limit().get() != binding.context_capacity {
            return Err(EngineError::authentication_failure(
                "directory manifest context differs from the signed registry decision",
            ));
        }
    }
    total_bytes = checked_total(total_bytes, manifest_bytes.len(), budget)?;

    let tokenizer_entry = required_entry(&catalog, "tokenizer.aiptok")?;
    ensure_entry_budget(total_bytes, tokenizer_entry.byte_length(), budget)?;
    let tokenizer = read_catalog_file(registration.package_path(), tokenizer_entry)?;
    if hash_package(&tokenizer) != manifest.tokenizer_hash() {
        return Err(corrupt("directory tokenizer hash differs from manifest"));
    }
    validate_models_v2_member(
        &registration,
        "tokenizer",
        hash_package(&tokenizer),
        tokenizer.len(),
    )?;
    if let Some(binding) = registration.registry_binding() {
        if manifest.tokenizer_hash() != binding.tokenizer_hash {
            return Err(EngineError::authentication_failure(
                "directory tokenizer identity differs from the signed registry decision",
            ));
        }
    }
    total_bytes = checked_total(total_bytes, tokenizer.len(), budget)?;
    let provenance_entry = required_entry(&catalog, "provenance.json")?;
    ensure_entry_budget(total_bytes, provenance_entry.byte_length(), budget)?;
    let provenance = read_catalog_file(registration.package_path(), provenance_entry)?;
    validate_provenance(&provenance, &registration)?;
    validate_models_v2_member(
        &registration,
        "provenance",
        hash_package(&provenance),
        provenance.len(),
    )?;
    if let Some(binding) = registration.registry_binding() {
        if hash_package(&provenance) != binding.provenance_hash {
            return Err(EngineError::authentication_failure(
                "directory provenance identity differs from the signed registry decision",
            ));
        }
    }
    total_bytes = checked_total(total_bytes, provenance.len(), budget)?;

    if manifest.shards().len()
        != catalog
            .entries()
            .iter()
            .filter(|entry| entry.kind() == PackageFileKind::TensorShard)
            .count()
    {
        return Err(corrupt("manifest and hash catalog shard counts differ"));
    }
    let mut shards = Vec::new();
    shards
        .try_reserve_exact(manifest.shards().len())
        .map_err(|_| EngineError::resource_exhaustion("mapped shard allocation failed"))?;
    for shard in manifest.shards() {
        check_cancelled(cancellation)?;
        let name = shard.file_name();
        let catalog_entry = required_entry(&catalog, &name)?;
        if catalog_entry.kind() != PackageFileKind::TensorShard
            || catalog_entry.index() != shard.index()
            || catalog_entry.byte_length() != shard.byte_length()
            || catalog_entry.content_hash() != shard.content_hash()
        {
            return Err(corrupt("manifest and hash catalog shard identities differ"));
        }
        ensure_entry_budget(total_bytes, shard.byte_length(), budget)?;
        total_bytes = checked_total(
            total_bytes,
            usize::try_from(shard.byte_length())
                .map_err(|_| EngineError::resource_exhaustion("shard size is not addressable"))?,
            budget,
        )?;
        let map = map_catalog_file(registration.package_path(), catalog_entry)?;
        if hash_package_cancellable(map.bytes(), cancellation)? != shard.content_hash() {
            return Err(EngineError::signature_failure("tensor shard hash verification failed"));
        }
        validate_models_v2_member(
            &registration,
            &format!("tensor_shard.{:04}", shard.index()),
            shard.content_hash(),
            usize::try_from(shard.byte_length()).map_err(|_| {
                EngineError::resource_exhaustion("tensor shard length is not addressable")
            })?,
        )?;
        shards.push(VerifiedShard { map, expected_hash: shard.content_hash() });
    }
    validate_models_v2_total(&registration, total_bytes)?;
    if let Some(binding) = registration.registry_binding() {
        let mut observed = manifest
            .shards()
            .iter()
            .map(personalai_engine_model_format::DirectoryShardRecord::content_hash)
            .collect::<Vec<_>>();
        observed.sort_unstable();
        if observed != binding.tensor_shard_hashes {
            return Err(EngineError::authentication_failure(
                "directory tensor-shard identities differ from the signed registry decision",
            ));
        }
    }
    let package =
        VerifiedDirectoryPackage { registration, manifest, tokenizer, provenance, shards };
    validate_tensor_hashes(&package, cancellation)?;
    check_cancelled(cancellation)?;
    let after = fs::symlink_metadata(package.registration.package_path())
        .map_err(|_| EngineError::authentication_failure("package changed during verification"))?;
    if after.dev() != package_metadata.dev() || after.ino() != package_metadata.ino() {
        return Err(EngineError::authentication_failure("package directory changed during load"));
    }
    if let Some(before) = artifact_metadata {
        let artifact =
            package.registration.package_path().parent().ok_or_else(|| {
                EngineError::internal_invariant("registry artifact parent is absent")
            })?;
        let after = fs::symlink_metadata(artifact)
            .map_err(|_| EngineError::authentication_failure("artifact directory changed"))?;
        if after.dev() != before.dev() || after.ino() != before.ino() {
            return Err(EngineError::authentication_failure(
                "artifact directory changed during package load",
            ));
        }
    }
    let _ = total_bytes;
    Ok(package)
}

fn validate_registry_artifact_parent(
    registry: &ModelRegistry,
    registration: &RegisteredModel,
) -> Result<Option<fs::Metadata>, EngineError> {
    let Some(artifact_id) = registration.registry_artifact_id() else {
        return Ok(None);
    };
    let artifact = registry.model_root().join(artifact_id);
    let expected_package = artifact.join(format!(
        "{}.{}",
        registration.model_id().as_str(),
        crate::MODEL_PACKAGE_EXTENSION
    ));
    if registration.package_path() != expected_package {
        return Err(EngineError::internal_invariant(
            "registry-backed package path is not canonically derived",
        ));
    }
    let metadata = fs::symlink_metadata(&artifact)
        .map_err(|_| EngineError::invalid_input("registry artifact directory is unavailable"))?;
    if !metadata.is_dir()
        || metadata.file_type().is_symlink()
        || metadata.permissions().mode() & 0o222 != 0
    {
        return Err(EngineError::authentication_failure(
            "registry artifact must be a protected non-symlink directory",
        ));
    }
    Ok(Some(metadata))
}

fn validate_root(root: &Path) -> Result<(), EngineError> {
    let metadata = fs::symlink_metadata(root)
        .map_err(|_| EngineError::invalid_input("configured model root is unavailable"))?;
    if !metadata.is_dir()
        || metadata.file_type().is_symlink()
        || metadata.permissions().mode() & 0o222 != 0
    {
        return Err(EngineError::authentication_failure(
            "model root must be a protected non-symlink directory",
        ));
    }
    Ok(())
}

fn validate_exact_file_set(directory: &Path, catalog: &HashCatalog) -> Result<(), EngineError> {
    let mut expected =
        catalog.entries().iter().map(|entry| entry.name().to_owned()).collect::<BTreeSet<_>>();
    expected.insert(HASH_CATALOG_FILE.to_owned());
    expected.insert(SIGNATURE_FILE.to_owned());
    let mut observed = BTreeSet::new();
    for entry in fs::read_dir(directory)
        .map_err(|_| EngineError::authentication_failure("package directory read failed"))?
    {
        let entry = entry
            .map_err(|_| EngineError::authentication_failure("package directory entry failed"))?;
        let name = entry
            .file_name()
            .into_string()
            .map_err(|_| corrupt("package filename is invalid UTF-8"))?;
        let metadata = fs::symlink_metadata(entry.path())
            .map_err(|_| EngineError::authentication_failure("package entry metadata failed"))?;
        if !metadata.is_file() || metadata.file_type().is_symlink() {
            return Err(EngineError::authentication_failure(
                "package entries must be non-symlink regular files",
            ));
        }
        observed.insert(name);
    }
    if observed != expected {
        return Err(corrupt("package contains a missing or unexpected trusted file"));
    }
    Ok(())
}

fn required_entry<'a>(
    catalog: &'a HashCatalog,
    name: &str,
) -> Result<&'a personalai_engine_model_format::HashCatalogEntry, EngineError> {
    catalog.entry(name).ok_or_else(|| corrupt("hash catalog omits required file"))
}

fn read_catalog_file(
    directory: &Path,
    entry: &personalai_engine_model_format::HashCatalogEntry,
) -> Result<Vec<u8>, EngineError> {
    let bytes = read_direct_file(directory, entry.name(), entry.byte_length())?;
    if usize_to_u64(bytes.len())? != entry.byte_length()
        || hash_package(&bytes) != entry.content_hash()
    {
        return Err(EngineError::signature_failure("package file hash verification failed"));
    }
    Ok(bytes)
}

fn read_direct_file(directory: &Path, name: &str, maximum: u64) -> Result<Vec<u8>, EngineError> {
    let (mut file, length) = open_direct_file(directory, name, maximum)?;
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(length)
        .map_err(|_| EngineError::resource_exhaustion("package file allocation failed"))?;
    bytes.resize(length, 0);
    file.read_exact(&mut bytes).map_err(|_| corrupt("package file is truncated"))?;
    let mut extra = [0_u8; 1];
    if file.read(&mut extra).map_err(|_| corrupt("package file trailing check failed"))? != 0 {
        return Err(corrupt("package file grew during read"));
    }
    Ok(bytes)
}

fn map_catalog_file(
    directory: &Path,
    entry: &personalai_engine_model_format::HashCatalogEntry,
) -> Result<ReadOnlyFileMap, EngineError> {
    let (file, length) = open_direct_file(directory, entry.name(), entry.byte_length())?;
    if usize_to_u64(length)? != entry.byte_length() {
        return Err(corrupt("mapped shard length differs from hash catalog"));
    }
    ReadOnlyFileMap::new(file, length)
}

fn open_direct_file(
    directory: &Path,
    name: &str,
    maximum: u64,
) -> Result<(File, usize), EngineError> {
    if name.is_empty() || name.contains('/') || name.contains('\\') || name.contains("..") {
        return Err(corrupt("package direct-child filename is invalid"));
    }
    let path = directory.join(name);
    let before = fs::symlink_metadata(&path)
        .map_err(|_| EngineError::invalid_input("registered package file is unavailable"))?;
    if !before.is_file()
        || before.file_type().is_symlink()
        || before.permissions().mode() & 0o222 != 0
    {
        return Err(EngineError::authentication_failure(
            "package file metadata or permissions are invalid",
        ));
    }
    if before.len() == 0 || before.len() > maximum {
        return Err(EngineError::resource_exhaustion(
            "package file exceeds its configured byte limit",
        ));
    }
    let file = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_CLOEXEC | libc::O_NOFOLLOW)
        .open(path)
        .map_err(|_| EngineError::authentication_failure("package file secure open failed"))?;
    let opened = file
        .metadata()
        .map_err(|_| EngineError::authentication_failure("opened package metadata failed"))?;
    if opened.dev() != before.dev()
        || opened.ino() != before.ino()
        || opened.len() != before.len()
        || !opened.is_file()
    {
        return Err(EngineError::authentication_failure("package file changed during secure open"));
    }
    let length = usize::try_from(opened.len())
        .map_err(|_| EngineError::resource_exhaustion("package file is not addressable"))?;
    Ok((file, length))
}

fn validate_architecture(manifest: &DirectoryManifest) -> Result<(), EngineError> {
    let architecture = manifest.architecture_id().as_str();
    if !matches!(
        architecture,
        DIRECTORY_DECODER_ARCHITECTURE_ID | DIRECTORY_QWEN2_DECODER_ARCHITECTURE_ID
    ) {
        return Err(EngineError::unsupported_architecture(manifest.architecture_id().clone()));
    }
    if architecture == DIRECTORY_QWEN2_DECODER_ARCHITECTURE_ID
        && (!manifest.tied_output() || manifest.rope_scaling().to_bits() != 1.0_f32.to_bits())
    {
        return Err(corrupt("Qwen2.5 package requires tied output and unscaled rotary positions"));
    }
    if architecture == DIRECTORY_QWEN2_DECODER_ARCHITECTURE_ID {
        validate_qwen25_15b_profile(manifest)?;
    }
    if !matches!(manifest.storage_dtype(), DType::F32 | DType::F16 | DType::BFloat16)
        || manifest.tensors().iter().any(|tensor| tensor.dtype() != manifest.storage_dtype())
    {
        return Err(corrupt("directory tensor storage dtype is unsupported or inconsistent"));
    }
    let expected = expected_shapes(manifest)?;
    let observed = manifest
        .tensors()
        .iter()
        .map(|tensor| (tensor.name(), tensor.shape()))
        .collect::<BTreeMap<_, _>>();
    if observed.len() != expected.len()
        || expected.iter().any(|(name, shape)| observed.get(name.as_str()) != Some(&shape))
    {
        return Err(corrupt("directory tensor inventory or shape is invalid"));
    }
    Ok(())
}

fn validate_models_v2_member(
    registration: &RegisteredModel,
    member_id: &str,
    observed_hash: personalai_engine_types::ModelPackageHash,
    observed_bytes: usize,
) -> Result<(), EngineError> {
    let Some(binding) = registration.qwen_registry_binding() else {
        return Ok(());
    };
    let member = descriptor_member(binding.descriptor(), member_id)?;
    let observed_bytes = u64::try_from(observed_bytes)
        .map_err(|_| EngineError::resource_exhaustion("package member length exceeds u64"))?;
    if !models_v2_digest_matches(&member.content.digest, observed_hash)
        || member.size_bytes != observed_bytes
    {
        return Err(EngineError::authentication_failure(
            "PAIDIR01 member differs from the canonical Models 2.0 descriptor",
        ));
    }
    Ok(())
}

fn models_v2_digest_matches(
    canonical_digest: &str,
    observed_hash: personalai_engine_types::ModelPackageHash,
) -> bool {
    canonical_digest == sha384_hex(observed_hash)
}

fn validate_models_v2_manifest(
    registration: &RegisteredModel,
    manifest: &DirectoryManifest,
    manifest_bytes: &[u8],
) -> Result<(), EngineError> {
    let is_qwen = manifest.architecture_id().as_str() == DIRECTORY_QWEN2_DECODER_ARCHITECTURE_ID;
    if is_qwen && registration.mode() != RegistrationMode::ExperimentalQwenRegistryV3 {
        return Err(EngineError::authentication_failure(
            "Qwen package execution requires the separate signed V3/Models 2.0 binding",
        ));
    }
    if !is_qwen && registration.mode() == RegistrationMode::ExperimentalQwenRegistryV3 {
        return Err(EngineError::authentication_failure(
            "Qwen V3 registration cannot authorize another architecture",
        ));
    }
    let Some(binding) = registration.qwen_registry_binding() else {
        return Ok(());
    };
    let descriptor = binding.descriptor();
    if manifest.architecture_id().as_str() != DIRECTORY_QWEN2_DECODER_ARCHITECTURE_ID
        || descriptor.architecture.architecture_id != manifest.architecture_id().as_str()
        || descriptor.architecture.context_limit != manifest.context_limit().get()
        || !models_v2_digest_matches(
            &descriptor.tokenizer.content.digest,
            manifest.tokenizer_hash(),
        )
        || descriptor.tensor_inventory.tensor_count
            != u32::try_from(manifest.tensors().len()).map_err(|_| {
                EngineError::resource_exhaustion("manifest tensor count exceeds u32")
            })?
    {
        return Err(EngineError::authentication_failure(
            "PAIDIR01 manifest differs from the canonical Models 2.0 descriptor",
        ));
    }
    validate_models_v2_member(
        registration,
        "manifest",
        hash_package(manifest_bytes),
        manifest_bytes.len(),
    )
}

fn validate_models_v2_total(
    registration: &RegisteredModel,
    observed_bytes: u64,
) -> Result<(), EngineError> {
    let Some(binding) = registration.qwen_registry_binding() else {
        return Ok(());
    };
    if binding.descriptor().package_size_bytes != observed_bytes {
        return Err(EngineError::authentication_failure(
            "PAIDIR01 byte total differs from the canonical Models 2.0 descriptor",
        ));
    }
    Ok(())
}

fn validate_qwen25_15b_profile(manifest: &DirectoryManifest) -> Result<(), EngineError> {
    let dimensions_match = manifest.vocabulary_size() == QWEN25_15B_VOCABULARY
        && manifest.hidden_dimension() == QWEN25_15B_HIDDEN
        && manifest.layer_count() == QWEN25_15B_LAYERS
        && manifest.attention_head_count() == QWEN25_15B_ATTENTION_HEADS
        && manifest.kv_head_count() == QWEN25_15B_KV_HEADS
        && manifest.feed_forward_dimension() == QWEN25_15B_FEED_FORWARD
        && manifest.rms_norm_epsilon().to_bits() == QWEN25_RMS_EPSILON.to_bits();
    let rope_profile_matches = matches!(
        (manifest.context_limit().get(), manifest.rope_theta().to_bits()),
        (QWEN25_STANDARD_CONTEXT, bits)
            if bits == QWEN25_STANDARD_ROPE_THETA.to_bits()
    ) || matches!(
        (manifest.context_limit().get(), manifest.rope_theta().to_bits()),
        (QWEN25_MATH_CONTEXT, bits) if bits == QWEN25_MATH_ROPE_THETA.to_bits()
    );
    if !dimensions_match || !rope_profile_matches {
        return Err(corrupt(
            "Qwen2.5 package configuration is outside the reviewed 1.5B bootstrap profiles",
        ));
    }
    Ok(())
}

fn expected_shapes(
    manifest: &DirectoryManifest,
) -> Result<BTreeMap<String, TensorShape>, EngineError> {
    let hidden = u64::from(manifest.hidden_dimension());
    let vocabulary = u64::from(manifest.vocabulary_size());
    let intermediate = u64::from(manifest.feed_forward_dimension());
    let head_dimension = hidden
        .checked_div(u64::from(manifest.attention_head_count()))
        .filter(|value| *value != 0)
        .ok_or_else(|| corrupt("attention dimensions are inconsistent"))?;
    let kv = u64::from(manifest.kv_head_count())
        .checked_mul(head_dimension)
        .ok_or_else(|| corrupt("KV dimension overflows"))?;
    let mut expected = BTreeMap::from([
        ("token_embedding.weight".to_owned(), TensorShape::new([vocabulary, hidden])?),
        ("final_norm.weight".to_owned(), TensorShape::new([hidden])?),
    ]);
    if !manifest.tied_output() {
        expected.insert("output.weight".to_owned(), TensorShape::new([vocabulary, hidden])?);
    }
    for layer in 0..manifest.layer_count() {
        let prefix = format!("layers.{layer:03}");
        for suffix in ["attn_norm.weight", "ffn_norm.weight"] {
            expected.insert(format!("{prefix}.{suffix}"), TensorShape::new([hidden])?);
        }
        for suffix in ["q_proj.weight", "o_proj.weight"] {
            expected.insert(format!("{prefix}.{suffix}"), TensorShape::new([hidden, hidden])?);
        }
        for suffix in ["k_proj.weight", "v_proj.weight"] {
            expected.insert(format!("{prefix}.{suffix}"), TensorShape::new([kv, hidden])?);
        }
        if manifest.architecture_id().as_str() == DIRECTORY_QWEN2_DECODER_ARCHITECTURE_ID {
            expected.insert(format!("{prefix}.q_proj.bias"), TensorShape::new([hidden])?);
            for suffix in ["k_proj.bias", "v_proj.bias"] {
                expected.insert(format!("{prefix}.{suffix}"), TensorShape::new([kv])?);
            }
        }
        for suffix in ["gate_proj.weight", "up_proj.weight"] {
            expected
                .insert(format!("{prefix}.{suffix}"), TensorShape::new([intermediate, hidden])?);
        }
        expected.insert(
            format!("{prefix}.down_proj.weight"),
            TensorShape::new([hidden, intermediate])?,
        );
    }
    Ok(expected)
}

fn validate_tensor_hashes(
    package: &VerifiedDirectoryPackage,
    cancellation: &dyn DirectoryLoadCancellationProbe,
) -> Result<(), EngineError> {
    for tensor in package.manifest.tensors() {
        check_cancelled(cancellation)?;
        let bytes = package.tensor_bytes_revalidated(tensor, cancellation)?;
        if matches!(tensor.quantization(), Quantization::AipQ8V1 | Quantization::AipQ4V1) {
            QuantizedTensor::parse(tensor.quantization(), tensor.shape(), bytes)?;
        }
    }
    Ok(())
}

fn hash_package_cancellable(
    bytes: &[u8],
    cancellation: &dyn DirectoryLoadCancellationProbe,
) -> Result<personalai_engine_types::ModelPackageHash, EngineError> {
    let mut accumulator = PackageHashAccumulator::new();
    for chunk in bytes.chunks(DIRECTORY_HASH_CHUNK_BYTES) {
        check_cancelled(cancellation)?;
        accumulator.update(chunk);
    }
    check_cancelled(cancellation)?;
    Ok(accumulator.finalize())
}

fn check_cancelled(cancellation: &dyn DirectoryLoadCancellationProbe) -> Result<(), EngineError> {
    if cancellation.is_cancelled() { Err(EngineError::cancellation(None)) } else { Ok(()) }
}

fn validate_provenance(bytes: &[u8], registration: &RegisteredModel) -> Result<(), EngineError> {
    if registration.mode() == RegistrationMode::ExperimentalQwenRegistryV3 {
        return validate_models_v2_provenance(bytes);
    }
    if bytes.is_empty()
        || bytes.len() > usize::try_from(MAX_PROVENANCE_BYTES).unwrap_or(usize::MAX)
        || bytes.first() != Some(&b'{')
        || !bytes.ends_with(b"}\n")
    {
        return Err(corrupt("provenance document is not canonical"));
    }
    let text = std::str::from_utf8(bytes).map_err(|_| corrupt("provenance is invalid UTF-8"))?;
    if text
        .chars()
        .any(|character| character.is_control() && !matches!(character, '\n' | '\r' | '\t'))
    {
        return Err(corrupt("provenance contains prohibited control data"));
    }
    Ok(())
}

fn validate_models_v2_provenance(bytes: &[u8]) -> Result<(), EngineError> {
    if bytes.is_empty()
        || bytes.len() > usize::try_from(MAX_PROVENANCE_BYTES).unwrap_or(usize::MAX)
        || bytes.first() != Some(&b'{')
        || bytes.last() != Some(&b'}')
    {
        return Err(corrupt("Models 2.0 provenance document is not canonical"));
    }
    let text = std::str::from_utf8(bytes)
        .map_err(|_| corrupt("Models 2.0 provenance is invalid UTF-8"))?;
    if text
        .chars()
        .any(|character| character.is_control() && !matches!(character, '\n' | '\r' | '\t'))
    {
        return Err(corrupt("Models 2.0 provenance contains prohibited control data"));
    }
    let value: serde_json::Value = serde_json::from_slice(bytes)
        .map_err(|_| corrupt("Models 2.0 provenance is invalid JSON"))?;
    let canonical = crate::canonical_json::to_vec(value)
        .map_err(|_| EngineError::internal_invariant("provenance serialization failed"))?;
    if canonical != bytes {
        return Err(corrupt("Models 2.0 provenance document is not canonical"));
    }
    Ok(())
}

fn checked_total(
    total: u64,
    additional: usize,
    budget: &ResourceBudget,
) -> Result<u64, EngineError> {
    let total = total
        .checked_add(usize_to_u64(additional)?)
        .ok_or_else(|| EngineError::resource_exhaustion("directory package size overflows"))?;
    if total > MAX_MODEL_PACKAGE_BYTES || total > budget.max_memory_bytes() {
        return Err(EngineError::resource_exhaustion(
            "directory package exceeds its configured byte limit",
        ));
    }
    Ok(total)
}

fn ensure_entry_budget(
    total: u64,
    additional: u64,
    budget: &ResourceBudget,
) -> Result<(), EngineError> {
    let projected = total
        .checked_add(additional)
        .ok_or_else(|| EngineError::resource_exhaustion("directory package size overflows"))?;
    if projected > MAX_MODEL_PACKAGE_BYTES || projected > budget.max_memory_bytes() {
        return Err(EngineError::resource_exhaustion(
            "directory package exceeds its configured byte limit",
        ));
    }
    Ok(())
}

fn rejection_reason(error: &EngineError) -> ModelPackageRejection {
    match error {
        EngineError::SignatureFailure(_) | EngineError::AuthenticationFailure(_) => {
            ModelPackageRejection::SignatureFailure
        }
        EngineError::UnsupportedArchitecture(_) => ModelPackageRejection::UnsupportedArchitecture,
        EngineError::UnsupportedTokenizer(_) => ModelPackageRejection::UnsupportedTokenizer,
        EngineError::ResourceExhaustion(_) => ModelPackageRejection::SizeLimit,
        _ => ModelPackageRejection::Corrupt,
    }
}

fn usize_to_u64(value: usize) -> Result<u64, EngineError> {
    u64::try_from(value)
        .map_err(|_| EngineError::resource_exhaustion("package size is not representable"))
}

fn corrupt(detail: impl AsRef<str>) -> EngineError {
    EngineError::corrupt_model_package(detail)
}

#[cfg(test)]
mod qwen25_tests {
    use super::*;
    use personalai_engine_model_format::{
        DIRECTORY_QWEN2_MINIMUM_ENGINE_VERSION, DirectoryManifestBuilder, DirectoryShardRecord,
        DirectoryTensorRecord, HashCatalogEntry,
    };
    use personalai_engine_types::{ContextLimit, ModelPackageHash};
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct CancelHashAfterChecks {
        checks: AtomicUsize,
        cancel_at: usize,
    }

    impl DirectoryLoadCancellationProbe for CancelHashAfterChecks {
        fn is_cancelled(&self) -> bool {
            self.checks.fetch_add(1, Ordering::AcqRel) + 1 >= self.cancel_at
        }
    }

    fn hash(byte: u8) -> ModelPackageHash {
        ModelPackageHash::from_sha384_bytes([byte; ModelPackageHash::BYTE_LEN])
    }

    fn record(name: String, shape: TensorShape, offset: &mut u64) -> DirectoryTensorRecord {
        let byte_length =
            shape.element_count().checked_mul(4).expect("fixture byte length must fit");
        let tensor = DirectoryTensorRecord::new(
            name,
            DType::F32,
            shape,
            0,
            *offset,
            byte_length,
            64,
            Quantization::None,
            hash(3),
        )
        .expect("fixture tensor must validate");
        *offset = offset.checked_add(byte_length).expect("fixture offset must fit");
        tensor
    }

    fn qwen25_manifest(omitted: Option<&str>, context: u32, rope_theta: f32) -> DirectoryManifest {
        let hidden = u64::from(QWEN25_15B_HIDDEN);
        let vocabulary = u64::from(QWEN25_15B_VOCABULARY);
        let intermediate = u64::from(QWEN25_15B_FEED_FORWARD);
        let kv = u64::from(QWEN25_15B_KV_HEADS) * (hidden / u64::from(QWEN25_15B_ATTENTION_HEADS));
        let mut specifications = vec![
            ("token_embedding.weight".to_owned(), TensorShape::new([vocabulary, hidden]).unwrap()),
            ("final_norm.weight".to_owned(), TensorShape::new([hidden]).unwrap()),
        ];
        for layer in 0..QWEN25_15B_LAYERS {
            let prefix = format!("layers.{layer:03}");
            for suffix in ["attn_norm.weight", "ffn_norm.weight"] {
                specifications
                    .push((format!("{prefix}.{suffix}"), TensorShape::new([hidden]).unwrap()));
            }
            for suffix in ["q_proj.weight", "o_proj.weight"] {
                specifications.push((
                    format!("{prefix}.{suffix}"),
                    TensorShape::new([hidden, hidden]).unwrap(),
                ));
            }
            for suffix in ["k_proj.weight", "v_proj.weight"] {
                specifications
                    .push((format!("{prefix}.{suffix}"), TensorShape::new([kv, hidden]).unwrap()));
            }
            specifications
                .push((format!("{prefix}.q_proj.bias"), TensorShape::new([hidden]).unwrap()));
            for suffix in ["k_proj.bias", "v_proj.bias"] {
                specifications
                    .push((format!("{prefix}.{suffix}"), TensorShape::new([kv]).unwrap()));
            }
            for suffix in ["gate_proj.weight", "up_proj.weight"] {
                specifications.push((
                    format!("{prefix}.{suffix}"),
                    TensorShape::new([intermediate, hidden]).unwrap(),
                ));
            }
            specifications.push((
                format!("{prefix}.down_proj.weight"),
                TensorShape::new([hidden, intermediate]).unwrap(),
            ));
        }
        specifications.sort_by(|left, right| left.0.cmp(&right.0));
        let mut offset = 0_u64;
        let tensors = specifications
            .into_iter()
            .filter(|(name, _)| omitted != Some(name.as_str()))
            .map(|(name, shape)| record(name, shape, &mut offset))
            .collect();
        DirectoryManifestBuilder::new_qwen2(
            ModelId::new("qwen25-bootstrap-fixture").unwrap(),
            DIRECTORY_QWEN2_MINIMUM_ENGINE_VERSION,
            ContextLimit::new(context).unwrap(),
            QWEN25_15B_VOCABULARY,
            QWEN25_15B_HIDDEN,
            QWEN25_15B_LAYERS,
            QWEN25_15B_ATTENTION_HEADS,
            QWEN25_15B_KV_HEADS,
            QWEN25_15B_FEED_FORWARD,
            QWEN25_RMS_EPSILON,
            rope_theta,
            DType::F32,
            Quantization::None,
            hash(1),
            vec![DirectoryShardRecord::new(0, offset, hash(2)).unwrap()],
            tensors,
        )
        .expect("Qwen2.5 manifest must validate")
        .manifest()
        .clone()
    }

    #[allow(
        clippy::too_many_lines,
        reason = "the cross-repository vector spells the complete reviewed 338-tensor inventory"
    )]
    fn foundry_qwen25_conformance_vector() -> (Vec<u8>, Vec<u8>) {
        let hidden = u64::from(QWEN25_15B_HIDDEN);
        let vocabulary = u64::from(QWEN25_15B_VOCABULARY);
        let intermediate = u64::from(QWEN25_15B_FEED_FORWARD);
        let kv = u64::from(QWEN25_15B_KV_HEADS) * (hidden / u64::from(QWEN25_15B_ATTENTION_HEADS));
        let mut specifications = vec![
            ("token_embedding.weight".to_owned(), TensorShape::new([vocabulary, hidden]).unwrap()),
            ("final_norm.weight".to_owned(), TensorShape::new([hidden]).unwrap()),
        ];
        for layer in 0..QWEN25_15B_LAYERS {
            let prefix = format!("layers.{layer:03}");
            for suffix in ["attn_norm.weight", "ffn_norm.weight"] {
                specifications
                    .push((format!("{prefix}.{suffix}"), TensorShape::new([hidden]).unwrap()));
            }
            for suffix in ["q_proj.weight", "o_proj.weight"] {
                specifications.push((
                    format!("{prefix}.{suffix}"),
                    TensorShape::new([hidden, hidden]).unwrap(),
                ));
            }
            for suffix in ["k_proj.weight", "v_proj.weight"] {
                specifications
                    .push((format!("{prefix}.{suffix}"), TensorShape::new([kv, hidden]).unwrap()));
            }
            specifications
                .push((format!("{prefix}.q_proj.bias"), TensorShape::new([hidden]).unwrap()));
            for suffix in ["k_proj.bias", "v_proj.bias"] {
                specifications
                    .push((format!("{prefix}.{suffix}"), TensorShape::new([kv]).unwrap()));
            }
            for suffix in ["gate_proj.weight", "up_proj.weight"] {
                specifications.push((
                    format!("{prefix}.{suffix}"),
                    TensorShape::new([intermediate, hidden]).unwrap(),
                ));
            }
            specifications.push((
                format!("{prefix}.down_proj.weight"),
                TensorShape::new([hidden, intermediate]).unwrap(),
            ));
        }
        specifications.sort_by(|left, right| left.0.cmp(&right.0));
        assert_eq!(specifications.len(), 338);

        let mut next_offset = 0_u64;
        let tensors = specifications
            .into_iter()
            .enumerate()
            .map(|(index, (name, shape))| {
                let offset =
                    next_offset.checked_add(63).expect("fixture offset").div_euclid(64) * 64;
                let byte_length = shape.element_count().checked_mul(2).expect("BF16 byte length");
                next_offset = offset.checked_add(byte_length).expect("next tensor offset");
                DirectoryTensorRecord::new(
                    name,
                    DType::BFloat16,
                    shape,
                    0,
                    offset,
                    byte_length,
                    64,
                    Quantization::None,
                    hash(u8::try_from(index % 251 + 1).expect("fixture hash byte")),
                )
                .expect("Foundry conformance tensor")
            })
            .collect::<Vec<_>>();
        assert_eq!(next_offset, 3_087_428_608);

        let tokenizer_hash = ModelPackageHash::from_sha384_hex(
            "f1b418104aedb90abc6ee892ef5b71db2b02c2c0c82a9f8e162487d933ffdd636b9ae077eed806f011c251cfea5187cc",
        )
        .expect("published AIPTOK02 identity");
        let manifest = DirectoryManifestBuilder::new_qwen2(
            ModelId::new("qwen2.5-1.5b-instruct-989aa798").unwrap(),
            DIRECTORY_QWEN2_MINIMUM_ENGINE_VERSION,
            ContextLimit::new(QWEN25_STANDARD_CONTEXT).unwrap(),
            QWEN25_15B_VOCABULARY,
            QWEN25_15B_HIDDEN,
            QWEN25_15B_LAYERS,
            QWEN25_15B_ATTENTION_HEADS,
            QWEN25_15B_KV_HEADS,
            QWEN25_15B_FEED_FORWARD,
            QWEN25_RMS_EPSILON,
            QWEN25_STANDARD_ROPE_THETA,
            DType::BFloat16,
            Quantization::None,
            tokenizer_hash,
            vec![DirectoryShardRecord::new(0, next_offset, hash(2)).unwrap()],
            tensors,
        )
        .expect("Foundry Qwen conformance manifest")
        .build_bytes()
        .expect("encode Foundry conformance manifest");
        let catalog = HashCatalog::new(vec![
            HashCatalogEntry::new(
                PackageFileKind::Manifest,
                0,
                "manifest.aipm",
                u64::try_from(manifest.len()).unwrap(),
                hash_package(&manifest),
            )
            .unwrap(),
            HashCatalogEntry::new(
                PackageFileKind::Tokenizer,
                0,
                "tokenizer.aiptok",
                4_613_135,
                tokenizer_hash,
            )
            .unwrap(),
            HashCatalogEntry::new(PackageFileKind::Provenance, 0, "provenance.json", 512, hash(3))
                .unwrap(),
            HashCatalogEntry::new(
                PackageFileKind::TensorShard,
                0,
                "tensors-0000.aipt",
                next_offset,
                hash(2),
            )
            .unwrap(),
        ])
        .expect("Foundry conformance catalog")
        .encode()
        .expect("encode Foundry conformance catalog");
        (manifest, catalog)
    }

    #[test]
    fn reviewed_qwen25_profiles_have_exact_bias_inventory() {
        let general = qwen25_manifest(None, QWEN25_STANDARD_CONTEXT, QWEN25_STANDARD_ROPE_THETA);
        validate_architecture(&general).expect("standard Qwen2.5 profile must validate");
        assert_eq!(
            general.required_engine_version(),
            DIRECTORY_LOADER_ENGINE_VERSION,
            "the reviewed v1.1 profile is bound to the 0.2 loader revision"
        );
        let math = qwen25_manifest(None, QWEN25_MATH_CONTEXT, QWEN25_MATH_ROPE_THETA);
        validate_architecture(&math).expect("math Qwen2.5 profile must validate");

        let expected = expected_shapes(&general).expect("expected inventory must derive");
        assert!(expected.contains_key("layers.000.q_proj.bias"));
        assert!(expected.contains_key("layers.027.k_proj.bias"));
        assert!(expected.contains_key("layers.027.v_proj.bias"));
        assert!(!expected.contains_key("output.weight"));

        let missing = qwen25_manifest(
            Some("layers.013.v_proj.bias"),
            QWEN25_STANDARD_CONTEXT,
            QWEN25_STANDARD_ROPE_THETA,
        );
        assert!(matches!(
            validate_architecture(&missing),
            Err(EngineError::CorruptModelPackage(_))
        ));
    }

    #[test]
    fn exact_foundry_qwen25_v1_1_vector_parses_and_matches_closed_profile() {
        let (manifest_bytes, catalog_bytes) = foundry_qwen25_conformance_vector();
        assert_eq!(manifest_bytes.len(), 65_472);
        assert_eq!(
            hash_package(&manifest_bytes),
            ModelPackageHash::from_sha384_hex(
                "a8f31f32b09d44fa66e103d004412e8af3787edc6d94f7cf1dd6ad0123151fb22ee2799db00818e904083d943e6d829e",
            )
            .unwrap()
        );
        let manifest = DirectoryManifest::parse(&manifest_bytes).expect("exact Foundry manifest");
        validate_architecture(&manifest).expect("exact Foundry Qwen profile");
        assert_eq!(manifest.tensors().len(), 338);
        assert_eq!(manifest.tensors().first().unwrap().name(), "final_norm.weight");
        assert_eq!(manifest.tensors().last().unwrap().name(), "token_embedding.weight");
        assert_eq!(manifest.storage_dtype(), DType::BFloat16);
        assert_eq!(manifest.required_engine_version(), (0, 2, 0));

        assert_eq!(catalog_bytes.len(), 528);
        assert_eq!(
            hash_package(&catalog_bytes),
            ModelPackageHash::from_sha384_hex(
                "86651018b34498e8639dac2276655d074a3a8ae8eaf976ca242431a569276e456d1626e5209456cf29517ab2c86fd11b",
            )
            .unwrap()
        );
        let catalog = HashCatalog::parse(&catalog_bytes).expect("exact Foundry catalog");
        let manifest_entry = catalog.entry("manifest.aipm").expect("manifest entry");
        assert_eq!(manifest_entry.byte_length(), 65_472);
        assert_eq!(manifest_entry.content_hash(), hash_package(&manifest_bytes));
        assert_eq!(catalog.entry("tensors-0000.aipt").unwrap().byte_length(), 3_087_428_608);

        if let Ok(directory) = std::env::var("PERSONALAI_FOUNDRY_QWEN_CONFORMANCE_DIR") {
            assert_eq!(
                std::fs::read(Path::new(&directory).join("manifest.aipm")).unwrap(),
                manifest_bytes,
                "engine-generated vector must be byte-identical to the Foundry encoder"
            );
            assert_eq!(
                std::fs::read(Path::new(&directory).join("hashes.aiph")).unwrap(),
                catalog_bytes,
                "engine-generated catalog must be byte-identical to the Foundry encoder"
            );
        }
    }

    #[test]
    fn package_hashing_is_exact_and_cooperatively_cancellable() {
        let bytes = vec![0x5a; DIRECTORY_HASH_CHUNK_BYTES * 2 + 17];
        assert_eq!(
            hash_package_cancellable(&bytes, &NeverCancelled).unwrap(),
            hash_package(&bytes)
        );
        let cancellation = CancelHashAfterChecks { checks: AtomicUsize::new(0), cancel_at: 2 };
        assert!(matches!(
            hash_package_cancellable(&bytes, &cancellation),
            Err(EngineError::Cancellation(_))
        ));
        assert_eq!(cancellation.checks.load(Ordering::Acquire), 2);
    }

    #[test]
    fn unreviewed_qwen25_configuration_fails_closed() {
        let context = qwen25_manifest(None, 8_192, QWEN25_STANDARD_ROPE_THETA);
        assert!(validate_architecture(&context).is_err());
        let theta = qwen25_manifest(None, QWEN25_STANDARD_CONTEXT, 10_000.0);
        assert!(validate_architecture(&theta).is_err());
    }

    #[test]
    fn models_v2_digests_are_unprefixed_while_internal_hashes_are_algorithm_tagged() {
        let observed = hash(0xab);
        let canonical = "ab".repeat(ModelPackageHash::BYTE_LEN);
        assert!(models_v2_digest_matches(&canonical, observed));
        assert_ne!(canonical, observed.to_string());
        assert!(!models_v2_digest_matches(&"cd".repeat(ModelPackageHash::BYTE_LEN), observed));
    }

    #[test]
    fn provenance_requires_recursive_canonical_json_without_a_trailing_newline() {
        let canonical = br#"{"a":{"c":2,"d":1},"b":[1,2]}"#;
        validate_models_v2_provenance(canonical).expect("canonical provenance");
        assert!(validate_models_v2_provenance(br#"{"b":[1,2],"a":{"c":2,"d":1}}"#).is_err());
        assert!(validate_models_v2_provenance(b"{\"a\":1}\n").is_err());
        assert!(validate_models_v2_provenance(br#"{"a":1,"a":1}"#).is_err());
    }
}
