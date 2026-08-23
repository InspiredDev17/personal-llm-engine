//! Registered-model resolution and cryptographic package admission.
//!
//! Requests contain only a [`ModelId`]. Owner configuration either pins the
//! exact package identity directly or selects one exact signed Model Foundry V2
//! decision. Registry-backed paths are derived only after strict verification;
//! request data never supplies paths, hashes, or trust identities.

mod canonical_json;
#[cfg(unix)]
mod directory_loader;
mod qwen_registry_v3;
#[cfg(unix)]
mod read_only_map;
mod registry_v2;

#[cfg(unix)]
pub use directory_loader::{
    DIRECTORY_LOADER_ENGINE_VERSION, DirectoryLoadCancellationProbe, DirectorySecurityEventSink,
    LoadedDirectoryTensors, VerifiedDirectoryPackage, VerifiedDirectoryReadinessMetadata,
};
#[cfg(unix)]
use qwen_registry_v3::read_verified_qwen_registry_selection_v3;
pub use qwen_registry_v3::{
    MAX_MODELS_V2_DESCRIPTOR_BYTES, MODELS_V2_CONTRACT_COMMIT, QWEN_MODELS_V2_ARCHITECTURE_ID,
    QWEN_MODELS_V2_MODEL_VOCABULARY, QWEN_MODELS_V2_PACKAGE_FORMAT_ID,
    QWEN_MODELS_V2_PACKAGE_FORMAT_VERSION, QWEN_MODELS_V2_POSITION_SEMANTICS_ID,
    QWEN_MODELS_V2_QUANTIZATION_PROFILE_ID, QWEN_MODELS_V2_SELECTABLE_TOKENS,
    QWEN_MODELS_V2_TOKENIZER_FORMAT_ID, QWEN_MODELS_V2_TOKENIZER_FORMAT_VERSION,
    QWEN_MODELS_V2_TOKENIZER_ID, QwenRegistrySelectionV3, VerifiedQwenRegistrySelectionV3,
    verify_qwen_registry_selection_v3,
};
#[cfg(unix)]
use registry_v2::read_verified_registry_selection_v2;
pub use registry_v2::{
    RegistrySelectionV2, VerifiedRegistrySelectionV2, verify_registry_selection_v2,
};

use personalai_engine_crypto::TrustedSigner;
#[cfg(test)]
use personalai_engine_crypto::{hash_package, hash_signed_content, verify_package_signature};
#[cfg(test)]
use personalai_engine_memory::{MemoryPool, MemoryReservation};
#[cfg(test)]
use personalai_engine_model_format::{ModelPackageHeader, ParsedModelPackage};
#[cfg(test)]
use personalai_engine_tensor::Tensor;
#[cfg(test)]
use personalai_engine_types::MAX_MODEL_PACKAGE_BYTES;
use personalai_engine_types::{EngineError, ModelId, ModelPackageHash, ResourceBudget};
use sovereign_ai_contracts::inference_runtime_v3::RuntimeIdentityV3;
use sovereign_ai_contracts::models_v2::ExecutionPackageDescriptor;
use std::collections::BTreeMap;
#[cfg(all(unix, test))]
use std::os::unix::fs::{MetadataExt, OpenOptionsExt, PermissionsExt};
use std::path::{Component, Path, PathBuf};
#[cfg(all(unix, test))]
use std::{fs, io::Read};

/// Required extension for first-party model packages.
pub const MODEL_PACKAGE_EXTENSION: &str = "paimodel";
/// Maximum number of model IDs in one local registry.
pub const MAX_REGISTERED_MODELS: usize = 1_024;

/// Explicit registration authority selected by owner configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RegistrationMode {
    /// Legacy exact package identity pinned directly by owner configuration.
    OwnerPinned,
    /// Exact signed Model Foundry V2 decision selected by owner configuration.
    ModelFoundryRegistryV2,
    /// Development-only Qwen profile bound to a signed V3 decision and canonical Models 2.0.
    ExperimentalQwenRegistryV3,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct RegistryBindingV2 {
    artifact_id: String,
    registry_sequence: u64,
    registry_entry_hash: ModelPackageHash,
    context_capacity: u32,
    tokenizer_hash: ModelPackageHash,
    provenance_hash: ModelPackageHash,
    tensor_shard_hashes: Vec<ModelPackageHash>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct QwenRegistryBindingV3 {
    descriptor: ExecutionPackageDescriptor,
    descriptor_hash: ModelPackageHash,
    registry_sequence: u64,
    registry_entry_hash: ModelPackageHash,
}

impl QwenRegistryBindingV3 {
    pub(crate) const fn descriptor(&self) -> &ExecutionPackageDescriptor {
        &self.descriptor
    }
}

/// One configuration-owned model package registration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RegisteredModel {
    model_id: ModelId,
    package_path: PathBuf,
    expected_hash: ModelPackageHash,
    trusted_signer: TrustedSigner,
    mode: RegistrationMode,
    registry_binding: Option<RegistryBindingV2>,
    qwen_registry_binding: Option<QwenRegistryBindingV3>,
}

impl RegisteredModel {
    /// Returns the configured model identifier.
    pub const fn model_id(&self) -> &ModelId {
        &self.model_id
    }

    /// Returns the engine-derived package path.
    ///
    /// This path is local configuration or verified-registry-derived data and
    /// must never be serialized into model context, generation output, or
    /// security telemetry.
    pub fn package_path(&self) -> &Path {
        &self.package_path
    }

    /// Returns the expected, algorithm-tagged package digest.
    pub const fn expected_hash(&self) -> &ModelPackageHash {
        &self.expected_hash
    }

    /// Returns the operator-configured package signer.
    pub const fn trusted_signer(&self) -> &TrustedSigner {
        &self.trusted_signer
    }

    /// Returns the explicit owner-selected admission mode.
    pub const fn mode(&self) -> RegistrationMode {
        self.mode
    }

    /// Returns the signed lifecycle artifact ID for registry-backed admission.
    pub fn registry_artifact_id(&self) -> Option<&str> {
        self.registry_binding.as_ref().map(|binding| binding.artifact_id.as_str()).or_else(|| {
            self.qwen_registry_binding
                .as_ref()
                .map(|binding| binding.descriptor.runtime_identity.artifact_id.as_str())
        })
    }

    /// Returns the exact selected registry sequence for registry-backed admission.
    pub fn registry_sequence(&self) -> Option<u64> {
        self.registry_binding.as_ref().map(|binding| binding.registry_sequence).or_else(|| {
            self.qwen_registry_binding.as_ref().map(|binding| binding.registry_sequence)
        })
    }

    /// Returns the exact selected registry-envelope identity.
    pub fn registry_entry_hash(&self) -> Option<ModelPackageHash> {
        self.registry_binding.as_ref().map(|binding| binding.registry_entry_hash).or_else(|| {
            self.qwen_registry_binding.as_ref().map(|binding| binding.registry_entry_hash)
        })
    }

    /// Returns the canonical Models 2.0 descriptor for a V3 Qwen registration.
    pub fn execution_package_descriptor(&self) -> Option<&ExecutionPackageDescriptor> {
        self.qwen_registry_binding.as_ref().map(|binding| &binding.descriptor)
    }

    /// Returns exact descriptor SHA-384 for a V3 Qwen registration.
    pub fn execution_package_descriptor_hash(&self) -> Option<ModelPackageHash> {
        self.qwen_registry_binding.as_ref().map(|binding| binding.descriptor_hash)
    }

    /// Returns the canonical runtime identity for a V3 Qwen registration.
    pub fn execution_runtime_identity(&self) -> Option<&RuntimeIdentityV3> {
        self.execution_package_descriptor().map(|descriptor| &descriptor.runtime_identity)
    }

    pub(crate) const fn registry_binding(&self) -> Option<&RegistryBindingV2> {
        self.registry_binding.as_ref()
    }

    pub(crate) const fn qwen_registry_binding(&self) -> Option<&QwenRegistryBindingV3> {
        self.qwen_registry_binding.as_ref()
    }
}

/// Defensive mirror of model IDs registered by local engine configuration.
#[derive(Debug)]
pub struct ModelRegistry {
    model_root: PathBuf,
    registrations: BTreeMap<String, RegisteredModel>,
}

impl ModelRegistry {
    /// Creates an empty registry rooted at a syntactically normalized absolute
    /// directory without touching the filesystem.
    pub fn new(model_root: impl Into<PathBuf>) -> Result<Self, EngineError> {
        let model_root = model_root.into();
        validate_absolute_normal_path(&model_root)?;
        Ok(Self { model_root, registrations: BTreeMap::new() })
    }

    /// Registers one configuration-owned relative package and trust policy.
    ///
    /// Request and protocol decoders must not expose this method to clients.
    pub fn register(
        &mut self,
        model_id: ModelId,
        relative_package_path: impl Into<PathBuf>,
        expected_hash: ModelPackageHash,
        trusted_signer: TrustedSigner,
    ) -> Result<(), EngineError> {
        if self.registrations.len() >= MAX_REGISTERED_MODELS {
            return Err(EngineError::resource_exhaustion(
                "local model registry reached its hard entry limit",
            ));
        }
        let relative_package_path = relative_package_path.into();
        validate_registered_package_path(&relative_package_path)?;
        let key = model_id.as_str().to_owned();
        if self.registrations.contains_key(&key) {
            return Err(EngineError::invalid_input("model identifier is already registered"));
        }
        let package_path = self.model_root.join(relative_package_path);
        let registration = RegisteredModel {
            model_id,
            package_path,
            expected_hash,
            trusted_signer,
            mode: RegistrationMode::OwnerPinned,
            registry_binding: None,
            qwen_registry_binding: None,
        };
        self.registrations.insert(key, registration);
        Ok(())
    }

    /// Verifies and registers one exact owner-selected Model Foundry V2 entry.
    ///
    /// Registry bytes are read only from the sequence-derived direct child of
    /// `<approved>/registry/`. The package path is then derived as
    /// `<approved>/<ArtifactId>/<ModelId>.paimodel`. Neither path can be
    /// supplied by a request, and failure never falls back to owner-pinned mode.
    #[cfg(unix)]
    pub fn register_registry_v2(
        &mut self,
        selection: &RegistrySelectionV2,
    ) -> Result<(), EngineError> {
        if self.registrations.len() >= MAX_REGISTERED_MODELS {
            return Err(EngineError::resource_exhaustion(
                "local model registry reached its hard entry limit",
            ));
        }
        let verified = read_verified_registry_selection_v2(&self.model_root, selection)?;
        let entry = verified.entry();
        let model_id = ModelId::new(entry.model_id())
            .map_err(|_| EngineError::corrupt_model_package("registry model ID is invalid"))?;
        let key = model_id.as_str().to_owned();
        if self.registrations.contains_key(&key) {
            return Err(EngineError::invalid_input("model identifier is already registered"));
        }
        let package_name = format!("{}.{}", model_id.as_str(), MODEL_PACKAGE_EXTENSION);
        let package_path = self.model_root.join(entry.artifact_id()).join(package_name);
        let registration = RegisteredModel {
            model_id,
            package_path,
            expected_hash: entry.package_catalog_hash(),
            trusted_signer: verified.package_signer().clone(),
            mode: RegistrationMode::ModelFoundryRegistryV2,
            registry_binding: Some(RegistryBindingV2 {
                artifact_id: entry.artifact_id().to_owned(),
                registry_sequence: entry.registry_sequence(),
                registry_entry_hash: verified.entry_hash(),
                context_capacity: entry.context_capacity(),
                tokenizer_hash: entry.tokenizer_hash(),
                provenance_hash: entry.provenance_hash(),
                tensor_shard_hashes: entry.tensor_shard_hashes().to_vec(),
            }),
            qwen_registry_binding: None,
        };
        self.registrations.insert(key, registration);
        Ok(())
    }

    /// Verifies and registers one development-only signed Qwen V3 decision.
    ///
    /// Registry and descriptor paths are derived from protected local state;
    /// clients provide neither paths nor package identities. Registration is
    /// evidence and package trust only. A separate explicit owner execution
    /// gate remains mandatory and production activation is unavailable.
    #[cfg(unix)]
    pub fn register_qwen_registry_v3(
        &mut self,
        selection: &QwenRegistrySelectionV3,
    ) -> Result<(), EngineError> {
        if self.registrations.len() >= MAX_REGISTERED_MODELS {
            return Err(EngineError::resource_exhaustion(
                "local model registry reached its hard entry limit",
            ));
        }
        let verified = read_verified_qwen_registry_selection_v3(&self.model_root, selection)?;
        let descriptor = verified.descriptor();
        let model_id = ModelId::new(descriptor.runtime_identity.provider_model_id.as_str())
            .map_err(|_| {
                EngineError::corrupt_model_package("Qwen descriptor model ID is invalid")
            })?;
        let key = model_id.as_str().to_owned();
        if self.registrations.contains_key(&key) {
            return Err(EngineError::invalid_input("model identifier is already registered"));
        }
        let artifact_id = descriptor.runtime_identity.artifact_id.as_str();
        let package_name = format!("{}.{}", model_id.as_str(), MODEL_PACKAGE_EXTENSION);
        let package_path = self.model_root.join(artifact_id).join(package_name);
        let registration = RegisteredModel {
            model_id,
            package_path,
            expected_hash: verified.package_catalog_hash(),
            trusted_signer: verified.package_signer().clone(),
            mode: RegistrationMode::ExperimentalQwenRegistryV3,
            registry_binding: None,
            qwen_registry_binding: Some(QwenRegistryBindingV3 {
                descriptor: descriptor.clone(),
                descriptor_hash: verified.descriptor_hash(),
                registry_sequence: verified.registry_sequence(),
                registry_entry_hash: verified.registry_entry_hash(),
            }),
        };
        self.registrations.insert(key, registration);
        Ok(())
    }

    /// Resolves a request model ID to configuration-owned package metadata.
    pub fn resolve(&self, model_id: &ModelId) -> Result<&RegisteredModel, EngineError> {
        self.registrations
            .get(model_id.as_str())
            .ok_or_else(|| EngineError::invalid_input("model identifier is not locally registered"))
    }

    /// Returns the number of local model registrations.
    pub fn len(&self) -> usize {
        self.registrations.len()
    }

    /// Returns whether no models are registered.
    pub fn is_empty(&self) -> bool {
        self.registrations.is_empty()
    }

    /// Iterates configuration-owned registrations in deterministic model-ID order.
    pub fn registrations(&self) -> impl ExactSizeIterator<Item = &RegisteredModel> {
        self.registrations.values()
    }

    /// Opens and reads one configuration-owned package without following a
    /// package symlink.
    ///
    /// Registrations are restricted to direct children of a non-symlink model
    /// root that is not writable by group or other users. The opened file's
    /// device and inode must match the pre-open metadata, and its exact length
    /// is bounded before allocation. This method never accepts a client path.
    #[cfg(all(unix, test))]
    pub fn read_registered_package(
        &self,
        model_id: &ModelId,
        budget: &ResourceBudget,
    ) -> Result<Vec<u8>, EngineError> {
        let registration = self.resolve(model_id)?;
        let root_metadata = fs::symlink_metadata(&self.model_root)
            .map_err(|_| EngineError::invalid_input("configured model root is unavailable"))?;
        if !root_metadata.is_dir()
            || root_metadata.file_type().is_symlink()
            || root_metadata.permissions().mode() & 0o022 != 0
        {
            return Err(EngineError::authentication_failure(
                "model root must be a non-symlink directory not writable by group or other",
            ));
        }
        let path = registration.package_path();
        let before = fs::symlink_metadata(path)
            .map_err(|_| EngineError::invalid_input("registered model package is unavailable"))?;
        if !before.file_type().is_file()
            || before.file_type().is_symlink()
            || before.permissions().mode() & 0o022 != 0
        {
            return Err(EngineError::authentication_failure(
                "registered package must be a non-symlink regular file not writable by group or other",
            ));
        }
        validate_package_budget(before.len(), budget)?;
        let mut file = fs::OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_CLOEXEC | libc::O_NOFOLLOW)
            .open(path)
            .map_err(|_| EngineError::authentication_failure("registered package open failed"))?;
        let opened = file
            .metadata()
            .map_err(|_| EngineError::corrupt_model_package("opened package metadata failed"))?;
        if !opened.is_file()
            || opened.dev() != before.dev()
            || opened.ino() != before.ino()
            || opened.len() != before.len()
        {
            return Err(EngineError::authentication_failure(
                "registered package changed during secure open",
            ));
        }
        let length = usize::try_from(opened.len()).map_err(|_| {
            EngineError::resource_exhaustion("registered package length is not addressable")
        })?;
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(length)
            .map_err(|_| EngineError::resource_exhaustion("package allocation failed"))?;
        bytes.resize(length, 0);
        file.read_exact(&mut bytes)
            .map_err(|_| EngineError::corrupt_model_package("registered package is truncated"))?;
        let mut extra = [0_u8; 1];
        if file
            .read(&mut extra)
            .map_err(|_| EngineError::corrupt_model_package("package trailing-byte check failed"))?
            != 0
        {
            return Err(EngineError::corrupt_model_package(
                "registered package grew during secure read",
            ));
        }
        Ok(bytes)
    }

    /// Opens, verifies, and read-only maps one registered directory package.
    ///
    /// Every failure is reported to the mandatory typed event sink before the
    /// error returns. The path remains configuration-owned and is never
    /// supplied by a client.
    #[cfg(unix)]
    pub fn load_registered_directory(
        &self,
        model_id: &ModelId,
        budget: &ResourceBudget,
        events: &mut impl DirectorySecurityEventSink,
    ) -> Result<VerifiedDirectoryPackage, EngineError> {
        directory_loader::load(self, model_id, budget, events)
    }

    /// Opens and verifies one registered directory package with cooperative cancellation.
    ///
    /// This preserves every ordinary package, engine-version, and identity gate;
    /// cancellation grants no ability to inspect or execute a future package.
    #[cfg(unix)]
    pub fn load_registered_directory_cancellable(
        &self,
        model_id: &ModelId,
        budget: &ResourceBudget,
        cancellation: &dyn DirectoryLoadCancellationProbe,
        events: &mut impl DirectorySecurityEventSink,
    ) -> Result<VerifiedDirectoryPackage, EngineError> {
        directory_loader::load_cancellable(self, model_id, budget, cancellation, events)
    }

    /// Verifies a registered directory package's signed metadata only.
    ///
    /// Tensor bytes are not mapped or hashed and the current engine-version
    /// admission gate is not bypassed. The result is therefore pre-activation
    /// evidence, never an executable package or qualification decision.
    #[cfg(unix)]
    pub fn inspect_registered_directory_readiness_metadata(
        &self,
        model_id: &ModelId,
        budget: &ResourceBudget,
        events: &mut impl DirectorySecurityEventSink,
    ) -> Result<VerifiedDirectoryReadinessMetadata, EngineError> {
        directory_loader::inspect_readiness_metadata(self, model_id, budget, events)
    }

    /// Produces a bounded metadata-only pre-load plan.
    #[cfg(test)]
    pub fn plan(
        &self,
        model_id: &ModelId,
        header: ModelPackageHeader,
        budget: &ResourceBudget,
    ) -> Result<ModelLoadPlan, EngineError> {
        let registration = self.resolve(model_id)?.clone();
        validate_package_budget(header.package_length(), budget)?;
        Ok(ModelLoadPlan { registration, header })
    }

    /// Verifies identity, canonical structure, and signature of bounded bytes.
    ///
    /// The caller must obtain `package_bytes` only from the registration-owned
    /// path through a future race-safe filesystem opener. No client path is
    /// accepted by this API.
    #[cfg(test)]
    pub fn verify_package_bytes<'a>(
        &self,
        model_id: &ModelId,
        package_bytes: &'a [u8],
        budget: &ResourceBudget,
    ) -> Result<VerifiedModelPackage<'a>, EngineError> {
        let registration = self.resolve(model_id)?.clone();
        let actual_length = u64::try_from(package_bytes.len())
            .map_err(|_| EngineError::resource_exhaustion("package length is not representable"))?;
        validate_package_budget(actual_length, budget)?;
        let actual_hash = hash_package(package_bytes);
        if actual_hash != *registration.expected_hash() {
            return Err(EngineError::signature_failure(
                "model package digest does not match local configuration",
            ));
        }
        let parsed = ParsedModelPackage::parse(package_bytes)?;
        let signed_range = parsed.header().signed_content_range()?;
        let content_digest = hash_signed_content(&package_bytes[signed_range]);
        verify_package_signature(
            registration.trusted_signer(),
            parsed.signature().key_id(),
            &content_digest,
            parsed.signature().signature(),
        )?;
        Ok(VerifiedModelPackage { registration, parsed, package_bytes })
    }
}

impl ModelRegistry {
    #[cfg(unix)]
    pub(crate) fn model_root(&self) -> &Path {
        &self.model_root
    }
}

/// Validated metadata required before package-byte verification.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg(test)]
pub struct ModelLoadPlan {
    registration: RegisteredModel,
    header: ModelPackageHeader,
}

#[cfg(test)]
impl ModelLoadPlan {
    /// Returns the local registration selected by model ID.
    pub const fn registration(&self) -> &RegisteredModel {
        &self.registration
    }

    /// Returns the validated fixed package header.
    pub const fn header(&self) -> &ModelPackageHeader {
        &self.header
    }
}

/// Borrowed package bytes admitted through every cryptographic gate.
#[derive(Debug)]
#[cfg(test)]
pub struct VerifiedModelPackage<'a> {
    registration: RegisteredModel,
    parsed: ParsedModelPackage,
    package_bytes: &'a [u8],
}

#[cfg(test)]
impl VerifiedModelPackage<'_> {
    /// Returns the exact local registration used for verification.
    pub const fn registration(&self) -> &RegisteredModel {
        &self.registration
    }

    /// Returns the fully validated package metadata.
    pub const fn parsed(&self) -> &ParsedModelPackage {
        &self.parsed
    }

    /// Returns the immutable verified package bytes.
    pub const fn package_bytes(&self) -> &[u8] {
        self.package_bytes
    }

    /// Copies all verified tensor bytes into aggregate-accounted safe storage.
    pub fn materialize_tensors(
        &self,
        pool: &MemoryPool,
    ) -> Result<LoadedModelTensors, EngineError> {
        let tensor_count = u64::try_from(self.parsed.tensors().len()).map_err(|_| {
            EngineError::resource_exhaustion("loaded tensor count is not representable")
        })?;
        let metadata_bytes = tensor_count.checked_mul(128).ok_or_else(|| {
            EngineError::resource_exhaustion("loaded tensor metadata bound overflows")
        })?;
        let metadata_reservation = pool.reserve(metadata_bytes)?;
        let mut tensors = BTreeMap::new();
        for record in self.parsed.tensors() {
            let source = self.parsed.tensor_data(self.package_bytes, record)?;
            let tensor = Tensor::convert_storage_to_f32(
                record.dtype(),
                record.shape().clone(),
                source,
                pool,
            )?;
            if tensors.insert(record.name().as_str().to_owned(), tensor).is_some() {
                return Err(EngineError::internal_invariant(
                    "validated tensor index contains a duplicate name",
                ));
            }
        }
        Ok(LoadedModelTensors { tensors, _metadata_reservation: metadata_reservation })
    }
}

/// Aggregate-accounted tensors copied from one verified package.
#[derive(Debug)]
#[cfg(test)]
pub struct LoadedModelTensors {
    tensors: BTreeMap<String, Tensor>,
    _metadata_reservation: MemoryReservation,
}

#[cfg(test)]
impl LoadedModelTensors {
    /// Returns one tensor by its canonical architecture-owned name.
    pub fn get(&self, name: &str) -> Result<&Tensor, EngineError> {
        self.tensors
            .get(name)
            .ok_or_else(|| EngineError::corrupt_model_package("required model tensor is missing"))
    }

    /// Returns the number of loaded tensors.
    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    /// Returns whether no tensors were loaded.
    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    /// Transfers the verified tensor set to an architecture constructor.
    pub fn into_tensors(self) -> BTreeMap<String, Tensor> {
        self.tensors
    }
}

#[cfg(test)]
fn validate_package_budget(
    package_length: u64,
    budget: &ResourceBudget,
) -> Result<(), EngineError> {
    if package_length == 0
        || package_length > MAX_MODEL_PACKAGE_BYTES
        || package_length > budget.max_memory_bytes()
    {
        return Err(EngineError::resource_exhaustion(
            "model package exceeds its configured memory bound",
        ));
    }
    Ok(())
}

fn validate_absolute_normal_path(path: &Path) -> Result<(), EngineError> {
    if !path.is_absolute() {
        return Err(EngineError::invalid_input("model root must be an absolute path"));
    }
    for component in path.components() {
        match component {
            Component::RootDir | Component::Normal(_) => {}
            Component::Prefix(_) | Component::CurDir | Component::ParentDir => {
                return Err(EngineError::invalid_input(
                    "model root must be syntactically normalized",
                ));
            }
        }
    }
    Ok(())
}

/// Validates the closed configuration-owned package-name grammar.
///
/// This pure parser performs no filesystem access and is suitable for bounded
/// fuzzing. Runtime requests never contain this value.
pub fn validate_registered_package_path(path: &Path) -> Result<(), EngineError> {
    if path.as_os_str().is_empty() || path.is_absolute() {
        return Err(EngineError::invalid_input("registered model package path must be relative"));
    }
    if path.extension().and_then(|value| value.to_str()) != Some(MODEL_PACKAGE_EXTENSION) {
        return Err(EngineError::invalid_input(
            "registered model package must use the first-party extension",
        ));
    }
    let name = path
        .file_name()
        .and_then(|value| value.to_str())
        .ok_or_else(|| EngineError::invalid_input("registered package name must be valid UTF-8"))?;
    if !name.is_ascii()
        || name.chars().any(char::is_control)
        || name.starts_with('.')
        || name.contains("..")
        || !name
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-'))
    {
        return Err(EngineError::invalid_input(
            "registered package name is outside the safe local grammar",
        ));
    }
    let mut components = path.components();
    if !matches!(components.next(), Some(Component::Normal(_))) || components.next().is_some() {
        return Err(EngineError::invalid_input(
            "registered model package must be one direct child of the configured model root",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use personalai_engine_model_format::MODEL_PACKAGE_HEADER_BYTES;
    use personalai_engine_testkit::PackageFixtureBuilder;

    #[test]
    fn registry_resolves_only_configured_model_ids() {
        let fixture = PackageFixtureBuilder::new().build().expect("fixture must build");
        let mut registry = registry(&fixture);
        let model_id = ModelId::new("local.test-model").expect("model id must be valid");
        registry
            .register(
                model_id.clone(),
                "model.paimodel",
                fixture.package_hash(),
                fixture.trusted_signer().clone(),
            )
            .expect("configured path must register");
        let resolved = registry.resolve(&model_id).expect("registered id must resolve");
        assert_eq!(resolved.package_path(), Path::new("/var/lib/personalai/models/model.paimodel"));
        let unknown = ModelId::new("local.unknown").expect("model id must be valid");
        assert!(matches!(registry.resolve(&unknown), Err(EngineError::InvalidInput(_))));
    }

    #[test]
    fn registry_rejects_client_style_paths_and_duplicates() {
        let fixture = PackageFixtureBuilder::new().build().expect("fixture must build");
        let mut registry = registry(&fixture);
        let model_id = ModelId::new("model-a").expect("model id must be valid");
        for path in ["/tmp/model.paimodel", "../model.paimodel", "nested/model.paimodel"] {
            assert!(
                registry
                    .register(
                        model_id.clone(),
                        path,
                        fixture.package_hash(),
                        fixture.trusted_signer().clone(),
                    )
                    .is_err()
            );
        }
        registry
            .register(
                model_id.clone(),
                "model.paimodel",
                fixture.package_hash(),
                fixture.trusted_signer().clone(),
            )
            .expect("normal configured path must register");
        assert!(
            registry
                .register(
                    model_id,
                    "other.paimodel",
                    fixture.package_hash(),
                    fixture.trusted_signer().clone(),
                )
                .is_err()
        );
    }

    #[cfg(unix)]
    #[test]
    fn registered_package_reader_is_bounded_and_rejects_symlinks() {
        use std::fs;
        use std::os::unix::fs::{PermissionsExt, symlink};
        use std::sync::atomic::{AtomicU64, Ordering};

        static NEXT_DIRECTORY: AtomicU64 = AtomicU64::new(1);
        let sequence = NEXT_DIRECTORY.fetch_add(1, Ordering::Relaxed);
        let root = PathBuf::from("/tmp")
            .join(format!("personalai-engine-loader-{}-{sequence}", std::process::id()));
        fs::create_dir(&root).expect("model root must create");
        fs::set_permissions(&root, fs::Permissions::from_mode(0o700))
            .expect("model root mode must set");
        let fixture = PackageFixtureBuilder::new().build().expect("fixture must build");
        let model_id = ModelId::new("secure-reader").expect("model ID must validate");
        let package_path = root.join("model.paimodel");
        fs::write(&package_path, fixture.bytes()).expect("fixture must write");
        fs::set_permissions(&package_path, fs::Permissions::from_mode(0o400))
            .expect("package mode must set");
        let mut registry = ModelRegistry::new(&root).expect("registry must construct");
        registry
            .register(
                model_id.clone(),
                "model.paimodel",
                fixture.package_hash(),
                fixture.trusted_signer().clone(),
            )
            .expect("package must register");
        let budget = ResourceBudget::new(1 << 20, 1 << 19, 1).expect("budget must validate");
        assert_eq!(
            registry
                .read_registered_package(&model_id, &budget)
                .expect("registered package must read"),
            fixture.bytes()
        );

        fs::remove_file(&package_path).expect("package must remove");
        let target = root.join("target.paimodel");
        fs::write(&target, fixture.bytes()).expect("target must write");
        symlink(&target, &package_path).expect("symlink must create");
        assert!(matches!(
            registry.read_registered_package(&model_id, &budget),
            Err(EngineError::AuthenticationFailure(_))
        ));
        fs::remove_file(&package_path).expect("symlink must remove");
        fs::remove_file(&target).expect("target must remove");
        fs::remove_dir(&root).expect("model root must remove");
    }

    #[test]
    fn exact_signed_package_verifies_and_any_tampering_fails_closed() {
        let fixture = PackageFixtureBuilder::new().build().expect("fixture must build");
        let mut registry = registry(&fixture);
        let model_id = ModelId::new("model-a").expect("model id must be valid");
        registry
            .register(
                model_id.clone(),
                "model.paimodel",
                fixture.package_hash(),
                fixture.trusted_signer().clone(),
            )
            .expect("registration must succeed");
        let budget = ResourceBudget::new(1 << 30, 1 << 29, 1).expect("budget must be valid");
        let verified = registry
            .verify_package_bytes(&model_id, fixture.bytes(), &budget)
            .expect("exact registered package must verify");
        assert_eq!(verified.registration().model_id(), &model_id);
        assert_eq!(verified.parsed().manifest().tensor_count(), 2);
        let pool = MemoryPool::new(1 << 20).expect("pool must validate");
        let loaded = verified.materialize_tensors(&pool).expect("tensors must materialize");
        assert_eq!(loaded.len(), 2);
        assert_eq!(
            loaded.get("tensor.000000").expect("tensor must exist").shape().dimensions(),
            &[4]
        );

        let mut tampered = fixture.to_bytes();
        tampered[MODEL_PACKAGE_HEADER_BYTES] ^= 1;
        assert!(matches!(
            registry.verify_package_bytes(&model_id, &tampered, &budget),
            Err(EngineError::SignatureFailure(_))
        ));

        for index in 0..fixture.bytes().len() {
            let mut mutation = fixture.to_bytes();
            mutation[index] ^= 1;
            assert!(
                registry.verify_package_bytes(&model_id, &mutation, &budget).is_err(),
                "single-byte package mutation at offset {index} was accepted"
            );
        }
        for end in 0..fixture.bytes().len() {
            assert!(
                registry.verify_package_bytes(&model_id, &fixture.bytes()[..end], &budget).is_err(),
                "package truncation at offset {end} was accepted"
            );
        }
    }

    #[test]
    fn budget_is_enforced_before_package_parsing() {
        let fixture = PackageFixtureBuilder::new().build().expect("fixture must build");
        let mut registry = registry(&fixture);
        let model_id = ModelId::new("model-a").expect("model id must be valid");
        registry
            .register(
                model_id.clone(),
                "model.paimodel",
                fixture.package_hash(),
                fixture.trusted_signer().clone(),
            )
            .expect("registration must succeed");
        let budget = ResourceBudget::new(1, 1, 1).expect("small budget remains structurally valid");
        assert!(matches!(
            registry.verify_package_bytes(&model_id, fixture.bytes(), &budget),
            Err(EngineError::ResourceExhaustion(_))
        ));
    }

    fn registry(_fixture: &personalai_engine_testkit::SignedPackageFixture) -> ModelRegistry {
        ModelRegistry::new("/var/lib/personalai/models").expect("absolute model root must validate")
    }
}
