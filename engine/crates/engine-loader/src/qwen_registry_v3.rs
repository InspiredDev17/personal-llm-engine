//! Exact development-only Qwen registry V3 and Models 2.0 binding.
//!
//! The signed V3 envelope is a transitional local adapter. Canonical package
//! identity comes from Sovereign Models 2.0, while package bytes remain
//! authenticated by the independent PAIDIR01 catalog signature.

use personalai_engine_crypto::{
    TrustedSigner, hash_package, hash_signed_content, verify_model_foundry_qwen_registry_signature,
};
use personalai_engine_model_format::{QWEN_REGISTRY_V3_MAX_BYTES, SignedQwenRegistryEntryV3};
use personalai_engine_types::{EngineError, ModelId, ModelPackageHash};
use sovereign_ai_contracts::Validate;
use sovereign_ai_contracts::identity::{ModelArtifactId, ModelId as LogicalModelId};
use sovereign_ai_contracts::inference_runtime_v3::ArtifactDigestAlgorithmV3;
use sovereign_ai_contracts::models_v2::{
    ExecutionPackageDescriptor, ExecutionPackageMember, ExecutionPackageMemberRole,
    PackageStorageDtype,
};
#[cfg(unix)]
use std::fs::{self, OpenOptions};
#[cfg(unix)]
use std::io::Read;
#[cfg(unix)]
use std::os::unix::fs::{MetadataExt, OpenOptionsExt, PermissionsExt};
#[cfg(unix)]
use std::path::Path;

/// Exact Contracts source for Models 2.0 and runtime-v3 identity.
pub const MODELS_V2_CONTRACT_COMMIT: &str = "dd3e18496b669a8aa28f822ca85e0299c16c9b99";
/// Maximum exact canonical descriptor bytes.
pub const MAX_MODELS_V2_DESCRIPTOR_BYTES: usize = 1024 * 1024;
/// Exact canonical package profile identifier.
pub const QWEN_MODELS_V2_PACKAGE_FORMAT_ID: &str = "personalai.directory_paimodel";
/// Exact canonical package profile revision.
pub const QWEN_MODELS_V2_PACKAGE_FORMAT_VERSION: &str = "1.1.0";
/// Exact canonical tokenizer identity shared by the reviewed Qwen2.5 profiles.
pub const QWEN_MODELS_V2_TOKENIZER_ID: &str = "tokenizer.qwen2_5.shared.989aa798";
/// Exact canonical tokenizer wire identity.
pub const QWEN_MODELS_V2_TOKENIZER_FORMAT_ID: &str = "personalai.aiptok";
/// Exact canonical tokenizer wire revision.
pub const QWEN_MODELS_V2_TOKENIZER_FORMAT_VERSION: &str = "2.0.0";
/// Exact reviewed model/logit vocabulary rows.
pub const QWEN_MODELS_V2_MODEL_VOCABULARY: u32 = 151_936;
/// Exact reviewed defined and selectable tokenizer rows.
pub const QWEN_MODELS_V2_SELECTABLE_TOKENS: u32 = 151_665;
/// Exact canonical Qwen decoder architecture.
pub const QWEN_MODELS_V2_ARCHITECTURE_ID: &str = "qwen2.5-rms-rope-swiglu-gqa-bias-tied-v1";
/// Exact Qwen split-half, unscaled position semantics.
pub const QWEN_MODELS_V2_POSITION_SEMANTICS_ID: &str = "rope.split_half.unscaled.v1";
/// Exact unquantized BF16 storage profile.
pub const QWEN_MODELS_V2_QUANTIZATION_PROFILE_ID: &str = "unquantized.bf16.v1";

const QWEN_MODELS_V2_BACKEND_ID: &str = "arm64";
const DESCRIPTOR_DIRECTORY: &str = "descriptors";
const REGISTRY_DIRECTORY: &str = "registry-v3";

/// Immutable owner selection for one signed development registry decision.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct QwenRegistrySelectionV3 {
    expected_model_id: ModelId,
    expected_logical_model_id: LogicalModelId,
    expected_artifact_id: ModelArtifactId,
    expected_entry_hash: ModelPackageHash,
    expected_sequence: u64,
    minimum_sequence: u64,
    registry_signer: TrustedSigner,
    package_signer: TrustedSigner,
}

impl QwenRegistrySelectionV3 {
    /// Constructs one exact, anti-rollback, two-key development selection.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        expected_model_id: ModelId,
        expected_logical_model_id: LogicalModelId,
        expected_artifact_id: ModelArtifactId,
        expected_entry_hash: ModelPackageHash,
        expected_sequence: u64,
        minimum_sequence: u64,
        registry_signer: TrustedSigner,
        package_signer: TrustedSigner,
    ) -> Result<Self, EngineError> {
        if minimum_sequence == 0 || expected_sequence < minimum_sequence {
            return Err(EngineError::invalid_input(
                "Qwen registry sequence and anti-rollback checkpoint are invalid",
            ));
        }
        if !registry_signer.is_active() || !package_signer.is_active() {
            return Err(EngineError::authentication_failure(
                "Qwen registry selection requires active public trust identities",
            ));
        }
        if registry_signer.key_id() == package_signer.key_id()
            || registry_signer.public_key() == package_signer.public_key()
        {
            return Err(EngineError::authentication_failure(
                "Qwen registry and package signing identities must be distinct",
            ));
        }
        Ok(Self {
            expected_model_id,
            expected_logical_model_id,
            expected_artifact_id,
            expected_entry_hash,
            expected_sequence,
            minimum_sequence,
            registry_signer,
            package_signer,
        })
    }

    /// Parses canonical logical and artifact identities from owner CLI text.
    #[allow(clippy::too_many_arguments)]
    pub fn from_owner_text(
        expected_model_id: ModelId,
        expected_logical_model_id: &str,
        expected_artifact_id: &str,
        expected_entry_hash: ModelPackageHash,
        expected_sequence: u64,
        minimum_sequence: u64,
        registry_signer: TrustedSigner,
        package_signer: TrustedSigner,
    ) -> Result<Self, EngineError> {
        Self::new(
            expected_model_id,
            LogicalModelId::new(expected_logical_model_id).map_err(|_| {
                EngineError::invalid_input("Qwen logical model identity is invalid")
            })?,
            ModelArtifactId::new(expected_artifact_id)
                .map_err(|_| EngineError::invalid_input("Qwen artifact identity is invalid"))?,
            expected_entry_hash,
            expected_sequence,
            minimum_sequence,
            registry_signer,
            package_signer,
        )
    }

    /// Returns the exact provider model selected by owner configuration.
    pub const fn expected_model_id(&self) -> &ModelId {
        &self.expected_model_id
    }

    /// Returns the exact canonical logical model selected by owner configuration.
    pub const fn expected_logical_model_id(&self) -> &LogicalModelId {
        &self.expected_logical_model_id
    }

    /// Returns the exact canonical artifact selected by owner configuration.
    pub const fn expected_artifact_id(&self) -> &ModelArtifactId {
        &self.expected_artifact_id
    }

    /// Returns SHA-384 of the exact signed registry envelope.
    pub const fn expected_entry_hash(&self) -> ModelPackageHash {
        self.expected_entry_hash
    }

    /// Returns the exact registry sequence.
    pub const fn expected_sequence(&self) -> u64 {
        self.expected_sequence
    }

    /// Returns the retained anti-rollback sequence floor.
    pub const fn minimum_sequence(&self) -> u64 {
        self.minimum_sequence
    }

    /// Detects forbidden trust-role reuse at startup.
    pub fn conflicts_with_identity(
        &self,
        key_id: personalai_engine_types::SignerKeyId,
        public_key: [u8; 32],
    ) -> bool {
        self.registry_signer.key_id() == key_id
            || self.package_signer.key_id() == key_id
            || self.registry_signer.public_key() == public_key
            || self.package_signer.public_key() == public_key
    }
}

/// Cryptographically verified V3 decision and canonical Models 2.0 descriptor.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VerifiedQwenRegistrySelectionV3 {
    descriptor: ExecutionPackageDescriptor,
    descriptor_hash: ModelPackageHash,
    package_catalog_hash: ModelPackageHash,
    registry_entry_hash: ModelPackageHash,
    registry_sequence: u64,
    package_signer: TrustedSigner,
}

impl VerifiedQwenRegistrySelectionV3 {
    /// Returns the canonical package descriptor.
    pub const fn descriptor(&self) -> &ExecutionPackageDescriptor {
        &self.descriptor
    }

    /// Returns SHA-384 of exact canonical descriptor bytes.
    pub const fn descriptor_hash(&self) -> ModelPackageHash {
        self.descriptor_hash
    }

    /// Returns SHA-384 of the independently signed PAIDIR01 catalog.
    pub const fn package_catalog_hash(&self) -> ModelPackageHash {
        self.package_catalog_hash
    }

    /// Returns SHA-384 of the signed V3 envelope.
    pub const fn registry_entry_hash(&self) -> ModelPackageHash {
        self.registry_entry_hash
    }

    /// Returns the signed monotonic development sequence.
    pub const fn registry_sequence(&self) -> u64 {
        self.registry_sequence
    }

    /// Returns package-verification trust selected by the signed entry and owner.
    pub const fn package_signer(&self) -> &TrustedSigner {
        &self.package_signer
    }
}

/// Verifies a signed development V3 decision and exact canonical Models 2.0 descriptor.
pub fn verify_qwen_registry_selection_v3(
    envelope: &[u8],
    descriptor_bytes: &[u8],
    selection: &QwenRegistrySelectionV3,
) -> Result<VerifiedQwenRegistrySelectionV3, EngineError> {
    if hash_package(envelope) != selection.expected_entry_hash {
        return Err(EngineError::authentication_failure(
            "Qwen registry entry identity differs from owner configuration",
        ));
    }
    let signed = SignedQwenRegistryEntryV3::parse(envelope)?;
    let entry = signed.entry();
    if entry.registry_sequence() != selection.expected_sequence
        || entry.registry_sequence() < selection.minimum_sequence
    {
        return Err(EngineError::authentication_failure(
            "Qwen registry sequence differs from owner selection or anti-rollback checkpoint",
        ));
    }
    if entry.model_id() != selection.expected_model_id.as_str()
        || entry.artifact_id() != selection.expected_artifact_id.as_str()
    {
        return Err(EngineError::authentication_failure(
            "Qwen registry identities differ from owner selection",
        ));
    }
    if entry.registry_signer() != selection.registry_signer.key_id()
        || entry.package_signer() != selection.package_signer.key_id()
    {
        return Err(EngineError::signature_failure(
            "Qwen registry trust identities differ from owner configuration",
        ));
    }
    verify_model_foundry_qwen_registry_signature(
        &selection.registry_signer,
        entry.registry_signer(),
        &hash_signed_content(signed.entry_bytes(envelope)?),
        signed.signature(),
    )?;
    let descriptor_hash = hash_package(descriptor_bytes);
    if descriptor_hash != entry.descriptor_hash() {
        return Err(EngineError::authentication_failure(
            "Models 2.0 descriptor identity differs from the signed Qwen registry entry",
        ));
    }
    let descriptor = decode_qwen_models_v2_descriptor(descriptor_bytes)?;
    validate_qwen_models_v2_profile(&descriptor, selection, entry.package_catalog_hash())?;
    Ok(VerifiedQwenRegistrySelectionV3 {
        descriptor,
        descriptor_hash,
        package_catalog_hash: entry.package_catalog_hash(),
        registry_entry_hash: selection.expected_entry_hash,
        registry_sequence: entry.registry_sequence(),
        package_signer: selection.package_signer.clone(),
    })
}

fn decode_qwen_models_v2_descriptor(
    bytes: &[u8],
) -> Result<ExecutionPackageDescriptor, EngineError> {
    if bytes.is_empty() || bytes.len() > MAX_MODELS_V2_DESCRIPTOR_BYTES {
        return Err(EngineError::resource_exhaustion(
            "Models 2.0 descriptor length is outside hard bounds",
        ));
    }
    let descriptor: ExecutionPackageDescriptor = serde_json::from_slice(bytes)
        .map_err(|_| EngineError::corrupt_model_package("Models 2.0 descriptor is invalid"))?;
    descriptor.validate().map_err(|_| {
        EngineError::corrupt_model_package("Models 2.0 descriptor validation failed")
    })?;
    let canonical = serde_json::to_vec(&descriptor).map_err(|_| {
        EngineError::internal_invariant("Models 2.0 descriptor serialization failed")
    })?;
    if canonical != bytes {
        return Err(EngineError::corrupt_model_package(
            "Models 2.0 descriptor is not canonical byte-for-byte",
        ));
    }
    Ok(descriptor)
}

fn validate_qwen_models_v2_profile(
    descriptor: &ExecutionPackageDescriptor,
    selection: &QwenRegistrySelectionV3,
    package_catalog_hash: ModelPackageHash,
) -> Result<(), EngineError> {
    let runtime = &descriptor.runtime_identity;
    if runtime.model_id != *selection.expected_logical_model_id()
        || runtime.provider_model_id != selection.expected_model_id.as_str()
        || runtime.artifact_id != *selection.expected_artifact_id()
        || runtime.artifact_digest_algorithm != ArtifactDigestAlgorithmV3::Sha384
        || runtime.artifact_digest != sha384_hex(package_catalog_hash)
        || runtime.backend_id != QWEN_MODELS_V2_BACKEND_ID
    {
        return Err(EngineError::authentication_failure(
            "Models 2.0 runtime identity differs from the exact Qwen owner/package binding",
        ));
    }
    if descriptor.package_identity_member_id != "catalog"
        || descriptor.package_format_id != QWEN_MODELS_V2_PACKAGE_FORMAT_ID
        || descriptor.package_format_version != QWEN_MODELS_V2_PACKAGE_FORMAT_VERSION
        || descriptor.tokenizer.tokenizer_id != QWEN_MODELS_V2_TOKENIZER_ID
        || descriptor.tokenizer.member_id != "tokenizer"
        || descriptor.tokenizer.format_id != QWEN_MODELS_V2_TOKENIZER_FORMAT_ID
        || descriptor.tokenizer.format_version != QWEN_MODELS_V2_TOKENIZER_FORMAT_VERSION
        || descriptor.tokenizer.model_vocabulary_size != QWEN_MODELS_V2_MODEL_VOCABULARY
        || descriptor.tokenizer.defined_token_count != QWEN_MODELS_V2_SELECTABLE_TOKENS
        || descriptor.tokenizer.selectable_token_count != QWEN_MODELS_V2_SELECTABLE_TOKENS
        || descriptor.architecture.architecture_id != QWEN_MODELS_V2_ARCHITECTURE_ID
        || descriptor.architecture.position_semantics_id != QWEN_MODELS_V2_POSITION_SEMANTICS_ID
        || descriptor.quantization.profile_id != QWEN_MODELS_V2_QUANTIZATION_PROFILE_ID
        || descriptor.quantization.storage_dtype != PackageStorageDtype::Bf16
        || descriptor.tensor_inventory.member_id != "manifest"
        || descriptor.tensor_inventory.tensor_count != 338
    {
        return Err(EngineError::corrupt_model_package(
            "Models 2.0 descriptor is outside the exact Qwen2.5 PAIDIR01 profile",
        ));
    }
    let expected = [
        ("manifest", 0, ExecutionPackageMemberRole::PackageManifest),
        ("tokenizer", 1, ExecutionPackageMemberRole::Tokenizer),
        ("provenance", 2, ExecutionPackageMemberRole::Provenance),
        ("tensor_shard.0000", 3, ExecutionPackageMemberRole::TensorShard),
        ("catalog", 4, ExecutionPackageMemberRole::IntegrityCatalog),
        ("signature", 5, ExecutionPackageMemberRole::IntegritySignature),
    ];
    if descriptor.members.len() != expected.len()
        || descriptor.members.iter().zip(expected).any(|(member, (id, ordinal, role))| {
            member.member_id != id
                || member.ordinal != ordinal
                || member.role != role
                || member.content.algorithm != ArtifactDigestAlgorithmV3::Sha384
        })
    {
        return Err(EngineError::corrupt_model_package(
            "Models 2.0 descriptor member set differs from the exact six-file Qwen package",
        ));
    }
    Ok(())
}

fn sha384_hex(digest: ModelPackageHash) -> String {
    lowercase_hex(digest.as_bytes())
}

fn lowercase_hex(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut encoded = String::with_capacity(bytes.len().saturating_mul(2));
    for byte in bytes {
        encoded.push(char::from(DIGITS[usize::from(byte >> 4)]));
        encoded.push(char::from(DIGITS[usize::from(byte & 0x0f)]));
    }
    encoded
}

pub(crate) fn descriptor_member<'a>(
    descriptor: &'a ExecutionPackageDescriptor,
    member_id: &str,
) -> Result<&'a ExecutionPackageMember, EngineError> {
    descriptor
        .members
        .iter()
        .find(|member| member.member_id == member_id)
        .ok_or_else(|| EngineError::internal_invariant("validated descriptor member disappeared"))
}

/// Securely reads and verifies the sequence-derived registry record and
/// digest-derived descriptor from protected local storage.
#[cfg(unix)]
pub(crate) fn read_verified_qwen_registry_selection_v3(
    approved_root: &Path,
    selection: &QwenRegistrySelectionV3,
) -> Result<VerifiedQwenRegistrySelectionV3, EngineError> {
    validate_protected_directory(approved_root, "Model Foundry approved root")?;
    let registry_root = approved_root.join(REGISTRY_DIRECTORY);
    validate_protected_directory(&registry_root, "Qwen registry root")?;
    let descriptor_root = approved_root.join(DESCRIPTOR_DIRECTORY);
    validate_protected_directory(&descriptor_root, "Models descriptor root")?;
    let envelope_name = format!("{:020}.qwen.registry.json", selection.expected_sequence);
    let envelope = read_protected_file(
        &registry_root.join(envelope_name),
        QWEN_REGISTRY_V3_MAX_BYTES,
        "Qwen registry entry",
    )?;
    let signed = SignedQwenRegistryEntryV3::parse(&envelope)?;
    let descriptor_hash = signed.entry().descriptor_hash();
    let descriptor_name = format!("{}.models-v2.json", sha384_hex(descriptor_hash));
    let descriptor = read_protected_file(
        &descriptor_root.join(descriptor_name),
        MAX_MODELS_V2_DESCRIPTOR_BYTES,
        "Models 2.0 descriptor",
    )?;
    verify_qwen_registry_selection_v3(&envelope, &descriptor, selection)
}

#[cfg(unix)]
fn validate_protected_directory(path: &Path, label: &str) -> Result<(), EngineError> {
    let metadata = fs::symlink_metadata(path)
        .map_err(|_| EngineError::invalid_input(format!("{label} is unavailable")))?;
    if !metadata.is_dir()
        || metadata.file_type().is_symlink()
        || metadata.permissions().mode() & 0o222 != 0
    {
        return Err(EngineError::authentication_failure(format!(
            "{label} must be a protected non-symlink directory"
        )));
    }
    Ok(())
}

#[cfg(unix)]
fn read_protected_file(path: &Path, maximum: usize, label: &str) -> Result<Vec<u8>, EngineError> {
    let before = fs::symlink_metadata(path)
        .map_err(|_| EngineError::invalid_input(format!("{label} is unavailable")))?;
    let maximum = u64::try_from(maximum)
        .map_err(|_| EngineError::internal_invariant("protected file bound exceeds u64"))?;
    if !before.is_file()
        || before.file_type().is_symlink()
        || before.permissions().mode() & 0o222 != 0
        || before.len() == 0
        || before.len() > maximum
    {
        return Err(EngineError::authentication_failure(format!("{label} metadata is invalid")));
    }
    let mut file = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_CLOEXEC | libc::O_NOFOLLOW)
        .open(path)
        .map_err(|_| EngineError::authentication_failure(format!("{label} secure open failed")))?;
    let opened = file
        .metadata()
        .map_err(|_| EngineError::authentication_failure(format!("{label} metadata failed")))?;
    if opened.dev() != before.dev()
        || opened.ino() != before.ino()
        || opened.len() != before.len()
        || !opened.is_file()
    {
        return Err(EngineError::authentication_failure(format!(
            "{label} changed during secure open"
        )));
    }
    let length = usize::try_from(opened.len())
        .map_err(|_| EngineError::resource_exhaustion(format!("{label} is not addressable")))?;
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(length)
        .map_err(|_| EngineError::resource_exhaustion(format!("{label} allocation failed")))?;
    bytes.resize(length, 0);
    file.read_exact(&mut bytes)
        .map_err(|_| EngineError::corrupt_model_package(format!("{label} is truncated")))?;
    let mut trailing = [0_u8; 1];
    if file
        .read(&mut trailing)
        .map_err(|_| EngineError::corrupt_model_package(format!("{label} trailing check failed")))?
        != 0
    {
        return Err(EngineError::corrupt_model_package(format!("{label} grew during secure read")));
    }
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ed25519_dalek::{Signer, SigningKey};
    use personalai_engine_crypto::model_foundry_qwen_registry_signature_message;
    use personalai_engine_types::SignerKeyId;
    use sovereign_ai_contracts::inference_runtime_v3::{
        ArtifactDigestAlgorithmV3, RuntimeIdentityV3,
    };
    use sovereign_ai_contracts::models_v2::{
        ArchitectureExecutionBinding, ExecutionPackageDescriptorContractVersion,
        ExecutionPackageDescriptorKind, ExecutionPackageMember, PackageDigest, QuantizationBinding,
        TensorInventoryBinding, TokenizerArtifactBinding,
    };

    fn package_hash(byte: u8) -> ModelPackageHash {
        ModelPackageHash::from_sha384_bytes([byte; 48])
    }

    fn member(
        id: &str,
        ordinal: u16,
        role: ExecutionPackageMemberRole,
        byte: u8,
    ) -> ExecutionPackageMember {
        ExecutionPackageMember {
            member_id: id.to_owned(),
            ordinal,
            role,
            content: PackageDigest {
                algorithm: ArtifactDigestAlgorithmV3::Sha384,
                digest: sha384_hex(package_hash(byte)),
            },
            size_bytes: u64::from(byte),
        }
    }

    fn descriptor() -> ExecutionPackageDescriptor {
        let members = vec![
            member("manifest", 0, ExecutionPackageMemberRole::PackageManifest, 1),
            member("tokenizer", 1, ExecutionPackageMemberRole::Tokenizer, 2),
            member("provenance", 2, ExecutionPackageMemberRole::Provenance, 3),
            member("tensor_shard.0000", 3, ExecutionPackageMemberRole::TensorShard, 4),
            member("catalog", 4, ExecutionPackageMemberRole::IntegrityCatalog, 5),
            member("signature", 5, ExecutionPackageMemberRole::IntegritySignature, 6),
        ];
        ExecutionPackageDescriptor {
            contract_version: ExecutionPackageDescriptorContractVersion,
            kind: ExecutionPackageDescriptorKind::ExecutionPackageDescriptor,
            runtime_identity: RuntimeIdentityV3 {
                model_id: LogicalModelId::new("mdl_qwen25_general_1_5b_989aa798").unwrap(),
                provider_model_id: "qwen2.5-1.5b-instruct-989aa798".to_owned(),
                artifact_id: ModelArtifactId::new("art_qwen25_general_1_5b_989aa798").unwrap(),
                artifact_digest_algorithm: ArtifactDigestAlgorithmV3::Sha384,
                artifact_digest: sha384_hex(package_hash(5)),
                backend_id: "arm64".to_owned(),
                backend_version: "version.0_2_0".to_owned(),
                backend_instance_id: "backend.arm64.local.dev1".to_owned(),
                engine_instance_id: "engine.local.dev1".to_owned(),
            },
            package_identity_member_id: "catalog".to_owned(),
            package_format_id: QWEN_MODELS_V2_PACKAGE_FORMAT_ID.to_owned(),
            package_format_version: QWEN_MODELS_V2_PACKAGE_FORMAT_VERSION.to_owned(),
            package_size_bytes: members.iter().map(|member| member.size_bytes).sum(),
            tokenizer: TokenizerArtifactBinding {
                tokenizer_id: QWEN_MODELS_V2_TOKENIZER_ID.to_owned(),
                member_id: "tokenizer".to_owned(),
                format_id: QWEN_MODELS_V2_TOKENIZER_FORMAT_ID.to_owned(),
                format_version: QWEN_MODELS_V2_TOKENIZER_FORMAT_VERSION.to_owned(),
                content: members[1].content.clone(),
                size_bytes: members[1].size_bytes,
                model_vocabulary_size: QWEN_MODELS_V2_MODEL_VOCABULARY,
                defined_token_count: QWEN_MODELS_V2_SELECTABLE_TOKENS,
                selectable_token_count: QWEN_MODELS_V2_SELECTABLE_TOKENS,
            },
            architecture: ArchitectureExecutionBinding {
                architecture_id: QWEN_MODELS_V2_ARCHITECTURE_ID.to_owned(),
                position_semantics_id: QWEN_MODELS_V2_POSITION_SEMANTICS_ID.to_owned(),
                context_limit: 32_768,
            },
            quantization: QuantizationBinding {
                profile_id: QWEN_MODELS_V2_QUANTIZATION_PROFILE_ID.to_owned(),
                storage_dtype: PackageStorageDtype::Bf16,
            },
            tensor_inventory: TensorInventoryBinding {
                member_id: "manifest".to_owned(),
                content: members[0].content.clone(),
                tensor_count: 338,
            },
            members,
            created_at: "2026-08-23T12:00:00.000Z".to_owned(),
        }
    }

    fn trusted(seed: u8, key_byte: u8) -> (SigningKey, TrustedSigner) {
        let signing = SigningKey::from_bytes(&[seed; 32]);
        let trusted = TrustedSigner::active(
            SignerKeyId::from_bytes([key_byte; 16]).unwrap(),
            signing.verifying_key().to_bytes(),
        )
        .unwrap();
        (signing, trusted)
    }

    fn envelope(
        descriptor_hash: ModelPackageHash,
        catalog_hash: ModelPackageHash,
        registry: &SigningKey,
        registry_id: SignerKeyId,
        package_id: SignerKeyId,
    ) -> Vec<u8> {
        let entry = format!(
            "{{\"approval_timestamp\":\"2026-08-23T12:00:00Z\",\"artifact_id\":\"art_qwen25_general_1_5b_989aa798\",\"descriptor_sha384\":\"{}\",\"development_execution_authorized\":true,\"engine_compatibility\":\"personalai.directory_paimodel+qwen2.5+aiptok02+bf16+arm64-v1\",\"minimum_engine_version\":\"0.2.0\",\"model_id\":\"qwen2.5-1.5b-instruct-989aa798\",\"package_catalog_sha384\":\"{}\",\"package_signer\":\"{package_id}\",\"production_activation\":false,\"registry_sequence\":1,\"registry_signer\":\"{registry_id}\",\"schema_version\":3,\"status\":\"development_fixture_authorized\"}}",
            sha384_hex(descriptor_hash),
            sha384_hex(catalog_hash),
        );
        let digest = hash_signed_content(entry.as_bytes());
        let message = model_foundry_qwen_registry_signature_message(&digest).unwrap();
        let signature = registry.sign(&message);
        format!("{{\"entry\":{entry},\"signature\":\"{}\"}}", hex(signature.to_bytes()))
            .into_bytes()
    }

    fn hex<const N: usize>(bytes: [u8; N]) -> String {
        lowercase_hex(&bytes)
    }

    fn selection(
        envelope_hash: ModelPackageHash,
        registry: TrustedSigner,
        package: TrustedSigner,
    ) -> QwenRegistrySelectionV3 {
        QwenRegistrySelectionV3::new(
            ModelId::new("qwen2.5-1.5b-instruct-989aa798").unwrap(),
            LogicalModelId::new("mdl_qwen25_general_1_5b_989aa798").unwrap(),
            ModelArtifactId::new("art_qwen25_general_1_5b_989aa798").unwrap(),
            envelope_hash,
            1,
            1,
            registry,
            package,
        )
        .unwrap()
    }

    #[test]
    fn signed_registry_and_canonical_descriptor_bind_exact_package() {
        let descriptor = serde_json::to_vec(&descriptor()).unwrap();
        let descriptor_hash = hash_package(&descriptor);
        let (registry_key, registry_trust) = trusted(7, 0x71);
        let (_, package_trust) = trusted(8, 0x72);
        let bytes = envelope(
            descriptor_hash,
            package_hash(5),
            &registry_key,
            registry_trust.key_id(),
            package_trust.key_id(),
        );
        let verified = verify_qwen_registry_selection_v3(
            &bytes,
            &descriptor,
            &selection(hash_package(&bytes), registry_trust, package_trust),
        )
        .expect("exact signed selection validates");
        assert_eq!(verified.descriptor_hash(), descriptor_hash);
        assert_eq!(verified.registry_sequence(), 1);
        assert_eq!(verified.descriptor().runtime_identity.backend_id, "arm64");
    }

    #[test]
    fn descriptor_or_runtime_substitution_fails_closed() {
        let original = descriptor();
        let descriptor = serde_json::to_vec(&original).unwrap();
        let descriptor_hash = hash_package(&descriptor);
        let (registry_key, registry_trust) = trusted(9, 0x73);
        let (_, package_trust) = trusted(10, 0x74);
        let bytes = envelope(
            descriptor_hash,
            package_hash(5),
            &registry_key,
            registry_trust.key_id(),
            package_trust.key_id(),
        );
        let selection = selection(hash_package(&bytes), registry_trust, package_trust);
        let mut changed = original;
        changed.runtime_identity.backend_instance_id = "backend.arm64.other.dev1".to_owned();
        assert!(
            verify_qwen_registry_selection_v3(
                &bytes,
                &serde_json::to_vec(&changed).unwrap(),
                &selection,
            )
            .is_err()
        );
        assert!(
            verify_qwen_registry_selection_v3(
                &bytes,
                &descriptor[..descriptor.len() - 1],
                &selection
            )
            .is_err()
        );
    }
}
