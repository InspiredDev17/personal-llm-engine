//! Canonical Sovereign Models 2.1 admission consumer.
//!
//! This module authenticates exact, content-free admission evidence against
//! owner-pinned trust and exact Models 2.0 package metadata. Authentication is
//! deliberately separate from package registration and execution. In
//! particular, a valid admission cannot construct a loader registration,
//! activate production, or make an unsupported architecture executable.

use crate::{canonical_json, qwen_registry_v3::decode_models_v2_descriptor};
use personalai_engine_crypto::{TrustedSigner, verify_canonical_contract_signature};
use personalai_engine_types::{EngineError, ModelPackageHash, Sha256Digest, SignerKeyId};
use sovereign_ai_contracts::Validate;
use sovereign_ai_contracts::inference_runtime_v3::RuntimeIdentityV3;
use sovereign_ai_contracts::model_admission_v1::{
    ModelAdmissionId, ModelAdmissionState, ModelExecutionAdmission,
    model_execution_admission_signing_transcript, validate_model_execution_admission_descriptor,
    validate_model_execution_admission_package_catalog,
};
use sovereign_ai_contracts::models_v2::ExecutionPackageDescriptor;

/// Exact published Contracts revision that owns Models 2.1 admission.
pub const MODEL_EXECUTION_ADMISSION_CONTRACT_COMMIT: &str =
    "f3ea60596c33faca673ad368304f162b30f2ac2f";
/// Maximum canonical admission document accepted by the engine boundary.
pub const MAX_MODEL_EXECUTION_ADMISSION_BYTES: usize = 1024 * 1024;
/// Maximum package-catalog bytes accepted for admission binding.
pub const MAX_MODEL_ADMISSION_CATALOG_BYTES: usize = 16 * 1024 * 1024;

/// Immutable owner selection used to authenticate one exact admission.
///
/// This is local trust and anti-rollback configuration, not a competing wire
/// contract. The canonical document and identities remain Contracts types.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModelExecutionAdmissionSelection {
    expected_admission_id: ModelAdmissionId,
    expected_admission_digest: Sha256Digest,
    expected_revision: u64,
    expected_registry_sequence: u64,
    minimum_registry_sequence: u64,
    expected_lineage_id: String,
    expected_runtime_identity: RuntimeIdentityV3,
    expected_engine_profile_id: String,
    expected_registry_authority_id: String,
    expected_package_authority_id: String,
    registry_signer: TrustedSigner,
    package_signer: TrustedSigner,
    owner_allows_development_execution: bool,
}

impl ModelExecutionAdmissionSelection {
    /// Constructs one exact, owner-pinned admission selection.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        expected_admission_id: ModelAdmissionId,
        expected_admission_digest: Sha256Digest,
        expected_revision: u64,
        expected_registry_sequence: u64,
        minimum_registry_sequence: u64,
        expected_lineage_id: impl Into<String>,
        expected_runtime_identity: RuntimeIdentityV3,
        expected_engine_profile_id: impl Into<String>,
        expected_registry_authority_id: impl Into<String>,
        expected_package_authority_id: impl Into<String>,
        registry_signer: TrustedSigner,
        package_signer: TrustedSigner,
        owner_allows_development_execution: bool,
    ) -> Result<Self, EngineError> {
        if expected_revision == 0
            || expected_registry_sequence == 0
            || minimum_registry_sequence == 0
            || expected_registry_sequence < minimum_registry_sequence
        {
            return Err(EngineError::invalid_input(
                "model admission revision or registry rollback floor is invalid",
            ));
        }
        if !registry_signer.is_active() || !package_signer.is_active() {
            return Err(EngineError::authentication_failure(
                "model admission requires active registry and package trust",
            ));
        }
        if registry_signer.key_id() == package_signer.key_id()
            || registry_signer.public_key() == package_signer.public_key()
        {
            return Err(EngineError::authentication_failure(
                "model admission registry and package trust must be distinct",
            ));
        }
        let expected_lineage_id = expected_lineage_id.into();
        let expected_engine_profile_id = expected_engine_profile_id.into();
        let expected_registry_authority_id = expected_registry_authority_id.into();
        let expected_package_authority_id = expected_package_authority_id.into();
        for (field, value) in [
            ("lineage_id", expected_lineage_id.as_str()),
            ("engine_profile_id", expected_engine_profile_id.as_str()),
            ("registry_authority_id", expected_registry_authority_id.as_str()),
            ("package_authority_id", expected_package_authority_id.as_str()),
        ] {
            if value.is_empty() || value.len() > 128 {
                return Err(EngineError::invalid_field(
                    field,
                    "owner-pinned identity is outside hard bounds",
                ));
            }
        }
        if expected_registry_authority_id == expected_package_authority_id {
            return Err(EngineError::authentication_failure(
                "model admission registry and package authorities must be distinct",
            ));
        }
        expected_runtime_identity.validate().map_err(|_| {
            EngineError::invalid_input("owner-pinned admission runtime identity is invalid")
        })?;
        Ok(Self {
            expected_admission_id,
            expected_admission_digest,
            expected_revision,
            expected_registry_sequence,
            minimum_registry_sequence,
            expected_lineage_id,
            expected_runtime_identity,
            expected_engine_profile_id,
            expected_registry_authority_id,
            expected_package_authority_id,
            registry_signer,
            package_signer,
            owner_allows_development_execution,
        })
    }
}

/// Authenticated admission evidence that still grants no execution authority.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AuthenticatedModelExecutionAdmission {
    admission: ModelExecutionAdmission,
    descriptor: ExecutionPackageDescriptor,
    descriptor_digest: ModelPackageHash,
    package_catalog_digest: ModelPackageHash,
    owner_allows_development_execution: bool,
}

impl AuthenticatedModelExecutionAdmission {
    /// Returns the exact canonical admission.
    pub const fn admission(&self) -> &ModelExecutionAdmission {
        &self.admission
    }

    /// Returns the exact Models 2.0 descriptor authenticated by the admission.
    pub const fn descriptor(&self) -> &ExecutionPackageDescriptor {
        &self.descriptor
    }

    /// Returns SHA-384 of exact canonical descriptor bytes.
    pub const fn descriptor_digest(&self) -> ModelPackageHash {
        self.descriptor_digest
    }

    /// Returns SHA-384 of exact package-catalog bytes.
    pub const fn package_catalog_digest(&self) -> ModelPackageHash {
        self.package_catalog_digest
    }

    /// Checks the signed development state and the separate owner gate.
    ///
    /// Success proves only that the release-state evidence is permissive. It
    /// does not register a package or prove that an execution adapter exists.
    pub fn require_development_release_state(&self) -> Result<(), EngineError> {
        if !self.owner_allows_development_execution {
            return Err(EngineError::authentication_failure(
                "owner has not enabled development execution for this admission",
            ));
        }
        if self.admission.payload.admission_state != ModelAdmissionState::DevelopmentAuthorized
            || !self.admission.payload.development_execution_authorized
            || self.admission.payload.production_activation
        {
            return Err(EngineError::authentication_failure(
                "canonical admission does not authorize development execution",
            ));
        }
        Ok(())
    }

    /// Fails closed because no canonical Models 2.1 registration/execution
    /// adapter has yet been reviewed for this engine release.
    pub fn require_execution_adapter(&self) -> Result<(), EngineError> {
        Err(EngineError::not_implemented(
            "canonical Models 2.1 package registration and execution adapter",
        ))
    }
}

/// Authenticates one canonical Models 2.1 admission and its exact package
/// metadata against owner-pinned selection and signer trust.
///
/// Restrictive states are returned as authenticated evidence so suspension or
/// revocation can be consumed. Callers must separately require a permissive
/// release state, package registration, backend support, and owner activation.
pub fn authenticate_model_execution_admission(
    admission_bytes: &[u8],
    descriptor_bytes: &[u8],
    package_catalog: &[u8],
    now: &str,
    selection: &ModelExecutionAdmissionSelection,
) -> Result<AuthenticatedModelExecutionAdmission, EngineError> {
    if admission_bytes.is_empty() || admission_bytes.len() > MAX_MODEL_EXECUTION_ADMISSION_BYTES {
        return Err(EngineError::corrupt_model_package(
            "model admission length is outside hard bounds",
        ));
    }
    if package_catalog.is_empty() || package_catalog.len() > MAX_MODEL_ADMISSION_CATALOG_BYTES {
        return Err(EngineError::corrupt_model_package(
            "model admission package catalog length is outside hard bounds",
        ));
    }
    let value: serde_json::Value = serde_json::from_slice(admission_bytes)
        .map_err(|_| EngineError::corrupt_model_package("model admission is invalid JSON"))?;
    let canonical = canonical_json::to_vec(value.clone())
        .map_err(|_| EngineError::internal_invariant("model admission canonicalization failed"))?;
    if canonical != admission_bytes {
        return Err(EngineError::corrupt_model_package(
            "model admission is not exact canonical JSON",
        ));
    }
    let admission: ModelExecutionAdmission = serde_json::from_value(value).map_err(|_| {
        EngineError::corrupt_model_package("model admission closed schema validation failed")
    })?;
    admission.validate_at(now).map_err(|_| {
        EngineError::authentication_failure("model admission is invalid, stale, or expired")
    })?;
    validate_selection(&admission, selection)?;

    let descriptor = decode_models_v2_descriptor(descriptor_bytes)?;
    validate_model_execution_admission_descriptor(&admission, &descriptor).map_err(|_| {
        EngineError::authentication_failure(
            "model admission execution descriptor binding was substituted",
        )
    })?;
    validate_model_execution_admission_package_catalog(&admission, package_catalog).map_err(
        |_| {
            EngineError::authentication_failure(
                "model admission package catalog binding was substituted",
            )
        },
    )?;

    let transcript = model_execution_admission_signing_transcript(&admission.canonical_digest)
        .map_err(|_| {
            EngineError::corrupt_model_package("model admission signing transcript is invalid")
        })?;
    let signature = decode_ed25519_base64(&admission.signature_envelope.signature)?;
    let signer_id = parse_signer_key_id(&admission.signature_envelope.key_id)?;
    verify_canonical_contract_signature(
        &selection.registry_signer,
        signer_id,
        &transcript,
        &signature,
    )?;

    let descriptor_digest = ModelPackageHash::from_sha384_hex(
        &admission.payload.execution_package_descriptor_digest.digest,
    )?;
    let package_catalog_digest =
        ModelPackageHash::from_sha384_hex(&admission.payload.package_catalog_digest.digest)?;
    Ok(AuthenticatedModelExecutionAdmission {
        admission,
        descriptor,
        descriptor_digest,
        package_catalog_digest,
        owner_allows_development_execution: selection.owner_allows_development_execution,
    })
}

fn validate_selection(
    admission: &ModelExecutionAdmission,
    selection: &ModelExecutionAdmissionSelection,
) -> Result<(), EngineError> {
    let payload = &admission.payload;
    let digest = Sha256Digest::from_hex(&admission.canonical_digest).map_err(|_| {
        EngineError::corrupt_model_package("model admission canonical digest is invalid")
    })?;
    if payload.admission_id != selection.expected_admission_id
        || digest != selection.expected_admission_digest
        || payload.revision != selection.expected_revision
        || payload.registry_sequence != selection.expected_registry_sequence
        || payload.registry_sequence < selection.minimum_registry_sequence
        || payload.minimum_registry_sequence < selection.minimum_registry_sequence
        || payload.lineage_id != selection.expected_lineage_id
        || payload.runtime_identity != selection.expected_runtime_identity
        || payload.engine_compatibility.profile_id != selection.expected_engine_profile_id
        || payload.registry_signer_identity.authority_id != selection.expected_registry_authority_id
        || payload.package_signer_identity.authority_id != selection.expected_package_authority_id
    {
        return Err(EngineError::authentication_failure(
            "model admission differs from exact owner selection",
        ));
    }
    let registry_key_id = parse_signer_key_id(&payload.registry_signer_identity.key_id)?;
    let package_key_id = parse_signer_key_id(&payload.package_signer_identity.key_id)?;
    if registry_key_id != selection.registry_signer.key_id()
        || package_key_id != selection.package_signer.key_id()
        || !selection.package_signer.is_active()
    {
        return Err(EngineError::authentication_failure(
            "model admission signer trust differs from owner configuration",
        ));
    }
    require_supported_engine_release(&payload.engine_compatibility.minimum_engine_version)
}

fn require_supported_engine_release(required: &str) -> Result<(), EngineError> {
    let required = stable_semver_core(required).ok_or_else(|| {
        EngineError::not_implemented("pre-release model admission engine compatibility")
    })?;
    let current = stable_semver_core(env!("CARGO_PKG_VERSION")).ok_or_else(|| {
        EngineError::internal_invariant("engine package version is not stable semantic version")
    })?;
    if required > current {
        return Err(EngineError::not_implemented(
            "model admission requires a newer native engine release",
        ));
    }
    Ok(())
}

fn stable_semver_core(value: &str) -> Option<(u64, u64, u64)> {
    if value.contains(['-', '+']) {
        return None;
    }
    let mut components = value.split('.');
    let major = parse_semver_component(components.next()?)?;
    let minor = parse_semver_component(components.next()?)?;
    let patch = parse_semver_component(components.next()?)?;
    if components.next().is_some() {
        return None;
    }
    Some((major, minor, patch))
}

fn parse_semver_component(value: &str) -> Option<u64> {
    if value.is_empty()
        || !value.bytes().all(|byte| byte.is_ascii_digit())
        || value.len() > 1 && value.starts_with('0')
    {
        return None;
    }
    value.parse().ok()
}

fn parse_signer_key_id(value: &str) -> Result<SignerKeyId, EngineError> {
    if value.len() != SignerKeyId::BYTE_LEN * 2 {
        return Err(EngineError::corrupt_model_package(
            "model admission signer key identity is invalid",
        ));
    }
    let mut result = [0_u8; SignerKeyId::BYTE_LEN];
    for (output, pair) in result.iter_mut().zip(value.as_bytes().chunks_exact(2)) {
        let high = decode_hex_nibble(pair[0]).ok_or_else(|| {
            EngineError::corrupt_model_package("model admission signer key identity is invalid")
        })?;
        let low = decode_hex_nibble(pair[1]).ok_or_else(|| {
            EngineError::corrupt_model_package("model admission signer key identity is invalid")
        })?;
        *output = (high << 4) | low;
    }
    SignerKeyId::from_bytes(result)
}

const fn decode_hex_nibble(value: u8) -> Option<u8> {
    match value {
        b'0'..=b'9' => Some(value - b'0'),
        b'a'..=b'f' => Some(value - b'a' + 10),
        _ => None,
    }
}

fn decode_ed25519_base64(value: &str) -> Result<[u8; 64], EngineError> {
    let encoded = value.as_bytes();
    if encoded.len() != 88 || encoded[86] != b'=' || encoded[87] != b'=' {
        return Err(EngineError::corrupt_model_package(
            "model admission signature encoding is invalid",
        ));
    }
    let mut decoded = [0_u8; 64];
    for group in 0..21 {
        let input = group * 4;
        let output = group * 3;
        let first = base64_value(encoded[input])?;
        let second = base64_value(encoded[input + 1])?;
        let third = base64_value(encoded[input + 2])?;
        let fourth = base64_value(encoded[input + 3])?;
        decoded[output] = (first << 2) | (second >> 4);
        decoded[output + 1] = (second << 4) | (third >> 2);
        decoded[output + 2] = (third << 6) | fourth;
    }
    let first = base64_value(encoded[84])?;
    let second = base64_value(encoded[85])?;
    if second & 0x0f != 0 {
        return Err(EngineError::corrupt_model_package(
            "model admission signature encoding is noncanonical",
        ));
    }
    decoded[63] = (first << 2) | (second >> 4);
    Ok(decoded)
}

fn base64_value(value: u8) -> Result<u8, EngineError> {
    match value {
        b'A'..=b'Z' => Ok(value - b'A'),
        b'a'..=b'z' => Ok(value - b'a' + 26),
        b'0'..=b'9' => Ok(value - b'0' + 52),
        b'+' => Ok(62),
        b'/' => Ok(63),
        _ => {
            Err(EngineError::corrupt_model_package("model admission signature encoding is invalid"))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ed25519_dalek::{Signer, SigningKey};
    use sovereign_ai_contracts::identity::{ModelArtifactId, ModelId as LogicalModelId};
    use sovereign_ai_contracts::inference_runtime_v3::ArtifactDigestAlgorithmV3;
    use sovereign_ai_contracts::model_admission_v1::{
        MODEL_EXECUTION_ADMISSION_CONTRACT_VERSION, ModelAdmissionSignatureEnvelope,
        ModelAdmissionSignerIdentity, ModelAdmissionTrustState, ModelEngineCompatibility,
        ModelExecutionAdmissionContractVersion, ModelExecutionAdmissionKind,
        ModelExecutionAdmissionPayload, canonical_model_execution_admission_digest,
    };
    use sovereign_ai_contracts::models_v2::{
        ArchitectureExecutionBinding, ExecutionPackageDescriptorContractVersion,
        ExecutionPackageDescriptorKind, ExecutionPackageMember, ExecutionPackageMemberRole,
        PackageDigest, PackageStorageDtype, QuantizationBinding, TensorInventoryBinding,
        TokenizerArtifactBinding,
    };

    const CATALOG: &[u8] = b"synthetic canonical package catalog";

    fn trusted(seed: u8, key_byte: u8) -> (SigningKey, TrustedSigner) {
        let signing = SigningKey::from_bytes(&[seed; 32]);
        let trusted = TrustedSigner::active(
            SignerKeyId::from_bytes([key_byte; SignerKeyId::BYTE_LEN]).unwrap(),
            signing.verifying_key().to_bytes(),
        )
        .unwrap();
        (signing, trusted)
    }

    fn digest(byte: u8) -> PackageDigest {
        PackageDigest {
            algorithm: ArtifactDigestAlgorithmV3::Sha384,
            digest: crate::qwen_registry_v3::sha384_hex(ModelPackageHash::from_sha384_bytes(
                [byte; 48],
            )),
        }
    }

    fn descriptor(catalog_digest: ModelPackageHash) -> ExecutionPackageDescriptor {
        let members = vec![
            ExecutionPackageMember {
                member_id: "manifest".into(),
                ordinal: 0,
                role: ExecutionPackageMemberRole::PackageManifest,
                content: digest(1),
                size_bytes: 1,
            },
            ExecutionPackageMember {
                member_id: "tokenizer".into(),
                ordinal: 1,
                role: ExecutionPackageMemberRole::Tokenizer,
                content: digest(2),
                size_bytes: 2,
            },
            ExecutionPackageMember {
                member_id: "catalog".into(),
                ordinal: 2,
                role: ExecutionPackageMemberRole::IntegrityCatalog,
                content: PackageDigest {
                    algorithm: ArtifactDigestAlgorithmV3::Sha384,
                    digest: crate::qwen_registry_v3::sha384_hex(catalog_digest),
                },
                size_bytes: u64::try_from(CATALOG.len()).unwrap(),
            },
            ExecutionPackageMember {
                member_id: "tensor_shard.0000".into(),
                ordinal: 3,
                role: ExecutionPackageMemberRole::TensorShard,
                content: digest(3),
                size_bytes: 3,
            },
        ];
        ExecutionPackageDescriptor {
            contract_version: ExecutionPackageDescriptorContractVersion,
            kind: ExecutionPackageDescriptorKind::ExecutionPackageDescriptor,
            runtime_identity: RuntimeIdentityV3 {
                model_id: LogicalModelId::new("mdl_admission_fixture").unwrap(),
                provider_model_id: "provider.admission.fixture".into(),
                artifact_id: ModelArtifactId::new("art_admission_fixture").unwrap(),
                artifact_digest_algorithm: ArtifactDigestAlgorithmV3::Sha384,
                artifact_digest: crate::qwen_registry_v3::sha384_hex(catalog_digest),
                backend_id: "arm64".into(),
                backend_version: "version.0_2_0".into(),
                backend_instance_id: "backend.arm64.fixture".into(),
                engine_instance_id: "engine.fixture".into(),
            },
            package_identity_member_id: "catalog".into(),
            package_format_id: "personalai.directory_paimodel".into(),
            package_format_version: "1.1.0".into(),
            package_size_bytes: members.iter().map(|member| member.size_bytes).sum(),
            tokenizer: TokenizerArtifactBinding {
                tokenizer_id: "tokenizer.fixture".into(),
                member_id: "tokenizer".into(),
                format_id: "personalai.aiptok".into(),
                format_version: "2.0.0".into(),
                content: members[1].content.clone(),
                size_bytes: 2,
                model_vocabulary_size: 256,
                defined_token_count: 256,
                selectable_token_count: 256,
            },
            architecture: ArchitectureExecutionBinding {
                architecture_id: "fixture.decoder.v1".into(),
                position_semantics_id: "fixture.position.v1".into(),
                context_limit: 1024,
            },
            quantization: QuantizationBinding {
                profile_id: "fixture.f32.v1".into(),
                storage_dtype: PackageStorageDtype::F32,
            },
            tensor_inventory: TensorInventoryBinding {
                member_id: "manifest".into(),
                content: members[0].content.clone(),
                tensor_count: 1,
            },
            members,
            created_at: "2026-08-25T12:00:00Z".into(),
        }
    }

    fn base64_signature(bytes: &[u8; 64]) -> String {
        const ALPHABET: &[u8; 64] =
            b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        let mut encoded = String::with_capacity(88);
        for chunk in bytes[..63].chunks_exact(3) {
            encoded.push(char::from(ALPHABET[usize::from(chunk[0] >> 2)]));
            encoded.push(char::from(
                ALPHABET[usize::from(((chunk[0] & 0x03) << 4) | (chunk[1] >> 4))],
            ));
            encoded.push(char::from(
                ALPHABET[usize::from(((chunk[1] & 0x0f) << 2) | (chunk[2] >> 6))],
            ));
            encoded.push(char::from(ALPHABET[usize::from(chunk[2] & 0x3f)]));
        }
        encoded.push(char::from(ALPHABET[usize::from(bytes[63] >> 2)]));
        encoded.push(char::from(ALPHABET[usize::from((bytes[63] & 0x03) << 4)]));
        encoded.push('=');
        encoded.push('=');
        encoded
    }

    fn fixture() -> (Vec<u8>, Vec<u8>, ModelExecutionAdmissionSelection, SigningKey) {
        let catalog_digest = personalai_engine_crypto::hash_package(CATALOG);
        let descriptor = descriptor(catalog_digest);
        let descriptor_bytes = canonical_json::to_vec(serde_json::to_value(&descriptor).unwrap())
            .expect("canonical descriptor");
        let descriptor_digest = personalai_engine_crypto::hash_package(&descriptor_bytes);
        let (registry_key, registry_trust) = trusted(0x31, 0x41);
        let (_, package_trust) = trusted(0x32, 0x42);
        let mut admission = ModelExecutionAdmission {
            contract_version: ModelExecutionAdmissionContractVersion,
            kind: ModelExecutionAdmissionKind::ModelExecutionAdmission,
            payload: ModelExecutionAdmissionPayload {
                manifest_version: MODEL_EXECUTION_ADMISSION_CONTRACT_VERSION.into(),
                admission_id: ModelAdmissionId::new("mad_engine_fixture").unwrap(),
                revision: 1,
                previous_revision_digest: None,
                lineage_id: "registry.engine.fixture.v1".into(),
                registry_sequence: 7,
                minimum_registry_sequence: 5,
                runtime_identity: descriptor.runtime_identity.clone(),
                execution_package_descriptor_digest: PackageDigest {
                    algorithm: ArtifactDigestAlgorithmV3::Sha384,
                    digest: crate::qwen_registry_v3::sha384_hex(descriptor_digest),
                },
                package_catalog_digest: PackageDigest {
                    algorithm: ArtifactDigestAlgorithmV3::Sha384,
                    digest: crate::qwen_registry_v3::sha384_hex(catalog_digest),
                },
                engine_compatibility: ModelEngineCompatibility {
                    profile_id: "engine.package.fixture.v1".into(),
                    minimum_engine_version: "0.2.0".into(),
                },
                admission_state: ModelAdmissionState::DevelopmentAuthorized,
                trust_state: ModelAdmissionTrustState::DevelopmentFixture,
                development_execution_authorized: true,
                production_activation: false,
                registry_signer_identity: ModelAdmissionSignerIdentity {
                    authority_id: "registry.authority.fixture".into(),
                    key_id: registry_trust.key_id().to_string(),
                },
                package_signer_identity: ModelAdmissionSignerIdentity {
                    authority_id: "package.authority.fixture".into(),
                    key_id: package_trust.key_id().to_string(),
                },
                approved_at: "2026-08-25T12:00:00Z".into(),
                issued_at: "2026-08-25T12:01:00Z".into(),
                expires_at: "2026-09-25T12:01:00Z".into(),
                extensions: None,
                required_extensions: None,
            },
            canonical_digest: String::new(),
            signature_envelope: ModelAdmissionSignatureEnvelope {
                algorithm: "ed25519".into(),
                key_id: registry_trust.key_id().to_string(),
                signature: "A".repeat(86) + "==",
            },
        };
        admission.canonical_digest =
            canonical_model_execution_admission_digest(&admission.payload).unwrap();
        let transcript =
            model_execution_admission_signing_transcript(&admission.canonical_digest).unwrap();
        admission.signature_envelope.signature =
            base64_signature(&registry_key.sign(&transcript).to_bytes());
        let admission_bytes = canonical_json::to_vec(serde_json::to_value(&admission).unwrap())
            .expect("canonical admission");
        let selection = ModelExecutionAdmissionSelection::new(
            admission.payload.admission_id.clone(),
            Sha256Digest::from_hex(&admission.canonical_digest).unwrap(),
            1,
            7,
            5,
            admission.payload.lineage_id.clone(),
            admission.payload.runtime_identity.clone(),
            admission.payload.engine_compatibility.profile_id.clone(),
            admission.payload.registry_signer_identity.authority_id.clone(),
            admission.payload.package_signer_identity.authority_id.clone(),
            registry_trust,
            package_trust,
            true,
        )
        .unwrap();
        (admission_bytes, descriptor_bytes, selection, registry_key)
    }

    #[test]
    fn canonical_admission_authenticates_but_cannot_activate_an_adapter() {
        let (admission, descriptor, selection, _) = fixture();
        let authenticated = authenticate_model_execution_admission(
            &admission,
            &descriptor,
            CATALOG,
            "2026-08-26T12:00:00Z",
            &selection,
        )
        .expect("canonical admission evidence");
        authenticated.require_development_release_state().expect("signed state plus owner gate");
        assert_eq!(
            crate::qwen_registry_v3::sha384_hex(authenticated.descriptor_digest()),
            authenticated.admission().payload.execution_package_descriptor_digest.digest
        );
        assert_eq!(
            crate::qwen_registry_v3::sha384_hex(authenticated.package_catalog_digest()),
            authenticated.admission().payload.package_catalog_digest.digest
        );
        assert!(authenticated.require_execution_adapter().is_err());
    }

    #[test]
    fn substitution_signature_and_noncanonical_input_fail_closed() {
        let (admission, descriptor, selection, _) = fixture();
        let mut wrong_catalog = CATALOG.to_vec();
        wrong_catalog.push(0);
        assert!(
            authenticate_model_execution_admission(
                &admission,
                &descriptor,
                &wrong_catalog,
                "2026-08-26T12:00:00Z",
                &selection,
            )
            .is_err()
        );

        let mut bad_signature: serde_json::Value = serde_json::from_slice(&admission).unwrap();
        bad_signature["signature_envelope"]["signature"] =
            serde_json::Value::String("A".repeat(86) + "==");
        let bad_signature = canonical_json::to_vec(bad_signature).unwrap();
        assert!(
            authenticate_model_execution_admission(
                &bad_signature,
                &descriptor,
                CATALOG,
                "2026-08-26T12:00:00Z",
                &selection,
            )
            .is_err()
        );

        let mut noncanonical = admission.clone();
        noncanonical.insert(0, b' ');
        assert!(
            authenticate_model_execution_admission(
                &noncanonical,
                &descriptor,
                CATALOG,
                "2026-08-26T12:00:00Z",
                &selection,
            )
            .is_err()
        );
    }

    #[test]
    fn owner_denial_expiry_and_newer_engine_requirement_fail_closed() {
        let (admission, descriptor, mut selection, _) = fixture();
        selection.owner_allows_development_execution = false;
        let authenticated = authenticate_model_execution_admission(
            &admission,
            &descriptor,
            CATALOG,
            "2026-08-26T12:00:00Z",
            &selection,
        )
        .expect("restrictive owner state still authenticates");
        assert!(authenticated.require_development_release_state().is_err());
        assert!(
            authenticate_model_execution_admission(
                &admission,
                &descriptor,
                CATALOG,
                "2026-10-01T00:00:00Z",
                &selection,
            )
            .is_err()
        );

        let mut newer: ModelExecutionAdmission = serde_json::from_slice(&admission).unwrap();
        newer.payload.engine_compatibility.minimum_engine_version = "0.3.0".into();
        newer.canonical_digest =
            canonical_model_execution_admission_digest(&newer.payload).unwrap();
        let (registry_key, _) = trusted(0x31, 0x41);
        let transcript =
            model_execution_admission_signing_transcript(&newer.canonical_digest).unwrap();
        newer.signature_envelope.signature =
            base64_signature(&registry_key.sign(&transcript).to_bytes());
        let newer = canonical_json::to_vec(serde_json::to_value(newer).unwrap()).unwrap();
        assert!(
            authenticate_model_execution_admission(
                &newer,
                &descriptor,
                CATALOG,
                "2026-08-26T12:00:00Z",
                &selection,
            )
            .is_err()
        );
    }
}
