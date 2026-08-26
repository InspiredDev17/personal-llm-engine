//! Experimental, provider-neutral signed model execution admission.
//!
//! The document binds Runtime V3 and Models 2.0 identities to registry
//! lineage and explicit release state. It does not verify external signer
//! trust, qualify a model, activate a provider, or execute a package.

use crate::common::{
    check_len, compare_timestamp_values, validate_extensions, validate_semver, validate_sha256,
    validate_timestamp, Extensions, RequiredExtensions,
};
use crate::error::{ContractError, Result, Validate};
use crate::inference_runtime_v3::{ArtifactDigestAlgorithmV3, RuntimeIdentityV3};
use crate::models_v2::{ExecutionPackageDescriptor, PackageDigest};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use sha2::{Digest, Sha256, Sha384};
use std::cmp::Ordering;

/// Experimental Models 2.1 signed-admission version.
pub const MODEL_EXECUTION_ADMISSION_CONTRACT_VERSION: &str = "2.1.0";
const DIGEST_NAMESPACE: &[u8] = b"SOVEREIGN-MODEL-EXECUTION-ADMISSION-V1\0";
const SIGNATURE_NAMESPACE: &[u8] = b"SOVEREIGN-MODEL-EXECUTION-ADMISSION-SIGNATURE-V1\0";

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ModelExecutionAdmissionContractVersion;

impl Serialize for ModelExecutionAdmissionContractVersion {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(MODEL_EXECUTION_ADMISSION_CONTRACT_VERSION)
    }
}

impl<'de> Deserialize<'de> for ModelExecutionAdmissionContractVersion {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        if value == MODEL_EXECUTION_ADMISSION_CONTRACT_VERSION {
            Ok(Self)
        } else {
            Err(serde::de::Error::custom(format!(
                "unsupported contract_version {value:?}; expected {MODEL_EXECUTION_ADMISSION_CONTRACT_VERSION}"
            )))
        }
    }
}

impl Validate for ModelExecutionAdmissionContractVersion {
    fn validate(&self) -> Result<()> {
        Ok(())
    }
}

/// Identifies one immutable admission lineage.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(transparent)]
pub struct ModelAdmissionId(String);

impl ModelAdmissionId {
    /// Constructs one bounded model-admission identity.
    ///
    /// # Errors
    ///
    /// Returns an error when the identifier is outside the published grammar.
    pub fn new(value: impl Into<String>) -> Result<Self> {
        let value = Self(value.into());
        value.validate()?;
        Ok(value)
    }

    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl Validate for ModelAdmissionId {
    fn validate(&self) -> Result<()> {
        check_len("admission_id", &self.0, 6, 68)?;
        if !self.0.starts_with("mad_")
            || self.0.len() < 6
            || !self.0[4..].bytes().all(|byte| {
                byte.is_ascii_lowercase() || byte.is_ascii_digit() || matches!(byte, b'_' | b'-')
            })
            || !self.0.as_bytes()[4].is_ascii_lowercase() && !self.0.as_bytes()[4].is_ascii_digit()
        {
            return Err(ContractError::field(
                "admission_id",
                "does not match the mad_ identifier grammar",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelAdmissionState {
    Unactivated,
    DevelopmentAuthorized,
    ProductionActivated,
    Suspended,
    Revoked,
}

impl Validate for ModelAdmissionState {
    fn validate(&self) -> Result<()> {
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelAdmissionTrustState {
    DevelopmentFixture,
    Verified,
}

impl Validate for ModelAdmissionTrustState {
    fn validate(&self) -> Result<()> {
        Ok(())
    }
}

/// Public signer identity only. Public keys and active trust remain external.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ModelAdmissionSignerIdentity {
    pub authority_id: String,
    pub key_id: String,
}

impl ModelAdmissionSignerIdentity {
    fn validate_for(&self, field: &'static str) -> Result<()> {
        validate_admission_identity(field, &self.authority_id)?;
        validate_lower_hex("signer_identity.key_id", &self.key_id, 32)
    }
}

impl Validate for ModelAdmissionSignerIdentity {
    fn validate(&self) -> Result<()> {
        self.validate_for("signer_identity.authority_id")
    }
}

/// Reviewed engine/package compatibility identity and minimum release.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ModelEngineCompatibility {
    pub profile_id: String,
    pub minimum_engine_version: String,
}

impl Validate for ModelEngineCompatibility {
    fn validate(&self) -> Result<()> {
        validate_admission_identity("engine_compatibility.profile_id", &self.profile_id)?;
        validate_semver(
            "engine_compatibility.minimum_engine_version",
            &self.minimum_engine_version,
        )
    }
}

/// Exact content-free binding and release state signed by registry authority.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ModelExecutionAdmissionPayload {
    pub manifest_version: String,
    pub admission_id: ModelAdmissionId,
    pub revision: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub previous_revision_digest: Option<String>,
    pub lineage_id: String,
    pub registry_sequence: u64,
    pub minimum_registry_sequence: u64,
    pub runtime_identity: RuntimeIdentityV3,
    pub execution_package_descriptor_digest: PackageDigest,
    pub package_catalog_digest: PackageDigest,
    pub engine_compatibility: ModelEngineCompatibility,
    pub admission_state: ModelAdmissionState,
    pub trust_state: ModelAdmissionTrustState,
    pub development_execution_authorized: bool,
    pub production_activation: bool,
    pub registry_signer_identity: ModelAdmissionSignerIdentity,
    pub package_signer_identity: ModelAdmissionSignerIdentity,
    pub approved_at: String,
    pub issued_at: String,
    pub expires_at: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extensions: Option<Extensions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub required_extensions: Option<RequiredExtensions>,
}

impl Validate for ModelExecutionAdmissionPayload {
    #[allow(clippy::too_many_lines)]
    fn validate(&self) -> Result<()> {
        if self.manifest_version != MODEL_EXECUTION_ADMISSION_CONTRACT_VERSION {
            return Err(ContractError::field("manifest_version", "must equal 2.1.0"));
        }
        self.admission_id.validate()?;
        if self.revision == 0 {
            return Err(ContractError::field("revision", "must be nonzero"));
        }
        validate_admission_identity("lineage_id", &self.lineage_id)?;
        if self.registry_sequence == 0
            || self.minimum_registry_sequence == 0
            || self.registry_sequence < self.minimum_registry_sequence
        {
            return Err(ContractError::field(
                "registry_sequence",
                "must be nonzero and meet the retained minimum",
            ));
        }
        self.runtime_identity.validate()?;
        self.execution_package_descriptor_digest.validate()?;
        self.package_catalog_digest.validate()?;
        if self.execution_package_descriptor_digest.algorithm != ArtifactDigestAlgorithmV3::Sha384
            || self.package_catalog_digest.algorithm != ArtifactDigestAlgorithmV3::Sha384
        {
            return Err(ContractError::field(
                "model_admission_digest",
                "descriptor and package catalog must use sha384",
            ));
        }
        if self.runtime_identity.artifact_digest_algorithm != self.package_catalog_digest.algorithm
            || self.runtime_identity.artifact_digest != self.package_catalog_digest.digest
        {
            return Err(ContractError::field(
                "package_catalog_digest",
                "must exactly match runtime_identity artifact digest",
            ));
        }
        self.engine_compatibility.validate()?;
        self.registry_signer_identity
            .validate_for("registry_signer_identity.authority_id")?;
        self.package_signer_identity
            .validate_for("package_signer_identity.authority_id")?;
        if self.registry_signer_identity == self.package_signer_identity
            || self.registry_signer_identity.key_id == self.package_signer_identity.key_id
        {
            return Err(ContractError::field(
                "signer_identity",
                "registry and package signer roles must be distinct",
            ));
        }
        validate_activation(self)?;
        validate_timestamp("approved_at", &self.approved_at)?;
        validate_timestamp("issued_at", &self.issued_at)?;
        validate_timestamp("expires_at", &self.expires_at)?;
        if compare_timestamp_values(&self.approved_at, &self.issued_at)? == Ordering::Greater {
            return Err(ContractError::field(
                "approved_at",
                "must not be later than issued_at",
            ));
        }
        if compare_timestamp_values(&self.issued_at, &self.expires_at)? != Ordering::Less {
            return Err(ContractError::field(
                "expires_at",
                "must be later than issued_at",
            ));
        }
        match (self.revision, self.previous_revision_digest.as_deref()) {
            (1, None) => {}
            (1, Some(_)) => {
                return Err(ContractError::field(
                    "previous_revision_digest",
                    "must be absent for revision one",
                ));
            }
            (_, Some(digest)) => validate_sha256("previous_revision_digest", digest)?,
            (_, None) => {
                return Err(ContractError::field(
                    "previous_revision_digest",
                    "is required after revision one",
                ));
            }
        }
        validate_extensions(self.extensions.as_ref(), self.required_extensions.as_ref())
    }
}

fn validate_activation(payload: &ModelExecutionAdmissionPayload) -> Result<()> {
    let (development, production) = match payload.admission_state {
        ModelAdmissionState::DevelopmentAuthorized => (true, false),
        ModelAdmissionState::ProductionActivated => (false, true),
        ModelAdmissionState::Unactivated
        | ModelAdmissionState::Suspended
        | ModelAdmissionState::Revoked => (false, false),
    };
    if payload.development_execution_authorized != development
        || payload.production_activation != production
    {
        return Err(ContractError::field(
            "admission_state",
            "does not match explicit development and production release flags",
        ));
    }
    if payload.trust_state == ModelAdmissionTrustState::DevelopmentFixture
        && payload.production_activation
    {
        return Err(ContractError::field(
            "production_activation",
            "development_fixture trust cannot activate production",
        ));
    }
    Ok(())
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ModelAdmissionSignatureEnvelope {
    pub algorithm: String,
    pub key_id: String,
    pub signature: String,
}

impl Validate for ModelAdmissionSignatureEnvelope {
    fn validate(&self) -> Result<()> {
        if self.algorithm != "ed25519" {
            return Err(ContractError::field(
                "signature_envelope.algorithm",
                "must equal ed25519",
            ));
        }
        validate_lower_hex("signature_envelope.key_id", &self.key_id, 32)?;
        validate_ed25519_base64(&self.signature)
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum ModelExecutionAdmissionKind {
    #[serde(rename = "model.execution_admission")]
    ModelExecutionAdmission,
}

/// Signed model admission. Signer trust remains an external consumer input.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ModelExecutionAdmission {
    pub contract_version: ModelExecutionAdmissionContractVersion,
    pub kind: ModelExecutionAdmissionKind,
    pub payload: ModelExecutionAdmissionPayload,
    pub canonical_digest: String,
    pub signature_envelope: ModelAdmissionSignatureEnvelope,
}

impl Validate for ModelExecutionAdmission {
    fn validate(&self) -> Result<()> {
        self.payload.validate()?;
        if self.canonical_digest != canonical_model_execution_admission_digest(&self.payload)? {
            return Err(ContractError::field(
                "canonical_digest",
                "does not match canonical payload",
            ));
        }
        self.signature_envelope.validate()?;
        if self.signature_envelope.key_id != self.payload.registry_signer_identity.key_id {
            return Err(ContractError::field(
                "signature_envelope.key_id",
                "must equal the registry signer key identity",
            ));
        }
        Ok(())
    }
}

impl ModelExecutionAdmission {
    /// Validates structure and proves the signed document is issued and fresh
    /// at the supplied timestamp. Signature trust remains external.
    ///
    /// # Errors
    ///
    /// Returns an error for an invalid, not-yet-issued, or expired document.
    pub fn validate_at(&self, now: &str) -> Result<()> {
        self.validate()?;
        validate_timestamp("now", now)?;
        if compare_timestamp_values(now, &self.payload.issued_at)? == Ordering::Less
            || compare_timestamp_values(now, &self.payload.expires_at)? != Ordering::Less
        {
            return Err(ContractError::field(
                "now",
                "must be at or after issued_at and before expires_at",
            ));
        }
        Ok(())
    }
}

/// Computes the namespaced SHA-256 digest of the canonical payload.
///
/// # Errors
///
/// Returns an error for an invalid payload or serialization failure.
pub fn canonical_model_execution_admission_digest(
    payload: &ModelExecutionAdmissionPayload,
) -> Result<String> {
    payload.validate()?;
    let canonical = serde_json::to_vec(&crate::common::sort_json_value(serde_json::to_value(
        payload,
    )?))?;
    let mut hasher = Sha256::new();
    hasher.update(DIGEST_NAMESPACE);
    hasher.update(canonical);
    Ok(format!("{:x}", hasher.finalize()))
}

/// Produces the fixed Ed25519 signing transcript.
///
/// # Errors
///
/// Returns an error unless the supplied digest is canonical lowercase SHA-256.
pub fn model_execution_admission_signing_transcript(digest: &str) -> Result<Vec<u8>> {
    validate_sha256("canonical_digest", digest)?;
    let mut transcript = Vec::with_capacity(SIGNATURE_NAMESPACE.len() + 32);
    transcript.extend_from_slice(SIGNATURE_NAMESPACE);
    transcript.extend(decode_hex(digest)?);
    Ok(transcript)
}

/// Verifies the exact canonical Models 2.0 descriptor and Runtime V3 binding.
///
/// # Errors
///
/// Returns an error for an invalid or substituted descriptor.
pub fn validate_model_execution_admission_descriptor(
    admission: &ModelExecutionAdmission,
    descriptor: &ExecutionPackageDescriptor,
) -> Result<()> {
    admission.validate()?;
    descriptor.validate()?;
    if descriptor.runtime_identity != admission.payload.runtime_identity {
        return Err(ContractError::field(
            "runtime_identity",
            "descriptor substitution",
        ));
    }
    let canonical = serde_json::to_vec(&crate::common::sort_json_value(serde_json::to_value(
        descriptor,
    )?))?;
    let digest = format!("{:x}", Sha384::digest(canonical));
    if digest != admission.payload.execution_package_descriptor_digest.digest {
        return Err(ContractError::field(
            "execution_package_descriptor_digest",
            "descriptor substitution",
        ));
    }
    Ok(())
}

/// Verifies exact package-catalog bytes without interpreting package internals.
///
/// # Errors
///
/// Returns an error for an invalid admission or substituted catalog.
pub fn validate_model_execution_admission_package_catalog(
    admission: &ModelExecutionAdmission,
    catalog: &[u8],
) -> Result<()> {
    admission.validate()?;
    let digest = format!("{:x}", Sha384::digest(catalog));
    if digest != admission.payload.package_catalog_digest.digest {
        return Err(ContractError::field(
            "package_catalog_digest",
            "catalog substitution",
        ));
    }
    Ok(())
}

/// Validates one exact adjacent admission revision.
///
/// # Errors
///
/// Returns an error for replay, rollback, substitution, trust downgrade,
/// expiry extension, or an unsafe state transition.
#[allow(clippy::too_many_lines)]
pub fn validate_model_execution_admission_revision(
    previous: &ModelExecutionAdmission,
    current: &ModelExecutionAdmission,
) -> Result<()> {
    previous.validate()?;
    let next_revision =
        previous
            .payload
            .revision
            .checked_add(1)
            .ok_or(ContractError::RevisionTerminal {
                revision: previous.payload.revision,
            })?;
    current.validate()?;
    if current.payload.admission_id != previous.payload.admission_id
        || current.payload.lineage_id != previous.payload.lineage_id
        || current.payload.runtime_identity != previous.payload.runtime_identity
        || current.payload.execution_package_descriptor_digest
            != previous.payload.execution_package_descriptor_digest
        || current.payload.package_catalog_digest != previous.payload.package_catalog_digest
        || current.payload.engine_compatibility != previous.payload.engine_compatibility
        || current.payload.registry_signer_identity != previous.payload.registry_signer_identity
        || current.payload.package_signer_identity != previous.payload.package_signer_identity
    {
        return Err(ContractError::field(
            "admission_revision",
            "identity, package, compatibility, lineage, or signer substitution",
        ));
    }
    if previous.payload.admission_state == ModelAdmissionState::Revoked {
        return Err(ContractError::field(
            "admission_revision",
            "revocation is terminal",
        ));
    }
    if current.payload.revision != next_revision
        || current.payload.registry_sequence <= previous.payload.registry_sequence
        || current.payload.minimum_registry_sequence < previous.payload.minimum_registry_sequence
        || current.payload.previous_revision_digest.as_deref() != Some(&previous.canonical_digest)
    {
        return Err(ContractError::field(
            "admission_revision",
            "replayed, stale, rolled-back, or broken lineage",
        ));
    }
    if compare_timestamp_values(&previous.payload.issued_at, &current.payload.issued_at)?
        != Ordering::Less
    {
        return Err(ContractError::field(
            "issued_at",
            "must advance across revisions",
        ));
    }
    if compare_timestamp_values(&previous.payload.expires_at, &current.payload.expires_at)?
        == Ordering::Less
    {
        return Err(ContractError::field(
            "expires_at",
            "cannot be extended within an existing admission lineage",
        ));
    }
    if previous.payload.trust_state == ModelAdmissionTrustState::Verified
        && current.payload.trust_state == ModelAdmissionTrustState::DevelopmentFixture
    {
        return Err(ContractError::field(
            "trust_state",
            "verified trust cannot roll back to development_fixture",
        ));
    }
    let state_changed = current.payload.admission_state != previous.payload.admission_state;
    let trust_changed = current.payload.trust_state != previous.payload.trust_state;
    let approved_order =
        compare_timestamp_values(&previous.payload.approved_at, &current.payload.approved_at)?;
    if state_changed || trust_changed {
        if approved_order != Ordering::Less
            || !allowed_transition(
                previous.payload.admission_state,
                current.payload.admission_state,
            )
        {
            return Err(ContractError::field(
                "admission_state",
                "unsupported or non-monotonic approved transition",
            ));
        }
    } else if approved_order != Ordering::Equal {
        return Err(ContractError::field(
            "approved_at",
            "must remain unchanged without a state or trust transition",
        ));
    }
    Ok(())
}

fn allowed_transition(previous: ModelAdmissionState, current: ModelAdmissionState) -> bool {
    matches!(
        (previous, current),
        (
            ModelAdmissionState::Unactivated,
            ModelAdmissionState::DevelopmentAuthorized
                | ModelAdmissionState::ProductionActivated
                | ModelAdmissionState::Suspended
                | ModelAdmissionState::Revoked
        ) | (
            ModelAdmissionState::DevelopmentAuthorized,
            ModelAdmissionState::ProductionActivated
                | ModelAdmissionState::Suspended
                | ModelAdmissionState::Revoked
        ) | (
            ModelAdmissionState::ProductionActivated,
            ModelAdmissionState::Suspended | ModelAdmissionState::Revoked
        ) | (ModelAdmissionState::Suspended, ModelAdmissionState::Revoked)
    )
}

fn validate_admission_identity(field: &'static str, value: &str) -> Result<()> {
    check_len(field, value, 2, 128)?;
    let mut bytes = value.bytes();
    if !bytes.next().is_some_and(|byte| byte.is_ascii_lowercase())
        || !bytes.all(|byte| {
            byte.is_ascii_lowercase()
                || byte.is_ascii_digit()
                || matches!(byte, b'.' | b'_' | b':' | b'-')
        })
        || value.ends_with(['.', '_', ':', '-'])
        || value
            .as_bytes()
            .windows(2)
            .any(|pair| matches!(pair, [b'.' | b'_' | b':' | b'-', b'.' | b'_' | b':' | b'-']))
    {
        return Err(ContractError::field(field, "invalid bounded identity"));
    }
    Ok(())
}

fn validate_lower_hex(field: &'static str, value: &str, length: usize) -> Result<()> {
    if value.len() != length
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || matches!(byte, b'a'..=b'f'))
        || value.bytes().all(|byte| byte == b'0')
    {
        return Err(ContractError::field(
            field,
            "must be nonzero lowercase hexadecimal of the exact length",
        ));
    }
    Ok(())
}

fn validate_ed25519_base64(value: &str) -> Result<()> {
    if value.len() != 88
        || !value.ends_with("==")
        || !value[..86]
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'+' | b'/'))
    {
        return Err(ContractError::field(
            "signature_envelope.signature",
            "must be canonical Ed25519 base64",
        ));
    }
    let final_value = base64_value(value.as_bytes()[85])?;
    if final_value & 0x0f != 0 {
        return Err(ContractError::field(
            "signature_envelope.signature",
            "must be canonical Ed25519 base64",
        ));
    }
    Ok(())
}

fn base64_value(value: u8) -> Result<u8> {
    match value {
        b'A'..=b'Z' => Ok(value - b'A'),
        b'a'..=b'z' => Ok(value - b'a' + 26),
        b'0'..=b'9' => Ok(value - b'0' + 52),
        b'+' => Ok(62),
        b'/' => Ok(63),
        _ => Err(ContractError::field(
            "signature_envelope.signature",
            "must be canonical Ed25519 base64",
        )),
    }
}

fn decode_hex(value: &str) -> Result<Vec<u8>> {
    value
        .as_bytes()
        .chunks_exact(2)
        .map(|pair| {
            let high = (pair[0] as char)
                .to_digit(16)
                .ok_or_else(|| ContractError::field("canonical_digest", "invalid hex"))?;
            let low = (pair[1] as char)
                .to_digit(16)
                .ok_or_else(|| ContractError::field("canonical_digest", "invalid hex"))?;
            u8::try_from((high << 4) | low)
                .map_err(|_| ContractError::field("canonical_digest", "invalid hex pair"))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const FIXTURE: &str =
        include_str!("../../fixtures/v1/valid/model-execution-admission-v2.1.json");
    const DESCRIPTOR_FIXTURE: &str =
        include_str!("../../fixtures/v1/valid/model-execution-package-descriptor-v2.json");
    const FIXTURE_DIGEST: &str = "70f7428e616544e2348b0567b6741f1aaffb761b7f046bcc205c27563fd1686f";

    fn fixture() -> ModelExecutionAdmission {
        let value: ModelExecutionAdmission =
            serde_json::from_str(FIXTURE).expect("decode model admission fixture");
        value.validate().expect("fixture must validate");
        value
    }

    fn finalize(value: &mut ModelExecutionAdmission) {
        value.canonical_digest = canonical_model_execution_admission_digest(&value.payload)
            .expect("canonical admission digest");
    }

    fn next(previous: &ModelExecutionAdmission) -> ModelExecutionAdmission {
        let mut current = previous.clone();
        current.payload.revision += 1;
        current.payload.registry_sequence += 1;
        current.payload.previous_revision_digest = Some(previous.canonical_digest.clone());
        current.payload.issued_at = "2026-08-25T12:02:00Z".into();
        finalize(&mut current);
        current
    }

    #[test]
    fn shared_fixture_digest_and_transcript_match_go() {
        let value = fixture();
        assert_eq!(value.canonical_digest, FIXTURE_DIGEST);
        let transcript = model_execution_admission_signing_transcript(&value.canonical_digest)
            .expect("signing transcript");
        let mut expected = SIGNATURE_NAMESPACE.to_vec();
        expected.extend(decode_hex(FIXTURE_DIGEST).expect("fixture digest hex"));
        assert_eq!(transcript, expected);
        value
            .validate_at("2026-08-26T12:00:00Z")
            .expect("fresh fixture");
        assert!(value.validate_at(&value.payload.expires_at).is_err());
    }

    #[test]
    fn development_fixture_cannot_activate_production_or_reuse_signer() {
        let mut value = fixture();
        value.payload.admission_state = ModelAdmissionState::ProductionActivated;
        value.payload.development_execution_authorized = false;
        value.payload.production_activation = true;
        assert!(value.validate().is_err());

        let mut value = fixture();
        value.payload.package_signer_identity = value.payload.registry_signer_identity.clone();
        assert!(value.validate().is_err());

        let mut value = fixture();
        value.signature_envelope.key_id = value.payload.package_signer_identity.key_id.clone();
        assert!(value.validate().is_err());
    }

    #[test]
    fn revision_rejects_replay_rollback_and_substitution() {
        let previous = fixture();
        let current = next(&previous);
        validate_model_execution_admission_revision(&previous, &current)
            .expect("valid adjacent revision");
        assert!(validate_model_execution_admission_revision(&previous, &previous).is_err());

        let mut substituted = next(&previous);
        substituted.payload.runtime_identity.provider_model_id = "provider.substituted".into();
        finalize(&mut substituted);
        assert!(validate_model_execution_admission_revision(&previous, &substituted).is_err());

        let mut rollback = next(&previous);
        rollback.payload.minimum_registry_sequence -= 1;
        finalize(&mut rollback);
        assert!(validate_model_execution_admission_revision(&previous, &rollback).is_err());

        let mut broken = next(&previous);
        broken.payload.previous_revision_digest = Some("a".repeat(64));
        finalize(&mut broken);
        assert!(validate_model_execution_admission_revision(&previous, &broken).is_err());
    }

    #[test]
    fn verified_production_and_terminal_revocation_are_bounded() {
        let previous = fixture();
        let mut current = next(&previous);
        current.payload.trust_state = ModelAdmissionTrustState::Verified;
        current.payload.admission_state = ModelAdmissionState::ProductionActivated;
        current.payload.development_execution_authorized = false;
        current.payload.production_activation = true;
        current.payload.approved_at = "2026-08-25T12:01:30Z".into();
        finalize(&mut current);
        validate_model_execution_admission_revision(&previous, &current)
            .expect("verified production transition");

        let mut revoked = current.clone();
        revoked.payload.revision += 1;
        revoked.payload.registry_sequence += 1;
        revoked.payload.previous_revision_digest = Some(current.canonical_digest.clone());
        revoked.payload.admission_state = ModelAdmissionState::Revoked;
        revoked.payload.production_activation = false;
        revoked.payload.approved_at = "2026-08-25T12:02:30Z".into();
        revoked.payload.issued_at = "2026-08-25T12:03:00Z".into();
        finalize(&mut revoked);
        validate_model_execution_admission_revision(&current, &revoked)
            .expect("restrictive revocation");

        let mut after = revoked.clone();
        after.payload.revision += 1;
        after.payload.registry_sequence += 1;
        after.payload.previous_revision_digest = Some(revoked.canonical_digest.clone());
        after.payload.issued_at = "2026-08-25T12:04:00Z".into();
        finalize(&mut after);
        assert!(validate_model_execution_admission_revision(&revoked, &after).is_err());
    }

    #[test]
    fn package_catalog_digest_is_exact() {
        let catalog = b"synthetic package catalog fixture";
        let mut value = fixture();
        value.payload.package_catalog_digest.digest = format!("{:x}", Sha384::digest(catalog));
        value.payload.runtime_identity.artifact_digest =
            value.payload.package_catalog_digest.digest.clone();
        finalize(&mut value);
        validate_model_execution_admission_package_catalog(&value, catalog).expect("exact catalog");
        assert!(validate_model_execution_admission_package_catalog(&value, b"other").is_err());
    }

    #[test]
    fn execution_package_descriptor_binding_is_exact() {
        let descriptor: ExecutionPackageDescriptor =
            serde_json::from_str(DESCRIPTOR_FIXTURE).expect("decode execution package descriptor");
        descriptor
            .validate()
            .expect("valid execution package descriptor");
        let mut value = fixture();
        value.payload.runtime_identity = descriptor.runtime_identity.clone();
        value.payload.package_catalog_digest = PackageDigest {
            algorithm: descriptor
                .runtime_identity
                .artifact_digest_algorithm
                .clone(),
            digest: descriptor.runtime_identity.artifact_digest.clone(),
        };
        let canonical = serde_json::to_vec(&crate::common::sort_json_value(
            serde_json::to_value(&descriptor).expect("serialize descriptor"),
        ))
        .expect("canonical descriptor");
        value.payload.execution_package_descriptor_digest = PackageDigest {
            algorithm: ArtifactDigestAlgorithmV3::Sha384,
            digest: format!("{:x}", Sha384::digest(canonical)),
        };
        finalize(&mut value);
        validate_model_execution_admission_descriptor(&value, &descriptor)
            .expect("exact descriptor binding");

        let mut substituted = descriptor;
        substituted.runtime_identity.provider_model_id = "provider.substituted".into();
        assert!(validate_model_execution_admission_descriptor(&value, &substituted).is_err());
    }
}
