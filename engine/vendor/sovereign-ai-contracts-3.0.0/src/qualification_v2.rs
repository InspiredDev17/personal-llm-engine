use crate::cognitive::{ResourceRequirementEnvelope, COGNITIVE_CONTRACT_VERSION};
use crate::common::{
    compare_timestamp_values, validate_sha256, validate_timestamp, validate_timestamp_order,
    RedactedReference,
};
use crate::error::{ContractError, Result, Validate};
use crate::identity::{EvaluationRevisionId, ModelArtifactId, ModelId, QualificationEvidenceId};
use crate::inference_runtime_v3::RuntimeIdentityV3;
use crate::qualification::{
    ConfidenceInterval, QualificationAntiRollback, QualificationManifestId,
    QualificationProducerIdentity, QualificationSignatureEnvelope,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::cmp::Ordering;
use std::collections::BTreeSet;

pub const MODEL_CAPABILITY_MANIFEST_CONTRACT_VERSION: &str = "2.0.0";
const DIGEST_NAMESPACE: &[u8] = b"SOVEREIGN-MODEL-CAPABILITY-MANIFEST-V2\0";
const SIGNATURE_NAMESPACE: &[u8] = b"SOVEREIGN-MODEL-CAPABILITY-SIGNATURE-V2\0";

fn valid_path(value: &str) -> bool {
    if value.is_empty() || value.len() > 160 || !value.is_ascii() {
        return false;
    }
    let parts: Vec<&str> = value.split('.').collect();
    !parts.is_empty()
        && parts.len() <= 12
        && parts.iter().all(|p| {
            !p.is_empty()
                && p.len() <= 48
                && p.as_bytes()[0].is_ascii_lowercase()
                && p.bytes()
                    .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_')
        })
}
fn extensible(field: &'static str, value: &str, core: &[&str]) -> Result<()> {
    if core.contains(&value) {
        return Ok(());
    }
    if valid_path(value) && value.starts_with("x.") && value.split('.').count() >= 3 {
        Ok(())
    } else {
        Err(ContractError::field(
            field,
            "must be core or x.<owner> namespaced",
        ))
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CapabilityQualificationV2 {
    pub capability_id: String,
    pub score_basis_points: u16,
    pub confidence: ConfidenceInterval,
    pub evaluation_revision: EvaluationRevisionId,
    pub evidence_ids: Vec<QualificationEvidenceId>,
}
impl Validate for CapabilityQualificationV2 {
    fn validate(&self) -> Result<()> {
        if !valid_path(&self.capability_id)
            || self.score_basis_points > 10_000
            || self.confidence.lower_basis_points > self.score_basis_points
            || self.confidence.upper_basis_points < self.score_basis_points
            || self.confidence.upper_basis_points > 10_000
            || !(5_000..=10_000).contains(&self.confidence.confidence_basis_points)
        {
            return Err(ContractError::field(
                "capability",
                "invalid path, score, or confidence",
            ));
        }
        self.evaluation_revision.validate()?;
        if self.evidence_ids.is_empty() || self.evidence_ids.len() > 32 {
            return Err(ContractError::field(
                "evidence_ids",
                "must contain 1 through 32",
            ));
        }
        let mut set = BTreeSet::new();
        for id in &self.evidence_ids {
            id.validate()?;
            if !set.insert(id) {
                return Err(ContractError::field("evidence_ids", "duplicate"));
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct QualificationEvidenceV2 {
    pub evidence_id: QualificationEvidenceId,
    pub evaluation_revision: EvaluationRevisionId,
    pub benchmark_suite_reference: RedactedReference,
    pub evidence_reference: RedactedReference,
    pub artifact_id: ModelArtifactId,
    pub observed_at: String,
}
impl QualificationEvidenceV2 {
    fn validate_for(&self, artifact: &ModelArtifactId) -> Result<()> {
        self.evidence_id.validate()?;
        self.evaluation_revision.validate()?;
        self.benchmark_suite_reference.validate()?;
        self.evidence_reference.validate()?;
        if &self.artifact_id != artifact {
            return Err(ContractError::field(
                "evidence.artifact_id",
                "must match manifest artifact",
            ));
        }
        validate_timestamp("observed_at", &self.observed_at)
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelQualificationState {
    Evaluated,
    Qualified,
    Rejected,
    Expired,
    Revoked,
    Experimental,
    Unactivated,
}
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelLocality {
    LocalOnly,
    LocallyAvailable,
    RemoteProvider,
    Hybrid,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ModelCapabilityPayload {
    pub manifest_version: String,
    pub manifest_id: QualificationManifestId,
    pub revision: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub previous_revision_digest: Option<String>,
    pub issued_at: String,
    pub expires_at: String,
    pub producer_identity: QualificationProducerIdentity,
    pub qualification_state: ModelQualificationState,
    pub logical_model_id: ModelId,
    pub runtime_identity: RuntimeIdentityV3,
    pub model_class: String,
    pub locality: ModelLocality,
    pub modalities: Vec<String>,
    pub taxonomy_revision: String,
    pub capabilities: Vec<CapabilityQualificationV2>,
    pub evaluation_revision: EvaluationRevisionId,
    pub qualification_timestamp: String,
    pub evidence: Vec<QualificationEvidenceV2>,
    pub resource_requirements: ResourceRequirementEnvelope,
    pub anti_rollback_metadata: QualificationAntiRollback,
}
impl Validate for ModelCapabilityPayload {
    #[allow(clippy::too_many_lines)]
    fn validate(&self) -> Result<()> {
        if self.manifest_version != MODEL_CAPABILITY_MANIFEST_CONTRACT_VERSION || self.revision == 0
        {
            return Err(ContractError::field(
                "manifest",
                "invalid version or revision",
            ));
        }
        self.manifest_id.validate()?;
        match (self.revision, &self.previous_revision_digest) {
            (1, None) => {}
            (2.., Some(digest)) => validate_sha256("previous_revision_digest", digest)?,
            _ => {
                return Err(ContractError::field(
                    "previous_revision_digest",
                    "omitted only for revision one and required afterward",
                ))
            }
        }
        validate_timestamp("issued_at", &self.issued_at)?;
        validate_timestamp("expires_at", &self.expires_at)?;
        validate_timestamp_order("issued_at", &self.issued_at, "expires_at", &self.expires_at)?;
        self.producer_identity.validate()?;
        self.logical_model_id.validate()?;
        self.runtime_identity.validate()?;
        if self.logical_model_id != self.runtime_identity.model_id
            || self.logical_model_id.as_str() == self.runtime_identity.provider_model_id
        {
            return Err(ContractError::field(
                "runtime_identity",
                "logical and provider identities are non-substitutable",
            ));
        }
        extensible(
            "model_class",
            &self.model_class,
            &[
                "language",
                "vision_language",
                "embedding",
                "reranker",
                "speech",
                "image",
                "video",
                "robotics_policy",
                "forecasting",
                "classifier",
            ],
        )?;
        if self.modalities.is_empty() || self.modalities.len() > 16 {
            return Err(ContractError::field(
                "modalities",
                "must contain 1 through 16",
            ));
        }
        let mut prior = None;
        for modality in &self.modalities {
            extensible(
                "modality",
                modality,
                &[
                    "text",
                    "image",
                    "audio",
                    "video",
                    "embedding",
                    "structured_data",
                    "control_policy",
                ],
            )?;
            if prior.is_some_and(|p: &String| p >= modality) {
                return Err(ContractError::field(
                    "modalities",
                    "must be sorted and unique",
                ));
            }
            prior = Some(modality);
        }
        if self.taxonomy_revision.is_empty() || self.taxonomy_revision.len() > 128 {
            return Err(ContractError::field("taxonomy_revision", "invalid length"));
        }
        self.evaluation_revision.validate()?;
        validate_timestamp("qualification_timestamp", &self.qualification_timestamp)?;
        if self.evidence.is_empty()
            || self.evidence.len() > 256
            || self.capabilities.is_empty()
            || self.capabilities.len() > 256
        {
            return Err(ContractError::field(
                "manifest",
                "capability or evidence bounds",
            ));
        }
        let mut evidence_ids = BTreeSet::new();
        for evidence in &self.evidence {
            evidence.validate_for(&self.runtime_identity.artifact_id)?;
            if evidence.evaluation_revision != self.evaluation_revision
                || !evidence_ids.insert(&evidence.evidence_id)
            {
                return Err(ContractError::field(
                    "evidence",
                    "revision mismatch or duplicate",
                ));
            }
            if compare_timestamp_values(&evidence.observed_at, &self.qualification_timestamp)?
                != Ordering::Less
            {
                return Err(ContractError::field(
                    "qualification_timestamp",
                    "must follow evidence",
                ));
            }
        }
        let mut capability_ids = BTreeSet::new();
        for capability in &self.capabilities {
            capability.validate()?;
            if capability.evaluation_revision != self.evaluation_revision
                || !capability_ids.insert(&capability.capability_id)
            {
                return Err(ContractError::field(
                    "capabilities",
                    "revision mismatch or duplicate",
                ));
            }
            if capability
                .evidence_ids
                .iter()
                .any(|id| !evidence_ids.contains(id))
            {
                return Err(ContractError::field(
                    "evidence_ids",
                    "references absent evidence",
                ));
            }
        }
        self.resource_requirements.validate()?;
        if self.anti_rollback_metadata.sequence != self.revision
            || self.anti_rollback_metadata.lineage_id.is_empty()
        {
            return Err(ContractError::field(
                "anti_rollback_metadata",
                "must bind nonzero manifest revision",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum ModelCapabilityManifestKind {
    #[serde(rename = "qualification.model_capability_manifest")]
    ModelCapabilityManifest,
}
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ModelCapabilityManifest {
    pub contract_version: String,
    pub kind: ModelCapabilityManifestKind,
    pub payload: ModelCapabilityPayload,
    pub canonical_digest: String,
    pub signature_envelope: QualificationSignatureEnvelope,
}
impl Validate for ModelCapabilityManifest {
    fn validate(&self) -> Result<()> {
        if self.contract_version != MODEL_CAPABILITY_MANIFEST_CONTRACT_VERSION {
            return Err(ContractError::field("contract_version", "must equal 2.0.0"));
        }
        self.payload.validate()?;
        if canonical_model_capability_digest(&self.payload)? != self.canonical_digest {
            return Err(ContractError::field(
                "canonical_digest",
                "does not match payload",
            ));
        }
        self.signature_envelope.validate()
    }
}
/// Computes the namespace-bound canonical payload digest.
///
/// # Errors
///
/// Returns an error when the payload is invalid or cannot be serialized canonically.
pub fn canonical_model_capability_digest(payload: &ModelCapabilityPayload) -> Result<String> {
    payload.validate()?;
    let value = serde_json::to_value(payload)?;
    let canonical = serde_json::to_vec(&crate::document::sort_value(value))?;
    let mut hasher = Sha256::new();
    hasher.update(DIGEST_NAMESPACE);
    hasher.update(canonical);
    Ok(format!("{:x}", hasher.finalize()))
}
/// Builds the fixed-namespace signing transcript for a canonical manifest digest.
///
/// # Errors
///
/// Returns an error when `digest` is not a canonical SHA-256 hexadecimal value.
pub fn model_capability_signing_transcript(digest: &str) -> Result<Vec<u8>> {
    validate_sha256("canonical_digest", digest)?;
    let mut transcript = SIGNATURE_NAMESPACE.to_vec();
    let bytes = digest.as_bytes();
    for pair in bytes.chunks_exact(2) {
        let high = (pair[0] as char)
            .to_digit(16)
            .ok_or_else(|| ContractError::field("canonical_digest", "invalid hex"))?;
        let low = (pair[1] as char)
            .to_digit(16)
            .ok_or_else(|| ContractError::field("canonical_digest", "invalid hex"))?;
        let byte = u8::try_from((high << 4) | low)
            .map_err(|_| ContractError::field("canonical_digest", "invalid hex byte"))?;
        transcript.push(byte);
    }
    Ok(transcript)
}
/// Validates immutable identity, digest lineage, and anti-rollback ordering between revisions.
///
/// # Errors
///
/// Returns an error when either manifest is invalid or the revision transition is unsafe.
pub fn validate_manifest_revision(
    previous: &ModelCapabilityManifest,
    current: &ModelCapabilityManifest,
) -> Result<()> {
    previous.validate()?;
    current.validate()?;
    if current.payload.manifest_id != previous.payload.manifest_id
        || current.payload.logical_model_id != previous.payload.logical_model_id
        || current.payload.runtime_identity.artifact_id
            != previous.payload.runtime_identity.artifact_id
        || current.payload.revision != previous.payload.revision + 1
        || current.payload.anti_rollback_metadata.sequence
            <= previous.payload.anti_rollback_metadata.sequence
        || current.payload.previous_revision_digest.as_deref() != Some(&previous.canonical_digest)
    {
        return Err(ContractError::field(
            "manifest_revision",
            "identity substitution, stale revision, or broken prior digest",
        ));
    }
    Ok(())
}

const _: &str = COGNITIVE_CONTRACT_VERSION;
