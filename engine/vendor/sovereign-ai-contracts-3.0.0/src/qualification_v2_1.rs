use crate::common::{compare_timestamp_values, validate_sha256};
use crate::error::{ContractError, Result, Validate};
use crate::qualification::QualificationLifecycleEvent;
use crate::qualification_v2::{
    checked_revision_successor, CapabilityQualificationV2, ModelCapabilityManifestKind,
    ModelCapabilityPayload, ModelQualificationState, QualificationEvidenceV2,
    MODEL_CAPABILITY_MANIFEST_CONTRACT_VERSION,
};
use crate::{
    cognitive::ResourceRequirementEnvelope,
    identity::{EvaluationRevisionId, ModelId},
    inference_runtime_v3::RuntimeIdentityV3,
    qualification::{
        QualificationAntiRollback, QualificationManifestId, QualificationProducerIdentity,
        QualificationSignatureEnvelope,
    },
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::cmp::Ordering;

pub const MODEL_CAPABILITY_MANIFEST_CONTRACT_VERSION_V21: &str = "2.1.0";
const DIGEST_NAMESPACE: &[u8] = b"SOVEREIGN-MODEL-CAPABILITY-MANIFEST-V2.1\0";
const SIGNATURE_NAMESPACE: &[u8] = b"SOVEREIGN-MODEL-CAPABILITY-SIGNATURE-V2.1\0";

/// Current eligibility state of one immutable qualification lineage.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelQualificationStateV21 {
    Evaluated,
    Qualified,
    Rejected,
    Suspended,
    Expired,
    Revoked,
    Experimental,
    Unactivated,
}

/// Evidence provenance classification. This is not signature verification or activation.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum QualificationTrustStateV21 {
    DevelopmentFixture,
    Verified,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ModelCapabilityPayloadV21 {
    pub manifest_version: String,
    pub manifest_id: QualificationManifestId,
    pub revision: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub previous_revision_digest: Option<String>,
    pub issued_at: String,
    pub expires_at: String,
    pub producer_identity: QualificationProducerIdentity,
    pub qualification_state: ModelQualificationStateV21,
    pub trust_state: QualificationTrustStateV21,
    pub state_transition: QualificationLifecycleEvent,
    pub logical_model_id: ModelId,
    pub runtime_identity: RuntimeIdentityV3,
    pub model_class: String,
    pub locality: crate::qualification_v2::ModelLocality,
    pub modalities: Vec<String>,
    pub taxonomy_revision: String,
    pub capabilities: Vec<CapabilityQualificationV2>,
    pub evaluation_revision: EvaluationRevisionId,
    pub qualification_timestamp: String,
    pub evidence: Vec<QualificationEvidenceV2>,
    pub resource_requirements: ResourceRequirementEnvelope,
    pub anti_rollback_metadata: QualificationAntiRollback,
}

impl Validate for ModelCapabilityPayloadV21 {
    fn validate(&self) -> Result<()> {
        if self.manifest_version != MODEL_CAPABILITY_MANIFEST_CONTRACT_VERSION_V21 {
            return Err(ContractError::field("manifest_version", "must equal 2.1.0"));
        }
        self.state_transition.validate()?;

        // Reuse the unchanged V2 structural validator through a private projection.
        let qualification_state = match self.qualification_state {
            ModelQualificationStateV21::Evaluated => ModelQualificationState::Evaluated,
            ModelQualificationStateV21::Qualified => ModelQualificationState::Qualified,
            ModelQualificationStateV21::Rejected => ModelQualificationState::Rejected,
            ModelQualificationStateV21::Expired => ModelQualificationState::Expired,
            ModelQualificationStateV21::Revoked => ModelQualificationState::Revoked,
            ModelQualificationStateV21::Experimental => ModelQualificationState::Experimental,
            ModelQualificationStateV21::Unactivated | ModelQualificationStateV21::Suspended => {
                ModelQualificationState::Unactivated
            }
        };
        let base = ModelCapabilityPayload {
            manifest_version: MODEL_CAPABILITY_MANIFEST_CONTRACT_VERSION.into(),
            manifest_id: self.manifest_id.clone(),
            revision: self.revision,
            previous_revision_digest: self.previous_revision_digest.clone(),
            issued_at: self.issued_at.clone(),
            expires_at: self.expires_at.clone(),
            producer_identity: self.producer_identity.clone(),
            qualification_state,
            logical_model_id: self.logical_model_id.clone(),
            runtime_identity: self.runtime_identity.clone(),
            model_class: self.model_class.clone(),
            locality: self.locality.clone(),
            modalities: self.modalities.clone(),
            taxonomy_revision: self.taxonomy_revision.clone(),
            capabilities: self.capabilities.clone(),
            evaluation_revision: self.evaluation_revision.clone(),
            qualification_timestamp: self.qualification_timestamp.clone(),
            evidence: self.evidence.clone(),
            resource_requirements: self.resource_requirements.clone(),
            anti_rollback_metadata: self.anti_rollback_metadata.clone(),
        };
        base.validate()?;
        if compare_timestamp_values(
            &self.qualification_timestamp,
            &self.state_transition.occurred_at,
        )? == Ordering::Greater
            || compare_timestamp_values(&self.state_transition.occurred_at, &self.issued_at)?
                == Ordering::Greater
        {
            return Err(ContractError::field(
                "state_transition.occurred_at",
                "must be between qualification_timestamp and issued_at",
            ));
        }
        if self.trust_state == QualificationTrustStateV21::DevelopmentFixture
            && matches!(
                self.qualification_state,
                ModelQualificationStateV21::Evaluated
                    | ModelQualificationStateV21::Qualified
                    | ModelQualificationStateV21::Suspended
            )
        {
            return Err(ContractError::field(
                "qualification_state",
                "development_fixture evidence cannot establish evaluated, qualified, or suspended state",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ModelCapabilityManifestV21 {
    pub contract_version: String,
    pub kind: ModelCapabilityManifestKind,
    pub payload: ModelCapabilityPayloadV21,
    pub canonical_digest: String,
    pub signature_envelope: QualificationSignatureEnvelope,
}

impl Validate for ModelCapabilityManifestV21 {
    fn validate(&self) -> Result<()> {
        if self.contract_version != MODEL_CAPABILITY_MANIFEST_CONTRACT_VERSION_V21 {
            return Err(ContractError::field("contract_version", "must equal 2.1.0"));
        }
        self.payload.validate()?;
        if canonical_model_capability_digest_v21(&self.payload)? != self.canonical_digest {
            return Err(ContractError::field(
                "canonical_digest",
                "does not match payload",
            ));
        }
        self.signature_envelope.validate()
    }
}

/// Computes the version-specific canonical payload digest.
///
/// # Errors
///
/// Returns an error when the payload is invalid or cannot be serialized canonically.
pub fn canonical_model_capability_digest_v21(
    payload: &ModelCapabilityPayloadV21,
) -> Result<String> {
    payload.validate()?;
    let value = serde_json::to_value(payload)?;
    let canonical = serde_json::to_vec(&crate::document::sort_value(value))?;
    let mut hasher = Sha256::new();
    hasher.update(DIGEST_NAMESPACE);
    hasher.update(canonical);
    Ok(format!("{:x}", hasher.finalize()))
}

/// Builds the fixed 2.1 signing transcript. It does not verify signatures.
///
/// # Errors
///
/// Returns an error when the digest is not canonical lowercase SHA-256 hexadecimal.
pub fn model_capability_signing_transcript_v21(digest: &str) -> Result<Vec<u8>> {
    validate_sha256("canonical_digest", digest)?;
    let mut transcript = SIGNATURE_NAMESPACE.to_vec();
    for pair in digest.as_bytes().chunks_exact(2) {
        let high = (pair[0] as char)
            .to_digit(16)
            .ok_or_else(|| ContractError::field("canonical_digest", "invalid hex"))?;
        let low = (pair[1] as char)
            .to_digit(16)
            .ok_or_else(|| ContractError::field("canonical_digest", "invalid hex"))?;
        transcript.push(
            u8::try_from((high << 4) | low)
                .map_err(|_| ContractError::field("canonical_digest", "invalid hex byte"))?,
        );
    }
    Ok(transcript)
}

/// Validates one exact adjacent lifecycle revision.
///
/// # Errors
///
/// Returns an error when either document or the lineage transition is invalid.
#[allow(clippy::too_many_lines)]
pub fn validate_manifest_revision_v21(
    previous: &ModelCapabilityManifestV21,
    current: &ModelCapabilityManifestV21,
) -> Result<()> {
    previous.validate()?;
    let next_revision = checked_revision_successor(previous.payload.revision)?;
    current.validate()?;
    if current.payload.manifest_id != previous.payload.manifest_id
        || current.payload.logical_model_id != previous.payload.logical_model_id
        || current.payload.runtime_identity != previous.payload.runtime_identity
        || current.payload.producer_identity != previous.payload.producer_identity
        || current.payload.model_class != previous.payload.model_class
        || current.payload.locality != previous.payload.locality
        || current.payload.anti_rollback_metadata.lineage_id
            != previous.payload.anti_rollback_metadata.lineage_id
    {
        return Err(ContractError::field(
            "manifest_revision",
            "identity or lineage substitution",
        ));
    }
    if previous.payload.qualification_state == ModelQualificationStateV21::Revoked {
        return Err(ContractError::field(
            "manifest_revision",
            "revocation is terminal",
        ));
    }
    if current.payload.revision != next_revision
        || current.payload.anti_rollback_metadata.sequence != current.payload.revision
        || current.payload.previous_revision_digest.as_deref() != Some(&previous.canonical_digest)
    {
        return Err(ContractError::field(
            "manifest_revision",
            "stale revision or broken prior digest",
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
    if previous.payload.trust_state == QualificationTrustStateV21::Verified
        && current.payload.trust_state == QualificationTrustStateV21::DevelopmentFixture
    {
        return Err(ContractError::field(
            "trust_state",
            "verified evidence cannot roll back to development_fixture",
        ));
    }

    let evaluation_changed =
        current.payload.evaluation_revision != previous.payload.evaluation_revision;
    let qualification_order = compare_timestamp_values(
        &previous.payload.qualification_timestamp,
        &current.payload.qualification_timestamp,
    )?;
    if qualification_order == Ordering::Greater
        || evaluation_changed != (qualification_order == Ordering::Less)
    {
        return Err(ContractError::field(
            "qualification_timestamp",
            "evaluation revision and timestamp must advance together and never roll back",
        ));
    }
    if current.payload.trust_state != previous.payload.trust_state && !evaluation_changed {
        return Err(ContractError::field(
            "trust_state",
            "evidence trust elevation requires a new evaluation",
        ));
    }
    if evaluation_changed {
        validate_fresh_evidence(&previous.payload.evidence, &current.payload.evidence)?;
    } else if current.payload.taxonomy_revision != previous.payload.taxonomy_revision
        || current.payload.modalities != previous.payload.modalities
        || current.payload.capabilities != previous.payload.capabilities
        || current.payload.evidence != previous.payload.evidence
        || current.payload.resource_requirements != previous.payload.resource_requirements
    {
        return Err(ContractError::field(
            "evaluation_revision",
            "evaluation-derived fields changed without a new evaluation",
        ));
    }
    let expiry_order =
        compare_timestamp_values(&previous.payload.expires_at, &current.payload.expires_at)?;
    if expiry_order == Ordering::Less
        && !evaluation_changed
        && !is_restrictive_transition(
            &previous.payload.qualification_state,
            &current.payload.qualification_state,
        )
    {
        return Err(ContractError::field(
            "expires_at",
            "qualification freshness cannot be extended without a new evaluation",
        ));
    }

    let state_changed = current.payload.qualification_state != previous.payload.qualification_state;
    let transition_order = compare_timestamp_values(
        &previous.payload.state_transition.occurred_at,
        &current.payload.state_transition.occurred_at,
    )?;
    if state_changed {
        if transition_order != Ordering::Less
            || !allowed_transition(
                &previous.payload.qualification_state,
                &current.payload.qualification_state,
            )
        {
            return Err(ContractError::field(
                "qualification_state",
                "unsupported or non-monotonic lifecycle transition",
            ));
        }
        if requires_fresh_evaluation_transition(
            &previous.payload.qualification_state,
            &current.payload.qualification_state,
        ) && !evaluation_changed
        {
            return Err(ContractError::field(
                "qualification_state",
                "transition requires new evidence and a later qualification timestamp",
            ));
        }
    } else if evaluation_changed {
        if transition_order != Ordering::Less {
            return Err(ContractError::field(
                "state_transition",
                "same-state re-evaluation must record a later lifecycle event",
            ));
        }
    } else if transition_order != Ordering::Equal
        || current.payload.state_transition.reason_code
            != previous.payload.state_transition.reason_code
    {
        return Err(ContractError::field(
            "state_transition",
            "must remain unchanged while qualification state and evaluation are unchanged",
        ));
    }
    Ok(())
}

fn allowed_transition(
    previous: &ModelQualificationStateV21,
    current: &ModelQualificationStateV21,
) -> bool {
    use ModelQualificationStateV21::{
        Evaluated, Experimental, Expired, Qualified, Rejected, Revoked, Suspended, Unactivated,
    };
    matches!(
        (previous, current),
        (
            Experimental,
            Unactivated | Evaluated | Rejected | Expired | Revoked
        ) | (
            Unactivated | Rejected | Suspended,
            Evaluated | Qualified | Rejected | Expired | Revoked
        ) | (
            Evaluated,
            Qualified | Rejected | Suspended | Expired | Revoked
        ) | (Qualified, Rejected | Suspended | Expired | Revoked)
            | (Expired, Evaluated | Qualified | Rejected | Revoked)
    )
}

fn requires_fresh_evaluation_transition(
    previous: &ModelQualificationStateV21,
    current: &ModelQualificationStateV21,
) -> bool {
    (*current == ModelQualificationStateV21::Rejected
        && matches!(
            previous,
            ModelQualificationStateV21::Qualified | ModelQualificationStateV21::Suspended
        ))
        || (matches!(
            current,
            ModelQualificationStateV21::Evaluated | ModelQualificationStateV21::Qualified
        ) && matches!(
            previous,
            ModelQualificationStateV21::Rejected
                | ModelQualificationStateV21::Suspended
                | ModelQualificationStateV21::Expired
        ))
}

fn validate_fresh_evidence(
    previous: &[QualificationEvidenceV2],
    current: &[QualificationEvidenceV2],
) -> Result<()> {
    for evidence in current {
        if previous
            .iter()
            .any(|prior| prior.evidence_id == evidence.evidence_id)
        {
            return Err(ContractError::field(
                "evidence.evidence_id",
                "new evaluation cannot relabel or reuse prior immutable evidence",
            ));
        }
        if previous.iter().any(|prior| {
            prior.evidence_reference.reference_id == evidence.evidence_reference.reference_id
                && prior.evidence_reference.digest == evidence.evidence_reference.digest
        }) {
            return Err(ContractError::field(
                "evidence.evidence_reference",
                "new evaluation cannot relabel or reuse a prior immutable evaluation result",
            ));
        }
    }
    Ok(())
}

fn is_restrictive_transition(
    previous: &ModelQualificationStateV21,
    current: &ModelQualificationStateV21,
) -> bool {
    previous != current
        && matches!(
            current,
            ModelQualificationStateV21::Rejected
                | ModelQualificationStateV21::Suspended
                | ModelQualificationStateV21::Expired
                | ModelQualificationStateV21::Revoked
        )
}
