use crate::common::{
    validate_extensions, validate_reason_code, validate_sha256, validate_timestamp,
    validate_timestamp_order, Component, ContractVersion, Extensions, RequiredExtensions,
};
use crate::error::{ContractError, Result, Validate};
use crate::identity::{ModelArtifactId, ModelId};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;

const DIGEST_NAMESPACE: &[u8] = b"SOVEREIGN-COGNITIVE-QUALIFICATION-V1\0";
const SIGNATURE_NAMESPACE: &[u8] = b"SOVEREIGN-COGNITIVE-QUALIFICATION-SIGNATURE-V1\0";
const QUALIFICATION_ROLES: &[&str] = &[
    "PLANNER",
    "ARCHITECT",
    "IMPLEMENTER",
    "RESEARCHER",
    "ANALYST",
    "SOLVER",
    "CRITIC",
    "VERIFIER",
    "SYNTHESIZER",
    "REVIEWER",
    "SECURITY_REVIEWER",
    "EVIDENCE_COLLECTOR",
    "TOOL_PLANNER",
    "FAST_PATH_RESPONDER",
    "DEEP_PATH_REASONER",
];
const QUALIFICATION_MODALITIES: &[&str] = &["text", "image", "audio", "video", "embedding"];

fn field(problem: impl Into<String>, name: &'static str) -> ContractError {
    ContractError::field(name, problem)
}

fn valid_identity(value: &str) -> bool {
    if value.is_empty() || value.len() > 128 || !value.is_ascii() {
        return false;
    }
    let bytes = value.as_bytes();
    if !bytes[0].is_ascii_lowercase() {
        return false;
    }
    let mut separator = false;
    for byte in bytes {
        if byte.is_ascii_lowercase() || byte.is_ascii_digit() {
            separator = false;
        } else if matches!(byte, b'.' | b'_' | b'-') && !separator {
            separator = true;
        } else {
            return false;
        }
    }
    !separator
}

fn validate_identity(field_name: &'static str, value: &str) -> Result<()> {
    if valid_identity(value) {
        Ok(())
    } else {
        Err(field("must be a bounded lowercase identity", field_name))
    }
}

fn valid_reference(value: &str) -> bool {
    if value.is_empty() || value.len() > 128 || !value.is_ascii() {
        return false;
    }
    value.as_bytes()[0].is_ascii_lowercase()
        && value.bytes().all(|byte| {
            byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b':' | b'/' | b'-')
        })
        && value
            .bytes()
            .any(|byte| matches!(byte, b'.' | b'_' | b':' | b'/' | b'-'))
}

fn validate_reference(field_name: &'static str, value: &str) -> Result<()> {
    if valid_reference(value) {
        Ok(())
    } else {
        Err(field(
            "must be a bounded redacted reference identity",
            field_name,
        ))
    }
}

fn valid_capability(value: &str) -> bool {
    if value.is_empty() || value.len() > 160 || !value.is_ascii() {
        return false;
    }
    let segments: Vec<&str> = value.split('.').collect();
    if segments.is_empty() || segments.len() > 12 {
        return false;
    }
    segments.iter().all(|segment| {
        !segment.is_empty()
            && segment.len() <= 48
            && segment.as_bytes()[0].is_ascii_lowercase()
            && segment
                .bytes()
                .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'_')
    })
}

fn unique_nonempty<'a>(
    field_name: &'static str,
    values: impl IntoIterator<Item = &'a str>,
) -> Result<()> {
    let mut seen = BTreeSet::new();
    for value in values {
        if !seen.insert(value) {
            return Err(field(
                format!("contains duplicate value {value:?}"),
                field_name,
            ));
        }
    }
    Ok(())
}

fn sorted_unique_nonempty<'a>(
    field_name: &'static str,
    values: impl IntoIterator<Item = &'a str>,
) -> Result<()> {
    let values: Vec<&str> = values.into_iter().collect();
    unique_nonempty(field_name, values.iter().copied())?;
    if values.windows(2).all(|pair| pair[0] < pair[1]) {
        Ok(())
    } else {
        Err(field(
            "must be strictly lexicographically ordered",
            field_name,
        ))
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(transparent)]
pub struct QualificationManifestId(String);

impl QualificationManifestId {
    /// Constructs and validates an opaque qualification manifest identifier.
    ///
    /// # Errors
    ///
    /// Returns an error when the identifier does not use the bounded `qmf_`
    /// grammar published by Cognitive Qualification Manifest V1.
    pub fn new(value: impl Into<String>) -> Result<Self> {
        let value = value.into();
        let identifier = Self(value);
        identifier.validate()?;
        Ok(identifier)
    }

    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl Validate for QualificationManifestId {
    fn validate(&self) -> Result<()> {
        let Some(suffix) = self.0.strip_prefix("qmf_") else {
            return Err(field("must start with qmf_", "manifest_id"));
        };
        if (2..=64).contains(&suffix.len())
            && suffix.as_bytes()[0].is_ascii_lowercase()
            && suffix.bytes().all(|byte| {
                byte.is_ascii_lowercase() || byte.is_ascii_digit() || matches!(byte, b'_' | b'-')
            })
        {
            Ok(())
        } else {
            Err(field("has invalid suffix", "manifest_id"))
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct QualificationProducerIdentity {
    pub component: Component,
    pub producer_id: String,
}

impl Validate for QualificationProducerIdentity {
    fn validate(&self) -> Result<()> {
        if self.component != Component::ModelFoundry {
            return Err(field(
                "must equal model_foundry",
                "producer_identity.component",
            ));
        }
        validate_identity("producer_identity.producer_id", &self.producer_id)
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct QualifiedModelIdentity {
    pub model_id: ModelId,
    pub family: String,
    pub revision: String,
}

impl Validate for QualifiedModelIdentity {
    fn validate(&self) -> Result<()> {
        self.model_id.validate()?;
        validate_identity("model_identity.family", &self.family)?;
        validate_identity("model_identity.revision", &self.revision)
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum QualificationLocality {
    Local,
    External,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct QualifiedProviderIdentity {
    pub provider_id: String,
    pub locality: QualificationLocality,
}

impl Validate for QualifiedProviderIdentity {
    fn validate(&self) -> Result<()> {
        validate_identity("provider_identity.provider_id", &self.provider_id)
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ArtifactDigestAlgorithm {
    Sha256,
    Sha384,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct QualifiedArtifactIdentity {
    pub artifact_id: ModelArtifactId,
    pub digest_algorithm: ArtifactDigestAlgorithm,
    pub digest: String,
    pub size_bytes: u64,
}

impl Validate for QualifiedArtifactIdentity {
    fn validate(&self) -> Result<()> {
        self.artifact_id.validate()?;
        let expected = match self.digest_algorithm {
            ArtifactDigestAlgorithm::Sha256 => 64,
            ArtifactDigestAlgorithm::Sha384 => 96,
        };
        if self.digest.len() != expected
            || !self
                .digest
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        {
            return Err(field(
                "has invalid lowercase hexadecimal length",
                "artifact_identity.digest",
            ));
        }
        if !(1..=1_099_511_627_776).contains(&self.size_bytes) {
            return Err(field(
                "is outside the published bound",
                "artifact_identity.size_bytes",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct QualifiedBackendIdentity {
    pub backend_id: String,
    pub backend_version: String,
}

impl Validate for QualifiedBackendIdentity {
    fn validate(&self) -> Result<()> {
        validate_identity("backend_identity.backend_id", &self.backend_id)?;
        validate_identity("backend_identity.backend_version", &self.backend_version)
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ConfidenceInterval {
    pub lower_basis_points: u16,
    pub upper_basis_points: u16,
    pub confidence_basis_points: u16,
}

impl ConfidenceInterval {
    fn validate_for(&self, score: u16) -> Result<()> {
        if self.lower_basis_points <= score
            && score <= self.upper_basis_points
            && self.upper_basis_points <= 10_000
            && (5_000..=10_000).contains(&self.confidence_basis_points)
        {
            Ok(())
        } else {
            Err(field(
                "must contain the score and use bounded basis points",
                "confidence",
            ))
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CapabilityQualification {
    pub capability_id: String,
    pub score_basis_points: u16,
    pub confidence: ConfidenceInterval,
    pub evaluation_revision: String,
    pub evidence_ids: Vec<String>,
}

impl Validate for CapabilityQualification {
    fn validate(&self) -> Result<()> {
        if !valid_capability(&self.capability_id) || self.score_basis_points > 10_000 {
            return Err(field(
                "contains an invalid capability or score",
                "capability",
            ));
        }
        self.confidence.validate_for(self.score_basis_points)?;
        validate_identity("evaluation_revision", &self.evaluation_revision)?;
        if self.evidence_ids.is_empty() || self.evidence_ids.len() > 16 {
            return Err(field("must contain one through 16 entries", "evidence_ids"));
        }
        sorted_unique_nonempty("evidence_ids", self.evidence_ids.iter().map(String::as_str))?;
        self.evidence_ids
            .iter()
            .try_for_each(|value| validate_reference("evidence_id", value))
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct QualificationContextLimits {
    pub maximum_input_tokens: u32,
    pub maximum_output_tokens: u32,
}

impl Validate for QualificationContextLimits {
    fn validate(&self) -> Result<()> {
        if (1..=2_000_000).contains(&self.maximum_input_tokens)
            && (1..=131_072).contains(&self.maximum_output_tokens)
        {
            Ok(())
        } else {
            Err(field(
                "contains out-of-range token limits",
                "context_limits",
            ))
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct QualificationOutputLimits {
    pub maximum_output_bytes: u64,
    pub structured_output: bool,
    pub tool_use: bool,
    pub streaming: bool,
}

impl Validate for QualificationOutputLimits {
    fn validate(&self) -> Result<()> {
        if (1..=16 * 1024 * 1024).contains(&self.maximum_output_bytes) {
            Ok(())
        } else {
            Err(field(
                "is outside the published bound",
                "maximum_output_bytes",
            ))
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct QualificationResourceLimits {
    pub minimum_inference_memory_bytes: u64,
    pub maximum_inference_memory_bytes: u64,
    pub maximum_concurrent_requests: u32,
    pub compute_classes: Vec<String>,
}

impl Validate for QualificationResourceLimits {
    fn validate(&self) -> Result<()> {
        if self.minimum_inference_memory_bytes == 0
            || self.maximum_inference_memory_bytes < self.minimum_inference_memory_bytes
            || self.maximum_inference_memory_bytes > 1_099_511_627_776
            || !(1..=4096).contains(&self.maximum_concurrent_requests)
            || self.compute_classes.is_empty()
            || self.compute_classes.len() > 16
        {
            return Err(field("contains invalid bounded values", "resource_limits"));
        }
        sorted_unique_nonempty(
            "compute_classes",
            self.compute_classes.iter().map(String::as_str),
        )?;
        self.compute_classes
            .iter()
            .try_for_each(|value| validate_identity("compute_class", value))
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct QualificationPerformance {
    pub latency_p50_milliseconds: u32,
    pub latency_p95_milliseconds: u32,
    pub throughput_milli_tokens_per_second: u64,
}

impl Validate for QualificationPerformance {
    fn validate(&self) -> Result<()> {
        if self.latency_p95_milliseconds >= self.latency_p50_milliseconds
            && self.latency_p95_milliseconds <= 3_600_000
            && self.throughput_milli_tokens_per_second <= 1_000_000_000
        {
            Ok(())
        } else {
            Err(field(
                "contains invalid latency or throughput",
                "performance",
            ))
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct QualificationReliability {
    pub success_basis_points: u16,
    pub sample_count: u64,
    pub window_seconds: u64,
}

impl Validate for QualificationReliability {
    fn validate(&self) -> Result<()> {
        if self.success_basis_points <= 10_000
            && (1..=1_000_000_000).contains(&self.sample_count)
            && (1..=31_536_000).contains(&self.window_seconds)
        {
            Ok(())
        } else {
            Err(field("contains invalid bounded values", "reliability"))
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum QualificationCostApplicability {
    NotApplicable,
    Priced,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct QualificationCost {
    pub applicability: QualificationCostApplicability,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub currency: Option<String>,
    pub input_micros_per_million_tokens: u64,
    pub output_micros_per_million_tokens: u64,
}

impl Validate for QualificationCost {
    fn validate(&self) -> Result<()> {
        match self.applicability {
            QualificationCostApplicability::NotApplicable
                if self.currency.is_none()
                    && self.input_micros_per_million_tokens == 0
                    && self.output_micros_per_million_tokens == 0 =>
            {
                Ok(())
            }
            QualificationCostApplicability::Priced
                if self.currency.as_ref().is_some_and(|currency| {
                    currency.len() == 3 && currency.bytes().all(|byte| byte.is_ascii_uppercase())
                }) && self.input_micros_per_million_tokens <= 1_000_000_000_000
                    && self.output_micros_per_million_tokens <= 1_000_000_000_000 =>
            {
                Ok(())
            }
            _ => Err(field("does not match applicability semantics", "cost")),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct QualificationEvidence {
    pub evidence_id: String,
    pub evidence_digest: String,
    pub artifact_id: ModelArtifactId,
    pub benchmark_revision: String,
    pub observed_at: String,
}

impl QualificationEvidence {
    fn validate_for(&self, artifact_id: &ModelArtifactId) -> Result<()> {
        validate_reference("evidence_id", &self.evidence_id)?;
        validate_sha256("evidence_digest", &self.evidence_digest)?;
        if &self.artifact_id != artifact_id {
            return Err(field(
                "must match manifest artifact",
                "evidence.artifact_id",
            ));
        }
        validate_identity("benchmark_revision", &self.benchmark_revision)?;
        validate_timestamp("observed_at", &self.observed_at)
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct QualificationLifecycleEvent {
    pub occurred_at: String,
    pub reason_code: String,
}

impl Validate for QualificationLifecycleEvent {
    fn validate(&self) -> Result<()> {
        validate_timestamp("occurred_at", &self.occurred_at)?;
        validate_reason_code("reason_code", &self.reason_code)
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct QualificationAntiRollback {
    pub lineage_id: String,
    pub sequence: u64,
}

impl QualificationAntiRollback {
    fn validate_for(&self, revision: u64) -> Result<()> {
        validate_reference("lineage_id", &self.lineage_id)?;
        if revision == 0 || self.sequence != revision {
            return Err(field(
                "must equal manifest revision",
                "anti_rollback.sequence",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum QualificationState {
    Qualified,
    Suspended,
    Revoked,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum QualificationTrustState {
    DevelopmentFixture,
    Verified,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CognitiveQualificationPayload {
    pub manifest_version: String,
    pub schema_version: String,
    pub manifest_id: QualificationManifestId,
    pub revision: u64,
    pub issued_at: String,
    pub expires_at: String,
    pub producer_identity: QualificationProducerIdentity,
    pub qualification_state: QualificationState,
    pub trust_state: QualificationTrustState,
    pub model_identity: QualifiedModelIdentity,
    pub provider_identity: QualifiedProviderIdentity,
    pub artifact_identity: QualifiedArtifactIdentity,
    pub backend_identity: QualifiedBackendIdentity,
    pub taxonomy_revision: String,
    pub capabilities: Vec<CapabilityQualification>,
    pub roles: Vec<String>,
    pub modalities: Vec<String>,
    pub context_limits: QualificationContextLimits,
    pub output_limits: QualificationOutputLimits,
    pub resource_limits: QualificationResourceLimits,
    pub performance: QualificationPerformance,
    pub reliability: QualificationReliability,
    pub cost: QualificationCost,
    pub evaluation_evidence: Vec<QualificationEvidence>,
    pub benchmark_revisions: Vec<String>,
    pub qualification_timestamp: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub suspension: Option<QualificationLifecycleEvent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub revocation: Option<QualificationLifecycleEvent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub previous_revision_digest: Option<String>,
    pub anti_rollback_metadata: QualificationAntiRollback,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extensions: Option<Extensions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub required_extensions: Option<RequiredExtensions>,
}

impl CognitiveQualificationPayload {
    fn validate_identity(&self) -> Result<()> {
        if self.manifest_version != "1.0.0" || self.schema_version != "1.0.0" || self.revision == 0
        {
            return Err(field(
                "versions must equal 1.0.0 and revision must be nonzero",
                "manifest",
            ));
        }
        self.manifest_id.validate()?;
        validate_timestamp("issued_at", &self.issued_at)?;
        validate_timestamp("expires_at", &self.expires_at)?;
        validate_timestamp_order("issued_at", &self.issued_at, "expires_at", &self.expires_at)?;
        self.producer_identity.validate()?;
        self.model_identity.validate()?;
        self.provider_identity.validate()?;
        self.artifact_identity.validate()?;
        self.backend_identity.validate()?;
        validate_identity("taxonomy_revision", &self.taxonomy_revision)
    }

    fn validate_capabilities(&self) -> Result<()> {
        if self.capabilities.is_empty() || self.capabilities.len() > 256 {
            return Err(field(
                "must contain one through 256 entries",
                "capabilities",
            ));
        }
        self.capabilities.iter().try_for_each(Validate::validate)?;
        sorted_unique_nonempty(
            "capabilities",
            self.capabilities
                .iter()
                .map(|value| value.capability_id.as_str()),
        )
    }

    fn validate_roles_and_modalities(&self) -> Result<()> {
        if self.roles.is_empty()
            || self.roles.len() > 32
            || self
                .roles
                .iter()
                .any(|role| !QUALIFICATION_ROLES.contains(&role.as_str()))
        {
            return Err(field("contains an unsupported or empty role set", "roles"));
        }
        sorted_unique_nonempty("roles", self.roles.iter().map(String::as_str))?;
        if self.modalities.is_empty()
            || self.modalities.len() > 8
            || self
                .modalities
                .iter()
                .any(|value| !QUALIFICATION_MODALITIES.contains(&value.as_str()))
        {
            return Err(field(
                "contains an unsupported or empty modality set",
                "modalities",
            ));
        }
        sorted_unique_nonempty("modalities", self.modalities.iter().map(String::as_str))
    }

    fn validate_limits(&self) -> Result<()> {
        self.context_limits.validate()?;
        self.output_limits.validate()?;
        self.resource_limits.validate()?;
        self.performance.validate()?;
        self.reliability.validate()?;
        self.cost.validate()
    }

    fn validate_evidence(&self) -> Result<()> {
        if self.evaluation_evidence.is_empty() || self.evaluation_evidence.len() > 256 {
            return Err(field(
                "must contain one through 256 entries",
                "evaluation_evidence",
            ));
        }
        for evidence in &self.evaluation_evidence {
            evidence.validate_for(&self.artifact_identity.artifact_id)?;
        }
        sorted_unique_nonempty(
            "evaluation_evidence",
            self.evaluation_evidence
                .iter()
                .map(|value| value.evidence_id.as_str()),
        )?;
        if self.benchmark_revisions.is_empty() || self.benchmark_revisions.len() > 64 {
            return Err(field(
                "must contain one through 64 entries",
                "benchmark_revisions",
            ));
        }
        sorted_unique_nonempty(
            "benchmark_revisions",
            self.benchmark_revisions.iter().map(String::as_str),
        )?;
        self.benchmark_revisions
            .iter()
            .try_for_each(|value| validate_identity("benchmark_revision", value))?;
        validate_timestamp("qualification_timestamp", &self.qualification_timestamp)?;
        validate_timestamp_order(
            "qualification_timestamp",
            &self.qualification_timestamp,
            "issued_at",
            &self.issued_at,
        )?;
        let evidence_ids: BTreeSet<&str> = self
            .evaluation_evidence
            .iter()
            .map(|evidence| evidence.evidence_id.as_str())
            .collect();
        let benchmark_revisions: BTreeSet<&str> = self
            .benchmark_revisions
            .iter()
            .map(String::as_str)
            .collect();
        for capability in &self.capabilities {
            if capability
                .evidence_ids
                .iter()
                .any(|evidence_id| !evidence_ids.contains(evidence_id.as_str()))
            {
                return Err(field(
                    "references undeclared evaluation evidence",
                    "capabilities.evidence_ids",
                ));
            }
        }
        if self
            .evaluation_evidence
            .iter()
            .any(|evidence| !benchmark_revisions.contains(evidence.benchmark_revision.as_str()))
        {
            return Err(field(
                "references an undeclared benchmark revision",
                "evaluation_evidence.benchmark_revision",
            ));
        }
        for evidence in &self.evaluation_evidence {
            validate_timestamp_order(
                "evaluation_evidence.observed_at",
                &evidence.observed_at,
                "qualification_timestamp",
                &self.qualification_timestamp,
            )?;
        }
        Ok(())
    }

    fn validate_lifecycle(&self) -> Result<()> {
        match self.qualification_state {
            QualificationState::Qualified
                if self.suspension.is_none() && self.revocation.is_none() => {}
            QualificationState::Suspended
                if self.suspension.is_some() && self.revocation.is_none() => {}
            QualificationState::Revoked if self.revocation.is_some() => {}
            _ => {
                return Err(field(
                    "does not match lifecycle evidence",
                    "qualification_state",
                ))
            }
        }
        if let Some(event) = &self.suspension {
            event.validate()?;
            validate_timestamp_order(
                "issued_at",
                &self.issued_at,
                "suspension.occurred_at",
                &event.occurred_at,
            )?;
            validate_timestamp_order(
                "suspension.occurred_at",
                &event.occurred_at,
                "expires_at",
                &self.expires_at,
            )?;
        }
        if let Some(event) = &self.revocation {
            event.validate()?;
            validate_timestamp_order(
                "issued_at",
                &self.issued_at,
                "revocation.occurred_at",
                &event.occurred_at,
            )?;
            validate_timestamp_order(
                "revocation.occurred_at",
                &event.occurred_at,
                "expires_at",
                &self.expires_at,
            )?;
        }
        Ok(())
    }

    fn validate_revision(&self) -> Result<()> {
        match (self.revision, self.previous_revision_digest.as_deref()) {
            (1, None) => {}
            (1, Some(_)) => {
                return Err(field(
                    "must be absent for revision one",
                    "previous_revision_digest",
                ))
            }
            (_, Some(digest)) => validate_sha256("previous_revision_digest", digest)?,
            (_, None) => {
                return Err(field(
                    "is required after revision one",
                    "previous_revision_digest",
                ))
            }
        }
        self.anti_rollback_metadata.validate_for(self.revision)?;
        validate_extensions(self.extensions.as_ref(), self.required_extensions.as_ref())
    }
}

impl Validate for CognitiveQualificationPayload {
    fn validate(&self) -> Result<()> {
        self.validate_identity()?;
        self.validate_capabilities()?;
        self.validate_roles_and_modalities()?;
        self.validate_limits()?;
        self.validate_evidence()?;
        self.validate_lifecycle()?;
        self.validate_revision()
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum CognitiveQualificationManifestKind {
    #[serde(rename = "qualification.cognitive_manifest")]
    CognitiveQualificationManifest,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct QualificationSignatureEnvelope {
    pub algorithm: String,
    pub key_id: String,
    pub signature: String,
}

impl Validate for QualificationSignatureEnvelope {
    fn validate(&self) -> Result<()> {
        validate_identity("signature.key_id", &self.key_id)?;
        if self.algorithm != "ed25519"
            || self.signature.len() != 88
            || !self.signature.ends_with("==")
            || !self.signature[..86]
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'+' | b'/'))
        {
            return Err(field("must be canonical Ed25519 base64", "signature"));
        }
        // Ed25519 signatures contain 64 bytes. Their final one-byte base64
        // quantum has four unused low bits in the second data character; a
        // canonical RFC 4648 encoding requires those bits to be zero.
        let final_value = base64_value(self.signature.as_bytes()[85])?;
        if final_value & 0x0f != 0 {
            return Err(field("must be canonical Ed25519 base64", "signature"));
        }
        Ok(())
    }
}

fn base64_value(value: u8) -> Result<u8> {
    match value {
        b'A'..=b'Z' => Ok(value - b'A'),
        b'a'..=b'z' => Ok(value - b'a' + 26),
        b'0'..=b'9' => Ok(value - b'0' + 52),
        b'+' => Ok(62),
        b'/' => Ok(63),
        _ => Err(field("must be canonical Ed25519 base64", "signature")),
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CognitiveQualificationManifest {
    pub contract_version: ContractVersion,
    pub kind: CognitiveQualificationManifestKind,
    pub payload: CognitiveQualificationPayload,
    pub canonical_digest: String,
    pub signature_envelope: QualificationSignatureEnvelope,
}

impl Validate for CognitiveQualificationManifest {
    fn validate(&self) -> Result<()> {
        self.payload.validate()?;
        let expected = canonical_qualification_digest(&self.payload)?;
        if self.canonical_digest != expected {
            return Err(field(
                "does not match canonical payload",
                "canonical_digest",
            ));
        }
        self.signature_envelope.validate()
    }
}

/// Returns the exact lowercase SHA-256 digest of the namespaced canonical payload.
///
/// # Errors
///
/// Returns an error when the payload violates the closed v1 contract or cannot
/// be represented as deterministic JSON.
pub fn canonical_qualification_digest(payload: &CognitiveQualificationPayload) -> Result<String> {
    payload.validate()?;
    let canonical =
        serde_json::to_vec(&crate::document::sort_value(serde_json::to_value(payload)?))?;
    let mut hasher = Sha256::new();
    hasher.update(DIGEST_NAMESPACE);
    hasher.update(canonical);
    Ok(format!("{:x}", hasher.finalize()))
}

/// Returns the exact fixed signing transcript for external Ed25519 providers.
///
/// # Errors
///
/// Returns an error when `canonical_digest` is not a lowercase SHA-256 digest.
pub fn qualification_signing_transcript(canonical_digest: &str) -> Result<Vec<u8>> {
    validate_sha256("canonical_digest", canonical_digest)?;
    let mut transcript = Vec::with_capacity(SIGNATURE_NAMESPACE.len() + 32);
    transcript.extend_from_slice(SIGNATURE_NAMESPACE);
    for pair in canonical_digest.as_bytes().chunks_exact(2) {
        let high = (pair[0] as char)
            .to_digit(16)
            .ok_or_else(|| field("invalid hex", "canonical_digest"))?;
        let low = (pair[1] as char)
            .to_digit(16)
            .ok_or_else(|| field("invalid hex", "canonical_digest"))?;
        transcript.push(
            u8::try_from((high << 4) | low)
                .map_err(|_| field("hexadecimal pair exceeds one byte", "canonical_digest"))?,
        );
    }
    Ok(transcript)
}

#[cfg(test)]
mod tests {
    use super::{
        canonical_qualification_digest, qualification_signing_transcript, valid_capability,
        valid_identity, CognitiveQualificationManifest, DIGEST_NAMESPACE,
    };
    use crate::Validate;

    const FIXTURE: &str =
        include_str!("../../fixtures/v1/valid/cognitive-qualification-manifest.json");
    const FIXTURE_DIGEST: &str = "4fa6d6dadd037b4c593caffc126bafc305fc1370fd71898dbcd060e524973299";

    fn fixture() -> CognitiveQualificationManifest {
        serde_json::from_str(FIXTURE).expect("decode qualification fixture")
    }

    #[test]
    fn capability_and_identity_grammars_fail_closed() {
        assert!(super::QualificationManifestId::new("qmf_fixture_v1").is_ok());
        assert!(super::QualificationManifestId::new("../fixture").is_err());
        assert!(valid_capability("software.debugging"));
        assert!(!valid_capability("software..debugging"));
        assert!(!valid_capability("Software.debugging"));
        assert!(valid_identity("cognitive-capabilities.v1"));
        assert!(!valid_identity("../taxonomy"));
    }

    #[test]
    fn digest_and_signing_transcript_match_go_fixture() {
        let manifest = fixture();
        manifest.validate().expect("validate qualification fixture");
        assert_eq!(
            canonical_qualification_digest(&manifest.payload).expect("digest payload"),
            FIXTURE_DIGEST
        );
        assert_eq!(
            qualification_signing_transcript(FIXTURE_DIGEST)
                .expect("build transcript")
                .len(),
            b"SOVEREIGN-COGNITIVE-QUALIFICATION-SIGNATURE-V1\0".len() + 32
        );
        assert_eq!(DIGEST_NAMESPACE.last(), Some(&0));
    }

    #[test]
    fn semantic_substitution_and_noncanonical_sets_fail_closed() {
        let mut manifest = fixture();
        manifest.payload.capabilities[0]
            .confidence
            .upper_basis_points = 7_999;
        assert!(manifest.payload.validate().is_err());

        let mut manifest = fixture();
        manifest.payload.capabilities[0].evidence_ids[0] = "evidence:missing".to_owned();
        assert!(manifest.payload.validate().is_err());

        let mut manifest = fixture();
        manifest.payload.roles.swap(0, 1);
        assert!(manifest.payload.validate().is_err());

        let mut manifest = fixture();
        manifest.payload.anti_rollback_metadata.sequence = 2;
        assert!(manifest.payload.validate().is_err());
    }

    #[test]
    fn signature_envelope_rejects_noncanonical_base64_pad_bits() {
        let mut signature = fixture().signature_envelope;
        assert_eq!(signature.signature.len(), 88);
        signature.signature.replace_range(85..86, "B");
        assert!(signature.validate().is_err());
    }
}
