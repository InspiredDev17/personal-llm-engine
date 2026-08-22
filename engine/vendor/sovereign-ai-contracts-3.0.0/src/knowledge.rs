use crate::common::{
    compare_timestamp_values, validate_extensions, validate_reason_code, validate_semver,
    validate_sha256, validate_timestamp, validate_timestamp_order, Component, ContractVersion,
    Extensions, RedactedReference, RequiredExtensions,
};
use crate::error::{ContractError, Result, Validate};
use crate::identity::{
    AgentId, CagCacheId, KnowledgeInvalidationEventId, KnowledgePlanId, KnowledgeQueryId,
    KnowledgeRevisionId, KnowledgeSourceId, ProjectId, RequestId, StaticKnowledgeBundleId, TaskId,
    UserId,
};
use crate::inference_runtime_v3::RuntimeIdentityV3;
use crate::memory::{Provenance, SensitivityClass};
use crate::qualification::QualificationSignatureEnvelope;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::cmp::Ordering;
use std::collections::BTreeSet;

pub const KNOWLEDGE_CONTRACT_VERSION: &str = "1.0.0";
const BUNDLE_DIGEST_NAMESPACE: &[u8] = b"SOVEREIGN-STATIC-KNOWLEDGE-BUNDLE-V1\0";
const INVALIDATION_DIGEST_NAMESPACE: &[u8] = b"SOVEREIGN-MEMORY-REVISION-INVALIDATED-V1\0";
const INVALIDATION_SIGNATURE_NAMESPACE: &[u8] =
    b"SOVEREIGN-MEMORY-REVISION-INVALIDATED-SIGNATURE-V1\0";
const MAXIMUM_TOKENS: u64 = 2_000_000;
const MAXIMUM_CACHE_BYTES: u64 = 1_099_511_627_776;

fn field(name: &'static str, message: impl Into<String>) -> ContractError {
    ContractError::field(name, message)
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

fn validate_identity(name: &'static str, value: &str) -> Result<()> {
    if valid_identity(value) {
        Ok(())
    } else {
        Err(field(name, "must be a bounded lowercase identity"))
    }
}

fn valid_domain(value: &str) -> bool {
    if value.is_empty() || value.len() > 160 || !value.is_ascii() {
        return false;
    }
    let segments: Vec<&str> = value.split('.').collect();
    !segments.is_empty()
        && segments.len() <= 12
        && segments.iter().all(|segment| {
            !segment.is_empty()
                && segment.len() <= 48
                && segment.as_bytes()[0].is_ascii_lowercase()
                && segment
                    .bytes()
                    .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'_')
        })
}

fn validate_domain(value: &str) -> Result<()> {
    if valid_domain(value) {
        Ok(())
    } else {
        Err(field("domain", "must be a bounded hierarchical path"))
    }
}

fn sensitivity_rank(value: &SensitivityClass) -> u8 {
    match value {
        SensitivityClass::Public => 0,
        SensitivityClass::Internal => 1,
        SensitivityClass::Personal => 2,
        SensitivityClass::Sensitive => 3,
        SensitivityClass::Restricted => 4,
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum KnowledgeMode {
    #[serde(rename = "DIRECT")]
    Direct,
    #[serde(rename = "RAG")]
    Rag,
    #[serde(rename = "CAG")]
    Cag,
    #[serde(rename = "HYBRID_RAG_CAG")]
    HybridRagCag,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum FreshnessClass {
    PinnedRevision,
    CurrentRevision,
    TimeWindow,
    Ephemeral,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CagEligibility {
    Eligible,
    EligibleWithRevisionPin,
    IneligibleDynamic,
    IneligibleSensitive,
    IneligibleTooLarge,
    IneligibleUnverified,
    IneligiblePolicy,
    IneligibleInvalidated,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct KnowledgeScope {
    pub owner_id: UserId,
    pub project_id: ProjectId,
}

impl Validate for KnowledgeScope {
    fn validate(&self) -> Result<()> {
        self.owner_id.validate()?;
        self.project_id.validate()
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct KnowledgeProducerIdentity {
    pub component: Component,
    pub producer_id: String,
}

impl KnowledgeProducerIdentity {
    fn validate_for(&self, expected: &Component) -> Result<()> {
        self.validate()?;
        if &self.component == expected {
            Ok(())
        } else {
            Err(field("producer.component", "unexpected producer component"))
        }
    }
}

impl Validate for KnowledgeProducerIdentity {
    fn validate(&self) -> Result<()> {
        if !matches!(
            self.component,
            Component::PersonalAiPlatform
                | Component::MemoryVault
                | Component::NativeInferenceEngine
        ) {
            return Err(field(
                "producer.component",
                "is not a knowledge-domain producer",
            ));
        }
        validate_identity("producer.producer_id", &self.producer_id)
    }
}

#[derive(Clone, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum KnowledgeClass {
    Static,
    Dynamic,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct KnowledgeRequirement {
    pub domain: String,
    pub knowledge_class: KnowledgeClass,
    pub freshness: FreshnessClass,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub current_after: Option<String>,
    pub required: bool,
    pub minimum_sources: u16,
}

impl Validate for KnowledgeRequirement {
    fn validate(&self) -> Result<()> {
        validate_domain(&self.domain)?;
        if self.minimum_sources > 256 {
            return Err(field("minimum_sources", "exceeds 256"));
        }
        if self.required && self.minimum_sources == 0 {
            return Err(field("minimum_sources", "must be nonzero when required"));
        }
        match (&self.freshness, &self.current_after) {
            (FreshnessClass::TimeWindow, Some(value)) => validate_timestamp("current_after", value),
            (FreshnessClass::TimeWindow, None) => {
                Err(field("current_after", "required for time_window"))
            }
            (_, None) => Ok(()),
            (_, Some(_)) => Err(field("current_after", "allowed only for time_window")),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ContextBudgetAllocation {
    pub maximum_context_tokens: u64,
    pub reserved_output_tokens: u64,
    pub system_security_tokens: u64,
    pub owner_request_tokens: u64,
    pub critical_task_tokens: u64,
    pub static_tokens: u64,
    pub dynamic_tokens: u64,
    pub optional_tokens: u64,
    pub final_input_tokens: u64,
}

impl Validate for ContextBudgetAllocation {
    fn validate(&self) -> Result<()> {
        if self.maximum_context_tokens == 0
            || self.maximum_context_tokens > MAXIMUM_TOKENS
            || self.reserved_output_tokens == 0
            || self.reserved_output_tokens > 131_072
            || self.owner_request_tokens == 0
        {
            return Err(field("budget", "invalid mandatory bound"));
        }
        let total = [
            self.system_security_tokens,
            self.owner_request_tokens,
            self.critical_task_tokens,
            self.static_tokens,
            self.dynamic_tokens,
            self.optional_tokens,
        ]
        .into_iter()
        .try_fold(0_u64, |sum, value| {
            if value > MAXIMUM_TOKENS {
                None
            } else {
                sum.checked_add(value)
                    .filter(|total| *total <= MAXIMUM_TOKENS)
            }
        })
        .ok_or_else(|| field("budget", "token total overflows or exceeds maximum"))?;
        if total != self.final_input_tokens
            || !total
                .checked_add(self.reserved_output_tokens)
                .is_some_and(|value| value <= self.maximum_context_tokens)
        {
            return Err(field("budget", "final input or ceiling is inconsistent"));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ExclusionCategory {
    Excluded,
    Truncated,
    Deduplicated,
    Unavailable,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SourceExclusion {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_id: Option<KnowledgeSourceId>,
    pub reason_code: String,
    pub category: ExclusionCategory,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub estimated_tokens: Option<u64>,
}

impl Validate for SourceExclusion {
    fn validate(&self) -> Result<()> {
        if let Some(source) = &self.source_id {
            source.validate()?;
        }
        validate_reason_code("reason_code", &self.reason_code)?;
        if self
            .estimated_tokens
            .is_some_and(|value| value > MAXIMUM_TOKENS)
        {
            return Err(field("estimated_tokens", "exceeds maximum"));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct StaticBundleReference {
    pub bundle_id: StaticKnowledgeBundleId,
    pub bundle_revision: u64,
    pub bundle_digest: String,
    pub ordered_prefix_digest: String,
}

impl Validate for StaticBundleReference {
    fn validate(&self) -> Result<()> {
        self.bundle_id.validate()?;
        if self.bundle_revision == 0 {
            return Err(field("bundle_revision", "must be nonzero"));
        }
        validate_sha256("bundle_digest", &self.bundle_digest)?;
        validate_sha256("ordered_prefix_digest", &self.ordered_prefix_digest)
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct KnowledgePlanRequest {
    pub contract_version: ContractVersion,
    pub kind: String,
    pub request_id: RequestId,
    pub task_id: TaskId,
    pub agent_id: AgentId,
    pub query_id: KnowledgeQueryId,
    pub query_digest: String,
    pub scope: KnowledgeScope,
    pub sensitivity_ceiling: SensitivityClass,
    pub requirements: Vec<KnowledgeRequirement>,
    pub maximum_context_tokens: u64,
    pub direct_safe_without_knowledge: bool,
    pub requested_at: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extensions: Option<Extensions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub required_extensions: Option<RequiredExtensions>,
}

impl Validate for KnowledgePlanRequest {
    fn validate(&self) -> Result<()> {
        if self.kind != "knowledge.plan_request" {
            return Err(field("kind", "must equal knowledge.plan_request"));
        }
        self.request_id.validate()?;
        self.task_id.validate()?;
        self.agent_id.validate()?;
        self.query_id.validate()?;
        validate_sha256("query_digest", &self.query_digest)?;
        self.scope.validate()?;
        if self.requirements.len() > 64 {
            return Err(field("requirements", "exceeds 64"));
        }
        let mut seen = BTreeSet::new();
        for requirement in &self.requirements {
            requirement.validate()?;
            if !seen.insert((&requirement.domain, &requirement.knowledge_class)) {
                return Err(field("requirements", "duplicate domain/class"));
            }
        }
        if !(1..=MAXIMUM_TOKENS).contains(&self.maximum_context_tokens) {
            return Err(field("maximum_context_tokens", "out of range"));
        }
        validate_timestamp("requested_at", &self.requested_at)?;
        validate_extensions(self.extensions.as_ref(), self.required_extensions.as_ref())
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum KnowledgePlanStatus {
    Resolved,
    Degraded,
    Failed,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CacheCompatibilityOutcome {
    Compatible,
    Missing,
    Incompatible,
    Stale,
    Unauthorized,
    Invalidated,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct KnowledgePlanResult {
    pub contract_version: ContractVersion,
    pub kind: String,
    pub plan_id: KnowledgePlanId,
    pub request_id: RequestId,
    pub query_id: KnowledgeQueryId,
    pub scope: KnowledgeScope,
    pub status: KnowledgePlanStatus,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mode: Option<KnowledgeMode>,
    pub reason_codes: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub static_bundle: Option<StaticBundleReference>,
    pub dynamic_results: Vec<KnowledgeSourceId>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_compatibility: Option<CacheCompatibilityOutcome>,
    pub excluded_sources: Vec<SourceExclusion>,
    pub budget: ContextBudgetAllocation,
    pub degraded: bool,
    pub policy_revision: String,
    pub decided_at: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extensions: Option<Extensions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub required_extensions: Option<RequiredExtensions>,
}

impl Validate for KnowledgePlanResult {
    fn validate(&self) -> Result<()> {
        if self.kind != "knowledge.plan_result" {
            return Err(field("kind", "must equal knowledge.plan_result"));
        }
        self.plan_id.validate()?;
        self.request_id.validate()?;
        self.query_id.validate()?;
        self.scope.validate()?;
        match (&self.status, &self.mode) {
            (KnowledgePlanStatus::Failed, None)
                if self.static_bundle.is_none()
                    && self.dynamic_results.is_empty()
                    && self.cache_compatibility.is_none()
                    && !self.degraded => {}
            (KnowledgePlanStatus::Failed, None) => {
                return Err(field(
                    "status",
                    "failed plan cannot carry usable knowledge or degraded state",
                ))
            }
            (KnowledgePlanStatus::Failed, Some(_)) => {
                return Err(field("mode", "must be absent for failed plan"))
            }
            (_, None) => return Err(field("mode", "required for resolved or degraded plan")),
            _ => {}
        }
        if matches!(self.status, KnowledgePlanStatus::Degraded) != self.degraded {
            return Err(field("degraded", "must match status"));
        }
        if self.reason_codes.is_empty() || self.reason_codes.len() > 16 {
            return Err(field("reason_codes", "must contain 1 through 16"));
        }
        let mut reasons = BTreeSet::new();
        for reason in &self.reason_codes {
            validate_reason_code("reason_code", reason)?;
            if !reasons.insert(reason) {
                return Err(field("reason_codes", "duplicate"));
            }
        }
        if self.dynamic_results.len() > 256 || self.excluded_sources.len() > 256 {
            return Err(field("plan", "collection bound exceeded"));
        }
        let mut dynamic = BTreeSet::new();
        for source in &self.dynamic_results {
            source.validate()?;
            if !dynamic.insert(source) {
                return Err(field("dynamic_results", "duplicate"));
            }
        }
        for exclusion in &self.excluded_sources {
            exclusion.validate()?;
        }
        if let Some(bundle) = &self.static_bundle {
            bundle.validate()?;
        }
        if let Some(mode) = &self.mode {
            match mode {
                KnowledgeMode::Direct
                    if self.static_bundle.is_some()
                        || !self.dynamic_results.is_empty()
                        || self.cache_compatibility.is_some() =>
                {
                    return Err(field("mode", "DIRECT cannot carry durable knowledge"))
                }
                KnowledgeMode::Rag if self.cache_compatibility.is_some() => {
                    return Err(field(
                        "cache_compatibility",
                        "RAG cannot carry CAG cache state",
                    ))
                }
                KnowledgeMode::Rag if self.dynamic_results.is_empty() && !self.degraded => {
                    return Err(field("dynamic_results", "RAG requires a result"));
                }
                KnowledgeMode::Cag
                    if self.static_bundle.is_none() || self.cache_compatibility.is_none() =>
                {
                    return Err(field(
                        "static_bundle",
                        "CAG requires a bundle and cache compatibility",
                    ))
                }
                KnowledgeMode::HybridRagCag
                    if self.static_bundle.is_none()
                        || self.cache_compatibility.is_none()
                        || (self.dynamic_results.is_empty() && !self.degraded) =>
                {
                    return Err(field(
                        "mode",
                        "hybrid requires bundle, cache state, and dynamic knowledge",
                    ))
                }
                _ => {}
            }
        }
        self.budget.validate()?;
        validate_identity("policy_revision", &self.policy_revision)?;
        validate_timestamp("decided_at", &self.decided_at)?;
        validate_extensions(self.extensions.as_ref(), self.required_extensions.as_ref())
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct StaticKnowledgeBundleMember {
    pub position: u16,
    pub source_id: KnowledgeSourceId,
    pub revision_id: KnowledgeRevisionId,
    pub content_digest: String,
    pub estimated_token_count: u64,
}

impl Validate for StaticKnowledgeBundleMember {
    fn validate(&self) -> Result<()> {
        if self.position > 255 || !(1..=MAXIMUM_TOKENS).contains(&self.estimated_token_count) {
            return Err(field("member", "position or token count out of range"));
        }
        self.source_id.validate()?;
        self.revision_id.validate()?;
        validate_sha256("content_digest", &self.content_digest)
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum BundleInvalidationState {
    Active,
    Invalidated,
    Deleted,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct StaticKnowledgeBundle {
    pub contract_version: ContractVersion,
    pub kind: String,
    pub bundle_id: StaticKnowledgeBundleId,
    pub bundle_revision: u64,
    pub ordered_members: Vec<StaticKnowledgeBundleMember>,
    pub ordered_prefix_digest: String,
    pub bundle_digest: String,
    pub scope: KnowledgeScope,
    pub sensitivity: SensitivityClass,
    pub producer: KnowledgeProducerIdentity,
    pub estimated_token_count: u64,
    pub cag_eligibility: CagEligibility,
    pub created_at: String,
    pub verified_at: String,
    pub invalidation_state: BundleInvalidationState,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub invalidated_at: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub invalidation_reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extensions: Option<Extensions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub required_extensions: Option<RequiredExtensions>,
}

impl StaticKnowledgeBundle {
    fn validate_without_digest(&self) -> Result<()> {
        if self.kind != "knowledge.static_bundle" {
            return Err(field("kind", "must equal knowledge.static_bundle"));
        }
        self.bundle_id.validate()?;
        if self.bundle_revision == 0
            || self.ordered_members.is_empty()
            || self.ordered_members.len() > 256
        {
            return Err(field("bundle", "invalid revision or member count"));
        }
        let mut tokens = 0_u64;
        let mut seen = BTreeSet::new();
        for (position, member) in self.ordered_members.iter().enumerate() {
            member.validate()?;
            if usize::from(member.position) != position {
                return Err(field("ordered_members.position", "must equal array index"));
            }
            if !seen.insert((&member.source_id, &member.revision_id)) {
                return Err(field("ordered_members", "duplicate source revision"));
            }
            tokens = tokens
                .checked_add(member.estimated_token_count)
                .filter(|value| *value <= MAXIMUM_TOKENS)
                .ok_or_else(|| field("estimated_token_count", "overflows"))?;
        }
        if tokens != self.estimated_token_count {
            return Err(field("estimated_token_count", "must equal member total"));
        }
        validate_sha256("ordered_prefix_digest", &self.ordered_prefix_digest)?;
        self.scope.validate()?;
        self.producer.validate_for(&Component::MemoryVault)?;
        validate_timestamp("created_at", &self.created_at)?;
        validate_timestamp_order(
            "created_at",
            &self.created_at,
            "verified_at",
            &self.verified_at,
        )?;
        match self.invalidation_state {
            BundleInvalidationState::Active => {
                if self.invalidated_at.is_some()
                    || self.invalidation_reason.is_some()
                    || self.cag_eligibility == CagEligibility::IneligibleInvalidated
                {
                    return Err(field(
                        "invalidation_state",
                        "active bundle has invalidation metadata",
                    ));
                }
            }
            BundleInvalidationState::Invalidated | BundleInvalidationState::Deleted => {
                let invalidated = self
                    .invalidated_at
                    .as_ref()
                    .ok_or_else(|| field("invalidated_at", "required"))?;
                let reason = self
                    .invalidation_reason
                    .as_ref()
                    .ok_or_else(|| field("invalidation_reason", "required"))?;
                if self.cag_eligibility != CagEligibility::IneligibleInvalidated {
                    return Err(field(
                        "cag_eligibility",
                        "must fail closed after invalidation",
                    ));
                }
                validate_timestamp_order(
                    "verified_at",
                    &self.verified_at,
                    "invalidated_at",
                    invalidated,
                )?;
                validate_reason_code("invalidation_reason", reason)?;
            }
        }
        validate_extensions(self.extensions.as_ref(), self.required_extensions.as_ref())
    }
}

impl Validate for StaticKnowledgeBundle {
    fn validate(&self) -> Result<()> {
        self.validate_without_digest()?;
        if canonical_static_knowledge_bundle_digest(self)? != self.bundle_digest {
            return Err(field("bundle_digest", "does not match canonical bundle"));
        }
        Ok(())
    }
}

/// Computes the namespaced canonical digest for one validated bundle.
///
/// # Errors
///
/// Returns an error when the bundle is invalid or cannot be canonically encoded.
pub fn canonical_static_knowledge_bundle_digest(bundle: &StaticKnowledgeBundle) -> Result<String> {
    bundle.validate_without_digest()?;
    let mut value = serde_json::to_value(bundle)?;
    value
        .as_object_mut()
        .ok_or_else(|| field("bundle", "must encode as object"))?
        .remove("bundle_digest");
    let canonical = serde_json::to_vec(&crate::document::sort_value(value))?;
    let mut hasher = Sha256::new();
    hasher.update(BUNDLE_DIGEST_NAMESPACE);
    hasher.update(canonical);
    Ok(format!("{:x}", hasher.finalize()))
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RetrievalOutcome {
    Included,
    Excluded,
    Truncated,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DynamicRetrievalResult {
    pub contract_version: ContractVersion,
    pub kind: String,
    pub query_id: KnowledgeQueryId,
    pub source_id: KnowledgeSourceId,
    pub revision_id: KnowledgeRevisionId,
    pub content_digest: String,
    pub retrieved_at: String,
    pub freshness: FreshnessClass,
    pub sensitivity: SensitivityClass,
    pub provenance: Provenance,
    pub score_millionths: u32,
    pub ranking_position: u16,
    pub estimated_token_count: u64,
    pub outcome: RetrievalOutcome,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason_code: Option<String>,
    pub domain: String,
    pub scope: KnowledgeScope,
    pub producer: KnowledgeProducerIdentity,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extensions: Option<Extensions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub required_extensions: Option<RequiredExtensions>,
}

impl Validate for DynamicRetrievalResult {
    fn validate(&self) -> Result<()> {
        if self.kind != "knowledge.dynamic_retrieval_result" {
            return Err(field(
                "kind",
                "must equal knowledge.dynamic_retrieval_result",
            ));
        }
        self.query_id.validate()?;
        self.source_id.validate()?;
        self.revision_id.validate()?;
        validate_sha256("content_digest", &self.content_digest)?;
        validate_timestamp("retrieved_at", &self.retrieved_at)?;
        self.provenance.validate()?;
        if self.score_millionths > 1_000_000
            || self.ranking_position == 0
            || self.ranking_position > 256
            || !(1..=MAXIMUM_TOKENS).contains(&self.estimated_token_count)
        {
            return Err(field(
                "retrieval_result",
                "score, rank, or tokens out of range",
            ));
        }
        match (&self.outcome, &self.reason_code) {
            (RetrievalOutcome::Included, None) => {}
            (RetrievalOutcome::Included, Some(_)) => {
                return Err(field("reason_code", "must be absent when included"))
            }
            (_, Some(reason)) => validate_reason_code("reason_code", reason)?,
            (_, None) => return Err(field("reason_code", "required when excluded or truncated")),
        }
        validate_domain(&self.domain)?;
        self.scope.validate()?;
        self.producer.validate_for(&Component::MemoryVault)?;
        validate_extensions(self.extensions.as_ref(), self.required_extensions.as_ref())
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct MemoryRevisionInvalidationPayload {
    pub event_id: KnowledgeInvalidationEventId,
    pub producer: KnowledgeProducerIdentity,
    pub sequence: u64,
    pub issued_at: String,
    pub source_id: KnowledgeSourceId,
    pub old_revision_id: KnowledgeRevisionId,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub replacement_revision_id: Option<KnowledgeRevisionId>,
    pub reason_code: String,
    pub deleted: bool,
    pub affected_bundles: Vec<StaticKnowledgeBundleId>,
    pub scope: KnowledgeScope,
    pub sensitivity: SensitivityClass,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extensions: Option<Extensions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub required_extensions: Option<RequiredExtensions>,
}

impl Validate for MemoryRevisionInvalidationPayload {
    fn validate(&self) -> Result<()> {
        self.event_id.validate()?;
        self.producer.validate_for(&Component::MemoryVault)?;
        if self.sequence == 0 {
            return Err(field("sequence", "must be nonzero"));
        }
        validate_timestamp("issued_at", &self.issued_at)?;
        self.source_id.validate()?;
        self.old_revision_id.validate()?;
        match (self.deleted, &self.replacement_revision_id) {
            (true, None) => {}
            (true, Some(_)) => {
                return Err(field("replacement_revision_id", "forbidden for deletion"))
            }
            (false, Some(replacement)) => {
                replacement.validate()?;
                if replacement == &self.old_revision_id {
                    return Err(field(
                        "replacement_revision_id",
                        "must differ from old revision",
                    ));
                }
            }
            (false, None) => return Err(field("replacement_revision_id", "required")),
        }
        validate_reason_code("reason_code", &self.reason_code)?;
        if self.affected_bundles.len() > 256 {
            return Err(field("affected_bundles", "exceeds 256"));
        }
        let mut prior: Option<&StaticKnowledgeBundleId> = None;
        for bundle in &self.affected_bundles {
            bundle.validate()?;
            if prior.is_some_and(|value| value >= bundle) {
                return Err(field("affected_bundles", "must be sorted and unique"));
            }
            prior = Some(bundle);
        }
        self.scope.validate()?;
        validate_extensions(self.extensions.as_ref(), self.required_extensions.as_ref())
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct MemoryRevisionInvalidated {
    pub contract_version: ContractVersion,
    pub kind: String,
    pub payload: MemoryRevisionInvalidationPayload,
    pub canonical_digest: String,
    pub signature_envelope: QualificationSignatureEnvelope,
}

impl Validate for MemoryRevisionInvalidated {
    fn validate(&self) -> Result<()> {
        if self.kind != "knowledge.memory_revision_invalidated" {
            return Err(field(
                "kind",
                "must equal knowledge.memory_revision_invalidated",
            ));
        }
        self.payload.validate()?;
        if canonical_memory_revision_invalidation_digest(&self.payload)? != self.canonical_digest {
            return Err(field(
                "canonical_digest",
                "does not match canonical payload",
            ));
        }
        self.signature_envelope.validate()
    }
}

/// Computes the namespaced canonical digest for one invalidation payload.
///
/// # Errors
///
/// Returns an error when the payload is invalid or cannot be canonically encoded.
pub fn canonical_memory_revision_invalidation_digest(
    payload: &MemoryRevisionInvalidationPayload,
) -> Result<String> {
    payload.validate()?;
    let value = serde_json::to_value(payload)?;
    let canonical = serde_json::to_vec(&crate::document::sort_value(value))?;
    let mut hasher = Sha256::new();
    hasher.update(INVALIDATION_DIGEST_NAMESPACE);
    hasher.update(canonical);
    Ok(format!("{:x}", hasher.finalize()))
}

/// Builds the domain-separated signature transcript for an invalidation digest.
///
/// # Errors
///
/// Returns an error unless `digest` is canonical lowercase SHA-256 hex.
pub fn memory_revision_invalidation_signing_transcript(digest: &str) -> Result<Vec<u8>> {
    validate_sha256("canonical_digest", digest)?;
    let mut transcript = INVALIDATION_SIGNATURE_NAMESPACE.to_vec();
    for pair in digest.as_bytes().chunks_exact(2) {
        let high = (pair[0] as char)
            .to_digit(16)
            .ok_or_else(|| field("canonical_digest", "invalid hex"))?;
        let low = (pair[1] as char)
            .to_digit(16)
            .ok_or_else(|| field("canonical_digest", "invalid hex"))?;
        transcript.push(
            u8::try_from((high << 4) | low)
                .map_err(|_| field("canonical_digest", "invalid hex byte"))?,
        );
    }
    Ok(transcript)
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum InvalidationDisposition {
    Next,
    Duplicate,
}

/// Classifies an exact duplicate or accepts the next event in one producer stream.
///
/// # Errors
///
/// Returns an error for invalid events, producer substitution, conflicts, gaps,
/// rollback, or non-increasing timestamps.
pub fn validate_memory_revision_invalidation_order(
    previous: &MemoryRevisionInvalidated,
    current: &MemoryRevisionInvalidated,
) -> Result<InvalidationDisposition> {
    previous.validate()?;
    current.validate()?;
    if previous.payload.producer != current.payload.producer {
        return Err(field("producer", "invalidation stream substitution"));
    }
    if current.payload.sequence == previous.payload.sequence {
        return if current.payload.event_id == previous.payload.event_id
            && current.canonical_digest == previous.canonical_digest
        {
            Ok(InvalidationDisposition::Duplicate)
        } else {
            Err(field("sequence", "conflicting replay"))
        };
    }
    if current.payload.sequence != previous.payload.sequence + 1 {
        return Err(field("sequence", "rollback or gap"));
    }
    if compare_timestamp_values(&previous.payload.issued_at, &current.payload.issued_at)?
        != Ordering::Less
    {
        return Err(field("issued_at", "must increase with sequence"));
    }
    Ok(InvalidationDisposition::Next)
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CacheImplementation {
    LocalNativeKv,
    ExternalPromptCache,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CacheBinding {
    pub runtime_identity: RuntimeIdentityV3,
    pub tokenizer_id: String,
    pub tokenizer_digest: String,
    pub architecture_id: String,
    pub cache_format_version: String,
    pub bundle: StaticBundleReference,
    pub token_count: u64,
    pub rope_semantics_id: String,
    pub scope: KnowledgeScope,
    pub sensitivity: SensitivityClass,
    pub cache_implementation: CacheImplementation,
}

impl Validate for CacheBinding {
    fn validate(&self) -> Result<()> {
        self.runtime_identity.validate()?;
        if self.runtime_identity.provider_model_id == self.runtime_identity.model_id.as_str() {
            return Err(field(
                "provider_model_id",
                "must not substitute logical model identity",
            ));
        }
        validate_identity("tokenizer_id", &self.tokenizer_id)?;
        validate_sha256("tokenizer_digest", &self.tokenizer_digest)?;
        validate_identity("architecture_id", &self.architecture_id)?;
        validate_semver("cache_format_version", &self.cache_format_version)?;
        self.bundle.validate()?;
        if !(1..=MAXIMUM_TOKENS).contains(&self.token_count) {
            return Err(field("token_count", "out of range"));
        }
        validate_identity("rope_semantics_id", &self.rope_semantics_id)?;
        self.scope.validate()?;
        if self.cache_implementation == CacheImplementation::ExternalPromptCache
            && sensitivity_rank(&self.sensitivity) > 1
        {
            return Err(field("sensitivity", "sensitive cache cannot be external"));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CacheStatus {
    Building,
    Ready,
    Invalidated,
    Expired,
    Failed,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CacheResidency {
    Resident,
    Available,
    Absent,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CagCacheDescriptor {
    pub contract_version: ContractVersion,
    pub kind: String,
    pub cache_id: CagCacheId,
    pub producer: KnowledgeProducerIdentity,
    pub binding: CacheBinding,
    pub status: CacheStatus,
    pub residency: CacheResidency,
    pub created_at: String,
    pub verified_at: String,
    pub expires_at: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason_code: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extensions: Option<Extensions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub required_extensions: Option<RequiredExtensions>,
}

impl Validate for CagCacheDescriptor {
    fn validate(&self) -> Result<()> {
        if self.kind != "knowledge.cag_cache_descriptor" {
            return Err(field("kind", "must equal knowledge.cag_cache_descriptor"));
        }
        self.cache_id.validate()?;
        self.producer
            .validate_for(&Component::NativeInferenceEngine)?;
        self.binding.validate()?;
        if self.producer.producer_id != self.binding.runtime_identity.engine_instance_id {
            return Err(field("producer_id", "must equal bound engine instance"));
        }
        validate_timestamp("created_at", &self.created_at)?;
        validate_timestamp_order(
            "created_at",
            &self.created_at,
            "verified_at",
            &self.verified_at,
        )?;
        validate_timestamp_order(
            "verified_at",
            &self.verified_at,
            "expires_at",
            &self.expires_at,
        )?;
        match (&self.status, &self.residency, &self.reason_code) {
            (CacheStatus::Ready, CacheResidency::Resident | CacheResidency::Available, None)
            | (CacheStatus::Building, _, None) => {}
            (
                CacheStatus::Invalidated | CacheStatus::Expired | CacheStatus::Failed,
                CacheResidency::Absent,
                Some(reason),
            ) => validate_reason_code("reason_code", reason)?,
            _ => {
                return Err(field(
                    "cache_state",
                    "incompatible status, residency, or reason",
                ))
            }
        }
        validate_extensions(self.extensions.as_ref(), self.required_extensions.as_ref())
    }
}

impl CagCacheDescriptor {
    /// Proves exact binding to an active bundle at one instant. This grants no
    /// cache access or execution authority.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid, mismatched, inactive, or expired inputs.
    pub fn validate_usable_against_bundle(
        &self,
        bundle: &StaticKnowledgeBundle,
        now: &str,
    ) -> Result<()> {
        self.validate()?;
        bundle.validate()?;
        validate_timestamp("now", now)?;
        let reference = StaticBundleReference {
            bundle_id: bundle.bundle_id.clone(),
            bundle_revision: bundle.bundle_revision,
            bundle_digest: bundle.bundle_digest.clone(),
            ordered_prefix_digest: bundle.ordered_prefix_digest.clone(),
        };
        if self.binding.bundle != reference
            || self.binding.scope != bundle.scope
            || sensitivity_rank(&self.binding.sensitivity) < sensitivity_rank(&bundle.sensitivity)
        {
            return Err(field(
                "cache_binding",
                "bundle identity, scope, or sensitivity mismatch",
            ));
        }
        if bundle.invalidation_state != BundleInvalidationState::Active
            || self.status != CacheStatus::Ready
            || compare_timestamp_values(now, &self.expires_at)? != Ordering::Less
        {
            return Err(field("cache_status", "cache is not usable"));
        }
        Ok(())
    }

    /// Rejects a cache named by a valid Memory Vault invalidation event.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid inputs or an affected cache.
    pub fn validate_unaffected_by_invalidation(
        &self,
        event: &MemoryRevisionInvalidated,
    ) -> Result<()> {
        self.validate()?;
        event.validate()?;
        if self.binding.scope == event.payload.scope
            && event
                .payload
                .affected_bundles
                .contains(&self.binding.bundle.bundle_id)
        {
            return Err(field(
                "cache_status",
                "cache is affected by revision invalidation",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CacheCompatibility {
    pub contract_version: ContractVersion,
    pub kind: String,
    pub request_id: RequestId,
    pub expected: CacheBinding,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_id: Option<CagCacheId>,
    pub outcome: CacheCompatibilityOutcome,
    pub mismatch_reasons: Vec<String>,
    pub evaluated_at: String,
}

impl Validate for CacheCompatibility {
    fn validate(&self) -> Result<()> {
        if self.kind != "knowledge.cache_compatibility" {
            return Err(field("kind", "must equal knowledge.cache_compatibility"));
        }
        self.request_id.validate()?;
        self.expected.validate()?;
        if self.mismatch_reasons.len() > 16 {
            return Err(field("mismatch_reasons", "exceeds 16"));
        }
        let mut seen = BTreeSet::new();
        for reason in &self.mismatch_reasons {
            validate_reason_code("mismatch_reason", reason)?;
            if !seen.insert(reason) {
                return Err(field("mismatch_reasons", "duplicate"));
            }
        }
        match (
            &self.outcome,
            &self.cache_id,
            self.mismatch_reasons.is_empty(),
        ) {
            (CacheCompatibilityOutcome::Compatible, Some(cache), true) => cache.validate()?,
            (CacheCompatibilityOutcome::Missing, None, _) => {}
            (CacheCompatibilityOutcome::Compatible | CacheCompatibilityOutcome::Missing, _, _) => {
                return Err(field(
                    "compatibility",
                    "invalid compatible or missing shape",
                ))
            }
            (_, Some(cache), false) => cache.validate()?,
            _ => return Err(field("compatibility", "mismatch requires cache and reason")),
        }
        validate_timestamp("evaluated_at", &self.evaluated_at)
    }
}

impl CacheCompatibility {
    /// Checks a claimed compatibility result against one exact descriptor.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid, expired, substituted, or misclassified input.
    pub fn validate_against(&self, descriptor: &CagCacheDescriptor, now: &str) -> Result<()> {
        self.validate()?;
        descriptor.validate()?;
        validate_timestamp("now", now)?;
        if self.cache_id.as_ref() != Some(&descriptor.cache_id) {
            return Err(field("cache_id", "does not bind descriptor"));
        }
        let compatible = self.expected == descriptor.binding
            && descriptor.status == CacheStatus::Ready
            && compare_timestamp_values(now, &descriptor.expires_at)? == Ordering::Less;
        if matches!(self.outcome, CacheCompatibilityOutcome::Compatible) != compatible {
            return Err(field(
                "outcome",
                "does not reflect exact descriptor compatibility",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CagCacheQueryRequest {
    pub contract_version: ContractVersion,
    pub kind: String,
    pub request_id: RequestId,
    pub engine_instance_id: String,
    pub scope: KnowledgeScope,
    pub expected: CacheBinding,
    pub maximum_results: u16,
    pub requested_at: String,
}

impl Validate for CagCacheQueryRequest {
    fn validate(&self) -> Result<()> {
        if self.kind != "knowledge.cag_cache_query_request" {
            return Err(field(
                "kind",
                "must equal knowledge.cag_cache_query_request",
            ));
        }
        self.request_id.validate()?;
        validate_identity("engine_instance_id", &self.engine_instance_id)?;
        self.scope.validate()?;
        self.expected.validate()?;
        if self.scope != self.expected.scope
            || self.engine_instance_id != self.expected.runtime_identity.engine_instance_id
            || !(1..=64).contains(&self.maximum_results)
        {
            return Err(field("query", "scope, engine, or bound mismatch"));
        }
        validate_timestamp("requested_at", &self.requested_at)
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CagCacheQueryResult {
    pub contract_version: ContractVersion,
    pub kind: String,
    pub request_id: RequestId,
    pub producer: KnowledgeProducerIdentity,
    pub engine_instance_id: String,
    pub scope: KnowledgeScope,
    pub caches: Vec<CagCacheDescriptor>,
    pub observed_at: String,
    pub expires_at: String,
}

impl Validate for CagCacheQueryResult {
    fn validate(&self) -> Result<()> {
        if self.kind != "knowledge.cag_cache_query_result" {
            return Err(field("kind", "must equal knowledge.cag_cache_query_result"));
        }
        self.request_id.validate()?;
        self.producer
            .validate_for(&Component::NativeInferenceEngine)?;
        validate_identity("engine_instance_id", &self.engine_instance_id)?;
        if self.producer.producer_id != self.engine_instance_id || self.caches.len() > 64 {
            return Err(field("query_result", "producer or collection mismatch"));
        }
        self.scope.validate()?;
        validate_timestamp("observed_at", &self.observed_at)?;
        let mut seen = BTreeSet::new();
        for cache in &self.caches {
            cache.validate()?;
            if cache.binding.scope != self.scope
                || cache.binding.runtime_identity.engine_instance_id != self.engine_instance_id
                || !seen.insert(&cache.cache_id)
            {
                return Err(field(
                    "caches",
                    "cross-scope, cross-engine, or duplicate cache",
                ));
            }
            if cache.status == CacheStatus::Ready
                && compare_timestamp_values(&self.observed_at, &cache.expires_at)? != Ordering::Less
            {
                return Err(field(
                    "caches",
                    "ready cache is expired at observation time",
                ));
            }
        }
        validate_timestamp_order(
            "observed_at",
            &self.observed_at,
            "expires_at",
            &self.expires_at,
        )
    }
}

impl CagCacheQueryResult {
    /// Checks this result against its originating bounded query.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid input or correlation, engine, scope, or bound mismatch.
    pub fn validate_against(&self, request: &CagCacheQueryRequest) -> Result<()> {
        self.validate()?;
        request.validate()?;
        if self.request_id != request.request_id
            || self.engine_instance_id != request.engine_instance_id
            || self.scope != request.scope
            || self.caches.len() > usize::from(request.maximum_results)
        {
            return Err(field("query_result", "request binding mismatch"));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CagCacheBuildRequest {
    pub contract_version: ContractVersion,
    pub kind: String,
    pub request_id: RequestId,
    pub binding: CacheBinding,
    pub prefix_reference: RedactedReference,
    pub maximum_cache_bytes: u64,
    pub idempotency_key: String,
    pub requested_at: String,
    pub expires_at: String,
}

impl Validate for CagCacheBuildRequest {
    fn validate(&self) -> Result<()> {
        if self.kind != "knowledge.cag_cache_build_request" {
            return Err(field(
                "kind",
                "must equal knowledge.cag_cache_build_request",
            ));
        }
        self.request_id.validate()?;
        self.binding.validate()?;
        if self.binding.cache_implementation != CacheImplementation::LocalNativeKv {
            return Err(field(
                "cache_implementation",
                "build request is native-only",
            ));
        }
        self.prefix_reference.validate()?;
        if self.prefix_reference.digest != self.binding.bundle.ordered_prefix_digest {
            return Err(field("prefix_reference", "must bind ordered prefix"));
        }
        if !(1..=MAXIMUM_CACHE_BYTES).contains(&self.maximum_cache_bytes) {
            return Err(field("maximum_cache_bytes", "out of range"));
        }
        validate_sha256("idempotency_key", &self.idempotency_key)?;
        validate_timestamp("requested_at", &self.requested_at)?;
        validate_timestamp_order(
            "requested_at",
            &self.requested_at,
            "expires_at",
            &self.expires_at,
        )
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CacheBuildStatus {
    Built,
    Reused,
    Rejected,
    Failed,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CagCacheBuildResult {
    pub contract_version: ContractVersion,
    pub kind: String,
    pub request_id: RequestId,
    pub producer: KnowledgeProducerIdentity,
    pub status: CacheBuildStatus,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache: Option<CagCacheDescriptor>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason_code: Option<String>,
    pub completed_at: String,
}

impl Validate for CagCacheBuildResult {
    fn validate(&self) -> Result<()> {
        if self.kind != "knowledge.cag_cache_build_result" {
            return Err(field("kind", "must equal knowledge.cag_cache_build_result"));
        }
        self.request_id.validate()?;
        self.producer
            .validate_for(&Component::NativeInferenceEngine)?;
        match (&self.status, &self.cache, &self.reason_code) {
            (CacheBuildStatus::Built | CacheBuildStatus::Reused, Some(cache), None) => {
                cache.validate()?;
                if cache.producer != self.producer {
                    return Err(field("cache.producer", "producer mismatch"));
                }
            }
            (CacheBuildStatus::Rejected | CacheBuildStatus::Failed, None, Some(reason)) => {
                validate_reason_code("reason_code", reason)?;
            }
            _ => return Err(field("build_result", "incompatible status payload")),
        }
        validate_timestamp("completed_at", &self.completed_at)
    }
}

impl CagCacheBuildResult {
    /// Checks this result against its originating native cache build request.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid input, substitution, escalation, or bad timing.
    pub fn validate_against(&self, request: &CagCacheBuildRequest) -> Result<()> {
        self.validate()?;
        request.validate()?;
        if self.request_id != request.request_id {
            return Err(field("request_id", "mismatch"));
        }
        if let Some(cache) = &self.cache {
            if cache.binding != request.binding
                || cache.producer.producer_id != request.binding.runtime_identity.engine_instance_id
            {
                return Err(field("cache", "identity or compatibility substitution"));
            }
        }
        validate_timestamp_order(
            "requested_at",
            &request.requested_at,
            "completed_at",
            &self.completed_at,
        )
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CagCacheInvalidationRequest {
    pub contract_version: ContractVersion,
    pub kind: String,
    pub request_id: RequestId,
    pub cache_id: CagCacheId,
    pub engine_instance_id: String,
    pub reason_code: String,
    pub idempotency_key: String,
    pub requested_at: String,
}

impl Validate for CagCacheInvalidationRequest {
    fn validate(&self) -> Result<()> {
        if self.kind != "knowledge.cag_cache_invalidation_request" {
            return Err(field(
                "kind",
                "must equal knowledge.cag_cache_invalidation_request",
            ));
        }
        self.request_id.validate()?;
        self.cache_id.validate()?;
        validate_identity("engine_instance_id", &self.engine_instance_id)?;
        validate_reason_code("reason_code", &self.reason_code)?;
        validate_sha256("idempotency_key", &self.idempotency_key)?;
        validate_timestamp("requested_at", &self.requested_at)
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CacheInvalidationStatus {
    Invalidated,
    AlreadyInvalidated,
    NotFound,
    Rejected,
    Failed,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CagCacheInvalidationResult {
    pub contract_version: ContractVersion,
    pub kind: String,
    pub request_id: RequestId,
    pub cache_id: CagCacheId,
    pub producer: KnowledgeProducerIdentity,
    pub status: CacheInvalidationStatus,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason_code: Option<String>,
    pub completed_at: String,
}

impl Validate for CagCacheInvalidationResult {
    fn validate(&self) -> Result<()> {
        if self.kind != "knowledge.cag_cache_invalidation_result" {
            return Err(field(
                "kind",
                "must equal knowledge.cag_cache_invalidation_result",
            ));
        }
        self.request_id.validate()?;
        self.cache_id.validate()?;
        self.producer
            .validate_for(&Component::NativeInferenceEngine)?;
        if matches!(
            self.status,
            CacheInvalidationStatus::Rejected | CacheInvalidationStatus::Failed
        ) && self.reason_code.is_none()
        {
            return Err(field("reason_code", "required for rejection or failure"));
        }
        if let Some(reason) = &self.reason_code {
            validate_reason_code("reason_code", reason)?;
        }
        validate_timestamp("completed_at", &self.completed_at)
    }
}

impl CagCacheInvalidationResult {
    /// Checks this result against its originating cache invalidation request.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid input, substitution, or bad timing.
    pub fn validate_against(&self, request: &CagCacheInvalidationRequest) -> Result<()> {
        self.validate()?;
        request.validate()?;
        if self.request_id != request.request_id
            || self.cache_id != request.cache_id
            || self.producer.producer_id != request.engine_instance_id
        {
            return Err(field(
                "invalidation_result",
                "request, cache, or engine mismatch",
            ));
        }
        validate_timestamp_order(
            "requested_at",
            &request.requested_at,
            "completed_at",
            &self.completed_at,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::{
        memory_revision_invalidation_signing_transcript,
        validate_memory_revision_invalidation_order, CagCacheDescriptor, InvalidationDisposition,
        MemoryRevisionInvalidated,
    };
    use crate::error::Validate;

    fn fixture<T: serde::de::DeserializeOwned + Validate>(name: &str) -> T {
        let path = format!("../fixtures/v1/valid/{name}");
        let data = std::fs::read(path).expect("read fixture");
        let value: T = serde_json::from_slice(&data).expect("decode fixture");
        value.validate().expect("valid fixture");
        value
    }

    #[test]
    fn golden_digest_transcript_and_duplicate_are_stable() {
        let descriptor: CagCacheDescriptor = fixture("knowledge-cache-descriptor.json");
        assert_eq!(
            descriptor.binding.bundle.bundle_digest,
            "f9f149216adc81b8845d676caa1ebb68f4036119e3074c371feb8e532dd43681"
        );
        let event: MemoryRevisionInvalidated = fixture("knowledge-revision-invalidated.json");
        assert_eq!(
            event.canonical_digest,
            "00b9190c9c99b5180cedfca54b2f011340d299cfb32f0bcc556ca34dca909d85"
        );
        let transcript = memory_revision_invalidation_signing_transcript(&event.canonical_digest)
            .expect("transcript");
        let mut expected = b"SOVEREIGN-MEMORY-REVISION-INVALIDATED-SIGNATURE-V1\0".to_vec();
        for pair in event.canonical_digest.as_bytes().chunks_exact(2) {
            let pair = std::str::from_utf8(pair).expect("ASCII digest");
            expected.push(u8::from_str_radix(pair, 16).expect("hex digest"));
        }
        assert_eq!(transcript, expected);
        assert_eq!(
            validate_memory_revision_invalidation_order(&event, &event).expect("duplicate"),
            InvalidationDisposition::Duplicate
        );
    }
}
