//! Experimental Knowledge 1.3 revocation-complete invalidation feed contracts.

use crate::common::{
    compare_timestamp_values, timestamp_unix_millis, validate_extensions, validate_reason_code,
    validate_sha256, validate_timestamp, validate_timestamp_order, Component, Extensions,
    RequiredExtensions,
};
use crate::error::{ContractError, Result, Validate};
use crate::identity::{
    KnowledgeInvalidationEventId, KnowledgeRevisionId, KnowledgeSourceId, RequestId, UserId,
};
use crate::knowledge::{KnowledgeProducerIdentity, StaticBundleReference};
use crate::knowledge_lifecycle::{CacheAuthorizationScope, CacheNamespace};
use crate::memory::SensitivityClass;
use crate::qualification::QualificationSignatureEnvelope;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::cmp::Ordering;
use std::collections::HashSet;

pub const KNOWLEDGE_INVALIDATION_FEED_CONTRACT_VERSION: &str = "1.3.0";

const MAXIMUM_SEQUENCE: u64 = 9_007_199_254_740_991;
const MAXIMUM_RESULTS: usize = 256;
const MAXIMUM_WINDOW_MILLIS: u64 = 5 * 60 * 1000;
const TOMBSTONE_DIGEST_NAMESPACE: &[u8] = b"SOVEREIGN-KNOWLEDGE-INVALIDATION-TOMBSTONE-V1\0";
const TOMBSTONE_SIGNATURE_NAMESPACE: &[u8] =
    b"SOVEREIGN-KNOWLEDGE-INVALIDATION-TOMBSTONE-SIGNATURE-V1\0";
const FEED_RESULT_DIGEST_NAMESPACE: &[u8] = b"SOVEREIGN-KNOWLEDGE-INVALIDATION-FEED-RESULT-V1\0";
const FEED_RESULT_SIGNATURE_NAMESPACE: &[u8] =
    b"SOVEREIGN-KNOWLEDGE-INVALIDATION-FEED-RESULT-SIGNATURE-V1\0";

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

fn sensitivity_rank(value: &SensitivityClass) -> u8 {
    match value {
        SensitivityClass::Public => 0,
        SensitivityClass::Internal => 1,
        SensitivityClass::Personal => 2,
        SensitivityClass::Sensitive => 3,
        SensitivityClass::Restricted => 4,
    }
}

fn validate_window(start: &str, end: &str) -> Result<()> {
    validate_timestamp("start", start)?;
    validate_timestamp_order("start", start, "expires_at", end)?;
    let start = timestamp_unix_millis("start", start)?;
    let end = timestamp_unix_millis("expires_at", end)?;
    if end - start > MAXIMUM_WINDOW_MILLIS {
        return Err(field("expires_at", "freshness window exceeds five minutes"));
    }
    Ok(())
}

fn validate_active(label: &'static str, start: &str, end: &str, now: &str) -> Result<()> {
    let start = timestamp_unix_millis("start", start)?;
    let end = timestamp_unix_millis("expires_at", end)?;
    let now = timestamp_unix_millis("now", now)?;
    if now < start {
        return Err(field(label, "is not active yet"));
    }
    if now >= end {
        return Err(field(label, "is expired"));
    }
    Ok(())
}

fn timestamp_after(left: &str, right: &str) -> Result<bool> {
    Ok(compare_timestamp_values(left, right)? == Ordering::Greater)
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(transparent)]
pub struct KnowledgeInvalidationAuthorizationViewId(String);

impl KnowledgeInvalidationAuthorizationViewId {
    /// Constructs one opaque, server-derived authorization-view identifier.
    ///
    /// # Errors
    ///
    /// Returns an error for a wrong prefix, invalid character, or length.
    pub fn new(value: impl Into<String>) -> Result<Self> {
        let value = value.into();
        let candidate = Self(value);
        candidate.validate()?;
        Ok(candidate)
    }

    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl Validate for KnowledgeInvalidationAuthorizationViewId {
    fn validate(&self) -> Result<()> {
        let Some(body) = self.0.strip_prefix("kav_") else {
            return Err(field("authorization_view_id", "must start with kav_"));
        };
        if !(2..=64).contains(&body.len())
            || !body
                .bytes()
                .next()
                .is_some_and(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit())
            || !body.bytes().all(|byte| {
                byte.is_ascii_lowercase() || byte.is_ascii_digit() || matches!(byte, b'_' | b'-')
            })
        {
            return Err(field(
                "authorization_view_id",
                "has invalid length or characters",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum KnowledgeInvalidationCoverage {
    RevocationCompleteOwner,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct KnowledgeInvalidationAuthorizationView {
    pub authorization_view_id: KnowledgeInvalidationAuthorizationViewId,
    pub consumer_component: Component,
    pub consumer_id: String,
    pub coverage: KnowledgeInvalidationCoverage,
    pub owner_id: UserId,
    pub authorization_revision: String,
    pub claims_digest: String,
    pub tombstone_sensitivity_ceiling: SensitivityClass,
}

impl Validate for KnowledgeInvalidationAuthorizationView {
    fn validate(&self) -> Result<()> {
        self.authorization_view_id.validate()?;
        if self.consumer_component != Component::PersonalAiPlatform {
            return Err(field(
                "consumer_component",
                "must equal personal_ai_platform",
            ));
        }
        validate_identity("consumer_id", &self.consumer_id)?;
        self.owner_id.validate()?;
        validate_identity("authorization_revision", &self.authorization_revision)?;
        validate_sha256("claims_digest", &self.claims_digest)?;
        if self.tombstone_sensitivity_ceiling != SensitivityClass::Restricted {
            return Err(field(
                "tombstone_sensitivity_ceiling",
                "must equal restricted for revocation-complete owner coverage",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum KnowledgeInvalidationEventKind {
    RevisionChanged,
    DocumentDeleted,
    KnowledgeDeleted,
    RetentionExpired,
    SensitivityChanged,
    SourceTrustChanged,
    AccessPolicyChanged,
    CagEligibilityChanged,
    BundleMembershipChanged,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum KnowledgeInvalidationStreamHistory {
    Complete,
    CutoverRequiresReconciliation,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum KnowledgeInvalidationTargetKind {
    SourceRevision,
    Bundle,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct KnowledgeInvalidationTarget {
    pub kind: KnowledgeInvalidationTargetKind,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_id: Option<KnowledgeSourceId>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub old_revision_id: Option<KnowledgeRevisionId>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub replacement_revision_id: Option<KnowledgeRevisionId>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bundle: Option<StaticBundleReference>,
}

impl Validate for KnowledgeInvalidationTarget {
    fn validate(&self) -> Result<()> {
        match self.kind {
            KnowledgeInvalidationTargetKind::SourceRevision => {
                let Some(source) = &self.source_id else {
                    return Err(field("target", "source_id is required"));
                };
                let Some(old_revision) = &self.old_revision_id else {
                    return Err(field("target", "old_revision_id is required"));
                };
                if self.bundle.is_some() {
                    return Err(field("target", "bundle is forbidden"));
                }
                source.validate()?;
                old_revision.validate()?;
                if let Some(replacement) = &self.replacement_revision_id {
                    replacement.validate()?;
                    if replacement == old_revision {
                        return Err(field(
                            "replacement_revision_id",
                            "must differ from old_revision_id",
                        ));
                    }
                }
            }
            KnowledgeInvalidationTargetKind::Bundle => {
                if self.source_id.is_some()
                    || self.old_revision_id.is_some()
                    || self.replacement_revision_id.is_some()
                {
                    return Err(field("target", "source identity is forbidden"));
                }
                self.bundle
                    .as_ref()
                    .ok_or_else(|| field("target", "bundle is required"))?
                    .validate()?;
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct KnowledgeInvalidationTombstonePayload {
    pub event_id: KnowledgeInvalidationEventId,
    pub producer: KnowledgeProducerIdentity,
    pub stream_epoch: String,
    pub sequence: u64,
    pub effective_at: String,
    pub issued_at: String,
    pub event_kind: KnowledgeInvalidationEventKind,
    pub target: KnowledgeInvalidationTarget,
    pub source_scope: CacheAuthorizationScope,
    pub source_sensitivity: SensitivityClass,
    pub reason_code: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extensions: Option<Extensions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub required_extensions: Option<RequiredExtensions>,
}

impl Validate for KnowledgeInvalidationTombstonePayload {
    fn validate(&self) -> Result<()> {
        self.event_id.validate()?;
        self.producer.validate()?;
        if self.producer.component != Component::MemoryVault {
            return Err(field("producer.component", "must equal memory_vault"));
        }
        validate_sha256("stream_epoch", &self.stream_epoch)?;
        if self.sequence == 0 || self.sequence > MAXIMUM_SEQUENCE {
            return Err(field("sequence", "out of range"));
        }
        validate_timestamp("effective_at", &self.effective_at)?;
        validate_timestamp("issued_at", &self.issued_at)?;
        if timestamp_after(&self.effective_at, &self.issued_at)? {
            return Err(field("issued_at", "must not precede effective_at"));
        }
        self.target.validate()?;
        match self.event_kind {
            KnowledgeInvalidationEventKind::RevisionChanged
            | KnowledgeInvalidationEventKind::SensitivityChanged
            | KnowledgeInvalidationEventKind::SourceTrustChanged
            | KnowledgeInvalidationEventKind::CagEligibilityChanged => {
                if self.target.kind != KnowledgeInvalidationTargetKind::SourceRevision
                    || self.target.replacement_revision_id.is_none()
                {
                    return Err(field(
                        "target",
                        "replacement event requires a source revision and replacement",
                    ));
                }
            }
            KnowledgeInvalidationEventKind::DocumentDeleted
            | KnowledgeInvalidationEventKind::KnowledgeDeleted
            | KnowledgeInvalidationEventKind::RetentionExpired => {
                if self.target.kind != KnowledgeInvalidationTargetKind::SourceRevision
                    || self.target.replacement_revision_id.is_some()
                {
                    return Err(field(
                        "target",
                        "terminal event requires an unreplaced source revision",
                    ));
                }
            }
            KnowledgeInvalidationEventKind::AccessPolicyChanged
            | KnowledgeInvalidationEventKind::BundleMembershipChanged => {
                if self.target.kind != KnowledgeInvalidationTargetKind::Bundle {
                    return Err(field(
                        "target",
                        "bundle event requires an exact bundle target",
                    ));
                }
            }
        }
        self.source_scope.validate()?;
        if !matches!(
            self.source_scope.namespace,
            CacheNamespace::Owner | CacheNamespace::Project
        ) {
            return Err(field("source_scope.namespace", "must be owner or project"));
        }
        validate_reason_code("reason_code", &self.reason_code)?;
        validate_extensions(self.extensions.as_ref(), self.required_extensions.as_ref())
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct KnowledgeInvalidationTombstone {
    pub contract_version: String,
    pub kind: String,
    pub payload: KnowledgeInvalidationTombstonePayload,
    pub canonical_digest: String,
    pub signature_envelope: QualificationSignatureEnvelope,
}

impl Validate for KnowledgeInvalidationTombstone {
    fn validate(&self) -> Result<()> {
        if self.contract_version != KNOWLEDGE_INVALIDATION_FEED_CONTRACT_VERSION
            || self.kind != "knowledge.invalidation_tombstone"
        {
            return Err(field(
                "envelope",
                "must identify knowledge invalidation tombstone v1.3",
            ));
        }
        self.payload.validate()?;
        if canonical_knowledge_invalidation_tombstone_digest(&self.payload)?
            != self.canonical_digest
        {
            return Err(field(
                "canonical_digest",
                "does not match canonical payload",
            ));
        }
        self.signature_envelope.validate()
    }
}

/// Computes the domain-separated canonical tombstone digest.
///
/// # Errors
///
/// Returns an error for an invalid payload or serialization failure.
pub fn canonical_knowledge_invalidation_tombstone_digest(
    payload: &KnowledgeInvalidationTombstonePayload,
) -> Result<String> {
    payload.validate()?;
    let canonical =
        serde_json::to_vec(&crate::document::sort_value(serde_json::to_value(payload)?))?;
    let mut hasher = Sha256::new();
    hasher.update(TOMBSTONE_DIGEST_NAMESPACE);
    hasher.update(canonical);
    Ok(format!("{:x}", hasher.finalize()))
}

/// Builds the domain-separated tombstone signature transcript.
///
/// # Errors
///
/// Returns an error unless `digest` is canonical lowercase SHA-256 hex.
pub fn knowledge_invalidation_tombstone_signing_transcript(digest: &str) -> Result<Vec<u8>> {
    signing_transcript(TOMBSTONE_SIGNATURE_NAMESPACE, digest)
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct KnowledgeInvalidationFeedRequest {
    pub contract_version: String,
    pub kind: String,
    pub request_id: RequestId,
    pub expected_producer: KnowledgeProducerIdentity,
    pub owner_id: UserId,
    pub expected_authorization_revision: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expected_authorization_view_id: Option<KnowledgeInvalidationAuthorizationViewId>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expected_stream_epoch: Option<String>,
    pub after_sequence: u64,
    pub limit: u16,
    pub idempotency_key: String,
    pub requested_at: String,
    pub expires_at: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extensions: Option<Extensions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub required_extensions: Option<RequiredExtensions>,
}

impl Validate for KnowledgeInvalidationFeedRequest {
    fn validate(&self) -> Result<()> {
        if self.contract_version != KNOWLEDGE_INVALIDATION_FEED_CONTRACT_VERSION
            || self.kind != "knowledge.invalidation_feed_request"
        {
            return Err(field(
                "envelope",
                "must identify knowledge invalidation feed request v1.3",
            ));
        }
        self.request_id.validate()?;
        self.expected_producer.validate()?;
        if self.expected_producer.component != Component::MemoryVault {
            return Err(field(
                "expected_producer.component",
                "must equal memory_vault",
            ));
        }
        self.owner_id.validate()?;
        validate_identity(
            "expected_authorization_revision",
            &self.expected_authorization_revision,
        )?;
        if self.expected_authorization_view_id.is_some() != self.expected_stream_epoch.is_some() {
            return Err(field(
                "expected_authorization_view_id",
                "view and stream epoch pins must appear together",
            ));
        }
        if let Some(view) = &self.expected_authorization_view_id {
            view.validate()?;
            validate_sha256(
                "expected_stream_epoch",
                self.expected_stream_epoch.as_deref().unwrap_or_default(),
            )?;
        }
        if self.after_sequence > MAXIMUM_SEQUENCE {
            return Err(field("after_sequence", "out of range"));
        }
        if self.after_sequence > 0 && self.expected_authorization_view_id.is_none() {
            return Err(field(
                "expected_authorization_view_id",
                "required after sequence zero",
            ));
        }
        if self.limit == 0 || usize::from(self.limit) > MAXIMUM_RESULTS {
            return Err(field("limit", "out of range"));
        }
        validate_sha256("idempotency_key", &self.idempotency_key)?;
        validate_window(&self.requested_at, &self.expires_at)?;
        validate_extensions(self.extensions.as_ref(), self.required_extensions.as_ref())
    }
}

impl KnowledgeInvalidationFeedRequest {
    /// Validates structure and activity at a caller-supplied time.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid, premature, stale, or expired requests.
    pub fn validate_at(&self, now: &str) -> Result<()> {
        self.validate()?;
        validate_active("request", &self.requested_at, &self.expires_at, now)
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct KnowledgeInvalidationFeedResultPayload {
    pub request_id: RequestId,
    pub producer: KnowledgeProducerIdentity,
    pub authorization_view: KnowledgeInvalidationAuthorizationView,
    pub stream_epoch: String,
    pub stream_history: KnowledgeInvalidationStreamHistory,
    pub epoch_started_at: String,
    pub after_sequence: u64,
    pub tombstones: Vec<KnowledgeInvalidationTombstone>,
    pub observed_through_sequence: u64,
    pub high_watermark_sequence: u64,
    pub has_more: bool,
    pub caught_up: bool,
    pub observed_at: String,
    pub expires_at: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extensions: Option<Extensions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub required_extensions: Option<RequiredExtensions>,
}

impl Validate for KnowledgeInvalidationFeedResultPayload {
    fn validate(&self) -> Result<()> {
        self.request_id.validate()?;
        self.producer.validate()?;
        if self.producer.component != Component::MemoryVault {
            return Err(field("producer.component", "must equal memory_vault"));
        }
        self.authorization_view.validate()?;
        validate_sha256("stream_epoch", &self.stream_epoch)?;
        validate_timestamp("epoch_started_at", &self.epoch_started_at)?;
        if self.after_sequence > MAXIMUM_SEQUENCE
            || self.observed_through_sequence > MAXIMUM_SEQUENCE
            || self.high_watermark_sequence > MAXIMUM_SEQUENCE
        {
            return Err(field("sequence", "out of range"));
        }
        if self.observed_through_sequence < self.after_sequence
            || self.observed_through_sequence > self.high_watermark_sequence
        {
            return Err(field(
                "observed_through_sequence",
                "must lie between after_sequence and high_watermark_sequence",
            ));
        }
        if self.has_more == self.caught_up {
            return Err(field("caught_up", "must be the inverse of has_more"));
        }
        if self.caught_up && self.observed_through_sequence != self.high_watermark_sequence {
            return Err(field(
                "caught_up",
                "requires observation through the high watermark",
            ));
        }
        if self.has_more && self.observed_through_sequence >= self.high_watermark_sequence {
            return Err(field("has_more", "requires an unobserved sequence range"));
        }
        if self.has_more && self.observed_through_sequence == self.after_sequence {
            return Err(field(
                "observed_through_sequence",
                "must advance when has_more is true",
            ));
        }
        if self.tombstones.len() > MAXIMUM_RESULTS {
            return Err(field("tombstones", "too many results"));
        }
        validate_window(&self.observed_at, &self.expires_at)?;
        if timestamp_after(&self.epoch_started_at, &self.observed_at)? {
            return Err(field("observed_at", "must not precede epoch_started_at"));
        }
        let mut previous_sequence = self.after_sequence;
        let mut seen_event_ids = HashSet::with_capacity(self.tombstones.len());
        let mut previous: Option<&KnowledgeInvalidationTombstone> = None;
        for event in &self.tombstones {
            event.validate()?;
            if event.payload.producer != self.producer
                || event.payload.stream_epoch != self.stream_epoch
                || event.payload.source_scope.owner_id != self.authorization_view.owner_id
                || sensitivity_rank(&event.payload.source_sensitivity)
                    > sensitivity_rank(&self.authorization_view.tombstone_sensitivity_ceiling)
            {
                return Err(field(
                    "tombstones",
                    "producer, epoch, owner, or sensitivity substitution",
                ));
            }
            if event.payload.sequence <= previous_sequence
                || event.payload.sequence > self.observed_through_sequence
            {
                return Err(field(
                    "tombstones",
                    "sequences must increase within the observed range",
                ));
            }
            if !seen_event_ids.insert(event.payload.event_id.clone()) {
                return Err(field("tombstones", "event identity reuse"));
            }
            if timestamp_after(&self.epoch_started_at, &event.payload.effective_at)? {
                return Err(field("tombstones", "event predates its stream epoch"));
            }
            if timestamp_after(&event.payload.issued_at, &self.observed_at)? {
                return Err(field(
                    "tombstones",
                    "event was issued after page observation",
                ));
            }
            if let Some(previous) = previous {
                if validate_knowledge_invalidation_tombstone_order(previous, event)?
                    != KnowledgeInvalidationDisposition::Next
                {
                    return Err(field("tombstones", "event order is invalid"));
                }
            }
            previous_sequence = event.payload.sequence;
            previous = Some(event);
        }
        validate_extensions(self.extensions.as_ref(), self.required_extensions.as_ref())
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct KnowledgeInvalidationFeedResult {
    pub contract_version: String,
    pub kind: String,
    pub payload: KnowledgeInvalidationFeedResultPayload,
    pub canonical_digest: String,
    pub signature_envelope: QualificationSignatureEnvelope,
}

impl Validate for KnowledgeInvalidationFeedResult {
    fn validate(&self) -> Result<()> {
        if self.contract_version != KNOWLEDGE_INVALIDATION_FEED_CONTRACT_VERSION
            || self.kind != "knowledge.invalidation_feed_result"
        {
            return Err(field(
                "envelope",
                "must identify knowledge invalidation feed result v1.3",
            ));
        }
        self.payload.validate()?;
        if canonical_knowledge_invalidation_feed_result_digest(&self.payload)?
            != self.canonical_digest
        {
            return Err(field(
                "canonical_digest",
                "does not match canonical payload",
            ));
        }
        self.signature_envelope.validate()
    }
}

impl KnowledgeInvalidationFeedResult {
    /// Validates structure and freshness at a caller-supplied time.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid, premature, stale, or expired results.
    pub fn validate_at(&self, now: &str) -> Result<()> {
        self.validate()?;
        validate_active(
            "result",
            &self.payload.observed_at,
            &self.payload.expires_at,
            now,
        )
    }

    /// Correlates a signed result with its exact request.
    ///
    /// # Errors
    ///
    /// Returns an error for request, producer, owner, view, epoch, cursor,
    /// result-count, or freshness substitution.
    pub fn validate_against(
        &self,
        request: &KnowledgeInvalidationFeedRequest,
        expected_consumer_id: &str,
    ) -> Result<()> {
        request.validate()?;
        validate_identity("expected_consumer_id", expected_consumer_id)?;
        self.validate()?;
        if self.payload.request_id != request.request_id
            || self.payload.producer != request.expected_producer
            || self.payload.authorization_view.owner_id != request.owner_id
            || self.payload.after_sequence != request.after_sequence
            || self.payload.tombstones.len() > usize::from(request.limit)
        {
            return Err(field(
                "result",
                "request, producer, owner, cursor, or limit substitution",
            ));
        }
        if self.payload.authorization_view.consumer_id != expected_consumer_id {
            return Err(field(
                "consumer_id",
                "does not match authenticated consumer",
            ));
        }
        if self.payload.authorization_view.authorization_revision
            != request.expected_authorization_revision
        {
            return Err(field(
                "authorization_revision",
                "does not match request pin",
            ));
        }
        if request
            .expected_authorization_view_id
            .as_ref()
            .is_some_and(|value| value != &self.payload.authorization_view.authorization_view_id)
        {
            return Err(field("authorization_view_id", "does not match request pin"));
        }
        if request
            .expected_stream_epoch
            .as_ref()
            .is_some_and(|value| value != &self.payload.stream_epoch)
        {
            return Err(field("stream_epoch", "does not match request pin"));
        }
        if timestamp_after(&request.requested_at, &self.payload.observed_at)?
            || timestamp_after(&self.payload.expires_at, &request.expires_at)?
        {
            return Err(field(
                "result",
                "falls outside the request freshness window",
            ));
        }
        Ok(())
    }
}

/// Computes the domain-separated canonical feed-result digest.
///
/// # Errors
///
/// Returns an error for an invalid payload or serialization failure.
pub fn canonical_knowledge_invalidation_feed_result_digest(
    payload: &KnowledgeInvalidationFeedResultPayload,
) -> Result<String> {
    payload.validate()?;
    let canonical =
        serde_json::to_vec(&crate::document::sort_value(serde_json::to_value(payload)?))?;
    let mut hasher = Sha256::new();
    hasher.update(FEED_RESULT_DIGEST_NAMESPACE);
    hasher.update(canonical);
    Ok(format!("{:x}", hasher.finalize()))
}

/// Builds the domain-separated feed-result signature transcript.
///
/// # Errors
///
/// Returns an error unless `digest` is canonical lowercase SHA-256 hex.
pub fn knowledge_invalidation_feed_result_signing_transcript(digest: &str) -> Result<Vec<u8>> {
    signing_transcript(FEED_RESULT_SIGNATURE_NAMESPACE, digest)
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum KnowledgeInvalidationDisposition {
    Next,
    Duplicate,
}

/// Accepts strictly increasing visible sequences; global gaps are expected.
///
/// # Errors
///
/// Returns an error for invalid events, substitution, rollback, conflicts, or
/// event-identity reuse.
pub fn validate_knowledge_invalidation_tombstone_order(
    previous: &KnowledgeInvalidationTombstone,
    current: &KnowledgeInvalidationTombstone,
) -> Result<KnowledgeInvalidationDisposition> {
    previous.validate()?;
    current.validate()?;
    if previous.payload.producer != current.payload.producer
        || previous.payload.stream_epoch != current.payload.stream_epoch
        || previous.payload.source_scope.owner_id != current.payload.source_scope.owner_id
    {
        return Err(field("stream", "producer, epoch, or owner substitution"));
    }
    if current.payload.sequence == previous.payload.sequence {
        if current.payload.event_id == previous.payload.event_id
            && current.canonical_digest == previous.canonical_digest
        {
            return Ok(KnowledgeInvalidationDisposition::Duplicate);
        }
        return Err(field("sequence", "conflicting replay"));
    }
    if current.payload.event_id == previous.payload.event_id
        || current.payload.sequence < previous.payload.sequence
    {
        return Err(field("sequence", "rollback or event identity reuse"));
    }
    if timestamp_after(&previous.payload.issued_at, &current.payload.issued_at)? {
        return Err(field("issued_at", "must not move backward"));
    }
    Ok(KnowledgeInvalidationDisposition::Next)
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum KnowledgeInvalidationFeedRequestDisposition {
    New,
    Duplicate,
}

/// Classifies exact request replay and rejects request/key conflicts.
///
/// # Errors
///
/// Returns an error for invalid requests, conflicting replay, or request-ID reuse.
pub fn validate_knowledge_invalidation_feed_request_replay(
    previous: &KnowledgeInvalidationFeedRequest,
    current: &KnowledgeInvalidationFeedRequest,
) -> Result<KnowledgeInvalidationFeedRequestDisposition> {
    previous.validate()?;
    current.validate()?;
    let previous_canonical = serde_json::to_vec(&crate::document::sort_value(
        serde_json::to_value(previous)?,
    ))?;
    let current_canonical =
        serde_json::to_vec(&crate::document::sort_value(serde_json::to_value(current)?))?;
    if previous.idempotency_key == current.idempotency_key {
        if previous_canonical == current_canonical {
            return Ok(KnowledgeInvalidationFeedRequestDisposition::Duplicate);
        }
        return Err(field("idempotency_key", "conflicting replay"));
    }
    if previous.request_id == current.request_id {
        return Err(field(
            "request_id",
            "reused with a different idempotency key",
        ));
    }
    Ok(KnowledgeInvalidationFeedRequestDisposition::New)
}

/// Validates that the current request is active before applying exact replay rules.
///
/// Producers must use this freshness-aware form before consulting or creating a
/// replay receipt so retained state can never revive an expired exact request.
///
/// # Errors
///
/// Returns an error for an inactive request, conflicting replay, or request-ID reuse.
pub fn validate_knowledge_invalidation_feed_request_replay_at(
    previous: &KnowledgeInvalidationFeedRequest,
    current: &KnowledgeInvalidationFeedRequest,
    now: &str,
) -> Result<KnowledgeInvalidationFeedRequestDisposition> {
    current.validate_at(now)?;
    validate_knowledge_invalidation_feed_request_replay(previous, current)
}

/// Validates page-to-page cursor, view, epoch, watermark, and time continuity.
///
/// # Errors
///
/// Returns an error for invalid pages, substitution, cursor discontinuity, or
/// rollback. An exact signed page replay is an idempotent duplicate.
pub fn validate_knowledge_invalidation_feed_page_order(
    previous: &KnowledgeInvalidationFeedResult,
    current: &KnowledgeInvalidationFeedResult,
) -> Result<KnowledgeInvalidationDisposition> {
    previous.validate()?;
    current.validate()?;
    if previous.canonical_digest == current.canonical_digest {
        return Ok(KnowledgeInvalidationDisposition::Duplicate);
    }
    if previous.payload.producer != current.payload.producer
        || previous.payload.authorization_view != current.payload.authorization_view
        || previous.payload.stream_epoch != current.payload.stream_epoch
        || previous.payload.stream_history != current.payload.stream_history
        || previous.payload.epoch_started_at != current.payload.epoch_started_at
    {
        return Err(field(
            "stream",
            "producer, authorization view, epoch, or history substitution",
        ));
    }
    if current.payload.after_sequence != previous.payload.observed_through_sequence {
        return Err(field(
            "after_sequence",
            "does not continue the observed cursor",
        ));
    }
    if current.payload.high_watermark_sequence < previous.payload.high_watermark_sequence
        || timestamp_after(&previous.payload.observed_at, &current.payload.observed_at)?
    {
        return Err(field(
            "result",
            "high watermark or observation time moved backward",
        ));
    }
    let previous_event_ids: HashSet<_> = previous
        .payload
        .tombstones
        .iter()
        .map(|event| &event.payload.event_id)
        .collect();
    if current
        .payload
        .tombstones
        .iter()
        .any(|event| previous_event_ids.contains(&event.payload.event_id))
    {
        return Err(field("tombstones", "event identity reused across pages"));
    }
    if let (Some(previous_event), Some(current_event)) = (
        previous.payload.tombstones.last(),
        current.payload.tombstones.first(),
    ) {
        if validate_knowledge_invalidation_tombstone_order(previous_event, current_event)?
            != KnowledgeInvalidationDisposition::Next
        {
            return Err(field("tombstones", "cross-page event order is invalid"));
        }
    }
    Ok(KnowledgeInvalidationDisposition::Next)
}

/// Returns the stricter handling class across the pre-event source and its
/// replacement. Terminal events retain the pre-event class.
#[must_use]
pub fn resolve_knowledge_invalidation_source_sensitivity(
    previous: &SensitivityClass,
    replacement: Option<&SensitivityClass>,
) -> SensitivityClass {
    match replacement {
        Some(replacement) if sensitivity_rank(replacement) > sensitivity_rank(previous) => {
            replacement.clone()
        }
        _ => previous.clone(),
    }
}

fn signing_transcript(namespace: &[u8], digest: &str) -> Result<Vec<u8>> {
    validate_sha256("canonical_digest", digest)?;
    let mut transcript = namespace.to_vec();
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
