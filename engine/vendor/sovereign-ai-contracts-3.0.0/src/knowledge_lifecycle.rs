//! Additive knowledge 1.1 cache lifecycle contracts.

use crate::common::{
    compare_timestamp_values, validate_reason_code, validate_sha256, validate_timestamp,
    validate_timestamp_order, Component, RedactedReference,
};
use crate::error::{ContractError, Result, Validate};
use crate::identity::{
    AgentId, CagCacheId, KnowledgeRevisionId, KnowledgeSourceId, ModelId, ProjectId, RequestId,
    TaskId, UserId,
};
use crate::knowledge::{
    CacheBinding, CagCacheDescriptor, KnowledgeProducerIdentity, KnowledgeScope,
};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::BTreeSet;

pub const KNOWLEDGE_CACHE_LIFECYCLE_CONTRACT_VERSION: &str = "1.1.0";
const MAXIMUM_TOKENS: u64 = 2_000_000;
const MAXIMUM_CACHE_BYTES: u64 = 1_099_511_627_776;
const MAXIMUM_TTL_SECONDS: u64 = 31_536_000;
const MAXIMUM_RESULTS: usize = 256;
const MAXIMUM_COUNTER: u64 = 9_007_199_254_740_991;

fn field(name: &'static str, message: impl Into<String>) -> ContractError {
    ContractError::field(name, message)
}

fn validate_envelope(version: &str, kind: &str, expected: &'static str) -> Result<()> {
    if version != KNOWLEDGE_CACHE_LIFECYCLE_CONTRACT_VERSION {
        return Err(field("contract_version", "must equal 1.1.0"));
    }
    if kind != expected {
        return Err(field("kind", format!("must equal {expected}")));
    }
    Ok(())
}

fn validate_request_time(requested_at: &str, expires_at: &str) -> Result<()> {
    validate_timestamp("requested_at", requested_at)?;
    validate_timestamp_order("requested_at", requested_at, "expires_at", expires_at)
}

fn valid_engine_identity(value: &str) -> bool {
    if value.is_empty() || value.len() > 128 || !value.is_ascii() {
        return false;
    }
    let mut separator = false;
    for (index, byte) in value.bytes().enumerate() {
        if (index == 0 && !byte.is_ascii_lowercase())
            || (!byte.is_ascii_lowercase()
                && !byte.is_ascii_digit()
                && !matches!(byte, b'.' | b'_' | b'-'))
            || (matches!(byte, b'.' | b'_' | b'-') && separator)
        {
            return false;
        }
        separator = matches!(byte, b'.' | b'_' | b'-');
    }
    !separator
}

fn validate_engine_identity(value: &str) -> Result<()> {
    if valid_engine_identity(value) {
        Ok(())
    } else {
        Err(field(
            "engine_instance_id",
            "must be a bounded lowercase identity",
        ))
    }
}

fn validate_native_producer(producer: &KnowledgeProducerIdentity) -> Result<()> {
    producer.validate()?;
    if producer.component == Component::NativeInferenceEngine {
        Ok(())
    } else {
        Err(field(
            "producer.component",
            "must be native_inference_engine",
        ))
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CacheNamespace {
    Owner,
    Project,
    Task,
    Agent,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum KvDtype {
    F32,
    F16,
    Bfloat16,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CacheAuthorizationScope {
    pub namespace: CacheNamespace,
    pub owner_id: UserId,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub project_id: Option<ProjectId>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub task_id: Option<TaskId>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub agent_id: Option<AgentId>,
}

impl Validate for CacheAuthorizationScope {
    fn validate(&self) -> Result<()> {
        self.owner_id.validate()?;
        if let Some(value) = &self.project_id {
            value.validate()?;
        }
        if let Some(value) = &self.task_id {
            value.validate()?;
        }
        if let Some(value) = &self.agent_id {
            value.validate()?;
        }
        let valid = match self.namespace {
            CacheNamespace::Owner => {
                self.project_id.is_none() && self.task_id.is_none() && self.agent_id.is_none()
            }
            CacheNamespace::Project => {
                self.project_id.is_some() && self.task_id.is_none() && self.agent_id.is_none()
            }
            CacheNamespace::Task => {
                self.project_id.is_some() && self.task_id.is_some() && self.agent_id.is_none()
            }
            CacheNamespace::Agent => {
                self.project_id.is_some() && self.agent_id.is_some() && self.task_id.is_none()
            }
        };
        if valid {
            Ok(())
        } else {
            Err(field(
                "authorization_scope",
                "namespace identity combination is invalid",
            ))
        }
    }
}

impl CacheAuthorizationScope {
    fn matches_knowledge_scope(&self, scope: &KnowledgeScope) -> bool {
        if self.owner_id != scope.owner_id {
            return false;
        }
        match self.namespace {
            CacheNamespace::Owner => true,
            _ => self.project_id.as_ref() == Some(&scope.project_id),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CacheLifecycleBinding {
    pub base: CacheBinding,
    pub authorization_scope: CacheAuthorizationScope,
    pub kv_dtype: KvDtype,
}

impl Validate for CacheLifecycleBinding {
    fn validate(&self) -> Result<()> {
        self.base.validate()?;
        self.authorization_scope.validate()?;
        if !self
            .authorization_scope
            .matches_knowledge_scope(&self.base.scope)
        {
            return Err(field(
                "authorization_scope",
                "does not bind the knowledge owner/project scope",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CagCacheLifecycleDescriptor {
    pub descriptor: CagCacheDescriptor,
    pub lifecycle_binding: CacheLifecycleBinding,
    pub byte_size: u64,
    pub hit_count: u64,
    pub last_accessed_at: String,
}

impl Validate for CagCacheLifecycleDescriptor {
    fn validate(&self) -> Result<()> {
        self.descriptor.validate()?;
        self.lifecycle_binding.validate()?;
        if self.lifecycle_binding.base != self.descriptor.binding {
            return Err(field(
                "lifecycle_binding.base",
                "does not equal descriptor binding",
            ));
        }
        if !(1..=MAXIMUM_CACHE_BYTES).contains(&self.byte_size) {
            return Err(field("byte_size", "out of range"));
        }
        if self.hit_count > MAXIMUM_COUNTER {
            return Err(field("hit_count", "exceeds canonical integer bound"));
        }
        validate_timestamp("last_accessed_at", &self.last_accessed_at)?;
        if compare_timestamp_values(&self.last_accessed_at, &self.descriptor.created_at)?
            == Ordering::Less
        {
            return Err(field("last_accessed_at", "precedes cache creation"));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CagCacheCreateRequest {
    pub contract_version: String,
    pub kind: String,
    pub request_id: RequestId,
    pub binding: CacheLifecycleBinding,
    pub prefix_reference: RedactedReference,
    pub maximum_cache_bytes: u64,
    pub ttl_seconds: u64,
    pub eviction_priority: u8,
    pub pinned: bool,
    pub idempotency_key: String,
    pub requested_at: String,
    pub expires_at: String,
}

impl Validate for CagCacheCreateRequest {
    fn validate(&self) -> Result<()> {
        validate_envelope(
            &self.contract_version,
            &self.kind,
            "knowledge.cag_cache_create_request",
        )?;
        self.request_id.validate()?;
        self.binding.validate()?;
        if self.binding.base.cache_implementation
            != crate::knowledge::CacheImplementation::LocalNativeKv
        {
            return Err(field("cache_implementation", "create is local-native only"));
        }
        self.prefix_reference.validate()?;
        if self.prefix_reference.digest != self.binding.base.bundle.ordered_prefix_digest {
            return Err(field(
                "prefix_reference.digest",
                "must bind the exact ordered prefix",
            ));
        }
        if !(1..=MAXIMUM_CACHE_BYTES).contains(&self.maximum_cache_bytes) {
            return Err(field("maximum_cache_bytes", "out of range"));
        }
        if !(1..=MAXIMUM_TTL_SECONDS).contains(&self.ttl_seconds) {
            return Err(field("ttl_seconds", "out of range"));
        }
        validate_sha256("idempotency_key", &self.idempotency_key)?;
        validate_request_time(&self.requested_at, &self.expires_at)
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CacheCreateStatus {
    Created,
    Reused,
    Rejected,
    Failed,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CagCacheCreateResult {
    pub contract_version: String,
    pub kind: String,
    pub request_id: RequestId,
    pub producer: KnowledgeProducerIdentity,
    pub status: CacheCreateStatus,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache: Option<CagCacheLifecycleDescriptor>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason_code: Option<String>,
    pub completed_at: String,
}

impl Validate for CagCacheCreateResult {
    fn validate(&self) -> Result<()> {
        validate_envelope(
            &self.contract_version,
            &self.kind,
            "knowledge.cag_cache_create_result",
        )?;
        self.request_id.validate()?;
        validate_native_producer(&self.producer)?;
        match (&self.status, &self.cache, &self.reason_code) {
            (CacheCreateStatus::Created | CacheCreateStatus::Reused, Some(cache), None) => {
                cache.validate()?;
                if cache.descriptor.producer != self.producer {
                    return Err(field("producer", "cache producer mismatch"));
                }
            }
            (CacheCreateStatus::Rejected | CacheCreateStatus::Failed, None, Some(reason)) => {
                validate_reason_code("reason_code", reason)?;
            }
            _ => {
                return Err(field(
                    "create_result",
                    "status payload combination is invalid",
                ))
            }
        }
        validate_timestamp("completed_at", &self.completed_at)
    }
}

impl CagCacheCreateResult {
    /// Correlates a result with its exact create request.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid input, substitution, or invalid timing.
    pub fn validate_against(&self, request: &CagCacheCreateRequest) -> Result<()> {
        self.validate()?;
        request.validate()?;
        if self.request_id != request.request_id
            || self.producer.producer_id != request.binding.base.runtime_identity.engine_instance_id
            || self
                .cache
                .as_ref()
                .is_some_and(|cache| cache.lifecycle_binding != request.binding)
        {
            return Err(field(
                "create_result",
                "request, engine, or binding substitution",
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

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CagCacheDescribeRequest {
    pub contract_version: String,
    pub kind: String,
    pub request_id: RequestId,
    pub cache_id: CagCacheId,
    pub engine_instance_id: String,
    pub authorization_scope: CacheAuthorizationScope,
    pub requested_at: String,
    pub expires_at: String,
}

impl Validate for CagCacheDescribeRequest {
    fn validate(&self) -> Result<()> {
        validate_envelope(
            &self.contract_version,
            &self.kind,
            "knowledge.cag_cache_describe_request",
        )?;
        self.request_id.validate()?;
        self.cache_id.validate()?;
        validate_engine_identity(&self.engine_instance_id)?;
        self.authorization_scope.validate()?;
        validate_request_time(&self.requested_at, &self.expires_at)
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CacheDescribeStatus {
    Found,
    Missing,
    Rejected,
    Failed,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CagCacheDescribeResult {
    pub contract_version: String,
    pub kind: String,
    pub request_id: RequestId,
    pub producer: KnowledgeProducerIdentity,
    pub status: CacheDescribeStatus,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache: Option<CagCacheLifecycleDescriptor>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason_code: Option<String>,
    pub observed_at: String,
}

impl Validate for CagCacheDescribeResult {
    fn validate(&self) -> Result<()> {
        validate_envelope(
            &self.contract_version,
            &self.kind,
            "knowledge.cag_cache_describe_result",
        )?;
        self.request_id.validate()?;
        validate_native_producer(&self.producer)?;
        match (&self.status, &self.cache, &self.reason_code) {
            (CacheDescribeStatus::Found, Some(cache), None) => {
                cache.validate()?;
                if cache.descriptor.producer != self.producer {
                    return Err(field("producer", "cache producer mismatch"));
                }
            }
            (CacheDescribeStatus::Missing, None, None) => {}
            (CacheDescribeStatus::Rejected | CacheDescribeStatus::Failed, None, Some(reason)) => {
                validate_reason_code("reason_code", reason)?;
            }
            _ => {
                return Err(field(
                    "describe_result",
                    "status payload combination is invalid",
                ))
            }
        }
        validate_timestamp("observed_at", &self.observed_at)
    }
}

impl CagCacheDescribeResult {
    /// Correlates a result with its exact describe request.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid input or identity, engine, or scope substitution.
    pub fn validate_against(&self, request: &CagCacheDescribeRequest) -> Result<()> {
        self.validate()?;
        request.validate()?;
        if self.request_id != request.request_id
            || self.producer.producer_id != request.engine_instance_id
            || self.cache.as_ref().is_some_and(|cache| {
                cache.descriptor.cache_id != request.cache_id
                    || cache.lifecycle_binding.authorization_scope != request.authorization_scope
            })
        {
            return Err(field(
                "describe_result",
                "request, cache, engine, or scope substitution",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CagCacheListRequest {
    pub contract_version: String,
    pub kind: String,
    pub request_id: RequestId,
    pub engine_instance_id: String,
    pub authorization_scope: CacheAuthorizationScope,
    pub maximum_results: u16,
    pub requested_at: String,
    pub expires_at: String,
}

impl Validate for CagCacheListRequest {
    fn validate(&self) -> Result<()> {
        validate_envelope(
            &self.contract_version,
            &self.kind,
            "knowledge.cag_cache_list_request",
        )?;
        self.request_id.validate()?;
        validate_engine_identity(&self.engine_instance_id)?;
        self.authorization_scope.validate()?;
        if self.maximum_results == 0 || usize::from(self.maximum_results) > MAXIMUM_RESULTS {
            return Err(field("maximum_results", "out of range"));
        }
        validate_request_time(&self.requested_at, &self.expires_at)
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CagCacheListResult {
    pub contract_version: String,
    pub kind: String,
    pub request_id: RequestId,
    pub producer: KnowledgeProducerIdentity,
    pub authorization_scope: CacheAuthorizationScope,
    pub caches: Vec<CagCacheLifecycleDescriptor>,
    pub truncated: bool,
    pub observed_at: String,
    pub expires_at: String,
}

impl Validate for CagCacheListResult {
    fn validate(&self) -> Result<()> {
        validate_envelope(
            &self.contract_version,
            &self.kind,
            "knowledge.cag_cache_list_result",
        )?;
        self.request_id.validate()?;
        validate_native_producer(&self.producer)?;
        self.authorization_scope.validate()?;
        if self.caches.len() > MAXIMUM_RESULTS {
            return Err(field("caches", "exceeds 256"));
        }
        validate_timestamp_order(
            "observed_at",
            &self.observed_at,
            "expires_at",
            &self.expires_at,
        )?;
        let mut seen = BTreeSet::new();
        for cache in &self.caches {
            cache.validate()?;
            if cache.descriptor.producer != self.producer
                || cache.lifecycle_binding.authorization_scope != self.authorization_scope
                || !seen.insert(cache.descriptor.cache_id.as_str())
            {
                return Err(field(
                    "caches",
                    "duplicate, cross-engine, or cross-scope cache",
                ));
            }
        }
        Ok(())
    }
}

impl CagCacheListResult {
    /// Correlates an inventory with its bounded authorized list request.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid input, substitution, or result-count escalation.
    pub fn validate_against(&self, request: &CagCacheListRequest) -> Result<()> {
        self.validate()?;
        request.validate()?;
        if self.request_id != request.request_id
            || self.producer.producer_id != request.engine_instance_id
            || self.authorization_scope != request.authorization_scope
            || self.caches.len() > usize::from(request.maximum_results)
        {
            return Err(field("list_result", "request binding mismatch"));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CagCacheAttachRequest {
    pub contract_version: String,
    pub kind: String,
    pub request_id: RequestId,
    pub target_request_id: RequestId,
    pub cache_id: CagCacheId,
    pub expected: CacheLifecycleBinding,
    pub idempotency_key: String,
    pub requested_at: String,
    pub expires_at: String,
}

impl Validate for CagCacheAttachRequest {
    fn validate(&self) -> Result<()> {
        validate_envelope(
            &self.contract_version,
            &self.kind,
            "knowledge.cag_cache_attach_request",
        )?;
        self.request_id.validate()?;
        self.target_request_id.validate()?;
        if self.request_id == self.target_request_id {
            return Err(field(
                "target_request_id",
                "must differ from operation request",
            ));
        }
        self.cache_id.validate()?;
        self.expected.validate()?;
        validate_sha256("idempotency_key", &self.idempotency_key)?;
        validate_request_time(&self.requested_at, &self.expires_at)
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CacheAttachStatus {
    Attached,
    Rejected,
    Failed,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CagCacheAttachResult {
    pub contract_version: String,
    pub kind: String,
    pub request_id: RequestId,
    pub target_request_id: RequestId,
    pub cache_id: CagCacheId,
    pub producer: KnowledgeProducerIdentity,
    pub status: CacheAttachStatus,
    pub token_count: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason_code: Option<String>,
    pub completed_at: String,
}

impl Validate for CagCacheAttachResult {
    fn validate(&self) -> Result<()> {
        validate_envelope(
            &self.contract_version,
            &self.kind,
            "knowledge.cag_cache_attach_result",
        )?;
        self.request_id.validate()?;
        self.target_request_id.validate()?;
        self.cache_id.validate()?;
        validate_native_producer(&self.producer)?;
        match (&self.status, self.token_count, &self.reason_code) {
            (CacheAttachStatus::Attached, 1..=MAXIMUM_TOKENS, None) => {}
            (CacheAttachStatus::Rejected | CacheAttachStatus::Failed, 0, Some(reason)) => {
                validate_reason_code("reason_code", reason)?;
            }
            _ => {
                return Err(field(
                    "attach_result",
                    "status payload combination is invalid",
                ))
            }
        }
        validate_timestamp("completed_at", &self.completed_at)
    }
}

impl CagCacheAttachResult {
    /// Correlates an attach result with its cache and inference request binding.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid input, substitution, or invalid timing.
    pub fn validate_against(&self, request: &CagCacheAttachRequest) -> Result<()> {
        self.validate()?;
        request.validate()?;
        if self.request_id != request.request_id
            || self.target_request_id != request.target_request_id
            || self.cache_id != request.cache_id
            || self.producer.producer_id
                != request.expected.base.runtime_identity.engine_instance_id
            || (matches!(self.status, CacheAttachStatus::Attached)
                && self.token_count != request.expected.base.token_count)
        {
            return Err(field(
                "attach_result",
                "request, cache, engine, or token substitution",
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

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CagCacheReleaseRequest {
    pub contract_version: String,
    pub kind: String,
    pub request_id: RequestId,
    pub target_request_id: RequestId,
    pub cache_id: CagCacheId,
    pub engine_instance_id: String,
    pub authorization_scope: CacheAuthorizationScope,
    pub idempotency_key: String,
    pub requested_at: String,
    pub expires_at: String,
}

impl Validate for CagCacheReleaseRequest {
    fn validate(&self) -> Result<()> {
        validate_envelope(
            &self.contract_version,
            &self.kind,
            "knowledge.cag_cache_release_request",
        )?;
        self.request_id.validate()?;
        self.target_request_id.validate()?;
        self.cache_id.validate()?;
        validate_engine_identity(&self.engine_instance_id)?;
        self.authorization_scope.validate()?;
        validate_sha256("idempotency_key", &self.idempotency_key)?;
        validate_request_time(&self.requested_at, &self.expires_at)
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CacheReleaseStatus {
    Released,
    AlreadyReleased,
    Rejected,
    Failed,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CagCacheReleaseResult {
    pub contract_version: String,
    pub kind: String,
    pub request_id: RequestId,
    pub target_request_id: RequestId,
    pub cache_id: CagCacheId,
    pub producer: KnowledgeProducerIdentity,
    pub status: CacheReleaseStatus,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason_code: Option<String>,
    pub completed_at: String,
}

impl Validate for CagCacheReleaseResult {
    fn validate(&self) -> Result<()> {
        validate_envelope(
            &self.contract_version,
            &self.kind,
            "knowledge.cag_cache_release_result",
        )?;
        self.request_id.validate()?;
        self.target_request_id.validate()?;
        self.cache_id.validate()?;
        validate_native_producer(&self.producer)?;
        match (&self.status, &self.reason_code) {
            (CacheReleaseStatus::Released | CacheReleaseStatus::AlreadyReleased, None) => {}
            (CacheReleaseStatus::Rejected | CacheReleaseStatus::Failed, Some(reason)) => {
                validate_reason_code("reason_code", reason)?;
            }
            _ => {
                return Err(field(
                    "release_result",
                    "status payload combination is invalid",
                ))
            }
        }
        validate_timestamp("completed_at", &self.completed_at)
    }
}

impl CagCacheReleaseResult {
    /// Correlates a release result with its exact request.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid input, substitution, or invalid timing.
    pub fn validate_against(&self, request: &CagCacheReleaseRequest) -> Result<()> {
        self.validate()?;
        request.validate()?;
        if self.request_id != request.request_id
            || self.target_request_id != request.target_request_id
            || self.cache_id != request.cache_id
            || self.producer.producer_id != request.engine_instance_id
        {
            return Err(field(
                "release_result",
                "request, cache, or engine substitution",
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

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CacheInvalidationKind {
    ExactCache,
    Scope,
    LogicalModel,
    SourceRevision,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CacheInvalidationSelector {
    pub kind: CacheInvalidationKind,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_id: Option<CagCacheId>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logical_model_id: Option<ModelId>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_id: Option<KnowledgeSourceId>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub revision_id: Option<KnowledgeRevisionId>,
}

impl Validate for CacheInvalidationSelector {
    fn validate(&self) -> Result<()> {
        match (
            &self.kind,
            &self.cache_id,
            &self.logical_model_id,
            &self.source_id,
            &self.revision_id,
        ) {
            (CacheInvalidationKind::ExactCache, Some(cache), None, None, None) => cache.validate(),
            (CacheInvalidationKind::Scope, None, None, None, None) => Ok(()),
            (CacheInvalidationKind::LogicalModel, None, Some(model), None, None) => {
                model.validate()
            }
            (CacheInvalidationKind::SourceRevision, None, None, Some(source), Some(revision)) => {
                source.validate()?;
                revision.validate()
            }
            _ => Err(field("selector", "kind and identity fields do not agree")),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CagCacheInvalidateRequest {
    pub contract_version: String,
    pub kind: String,
    pub request_id: RequestId,
    pub engine_instance_id: String,
    pub authorization_scope: CacheAuthorizationScope,
    pub selector: CacheInvalidationSelector,
    pub reason_code: String,
    pub idempotency_key: String,
    pub requested_at: String,
    pub expires_at: String,
}

impl Validate for CagCacheInvalidateRequest {
    fn validate(&self) -> Result<()> {
        validate_envelope(
            &self.contract_version,
            &self.kind,
            "knowledge.cag_cache_invalidate_request",
        )?;
        self.request_id.validate()?;
        validate_engine_identity(&self.engine_instance_id)?;
        self.authorization_scope.validate()?;
        self.selector.validate()?;
        validate_reason_code("reason_code", &self.reason_code)?;
        validate_sha256("idempotency_key", &self.idempotency_key)?;
        validate_request_time(&self.requested_at, &self.expires_at)
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CacheInvalidateStatus {
    Invalidated,
    NoMatch,
    Rejected,
    Failed,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CagCacheInvalidateResult {
    pub contract_version: String,
    pub kind: String,
    pub request_id: RequestId,
    pub producer: KnowledgeProducerIdentity,
    pub status: CacheInvalidateStatus,
    pub invalidated_count: u16,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason_code: Option<String>,
    pub completed_at: String,
}

impl Validate for CagCacheInvalidateResult {
    fn validate(&self) -> Result<()> {
        validate_envelope(
            &self.contract_version,
            &self.kind,
            "knowledge.cag_cache_invalidate_result",
        )?;
        self.request_id.validate()?;
        validate_native_producer(&self.producer)?;
        match (&self.status, self.invalidated_count, &self.reason_code) {
            (CacheInvalidateStatus::Invalidated, 1..=256, None)
            | (CacheInvalidateStatus::NoMatch, 0, None) => {}
            (CacheInvalidateStatus::Rejected | CacheInvalidateStatus::Failed, 0, Some(reason)) => {
                validate_reason_code("reason_code", reason)?;
            }
            _ => {
                return Err(field(
                    "invalidate_result",
                    "status payload combination is invalid",
                ))
            }
        }
        validate_timestamp("completed_at", &self.completed_at)
    }
}

impl CagCacheInvalidateResult {
    /// Correlates an invalidation result with its exact request.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid input, substitution, or invalid timing.
    pub fn validate_against(&self, request: &CagCacheInvalidateRequest) -> Result<()> {
        self.validate()?;
        request.validate()?;
        if self.request_id != request.request_id
            || self.producer.producer_id != request.engine_instance_id
        {
            return Err(field("invalidate_result", "request or engine substitution"));
        }
        validate_timestamp_order(
            "requested_at",
            &request.requested_at,
            "completed_at",
            &self.completed_at,
        )
    }
}

#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CacheMetrics {
    pub lookups: u64,
    pub hits: u64,
    pub misses: u64,
    pub exact_prefix_hits: u64,
    pub partial_prefix_hits: u64,
    pub tokens_reused: u64,
    pub tokens_recomputed: u64,
    pub bytes_reused: u64,
    pub cpu_cache_bytes: u64,
    pub gpu_cache_bytes: u64,
    pub evictions: u64,
    pub invalidations: u64,
    pub transfer_nanoseconds: u64,
    pub measured_time_saved_nanoseconds: u64,
    pub time_to_first_token_nanoseconds: u64,
    pub prefill_nanoseconds: u64,
}

impl Validate for CacheMetrics {
    fn validate(&self) -> Result<()> {
        let values = [
            self.lookups,
            self.hits,
            self.misses,
            self.exact_prefix_hits,
            self.partial_prefix_hits,
            self.tokens_reused,
            self.tokens_recomputed,
            self.bytes_reused,
            self.cpu_cache_bytes,
            self.gpu_cache_bytes,
            self.evictions,
            self.invalidations,
            self.transfer_nanoseconds,
            self.measured_time_saved_nanoseconds,
            self.time_to_first_token_nanoseconds,
            self.prefill_nanoseconds,
        ];
        if values.iter().any(|value| *value > MAXIMUM_COUNTER) {
            return Err(field("metrics", "counter exceeds canonical integer bound"));
        }
        if self.hits > self.lookups
            || self.misses > self.lookups
            || self.exact_prefix_hits > self.hits
            || self.partial_prefix_hits > self.hits
        {
            return Err(field("metrics", "counter relationship is impossible"));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CagCacheMetricsRequest {
    pub contract_version: String,
    pub kind: String,
    pub request_id: RequestId,
    pub engine_instance_id: String,
    pub logical_model_id: ModelId,
    pub authorization_scope: CacheAuthorizationScope,
    pub requested_at: String,
    pub expires_at: String,
}

impl Validate for CagCacheMetricsRequest {
    fn validate(&self) -> Result<()> {
        validate_envelope(
            &self.contract_version,
            &self.kind,
            "knowledge.cag_cache_metrics_request",
        )?;
        self.request_id.validate()?;
        validate_engine_identity(&self.engine_instance_id)?;
        self.logical_model_id.validate()?;
        self.authorization_scope.validate()?;
        validate_request_time(&self.requested_at, &self.expires_at)
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CagCacheMetricsResult {
    pub contract_version: String,
    pub kind: String,
    pub request_id: RequestId,
    pub producer: KnowledgeProducerIdentity,
    pub logical_model_id: ModelId,
    pub authorization_scope: CacheAuthorizationScope,
    pub metrics: CacheMetrics,
    pub window_started_at: String,
    pub observed_at: String,
}

impl Validate for CagCacheMetricsResult {
    fn validate(&self) -> Result<()> {
        validate_envelope(
            &self.contract_version,
            &self.kind,
            "knowledge.cag_cache_metrics_result",
        )?;
        self.request_id.validate()?;
        validate_native_producer(&self.producer)?;
        self.logical_model_id.validate()?;
        self.authorization_scope.validate()?;
        self.metrics.validate()?;
        validate_timestamp_order(
            "window_started_at",
            &self.window_started_at,
            "observed_at",
            &self.observed_at,
        )
    }
}

impl CagCacheMetricsResult {
    /// Correlates content-free cache metrics with their bounded request.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid input, substitution, or invalid timing.
    pub fn validate_against(&self, request: &CagCacheMetricsRequest) -> Result<()> {
        self.validate()?;
        request.validate()?;
        if self.request_id != request.request_id
            || self.producer.producer_id != request.engine_instance_id
            || self.logical_model_id != request.logical_model_id
            || self.authorization_scope != request.authorization_scope
        {
            return Err(field(
                "metrics_result",
                "request, engine, model, or scope substitution",
            ));
        }
        validate_timestamp_order(
            "requested_at",
            &request.requested_at,
            "observed_at",
            &self.observed_at,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::{
        CagCacheAttachRequest, CagCacheAttachResult, CagCacheCreateRequest, CagCacheCreateResult,
        CagCacheDescribeRequest, CagCacheDescribeResult, CagCacheInvalidateRequest,
        CagCacheInvalidateResult, CagCacheListRequest, CagCacheListResult, CagCacheMetricsRequest,
        CagCacheMetricsResult, CagCacheReleaseRequest, CagCacheReleaseResult,
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
    fn golden_pairs_reject_substitution_and_impossible_metrics() {
        let create_request: CagCacheCreateRequest =
            fixture("knowledge-cache-create-request-1.1.0.json");
        let create_result: CagCacheCreateResult =
            fixture("knowledge-cache-create-result-1.1.0.json");
        create_result
            .validate_against(&create_request)
            .expect("create pair");
        let mut substituted_create = create_request.clone();
        substituted_create
            .binding
            .base
            .runtime_identity
            .engine_instance_id = "engine.substituted".into();
        assert!(create_result.validate_against(&substituted_create).is_err());

        let describe_request: CagCacheDescribeRequest =
            fixture("knowledge-cache-describe-request-1.1.0.json");
        let describe_result: CagCacheDescribeResult =
            fixture("knowledge-cache-describe-result-1.1.0.json");
        describe_result
            .validate_against(&describe_request)
            .expect("describe pair");

        let list_request: CagCacheListRequest = fixture("knowledge-cache-list-request-1.1.0.json");
        let list_result: CagCacheListResult = fixture("knowledge-cache-list-result-1.1.0.json");
        list_result
            .validate_against(&list_request)
            .expect("list pair");

        let attach_request: CagCacheAttachRequest =
            fixture("knowledge-cache-attach-request-1.1.0.json");
        let mut attach_result: CagCacheAttachResult =
            fixture("knowledge-cache-attach-result-1.1.0.json");
        attach_result
            .validate_against(&attach_request)
            .expect("attach pair");
        attach_result.token_count += 1;
        assert!(attach_result.validate_against(&attach_request).is_err());

        let release_request: CagCacheReleaseRequest =
            fixture("knowledge-cache-release-request-1.1.0.json");
        let release_result: CagCacheReleaseResult =
            fixture("knowledge-cache-release-result-1.1.0.json");
        release_result
            .validate_against(&release_request)
            .expect("release pair");

        let invalidate_request: CagCacheInvalidateRequest =
            fixture("knowledge-cache-invalidate-request-1.1.0.json");
        let invalidate_result: CagCacheInvalidateResult =
            fixture("knowledge-cache-invalidate-result-1.1.0.json");
        invalidate_result
            .validate_against(&invalidate_request)
            .expect("invalidate pair");

        let metrics_request: CagCacheMetricsRequest =
            fixture("knowledge-cache-metrics-request-1.1.0.json");
        let mut metrics_result: CagCacheMetricsResult =
            fixture("knowledge-cache-metrics-result-1.1.0.json");
        metrics_result
            .validate_against(&metrics_request)
            .expect("metrics pair");
        metrics_result.metrics.hits = metrics_result.metrics.lookups + 1;
        assert!(metrics_result.validate_against(&metrics_request).is_err());
    }
}
