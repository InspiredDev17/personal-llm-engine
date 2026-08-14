//! Provider-neutral, identity-bound inference runtime contract major version 2.
//!
//! These types contain no scheduler, model loader, qualification authority, or
//! transport authentication. They validate only bounded cross-repository data.

use crate::common::{
    check_len, check_max, check_range, validate_extensions, validate_reason_code, validate_sha256,
    validate_timestamp, validate_timestamp_window, Extensions, RequiredExtensions,
};
use crate::error::{ContractError, Result, Validate};
use crate::identity::{ModelArtifactId, ModelId, RequestId, TaskId};
use crate::inference::{GenerationParameters, GenerationUsage, InferenceMessage};
use crate::INFERENCE_RUNTIME_CONTRACT_VERSION;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;

const MAX_MEMORY_BYTES: u64 = 1 << 50;
const MAX_DURATION_MILLIS: u64 = 86_400_000;
const MAX_TOKEN_COUNT: u64 = 10_000_000;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct InferenceRuntimeContractVersion;

impl Serialize for InferenceRuntimeContractVersion {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(INFERENCE_RUNTIME_CONTRACT_VERSION)
    }
}

impl<'de> Deserialize<'de> for InferenceRuntimeContractVersion {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        if value == INFERENCE_RUNTIME_CONTRACT_VERSION {
            Ok(Self)
        } else {
            Err(serde::de::Error::custom(format!(
                "unsupported contract_version {value:?}; expected {INFERENCE_RUNTIME_CONTRACT_VERSION}"
            )))
        }
    }
}

impl Validate for InferenceRuntimeContractVersion {
    fn validate(&self) -> Result<()> {
        Ok(())
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
pub struct RuntimeIdentity {
    pub model_id: ModelId,
    pub artifact_id: ModelArtifactId,
    pub artifact_digest_algorithm: ArtifactDigestAlgorithm,
    pub artifact_digest: String,
    pub backend_id: String,
    pub backend_version: String,
    pub backend_instance_id: String,
    pub engine_instance_id: String,
}

impl Validate for RuntimeIdentity {
    fn validate(&self) -> Result<()> {
        match self.artifact_digest_algorithm {
            ArtifactDigestAlgorithm::Sha256 => {
                validate_sha256("artifact_digest", &self.artifact_digest)?;
            }
            ArtifactDigestAlgorithm::Sha384 => {
                validate_lower_hex("artifact_digest", &self.artifact_digest, 96)?;
            }
        }
        for (field, value) in [
            ("backend_id", self.backend_id.as_str()),
            ("backend_version", self.backend_version.as_str()),
            ("backend_instance_id", self.backend_instance_id.as_str()),
            ("engine_instance_id", self.engine_instance_id.as_str()),
        ] {
            validate_runtime_id(field, value)?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeResourceBudget {
    pub maximum_input_tokens: u64,
    pub maximum_output_tokens: u64,
    pub maximum_inference_memory_bytes: u64,
    pub maximum_execution_millis: u64,
}

impl Validate for RuntimeResourceBudget {
    fn validate(&self) -> Result<()> {
        check_range(
            "maximum_input_tokens",
            self.maximum_input_tokens,
            1,
            MAX_TOKEN_COUNT,
        )?;
        check_range(
            "maximum_output_tokens",
            self.maximum_output_tokens,
            1,
            MAX_TOKEN_COUNT,
        )?;
        check_range(
            "maximum_inference_memory_bytes",
            self.maximum_inference_memory_bytes,
            1,
            MAX_MEMORY_BYTES,
        )?;
        check_range(
            "maximum_execution_millis",
            self.maximum_execution_millis,
            1,
            MAX_DURATION_MILLIS,
        )
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeModelState {
    Loaded,
    Loadable,
    Draining,
    Unavailable,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeModelStatus {
    pub identity: RuntimeIdentity,
    pub state: RuntimeModelState,
    pub available_memory_bytes: u64,
    pub active_requests: u16,
    pub maximum_concurrent_requests: u16,
}

impl Validate for RuntimeModelStatus {
    fn validate(&self) -> Result<()> {
        self.identity.validate()?;
        check_range(
            "available_memory_bytes",
            self.available_memory_bytes,
            0,
            MAX_MEMORY_BYTES,
        )?;
        check_range(
            "maximum_concurrent_requests",
            self.maximum_concurrent_requests,
            1,
            u16::MAX,
        )?;
        if self.active_requests > self.maximum_concurrent_requests {
            return Err(ContractError::field(
                "active_requests",
                "must not exceed maximum_concurrent_requests",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeState {
    Available,
    Constrained,
    Draining,
    Unavailable,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum RuntimeAvailabilityKind {
    #[serde(rename = "inference.runtime_availability")]
    RuntimeAvailability,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeAvailability {
    pub contract_version: InferenceRuntimeContractVersion,
    pub kind: RuntimeAvailabilityKind,
    pub engine_instance_id: String,
    pub status_revision: u64,
    pub state: RuntimeState,
    pub models: Vec<RuntimeModelStatus>,
    pub observed_at: String,
    pub expires_at: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extensions: Option<Extensions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub required_extensions: Option<RequiredExtensions>,
}

impl Validate for RuntimeAvailability {
    fn validate(&self) -> Result<()> {
        validate_runtime_id("engine_instance_id", &self.engine_instance_id)?;
        check_range("status_revision", self.status_revision, 1, u64::MAX)?;
        check_max("models", self.models.len(), 256)?;
        let mut models = BTreeSet::new();
        for model in &self.models {
            model.validate()?;
            if model.identity.engine_instance_id != self.engine_instance_id {
                return Err(ContractError::field(
                    "models",
                    "contains a cross-engine identity",
                ));
            }
            if !models.insert(model.identity.model_id.as_str()) {
                return Err(ContractError::field(
                    "models",
                    "contains duplicate model identity",
                ));
            }
        }
        validate_timestamp_window(
            "observed_at",
            &self.observed_at,
            "expires_at",
            &self.expires_at,
            300,
        )?;
        validate_extensions(self.extensions.as_ref(), self.required_extensions.as_ref())
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum ReservationRequestKind {
    #[serde(rename = "inference.reservation_request")]
    ReservationRequest,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct InferenceReservationRequest {
    pub contract_version: InferenceRuntimeContractVersion,
    pub kind: ReservationRequestKind,
    pub reservation_id: String,
    pub request_id: RequestId,
    pub task_id: TaskId,
    pub identity: RuntimeIdentity,
    pub budget: RuntimeResourceBudget,
    pub idempotency_key: String,
    pub created_at: String,
    pub expires_at: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extensions: Option<Extensions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub required_extensions: Option<RequiredExtensions>,
}

impl Validate for InferenceReservationRequest {
    fn validate(&self) -> Result<()> {
        validate_reservation_id(&self.reservation_id)?;
        self.identity.validate()?;
        self.budget.validate()?;
        validate_sha256("idempotency_key", &self.idempotency_key)?;
        validate_timestamp_window(
            "created_at",
            &self.created_at,
            "expires_at",
            &self.expires_at,
            600,
        )?;
        validate_extensions(self.extensions.as_ref(), self.required_extensions.as_ref())
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ReservationDecision {
    Accepted,
    Denied,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum ReservationDecisionKind {
    #[serde(rename = "inference.reservation_decision")]
    ReservationDecision,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct InferenceReservationDecision {
    pub contract_version: InferenceRuntimeContractVersion,
    pub kind: ReservationDecisionKind,
    pub reservation_id: String,
    pub request_id: RequestId,
    pub task_id: TaskId,
    pub identity: RuntimeIdentity,
    pub decision: ReservationDecision,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub granted_budget: Option<RuntimeResourceBudget>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub denial_code: Option<String>,
    pub status_revision: u64,
    pub observed_at: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expires_at: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extensions: Option<Extensions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub required_extensions: Option<RequiredExtensions>,
}

impl Validate for InferenceReservationDecision {
    fn validate(&self) -> Result<()> {
        validate_reservation_id(&self.reservation_id)?;
        self.identity.validate()?;
        check_range("status_revision", self.status_revision, 1, u64::MAX)?;
        validate_timestamp("observed_at", &self.observed_at)?;
        match self.decision {
            ReservationDecision::Accepted => {
                self.granted_budget
                    .as_ref()
                    .ok_or_else(|| {
                        ContractError::field("granted_budget", "is required for accepted decisions")
                    })?
                    .validate()?;
                if self.denial_code.is_some() {
                    return Err(ContractError::field(
                        "denial_code",
                        "is forbidden for accepted decisions",
                    ));
                }
                validate_timestamp_window(
                    "observed_at",
                    &self.observed_at,
                    "expires_at",
                    self.expires_at.as_deref().unwrap_or_default(),
                    600,
                )?;
            }
            ReservationDecision::Denied => {
                if self.granted_budget.is_some() || self.expires_at.is_some() {
                    return Err(ContractError::field(
                        "reservation_decision",
                        "denied decisions cannot grant a budget or expiry",
                    ));
                }
                validate_reason_code(
                    "denial_code",
                    self.denial_code.as_deref().unwrap_or_default(),
                )?;
            }
        }
        validate_extensions(self.extensions.as_ref(), self.required_extensions.as_ref())
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub enum ExecutionRequestKind {
    #[serde(rename = "inference.execution_request")]
    ExecutionRequest,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct InferenceExecutionRequest {
    pub contract_version: InferenceRuntimeContractVersion,
    pub kind: ExecutionRequestKind,
    pub reservation_id: String,
    pub request_id: RequestId,
    pub task_id: TaskId,
    pub identity: RuntimeIdentity,
    pub messages: Vec<InferenceMessage>,
    pub parameters: GenerationParameters,
    pub idempotency_key: String,
    pub created_at: String,
    pub deadline_at: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extensions: Option<Extensions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub required_extensions: Option<RequiredExtensions>,
}

impl Validate for InferenceExecutionRequest {
    fn validate(&self) -> Result<()> {
        validate_reservation_id(&self.reservation_id)?;
        self.identity.validate()?;
        check_range("messages", self.messages.len(), 1, 64)?;
        self.messages.iter().try_for_each(Validate::validate)?;
        self.parameters.validate()?;
        validate_sha256("idempotency_key", &self.idempotency_key)?;
        validate_timestamp_window(
            "created_at",
            &self.created_at,
            "deadline_at",
            &self.deadline_at,
            86_400,
        )?;
        validate_extensions(self.extensions.as_ref(), self.required_extensions.as_ref())
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Ord, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeResource {
    InferenceMemory,
    WallTime,
    CpuTime,
    AcceleratorTime,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeResourceUnit {
    Bytes,
    Milliseconds,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeResourceUsage {
    pub resource: RuntimeResource,
    pub units: u64,
    pub unit: RuntimeResourceUnit,
}

impl Validate for RuntimeResourceUsage {
    fn validate(&self) -> Result<()> {
        match (&self.resource, &self.unit) {
            (RuntimeResource::InferenceMemory, RuntimeResourceUnit::Bytes) => {
                check_range("units", self.units, 0, MAX_MEMORY_BYTES)
            }
            (
                RuntimeResource::WallTime
                | RuntimeResource::CpuTime
                | RuntimeResource::AcceleratorTime,
                RuntimeResourceUnit::Milliseconds,
            ) => check_range("units", self.units, 0, MAX_DURATION_MILLIS),
            _ => Err(ContractError::field("unit", "does not match resource")),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeFailureClass {
    ModelUnavailable,
    BackendUnavailable,
    ResourceUnavailable,
    AdmissionDenied,
    InvalidIdentity,
    Cancelled,
    DeadlineExceeded,
    InvalidRequest,
    InternalFailure,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeFailure {
    pub class: RuntimeFailureClass,
    pub code: String,
    pub retryable: bool,
}

impl Validate for RuntimeFailure {
    fn validate(&self) -> Result<()> {
        validate_reason_code("failure.code", &self.code)?;
        if matches!(
            self.class,
            RuntimeFailureClass::InvalidIdentity
                | RuntimeFailureClass::Cancelled
                | RuntimeFailureClass::DeadlineExceeded
                | RuntimeFailureClass::InvalidRequest
        ) && self.retryable
        {
            return Err(ContractError::field(
                "failure.retryable",
                "security, cancellation, deadline, and invalid-input failures are non-retryable",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionStatus {
    Succeeded,
    Failed,
    Cancelled,
    DeadlineExceeded,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum ExecutionResultKind {
    #[serde(rename = "inference.execution_result")]
    ExecutionResult,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct InferenceExecutionResult {
    pub contract_version: InferenceRuntimeContractVersion,
    pub kind: ExecutionResultKind,
    pub reservation_id: String,
    pub request_id: RequestId,
    pub task_id: TaskId,
    pub identity: RuntimeIdentity,
    pub status: ExecutionStatus,
    pub started_at: String,
    pub completed_at: String,
    pub usage: GenerationUsage,
    pub resource_usage: Vec<RuntimeResourceUsage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_digest: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub failure: Option<RuntimeFailure>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extensions: Option<Extensions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub required_extensions: Option<RequiredExtensions>,
}

impl Validate for InferenceExecutionResult {
    fn validate(&self) -> Result<()> {
        validate_reservation_id(&self.reservation_id)?;
        self.identity.validate()?;
        self.usage.validate()?;
        if self.usage.total_tokens != self.usage.input_tokens + self.usage.output_tokens {
            return Err(ContractError::field(
                "usage",
                "total_tokens must equal input_tokens plus output_tokens",
            ));
        }
        check_max("resource_usage", self.resource_usage.len(), 16)?;
        let mut resources = BTreeSet::new();
        for usage in &self.resource_usage {
            usage.validate()?;
            if !resources.insert(&usage.resource) {
                return Err(ContractError::field(
                    "resource_usage",
                    "contains duplicate resource",
                ));
            }
        }
        validate_timestamp_window(
            "started_at",
            &self.started_at,
            "completed_at",
            &self.completed_at,
            86_400,
        )?;
        match self.status {
            ExecutionStatus::Succeeded => {
                if self.failure.is_some() {
                    return Err(ContractError::field(
                        "failure",
                        "is forbidden for succeeded results",
                    ));
                }
                let output = self.output.as_deref().ok_or_else(|| {
                    ContractError::field("output", "is required for succeeded results")
                })?;
                check_len("output", output, 1, 1_048_576)?;
                if output.len() > 1_048_576 {
                    return Err(ContractError::field(
                        "output",
                        "UTF-8 encoding exceeds 1048576 bytes",
                    ));
                }
                let digest = self.output_digest.as_deref().ok_or_else(|| {
                    ContractError::field("output_digest", "is required for succeeded results")
                })?;
                validate_sha256("output_digest", digest)?;
                if format!("{:x}", Sha256::digest(output.as_bytes())) != digest {
                    return Err(ContractError::field(
                        "output_digest",
                        "does not match exact UTF-8 output bytes",
                    ));
                }
            }
            ExecutionStatus::Failed
            | ExecutionStatus::Cancelled
            | ExecutionStatus::DeadlineExceeded => {
                if self.output.is_some() || self.output_digest.is_some() {
                    return Err(ContractError::field(
                        "output",
                        "is forbidden for non-success results",
                    ));
                }
                let failure = self.failure.as_ref().ok_or_else(|| {
                    ContractError::field("failure", "is required for non-success results")
                })?;
                failure.validate()?;
                if matches!(self.status, ExecutionStatus::Cancelled)
                    != matches!(failure.class, RuntimeFailureClass::Cancelled)
                    || matches!(self.status, ExecutionStatus::DeadlineExceeded)
                        != matches!(failure.class, RuntimeFailureClass::DeadlineExceeded)
                {
                    return Err(ContractError::field(
                        "failure.class",
                        "does not match terminal status",
                    ));
                }
            }
        }
        validate_extensions(self.extensions.as_ref(), self.required_extensions.as_ref())
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum ReservationReleaseKind {
    #[serde(rename = "inference.reservation_release")]
    ReservationRelease,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct InferenceReservationRelease {
    pub contract_version: InferenceRuntimeContractVersion,
    pub kind: ReservationReleaseKind,
    pub reservation_id: String,
    pub request_id: RequestId,
    pub task_id: TaskId,
    pub reason_code: String,
    pub requested_at: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extensions: Option<Extensions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub required_extensions: Option<RequiredExtensions>,
}
impl Validate for InferenceReservationRelease {
    fn validate(&self) -> Result<()> {
        validate_reservation_id(&self.reservation_id)?;
        validate_reason_code("reason_code", &self.reason_code)?;
        validate_timestamp("requested_at", &self.requested_at)?;
        validate_extensions(self.extensions.as_ref(), self.required_extensions.as_ref())
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum RuntimeCancellationRequestKind {
    #[serde(rename = "inference.runtime_cancellation_request")]
    RuntimeCancellationRequest,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeCancellationRequest {
    pub contract_version: InferenceRuntimeContractVersion,
    pub kind: RuntimeCancellationRequestKind,
    pub reservation_id: String,
    pub request_id: RequestId,
    pub task_id: TaskId,
    pub cancellation_request_id: RequestId,
    pub reason_code: String,
    pub requested_at: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extensions: Option<Extensions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub required_extensions: Option<RequiredExtensions>,
}
impl Validate for RuntimeCancellationRequest {
    fn validate(&self) -> Result<()> {
        validate_reservation_id(&self.reservation_id)?;
        if self.cancellation_request_id == self.request_id {
            return Err(ContractError::field(
                "cancellation_request_id",
                "must differ from request_id",
            ));
        }
        validate_reason_code("reason_code", &self.reason_code)?;
        validate_timestamp("requested_at", &self.requested_at)?;
        validate_extensions(self.extensions.as_ref(), self.required_extensions.as_ref())
    }
}

fn validate_runtime_id(field: &'static str, value: &str) -> Result<()> {
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
        return Err(ContractError::field(
            field,
            "invalid path-free runtime identifier",
        ));
    }
    Ok(())
}

fn validate_reservation_id(value: &str) -> Result<()> {
    check_len("reservation_id", value, 5, 68)?;
    let Some(body) = value.strip_prefix("rsv_") else {
        return Err(ContractError::field(
            "reservation_id",
            "must start with rsv_",
        ));
    };
    if body.len() < 2
        || !body.bytes().all(|byte| {
            byte.is_ascii_lowercase() || byte.is_ascii_digit() || matches!(byte, b'_' | b'-')
        })
        || !body.as_bytes()[0].is_ascii_alphanumeric()
    {
        return Err(ContractError::field("reservation_id", "has invalid syntax"));
    }
    Ok(())
}

fn validate_lower_hex(field: &'static str, value: &str, length: usize) -> Result<()> {
    if value.len() == length
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || matches!(byte, b'a'..=b'f'))
    {
        Ok(())
    } else {
        Err(ContractError::field(
            field,
            "must be lowercase hexadecimal with the declared length",
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::{InferenceExecutionResult, RuntimeAvailability};
    use crate::Validate;

    #[test]
    fn shared_valid_fixtures_decode() {
        let availability: RuntimeAvailability = serde_json::from_str(include_str!(
            "../../fixtures/v1/valid/inference-v2-runtime-availability.json"
        ))
        .expect("decode availability");
        availability.validate().expect("validate availability");
        let result: InferenceExecutionResult = serde_json::from_str(include_str!(
            "../../fixtures/v1/valid/inference-v2-execution-result.json"
        ))
        .expect("decode result");
        result.validate().expect("validate result");
    }

    #[test]
    fn output_substitution_is_rejected() {
        let result: InferenceExecutionResult = serde_json::from_str(include_str!(
            "../../fixtures/v1/invalid/inference-v2-execution-result-digest-mismatch.json"
        ))
        .expect("decode invalid fixture shape");
        assert!(result.validate().is_err());
    }
}
