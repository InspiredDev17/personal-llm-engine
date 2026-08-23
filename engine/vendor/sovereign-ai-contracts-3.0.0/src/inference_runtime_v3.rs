//! Provider-neutral, identity-bound inference runtime contract major version 3.
//!
//! These types contain no scheduler, model loader, qualification authority, or
//! transport authentication. They validate only bounded cross-repository data.

use crate::common::{
    check_len, check_max, check_range, compare_timestamp_values, format_unix_millis,
    timestamp_unix_millis, validate_extensions, validate_reason_code, validate_sha256,
    validate_timestamp, validate_timestamp_window, Extensions, RequiredExtensions,
};
use crate::error::{ContractError, Result, Validate};
use crate::identity::{ModelArtifactId, ModelId, RequestId, TaskId};
use crate::inference::{GenerationParameters, GenerationUsage, InferenceMessage};
use crate::INFERENCE_RUNTIME_CONTRACT_VERSION_V3;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use sha2::{Digest, Sha256};
use std::cmp::Ordering;
use std::collections::BTreeSet;

/// Formats a Unix-millisecond clock value into the canonical UTC timestamp
/// representation used by runtime-v3 documents.
///
/// # Errors
///
/// Returns an error when the value lies beyond the supported RFC3339 year range.
pub fn runtime_v3_timestamp_from_unix_millis(value: u64) -> Result<String> {
    format_unix_millis("runtime_timestamp", value)
}

fn decode_runtime_v3<T: DeserializeOwned + Validate>(source: &[u8]) -> Result<T> {
    let value: T = serde_json::from_slice(source)?;
    value.validate()?;
    Ok(value)
}

fn encode_runtime_v3<T: Serialize + Validate>(value: &T) -> Result<Vec<u8>> {
    value.validate()?;
    let encoded = serde_json::to_value(value)?;
    Ok(serde_json::to_vec(&crate::common::sort_json_value(
        encoded,
    ))?)
}

macro_rules! runtime_v3_codec {
    ($decode:ident, $encode:ident, $type:ty) => {
        /// Strictly decodes and validates one bounded runtime-v3 document.
        ///
        /// # Errors
        ///
        /// Returns an error for malformed JSON or any contract violation.
        pub fn $decode(source: &[u8]) -> Result<$type> {
            decode_runtime_v3(source)
        }

        /// Validates and deterministically serializes one runtime-v3 document.
        ///
        /// # Errors
        ///
        /// Returns an error when the document violates its contract.
        pub fn $encode(value: &$type) -> Result<Vec<u8>> {
            encode_runtime_v3(value)
        }
    };
}

const MAX_MEMORY_BYTES: u64 = 1 << 50;
const MAX_DURATION_MILLIS: u64 = 86_400_000;
const MAX_TOKEN_COUNT: u64 = 10_000_000;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct InferenceRuntimeContractVersionV3;

impl Serialize for InferenceRuntimeContractVersionV3 {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(INFERENCE_RUNTIME_CONTRACT_VERSION_V3)
    }
}

impl<'de> Deserialize<'de> for InferenceRuntimeContractVersionV3 {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        if value == INFERENCE_RUNTIME_CONTRACT_VERSION_V3 {
            Ok(Self)
        } else {
            Err(serde::de::Error::custom(format!(
                "unsupported contract_version {value:?}; expected {INFERENCE_RUNTIME_CONTRACT_VERSION_V3}"
            )))
        }
    }
}

impl Validate for InferenceRuntimeContractVersionV3 {
    fn validate(&self) -> Result<()> {
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ArtifactDigestAlgorithmV3 {
    Sha256,
    Sha384,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeIdentityV3 {
    pub model_id: ModelId,
    pub provider_model_id: String,
    pub artifact_id: ModelArtifactId,
    pub artifact_digest_algorithm: ArtifactDigestAlgorithmV3,
    pub artifact_digest: String,
    pub backend_id: String,
    pub backend_version: String,
    pub backend_instance_id: String,
    pub engine_instance_id: String,
}

impl Validate for RuntimeIdentityV3 {
    fn validate(&self) -> Result<()> {
        validate_runtime_id_v3("provider_model_id", &self.provider_model_id)?;
        match self.artifact_digest_algorithm {
            ArtifactDigestAlgorithmV3::Sha256 => {
                validate_sha256("artifact_digest", &self.artifact_digest)?;
            }
            ArtifactDigestAlgorithmV3::Sha384 => {
                validate_lower_hex_v3("artifact_digest", &self.artifact_digest, 96)?;
            }
        }
        for (field, value) in [
            ("backend_id", self.backend_id.as_str()),
            ("backend_version", self.backend_version.as_str()),
            ("backend_instance_id", self.backend_instance_id.as_str()),
            ("engine_instance_id", self.engine_instance_id.as_str()),
        ] {
            validate_runtime_id_v3(field, value)?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeResourceBudgetV3 {
    pub maximum_input_tokens: u64,
    pub maximum_output_tokens: u64,
    pub maximum_inference_memory_bytes: u64,
    pub maximum_execution_millis: u64,
}

impl Validate for RuntimeResourceBudgetV3 {
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
pub enum RuntimeModelStateV3 {
    Loaded,
    Loadable,
    Draining,
    Unavailable,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeModelStatusV3 {
    pub identity: RuntimeIdentityV3,
    pub state: RuntimeModelStateV3,
    pub available_memory_bytes: u64,
    pub active_requests: u16,
    pub maximum_concurrent_requests: u16,
}

impl Validate for RuntimeModelStatusV3 {
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
pub enum RuntimeStateV3 {
    Available,
    Constrained,
    Draining,
    Unavailable,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum RuntimeAvailabilityKindV3 {
    #[serde(rename = "inference.runtime_availability")]
    RuntimeAvailabilityV3,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeAvailabilityV3 {
    pub contract_version: InferenceRuntimeContractVersionV3,
    pub kind: RuntimeAvailabilityKindV3,
    pub engine_instance_id: String,
    pub status_revision: u64,
    pub state: RuntimeStateV3,
    pub models: Vec<RuntimeModelStatusV3>,
    pub observed_at: String,
    pub expires_at: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extensions: Option<Extensions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub required_extensions: Option<RequiredExtensions>,
}

impl Validate for RuntimeAvailabilityV3 {
    fn validate(&self) -> Result<()> {
        validate_runtime_id_v3("engine_instance_id", &self.engine_instance_id)?;
        check_range("status_revision", self.status_revision, 1, u64::MAX)?;
        check_max("models", self.models.len(), 256)?;
        let mut models = BTreeSet::new();
        let mut provider_models = BTreeSet::new();
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
            if !provider_models.insert(model.identity.provider_model_id.as_str()) {
                return Err(ContractError::field(
                    "models",
                    "contains duplicate provider model identity",
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

impl RuntimeAvailabilityV3 {
    /// Rejects stale or not-yet-observed availability at a deterministic caller-supplied instant.
    ///
    /// # Errors
    ///
    /// Returns an error when the availability is invalid, stale, or not yet observable.
    pub fn validate_fresh_at(&self, now: &str) -> Result<()> {
        self.validate()?;
        validate_timestamp("now", now)?;
        if compare_timestamp_values(now, &self.observed_at)? == Ordering::Less
            || compare_timestamp_values(now, &self.expires_at)? != Ordering::Less
        {
            return Err(ContractError::field(
                "runtime_availability",
                "is not fresh at the supplied instant",
            ));
        }
        Ok(())
    }
    /// Returns the canonical observation instant as Unix milliseconds.
    ///
    /// # Errors
    ///
    /// Returns an error when the stored timestamp is invalid or before 1970.
    pub fn observed_unix_millis(&self) -> Result<u64> {
        timestamp_unix_millis("observed_at", &self.observed_at)
    }

    /// Returns the canonical expiry instant as Unix milliseconds.
    ///
    /// # Errors
    ///
    /// Returns an error when the stored timestamp is invalid or before 1970.
    pub fn expires_unix_millis(&self) -> Result<u64> {
        timestamp_unix_millis("expires_at", &self.expires_at)
    }
}

runtime_v3_codec!(
    decode_runtime_availability_v3,
    encode_runtime_availability_v3,
    RuntimeAvailabilityV3
);

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum ReservationRequestKindV3 {
    #[serde(rename = "inference.reservation_request")]
    ReservationRequest,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct InferenceReservationRequestV3 {
    pub contract_version: InferenceRuntimeContractVersionV3,
    pub kind: ReservationRequestKindV3,
    pub reservation_id: String,
    pub request_id: RequestId,
    pub task_id: TaskId,
    pub identity: RuntimeIdentityV3,
    pub budget: RuntimeResourceBudgetV3,
    pub idempotency_key: String,
    pub created_at: String,
    pub expires_at: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extensions: Option<Extensions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub required_extensions: Option<RequiredExtensions>,
}

impl Validate for InferenceReservationRequestV3 {
    fn validate(&self) -> Result<()> {
        validate_reservation_id_v3(&self.reservation_id)?;
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

impl InferenceReservationRequestV3 {
    /// Returns the canonical creation instant as Unix milliseconds.
    ///
    /// # Errors
    ///
    /// Returns an error when the stored timestamp is invalid or before 1970.
    pub fn created_unix_millis(&self) -> Result<u64> {
        timestamp_unix_millis("created_at", &self.created_at)
    }

    /// Returns the canonical expiry instant as Unix milliseconds.
    ///
    /// # Errors
    ///
    /// Returns an error when the stored timestamp is invalid or before 1970.
    pub fn expires_unix_millis(&self) -> Result<u64> {
        timestamp_unix_millis("expires_at", &self.expires_at)
    }
}

runtime_v3_codec!(
    decode_inference_reservation_request_v3,
    encode_inference_reservation_request_v3,
    InferenceReservationRequestV3
);

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ReservationDecisionV3 {
    Accepted,
    Denied,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum ReservationDecisionKindV3 {
    #[serde(rename = "inference.reservation_decision")]
    ReservationDecisionV3,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct InferenceReservationDecisionV3 {
    pub contract_version: InferenceRuntimeContractVersionV3,
    pub kind: ReservationDecisionKindV3,
    pub reservation_id: String,
    pub request_id: RequestId,
    pub task_id: TaskId,
    pub identity: RuntimeIdentityV3,
    pub decision: ReservationDecisionV3,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub granted_budget: Option<RuntimeResourceBudgetV3>,
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

impl Validate for InferenceReservationDecisionV3 {
    fn validate(&self) -> Result<()> {
        validate_reservation_id_v3(&self.reservation_id)?;
        self.identity.validate()?;
        check_range("status_revision", self.status_revision, 1, u64::MAX)?;
        validate_timestamp("observed_at", &self.observed_at)?;
        match self.decision {
            ReservationDecisionV3::Accepted => {
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
            ReservationDecisionV3::Denied => {
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

impl InferenceReservationDecisionV3 {
    /// Proves that this decision is an exact, non-escalating response to the
    /// supplied request. Individual document validation is not sufficient for
    /// this cross-document authority check.
    ///
    /// # Errors
    ///
    /// Returns an error when either document is invalid or when correlation,
    /// identity, time, or resource authority differs from the request.
    pub fn validate_against(&self, request: &InferenceReservationRequestV3) -> Result<()> {
        request.validate()?;
        self.validate()?;
        if self.reservation_id != request.reservation_id
            || self.request_id != request.request_id
            || self.task_id != request.task_id
            || self.identity != request.identity
        {
            return Err(ContractError::field(
                "reservation_decision",
                "does not match the reservation request correlation or identity",
            ));
        }
        if compare_timestamp_values(&self.observed_at, &request.created_at)? == Ordering::Less
            || compare_timestamp_values(&self.observed_at, &request.expires_at)?
                == Ordering::Greater
        {
            return Err(ContractError::field(
                "observed_at",
                "must fall within the reservation request window",
            ));
        }
        if matches!(self.decision, ReservationDecisionV3::Denied) {
            return Ok(());
        }
        let Some(expiry) = self.expires_at.as_deref() else {
            return Err(ContractError::field(
                "expires_at",
                "is required for accepted decisions",
            ));
        };
        if compare_timestamp_values(expiry, &request.expires_at)? == Ordering::Greater {
            return Err(ContractError::field(
                "expires_at",
                "must not exceed the requested reservation expiry",
            ));
        }
        let Some(granted) = self.granted_budget.as_ref() else {
            return Err(ContractError::field(
                "granted_budget",
                "is required for accepted decisions",
            ));
        };
        if granted.maximum_input_tokens > request.budget.maximum_input_tokens
            || granted.maximum_output_tokens > request.budget.maximum_output_tokens
            || granted.maximum_inference_memory_bytes
                > request.budget.maximum_inference_memory_bytes
            || granted.maximum_execution_millis > request.budget.maximum_execution_millis
        {
            return Err(ContractError::field(
                "granted_budget",
                "must not exceed any requested resource ceiling",
            ));
        }
        Ok(())
    }

    /// Rejects an unaccepted or expired reservation at a deterministic instant.
    ///
    /// # Errors
    ///
    /// Returns an error when the decision is invalid, denied, or outside its usable window.
    pub fn validate_usable_at(&self, now: &str) -> Result<()> {
        self.validate()?;
        if self.decision != ReservationDecisionV3::Accepted {
            return Err(ContractError::field(
                "reservation_decision",
                "denied reservation is not usable",
            ));
        }
        validate_timestamp("now", now)?;
        let expires = self.expires_at.as_ref().ok_or_else(|| {
            ContractError::field("expires_at", "accepted reservation requires expiry")
        })?;
        if compare_timestamp_values(now, &self.observed_at)? == Ordering::Less
            || compare_timestamp_values(now, expires)? != Ordering::Less
        {
            return Err(ContractError::field(
                "reservation_decision",
                "is outside its permitted window",
            ));
        }
        Ok(())
    }

    /// Returns the canonical decision observation instant as Unix milliseconds.
    ///
    /// # Errors
    ///
    /// Returns an error when the stored timestamp is invalid or before 1970.
    pub fn observed_unix_millis(&self) -> Result<u64> {
        timestamp_unix_millis("observed_at", &self.observed_at)
    }

    /// Returns the accepted reservation expiry as Unix milliseconds.
    ///
    /// # Errors
    ///
    /// Returns an error when a present timestamp is invalid or before 1970.
    pub fn expires_unix_millis(&self) -> Result<Option<u64>> {
        self.expires_at
            .as_deref()
            .map(|value| timestamp_unix_millis("expires_at", value))
            .transpose()
    }
}

runtime_v3_codec!(
    decode_inference_reservation_decision_v3,
    encode_inference_reservation_decision_v3,
    InferenceReservationDecisionV3
);

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub enum ExecutionRequestKindV3 {
    #[serde(rename = "inference.execution_request")]
    ExecutionRequest,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct InferenceExecutionRequestV3 {
    pub contract_version: InferenceRuntimeContractVersionV3,
    pub kind: ExecutionRequestKindV3,
    pub reservation_id: String,
    pub request_id: RequestId,
    pub task_id: TaskId,
    pub identity: RuntimeIdentityV3,
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

impl Validate for InferenceExecutionRequestV3 {
    fn validate(&self) -> Result<()> {
        validate_reservation_id_v3(&self.reservation_id)?;
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

impl InferenceExecutionRequestV3 {
    /// Proves that this request consumes the exact accepted reservation without
    /// changing identity, correlation, time, or output scope.
    ///
    /// # Errors
    ///
    /// Returns an error when either document is invalid or when the request
    /// exceeds the accepted reservation's identity, time, or output authority.
    pub fn validate_against(&self, decision: &InferenceReservationDecisionV3) -> Result<()> {
        decision.validate()?;
        self.validate()?;
        if !matches!(decision.decision, ReservationDecisionV3::Accepted) {
            return Err(ContractError::field(
                "reservation_decision",
                "must be accepted before execution",
            ));
        }
        if self.reservation_id != decision.reservation_id
            || self.request_id != decision.request_id
            || self.task_id != decision.task_id
            || self.identity != decision.identity
        {
            return Err(ContractError::field(
                "execution_request",
                "does not match the accepted reservation correlation or identity",
            ));
        }
        let observed = &decision.observed_at;
        let Some(expiry) = decision.expires_at.as_deref() else {
            return Err(ContractError::field(
                "expires_at",
                "is required for accepted decisions",
            ));
        };
        if compare_timestamp_values(&self.created_at, observed)? == Ordering::Less
            || compare_timestamp_values(&self.created_at, expiry)? != Ordering::Less
            || compare_timestamp_values(&self.deadline_at, expiry)? == Ordering::Greater
        {
            return Err(ContractError::field(
                "execution_request",
                "must be created and complete within the accepted reservation window",
            ));
        }
        let Some(granted) = decision.granted_budget.as_ref() else {
            return Err(ContractError::field(
                "granted_budget",
                "is required for accepted decisions",
            ));
        };
        if u64::from(self.parameters.max_output_tokens) > granted.maximum_output_tokens {
            return Err(ContractError::field(
                "parameters.max_output_tokens",
                "must not exceed the granted output ceiling",
            ));
        }
        Ok(())
    }

    /// Returns the canonical request creation instant as Unix milliseconds.
    ///
    /// # Errors
    ///
    /// Returns an error when the stored timestamp is invalid or before 1970.
    pub fn created_unix_millis(&self) -> Result<u64> {
        timestamp_unix_millis("created_at", &self.created_at)
    }

    /// Returns the canonical execution deadline as Unix milliseconds.
    ///
    /// # Errors
    ///
    /// Returns an error when the stored timestamp is invalid or before 1970.
    pub fn deadline_unix_millis(&self) -> Result<u64> {
        timestamp_unix_millis("deadline_at", &self.deadline_at)
    }
}

runtime_v3_codec!(
    decode_inference_execution_request_v3,
    encode_inference_execution_request_v3,
    InferenceExecutionRequestV3
);

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Ord, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeResourceV3 {
    InferenceMemory,
    WallTime,
    CpuTime,
    AcceleratorTime,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeResourceUnitV3 {
    Bytes,
    Milliseconds,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeResourceUsageV3 {
    pub resource: RuntimeResourceV3,
    pub units: u64,
    pub unit: RuntimeResourceUnitV3,
}

impl Validate for RuntimeResourceUsageV3 {
    fn validate(&self) -> Result<()> {
        match (&self.resource, &self.unit) {
            (RuntimeResourceV3::InferenceMemory, RuntimeResourceUnitV3::Bytes) => {
                check_range("units", self.units, 0, MAX_MEMORY_BYTES)
            }
            (
                RuntimeResourceV3::WallTime
                | RuntimeResourceV3::CpuTime
                | RuntimeResourceV3::AcceleratorTime,
                RuntimeResourceUnitV3::Milliseconds,
            ) => check_range("units", self.units, 0, MAX_DURATION_MILLIS),
            _ => Err(ContractError::field("unit", "does not match resource")),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeFailureClassV3 {
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
pub struct RuntimeFailureV3 {
    pub class: RuntimeFailureClassV3,
    pub code: String,
    pub retryable: bool,
}

impl Validate for RuntimeFailureV3 {
    fn validate(&self) -> Result<()> {
        validate_reason_code("failure.code", &self.code)?;
        if matches!(
            self.class,
            RuntimeFailureClassV3::InvalidIdentity
                | RuntimeFailureClassV3::Cancelled
                | RuntimeFailureClassV3::DeadlineExceeded
                | RuntimeFailureClassV3::InvalidRequest
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
pub enum ExecutionStatusV3 {
    Succeeded,
    Failed,
    Cancelled,
    DeadlineExceeded,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum ExecutionResultKindV3 {
    #[serde(rename = "inference.execution_result")]
    ExecutionResult,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct InferenceExecutionResultV3 {
    pub contract_version: InferenceRuntimeContractVersionV3,
    pub kind: ExecutionResultKindV3,
    pub reservation_id: String,
    pub request_id: RequestId,
    pub task_id: TaskId,
    pub identity: RuntimeIdentityV3,
    pub status: ExecutionStatusV3,
    pub started_at: String,
    pub completed_at: String,
    pub usage: GenerationUsage,
    pub resource_usage: Vec<RuntimeResourceUsageV3>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_digest: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub failure: Option<RuntimeFailureV3>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extensions: Option<Extensions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub required_extensions: Option<RequiredExtensions>,
}

impl Validate for InferenceExecutionResultV3 {
    fn validate(&self) -> Result<()> {
        validate_reservation_id_v3(&self.reservation_id)?;
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
            ExecutionStatusV3::Succeeded => {
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
            ExecutionStatusV3::Failed
            | ExecutionStatusV3::Cancelled
            | ExecutionStatusV3::DeadlineExceeded => {
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
                if matches!(self.status, ExecutionStatusV3::Cancelled)
                    != matches!(failure.class, RuntimeFailureClassV3::Cancelled)
                    || matches!(self.status, ExecutionStatusV3::DeadlineExceeded)
                        != matches!(failure.class, RuntimeFailureClassV3::DeadlineExceeded)
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

impl InferenceExecutionResultV3 {
    /// Proves that this result belongs to the request and did not exceed its
    /// identity, deadline, or output-token authority.
    ///
    /// # Errors
    ///
    /// Returns an error when either document is invalid or when the result
    /// changes correlation or identity, misses the deadline, or exceeds output.
    pub fn validate_against(&self, request: &InferenceExecutionRequestV3) -> Result<()> {
        request.validate()?;
        self.validate()?;
        if self.reservation_id != request.reservation_id
            || self.request_id != request.request_id
            || self.task_id != request.task_id
            || self.identity != request.identity
        {
            return Err(ContractError::field(
                "execution_result",
                "does not match the execution request correlation or identity",
            ));
        }
        if compare_timestamp_values(&self.started_at, &request.created_at)? == Ordering::Less
            || compare_timestamp_values(&self.completed_at, &request.deadline_at)?
                == Ordering::Greater
        {
            return Err(ContractError::field(
                "execution_result",
                "must not start before creation and must complete by the request deadline",
            ));
        }
        if self.usage.output_tokens > u64::from(request.parameters.max_output_tokens) {
            return Err(ContractError::field(
                "usage.output_tokens",
                "must not exceed the requested output ceiling",
            ));
        }
        Ok(())
    }

    /// Returns the canonical execution start instant as Unix milliseconds.
    ///
    /// # Errors
    ///
    /// Returns an error when the stored timestamp is invalid or before 1970.
    pub fn started_unix_millis(&self) -> Result<u64> {
        timestamp_unix_millis("started_at", &self.started_at)
    }

    /// Returns the canonical completion instant as Unix milliseconds.
    ///
    /// # Errors
    ///
    /// Returns an error when the stored timestamp is invalid or before 1970.
    pub fn completed_unix_millis(&self) -> Result<u64> {
        timestamp_unix_millis("completed_at", &self.completed_at)
    }
}

runtime_v3_codec!(
    decode_inference_execution_result_v3,
    encode_inference_execution_result_v3,
    InferenceExecutionResultV3
);

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum ReservationReleaseKindV3 {
    #[serde(rename = "inference.reservation_release")]
    ReservationRelease,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct InferenceReservationReleaseV3 {
    pub contract_version: InferenceRuntimeContractVersionV3,
    pub kind: ReservationReleaseKindV3,
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
impl Validate for InferenceReservationReleaseV3 {
    fn validate(&self) -> Result<()> {
        validate_reservation_id_v3(&self.reservation_id)?;
        validate_reason_code("reason_code", &self.reason_code)?;
        validate_timestamp("requested_at", &self.requested_at)?;
        validate_extensions(self.extensions.as_ref(), self.required_extensions.as_ref())
    }
}

impl InferenceReservationReleaseV3 {
    /// Returns the canonical release request instant as Unix milliseconds.
    ///
    /// # Errors
    ///
    /// Returns an error when the stored timestamp is invalid or before 1970.
    pub fn requested_unix_millis(&self) -> Result<u64> {
        timestamp_unix_millis("requested_at", &self.requested_at)
    }
}

runtime_v3_codec!(
    decode_inference_reservation_release_v3,
    encode_inference_reservation_release_v3,
    InferenceReservationReleaseV3
);

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum RuntimeCancellationRequestKindV3 {
    #[serde(rename = "inference.runtime_cancellation_request")]
    RuntimeCancellationRequestV3,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeCancellationRequestV3 {
    pub contract_version: InferenceRuntimeContractVersionV3,
    pub kind: RuntimeCancellationRequestKindV3,
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
impl Validate for RuntimeCancellationRequestV3 {
    fn validate(&self) -> Result<()> {
        validate_reservation_id_v3(&self.reservation_id)?;
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

impl RuntimeCancellationRequestV3 {
    /// Returns the canonical cancellation request instant as Unix milliseconds.
    ///
    /// # Errors
    ///
    /// Returns an error when the stored timestamp is invalid or before 1970.
    pub fn requested_unix_millis(&self) -> Result<u64> {
        timestamp_unix_millis("requested_at", &self.requested_at)
    }
}

runtime_v3_codec!(
    decode_runtime_cancellation_request_v3,
    encode_runtime_cancellation_request_v3,
    RuntimeCancellationRequestV3
);

fn validate_runtime_id_v3(field: &'static str, value: &str) -> Result<()> {
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

fn validate_reservation_id_v3(value: &str) -> Result<()> {
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

fn validate_lower_hex_v3(field: &'static str, value: &str, length: usize) -> Result<()> {
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
    use super::{
        decode_inference_execution_request_v3, decode_inference_execution_result_v3,
        decode_inference_reservation_decision_v3, decode_inference_reservation_release_v3,
        decode_inference_reservation_request_v3, decode_runtime_availability_v3,
        decode_runtime_cancellation_request_v3, encode_inference_execution_request_v3,
        encode_inference_execution_result_v3, encode_inference_reservation_decision_v3,
        encode_inference_reservation_release_v3, encode_inference_reservation_request_v3,
        encode_runtime_availability_v3, encode_runtime_cancellation_request_v3,
        runtime_v3_timestamp_from_unix_millis, InferenceExecutionRequestV3,
        InferenceExecutionResultV3, InferenceReservationDecisionV3, InferenceReservationReleaseV3,
        InferenceReservationRequestV3, RuntimeAvailabilityV3, RuntimeCancellationRequestV3,
    };
    use crate::{identity::ModelId, Result, Validate};

    fn assert_canonical_fixture<T>(
        path: &str,
        source: &[u8],
        decode: fn(&[u8]) -> Result<T>,
        encode: fn(&T) -> Result<Vec<u8>>,
    ) {
        let value = decode(source).expect("decode shared Runtime V3 fixture");
        let actual = encode(&value).expect("encode canonical Runtime V3 fixture");
        let prefix = format!("v1/valid/{path}\t");
        let expected = include_str!("../../fixtures/canonical.jsonl")
            .lines()
            .find_map(|line| line.strip_prefix(&prefix))
            .unwrap_or_else(|| panic!("canonical fixture entry is missing for {path}"));
        assert_eq!(actual, expected.as_bytes(), "canonical bytes for {path}");
    }

    #[test]
    fn shared_valid_fixtures_decode() {
        let availability: RuntimeAvailabilityV3 = serde_json::from_str(include_str!(
            "../../fixtures/v1/valid/inference-v3-runtime-availability.json"
        ))
        .expect("decode availability");
        availability.validate().expect("validate availability");
        let result: InferenceExecutionResultV3 = serde_json::from_str(include_str!(
            "../../fixtures/v1/valid/inference-v3-execution-result.json"
        ))
        .expect("decode result");
        result.validate().expect("validate result");
    }

    #[test]
    fn output_substitution_is_rejected() {
        let result: InferenceExecutionResultV3 = serde_json::from_str(include_str!(
            "../../fixtures/v1/invalid/inference-v3-execution-result-digest-mismatch.json"
        ))
        .expect("decode invalid fixture shape");
        assert!(result.validate().is_err());
    }

    #[test]
    fn duplicate_provider_model_identity_is_rejected() {
        let mut availability: RuntimeAvailabilityV3 = serde_json::from_str(include_str!(
            "../../fixtures/v1/valid/inference-v3-runtime-availability.json"
        ))
        .expect("decode availability");
        let mut duplicate = availability.models[0].clone();
        duplicate.identity.model_id = ModelId::new("mdl_other").expect("model ID");
        availability.models.push(duplicate);
        assert!(availability.validate().is_err());
    }

    #[test]
    fn reservation_execution_chain_rejects_escalation_and_substitution() {
        let request: InferenceReservationRequestV3 = serde_json::from_str(include_str!(
            "../../fixtures/v1/valid/inference-v3-reservation-request.json"
        ))
        .expect("decode reservation request");
        let decision: InferenceReservationDecisionV3 = serde_json::from_str(include_str!(
            "../../fixtures/v1/valid/inference-v3-reservation-decision.json"
        ))
        .expect("decode reservation decision");
        decision
            .validate_against(&request)
            .expect("bind reservation decision");

        let mut escalated = decision.clone();
        escalated
            .granted_budget
            .as_mut()
            .expect("accepted budget")
            .maximum_inference_memory_bytes += 1;
        assert!(escalated.validate_against(&request).is_err());

        let execution: InferenceExecutionRequestV3 = serde_json::from_str(include_str!(
            "../../fixtures/v1/valid/inference-v3-execution-request.json"
        ))
        .expect("decode execution request");
        execution
            .validate_against(&decision)
            .expect("bind execution request");

        let result: InferenceExecutionResultV3 = serde_json::from_str(include_str!(
            "../../fixtures/v1/valid/inference-v3-execution-result.json"
        ))
        .expect("decode result");
        result
            .validate_against(&execution)
            .expect("bind execution result");

        let mut substituted = result.clone();
        substituted.identity.provider_model_id = "runtime-model-2".to_owned();
        assert!(substituted.validate_against(&execution).is_err());

        let mut over_budget = result;
        over_budget.usage.output_tokens = 513;
        over_budget.usage.total_tokens = over_budget.usage.input_tokens + 513;
        assert!(over_budget.validate_against(&execution).is_err());
    }

    #[test]
    fn runtime_timestamp_projection_is_canonical_and_offset_stable() {
        let request: InferenceReservationRequestV3 = serde_json::from_str(include_str!(
            "../../fixtures/v1/valid/inference-v3-reservation-request.json"
        ))
        .expect("decode reservation request");
        let created = request
            .created_unix_millis()
            .expect("project creation time");
        assert_eq!(
            runtime_v3_timestamp_from_unix_millis(created).expect("format creation time"),
            "2026-08-14T12:00:00.000Z"
        );
        let mut offset = request;
        offset.created_at = "2026-08-14T08:00:00-04:00".to_owned();
        assert_eq!(
            offset.created_unix_millis().expect("project offset"),
            created
        );
        assert!(runtime_v3_timestamp_from_unix_millis(u64::MAX).is_err());
    }

    #[test]
    fn terminal_request_timestamp_projections_are_canonical() {
        let release: InferenceReservationReleaseV3 = serde_json::from_str(
            r#"{"contract_version":"3.0.0","kind":"inference.reservation_release","reservation_id":"rsv_fixture-1","request_id":"req_11111111111111111111111111111111","task_id":"tsk_fixture","reason_code":"complete","requested_at":"2026-08-14T08:00:01-04:00"}"#,
        )
        .expect("decode release");
        release.validate().expect("validate release");
        assert_eq!(
            runtime_v3_timestamp_from_unix_millis(
                release.requested_unix_millis().expect("release time")
            )
            .expect("format release time"),
            "2026-08-14T12:00:01.000Z"
        );

        let cancellation: RuntimeCancellationRequestV3 = serde_json::from_str(
            r#"{"contract_version":"3.0.0","kind":"inference.runtime_cancellation_request","reservation_id":"rsv_fixture-1","request_id":"req_11111111111111111111111111111111","task_id":"tsk_fixture","cancellation_request_id":"req_22222222222222222222222222222222","reason_code":"owner_cancelled","requested_at":"2026-08-14T12:00:02Z"}"#,
        )
        .expect("decode cancellation");
        cancellation.validate().expect("validate cancellation");
        assert_eq!(
            cancellation
                .requested_unix_millis()
                .expect("cancellation time"),
            release.requested_unix_millis().expect("release time") + 1_000
        );
    }

    #[test]
    fn strict_runtime_codec_round_trips_and_rejects_duplicates() {
        let source =
            include_bytes!("../../fixtures/v1/valid/inference-v3-reservation-request.json");
        let request = decode_inference_reservation_request_v3(source).expect("strict decode");
        let encoded = encode_inference_reservation_request_v3(&request).expect("encode");
        assert_eq!(
            decode_inference_reservation_request_v3(&encoded).expect("round trip"),
            request
        );
        let duplicated = String::from_utf8(encoded).expect("utf8").replacen(
            '{',
            "{\"contract_version\":\"3.0.0\",",
            1,
        );
        assert!(decode_inference_reservation_request_v3(duplicated.as_bytes()).is_err());
    }

    #[test]
    fn every_runtime_v3_encoder_matches_shared_go_canonical_fixtures() {
        assert_canonical_fixture(
            "inference-v3-runtime-availability.json",
            include_bytes!("../../fixtures/v1/valid/inference-v3-runtime-availability.json"),
            decode_runtime_availability_v3,
            encode_runtime_availability_v3,
        );
        assert_canonical_fixture(
            "inference-v3-reservation-request.json",
            include_bytes!("../../fixtures/v1/valid/inference-v3-reservation-request.json"),
            decode_inference_reservation_request_v3,
            encode_inference_reservation_request_v3,
        );
        assert_canonical_fixture(
            "inference-v3-reservation-decision.json",
            include_bytes!("../../fixtures/v1/valid/inference-v3-reservation-decision.json"),
            decode_inference_reservation_decision_v3,
            encode_inference_reservation_decision_v3,
        );
        assert_canonical_fixture(
            "inference-v3-execution-request.json",
            include_bytes!("../../fixtures/v1/valid/inference-v3-execution-request.json"),
            decode_inference_execution_request_v3,
            encode_inference_execution_request_v3,
        );
        assert_canonical_fixture(
            "inference-v3-execution-request-integral-generation.json",
            include_bytes!(
                "../../fixtures/v1/valid/inference-v3-execution-request-integral-generation.json"
            ),
            decode_inference_execution_request_v3,
            encode_inference_execution_request_v3,
        );
        assert_canonical_fixture(
            "inference-v3-execution-request-negative-zero-generation.json",
            include_bytes!(
                "../../fixtures/v1/valid/inference-v3-execution-request-negative-zero-generation.json"
            ),
            decode_inference_execution_request_v3,
            encode_inference_execution_request_v3,
        );
        assert_canonical_fixture(
            "inference-v3-execution-result.json",
            include_bytes!("../../fixtures/v1/valid/inference-v3-execution-result.json"),
            decode_inference_execution_result_v3,
            encode_inference_execution_result_v3,
        );
        assert_canonical_fixture(
            "inference-v3-reservation-release.json",
            include_bytes!("../../fixtures/v1/valid/inference-v3-reservation-release.json"),
            decode_inference_reservation_release_v3,
            encode_inference_reservation_release_v3,
        );
        assert_canonical_fixture(
            "inference-v3-runtime-cancellation-request.json",
            include_bytes!(
                "../../fixtures/v1/valid/inference-v3-runtime-cancellation-request.json"
            ),
            decode_runtime_cancellation_request_v3,
            encode_runtime_cancellation_request_v3,
        );
    }
}
