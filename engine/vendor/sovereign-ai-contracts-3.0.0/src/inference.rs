use crate::common::{
    check_len, check_max, check_range, validate_extensions, validate_reason_code, validate_semver,
    validate_timestamp, ContractVersion, Extensions, RequiredExtensions,
};
use crate::error::{ContractError, Result, Validate};
use crate::identity::{ModelId, NodeId, RequestId, SessionId, TaskId};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct GenerationParameters {
    pub max_output_tokens: u32,
    pub temperature: f64,
    pub top_p: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stop_sequences: Option<Vec<String>>,
}

impl Validate for GenerationParameters {
    fn validate(&self) -> Result<()> {
        check_range("max_output_tokens", self.max_output_tokens, 1, 131_072)?;
        validate_f64("temperature", self.temperature, 0.0, 2.0, true)?;
        validate_f64("top_p", self.top_p, 0.0, 1.0, false)?;
        if let Some(stops) = &self.stop_sequences {
            check_max("stop_sequences", stops.len(), 16)?;
            let mut unique = BTreeSet::new();
            for stop in stops {
                check_len("stop_sequences", stop, 1, 128)?;
                if !unique.insert(stop) {
                    return Err(ContractError::field(
                        "stop_sequences",
                        "values must be unique",
                    ));
                }
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum InferenceRole {
    System,
    User,
    Assistant,
    Tool,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct InferenceMessage {
    pub role: InferenceRole,
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

impl Validate for InferenceMessage {
    fn validate(&self) -> Result<()> {
        check_len("content", &self.content, 1, 65_536)?;
        if let Some(name) = &self.name {
            check_len("name", name, 1, 64)?;
            validate_identifier_name("name", name)?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum InferenceRequestKind {
    #[serde(rename = "inference.request")]
    InferenceRequest,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct InferenceRequest {
    pub contract_version: ContractVersion,
    pub kind: InferenceRequestKind,
    pub request_id: RequestId,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub session_id: Option<SessionId>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub task_id: Option<TaskId>,
    pub model_id: ModelId,
    pub messages: Vec<InferenceMessage>,
    pub parameters: GenerationParameters,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub deadline_at: Option<String>,
    pub created_at: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extensions: Option<Extensions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub required_extensions: Option<RequiredExtensions>,
}

impl Validate for InferenceRequest {
    fn validate(&self) -> Result<()> {
        check_range("messages", self.messages.len(), 1, 64)?;
        self.messages.iter().try_for_each(Validate::validate)?;
        self.parameters.validate()?;
        if let Some(deadline) = &self.deadline_at {
            validate_timestamp("deadline_at", deadline)?;
        }
        validate_timestamp("created_at", &self.created_at)?;
        validate_extensions(self.extensions.as_ref(), self.required_extensions.as_ref())
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct GenerationUsage {
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub total_tokens: u64,
}

impl Validate for GenerationUsage {
    fn validate(&self) -> Result<()> {
        check_range("input_tokens", self.input_tokens, 0, 10_000_000)?;
        check_range("output_tokens", self.output_tokens, 0, 10_000_000)?;
        check_range("total_tokens", self.total_tokens, 0, 20_000_000)
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum GenerationEventKind {
    #[serde(rename = "generation.event")]
    GenerationEvent,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum GenerationEventType {
    Started,
    TextDelta,
    Completed,
    Cancelled,
    Failed,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    Stop,
    Length,
    Cancelled,
    Error,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct GenerationEvent {
    pub contract_version: ContractVersion,
    pub kind: GenerationEventKind,
    pub request_id: RequestId,
    pub sequence: u32,
    pub event_type: GenerationEventType,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text_delta: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<FinishReason>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<GenerationUsage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error_code: Option<String>,
    pub occurred_at: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extensions: Option<Extensions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub required_extensions: Option<RequiredExtensions>,
}

impl Validate for GenerationEvent {
    fn validate(&self) -> Result<()> {
        if let Some(delta) = &self.text_delta {
            check_len("text_delta", delta, 1, 16_384)?;
        }
        if let Some(usage) = &self.usage {
            usage.validate()?;
        }
        if let Some(error) = &self.error_code {
            validate_reason_code("error_code", error)?;
        }
        match self.event_type {
            GenerationEventType::TextDelta if self.text_delta.is_none() => {
                return Err(ContractError::field(
                    "text_delta",
                    "is required for a text_delta event",
                ));
            }
            GenerationEventType::Completed
                if self.finish_reason.is_none() || self.usage.is_none() =>
            {
                return Err(ContractError::field(
                    "generation_event",
                    "completed events require finish_reason and usage",
                ));
            }
            GenerationEventType::Failed if self.error_code.is_none() => {
                return Err(ContractError::field(
                    "error_code",
                    "is required for a failed event",
                ));
            }
            _ => {}
        }
        validate_timestamp("occurred_at", &self.occurred_at)?;
        validate_extensions(self.extensions.as_ref(), self.required_extensions.as_ref())
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelCapabilityName {
    TextGeneration,
    Embeddings,
    StructuredOutput,
    ToolCalling,
    VisionInput,
    AudioInput,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ModelCapability {
    pub name: ModelCapabilityName,
    pub supported: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub limit: Option<u64>,
}

impl Validate for ModelCapability {
    fn validate(&self) -> Result<()> {
        if let Some(limit) = self.limit {
            check_range("limit", limit, 1, 10_000_000)?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct BackendCapability {
    pub backend_name: String,
    pub backend_version: String,
    pub supports_cancellation: bool,
    pub max_concurrent_requests: u16,
    pub model_capabilities: Vec<ModelCapability>,
}

impl Validate for BackendCapability {
    fn validate(&self) -> Result<()> {
        check_len("backend_name", &self.backend_name, 1, 64)?;
        validate_lower_name("backend_name", &self.backend_name)?;
        validate_semver("backend_version", &self.backend_version)?;
        check_range(
            "max_concurrent_requests",
            self.max_concurrent_requests,
            1,
            u16::MAX,
        )?;
        check_max("model_capabilities", self.model_capabilities.len(), 32)?;
        self.model_capabilities
            .iter()
            .try_for_each(Validate::validate)
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Resource {
    Cpu,
    Memory,
    Gpu,
    Accelerator,
    ModelCache,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ResourceState {
    Available,
    Constrained,
    Exhausted,
    Unavailable,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ResourceUnit {
    Percent,
    Bytes,
    Count,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ResourceStatus {
    pub resource: Resource,
    pub state: ResourceState,
    pub used_units: f64,
    pub total_units: f64,
    pub unit: ResourceUnit,
}

impl Validate for ResourceStatus {
    fn validate(&self) -> Result<()> {
        validate_f64(
            "used_units",
            self.used_units,
            0.0,
            1_000_000_000_000_000.0,
            true,
        )?;
        validate_f64(
            "total_units",
            self.total_units,
            0.0,
            1_000_000_000_000_000.0,
            false,
        )
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum EngineHealthKind {
    #[serde(rename = "engine.health")]
    EngineHealth,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum EngineStatus {
    Healthy,
    Degraded,
    Unavailable,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EngineHealth {
    pub contract_version: ContractVersion,
    pub kind: EngineHealthKind,
    pub node_id: NodeId,
    pub status: EngineStatus,
    pub observed_at: String,
    pub backend: BackendCapability,
    pub resources: Vec<ResourceStatus>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extensions: Option<Extensions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub required_extensions: Option<RequiredExtensions>,
}

impl Validate for EngineHealth {
    fn validate(&self) -> Result<()> {
        validate_timestamp("observed_at", &self.observed_at)?;
        self.backend.validate()?;
        check_max("resources", self.resources.len(), 16)?;
        self.resources.iter().try_for_each(Validate::validate)?;
        validate_extensions(self.extensions.as_ref(), self.required_extensions.as_ref())
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum CancellationRequestKind {
    #[serde(rename = "inference.cancellation_request")]
    CancellationRequest,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CancellationRequest {
    pub contract_version: ContractVersion,
    pub kind: CancellationRequestKind,
    pub request_id: RequestId,
    pub cancellation_request_id: RequestId,
    pub reason_code: String,
    pub requested_at: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extensions: Option<Extensions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub required_extensions: Option<RequiredExtensions>,
}

impl Validate for CancellationRequest {
    fn validate(&self) -> Result<()> {
        validate_reason_code("reason_code", &self.reason_code)?;
        validate_timestamp("requested_at", &self.requested_at)?;
        validate_extensions(self.extensions.as_ref(), self.required_extensions.as_ref())
    }
}

fn validate_f64(
    field: &'static str,
    value: f64,
    minimum: f64,
    maximum: f64,
    include_minimum: bool,
) -> Result<()> {
    let minimum_ok = if include_minimum {
        value >= minimum
    } else {
        value > minimum
    };
    if value.is_finite() && minimum_ok && value <= maximum {
        Ok(())
    } else {
        Err(ContractError::field(
            field,
            format!("out of bounds: {value}"),
        ))
    }
}

fn validate_identifier_name(field: &'static str, value: &str) -> Result<()> {
    let mut bytes = value.bytes();
    if !bytes.next().is_some_and(|byte| byte.is_ascii_alphabetic())
        || !bytes.all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-'))
    {
        Err(ContractError::field(
            field,
            "invalid identifier-name syntax",
        ))
    } else {
        Ok(())
    }
}

fn validate_lower_name(field: &'static str, value: &str) -> Result<()> {
    let mut bytes = value.bytes();
    if !bytes.next().is_some_and(|byte| byte.is_ascii_lowercase())
        || !bytes.all(|byte| {
            byte.is_ascii_lowercase() || byte.is_ascii_digit() || matches!(byte, b'_' | b'-')
        })
    {
        Err(ContractError::field(field, "invalid lowercase name syntax"))
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{GenerationParameters, InferenceRequest};
    use crate::Validate;

    #[test]
    fn closed_request_rejects_unknown_fields() {
        let source = r#"{
          "contract_version":"1.0.0","kind":"inference.request",
          "request_id":"req_ab","model_id":"mdl_ab",
          "messages":[{"role":"user","content":"hello"}],
          "parameters":{"max_output_tokens":1,"temperature":0,"top_p":1},
          "created_at":"2026-08-12T00:00:00Z","surprise":true
        }"#;
        assert!(serde_json::from_str::<InferenceRequest>(source).is_err());
    }

    #[test]
    fn generation_parameters_enforce_bounds_and_uniqueness() {
        let params = GenerationParameters {
            max_output_tokens: 1,
            temperature: 0.0,
            top_p: 1.0,
            seed: None,
            stop_sequences: Some(vec!["done".to_owned(), "done".to_owned()]),
        };
        assert!(params.validate().is_err());
    }

    #[test]
    fn request_validates_nested_messages() {
        let source = r#"{
          "contract_version":"1.0.0","kind":"inference.request",
          "request_id":"req_ab","model_id":"mdl_ab",
          "messages":[{"role":"user","content":"hello"}],
          "parameters":{"max_output_tokens":128,"temperature":0.5,"top_p":1},
          "created_at":"2026-08-12T00:00:00Z"
        }"#;
        let request: InferenceRequest = serde_json::from_str(source).expect("decode");
        request.validate().expect("validate");
    }
}
