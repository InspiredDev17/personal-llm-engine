use crate::common::{
    check_len, check_range, validate_extensions, validate_reason_code, validate_sha256,
    validate_timestamp, Component, ContractVersion, Extensions, RedactedReference,
    RequiredExtensions,
};
use crate::error::{ContractError, Result, Validate};
use crate::identity::{
    AuditEventId, MemoryId, ModelArtifactId, ModelId, NodeId, RequestId, SecurityEventId, ToolId,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum EventStatus {
    Requested,
    Accepted,
    Completed,
    Denied,
    Cancelled,
    Failed,
    Expired,
}

macro_rules! payload_kind {
    ($name:ident, $wire:literal, $variant:ident) => {
        #[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
        pub enum $name {
            #[serde(rename = $wire)]
            $variant,
        }
    };
}

payload_kind!(InferencePayloadType, "inference", Inference);
payload_kind!(
    MemoryRetrievalPayloadType,
    "memory_retrieval",
    MemoryRetrieval
);
payload_kind!(MemoryWritePayloadType, "memory_write", MemoryWrite);
payload_kind!(ToolRequestPayloadType, "tool_request", ToolRequest);
payload_kind!(ToolExecutionPayloadType, "tool_execution", ToolExecution);
payload_kind!(ApprovalPayloadType, "approval", Approval);
payload_kind!(
    SecurityDecisionPayloadType,
    "security_decision",
    SecurityDecision
);
payload_kind!(ModelImportPayloadType, "model_import", ModelImport);
payload_kind!(
    ModelActivationPayloadType,
    "model_activation",
    ModelActivation
);
payload_kind!(NodeActionPayloadType, "node_action", NodeAction);
payload_kind!(
    ConfigurationChangePayloadType,
    "configuration_change",
    ConfigurationChange
);

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct InferenceAuditPayload {
    pub payload_type: InferencePayloadType,
    pub request_id: RequestId,
    pub model_id: ModelId,
    pub status: EventStatus,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_digest: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_digest: Option<String>,
}

impl Validate for InferenceAuditPayload {
    fn validate(&self) -> Result<()> {
        validate_optional_digest("input_digest", self.input_digest.as_deref())?;
        validate_optional_digest("output_digest", self.output_digest.as_deref())
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct MemoryRetrievalAuditPayload {
    pub payload_type: MemoryRetrievalPayloadType,
    pub request_id: RequestId,
    pub memory_ids: Vec<MemoryId>,
    pub status: EventStatus,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub query_digest: Option<String>,
}

impl Validate for MemoryRetrievalAuditPayload {
    fn validate(&self) -> Result<()> {
        check_unique("memory_ids", &self.memory_ids, 0, 100)?;
        validate_optional_digest("query_digest", self.query_digest.as_deref())
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct MemoryWriteAuditPayload {
    pub payload_type: MemoryWritePayloadType,
    pub proposal_id: RequestId,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub memory_id: Option<MemoryId>,
    pub status: EventStatus,
    pub proposal_digest: String,
}

impl Validate for MemoryWriteAuditPayload {
    fn validate(&self) -> Result<()> {
        validate_sha256("proposal_digest", &self.proposal_digest)
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ToolRequestAuditPayload {
    pub payload_type: ToolRequestPayloadType,
    pub request_id: RequestId,
    pub tool_id: ToolId,
    pub operation: String,
    pub status: EventStatus,
    pub request_digest: String,
}

impl Validate for ToolRequestAuditPayload {
    fn validate(&self) -> Result<()> {
        validate_operation_name(&self.operation)?;
        validate_sha256("request_digest", &self.request_digest)
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ToolExecutionAuditPayload {
    pub payload_type: ToolExecutionPayloadType,
    pub execution_id: RequestId,
    pub request_id: RequestId,
    pub tool_id: ToolId,
    pub status: EventStatus,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result_digest: Option<String>,
}

impl Validate for ToolExecutionAuditPayload {
    fn validate(&self) -> Result<()> {
        validate_optional_digest("result_digest", self.result_digest.as_deref())
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ApprovalDecision {
    Approved,
    Denied,
    Expired,
    Revoked,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ApprovalAuditPayload {
    pub payload_type: ApprovalPayloadType,
    pub approval_id: RequestId,
    pub request_id: RequestId,
    pub decision: ApprovalDecision,
    pub approver_reference: RedactedReference,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason_code: Option<String>,
}

impl Validate for ApprovalAuditPayload {
    fn validate(&self) -> Result<()> {
        self.approver_reference.validate()?;
        if let Some(reason_code) = &self.reason_code {
            validate_reason_code("reason_code", reason_code)?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum AuditSecurityDecision {
    Allow,
    Deny,
    Escalate,
    ContainmentRecommended,
    ContainmentAuthorized,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SecurityDecisionAuditPayload {
    pub payload_type: SecurityDecisionPayloadType,
    pub decision_id: RequestId,
    pub security_event_id: SecurityEventId,
    pub decision: AuditSecurityDecision,
    pub decision_digest: String,
}

impl Validate for SecurityDecisionAuditPayload {
    fn validate(&self) -> Result<()> {
        validate_sha256("decision_digest", &self.decision_digest)
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ModelImportAuditPayload {
    pub payload_type: ModelImportPayloadType,
    pub model_id: ModelId,
    pub artifact_id: ModelArtifactId,
    pub artifact_digest: String,
    pub status: EventStatus,
}

impl Validate for ModelImportAuditPayload {
    fn validate(&self) -> Result<()> {
        validate_sha256("artifact_digest", &self.artifact_digest)
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ModelActivationAuditPayload {
    pub payload_type: ModelActivationPayloadType,
    pub request_id: RequestId,
    pub model_id: ModelId,
    pub artifact_id: ModelArtifactId,
    pub node_id: NodeId,
    pub status: EventStatus,
}

impl Validate for ModelActivationAuditPayload {
    fn validate(&self) -> Result<()> {
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum AuditNodeAction {
    HealthCheck,
    CacheArtifact,
    EvictArtifact,
    ActivateModel,
    DeactivateModel,
    Drain,
    SecurityContainment,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct NodeActionAuditPayload {
    pub payload_type: NodeActionPayloadType,
    pub request_id: RequestId,
    pub node_id: NodeId,
    pub action: AuditNodeAction,
    pub status: EventStatus,
}

impl Validate for NodeActionAuditPayload {
    fn validate(&self) -> Result<()> {
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ConfigurationChangeAuditPayload {
    pub payload_type: ConfigurationChangePayloadType,
    pub change_id: RequestId,
    pub component: Component,
    pub changed_keys: Vec<String>,
    pub before_digest: String,
    pub after_digest: String,
    pub authorization_reference: RedactedReference,
}

impl Validate for ConfigurationChangeAuditPayload {
    fn validate(&self) -> Result<()> {
        check_range("changed_keys", self.changed_keys.len(), 1, 64)?;
        let mut unique = BTreeSet::new();
        for key in &self.changed_keys {
            check_len("changed_keys", key, 1, 128)?;
            validate_configuration_key(key)?;
            if !unique.insert(key) {
                return Err(ContractError::field(
                    "changed_keys",
                    "values must be unique",
                ));
            }
        }
        validate_sha256("before_digest", &self.before_digest)?;
        validate_sha256("after_digest", &self.after_digest)?;
        self.authorization_reference.validate()
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(untagged)]
pub enum AuditPayload {
    Inference(InferenceAuditPayload),
    MemoryRetrieval(MemoryRetrievalAuditPayload),
    MemoryWrite(MemoryWriteAuditPayload),
    ToolRequest(ToolRequestAuditPayload),
    ToolExecution(ToolExecutionAuditPayload),
    Approval(ApprovalAuditPayload),
    SecurityDecision(SecurityDecisionAuditPayload),
    ModelImport(ModelImportAuditPayload),
    ModelActivation(ModelActivationAuditPayload),
    NodeAction(NodeActionAuditPayload),
    ConfigurationChange(ConfigurationChangeAuditPayload),
}

impl Validate for AuditPayload {
    fn validate(&self) -> Result<()> {
        match self {
            Self::Inference(value) => value.validate(),
            Self::MemoryRetrieval(value) => value.validate(),
            Self::MemoryWrite(value) => value.validate(),
            Self::ToolRequest(value) => value.validate(),
            Self::ToolExecution(value) => value.validate(),
            Self::Approval(value) => value.validate(),
            Self::SecurityDecision(value) => value.validate(),
            Self::ModelImport(value) => value.validate(),
            Self::ModelActivation(value) => value.validate(),
            Self::NodeAction(value) => value.validate(),
            Self::ConfigurationChange(value) => value.validate(),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum AuditEventType {
    Inference,
    MemoryRetrieval,
    MemoryWrite,
    ToolRequest,
    ToolExecution,
    Approval,
    SecurityDecision,
    ModelImport,
    ModelActivation,
    NodeAction,
    ConfigurationChange,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum AuditEventKind {
    #[serde(rename = "audit.event")]
    AuditEvent,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum ImmutabilityState {
    #[serde(rename = "append_only_event")]
    AppendOnlyEvent,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct AuditEvent {
    pub contract_version: ContractVersion,
    pub kind: AuditEventKind,
    pub audit_event_id: AuditEventId,
    pub event_type: AuditEventType,
    pub sequence: u64,
    pub recorded_by: Component,
    pub occurred_at: String,
    pub recorded_at: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub previous_event_digest: Option<String>,
    pub event_digest: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub correction_of: Option<AuditEventId>,
    pub immutability_state: ImmutabilityState,
    pub payload: AuditPayload,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extensions: Option<Extensions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub required_extensions: Option<RequiredExtensions>,
}

impl Validate for AuditEvent {
    fn validate(&self) -> Result<()> {
        check_range("sequence", self.sequence, 0, 9_007_199_254_740_991)?;
        validate_timestamp("occurred_at", &self.occurred_at)?;
        validate_timestamp("recorded_at", &self.recorded_at)?;
        validate_optional_digest(
            "previous_event_digest",
            self.previous_event_digest.as_deref(),
        )?;
        validate_sha256("event_digest", &self.event_digest)?;
        self.payload.validate()?;
        let matches = matches!(
            (&self.event_type, &self.payload),
            (AuditEventType::Inference, AuditPayload::Inference(_))
                | (
                    AuditEventType::MemoryRetrieval,
                    AuditPayload::MemoryRetrieval(_)
                )
                | (AuditEventType::MemoryWrite, AuditPayload::MemoryWrite(_))
                | (AuditEventType::ToolRequest, AuditPayload::ToolRequest(_))
                | (
                    AuditEventType::ToolExecution,
                    AuditPayload::ToolExecution(_)
                )
                | (AuditEventType::Approval, AuditPayload::Approval(_))
                | (
                    AuditEventType::SecurityDecision,
                    AuditPayload::SecurityDecision(_)
                )
                | (AuditEventType::ModelImport, AuditPayload::ModelImport(_))
                | (
                    AuditEventType::ModelActivation,
                    AuditPayload::ModelActivation(_)
                )
                | (AuditEventType::NodeAction, AuditPayload::NodeAction(_))
                | (
                    AuditEventType::ConfigurationChange,
                    AuditPayload::ConfigurationChange(_)
                )
        );
        if !matches {
            return Err(ContractError::field(
                "payload",
                "payload_type must match event_type",
            ));
        }
        validate_extensions(self.extensions.as_ref(), self.required_extensions.as_ref())
    }
}

fn validate_optional_digest(field: &'static str, digest: Option<&str>) -> Result<()> {
    if let Some(digest) = digest {
        validate_sha256(field, digest)?;
    }
    Ok(())
}

fn check_unique<T>(field: &'static str, values: &[T], minimum: usize, maximum: usize) -> Result<()>
where
    T: Ord,
{
    check_range(field, values.len(), minimum, maximum)?;
    let mut unique = BTreeSet::new();
    for value in values {
        if !unique.insert(value) {
            return Err(ContractError::field(field, "values must be unique"));
        }
    }
    Ok(())
}

fn validate_operation_name(value: &str) -> Result<()> {
    check_len("operation", value, 3, 96)?;
    let segments: Vec<&str> = value.split(['.', '_', '-']).collect();
    if segments.len() < 2
        || segments.iter().any(|segment| {
            let mut bytes = segment.bytes();
            !bytes.next().is_some_and(|byte| byte.is_ascii_lowercase())
                || !bytes.all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit())
        })
    {
        Err(ContractError::field("operation", "invalid operation name"))
    } else {
        Ok(())
    }
}

fn validate_configuration_key(value: &str) -> Result<()> {
    let mut bytes = value.bytes();
    if !bytes.next().is_some_and(|byte| byte.is_ascii_alphabetic())
        || !bytes.all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'.' | b'-'))
    {
        Err(ContractError::field(
            "changed_keys",
            "invalid configuration key",
        ))
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::AuditEvent;
    use crate::Validate;

    #[test]
    fn event_type_must_match_payload_type() {
        let event: AuditEvent = serde_json::from_str(
            r#"{
              "contract_version":"1.0.0","kind":"audit.event",
              "audit_event_id":"aud_event","event_type":"inference","sequence":1,
              "recorded_by":"personal_ai_platform",
              "occurred_at":"2026-08-12T00:00:00Z","recorded_at":"2026-08-12T00:00:01Z",
              "event_digest":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
              "immutability_state":"append_only_event",
              "payload":{"payload_type":"memory_write","proposal_id":"req_proposal","status":"requested","proposal_digest":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"}
            }"#,
        )
        .expect("decode");
        assert!(event.validate().is_err());
    }

    #[test]
    fn payload_is_closed() {
        let source = r#"{
          "contract_version":"1.0.0","kind":"audit.event",
          "audit_event_id":"aud_event","event_type":"inference","sequence":1,
          "recorded_by":"personal_ai_platform",
          "occurred_at":"2026-08-12T00:00:00Z","recorded_at":"2026-08-12T00:00:01Z",
          "event_digest":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
          "immutability_state":"append_only_event",
          "payload":{"payload_type":"inference","request_id":"req_one","model_id":"mdl_one","status":"requested","unexpected":true}
        }"#;
        assert!(serde_json::from_str::<AuditEvent>(source).is_err());
    }
}
