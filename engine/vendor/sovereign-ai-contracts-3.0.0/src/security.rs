use crate::common::{
    check_range, validate_extensions, validate_reason_code, validate_sha256, validate_timestamp,
    validate_timestamp_order, Component, ContractVersion, Extensions, RedactedReference,
    RequiredExtensions, ScreenedText,
};
use crate::error::{ContractError, Result, Validate};
use crate::identity::{AgentId, ModelId, NodeId, RequestId, SecurityEventId, SessionId, ToolId};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SecuritySeverity {
    Informational,
    Low,
    Medium,
    High,
    Critical,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceTrust {
    UntrustedModelObservation,
    UntrustedSensorObservation,
    VerifiedControlSignal,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SecurityEvidence {
    pub evidence_reference: RedactedReference,
    pub source_component: Component,
    pub source_trust: EvidenceTrust,
    pub summary: ScreenedText,
    pub observed_at: String,
}

impl Validate for SecurityEvidence {
    fn validate(&self) -> Result<()> {
        self.evidence_reference.validate()?;
        self.summary.validate()?;
        validate_timestamp("observed_at", &self.observed_at)
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SecurityCategory {
    Authentication,
    Authorization,
    DataExposure,
    Integrity,
    Availability,
    MaliciousInput,
    PolicyViolation,
    SupplyChain,
    AnomalousBehavior,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum SecurityEventKind {
    #[serde(rename = "security.event")]
    SecurityEvent,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SecurityEvent {
    pub contract_version: ContractVersion,
    pub kind: SecurityEventKind,
    pub security_event_id: SecurityEventId,
    pub category: SecurityCategory,
    pub severity: SecuritySeverity,
    pub reported_by: Component,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub subject_node_id: Option<NodeId>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub subject_agent_id: Option<AgentId>,
    pub evidence: Vec<SecurityEvidence>,
    pub occurred_at: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extensions: Option<Extensions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub required_extensions: Option<RequiredExtensions>,
}

impl Validate for SecurityEvent {
    fn validate(&self) -> Result<()> {
        check_range("evidence", self.evidence.len(), 1, 32)?;
        self.evidence.iter().try_for_each(Validate::validate)?;
        validate_timestamp("occurred_at", &self.occurred_at)?;
        validate_extensions(self.extensions.as_ref(), self.required_extensions.as_ref())
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SecurityDecisionValue {
    Allow,
    Deny,
    Escalate,
    ContainmentRecommended,
    ContainmentAuthorized,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SecurityAuthorityType {
    SecurityController,
    HumanOperator,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum SecurityDecisionKind {
    #[serde(rename = "security.decision")]
    SecurityDecision,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SecurityDecision {
    pub contract_version: ContractVersion,
    pub kind: SecurityDecisionKind,
    pub decision_id: RequestId,
    pub security_event_id: SecurityEventId,
    pub decision: SecurityDecisionValue,
    pub decided_by_type: SecurityAuthorityType,
    pub decided_by_reference: RedactedReference,
    pub reason_code: String,
    pub evidence_digests: Vec<String>,
    pub decided_at: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expires_at: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extensions: Option<Extensions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub required_extensions: Option<RequiredExtensions>,
}

impl Validate for SecurityDecision {
    fn validate(&self) -> Result<()> {
        self.decided_by_reference.validate()?;
        validate_reason_code("reason_code", &self.reason_code)?;
        check_range("evidence_digests", self.evidence_digests.len(), 1, 32)?;
        let mut unique = BTreeSet::new();
        for digest in &self.evidence_digests {
            validate_sha256("evidence_digests", digest)?;
            if !unique.insert(digest) {
                return Err(ContractError::field(
                    "evidence_digests",
                    "values must be unique",
                ));
            }
        }
        validate_timestamp("decided_at", &self.decided_at)?;
        if let Some(expires_at) = &self.expires_at {
            validate_timestamp("expires_at", expires_at)?;
            validate_timestamp_order("decided_at", &self.decided_at, "expires_at", expires_at)?;
        }
        if self.decision == SecurityDecisionValue::ContainmentAuthorized
            && self.expires_at.is_none()
        {
            return Err(ContractError::field(
                "expires_at",
                "is required for containment_authorized",
            ));
        }
        validate_extensions(self.extensions.as_ref(), self.required_extensions.as_ref())
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ContainmentTargetType {
    Node,
    Tool,
    Session,
    Model,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ContainmentTarget {
    pub target_type: ContainmentTargetType,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub node_id: Option<NodeId>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_id: Option<ToolId>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub session_id: Option<SessionId>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_id: Option<ModelId>,
}

impl Validate for ContainmentTarget {
    fn validate(&self) -> Result<()> {
        let present = usize::from(self.node_id.is_some())
            + usize::from(self.tool_id.is_some())
            + usize::from(self.session_id.is_some())
            + usize::from(self.model_id.is_some());
        if present != 1 {
            return Err(ContractError::field(
                "target",
                "must contain exactly one typed target identifier",
            ));
        }
        let matches = matches!(
            (
                &self.target_type,
                &self.node_id,
                &self.tool_id,
                &self.session_id,
                &self.model_id
            ),
            (ContainmentTargetType::Node, Some(_), None, None, None)
                | (ContainmentTargetType::Tool, None, Some(_), None, None)
                | (ContainmentTargetType::Session, None, None, Some(_), None)
                | (ContainmentTargetType::Model, None, None, None, Some(_))
        );
        if matches {
            Ok(())
        } else {
            Err(ContractError::field(
                "target",
                "identifier does not match target_type",
            ))
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ContainmentAction {
    DisableTool,
    SuspendSession,
    QuarantineNode,
    DisableModel,
    BlockNetworkEgress,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum ContainmentAuthorityState {
    #[serde(rename = "requires_authenticated_external_authorization")]
    RequiresAuthenticatedExternalAuthorization,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum ContainmentRequestKind {
    #[serde(rename = "security.containment_request")]
    ContainmentRequest,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ContainmentRequest {
    pub contract_version: ContractVersion,
    pub kind: ContainmentRequestKind,
    pub containment_request_id: RequestId,
    pub security_event_id: SecurityEventId,
    pub decision_id: RequestId,
    pub requested_by_type: SecurityAuthorityType,
    pub requested_by_reference: RedactedReference,
    pub target: ContainmentTarget,
    pub action: ContainmentAction,
    pub nonce_digest: String,
    pub issued_at: String,
    pub expires_at: String,
    pub authority_state: ContainmentAuthorityState,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extensions: Option<Extensions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub required_extensions: Option<RequiredExtensions>,
}

impl Validate for ContainmentRequest {
    fn validate(&self) -> Result<()> {
        self.requested_by_reference.validate()?;
        self.target.validate()?;
        let action_matches_target = matches!(
            (&self.action, &self.target.target_type),
            (ContainmentAction::DisableTool, ContainmentTargetType::Tool)
                | (
                    ContainmentAction::SuspendSession,
                    ContainmentTargetType::Session
                )
                | (
                    ContainmentAction::QuarantineNode | ContainmentAction::BlockNetworkEgress,
                    ContainmentTargetType::Node
                )
                | (
                    ContainmentAction::DisableModel,
                    ContainmentTargetType::Model
                )
        );
        if !action_matches_target {
            return Err(ContractError::field(
                "action",
                "does not apply to the declared target_type",
            ));
        }
        validate_sha256("nonce_digest", &self.nonce_digest)?;
        validate_timestamp("issued_at", &self.issued_at)?;
        validate_timestamp("expires_at", &self.expires_at)?;
        validate_timestamp_order("issued_at", &self.issued_at, "expires_at", &self.expires_at)?;
        validate_extensions(self.extensions.as_ref(), self.required_extensions.as_ref())
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ContainmentStatus {
    Completed,
    Rejected,
    Expired,
    Failed,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ContainmentExecutor {
    AgentGreen,
    NodeRuntime,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum ContainmentResultKind {
    #[serde(rename = "security.containment_result")]
    ContainmentResult,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ContainmentResult {
    pub contract_version: ContractVersion,
    pub kind: ContainmentResultKind,
    pub containment_request_id: RequestId,
    pub status: ContainmentStatus,
    pub executed_by: ContainmentExecutor,
    pub result_reference: RedactedReference,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason_code: Option<String>,
    pub completed_at: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extensions: Option<Extensions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub required_extensions: Option<RequiredExtensions>,
}

impl Validate for ContainmentResult {
    fn validate(&self) -> Result<()> {
        self.result_reference.validate()?;
        if let Some(reason_code) = &self.reason_code {
            validate_reason_code("reason_code", reason_code)?;
        }
        validate_timestamp("completed_at", &self.completed_at)?;
        validate_extensions(self.extensions.as_ref(), self.required_extensions.as_ref())
    }
}

#[cfg(test)]
mod tests {
    use super::{ContainmentRequest, ContainmentTarget, SecurityDecision};
    use crate::Validate;

    #[test]
    fn target_type_requires_matching_typed_id() {
        let target: ContainmentTarget =
            serde_json::from_str(r#"{"target_type":"node","tool_id":"tol_ab"}"#).expect("decode");
        assert!(target.validate().is_err());
    }

    #[test]
    fn authorized_containment_decision_expires() {
        let decision: SecurityDecision = serde_json::from_str(
            r#"{
              "contract_version":"1.0.0","kind":"security.decision",
              "decision_id":"req_decision","security_event_id":"sec_event",
              "decision":"containment_authorized","decided_by_type":"human_operator",
              "decided_by_reference":{"reference_id":"operator:one","digest":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","redaction_state":"reference_only"},
              "reason_code":"policy.violation",
              "evidence_digests":["bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"],
              "decided_at":"2026-08-12T00:00:00Z"
            }"#,
        )
        .expect("decode");
        assert!(decision.validate().is_err());
    }

    #[test]
    fn containment_action_must_match_target_and_expire_after_issue() {
        let request: ContainmentRequest = serde_json::from_str(
            r#"{
              "contract_version":"1.0.0","kind":"security.containment_request",
              "containment_request_id":"req_contain","security_event_id":"sec_event",
              "decision_id":"req_decision","requested_by_type":"human_operator",
              "requested_by_reference":{"reference_id":"operator:one","digest":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","redaction_state":"reference_only"},
              "target":{"target_type":"tool","tool_id":"tol_terminal"},
              "action":"quarantine_node",
              "nonce_digest":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
              "issued_at":"2026-08-12T00:01:00Z","expires_at":"2026-08-12T00:01:00Z",
              "authority_state":"requires_authenticated_external_authorization"
            }"#,
        )
        .expect("decode");
        assert!(request.validate().is_err());
    }
}
