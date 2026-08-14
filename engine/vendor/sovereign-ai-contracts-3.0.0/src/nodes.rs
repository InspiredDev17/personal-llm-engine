use crate::common::{
    check_max, validate_extensions, validate_reason_code, validate_semver, validate_timestamp,
    validate_timestamp_order, ContractVersion, Extensions, RedactedReference, RequiredExtensions,
};
use crate::error::{ContractError, Result, Validate};
use crate::identity::{ModelArtifactId, ModelId, NodeId, RequestId};
use crate::inference::ResourceStatus;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

#[derive(Clone, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum NodeCapability {
    Inference,
    ModelCache,
    ToolExecutor,
    SecuritySensor,
    MemoryReplica,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum NodeStatus {
    Available,
    Degraded,
    Draining,
    Offline,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum NodeDescriptorKind {
    #[serde(rename = "node.descriptor")]
    NodeDescriptor,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct NodeDescriptor {
    pub contract_version: ContractVersion,
    pub kind: NodeDescriptorKind,
    pub node_id: NodeId,
    pub runtime_version: String,
    pub status: NodeStatus,
    pub capabilities: Vec<NodeCapability>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub resources: Option<Vec<ResourceStatus>>,
    pub attestation_reference: RedactedReference,
    pub observed_at: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extensions: Option<Extensions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub required_extensions: Option<RequiredExtensions>,
}

impl Validate for NodeDescriptor {
    fn validate(&self) -> Result<()> {
        validate_semver("runtime_version", &self.runtime_version)?;
        check_max("capabilities", self.capabilities.len(), 32)?;
        let mut unique = BTreeSet::new();
        for capability in &self.capabilities {
            if !unique.insert(capability) {
                return Err(ContractError::field(
                    "capabilities",
                    "values must be unique",
                ));
            }
        }
        if let Some(resources) = &self.resources {
            check_max("resources", resources.len(), 16)?;
            resources.iter().try_for_each(Validate::validate)?;
        }
        self.attestation_reference.validate()?;
        validate_timestamp("observed_at", &self.observed_at)?;
        validate_extensions(self.extensions.as_ref(), self.required_extensions.as_ref())
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum NodeAction {
    HealthCheck,
    CacheArtifact,
    EvictArtifact,
    ActivateModel,
    DeactivateModel,
    Drain,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum NodeActionIssuer {
    ActionBroker,
    PersonalAiPlatform,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum NodeActionRequestKind {
    #[serde(rename = "node.action_request")]
    NodeActionRequest,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct NodeActionRequest {
    pub contract_version: ContractVersion,
    pub kind: NodeActionRequestKind,
    pub request_id: RequestId,
    pub node_id: NodeId,
    pub action: NodeAction,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_artifact_id: Option<ModelArtifactId>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_id: Option<ModelId>,
    pub authorization_reference: RedactedReference,
    pub issued_by: NodeActionIssuer,
    pub issued_at: String,
    pub expires_at: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extensions: Option<Extensions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub required_extensions: Option<RequiredExtensions>,
}

impl Validate for NodeActionRequest {
    fn validate(&self) -> Result<()> {
        self.authorization_reference.validate()?;
        validate_timestamp("issued_at", &self.issued_at)?;
        validate_timestamp("expires_at", &self.expires_at)?;
        validate_timestamp_order("issued_at", &self.issued_at, "expires_at", &self.expires_at)?;
        validate_extensions(self.extensions.as_ref(), self.required_extensions.as_ref())
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum NodeActionStatus {
    Completed,
    Rejected,
    Expired,
    Failed,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum NodeActionResultKind {
    #[serde(rename = "node.action_result")]
    NodeActionResult,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct NodeActionResult {
    pub contract_version: ContractVersion,
    pub kind: NodeActionResultKind,
    pub request_id: RequestId,
    pub node_id: NodeId,
    pub status: NodeActionStatus,
    pub result_reference: RedactedReference,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason_code: Option<String>,
    pub completed_at: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extensions: Option<Extensions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub required_extensions: Option<RequiredExtensions>,
}

impl Validate for NodeActionResult {
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
    use super::NodeActionRequest;
    use crate::Validate;

    #[test]
    fn node_action_expiry_must_follow_issue() {
        let request: NodeActionRequest = serde_json::from_str(
            r#"{
              "contract_version":"1.0.0","kind":"node.action_request",
              "request_id":"req_nodeaction","node_id":"nod_runtime",
              "action":"health_check",
              "authorization_reference":{"reference_id":"approval:one","digest":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","redaction_state":"reference_only"},
              "issued_by":"action_broker","issued_at":"2026-08-12T00:01:00Z",
              "expires_at":"2026-08-11T23:59:00Z"
            }"#,
        )
        .expect("decode");
        assert!(request.validate().is_err());
    }
}
