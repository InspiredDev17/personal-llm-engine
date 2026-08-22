use crate::common::{
    check_max, check_range, validate_extensions, validate_reason_code, validate_sha256,
    validate_timestamp, ContractVersion, Extensions, RedactedReference, RequiredExtensions,
};
use crate::error::{Result, Validate};
use crate::identity::{ModelArtifactId, ModelId, NodeId, RequestId};
use crate::inference::ModelCapability;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelArtifactFormat {
    Safetensors,
    Gguf,
    Onnx,
    Coreml,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum ModelArtifactDescriptorKind {
    #[serde(rename = "model.artifact_descriptor")]
    ModelArtifactDescriptor,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ModelArtifactDescriptor {
    pub contract_version: ContractVersion,
    pub kind: ModelArtifactDescriptorKind,
    pub model_id: ModelId,
    pub artifact_id: ModelArtifactId,
    pub format: ModelArtifactFormat,
    pub artifact_digest: String,
    pub size_bytes: u64,
    pub source_reference: RedactedReference,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub capabilities: Option<Vec<ModelCapability>>,
    pub created_at: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extensions: Option<Extensions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub required_extensions: Option<RequiredExtensions>,
}

impl Validate for ModelArtifactDescriptor {
    fn validate(&self) -> Result<()> {
        validate_sha256("artifact_digest", &self.artifact_digest)?;
        check_range("size_bytes", self.size_bytes, 1, 1_099_511_627_776)?;
        self.source_reference.validate()?;
        if let Some(capabilities) = &self.capabilities {
            check_max("capabilities", capabilities.len(), 32)?;
            capabilities.iter().try_for_each(Validate::validate)?;
        }
        validate_timestamp("created_at", &self.created_at)?;
        validate_extensions(self.extensions.as_ref(), self.required_extensions.as_ref())
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum ModelActivationRequestKind {
    #[serde(rename = "model.activation_request")]
    ModelActivationRequest,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ModelActivationRequest {
    pub contract_version: ContractVersion,
    pub kind: ModelActivationRequestKind,
    pub request_id: RequestId,
    pub model_id: ModelId,
    pub artifact_id: ModelArtifactId,
    pub target_node_id: NodeId,
    pub artifact_digest: String,
    pub authorization_reference: RedactedReference,
    pub requested_at: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extensions: Option<Extensions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub required_extensions: Option<RequiredExtensions>,
}

impl Validate for ModelActivationRequest {
    fn validate(&self) -> Result<()> {
        validate_sha256("artifact_digest", &self.artifact_digest)?;
        self.authorization_reference.validate()?;
        validate_timestamp("requested_at", &self.requested_at)?;
        validate_extensions(self.extensions.as_ref(), self.required_extensions.as_ref())
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelActivationStatus {
    Pending,
    Loading,
    Active,
    Inactive,
    Failed,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum ModelActivationStateKind {
    #[serde(rename = "model.activation_state")]
    ModelActivationState,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ModelActivationState {
    pub contract_version: ContractVersion,
    pub kind: ModelActivationStateKind,
    pub request_id: RequestId,
    pub model_id: ModelId,
    pub artifact_id: ModelArtifactId,
    pub node_id: NodeId,
    pub state: ModelActivationStatus,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason_code: Option<String>,
    pub observed_at: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extensions: Option<Extensions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub required_extensions: Option<RequiredExtensions>,
}

impl Validate for ModelActivationState {
    fn validate(&self) -> Result<()> {
        if let Some(reason_code) = &self.reason_code {
            validate_reason_code("reason_code", reason_code)?;
        }
        validate_timestamp("observed_at", &self.observed_at)?;
        validate_extensions(self.extensions.as_ref(), self.required_extensions.as_ref())
    }
}
