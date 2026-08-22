use crate::common::{
    check_len, check_max, check_range, validate_extensions, validate_reason_code, validate_sha256,
    validate_timestamp, validate_timestamp_order, ContractVersion, Extensions, RedactedReference,
    RequiredExtensions, ScreenedText,
};
use crate::error::{ContractError, Result, Validate};
use crate::identity::{AgentId, RequestId, ToolId};
use crate::memory::SensitivityClass;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(untagged)]
pub enum ToolArgumentValue {
    String(String),
    Integer(i64),
    Number(f64),
    Boolean(bool),
    StringList(Vec<String>),
}

impl Validate for ToolArgumentValue {
    fn validate(&self) -> Result<()> {
        match self {
            Self::String(value) => validate_untrusted_string("value", value, 4_096),
            Self::Integer(value) => check_range(
                "value",
                *value,
                -9_007_199_254_740_991,
                9_007_199_254_740_991,
            ),
            Self::Number(value) => {
                if value.is_finite() && (-1_000_000_000_000.0..=1_000_000_000_000.0).contains(value)
                {
                    Ok(())
                } else {
                    Err(ContractError::field(
                        "value",
                        "numeric argument out of bounds",
                    ))
                }
            }
            Self::Boolean(_) => Ok(()),
            Self::StringList(values) => {
                check_max("value", values.len(), 32)?;
                values
                    .iter()
                    .try_for_each(|value| validate_untrusted_string("value", value, 1_024))
            }
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolArgumentType {
    String,
    Integer,
    Number,
    Boolean,
    StringList,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ToolArgument {
    pub name: String,
    pub value_type: ToolArgumentType,
    pub value: ToolArgumentValue,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sensitivity: Option<SensitivityClass>,
}

impl Validate for ToolArgument {
    fn validate(&self) -> Result<()> {
        validate_argument_name(&self.name)?;
        self.value.validate()?;
        let matches = matches!(
            (&self.value_type, &self.value),
            (ToolArgumentType::String, ToolArgumentValue::String(_))
                | (ToolArgumentType::Integer, ToolArgumentValue::Integer(_))
                | (
                    ToolArgumentType::Number,
                    ToolArgumentValue::Number(_) | ToolArgumentValue::Integer(_)
                )
                | (ToolArgumentType::Boolean, ToolArgumentValue::Boolean(_))
                | (
                    ToolArgumentType::StringList,
                    ToolArgumentValue::StringList(_)
                )
        );
        if matches {
            Ok(())
        } else {
            Err(ContractError::field(
                "value",
                "does not match declared value_type",
            ))
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ToolArgumentDescriptor {
    pub name: String,
    pub value_type: ToolArgumentType,
    pub required: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_length: Option<u16>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
}

impl Validate for ToolArgumentDescriptor {
    fn validate(&self) -> Result<()> {
        validate_argument_name(&self.name)?;
        if let Some(maximum) = self.max_length {
            check_range("max_length", maximum, 1, 4_096)?;
        }
        if let Some(description) = &self.description {
            check_len("description", description, 1, 512)?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PermissionScope {
    Read,
    Write,
    Network,
    Device,
    Compute,
    ExternalService,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ToolPermission {
    pub name: String,
    pub scope: PermissionScope,
}

impl Validate for ToolPermission {
    fn validate(&self) -> Result<()> {
        validate_operation_name("name", &self.name)
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ApprovalMode {
    None,
    User,
    Administrator,
    TwoPerson,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ApprovalRequirement {
    pub mode: ApprovalMode,
    pub reason_code: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub freshness_seconds: Option<u32>,
}

impl Validate for ApprovalRequirement {
    fn validate(&self) -> Result<()> {
        validate_reason_code("reason_code", &self.reason_code)?;
        if let Some(freshness) = self.freshness_seconds {
            check_range("freshness_seconds", freshness, 1, 86_400)?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum AuthorityMetadataState {
    #[serde(rename = "metadata_only_not_authorization")]
    MetadataOnlyNotAuthorization,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ExecutionCapability {
    pub tool_id: ToolId,
    pub operation: String,
    pub permissions: Vec<ToolPermission>,
    pub approval_requirement: ApprovalRequirement,
    pub authority_state: AuthorityMetadataState,
}

impl Validate for ExecutionCapability {
    fn validate(&self) -> Result<()> {
        validate_operation_name("operation", &self.operation)?;
        check_max("permissions", self.permissions.len(), 32)?;
        self.permissions.iter().try_for_each(Validate::validate)?;
        self.approval_requirement.validate()
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ToolOperation {
    pub capability: ExecutionCapability,
    pub arguments: Vec<ToolArgumentDescriptor>,
}

impl Validate for ToolOperation {
    fn validate(&self) -> Result<()> {
        self.capability.validate()?;
        check_max("arguments", self.arguments.len(), 64)?;
        self.arguments.iter().try_for_each(Validate::validate)
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum ToolDescriptorKind {
    #[serde(rename = "tool.descriptor")]
    ToolDescriptor,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ToolDescriptor {
    pub contract_version: ContractVersion,
    pub kind: ToolDescriptorKind,
    pub tool_id: ToolId,
    pub name: String,
    pub description: String,
    pub operations: Vec<ToolOperation>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extensions: Option<Extensions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub required_extensions: Option<RequiredExtensions>,
}

impl Validate for ToolDescriptor {
    fn validate(&self) -> Result<()> {
        check_len("name", &self.name, 1, 96)?;
        check_len("description", &self.description, 1, 512)?;
        check_range("operations", self.operations.len(), 1, 64)?;
        self.operations.iter().try_for_each(Validate::validate)?;
        validate_extensions(self.extensions.as_ref(), self.required_extensions.as_ref())
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum ToolRequestKind {
    #[serde(rename = "tool.request")]
    ToolRequest,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ToolRequest {
    pub contract_version: ContractVersion,
    pub kind: ToolRequestKind,
    pub request_id: RequestId,
    pub tool_id: ToolId,
    pub operation: String,
    pub requested_by: AgentId,
    pub arguments: Vec<ToolArgument>,
    pub requested_permissions: Vec<ToolPermission>,
    pub requested_at: String,
    pub trust_state: ToolRequestTrustState,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extensions: Option<Extensions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub required_extensions: Option<RequiredExtensions>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum ToolRequestTrustState {
    #[serde(rename = "untrusted_intent")]
    UntrustedIntent,
}

impl Validate for ToolRequest {
    fn validate(&self) -> Result<()> {
        validate_operation_name("operation", &self.operation)?;
        check_max("arguments", self.arguments.len(), 64)?;
        self.arguments.iter().try_for_each(Validate::validate)?;
        check_max(
            "requested_permissions",
            self.requested_permissions.len(),
            32,
        )?;
        self.requested_permissions
            .iter()
            .try_for_each(Validate::validate)?;
        validate_timestamp("requested_at", &self.requested_at)?;
        validate_extensions(self.extensions.as_ref(), self.required_extensions.as_ref())
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum ExecutionRequestKind {
    #[serde(rename = "tool.execution_request")]
    ExecutionRequest,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum ExecutionIssuer {
    #[serde(rename = "action_broker")]
    ActionBroker,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ExecutionRequest {
    pub contract_version: ContractVersion,
    pub kind: ExecutionRequestKind,
    pub execution_id: RequestId,
    pub request_id: RequestId,
    pub tool_id: ToolId,
    pub operation: String,
    pub request_digest: String,
    pub authorization_reference: RedactedReference,
    pub issued_by: ExecutionIssuer,
    pub issued_at: String,
    pub expires_at: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extensions: Option<Extensions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub required_extensions: Option<RequiredExtensions>,
}

impl Validate for ExecutionRequest {
    fn validate(&self) -> Result<()> {
        validate_operation_name("operation", &self.operation)?;
        if is_containment_operation(&self.operation) {
            return Err(ContractError::field(
                "operation",
                "containment cannot be executed through the tool protocol",
            ));
        }
        validate_sha256("request_digest", &self.request_digest)?;
        self.authorization_reference.validate()?;
        validate_timestamp("issued_at", &self.issued_at)?;
        validate_timestamp("expires_at", &self.expires_at)?;
        validate_timestamp_order("issued_at", &self.issued_at, "expires_at", &self.expires_at)?;
        validate_extensions(self.extensions.as_ref(), self.required_extensions.as_ref())
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum ExecutionResultKind {
    #[serde(rename = "tool.execution_result")]
    ExecutionResult,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum ExecutionSuccessStatus {
    #[serde(rename = "succeeded")]
    Succeeded,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ExecutionResult {
    pub contract_version: ContractVersion,
    pub kind: ExecutionResultKind,
    pub execution_id: RequestId,
    pub status: ExecutionSuccessStatus,
    pub output: ScreenedText,
    pub completed_at: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extensions: Option<Extensions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub required_extensions: Option<RequiredExtensions>,
}

impl Validate for ExecutionResult {
    fn validate(&self) -> Result<()> {
        self.output.validate()?;
        validate_timestamp("completed_at", &self.completed_at)?;
        validate_extensions(self.extensions.as_ref(), self.required_extensions.as_ref())
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum ExecutionErrorKind {
    #[serde(rename = "tool.execution_error")]
    ExecutionError,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum ExecutionFailureStatus {
    #[serde(rename = "failed")]
    Failed,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ExecutionError {
    pub contract_version: ContractVersion,
    pub kind: ExecutionErrorKind,
    pub execution_id: RequestId,
    pub status: ExecutionFailureStatus,
    pub code: String,
    pub message: String,
    pub retryable: bool,
    pub failed_at: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extensions: Option<Extensions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub required_extensions: Option<RequiredExtensions>,
}

impl Validate for ExecutionError {
    fn validate(&self) -> Result<()> {
        validate_reason_code("code", &self.code)?;
        check_len("message", &self.message, 1, 1_024)?;
        validate_timestamp("failed_at", &self.failed_at)?;
        validate_extensions(self.extensions.as_ref(), self.required_extensions.as_ref())
    }
}

fn validate_argument_name(value: &str) -> Result<()> {
    check_len("name", value, 1, 64)?;
    let mut bytes = value.bytes();
    if !bytes.next().is_some_and(|byte| byte.is_ascii_lowercase())
        || !bytes.all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'_')
    {
        Err(ContractError::field("name", "invalid argument name"))
    } else {
        Ok(())
    }
}

fn validate_operation_name(field: &'static str, value: &str) -> Result<()> {
    check_len(field, value, 3, 96)?;
    let segments: Vec<&str> = value.split(['.', '_', '-']).collect();
    if segments.len() < 2
        || segments.iter().any(|segment| {
            let mut bytes = segment.bytes();
            !bytes.next().is_some_and(|byte| byte.is_ascii_lowercase())
                || !bytes.all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit())
        })
    {
        Err(ContractError::field(field, "invalid operation name"))
    } else {
        Ok(())
    }
}

fn validate_untrusted_string(field: &'static str, value: &str, maximum: usize) -> Result<()> {
    check_len(field, value, 1, maximum)?;
    if value.chars().any(char::is_control) {
        Err(ContractError::field(
            field,
            "control characters are forbidden",
        ))
    } else {
        Ok(())
    }
}

fn is_containment_operation(operation: &str) -> bool {
    let mut segments = operation.split(['.', '_', '-']);
    let first = segments.next().unwrap_or_default();
    let candidate = if first == "security" {
        segments.next().unwrap_or_default()
    } else {
        first
    };
    matches!(
        candidate,
        "contain" | "containment" | "isolate" | "quarantine" | "terminate"
    )
}

#[cfg(test)]
mod tests {
    use super::{ExecutionRequest, ToolArgument};
    use crate::Validate;

    #[test]
    fn tool_argument_type_must_match_value() {
        let argument: ToolArgument =
            serde_json::from_str(r#"{"name":"count","value_type":"integer","value":"five"}"#)
                .expect("strict shape");
        assert!(argument.validate().is_err());
    }

    #[test]
    fn number_accepts_integral_json_but_integer_rejects_fractional_json() {
        let number: ToolArgument =
            serde_json::from_str(r#"{"name":"amount","value_type":"number","value":1}"#)
                .expect("decode");
        assert!(number.validate().is_ok());

        let integer: ToolArgument =
            serde_json::from_str(r#"{"name":"amount","value_type":"integer","value":1.5}"#)
                .expect("decode");
        assert!(integer.validate().is_err());
    }

    #[test]
    fn execution_request_cannot_convey_containment_authority() {
        let request: ExecutionRequest = serde_json::from_str(
            r#"{
              "contract_version":"1.0.0","kind":"tool.execution_request",
              "execution_id":"req_execute","request_id":"req_original",
              "tool_id":"tol_security","operation":"security.quarantine.node",
              "request_digest":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
              "authorization_reference":{"reference_id":"auth:one","digest":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb","redaction_state":"reference_only"},
              "issued_by":"action_broker","issued_at":"2026-08-12T00:00:00Z",
              "expires_at":"2026-08-12T00:01:00Z"
            }"#,
        )
        .expect("decode");
        assert!(request.validate().is_err());
    }

    #[test]
    fn execution_authorization_expiry_must_follow_issue() {
        let request: ExecutionRequest = serde_json::from_str(
            r#"{
              "contract_version":"1.0.0","kind":"tool.execution_request",
              "execution_id":"req_execute","request_id":"req_original",
              "tool_id":"tol_terminal","operation":"system.run_command",
              "request_digest":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
              "authorization_reference":{"reference_id":"auth:one","digest":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb","redaction_state":"reference_only"},
              "issued_by":"action_broker","issued_at":"2026-08-12T00:01:00Z",
              "expires_at":"2026-08-12T00:01:00Z"
            }"#,
        )
        .expect("decode");
        assert!(request.validate().is_err());
    }
}
