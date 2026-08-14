use crate::common::{
    validate_extensions, validate_reason_code, validate_timestamp, ContractVersion, Extensions,
    RequiredExtensions, ScreenedText,
};
use crate::error::{Result, Validate};
use crate::identity::{AgentId, RequestId, TaskId, UserId};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum TaskEnvelopeKind {
    #[serde(rename = "task.envelope")]
    TaskEnvelope,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TaskEnvelope {
    pub contract_version: ContractVersion,
    pub kind: TaskEnvelopeKind,
    pub task_id: TaskId,
    pub request_id: RequestId,
    pub owner_user_id: UserId,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub assigned_agent_id: Option<AgentId>,
    pub objective: ScreenedText,
    pub created_at: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub deadline_at: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extensions: Option<Extensions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub required_extensions: Option<RequiredExtensions>,
}

impl Validate for TaskEnvelope {
    fn validate(&self) -> Result<()> {
        self.objective.validate()?;
        validate_timestamp("created_at", &self.created_at)?;
        if let Some(deadline_at) = &self.deadline_at {
            validate_timestamp("deadline_at", deadline_at)?;
        }
        validate_extensions(self.extensions.as_ref(), self.required_extensions.as_ref())
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskState {
    Pending,
    Running,
    WaitingForApproval,
    Completed,
    Cancelled,
    Failed,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum TaskStatusKind {
    #[serde(rename = "task.status")]
    TaskStatus,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TaskStatus {
    pub contract_version: ContractVersion,
    pub kind: TaskStatusKind,
    pub task_id: TaskId,
    pub state: TaskState,
    pub sequence: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason_code: Option<String>,
    pub observed_at: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extensions: Option<Extensions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub required_extensions: Option<RequiredExtensions>,
}

impl Validate for TaskStatus {
    fn validate(&self) -> Result<()> {
        if let Some(reason_code) = &self.reason_code {
            validate_reason_code("reason_code", reason_code)?;
        }
        validate_timestamp("observed_at", &self.observed_at)?;
        validate_extensions(self.extensions.as_ref(), self.required_extensions.as_ref())
    }
}
