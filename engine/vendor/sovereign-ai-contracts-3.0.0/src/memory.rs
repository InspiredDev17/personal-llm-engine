use crate::common::{
    check_len, check_max, check_range, validate_extensions, validate_reason_code, validate_sha256,
    validate_timestamp, validate_timestamp_order, Component, ContractVersion, Extensions,
    RedactedReference, RequiredExtensions, ScreenedText,
};
use crate::error::{ContractError, Result, Validate};
use crate::identity::{AgentId, MemoryId, RequestId, UserId};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SensitivityClass {
    Public,
    Internal,
    Personal,
    Sensitive,
    Restricted,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RetentionMode {
    Session,
    Until,
    DurationDays,
    Indefinite,
    LegalHold,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RetentionPolicy {
    pub mode: RetentionMode,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expires_at: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub duration_days: Option<u16>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason_code: Option<String>,
}

impl Validate for RetentionPolicy {
    fn validate(&self) -> Result<()> {
        if let Some(expires_at) = &self.expires_at {
            validate_timestamp("expires_at", expires_at)?;
        }
        if let Some(days) = self.duration_days {
            check_range("duration_days", days, 1, 36_500)?;
        }
        if let Some(reason_code) = &self.reason_code {
            validate_reason_code("reason_code", reason_code)?;
        }
        match self.mode {
            RetentionMode::Until if self.expires_at.is_none() => Err(ContractError::field(
                "expires_at",
                "is required for until retention",
            )),
            RetentionMode::DurationDays if self.duration_days.is_none() => Err(
                ContractError::field("duration_days", "is required for duration_days retention"),
            ),
            RetentionMode::LegalHold if self.reason_code.is_none() => Err(ContractError::field(
                "reason_code",
                "is required for legal_hold retention",
            )),
            _ => Ok(()),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum MemorySourceType {
    UserStatement,
    Conversation,
    Document,
    ToolResult,
    SystemObservation,
    Import,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct MemorySource {
    pub source_type: MemorySourceType,
    pub source_component: Component,
    pub source_reference: RedactedReference,
}

impl Validate for MemorySource {
    fn validate(&self) -> Result<()> {
        self.source_reference.validate()
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ProvenanceHop {
    pub component: Component,
    pub operation: String,
    pub occurred_at: String,
    pub input_digest: String,
    pub output_digest: String,
}

impl Validate for ProvenanceHop {
    fn validate(&self) -> Result<()> {
        validate_reason_code("operation", &self.operation)?;
        validate_timestamp("occurred_at", &self.occurred_at)?;
        validate_sha256("input_digest", &self.input_digest)?;
        validate_sha256("output_digest", &self.output_digest)
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Provenance {
    pub origin: MemorySource,
    pub chain: Vec<ProvenanceHop>,
}

impl Validate for Provenance {
    fn validate(&self) -> Result<()> {
        self.origin.validate()?;
        check_max("chain", self.chain.len(), 16)?;
        self.chain.iter().try_for_each(Validate::validate)
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum MemoryQueryKind {
    #[serde(rename = "memory.query")]
    MemoryQuery,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct MemoryQuery {
    pub contract_version: ContractVersion,
    pub kind: MemoryQueryKind,
    pub request_id: RequestId,
    pub user_id: UserId,
    pub query: ScreenedText,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub namespaces: Option<Vec<String>>,
    pub max_results: u8,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub minimum_relevance: Option<f64>,
    pub sensitivity_ceiling: SensitivityClass,
    pub requested_at: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extensions: Option<Extensions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub required_extensions: Option<RequiredExtensions>,
}

impl Validate for MemoryQuery {
    fn validate(&self) -> Result<()> {
        self.query.validate()?;
        if let Some(namespaces) = &self.namespaces {
            check_max("namespaces", namespaces.len(), 16)?;
            let mut unique = BTreeSet::new();
            for namespace in namespaces {
                check_len("namespaces", namespace, 1, 64)?;
                validate_lower_name("namespaces", namespace)?;
                if !unique.insert(namespace) {
                    return Err(ContractError::field("namespaces", "values must be unique"));
                }
            }
        }
        check_range("max_results", self.max_results, 1, 100)?;
        if let Some(relevance) = self.minimum_relevance {
            validate_relevance("minimum_relevance", relevance)?;
        }
        validate_timestamp("requested_at", &self.requested_at)?;
        validate_extensions(self.extensions.as_ref(), self.required_extensions.as_ref())
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum MemoryCandidateKind {
    #[serde(rename = "memory.candidate")]
    MemoryCandidate,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct MemoryCandidate {
    pub contract_version: ContractVersion,
    pub kind: MemoryCandidateKind,
    pub request_id: RequestId,
    pub memory_id: MemoryId,
    pub relevance: f64,
    pub excerpt: ScreenedText,
    pub source: MemorySource,
    pub provenance: Provenance,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extensions: Option<Extensions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub required_extensions: Option<RequiredExtensions>,
}

impl Validate for MemoryCandidate {
    fn validate(&self) -> Result<()> {
        validate_relevance("relevance", self.relevance)?;
        self.excerpt.validate()?;
        self.source.validate()?;
        self.provenance.validate()?;
        validate_extensions(self.extensions.as_ref(), self.required_extensions.as_ref())
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum MemoryWriteProposalKind {
    #[serde(rename = "memory.write_proposal")]
    MemoryWriteProposal,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct MemoryWriteProposal {
    pub contract_version: ContractVersion,
    pub kind: MemoryWriteProposalKind,
    pub proposal_id: RequestId,
    pub proposed_by: AgentId,
    pub user_id: UserId,
    pub content: ScreenedText,
    pub source: MemorySource,
    pub retention: RetentionPolicy,
    pub sensitivity: SensitivityClass,
    pub provenance: Provenance,
    pub proposed_at: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extensions: Option<Extensions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub required_extensions: Option<RequiredExtensions>,
}

impl Validate for MemoryWriteProposal {
    fn validate(&self) -> Result<()> {
        self.content.validate()?;
        self.source.validate()?;
        self.retention.validate()?;
        self.provenance.validate()?;
        validate_timestamp("proposed_at", &self.proposed_at)?;
        validate_extensions(self.extensions.as_ref(), self.required_extensions.as_ref())
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum ApprovedMemoryWriteKind {
    #[serde(rename = "memory.approved_write")]
    ApprovedMemoryWrite,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ApprovedMemoryWrite {
    pub contract_version: ContractVersion,
    pub kind: ApprovedMemoryWriteKind,
    pub proposal_id: RequestId,
    pub approval_id: RequestId,
    pub approved_by: UserId,
    pub approved_at: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expires_at: Option<String>,
    pub proposal_digest: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extensions: Option<Extensions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub required_extensions: Option<RequiredExtensions>,
}

impl Validate for ApprovedMemoryWrite {
    fn validate(&self) -> Result<()> {
        validate_timestamp("approved_at", &self.approved_at)?;
        if let Some(expires_at) = &self.expires_at {
            validate_timestamp("expires_at", expires_at)?;
            validate_timestamp_order("approved_at", &self.approved_at, "expires_at", expires_at)?;
        }
        validate_sha256("proposal_digest", &self.proposal_digest)?;
        validate_extensions(self.extensions.as_ref(), self.required_extensions.as_ref())
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum MemoryRecordKind {
    #[serde(rename = "memory.record")]
    MemoryRecord,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct MemoryRecord {
    pub contract_version: ContractVersion,
    pub kind: MemoryRecordKind,
    pub memory_id: MemoryId,
    pub user_id: UserId,
    pub content: ScreenedText,
    pub source: MemorySource,
    pub retention: RetentionPolicy,
    pub sensitivity: SensitivityClass,
    pub provenance: Provenance,
    pub created_at: String,
    pub record_digest: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extensions: Option<Extensions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub required_extensions: Option<RequiredExtensions>,
}

impl Validate for MemoryRecord {
    fn validate(&self) -> Result<()> {
        self.content.validate()?;
        self.source.validate()?;
        self.retention.validate()?;
        self.provenance.validate()?;
        validate_timestamp("created_at", &self.created_at)?;
        validate_sha256("record_digest", &self.record_digest)?;
        validate_extensions(self.extensions.as_ref(), self.required_extensions.as_ref())
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

fn validate_relevance(field: &'static str, value: f64) -> Result<()> {
    if value.is_finite() && (0.0..=1.0).contains(&value) {
        Ok(())
    } else {
        Err(ContractError::field(field, "must be between zero and one"))
    }
}

#[cfg(test)]
mod tests {
    use super::{ApprovedMemoryWrite, RetentionMode, RetentionPolicy};
    use crate::Validate;

    #[test]
    fn conditional_retention_fields_are_required() {
        let policy = RetentionPolicy {
            mode: RetentionMode::Until,
            expires_at: None,
            duration_days: None,
            reason_code: None,
        };
        assert!(policy.validate().is_err());
    }

    #[test]
    fn memory_approval_expiry_must_follow_approval() {
        let approval: ApprovedMemoryWrite = serde_json::from_str(
            r#"{
              "contract_version":"1.0.0","kind":"memory.approved_write",
              "proposal_id":"req_proposal","approval_id":"req_approval",
              "approved_by":"usr_owner","approved_at":"2026-08-12T10:00:00Z",
              "expires_at":"2026-08-12T10:00:00Z",
              "proposal_digest":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
            }"#,
        )
        .expect("decode");
        assert!(approval.validate().is_err());
    }
}
