use crate::common::{
    check_len, compare_timestamp_values, validate_reason_code, validate_sha256, validate_timestamp,
    validate_timestamp_order, RedactedReference,
};
use crate::error::{ContractError, Result, Validate};
use crate::identity::{
    CognitiveExecutionResultId, CognitivePlanId, CognitivePlanNodeId, ModelId, RequestId,
};
use crate::inference_runtime_v3::{RuntimeIdentityV3, RuntimeResourceUsageV3, RuntimeResourceV3};
use crate::qualification::QualificationManifestId;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};

pub const COGNITIVE_CONTRACT_VERSION: &str = "1.0.0";

fn valid_path(value: &str) -> bool {
    if value.is_empty() || value.len() > 160 || !value.is_ascii() {
        return false;
    }
    let parts: Vec<&str> = value.split('.').collect();
    if parts.len() > 12
        || parts.iter().any(|p| {
            p.is_empty()
                || p.len() > 48
                || !p.as_bytes()[0].is_ascii_lowercase()
                || !p
                    .bytes()
                    .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_')
        })
    {
        return false;
    }
    parts[0] != "x" || parts.len() >= 3
}
fn validate_path(field: &'static str, value: &str) -> Result<()> {
    if valid_path(value) {
        Ok(())
    } else {
        Err(ContractError::field(
            field,
            "must be a normalized bounded capability path",
        ))
    }
}
fn extensible(field: &'static str, value: &str, core: &[&str]) -> Result<()> {
    if core.contains(&value) {
        return Ok(());
    }
    validate_path(field, value)?;
    if !value.starts_with("x.") {
        return Err(ContractError::field(
            field,
            "non-core value must use x.<owner> namespace",
        ));
    }
    Ok(())
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CapabilityRequirement {
    pub capability_id: String,
    pub minimum_score_basis_points: u16,
    pub minimum_confidence_basis_points: u16,
}
impl Validate for CapabilityRequirement {
    fn validate(&self) -> Result<()> {
        validate_path("capability_id", &self.capability_id)?;
        if self.minimum_score_basis_points > 10_000 || self.minimum_confidence_basis_points > 10_000
        {
            return Err(ContractError::field(
                "capability_requirement",
                "basis points exceed 10000",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ResourceRequirementEnvelope {
    pub minimum_memory_bytes: u64,
    pub expected_memory_bytes: u64,
    pub minimum_cpu_millicores: u32,
    pub minimum_accelerators: u16,
    pub device_classes: Vec<String>,
    pub context_tokens: u64,
    pub temporary_storage_bytes: u64,
    pub maximum_concurrency: u16,
}
impl Validate for ResourceRequirementEnvelope {
    fn validate(&self) -> Result<()> {
        if self.minimum_memory_bytes == 0
            || self.expected_memory_bytes < self.minimum_memory_bytes
            || self.expected_memory_bytes > 1_099_511_627_776
            || self.minimum_cpu_millicores > 262_144_000
            || self.minimum_accelerators > 64
            || self.context_tokens == 0
            || self.context_tokens > 2_000_000
            || self.temporary_storage_bytes > 1_099_511_627_776
            || self.maximum_concurrency == 0
            || self.maximum_concurrency > 4096
            || self.device_classes.len() > 16
        {
            return Err(ContractError::field(
                "resource_requirements",
                "contains impossible or out-of-range values",
            ));
        }
        let mut prior = None;
        for v in &self.device_classes {
            extensible("device_class", v, &["cpu", "gpu", "npu", "tpu"])?;
            if prior.is_some_and(|p: &String| p >= v) {
                return Err(ContractError::field(
                    "device_classes",
                    "must be sorted and unique",
                ));
            }
            prior = Some(v);
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ExecutionResourceCeiling {
    pub maximum_wall_clock_millis: u64,
    pub maximum_memory_bytes: u64,
    pub maximum_input_tokens: u64,
    pub maximum_output_tokens: u64,
    pub maximum_requests: u32,
    pub maximum_cost_microunits: u64,
    pub maximum_artifact_bytes: u64,
    pub maximum_output_bytes: u64,
    pub maximum_concurrency: u16,
}
impl Validate for ExecutionResourceCeiling {
    fn validate(&self) -> Result<()> {
        if self.maximum_wall_clock_millis == 0
            || self.maximum_wall_clock_millis > 86_400_000
            || self.maximum_memory_bytes == 0
            || self.maximum_memory_bytes > 1_099_511_627_776
            || self.maximum_input_tokens == 0
            || self.maximum_input_tokens > 2_000_000
            || self.maximum_output_tokens == 0
            || self.maximum_output_tokens > 2_000_000
            || self.maximum_requests == 0
            || self.maximum_requests > 10_000
            || self.maximum_cost_microunits > 1_000_000_000_000
            || self.maximum_artifact_bytes > 1_099_511_627_776
            || self.maximum_output_bytes == 0
            || self.maximum_output_bytes > 1_073_741_824
            || self.maximum_concurrency == 0
            || self.maximum_concurrency > 4096
        {
            return Err(ContractError::field(
                "resource_ceiling",
                "contains zero, impossible, or out-of-range limits",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct QualificationManifestReference {
    pub manifest_id: QualificationManifestId,
    pub revision: u64,
    pub digest: String,
}
impl Validate for QualificationManifestReference {
    fn validate(&self) -> Result<()> {
        self.manifest_id.validate()?;
        if self.revision == 0 {
            return Err(ContractError::field("manifest_revision", "must be nonzero"));
        }
        validate_sha256("manifest_digest", &self.digest)
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct VerificationRequirement {
    pub required: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub required_capability: Option<CapabilityRequirement>,
    pub evidence_class: String,
    pub independent_from_primary: bool,
}
impl Validate for VerificationRequirement {
    fn validate(&self) -> Result<()> {
        if self.required {
            self.required_capability
                .as_ref()
                .ok_or_else(|| ContractError::field("required_capability", "is required"))?
                .validate()?;
            extensible(
                "evidence_class",
                &self.evidence_class,
                &[
                    "contract_validation",
                    "code_verification",
                    "factual_verification",
                    "safety_review",
                ],
            )
        } else if self.required_capability.is_none()
            && self.evidence_class == "none"
            && !self.independent_from_primary
        {
            Ok(())
        } else {
            Err(ContractError::field(
                "verification_requirement",
                "disabled verification carries incompatible data",
            ))
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SynthesisRequirement {
    pub required: bool,
    pub input_node_ids: Vec<CognitivePlanNodeId>,
    pub required_role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub required_capability: Option<CapabilityRequirement>,
    pub aggregate_evidence: bool,
}
impl Validate for SynthesisRequirement {
    fn validate(&self) -> Result<()> {
        if self.required {
            if self.input_node_ids.is_empty() || self.input_node_ids.len() > 64 {
                return Err(ContractError::field(
                    "input_node_ids",
                    "must be bounded and nonempty",
                ));
            }
            extensible("required_role", &self.required_role, &["synthesizer"])?;
            self.required_capability
                .as_ref()
                .ok_or_else(|| ContractError::field("required_capability", "is required"))?
                .validate()?;
            let mut set = BTreeSet::new();
            for id in &self.input_node_ids {
                id.validate()?;
                if !set.insert(id) {
                    return Err(ContractError::field("input_node_ids", "duplicate"));
                }
            }
            Ok(())
        } else if self.input_node_ids.is_empty()
            && self.required_role == "none"
            && self.required_capability.is_none()
            && !self.aggregate_evidence
        {
            Ok(())
        } else {
            Err(ContractError::field(
                "synthesis_requirement",
                "disabled synthesis carries incompatible data",
            ))
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SpecialistAssignment {
    pub role: String,
    pub assignment_class: String,
    pub required_capabilities: Vec<CapabilityRequirement>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub assigned_model_id: Option<ModelId>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub qualification_manifest: Option<QualificationManifestReference>,
    pub resource_ceiling: ExecutionResourceCeiling,
    pub locality_constraint: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fallback_node_id: Option<CognitivePlanNodeId>,
}
impl Validate for SpecialistAssignment {
    fn validate(&self) -> Result<()> {
        extensible(
            "role",
            &self.role,
            &["specialist", "critic", "verifier", "synthesizer"],
        )?;
        if ![
            "primary",
            "secondary",
            "fallback",
            "local_fallback",
            "fast_path",
            "deep_path",
        ]
        .contains(&self.assignment_class.as_str())
        {
            return Err(ContractError::field(
                "assignment_class",
                "unsupported value",
            ));
        }
        if self.required_capabilities.is_empty() || self.required_capabilities.len() > 32 {
            return Err(ContractError::field(
                "required_capabilities",
                "must contain 1 through 32",
            ));
        }
        let mut set = BTreeSet::new();
        for r in &self.required_capabilities {
            r.validate()?;
            if !set.insert(&r.capability_id) {
                return Err(ContractError::field("required_capabilities", "duplicate"));
            }
        }
        if self.assigned_model_id.is_some() != self.qualification_manifest.is_some() {
            return Err(ContractError::field(
                "assignment_resolution",
                "model and manifest resolve together",
            ));
        }
        if let Some(id) = &self.assigned_model_id {
            id.validate()?;
        }
        if let Some(reference) = &self.qualification_manifest {
            reference.validate()?;
        }
        self.resource_ceiling.validate()?;
        if ![
            "any",
            "local_only",
            "locally_available",
            "remote_provider",
            "hybrid",
        ]
        .contains(&self.locality_constraint.as_str())
        {
            return Err(ContractError::field(
                "locality_constraint",
                "unsupported value",
            ));
        }
        if let Some(id) = &self.fallback_node_id {
            id.validate()?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CognitiveDependency {
    pub node_id: CognitivePlanNodeId,
    pub required_status: String,
}
impl Validate for CognitiveDependency {
    fn validate(&self) -> Result<()> {
        self.node_id.validate()?;
        if !["succeeded", "verified"].contains(&self.required_status.as_str()) {
            return Err(ContractError::field("required_status", "unsupported value"));
        }
        Ok(())
    }
}
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CognitivePlanNode {
    pub node_id: CognitivePlanNodeId,
    pub dependencies: Vec<CognitiveDependency>,
    pub assignment: SpecialistAssignment,
    pub verification: VerificationRequirement,
    pub synthesis: SynthesisRequirement,
    pub input_references: Vec<RedactedReference>,
}
impl Validate for CognitivePlanNode {
    fn validate(&self) -> Result<()> {
        self.node_id.validate()?;
        if self.dependencies.len() > 64 || self.input_references.len() > 64 {
            return Err(ContractError::field("node", "bounded collection exceeded"));
        }
        let mut set = BTreeSet::new();
        for d in &self.dependencies {
            d.validate()?;
            if d.node_id == self.node_id || !set.insert(&d.node_id) {
                return Err(ContractError::field(
                    "dependencies",
                    "self or duplicate dependency",
                ));
            }
        }
        self.assignment.validate()?;
        self.verification.validate()?;
        self.synthesis.validate()?;
        for r in &self.input_references {
            r.validate()?;
        }
        Ok(())
    }
}

fn visit_plan_node<'a>(
    id: &'a CognitivePlanNodeId,
    map: &BTreeMap<&'a CognitivePlanNodeId, &'a CognitivePlanNode>,
    state: &mut BTreeMap<&'a CognitivePlanNodeId, u8>,
) -> Result<()> {
    if state.get(id) == Some(&1) {
        return Err(ContractError::field("nodes", "dependency cycle"));
    }
    if state.get(id) == Some(&2) {
        return Ok(());
    }
    state.insert(id, 1);
    for dependency in &map[id].dependencies {
        visit_plan_node(&dependency.node_id, map, state)?;
    }
    state.insert(id, 2);
    Ok(())
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum CognitivePlanKind {
    #[serde(rename = "cognitive.plan")]
    Plan,
}
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CognitivePlan {
    pub contract_version: String,
    pub kind: CognitivePlanKind,
    pub plan_id: CognitivePlanId,
    pub revision: u64,
    pub request_id: RequestId,
    pub created_at: String,
    pub nodes: Vec<CognitivePlanNode>,
    pub root_node_ids: Vec<CognitivePlanNodeId>,
}
impl Validate for CognitivePlan {
    fn validate(&self) -> Result<()> {
        if self.contract_version != COGNITIVE_CONTRACT_VERSION
            || self.revision == 0
            || self.nodes.is_empty()
            || self.nodes.len() > 256
            || self.root_node_ids.is_empty()
            || self.root_node_ids.len() > 64
        {
            return Err(ContractError::field(
                "plan",
                "invalid version, revision, or bounds",
            ));
        }
        self.plan_id.validate()?;
        self.request_id.validate()?;
        validate_timestamp("created_at", &self.created_at)?;
        let mut map = BTreeMap::new();
        for n in &self.nodes {
            n.validate()?;
            if map.insert(&n.node_id, n).is_some() {
                return Err(ContractError::field("nodes", "duplicate node id"));
            }
        }
        for n in &self.nodes {
            for d in &n.dependencies {
                if !map.contains_key(&d.node_id) {
                    return Err(ContractError::field("dependencies", "absent node"));
                }
            }
            if let Some(id) = &n.assignment.fallback_node_id {
                if !map.contains_key(id) {
                    return Err(ContractError::field("fallback_node_id", "absent node"));
                }
            }
            for id in &n.synthesis.input_node_ids {
                if !map.contains_key(id) {
                    return Err(ContractError::field(
                        "synthesis.input_node_ids",
                        "absent node",
                    ));
                }
            }
        }
        for id in &self.root_node_ids {
            let valid_root = match map.get(id) {
                Some(node) => node.dependencies.is_empty(),
                None => false,
            };
            if !valid_root {
                return Err(ContractError::field(
                    "root_node_ids",
                    "must reference dependency-free nodes",
                ));
            }
        }
        let mut state = BTreeMap::new();
        for id in map.keys() {
            visit_plan_node(id, &map, &mut state)?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum CognitiveRequestKind {
    #[serde(rename = "cognitive.execution_request")]
    ExecutionRequest,
}
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CognitiveExecutionRequest {
    pub contract_version: String,
    pub kind: CognitiveRequestKind,
    pub request_id: RequestId,
    pub plan_id: CognitivePlanId,
    pub plan_revision: u64,
    pub plan_node_id: CognitivePlanNodeId,
    pub runtime_identity: RuntimeIdentityV3,
    pub qualification_manifest: QualificationManifestReference,
    pub input_references: Vec<RedactedReference>,
    pub resource_ceiling: ExecutionResourceCeiling,
    pub reservation_id: String,
    pub idempotency_key: String,
    pub immutable_request_digest: String,
    pub issued_at: String,
    pub deadline_at: String,
}
impl Validate for CognitiveExecutionRequest {
    fn validate(&self) -> Result<()> {
        if self.contract_version != COGNITIVE_CONTRACT_VERSION || self.plan_revision == 0 {
            return Err(ContractError::field(
                "request",
                "invalid version or plan revision",
            ));
        }
        self.request_id.validate()?;
        self.plan_id.validate()?;
        self.plan_node_id.validate()?;
        self.runtime_identity.validate()?;
        if self.runtime_identity.model_id.as_str() == self.runtime_identity.provider_model_id {
            return Err(ContractError::field(
                "runtime_identity",
                "logical and provider identities are non-substitutable",
            ));
        }
        self.qualification_manifest.validate()?;
        if self.input_references.is_empty() || self.input_references.len() > 64 {
            return Err(ContractError::field(
                "input_references",
                "must contain 1 through 64",
            ));
        }
        for r in &self.input_references {
            r.validate()?;
        }
        self.resource_ceiling.validate()?;
        check_len("reservation_id", &self.reservation_id, 2, 128)?;
        check_len("idempotency_key", &self.idempotency_key, 8, 128)?;
        validate_sha256("immutable_request_digest", &self.immutable_request_digest)?;
        validate_timestamp("issued_at", &self.issued_at)?;
        validate_timestamp("deadline_at", &self.deadline_at)?;
        validate_timestamp_order(
            "issued_at",
            &self.issued_at,
            "deadline_at",
            &self.deadline_at,
        )
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CognitiveExecutionFailure {
    pub class: String,
    pub code: String,
    pub retryable: bool,
}
impl Validate for CognitiveExecutionFailure {
    fn validate(&self) -> Result<()> {
        if ![
            "invalid_request",
            "identity_mismatch",
            "qualification_unavailable",
            "resource_exhausted",
            "deadline_exceeded",
            "cancelled",
            "runtime_failure",
        ]
        .contains(&self.class.as_str())
        {
            return Err(ContractError::field("failure.class", "unsupported value"));
        }
        validate_reason_code("failure.code", &self.code)
    }
}
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum CognitiveResultKind {
    #[serde(rename = "cognitive.execution_result")]
    ExecutionResult,
}
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CognitiveExecutionResult {
    pub contract_version: String,
    pub kind: CognitiveResultKind,
    pub result_id: CognitiveExecutionResultId,
    pub request_id: RequestId,
    pub plan_id: CognitivePlanId,
    pub plan_revision: u64,
    pub plan_node_id: CognitivePlanNodeId,
    pub runtime_identity: RuntimeIdentityV3,
    pub status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub started_at: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completed_at: Option<String>,
    pub resource_usage: Vec<RuntimeResourceUsageV3>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_reference: Option<RedactedReference>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub failure: Option<CognitiveExecutionFailure>,
    pub idempotency_key: String,
    pub immutable_request_digest: String,
}
impl Validate for CognitiveExecutionResult {
    fn validate(&self) -> Result<()> {
        if self.contract_version != COGNITIVE_CONTRACT_VERSION || self.plan_revision == 0 {
            return Err(ContractError::field(
                "result",
                "invalid version or revision",
            ));
        }
        self.result_id.validate()?;
        self.request_id.validate()?;
        self.plan_id.validate()?;
        self.plan_node_id.validate()?;
        self.runtime_identity.validate()?;
        if self.runtime_identity.model_id.as_str() == self.runtime_identity.provider_model_id {
            return Err(ContractError::field(
                "runtime_identity",
                "identity substitution",
            ));
        }
        if self.resource_usage.len() > 32 {
            return Err(ContractError::field("resource_usage", "exceeds 32"));
        }
        for usage in &self.resource_usage {
            usage.validate()?;
        }
        match self.status.as_str() {
            "pending"
                if self.started_at.is_none()
                    && self.completed_at.is_none()
                    && self.output_reference.is_none()
                    && self.failure.is_none() => {}
            "running"
                if self.started_at.is_some()
                    && self.completed_at.is_none()
                    && self.output_reference.is_none()
                    && self.failure.is_none() => {}
            "succeeded"
                if self.started_at.is_some()
                    && self.completed_at.is_some()
                    && self.output_reference.is_some()
                    && self.failure.is_none() => {}
            "failed" | "cancelled"
                if self.started_at.is_some()
                    && self.completed_at.is_some()
                    && self.output_reference.is_none()
                    && self.failure.is_some() => {}
            _ => {
                return Err(ContractError::field(
                    "status",
                    "incompatible status and payload",
                ))
            }
        }
        if let Some(value) = &self.started_at {
            validate_timestamp("started_at", value)?;
        }
        if let Some(value) = &self.completed_at {
            validate_timestamp("completed_at", value)?;
            validate_timestamp_order(
                "started_at",
                self.started_at.as_ref().expect("terminal has start"),
                "completed_at",
                value,
            )?;
        }
        if let Some(value) = &self.output_reference {
            value.validate()?;
        }
        if let Some(value) = &self.failure {
            value.validate()?;
        }
        check_len("idempotency_key", &self.idempotency_key, 8, 128)?;
        validate_sha256("immutable_request_digest", &self.immutable_request_digest)
    }
}

/// Compares one result with the exact request identity, replay, resource, and deadline bounds.
///
/// # Errors
///
/// Returns an error when either document is invalid or any cross-document binding differs.
pub fn validate_execution_against_request(
    request: &CognitiveExecutionRequest,
    result: &CognitiveExecutionResult,
) -> Result<()> {
    request.validate()?;
    result.validate()?;
    if request.request_id != result.request_id
        || request.plan_id != result.plan_id
        || request.plan_revision != result.plan_revision
        || request.plan_node_id != result.plan_node_id
        || request.runtime_identity != result.runtime_identity
        || request.idempotency_key != result.idempotency_key
        || request.immutable_request_digest != result.immutable_request_digest
    {
        return Err(ContractError::field(
            "execution_result",
            "request, plan, runtime identity, or replay binding mismatch",
        ));
    }
    for usage in &result.resource_usage {
        if (usage.resource == RuntimeResourceV3::InferenceMemory
            && usage.units > request.resource_ceiling.maximum_memory_bytes)
            || (usage.resource == RuntimeResourceV3::WallTime
                && usage.units > request.resource_ceiling.maximum_wall_clock_millis)
        {
            return Err(ContractError::field(
                "resource_usage",
                "exceeds request ceiling",
            ));
        }
    }
    if let Some(completed) = &result.completed_at {
        if compare_timestamp_values(completed, &request.deadline_at)? == Ordering::Greater {
            return Err(ContractError::field("completed_at", "exceeds deadline"));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::valid_path;
    #[test]
    fn capability_path_parser_is_bounded() {
        assert!(valid_path("software.rust"));
        assert!(valid_path("x.example.future"));
        assert!(!valid_path("Software.Rust"));
        assert!(!valid_path("x.owner"));
        assert!(!valid_path(&format!("a.{}", "b.".repeat(20))));
    }
}
