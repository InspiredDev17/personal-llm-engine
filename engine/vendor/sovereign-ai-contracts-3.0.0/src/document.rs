//! Strict standalone document validation and deterministic normalization.
//!
//! This module has no execution or authorization behavior. It validates only
//! the language-neutral contract representation through the Rust bindings.

use crate::audit::AuditEvent;
use crate::cognitive::{CognitiveExecutionRequest, CognitiveExecutionResult, CognitivePlan};
use crate::inference::{CancellationRequest, EngineHealth, GenerationEvent, InferenceRequest};
use crate::inference_runtime_v2::{
    InferenceExecutionRequest, InferenceExecutionResult, InferenceReservationDecision,
    InferenceReservationRelease, InferenceReservationRequest, RuntimeAvailability,
    RuntimeCancellationRequest,
};
use crate::inference_runtime_v3::{
    InferenceExecutionRequestV3, InferenceExecutionResultV3, InferenceReservationDecisionV3,
    InferenceReservationReleaseV3, InferenceReservationRequestV3, RuntimeAvailabilityV3,
    RuntimeCancellationRequestV3,
};
use crate::knowledge::{
    CacheCompatibility, CagCacheBuildRequest, CagCacheBuildResult, CagCacheDescriptor,
    CagCacheInvalidationRequest, CagCacheInvalidationResult, CagCacheQueryRequest,
    CagCacheQueryResult, DynamicRetrievalResult, KnowledgePlanRequest, KnowledgePlanResult,
    MemoryRevisionInvalidated, StaticKnowledgeBundle,
};
use crate::knowledge_lifecycle::{
    CagCacheAttachRequest, CagCacheAttachResult, CagCacheCreateRequest, CagCacheCreateResult,
    CagCacheDescribeRequest, CagCacheDescribeResult, CagCacheInvalidateRequest,
    CagCacheInvalidateResult, CagCacheListRequest, CagCacheListResult, CagCacheMetricsRequest,
    CagCacheMetricsResult, CagCacheReleaseRequest, CagCacheReleaseResult,
};
use crate::knowledge_prefix::{StaticPrefixMaterialRequest, StaticPrefixMaterialResult};
use crate::memory::{
    ApprovedMemoryWrite, MemoryCandidate, MemoryQuery, MemoryRecord, MemoryWriteProposal,
};
use crate::models::{ModelActivationRequest, ModelActivationState, ModelArtifactDescriptor};
use crate::negotiation::{
    ContractVersionOffer, ContractVersionSelection, VersionNegotiationTranscript,
};
use crate::negotiation_v2::{
    ProductionVersionOffer, ProductionVersionSelection, ProductionVersionTranscript,
};
use crate::nodes::{NodeActionRequest, NodeActionResult, NodeDescriptor};
use crate::qualification::CognitiveQualificationManifest;
use crate::qualification_v2::ModelCapabilityManifest;
use crate::security::{ContainmentRequest, ContainmentResult, SecurityDecision, SecurityEvent};
use crate::tasks::{TaskEnvelope, TaskStatus};
use crate::tools::{
    ExecutionError, ExecutionRequest, ExecutionResult, ToolDescriptor, ToolRequest,
};
use crate::{ContractError, Result, Validate};
use serde::de::{DeserializeOwned, Error as DeError, MapAccess, SeqAccess, Visitor};
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::{Map, Number, Value};
use std::fmt::Formatter;

/// Successful strict validation of one standalone contract document.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DocumentValidation {
    pub kind: String,
    pub contract_version: String,
    pub schema_path: &'static str,
    pub canonical_json: String,
}

/// One supported standalone wire kind and its exact versioned schema path.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct DocumentRegistryEntry {
    pub contract_version: &'static str,
    pub kind: &'static str,
    pub schema: &'static str,
}

const DOCUMENT_REGISTRY: &[DocumentRegistryEntry] = &[
    DocumentRegistryEntry {
        contract_version: "1.0.0",
        kind: "audit.event",
        schema: "audit/v1/audit.schema.json",
    },
    DocumentRegistryEntry {
        contract_version: "1.0.0",
        kind: "cognitive.execution_request",
        schema: "cognitive/v1/cognitive.schema.json",
    },
    DocumentRegistryEntry {
        contract_version: "1.0.0",
        kind: "cognitive.execution_result",
        schema: "cognitive/v1/cognitive.schema.json",
    },
    DocumentRegistryEntry {
        contract_version: "1.0.0",
        kind: "cognitive.plan",
        schema: "cognitive/v1/cognitive.schema.json",
    },
    DocumentRegistryEntry {
        contract_version: "1.0.0",
        kind: "engine.health",
        schema: "inference/v1/inference.schema.json",
    },
    DocumentRegistryEntry {
        contract_version: "1.0.0",
        kind: "generation.event",
        schema: "inference/v1/inference.schema.json",
    },
    DocumentRegistryEntry {
        contract_version: "1.0.0",
        kind: "inference.cancellation_request",
        schema: "inference/v1/inference.schema.json",
    },
    DocumentRegistryEntry {
        contract_version: "1.0.0",
        kind: "inference.request",
        schema: "inference/v1/inference.schema.json",
    },
    DocumentRegistryEntry {
        contract_version: "1.0.0",
        kind: "knowledge.cache_compatibility",
        schema: "knowledge/v1/knowledge.schema.json",
    },
    DocumentRegistryEntry {
        contract_version: "1.0.0",
        kind: "knowledge.cag_cache_build_request",
        schema: "knowledge/v1/knowledge.schema.json",
    },
    DocumentRegistryEntry {
        contract_version: "1.0.0",
        kind: "knowledge.cag_cache_build_result",
        schema: "knowledge/v1/knowledge.schema.json",
    },
    DocumentRegistryEntry {
        contract_version: "1.0.0",
        kind: "knowledge.cag_cache_descriptor",
        schema: "knowledge/v1/knowledge.schema.json",
    },
    DocumentRegistryEntry {
        contract_version: "1.0.0",
        kind: "knowledge.cag_cache_invalidation_request",
        schema: "knowledge/v1/knowledge.schema.json",
    },
    DocumentRegistryEntry {
        contract_version: "1.0.0",
        kind: "knowledge.cag_cache_invalidation_result",
        schema: "knowledge/v1/knowledge.schema.json",
    },
    DocumentRegistryEntry {
        contract_version: "1.0.0",
        kind: "knowledge.cag_cache_query_request",
        schema: "knowledge/v1/knowledge.schema.json",
    },
    DocumentRegistryEntry {
        contract_version: "1.0.0",
        kind: "knowledge.cag_cache_query_result",
        schema: "knowledge/v1/knowledge.schema.json",
    },
    DocumentRegistryEntry {
        contract_version: "1.0.0",
        kind: "knowledge.dynamic_retrieval_result",
        schema: "knowledge/v1/knowledge.schema.json",
    },
    DocumentRegistryEntry {
        contract_version: "1.0.0",
        kind: "knowledge.memory_revision_invalidated",
        schema: "knowledge/v1/knowledge.schema.json",
    },
    DocumentRegistryEntry {
        contract_version: "1.0.0",
        kind: "knowledge.plan_request",
        schema: "knowledge/v1/knowledge.schema.json",
    },
    DocumentRegistryEntry {
        contract_version: "1.0.0",
        kind: "knowledge.plan_result",
        schema: "knowledge/v1/knowledge.schema.json",
    },
    DocumentRegistryEntry {
        contract_version: "1.0.0",
        kind: "knowledge.static_bundle",
        schema: "knowledge/v1/knowledge.schema.json",
    },
    DocumentRegistryEntry {
        contract_version: "1.0.0",
        kind: "memory.approved_write",
        schema: "memory/v1/memory.schema.json",
    },
    DocumentRegistryEntry {
        contract_version: "1.0.0",
        kind: "memory.candidate",
        schema: "memory/v1/memory.schema.json",
    },
    DocumentRegistryEntry {
        contract_version: "1.0.0",
        kind: "memory.query",
        schema: "memory/v1/memory.schema.json",
    },
    DocumentRegistryEntry {
        contract_version: "1.0.0",
        kind: "memory.record",
        schema: "memory/v1/memory.schema.json",
    },
    DocumentRegistryEntry {
        contract_version: "1.0.0",
        kind: "memory.write_proposal",
        schema: "memory/v1/memory.schema.json",
    },
    DocumentRegistryEntry {
        contract_version: "1.0.0",
        kind: "model.activation_request",
        schema: "models/v1/models.schema.json",
    },
    DocumentRegistryEntry {
        contract_version: "1.0.0",
        kind: "model.activation_state",
        schema: "models/v1/models.schema.json",
    },
    DocumentRegistryEntry {
        contract_version: "1.0.0",
        kind: "model.artifact_descriptor",
        schema: "models/v1/models.schema.json",
    },
    DocumentRegistryEntry {
        contract_version: "1.0.0",
        kind: "node.action_request",
        schema: "nodes/v1/nodes.schema.json",
    },
    DocumentRegistryEntry {
        contract_version: "1.0.0",
        kind: "node.action_result",
        schema: "nodes/v1/nodes.schema.json",
    },
    DocumentRegistryEntry {
        contract_version: "1.0.0",
        kind: "node.descriptor",
        schema: "nodes/v1/nodes.schema.json",
    },
    DocumentRegistryEntry {
        contract_version: "1.0.0",
        kind: "qualification.cognitive_manifest",
        schema: "qualification/v1/qualification.schema.json",
    },
    DocumentRegistryEntry {
        contract_version: "1.0.0",
        kind: "security.containment_request",
        schema: "security/v1/security.schema.json",
    },
    DocumentRegistryEntry {
        contract_version: "1.0.0",
        kind: "security.containment_result",
        schema: "security/v1/security.schema.json",
    },
    DocumentRegistryEntry {
        contract_version: "1.0.0",
        kind: "security.decision",
        schema: "security/v1/security.schema.json",
    },
    DocumentRegistryEntry {
        contract_version: "1.0.0",
        kind: "security.event",
        schema: "security/v1/security.schema.json",
    },
    DocumentRegistryEntry {
        contract_version: "1.0.0",
        kind: "task.envelope",
        schema: "tasks/v1/tasks.schema.json",
    },
    DocumentRegistryEntry {
        contract_version: "1.0.0",
        kind: "task.status",
        schema: "tasks/v1/tasks.schema.json",
    },
    DocumentRegistryEntry {
        contract_version: "1.0.0",
        kind: "tool.descriptor",
        schema: "tools/v1/tools.schema.json",
    },
    DocumentRegistryEntry {
        contract_version: "1.0.0",
        kind: "tool.execution_error",
        schema: "tools/v1/tools.schema.json",
    },
    DocumentRegistryEntry {
        contract_version: "1.0.0",
        kind: "tool.execution_request",
        schema: "tools/v1/tools.schema.json",
    },
    DocumentRegistryEntry {
        contract_version: "1.0.0",
        kind: "tool.execution_result",
        schema: "tools/v1/tools.schema.json",
    },
    DocumentRegistryEntry {
        contract_version: "1.0.0",
        kind: "tool.request",
        schema: "tools/v1/tools.schema.json",
    },
    DocumentRegistryEntry {
        contract_version: "1.1.0",
        kind: "contract.version_offer",
        schema: "common/v1/common-1.1.0.schema.json",
    },
    DocumentRegistryEntry {
        contract_version: "1.1.0",
        kind: "contract.version_selection",
        schema: "common/v1/common-1.1.0.schema.json",
    },
    DocumentRegistryEntry {
        contract_version: "1.1.0",
        kind: "contract.version_transcript",
        schema: "common/v1/common-1.1.0.schema.json",
    },
    DocumentRegistryEntry {
        contract_version: "1.1.0",
        kind: "knowledge.cag_cache_attach_request",
        schema: "knowledge/v1/knowledge-1.1.0.schema.json",
    },
    DocumentRegistryEntry {
        contract_version: "1.1.0",
        kind: "knowledge.cag_cache_attach_result",
        schema: "knowledge/v1/knowledge-1.1.0.schema.json",
    },
    DocumentRegistryEntry {
        contract_version: "1.1.0",
        kind: "knowledge.cag_cache_create_request",
        schema: "knowledge/v1/knowledge-1.1.0.schema.json",
    },
    DocumentRegistryEntry {
        contract_version: "1.1.0",
        kind: "knowledge.cag_cache_create_result",
        schema: "knowledge/v1/knowledge-1.1.0.schema.json",
    },
    DocumentRegistryEntry {
        contract_version: "1.1.0",
        kind: "knowledge.cag_cache_describe_request",
        schema: "knowledge/v1/knowledge-1.1.0.schema.json",
    },
    DocumentRegistryEntry {
        contract_version: "1.1.0",
        kind: "knowledge.cag_cache_describe_result",
        schema: "knowledge/v1/knowledge-1.1.0.schema.json",
    },
    DocumentRegistryEntry {
        contract_version: "1.1.0",
        kind: "knowledge.cag_cache_invalidate_request",
        schema: "knowledge/v1/knowledge-1.1.0.schema.json",
    },
    DocumentRegistryEntry {
        contract_version: "1.1.0",
        kind: "knowledge.cag_cache_invalidate_result",
        schema: "knowledge/v1/knowledge-1.1.0.schema.json",
    },
    DocumentRegistryEntry {
        contract_version: "1.1.0",
        kind: "knowledge.cag_cache_list_request",
        schema: "knowledge/v1/knowledge-1.1.0.schema.json",
    },
    DocumentRegistryEntry {
        contract_version: "1.1.0",
        kind: "knowledge.cag_cache_list_result",
        schema: "knowledge/v1/knowledge-1.1.0.schema.json",
    },
    DocumentRegistryEntry {
        contract_version: "1.1.0",
        kind: "knowledge.cag_cache_metrics_request",
        schema: "knowledge/v1/knowledge-1.1.0.schema.json",
    },
    DocumentRegistryEntry {
        contract_version: "1.1.0",
        kind: "knowledge.cag_cache_metrics_result",
        schema: "knowledge/v1/knowledge-1.1.0.schema.json",
    },
    DocumentRegistryEntry {
        contract_version: "1.1.0",
        kind: "knowledge.cag_cache_release_request",
        schema: "knowledge/v1/knowledge-1.1.0.schema.json",
    },
    DocumentRegistryEntry {
        contract_version: "1.1.0",
        kind: "knowledge.cag_cache_release_result",
        schema: "knowledge/v1/knowledge-1.1.0.schema.json",
    },
    DocumentRegistryEntry {
        contract_version: "1.2.0",
        kind: "knowledge.static_prefix_material_request",
        schema: "knowledge/v1/knowledge-1.2.0.schema.json",
    },
    DocumentRegistryEntry {
        contract_version: "1.2.0",
        kind: "knowledge.static_prefix_material_result",
        schema: "knowledge/v1/knowledge-1.2.0.schema.json",
    },
    DocumentRegistryEntry {
        contract_version: "2.0.0",
        kind: "contract.production_version_offer",
        schema: "common/v2/common.schema.json",
    },
    DocumentRegistryEntry {
        contract_version: "2.0.0",
        kind: "contract.production_version_selection",
        schema: "common/v2/common.schema.json",
    },
    DocumentRegistryEntry {
        contract_version: "2.0.0",
        kind: "contract.production_version_transcript",
        schema: "common/v2/common.schema.json",
    },
    DocumentRegistryEntry {
        contract_version: "2.0.0",
        kind: "inference.execution_request",
        schema: "inference/v2/inference.schema.json",
    },
    DocumentRegistryEntry {
        contract_version: "2.0.0",
        kind: "inference.execution_result",
        schema: "inference/v2/inference.schema.json",
    },
    DocumentRegistryEntry {
        contract_version: "2.0.0",
        kind: "inference.reservation_decision",
        schema: "inference/v2/inference.schema.json",
    },
    DocumentRegistryEntry {
        contract_version: "2.0.0",
        kind: "inference.reservation_release",
        schema: "inference/v2/inference.schema.json",
    },
    DocumentRegistryEntry {
        contract_version: "2.0.0",
        kind: "inference.reservation_request",
        schema: "inference/v2/inference.schema.json",
    },
    DocumentRegistryEntry {
        contract_version: "2.0.0",
        kind: "inference.runtime_availability",
        schema: "inference/v2/inference.schema.json",
    },
    DocumentRegistryEntry {
        contract_version: "2.0.0",
        kind: "inference.runtime_cancellation_request",
        schema: "inference/v2/inference.schema.json",
    },
    DocumentRegistryEntry {
        contract_version: "2.0.0",
        kind: "qualification.model_capability_manifest",
        schema: "qualification/v2/qualification.schema.json",
    },
    DocumentRegistryEntry {
        contract_version: "3.0.0",
        kind: "inference.execution_request",
        schema: "inference/v3/inference.schema.json",
    },
    DocumentRegistryEntry {
        contract_version: "3.0.0",
        kind: "inference.execution_result",
        schema: "inference/v3/inference.schema.json",
    },
    DocumentRegistryEntry {
        contract_version: "3.0.0",
        kind: "inference.reservation_decision",
        schema: "inference/v3/inference.schema.json",
    },
    DocumentRegistryEntry {
        contract_version: "3.0.0",
        kind: "inference.reservation_release",
        schema: "inference/v3/inference.schema.json",
    },
    DocumentRegistryEntry {
        contract_version: "3.0.0",
        kind: "inference.reservation_request",
        schema: "inference/v3/inference.schema.json",
    },
    DocumentRegistryEntry {
        contract_version: "3.0.0",
        kind: "inference.runtime_availability",
        schema: "inference/v3/inference.schema.json",
    },
    DocumentRegistryEntry {
        contract_version: "3.0.0",
        kind: "inference.runtime_cancellation_request",
        schema: "inference/v3/inference.schema.json",
    },
];

/// Returns every supported standalone mapping in deterministic order.
#[must_use]
pub const fn document_registry() -> &'static [DocumentRegistryEntry] {
    DOCUMENT_REGISTRY
}

/// Validates a standalone document through its exact version and kind binding.
///
/// # Errors
///
/// Returns an error for malformed JSON, duplicate object keys, unknown
/// kind/version pairs, strict deserialization failures, or semantic failures.
pub fn validate_document(source: &[u8]) -> Result<DocumentValidation> {
    let value = parse_strict_value(source)?;
    let object = value
        .as_object()
        .ok_or_else(|| ContractError::field("document", "must be a JSON object"))?;
    let kind = required_string(object, "kind")?.to_owned();
    let contract_version = required_string(object, "contract_version")?.to_owned();
    let schema_path = schema_for(&contract_version, &kind).ok_or_else(|| {
        ContractError::field(
            "contract_version",
            format!("unsupported contract kind/version pair {kind:?} at {contract_version:?}"),
        )
    })?;
    let normalized = normalize_value(&contract_version, &kind, value)?;
    let canonical_json = serde_json::to_string(&sort_value(normalized))?;
    Ok(DocumentValidation {
        kind,
        contract_version,
        schema_path,
        canonical_json,
    })
}

fn required_string<'a>(object: &'a Map<String, Value>, field: &'static str) -> Result<&'a str> {
    object
        .get(field)
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| ContractError::field(field, "is required and must be a non-empty string"))
}

fn schema_for(version: &str, kind: &str) -> Option<&'static str> {
    DOCUMENT_REGISTRY
        .iter()
        .find(|entry| entry.contract_version == version && entry.kind == kind)
        .map(|entry| entry.schema)
}

fn normalize_value(contract_version: &str, kind: &str, value: Value) -> Result<Value> {
    if contract_version == "1.1.0" && kind.starts_with("knowledge.cag_cache_") {
        return normalize_knowledge_lifecycle(kind, value);
    }
    if contract_version == "1.2.0" && kind.starts_with("knowledge.static_prefix_material_") {
        return match kind {
            "knowledge.static_prefix_material_request" => {
                normalize::<StaticPrefixMaterialRequest>(value)
            }
            "knowledge.static_prefix_material_result" => {
                normalize::<StaticPrefixMaterialResult>(value)
            }
            _ => Err(ContractError::field(
                "kind",
                "unsupported knowledge 1.2 kind",
            )),
        };
    }
    if contract_version == "1.0.0" && kind.starts_with("knowledge.") {
        return normalize_knowledge_v1(kind, value);
    }
    match (contract_version, kind) {
        ("1.0.0", "cognitive.plan") => normalize::<CognitivePlan>(value),
        ("1.0.0", "cognitive.execution_request") => normalize::<CognitiveExecutionRequest>(value),
        ("1.0.0", "cognitive.execution_result") => normalize::<CognitiveExecutionResult>(value),
        ("2.0.0", "contract.production_version_offer") => {
            normalize::<ProductionVersionOffer>(value)
        }
        ("2.0.0", "contract.production_version_selection") => {
            normalize::<ProductionVersionSelection>(value)
        }
        ("2.0.0", "contract.production_version_transcript") => {
            normalize::<ProductionVersionTranscript>(value)
        }
        ("2.0.0", "qualification.model_capability_manifest") => {
            normalize::<ModelCapabilityManifest>(value)
        }
        ("3.0.0", "inference.runtime_availability") => normalize::<RuntimeAvailabilityV3>(value),
        ("3.0.0", "inference.reservation_request") => {
            normalize::<InferenceReservationRequestV3>(value)
        }
        ("3.0.0", "inference.reservation_decision") => {
            normalize::<InferenceReservationDecisionV3>(value)
        }
        ("3.0.0", "inference.execution_request") => normalize::<InferenceExecutionRequestV3>(value),
        ("3.0.0", "inference.execution_result") => normalize::<InferenceExecutionResultV3>(value),
        ("3.0.0", "inference.reservation_release") => {
            normalize::<InferenceReservationReleaseV3>(value)
        }
        ("3.0.0", "inference.runtime_cancellation_request") => {
            normalize::<RuntimeCancellationRequestV3>(value)
        }
        (_, "contract.version_offer") => normalize::<ContractVersionOffer>(value),
        (_, "contract.version_selection") => normalize::<ContractVersionSelection>(value),
        (_, "contract.version_transcript") => normalize::<VersionNegotiationTranscript>(value),
        (_, "inference.request") => normalize::<InferenceRequest>(value),
        (_, "generation.event") => normalize::<GenerationEvent>(value),
        (_, "engine.health") => normalize::<EngineHealth>(value),
        (_, "inference.cancellation_request") => normalize::<CancellationRequest>(value),
        (_, "inference.runtime_availability") => normalize::<RuntimeAvailability>(value),
        (_, "inference.reservation_request") => normalize::<InferenceReservationRequest>(value),
        (_, "inference.reservation_decision") => normalize::<InferenceReservationDecision>(value),
        (_, "inference.execution_request") => normalize::<InferenceExecutionRequest>(value),
        (_, "inference.execution_result") => normalize::<InferenceExecutionResult>(value),
        (_, "inference.reservation_release") => normalize::<InferenceReservationRelease>(value),
        (_, "inference.runtime_cancellation_request") => {
            normalize::<RuntimeCancellationRequest>(value)
        }
        (_, "memory.query") => normalize::<MemoryQuery>(value),
        (_, "memory.candidate") => normalize::<MemoryCandidate>(value),
        (_, "memory.write_proposal") => normalize::<MemoryWriteProposal>(value),
        (_, "memory.approved_write") => normalize::<ApprovedMemoryWrite>(value),
        (_, "memory.record") => normalize::<MemoryRecord>(value),
        (_, "tool.descriptor") => normalize::<ToolDescriptor>(value),
        (_, "tool.request") => normalize::<ToolRequest>(value),
        (_, "tool.execution_request") => normalize::<ExecutionRequest>(value),
        (_, "tool.execution_result") => normalize::<ExecutionResult>(value),
        (_, "tool.execution_error") => normalize::<ExecutionError>(value),
        (_, "security.event") => normalize::<SecurityEvent>(value),
        (_, "security.decision") => normalize::<SecurityDecision>(value),
        (_, "security.containment_request") => normalize::<ContainmentRequest>(value),
        (_, "security.containment_result") => normalize::<ContainmentResult>(value),
        (_, "model.artifact_descriptor") => normalize::<ModelArtifactDescriptor>(value),
        (_, "model.activation_request") => normalize::<ModelActivationRequest>(value),
        (_, "model.activation_state") => normalize::<ModelActivationState>(value),
        (_, "qualification.cognitive_manifest") => {
            normalize::<CognitiveQualificationManifest>(value)
        }
        (_, "task.envelope") => normalize::<TaskEnvelope>(value),
        (_, "task.status") => normalize::<TaskStatus>(value),
        (_, "node.descriptor") => normalize::<NodeDescriptor>(value),
        (_, "node.action_request") => normalize::<NodeActionRequest>(value),
        (_, "node.action_result") => normalize::<NodeActionResult>(value),
        (_, "audit.event") => normalize::<AuditEvent>(value),
        _ => Err(ContractError::field(
            "kind",
            format!("unsupported contract kind {kind:?}"),
        )),
    }
}

fn normalize_knowledge_v1(kind: &str, value: Value) -> Result<Value> {
    match kind {
        "knowledge.plan_request" => normalize::<KnowledgePlanRequest>(value),
        "knowledge.plan_result" => normalize::<KnowledgePlanResult>(value),
        "knowledge.static_bundle" => normalize::<StaticKnowledgeBundle>(value),
        "knowledge.dynamic_retrieval_result" => normalize::<DynamicRetrievalResult>(value),
        "knowledge.memory_revision_invalidated" => normalize::<MemoryRevisionInvalidated>(value),
        "knowledge.cag_cache_descriptor" => normalize::<CagCacheDescriptor>(value),
        "knowledge.cag_cache_query_request" => normalize::<CagCacheQueryRequest>(value),
        "knowledge.cag_cache_query_result" => normalize::<CagCacheQueryResult>(value),
        "knowledge.cache_compatibility" => normalize::<CacheCompatibility>(value),
        "knowledge.cag_cache_build_request" => normalize::<CagCacheBuildRequest>(value),
        "knowledge.cag_cache_build_result" => normalize::<CagCacheBuildResult>(value),
        "knowledge.cag_cache_invalidation_request" => {
            normalize::<CagCacheInvalidationRequest>(value)
        }
        "knowledge.cag_cache_invalidation_result" => normalize::<CagCacheInvalidationResult>(value),
        _ => Err(ContractError::field(
            "kind",
            format!("unsupported knowledge v1 kind {kind:?}"),
        )),
    }
}

fn normalize_knowledge_lifecycle(kind: &str, value: Value) -> Result<Value> {
    match kind {
        "knowledge.cag_cache_create_request" => normalize::<CagCacheCreateRequest>(value),
        "knowledge.cag_cache_create_result" => normalize::<CagCacheCreateResult>(value),
        "knowledge.cag_cache_describe_request" => normalize::<CagCacheDescribeRequest>(value),
        "knowledge.cag_cache_describe_result" => normalize::<CagCacheDescribeResult>(value),
        "knowledge.cag_cache_list_request" => normalize::<CagCacheListRequest>(value),
        "knowledge.cag_cache_list_result" => normalize::<CagCacheListResult>(value),
        "knowledge.cag_cache_attach_request" => normalize::<CagCacheAttachRequest>(value),
        "knowledge.cag_cache_attach_result" => normalize::<CagCacheAttachResult>(value),
        "knowledge.cag_cache_release_request" => normalize::<CagCacheReleaseRequest>(value),
        "knowledge.cag_cache_release_result" => normalize::<CagCacheReleaseResult>(value),
        "knowledge.cag_cache_invalidate_request" => normalize::<CagCacheInvalidateRequest>(value),
        "knowledge.cag_cache_invalidate_result" => normalize::<CagCacheInvalidateResult>(value),
        "knowledge.cag_cache_metrics_request" => normalize::<CagCacheMetricsRequest>(value),
        "knowledge.cag_cache_metrics_result" => normalize::<CagCacheMetricsResult>(value),
        _ => Err(ContractError::field(
            "kind",
            format!("unsupported knowledge lifecycle kind {kind:?}"),
        )),
    }
}

fn normalize<T>(value: Value) -> Result<Value>
where
    T: DeserializeOwned + Serialize + Validate,
{
    let typed: T = serde_json::from_value(value)?;
    typed.validate()?;
    Ok(serde_json::to_value(typed)?)
}

pub(crate) fn sort_value(value: Value) -> Value {
    match value {
        Value::Object(object) => {
            let mut entries: Vec<(String, Value)> = object.into_iter().collect();
            entries.sort_by(|left, right| left.0.cmp(&right.0));
            Value::Object(
                entries
                    .into_iter()
                    .map(|(key, value)| (key, sort_value(value)))
                    .collect(),
            )
        }
        Value::Array(values) => Value::Array(values.into_iter().map(sort_value).collect()),
        other => other,
    }
}

pub(crate) fn parse_strict_value(source: &[u8]) -> Result<Value> {
    let mut deserializer = serde_json::Deserializer::from_slice(source);
    let value = StrictValue::deserialize(&mut deserializer)?.0;
    deserializer.end()?;
    Ok(value)
}

struct StrictValue(Value);

impl<'de> Deserialize<'de> for StrictValue {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_any(StrictValueVisitor)
    }
}

struct StrictValueVisitor;

impl<'de> Visitor<'de> for StrictValueVisitor {
    type Value = StrictValue;

    fn expecting(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("a JSON value without duplicate object keys")
    }

    fn visit_bool<E>(self, value: bool) -> std::result::Result<Self::Value, E> {
        Ok(StrictValue(Value::Bool(value)))
    }

    fn visit_i64<E>(self, value: i64) -> std::result::Result<Self::Value, E> {
        Ok(StrictValue(Value::Number(Number::from(value))))
    }

    fn visit_u64<E>(self, value: u64) -> std::result::Result<Self::Value, E> {
        Ok(StrictValue(Value::Number(Number::from(value))))
    }

    fn visit_f64<E>(self, value: f64) -> std::result::Result<Self::Value, E>
    where
        E: DeError,
    {
        Number::from_f64(value)
            .map(Value::Number)
            .map(StrictValue)
            .ok_or_else(|| E::custom("non-finite JSON number"))
    }

    fn visit_str<E>(self, value: &str) -> std::result::Result<Self::Value, E> {
        Ok(StrictValue(Value::String(value.to_owned())))
    }

    fn visit_string<E>(self, value: String) -> std::result::Result<Self::Value, E> {
        Ok(StrictValue(Value::String(value)))
    }

    fn visit_none<E>(self) -> std::result::Result<Self::Value, E> {
        Ok(StrictValue(Value::Null))
    }

    fn visit_unit<E>(self) -> std::result::Result<Self::Value, E> {
        Ok(StrictValue(Value::Null))
    }

    fn visit_some<D>(self, deserializer: D) -> std::result::Result<Self::Value, D::Error>
    where
        D: Deserializer<'de>,
    {
        StrictValue::deserialize(deserializer)
    }

    fn visit_seq<A>(self, mut sequence: A) -> std::result::Result<Self::Value, A::Error>
    where
        A: SeqAccess<'de>,
    {
        let mut values = Vec::with_capacity(sequence.size_hint().unwrap_or(0).min(4_096));
        while let Some(value) = sequence.next_element::<StrictValue>()? {
            values.push(value.0);
        }
        Ok(StrictValue(Value::Array(values)))
    }

    fn visit_map<A>(self, mut object: A) -> std::result::Result<Self::Value, A::Error>
    where
        A: MapAccess<'de>,
    {
        let mut values = Map::new();
        while let Some(key) = object.next_key::<String>()? {
            if values.contains_key(&key) {
                return Err(A::Error::custom(format!("duplicate object key {key:?}")));
            }
            let value = object.next_value::<StrictValue>()?;
            values.insert(key, value.0);
        }
        Ok(StrictValue(Value::Object(values)))
    }
}

#[cfg(test)]
mod tests {
    use super::{document_registry, validate_document};
    use std::collections::BTreeSet;

    #[test]
    fn registry_is_sorted_unique_and_complete() {
        let registry = document_registry();
        assert!(!registry.is_empty());
        let identities: Vec<(&str, &str)> = registry
            .iter()
            .map(|entry| (entry.contract_version, entry.kind))
            .collect();
        assert!(identities.windows(2).all(|pair| pair[0] < pair[1]));
        assert_eq!(
            identities.iter().copied().collect::<BTreeSet<_>>().len(),
            registry.len()
        );
    }

    #[test]
    fn rejects_duplicate_keys_before_typed_decoding() {
        let source = br#"{"contract_version":"1.0.0","kind":"audit.event","kind":"tool.request"}"#;
        let error = validate_document(source).expect_err("duplicate keys must fail");
        assert!(error.to_string().contains("duplicate object key"));
    }

    #[test]
    fn rejects_unknown_kind_version_pairs() {
        let source = br#"{"contract_version":"1.1.0","kind":"inference.request"}"#;
        let error = validate_document(source).expect_err("cross-version pair must fail");
        assert!(error
            .to_string()
            .contains("unsupported contract kind/version pair"));
    }
}
