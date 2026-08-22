//! Compile-bound registry for required reusable contract types.

use serde::Serialize;

/// One required reusable schema definition and its first-party binding names.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct PublicTypeRegistryEntry {
    pub contract_version: &'static str,
    pub domain: &'static str,
    pub definition: &'static str,
    pub schema: &'static str,
    pub go_type: &'static str,
    pub rust_type: &'static str,
}

fn binding_name<T>() -> &'static str {
    std::any::type_name::<T>()
        .rsplit("::")
        .next()
        .expect("a Rust type name is never empty")
}

fn entry<T>(
    domain: &'static str,
    definition: &'static str,
    schema: &'static str,
    go_type: &'static str,
) -> PublicTypeRegistryEntry {
    entry_version::<T>(crate::CONTRACT_VERSION, domain, definition, schema, go_type)
}

fn entry_version<T>(
    version: &'static str,
    domain: &'static str,
    definition: &'static str,
    schema: &'static str,
    go_type: &'static str,
) -> PublicTypeRegistryEntry {
    PublicTypeRegistryEntry {
        contract_version: version,
        domain,
        definition,
        schema,
        go_type,
        rust_type: binding_name::<T>(),
    }
}

macro_rules! type_entries {
    ($domain:literal, $schema:literal; $(($binding:ty, $definition:literal, $go_type:literal)),+ $(,)?) => {
        vec![$(entry::<$binding>($domain, $definition, $schema, $go_type)),+]
    };
}

fn runtime_type_entries() -> Vec<PublicTypeRegistryEntry> {
    vec![
        entry_version::<crate::inference_runtime_v2::RuntimeIdentity>(
            crate::INFERENCE_RUNTIME_CONTRACT_VERSION,
            "inference",
            "RuntimeIdentity",
            "inference/v2/inference.schema.json",
            "RuntimeIdentity",
        ),
        entry_version::<crate::inference_runtime_v2::RuntimeResourceBudget>(
            crate::INFERENCE_RUNTIME_CONTRACT_VERSION,
            "inference",
            "RuntimeResourceBudget",
            "inference/v2/inference.schema.json",
            "RuntimeResourceBudget",
        ),
        entry_version::<crate::inference_runtime_v2::RuntimeModelStatus>(
            crate::INFERENCE_RUNTIME_CONTRACT_VERSION,
            "inference",
            "RuntimeModelStatus",
            "inference/v2/inference.schema.json",
            "RuntimeModelStatus",
        ),
        entry_version::<crate::inference_runtime_v2::RuntimeResourceUsage>(
            crate::INFERENCE_RUNTIME_CONTRACT_VERSION,
            "inference",
            "RuntimeResourceUsage",
            "inference/v2/inference.schema.json",
            "RuntimeResourceUsage",
        ),
        entry_version::<crate::inference_runtime_v2::RuntimeFailure>(
            crate::INFERENCE_RUNTIME_CONTRACT_VERSION,
            "inference",
            "RuntimeFailure",
            "inference/v2/inference.schema.json",
            "RuntimeFailure",
        ),
        entry_version::<crate::inference_runtime_v3::RuntimeIdentityV3>(
            crate::INFERENCE_RUNTIME_CONTRACT_VERSION_V3,
            "inference",
            "RuntimeIdentityV3",
            "inference/v3/inference.schema.json",
            "RuntimeIdentityV3",
        ),
        entry_version::<crate::inference_runtime_v3::RuntimeResourceBudgetV3>(
            crate::INFERENCE_RUNTIME_CONTRACT_VERSION_V3,
            "inference",
            "RuntimeResourceBudgetV3",
            "inference/v3/inference.schema.json",
            "RuntimeResourceBudgetV3",
        ),
        entry_version::<crate::inference_runtime_v3::RuntimeModelStatusV3>(
            crate::INFERENCE_RUNTIME_CONTRACT_VERSION_V3,
            "inference",
            "RuntimeModelStatusV3",
            "inference/v3/inference.schema.json",
            "RuntimeModelStatusV3",
        ),
        entry_version::<crate::inference_runtime_v3::RuntimeResourceUsageV3>(
            crate::INFERENCE_RUNTIME_CONTRACT_VERSION_V3,
            "inference",
            "RuntimeResourceUsageV3",
            "inference/v3/inference.schema.json",
            "RuntimeResourceUsageV3",
        ),
        entry_version::<crate::inference_runtime_v3::RuntimeFailureV3>(
            crate::INFERENCE_RUNTIME_CONTRACT_VERSION_V3,
            "inference",
            "RuntimeFailureV3",
            "inference/v3/inference.schema.json",
            "RuntimeFailureV3",
        ),
    ]
}

/// Returns every required reusable type in deterministic order.
///
/// Each generic entry directly references its concrete Rust binding, so
/// removing or renaming that type is a compile-time failure.
#[must_use]
#[allow(clippy::too_many_lines)]
pub fn public_type_registry() -> Vec<PublicTypeRegistryEntry> {
    let mut registry = type_entries!(
        "audit", "audit/v1/audit.schema.json";
        (crate::audit::ApprovalAuditPayload, "ApprovalAuditPayload", "ApprovalAuditPayload"),
        (crate::audit::ConfigurationChangeAuditPayload, "ConfigurationChangeAuditPayload", "ConfigurationChangeAuditPayload"),
        (crate::audit::InferenceAuditPayload, "InferenceAuditPayload", "InferenceAuditPayload"),
        (crate::audit::MemoryRetrievalAuditPayload, "MemoryRetrievalAuditPayload", "MemoryRetrievalAuditPayload"),
        (crate::audit::MemoryWriteAuditPayload, "MemoryWriteAuditPayload", "MemoryWriteAuditPayload"),
        (crate::audit::ModelActivationAuditPayload, "ModelActivationAuditPayload", "ModelActivationAuditPayload"),
        (crate::audit::ModelImportAuditPayload, "ModelImportAuditPayload", "ModelImportAuditPayload"),
        (crate::audit::NodeActionAuditPayload, "NodeActionAuditPayload", "NodeActionAuditPayload"),
        (crate::audit::SecurityDecisionAuditPayload, "SecurityDecisionAuditPayload", "SecurityDecisionAuditPayload"),
        (crate::audit::ToolExecutionAuditPayload, "ToolExecutionAuditPayload", "ToolExecutionAuditPayload"),
        (crate::audit::ToolRequestAuditPayload, "ToolRequestAuditPayload", "ToolRequestAuditPayload"),
    );
    registry.extend(type_entries!(
        "identity", "identity/v1/identity.schema.json";
        (crate::identity::AgentId, "AgentId", "AgentID"),
        (crate::identity::AuditEventId, "AuditEventId", "AuditEventID"),
        (crate::identity::MemoryId, "MemoryId", "MemoryID"),
        (crate::identity::ModelArtifactId, "ModelArtifactId", "ModelArtifactID"),
        (crate::identity::ModelId, "ModelId", "ModelID"),
        (crate::identity::NodeId, "NodeId", "NodeID"),
        (crate::identity::RequestId, "RequestId", "RequestID"),
        (crate::identity::SecurityEventId, "SecurityEventId", "SecurityEventID"),
        (crate::identity::SessionId, "SessionId", "SessionID"),
        (crate::identity::TaskId, "TaskId", "TaskID"),
        (crate::identity::ToolId, "ToolId", "ToolID"),
        (crate::identity::UserId, "UserId", "UserID"),
    ));
    registry.extend(type_entries!(
        "inference", "inference/v1/inference.schema.json";
        (crate::inference::BackendCapability, "BackendCapability", "BackendCapability"),
        (crate::inference::GenerationParameters, "GenerationParameters", "GenerationParameters"),
        (crate::inference::GenerationUsage, "GenerationUsage", "GenerationUsage"),
        (crate::inference::ModelCapability, "ModelCapability", "ModelCapability"),
        (crate::inference::ResourceStatus, "ResourceStatus", "ResourceStatus"),
    ));
    registry.extend(runtime_type_entries());
    registry.extend(vec![
        entry_version::<crate::cognitive::CapabilityRequirement>(
            "1.0.0",
            "cognitive",
            "CapabilityRequirement",
            "cognitive/v1/cognitive.schema.json",
            "CapabilityRequirement",
        ),
        entry_version::<crate::cognitive::ResourceRequirementEnvelope>(
            "1.0.0",
            "cognitive",
            "ResourceRequirementEnvelope",
            "cognitive/v1/cognitive.schema.json",
            "ResourceRequirementEnvelope",
        ),
        entry_version::<crate::cognitive::ExecutionResourceCeiling>(
            "1.0.0",
            "cognitive",
            "ExecutionResourceCeiling",
            "cognitive/v1/cognitive.schema.json",
            "ExecutionResourceCeiling",
        ),
        entry_version::<crate::cognitive::QualificationManifestReference>(
            "1.0.0",
            "cognitive",
            "QualificationManifestReference",
            "cognitive/v1/cognitive.schema.json",
            "QualificationManifestReference",
        ),
        entry_version::<crate::cognitive::SpecialistAssignment>(
            "1.0.0",
            "cognitive",
            "SpecialistAssignment",
            "cognitive/v1/cognitive.schema.json",
            "SpecialistAssignment",
        ),
        entry_version::<crate::cognitive::VerificationRequirement>(
            "1.0.0",
            "cognitive",
            "VerificationRequirement",
            "cognitive/v1/cognitive.schema.json",
            "VerificationRequirement",
        ),
        entry_version::<crate::cognitive::SynthesisRequirement>(
            "1.0.0",
            "cognitive",
            "SynthesisRequirement",
            "cognitive/v1/cognitive.schema.json",
            "SynthesisRequirement",
        ),
        entry_version::<crate::cognitive::CognitivePlanNode>(
            "1.0.0",
            "cognitive",
            "CognitivePlanNode",
            "cognitive/v1/cognitive.schema.json",
            "CognitivePlanNode",
        ),
        entry_version::<crate::negotiation_v2::ProductionContractCandidate>(
            "2.0.0",
            "common",
            "ProductionContractCandidate",
            "common/v2/common.schema.json",
            "ProductionContractCandidate",
        ),
        entry_version::<crate::identity::CognitiveExecutionResultId>(
            "1.1.0",
            "identity",
            "CognitiveExecutionResultId",
            "identity/v1/identity-1.1.0.schema.json",
            "CognitiveExecutionResultID",
        ),
        entry_version::<crate::identity::CognitivePlanId>(
            "1.1.0",
            "identity",
            "CognitivePlanId",
            "identity/v1/identity-1.1.0.schema.json",
            "CognitivePlanID",
        ),
        entry_version::<crate::identity::CognitivePlanNodeId>(
            "1.1.0",
            "identity",
            "CognitivePlanNodeId",
            "identity/v1/identity-1.1.0.schema.json",
            "CognitivePlanNodeID",
        ),
        entry_version::<crate::identity::EvaluationRevisionId>(
            "1.1.0",
            "identity",
            "EvaluationRevisionId",
            "identity/v1/identity-1.1.0.schema.json",
            "EvaluationRevisionID",
        ),
        entry_version::<crate::identity::QualificationEvidenceId>(
            "1.1.0",
            "identity",
            "QualificationEvidenceId",
            "identity/v1/identity-1.1.0.schema.json",
            "QualificationEvidenceID",
        ),
        entry_version::<crate::identity::ProjectId>(
            "1.2.0",
            "identity",
            "ProjectId",
            "identity/v1/identity-1.2.0.schema.json",
            "ProjectID",
        ),
        entry_version::<crate::identity::KnowledgePlanId>(
            "1.2.0",
            "identity",
            "KnowledgePlanId",
            "identity/v1/identity-1.2.0.schema.json",
            "KnowledgePlanID",
        ),
        entry_version::<crate::identity::KnowledgeQueryId>(
            "1.2.0",
            "identity",
            "KnowledgeQueryId",
            "identity/v1/identity-1.2.0.schema.json",
            "KnowledgeQueryID",
        ),
        entry_version::<crate::identity::KnowledgeSourceId>(
            "1.2.0",
            "identity",
            "KnowledgeSourceId",
            "identity/v1/identity-1.2.0.schema.json",
            "KnowledgeSourceID",
        ),
        entry_version::<crate::identity::KnowledgeRevisionId>(
            "1.2.0",
            "identity",
            "KnowledgeRevisionId",
            "identity/v1/identity-1.2.0.schema.json",
            "KnowledgeRevisionID",
        ),
        entry_version::<crate::identity::StaticKnowledgeBundleId>(
            "1.2.0",
            "identity",
            "StaticKnowledgeBundleId",
            "identity/v1/identity-1.2.0.schema.json",
            "StaticKnowledgeBundleID",
        ),
        entry_version::<crate::identity::CagCacheId>(
            "1.2.0",
            "identity",
            "CagCacheId",
            "identity/v1/identity-1.2.0.schema.json",
            "CagCacheID",
        ),
        entry_version::<crate::identity::KnowledgeInvalidationEventId>(
            "1.2.0",
            "identity",
            "KnowledgeInvalidationEventId",
            "identity/v1/identity-1.2.0.schema.json",
            "KnowledgeInvalidationEventID",
        ),
        entry_version::<crate::knowledge::KnowledgeMode>(
            "1.0.0",
            "knowledge",
            "KnowledgeMode",
            "knowledge/v1/knowledge.schema.json",
            "KnowledgeMode",
        ),
        entry_version::<crate::knowledge::FreshnessClass>(
            "1.0.0",
            "knowledge",
            "FreshnessClass",
            "knowledge/v1/knowledge.schema.json",
            "FreshnessClass",
        ),
        entry_version::<crate::knowledge::CagEligibility>(
            "1.0.0",
            "knowledge",
            "CagEligibility",
            "knowledge/v1/knowledge.schema.json",
            "CagEligibility",
        ),
        entry_version::<crate::knowledge::KnowledgeScope>(
            "1.0.0",
            "knowledge",
            "KnowledgeScope",
            "knowledge/v1/knowledge.schema.json",
            "KnowledgeScope",
        ),
        entry_version::<crate::knowledge::KnowledgeProducerIdentity>(
            "1.0.0",
            "knowledge",
            "KnowledgeProducerIdentity",
            "knowledge/v1/knowledge.schema.json",
            "KnowledgeProducerIdentity",
        ),
        entry_version::<crate::knowledge::KnowledgeRequirement>(
            "1.0.0",
            "knowledge",
            "KnowledgeRequirement",
            "knowledge/v1/knowledge.schema.json",
            "KnowledgeRequirement",
        ),
        entry_version::<crate::knowledge::ContextBudgetAllocation>(
            "1.0.0",
            "knowledge",
            "ContextBudgetAllocation",
            "knowledge/v1/knowledge.schema.json",
            "ContextBudgetAllocation",
        ),
        entry_version::<crate::knowledge::SourceExclusion>(
            "1.0.0",
            "knowledge",
            "SourceExclusion",
            "knowledge/v1/knowledge.schema.json",
            "SourceExclusion",
        ),
        entry_version::<crate::knowledge::StaticBundleReference>(
            "1.0.0",
            "knowledge",
            "StaticBundleReference",
            "knowledge/v1/knowledge.schema.json",
            "StaticBundleReference",
        ),
        entry_version::<crate::knowledge::CacheBinding>(
            "1.0.0",
            "knowledge",
            "CacheBinding",
            "knowledge/v1/knowledge.schema.json",
            "CacheBinding",
        ),
        entry_version::<crate::knowledge::CacheCompatibility>(
            "1.0.0",
            "knowledge",
            "CacheCompatibility",
            "knowledge/v1/knowledge.schema.json",
            "CacheCompatibility",
        ),
        entry_version::<crate::knowledge_lifecycle::CacheNamespace>(
            "1.1.0",
            "knowledge",
            "CacheNamespace",
            "knowledge/v1/knowledge-1.1.0.schema.json",
            "CacheNamespace",
        ),
        entry_version::<crate::knowledge_lifecycle::KvDtype>(
            "1.1.0",
            "knowledge",
            "KvDtype",
            "knowledge/v1/knowledge-1.1.0.schema.json",
            "KVDtype",
        ),
        entry_version::<crate::knowledge_lifecycle::CacheAuthorizationScope>(
            "1.1.0",
            "knowledge",
            "CacheAuthorizationScope",
            "knowledge/v1/knowledge-1.1.0.schema.json",
            "CacheAuthorizationScope",
        ),
        entry_version::<crate::knowledge_lifecycle::CacheLifecycleBinding>(
            "1.1.0",
            "knowledge",
            "CacheLifecycleBinding",
            "knowledge/v1/knowledge-1.1.0.schema.json",
            "CacheLifecycleBinding",
        ),
        entry_version::<crate::knowledge_lifecycle::CagCacheLifecycleDescriptor>(
            "1.1.0",
            "knowledge",
            "CagCacheLifecycleDescriptor",
            "knowledge/v1/knowledge-1.1.0.schema.json",
            "CagCacheLifecycleDescriptor",
        ),
        entry_version::<crate::knowledge_lifecycle::CacheInvalidationSelector>(
            "1.1.0",
            "knowledge",
            "CacheInvalidationSelector",
            "knowledge/v1/knowledge-1.1.0.schema.json",
            "CacheInvalidationSelector",
        ),
        entry_version::<crate::knowledge_lifecycle::CacheMetrics>(
            "1.1.0",
            "knowledge",
            "CacheMetrics",
            "knowledge/v1/knowledge-1.1.0.schema.json",
            "CacheMetrics",
        ),
        entry_version::<crate::knowledge_prefix::StaticPrefixMaterialStatus>(
            crate::knowledge_prefix::KNOWLEDGE_STATIC_PREFIX_CONTRACT_VERSION,
            "knowledge",
            "StaticPrefixMaterialStatus",
            "knowledge/v1/knowledge-1.2.0.schema.json",
            "StaticPrefixMaterialStatus",
        ),
        entry_version::<crate::knowledge_prefix::StaticPrefixMaterialRequest>(
            crate::knowledge_prefix::KNOWLEDGE_STATIC_PREFIX_CONTRACT_VERSION,
            "knowledge",
            "StaticPrefixMaterialRequest",
            "knowledge/v1/knowledge-1.2.0.schema.json",
            "StaticPrefixMaterialRequest",
        ),
        entry_version::<crate::knowledge_prefix::StaticPrefixMaterialResult>(
            crate::knowledge_prefix::KNOWLEDGE_STATIC_PREFIX_CONTRACT_VERSION,
            "knowledge",
            "StaticPrefixMaterialResult",
            "knowledge/v1/knowledge-1.2.0.schema.json",
            "StaticPrefixMaterialResult",
        ),
        entry_version::<crate::qualification_v2::CapabilityQualificationV2>(
            "2.0.0",
            "qualification",
            "CapabilityQualificationV2",
            "qualification/v2/qualification.schema.json",
            "CapabilityQualificationV2",
        ),
        entry_version::<crate::qualification_v2::QualificationEvidenceV2>(
            "2.0.0",
            "qualification",
            "QualificationEvidenceV2",
            "qualification/v2/qualification.schema.json",
            "QualificationEvidenceV2",
        ),
        entry_version::<crate::qualification_v2::ModelCapabilityPayload>(
            "2.0.0",
            "qualification",
            "ModelCapabilityPayload",
            "qualification/v2/qualification.schema.json",
            "ModelCapabilityPayload",
        ),
    ]);
    registry.extend(type_entries!(
        "memory", "memory/v1/memory.schema.json";
        (crate::memory::MemorySource, "MemorySource", "MemorySource"),
        (crate::memory::Provenance, "Provenance", "Provenance"),
        (crate::memory::RetentionPolicy, "RetentionPolicy", "RetentionPolicy"),
        (crate::memory::SensitivityClass, "SensitivityClass", "SensitivityClass"),
    ));
    registry.extend(type_entries!(
        "security", "security/v1/security.schema.json";
        (crate::security::SecurityEvidence, "SecurityEvidence", "SecurityEvidence"),
        (crate::security::SecuritySeverity, "SecuritySeverity", "SecuritySeverity"),
    ));
    registry.extend(type_entries!(
        "qualification", "qualification/v1/qualification.schema.json";
        (crate::qualification::QualificationManifestId, "QualificationManifestId", "QualificationManifestID"),
        (crate::qualification::QualificationProducerIdentity, "ProducerIdentity", "QualificationProducerIdentity"),
        (crate::qualification::QualifiedModelIdentity, "ModelIdentity", "QualifiedModelIdentity"),
        (crate::qualification::QualifiedProviderIdentity, "ProviderIdentity", "QualifiedProviderIdentity"),
        (crate::qualification::QualifiedArtifactIdentity, "ArtifactIdentity", "QualifiedArtifactIdentity"),
        (crate::qualification::QualifiedBackendIdentity, "BackendIdentity", "QualifiedBackendIdentity"),
        (crate::qualification::ConfidenceInterval, "ConfidenceInterval", "ConfidenceInterval"),
        (crate::qualification::CapabilityQualification, "CapabilityQualification", "CapabilityQualification"),
        (crate::qualification::QualificationContextLimits, "ContextLimits", "QualificationContextLimits"),
        (crate::qualification::QualificationOutputLimits, "OutputLimits", "QualificationOutputLimits"),
        (crate::qualification::QualificationResourceLimits, "ResourceLimits", "QualificationResourceLimits"),
        (crate::qualification::QualificationPerformance, "Performance", "QualificationPerformance"),
        (crate::qualification::QualificationReliability, "Reliability", "QualificationReliability"),
        (crate::qualification::QualificationCost, "Cost", "QualificationCost"),
        (crate::qualification::QualificationEvidence, "EvaluationEvidence", "QualificationEvidence"),
        (crate::qualification::QualificationLifecycleEvent, "LifecycleEvent", "QualificationLifecycleEvent"),
        (crate::qualification::QualificationAntiRollback, "AntiRollbackMetadata", "QualificationAntiRollback"),
        (crate::qualification::CognitiveQualificationPayload, "CognitiveQualificationPayload", "CognitiveQualificationPayload"),
        (crate::qualification::QualificationSignatureEnvelope, "SignatureEnvelope", "QualificationSignatureEnvelope"),
    ));
    registry.extend(type_entries!(
        "tools", "tools/v1/tools.schema.json";
        (crate::tools::ApprovalRequirement, "ApprovalRequirement", "ApprovalRequirement"),
        (crate::tools::ExecutionCapability, "ExecutionCapability", "ExecutionCapability"),
        (crate::tools::ToolArgument, "ToolArgument", "ToolArgument"),
        (crate::tools::ToolPermission, "ToolPermission", "ToolPermission"),
    ));
    registry.sort_by(|left, right| {
        (left.contract_version, left.domain, left.definition).cmp(&(
            right.contract_version,
            right.domain,
            right.definition,
        ))
    });
    registry
}

#[cfg(test)]
mod tests {
    use super::public_type_registry;

    #[test]
    fn registry_is_nonempty_sorted_and_unique() {
        let registry = public_type_registry();
        assert!(!registry.is_empty());
        for pair in registry.windows(2) {
            let left = (
                &pair[0].contract_version,
                &pair[0].domain,
                &pair[0].definition,
            );
            let right = (
                &pair[1].contract_version,
                &pair[1].domain,
                &pair[1].definition,
            );
            assert!(
                left < right,
                "registry is not strictly sorted: {left:?} {right:?}"
            );
        }
        assert!(registry.iter().all(|entry| {
            !entry.go_type.is_empty()
                && !entry.rust_type.is_empty()
                && entry.schema.ends_with(".schema.json")
        }));
    }
}
