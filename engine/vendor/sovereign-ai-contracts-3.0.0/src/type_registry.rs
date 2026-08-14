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
