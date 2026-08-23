//! Compile-bound descriptive producer and consumer registry.

use serde::Serialize;

pub(crate) const ALL_COMPONENTS: &[&str] = &[
    "action_broker",
    "agent_green",
    "creator_os",
    "memory_vault",
    "model_foundry",
    "native_inference_engine",
    "node_runtime",
    "personal_ai_platform",
    "tradeforge",
];

/// One released schema and its expected producer and consumer endpoints.
/// Presence in this registry never grants access or authority.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct ComponentContractRegistryEntry {
    pub contract_version: &'static str,
    pub domain: &'static str,
    pub schema: &'static str,
    pub producers: &'static [&'static str],
    pub consumers: &'static [&'static str],
}

const fn entry(
    contract_version: &'static str,
    domain: &'static str,
    schema: &'static str,
    producers: &'static [&'static str],
    consumers: &'static [&'static str],
) -> ComponentContractRegistryEntry {
    ComponentContractRegistryEntry {
        contract_version,
        domain,
        schema,
        producers,
        consumers,
    }
}

const COMPONENT_CONTRACT_REGISTRY: &[ComponentContractRegistryEntry] = &[
    entry(
        "1.0.0",
        "audit",
        "schemas/audit/v1/audit.schema.json",
        ALL_COMPONENTS,
        &["authorized_audit_store"],
    ),
    entry(
        "1.0.0",
        "cognitive",
        "schemas/cognitive/v1/cognitive.schema.json",
        &["personal_ai_platform"],
        &["personal_ai_platform"],
    ),
    entry(
        "1.0.0",
        "common",
        "schemas/common/v1/common.schema.json",
        ALL_COMPONENTS,
        ALL_COMPONENTS,
    ),
    entry(
        "1.0.0",
        "identity",
        "schemas/identity/v1/identity.schema.json",
        ALL_COMPONENTS,
        ALL_COMPONENTS,
    ),
    entry(
        "1.0.0",
        "inference",
        "schemas/inference/v1/inference.schema.json",
        &[
            "creator_os",
            "native_inference_engine",
            "personal_ai_platform",
            "tradeforge",
        ],
        &[
            "creator_os",
            "native_inference_engine",
            "personal_ai_platform",
            "tradeforge",
        ],
    ),
    entry(
        "1.0.0",
        "knowledge",
        "schemas/knowledge/v1/knowledge.schema.json",
        &[
            "memory_vault",
            "native_inference_engine",
            "personal_ai_platform",
        ],
        &[
            "memory_vault",
            "native_inference_engine",
            "personal_ai_platform",
        ],
    ),
    entry(
        "1.0.0",
        "memory",
        "schemas/memory/v1/memory.schema.json",
        &["memory_vault", "personal_ai_platform"],
        &["memory_vault", "personal_ai_platform"],
    ),
    entry(
        "1.0.0",
        "models",
        "schemas/models/v1/models.schema.json",
        &[
            "model_foundry",
            "native_inference_engine",
            "personal_ai_platform",
        ],
        &[
            "model_foundry",
            "native_inference_engine",
            "node_runtime",
            "personal_ai_platform",
        ],
    ),
    entry(
        "1.0.0",
        "nodes",
        "schemas/nodes/v1/nodes.schema.json",
        &["action_broker", "node_runtime", "personal_ai_platform"],
        &[
            "action_broker",
            "agent_green",
            "node_runtime",
            "personal_ai_platform",
        ],
    ),
    entry(
        "1.0.0",
        "qualification",
        "schemas/qualification/v1/qualification.schema.json",
        &["model_foundry"],
        &["personal_ai_platform"],
    ),
    entry(
        "1.0.0",
        "security",
        "schemas/security/v1/security.schema.json",
        &["agent_green", "node_runtime"],
        &[
            "action_broker",
            "agent_green",
            "node_runtime",
            "personal_ai_platform",
        ],
    ),
    entry(
        "1.0.0",
        "tasks",
        "schemas/tasks/v1/tasks.schema.json",
        &[
            "creator_os",
            "node_runtime",
            "personal_ai_platform",
            "tradeforge",
        ],
        &[
            "creator_os",
            "node_runtime",
            "personal_ai_platform",
            "tradeforge",
        ],
    ),
    entry(
        "1.0.0",
        "tools",
        "schemas/tools/v1/tools.schema.json",
        &[
            "action_broker",
            "creator_os",
            "personal_ai_platform",
            "tradeforge",
        ],
        &[
            "action_broker",
            "creator_os",
            "node_runtime",
            "personal_ai_platform",
            "tradeforge",
        ],
    ),
    entry(
        "1.1.0",
        "common",
        "schemas/common/v1/common-1.1.0.schema.json",
        ALL_COMPONENTS,
        ALL_COMPONENTS,
    ),
    entry(
        "1.1.0",
        "identity",
        "schemas/identity/v1/identity-1.1.0.schema.json",
        &["model_foundry", "personal_ai_platform"],
        &[
            "model_foundry",
            "native_inference_engine",
            "personal_ai_platform",
        ],
    ),
    entry(
        "1.1.0",
        "knowledge",
        "schemas/knowledge/v1/knowledge-1.1.0.schema.json",
        &["native_inference_engine", "personal_ai_platform"],
        &["native_inference_engine", "personal_ai_platform"],
    ),
    entry(
        "1.2.0",
        "identity",
        "schemas/identity/v1/identity-1.2.0.schema.json",
        &[
            "memory_vault",
            "native_inference_engine",
            "personal_ai_platform",
        ],
        &[
            "memory_vault",
            "native_inference_engine",
            "personal_ai_platform",
        ],
    ),
    entry(
        "1.2.0",
        "knowledge",
        "schemas/knowledge/v1/knowledge-1.2.0.schema.json",
        &["memory_vault", "personal_ai_platform"],
        &["memory_vault", "personal_ai_platform"],
    ),
    entry(
        "1.3.0",
        "knowledge",
        "schemas/knowledge/v1/knowledge-1.3.0.schema.json",
        &["memory_vault", "personal_ai_platform"],
        &["memory_vault", "personal_ai_platform"],
    ),
    entry(
        "2.0.0",
        "common",
        "schemas/common/v2/common.schema.json",
        ALL_COMPONENTS,
        ALL_COMPONENTS,
    ),
    entry(
        "2.0.0",
        "inference",
        "schemas/inference/v2/inference.schema.json",
        &["native_inference_engine", "personal_ai_platform"],
        &["native_inference_engine", "personal_ai_platform"],
    ),
    entry(
        "2.0.0",
        "models",
        "schemas/models/v2/models.schema.json",
        &["model_foundry"],
        &["native_inference_engine", "personal_ai_platform"],
    ),
    entry(
        "2.0.0",
        "qualification",
        "schemas/qualification/v2/qualification.schema.json",
        &["model_foundry"],
        &["personal_ai_platform"],
    ),
    entry(
        "2.1.0",
        "qualification",
        "schemas/qualification/v2/qualification-2.1.0.schema.json",
        &["model_foundry"],
        &["personal_ai_platform"],
    ),
    entry(
        "3.0.0",
        "inference",
        "schemas/inference/v3/inference.schema.json",
        &["native_inference_engine", "personal_ai_platform"],
        &["native_inference_engine", "personal_ai_platform"],
    ),
];

/// Returns the complete deterministic routing registry.
#[must_use]
pub fn component_contract_registry() -> Vec<ComponentContractRegistryEntry> {
    COMPONENT_CONTRACT_REGISTRY.to_vec()
}

#[cfg(test)]
mod tests {
    use super::component_contract_registry;

    #[test]
    fn registry_is_nonempty_sorted_unique_and_bounded() {
        let registry = component_contract_registry();
        assert!(!registry.is_empty());
        for pair in registry.windows(2) {
            assert!(
                (pair[0].contract_version, pair[0].domain)
                    < (pair[1].contract_version, pair[1].domain)
            );
        }
        for entry in registry {
            assert!(!entry.producers.is_empty() && entry.producers.len() <= 10);
            assert!(!entry.consumers.is_empty() && entry.consumers.len() <= 10);
            assert!(entry.producers.windows(2).all(|pair| pair[0] < pair[1]));
            assert!(entry.consumers.windows(2).all(|pair| pair[0] < pair[1]));
            assert!(entry.schema.starts_with("schemas/"));
            assert!(entry.schema.ends_with(".schema.json"));
        }
    }
}
