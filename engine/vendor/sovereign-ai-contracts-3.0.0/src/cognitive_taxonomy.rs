//! Canonical provider-neutral cognitive capability taxonomy revision v1.
//!
//! This module is immutable vocabulary only. It contains no model rankings,
//! qualification evidence, routing policy, or execution authority.

use crate::{ContractError, Result, Validate};
use serde::{Deserialize, Serialize};

pub const COGNITIVE_CAPABILITY_TAXONOMY_CONTRACT_VERSION: &str = "1.1.0";
pub const COGNITIVE_CAPABILITY_TAXONOMY_REVISION: &str = "cognitive-capabilities.v1";

pub const COGNITIVE_CAPABILITY_IDS: &[&str] = &[
    "additive_manufacturing",
    "anomaly_detection",
    "audio_understanding",
    "biology",
    "cad_geometry",
    "chemistry",
    "cnc_cam",
    "communication",
    "computer_science",
    "computer_use",
    "controls",
    "creative_ideation",
    "criticism",
    "cybersecurity",
    "document_intelligence",
    "electrical",
    "electrical.design",
    "electronics",
    "embeddings",
    "factual_validation",
    "forecasting",
    "image",
    "image.editing",
    "image.generation",
    "language",
    "manufacturing",
    "materials_science",
    "mathematics",
    "mechanical",
    "mechanical.design",
    "physics",
    "planning",
    "project_decomposition",
    "reasoning",
    "reasoning.deep",
    "reasoning.general",
    "reranking",
    "research",
    "retrieval",
    "robotics",
    "robotics.system_design",
    "simulation_interpretation",
    "software",
    "software.architecture",
    "software.compilers",
    "software.concurrency",
    "software.cpp",
    "software.cuda",
    "software.databases",
    "software.debugging",
    "software.distributed_systems",
    "software.embedded",
    "software.go",
    "software.implementation",
    "software.metal",
    "software.objective_cpp",
    "software.performance",
    "software.python",
    "software.refactoring",
    "software.repository_understanding",
    "software.review",
    "software.robotics",
    "software.rust",
    "software.security",
    "software.testing",
    "spatial_reasoning",
    "speech",
    "speech.recognition",
    "speech.synthesis",
    "time_series_reasoning",
    "tool_use",
    "translation",
    "verification",
    "video_generation",
    "video_understanding",
    "vision",
    "welding_joining",
];

#[must_use]
pub fn is_cognitive_capability(id: &str) -> bool {
    COGNITIVE_CAPABILITY_IDS.binary_search(&id).is_ok()
}

#[derive(Clone, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(transparent)]
pub struct CognitiveCapabilityId(pub String);

impl Validate for CognitiveCapabilityId {
    fn validate(&self) -> Result<()> {
        if is_cognitive_capability(&self.0) {
            Ok(())
        } else {
            Err(ContractError::field(
                "capability_id",
                format!("is not present in {COGNITIVE_CAPABILITY_TAXONOMY_REVISION}"),
            ))
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CognitiveCapabilityTaxonomy {
    pub contract_version: String,
    pub kind: String,
    pub taxonomy_revision: String,
    pub capabilities: Vec<CognitiveCapabilityId>,
}

impl Validate for CognitiveCapabilityTaxonomy {
    fn validate(&self) -> Result<()> {
        if self.contract_version != COGNITIVE_CAPABILITY_TAXONOMY_CONTRACT_VERSION {
            return Err(ContractError::field(
                "contract_version",
                format!("must equal {COGNITIVE_CAPABILITY_TAXONOMY_CONTRACT_VERSION}"),
            ));
        }
        if self.kind != "cognitive.capability_taxonomy" {
            return Err(ContractError::field(
                "kind",
                "must equal cognitive.capability_taxonomy",
            ));
        }
        if self.taxonomy_revision != COGNITIVE_CAPABILITY_TAXONOMY_REVISION {
            return Err(ContractError::field(
                "taxonomy_revision",
                format!("must equal {COGNITIVE_CAPABILITY_TAXONOMY_REVISION}"),
            ));
        }
        if self.capabilities.len() != COGNITIVE_CAPABILITY_IDS.len()
            || self
                .capabilities
                .iter()
                .zip(COGNITIVE_CAPABILITY_IDS)
                .any(|(actual, expected)| actual.0 != *expected)
        {
            return Err(ContractError::field(
                "capabilities",
                "must exactly equal the canonical v1 catalog",
            ));
        }
        Ok(())
    }
}

#[must_use]
pub fn canonical_cognitive_capability_taxonomy() -> CognitiveCapabilityTaxonomy {
    CognitiveCapabilityTaxonomy {
        contract_version: COGNITIVE_CAPABILITY_TAXONOMY_CONTRACT_VERSION.to_owned(),
        kind: "cognitive.capability_taxonomy".to_owned(),
        taxonomy_revision: COGNITIVE_CAPABILITY_TAXONOMY_REVISION.to_owned(),
        capabilities: COGNITIVE_CAPABILITY_IDS
            .iter()
            .map(|id| CognitiveCapabilityId((*id).to_owned()))
            .collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::{
        canonical_cognitive_capability_taxonomy, is_cognitive_capability,
        CognitiveCapabilityTaxonomy, COGNITIVE_CAPABILITY_IDS,
    };
    use crate::Validate;

    #[test]
    fn canonical_v1_membership_is_complete_and_exact() {
        let taxonomy = canonical_cognitive_capability_taxonomy();
        taxonomy.validate().expect("canonical taxonomy");
        assert_eq!(COGNITIVE_CAPABILITY_IDS.len(), 77);
        assert!(is_cognitive_capability("mathematics"));
        assert!(is_cognitive_capability("reasoning.general"));
        assert!(is_cognitive_capability("software.implementation"));
        assert!(!is_cognitive_capability("software.golang"));
        assert!(COGNITIVE_CAPABILITY_IDS
            .windows(2)
            .all(|pair| pair[0] < pair[1]));
    }

    #[test]
    fn shared_fixture_matches_compiled_taxonomy() {
        let raw = include_bytes!("../../fixtures/v1/valid/cognitive-capability-taxonomy-v1.json");
        let fixture: CognitiveCapabilityTaxonomy =
            serde_json::from_slice(raw).expect("decode taxonomy fixture");
        fixture.validate().expect("validate taxonomy fixture");
        assert_eq!(fixture, canonical_cognitive_capability_taxonomy());
    }
}
