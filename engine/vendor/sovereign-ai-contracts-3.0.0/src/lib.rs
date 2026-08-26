//! Strict Rust bindings for Sovereign AI inter-component contracts.
//!
//! This crate deliberately contains no application or business logic. It only
//! decodes, validates, and deterministically re-encodes published contracts.

#[cfg(feature = "full")]
pub mod audit;
#[cfg(feature = "full")]
pub mod cognitive;
#[cfg(feature = "full")]
pub mod cognitive_taxonomy;
#[cfg(feature = "runtime-v3")]
pub mod common;
#[cfg(feature = "full")]
pub mod component_kit;
#[cfg(feature = "full")]
pub mod component_matrix;
#[cfg(feature = "full")]
pub mod component_profile;
#[cfg(feature = "full")]
pub mod component_registry;
#[cfg(feature = "full")]
pub mod document;
#[cfg(feature = "runtime-v3")]
pub mod error;
#[cfg(feature = "full")]
pub mod frame;
#[cfg(feature = "runtime-v3")]
pub mod identity;
#[cfg(feature = "runtime-v3")]
pub mod inference;
#[cfg(feature = "full")]
pub mod inference_runtime_v2;
#[cfg(feature = "runtime-v3")]
pub mod inference_runtime_v3;
#[cfg(feature = "full")]
pub mod knowledge;
#[cfg(feature = "full")]
pub mod knowledge_invalidation_feed;
#[cfg(feature = "full")]
pub mod knowledge_lifecycle;
#[cfg(feature = "full")]
pub mod knowledge_prefix;
#[cfg(feature = "full")]
pub mod memory;
#[cfg(feature = "model-admission-v1")]
pub mod model_admission_v1;
#[cfg(feature = "full")]
pub mod models;
#[cfg(feature = "models-v2")]
pub mod models_v2;
#[cfg(feature = "full")]
pub mod negotiation;
#[cfg(feature = "full")]
pub mod negotiation_v2;
#[cfg(feature = "full")]
pub mod nodes;
#[cfg(feature = "full")]
pub mod pairsev1;
#[cfg(feature = "full")]
pub mod qualification;
#[cfg(feature = "full")]
pub mod qualification_v2;
#[cfg(feature = "full")]
pub mod qualification_v2_1;
#[cfg(feature = "full")]
pub mod security;
#[cfg(feature = "full")]
pub mod tasks;
#[cfg(feature = "full")]
pub mod tools;
#[cfg(feature = "full")]
pub mod type_registry;

#[cfg(feature = "runtime-v3")]
pub use error::{ContractError, Result, Validate};

/// The only semantic contract version accepted by these v1 bindings.
pub const CONTRACT_VERSION: &str = "1.0.0";

/// Additive version-negotiation contract release accepted by the negotiation module.
pub const NEGOTIATION_CONTRACT_VERSION: &str = "1.1.0";

/// Identity-bound inference runtime contract release.
pub const INFERENCE_RUNTIME_CONTRACT_VERSION: &str = "2.0.0";

/// Dual-model-identity inference runtime contract release.
pub const INFERENCE_RUNTIME_CONTRACT_VERSION_V3: &str = "3.0.0";

/// The complete first-party binding release represented by this crate.
pub const BINDING_VERSION: &str = env!("CARGO_PKG_VERSION");
