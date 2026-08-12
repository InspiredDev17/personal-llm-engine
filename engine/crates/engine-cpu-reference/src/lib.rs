//! Safe deterministic scalar CPU reference backend.
//!
//! The backend is the engine's correctness oracle. It is deliberately
//! single-threaded, performs no operator fusion, uses no SIMD or platform
//! acceleration, and accepts only first-party allowlisted graph operations.

mod graph;
mod operations;

pub use graph::{
    Backend, ExecutionPlan, GraphNode, GraphNodeId, InferenceGraph, Operation, TensorSpec, ValueId,
};
pub use personalai_engine_tensor::Device;

use personalai_engine_types::{BackendId, EngineError};

/// Identifier advertised for the safe CPU reference backend.
pub const CPU_REFERENCE_BACKEND_ID: &str = "cpu-reference";
/// Hard ceiling on scalar iterations in one reference operation.
pub const MAX_CPU_REFERENCE_OPERATIONS: u64 = 100_000_000;

/// Stateless safe CPU reference backend.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CpuReferenceBackend {
    id: BackendId,
}

impl CpuReferenceBackend {
    /// Constructs the compiled CPU reference backend identity.
    pub fn new() -> Result<Self, EngineError> {
        Ok(Self { id: BackendId::new(CPU_REFERENCE_BACKEND_ID)? })
    }

    /// Returns the backend identifier.
    pub const fn id(&self) -> &BackendId {
        &self.id
    }

    /// Returns the only device supported by this reference backend.
    pub const fn device(&self) -> Device {
        Device::Cpu
    }
}
