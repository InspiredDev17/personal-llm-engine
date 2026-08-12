//! Validated transformer configuration and backend capability boundaries.

#[cfg(feature = "cpu-reference")]
mod cpu_decoder;

#[cfg(feature = "cpu-reference")]
pub use cpu_decoder::{CpuDecoderModel, CpuDecoderSession};

use personalai_engine_kv_cache::{
    KvCachePlan, MAX_HEAD_DIMENSION, MAX_KV_HEAD_COUNT, MAX_LAYER_COUNT,
};
use personalai_engine_model_format::{DirectoryManifest, ModelManifest};
use personalai_engine_types::{
    ArchitectureId, BackendId, ContextLimit, DType, EngineError, MAX_VOCABULARY_SIZE,
    ResourceBudget,
};

#[cfg(feature = "cpu-reference")]
use personalai_engine_cpu_reference::{
    GraphNode, GraphNodeId, InferenceGraph, Operation, TensorSpec, ValueId,
};
#[cfg(feature = "cpu-reference")]
use personalai_engine_types::TensorShape;

/// Maximum model hidden dimension accepted by transformer planning.
pub const MAX_HIDDEN_DIMENSION: u32 = 1_048_576;
/// Maximum feed-forward intermediate dimension.
pub const MAX_INTERMEDIATE_DIMENSION: u32 = 4_194_304;
/// Exact first-party decoder architecture implemented by the CPU reference path.
pub const FIRST_PARTY_DECODER_ARCHITECTURE_ID: &str = "decoder-rms-rope-swiglu-gqa-v1";
/// ARM64 feature/backend identifier.
pub const ARM64_BACKEND_ID: &str = "arm64";
/// Apple Metal feature/backend identifier.
pub const METAL_BACKEND_ID: &str = "metal";
/// NVIDIA CUDA feature/backend identifier.
pub const CUDA_BACKEND_ID: &str = "cuda";

/// Compile-time and implementation status for a backend boundary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BackendStatus {
    backend_id: BackendId,
    compiled: bool,
    model_execution_implemented: bool,
    default_backend: bool,
}

impl BackendStatus {
    /// Returns the backend identifier.
    pub const fn backend_id(&self) -> &BackendId {
        &self.backend_id
    }

    /// Returns whether its feature is compiled into this build.
    pub const fn compiled(&self) -> bool {
        self.compiled
    }

    /// Returns whether this backend can execute a supported model architecture.
    pub const fn model_execution_implemented(&self) -> bool {
        self.model_execution_implemented
    }

    /// Returns whether it is the default feature/backend.
    pub const fn default_backend(&self) -> bool {
        self.default_backend
    }
}

/// Returns truthful status for a known backend identifier.
pub fn backend_status(backend_id: &BackendId) -> Result<BackendStatus, EngineError> {
    let id = backend_id.as_str();
    if id == cpu_reference_backend_id() {
        return Ok(BackendStatus {
            backend_id: backend_id.clone(),
            compiled: cfg!(feature = "cpu-reference"),
            model_execution_implemented: cfg!(feature = "cpu-reference"),
            default_backend: cfg!(feature = "cpu-reference"),
        });
    }
    let compiled = match id {
        ARM64_BACKEND_ID => cfg!(feature = "arm64"),
        METAL_BACKEND_ID => cfg!(feature = "metal"),
        CUDA_BACKEND_ID => cfg!(feature = "cuda"),
        _ => {
            return Err(EngineError::invalid_input("requested backend identifier is unknown"));
        }
    };
    Ok(BackendStatus {
        backend_id: backend_id.clone(),
        compiled,
        model_execution_implemented: false,
        default_backend: false,
    })
}

#[cfg(feature = "cpu-reference")]
fn cpu_reference_backend_id() -> &'static str {
    personalai_engine_cpu_reference::CPU_REFERENCE_BACKEND_ID
}

#[cfg(not(feature = "cpu-reference"))]
const fn cpu_reference_backend_id() -> &'static str {
    "cpu-reference"
}

/// Validated architecture metadata required to plan transformer resources.
#[derive(Debug, Clone, PartialEq)]
pub struct TransformerConfig {
    architecture_id: ArchitectureId,
    layer_count: u32,
    hidden_dimension: u32,
    attention_head_count: u32,
    kv_head_count: u32,
    vocabulary_size: u32,
    context_limit: ContextLimit,
    cache_dtype: DType,
    intermediate_dimension: u32,
    rms_norm_epsilon: f32,
    rope_theta: f32,
}

impl TransformerConfig {
    /// Constructs the exact decoder configuration from a verified directory manifest.
    pub fn from_directory_manifest(manifest: &DirectoryManifest) -> Result<Self, EngineError> {
        if manifest.rope_scaling().to_bits() != 1.0_f32.to_bits() {
            return Err(EngineError::not_implemented(
                "directory manifest requests unsupported RoPE scaling",
            ));
        }
        Self::new_decoder(
            manifest.architecture_id().clone(),
            manifest.layer_count(),
            manifest.hidden_dimension(),
            manifest.attention_head_count(),
            manifest.kv_head_count(),
            manifest.vocabulary_size(),
            manifest.context_limit(),
            DType::F32,
            manifest.feed_forward_dimension(),
            1.0e-5,
            manifest.rope_theta(),
        )
    }

    /// Constructs the exact decoder configuration from a verified manifest.
    pub fn from_manifest(manifest: &ModelManifest) -> Result<Self, EngineError> {
        Self::new_decoder(
            manifest.architecture_id().clone(),
            manifest.layer_count(),
            manifest.hidden_dimension(),
            manifest.attention_head_count(),
            manifest.kv_head_count(),
            manifest.vocabulary_size(),
            manifest.context_limit(),
            DType::F32,
            manifest.intermediate_dimension(),
            manifest.rms_norm_epsilon(),
            manifest.rope_theta(),
        )
    }

    /// Validates bounded transformer dimensions and divisibility invariants.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        architecture_id: ArchitectureId,
        layer_count: u32,
        hidden_dimension: u32,
        attention_head_count: u32,
        kv_head_count: u32,
        vocabulary_size: u32,
        context_limit: ContextLimit,
        cache_dtype: DType,
    ) -> Result<Self, EngineError> {
        let intermediate_dimension = hidden_dimension.checked_mul(4).ok_or_else(|| {
            EngineError::corrupt_model_package("transformer intermediate dimension overflows")
        })?;
        Self::new_decoder(
            architecture_id,
            layer_count,
            hidden_dimension,
            attention_head_count,
            kv_head_count,
            vocabulary_size,
            context_limit,
            cache_dtype,
            intermediate_dimension,
            1.0e-5,
            10_000.0,
        )
    }

    /// Validates all first-party decoder hyperparameters.
    #[allow(clippy::too_many_arguments)]
    pub fn new_decoder(
        architecture_id: ArchitectureId,
        layer_count: u32,
        hidden_dimension: u32,
        attention_head_count: u32,
        kv_head_count: u32,
        vocabulary_size: u32,
        context_limit: ContextLimit,
        cache_dtype: DType,
        intermediate_dimension: u32,
        rms_norm_epsilon: f32,
        rope_theta: f32,
    ) -> Result<Self, EngineError> {
        if layer_count == 0 || layer_count > MAX_LAYER_COUNT {
            return Err(EngineError::corrupt_model_package(
                "transformer layer count is outside hard bounds",
            ));
        }
        if hidden_dimension == 0 || hidden_dimension > MAX_HIDDEN_DIMENSION {
            return Err(EngineError::corrupt_model_package(
                "transformer hidden dimension is outside hard bounds",
            ));
        }
        if attention_head_count == 0 || attention_head_count > MAX_KV_HEAD_COUNT {
            return Err(EngineError::corrupt_model_package(
                "transformer attention head count is outside hard bounds",
            ));
        }
        if kv_head_count == 0
            || kv_head_count > attention_head_count
            || attention_head_count % kv_head_count != 0
        {
            return Err(EngineError::corrupt_model_package(
                "transformer key/value head metadata is inconsistent",
            ));
        }
        if hidden_dimension % attention_head_count != 0 {
            return Err(EngineError::corrupt_model_package(
                "transformer hidden dimension is not divisible by its head count",
            ));
        }
        let head_dimension = hidden_dimension / attention_head_count;
        if head_dimension == 0 || head_dimension > MAX_HEAD_DIMENSION {
            return Err(EngineError::corrupt_model_package(
                "transformer head dimension is outside hard bounds",
            ));
        }
        if vocabulary_size == 0 || vocabulary_size > MAX_VOCABULARY_SIZE {
            return Err(EngineError::corrupt_model_package(
                "transformer vocabulary size is outside hard bounds",
            ));
        }
        if intermediate_dimension == 0 || intermediate_dimension > MAX_INTERMEDIATE_DIMENSION {
            return Err(EngineError::corrupt_model_package(
                "transformer intermediate dimension is outside hard bounds",
            ));
        }
        if !rms_norm_epsilon.is_finite() || !(0.0..=1.0).contains(&rms_norm_epsilon) {
            return Err(EngineError::corrupt_model_package(
                "transformer RMS normalization epsilon is invalid",
            ));
        }
        if rms_norm_epsilon == 0.0 || !rope_theta.is_finite() || rope_theta <= 1.0 {
            return Err(EngineError::corrupt_model_package(
                "transformer numeric hyperparameters are invalid",
            ));
        }
        Ok(Self {
            architecture_id,
            layer_count,
            hidden_dimension,
            attention_head_count,
            kv_head_count,
            vocabulary_size,
            context_limit,
            cache_dtype,
            intermediate_dimension,
            rms_norm_epsilon,
            rope_theta,
        })
    }

    /// Returns the package architecture identifier.
    pub const fn architecture_id(&self) -> &ArchitectureId {
        &self.architecture_id
    }

    /// Returns the transformer layer count.
    pub const fn layer_count(&self) -> u32 {
        self.layer_count
    }

    /// Returns the hidden dimension.
    pub const fn hidden_dimension(&self) -> u32 {
        self.hidden_dimension
    }

    /// Returns the attention head count.
    pub const fn attention_head_count(&self) -> u32 {
        self.attention_head_count
    }

    /// Returns the grouped-query key/value head count.
    pub const fn kv_head_count(&self) -> u32 {
        self.kv_head_count
    }

    /// Returns the vocabulary size.
    pub const fn vocabulary_size(&self) -> u32 {
        self.vocabulary_size
    }

    /// Returns the context window limit.
    pub const fn context_limit(&self) -> ContextLimit {
        self.context_limit
    }

    /// Returns the key/value cache data type.
    pub const fn cache_dtype(&self) -> DType {
        self.cache_dtype
    }

    /// Returns the feed-forward intermediate dimension.
    pub const fn intermediate_dimension(&self) -> u32 {
        self.intermediate_dimension
    }

    /// Returns the RMS normalization epsilon.
    pub const fn rms_norm_epsilon(&self) -> f32 {
        self.rms_norm_epsilon
    }

    /// Returns the rotary-position base theta.
    pub const fn rope_theta(&self) -> f32 {
        self.rope_theta
    }

    /// Returns the per-attention-head dimension.
    pub const fn head_dimension(&self) -> u32 {
        self.hidden_dimension / self.attention_head_count
    }

    /// Constructs one immutable, first-party scalar decoder-block graph.
    ///
    /// Runtime values 1 through 10 are respectively the hidden state,
    /// attention-normalization weight, query projection, immutable key cache,
    /// immutable value cache, attention output projection, feed-forward
    /// normalization weight, gate projection, up projection, and down
    /// projection. The sole output is value 33. Package data cannot add or
    /// redefine operations.
    #[cfg(feature = "cpu-reference")]
    pub fn reference_block_graph(&self, position: u32) -> Result<InferenceGraph, EngineError> {
        if position >= self.context_limit.get() {
            return Err(EngineError::invalid_input(
                "reference graph position exceeds context limit",
            ));
        }
        let hidden = u64::from(self.hidden_dimension);
        let kv_heads = u64::from(self.kv_head_count);
        let head_dimension = u64::from(self.head_dimension());
        let intermediate = u64::from(self.intermediate_dimension);
        let sequence = u64::from(position).checked_add(1).ok_or_else(|| {
            EngineError::resource_exhaustion("reference graph position overflows")
        })?;
        let f32_spec = |dimensions| TensorSpec::new(DType::F32, TensorShape::new(dimensions)?);
        let value = |identifier| ValueId::new(identifier);
        let node = |identifier| GraphNodeId::new(identifier);
        let inputs = vec![
            (value(1)?, f32_spec(vec![hidden])?),
            (value(2)?, f32_spec(vec![hidden])?),
            (value(3)?, f32_spec(vec![hidden, hidden])?),
            (value(4)?, f32_spec(vec![sequence, kv_heads, head_dimension])?),
            (value(5)?, f32_spec(vec![sequence, kv_heads, head_dimension])?),
            (value(6)?, f32_spec(vec![hidden, hidden])?),
            (value(7)?, f32_spec(vec![hidden])?),
            (value(8)?, f32_spec(vec![intermediate, hidden])?),
            (value(9)?, f32_spec(vec![intermediate, hidden])?),
            (value(10)?, f32_spec(vec![hidden, intermediate])?),
        ];
        let nodes = vec![
            GraphNode::new(
                node(1)?,
                Operation::RmsNorm { epsilon: self.rms_norm_epsilon },
                [value(1)?, value(2)?],
                value(20)?,
            )?,
            GraphNode::new(node(2)?, Operation::MatrixVector, [value(3)?, value(20)?], value(21)?)?,
            GraphNode::new(
                node(3)?,
                Operation::HeadReshape { head_count: self.attention_head_count },
                [value(21)?],
                value(22)?,
            )?,
            GraphNode::new(
                node(4)?,
                Operation::Rope { position, theta: self.rope_theta },
                [value(22)?],
                value(23)?,
            )?,
            GraphNode::new(
                node(5)?,
                Operation::GroupedQueryAttention { causal_position: position },
                [value(23)?, value(4)?, value(5)?],
                value(24)?,
            )?,
            GraphNode::new(
                node(6)?,
                Operation::HeadReshape { head_count: self.attention_head_count },
                [value(24)?],
                value(25)?,
            )?,
            GraphNode::new(node(7)?, Operation::MatrixVector, [value(6)?, value(25)?], value(26)?)?,
            GraphNode::new(node(8)?, Operation::ResidualAdd, [value(1)?, value(26)?], value(27)?)?,
            GraphNode::new(
                node(9)?,
                Operation::RmsNorm { epsilon: self.rms_norm_epsilon },
                [value(27)?, value(7)?],
                value(28)?,
            )?,
            GraphNode::new(
                node(10)?,
                Operation::MatrixVector,
                [value(8)?, value(28)?],
                value(29)?,
            )?,
            GraphNode::new(
                node(11)?,
                Operation::MatrixVector,
                [value(9)?, value(28)?],
                value(30)?,
            )?,
            GraphNode::new(node(12)?, Operation::SwiGlu, [value(29)?, value(30)?], value(31)?)?,
            GraphNode::new(
                node(13)?,
                Operation::MatrixVector,
                [value(10)?, value(31)?],
                value(32)?,
            )?,
            GraphNode::new(
                node(14)?,
                Operation::ResidualAdd,
                [value(27)?, value(32)?],
                value(33)?,
            )?,
        ];
        InferenceGraph::new(inputs, nodes, vec![value(33)?])
    }
}

/// Resource plan for an otherwise unsupported transformer architecture.
#[derive(Debug, Clone, PartialEq)]
pub struct TransformerPlan {
    config: TransformerConfig,
    kv_cache: KvCachePlan,
}

impl TransformerPlan {
    /// Produces a checked single-sequence cache plan.
    pub fn new(config: TransformerConfig, budget: &ResourceBudget) -> Result<Self, EngineError> {
        let kv_cache = KvCachePlan::new(
            config.layer_count(),
            config.kv_head_count(),
            config.head_dimension(),
            config.context_limit().get(),
            config.cache_dtype(),
            budget,
        )?;
        Ok(Self { config, kv_cache })
    }

    /// Returns validated architecture metadata.
    pub const fn config(&self) -> &TransformerConfig {
        &self.config
    }

    /// Returns the checked cache plan.
    pub const fn kv_cache(&self) -> &KvCachePlan {
        &self.kv_cache
    }

    /// Confirms that this exact architecture has a compiled implementation.
    pub fn require_executable_architecture(&self) -> Result<(), EngineError> {
        if cfg!(feature = "cpu-reference")
            && self.config.architecture_id().as_str() == FIRST_PARTY_DECODER_ARCHITECTURE_ID
        {
            Ok(())
        } else {
            Err(EngineError::unsupported_architecture(self.config.architecture_id().clone()))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "cpu-reference")]
    use personalai_engine_cpu_reference::CpuReferenceBackend;
    #[cfg(feature = "cpu-reference")]
    use personalai_engine_memory::MemoryPool;
    #[cfg(feature = "cpu-reference")]
    use personalai_engine_tensor::MutableTensor;
    #[cfg(feature = "cpu-reference")]
    use std::collections::BTreeMap;

    #[test]
    fn validates_transformer_and_cache_dimensions() {
        let config = config();
        let budget =
            ResourceBudget::new(1 << 30, 1 << 29, 1).expect("resource budget must validate");
        let plan = TransformerPlan::new(config, &budget).expect("small plan must validate");
        assert_eq!(plan.kv_cache().head_dimension(), 64);
        assert!(matches!(
            plan.require_executable_architecture(),
            Err(EngineError::UnsupportedArchitecture(_))
        ));
    }

    #[test]
    fn rejects_invalid_head_divisibility() {
        let result = TransformerConfig::new(
            ArchitectureId::new("first-party.test").expect("id must validate"),
            2,
            127,
            4,
            2,
            1_024,
            ContextLimit::new(128).expect("context must validate"),
            DType::F32,
        );
        assert!(result.is_err());
    }

    #[test]
    fn accelerated_features_never_claim_implementation() {
        for id in [ARM64_BACKEND_ID, METAL_BACKEND_ID, CUDA_BACKEND_ID] {
            let backend_id = BackendId::new(id).expect("backend id must validate");
            let status = backend_status(&backend_id).expect("known backend must resolve");
            assert!(!status.model_execution_implemented());
            assert!(!status.default_backend());
        }
    }

    #[test]
    fn cpu_feature_is_default_and_implements_the_first_party_decoder() {
        let backend_id =
            BackendId::new(cpu_reference_backend_id()).expect("backend id must validate");
        let status = backend_status(&backend_id).expect("CPU backend must resolve");
        assert_eq!(status.compiled(), cfg!(feature = "cpu-reference"));
        assert_eq!(status.default_backend(), cfg!(feature = "cpu-reference"));
        assert_eq!(status.model_execution_implemented(), cfg!(feature = "cpu-reference"));
    }

    #[cfg(feature = "cpu-reference")]
    #[test]
    fn first_party_architecture_constructs_and_executes_a_tiny_immutable_block() {
        let config = TransformerConfig::new_decoder(
            ArchitectureId::new(FIRST_PARTY_DECODER_ARCHITECTURE_ID).unwrap(),
            1,
            4,
            2,
            1,
            8,
            ContextLimit::new(4).unwrap(),
            DType::F32,
            8,
            1.0e-5,
            10_000.0,
        )
        .unwrap();
        let graph = config.reference_block_graph(0).unwrap();
        assert_eq!(graph.nodes().len(), 14);
        assert_eq!(graph.outputs(), [ValueId::new(33).unwrap()]);
        let pool = MemoryPool::new(1 << 20).unwrap();
        let mut inputs = BTreeMap::new();
        for (identifier, specification) in graph.inputs() {
            let count = usize::try_from(specification.shape().element_count()).unwrap();
            let values = vec![0.125_f32; count];
            inputs.insert(
                *identifier,
                MutableTensor::from_f32_values(specification.shape().clone(), &values, &pool)
                    .unwrap(),
            );
        }
        let result = graph.execute(&CpuReferenceBackend::new().unwrap(), inputs, &pool).unwrap();
        let output = &result[&ValueId::new(33).unwrap()];
        assert_eq!(output.shape().dimensions(), [4]);
        for index in 0..4 {
            assert!(output.f32_value(index).unwrap().is_finite());
        }
        assert!(config.reference_block_graph(4).is_err());
    }

    fn config() -> TransformerConfig {
        TransformerConfig::new(
            ArchitectureId::new("first-party.test").expect("id must validate"),
            2,
            256,
            4,
            2,
            1_024,
            ContextLimit::new(128).expect("context must validate"),
            DType::F32,
        )
        .expect("config must validate")
    }
}
