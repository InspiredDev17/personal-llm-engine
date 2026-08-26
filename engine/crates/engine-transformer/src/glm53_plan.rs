//! Experimental, non-executable GLM-5.3-Flash architecture and state planning.
//!
//! The plan encodes reviewed source facts and checked minimum state accounting.
//! It is deliberately separate from the implemented dense/Qwen decoder paths:
//! no package decoder, tokenizer runtime, KDA/DSA/mHC kernel, FP8 decoder,
//! multimodal processor, Metal path, CUDA path, or qualification is implied.

use personalai_engine_types::{
    ArchitectureId, EngineError, MAX_CONTEXT_TOKENS, ResourceBudget, ResourceKind,
};

/// Experimental internal identity. It is not an activated package-format profile.
pub const GLM53_FLASH_ARCHITECTURE_ID: &str = "glm5-next-flash-experimental-v1";
/// Context ceiling declared by the exact reviewed source.
pub const GLM53_DECLARED_CONTEXT_TOKENS: u32 = 1_048_576;
/// Exact text layer count.
pub const GLM53_LAYER_COUNT: u32 = 45;
/// Kimi Delta Attention layer count.
pub const GLM53_KDA_LAYER_COUNT: u32 = 34;
/// DeepSeek Sparse Attention layer count.
pub const GLM53_DSA_LAYER_COUNT: u32 = 11;
/// Exact hidden dimension.
pub const GLM53_HIDDEN_DIMENSION: u32 = 4_096;
/// Exact text vocabulary rows.
pub const GLM53_VOCABULARY_SIZE: u32 = 154_880;
/// Routed experts in each sparse MoE layer.
pub const GLM53_ROUTED_EXPERT_COUNT: u32 = 288;
/// Routed experts selected per token.
pub const GLM53_ACTIVE_ROUTED_EXPERT_COUNT: u32 = 8;
/// Shared experts applied per sparse token.
pub const GLM53_SHARED_EXPERT_COUNT: u32 = 1;
/// Exact dense leading layer count.
pub const GLM53_DENSE_MLP_LAYER_COUNT: u32 = 3;
/// Exact MoE expert intermediate dimension.
pub const GLM53_MOE_INTERMEDIATE_DIMENSION: u32 = 2_048;
/// Exact dense MLP intermediate dimension.
pub const GLM53_DENSE_INTERMEDIATE_DIMENSION: u32 = 12_288;
/// Exact KDA head count.
pub const GLM53_KDA_HEAD_COUNT: u32 = 64;
/// Exact KDA head dimension.
pub const GLM53_KDA_HEAD_DIMENSION: u32 = 128;
/// Exact KDA convolution state width retained by the reviewed reference cache.
pub const GLM53_KDA_CONV_KERNEL: u32 = 4;
/// Exact DSA compressed KV rank.
pub const GLM53_DSA_KV_LORA_RANK: u32 = 512;
/// Exact DSA query rank.
pub const GLM53_DSA_Q_LORA_RANK: u32 = 1_536;
/// Exact DSA index head count.
pub const GLM53_DSA_INDEX_HEAD_COUNT: u32 = 32;
/// Exact DSA index head dimension.
pub const GLM53_DSA_INDEX_HEAD_DIMENSION: u32 = 128;
/// Exact DSA candidate token ceiling.
pub const GLM53_DSA_INDEX_TOPK: u32 = 2_048;
/// Exact DSA index pooling width.
pub const GLM53_DSA_INDEX_KPOOL: u32 = 4;
/// Exact mHC expansion multiplier.
pub const GLM53_MHC_MULTIPLIER: u32 = 4;
/// Exact mHC Sinkhorn iteration count.
pub const GLM53_MHC_SINKHORN_ITERATIONS: u32 = 20;
/// Exact mHC numerical epsilon from the reviewed source.
pub const GLM53_MHC_EPSILON: f32 = 1.0e-6;
/// Exact DSA non-RoPE query/key dimension.
pub const GLM53_DSA_QK_NOPE_HEAD_DIMENSION: u32 = 256;
/// Exact DSA RoPE dimension (the reviewed profile is NoPE).
pub const GLM53_DSA_QK_ROPE_HEAD_DIMENSION: u32 = 0;
/// Exact DSA value-head dimension.
pub const GLM53_DSA_VALUE_HEAD_DIMENSION: u32 = 256;
/// Exact source router scaling after normalized selected sigmoid scores.
pub const GLM53_ROUTED_SCALING_FACTOR: f32 = 2.5;
/// Exact reviewed source weight payload bytes, excluding metadata files.
pub const GLM53_SOURCE_WEIGHT_BYTES: u64 = 328_326_771_576;

const KDA_LAYERS: [u32; 34] = [
    0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 20, 21, 22, 24, 25, 26, 28, 29, 30, 32, 33,
    34, 36, 37, 38, 40, 41, 42, 44,
];
const DSA_LAYERS: [u32; 11] = [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43];

/// Exact attention family assigned to one text layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Glm53AttentionKind {
    /// Kimi Delta Attention with convolution and recurrent state.
    KimiDelta,
    /// DeepSeek Sparse Attention with compressed MLA and index state.
    DeepSeekSparse,
}

/// Exact MLP family assigned to one text layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Glm53MlpKind {
    /// Dense SwiGLU layer.
    Dense,
    /// Sparse routed MoE layer plus one shared expert.
    SparseMoe,
}

/// Closed semantic description of the reviewed sparse-router policy.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Glm53RouterPlan {
    routed_experts: u32,
    selected_experts: u32,
    shared_experts: u32,
    scaling_factor: f32,
}

impl Glm53RouterPlan {
    /// Returns the exact reviewed sigmoid/noaux_tc router.
    pub const fn exact() -> Self {
        Self {
            routed_experts: GLM53_ROUTED_EXPERT_COUNT,
            selected_experts: GLM53_ACTIVE_ROUTED_EXPERT_COUNT,
            shared_experts: GLM53_SHARED_EXPERT_COUNT,
            scaling_factor: GLM53_ROUTED_SCALING_FACTOR,
        }
    }

    /// Returns the available routed-expert count.
    pub const fn routed_experts(self) -> u32 {
        self.routed_experts
    }

    /// Returns the sigmoid-ranked selected routed-expert count.
    pub const fn selected_experts(self) -> u32 {
        self.selected_experts
    }

    /// Returns the always-applied shared-expert count.
    pub const fn shared_experts(self) -> u32 {
        self.shared_experts
    }

    /// Returns the post-normalization selected-score scaling factor.
    pub const fn scaling_factor(self) -> f32 {
        self.scaling_factor
    }

    /// The reviewed source uses sigmoid scoring rather than softmax scoring.
    pub const fn uses_sigmoid_scores(self) -> bool {
        true
    }

    /// The reviewed source uses `noaux_tc` selection.
    pub const fn uses_noaux_tc(self) -> bool {
        true
    }

    /// The reviewed source normalizes selected top-k probabilities.
    pub const fn normalizes_selected_scores(self) -> bool {
        true
    }
}

/// Immutable layer schedule entry derived from the reviewed source profile.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Glm53LayerPlan {
    index: u32,
    attention: Glm53AttentionKind,
    mlp: Glm53MlpKind,
}

impl Glm53LayerPlan {
    /// Returns the zero-based source layer index.
    pub const fn index(self) -> u32 {
        self.index
    }

    /// Returns the exact attention family.
    pub const fn attention(self) -> Glm53AttentionKind {
        self.attention
    }

    /// Returns the exact MLP family.
    pub const fn mlp(self) -> Glm53MlpKind {
        self.mlp
    }
}

/// Exact, source-pinned architecture facts with no executable authority.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Glm53ArchitecturePlan {
    architecture_id: ArchitectureId,
    layers: Vec<Glm53LayerPlan>,
}

impl Glm53ArchitecturePlan {
    /// Constructs the sole reviewed GLM-5.3-Flash schedule.
    pub fn exact_experimental() -> Result<Self, EngineError> {
        let mut layers = Vec::new();
        layers
            .try_reserve_exact(usize::try_from(GLM53_LAYER_COUNT).map_err(|_| {
                EngineError::internal_invariant("GLM layer count is not addressable")
            })?)
            .map_err(|_| EngineError::resource_exhaustion("GLM layer plan allocation failed"))?;
        for index in 0..GLM53_LAYER_COUNT {
            let attention = if DSA_LAYERS.contains(&index) {
                Glm53AttentionKind::DeepSeekSparse
            } else if KDA_LAYERS.contains(&index) {
                Glm53AttentionKind::KimiDelta
            } else {
                return Err(EngineError::internal_invariant("GLM layer schedule is incomplete"));
            };
            let mlp = if index < GLM53_DENSE_MLP_LAYER_COUNT {
                Glm53MlpKind::Dense
            } else {
                Glm53MlpKind::SparseMoe
            };
            layers.push(Glm53LayerPlan { index, attention, mlp });
        }
        if layers.iter().filter(|layer| layer.attention == Glm53AttentionKind::KimiDelta).count()
            != usize::try_from(GLM53_KDA_LAYER_COUNT).unwrap_or(usize::MAX)
            || layers
                .iter()
                .filter(|layer| layer.attention == Glm53AttentionKind::DeepSeekSparse)
                .count()
                != usize::try_from(GLM53_DSA_LAYER_COUNT).unwrap_or(usize::MAX)
        {
            return Err(EngineError::internal_invariant("GLM layer schedule count differs"));
        }
        Ok(Self {
            architecture_id: ArchitectureId::new(GLM53_FLASH_ARCHITECTURE_ID).map_err(|_| {
                EngineError::internal_invariant("GLM architecture identifier is invalid")
            })?,
            layers,
        })
    }

    /// Returns the non-executable internal architecture identity.
    pub const fn architecture_id(&self) -> &ArchitectureId {
        &self.architecture_id
    }

    /// Returns the exact 45-layer KDA/DSA and dense/MoE schedule.
    pub fn layers(&self) -> &[Glm53LayerPlan] {
        &self.layers
    }

    /// Always fails closed until a first-party decoder is implemented and qualified.
    pub fn require_executable(&self) -> Result<(), EngineError> {
        Err(EngineError::unsupported_architecture(self.architecture_id.clone()))
    }
}

/// One bounded, content-free reason execution is unavailable.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Glm53ReadinessBlocker {
    /// No signed package profile/decoder exists for this architecture.
    PackageDecoderUnavailable,
    /// The exact tokenizer runtime has not been implemented and conformance-tested.
    TokenizerRuntimeUnavailable,
    /// The KDA recurrent/conv execution path is absent.
    KdaRuntimeUnavailable,
    /// The DSA/MLA/index execution path is absent.
    DsaRuntimeUnavailable,
    /// mHC execution is absent.
    MhcRuntimeUnavailable,
    /// FP8 E4M3 scale-bound decoding/execution is absent.
    Fp8RuntimeUnavailable,
    /// Image/video processor and vision tower execution are absent.
    MultimodalRuntimeUnavailable,
    /// The declared million-token context exceeds the current runtime ceiling.
    RuntimeContextCeilingTooLow,
    /// No backend/model qualification evidence exists.
    QualificationUnavailable,
}

/// Content-free, deterministic readiness report for the experimental profile.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Glm53ReadinessReport {
    architecture_id: ArchitectureId,
    blockers: Vec<Glm53ReadinessBlocker>,
}

impl Glm53ReadinessReport {
    /// Produces the truthful current non-executable status.
    pub fn current() -> Result<Self, EngineError> {
        let blockers = vec![
            Glm53ReadinessBlocker::PackageDecoderUnavailable,
            Glm53ReadinessBlocker::TokenizerRuntimeUnavailable,
            Glm53ReadinessBlocker::KdaRuntimeUnavailable,
            Glm53ReadinessBlocker::DsaRuntimeUnavailable,
            Glm53ReadinessBlocker::MhcRuntimeUnavailable,
            Glm53ReadinessBlocker::Fp8RuntimeUnavailable,
            Glm53ReadinessBlocker::MultimodalRuntimeUnavailable,
            Glm53ReadinessBlocker::RuntimeContextCeilingTooLow,
            Glm53ReadinessBlocker::QualificationUnavailable,
        ];
        Ok(Self {
            architecture_id: ArchitectureId::new(GLM53_FLASH_ARCHITECTURE_ID).map_err(|_| {
                EngineError::internal_invariant("GLM architecture identifier is invalid")
            })?,
            blockers,
        })
    }

    /// Returns the architecture identity without model/source content.
    pub const fn architecture_id(&self) -> &ArchitectureId {
        &self.architecture_id
    }

    /// Returns the finite, content-free blockers.
    pub fn blockers(&self) -> &[Glm53ReadinessBlocker] {
        &self.blockers
    }

    /// Returns false until every independent runtime and qualification gate is satisfied.
    pub const fn executable(&self) -> bool {
        false
    }
}

/// Checked minimum logical state accounting for one experimental plan.
///
/// The plan is not an allocation recipe. It excludes model weights, backend
/// layout/alignment, graph workspaces, vision state, allocator overhead, and
/// kernel-specific scratch because none of those implementations exist yet.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Glm53ResourcePlan {
    context_tokens: u32,
    concurrent_sequences: u32,
    kda_conv_state_bytes: u64,
    kda_recurrent_state_bytes: u64,
    dsa_mla_state_bytes: u64,
    dsa_index_state_bytes: u64,
    mhc_minimum_working_bytes: u64,
    moe_minimum_working_bytes: u64,
    minimum_persistent_state_bytes: u64,
    minimum_total_state_bytes: u64,
}

impl Glm53ResourcePlan {
    /// Plans checked minimum state for a bounded number of tokens and sequences.
    #[allow(
        clippy::too_many_lines,
        reason = "all exact GLM state components and budget gates remain visible in one audit boundary"
    )]
    pub fn new(
        context_tokens: u32,
        concurrent_sequences: u32,
        budget: &ResourceBudget,
        is_cancelled: &impl Fn() -> bool,
    ) -> Result<Self, EngineError> {
        check_cancelled(is_cancelled)?;
        if context_tokens == 0 || context_tokens > GLM53_DECLARED_CONTEXT_TOKENS {
            return Err(EngineError::resource_limit(
                ResourceKind::ContextTokens,
                u64::from(context_tokens),
                u64::from(GLM53_DECLARED_CONTEXT_TOKENS),
            ));
        }
        if concurrent_sequences == 0 || concurrent_sequences > budget.max_concurrent_sequences() {
            return Err(EngineError::resource_limit(
                ResourceKind::ConcurrentSequences,
                u64::from(concurrent_sequences),
                u64::from(budget.max_concurrent_sequences()),
            ));
        }
        let sequences = u64::from(concurrent_sequences);
        let tokens = u64::from(context_tokens);
        check_cancelled(is_cancelled)?;

        // The reviewed reference cache retains three KDA Q/K/V convolution
        // streams at kernel width four, using the source activation width.
        let kda_conv_state_bytes = checked_product(&[
            sequences,
            u64::from(GLM53_KDA_LAYER_COUNT),
            3,
            u64::from(GLM53_KDA_HEAD_COUNT),
            u64::from(GLM53_KDA_HEAD_DIMENSION),
            u64::from(GLM53_KDA_CONV_KERNEL),
            2,
        ])?;
        // KDA recurrent state is explicitly FP32: [H, K, V].
        let kda_recurrent_state_bytes = checked_product(&[
            sequences,
            u64::from(GLM53_KDA_LAYER_COUNT),
            u64::from(GLM53_KDA_HEAD_COUNT),
            u64::from(GLM53_KDA_HEAD_DIMENSION),
            u64::from(GLM53_KDA_HEAD_DIMENSION),
            4,
        ])?;
        // DSA caches the 512-wide MLA latent per token. RoPE dimension is zero.
        let dsa_mla_state_bytes = checked_product(&[
            sequences,
            tokens,
            u64::from(GLM53_DSA_LAYER_COUNT),
            u64::from(GLM53_DSA_KV_LORA_RANK),
            2,
        ])?;
        // The reference index cache retains key, gate, and validity channels.
        let index_channels = u64::from(GLM53_DSA_INDEX_HEAD_DIMENSION)
            .checked_mul(2)
            .and_then(|value| value.checked_add(1))
            .ok_or_else(resource_overflow)?;
        let dsa_index_state_bytes = checked_product(&[
            sequences,
            tokens,
            u64::from(GLM53_DSA_LAYER_COUNT),
            index_channels,
            2,
        ])?;
        check_cancelled(is_cancelled)?;

        let mhc_minimum_working_bytes = checked_product(&[
            sequences,
            u64::from(GLM53_MHC_MULTIPLIER),
            u64::from(GLM53_HIDDEN_DIMENSION),
            4,
        ])?;
        let moe_minimum_working_bytes = checked_product(&[
            sequences,
            u64::from(GLM53_ACTIVE_ROUTED_EXPERT_COUNT + GLM53_SHARED_EXPERT_COUNT),
            u64::from(GLM53_MOE_INTERMEDIATE_DIMENSION),
            4,
        ])?;
        let minimum_persistent_state_bytes = checked_sum(&[
            kda_conv_state_bytes,
            kda_recurrent_state_bytes,
            dsa_mla_state_bytes,
            dsa_index_state_bytes,
        ])?;
        let minimum_total_state_bytes = checked_sum(&[
            minimum_persistent_state_bytes,
            mhc_minimum_working_bytes,
            moe_minimum_working_bytes,
        ])?;
        if minimum_persistent_state_bytes > budget.max_kv_cache_bytes() {
            return Err(EngineError::resource_limit(
                ResourceKind::MemoryBytes,
                minimum_persistent_state_bytes,
                budget.max_kv_cache_bytes(),
            ));
        }
        if minimum_total_state_bytes > budget.max_memory_bytes() {
            return Err(EngineError::resource_limit(
                ResourceKind::MemoryBytes,
                minimum_total_state_bytes,
                budget.max_memory_bytes(),
            ));
        }
        check_cancelled(is_cancelled)?;
        Ok(Self {
            context_tokens,
            concurrent_sequences,
            kda_conv_state_bytes,
            kda_recurrent_state_bytes,
            dsa_mla_state_bytes,
            dsa_index_state_bytes,
            mhc_minimum_working_bytes,
            moe_minimum_working_bytes,
            minimum_persistent_state_bytes,
            minimum_total_state_bytes,
        })
    }

    /// Returns the planned token count, which may exceed current executable limits.
    pub const fn context_tokens(&self) -> u32 {
        self.context_tokens
    }

    /// Returns the bounded sequence count.
    pub const fn concurrent_sequences(&self) -> u32 {
        self.concurrent_sequences
    }

    /// Returns the checked KDA convolution state bytes.
    pub const fn kda_conv_state_bytes(&self) -> u64 {
        self.kda_conv_state_bytes
    }

    /// Returns the checked KDA FP32 recurrent state bytes.
    pub const fn kda_recurrent_state_bytes(&self) -> u64 {
        self.kda_recurrent_state_bytes
    }

    /// Returns checked DSA compressed MLA state bytes.
    pub const fn dsa_mla_state_bytes(&self) -> u64 {
        self.dsa_mla_state_bytes
    }

    /// Returns checked DSA index state bytes.
    pub const fn dsa_index_state_bytes(&self) -> u64 {
        self.dsa_index_state_bytes
    }

    /// Returns bounded lower-bound mHC working state bytes.
    pub const fn mhc_minimum_working_bytes(&self) -> u64 {
        self.mhc_minimum_working_bytes
    }

    /// Returns bounded lower-bound active/shared expert working bytes.
    pub const fn moe_minimum_working_bytes(&self) -> u64 {
        self.moe_minimum_working_bytes
    }

    /// Returns the minimum persistent recurrent/attention state bytes.
    pub const fn minimum_persistent_state_bytes(&self) -> u64 {
        self.minimum_persistent_state_bytes
    }

    /// Returns minimum persistent plus explicitly modeled working state bytes.
    pub const fn minimum_total_state_bytes(&self) -> u64 {
        self.minimum_total_state_bytes
    }

    /// Reports that the reviewed full context exceeds the current executable hard limit.
    pub const fn exceeds_current_runtime_context_limit(&self) -> bool {
        self.context_tokens > MAX_CONTEXT_TOKENS
    }

    /// Always fails closed: checked planning is not decoder or backend evidence.
    pub fn require_executable(&self) -> Result<(), EngineError> {
        let architecture = ArchitectureId::new(GLM53_FLASH_ARCHITECTURE_ID).map_err(|_| {
            EngineError::internal_invariant("GLM architecture identifier is invalid")
        })?;
        Err(EngineError::unsupported_architecture(architecture))
    }
}

fn checked_product(values: &[u64]) -> Result<u64, EngineError> {
    values
        .iter()
        .try_fold(1_u64, |total, value| total.checked_mul(*value).ok_or_else(resource_overflow))
}

fn checked_sum(values: &[u64]) -> Result<u64, EngineError> {
    values
        .iter()
        .try_fold(0_u64, |total, value| total.checked_add(*value).ok_or_else(resource_overflow))
}

fn resource_overflow() -> EngineError {
    EngineError::resource_exhaustion("GLM experimental state accounting overflows")
}

fn check_cancelled(is_cancelled: &impl Fn() -> bool) -> Result<(), EngineError> {
    if is_cancelled() { Err(EngineError::cancellation(None)) } else { Ok(()) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_layer_schedule_is_complete_and_disjoint() {
        assert_eq!(
            GLM53_FLASH_ARCHITECTURE_ID,
            personalai_engine_model_importer::GLM53_FLASH_ARCHITECTURE_ID
        );
        let plan = Glm53ArchitecturePlan::exact_experimental().unwrap();
        assert_eq!(plan.layers().len(), 45);
        assert_eq!(plan.layers()[3].attention(), Glm53AttentionKind::DeepSeekSparse);
        assert_eq!(plan.layers()[4].attention(), Glm53AttentionKind::KimiDelta);
        assert_eq!(plan.layers()[2].mlp(), Glm53MlpKind::Dense);
        assert_eq!(plan.layers()[3].mlp(), Glm53MlpKind::SparseMoe);
        assert_eq!(KDA_LAYERS.len() + DSA_LAYERS.len(), 45);
        assert!(KDA_LAYERS.iter().all(|layer| !DSA_LAYERS.contains(layer)));
        let router = Glm53RouterPlan::exact();
        assert_eq!(router.routed_experts(), 288);
        assert_eq!(router.selected_experts(), 8);
        assert_eq!(router.shared_experts(), 1);
        assert!(router.uses_sigmoid_scores());
        assert!(router.uses_noaux_tc());
        assert!(router.normalizes_selected_scores());
        assert_eq!(router.scaling_factor().to_bits(), 2.5_f32.to_bits());
        assert!(matches!(plan.require_executable(), Err(EngineError::UnsupportedArchitecture(_))));
    }

    #[test]
    fn tiny_resource_plan_is_deterministic_and_checked() {
        let budget = ResourceBudget::new(1_u64 << 30, 1_u64 << 30, 2).unwrap();
        let first = Glm53ResourcePlan::new(8, 1, &budget, &|| false).unwrap();
        let second = Glm53ResourcePlan::new(8, 1, &budget, &|| false).unwrap();
        assert_eq!(first, second);
        assert_eq!(first.kda_conv_state_bytes(), 6_684_672);
        assert_eq!(first.kda_recurrent_state_bytes(), 142_606_336);
        assert_eq!(first.dsa_mla_state_bytes(), 90_112);
        assert_eq!(first.dsa_index_state_bytes(), 45_232);
        assert!(!first.exceeds_current_runtime_context_limit());
        assert!(matches!(first.require_executable(), Err(EngineError::UnsupportedArchitecture(_))));
    }

    #[test]
    fn context_substitution_budget_denial_and_cancellation_fail_closed() {
        let budget = ResourceBudget::new(1_u64 << 30, 1_u64 << 30, 1).unwrap();
        for tokens in [0, GLM53_DECLARED_CONTEXT_TOKENS + 1] {
            assert!(matches!(
                Glm53ResourcePlan::new(tokens, 1, &budget, &|| false),
                Err(EngineError::ResourceExhaustion(_))
            ));
        }
        assert!(matches!(
            Glm53ResourcePlan::new(8, 2, &budget, &|| false),
            Err(EngineError::ResourceExhaustion(_))
        ));
        assert!(matches!(
            Glm53ResourcePlan::new(8, 1, &budget, &|| true),
            Err(EngineError::Cancellation(_))
        ));
    }

    #[test]
    fn full_context_is_plannable_but_truthfully_not_current_runtime_evidence() {
        let budget = ResourceBudget::new(64_u64 << 30, 64_u64 << 30, 1).unwrap();
        let plan =
            Glm53ResourcePlan::new(GLM53_DECLARED_CONTEXT_TOKENS, 1, &budget, &|| false).unwrap();
        assert!(plan.exceeds_current_runtime_context_limit());
        assert!(plan.minimum_persistent_state_bytes() > 16_u64 << 30);
        let report = Glm53ReadinessReport::current().unwrap();
        assert!(!report.executable());
        assert!(report.blockers().contains(&Glm53ReadinessBlocker::RuntimeContextCeilingTooLow));
    }

    #[test]
    fn overflow_helpers_never_wrap() {
        assert!(matches!(checked_product(&[u64::MAX, 2]), Err(EngineError::ResourceExhaustion(_))));
        assert!(matches!(checked_sum(&[u64::MAX, 1]), Err(EngineError::ResourceExhaustion(_))));
    }
}
