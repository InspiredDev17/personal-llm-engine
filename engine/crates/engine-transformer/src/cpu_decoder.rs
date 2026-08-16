//! Safe scalar implementation of the exact first-party decoder architecture.

use crate::{FIRST_PARTY_DECODER_ARCHITECTURE_ID, TransformerConfig, TransformerPlan};
#[cfg(feature = "arm64")]
use personalai_engine_arm64::Arm64Backend;
use personalai_engine_cpu_reference::MAX_CPU_REFERENCE_OPERATIONS;
#[cfg(feature = "arm64")]
use personalai_engine_cpu_reference::{Backend, Operation};
use personalai_engine_kv_cache::{
    ContiguousKvCache, KvCacheMetrics, KvTokenWriter, SequenceHandle,
};
use personalai_engine_loader::{DirectorySecurityEventSink, VerifiedDirectoryPackage};
use personalai_engine_memory::{MemoryOwner, MemoryPool};
use personalai_engine_placement::{
    ExecutionPlacementPlan, PlacementCancellation, PlacementFaultInjector, PlacementStage,
    PlacementStageKind, PlacementTensorRange, WeightResidencyMode, run_layer_streamed,
};
use personalai_engine_tensor::Tensor;
use personalai_engine_types::{
    DType, EngineError, MAX_CONCURRENT_SEQUENCES, ModelId, RequestId, ResourceBudget, SequenceId,
    TensorShape,
};
use std::collections::BTreeMap;
use std::sync::Arc;

const WEIGHTS_PER_LAYER: usize = 9;
const GLOBAL_WEIGHT_COUNT: usize = 3;

/// Fully validated, aggregate-accounted CPU decoder model.
///
/// Clones share immutable weights, configuration, memory accounting, and the
/// synchronized sequence cache. A session owns one clone so model-registry map
/// mutation cannot invalidate active inference state.
#[derive(Clone, Debug)]
pub struct CpuDecoderModel {
    inner: Arc<CpuDecoderModelInner>,
}

#[derive(Debug)]
struct CpuDecoderModelInner {
    config: TransformerConfig,
    weights: DecoderWeights,
    pool: MemoryPool,
    cache: ContiguousKvCache,
    backend: DecoderCpuBackend,
}

#[derive(Debug)]
enum DecoderWeights {
    Resident(BTreeMap<String, Tensor>),
    Streamed { package: Arc<VerifiedDirectoryPackage>, plan: Arc<ExecutionPlacementPlan> },
}

#[derive(Clone, Debug)]
enum DecoderCpuBackend {
    Scalar,
    #[cfg(feature = "arm64")]
    Arm64(Arm64Backend),
}

impl CpuDecoderModel {
    /// Validates the exact architecture and every required tensor shape.
    pub fn new(
        config: TransformerConfig,
        weights: BTreeMap<String, Tensor>,
        pool: &MemoryPool,
    ) -> Result<Self, EngineError> {
        Self::new_registered(ModelId::new("unregistered-test-model")?, config, weights, pool)
    }

    /// Validates a production model under its configuration-owned identity.
    pub fn new_registered(
        model_id: ModelId,
        config: TransformerConfig,
        weights: BTreeMap<String, Tensor>,
        pool: &MemoryPool,
    ) -> Result<Self, EngineError> {
        Self::new_registered_with_backend(
            model_id,
            config,
            weights,
            pool,
            DecoderCpuBackend::Scalar,
        )
    }

    /// Validates a production model for owner-selected ARM64 execution.
    #[cfg(feature = "arm64")]
    pub fn new_registered_arm64(
        model_id: ModelId,
        config: TransformerConfig,
        weights: BTreeMap<String, Tensor>,
        pool: &MemoryPool,
        backend: Arm64Backend,
    ) -> Result<Self, EngineError> {
        Self::new_registered_with_backend(
            model_id,
            config,
            weights,
            pool,
            DecoderCpuBackend::Arm64(backend),
        )
    }

    fn new_registered_with_backend(
        model_id: ModelId,
        config: TransformerConfig,
        weights: BTreeMap<String, Tensor>,
        pool: &MemoryPool,
        backend: DecoderCpuBackend,
    ) -> Result<Self, EngineError> {
        validate_model_configuration(&config)?;
        let layer_count = usize::try_from(config.layer_count()).map_err(|_| {
            EngineError::corrupt_model_package("decoder layer count is not addressable")
        })?;
        let expected_count = layer_count
            .checked_mul(WEIGHTS_PER_LAYER)
            .and_then(|count| count.checked_add(GLOBAL_WEIGHT_COUNT))
            .ok_or_else(|| EngineError::corrupt_model_package("decoder weight count overflows"))?;
        if weights.len() != expected_count {
            return Err(EngineError::corrupt_model_package(
                "decoder tensor count does not match the exact architecture",
            ));
        }
        validate_weights(&config, &weights, pool)?;
        Self::finish_new(model_id, config, DecoderWeights::Resident(weights), pool, backend)
    }

    /// Constructs a scalar decoder whose weights are materialized one trusted
    /// architecture stage at a time from one exact verified package.
    pub fn new_registered_streamed(
        model_id: ModelId,
        config: TransformerConfig,
        package: Arc<VerifiedDirectoryPackage>,
        plan: Arc<ExecutionPlacementPlan>,
        pool: &MemoryPool,
    ) -> Result<Self, EngineError> {
        validate_model_configuration(&config)?;
        plan.verify_package_binding(&package)
            .map_err(personalai_engine_placement::PlacementError::into_engine_error)?;
        if plan.mode() != WeightResidencyMode::LayerStreamed
            || plan.model_id() != &model_id
            || plan.architecture_id() != config.architecture_id()
            || plan.backend_id().as_str() != "cpu-reference"
        {
            return Err(EngineError::invalid_input(
                "streamed decoder plan does not match the exact model and scalar backend",
            ));
        }
        if plan.required_envelope_bytes() > pool.limit_bytes() {
            return Err(EngineError::resource_exhaustion(
                "streamed decoder pool is below the guaranteed placement envelope",
            ));
        }
        let package_config = TransformerConfig::from_directory_manifest(package.manifest())?;
        if package_config != config {
            return Err(EngineError::corrupt_model_package(
                "streamed decoder configuration differs from the verified package",
            ));
        }
        Self::finish_new(
            model_id,
            config,
            DecoderWeights::Streamed { package, plan },
            pool,
            DecoderCpuBackend::Scalar,
        )
    }

    fn finish_new(
        model_id: ModelId,
        config: TransformerConfig,
        weights: DecoderWeights,
        pool: &MemoryPool,
        backend: DecoderCpuBackend,
    ) -> Result<Self, EngineError> {
        let cache_budget =
            ResourceBudget::new(pool.limit_bytes(), pool.limit_bytes(), MAX_CONCURRENT_SEQUENCES)?;
        let cache_plan = personalai_engine_kv_cache::KvCachePlan::new(
            config.layer_count(),
            config.kv_head_count(),
            config.head_dimension(),
            config.context_limit().get(),
            config.cache_dtype(),
            &cache_budget,
        )?;
        let cache =
            ContiguousKvCache::new(model_id, cache_plan, MAX_CONCURRENT_SEQUENCES, pool.clone())?;
        Ok(Self {
            inner: Arc::new(CpuDecoderModelInner {
                config,
                weights,
                pool: pool.clone(),
                cache,
                backend,
            }),
        })
    }

    fn resident_weights(&self) -> Result<&BTreeMap<String, Tensor>, EngineError> {
        match &self.inner.weights {
            DecoderWeights::Resident(weights) => Ok(weights),
            DecoderWeights::Streamed { .. } => Err(EngineError::not_implemented(
                "streamed decoder requires the cancellation-aware staged execution API",
            )),
        }
    }

    fn streamed_weights(
        &self,
    ) -> Result<(Arc<VerifiedDirectoryPackage>, Arc<ExecutionPlacementPlan>), EngineError> {
        match &self.inner.weights {
            DecoderWeights::Streamed { package, plan } => {
                Ok((Arc::clone(package), Arc::clone(plan)))
            }
            DecoderWeights::Resident(_) => Err(EngineError::invalid_input(
                "resident decoder cannot use the streamed execution API",
            )),
        }
    }

    /// Returns the exact trusted execution backend selected for this model.
    pub fn backend_id(&self) -> Result<personalai_engine_types::BackendId, EngineError> {
        match &self.inner.backend {
            DecoderCpuBackend::Scalar => personalai_engine_types::BackendId::new("cpu-reference"),
            #[cfg(feature = "arm64")]
            DecoderCpuBackend::Arm64(backend) => Ok(backend.id().clone()),
        }
    }

    /// Returns the validated architecture configuration.
    pub fn config(&self) -> &TransformerConfig {
        &self.inner.config
    }

    /// Returns the aggregate pool used for request-scoped runtime accounting.
    pub fn memory_pool(&self) -> &MemoryPool {
        &self.inner.pool
    }

    /// Returns exact live and peak contiguous KV-cache accounting.
    pub fn kv_cache_metrics(&self) -> Result<KvCacheMetrics, EngineError> {
        self.inner.cache.metrics()
    }

    /// Allocates one bounded sequence cache under the model's aggregate pool.
    pub fn start_session(
        &self,
        budget: &ResourceBudget,
        request_id: RequestId,
        sequence_id: SequenceId,
    ) -> Result<CpuDecoderSession, EngineError> {
        let plan = TransformerPlan::new(self.inner.config.clone(), budget)?;
        plan.require_executable_architecture()?;
        if plan.kv_cache() != self.inner.cache.plan() {
            return Err(EngineError::internal_invariant(
                "request and model KV-cache layouts differ",
            ));
        }
        if self.inner.cache.metrics()?.active_sequence_count >= budget.max_concurrent_sequences() {
            return Err(EngineError::resource_exhaustion(
                "request sequence concurrency budget is exhausted",
            ));
        }
        let handle = self.inner.cache.create_sequence(request_id, sequence_id)?;
        Ok(CpuDecoderSession {
            model: self.clone(),
            handle,
            scratch_pool: self.inner.pool.scoped(MemoryOwner::RequestScratch(request_id)),
            output_pool: self.inner.pool.scoped(MemoryOwner::OutputLogits(request_id)),
            token_count: 0,
        })
    }
}

/// Mutable per-sequence CPU decoder state and materialized KV cache.
///
/// The session owns a shared model handle and destroys only its own sequence
/// slot on drop.
#[derive(Debug)]
pub struct CpuDecoderSession {
    model: CpuDecoderModel,
    handle: SequenceHandle,
    scratch_pool: MemoryPool,
    output_pool: MemoryPool,
    token_count: u32,
}

impl CpuDecoderSession {
    /// Returns the number of tokens already committed to this sequence.
    pub const fn token_count(&self) -> u32 {
        self.token_count
    }

    /// Processes a non-empty prompt and returns logits for its final position.
    pub fn prefill(&mut self, tokens: &[u32]) -> Result<Tensor, EngineError> {
        if tokens.is_empty() {
            return Err(EngineError::invalid_input("decoder prefill requires at least one token"));
        }
        let additional = u32::try_from(tokens.len())
            .map_err(|_| EngineError::resource_exhaustion("prefill token count exceeds u32"))?;
        if self
            .token_count
            .checked_add(additional)
            .is_none_or(|total| total > self.model.inner.config.context_limit().get())
        {
            return Err(EngineError::resource_exhaustion("decoder prefill exceeds context"));
        }
        let mut final_logits = None;
        for token in tokens {
            final_logits = Some(self.decode(*token)?);
        }
        final_logits.ok_or_else(|| EngineError::internal_invariant("prefill produced no logits"))
    }

    /// Processes exactly one token at the next sequence position.
    pub fn decode(&mut self, token: u32) -> Result<Tensor, EngineError> {
        self.forward_token(token)
    }

    /// Executes one token and returns finite next-token logits.
    pub fn forward_token(&mut self, token: u32) -> Result<Tensor, EngineError> {
        let weights = self.model.resident_weights()?;
        if token >= self.model.inner.config.vocabulary_size() {
            return Err(EngineError::invalid_input("input token is outside the model vocabulary"));
        }
        if self.token_count >= self.model.inner.config.context_limit().get() {
            return Err(EngineError::resource_exhaustion("decoder context capacity is exhausted"));
        }
        check_operation_count(&self.model.inner.config, self.token_count)?;
        let workspace_bytes = workspace_bytes(&self.model.inner.config, self.token_count)?;
        let _workspace = self.scratch_pool.reserve(workspace_bytes)?;
        let hidden = usize_from_u32(self.model.inner.config.hidden_dimension())?;
        let mut state = embedding_row(
            required_weight(weights, "token_embedding.weight")?,
            usize::try_from(token)
                .map_err(|_| EngineError::invalid_input("token identifier is not addressable"))?,
            hidden,
        )?;

        let mut writer = self.model.inner.cache.begin_token(self.handle)?;
        for layer in 0..self.model.inner.config.layer_count() {
            self.forward_layer(layer, &mut writer, &mut state, weights)?;
        }
        let normalized = self.rms_norm(
            &state,
            required_weight(weights, "final_norm.weight")?,
            self.model.inner.config.rms_norm_epsilon(),
        )?;
        let logits = self.linear(
            required_weight(weights, "output.weight")?,
            &normalized,
            usize_from_u32(self.model.inner.config.vocabulary_size())?,
            hidden,
        )?;
        let output = Tensor::from_f32_values(
            TensorShape::new([u64::from(self.model.inner.config.vocabulary_size())])?,
            &logits,
            &self.output_pool,
        )?;
        writer.commit()?;
        self.token_count = self
            .token_count
            .checked_add(1)
            .ok_or_else(|| EngineError::internal_invariant("decoder token count overflows"))?;
        Ok(output)
    }

    /// Processes a prompt through the cancellation-aware streamed-weight path.
    pub fn prefill_streamed(
        &mut self,
        tokens: &[u32],
        cancellation: &PlacementCancellation,
        faults: &dyn PlacementFaultInjector,
        events: &mut impl DirectorySecurityEventSink,
    ) -> Result<Tensor, EngineError> {
        if tokens.is_empty() {
            return Err(EngineError::invalid_input(
                "streamed decoder prefill requires at least one token",
            ));
        }
        let additional = u32::try_from(tokens.len())
            .map_err(|_| EngineError::resource_exhaustion("prefill token count exceeds u32"))?;
        if self
            .token_count
            .checked_add(additional)
            .is_none_or(|total| total > self.model.inner.config.context_limit().get())
        {
            return Err(EngineError::resource_exhaustion("decoder prefill exceeds context"));
        }
        let mut final_logits = None;
        for token in tokens {
            final_logits =
                Some(self.forward_token_streamed(*token, cancellation, faults, events)?);
        }
        final_logits.ok_or_else(|| EngineError::internal_invariant("prefill produced no logits"))
    }

    /// Executes one token while materializing only the current and optionally
    /// next trusted architecture stages.
    pub fn forward_token_streamed(
        &mut self,
        token: u32,
        cancellation: &PlacementCancellation,
        faults: &dyn PlacementFaultInjector,
        events: &mut impl DirectorySecurityEventSink,
    ) -> Result<Tensor, EngineError> {
        if cancellation.is_cancelled() {
            return Err(EngineError::cancellation(None));
        }
        if token >= self.model.inner.config.vocabulary_size() {
            return Err(EngineError::invalid_input("input token is outside the model vocabulary"));
        }
        if self.token_count >= self.model.inner.config.context_limit().get() {
            return Err(EngineError::resource_exhaustion("decoder context capacity is exhausted"));
        }
        let (package, plan) = self.model.streamed_weights()?;
        check_operation_count(&self.model.inner.config, self.token_count)?;
        let workspace_bytes = workspace_bytes(&self.model.inner.config, self.token_count)?;
        let _workspace = self.scratch_pool.reserve(workspace_bytes)?;
        let hidden = usize_from_u32(self.model.inner.config.hidden_dimension())?;
        let token_index = usize::try_from(token)
            .map_err(|_| EngineError::invalid_input("token identifier is not addressable"))?;
        let mut writer = self.model.inner.cache.begin_token(self.handle)?;
        let mut state = None;
        let mut logits = None;
        run_layer_streamed(
            &plan,
            &package,
            &self.model.inner.pool,
            cancellation,
            faults,
            events,
            |stage, loaded| match stage.kind() {
                PlacementStageKind::Embedding => {
                    if state.is_some() {
                        return Err(EngineError::internal_invariant(
                            "streamed embedding stage executed more than once",
                        ));
                    }
                    state = Some(embedding_row(
                        stage_weight(stage, loaded.tensors(), "token_embedding.weight")?,
                        token_index,
                        hidden,
                    )?);
                    Ok(())
                }
                PlacementStageKind::DecoderLayer(layer) => {
                    let current = state.as_mut().ok_or_else(|| {
                        EngineError::internal_invariant(
                            "streamed decoder layer preceded the embedding stage",
                        )
                    })?;
                    self.forward_layer(layer, &mut writer, current, loaded.tensors())
                }
                PlacementStageKind::Output => {
                    let current = state.as_ref().ok_or_else(|| {
                        EngineError::internal_invariant(
                            "streamed output stage preceded decoder state",
                        )
                    })?;
                    let normalized = self.rms_norm(
                        current,
                        stage_weight(stage, loaded.tensors(), "final_norm.weight")?,
                        self.model.inner.config.rms_norm_epsilon(),
                    )?;
                    logits = Some(self.linear(
                        stage_weight(stage, loaded.tensors(), "output.weight")?,
                        &normalized,
                        usize_from_u32(self.model.inner.config.vocabulary_size())?,
                        hidden,
                    )?);
                    Ok(())
                }
            },
        )
        .map_err(personalai_engine_placement::PlacementError::into_engine_error)?;
        let logits = logits.ok_or_else(|| {
            EngineError::internal_invariant("streamed decoder produced no output stage")
        })?;
        let output = Tensor::from_f32_values(
            TensorShape::new([u64::from(self.model.inner.config.vocabulary_size())])?,
            &logits,
            &self.output_pool,
        )?;
        writer.commit()?;
        self.token_count = self
            .token_count
            .checked_add(1)
            .ok_or_else(|| EngineError::internal_invariant("decoder token count overflows"))?;
        Ok(output)
    }

    fn forward_layer(
        &self,
        layer: u32,
        writer: &mut KvTokenWriter<'_>,
        state: &mut [f32],
        weights: &BTreeMap<String, Tensor>,
    ) -> Result<(), EngineError> {
        let hidden = usize_from_u32(self.model.inner.config.hidden_dimension())?;
        let kv_dimension = usize_from_u32(
            self.model
                .inner
                .config
                .kv_head_count()
                .checked_mul(self.model.inner.config.head_dimension())
                .ok_or_else(|| EngineError::internal_invariant("KV dimension overflows"))?,
        )?;
        let prefix = format!("layers.{layer:03}");
        let attention_input = self.rms_norm(
            state,
            required_weight(weights, &format!("{prefix}.attn_norm.weight"))?,
            self.model.inner.config.rms_norm_epsilon(),
        )?;
        let mut query = self.linear(
            required_weight(weights, &format!("{prefix}.q_proj.weight"))?,
            &attention_input,
            hidden,
            hidden,
        )?;
        let mut key = self.linear(
            required_weight(weights, &format!("{prefix}.k_proj.weight"))?,
            &attention_input,
            kv_dimension,
            hidden,
        )?;
        let value = self.linear(
            required_weight(weights, &format!("{prefix}.v_proj.weight"))?,
            &attention_input,
            kv_dimension,
            hidden,
        )?;
        self.apply_rope(
            &mut query,
            self.model.inner.config.attention_head_count(),
            self.model.inner.config.head_dimension(),
            self.token_count,
            self.model.inner.config.rope_theta(),
        )?;
        self.apply_rope(
            &mut key,
            self.model.inner.config.kv_head_count(),
            self.model.inner.config.head_dimension(),
            self.token_count,
            self.model.inner.config.rope_theta(),
        )?;
        writer.write_layer(layer, &key, &value)?;
        let attention = self.attention(layer, writer, &query)?;
        let projected = self.linear(
            required_weight(weights, &format!("{prefix}.o_proj.weight"))?,
            &attention,
            hidden,
            hidden,
        )?;
        self.add_in_place(state, &projected)?;

        let ffn_input = self.rms_norm(
            state,
            required_weight(weights, &format!("{prefix}.ffn_norm.weight"))?,
            self.model.inner.config.rms_norm_epsilon(),
        )?;
        let intermediate = usize_from_u32(self.model.inner.config.intermediate_dimension())?;
        let gate = self.linear(
            required_weight(weights, &format!("{prefix}.gate_proj.weight"))?,
            &ffn_input,
            intermediate,
            hidden,
        )?;
        let up = self.linear(
            required_weight(weights, &format!("{prefix}.up_proj.weight"))?,
            &ffn_input,
            intermediate,
            hidden,
        )?;
        let activated = self.swiglu(&gate, &up)?;
        let down = self.linear(
            required_weight(weights, &format!("{prefix}.down_proj.weight"))?,
            &activated,
            hidden,
            intermediate,
        )?;
        self.add_in_place(state, &down)
    }

    fn attention(
        &self,
        layer: u32,
        writer: &KvTokenWriter<'_>,
        query: &[f32],
    ) -> Result<Vec<f32>, EngineError> {
        #[cfg(feature = "arm64")]
        if let DecoderCpuBackend::Arm64(backend) = &self.model.inner.backend {
            return self.arm64_attention(layer, writer, query, backend);
        }
        let attention_heads = usize_from_u32(self.model.inner.config.attention_head_count())?;
        let kv_heads = usize_from_u32(self.model.inner.config.kv_head_count())?;
        let head_dimension = usize_from_u32(self.model.inner.config.head_dimension())?;
        let sequence_length = usize_from_u32(
            self.token_count
                .checked_add(1)
                .ok_or_else(|| EngineError::internal_invariant("sequence length overflows"))?,
        )?;
        let group_size = attention_heads
            .checked_div(kv_heads)
            .ok_or_else(|| EngineError::internal_invariant("attention group size is invalid"))?;
        let mut output =
            zero_f32(attention_heads.checked_mul(head_dimension).ok_or_else(|| {
                EngineError::resource_exhaustion("attention output length overflows")
            })?)?;
        let scale = f64::from(self.model.inner.config.head_dimension()).sqrt();
        for query_head in 0..attention_heads {
            let kv_head = query_head / group_size;
            let mut scores = zero_f32(sequence_length)?;
            for (position, score) in scores.iter_mut().enumerate() {
                let mut dot = 0.0_f64;
                for dimension in 0..head_dimension {
                    let query_index = query_head
                        .checked_mul(head_dimension)
                        .and_then(|offset| offset.checked_add(dimension))
                        .ok_or_else(|| EngineError::internal_invariant("query index overflows"))?;
                    let cached = writer.read_key_value(
                        layer,
                        u32_from_usize(position)?,
                        u32_from_usize(kv_head)?,
                        u32_from_usize(dimension)?,
                    )?;
                    dot += f64::from(finite(query[query_index])?) * f64::from(finite(cached)?);
                }
                *score = f32_from_f64(dot / scale)?;
            }
            softmax(&mut scores)?;
            for dimension in 0..head_dimension {
                let mut value = 0.0_f64;
                for (position, probability) in scores.iter().copied().enumerate() {
                    let cached = writer.read_cached_value(
                        layer,
                        u32_from_usize(position)?,
                        u32_from_usize(kv_head)?,
                        u32_from_usize(dimension)?,
                    )?;
                    value += f64::from(finite(probability)?) * f64::from(finite(cached)?);
                }
                let output_index = query_head
                    .checked_mul(head_dimension)
                    .and_then(|offset| offset.checked_add(dimension))
                    .ok_or_else(|| EngineError::internal_invariant("attention index overflows"))?;
                output[output_index] = f32_from_f64(value)?;
            }
        }
        Ok(output)
    }

    #[cfg(feature = "arm64")]
    fn arm64_attention(
        &self,
        layer: u32,
        writer: &KvTokenWriter<'_>,
        query: &[f32],
        backend: &Arm64Backend,
    ) -> Result<Vec<f32>, EngineError> {
        let attention_heads = usize_from_u32(self.model.inner.config.attention_head_count())?;
        let kv_heads = usize_from_u32(self.model.inner.config.kv_head_count())?;
        let head_dimension = usize_from_u32(self.model.inner.config.head_dimension())?;
        let sequence_length = usize_from_u32(
            self.token_count
                .checked_add(1)
                .ok_or_else(|| EngineError::internal_invariant("sequence length overflows"))?,
        )?;
        let kv_values = sequence_length
            .checked_mul(kv_heads)
            .and_then(|count| count.checked_mul(head_dimension))
            .ok_or_else(|| EngineError::resource_exhaustion("attention cache tensor overflows"))?;
        let mut keys = zero_f32(kv_values)?;
        let mut values = zero_f32(kv_values)?;
        for position in 0..sequence_length {
            for head in 0..kv_heads {
                for dimension in 0..head_dimension {
                    let index = position
                        .checked_mul(kv_heads)
                        .and_then(|offset| offset.checked_add(head))
                        .and_then(|offset| offset.checked_mul(head_dimension))
                        .and_then(|offset| offset.checked_add(dimension))
                        .ok_or_else(|| {
                            EngineError::internal_invariant("attention cache index overflows")
                        })?;
                    keys[index] = writer.read_key_value(
                        layer,
                        u32_from_usize(position)?,
                        u32_from_usize(head)?,
                        u32_from_usize(dimension)?,
                    )?;
                    values[index] = writer.read_cached_value(
                        layer,
                        u32_from_usize(position)?,
                        u32_from_usize(head)?,
                        u32_from_usize(dimension)?,
                    )?;
                }
            }
        }
        let query_tensor = Tensor::from_f32_values(
            TensorShape::new([attention_heads as u64, head_dimension as u64])?,
            query,
            &self.scratch_pool,
        )?;
        let key_tensor = Tensor::from_f32_values(
            TensorShape::new([sequence_length as u64, kv_heads as u64, head_dimension as u64])?,
            &keys,
            &self.scratch_pool,
        )?;
        let value_tensor =
            Tensor::from_f32_values(key_tensor.shape().clone(), &values, &self.scratch_pool)?;
        let output = backend.execute_operation(
            &Operation::GroupedQueryAttention { causal_position: self.token_count },
            &[&query_tensor, &key_tensor, &value_tensor],
            &self.scratch_pool,
        )?;
        tensor_f32_values(&output)
    }

    fn linear(
        &self,
        weight: &Tensor,
        input: &[f32],
        output_dimension: usize,
        input_dimension: usize,
    ) -> Result<Vec<f32>, EngineError> {
        #[cfg(feature = "arm64")]
        if let DecoderCpuBackend::Arm64(backend) = &self.model.inner.backend {
            if input.len() != input_dimension {
                return Err(EngineError::internal_invariant("linear input dimension differs"));
            }
            let input_tensor = Tensor::from_f32_values(
                TensorShape::new([input_dimension as u64])?,
                input,
                &self.scratch_pool,
            )?;
            let output = backend.execute_operation(
                &Operation::MatrixVector,
                &[weight, &input_tensor],
                &self.scratch_pool,
            )?;
            if output.shape().dimensions() != [output_dimension as u64] {
                return Err(EngineError::internal_invariant("ARM64 linear output shape differs"));
            }
            return tensor_f32_values(&output);
        }
        linear(weight, input, output_dimension, input_dimension)
    }

    fn rms_norm(
        &self,
        input: &[f32],
        weight: &Tensor,
        epsilon: f32,
    ) -> Result<Vec<f32>, EngineError> {
        #[cfg(feature = "arm64")]
        if let DecoderCpuBackend::Arm64(backend) = &self.model.inner.backend {
            let input_tensor = Tensor::from_f32_values(
                TensorShape::new([input.len() as u64])?,
                input,
                &self.scratch_pool,
            )?;
            let output = backend.execute_operation(
                &Operation::RmsNorm { epsilon },
                &[&input_tensor, weight],
                &self.scratch_pool,
            )?;
            return tensor_f32_values(&output);
        }
        rms_norm(input, weight, epsilon)
    }

    fn apply_rope(
        &self,
        values: &mut [f32],
        head_count: u32,
        head_dimension: u32,
        position: u32,
        theta: f32,
    ) -> Result<(), EngineError> {
        #[cfg(feature = "arm64")]
        if let DecoderCpuBackend::Arm64(backend) = &self.model.inner.backend {
            let input = Tensor::from_f32_values(
                TensorShape::new([u64::from(head_count), u64::from(head_dimension)])?,
                values,
                &self.scratch_pool,
            )?;
            let output = backend.execute_operation(
                &Operation::Rope { position, theta },
                &[&input],
                &self.scratch_pool,
            )?;
            values.copy_from_slice(&tensor_f32_values(&output)?);
            return Ok(());
        }
        apply_rope(values, head_count, head_dimension, position, theta)
    }

    fn swiglu(&self, gate: &[f32], up: &[f32]) -> Result<Vec<f32>, EngineError> {
        #[cfg(feature = "arm64")]
        if let DecoderCpuBackend::Arm64(backend) = &self.model.inner.backend {
            let shape = TensorShape::new([gate.len() as u64])?;
            let gate_tensor = Tensor::from_f32_values(shape.clone(), gate, &self.scratch_pool)?;
            let up_tensor = Tensor::from_f32_values(shape, up, &self.scratch_pool)?;
            let output = backend.execute_operation(
                &Operation::SwiGlu,
                &[&gate_tensor, &up_tensor],
                &self.scratch_pool,
            )?;
            return tensor_f32_values(&output);
        }
        swiglu(gate, up)
    }

    fn add_in_place(&self, left: &mut [f32], right: &[f32]) -> Result<(), EngineError> {
        #[cfg(feature = "arm64")]
        if let DecoderCpuBackend::Arm64(backend) = &self.model.inner.backend {
            let shape = TensorShape::new([left.len() as u64])?;
            let left_tensor = Tensor::from_f32_values(shape.clone(), left, &self.scratch_pool)?;
            let right_tensor = Tensor::from_f32_values(shape, right, &self.scratch_pool)?;
            let output = backend.execute_operation(
                &Operation::Add,
                &[&left_tensor, &right_tensor],
                &self.scratch_pool,
            )?;
            left.copy_from_slice(&tensor_f32_values(&output)?);
            return Ok(());
        }
        add_in_place(left, right)
    }
}

impl Drop for CpuDecoderSession {
    fn drop(&mut self) {
        let _ = self.model.inner.cache.destroy_sequence(self.handle);
    }
}

fn validate_model_configuration(config: &TransformerConfig) -> Result<(), EngineError> {
    if config.architecture_id().as_str() != FIRST_PARTY_DECODER_ARCHITECTURE_ID {
        return Err(EngineError::unsupported_architecture(config.architecture_id().clone()));
    }
    if config.cache_dtype() != DType::F32 {
        return Err(EngineError::not_implemented(
            "CPU decoder currently supports only f32 model and cache tensors",
        ));
    }
    Ok(())
}

fn required_weight<'a>(
    weights: &'a BTreeMap<String, Tensor>,
    name: &str,
) -> Result<&'a Tensor, EngineError> {
    weights
        .get(name)
        .ok_or_else(|| EngineError::internal_invariant("validated decoder weight is missing"))
}

fn stage_weight<'a>(
    stage: &PlacementStage,
    weights: &'a BTreeMap<String, Tensor>,
    logical_name: &str,
) -> Result<&'a Tensor, EngineError> {
    let source_name = stage
        .tensors()
        .iter()
        .find(|tensor| tensor.logical_name() == logical_name)
        .map(PlacementTensorRange::source_name)
        .ok_or_else(|| {
            EngineError::internal_invariant("streamed stage lacks a required logical tensor")
        })?;
    required_weight(weights, source_name)
}

fn validate_weights(
    config: &TransformerConfig,
    weights: &BTreeMap<String, Tensor>,
    pool: &MemoryPool,
) -> Result<(), EngineError> {
    let hidden = u64::from(config.hidden_dimension());
    let vocabulary = u64::from(config.vocabulary_size());
    let intermediate = u64::from(config.intermediate_dimension());
    let kv_dimension = u64::from(config.kv_head_count())
        .checked_mul(u64::from(config.head_dimension()))
        .ok_or_else(|| EngineError::corrupt_model_package("KV weight dimension overflows"))?;
    validate_weight(weights, "token_embedding.weight", &[vocabulary, hidden], pool)?;
    validate_weight(weights, "final_norm.weight", &[hidden], pool)?;
    validate_weight(weights, "output.weight", &[vocabulary, hidden], pool)?;
    for layer in 0..config.layer_count() {
        let prefix = format!("layers.{layer:03}");
        for suffix in ["attn_norm.weight", "ffn_norm.weight"] {
            validate_weight(weights, &format!("{prefix}.{suffix}"), &[hidden], pool)?;
        }
        validate_weight(weights, &format!("{prefix}.q_proj.weight"), &[hidden, hidden], pool)?;
        for suffix in ["k_proj.weight", "v_proj.weight"] {
            validate_weight(weights, &format!("{prefix}.{suffix}"), &[kv_dimension, hidden], pool)?;
        }
        validate_weight(weights, &format!("{prefix}.o_proj.weight"), &[hidden, hidden], pool)?;
        for suffix in ["gate_proj.weight", "up_proj.weight"] {
            validate_weight(weights, &format!("{prefix}.{suffix}"), &[intermediate, hidden], pool)?;
        }
        validate_weight(
            weights,
            &format!("{prefix}.down_proj.weight"),
            &[hidden, intermediate],
            pool,
        )?;
    }
    Ok(())
}

fn validate_weight(
    weights: &BTreeMap<String, Tensor>,
    name: &str,
    dimensions: &[u64],
    pool: &MemoryPool,
) -> Result<(), EngineError> {
    let tensor = weights
        .get(name)
        .ok_or_else(|| EngineError::corrupt_model_package("required decoder tensor is missing"))?;
    if tensor.dtype() != DType::F32 || tensor.shape().dimensions() != dimensions {
        return Err(EngineError::corrupt_model_package("decoder tensor dtype or shape is invalid"));
    }
    if !tensor.accounted_by(pool) {
        return Err(EngineError::invalid_input(
            "decoder weights must use the model aggregate memory pool",
        ));
    }
    Ok(())
}

fn embedding_row(weight: &Tensor, row: usize, hidden: usize) -> Result<Vec<f32>, EngineError> {
    let start = row
        .checked_mul(hidden)
        .ok_or_else(|| EngineError::internal_invariant("embedding row offset overflows"))?;
    let mut output = zero_f32(hidden)?;
    for (column, value) in output.iter_mut().enumerate() {
        *value =
            finite(
                weight.f32_value(start.checked_add(column).ok_or_else(|| {
                    EngineError::internal_invariant("embedding index overflows")
                })?)?,
            )?;
    }
    Ok(output)
}

fn linear(
    weight: &Tensor,
    input: &[f32],
    output_dimension: usize,
    input_dimension: usize,
) -> Result<Vec<f32>, EngineError> {
    if input.len() != input_dimension {
        return Err(EngineError::internal_invariant("linear input dimension differs"));
    }
    let mut output = zero_f32(output_dimension)?;
    for (row, result) in output.iter_mut().enumerate() {
        let row_offset = row
            .checked_mul(input_dimension)
            .ok_or_else(|| EngineError::internal_invariant("linear row offset overflows"))?;
        let mut accumulator = 0.0_f64;
        for (column, input_value) in input.iter().copied().enumerate() {
            let weight_value = weight.f32_value(
                row_offset
                    .checked_add(column)
                    .ok_or_else(|| EngineError::internal_invariant("linear index overflows"))?,
            )?;
            accumulator += f64::from(finite(weight_value)?) * f64::from(finite(input_value)?);
        }
        *result = f32_from_f64(accumulator)?;
    }
    Ok(output)
}

fn rms_norm(input: &[f32], weight: &Tensor, epsilon: f32) -> Result<Vec<f32>, EngineError> {
    let mut scale = 0.0_f64;
    let mut sum_scaled_squares = 1.0_f64;
    for value in input.iter().copied() {
        let absolute = f64::from(finite(value)?.abs());
        if absolute != 0.0 {
            if scale < absolute {
                let ratio = scale / absolute;
                sum_scaled_squares = 1.0 + sum_scaled_squares * ratio * ratio;
                scale = absolute;
            } else {
                let ratio = absolute / scale;
                sum_scaled_squares += ratio * ratio;
            }
        }
    }
    let count = u32_from_usize(input.len())?;
    let inverse_rms = if scale == 0.0 {
        1.0_f64 / f64::from(epsilon).sqrt()
    } else {
        let normalized_epsilon = f64::from(epsilon) / (scale * scale);
        (1.0 / scale) / (sum_scaled_squares / f64::from(count) + normalized_epsilon).sqrt()
    };
    let mut output = zero_f32(input.len())?;
    for (index, value) in input.iter().copied().enumerate() {
        output[index] = f32_from_f64(
            f64::from(finite(value)?) * inverse_rms * f64::from(finite(weight.f32_value(index)?)?),
        )?;
    }
    Ok(output)
}

fn apply_rope(
    values: &mut [f32],
    head_count: u32,
    head_dimension: u32,
    position: u32,
    theta: f32,
) -> Result<(), EngineError> {
    if head_dimension % 2 != 0 {
        return Err(EngineError::unsupported_architecture(
            personalai_engine_types::ArchitectureId::new(FIRST_PARTY_DECODER_ARCHITECTURE_ID)?,
        ));
    }
    let dimension = usize_from_u32(head_dimension)?;
    for head in 0..usize_from_u32(head_count)? {
        let base = head
            .checked_mul(dimension)
            .ok_or_else(|| EngineError::internal_invariant("RoPE head offset overflows"))?;
        for pair in (0..dimension).step_by(2) {
            let exponent = f64::from(u32_from_usize(pair)?) / f64::from(head_dimension);
            let angle = f64::from(position) / f64::from(theta).powf(exponent);
            let (sine, cosine) = angle.sin_cos();
            let first = f64::from(finite(values[base + pair])?);
            let second = f64::from(finite(values[base + pair + 1])?);
            values[base + pair] = f32_from_f64(first * cosine - second * sine)?;
            values[base + pair + 1] = f32_from_f64(first * sine + second * cosine)?;
        }
    }
    Ok(())
}

fn softmax(values: &mut [f32]) -> Result<(), EngineError> {
    let maximum = values
        .iter()
        .copied()
        .try_fold(f32::NEG_INFINITY, |maximum, value| Ok(maximum.max(finite(value)?)))?;
    let mut sum = 0.0_f64;
    for value in values.iter_mut() {
        *value = f32_from_f64(f64::from(*value - maximum).exp())?;
        sum += f64::from(*value);
    }
    if sum <= 0.0 {
        return Err(EngineError::backend_failure("softmax normalization is nonpositive"));
    }
    for value in values {
        *value = f32_from_f64(f64::from(*value) / sum)?;
    }
    Ok(())
}

fn swiglu(gate: &[f32], up: &[f32]) -> Result<Vec<f32>, EngineError> {
    if gate.len() != up.len() {
        return Err(EngineError::internal_invariant("SwiGLU dimensions differ"));
    }
    let mut output = zero_f32(gate.len())?;
    for index in 0..gate.len() {
        let gate_value = f64::from(finite(gate[index])?);
        let silu = gate_value / (1.0 + (-gate_value).exp());
        output[index] = f32_from_f64(silu * f64::from(finite(up[index])?))?;
    }
    Ok(output)
}

fn add_in_place(left: &mut [f32], right: &[f32]) -> Result<(), EngineError> {
    if left.len() != right.len() {
        return Err(EngineError::internal_invariant("residual dimensions differ"));
    }
    for (left_value, right_value) in left.iter_mut().zip(right) {
        *left_value = finite(*left_value + finite(*right_value)?)?;
    }
    Ok(())
}

fn finite(value: f32) -> Result<f32, EngineError> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(EngineError::backend_failure("CPU decoder encountered a non-finite value"))
    }
}

fn f32_from_f64(value: f64) -> Result<f32, EngineError> {
    if !value.is_finite() || value > f64::from(f32::MAX) || value < f64::from(f32::MIN) {
        return Err(EngineError::backend_failure("CPU decoder accumulator is outside f32 range"));
    }
    #[allow(clippy::cast_possible_truncation)]
    let converted = value as f32;
    Ok(converted)
}

fn zero_f32(length: usize) -> Result<Vec<f32>, EngineError> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(length)
        .map_err(|_| EngineError::resource_exhaustion("CPU decoder allocation failed"))?;
    values.resize(length, 0.0);
    Ok(values)
}

#[cfg(feature = "arm64")]
fn tensor_f32_values(tensor: &Tensor) -> Result<Vec<f32>, EngineError> {
    let count = usize::try_from(tensor.shape().element_count()).map_err(|_| {
        EngineError::resource_exhaustion("decoder tensor length is not addressable")
    })?;
    let mut values = zero_f32(count)?;
    for (index, value) in values.iter_mut().enumerate() {
        *value = finite(tensor.f32_value(index)?)?;
    }
    Ok(values)
}

fn check_operation_count(config: &TransformerConfig, position: u32) -> Result<(), EngineError> {
    let hidden = u64::from(config.hidden_dimension());
    let intermediate = u64::from(config.intermediate_dimension());
    let kv_dimension = u64::from(config.kv_head_count())
        .checked_mul(u64::from(config.head_dimension()))
        .ok_or_else(|| EngineError::resource_exhaustion("KV operation dimension overflows"))?;
    let linear = hidden
        .checked_mul(hidden)
        .and_then(|value| value.checked_mul(2))
        .and_then(|value| {
            kv_dimension
                .checked_mul(hidden)
                .and_then(|part| part.checked_mul(2))
                .and_then(|part| value.checked_add(part))
        })
        .and_then(|value| {
            intermediate
                .checked_mul(hidden)
                .and_then(|part| part.checked_mul(3))
                .and_then(|part| value.checked_add(part))
        })
        .ok_or_else(|| EngineError::resource_exhaustion("linear operation bound overflows"))?;
    let sequence = u64::from(position)
        .checked_add(1)
        .ok_or_else(|| EngineError::resource_exhaustion("attention length overflows"))?;
    let attention = 2_u64
        .checked_mul(u64::from(config.attention_head_count()))
        .and_then(|value| value.checked_mul(sequence))
        .and_then(|value| value.checked_mul(u64::from(config.head_dimension())))
        .ok_or_else(|| EngineError::resource_exhaustion("attention operation bound overflows"))?;
    let layers = linear
        .checked_add(attention)
        .and_then(|value| value.checked_mul(u64::from(config.layer_count())))
        .ok_or_else(|| EngineError::resource_exhaustion("layer operation bound overflows"))?;
    let total =
        layers
            .checked_add(u64::from(config.vocabulary_size()).checked_mul(hidden).ok_or_else(
                || EngineError::resource_exhaustion("logit operation bound overflows"),
            )?)
            .ok_or_else(|| EngineError::resource_exhaustion("decoder operation bound overflows"))?;
    if total > MAX_CPU_REFERENCE_OPERATIONS {
        return Err(EngineError::resource_exhaustion(
            "CPU decoder step exceeds the reference compute bound",
        ));
    }
    Ok(())
}

fn workspace_bytes(config: &TransformerConfig, position: u32) -> Result<u64, EngineError> {
    let hidden = u64::from(config.hidden_dimension());
    let intermediate = u64::from(config.intermediate_dimension());
    let kv_dimension = u64::from(config.kv_head_count())
        .checked_mul(u64::from(config.head_dimension()))
        .ok_or_else(|| EngineError::resource_exhaustion("KV workspace dimension overflows"))?;
    let score_count = u64::from(config.attention_head_count())
        .checked_mul(
            u64::from(position)
                .checked_add(1)
                .ok_or_else(|| EngineError::resource_exhaustion("score length overflows"))?,
        )
        .ok_or_else(|| EngineError::resource_exhaustion("score workspace overflows"))?;
    let elements = hidden
        .checked_mul(16)
        .and_then(|value| intermediate.checked_mul(6).and_then(|part| value.checked_add(part)))
        .and_then(|value| kv_dimension.checked_mul(4).and_then(|part| value.checked_add(part)))
        .and_then(|value| score_count.checked_mul(2).and_then(|part| value.checked_add(part)))
        .and_then(|value| value.checked_add(u64::from(config.vocabulary_size())))
        .ok_or_else(|| EngineError::resource_exhaustion("decoder workspace bound overflows"))?;
    elements
        .checked_mul(DType::F32.byte_width())
        .filter(|bytes| *bytes > 0)
        .ok_or_else(|| EngineError::resource_exhaustion("decoder workspace bytes overflow"))
}

fn usize_from_u32(value: u32) -> Result<usize, EngineError> {
    usize::try_from(value)
        .map_err(|_| EngineError::resource_exhaustion("model dimension is not addressable"))
}

fn u32_from_usize(value: usize) -> Result<u32, EngineError> {
    u32::try_from(value)
        .map_err(|_| EngineError::resource_exhaustion("model index exceeds its field"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use personalai_engine_types::{ArchitectureId, ContextLimit};

    #[test]
    fn tiny_decoder_executes_deterministically_and_advances_cache() {
        let pool = MemoryPool::new(1 << 20).expect("pool must validate");
        let config = tiny_config();
        let model = CpuDecoderModel::new(config, tiny_weights(&pool), &pool)
            .expect("tiny exact architecture must construct");
        let budget = ResourceBudget::new(1 << 20, 1 << 18, 1).expect("budget must validate");
        let mut session = model
            .start_session(&budget, RequestId::new(1).unwrap(), SequenceId::new(1).unwrap())
            .expect("session must construct");
        let first = session.forward_token(1).expect("first token must execute");
        let second = session.forward_token(2).expect("second token must execute");
        assert_eq!(first.shape().dimensions(), &[8]);
        assert_eq!(second.shape().dimensions(), &[8]);
        assert_eq!(session.token_count(), 2);
        for index in 0..8 {
            assert!(first.f32_value(index).expect("logit must read").is_finite());
            assert!(second.f32_value(index).expect("logit must read").is_finite());
        }
    }

    #[test]
    fn prefill_matches_incremental_decode_and_enforces_context() {
        let pool = MemoryPool::new(1 << 20).expect("pool must validate");
        let model = CpuDecoderModel::new(tiny_config(), tiny_weights(&pool), &pool)
            .expect("tiny exact architecture must construct");
        let budget = ResourceBudget::new(1 << 20, 1 << 18, 2).expect("budget must validate");
        let mut prefilled = model
            .start_session(&budget, RequestId::new(1).unwrap(), SequenceId::new(1).unwrap())
            .expect("prefill session must construct");
        let mut incremental = model
            .start_session(&budget, RequestId::new(2).unwrap(), SequenceId::new(2).unwrap())
            .expect("incremental session must construct");
        let expected = prefilled.prefill(&[1, 2, 3]).expect("prefill must execute");
        let mut actual = None;
        for token in [1, 2, 3] {
            actual = Some(incremental.decode(token).expect("decode must execute"));
        }
        let actual = actual.expect("incremental logits must exist");
        for index in 0..8 {
            let difference =
                (expected.f32_value(index).unwrap() - actual.f32_value(index).unwrap()).abs();
            assert!(difference <= 1.0e-6);
        }
        assert!(prefilled.prefill(&[]).is_err());
        assert!(prefilled.prefill(&[1; 6]).is_err());
        let cached = incremental.decode(4).expect("cached decode must execute");
        drop(prefilled);
        let mut full_sequence = model
            .start_session(&budget, RequestId::new(3).unwrap(), SequenceId::new(3).unwrap())
            .expect("full-sequence session must construct");
        let recomputed =
            full_sequence.prefill(&[1, 2, 3, 4]).expect("full sequence must recompute");
        for index in 0..8 {
            let difference =
                (cached.f32_value(index).unwrap() - recomputed.f32_value(index).unwrap()).abs();
            assert!(difference <= 1.0e-6);
        }
        assert!(incremental.decode(8).is_err());
        drop(full_sequence);
        drop(incremental);
        let mut maximum = model
            .start_session(&budget, RequestId::new(4).unwrap(), SequenceId::new(4).unwrap())
            .expect("maximum session must construct");
        maximum.prefill(&[1; 8]).expect("exact context must execute");
        assert!(maximum.decode(1).is_err());
    }

    #[test]
    fn missing_or_unaccounted_weights_fail_closed() {
        let pool = MemoryPool::new(1 << 20).expect("pool must validate");
        let mut weights = tiny_weights(&pool);
        weights.remove("output.weight");
        assert!(matches!(
            CpuDecoderModel::new(tiny_config(), weights, &pool),
            Err(EngineError::CorruptModelPackage(_))
        ));
    }

    #[cfg(all(feature = "arm64", target_arch = "aarch64"))]
    #[test]
    fn complete_arm64_decoder_matches_scalar_and_cached_execution() {
        let scalar_pool = MemoryPool::new(16 << 20).expect("scalar pool must validate");
        let arm64_pool = MemoryPool::new(16 << 20).expect("ARM64 pool must validate");
        let config = arm64_config();
        let scalar = CpuDecoderModel::new_registered(
            ModelId::new("scalar-model").unwrap(),
            config.clone(),
            arm64_weights(&scalar_pool),
            &scalar_pool,
        )
        .expect("scalar model must construct");
        let arm64 = CpuDecoderModel::new_registered_arm64(
            ModelId::new("arm64-model").unwrap(),
            config,
            arm64_weights(&arm64_pool),
            &arm64_pool,
            Arm64Backend::new().expect("ARM64 backend must construct"),
        )
        .expect("ARM64 model must construct");
        assert_eq!(arm64.backend_id().unwrap().as_str(), "arm64");
        let budget = ResourceBudget::new(16 << 20, 2 << 20, 2).unwrap();
        let mut scalar_session = scalar
            .start_session(&budget, RequestId::new(11).unwrap(), SequenceId::new(11).unwrap())
            .unwrap();
        let mut arm64_session = arm64
            .start_session(&budget, RequestId::new(12).unwrap(), SequenceId::new(12).unwrap())
            .unwrap();
        let scalar_logits = scalar_session.prefill(&[1, 2, 3]).unwrap();
        let arm64_logits = arm64_session.prefill(&[1, 2, 3]).unwrap();
        assert_tensors_close(&scalar_logits, &arm64_logits, 2.0e-4, 2.0e-4);

        let scalar_next = scalar_session.decode(4).unwrap();
        let arm64_next = arm64_session.decode(4).unwrap();
        assert_tensors_close(&scalar_next, &arm64_next, 2.0e-4, 2.0e-4);
    }

    fn tiny_config() -> TransformerConfig {
        TransformerConfig::new_decoder(
            ArchitectureId::new(FIRST_PARTY_DECODER_ARCHITECTURE_ID)
                .expect("architecture id must validate"),
            1,
            4,
            1,
            1,
            8,
            ContextLimit::new(8).expect("context must validate"),
            DType::F32,
            8,
            1.0e-5,
            10_000.0,
        )
        .expect("tiny config must validate")
    }

    #[cfg(all(feature = "arm64", target_arch = "aarch64"))]
    fn arm64_config() -> TransformerConfig {
        TransformerConfig::new_decoder(
            ArchitectureId::new(FIRST_PARTY_DECODER_ARCHITECTURE_ID).unwrap(),
            1,
            32,
            4,
            2,
            32,
            ContextLimit::new(4).unwrap(),
            DType::F32,
            64,
            1.0e-5,
            10_000.0,
        )
        .unwrap()
    }

    #[cfg(all(feature = "arm64", target_arch = "aarch64"))]
    fn arm64_weights(pool: &MemoryPool) -> BTreeMap<String, Tensor> {
        let mut weights = BTreeMap::new();
        insert(&mut weights, "token_embedding.weight", [32, 32], &repeated(1024, 0.0005), pool);
        insert(&mut weights, "final_norm.weight", [32], &[1.0; 32], pool);
        insert(&mut weights, "output.weight", [32, 32], &repeated(1024, 0.0003), pool);
        insert(&mut weights, "layers.000.attn_norm.weight", [32], &[1.0; 32], pool);
        insert(&mut weights, "layers.000.q_proj.weight", [32, 32], &identity(32), pool);
        insert(&mut weights, "layers.000.k_proj.weight", [16, 32], &repeated(512, 0.0004), pool);
        insert(&mut weights, "layers.000.v_proj.weight", [16, 32], &repeated(512, 0.0002), pool);
        insert(&mut weights, "layers.000.o_proj.weight", [32, 32], &identity(32), pool);
        insert(&mut weights, "layers.000.ffn_norm.weight", [32], &[1.0; 32], pool);
        insert(
            &mut weights,
            "layers.000.gate_proj.weight",
            [64, 32],
            &repeated(2048, 0.0001),
            pool,
        );
        insert(&mut weights, "layers.000.up_proj.weight", [64, 32], &repeated(2048, 0.00015), pool);
        insert(
            &mut weights,
            "layers.000.down_proj.weight",
            [32, 64],
            &repeated(2048, 0.0001),
            pool,
        );
        weights
    }

    #[cfg(all(feature = "arm64", target_arch = "aarch64"))]
    fn assert_tensors_close(left: &Tensor, right: &Tensor, absolute: f32, relative: f32) {
        assert_eq!(left.shape(), right.shape());
        for index in 0..usize::try_from(left.shape().element_count()).unwrap() {
            let left = left.f32_value(index).unwrap();
            let right = right.f32_value(index).unwrap();
            let difference = (left - right).abs();
            let scale = left.abs().max(right.abs());
            assert!(
                difference <= absolute + relative * scale,
                "index {index}: scalar={left} arm64={right} difference={difference}"
            );
        }
    }

    fn tiny_weights(pool: &MemoryPool) -> BTreeMap<String, Tensor> {
        let mut weights = BTreeMap::new();
        insert(&mut weights, "token_embedding.weight", [8, 4], &repeated(32, 0.05), pool);
        insert(&mut weights, "final_norm.weight", [4], &[1.0; 4], pool);
        insert(&mut weights, "output.weight", [8, 4], &repeated(32, 0.03), pool);
        insert(&mut weights, "layers.000.attn_norm.weight", [4], &[1.0; 4], pool);
        insert(&mut weights, "layers.000.q_proj.weight", [4, 4], &identity(4), pool);
        insert(&mut weights, "layers.000.k_proj.weight", [4, 4], &identity(4), pool);
        insert(&mut weights, "layers.000.v_proj.weight", [4, 4], &identity(4), pool);
        insert(&mut weights, "layers.000.o_proj.weight", [4, 4], &identity(4), pool);
        insert(&mut weights, "layers.000.ffn_norm.weight", [4], &[1.0; 4], pool);
        insert(&mut weights, "layers.000.gate_proj.weight", [8, 4], &repeated(32, 0.02), pool);
        insert(&mut weights, "layers.000.up_proj.weight", [8, 4], &repeated(32, 0.01), pool);
        insert(&mut weights, "layers.000.down_proj.weight", [4, 8], &repeated(32, 0.01), pool);
        weights
    }

    fn insert<const R: usize>(
        weights: &mut BTreeMap<String, Tensor>,
        name: &str,
        dimensions: [u64; R],
        values: &[f32],
        pool: &MemoryPool,
    ) {
        let tensor = Tensor::from_f32_values(
            TensorShape::new(dimensions).expect("shape must validate"),
            values,
            pool,
        )
        .expect("weight must construct");
        weights.insert(name.to_owned(), tensor);
    }

    fn identity(dimension: usize) -> Vec<f32> {
        let mut values = vec![0.0; dimension * dimension];
        for index in 0..dimension {
            values[index * dimension + index] = 1.0;
        }
        values
    }

    fn repeated(length: usize, scale: f32) -> Vec<f32> {
        (0..length)
            .map(|index| {
                let bounded = u16::try_from(index).expect("tiny test index must fit");
                (f32::from(bounded) + 1.0) * scale
            })
            .collect()
    }
}
