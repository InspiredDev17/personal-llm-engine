//! Complete first-party decoder execution over the reviewed Metal operation backend.

use crate::{FIRST_PARTY_DECODER_ARCHITECTURE_ID, TransformerConfig, TransformerPlan};
use personalai_engine_cpu_reference::{Backend, Operation};
use personalai_engine_kv_cache::KvCachePlan;
use personalai_engine_memory::{MemoryOwner, MemoryPool};
use personalai_engine_metal::{
    MetalBackend, MetalCancellation, MetalKvCache, MetalKvCacheMetrics, MetalKvSequenceHandle,
    MetalModelWeights,
};
use personalai_engine_tensor::Tensor;
use personalai_engine_types::{
    DType, EngineError, MAX_CONCURRENT_SEQUENCES, ModelId, RequestId, ResourceBudget, SequenceId,
    TensorShape,
};
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex, MutexGuard};

const WEIGHTS_PER_LAYER: usize = 9;
const GLOBAL_WEIGHT_COUNT: usize = 3;

/// Validated decoder model whose allowlisted operations execute through Metal.
///
/// Package validation and F32 materialization happen before construction. Model
/// metadata never selects operations or kernels: this type contains the only
/// execution recipe for `decoder-rms-rope-swiglu-gqa-v1`.
#[derive(Clone, Debug)]
pub struct MetalDecoderModel {
    inner: Arc<MetalDecoderModelInner>,
}

#[derive(Debug)]
struct MetalDecoderModelInner {
    config: TransformerConfig,
    pool: MemoryPool,
    backend: MetalBackend,
    device_weights: Mutex<Option<MetalModelWeights>>,
    cache: MetalKvCache,
}

impl MetalDecoderModel {
    /// Validates the architecture and creates a device-cache owner.
    pub fn new_registered(
        model_id: ModelId,
        config: TransformerConfig,
        weights: BTreeMap<String, Tensor>,
        pool: &MemoryPool,
        backend: MetalBackend,
    ) -> Result<Self, EngineError> {
        validate_weights(&config, &weights, pool)?;
        let cache_budget =
            ResourceBudget::new(pool.limit_bytes(), pool.limit_bytes(), MAX_CONCURRENT_SEQUENCES)?;
        let cache_plan = KvCachePlan::new(
            config.layer_count(),
            config.kv_head_count(),
            config.head_dimension(),
            config.context_limit().get(),
            config.cache_dtype(),
            &cache_budget,
        )?;
        let cache =
            MetalKvCache::new(model_id, cache_plan, MAX_CONCURRENT_SEQUENCES, backend.clone())?;
        let device_weights = backend.load_model_weights(&weights)?;
        drop(weights);
        Ok(Self {
            inner: Arc::new(MetalDecoderModelInner {
                config,
                pool: pool.clone(),
                backend,
                device_weights: Mutex::new(Some(device_weights)),
                cache,
            }),
        })
    }

    /// Returns the validated architecture configuration.
    pub fn config(&self) -> &TransformerConfig {
        &self.inner.config
    }

    /// Returns aggregate host-memory accounting for weights and intermediates.
    pub fn memory_pool(&self) -> &MemoryPool {
        &self.inner.pool
    }

    /// Returns device KV-cache pressure without model or prompt content.
    pub fn kv_cache_metrics(&self) -> Result<MetalKvCacheMetrics, EngineError> {
        self.inner.cache.metrics()
    }

    /// Starts one sequence with independent device KV state and cancellation.
    pub fn start_session(
        &self,
        budget: &ResourceBudget,
        request_id: RequestId,
        sequence_id: SequenceId,
    ) -> Result<MetalDecoderSession, EngineError> {
        let plan = TransformerPlan::new(self.inner.config.clone(), budget)?;
        plan.require_executable_architecture()?;
        if plan.kv_cache() != self.inner.cache.plan() {
            return Err(EngineError::internal_invariant(
                "request and Metal model KV layouts differ",
            ));
        }
        if self.inner.cache.metrics()?.active_sequence_count >= budget.max_concurrent_sequences() {
            return Err(EngineError::resource_exhaustion(
                "request Metal sequence concurrency budget is exhausted",
            ));
        }
        drop(self.resident_weights()?);
        let handle = self.inner.cache.create_sequence(request_id, sequence_id)?;
        Ok(MetalDecoderSession {
            model: self.clone(),
            handle: Some(handle),
            scratch_pool: self.inner.pool.scoped(MemoryOwner::RequestScratch(request_id)),
            output_pool: self.inner.pool.scoped(MemoryOwner::OutputLogits(request_id)),
            cancellation: MetalCancellation::new(),
            token_count: 0,
            failed: false,
        })
    }

    /// Unloads only when no sequence remains active, then reclaims idle buffers.
    pub fn unload_if_idle(&self) -> Result<u64, EngineError> {
        if self.inner.cache.metrics()?.active_sequence_count != 0 {
            return Err(EngineError::resource_exhaustion(
                "Metal model unload rejected while sequences are active",
            ));
        }
        self.inner.cache.unload_model()?;
        self.inner
            .device_weights
            .lock()
            .map_err(|_| EngineError::internal_invariant("Metal weight lock is poisoned"))?
            .take();
        self.inner.backend.reclaim()
    }

    fn resident_weights(&self) -> Result<MutexGuard<'_, Option<MetalModelWeights>>, EngineError> {
        let guard = self
            .inner
            .device_weights
            .lock()
            .map_err(|_| EngineError::internal_invariant("Metal weight lock is poisoned"))?;
        if guard.is_none() {
            return Err(EngineError::backend_failure("Metal model is unloaded"));
        }
        Ok(guard)
    }
}

/// Per-sequence Metal decoder state with device KV ownership.
#[derive(Debug)]
pub struct MetalDecoderSession {
    model: MetalDecoderModel,
    handle: Option<MetalKvSequenceHandle>,
    scratch_pool: MemoryPool,
    output_pool: MemoryPool,
    cancellation: MetalCancellation,
    token_count: u32,
    failed: bool,
}

impl MetalDecoderSession {
    /// Returns committed positions.
    pub const fn token_count(&self) -> u32 {
        self.token_count
    }

    /// Returns a cloneable cooperative cancellation capability.
    pub fn cancellation(&self) -> MetalCancellation {
        self.cancellation.clone()
    }

    /// Clears and invalidates this sequence immediately.
    pub fn cancel(&mut self) -> Result<(), EngineError> {
        self.cancellation.cancel();
        self.invalidate()
    }

    /// Processes a non-empty prompt and returns final-position logits.
    pub fn prefill(&mut self, tokens: &[u32]) -> Result<Tensor, EngineError> {
        if tokens.is_empty() {
            return Err(EngineError::invalid_input("Metal prefill requires at least one token"));
        }
        let additional = u32::try_from(tokens.len())
            .map_err(|_| EngineError::resource_exhaustion("Metal prefill count exceeds u32"))?;
        if self
            .token_count
            .checked_add(additional)
            .is_none_or(|total| total > self.model.inner.config.context_limit().get())
        {
            return Err(EngineError::resource_exhaustion("Metal prefill exceeds context"));
        }
        let mut logits = None;
        for token in tokens {
            logits = Some(self.decode(*token)?);
        }
        logits.ok_or_else(|| EngineError::internal_invariant("Metal prefill produced no logits"))
    }

    /// Executes exactly one cached decode position.
    pub fn decode(&mut self, token: u32) -> Result<Tensor, EngineError> {
        let result = self.forward_token(token);
        if matches!(result, Err(EngineError::Cancellation(_) | EngineError::BackendFailure(_))) {
            let _ = self.invalidate();
        }
        result
    }

    #[allow(clippy::too_many_lines)]
    fn forward_token(&mut self, token: u32) -> Result<Tensor, EngineError> {
        self.check_live()?;
        if token >= self.model.inner.config.vocabulary_size() {
            return Err(EngineError::invalid_input("input token exceeds Metal model vocabulary"));
        }
        if self.token_count >= self.model.inner.config.context_limit().get() {
            return Err(EngineError::resource_exhaustion("Metal decoder context is exhausted"));
        }
        let handle = self.handle.ok_or_else(|| EngineError::cancellation(None))?;
        let hidden = self.model.inner.config.hidden_dimension();
        let mut state = self.embedding(token)?;
        let layer_width = checked_product(
            self.model.inner.config.kv_head_count(),
            self.model.inner.config.head_dimension(),
            "Metal KV layer width overflows",
        )?;
        let all_layers = usize::try_from(self.model.inner.config.layer_count())
            .ok()
            .and_then(|layers| layers.checked_mul(layer_width))
            .ok_or_else(|| EngineError::resource_exhaustion("Metal KV token width overflows"))?;
        let mut token_keys = zero_f32(all_layers)?;
        let mut token_values = zero_f32(all_layers)?;

        for layer in 0..self.model.inner.config.layer_count() {
            let prefix = format!("layers.{layer:03}");
            let attention_input = self.rms_norm(&format!("{prefix}.attn_norm.weight"), &state)?;
            let query = self.linear(&format!("{prefix}.q_proj.weight"), &attention_input)?;
            let key = self.linear(&format!("{prefix}.k_proj.weight"), &attention_input)?;
            let value = self.linear(&format!("{prefix}.v_proj.weight"), &attention_input)?;
            let query = reshape_copy(
                &query,
                TensorShape::new([
                    u64::from(self.model.inner.config.attention_head_count()),
                    u64::from(self.model.inner.config.head_dimension()),
                ])?,
                &self.scratch_pool,
            )?;
            let key = reshape_copy(
                &key,
                TensorShape::new([
                    u64::from(self.model.inner.config.kv_head_count()),
                    u64::from(self.model.inner.config.head_dimension()),
                ])?,
                &self.scratch_pool,
            )?;
            let query = self.execute(
                &Operation::Rope {
                    position: self.token_count,
                    theta: self.model.inner.config.rope_theta(),
                },
                &[&query],
                &self.scratch_pool,
            )?;
            let key = self.execute(
                &Operation::Rope {
                    position: self.token_count,
                    theta: self.model.inner.config.rope_theta(),
                },
                &[&key],
                &self.scratch_pool,
            )?;
            let layer_offset = usize::try_from(layer)
                .ok()
                .and_then(|value| value.checked_mul(layer_width))
                .ok_or_else(|| {
                    EngineError::resource_exhaustion("Metal KV layer offset overflows")
                })?;
            copy_tensor_values(&key, &mut token_keys[layer_offset..layer_offset + layer_width])?;
            copy_tensor_values(
                &value,
                &mut token_values[layer_offset..layer_offset + layer_width],
            )?;
            let (keys, values) =
                self.attention_history(handle, layer, &key, &value, layer_width)?;
            let attention = self.execute(
                &Operation::GroupedQueryAttention { causal_position: self.token_count },
                &[&query, &keys, &values],
                &self.scratch_pool,
            )?;
            let attention = reshape_copy(
                &attention,
                TensorShape::new([u64::from(hidden)])?,
                &self.scratch_pool,
            )?;
            let projected = self.linear(&format!("{prefix}.o_proj.weight"), &attention)?;
            state =
                self.execute(&Operation::ResidualAdd, &[&state, &projected], &self.scratch_pool)?;
            let ffn_input = self.rms_norm(&format!("{prefix}.ffn_norm.weight"), &state)?;
            let gate = self.linear(&format!("{prefix}.gate_proj.weight"), &ffn_input)?;
            let up = self.linear(&format!("{prefix}.up_proj.weight"), &ffn_input)?;
            let activated = self.execute(&Operation::SwiGlu, &[&gate, &up], &self.scratch_pool)?;
            let down = self.linear(&format!("{prefix}.down_proj.weight"), &activated)?;
            state = self.execute(&Operation::ResidualAdd, &[&state, &down], &self.scratch_pool)?;
        }

        let normalized = self.rms_norm("final_norm.weight", &state)?;
        let logits = self.linear_to("output.weight", &normalized, &self.output_pool)?;
        self.check_live()?;
        self.model.inner.cache.append_token(handle, &token_keys, &token_values)?;
        self.token_count = self
            .token_count
            .checked_add(1)
            .ok_or_else(|| EngineError::internal_invariant("Metal token count overflows"))?;
        Ok(logits)
    }

    fn attention_history(
        &self,
        handle: MetalKvSequenceHandle,
        layer: u32,
        key: &Tensor,
        value: &Tensor,
        layer_width: usize,
    ) -> Result<(Tensor, Tensor), EngineError> {
        let positions = self
            .token_count
            .checked_add(1)
            .ok_or_else(|| EngineError::resource_exhaustion("Metal attention length overflows"))?;
        let total = usize::try_from(positions)
            .ok()
            .and_then(|count| count.checked_mul(layer_width))
            .ok_or_else(|| EngineError::resource_exhaustion("Metal attention history overflows"))?;
        let mut keys = zero_f32(total)?;
        let mut values = zero_f32(total)?;
        for position in 0..self.token_count {
            let start = usize::try_from(position)
                .ok()
                .and_then(|value| value.checked_mul(layer_width))
                .ok_or_else(|| {
                    EngineError::resource_exhaustion("Metal history offset overflows")
                })?;
            self.model.inner.cache.read_key_into(
                handle,
                layer,
                position,
                &mut keys[start..start + layer_width],
            )?;
            self.model.inner.cache.read_value_into(
                handle,
                layer,
                position,
                &mut values[start..start + layer_width],
            )?;
        }
        let current = total - layer_width;
        copy_tensor_values(key, &mut keys[current..])?;
        copy_tensor_values(value, &mut values[current..])?;
        let shape = TensorShape::new([
            u64::from(positions),
            u64::from(self.model.inner.config.kv_head_count()),
            u64::from(self.model.inner.config.head_dimension()),
        ])?;
        Ok((
            Tensor::from_f32_values(shape.clone(), &keys, &self.scratch_pool)?,
            Tensor::from_f32_values(shape, &values, &self.scratch_pool)?,
        ))
    }

    fn execute(
        &self,
        operation: &Operation,
        inputs: &[&Tensor],
        pool: &MemoryPool,
    ) -> Result<Tensor, EngineError> {
        self.check_live()?;
        let output = self.model.inner.backend.execute_operation(operation, inputs, pool)?;
        self.check_live()?;
        Ok(output)
    }

    fn embedding(&self, token: u32) -> Result<Tensor, EngineError> {
        self.check_live()?;
        let weights = self.model.resident_weights()?;
        self.model.inner.backend.resident_embedding(
            weights.as_ref().expect("checked resident weights"),
            "token_embedding.weight",
            token,
            &self.scratch_pool,
            &self.cancellation,
        )
    }

    fn rms_norm(&self, name: &str, input: &Tensor) -> Result<Tensor, EngineError> {
        self.check_live()?;
        let weights = self.model.resident_weights()?;
        self.model.inner.backend.resident_rms_norm(
            weights.as_ref().expect("checked resident weights"),
            name,
            input,
            self.model.inner.config.rms_norm_epsilon(),
            &self.scratch_pool,
            &self.cancellation,
        )
    }

    fn linear(&self, name: &str, input: &Tensor) -> Result<Tensor, EngineError> {
        self.linear_to(name, input, &self.scratch_pool)
    }

    fn linear_to(
        &self,
        name: &str,
        input: &Tensor,
        pool: &MemoryPool,
    ) -> Result<Tensor, EngineError> {
        self.check_live()?;
        let weights = self.model.resident_weights()?;
        self.model.inner.backend.resident_matrix_vector(
            weights.as_ref().expect("checked resident weights"),
            name,
            input,
            pool,
            &self.cancellation,
        )
    }

    fn check_live(&self) -> Result<(), EngineError> {
        if self.failed || self.handle.is_none() {
            return Err(EngineError::cancellation(None));
        }
        if self.cancellation.is_cancelled() {
            return Err(EngineError::cancellation(None));
        }
        Ok(())
    }

    fn invalidate(&mut self) -> Result<(), EngineError> {
        self.failed = true;
        if let Some(handle) = self.handle.take() {
            self.model.inner.cache.cancel_sequence(handle)?;
            self.model.inner.cache.destroy_sequence(handle)?;
        }
        Ok(())
    }
}

impl Drop for MetalDecoderSession {
    fn drop(&mut self) {
        if let Some(handle) = self.handle.take() {
            let _ = self.model.inner.cache.destroy_sequence(handle);
        }
    }
}

fn validate_weights(
    config: &TransformerConfig,
    weights: &BTreeMap<String, Tensor>,
    pool: &MemoryPool,
) -> Result<(), EngineError> {
    if config.architecture_id().as_str() != FIRST_PARTY_DECODER_ARCHITECTURE_ID {
        return Err(EngineError::unsupported_architecture(config.architecture_id().clone()));
    }
    if config.cache_dtype() != DType::F32 {
        return Err(EngineError::not_implemented("Metal decoder requires F32 cache storage"));
    }
    let expected = usize::try_from(config.layer_count())
        .ok()
        .and_then(|layers| layers.checked_mul(WEIGHTS_PER_LAYER))
        .and_then(|count| count.checked_add(GLOBAL_WEIGHT_COUNT))
        .ok_or_else(|| EngineError::corrupt_model_package("Metal weight count overflows"))?;
    if weights.len() != expected {
        return Err(EngineError::corrupt_model_package(
            "Metal tensor count differs from the exact architecture",
        ));
    }
    let hidden = u64::from(config.hidden_dimension());
    let vocabulary = u64::from(config.vocabulary_size());
    let intermediate = u64::from(config.intermediate_dimension());
    let kv = u64::from(config.kv_head_count())
        .checked_mul(u64::from(config.head_dimension()))
        .ok_or_else(|| EngineError::corrupt_model_package("Metal KV dimension overflows"))?;
    validate_weight(weights, "token_embedding.weight", &[vocabulary, hidden], pool)?;
    validate_weight(weights, "final_norm.weight", &[hidden], pool)?;
    validate_weight(weights, "output.weight", &[vocabulary, hidden], pool)?;
    for layer in 0..config.layer_count() {
        let prefix = format!("layers.{layer:03}");
        validate_weight(weights, &format!("{prefix}.attn_norm.weight"), &[hidden], pool)?;
        validate_weight(weights, &format!("{prefix}.ffn_norm.weight"), &[hidden], pool)?;
        validate_weight(weights, &format!("{prefix}.q_proj.weight"), &[hidden, hidden], pool)?;
        validate_weight(weights, &format!("{prefix}.k_proj.weight"), &[kv, hidden], pool)?;
        validate_weight(weights, &format!("{prefix}.v_proj.weight"), &[kv, hidden], pool)?;
        validate_weight(weights, &format!("{prefix}.o_proj.weight"), &[hidden, hidden], pool)?;
        validate_weight(
            weights,
            &format!("{prefix}.gate_proj.weight"),
            &[intermediate, hidden],
            pool,
        )?;
        validate_weight(
            weights,
            &format!("{prefix}.up_proj.weight"),
            &[intermediate, hidden],
            pool,
        )?;
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
        .ok_or_else(|| EngineError::corrupt_model_package("required Metal tensor is missing"))?;
    if tensor.dtype() != DType::F32 || tensor.shape().dimensions() != dimensions {
        return Err(EngineError::corrupt_model_package("Metal tensor dtype or shape is invalid"));
    }
    if !tensor.accounted_by(pool) {
        return Err(EngineError::invalid_input(
            "Metal weights must use the model aggregate memory pool",
        ));
    }
    Ok(())
}

fn reshape_copy(
    tensor: &Tensor,
    shape: TensorShape,
    pool: &MemoryPool,
) -> Result<Tensor, EngineError> {
    if tensor.shape().element_count() != shape.element_count() {
        return Err(EngineError::internal_invariant("Metal reshape element count differs"));
    }
    Tensor::copy_from_bytes(DType::F32, shape, tensor.as_bytes(), pool)
}

fn copy_tensor_values(tensor: &Tensor, destination: &mut [f32]) -> Result<(), EngineError> {
    if usize::try_from(tensor.shape().element_count()).ok() != Some(destination.len()) {
        return Err(EngineError::internal_invariant("Metal tensor copy length differs"));
    }
    for (index, value) in destination.iter_mut().enumerate() {
        *value = tensor.f32_value(index)?;
        if !value.is_finite() {
            return Err(EngineError::backend_failure("Metal decoder output is non-finite"));
        }
    }
    Ok(())
}

fn checked_product(left: u32, right: u32, detail: &'static str) -> Result<usize, EngineError> {
    usize::try_from(left)
        .ok()
        .and_then(|left| left.checked_mul(usize::try_from(right).ok()?))
        .ok_or_else(|| EngineError::resource_exhaustion(detail))
}

fn zero_f32(length: usize) -> Result<Vec<f32>, EngineError> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(length)
        .map_err(|_| EngineError::resource_exhaustion("Metal decoder host allocation failed"))?;
    values.resize(length, 0.0);
    Ok(values)
}
