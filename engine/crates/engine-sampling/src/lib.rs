//! First-party bounded deterministic token sampling.
//!
//! Filtering order is fixed: allow/deny constraints, repetition penalty,
//! temperature, top-k, min-p, then top-p. Equal scores use the lower token ID.
//! [`DeterministicRng`] is reproducible xorshift64* state and is explicitly not
//! suitable for cryptography, secrets, identifiers, or authorization.

use personalai_engine_memory::{MemoryOwner, MemoryPool};
use personalai_engine_tensor::Tensor;
use personalai_engine_types::{
    DType, EngineError, MAX_TOKEN_COUNT, MAX_VOCABULARY_SIZE, RequestId,
};
use std::cmp::Ordering;
use std::mem::size_of;

/// Maximum accepted temperature.
pub const MAX_TEMPERATURE: f32 = 2.0;
/// Minimum accepted non-neutral repetition penalty.
pub const MIN_REPETITION_PENALTY: f32 = 0.01;
/// Maximum accepted repetition penalty.
pub const MAX_REPETITION_PENALTY: f32 = 100.0;

/// Validated token sampling and stop settings.
#[derive(Debug, Clone, PartialEq)]
pub struct SamplingConfig {
    temperature: f32,
    top_k: Option<u32>,
    top_p: f32,
    min_p: f32,
    repetition_penalty: f32,
    seed: u64,
    allowed_token_ids: Box<[u32]>,
    disallowed_token_ids: Box<[u32]>,
    stop_token_ids: Box<[u32]>,
    maximum_generated_tokens: u32,
}

impl SamplingConfig {
    /// Creates settings with neutral filters, seed zero, and the global token cap.
    pub fn new(temperature: f32) -> Result<Self, EngineError> {
        validate_temperature(temperature)?;
        Ok(Self {
            temperature,
            top_k: None,
            top_p: 1.0,
            min_p: 0.0,
            repetition_penalty: 1.0,
            seed: 0,
            allowed_token_ids: Box::new([]),
            disallowed_token_ids: Box::new([]),
            stop_token_ids: Box::new([]),
            maximum_generated_tokens: MAX_TOKEN_COUNT,
        })
    }

    /// Returns deterministic temperature-zero settings.
    pub fn greedy() -> Self {
        Self::new(0.0).expect("zero temperature is an invariant")
    }

    /// Configures top-k; `None` disables it and zero is rejected.
    pub fn with_top_k(mut self, top_k: Option<u32>) -> Result<Self, EngineError> {
        if top_k.is_some_and(|value| value == 0 || value > MAX_VOCABULARY_SIZE) {
            return Err(EngineError::invalid_input("sampling top-k is outside hard bounds"));
        }
        self.top_k = top_k;
        Ok(self)
    }

    /// Configures nucleus probability in `(0, 1]`.
    pub fn with_top_p(mut self, top_p: f32) -> Result<Self, EngineError> {
        if !(top_p.is_finite() && 0.0 < top_p && top_p <= 1.0) {
            return Err(EngineError::invalid_input("sampling top-p is outside hard bounds"));
        }
        self.top_p = top_p;
        Ok(self)
    }

    /// Configures min-p relative to the most likely remaining token in `[0, 1]`.
    pub fn with_min_p(mut self, min_p: f32) -> Result<Self, EngineError> {
        if !min_p.is_finite() || !(0.0..=1.0).contains(&min_p) {
            return Err(EngineError::invalid_input("sampling min-p is outside hard bounds"));
        }
        self.min_p = min_p;
        Ok(self)
    }

    /// Configures the standard sign-aware repetition penalty.
    pub fn with_repetition_penalty(mut self, penalty: f32) -> Result<Self, EngineError> {
        if !penalty.is_finite()
            || !(MIN_REPETITION_PENALTY..=MAX_REPETITION_PENALTY).contains(&penalty)
        {
            return Err(EngineError::invalid_input(
                "sampling repetition penalty is outside hard bounds",
            ));
        }
        self.repetition_penalty = penalty;
        Ok(self)
    }

    /// Sets the reproducible non-cryptographic random seed.
    #[must_use]
    pub const fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Restricts selection to these unique IDs; an empty list allows all IDs.
    pub fn with_allowed_token_ids(mut self, ids: &[u32]) -> Result<Self, EngineError> {
        self.allowed_token_ids = validate_id_set(ids, "allowed token IDs")?;
        validate_disjoint(&self.allowed_token_ids, &self.disallowed_token_ids)?;
        Ok(self)
    }

    /// Excludes these unique IDs.
    pub fn with_disallowed_token_ids(mut self, ids: &[u32]) -> Result<Self, EngineError> {
        self.disallowed_token_ids = validate_id_set(ids, "disallowed token IDs")?;
        validate_disjoint(&self.allowed_token_ids, &self.disallowed_token_ids)?;
        Ok(self)
    }

    /// Configures unique stop-token IDs. Stop tokens are still emitted.
    pub fn with_stop_token_ids(mut self, ids: &[u32]) -> Result<Self, EngineError> {
        self.stop_token_ids = validate_id_set(ids, "stop token IDs")?;
        Ok(self)
    }

    /// Configures the hard generated-token ceiling.
    pub fn with_maximum_generated_tokens(mut self, maximum: u32) -> Result<Self, EngineError> {
        if maximum == 0 || maximum > MAX_TOKEN_COUNT {
            return Err(EngineError::invalid_input(
                "sampling generated-token limit is outside hard bounds",
            ));
        }
        self.maximum_generated_tokens = maximum;
        Ok(self)
    }

    /// Returns temperature. Zero means deterministic greedy selection after constraints.
    pub const fn temperature(&self) -> f32 {
        self.temperature
    }

    /// Returns the optional top-k limit.
    pub const fn top_k(&self) -> Option<u32> {
        self.top_k
    }

    /// Returns the nucleus probability.
    pub const fn top_p(&self) -> f32 {
        self.top_p
    }

    /// Returns the relative minimum probability.
    pub const fn min_p(&self) -> f32 {
        self.min_p
    }

    /// Returns the repetition penalty.
    pub const fn repetition_penalty(&self) -> f32 {
        self.repetition_penalty
    }

    /// Returns the reproducibility seed.
    pub const fn seed(&self) -> u64 {
        self.seed
    }

    /// Returns the hard generated-token ceiling.
    pub const fn maximum_generated_tokens(&self) -> u32 {
        self.maximum_generated_tokens
    }

    /// Reports whether one generated token is a configured stop token.
    pub fn is_stop_token(&self, token_id: u32) -> bool {
        self.stop_token_ids.binary_search(&token_id).is_ok()
    }
}

/// Deterministic xorshift64* source for reproducible sampling only.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeterministicRng {
    state: u64,
}

impl DeterministicRng {
    /// Creates state; zero maps to a documented fixed nonzero constant.
    pub const fn new(seed: u64) -> Self {
        Self { state: if seed == 0 { 0x9e37_79b9_7f4a_7c15 } else { seed } }
    }

    /// Returns the next reproducible 64-bit word.
    pub fn next_u64(&mut self) -> u64 {
        let mut value = self.state;
        value ^= value >> 12;
        value ^= value << 25;
        value ^= value >> 27;
        self.state = value;
        value.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }

    fn unit_f64(&mut self) -> f64 {
        const SCALE: f64 = 1.0 / 9_007_199_254_740_992.0;
        let mantissa = self.next_u64() >> 11;
        #[allow(clippy::cast_precision_loss)]
        let value = mantissa as f64;
        value * SCALE
    }
}

#[derive(Debug, Clone, Copy)]
struct Candidate {
    token_id: u32,
    score: f64,
    weight: f64,
}

/// Request-owned deterministic sampler with explicit PRNG state.
#[derive(Debug)]
pub struct Sampler {
    config: SamplingConfig,
    random: DeterministicRng,
    pool: MemoryPool,
}

impl Sampler {
    /// Creates a sampler whose temporary tables are request-accounted.
    pub fn new(config: SamplingConfig, request_id: RequestId, pool: &MemoryPool) -> Self {
        let random = DeterministicRng::new(config.seed());
        Self { config, random, pool: pool.scoped(MemoryOwner::Sampling(request_id)) }
    }

    /// Returns the immutable validated settings.
    pub const fn config(&self) -> &SamplingConfig {
        &self.config
    }

    /// Selects from one rank-one F32 logit tensor and prior token history.
    pub fn sample(&mut self, logits: &Tensor, history: &[u32]) -> Result<u32, EngineError> {
        if logits.dtype() != DType::F32 || logits.shape().rank() != 1 {
            return Err(EngineError::invalid_input(
                "sampling requires one rank-one f32 logit tensor",
            ));
        }
        let count = usize::try_from(logits.shape().element_count())
            .map_err(|_| EngineError::resource_exhaustion("logit count is not addressable"))?;
        validate_logit_count(count)?;
        if history.len() > usize::try_from(MAX_TOKEN_COUNT).unwrap_or(usize::MAX) {
            return Err(EngineError::resource_exhaustion("sampling history exceeds token bound"));
        }
        validate_config_ids_for_vocabulary(&self.config, count)?;

        let table_bytes = count
            .checked_mul(size_of::<Candidate>())
            .and_then(|value| value.checked_add(count))
            .and_then(|value| u64::try_from(value).ok())
            .ok_or_else(|| EngineError::resource_exhaustion("sampling table size overflows"))?;
        let _reservation = self.pool.reserve(table_bytes)?;
        let repeated = repeated_table(history, count)?;
        let mut candidates = Vec::new();
        candidates.try_reserve_exact(count).map_err(|_| {
            EngineError::resource_exhaustion("sampling candidate allocation failed")
        })?;
        for (token_index, repetition_count) in repeated.iter().copied().enumerate() {
            let token_id = u32::try_from(token_index)
                .map_err(|_| EngineError::internal_invariant("token index exceeds u32"))?;
            if !eligible(&self.config, token_id) {
                continue;
            }
            let raw = logits.f32_value(token_index)?;
            if raw.is_nan() || raw == f32::INFINITY {
                return Err(EngineError::backend_failure(
                    "sampling received NaN or positive-infinite logits",
                ));
            }
            if raw == f32::NEG_INFINITY {
                continue;
            }
            let mut score = f64::from(raw);
            if repetition_count != 0
                && self.config.repetition_penalty().to_bits() != 1.0_f32.to_bits()
            {
                let penalty = f64::from(self.config.repetition_penalty());
                score = if score < 0.0 { score * penalty } else { score / penalty };
            }
            candidates.push(Candidate { token_id, score, weight: 0.0 });
        }
        if candidates.is_empty() {
            return Err(EngineError::invalid_input(
                "sampling constraints leave no selectable token",
            ));
        }
        sort_candidates(&mut candidates);
        if self.config.temperature() == 0.0 {
            return Ok(candidates[0].token_id);
        }
        self.sample_stochastic(&mut candidates)
    }

    fn sample_stochastic(&mut self, candidates: &mut Vec<Candidate>) -> Result<u32, EngineError> {
        if let Some(top_k) = self.config.top_k() {
            let keep = usize::try_from(top_k)
                .map_err(|_| EngineError::resource_exhaustion("top-k exceeds usize"))?
                .min(candidates.len());
            candidates.truncate(keep);
        }
        let temperature = f64::from(self.config.temperature());
        let maximum = candidates[0].score / temperature;
        for candidate in candidates.iter_mut() {
            candidate.weight = (candidate.score / temperature - maximum).exp();
            if !candidate.weight.is_finite() || candidate.weight < 0.0 {
                return Err(EngineError::backend_failure(
                    "sampling produced an invalid exponential weight",
                ));
            }
        }
        let min_weight = candidates[0].weight * f64::from(self.config.min_p());
        candidates.retain(|candidate| candidate.weight >= min_weight && candidate.weight > 0.0);
        if candidates.is_empty() {
            return Err(EngineError::backend_failure("min-p removed every sampling candidate"));
        }
        let total = weight_sum(candidates)?;
        if self.config.top_p() < 1.0 {
            let threshold = total * f64::from(self.config.top_p());
            let mut cumulative = 0.0_f64;
            let mut keep = 0_usize;
            for candidate in candidates.iter() {
                cumulative += candidate.weight;
                keep = keep.checked_add(1).ok_or_else(|| {
                    EngineError::resource_exhaustion("top-p candidate count overflows")
                })?;
                if cumulative >= threshold {
                    break;
                }
            }
            candidates.truncate(keep);
        }
        let filtered_total = weight_sum(candidates)?;
        let target = self.random.unit_f64() * filtered_total;
        let mut cumulative = 0.0_f64;
        for candidate in candidates.iter() {
            cumulative += candidate.weight;
            if target < cumulative {
                return Ok(candidate.token_id);
            }
        }
        candidates
            .last()
            .map(|candidate| candidate.token_id)
            .ok_or_else(|| EngineError::internal_invariant("sampling candidate table disappeared"))
    }
}

/// Compatibility greedy selection from a bounded slice.
pub fn select_next_token(logits: &[f32], config: &SamplingConfig) -> Result<u32, EngineError> {
    validate_logit_count(logits.len())?;
    if config.temperature() != 0.0
        || config.top_k().is_some()
        || config.top_p().to_bits() != 1.0_f32.to_bits()
        || config.min_p().to_bits() != 0.0_f32.to_bits()
        || config.repetition_penalty().to_bits() != 1.0_f32.to_bits()
        || !config.allowed_token_ids.is_empty()
        || !config.disallowed_token_ids.is_empty()
    {
        return Err(EngineError::invalid_input(
            "slice compatibility selection accepts neutral greedy settings only",
        ));
    }
    let mut selected = None;
    for (index, value) in logits.iter().copied().enumerate() {
        if !value.is_finite() {
            return Err(EngineError::backend_failure("sampling received a non-finite logit"));
        }
        if selected.is_none_or(|(_, best)| value > best) {
            selected = Some((index, value));
        }
    }
    let selected =
        selected.ok_or_else(|| EngineError::internal_invariant("validated logits became empty"))?;
    u32::try_from(selected.0)
        .map_err(|_| EngineError::internal_invariant("selected token ID exceeds u32"))
}

/// Compatibility greedy selection directly from one tensor.
pub fn select_next_token_tensor(
    logits: &Tensor,
    config: &SamplingConfig,
) -> Result<u32, EngineError> {
    if logits.dtype() != DType::F32 || logits.shape().rank() != 1 {
        return Err(EngineError::invalid_input("sampling requires one rank-one f32 logit tensor"));
    }
    if config != &SamplingConfig::greedy() {
        return Err(EngineError::invalid_input(
            "tensor compatibility selection accepts neutral greedy settings only",
        ));
    }
    let count = usize::try_from(logits.shape().element_count())
        .map_err(|_| EngineError::resource_exhaustion("logit count is not addressable"))?;
    validate_logit_count(count)?;
    let mut selected_index = 0_usize;
    let mut selected_value = logits.f32_value(0)?;
    if !selected_value.is_finite() {
        return Err(EngineError::backend_failure("sampling received a non-finite logit"));
    }
    for index in 1..count {
        let value = logits.f32_value(index)?;
        if !value.is_finite() {
            return Err(EngineError::backend_failure("sampling received a non-finite logit"));
        }
        if value > selected_value {
            selected_index = index;
            selected_value = value;
        }
    }
    u32::try_from(selected_index)
        .map_err(|_| EngineError::internal_invariant("selected token ID exceeds u32"))
}

fn validate_temperature(temperature: f32) -> Result<(), EngineError> {
    if !temperature.is_finite() || !(0.0..=MAX_TEMPERATURE).contains(&temperature) {
        return Err(EngineError::invalid_input("sampling temperature is outside hard bounds"));
    }
    Ok(())
}

fn validate_logit_count(count: usize) -> Result<(), EngineError> {
    let count =
        u32::try_from(count).map_err(|_| EngineError::invalid_input("logit count exceeds u32"))?;
    if count == 0 || count > MAX_VOCABULARY_SIZE {
        return Err(EngineError::invalid_input("logit count is outside sampling hard bounds"));
    }
    Ok(())
}

fn validate_id_set(ids: &[u32], label: &'static str) -> Result<Box<[u32]>, EngineError> {
    if ids.len() > usize::try_from(MAX_VOCABULARY_SIZE).unwrap_or(usize::MAX) {
        return Err(EngineError::invalid_input(label));
    }
    let mut values = Vec::new();
    values
        .try_reserve_exact(ids.len())
        .map_err(|_| EngineError::resource_exhaustion("sampling token set allocation failed"))?;
    values.extend_from_slice(ids);
    values.sort_unstable();
    if values.iter().any(|id| *id >= MAX_VOCABULARY_SIZE)
        || values.windows(2).any(|window| window[0] == window[1])
    {
        return Err(EngineError::invalid_input(label));
    }
    Ok(values.into_boxed_slice())
}

fn validate_disjoint(allowed: &[u32], disallowed: &[u32]) -> Result<(), EngineError> {
    if allowed.iter().any(|id| disallowed.binary_search(id).is_ok()) {
        return Err(EngineError::invalid_input("allowed and disallowed token sets overlap"));
    }
    Ok(())
}

fn validate_config_ids_for_vocabulary(
    config: &SamplingConfig,
    vocabulary_size: usize,
) -> Result<(), EngineError> {
    let vocabulary_size = u32::try_from(vocabulary_size)
        .map_err(|_| EngineError::resource_exhaustion("vocabulary size exceeds u32"))?;
    if config
        .allowed_token_ids
        .iter()
        .chain(config.disallowed_token_ids.iter())
        .chain(config.stop_token_ids.iter())
        .any(|id| *id >= vocabulary_size)
    {
        return Err(EngineError::invalid_input(
            "sampling token constraint exceeds model vocabulary",
        ));
    }
    if config.top_k().is_some_and(|top_k| top_k > vocabulary_size) {
        return Err(EngineError::invalid_input("sampling top-k exceeds model vocabulary"));
    }
    Ok(())
}

fn repeated_table(history: &[u32], vocabulary_size: usize) -> Result<Vec<u8>, EngineError> {
    let mut repeated = Vec::new();
    repeated
        .try_reserve_exact(vocabulary_size)
        .map_err(|_| EngineError::resource_exhaustion("repetition table allocation failed"))?;
    repeated.resize(vocabulary_size, 0);
    for token in history {
        let index = usize::try_from(*token)
            .map_err(|_| EngineError::invalid_input("history token exceeds usize"))?;
        let slot = repeated
            .get_mut(index)
            .ok_or_else(|| EngineError::invalid_input("history token exceeds vocabulary"))?;
        *slot = 1;
    }
    Ok(repeated)
}

fn eligible(config: &SamplingConfig, token_id: u32) -> bool {
    (config.allowed_token_ids.is_empty()
        || config.allowed_token_ids.binary_search(&token_id).is_ok())
        && config.disallowed_token_ids.binary_search(&token_id).is_err()
}

fn sort_candidates(candidates: &mut [Candidate]) {
    candidates.sort_by(|left, right| {
        right
            .score
            .partial_cmp(&left.score)
            .unwrap_or(Ordering::Equal)
            .then_with(|| left.token_id.cmp(&right.token_id))
    });
}

fn weight_sum(candidates: &[Candidate]) -> Result<f64, EngineError> {
    let total = candidates.iter().try_fold(0.0_f64, |sum, candidate| {
        let next = sum + candidate.weight;
        if next.is_finite() {
            Ok(next)
        } else {
            Err(EngineError::backend_failure("sampling probability sum is invalid"))
        }
    })?;
    if total <= 0.0 {
        return Err(EngineError::backend_failure("sampling probability distribution is empty"));
    }
    Ok(total)
}

#[cfg(test)]
mod tests {
    use super::*;
    use personalai_engine_types::TensorShape;

    fn tensor(values: &[f32], pool: &MemoryPool) -> Tensor {
        Tensor::from_f32_values(TensorShape::new([values.len() as u64]).unwrap(), values, pool)
            .unwrap()
    }

    #[test]
    fn greedy_temperature_zero_and_ties_are_exact() {
        let config = SamplingConfig::greedy();
        assert_eq!(select_next_token(&[0.5, 2.0, 2.0, -1.0], &config).unwrap(), 1);
        let pool = MemoryPool::new(4096).unwrap();
        let logits = tensor(&[0.5, 2.0, 2.0, -1.0], &pool);
        let mut sampler = Sampler::new(config, RequestId::new(1).unwrap(), &pool);
        assert_eq!(sampler.sample(&logits, &[]).unwrap(), 1);
    }

    #[test]
    fn seeded_sampling_is_reproducible_and_seed_sensitive() {
        let pool = MemoryPool::new(1 << 20).unwrap();
        let logits = tensor(&[1.0, 1.1, 1.2, 1.3], &pool);
        let config = SamplingConfig::new(1.0)
            .unwrap()
            .with_top_k(Some(4))
            .unwrap()
            .with_top_p(0.95)
            .unwrap()
            .with_min_p(0.1)
            .unwrap()
            .with_seed(42);
        let mut first = Sampler::new(config.clone(), RequestId::new(1).unwrap(), &pool);
        let mut second = Sampler::new(config, RequestId::new(2).unwrap(), &pool);
        let first_sequence =
            (0..16).map(|_| first.sample(&logits, &[]).unwrap()).collect::<Vec<_>>();
        let second_sequence =
            (0..16).map(|_| second.sample(&logits, &[]).unwrap()).collect::<Vec<_>>();
        assert_eq!(first_sequence, second_sequence);
        assert!(first_sequence.iter().all(|token| *token < 4));
    }

    #[test]
    fn constraints_repetition_and_stop_policy_are_enforced() {
        let pool = MemoryPool::new(1 << 20).unwrap();
        let logits = tensor(&[10.0, 9.0, 8.0, 7.0], &pool);
        let config = SamplingConfig::greedy()
            .with_allowed_token_ids(&[2, 3])
            .unwrap()
            .with_disallowed_token_ids(&[1])
            .unwrap()
            .with_repetition_penalty(100.0)
            .unwrap()
            .with_stop_token_ids(&[3])
            .unwrap()
            .with_maximum_generated_tokens(7)
            .unwrap();
        let mut sampler = Sampler::new(config.clone(), RequestId::new(1).unwrap(), &pool);
        assert_eq!(sampler.sample(&logits, &[2]).unwrap(), 3);
        assert!(config.is_stop_token(3));
        assert_eq!(config.maximum_generated_tokens(), 7);
    }

    #[test]
    fn invalid_configs_distributions_and_tokens_fail_closed() {
        assert!(SamplingConfig::new(f32::NAN).is_err());
        assert!(SamplingConfig::new(1.0).unwrap().with_top_k(Some(0)).is_err());
        assert!(SamplingConfig::new(1.0).unwrap().with_top_p(0.0).is_err());
        assert!(SamplingConfig::new(1.0).unwrap().with_min_p(1.1).is_err());
        assert!(SamplingConfig::new(1.0).unwrap().with_repetition_penalty(0.0).is_err());
        assert!(SamplingConfig::greedy().with_allowed_token_ids(&[1, 1]).is_err());
        assert!(
            SamplingConfig::greedy()
                .with_allowed_token_ids(&[1])
                .unwrap()
                .with_disallowed_token_ids(&[1])
                .is_err()
        );

        let pool = MemoryPool::new(1 << 20).unwrap();
        for values in [[0.0, f32::NAN], [0.0, f32::INFINITY], [f32::NEG_INFINITY; 2]] {
            let logits = tensor(&values, &pool);
            let mut sampler =
                Sampler::new(SamplingConfig::greedy(), RequestId::new(1).unwrap(), &pool);
            assert!(sampler.sample(&logits, &[]).is_err());
        }
        let logits = tensor(&[1.0, 2.0], &pool);
        let mut sampler = Sampler::new(
            SamplingConfig::greedy().with_allowed_token_ids(&[3]).unwrap(),
            RequestId::new(2).unwrap(),
            &pool,
        );
        assert!(sampler.sample(&logits, &[]).is_err());
    }

    #[test]
    fn tensor_and_slice_compatibility_greedy_match() {
        let pool = MemoryPool::new(64).unwrap();
        let logits = tensor(&[0.5, 2.0, 2.0, -1.0], &pool);
        assert_eq!(
            select_next_token_tensor(&logits, &SamplingConfig::greedy()).unwrap(),
            select_next_token(&[0.5, 2.0, 2.0, -1.0], &SamplingConfig::greedy()).unwrap()
        );
    }
}
