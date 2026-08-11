//! Bounded deterministic sampling primitives.
//!
//! Only greedy selection is implemented. Stochastic sampling requires a
//! reviewed first-party random source and explicit reproducibility contract.

use personalai_engine_tensor::Tensor;
use personalai_engine_types::{DType, EngineError, MAX_VOCABULARY_SIZE};
/// Maximum accepted temperature.
pub const MAX_TEMPERATURE: f32 = 2.0;

/// Validated token sampling settings.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SamplingConfig {
    temperature: f32,
}

impl SamplingConfig {
    /// Creates sampling settings after finite range validation.
    pub fn new(temperature: f32) -> Result<Self, EngineError> {
        if !temperature.is_finite() || !(0.0..=MAX_TEMPERATURE).contains(&temperature) {
            return Err(EngineError::invalid_input("sampling temperature is outside hard bounds"));
        }
        Ok(Self { temperature })
    }

    /// Returns deterministic greedy settings.
    pub const fn greedy() -> Self {
        Self { temperature: 0.0 }
    }

    /// Returns the configured temperature.
    pub const fn temperature(self) -> f32 {
        self.temperature
    }
}

/// Selects the next token from bounded logits.
///
/// Equal logits resolve to the lowest token ID. Non-finite input is rejected to
/// avoid platform-dependent ordering. Nonzero temperature is explicitly not
/// implemented in the foundation stage.
pub fn select_next_token(logits: &[f32], config: SamplingConfig) -> Result<u32, EngineError> {
    let logit_count = u32::try_from(logits.len())
        .map_err(|_| EngineError::invalid_input("logit count exceeds u32"))?;
    if logits.is_empty() || logit_count > MAX_VOCABULARY_SIZE {
        return Err(EngineError::invalid_input("logit count is outside sampling hard bounds"));
    }
    if config.temperature() != 0.0 {
        return Err(EngineError::not_implemented("stochastic token sampling is not implemented"));
    }
    if logits.iter().any(|value| !value.is_finite()) {
        return Err(EngineError::backend_failure("sampling received a non-finite logit"));
    }

    let mut selected_index = 0_usize;
    let mut selected_value = logits[0];
    for (index, value) in logits.iter().copied().enumerate().skip(1) {
        if value > selected_value {
            selected_index = index;
            selected_value = value;
        }
    }
    u32::try_from(selected_index)
        .map_err(|_| EngineError::internal_invariant("selected token ID exceeds u32"))
}

/// Selects directly from one rank-one accounted f32 logit tensor.
pub fn select_next_token_tensor(
    logits: &Tensor,
    config: SamplingConfig,
) -> Result<u32, EngineError> {
    if logits.dtype() != DType::F32 || logits.shape().rank() != 1 {
        return Err(EngineError::invalid_input("sampling requires one rank-one f32 logit tensor"));
    }
    if config.temperature() != 0.0 {
        return Err(EngineError::not_implemented("stochastic token sampling is not implemented"));
    }
    let count = logits.shape().element_count();
    if count == 0 || count > u64::from(MAX_VOCABULARY_SIZE) {
        return Err(EngineError::invalid_input("logit count is outside sampling hard bounds"));
    }
    let count = usize::try_from(count)
        .map_err(|_| EngineError::resource_exhaustion("logit count is not addressable"))?;
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

#[cfg(test)]
mod tests {
    use super::*;
    use personalai_engine_memory::MemoryPool;
    use personalai_engine_types::TensorShape;

    #[test]
    fn greedy_selection_is_deterministic_and_prefers_lowest_tie() {
        let token = select_next_token(&[0.5, 2.0, 2.0, -1.0], SamplingConfig::greedy())
            .expect("finite logits must select");
        assert_eq!(token, 1);
    }

    #[test]
    fn rejects_empty_and_non_finite_logits() {
        assert!(select_next_token(&[], SamplingConfig::greedy()).is_err());
        assert!(select_next_token(&[0.0, f32::NAN], SamplingConfig::greedy()).is_err());
    }

    #[test]
    fn stochastic_sampling_is_explicitly_unimplemented() {
        let config = SamplingConfig::new(0.7).expect("temperature must validate");
        assert!(matches!(
            select_next_token(&[1.0, 2.0], config),
            Err(EngineError::NotImplemented(_))
        ));
    }

    #[test]
    fn tensor_sampling_matches_slice_sampling_without_a_decoded_copy() {
        let pool = MemoryPool::new(16).expect("pool must validate");
        let logits = Tensor::from_f32_values(
            TensorShape::new([4]).expect("shape must validate"),
            &[0.5, 2.0, 2.0, -1.0],
            &pool,
        )
        .expect("logits must construct");
        assert_eq!(
            select_next_token_tensor(&logits, SamplingConfig::greedy())
                .expect("tensor logits must select"),
            1
        );
    }
}
