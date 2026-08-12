//! Checked KV-cache planning and contiguous CPU sequence storage.

use personalai_engine_memory::{MemoryPool, MemoryReservation};
use personalai_engine_types::{DType, EngineError, MAX_CONTEXT_TOKENS, ResourceBudget, SequenceId};

mod contiguous;

pub use contiguous::{
    ContiguousKvCache, KvCacheMetrics, KvCacheStatus, KvTokenWriter, SequenceHandle, SequenceStatus,
};

/// Maximum transformer layer count accepted by cache planning.
pub const MAX_LAYER_COUNT: u32 = 512;
/// Maximum key/value head count accepted by cache planning.
pub const MAX_KV_HEAD_COUNT: u32 = 512;
/// Maximum elements per attention head accepted by cache planning.
pub const MAX_HEAD_DIMENSION: u32 = 8_192;

/// Validated byte layout for one sequence's key/value cache.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KvCachePlan {
    layer_count: u32,
    kv_head_count: u32,
    head_dimension: u32,
    token_capacity: u32,
    dtype: DType,
    bytes: u64,
}

impl KvCachePlan {
    /// Calculates cache bytes with checked arithmetic and request budgets.
    pub fn new(
        layer_count: u32,
        kv_head_count: u32,
        head_dimension: u32,
        token_capacity: u32,
        dtype: DType,
        budget: &ResourceBudget,
    ) -> Result<Self, EngineError> {
        if layer_count == 0 || layer_count > MAX_LAYER_COUNT {
            return Err(EngineError::invalid_input("KV-cache layer count is outside hard bounds"));
        }
        if kv_head_count == 0 || kv_head_count > MAX_KV_HEAD_COUNT {
            return Err(EngineError::invalid_input("KV-cache head count is outside hard bounds"));
        }
        if head_dimension == 0 || head_dimension > MAX_HEAD_DIMENSION {
            return Err(EngineError::invalid_input(
                "KV-cache head dimension is outside hard bounds",
            ));
        }
        if token_capacity == 0 || token_capacity > MAX_CONTEXT_TOKENS {
            return Err(EngineError::invalid_input(
                "KV-cache token capacity is outside hard bounds",
            ));
        }
        if !matches!(dtype, DType::F32 | DType::F16 | DType::BFloat16) {
            return Err(EngineError::not_implemented(
                "KV-cache storage supports only F32, F16, and BF16",
            ));
        }

        let bytes = 2_u64
            .checked_mul(u64::from(layer_count))
            .and_then(|value| value.checked_mul(u64::from(kv_head_count)))
            .and_then(|value| value.checked_mul(u64::from(head_dimension)))
            .and_then(|value| value.checked_mul(u64::from(token_capacity)))
            .and_then(|value| value.checked_mul(dtype.byte_width()))
            .ok_or_else(|| EngineError::resource_exhaustion("KV-cache byte length overflows"))?;
        if bytes > budget.max_kv_cache_bytes() || bytes > budget.max_memory_bytes() {
            return Err(EngineError::resource_exhaustion(
                "KV-cache plan exceeds the request resource budget",
            ));
        }
        Ok(Self { layer_count, kv_head_count, head_dimension, token_capacity, dtype, bytes })
    }

    /// Returns the transformer layer count.
    pub const fn layer_count(&self) -> u32 {
        self.layer_count
    }

    /// Returns the key/value head count.
    pub const fn kv_head_count(&self) -> u32 {
        self.kv_head_count
    }

    /// Returns the per-head element dimension.
    pub const fn head_dimension(&self) -> u32 {
        self.head_dimension
    }

    /// Returns the sequence token capacity.
    pub const fn token_capacity(&self) -> u32 {
        self.token_capacity
    }

    /// Returns the cache data type.
    pub const fn dtype(&self) -> DType {
        self.dtype
    }

    /// Returns the total planned bytes for keys and values.
    pub const fn bytes(&self) -> u64 {
        self.bytes
    }

    /// Reserves the plan's bytes in a checked accounting pool.
    pub fn reserve(self, pool: &MemoryPool) -> Result<KvCacheAllocation, EngineError> {
        let reservation = pool.reserve(self.bytes)?;
        Ok(KvCacheAllocation { plan: self, reservation })
    }
}

/// Accounted KV-cache allocation without backend tensor storage.
#[derive(Debug)]
pub struct KvCacheAllocation {
    plan: KvCachePlan,
    reservation: MemoryReservation,
}

impl KvCacheAllocation {
    /// Returns the validated cache layout.
    pub const fn plan(&self) -> &KvCachePlan {
        &self.plan
    }

    /// Returns the accounted reservation byte count.
    pub const fn reserved_bytes(&self) -> u64 {
        self.reservation.bytes()
    }
}

/// Bounded logical occupancy for one generated sequence.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SequenceCacheState {
    sequence_id: SequenceId,
    token_capacity: u32,
    token_count: u32,
}

impl SequenceCacheState {
    /// Creates an empty logical cache state for a sequence.
    pub fn new(sequence_id: SequenceId, token_capacity: u32) -> Result<Self, EngineError> {
        if token_capacity == 0 || token_capacity > MAX_CONTEXT_TOKENS {
            return Err(EngineError::invalid_input(
                "sequence cache capacity is outside hard bounds",
            ));
        }
        Ok(Self { sequence_id, token_capacity, token_count: 0 })
    }

    /// Advances logical occupancy without writing any tensor data.
    pub fn append_tokens(&mut self, additional_tokens: u32) -> Result<(), EngineError> {
        if additional_tokens == 0 {
            return Err(EngineError::invalid_input(
                "sequence cache append must include at least one token",
            ));
        }
        let next = self.token_count.checked_add(additional_tokens).ok_or_else(|| {
            EngineError::resource_exhaustion("sequence cache token count overflows")
        })?;
        if next > self.token_capacity {
            return Err(EngineError::resource_exhaustion("sequence cache capacity is exhausted"));
        }
        self.token_count = next;
        Ok(())
    }

    /// Returns the sequence identifier.
    pub const fn sequence_id(&self) -> SequenceId {
        self.sequence_id
    }

    /// Returns the current logical token occupancy.
    pub const fn token_count(&self) -> u32 {
        self.token_count
    }

    /// Returns the maximum logical token occupancy.
    pub const fn token_capacity(&self) -> u32 {
        self.token_capacity
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn budget() -> ResourceBudget {
        ResourceBudget::new(1 << 30, 1 << 29, 2).expect("budget must validate")
    }

    #[test]
    fn cache_plan_uses_checked_exact_arithmetic() {
        let plan = KvCachePlan::new(2, 4, 8, 16, DType::F32, &budget())
            .expect("small cache plan must validate");
        assert_eq!(plan.bytes(), 2 * 2 * 4 * 8 * 16 * 4);
        let pool = MemoryPool::new(plan.bytes()).expect("pool must validate");
        let allocation = plan.reserve(&pool).expect("plan must reserve");
        assert_eq!(allocation.reserved_bytes(), pool.limit_bytes());
    }

    #[test]
    fn plan_rejects_budget_exhaustion() {
        let tiny = ResourceBudget::new(1_024, 512, 1).expect("budget must validate");
        assert!(KvCachePlan::new(8, 8, 64, 64, DType::F32, &tiny).is_err());
    }

    #[test]
    fn sequence_occupancy_is_bounded() {
        let mut state = SequenceCacheState::new(SequenceId::new(1).expect("id must validate"), 4)
            .expect("state must validate");
        state.append_tokens(3).expect("append must fit");
        assert_eq!(state.token_count(), 3);
        assert!(state.append_tokens(2).is_err());
    }
}
