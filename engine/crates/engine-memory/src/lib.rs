//! Checked process-local memory accounting.
//!
//! This crate accounts tensor and cache bytes only. It is unrelated to the Go
//! platform's personal memory, conversation storage, or retrieval authority.

use personalai_engine_types::{
    BackendId, EngineError, MAX_RESOURCE_MEMORY_BYTES, ModelId, RequestId, ResourceBudget,
    SequenceId,
};
use std::sync::{Arc, Mutex};

mod runtime;

pub use runtime::{
    BackendResources, CpuMemoryBackend, DeviceBuffer, DeviceMemory, DeviceMemoryKind, HostBuffer,
    MemoryAdmission, MemoryRequirements, ScratchArena, ScratchBufferId, ScratchMemoryManager,
    ScratchMetrics,
};

const OWNER_CATEGORY_COUNT: usize = 16;

/// Explicit owner for one engine inference allocation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MemoryOwner {
    /// Process-wide engine bookkeeping.
    Engine,
    /// Verified immutable model weight mappings or copies.
    ModelWeights(ModelId),
    /// Parsed immutable model metadata.
    ModelMetadata(ModelId),
    /// Immutable tokenizer tables.
    Tokenizer(ModelId),
    /// Other immutable per-model state.
    ModelState(ModelId),
    /// Bounded layer-streaming and backend-prefetch staging owned by one model.
    PlacementStaging(ModelId),
    /// Immutable routed-expert payloads cached for one verified model.
    SparseExpertCache(ModelId),
    /// Durable staging and safety capacity held for sparse-expert execution.
    SparseExpertStaging(ModelId),
    /// Immutable reusable prefix K/V state owned by one verified model.
    ReusableKvCache(ModelId),
    /// Request prompt/token state.
    RequestPrompt(RequestId),
    /// Sequence-owned key/value state.
    SequenceKv(SequenceId),
    /// Temporary request operation storage.
    RequestScratch(RequestId),
    /// Request output logits.
    OutputLogits(RequestId),
    /// Request sampling state.
    Sampling(RequestId),
    /// Request streaming decoder state.
    StreamingDecoder(RequestId),
    /// Backend-owned resources.
    Backend(BackendId),
}

impl MemoryOwner {
    const fn category_index(&self) -> usize {
        match self {
            Self::Engine => 0,
            Self::ModelWeights(_) => 1,
            Self::ModelMetadata(_) => 2,
            Self::Tokenizer(_) => 3,
            Self::ModelState(_) => 4,
            Self::PlacementStaging(_) => 5,
            Self::SparseExpertCache(_) => 6,
            Self::SparseExpertStaging(_) => 7,
            Self::ReusableKvCache(_) => 8,
            Self::RequestPrompt(_) => 9,
            Self::SequenceKv(_) => 10,
            Self::RequestScratch(_) => 11,
            Self::OutputLogits(_) => 12,
            Self::Sampling(_) => 13,
            Self::StreamingDecoder(_) => 14,
            Self::Backend(_) => 15,
        }
    }
}

#[derive(Debug)]
struct MemoryState {
    used_bytes: u64,
    peak_bytes: u64,
    allocation_count: u64,
    by_category: [u64; OWNER_CATEGORY_COUNT],
}

/// Thread-safe bounded memory accounting pool.
#[derive(Debug, Clone)]
pub struct MemoryPool {
    limit_bytes: u64,
    default_owner: MemoryOwner,
    state: Arc<Mutex<MemoryState>>,
}

impl MemoryPool {
    /// Creates a memory pool with a positive hard-bounded limit.
    pub fn new(limit_bytes: u64) -> Result<Self, EngineError> {
        Self::new_scoped(limit_bytes, MemoryOwner::Engine)
    }

    /// Creates a pool whose ordinary reservations have one explicit owner.
    pub fn new_scoped(limit_bytes: u64, default_owner: MemoryOwner) -> Result<Self, EngineError> {
        if limit_bytes == 0 || limit_bytes > MAX_RESOURCE_MEMORY_BYTES {
            return Err(EngineError::invalid_input("memory pool limit is outside hard bounds"));
        }
        Ok(Self {
            limit_bytes,
            default_owner,
            state: Arc::new(Mutex::new(MemoryState {
                used_bytes: 0,
                peak_bytes: 0,
                allocation_count: 0,
                by_category: [0; OWNER_CATEGORY_COUNT],
            })),
        })
    }

    /// Creates an allocation pool from the request's memory budget.
    pub fn from_budget(budget: &ResourceBudget) -> Result<Self, EngineError> {
        Self::new(budget.max_memory_bytes())
    }

    /// Creates a view of this exact pool with a different default owner.
    ///
    /// Limits and counters remain shared; only subsequent ordinary reservations
    /// receive the supplied ownership label.
    #[must_use]
    pub fn scoped(&self, default_owner: MemoryOwner) -> Self {
        Self { limit_bytes: self.limit_bytes, default_owner, state: Arc::clone(&self.state) }
    }

    /// Reserves positive bytes and returns an automatically released guard.
    pub fn reserve(&self, bytes: u64) -> Result<MemoryReservation, EngineError> {
        self.reserve_for(bytes, self.default_owner.clone())
    }

    /// Reserves positive bytes under an explicit inference-state owner.
    pub fn reserve_for(
        &self,
        bytes: u64,
        owner: MemoryOwner,
    ) -> Result<MemoryReservation, EngineError> {
        self.reserve_for_preserving(bytes, 0, owner)
    }

    /// Reserves positive bytes while preserving unreserved capacity at admission.
    ///
    /// The availability check is atomic with the reservation: current usage,
    /// `bytes`, and `preserved_headroom_bytes` must fit beneath the pool limit
    /// while the accounting lock is held. Only `bytes` are charged to the
    /// returned reservation; headroom remains available to other owners. This
    /// is a point-in-time admission check, not a durable partition. Callers
    /// requiring a lifetime guarantee must retain an explicit reservation for
    /// that capacity.
    pub fn reserve_for_preserving(
        &self,
        bytes: u64,
        preserved_headroom_bytes: u64,
        owner: MemoryOwner,
    ) -> Result<MemoryReservation, EngineError> {
        if bytes == 0 {
            return Err(EngineError::invalid_input(
                "memory reservation must contain at least one byte",
            ));
        }
        let mut state = self
            .state
            .lock()
            .map_err(|_| EngineError::internal_invariant("memory accounting lock is poisoned"))?;
        let next = state.used_bytes.checked_add(bytes).ok_or_else(|| {
            EngineError::resource_exhaustion("memory reservation accounting overflows")
        })?;
        let required = next.checked_add(preserved_headroom_bytes).ok_or_else(|| {
            EngineError::resource_exhaustion("memory reservation headroom accounting overflows")
        })?;
        if required > self.limit_bytes {
            let message = if preserved_headroom_bytes == 0 {
                "memory reservation exceeds the configured pool"
            } else {
                "memory reservation would consume preserved pool headroom"
            };
            return Err(EngineError::resource_exhaustion(message));
        }
        let allocation_count = state.allocation_count.checked_add(1).ok_or_else(|| {
            EngineError::resource_exhaustion("memory allocation counter overflows")
        })?;
        let category = owner.category_index();
        let category_bytes = state.by_category[category]
            .checked_add(bytes)
            .ok_or_else(|| EngineError::resource_exhaustion("memory owner accounting overflows"))?;
        state.used_bytes = next;
        state.peak_bytes = state.peak_bytes.max(next);
        state.allocation_count = allocation_count;
        state.by_category[category] = category_bytes;
        drop(state);
        Ok(MemoryReservation { bytes, owner, state: Arc::clone(&self.state), released: false })
    }

    /// Extends one owned reservation after an allocator reports spare capacity.
    pub fn grow_reservation(
        &self,
        reservation: &mut MemoryReservation,
        additional_bytes: u64,
    ) -> Result<(), EngineError> {
        self.grow_reservation_preserving(reservation, additional_bytes, 0)
    }

    /// Extends one owned reservation while preserving unreserved capacity at admission.
    ///
    /// As with [`Self::reserve_for_preserving`], the check and accounting
    /// update occur under the same pool lock and only `additional_bytes` are
    /// charged to the reservation.
    pub fn grow_reservation_preserving(
        &self,
        reservation: &mut MemoryReservation,
        additional_bytes: u64,
        preserved_headroom_bytes: u64,
    ) -> Result<(), EngineError> {
        if additional_bytes == 0 {
            return Ok(());
        }
        if !self.owns(reservation) {
            return Err(EngineError::invalid_input(
                "memory reservation belongs to a different pool",
            ));
        }
        let reservation_bytes = reservation
            .bytes
            .checked_add(additional_bytes)
            .ok_or_else(|| EngineError::resource_exhaustion("memory reservation size overflows"))?;
        let mut state = self
            .state
            .lock()
            .map_err(|_| EngineError::internal_invariant("memory accounting lock is poisoned"))?;
        let used_bytes = state.used_bytes.checked_add(additional_bytes).ok_or_else(|| {
            EngineError::resource_exhaustion("memory reservation accounting overflows")
        })?;
        let required = used_bytes.checked_add(preserved_headroom_bytes).ok_or_else(|| {
            EngineError::resource_exhaustion("memory reservation headroom accounting overflows")
        })?;
        if required > self.limit_bytes {
            let message = if preserved_headroom_bytes == 0 {
                "memory reservation exceeds the configured pool"
            } else {
                "memory reservation would consume preserved pool headroom"
            };
            return Err(EngineError::resource_exhaustion(message));
        }
        let category = reservation.owner.category_index();
        let category_bytes = state.by_category[category]
            .checked_add(additional_bytes)
            .ok_or_else(|| EngineError::resource_exhaustion("memory owner accounting overflows"))?;
        state.used_bytes = used_bytes;
        state.peak_bytes = state.peak_bytes.max(used_bytes);
        state.by_category[category] = category_bytes;
        reservation.bytes = reservation_bytes;
        Ok(())
    }

    /// Shrinks one owned reservation after an allocator releases spare capacity.
    pub fn shrink_reservation(
        &self,
        reservation: &mut MemoryReservation,
        released_bytes: u64,
    ) -> Result<(), EngineError> {
        if released_bytes == 0 {
            return Ok(());
        }
        if !self.owns(reservation) || released_bytes >= reservation.bytes {
            return Err(EngineError::invalid_input(
                "memory reservation shrink is invalid for this pool",
            ));
        }
        let reservation_bytes = reservation.bytes.checked_sub(released_bytes).ok_or_else(|| {
            EngineError::internal_invariant("memory reservation shrink underflows")
        })?;
        let mut state = self
            .state
            .lock()
            .map_err(|_| EngineError::internal_invariant("memory accounting lock is poisoned"))?;
        let used_bytes = state.used_bytes.checked_sub(released_bytes).ok_or_else(|| {
            EngineError::internal_invariant("memory accounting shrink underflows")
        })?;
        let category = reservation.owner.category_index();
        let category_bytes = state.by_category[category]
            .checked_sub(released_bytes)
            .ok_or_else(|| EngineError::internal_invariant("memory owner shrink underflows"))?;
        state.used_bytes = used_bytes;
        state.by_category[category] = category_bytes;
        reservation.bytes = reservation_bytes;
        Ok(())
    }

    /// Transfers an existing reservation to another explicit owner.
    pub fn reassign_reservation(
        &self,
        reservation: &mut MemoryReservation,
        owner: MemoryOwner,
    ) -> Result<(), EngineError> {
        if !self.owns(reservation) {
            return Err(EngineError::invalid_input(
                "memory reservation belongs to a different pool",
            ));
        }
        let mut state = self
            .state
            .lock()
            .map_err(|_| EngineError::internal_invariant("memory accounting lock is poisoned"))?;
        let old_category = reservation.owner.category_index();
        let new_category = owner.category_index();
        if old_category != new_category {
            let old_bytes =
                state.by_category[old_category].checked_sub(reservation.bytes).ok_or_else(
                    || EngineError::internal_invariant("memory owner transfer underflows"),
                )?;
            let new_bytes =
                state.by_category[new_category].checked_add(reservation.bytes).ok_or_else(
                    || EngineError::resource_exhaustion("memory owner transfer overflows"),
                )?;
            state.by_category[old_category] = old_bytes;
            state.by_category[new_category] = new_bytes;
        }
        reservation.owner = owner;
        Ok(())
    }

    /// Returns the configured pool limit.
    pub const fn limit_bytes(&self) -> u64 {
        self.limit_bytes
    }

    /// Returns the currently reserved byte count.
    pub fn used_bytes(&self) -> Result<u64, EngineError> {
        let state = self
            .state
            .lock()
            .map_err(|_| EngineError::internal_invariant("memory accounting lock is poisoned"))?;
        Ok(state.used_bytes)
    }

    /// Returns the unreserved byte count.
    pub fn available_bytes(&self) -> Result<u64, EngineError> {
        self.limit_bytes
            .checked_sub(self.used_bytes()?)
            .ok_or_else(|| EngineError::internal_invariant("memory accounting exceeds its limit"))
    }

    /// Returns whether a reservation is owned by this pool.
    pub fn owns(&self, reservation: &MemoryReservation) -> bool {
        Arc::ptr_eq(&self.state, &reservation.state)
    }

    /// Returns an atomic snapshot used by admission control and metrics.
    pub fn statistics(&self) -> Result<MemoryStatistics, EngineError> {
        let state = self
            .state
            .lock()
            .map_err(|_| EngineError::internal_invariant("memory accounting lock is poisoned"))?;
        Ok(MemoryStatistics {
            limit_bytes: self.limit_bytes,
            reserved_bytes: state.used_bytes,
            peak_bytes: state.peak_bytes,
            allocation_count: state.allocation_count,
            by_category: state.by_category,
        })
    }
}

/// Immutable process-local memory metrics snapshot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemoryStatistics {
    limit_bytes: u64,
    reserved_bytes: u64,
    peak_bytes: u64,
    allocation_count: u64,
    by_category: [u64; OWNER_CATEGORY_COUNT],
}

impl MemoryStatistics {
    /// Configured hard limit.
    pub const fn limit_bytes(self) -> u64 {
        self.limit_bytes
    }

    /// Bytes currently reserved.
    pub const fn reserved_bytes(self) -> u64 {
        self.reserved_bytes
    }

    /// Maximum simultaneous reservation observed.
    pub const fn peak_bytes(self) -> u64 {
        self.peak_bytes
    }

    /// Successful reservation count since pool creation.
    pub const fn allocation_count(self) -> u64 {
        self.allocation_count
    }

    /// Current bytes attributed to the supplied ownership category.
    pub const fn bytes_for(self, owner: &MemoryOwner) -> u64 {
        self.by_category[owner.category_index()]
    }
}

/// RAII guard for an exact memory reservation.
#[derive(Debug)]
pub struct MemoryReservation {
    bytes: u64,
    owner: MemoryOwner,
    state: Arc<Mutex<MemoryState>>,
    released: bool,
}

impl MemoryReservation {
    /// Returns the exact reserved byte count.
    pub const fn bytes(&self) -> u64 {
        self.bytes
    }

    /// Returns the explicit allocation owner.
    pub const fn owner(&self) -> &MemoryOwner {
        &self.owner
    }

    /// Releases the reservation immediately.
    pub fn release(mut self) -> Result<(), EngineError> {
        self.release_inner()?;
        self.released = true;
        Ok(())
    }

    fn release_inner(&self) -> Result<(), EngineError> {
        let mut state = match self.state.lock() {
            Ok(state) => state,
            Err(poisoned) => poisoned.into_inner(),
        };
        let used_bytes = state.used_bytes.checked_sub(self.bytes).ok_or_else(|| {
            EngineError::internal_invariant("memory reservation release underflows")
        })?;
        let category = self.owner.category_index();
        let category_bytes = state.by_category[category]
            .checked_sub(self.bytes)
            .ok_or_else(|| EngineError::internal_invariant("memory owner release underflows"))?;
        state.used_bytes = used_bytes;
        state.by_category[category] = category_bytes;
        Ok(())
    }
}

impl Drop for MemoryReservation {
    fn drop(&mut self) {
        if !self.released {
            let _release_result = self.release_inner();
            self.released = true;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reservations_are_bounded_and_drop_releases_them() {
        let pool = MemoryPool::new(100).expect("pool must validate");
        let first = pool.reserve(60).expect("first reservation must fit");
        assert_eq!(first.bytes(), 60);
        assert_eq!(pool.used_bytes().expect("accounting must be readable"), 60);
        assert!(pool.reserve(41).is_err());
        drop(first);
        assert_eq!(pool.available_bytes().expect("accounting must be readable"), 100);
    }

    #[test]
    fn explicit_release_is_exact() {
        let pool = MemoryPool::new(64).expect("pool must validate");
        let reservation = pool.reserve(64).expect("reservation must fit");
        reservation.release().expect("reservation must release");
        assert_eq!(pool.used_bytes().expect("accounting must be readable"), 0);
    }

    #[test]
    fn zero_and_over_limit_reservations_fail_closed() {
        let pool = MemoryPool::new(16).expect("pool must validate");
        assert!(pool.reserve(0).is_err());
        assert!(pool.reserve(17).is_err());
    }

    #[test]
    fn reservation_preserves_headroom_against_preused_capacity() {
        let pool = MemoryPool::new(100).expect("pool must validate");
        let existing = pool.reserve(30).expect("existing use must fit");
        let reservation = pool
            .reserve_for_preserving(40, 30, MemoryOwner::Engine)
            .expect("request plus headroom must exactly fit");
        assert_eq!(pool.used_bytes().unwrap(), 70);
        assert_eq!(reservation.bytes(), 40);
        assert!(pool.reserve_for_preserving(1, 30, MemoryOwner::Engine).is_err());
        assert_eq!(pool.used_bytes().unwrap(), 70);
        drop(reservation);
        assert_eq!(pool.used_bytes().unwrap(), 30);
        drop(existing);
        assert_eq!(pool.used_bytes().unwrap(), 0);
    }

    #[test]
    fn preserved_headroom_overflow_fails_without_accounting_changes() {
        let pool = MemoryPool::new(16).expect("pool must validate");
        assert!(pool.reserve_for_preserving(1, u64::MAX, MemoryOwner::Engine).is_err());
        let statistics = pool.statistics().unwrap();
        assert_eq!(statistics.reserved_bytes(), 0);
        assert_eq!(statistics.allocation_count(), 0);

        let mut reservation = pool.reserve(1).expect("baseline reservation must fit");
        assert!(pool.grow_reservation_preserving(&mut reservation, 1, u64::MAX).is_err());
        assert_eq!(reservation.bytes(), 1);
        assert_eq!(pool.used_bytes().unwrap(), 1);
    }

    #[test]
    fn reservation_identity_is_pool_specific() {
        let pool = MemoryPool::new(16).expect("pool must validate");
        let clone = pool.clone();
        let other = MemoryPool::new(16).expect("pool must validate");
        let reservation = pool.reserve(8).expect("reservation must fit");
        assert!(pool.owns(&reservation));
        assert!(clone.owns(&reservation));
        assert!(!other.owns(&reservation));
    }

    #[test]
    fn reservation_growth_is_bounded_and_owned() {
        let pool = MemoryPool::new(16).expect("pool must validate");
        let other = MemoryPool::new(16).expect("pool must validate");
        let mut reservation = pool.reserve(8).expect("reservation must fit");
        pool.grow_reservation(&mut reservation, 8).expect("growth must fit");
        assert_eq!(reservation.bytes(), 16);
        assert!(pool.grow_reservation(&mut reservation, 1).is_err());
        assert!(other.grow_reservation(&mut reservation, 1).is_err());
        pool.shrink_reservation(&mut reservation, 8).expect("shrink must release spare bytes");
        assert_eq!(reservation.bytes(), 8);
        assert_eq!(pool.used_bytes().unwrap(), 8);
        assert!(pool.shrink_reservation(&mut reservation, 8).is_err());
    }

    #[test]
    fn reservation_growth_preserves_requested_headroom() {
        let pool = MemoryPool::new(32).expect("pool must validate");
        let _existing = pool.reserve(8).expect("existing use must fit");
        let mut reservation = pool.reserve(8).expect("reservation must fit");
        pool.grow_reservation_preserving(&mut reservation, 8, 8)
            .expect("growth and headroom must fit");
        assert_eq!(reservation.bytes(), 16);
        assert!(pool.grow_reservation_preserving(&mut reservation, 1, 8).is_err());
        assert_eq!(reservation.bytes(), 16);
        assert_eq!(pool.used_bytes().unwrap(), 24);
    }

    #[test]
    fn owner_transfer_preserves_totals_and_updates_categories() {
        let request = RequestId::new(1).unwrap();
        let sequence = SequenceId::new(1).unwrap();
        let pool = MemoryPool::new_scoped(16, MemoryOwner::RequestScratch(request)).unwrap();
        let mut reservation = pool.reserve(8).unwrap();
        assert_eq!(pool.statistics().unwrap().bytes_for(reservation.owner()), 8);
        pool.reassign_reservation(&mut reservation, MemoryOwner::SequenceKv(sequence)).unwrap();
        let statistics = pool.statistics().unwrap();
        assert_eq!(statistics.reserved_bytes(), 8);
        assert_eq!(statistics.bytes_for(&MemoryOwner::RequestScratch(request)), 0);
        assert_eq!(statistics.bytes_for(&MemoryOwner::SequenceKv(sequence)), 8);
    }
}
