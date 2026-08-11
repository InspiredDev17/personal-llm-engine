//! Checked process-local memory accounting.
//!
//! This crate accounts tensor and cache bytes only. It is unrelated to the Go
//! platform's personal memory, conversation storage, or retrieval authority.

use personalai_engine_types::{EngineError, MAX_RESOURCE_MEMORY_BYTES, ResourceBudget};
use std::sync::{Arc, Mutex};

#[derive(Debug)]
struct MemoryState {
    used_bytes: u64,
}

/// Thread-safe bounded memory accounting pool.
#[derive(Debug, Clone)]
pub struct MemoryPool {
    limit_bytes: u64,
    state: Arc<Mutex<MemoryState>>,
}

impl MemoryPool {
    /// Creates a memory pool with a positive hard-bounded limit.
    pub fn new(limit_bytes: u64) -> Result<Self, EngineError> {
        if limit_bytes == 0 || limit_bytes > MAX_RESOURCE_MEMORY_BYTES {
            return Err(EngineError::invalid_input("memory pool limit is outside hard bounds"));
        }
        Ok(Self { limit_bytes, state: Arc::new(Mutex::new(MemoryState { used_bytes: 0 })) })
    }

    /// Creates an allocation pool from the request's memory budget.
    pub fn from_budget(budget: &ResourceBudget) -> Result<Self, EngineError> {
        Self::new(budget.max_memory_bytes())
    }

    /// Reserves positive bytes and returns an automatically released guard.
    pub fn reserve(&self, bytes: u64) -> Result<MemoryReservation, EngineError> {
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
        if next > self.limit_bytes {
            return Err(EngineError::resource_exhaustion(
                "memory reservation exceeds the configured pool",
            ));
        }
        state.used_bytes = next;
        drop(state);
        Ok(MemoryReservation { bytes, state: Arc::clone(&self.state), released: false })
    }

    /// Extends one owned reservation after an allocator reports spare capacity.
    pub fn grow_reservation(
        &self,
        reservation: &mut MemoryReservation,
        additional_bytes: u64,
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
        if used_bytes > self.limit_bytes {
            return Err(EngineError::resource_exhaustion(
                "memory reservation exceeds the configured pool",
            ));
        }
        state.used_bytes = used_bytes;
        reservation.bytes = reservation_bytes;
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
}

/// RAII guard for an exact memory reservation.
#[derive(Debug)]
pub struct MemoryReservation {
    bytes: u64,
    state: Arc<Mutex<MemoryState>>,
    released: bool,
}

impl MemoryReservation {
    /// Returns the exact reserved byte count.
    pub const fn bytes(&self) -> u64 {
        self.bytes
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
        state.used_bytes = state.used_bytes.checked_sub(self.bytes).ok_or_else(|| {
            EngineError::internal_invariant("memory reservation release underflows")
        })?;
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
    }
}
