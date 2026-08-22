//! Bounded scratch storage, admission calculations, and device-memory contracts.

use super::{MemoryOwner, MemoryPool, MemoryReservation, MemoryStatistics};
use personalai_engine_types::{BackendId, EngineError, RequestId, ResourceBudget};
use std::sync::{Arc, Mutex};

/// Backend-neutral memory domain.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceMemoryKind {
    /// Ordinary non-executable process memory.
    Cpu,
    /// Apple Metal device memory, not implemented in this stage.
    Metal,
    /// NVIDIA CUDA device memory, not implemented in this stage.
    Cuda,
}

/// Owned, zero-initialized host buffer with RAII accounting.
#[derive(Debug)]
pub struct HostBuffer {
    bytes: Box<[u8]>,
    reservation: MemoryReservation,
}

impl HostBuffer {
    fn allocate(pool: &MemoryPool, bytes: usize, owner: MemoryOwner) -> Result<Self, EngineError> {
        Self::allocate_preserving(pool, bytes, owner, 0)
    }

    /// Allocates initialized host memory while preserving pool headroom.
    ///
    /// The final allocation is an exact-length boxed slice. Any temporary
    /// spare capacity reported by the fallible vector allocator is also
    /// admitted under the same headroom rule before initialization, then
    /// released after conversion to exact-length storage.
    pub fn allocate_preserving(
        pool: &MemoryPool,
        bytes: usize,
        owner: MemoryOwner,
        preserved_headroom_bytes: u64,
    ) -> Result<Self, EngineError> {
        let bytes_u64 = u64::try_from(bytes)
            .map_err(|_| EngineError::resource_exhaustion("host buffer length overflows"))?;
        let mut reservation =
            pool.reserve_for_preserving(bytes_u64, preserved_headroom_bytes, owner)?;
        let mut allocation = Vec::new();
        allocation
            .try_reserve_exact(bytes)
            .map_err(|_| EngineError::resource_exhaustion("host buffer allocation failed"))?;
        let spare = allocation.capacity().checked_sub(bytes).ok_or_else(|| {
            EngineError::internal_invariant("host buffer capacity is below initialized length")
        })?;
        let spare_u64 = u64::try_from(spare).map_err(|_| {
            EngineError::resource_exhaustion("host buffer spare capacity overflows")
        })?;
        if spare_u64 != 0 {
            pool.grow_reservation_preserving(
                &mut reservation,
                spare_u64,
                preserved_headroom_bytes,
            )?;
        }
        allocation.resize(bytes, 0);
        let bytes = allocation.into_boxed_slice();
        if spare_u64 != 0 {
            pool.shrink_reservation(&mut reservation, spare_u64)?;
        }
        Ok(Self { bytes, reservation })
    }

    /// Returns initialized immutable bytes.
    pub fn as_slice(&self) -> &[u8] {
        &self.bytes
    }

    /// Returns initialized mutable bytes to trusted engine code.
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        &mut self.bytes
    }

    /// Replaces the complete buffer; partial client offsets are not accepted.
    pub fn write_all(&mut self, source: &[u8]) -> Result<(), EngineError> {
        if source.len() != self.bytes.len() {
            return Err(EngineError::invalid_input("host transfer length does not match buffer"));
        }
        self.bytes.copy_from_slice(source);
        Ok(())
    }

    /// Returns accounted allocation bytes.
    pub const fn reserved_bytes(&self) -> u64 {
        self.reservation.bytes()
    }

    fn clear(&mut self) {
        self.bytes.fill(0);
    }
}

impl Drop for HostBuffer {
    fn drop(&mut self) {
        self.clear();
    }
}

/// Opaque backend allocation. CPU storage is real; GPU variants are absent.
#[derive(Debug)]
pub struct DeviceBuffer {
    kind: DeviceMemoryKind,
    storage: HostBuffer,
}

impl DeviceBuffer {
    /// Returns the actual memory domain.
    pub const fn kind(&self) -> DeviceMemoryKind {
        self.kind
    }

    /// Returns buffer length.
    pub fn len(&self) -> usize {
        self.storage.bytes.len()
    }

    /// Reports whether the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.storage.bytes.is_empty()
    }
}

/// Backend-neutral memory operations required by future accelerated backends.
pub trait DeviceMemory: Send + Sync {
    /// Allocates initialized host memory.
    fn allocate_host(&self, bytes: usize, owner: MemoryOwner) -> Result<HostBuffer, EngineError>;
    /// Allocates initialized device memory.
    fn allocate_device(
        &self,
        bytes: usize,
        owner: MemoryOwner,
    ) -> Result<DeviceBuffer, EngineError>;
    /// Copies an entire host buffer to an equal-sized device buffer.
    fn host_to_device(
        &self,
        source: &HostBuffer,
        destination: &mut DeviceBuffer,
    ) -> Result<(), EngineError>;
    /// Copies an entire device buffer to an equal-sized host buffer.
    fn device_to_host(
        &self,
        source: &DeviceBuffer,
        destination: &mut HostBuffer,
    ) -> Result<(), EngineError>;
    /// Releases and clears one device allocation.
    fn release_device(&self, buffer: DeviceBuffer) -> Result<(), EngineError>;
    /// Waits for prior memory operations.
    fn synchronize(&self) -> Result<(), EngineError>;
    /// Returns current bounded memory statistics.
    fn memory_statistics(&self) -> Result<MemoryStatistics, EngineError>;
}

/// Backend resource bundle with deterministic release after synchronization.
#[derive(Debug)]
pub struct BackendResources<B: DeviceMemory> {
    backend: B,
    buffers: Vec<DeviceBuffer>,
}

impl<B: DeviceMemory> BackendResources<B> {
    /// Creates empty backend-owned resource state.
    pub const fn new(backend: B) -> Self {
        Self { backend, buffers: Vec::new() }
    }

    /// Allocates and retains one device buffer.
    pub fn allocate(&mut self, bytes: usize, owner: MemoryOwner) -> Result<usize, EngineError> {
        self.buffers.try_reserve(1).map_err(|_| {
            EngineError::resource_exhaustion("backend resource table allocation failed")
        })?;
        let buffer = self.backend.allocate_device(bytes, owner)?;
        self.buffers.push(buffer);
        Ok(self.buffers.len() - 1)
    }

    /// Returns one live backend allocation.
    pub fn buffer(&self, index: usize) -> Result<&DeviceBuffer, EngineError> {
        self.buffers
            .get(index)
            .ok_or_else(|| EngineError::invalid_input("backend buffer identifier is invalid"))
    }

    /// Synchronizes and releases every backend allocation.
    pub fn release_all(&mut self) -> Result<(), EngineError> {
        self.backend.synchronize()?;
        while let Some(buffer) = self.buffers.pop() {
            self.backend.release_device(buffer)?;
        }
        Ok(())
    }

    /// Returns current backend memory statistics.
    pub fn statistics(&self) -> Result<MemoryStatistics, EngineError> {
        self.backend.memory_statistics()
    }
}

/// Real scalar CPU memory implementation; it performs no GPU emulation.
#[derive(Debug, Clone)]
pub struct CpuMemoryBackend {
    pool: MemoryPool,
}

impl CpuMemoryBackend {
    /// Creates a CPU backend over the supplied accounting pool.
    pub const fn new(pool: MemoryPool) -> Self {
        Self { pool }
    }

    /// Rejects unimplemented device kinds explicitly.
    pub fn for_kind(kind: DeviceMemoryKind, limit_bytes: u64) -> Result<Self, EngineError> {
        if kind != DeviceMemoryKind::Cpu {
            return Err(EngineError::not_implemented(
                "requested device-memory backend is not implemented",
            ));
        }
        let backend = BackendId::new("cpu-reference")?;
        Ok(Self::new(MemoryPool::new_scoped(limit_bytes, MemoryOwner::Backend(backend))?))
    }
}

impl DeviceMemory for CpuMemoryBackend {
    fn allocate_host(&self, bytes: usize, owner: MemoryOwner) -> Result<HostBuffer, EngineError> {
        HostBuffer::allocate(&self.pool, bytes, owner)
    }

    fn allocate_device(
        &self,
        bytes: usize,
        owner: MemoryOwner,
    ) -> Result<DeviceBuffer, EngineError> {
        Ok(DeviceBuffer {
            kind: DeviceMemoryKind::Cpu,
            storage: HostBuffer::allocate(&self.pool, bytes, owner)?,
        })
    }

    fn host_to_device(
        &self,
        source: &HostBuffer,
        destination: &mut DeviceBuffer,
    ) -> Result<(), EngineError> {
        if destination.kind != DeviceMemoryKind::Cpu {
            return Err(EngineError::backend_failure("device buffer is not CPU memory"));
        }
        destination.storage.write_all(source.as_slice())
    }

    fn device_to_host(
        &self,
        source: &DeviceBuffer,
        destination: &mut HostBuffer,
    ) -> Result<(), EngineError> {
        if source.kind != DeviceMemoryKind::Cpu {
            return Err(EngineError::backend_failure("device buffer is not CPU memory"));
        }
        destination.write_all(source.storage.as_slice())
    }

    fn release_device(&self, mut buffer: DeviceBuffer) -> Result<(), EngineError> {
        buffer.storage.clear();
        drop(buffer);
        Ok(())
    }

    fn synchronize(&self) -> Result<(), EngineError> {
        Ok(())
    }

    fn memory_statistics(&self) -> Result<MemoryStatistics, EngineError> {
        self.pool.statistics()
    }
}

#[derive(Debug)]
struct CachedScratch {
    bytes: HostBuffer,
}

#[derive(Debug)]
struct ScratchState {
    free: Vec<CachedScratch>,
    active_bytes: u64,
    peak_active_bytes: u64,
}

/// Thread-safe bounded pool for reusable, cleared request scratch buffers.
#[derive(Debug, Clone)]
pub struct ScratchMemoryManager {
    pool: MemoryPool,
    idle_owner: MemoryOwner,
    state: Arc<Mutex<ScratchState>>,
}

impl ScratchMemoryManager {
    /// Creates an engine-wide scratch pool.
    pub fn new(engine_limit_bytes: u64) -> Result<Self, EngineError> {
        let backend = BackendId::new("cpu-reference-scratch")?;
        let idle_owner = MemoryOwner::Backend(backend);
        Ok(Self {
            pool: MemoryPool::new_scoped(engine_limit_bytes, idle_owner.clone())?,
            idle_owner,
            state: Arc::new(Mutex::new(ScratchState {
                free: Vec::new(),
                active_bytes: 0,
                peak_active_bytes: 0,
            })),
        })
    }

    /// Starts request-scoped scratch ownership with a hard request limit.
    pub fn begin_request(
        &self,
        request_id: RequestId,
        request_limit_bytes: u64,
    ) -> Result<ScratchArena, EngineError> {
        if request_limit_bytes == 0 || request_limit_bytes > self.pool.limit_bytes() {
            return Err(EngineError::invalid_input("scratch request limit is outside pool bounds"));
        }
        Ok(ScratchArena {
            request_id,
            request_limit_bytes,
            used_bytes: 0,
            generation: 1,
            allocations: Vec::new(),
            manager: self.clone(),
            closed: false,
        })
    }

    /// Returns scheduler-facing reserved, active, cached, and peak bytes.
    pub fn metrics(&self) -> Result<ScratchMetrics, EngineError> {
        let state = self
            .state
            .lock()
            .map_err(|_| EngineError::internal_invariant("scratch accounting lock is poisoned"))?;
        let reserved_bytes = self.pool.used_bytes()?;
        let cached_bytes = reserved_bytes.checked_sub(state.active_bytes).ok_or_else(|| {
            EngineError::internal_invariant("scratch active bytes exceed reservations")
        })?;
        Ok(ScratchMetrics {
            limit_bytes: self.pool.limit_bytes(),
            reserved_bytes,
            active_bytes: state.active_bytes,
            cached_bytes,
            peak_active_bytes: state.peak_active_bytes,
        })
    }

    /// Releases all currently idle buffers and returns reclaimed bytes.
    pub fn reclaim_idle(&self) -> Result<u64, EngineError> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| EngineError::internal_invariant("scratch accounting lock is poisoned"))?;
        let before = self.pool.used_bytes()?;
        state.free.clear();
        let after = self.pool.used_bytes()?;
        before
            .checked_sub(after)
            .ok_or_else(|| EngineError::internal_invariant("scratch reclaim accounting increased"))
    }
}

/// Immutable scratch-memory metrics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScratchMetrics {
    /// Engine-wide scratch ceiling.
    pub limit_bytes: u64,
    /// Active plus cached physical reservations.
    pub reserved_bytes: u64,
    /// Bytes currently assigned to requests.
    pub active_bytes: u64,
    /// Cleared idle bytes immediately reclaimable.
    pub cached_bytes: u64,
    /// Highest simultaneous active usage.
    pub peak_active_bytes: u64,
}

/// Opaque request-local scratch slot identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScratchBufferId {
    slot: u32,
    generation: u64,
}

/// Request-owned scratch arena. Drop, cancel, and finish all clear its buffers.
#[derive(Debug)]
pub struct ScratchArena {
    request_id: RequestId,
    request_limit_bytes: u64,
    used_bytes: u64,
    generation: u64,
    allocations: Vec<HostBuffer>,
    manager: ScratchMemoryManager,
    closed: bool,
}

impl ScratchArena {
    /// Allocates or reuses an initialized request-local scratch buffer.
    pub fn allocate_zeroed(&mut self, bytes: usize) -> Result<ScratchBufferId, EngineError> {
        if self.closed || bytes == 0 {
            return Err(EngineError::invalid_input("scratch arena is closed or size is zero"));
        }
        let bytes_u64 = u64::try_from(bytes)
            .map_err(|_| EngineError::resource_exhaustion("scratch size overflows"))?;
        let next = self
            .used_bytes
            .checked_add(bytes_u64)
            .ok_or_else(|| EngineError::resource_exhaustion("request scratch usage overflows"))?;
        if next > self.request_limit_bytes {
            return Err(EngineError::resource_exhaustion("request scratch limit is exhausted"));
        }
        self.allocations.try_reserve(1).map_err(|_| {
            EngineError::resource_exhaustion("scratch allocation table growth failed")
        })?;
        let mut state =
            self.manager.state.lock().map_err(|_| {
                EngineError::internal_invariant("scratch accounting lock is poisoned")
            })?;
        let mut buffer = if let Some(index) =
            state.free.iter().position(|candidate| candidate.bytes.as_slice().len() == bytes)
        {
            state.free.swap_remove(index).bytes
        } else {
            HostBuffer::allocate(
                &self.manager.pool,
                bytes,
                MemoryOwner::RequestScratch(self.request_id),
            )?
        };
        self.manager.pool.reassign_reservation(
            &mut buffer.reservation,
            MemoryOwner::RequestScratch(self.request_id),
        )?;
        buffer.clear();
        state.active_bytes = state
            .active_bytes
            .checked_add(bytes_u64)
            .ok_or_else(|| EngineError::resource_exhaustion("engine scratch usage overflows"))?;
        state.peak_active_bytes = state.peak_active_bytes.max(state.active_bytes);
        drop(state);
        self.allocations.push(buffer);
        self.used_bytes = next;
        let slot = u32::try_from(self.allocations.len() - 1)
            .map_err(|_| EngineError::resource_exhaustion("scratch slot index overflows"))?;
        Ok(ScratchBufferId { slot, generation: self.generation })
    }

    /// Returns initialized immutable bytes for a live request-local slot.
    pub fn get(&self, id: ScratchBufferId) -> Result<&[u8], EngineError> {
        self.validate_id(id).map(HostBuffer::as_slice)
    }

    /// Returns initialized mutable bytes for a live request-local slot.
    pub fn get_mut(&mut self, id: ScratchBufferId) -> Result<&mut [u8], EngineError> {
        if self.closed || id.generation != self.generation {
            return Err(EngineError::invalid_input("scratch buffer handle is stale"));
        }
        self.allocations
            .get_mut(id.slot as usize)
            .map(HostBuffer::as_mut_slice)
            .ok_or_else(|| EngineError::invalid_input("scratch buffer handle is invalid"))
    }

    /// Finishes the request, clearing and returning buffers for safe reuse.
    pub fn finish(mut self) -> Result<(), EngineError> {
        self.release_all()?;
        self.closed = true;
        Ok(())
    }

    /// Cancels the request identically to normal cleanup.
    pub fn cancel(self) -> Result<(), EngineError> {
        self.finish()
    }

    /// Returns logical bytes currently assigned to this request.
    pub const fn used_bytes(&self) -> u64 {
        self.used_bytes
    }

    fn validate_id(&self, id: ScratchBufferId) -> Result<&HostBuffer, EngineError> {
        if self.closed || id.generation != self.generation {
            return Err(EngineError::invalid_input("scratch buffer handle is stale"));
        }
        self.allocations
            .get(id.slot as usize)
            .ok_or_else(|| EngineError::invalid_input("scratch buffer handle is invalid"))
    }

    fn release_all(&mut self) -> Result<(), EngineError> {
        if self.allocations.is_empty() {
            self.used_bytes = 0;
            return Ok(());
        }
        let mut state =
            self.manager.state.lock().map_err(|_| {
                EngineError::internal_invariant("scratch accounting lock is poisoned")
            })?;
        if state.free.try_reserve(self.allocations.len()).is_err() {
            for mut allocation in self.allocations.drain(..) {
                allocation.clear();
            }
            state.active_bytes =
                state.active_bytes.checked_sub(self.used_bytes).ok_or_else(|| {
                    EngineError::internal_invariant("scratch release accounting underflows")
                })?;
            self.used_bytes = 0;
            return Err(EngineError::resource_exhaustion(
                "scratch reuse table growth failed; buffers were released",
            ));
        }
        for mut allocation in self.allocations.drain(..) {
            allocation.clear();
            self.manager.pool.reassign_reservation(
                &mut allocation.reservation,
                self.manager.idle_owner.clone(),
            )?;
            state.free.push(CachedScratch { bytes: allocation });
        }
        state.active_bytes = state.active_bytes.checked_sub(self.used_bytes).ok_or_else(|| {
            EngineError::internal_invariant("scratch release accounting underflows")
        })?;
        self.used_bytes = 0;
        self.generation = self.generation.checked_add(1).ok_or_else(|| {
            EngineError::resource_exhaustion("scratch handle generation overflows")
        })?;
        Ok(())
    }
}

impl Drop for ScratchArena {
    fn drop(&mut self) {
        if !self.closed {
            let _result = self.release_all();
            self.closed = true;
        }
    }
}

/// Checked memory requirements supplied to admission control.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemoryRequirements {
    /// Immutable model bytes.
    pub model_bytes: u64,
    /// Request prompt bytes.
    pub prompt_bytes: u64,
    /// Sequence KV reservation bytes.
    pub kv_cache_bytes: u64,
    /// Peak scratch bytes.
    pub scratch_bytes: u64,
    /// Output logits and sampling bytes.
    pub output_bytes: u64,
}

impl MemoryRequirements {
    /// Returns checked aggregate bytes.
    pub fn total_bytes(self) -> Result<u64, EngineError> {
        self.model_bytes
            .checked_add(self.prompt_bytes)
            .and_then(|value| value.checked_add(self.kv_cache_bytes))
            .and_then(|value| value.checked_add(self.scratch_bytes))
            .and_then(|value| value.checked_add(self.output_bytes))
            .ok_or_else(|| EngineError::resource_exhaustion("memory requirement total overflows"))
    }
}

/// Stateless scheduler-facing resource admission calculations.
#[derive(Debug, Clone, Copy)]
pub struct MemoryAdmission;

impl MemoryAdmission {
    /// Reports whether immutable model state fits current engine capacity.
    pub fn can_load_model(
        statistics: MemoryStatistics,
        immutable_bytes: u64,
    ) -> Result<bool, EngineError> {
        let next = statistics.reserved_bytes().checked_add(immutable_bytes).ok_or_else(|| {
            EngineError::resource_exhaustion("model admission arithmetic overflows")
        })?;
        Ok(next <= statistics.limit_bytes())
    }

    /// Reports whether a request fits both its request budget and engine pool.
    pub fn can_admit_request(
        statistics: MemoryStatistics,
        budget: &ResourceBudget,
        requirements: MemoryRequirements,
    ) -> Result<bool, EngineError> {
        let total = requirements.total_bytes()?;
        if total > budget.max_memory_bytes()
            || requirements.kv_cache_bytes > budget.max_kv_cache_bytes()
        {
            return Ok(false);
        }
        Self::can_load_model(statistics, total)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use personalai_engine_types::{ModelId, SequenceId};

    #[test]
    fn cpu_transfers_are_exact_and_gpu_is_explicitly_unsupported() {
        let backend = CpuMemoryBackend::for_kind(DeviceMemoryKind::Cpu, 64).unwrap();
        let owner = MemoryOwner::SequenceKv(SequenceId::new(1).unwrap());
        let mut host = backend.allocate_host(8, owner.clone()).unwrap();
        host.write_all(&[1; 8]).unwrap();
        let mut device = backend.allocate_device(8, owner).unwrap();
        backend.host_to_device(&host, &mut device).unwrap();
        let mut output = backend
            .allocate_host(8, MemoryOwner::ModelState(ModelId::new("tiny").unwrap()))
            .unwrap();
        backend.device_to_host(&device, &mut output).unwrap();
        assert_eq!(output.as_slice(), &[1; 8]);
        assert!(CpuMemoryBackend::for_kind(DeviceMemoryKind::Metal, 64).is_err());
    }

    #[test]
    fn backend_resources_release_every_owned_buffer() {
        let backend = CpuMemoryBackend::for_kind(DeviceMemoryKind::Cpu, 64).unwrap();
        let mut resources = BackendResources::new(backend);
        let index =
            resources.allocate(32, MemoryOwner::SequenceKv(SequenceId::new(1).unwrap())).unwrap();
        assert_eq!(resources.buffer(index).unwrap().len(), 32);
        assert_eq!(resources.statistics().unwrap().reserved_bytes(), 32);
        resources.release_all().unwrap();
        assert_eq!(resources.statistics().unwrap().reserved_bytes(), 0);
    }

    #[test]
    fn host_buffer_preserves_headroom_accounts_exact_storage_and_releases() {
        let pool = MemoryPool::new(256).unwrap();
        let existing = pool.reserve(32).unwrap();
        let owner = MemoryOwner::ModelState(ModelId::new("headroom-fixture").unwrap());
        let buffer = HostBuffer::allocate_preserving(&pool, 48, owner.clone(), 128).unwrap();

        assert_eq!(buffer.as_slice(), &[0; 48]);
        assert_eq!(buffer.reserved_bytes(), 48);
        let statistics = pool.statistics().unwrap();
        assert_eq!(statistics.reserved_bytes(), 80);
        assert_eq!(statistics.bytes_for(&owner), 48);

        drop(buffer);
        let statistics = pool.statistics().unwrap();
        assert_eq!(statistics.reserved_bytes(), 32);
        assert_eq!(statistics.bytes_for(&owner), 0);
        drop(existing);
        assert_eq!(pool.used_bytes().unwrap(), 0);
    }

    #[test]
    fn scratch_is_bounded_cleared_and_reclaimed() {
        let manager = ScratchMemoryManager::new(32).unwrap();
        let request = RequestId::new(1).unwrap();
        let mut arena = manager.begin_request(request, 16).unwrap();
        let id = arena.allocate_zeroed(16).unwrap();
        arena.get_mut(id).unwrap().fill(0xaa);
        assert!(arena.allocate_zeroed(1).is_err());
        arena.cancel().unwrap();
        assert_eq!(manager.metrics().unwrap().cached_bytes, 16);
        let mut second = manager.begin_request(RequestId::new(2).unwrap(), 16).unwrap();
        let reused = second.allocate_zeroed(16).unwrap();
        assert_eq!(second.get(reused).unwrap(), &[0; 16]);
        drop(second);
        assert_eq!(manager.reclaim_idle().unwrap(), 16);
    }

    #[test]
    fn admission_uses_engine_and_request_limits() {
        let pool = MemoryPool::new(100).unwrap();
        let stats = pool.statistics().unwrap();
        let requirements = MemoryRequirements {
            model_bytes: 0,
            prompt_bytes: 10,
            kv_cache_bytes: 20,
            scratch_bytes: 10,
            output_bytes: 10,
        };
        let budget = ResourceBudget::new(50, 20, 1).unwrap();
        assert!(MemoryAdmission::can_admit_request(stats, &budget, requirements).unwrap());
        assert!(!MemoryAdmission::can_load_model(stats, 101).unwrap());
        let overflow = MemoryRequirements {
            model_bytes: u64::MAX,
            prompt_bytes: 1,
            kv_cache_bytes: 0,
            scratch_bytes: 0,
            output_bytes: 0,
        };
        assert!(overflow.total_bytes().is_err());
    }

    #[test]
    fn deterministic_allocation_pressure_leaves_accounting_exact() {
        let manager = ScratchMemoryManager::new(16).unwrap();
        let mut first = manager.begin_request(RequestId::new(1).unwrap(), 16).unwrap();
        first.allocate_zeroed(16).unwrap();
        let mut second = manager.begin_request(RequestId::new(2).unwrap(), 16).unwrap();
        assert!(second.allocate_zeroed(16).is_err());
        let metrics = manager.metrics().unwrap();
        assert_eq!(metrics.active_bytes, 16);
        assert_eq!(metrics.reserved_bytes, 16);
    }
}
