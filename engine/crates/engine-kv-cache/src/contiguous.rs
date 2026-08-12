//! Correctness-first contiguous key/value cache.

use super::KvCachePlan;
use personalai_engine_memory::{MemoryOwner, MemoryPool};
use personalai_engine_tensor::{MutableTensor, TensorShape, f32_to_bf16_bits, f32_to_f16_bits};
use personalai_engine_types::{DType, EngineError, ModelId, RequestId, SequenceId};
use std::collections::BTreeMap;
use std::sync::{Mutex, MutexGuard};

/// Opaque generation-checked sequence ownership handle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SequenceHandle {
    sequence_id: SequenceId,
    generation: u64,
}

impl SequenceHandle {
    /// Returns the scheduler-assigned sequence identifier.
    pub const fn sequence_id(self) -> SequenceId {
        self.sequence_id
    }
}

/// Sequence lifecycle visible to scheduler metrics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SequenceStatus {
    /// Cache may be extended or read.
    Active,
    /// Cache was cleared following cancellation or incomplete mutation.
    Cancelled,
}

#[derive(Debug)]
struct SequenceEntry {
    request_id: RequestId,
    generation: u64,
    token_count: u32,
    status: SequenceStatus,
    storage: MutableTensor,
}

#[derive(Debug)]
struct CacheState {
    loaded: bool,
    next_generation: u64,
    peak_reserved_bytes: u64,
    sequences: BTreeMap<SequenceId, SequenceEntry>,
    #[cfg(test)]
    fail_extension_after_layer: Option<u32>,
}

/// Scheduler-facing snapshot of contiguous cache pressure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KvCacheMetrics {
    /// Live sequence objects, including cancelled objects awaiting destruction.
    pub sequence_count: u32,
    /// Active sequence objects.
    pub active_sequence_count: u32,
    /// Physically reserved bytes.
    pub reserved_bytes: u64,
    /// Bytes containing committed K/V positions.
    pub used_bytes: u64,
    /// Bytes reclaimable by destroying all sequence objects.
    pub reclaimable_bytes: u64,
    /// Highest physical reservation observed.
    pub peak_reserved_bytes: u64,
}

/// Cache/model lifecycle state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvCacheStatus {
    /// Model identity remains loaded and sequences may be created.
    Loaded,
    /// Model unload invalidated every sequence; this cache cannot be reused.
    Unloaded,
}

/// One-model, multi-sequence contiguous CPU KV cache.
#[derive(Debug)]
pub struct ContiguousKvCache {
    model_id: ModelId,
    plan: KvCachePlan,
    maximum_sequences: u32,
    pool: MemoryPool,
    state: Mutex<CacheState>,
}

impl ContiguousKvCache {
    /// Creates an empty cache after checking the exact per-sequence layout.
    pub fn new(
        model_id: ModelId,
        plan: KvCachePlan,
        maximum_sequences: u32,
        pool: MemoryPool,
    ) -> Result<Self, EngineError> {
        if maximum_sequences == 0
            || maximum_sequences > personalai_engine_types::MAX_CONCURRENT_SEQUENCES
        {
            return Err(EngineError::invalid_input(
                "KV-cache sequence limit is outside hard bounds",
            ));
        }
        Ok(Self {
            model_id,
            plan,
            maximum_sequences,
            pool,
            state: Mutex::new(CacheState {
                loaded: true,
                next_generation: 1,
                peak_reserved_bytes: 0,
                sequences: BTreeMap::new(),
                #[cfg(test)]
                fail_extension_after_layer: None,
            }),
        })
    }

    /// Returns the verified model identity that owns every sequence.
    pub const fn model_id(&self) -> &ModelId {
        &self.model_id
    }

    /// Returns the checked immutable layout.
    pub const fn plan(&self) -> &KvCachePlan {
        &self.plan
    }

    /// Returns bytes reserved for one full contiguous sequence.
    pub const fn sequence_allocation_bytes(&self) -> u64 {
        self.plan.bytes()
    }

    /// Returns logical K/V bytes consumed by each additional token.
    pub fn additional_token_bytes(&self) -> Result<u64, EngineError> {
        self.plan
            .bytes()
            .checked_div(u64::from(self.plan.token_capacity()))
            .ok_or_else(|| EngineError::internal_invariant("KV token capacity is zero"))
    }

    /// Returns logical bytes a prompt of `tokens` will occupy.
    pub fn prefill_bytes(&self, tokens: u32) -> Result<u64, EngineError> {
        if tokens > self.plan.token_capacity() {
            return Err(EngineError::resource_exhaustion("prefill exceeds context capacity"));
        }
        self.additional_token_bytes()?
            .checked_mul(u64::from(tokens))
            .ok_or_else(|| EngineError::resource_exhaustion("prefill byte count overflows"))
    }

    /// Reports whether a new full-capacity sequence reservation currently fits.
    pub fn can_admit_sequence(&self) -> Result<bool, EngineError> {
        let state = self.lock_state()?;
        if !state.loaded {
            return Ok(false);
        }
        let sequence_count = u32::try_from(state.sequences.len())
            .map_err(|_| EngineError::internal_invariant("sequence count exceeds u32"))?;
        if sequence_count >= self.maximum_sequences {
            return Ok(false);
        }
        Ok(self.plan.bytes() <= self.pool.available_bytes()?)
    }

    /// Allocates a zeroed sequence bound to a distinct request identifier.
    pub fn create_sequence(
        &self,
        request_id: RequestId,
        sequence_id: SequenceId,
    ) -> Result<SequenceHandle, EngineError> {
        let mut state = self.lock_state()?;
        ensure_loaded(&state)?;
        if state.sequences.contains_key(&sequence_id) {
            return Err(EngineError::protocol_violation("sequence identifier is already active"));
        }
        let sequence_count = u32::try_from(state.sequences.len())
            .map_err(|_| EngineError::internal_invariant("sequence count exceeds u32"))?;
        if sequence_count >= self.maximum_sequences {
            return Err(EngineError::resource_exhaustion(
                "KV-cache active sequence limit is reached",
            ));
        }
        let generation = state.next_generation;
        state.next_generation = generation
            .checked_add(1)
            .ok_or_else(|| EngineError::resource_exhaustion("sequence generation counter wraps"))?;
        let sequence_pool = self.pool.scoped(MemoryOwner::SequenceKv(sequence_id));
        let storage = MutableTensor::zeros(
            self.plan.dtype(),
            TensorShape::new([
                2,
                u64::from(self.plan.layer_count()),
                u64::from(self.plan.token_capacity()),
                u64::from(self.plan.kv_head_count()),
                u64::from(self.plan.head_dimension()),
            ])?,
            &sequence_pool,
        )?;
        state.sequences.insert(
            sequence_id,
            SequenceEntry {
                request_id,
                generation,
                token_count: 0,
                status: SequenceStatus::Active,
                storage,
            },
        );
        let reserved = state.sequences.values().try_fold(0_u64, |total, entry| {
            total.checked_add(entry.storage.reserved_bytes()).ok_or_else(|| {
                EngineError::resource_exhaustion("KV reserved-byte accounting overflows")
            })
        })?;
        state.peak_reserved_bytes = state.peak_reserved_bytes.max(reserved);
        Ok(SequenceHandle { sequence_id, generation })
    }

    /// Begins the next engine-selected position. Dropping without commit clears
    /// and cancels the complete sequence.
    pub fn begin_token(&self, handle: SequenceHandle) -> Result<KvTokenWriter<'_>, EngineError> {
        let state = self.lock_state()?;
        ensure_loaded(&state)?;
        let sequence = sequence(&state, handle)?;
        ensure_active(sequence)?;
        if sequence.token_count >= self.plan.token_capacity() {
            return Err(EngineError::resource_exhaustion("KV context capacity is exhausted"));
        }
        let position = sequence.token_count;
        let layer_count = usize::try_from(self.plan.layer_count())
            .map_err(|_| EngineError::resource_exhaustion("layer count exceeds usize"))?;
        let mut written = Vec::new();
        written
            .try_reserve_exact(layer_count)
            .map_err(|_| EngineError::resource_exhaustion("KV write state allocation failed"))?;
        written.resize(layer_count, false);
        Ok(KvTokenWriter { cache: self, state, handle, position, written, committed: false })
    }

    /// Deterministically prefills complete token-major, layer-major K/V values.
    pub fn prefill(
        &self,
        handle: SequenceHandle,
        token_count: u32,
        keys: &[f32],
        values: &[f32],
    ) -> Result<(), EngineError> {
        if token_count == 0 {
            return Err(EngineError::invalid_input("prefill must contain at least one token"));
        }
        let layer_width = self.layer_width()?;
        let expected = usize::try_from(token_count)
            .ok()
            .and_then(|tokens| tokens.checked_mul(usize::try_from(self.plan.layer_count()).ok()?))
            .and_then(|items| items.checked_mul(layer_width))
            .ok_or_else(|| EngineError::resource_exhaustion("prefill element count overflows"))?;
        if keys.len() != expected || values.len() != expected {
            return Err(EngineError::invalid_input(
                "prefill K/V lengths do not match cache layout",
            ));
        }
        if keys.iter().chain(values).any(|value| !value.is_finite()) {
            return Err(EngineError::invalid_input("KV cache rejects non-finite input"));
        }
        if self
            .token_count(handle)?
            .checked_add(token_count)
            .is_none_or(|count| count > self.plan.token_capacity())
        {
            return Err(EngineError::resource_exhaustion("prefill exceeds context capacity"));
        }
        let layers = usize::try_from(self.plan.layer_count())
            .map_err(|_| EngineError::resource_exhaustion("layer count exceeds usize"))?;
        for token in 0..usize::try_from(token_count)
            .map_err(|_| EngineError::resource_exhaustion("token count exceeds usize"))?
        {
            let mut writer = self.begin_token(handle)?;
            for layer in 0..layers {
                let start = token
                    .checked_mul(layers)
                    .and_then(|value| value.checked_add(layer))
                    .and_then(|value| value.checked_mul(layer_width))
                    .ok_or_else(|| EngineError::resource_exhaustion("prefill index overflows"))?;
                let end = start
                    .checked_add(layer_width)
                    .ok_or_else(|| EngineError::resource_exhaustion("prefill range overflows"))?;
                writer.write_layer(
                    u32::try_from(layer)
                        .map_err(|_| EngineError::resource_exhaustion("layer index overflows"))?,
                    &keys[start..end],
                    &values[start..end],
                )?;
            }
            writer.commit()?;
        }
        Ok(())
    }

    /// Reads one committed layer key into an exact destination.
    pub fn read_key_into(
        &self,
        handle: SequenceHandle,
        layer: u32,
        position: u32,
        destination: &mut [f32],
    ) -> Result<(), EngineError> {
        self.read_into(handle, 0, layer, position, destination)
    }

    /// Reads one committed layer value into an exact destination.
    pub fn read_value_into(
        &self,
        handle: SequenceHandle,
        layer: u32,
        position: u32,
        destination: &mut [f32],
    ) -> Result<(), EngineError> {
        self.read_into(handle, 1, layer, position, destination)
    }

    /// Returns committed positions for a live sequence.
    pub fn token_count(&self, handle: SequenceHandle) -> Result<u32, EngineError> {
        let state = self.lock_state()?;
        ensure_loaded(&state)?;
        let entry = sequence(&state, handle)?;
        ensure_active(entry)?;
        Ok(entry.token_count)
    }

    /// Clears committed and spare storage while retaining sequence ownership.
    pub fn reset_sequence(&self, handle: SequenceHandle) -> Result<(), EngineError> {
        let mut state = self.lock_state()?;
        ensure_loaded(&state)?;
        let entry = sequence_mut(&mut state, handle)?;
        ensure_active(entry)?;
        entry.storage.clear();
        entry.token_count = 0;
        entry.status = SequenceStatus::Active;
        Ok(())
    }

    /// Clears and invalidates a sequence after cancellation or backend failure.
    pub fn cancel_sequence(&self, handle: SequenceHandle) -> Result<(), EngineError> {
        let mut state = self.lock_state()?;
        ensure_loaded(&state)?;
        let entry = sequence_mut(&mut state, handle)?;
        clear_and_cancel(entry);
        Ok(())
    }

    /// Clears and releases one sequence allocation.
    pub fn destroy_sequence(&self, handle: SequenceHandle) -> Result<(), EngineError> {
        let mut state = self.lock_state()?;
        ensure_loaded(&state)?;
        let existing = sequence(&state, handle)?;
        if existing.generation != handle.generation {
            return Err(EngineError::protocol_violation("sequence handle is stale"));
        }
        let mut entry = state
            .sequences
            .remove(&handle.sequence_id)
            .ok_or_else(|| EngineError::protocol_violation("sequence identifier is unknown"))?;
        entry.storage.clear();
        drop(entry);
        Ok(())
    }

    /// Clears and releases every sequence, permanently invalidating the model cache.
    pub fn unload_model(&self) -> Result<(), EngineError> {
        let mut state = self.lock_state()?;
        if !state.loaded {
            return Err(EngineError::protocol_violation("model cache is already unloaded"));
        }
        for entry in state.sequences.values_mut() {
            clear_and_cancel(entry);
        }
        state.sequences.clear();
        state.loaded = false;
        Ok(())
    }

    /// Returns whether model-scoped cache state remains valid.
    pub fn status(&self) -> Result<KvCacheStatus, EngineError> {
        Ok(if self.lock_state()?.loaded { KvCacheStatus::Loaded } else { KvCacheStatus::Unloaded })
    }

    /// Returns exact pressure and reclaimability metrics.
    pub fn metrics(&self) -> Result<KvCacheMetrics, EngineError> {
        let state = self.lock_state()?;
        let sequence_count = u32::try_from(state.sequences.len())
            .map_err(|_| EngineError::internal_invariant("sequence count exceeds u32"))?;
        let active_sequence_count = u32::try_from(
            state.sequences.values().filter(|entry| entry.status == SequenceStatus::Active).count(),
        )
        .map_err(|_| EngineError::internal_invariant("active sequence count exceeds u32"))?;
        let used_bytes = state.sequences.values().try_fold(0_u64, |total, entry| {
            self.prefill_bytes(entry.token_count).and_then(|bytes| {
                total.checked_add(bytes).ok_or_else(|| {
                    EngineError::resource_exhaustion("KV used-byte accounting overflows")
                })
            })
        })?;
        let reserved_bytes = state.sequences.values().try_fold(0_u64, |total, entry| {
            total.checked_add(entry.storage.reserved_bytes()).ok_or_else(|| {
                EngineError::resource_exhaustion("KV reserved-byte accounting overflows")
            })
        })?;
        Ok(KvCacheMetrics {
            sequence_count,
            active_sequence_count,
            reserved_bytes,
            used_bytes,
            reclaimable_bytes: reserved_bytes,
            peak_reserved_bytes: state.peak_reserved_bytes,
        })
    }

    fn layer_width(&self) -> Result<usize, EngineError> {
        usize::try_from(self.plan.kv_head_count())
            .ok()
            .and_then(|heads| heads.checked_mul(usize::try_from(self.plan.head_dimension()).ok()?))
            .ok_or_else(|| EngineError::resource_exhaustion("KV layer width overflows"))
    }

    fn read_into(
        &self,
        handle: SequenceHandle,
        side: u32,
        layer: u32,
        position: u32,
        destination: &mut [f32],
    ) -> Result<(), EngineError> {
        let state = self.lock_state()?;
        ensure_loaded(&state)?;
        let entry = sequence(&state, handle)?;
        ensure_active(entry)?;
        if layer >= self.plan.layer_count() || position >= entry.token_count {
            return Err(EngineError::invalid_input("KV read location is outside committed state"));
        }
        if destination.len() != self.layer_width()? {
            return Err(EngineError::invalid_input("KV read destination length is invalid"));
        }
        let start = self.element_offset(side, layer, position, 0)?;
        for (index, destination_value) in destination.iter_mut().enumerate() {
            let element = start.checked_add(index).ok_or_else(|| {
                EngineError::resource_exhaustion("KV read element index overflows")
            })?;
            *destination_value = entry.storage.numeric_f32_value(element)?;
        }
        Ok(())
    }

    fn element_offset(
        &self,
        side: u32,
        layer: u32,
        position: u32,
        layer_element: usize,
    ) -> Result<usize, EngineError> {
        if side > 1 || layer >= self.plan.layer_count() || position >= self.plan.token_capacity() {
            return Err(EngineError::invalid_input("KV internal index is outside cache layout"));
        }
        let layer_width = self.layer_width()?;
        if layer_element >= layer_width {
            return Err(EngineError::invalid_input("KV layer element is outside cache layout"));
        }
        usize::try_from(side)
            .ok()
            .and_then(|value| value.checked_mul(usize::try_from(self.plan.layer_count()).ok()?))
            .and_then(|value| value.checked_add(usize::try_from(layer).ok()?))
            .and_then(|value| value.checked_mul(usize::try_from(self.plan.token_capacity()).ok()?))
            .and_then(|value| value.checked_add(usize::try_from(position).ok()?))
            .and_then(|value| value.checked_mul(layer_width))
            .and_then(|value| value.checked_add(layer_element))
            .ok_or_else(|| EngineError::resource_exhaustion("KV element index overflows"))
    }

    fn lock_state(&self) -> Result<MutexGuard<'_, CacheState>, EngineError> {
        self.state
            .lock()
            .map_err(|_| EngineError::internal_invariant("KV-cache lifecycle lock is poisoned"))
    }
}

/// In-progress all-layer mutation for one engine-selected token position.
#[derive(Debug)]
pub struct KvTokenWriter<'a> {
    cache: &'a ContiguousKvCache,
    state: MutexGuard<'a, CacheState>,
    handle: SequenceHandle,
    position: u32,
    written: Vec<bool>,
    committed: bool,
}

impl KvTokenWriter<'_> {
    /// Reads one scalar from committed state or the current fully written layer.
    /// Coordinates are semantic cache dimensions, never client-provided byte offsets.
    pub fn read_key_value(
        &self,
        layer: u32,
        position: u32,
        head: u32,
        dimension: u32,
    ) -> Result<f32, EngineError> {
        self.read_value_at(0, layer, position, head, dimension)
    }

    /// Reads one value scalar from committed state or the current fully written layer.
    pub fn read_cached_value(
        &self,
        layer: u32,
        position: u32,
        head: u32,
        dimension: u32,
    ) -> Result<f32, EngineError> {
        self.read_value_at(1, layer, position, head, dimension)
    }

    /// Writes one complete layer K/V pair at the implicit next position.
    pub fn write_layer(
        &mut self,
        layer: u32,
        key: &[f32],
        value: &[f32],
    ) -> Result<(), EngineError> {
        let layer_index = usize::try_from(layer)
            .map_err(|_| EngineError::invalid_input("KV layer index exceeds usize"))?;
        if layer >= self.cache.plan.layer_count()
            || self.written.get(layer_index).copied().unwrap_or(true)
        {
            return Err(EngineError::invalid_input("KV layer is invalid or already written"));
        }
        let width = self.cache.layer_width()?;
        if key.len() != width || value.len() != width {
            return Err(EngineError::invalid_input("KV layer value count is invalid"));
        }
        if key.iter().chain(value).any(|element| !element.is_finite()) {
            return Err(EngineError::invalid_input("KV cache rejects non-finite input"));
        }
        #[cfg(test)]
        if self.state.fail_extension_after_layer == Some(layer) {
            return Err(EngineError::backend_failure("injected KV extension failure"));
        }
        let dtype = self.cache.plan.dtype();
        let entry = sequence_mut(&mut self.state, self.handle)?;
        for (side, source) in [(0_u32, key), (1_u32, value)] {
            for (index, element) in source.iter().copied().enumerate() {
                let offset = self.cache.element_offset(side, layer, self.position, index)?;
                match dtype {
                    DType::F32 => {
                        entry.storage.set_element_bytes(offset, &element.to_le_bytes())?;
                    }
                    DType::F16 => {
                        entry
                            .storage
                            .set_element_bytes(offset, &f32_to_f16_bits(element).to_le_bytes())?;
                    }
                    DType::BFloat16 => {
                        entry
                            .storage
                            .set_element_bytes(offset, &f32_to_bf16_bits(element).to_le_bytes())?;
                    }
                    _ => {
                        return Err(EngineError::internal_invariant(
                            "validated KV dtype became unsupported",
                        ));
                    }
                }
            }
        }
        self.written[layer_index] = true;
        Ok(())
    }

    /// Atomically exposes the position only after every layer is written.
    pub fn commit(mut self) -> Result<(), EngineError> {
        if self.written.iter().any(|written| !written) {
            return Err(EngineError::invalid_input("KV token is missing one or more layers"));
        }
        let entry = sequence_mut(&mut self.state, self.handle)?;
        entry.token_count = entry
            .token_count
            .checked_add(1)
            .ok_or_else(|| EngineError::resource_exhaustion("KV token count overflows"))?;
        self.committed = true;
        Ok(())
    }

    fn read_value_at(
        &self,
        side: u32,
        layer: u32,
        position: u32,
        head: u32,
        dimension: u32,
    ) -> Result<f32, EngineError> {
        if head >= self.cache.plan.kv_head_count()
            || dimension >= self.cache.plan.head_dimension()
            || position > self.position
        {
            return Err(EngineError::invalid_input("KV read coordinate is outside sequence state"));
        }
        let layer_index = usize::try_from(layer)
            .map_err(|_| EngineError::invalid_input("KV layer index exceeds usize"))?;
        if layer >= self.cache.plan.layer_count()
            || (position == self.position
                && !self.written.get(layer_index).copied().unwrap_or(false))
        {
            return Err(EngineError::invalid_input("KV layer is not available at this position"));
        }
        let layer_element = usize::try_from(head)
            .ok()
            .and_then(|head| {
                head.checked_mul(usize::try_from(self.cache.plan.head_dimension()).ok()?)
            })
            .and_then(|offset| offset.checked_add(usize::try_from(dimension).ok()?))
            .ok_or_else(|| EngineError::resource_exhaustion("KV coordinate overflows"))?;
        let offset = self.cache.element_offset(side, layer, position, layer_element)?;
        let entry = sequence(&self.state, self.handle)?;
        entry.storage.numeric_f32_value(offset)
    }
}

impl Drop for KvTokenWriter<'_> {
    fn drop(&mut self) {
        if !self.committed {
            if let Ok(entry) = sequence_mut(&mut self.state, self.handle) {
                clear_and_cancel(entry);
            }
        }
    }
}

fn ensure_loaded(state: &CacheState) -> Result<(), EngineError> {
    if state.loaded {
        Ok(())
    } else {
        Err(EngineError::protocol_violation("model cache has been unloaded"))
    }
}

fn ensure_active(entry: &SequenceEntry) -> Result<(), EngineError> {
    if entry.status == SequenceStatus::Active {
        Ok(())
    } else {
        Err(EngineError::cancellation(Some(entry.request_id)))
    }
}

fn sequence(state: &CacheState, handle: SequenceHandle) -> Result<&SequenceEntry, EngineError> {
    let entry = state
        .sequences
        .get(&handle.sequence_id)
        .ok_or_else(|| EngineError::protocol_violation("sequence identifier is unknown"))?;
    if entry.generation != handle.generation {
        return Err(EngineError::protocol_violation("sequence handle is stale"));
    }
    Ok(entry)
}

fn sequence_mut(
    state: &mut CacheState,
    handle: SequenceHandle,
) -> Result<&mut SequenceEntry, EngineError> {
    let entry = state
        .sequences
        .get_mut(&handle.sequence_id)
        .ok_or_else(|| EngineError::protocol_violation("sequence identifier is unknown"))?;
    if entry.generation != handle.generation {
        return Err(EngineError::protocol_violation("sequence handle is stale"));
    }
    Ok(entry)
}

fn clear_and_cancel(entry: &mut SequenceEntry) {
    entry.storage.clear();
    entry.token_count = 0;
    entry.status = SequenceStatus::Cancelled;
}

#[cfg(test)]
mod tests {
    use super::*;
    use personalai_engine_types::ResourceBudget;
    use std::sync::Arc;
    use std::thread;

    fn kv_cache(dtype: DType, sequences: u64) -> ContiguousKvCache {
        let per_sequence = 2 * 2 * 4 * 2 * 2 * dtype.byte_width();
        let budget = ResourceBudget::new(per_sequence * sequences, per_sequence, 16).unwrap();
        let pool = MemoryPool::new(per_sequence * sequences).unwrap();
        let plan = KvCachePlan::new(2, 2, 2, 4, dtype, &budget).unwrap();
        ContiguousKvCache::new(
            ModelId::new("tiny").unwrap(),
            plan,
            u32::try_from(sequences).unwrap(),
            pool,
        )
        .unwrap()
    }

    fn extend(cache: &ContiguousKvCache, handle: SequenceHandle, base: f32) {
        let mut writer = cache.begin_token(handle).unwrap();
        writer.write_layer(0, &[base, 2.0, 3.0, 4.0], &[5.0, 6.0, 7.0, 8.0]).unwrap();
        writer.write_layer(1, &[9.0, 10.0, 11.0, 12.0], &[13.0, 14.0, 15.0, 16.0]).unwrap();
        writer.commit().unwrap();
    }

    #[test]
    fn writes_reads_layers_heads_and_positions() {
        for dtype in [DType::F32, DType::F16, DType::BFloat16] {
            let cache = kv_cache(dtype, 1);
            let handle = cache
                .create_sequence(RequestId::new(1).unwrap(), SequenceId::new(1).unwrap())
                .unwrap();
            extend(&cache, handle, 1.0);
            let mut key = [0.0; 4];
            let mut value = [0.0; 4];
            cache.read_key_into(handle, 1, 0, &mut key).unwrap();
            cache.read_value_into(handle, 1, 0, &mut value).unwrap();
            assert!(key.iter().zip([9.0, 10.0, 11.0, 12.0]).all(|(a, b)| (*a - b).abs() < 0.01));
            assert!(value.iter().zip([13.0, 14.0, 15.0, 16.0]).all(|(a, b)| (*a - b).abs() < 0.01));
        }
    }

    #[test]
    fn sequences_are_isolated_and_stale_handles_fail() {
        let cache = kv_cache(DType::F32, 2);
        let first =
            cache.create_sequence(RequestId::new(1).unwrap(), SequenceId::new(7).unwrap()).unwrap();
        let second =
            cache.create_sequence(RequestId::new(2).unwrap(), SequenceId::new(8).unwrap()).unwrap();
        extend(&cache, first, 21.0);
        assert_eq!(cache.token_count(second).unwrap(), 0);
        cache.destroy_sequence(first).unwrap();
        let replacement =
            cache.create_sequence(RequestId::new(3).unwrap(), SequenceId::new(7).unwrap()).unwrap();
        assert!(cache.token_count(first).is_err());
        assert_eq!(cache.token_count(replacement).unwrap(), 0);
        extend(&cache, replacement, 99.0);
        let mut key = [0.0; 4];
        cache.read_key_into(replacement, 0, 0, &mut key).unwrap();
        assert!((key[0] - 99.0).abs() < f32::EPSILON);
    }

    #[test]
    fn cancellation_reset_context_and_unload_fail_closed() {
        let cache = kv_cache(DType::F32, 1);
        let handle =
            cache.create_sequence(RequestId::new(1).unwrap(), SequenceId::new(1).unwrap()).unwrap();
        extend(&cache, handle, 1.0);
        cache.reset_sequence(handle).unwrap();
        assert_eq!(cache.token_count(handle).unwrap(), 0);
        for _ in 0..4 {
            extend(&cache, handle, 1.0);
        }
        assert!(cache.begin_token(handle).is_err());
        cache.cancel_sequence(handle).unwrap();
        assert!(cache.token_count(handle).is_err());
        assert!(cache.reset_sequence(handle).is_err());
        cache.unload_model().unwrap();
        assert!(
            cache.create_sequence(RequestId::new(2).unwrap(), SequenceId::new(2).unwrap()).is_err()
        );
    }

    #[test]
    fn dropped_or_failed_extension_clears_and_cancels() {
        let cache = kv_cache(DType::F32, 1);
        let handle =
            cache.create_sequence(RequestId::new(1).unwrap(), SequenceId::new(1).unwrap()).unwrap();
        {
            let mut writer = cache.begin_token(handle).unwrap();
            writer.write_layer(0, &[1.0; 4], &[2.0; 4]).unwrap();
        }
        assert!(cache.token_count(handle).is_err());

        let cache = kv_cache(DType::F32, 1);
        let handle =
            cache.create_sequence(RequestId::new(1).unwrap(), SequenceId::new(1).unwrap()).unwrap();
        cache.state.lock().unwrap().fail_extension_after_layer = Some(1);
        let mut writer = cache.begin_token(handle).unwrap();
        writer.write_layer(0, &[1.0; 4], &[2.0; 4]).unwrap();
        assert!(writer.write_layer(1, &[3.0; 4], &[4.0; 4]).is_err());
        drop(writer);
        assert!(cache.token_count(handle).is_err());
    }

    #[test]
    fn prefill_and_pressure_metrics_are_exact() {
        let cache = kv_cache(DType::F32, 1);
        let handle =
            cache.create_sequence(RequestId::new(1).unwrap(), SequenceId::new(1).unwrap()).unwrap();
        cache.prefill(handle, 2, &[1.0; 16], &[2.0; 16]).unwrap();
        assert_eq!(cache.token_count(handle).unwrap(), 2);
        let metrics = cache.metrics().unwrap();
        assert_eq!(metrics.reserved_bytes, cache.sequence_allocation_bytes());
        assert_eq!(metrics.used_bytes, cache.prefill_bytes(2).unwrap());
        assert!(!cache.can_admit_sequence().unwrap());
    }

    #[test]
    fn concurrent_lifecycle_stress_remains_isolated() {
        let cache = Arc::new(kv_cache(DType::F32, 8));
        let mut threads = Vec::new();
        for id in 1..=8_u64 {
            let cache = Arc::clone(&cache);
            threads.push(thread::spawn(move || {
                for cycle in 0..32_u128 {
                    let handle = cache
                        .create_sequence(
                            RequestId::new(u128::from(id) * 100 + cycle + 1).unwrap(),
                            SequenceId::new(id).unwrap(),
                        )
                        .unwrap();
                    let base = f32::from(u16::try_from(id).unwrap());
                    extend(&cache, handle, base);
                    if cycle % 2 == 0 {
                        cache.cancel_sequence(handle).unwrap();
                    }
                    cache.destroy_sequence(handle).unwrap();
                }
            }));
        }
        for thread in threads {
            thread.join().unwrap();
        }
        assert_eq!(cache.metrics().unwrap().sequence_count, 0);
    }

    #[test]
    fn malformed_limits_and_locations_are_rejected() {
        let cache = kv_cache(DType::F32, 1);
        let handle =
            cache.create_sequence(RequestId::new(1).unwrap(), SequenceId::new(1).unwrap()).unwrap();
        assert!(
            cache.create_sequence(RequestId::new(2).unwrap(), SequenceId::new(1).unwrap()).is_err()
        );
        let mut output = [0.0; 4];
        assert!(cache.read_key_into(handle, 0, 0, &mut output).is_err());
        assert!(cache.prefill(handle, 5, &[0.0; 40], &[0.0; 40]).is_err());
        assert!(cache.prefill(handle, 1, &[f32::NAN; 8], &[0.0; 8]).is_err());
    }

    #[test]
    fn allocation_pressure_and_generation_wrap_leave_no_partial_sequence() {
        let cache = kv_cache(DType::F32, 1);
        cache.create_sequence(RequestId::new(1).unwrap(), SequenceId::new(1).unwrap()).unwrap();
        assert!(
            cache.create_sequence(RequestId::new(2).unwrap(), SequenceId::new(2).unwrap()).is_err()
        );
        assert_eq!(cache.metrics().unwrap().sequence_count, 1);

        let cache = kv_cache(DType::F32, 1);
        cache.state.lock().unwrap().next_generation = u64::MAX;
        assert!(
            cache.create_sequence(RequestId::new(1).unwrap(), SequenceId::new(1).unwrap()).is_err()
        );
        assert_eq!(cache.metrics().unwrap().sequence_count, 0);

        let per_sequence = 2 * 2 * 4 * 2 * 2 * DType::F32.byte_width();
        let budget = ResourceBudget::new(per_sequence * 2, per_sequence, 2).unwrap();
        let plan = KvCachePlan::new(2, 2, 2, 4, DType::F32, &budget).unwrap();
        let cache = ContiguousKvCache::new(
            ModelId::new("limited").unwrap(),
            plan,
            1,
            MemoryPool::new(per_sequence * 2).unwrap(),
        )
        .unwrap();
        cache.create_sequence(RequestId::new(1).unwrap(), SequenceId::new(1).unwrap()).unwrap();
        assert!(!cache.can_admit_sequence().unwrap());
        assert!(
            cache.create_sequence(RequestId::new(2).unwrap(), SequenceId::new(2).unwrap()).is_err()
        );
    }
}
