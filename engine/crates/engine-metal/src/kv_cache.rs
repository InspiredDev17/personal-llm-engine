//! Sequence-isolated contiguous Metal key/value storage.

use super::{Allocation, MetalBackend};
use personalai_engine_kv_cache::KvCachePlan;
use personalai_engine_tensor::{
    bf16_bits_to_f32, f16_bits_to_f32, f32_to_bf16_bits, f32_to_f16_bits,
};
use personalai_engine_types::{DType, EngineError, ModelId, RequestId, SequenceId};
use std::collections::BTreeMap;
use std::sync::{Mutex, MutexGuard};

/// Generation-checked ownership capability for one Metal KV sequence.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MetalKvSequenceHandle {
    sequence_id: SequenceId,
    generation: u64,
}

impl MetalKvSequenceHandle {
    /// Returns the scheduler-assigned sequence identity.
    pub const fn sequence_id(self) -> SequenceId {
        self.sequence_id
    }
}

/// Prompt-free scheduler and device-memory accounting snapshot.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MetalKvCacheMetrics {
    /// Live sequence allocations.
    pub sequence_count: u32,
    /// Active, non-cancelled sequences.
    pub active_sequence_count: u32,
    /// Device bytes reserved for live sequences.
    pub reserved_bytes: u64,
    /// Logical committed K/V bytes.
    pub used_bytes: u64,
    /// Highest live reservation observed.
    pub peak_reserved_bytes: u64,
}

#[derive(Debug)]
struct Sequence {
    request_id: RequestId,
    generation: u64,
    tokens: u32,
    cancelled: bool,
    allocation: Allocation,
}

#[derive(Debug)]
struct State {
    loaded: bool,
    next_generation: u64,
    peak_reserved_bytes: u64,
    sequences: BTreeMap<SequenceId, Sequence>,
}

/// One-model contiguous Metal KV cache with explicit sequence ownership.
#[derive(Debug)]
pub struct MetalKvCache {
    model_id: ModelId,
    plan: KvCachePlan,
    maximum_sequences: u32,
    backend: MetalBackend,
    state: Mutex<State>,
}

impl MetalKvCache {
    /// Creates an empty device cache for a previously validated model layout.
    pub fn new(
        model_id: ModelId,
        plan: KvCachePlan,
        maximum_sequences: u32,
        backend: MetalBackend,
    ) -> Result<Self, EngineError> {
        if maximum_sequences == 0
            || maximum_sequences > personalai_engine_types::MAX_CONCURRENT_SEQUENCES
        {
            return Err(EngineError::invalid_input(
                "Metal KV sequence limit is outside hard bounds",
            ));
        }
        Ok(Self {
            model_id,
            plan,
            maximum_sequences,
            backend,
            state: Mutex::new(State {
                loaded: true,
                next_generation: 1,
                peak_reserved_bytes: 0,
                sequences: BTreeMap::new(),
            }),
        })
    }

    /// Returns the verified model identity owning every allocation.
    pub const fn model_id(&self) -> &ModelId {
        &self.model_id
    }

    /// Allocates one zeroed, request-bound device sequence.
    pub fn create_sequence(
        &self,
        request_id: RequestId,
        sequence_id: SequenceId,
    ) -> Result<MetalKvSequenceHandle, EngineError> {
        let mut state = self.lock()?;
        ensure_loaded(&state)?;
        if state.sequences.contains_key(&sequence_id) {
            return Err(EngineError::protocol_violation(
                "Metal KV sequence identity is already active",
            ));
        }
        if u32::try_from(state.sequences.len())
            .map_err(|_| EngineError::resource_exhaustion("Metal KV sequence count overflows"))?
            >= self.maximum_sequences
        {
            return Err(EngineError::resource_exhaustion("Metal KV sequence limit is reached"));
        }
        let bytes = usize::try_from(self.plan.bytes())
            .map_err(|_| EngineError::resource_exhaustion("Metal KV size is not addressable"))?;
        let allocation = self.backend.acquire(bytes)?;
        allocation.native.clear()?;
        let generation = state.next_generation;
        state.next_generation = generation
            .checked_add(1)
            .ok_or_else(|| EngineError::resource_exhaustion("Metal KV generation wraps"))?;
        state.sequences.insert(
            sequence_id,
            Sequence { request_id, generation, tokens: 0, cancelled: false, allocation },
        );
        let reserved = self.reserved_bytes(&state)?;
        state.peak_reserved_bytes = state.peak_reserved_bytes.max(reserved);
        Ok(MetalKvSequenceHandle { sequence_id, generation })
    }

    /// Appends complete layer-major K and V values for exactly one position.
    pub fn append_token(
        &self,
        handle: MetalKvSequenceHandle,
        keys: &[f32],
        values: &[f32],
    ) -> Result<(), EngineError> {
        let width = self.layer_width()?;
        let expected = usize::try_from(self.plan.layer_count())
            .ok()
            .and_then(|layers| layers.checked_mul(width))
            .ok_or_else(|| EngineError::resource_exhaustion("Metal KV token size overflows"))?;
        if keys.len() != expected
            || values.len() != expected
            || keys.iter().chain(values).any(|value| !value.is_finite())
        {
            return Err(EngineError::invalid_input("Metal KV token data is invalid"));
        }
        let mut state = self.lock()?;
        ensure_loaded(&state)?;
        let sequence = sequence_mut(&mut state, handle)?;
        ensure_active(sequence)?;
        if sequence.tokens >= self.plan.token_capacity() {
            return Err(EngineError::resource_exhaustion("Metal KV context capacity is exhausted"));
        }
        let position = sequence.tokens;
        let encoded_keys = encode_values(keys, self.plan.dtype())?;
        let encoded_values = encode_values(values, self.plan.dtype())?;
        let result = self
            .write_side(sequence, 0, position, &encoded_keys)
            .and_then(|()| self.write_side(sequence, 1, position, &encoded_values));
        if let Err(error) = result {
            sequence.allocation.native.clear()?;
            sequence.cancelled = true;
            sequence.tokens = 0;
            return Err(error);
        }
        sequence.tokens = sequence
            .tokens
            .checked_add(1)
            .ok_or_else(|| EngineError::resource_exhaustion("Metal KV token count wraps"))?;
        Ok(())
    }

    /// Reads one committed layer key into an exact finite F32 destination.
    pub fn read_key_into(
        &self,
        handle: MetalKvSequenceHandle,
        layer: u32,
        position: u32,
        destination: &mut [f32],
    ) -> Result<(), EngineError> {
        self.read_side(handle, 0, layer, position, destination)
    }

    /// Reads one committed layer value into an exact finite F32 destination.
    pub fn read_value_into(
        &self,
        handle: MetalKvSequenceHandle,
        layer: u32,
        position: u32,
        destination: &mut [f32],
    ) -> Result<(), EngineError> {
        self.read_side(handle, 1, layer, position, destination)
    }

    /// Clears a sequence while retaining its ownership capability.
    pub fn reset_sequence(&self, handle: MetalKvSequenceHandle) -> Result<(), EngineError> {
        let mut state = self.lock()?;
        ensure_loaded(&state)?;
        let sequence = sequence_mut(&mut state, handle)?;
        sequence.allocation.native.clear()?;
        sequence.tokens = 0;
        sequence.cancelled = false;
        Ok(())
    }

    /// Clears and makes a sequence unusable until destruction.
    pub fn cancel_sequence(&self, handle: MetalKvSequenceHandle) -> Result<(), EngineError> {
        let mut state = self.lock()?;
        ensure_loaded(&state)?;
        let sequence = sequence_mut(&mut state, handle)?;
        sequence.allocation.native.clear()?;
        sequence.tokens = 0;
        sequence.cancelled = true;
        Ok(())
    }

    /// Clears and releases one exact generation-checked sequence.
    pub fn destroy_sequence(&self, handle: MetalKvSequenceHandle) -> Result<(), EngineError> {
        let mut state = self.lock()?;
        ensure_loaded(&state)?;
        sequence(&state, handle)?;
        let sequence = state
            .sequences
            .remove(&handle.sequence_id)
            .ok_or_else(|| EngineError::protocol_violation("Metal KV sequence is unknown"))?;
        sequence.allocation.native.clear()?;
        drop(sequence);
        Ok(())
    }

    /// Synchronizes, clears, and invalidates all model-owned sequences.
    pub fn unload_model(&self) -> Result<(), EngineError> {
        let mut state = self.lock()?;
        ensure_loaded(&state)?;
        self.backend.inner.context.synchronize()?;
        for sequence in state.sequences.values() {
            sequence.allocation.native.clear()?;
        }
        state.sequences.clear();
        state.loaded = false;
        Ok(())
    }

    /// Returns bounded pressure metrics without content.
    pub fn metrics(&self) -> Result<MetalKvCacheMetrics, EngineError> {
        let state = self.lock()?;
        let sequence_count = u32::try_from(state.sequences.len())
            .map_err(|_| EngineError::resource_exhaustion("Metal KV metrics overflow"))?;
        let active_sequence_count =
            u32::try_from(state.sequences.values().filter(|sequence| !sequence.cancelled).count())
                .map_err(|_| {
                    EngineError::resource_exhaustion("Metal KV active metrics overflow")
                })?;
        let used_bytes = state.sequences.values().try_fold(0_u64, |total, sequence| {
            self.token_bytes()?
                .checked_mul(u64::from(sequence.tokens))
                .and_then(|bytes| total.checked_add(bytes))
                .ok_or_else(|| EngineError::resource_exhaustion("Metal KV used bytes overflow"))
        })?;
        Ok(MetalKvCacheMetrics {
            sequence_count,
            active_sequence_count,
            reserved_bytes: self.reserved_bytes(&state)?,
            used_bytes,
            peak_reserved_bytes: state.peak_reserved_bytes,
        })
    }

    fn read_side(
        &self,
        handle: MetalKvSequenceHandle,
        side: u32,
        layer: u32,
        position: u32,
        destination: &mut [f32],
    ) -> Result<(), EngineError> {
        let width = self.layer_width()?;
        if destination.len() != width {
            return Err(EngineError::invalid_input("Metal KV read destination length is invalid"));
        }
        let state = self.lock()?;
        ensure_loaded(&state)?;
        let sequence = sequence(&state, handle)?;
        ensure_active(sequence)?;
        if layer >= self.plan.layer_count() || position >= sequence.tokens {
            return Err(EngineError::invalid_input("Metal KV read index is invalid"));
        }
        let length = width
            .checked_mul(byte_width(self.plan.dtype())?)
            .ok_or_else(|| EngineError::resource_exhaustion("Metal KV read size overflows"))?;
        let bytes = sequence
            .allocation
            .native
            .download_range(self.offset(side, layer, position)?, length)?;
        decode_values(&bytes, self.plan.dtype(), destination)
    }

    fn write_side(
        &self,
        sequence: &Sequence,
        side: u32,
        position: u32,
        bytes: &[u8],
    ) -> Result<(), EngineError> {
        let width_bytes = self
            .layer_width()?
            .checked_mul(byte_width(self.plan.dtype())?)
            .ok_or_else(|| EngineError::resource_exhaustion("Metal KV layer bytes overflow"))?;
        for layer in 0..self.plan.layer_count() {
            let start = usize::try_from(layer)
                .ok()
                .and_then(|value| value.checked_mul(width_bytes))
                .ok_or_else(|| {
                    EngineError::resource_exhaustion("Metal KV source offset overflows")
                })?;
            let end = start.checked_add(width_bytes).ok_or_else(|| {
                EngineError::resource_exhaustion("Metal KV source range overflows")
            })?;
            sequence
                .allocation
                .native
                .upload_range(self.offset(side, layer, position)?, &bytes[start..end])?;
        }
        Ok(())
    }

    fn offset(&self, side: u32, layer: u32, position: u32) -> Result<usize, EngineError> {
        if side > 1 || layer >= self.plan.layer_count() || position >= self.plan.token_capacity() {
            return Err(EngineError::invalid_input("Metal KV index is invalid"));
        }
        let layer_width = u64::try_from(self.layer_width()?)
            .map_err(|_| EngineError::resource_exhaustion("Metal KV layer width overflows"))?;
        let element = u64::from(side)
            .checked_mul(u64::from(self.plan.layer_count()))
            .and_then(|value| value.checked_add(u64::from(layer)))
            .and_then(|value| value.checked_mul(u64::from(self.plan.token_capacity())))
            .and_then(|value| value.checked_add(u64::from(position)))
            .and_then(|value| value.checked_mul(layer_width))
            .and_then(|value| value.checked_mul(self.plan.dtype().byte_width()))
            .ok_or_else(|| EngineError::resource_exhaustion("Metal KV byte offset overflows"))?;
        usize::try_from(element)
            .map_err(|_| EngineError::resource_exhaustion("Metal KV offset is not addressable"))
    }

    fn layer_width(&self) -> Result<usize, EngineError> {
        usize::try_from(self.plan.kv_head_count())
            .ok()
            .and_then(|heads| heads.checked_mul(usize::try_from(self.plan.head_dimension()).ok()?))
            .ok_or_else(|| EngineError::resource_exhaustion("Metal KV layer width overflows"))
    }

    fn token_bytes(&self) -> Result<u64, EngineError> {
        self.plan
            .bytes()
            .checked_div(u64::from(self.plan.token_capacity()))
            .ok_or_else(|| EngineError::internal_invariant("Metal KV token capacity is zero"))
    }

    fn reserved_bytes(&self, state: &State) -> Result<u64, EngineError> {
        u64::try_from(state.sequences.len())
            .ok()
            .and_then(|count| count.checked_mul(self.plan.bytes()))
            .ok_or_else(|| EngineError::resource_exhaustion("Metal KV reserved bytes overflow"))
    }

    fn lock(&self) -> Result<MutexGuard<'_, State>, EngineError> {
        self.state
            .lock()
            .map_err(|_| EngineError::internal_invariant("Metal KV state lock is poisoned"))
    }
}

fn ensure_loaded(state: &State) -> Result<(), EngineError> {
    if state.loaded {
        Ok(())
    } else {
        Err(EngineError::backend_failure("Metal KV model is unloaded"))
    }
}

fn ensure_active(sequence: &Sequence) -> Result<(), EngineError> {
    if sequence.cancelled {
        Err(EngineError::cancellation(Some(sequence.request_id)))
    } else {
        Ok(())
    }
}

fn sequence(state: &State, handle: MetalKvSequenceHandle) -> Result<&Sequence, EngineError> {
    let value = state
        .sequences
        .get(&handle.sequence_id)
        .ok_or_else(|| EngineError::protocol_violation("Metal KV sequence is unknown"))?;
    if value.generation != handle.generation {
        return Err(EngineError::protocol_violation("Metal KV sequence handle is stale"));
    }
    Ok(value)
}

fn sequence_mut(
    state: &mut State,
    handle: MetalKvSequenceHandle,
) -> Result<&mut Sequence, EngineError> {
    let value = state
        .sequences
        .get_mut(&handle.sequence_id)
        .ok_or_else(|| EngineError::protocol_violation("Metal KV sequence is unknown"))?;
    if value.generation != handle.generation {
        return Err(EngineError::protocol_violation("Metal KV sequence handle is stale"));
    }
    Ok(value)
}

fn byte_width(dtype: DType) -> Result<usize, EngineError> {
    usize::try_from(dtype.byte_width())
        .map_err(|_| EngineError::resource_exhaustion("Metal KV data width overflows"))
}

fn encode_values(values: &[f32], dtype: DType) -> Result<Vec<u8>, EngineError> {
    let length = values
        .len()
        .checked_mul(byte_width(dtype)?)
        .ok_or_else(|| EngineError::resource_exhaustion("Metal KV encoding size overflows"))?;
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(length)
        .map_err(|_| EngineError::resource_exhaustion("Metal KV encoding allocation failed"))?;
    for value in values {
        match dtype {
            DType::F32 => bytes.extend_from_slice(&value.to_le_bytes()),
            DType::F16 => bytes.extend_from_slice(&f32_to_f16_bits(*value).to_le_bytes()),
            DType::BFloat16 => bytes.extend_from_slice(&f32_to_bf16_bits(*value).to_le_bytes()),
            _ => return Err(EngineError::not_implemented("Metal KV data type is unsupported")),
        }
    }
    Ok(bytes)
}

fn decode_values(bytes: &[u8], dtype: DType, destination: &mut [f32]) -> Result<(), EngineError> {
    let width = byte_width(dtype)?;
    if bytes.len()
        != destination
            .len()
            .checked_mul(width)
            .ok_or_else(|| EngineError::resource_exhaustion("Metal KV decoding size overflows"))?
    {
        return Err(EngineError::backend_failure("Metal KV downloaded length is invalid"));
    }
    for (index, destination) in destination.iter_mut().enumerate() {
        let start = index
            .checked_mul(width)
            .ok_or_else(|| EngineError::resource_exhaustion("Metal KV decode offset overflows"))?;
        *destination = match dtype {
            DType::F32 => f32::from_le_bytes(
                bytes[start..start + 4]
                    .try_into()
                    .map_err(|_| EngineError::backend_failure("Metal KV F32 data is truncated"))?,
            ),
            DType::F16 => f16_bits_to_f32(u16::from_le_bytes(
                bytes[start..start + 2]
                    .try_into()
                    .map_err(|_| EngineError::backend_failure("Metal KV F16 data is truncated"))?,
            )),
            DType::BFloat16 => {
                bf16_bits_to_f32(u16::from_le_bytes(bytes[start..start + 2].try_into().map_err(
                    |_| EngineError::backend_failure("Metal KV BF16 data is truncated"),
                )?))
            }
            _ => return Err(EngineError::not_implemented("Metal KV data type is unsupported")),
        };
        if !destination.is_finite() {
            return Err(EngineError::backend_failure("Metal KV contains non-finite data"));
        }
    }
    Ok(())
}
