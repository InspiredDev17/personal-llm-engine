//! First-party Apple Metal graph backend.
//!
//! Shaders are compiled into an embedded metallib at build time. The safe Rust
//! facade selects only a closed kernel enum, performs checked shape/byte
//! validation, accounts all resident GPU buffers, records every transfer, and
//! falls back to the scalar oracle only for unsupported validated operations.

#[cfg(personalai_metal_native)]
mod ffi;
#[cfg(personalai_metal_native)]
mod kv_cache;
#[cfg(personalai_metal_native)]
pub use kv_cache::{MetalKvCache, MetalKvCacheMetrics, MetalKvSequenceHandle};

use personalai_engine_cpu_reference::{Backend, CpuReferenceBackend, Operation};
use personalai_engine_memory::MemoryPool;
#[cfg(personalai_metal_native)]
use personalai_engine_memory::{MemoryOwner, MemoryReservation};
#[cfg(personalai_metal_native)]
use personalai_engine_tensor::TensorShape;
use personalai_engine_tensor::{Device, Tensor};
#[cfg(personalai_metal_native)]
use personalai_engine_types::DType;
use personalai_engine_types::{BackendId, EngineError};
use std::sync::Arc;
#[cfg(personalai_metal_native)]
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};

/// Stable identifier for the first-party GPU backend.
pub const METAL_BACKEND_ID: &str = "metal";
/// Maximum retained reusable device allocations.
pub const MAX_METAL_POOL_BUFFERS: usize = 64;
/// Required buffer alignment at the bridge boundary.
pub const METAL_BUFFER_ALIGNMENT: usize = 256;
/// F32 differential tolerance for GPU reductions.
pub const METAL_F32_TOLERANCE: f32 = 5.0e-5;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
#[cfg(personalai_metal_native)]
enum Kernel {
    Copy = 0,
    Add = 1,
    Multiply = 2,
    Scalar = 3,
    MatVec = 4,
    MatMul = 5,
    F16MatVec = 6,
    F16MatMul = 7,
    RmsNorm = 8,
    Softmax = 9,
    Silu = 10,
    SwiGlu = 11,
    Rope = 12,
    Embedding = 13,
    AttentionScores = 14,
    CausalMask = 15,
    AttentionValues = 16,
    Q8MatVec = 17,
    Q4MatVec = 18,
}

/// Cooperative request-local cancellation flag.
#[derive(Clone, Debug, Default)]
pub struct MetalCancellation(Arc<AtomicBool>);
impl MetalCancellation {
    /// Creates a live flag.
    pub fn new() -> Self {
        Self::default()
    }
    /// Requests cancellation.
    pub fn cancel(&self) {
        self.0.store(true, Ordering::Release);
    }
    fn check(&self) -> Result<(), EngineError> {
        if self.0.load(Ordering::Acquire) { Err(EngineError::cancellation(None)) } else { Ok(()) }
    }
}

/// Runtime-reported Apple GPU capabilities.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MetalDeviceInfo {
    /// Registry identity.
    pub registry_id: u64,
    /// Human-readable local device name.
    pub name: String,
    /// Recommended working-set ceiling.
    pub recommended_working_set_bytes: u64,
    /// Maximum single buffer size.
    pub maximum_buffer_bytes: u64,
    /// Unified-memory report.
    pub unified_memory: bool,
    /// F16 arithmetic support.
    pub supports_f16: bool,
}
/// Transfer, allocation, and dispatch counters with no prompt or model data.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MetalStatistics {
    /// Current native bytes.
    pub current_allocated_bytes: u64,
    /// Peak native bytes.
    pub peak_allocated_bytes: u64,
    /// Total allocations.
    pub allocation_count: u64,
    /// Explicit upload bytes.
    pub host_to_device_bytes: u64,
    /// Explicit download bytes.
    pub device_to_host_bytes: u64,
    /// Completed dispatch count.
    pub dispatch_count: u64,
    /// Failed command count.
    pub command_failures: u64,
    /// Currently pooled bytes.
    pub pooled_bytes: u64,
    /// Currently pooled buffers.
    pub pooled_buffers: u64,
}

#[cfg(personalai_metal_native)]
#[derive(Debug)]
struct Allocation {
    native: ffi::NativeBuffer,
    _reservation: MemoryReservation,
}
#[cfg(personalai_metal_native)]
struct Inner {
    context: Arc<ffi::NativeContext>,
    pool: MemoryPool,
    reusable: Mutex<Vec<Allocation>>,
    max_pool_buffers: usize,
}
#[cfg(personalai_metal_native)]
impl Drop for Inner {
    fn drop(&mut self) {
        if let Ok(mut buffers) = self.reusable.lock() {
            buffers.clear();
        }
        let _ = self.context.synchronize();
    }
}

/// Safe synchronous Metal graph backend with scalar fallback.
#[derive(Clone)]
pub struct MetalBackend {
    id: BackendId,
    scalar: CpuReferenceBackend,
    #[cfg(personalai_metal_native)]
    inner: Arc<Inner>,
}
impl std::fmt::Debug for MetalBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MetalBackend").field("id", &self.id).finish_non_exhaustive()
    }
}

impl MetalBackend {
    /// Enumerates documented Metal devices without creating a context.
    pub fn devices() -> Result<Vec<MetalDeviceInfo>, EngineError> {
        #[cfg(personalai_metal_native)]
        {
            let mut out = Vec::new();
            for index in 0..ffi::device_count() {
                let d = ffi::device_info(index)?;
                out.push(MetalDeviceInfo {
                    registry_id: d.registry_id,
                    name: d.name,
                    recommended_working_set_bytes: d.recommended_bytes,
                    maximum_buffer_bytes: d.maximum_buffer_bytes,
                    unified_memory: d.unified_memory,
                    supports_f16: d.f16,
                });
            }
            Ok(out)
        }
        #[cfg(not(personalai_metal_native))]
        {
            Err(EngineError::not_implemented("Metal backend requires macOS"))
        }
    }
    /// Creates one context, queue, closed pipeline set, and bounded device pool.
    pub fn new(
        device_index: usize,
        pool_limit_bytes: u64,
        max_pool_buffers: usize,
    ) -> Result<Self, EngineError> {
        if max_pool_buffers == 0 || max_pool_buffers > MAX_METAL_POOL_BUFFERS {
            return Err(EngineError::invalid_input("Metal buffer-pool count is outside bounds"));
        }
        #[cfg(personalai_metal_native)]
        {
            static LIBRARY: &[u8] =
                include_bytes!(concat!(env!("OUT_DIR"), "/personalai.metallib"));
            static DIGEST: &[u8] =
                include_bytes!(concat!(env!("OUT_DIR"), "/personalai.metallib.sha384"));
            if personalai_engine_crypto::hash_signed_content(LIBRARY).as_slice() != DIGEST {
                return Err(EngineError::signature_failure("embedded metallib identity mismatch"));
            }
            let context = ffi::create_context(device_index, LIBRARY)?;
            let id = BackendId::new(METAL_BACKEND_ID)?;
            let pool = MemoryPool::new_scoped(pool_limit_bytes, MemoryOwner::Backend(id.clone()))?;
            Ok(Self {
                id,
                scalar: CpuReferenceBackend::new()?,
                inner: Arc::new(Inner {
                    context,
                    pool,
                    reusable: Mutex::new(Vec::new()),
                    max_pool_buffers,
                }),
            })
        }
        #[cfg(not(personalai_metal_native))]
        {
            let _ = (device_index, pool_limit_bytes);
            Err(EngineError::not_implemented("Metal backend requires macOS"))
        }
    }
    /// Returns the exact backend identifier.
    pub const fn id(&self) -> &BackendId {
        &self.id
    }
    /// Returns native and pool statistics.
    pub fn statistics(&self) -> Result<MetalStatistics, EngineError> {
        #[cfg(personalai_metal_native)]
        {
            let n = self.inner.context.statistics()?;
            let buffers = self
                .inner
                .reusable
                .lock()
                .map_err(|_| EngineError::internal_invariant("Metal pool lock poisoned"))?;
            let pooled_bytes = buffers.iter().try_fold(0u64, |sum, b| {
                sum.checked_add(b.native.len() as u64)
                    .ok_or_else(|| EngineError::resource_exhaustion("Metal pool metrics overflow"))
            })?;
            Ok(MetalStatistics {
                current_allocated_bytes: n.current_bytes,
                peak_allocated_bytes: n.peak_bytes,
                allocation_count: n.allocations,
                host_to_device_bytes: n.host_to_device_bytes,
                device_to_host_bytes: n.device_to_host_bytes,
                dispatch_count: n.dispatches,
                command_failures: n.command_failures,
                pooled_bytes,
                pooled_buffers: buffers.len() as u64,
            })
        }
        #[cfg(not(personalai_metal_native))]
        {
            Err(EngineError::not_implemented("Metal backend requires macOS"))
        }
    }
    /// Releases every idle pooled buffer after synchronization.
    pub fn reclaim(&self) -> Result<u64, EngineError> {
        #[cfg(personalai_metal_native)]
        {
            self.inner.context.synchronize()?;
            let mut pool = self
                .inner
                .reusable
                .lock()
                .map_err(|_| EngineError::internal_invariant("Metal pool lock poisoned"))?;
            let bytes = pool.iter().map(|b| b.native.len() as u64).sum();
            pool.clear();
            Ok(bytes)
        }
        #[cfg(not(personalai_metal_native))]
        {
            Err(EngineError::not_implemented("Metal backend requires macOS"))
        }
    }

    /// Multiplies an explicitly little-endian IEEE F16 matrix and vector.
    pub fn f16_matrix_vector(
        &self,
        matrix: &[u16],
        rows: u32,
        columns: u32,
        vector: &[u16],
        cancel: &MetalCancellation,
    ) -> Result<Vec<f32>, EngineError> {
        #[cfg(personalai_metal_native)]
        {
            let elements = checked_usize_product(rows, columns, "Metal F16 matrix size overflows")?;
            if matrix.len() != elements
                || vector.len()
                    != usize::try_from(columns).map_err(|_| {
                        EngineError::resource_exhaustion("Metal F16 vector size overflows")
                    })?
            {
                return Err(EngineError::invalid_input("Metal F16 matrix-vector shape is invalid"));
            }
            let matrix_bytes = u16_bytes(matrix);
            let vector_bytes = u16_bytes(vector);
            let output_len =
                usize::try_from(rows).ok().and_then(|v| v.checked_mul(4)).ok_or_else(|| {
                    EngineError::resource_exhaustion("Metal F16 output size overflows")
                })?;
            self.raw(
                Kernel::F16MatVec,
                &[&matrix_bytes, &vector_bytes],
                output_len,
                ffi::RawParams { rows, columns, ..Default::default() },
                [rows, 1, 1],
                cancel,
            )
            .map(|bytes| bytes_to_f32(&bytes))?
        }
        #[cfg(not(personalai_metal_native))]
        {
            let _ = (matrix, rows, columns, vector, cancel);
            Err(EngineError::not_implemented("Metal backend requires macOS"))
        }
    }

    /// Multiplies two explicitly little-endian IEEE F16 rank-two matrices.
    pub fn f16_matrix_matrix(
        &self,
        left: &[u16],
        rows: u32,
        inner: u32,
        right: &[u16],
        columns: u32,
        cancel: &MetalCancellation,
    ) -> Result<Vec<f32>, EngineError> {
        #[cfg(personalai_metal_native)]
        {
            if left.len()
                != checked_usize_product(rows, inner, "Metal F16 left matrix size overflows")?
                || right.len()
                    != checked_usize_product(
                        inner,
                        columns,
                        "Metal F16 right matrix size overflows",
                    )?
            {
                return Err(EngineError::invalid_input("Metal F16 matrix-matrix shape is invalid"));
            }
            let left_bytes = u16_bytes(left);
            let right_bytes = u16_bytes(right);
            let output_len =
                checked_usize_product(rows, columns, "Metal F16 output size overflows")?
                    .checked_mul(4)
                    .ok_or_else(|| {
                        EngineError::resource_exhaustion("Metal F16 output bytes overflow")
                    })?;
            self.raw(
                Kernel::F16MatMul,
                &[&left_bytes, &right_bytes],
                output_len,
                ffi::RawParams { rows, columns, inner, ..Default::default() },
                [columns, rows, 1],
                cancel,
            )
            .map(|bytes| bytes_to_f32(&bytes))?
        }
        #[cfg(not(personalai_metal_native))]
        {
            let _ = (left, rows, inner, right, columns, cancel);
            Err(EngineError::not_implemented("Metal backend requires macOS"))
        }
    }

    /// Executes the reviewed AIPQ8 or AIPQ4 block matrix-vector kernel.
    #[allow(clippy::too_many_arguments)]
    pub fn quantized_matrix_vector(
        &self,
        q4: bool,
        scales: &[f32],
        codes: &[u8],
        rows: u32,
        columns: u32,
        vector: &[f32],
        cancel: &MetalCancellation,
    ) -> Result<Vec<f32>, EngineError> {
        #[cfg(personalai_metal_native)]
        {
            let blocks_per_row = usize::try_from(columns)
                .map_err(|_| EngineError::resource_exhaustion("Metal quantized columns overflow"))?
                .checked_add(31)
                .ok_or_else(|| {
                    EngineError::resource_exhaustion("Metal quantized block count overflows")
                })?
                / 32;
            let blocks = usize::try_from(rows)
                .map_err(|_| EngineError::resource_exhaustion("Metal quantized rows overflow"))?
                .checked_mul(blocks_per_row)
                .ok_or_else(|| {
                    EngineError::resource_exhaustion("Metal quantized block count overflows")
                })?;
            let code_len = blocks.checked_mul(if q4 { 16 } else { 32 }).ok_or_else(|| {
                EngineError::resource_exhaustion("Metal quantized packed size overflows")
            })?;
            if scales.len() != blocks
                || codes.len() != code_len
                || vector.len()
                    != usize::try_from(columns).map_err(|_| {
                        EngineError::resource_exhaustion("Metal quantized vector size overflows")
                    })?
                || scales.iter().any(|v| !v.is_finite() || *v < 0.0)
                || vector.iter().any(|v| !v.is_finite())
            {
                return Err(EngineError::invalid_input(
                    "Metal quantized matrix-vector input is invalid",
                ));
            }
            let scale_bytes = f32_bytes(scales);
            let vector_bytes = f32_bytes(vector);
            let output_len =
                usize::try_from(rows).ok().and_then(|v| v.checked_mul(4)).ok_or_else(|| {
                    EngineError::resource_exhaustion("Metal quantized output size overflows")
                })?;
            let kernel = if q4 { Kernel::Q4MatVec } else { Kernel::Q8MatVec };
            self.raw(
                kernel,
                &[&scale_bytes, codes, &vector_bytes],
                output_len,
                ffi::RawParams {
                    rows,
                    columns,
                    block_count: u32::try_from(blocks).map_err(|_| {
                        EngineError::resource_exhaustion("Metal quantized block count exceeds u32")
                    })?,
                    ..Default::default()
                },
                [rows, 1, 1],
                cancel,
            )
            .map(|bytes| bytes_to_f32(&bytes))?
        }
        #[cfg(not(personalai_metal_native))]
        {
            let _ = (q4, scales, codes, rows, columns, vector, cancel);
            Err(EngineError::not_implemented("Metal backend requires macOS"))
        }
    }

    #[cfg(personalai_metal_native)]
    fn acquire(&self, length: usize) -> Result<Allocation, EngineError> {
        let mut pool = self
            .inner
            .reusable
            .lock()
            .map_err(|_| EngineError::internal_invariant("Metal pool lock poisoned"))?;
        if let Some(index) = pool.iter().position(|b| b.native.len() == length) {
            return Ok(pool.swap_remove(index));
        }
        drop(pool);
        let reservation = self.inner.pool.reserve(length as u64)?;
        let native = self.inner.context.allocate(length, METAL_BUFFER_ALIGNMENT, false)?;
        Ok(Allocation { native, _reservation: reservation })
    }
    #[cfg(personalai_metal_native)]
    fn release(&self, allocation: Allocation) -> Result<(), EngineError> {
        allocation.native.clear()?;
        let mut pool = self
            .inner
            .reusable
            .lock()
            .map_err(|_| EngineError::internal_invariant("Metal pool lock poisoned"))?;
        if pool.len() < self.inner.max_pool_buffers {
            pool.push(allocation);
        }
        Ok(())
    }
    #[cfg(personalai_metal_native)]
    fn raw(
        &self,
        kernel: Kernel,
        inputs: &[&[u8]],
        output_len: usize,
        params: ffi::RawParams,
        grid: [u32; 3],
        cancel: &MetalCancellation,
    ) -> Result<Vec<u8>, EngineError> {
        cancel.check()?;
        let mut allocations = Vec::new();
        for input in inputs {
            let b = self.acquire(input.len())?;
            b.native.upload(input)?;
            allocations.push(b);
        }
        let output = self.acquire(output_len)?;
        allocations.push(output);
        let refs = allocations.iter().map(|b| &b.native).collect::<Vec<_>>();
        let result = self
            .inner
            .context
            .dispatch(kernel as i32, &refs, &params, grid)
            .and_then(|()| cancel.check())
            .and_then(|()| allocations.last().expect("output exists").native.download());
        let mut release_error = None;
        for allocation in allocations {
            if let Err(error) = self.release(allocation) {
                release_error = Some(error);
            }
        }
        match (result, release_error) {
            (Ok(bytes), None) => Ok(bytes),
            (Err(error), _) | (Ok(_), Some(error)) => Err(error),
        }
    }
    #[cfg(personalai_metal_native)]
    fn tensor_from_bytes(
        shape: TensorShape,
        bytes: Vec<u8>,
        pool: &MemoryPool,
    ) -> Result<Tensor, EngineError> {
        if bytes
            .chunks_exact(4)
            .any(|v| !f32::from_le_bytes(v.try_into().expect("chunk")).is_finite())
        {
            return Err(EngineError::backend_failure("Metal output is non-finite"));
        }
        Tensor::from_bytes(DType::F32, shape, bytes, pool)
    }
    #[cfg(personalai_metal_native)]
    fn masked_tensor_from_bytes(
        shape: TensorShape,
        bytes: Vec<u8>,
        pool: &MemoryPool,
    ) -> Result<Tensor, EngineError> {
        if bytes.chunks_exact(4).any(|value| {
            let decoded = f32::from_le_bytes(value.try_into().expect("chunk"));
            decoded.is_nan() || decoded == f32::INFINITY
        }) {
            return Err(EngineError::backend_failure("Metal mask output is invalid"));
        }
        Tensor::from_bytes(DType::F32, shape, bytes, pool)
    }
}

impl Backend for MetalBackend {
    fn device(&self) -> Device {
        Device::Metal
    }
    #[allow(clippy::too_many_lines)]
    fn execute_operation(
        &self,
        operation: &Operation,
        inputs: &[&Tensor],
        pool: &MemoryPool,
    ) -> Result<Tensor, EngineError> {
        #[cfg(not(personalai_metal_native))]
        {
            return self.scalar.execute_operation(operation, inputs, pool);
        }
        #[cfg(personalai_metal_native)]
        {
            if inputs.iter().any(|t| {
                t.dtype() != DType::F32
                    || !t.metadata().layout().is_contiguous()
                    || !t.accounted_by(pool)
            }) {
                return self.scalar.execute_operation(operation, inputs, pool);
            }
            let cancel = MetalCancellation::new();
            let count = |t: &Tensor| {
                u32::try_from(t.shape().element_count()).map_err(|_| {
                    EngineError::resource_exhaustion("Metal element count exceeds u32")
                })
            };
            let run = |kernel, source: Vec<&[u8]>, shape: TensorShape, params, grid| {
                self.raw(
                    kernel,
                    &source,
                    usize::try_from(shape.byte_len(DType::F32)?).map_err(|_| {
                        EngineError::resource_exhaustion("Metal output is not addressable")
                    })?,
                    params,
                    grid,
                    &cancel,
                )
                .and_then(|b| Self::tensor_from_bytes(shape, b, pool))
            };
            match operation {
                Operation::Copy => {
                    let bytes = u32::try_from(inputs[0].as_bytes().len()).map_err(|_| {
                        EngineError::resource_exhaustion("Metal copy size exceeds u32")
                    })?;
                    run(
                        Kernel::Copy,
                        vec![inputs[0].as_bytes()],
                        inputs[0].shape().clone(),
                        ffi::RawParams { count: bytes, ..Default::default() },
                        [bytes, 1, 1],
                    )
                }
                Operation::Add | Operation::ResidualAdd
                    if inputs.len() == 2 && inputs[0].shape() == inputs[1].shape() =>
                {
                    let n = count(inputs[0])?;
                    run(
                        Kernel::Add,
                        vec![inputs[0].as_bytes(), inputs[1].as_bytes()],
                        inputs[0].shape().clone(),
                        ffi::RawParams { count: n, ..Default::default() },
                        [n, 1, 1],
                    )
                }
                Operation::Multiply
                    if inputs.len() == 2 && inputs[0].shape() == inputs[1].shape() =>
                {
                    let n = count(inputs[0])?;
                    run(
                        Kernel::Multiply,
                        vec![inputs[0].as_bytes(), inputs[1].as_bytes()],
                        inputs[0].shape().clone(),
                        ffi::RawParams { count: n, ..Default::default() },
                        [n, 1, 1],
                    )
                }
                Operation::ScalarMultiply(s) if s.is_finite() => {
                    let n = count(inputs[0])?;
                    run(
                        Kernel::Scalar,
                        vec![inputs[0].as_bytes()],
                        inputs[0].shape().clone(),
                        ffi::RawParams { count: n, scalar0: *s, ..Default::default() },
                        [n, 1, 1],
                    )
                }
                Operation::MatrixVector | Operation::LogitProjection
                    if inputs.len() == 2
                        && inputs[0].shape().rank() == 2
                        && inputs[1].shape().rank() == 1 =>
                {
                    let rows = u32dim(inputs[0], 0)?;
                    let columns = u32dim(inputs[0], 1)?;
                    run(
                        Kernel::MatVec,
                        vec![inputs[0].as_bytes(), inputs[1].as_bytes()],
                        TensorShape::new([u64::from(rows)])?,
                        ffi::RawParams { rows, columns, ..Default::default() },
                        [rows, 1, 1],
                    )
                }
                Operation::MatrixMatrix
                    if inputs.len() == 2
                        && inputs[0].shape().rank() == 2
                        && inputs[1].shape().rank() == 2 =>
                {
                    let rows = u32dim(inputs[0], 0)?;
                    let inner = u32dim(inputs[0], 1)?;
                    let columns = u32dim(inputs[1], 1)?;
                    run(
                        Kernel::MatMul,
                        vec![inputs[0].as_bytes(), inputs[1].as_bytes()],
                        TensorShape::new([u64::from(rows), u64::from(columns)])?,
                        ffi::RawParams { rows, columns, inner, ..Default::default() },
                        [columns, rows, 1],
                    )
                }
                Operation::EmbeddingLookup(token)
                    if inputs.len() == 1 && inputs[0].shape().rank() == 2 =>
                {
                    let rows = u32dim(inputs[0], 0)?;
                    let columns = u32dim(inputs[0], 1)?;
                    if *token >= rows {
                        return Err(EngineError::invalid_input(
                            "Metal embedding token is out of range",
                        ));
                    }
                    run(
                        Kernel::Embedding,
                        vec![inputs[0].as_bytes()],
                        TensorShape::new([u64::from(columns)])?,
                        ffi::RawParams { position: *token, columns, ..Default::default() },
                        [columns, 1, 1],
                    )
                }
                Operation::RmsNorm { epsilon } if inputs.len() == 2 => {
                    let columns = u32dim(inputs[0], inputs[0].shape().rank() - 1)?;
                    let total = count(inputs[0])?;
                    let rows = total / columns;
                    run(
                        Kernel::RmsNorm,
                        vec![inputs[0].as_bytes(), inputs[1].as_bytes()],
                        inputs[0].shape().clone(),
                        ffi::RawParams { rows, columns, scalar0: *epsilon, ..Default::default() },
                        [rows, 1, 1],
                    )
                }
                Operation::Softmax => {
                    let columns = u32dim(inputs[0], inputs[0].shape().rank() - 1)?;
                    let rows = count(inputs[0])? / columns;
                    run(
                        Kernel::Softmax,
                        vec![inputs[0].as_bytes()],
                        inputs[0].shape().clone(),
                        ffi::RawParams { rows, columns, ..Default::default() },
                        [rows, 1, 1],
                    )
                }
                Operation::Silu => {
                    let n = count(inputs[0])?;
                    run(
                        Kernel::Silu,
                        vec![inputs[0].as_bytes()],
                        inputs[0].shape().clone(),
                        ffi::RawParams { count: n, ..Default::default() },
                        [n, 1, 1],
                    )
                }
                Operation::SwiGlu
                    if inputs.len() == 2 && inputs[0].shape() == inputs[1].shape() =>
                {
                    let n = count(inputs[0])?;
                    run(
                        Kernel::SwiGlu,
                        vec![inputs[0].as_bytes(), inputs[1].as_bytes()],
                        inputs[0].shape().clone(),
                        ffi::RawParams { count: n, ..Default::default() },
                        [n, 1, 1],
                    )
                }
                Operation::Rope { position, theta } if inputs[0].shape().rank() == 2 => {
                    let heads = u32dim(inputs[0], 0)?;
                    let head_dimension = u32dim(inputs[0], 1)?;
                    run(
                        Kernel::Rope,
                        vec![inputs[0].as_bytes()],
                        inputs[0].shape().clone(),
                        ffi::RawParams {
                            position: *position,
                            head_dimension,
                            head_count: heads,
                            scalar0: *theta,
                            ..Default::default()
                        },
                        [heads * head_dimension / 2, 1, 1],
                    )
                }
                Operation::CausalMask { first_query_position } if inputs[0].shape().rank() == 2 => {
                    let rows = u32dim(inputs[0], 0)?;
                    let columns = u32dim(inputs[0], 1)?;
                    let shape = inputs[0].shape().clone();
                    let bytes = self.raw(
                        Kernel::CausalMask,
                        &[inputs[0].as_bytes()],
                        usize::try_from(shape.byte_len(DType::F32)?).map_err(|_| {
                            EngineError::resource_exhaustion("Metal mask output is not addressable")
                        })?,
                        ffi::RawParams {
                            rows,
                            columns,
                            query_count: rows,
                            first_query_position: *first_query_position,
                            ..Default::default()
                        },
                        [columns, rows, 1],
                        &cancel,
                    )?;
                    Self::masked_tensor_from_bytes(shape, bytes, pool)
                }
                Operation::GroupedQueryAttention { causal_position }
                    if inputs.len() == 3
                        && inputs[0].shape().rank() == 2
                        && inputs[1].shape().rank() == 3
                        && inputs[1].shape() == inputs[2].shape() =>
                {
                    let head_count = u32dim(inputs[0], 0)?;
                    let head_dimension = u32dim(inputs[0], 1)?;
                    let sequence_length = causal_position.checked_add(1).ok_or_else(|| {
                        EngineError::resource_exhaustion("Metal attention sequence overflows")
                    })?;
                    let kv_head_count = u32dim(inputs[1], 1)?;
                    if u64::from(sequence_length) > inputs[1].shape().dimensions()[0]
                        || kv_head_count == 0
                        || head_count % kv_head_count != 0
                        || u64::from(head_dimension) != inputs[1].shape().dimensions()[2]
                    {
                        return self.scalar.execute_operation(operation, inputs, pool);
                    }
                    #[allow(clippy::cast_precision_loss)]
                    let scale = 1.0 / (head_dimension as f32).sqrt();
                    let params = ffi::RawParams {
                        head_count,
                        head_dimension,
                        kv_head_count,
                        sequence_length,
                        scalar0: scale,
                        ..Default::default()
                    };
                    let score_len = checked_usize_product(
                        head_count,
                        sequence_length,
                        "Metal attention score size overflows",
                    )?
                    .checked_mul(4)
                    .ok_or_else(|| {
                        EngineError::resource_exhaustion("Metal attention score bytes overflow")
                    })?;
                    let scores = self.raw(
                        Kernel::AttentionScores,
                        &[inputs[0].as_bytes(), inputs[1].as_bytes()],
                        score_len,
                        params,
                        [sequence_length, head_count, 1],
                        &cancel,
                    )?;
                    let probabilities = self.raw(
                        Kernel::Softmax,
                        &[&scores],
                        score_len,
                        ffi::RawParams {
                            rows: head_count,
                            columns: sequence_length,
                            ..Default::default()
                        },
                        [head_count, 1, 1],
                        &cancel,
                    )?;
                    let output_len = checked_usize_product(
                        head_count,
                        head_dimension,
                        "Metal attention output size overflows",
                    )?
                    .checked_mul(4)
                    .ok_or_else(|| {
                        EngineError::resource_exhaustion("Metal attention output bytes overflow")
                    })?;
                    let output = self.raw(
                        Kernel::AttentionValues,
                        &[&probabilities, inputs[2].as_bytes()],
                        output_len,
                        params,
                        [head_dimension, head_count, 1],
                        &cancel,
                    )?;
                    Self::tensor_from_bytes(inputs[0].shape().clone(), output, pool)
                }
                _ => self.scalar.execute_operation(operation, inputs, pool),
            }
        }
    }
}
#[cfg(personalai_metal_native)]
fn u32dim(tensor: &Tensor, index: usize) -> Result<u32, EngineError> {
    u32::try_from(tensor.shape().dimensions()[index])
        .map_err(|_| EngineError::resource_exhaustion("Metal dimension exceeds u32"))
}

#[cfg(personalai_metal_native)]
fn checked_usize_product(
    left: u32,
    right: u32,
    detail: &'static str,
) -> Result<usize, EngineError> {
    usize::try_from(left)
        .ok()
        .and_then(|l| usize::try_from(right).ok().and_then(|r| l.checked_mul(r)))
        .ok_or_else(|| EngineError::resource_exhaustion(detail))
}
#[cfg(personalai_metal_native)]
fn u16_bytes(values: &[u16]) -> Vec<u8> {
    values.iter().flat_map(|value| value.to_le_bytes()).collect()
}
#[cfg(personalai_metal_native)]
fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|value| value.to_le_bytes()).collect()
}
#[cfg(personalai_metal_native)]
fn bytes_to_f32(bytes: &[u8]) -> Result<Vec<f32>, EngineError> {
    if bytes.len() % 4 != 0 {
        return Err(EngineError::backend_failure("Metal F32 output length is invalid"));
    }
    let values = bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("four-byte chunk")))
        .collect::<Vec<_>>();
    if values.iter().any(|value| !value.is_finite()) {
        return Err(EngineError::backend_failure("Metal output is non-finite"));
    }
    Ok(values)
}

/// Returns truthful compile-time availability without creating a device.
pub const fn native_compiled() -> bool {
    cfg!(personalai_metal_native)
}

#[cfg(all(test, personalai_metal_native))]
mod tests {
    use super::*;
    use personalai_engine_arm64::Arm64Backend;
    use personalai_engine_tensor::MutableTensor;
    use personalai_engine_testkit::compare_tensors;

    fn backend() -> Option<MetalBackend> {
        if MetalBackend::devices().ok()?.is_empty() {
            return None;
        }
        MetalBackend::new(0, 8 << 20, 8).ok()
    }

    fn tensor(pool: &MemoryPool, shape: impl IntoIterator<Item = u64>, values: &[f32]) -> Tensor {
        MutableTensor::from_f32_values(TensorShape::new(shape).unwrap(), values, pool).unwrap()
    }

    #[test]
    fn device_enumeration_context_identity_and_pool_metrics_are_real() {
        let devices = MetalBackend::devices().unwrap();
        if devices.is_empty() {
            return;
        }
        assert!(!devices[0].name.is_empty());
        assert!(devices[0].maximum_buffer_bytes > 0);
        let backend = backend().unwrap();
        assert_eq!(backend.device(), Device::Metal);
        assert_eq!(backend.id().as_str(), METAL_BACKEND_ID);
        assert!(native_compiled());
        let initial = backend.statistics().unwrap();
        assert_eq!(initial.current_allocated_bytes, 0);
        assert_eq!(backend.reclaim().unwrap(), 0);
    }

    #[test]
    fn graph_kernels_match_scalar_and_arm64_oracle() {
        let pool = MemoryPool::new(32 << 20).unwrap();
        let Some(metal) = backend() else { return };
        let scalar = CpuReferenceBackend::new().unwrap();
        let arm64 = Arm64Backend::new().unwrap();
        let vector = tensor(&pool, [4], &[0.25, -0.5, 1.0, 2.0]);
        let other = tensor(&pool, [4], &[1.5, 2.0, -0.5, 0.25]);
        let matrix = tensor(
            &pool,
            [3, 4],
            &[1.0, 0.5, -0.25, 2.0, -1.0, 1.0, 0.0, 0.5, 0.25, -0.5, 2.0, 1.0],
        );
        let embedding =
            tensor(&pool, [3, 4], &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0]);
        let weight = tensor(&pool, [4], &[1.0, 0.75, 1.25, 0.5]);
        let scores = tensor(&pool, [2, 4], &[1.0, 2.0, 3.0, 4.0, -1.0, 0.0, 1.0, 2.0]);
        let cases: Vec<(Operation, Vec<&Tensor>, f32)> = vec![
            (Operation::Copy, vec![&vector], 0.0),
            (Operation::Add, vec![&vector, &other], 0.0),
            (Operation::Multiply, vec![&vector, &other], 0.0),
            (Operation::ScalarMultiply(0.5), vec![&vector], 0.0),
            (Operation::MatrixVector, vec![&matrix, &vector], 2.0e-6),
            (Operation::EmbeddingLookup(2), vec![&embedding], 0.0),
            (Operation::RmsNorm { epsilon: 1.0e-5 }, vec![&vector, &weight], METAL_F32_TOLERANCE),
            (Operation::Softmax, vec![&scores], METAL_F32_TOLERANCE),
            (Operation::Silu, vec![&vector], METAL_F32_TOLERANCE),
            (Operation::SwiGlu, vec![&vector, &other], METAL_F32_TOLERANCE),
            (Operation::Rope { position: 3, theta: 10_000.0 }, vec![&scores], METAL_F32_TOLERANCE),
        ];
        for (operation, inputs, tolerance) in cases {
            let expected = scalar.execute_operation(&operation, &inputs, &pool).unwrap();
            let actual = metal.execute_operation(&operation, &inputs, &pool).unwrap();
            compare_tensors(&actual, &expected, tolerance, tolerance).unwrap();
            let arm = arm64.execute_operation(&operation, &inputs, &pool).unwrap();
            compare_tensors(&actual, &arm, tolerance, tolerance).unwrap();
        }
        let mask = Operation::CausalMask { first_query_position: 1 };
        let expected = scalar.execute_operation(&mask, &[&scores], &pool).unwrap();
        let actual = metal.execute_operation(&mask, &[&scores], &pool).unwrap();
        assert_eq!(actual.as_bytes(), expected.as_bytes());
        assert!(metal.statistics().unwrap().dispatch_count >= 12);
    }

    #[test]
    #[allow(clippy::cast_precision_loss)]
    fn grouped_query_attention_and_irregular_matrix_shapes_match_scalar() {
        let pool = MemoryPool::new(16 << 20).unwrap();
        let Some(metal) = backend() else { return };
        let scalar = CpuReferenceBackend::new().unwrap();
        let query =
            tensor(&pool, [4, 3], &[1.0, 0.0, 0.5, 0.0, 1.0, -0.5, 0.5, 0.5, 1.0, -1.0, 0.5, 0.25]);
        let key = tensor(
            &pool,
            [2, 2, 3],
            &[1.0, 0.0, 0.5, 0.0, 1.0, 0.5, 0.5, 0.25, 1.0, 1.0, -0.5, 0.0],
        );
        let value = tensor(
            &pool,
            [2, 2, 3],
            &[1.0, 2.0, 3.0, -1.0, 0.5, 1.0, 2.0, 0.0, -1.0, 0.25, 1.5, 2.5],
        );
        let operation = Operation::GroupedQueryAttention { causal_position: 1 };
        let expected =
            scalar.execute_operation(&operation, &[&query, &key, &value], &pool).unwrap();
        let actual = metal.execute_operation(&operation, &[&query, &key, &value], &pool).unwrap();
        compare_tensors(&actual, &expected, 1.0e-4, 1.0e-4).unwrap();

        let left = tensor(&pool, [3, 5], &(0..15).map(|n| n as f32 / 16.0).collect::<Vec<_>>());
        let right =
            tensor(&pool, [5, 7], &(0..35).map(|n| (n as f32 - 15.0) / 32.0).collect::<Vec<_>>());
        let expected =
            scalar.execute_operation(&Operation::MatrixMatrix, &[&left, &right], &pool).unwrap();
        let actual =
            metal.execute_operation(&Operation::MatrixMatrix, &[&left, &right], &pool).unwrap();
        compare_tensors(&actual, &expected, 1.0e-5, 1.0e-5).unwrap();
    }

    #[test]
    fn f16_and_quantized_kernels_have_known_outputs_and_validate_inputs() {
        let Some(backend) = backend() else { return };
        let cancel = MetalCancellation::new();
        let half = [0x3c00, 0x4000, 0x4200, 0x4400];
        let vector = [0x3c00, 0x3800];
        let output = backend.f16_matrix_vector(&half, 2, 2, &vector, &cancel).unwrap();
        assert_eq!(output, [2.0, 5.0]);
        let identity = [0x3c00, 0, 0, 0x3c00];
        assert_eq!(
            backend.f16_matrix_matrix(&half, 2, 2, &identity, 2, &cancel).unwrap(),
            [1.0, 2.0, 3.0, 4.0]
        );

        let mut q8 = [0_u8; 32];
        q8[0] = 1_u8;
        q8[1] = (-2_i8).to_ne_bytes()[0];
        let mut input = [0.0_f32; 32];
        input[0] = 2.0;
        input[1] = 3.0;
        assert_eq!(
            backend.quantized_matrix_vector(false, &[0.5], &q8, 1, 32, &input, &cancel).unwrap(),
            [-2.0]
        );
        let mut q4 = [0_u8; 16];
        q4[0] = 0xe1;
        assert_eq!(
            backend.quantized_matrix_vector(true, &[0.5], &q4, 1, 32, &input, &cancel).unwrap(),
            [-0.5]
        );

        cancel.cancel();
        assert!(matches!(
            backend.f16_matrix_vector(&half, 2, 2, &vector, &cancel),
            Err(EngineError::Cancellation(_))
        ));
        assert!(
            backend
                .quantized_matrix_vector(
                    false,
                    &[f32::NAN],
                    &q8,
                    1,
                    32,
                    &input,
                    &MetalCancellation::new()
                )
                .is_err()
        );
    }

    #[test]
    fn unsupported_dtype_falls_back_and_pool_reuse_is_zeroed() {
        let pool = MemoryPool::new(8 << 20).unwrap();
        let Some(metal) = backend() else { return };
        let source = tensor(&pool, [2], &[1.0, 2.0]);
        let f16 = CpuReferenceBackend::new()
            .unwrap()
            .execute_operation(&Operation::Convert(DType::F16), &[&source], &pool)
            .unwrap();
        let copied = metal.execute_operation(&Operation::Copy, &[&f16], &pool).unwrap();
        assert_eq!(copied.as_bytes(), f16.as_bytes());
        let first = metal.execute_operation(&Operation::Copy, &[&source], &pool).unwrap();
        let zero = tensor(&pool, [2], &[0.0, 0.0]);
        let second = metal.execute_operation(&Operation::Copy, &[&zero], &pool).unwrap();
        assert_ne!(first.as_bytes(), second.as_bytes());
        assert!(metal.statistics().unwrap().pooled_buffers > 0);
        assert!(metal.reclaim().unwrap() > 0);
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn metal_kv_cache_isolates_clears_cancels_and_unloads_sequences() {
        use personalai_engine_kv_cache::KvCachePlan;
        use personalai_engine_types::{ModelId, RequestId, ResourceBudget, SequenceId};

        let per_sequence = 2_u64 * 2 * 2 * 3 * 4 * 4;
        let budget = ResourceBudget::new(per_sequence * 2, per_sequence, 2).unwrap();
        let plan = KvCachePlan::new(2, 2, 3, 4, DType::F32, &budget).unwrap();
        let Some(backend) = backend() else { return };
        let cache =
            MetalKvCache::new(ModelId::new("tiny-metal").unwrap(), plan, 2, backend).unwrap();
        let first =
            cache.create_sequence(RequestId::new(1).unwrap(), SequenceId::new(1).unwrap()).unwrap();
        let second =
            cache.create_sequence(RequestId::new(2).unwrap(), SequenceId::new(2).unwrap()).unwrap();
        let keys = (1_u8..=12).map(f32::from).collect::<Vec<_>>();
        let values = (21_u8..=32).map(f32::from).collect::<Vec<_>>();
        cache.append_token(first, &keys, &values).unwrap();
        cache.append_token(second, &[9.0; 12], &[8.0; 12]).unwrap();
        let mut output = [0.0; 6];
        cache.read_key_into(first, 1, 0, &mut output).unwrap();
        assert_eq!(output, [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
        cache.read_key_into(second, 1, 0, &mut output).unwrap();
        assert_eq!(output, [9.0; 6]);
        let metrics = cache.metrics().unwrap();
        assert_eq!(metrics.sequence_count, 2);
        assert_eq!(metrics.reserved_bytes, per_sequence * 2);
        assert_eq!(metrics.used_bytes, per_sequence / 2);

        cache.cancel_sequence(first).unwrap();
        assert!(matches!(
            cache.read_key_into(first, 0, 0, &mut output),
            Err(EngineError::Cancellation(_))
        ));
        cache.reset_sequence(first).unwrap();
        assert!(cache.read_key_into(first, 0, 0, &mut output).is_err());
        cache.destroy_sequence(first).unwrap();
        assert!(cache.destroy_sequence(first).is_err());
        cache.unload_model().unwrap();
        assert!(cache.append_token(second, &keys, &values).is_err());
        assert_eq!(cache.metrics().unwrap().reserved_bytes, 0);
    }
}
