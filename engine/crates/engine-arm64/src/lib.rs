//! First-party ARM64 graph backend.
//!
//! The backend accelerates the closed baseline graph operation set with
//! isolated ARM64 Advanced SIMD kernels and a bounded worker pool. Unsupported
//! dtype, layout, shape, size, or operation cases use the scalar correctness
//! implementation. It has no model, network, filesystem, dynamic-loading, or
//! kernel-selection input.

#[cfg(target_arch = "aarch64")]
mod intrinsics;
mod worker_pool;

pub use worker_pool::{CancellationToken, MAX_ARM64_WORKERS, WorkerPoolMetrics};

use personalai_engine_cpu_reference::{
    Backend, CpuReferenceBackend, MAX_CPU_REFERENCE_OPERATIONS, Operation,
};
use personalai_engine_memory::{MemoryOwner, MemoryPool, MemoryReservation};
use personalai_engine_model_format::Quantization;
use personalai_engine_quantization::QuantizedTensor;
use personalai_engine_tensor::{DType, Device, MutableTensor, Tensor, TensorRead, TensorShape};
use personalai_engine_types::{BackendId, EngineError};
use std::mem::size_of;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};
use worker_pool::WorkerPool;

/// Exact identifier for this trusted host backend.
pub const ARM64_BACKEND_ID: &str = "arm64";
/// Minimum contiguous elements used before vector dispatch wins over scalar setup.
pub const MIN_VECTOR_ELEMENTS: usize = 8;
/// Fixed row tile used by the first cache-aware matrix kernel.
pub const MATMUL_ROW_TILE: usize = 2;
/// Fixed column tile matching one 128-bit Advanced SIMD register.
pub const MATMUL_COLUMN_TILE: usize = 4;
/// Fixed logical value count in one canonical AIPQ v1 block.
const AIPQ_V1_BLOCK_VALUES: usize = 32;
/// Exact work ceiling for the reviewed Qwen2.5 1.5B tied output projection.
///
/// This does not widen the generic matrix-vector or scalar-reference bounds.
/// Only the engine-owned `LogitProjection` operation may use this ceiling.
const MAX_ARM64_LOGIT_PROJECTION_OPERATIONS: u64 = 233_373_696;
/// Fixed output-row quantum between cooperative cancellation polls.
const LOGIT_PROJECTION_ROW_CHUNK: usize = 64;
/// Fixed input quantum between cooperative cancellation polls.
const LOGIT_PROJECTION_COLUMN_CHUNK: usize = 256;

/// Trusted execution policy selected by local engine configuration.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Arm64ExecutionConfig {
    worker_count: usize,
    deterministic: bool,
}

impl Arm64ExecutionConfig {
    /// Validates a fixed worker count and deterministic policy.
    pub fn new(worker_count: usize, deterministic: bool) -> Result<Self, EngineError> {
        if worker_count == 0 || worker_count > MAX_ARM64_WORKERS {
            return Err(EngineError::invalid_input("ARM64 worker count is outside its hard bound"));
        }
        if deterministic && worker_count != 1 {
            return Err(EngineError::invalid_input(
                "deterministic ARM64 execution requires exactly one worker",
            ));
        }
        Ok(Self { worker_count, deterministic })
    }

    /// Returns the fixed worker count.
    pub const fn worker_count(self) -> usize {
        self.worker_count
    }

    /// Returns whether deterministic single-thread dispatch is required.
    pub const fn deterministic(self) -> bool {
        self.deterministic
    }
}

impl Default for Arm64ExecutionConfig {
    fn default() -> Self {
        Self { worker_count: 1, deterministic: true }
    }
}

/// Closed set of operations with an ARM64-specific implementation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Arm64Operation {
    /// Contiguous F32 addition.
    VectorAdd,
    /// Contiguous F32 multiplication.
    VectorMultiply,
    /// Contiguous F32 reduction.
    DotProduct,
    /// Row-major F32 matrix-vector multiplication.
    MatrixVector,
    /// Cache-tiled row-major F32 matrix multiplication.
    MatrixMatrix,
    /// Stable F32 RMS normalization.
    RmsNorm,
    /// Stable F32 softmax with vector normalization.
    Softmax,
    /// F32 SiLU with vector final multiplication.
    Silu,
    /// F32 SwiGLU with vector final multiplication.
    SwiGlu,
    /// F32 rotary positional encoding.
    Rope,
    /// Grouped-query attention dot products.
    AttentionScores,
    /// Grouped-query attention weighted accumulation.
    AttentionValues,
    /// IEEE F16-to-F32 storage conversion.
    F16ToF32,
    /// IEEE BF16-to-F32 storage conversion.
    Bf16ToF32,
    /// AIPQ8 block dequantization and matrix multiplication.
    AipQ8Matrix,
    /// AIPQ4 block dequantization and matrix multiplication.
    AipQ4Matrix,
}

/// Runtime capability evidence for this host and backend configuration.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Arm64Capabilities {
    advanced_simd: bool,
    fp16_arithmetic: bool,
    operations: Box<[Arm64Operation]>,
    worker_count: usize,
    deterministic: bool,
}

/// Trusted host-CPU backend selected at runtime with a scalar fallback.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HostCpuBackend {
    /// Portable correctness backend used when ARM64 acceleration is unavailable.
    Scalar(CpuReferenceBackend),
    /// Runtime-validated Apple Silicon/AArch64 backend.
    Arm64(Arm64Backend),
}

impl HostCpuBackend {
    /// Selects ARM64 only when its fixed runtime requirements are present.
    /// Any unsupported host safely receives the scalar correctness backend.
    pub fn select(config: Arm64ExecutionConfig) -> Result<Self, EngineError> {
        if cfg!(target_arch = "aarch64") && runtime_capabilities(config).advanced_simd() {
            Ok(Self::Arm64(Arm64Backend::with_config(config)?))
        } else {
            Ok(Self::Scalar(CpuReferenceBackend::new()?))
        }
    }

    /// Returns the exact selected backend identifier.
    pub fn backend_id(&self) -> &BackendId {
        match self {
            Self::Scalar(backend) => backend.id(),
            Self::Arm64(backend) => backend.id(),
        }
    }
}

impl Backend for HostCpuBackend {
    fn device(&self) -> Device {
        match self {
            Self::Scalar(backend) => backend.device(),
            Self::Arm64(backend) => backend.device(),
        }
    }

    fn execute_operation(
        &self,
        operation: &Operation,
        inputs: &[&Tensor],
        pool: &MemoryPool,
    ) -> Result<Tensor, EngineError> {
        match self {
            Self::Scalar(backend) => backend.execute_operation(operation, inputs, pool),
            Self::Arm64(backend) => backend.execute_operation(operation, inputs, pool),
        }
    }
}

impl Arm64Capabilities {
    /// Returns whether the mandatory AArch64 Advanced SIMD facility is available.
    pub const fn advanced_simd(&self) -> bool {
        self.advanced_simd
    }

    /// Returns whether optional FP16 arithmetic is reported by the host.
    pub const fn fp16_arithmetic(&self) -> bool {
        self.fp16_arithmetic
    }

    /// Returns the closed optimized operation set.
    pub fn operations(&self) -> &[Arm64Operation] {
        &self.operations
    }

    /// Returns the configured fixed worker count.
    pub const fn worker_count(&self) -> usize {
        self.worker_count
    }

    /// Returns whether single-thread deterministic execution is required.
    pub const fn deterministic(&self) -> bool {
        self.deterministic
    }
}

/// ARM64 graph backend with a scalar correctness fallback for closed operations.
#[derive(Debug, Clone)]
pub struct Arm64Backend {
    id: BackendId,
    scalar: CpuReferenceBackend,
    config: Arm64ExecutionConfig,
    capabilities: Arm64Capabilities,
    workers: Arc<WorkerPool>,
}

impl PartialEq for Arm64Backend {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id && self.config == other.config
    }
}

impl Eq for Arm64Backend {}

impl Arm64Backend {
    /// Constructs the backend only on an ARM64 target.
    pub fn new() -> Result<Self, EngineError> {
        Self::with_config(Arm64ExecutionConfig::default())
    }

    /// Constructs an ARM64 backend with one validated local execution policy.
    pub fn with_config(config: Arm64ExecutionConfig) -> Result<Self, EngineError> {
        if !cfg!(target_arch = "aarch64") {
            return Err(EngineError::not_implemented("ARM64 backend requires an aarch64 target"));
        }
        let capabilities = runtime_capabilities(config);
        if !capabilities.advanced_simd {
            return Err(EngineError::not_implemented(
                "ARM64 backend requires runtime Advanced SIMD support",
            ));
        }
        Ok(Self {
            id: BackendId::new(ARM64_BACKEND_ID)?,
            scalar: CpuReferenceBackend::new()?,
            config,
            capabilities,
            workers: Arc::new(WorkerPool::new(config.worker_count)?),
        })
    }

    /// Returns the fixed backend identifier.
    pub const fn id(&self) -> &BackendId {
        &self.id
    }

    /// Returns the trusted host execution device.
    pub const fn device(&self) -> Device {
        Device::Arm64
    }

    /// Returns immutable runtime capability evidence.
    pub const fn capabilities(&self) -> &Arm64Capabilities {
        &self.capabilities
    }

    /// Returns current bounded worker resource metrics.
    pub fn worker_metrics(&self) -> WorkerPoolMetrics {
        self.workers.metrics()
    }

    fn staging_pool(&self, pool: &MemoryPool) -> MemoryPool {
        pool.scoped(MemoryOwner::Backend(self.id.clone()))
    }

    fn add(&self, left: &Tensor, right: &Tensor, pool: &MemoryPool) -> Result<Tensor, EngineError> {
        if !same_contiguous_f32(left, right, pool)?
            || left.shape().element_count() < MIN_VECTOR_ELEMENTS as u64
        {
            return self.scalar.add(left, right, pool);
        }
        let shape = left.shape().clone();
        let staging = self.staging_pool(pool);
        let left = f32_values(left, &staging)?;
        let right = f32_values(right, &staging)?;
        let mut output = StagedF32::zeros(left.len(), &staging)?;
        accelerated_add(left.as_slice(), right.as_slice(), output.as_mut_slice())?;
        checked_tensor(left.len(), output.as_slice(), shape, pool)
    }

    fn multiply(
        &self,
        left: &Tensor,
        right: &Tensor,
        pool: &MemoryPool,
    ) -> Result<Tensor, EngineError> {
        if !same_contiguous_f32(left, right, pool)?
            || left.shape().element_count() < MIN_VECTOR_ELEMENTS as u64
        {
            return self.scalar.multiply(left, right, pool);
        }
        let shape = left.shape().clone();
        let staging = self.staging_pool(pool);
        let left = f32_values(left, &staging)?;
        let right = f32_values(right, &staging)?;
        let mut output = StagedF32::zeros(left.len(), &staging)?;
        accelerated_multiply(left.as_slice(), right.as_slice(), output.as_mut_slice())?;
        checked_tensor(left.len(), output.as_slice(), shape, pool)
    }

    fn scalar_multiply(
        &self,
        input: &Tensor,
        scalar: f32,
        pool: &MemoryPool,
    ) -> Result<Tensor, EngineError> {
        require_contiguous_f32(input, pool)?;
        if input.shape().element_count() < MIN_VECTOR_ELEMENTS as u64 {
            return self.scalar.scalar_multiply(input, scalar, pool);
        }
        require_finite(scalar)?;
        let shape = input.shape().clone();
        let staging = self.staging_pool(pool);
        let input = f32_values(input, &staging)?;
        let mut output = StagedF32::zeros(input.len(), &staging)?;
        accelerated_scale(input.as_slice(), scalar, output.as_mut_slice())?;
        checked_tensor(input.len(), output.as_slice(), shape, pool)
    }

    fn matrix_vector(
        &self,
        matrix: &Tensor,
        vector: &Tensor,
        pool: &MemoryPool,
    ) -> Result<Tensor, EngineError> {
        require_contiguous_f32(matrix, pool)?;
        require_contiguous_f32(vector, pool)?;
        if matrix.shape().rank() != 2 || vector.shape().rank() != 1 {
            return Err(EngineError::invalid_input("ARM64 matrix-vector ranks are invalid"));
        }
        let rows_u64 = matrix.shape().dimensions()[0];
        let columns = matrix.shape().dimensions()[1];
        if vector.shape().dimensions()[0] != columns {
            return Err(EngineError::invalid_input("ARM64 matrix-vector dimensions differ"));
        }
        if columns < MIN_VECTOR_ELEMENTS as u64 {
            return self.scalar.matrix_vector(matrix, vector, pool);
        }
        validate_matrix_vector_work(
            rows_u64,
            columns,
            MAX_CPU_REFERENCE_OPERATIONS,
            "ARM64 matrix-vector exceeds its work ceiling",
        )?;
        let rows = usize::try_from(rows_u64)
            .map_err(|_| EngineError::resource_exhaustion("ARM64 row count is not addressable"))?;
        let columns = usize::try_from(columns).map_err(|_| {
            EngineError::resource_exhaustion("ARM64 column count is not addressable")
        })?;
        let staging = self.staging_pool(pool);
        let matrix = f32_values(matrix, &staging)?;
        let vector = f32_values(vector, &staging)?;
        let mut output = StagedF32::zeros(rows, &staging)?;
        for (row, destination) in output.as_mut_slice().iter_mut().enumerate() {
            let start = row.checked_mul(columns).ok_or_else(|| {
                EngineError::resource_exhaustion("ARM64 matrix row offset overflows")
            })?;
            let end = start.checked_add(columns).ok_or_else(|| {
                EngineError::resource_exhaustion("ARM64 matrix row end overflows")
            })?;
            *destination = accelerated_dot(&matrix.as_slice()[start..end], vector.as_slice())?;
            require_finite(*destination)?;
        }
        MutableTensor::from_f32_values(TensorShape::new([rows_u64])?, output.as_slice(), pool)
    }

    /// Executes the engine-owned hidden-to-vocabulary projection with bounded
    /// scratch and cooperative cancellation.
    ///
    /// The separate ceiling is the exact work required by the already-reviewed
    /// Qwen2.5 1.5B tied embedding/output shape. Generic matrix-vector and
    /// scalar-reference operations retain their existing lower ceiling.
    pub fn logit_projection_cancellable(
        &self,
        matrix: &Tensor,
        vector: &Tensor,
        pool: &MemoryPool,
        cancelled: &dyn Fn() -> bool,
    ) -> Result<Tensor, EngineError> {
        require_contiguous_f32_with_limit(
            matrix,
            pool,
            MAX_ARM64_LOGIT_PROJECTION_OPERATIONS,
            "ARM64 logit projection exceeds its exact work ceiling",
        )?;
        require_contiguous_f32(vector, pool)?;
        if matrix.shape().rank() != 2 || vector.shape().rank() != 1 {
            return Err(EngineError::invalid_input("ARM64 logit projection ranks are invalid"));
        }
        let rows_u64 = matrix.shape().dimensions()[0];
        let columns_u64 = matrix.shape().dimensions()[1];
        if vector.shape().dimensions()[0] != columns_u64 {
            return Err(EngineError::invalid_input("ARM64 logit projection dimensions differ"));
        }
        validate_matrix_vector_work(
            rows_u64,
            columns_u64,
            MAX_ARM64_LOGIT_PROJECTION_OPERATIONS,
            "ARM64 logit projection exceeds its exact work ceiling",
        )?;
        if columns_u64 < MIN_VECTOR_ELEMENTS as u64 {
            if cancelled() {
                return Err(EngineError::cancellation(None));
            }
            return self.scalar.logit_projection(matrix, vector, pool);
        }
        let rows = addressable(rows_u64, "ARM64 logit row count is not addressable")?;
        let columns = addressable(columns_u64, "ARM64 logit width is not addressable")?;
        let staging = self.staging_pool(pool);
        let vector = f32_values(vector, &staging)?;
        let mut output = StagedF32::zeros(rows, &staging)?;
        for first_row in (0..rows).step_by(LOGIT_PROJECTION_ROW_CHUNK) {
            if cancelled() {
                return Err(EngineError::cancellation(None));
            }
            let row_end = first_row
                .checked_add(LOGIT_PROJECTION_ROW_CHUNK)
                .ok_or_else(|| EngineError::resource_exhaustion("ARM64 logit row chunk overflows"))?
                .min(rows);
            for (row, destination) in
                output.as_mut_slice()[first_row..row_end].iter_mut().enumerate()
            {
                let row = first_row.checked_add(row).ok_or_else(|| {
                    EngineError::resource_exhaustion("ARM64 logit row index overflows")
                })?;
                let row_start = row.checked_mul(columns).ok_or_else(|| {
                    EngineError::resource_exhaustion("ARM64 logit row offset overflows")
                })?;
                let mut accumulator = 0.0_f64;
                for first_column in (0..columns).step_by(LOGIT_PROJECTION_COLUMN_CHUNK) {
                    if cancelled() {
                        return Err(EngineError::cancellation(None));
                    }
                    let width = (columns - first_column).min(LOGIT_PROJECTION_COLUMN_CHUNK);
                    let input_end = first_column.checked_add(width).ok_or_else(|| {
                        EngineError::resource_exhaustion("ARM64 logit input range overflows")
                    })?;
                    let weight_offset = row_start.checked_add(first_column).ok_or_else(|| {
                        EngineError::resource_exhaustion("ARM64 logit weight index overflows")
                    })?;
                    let weight_bytes = contiguous_f32_byte_chunk(matrix, weight_offset, width)?;
                    let partial = accelerated_dot_canonical_f32_bytes(
                        weight_bytes,
                        &vector.as_slice()[first_column..input_end],
                    )?;
                    accumulator += f64::from(require_finite(partial)?);
                    if !accumulator.is_finite() {
                        return Err(EngineError::backend_failure(
                            "ARM64 logit projection accumulator is non-finite",
                        ));
                    }
                }
                *destination = f32_from_f64(accumulator)?;
            }
        }
        if cancelled() {
            return Err(EngineError::cancellation(None));
        }
        checked_tensor(rows, output.as_slice(), TensorShape::new([rows_u64])?, pool)
    }

    fn matrix_matrix(
        &self,
        left: &Tensor,
        right: &Tensor,
        pool: &MemoryPool,
        cancellation: &CancellationToken,
    ) -> Result<Tensor, EngineError> {
        if !can_accelerate_matrix_matrix(left, right, pool) {
            return self.scalar.matrix_matrix(left, right, pool);
        }
        let rows_u64 = left.shape().dimensions()[0];
        let shared_u64 = left.shape().dimensions()[1];
        let columns_u64 = right.shape().dimensions()[1];
        checked_work(&[rows_u64, shared_u64, columns_u64])?;
        let rows = addressable(rows_u64, "ARM64 matrix row count is not addressable")?;
        let shared = addressable(shared_u64, "ARM64 matrix shared dimension is not addressable")?;
        let columns = addressable(columns_u64, "ARM64 matrix column count is not addressable")?;
        let staging = self.staging_pool(pool);
        let left = Arc::new(f32_values(left, &staging)?.values);
        let right = Arc::new(f32_values(right, &staging)?.values);
        let output_count = rows
            .checked_mul(columns)
            .ok_or_else(|| EngineError::resource_exhaustion("ARM64 matrix output overflows"))?;
        let output = Arc::new(StagedAtomicF32::zeros(output_count, &staging)?);
        let partitions = self.config.worker_count.min(rows.div_ceil(MATMUL_ROW_TILE)).max(1);
        let mut jobs = Vec::new();
        jobs.try_reserve_exact(partitions).map_err(|_| {
            EngineError::resource_exhaustion("ARM64 work partition allocation failed")
        })?;
        for partition in 0..partitions {
            let left = Arc::clone(&left);
            let right = Arc::clone(&right);
            let output = Arc::clone(&output);
            let cancellation = cancellation.clone();
            jobs.push(Box::new(move || {
                let mut row = partition * MATMUL_ROW_TILE;
                while row < rows {
                    if cancellation.is_cancelled() {
                        return Err(EngineError::cancellation(None));
                    }
                    for tile_row in row..(row + MATMUL_ROW_TILE).min(rows) {
                        let start = tile_row.checked_mul(shared).ok_or_else(|| {
                            EngineError::resource_exhaustion("ARM64 matrix row offset overflows")
                        })?;
                        let end = start.checked_add(shared).ok_or_else(|| {
                            EngineError::resource_exhaustion("ARM64 matrix row end overflows")
                        })?;
                        for first_column in (0..columns).step_by(MATMUL_COLUMN_TILE) {
                            let values = accelerated_matrix_matrix_block(
                                &left[start..end],
                                &right,
                                columns,
                                first_column,
                            )?;
                            for (offset, value) in values
                                .into_iter()
                                .take((columns - first_column).min(MATMUL_COLUMN_TILE))
                                .enumerate()
                            {
                                require_finite(value)?;
                                output.store(tile_row * columns + first_column + offset, value)?;
                            }
                        }
                    }
                    row = row.checked_add(partitions * MATMUL_ROW_TILE).ok_or_else(|| {
                        EngineError::resource_exhaustion("ARM64 row stride overflows")
                    })?;
                }
                Ok(())
            }) as Box<dyn FnOnce() -> Result<(), EngineError> + Send>);
        }
        self.workers.run(jobs, cancellation)?;
        let values = output.collect(&staging)?;
        MutableTensor::from_f32_values(
            TensorShape::new([rows_u64, columns_u64])?,
            values.as_slice(),
            pool,
        )
    }

    fn convert(
        &self,
        input: &Tensor,
        dtype: DType,
        pool: &MemoryPool,
    ) -> Result<Tensor, EngineError> {
        if dtype != DType::F32
            || !matches!(input.dtype(), DType::F16 | DType::BFloat16)
            || !input.metadata().layout().is_contiguous()
            || !input.accounted_by(pool)
            || usize::try_from(input.shape().element_count()).unwrap_or(0) < MIN_VECTOR_ELEMENTS
        {
            return self.scalar.convert(input, dtype, pool);
        }
        let count = addressable(
            input.shape().element_count(),
            "ARM64 conversion length is not addressable",
        )?;
        let staging = self.staging_pool(pool);
        let mut words = StagedU16::zeros(count, &staging)?;
        for index in 0..count {
            let bytes = input.element_bytes(index)?;
            words.values[index] = u16::from_le_bytes(bytes.try_into().map_err(|_| {
                EngineError::internal_invariant("validated half tensor element is truncated")
            })?);
        }
        let mut output = StagedF32::zeros(count, &staging)?;
        accelerated_half_to_f32(input.dtype(), &words.values, output.as_mut_slice())?;
        MutableTensor::from_f32_values(input.shape().clone(), output.as_slice(), pool)
    }

    fn rms_norm(
        &self,
        input: &Tensor,
        weight: &Tensor,
        epsilon: f32,
        pool: &MemoryPool,
    ) -> Result<Tensor, EngineError> {
        if !same_rms_shape(input, weight, epsilon, pool) {
            return self.scalar.rms_norm(input, weight, epsilon, pool);
        }
        let width =
            addressable(weight.shape().element_count(), "ARM64 RMS width is not addressable")?;
        let staging = self.staging_pool(pool);
        let input_values = f32_values(input, &staging)?;
        let weight_values = f32_values(weight, &staging)?;
        let mut output = StagedF32::zeros(input_values.len(), &staging)?;
        for (row_input, row_output) in input_values
            .as_slice()
            .chunks_exact(width)
            .zip(output.as_mut_slice().chunks_exact_mut(width))
        {
            let inverse = robust_rms_inverse(row_input, epsilon)?;
            accelerated_rms_apply(row_input, weight_values.as_slice(), inverse, row_output)?;
        }
        checked_tensor(output.len(), output.as_slice(), input.shape().clone(), pool)
    }

    fn softmax(&self, input: &Tensor, pool: &MemoryPool) -> Result<Tensor, EngineError> {
        if !can_accelerate_unary(input, pool)? || input.shape().rank() == 0 {
            return self.scalar.softmax(input, pool);
        }
        let width =
            addressable(
                *input.shape().dimensions().last().ok_or_else(|| {
                    EngineError::invalid_input("ARM64 softmax input rank is invalid")
                })?,
                "ARM64 softmax width is not addressable",
            )?;
        let staging = self.staging_pool(pool);
        let input_values = f32_values_allow_negative_infinity(input, &staging)?;
        let mut output = StagedF32::zeros(input_values.len(), &staging)?;
        for (row_input, row_output) in input_values
            .as_slice()
            .chunks_exact(width)
            .zip(output.as_mut_slice().chunks_exact_mut(width))
        {
            let maximum = row_input.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            if maximum == f32::NEG_INFINITY {
                return Err(EngineError::backend_failure("ARM64 softmax row is fully masked"));
            }
            let mut sum = 0.0_f64;
            for (value, destination) in row_input.iter().copied().zip(row_output.iter_mut()) {
                let exponential =
                    if value == f32::NEG_INFINITY { 0.0 } else { f64::from(value - maximum).exp() };
                *destination = f32_from_f64(exponential)?;
                sum += exponential;
            }
            if !sum.is_finite() || sum <= 0.0 {
                return Err(EngineError::backend_failure("ARM64 softmax normalization is invalid"));
            }
            accelerated_scale_in_place(row_output, f32_from_f64(1.0 / sum)?)?;
        }
        checked_tensor(output.len(), output.as_slice(), input.shape().clone(), pool)
    }

    fn silu(&self, input: &Tensor, pool: &MemoryPool) -> Result<Tensor, EngineError> {
        if !can_accelerate_unary(input, pool)? {
            return self.scalar.silu(input, pool);
        }
        let staging = self.staging_pool(pool);
        let input_values = f32_values(input, &staging)?;
        let mut sigmoid = StagedF32::zeros(input_values.len(), &staging)?;
        for (value, destination) in input_values.as_slice().iter().zip(sigmoid.as_mut_slice()) {
            *destination = f32_from_f64(1.0 / (1.0 + (-f64::from(*value)).exp()))?;
        }
        let mut output = StagedF32::zeros(input_values.len(), &staging)?;
        accelerated_multiply(input_values.as_slice(), sigmoid.as_slice(), output.as_mut_slice())?;
        checked_tensor(output.len(), output.as_slice(), input.shape().clone(), pool)
    }

    fn swiglu(&self, gate: &Tensor, up: &Tensor, pool: &MemoryPool) -> Result<Tensor, EngineError> {
        if !same_contiguous_f32(gate, up, pool)?
            || gate.shape().element_count() < MIN_VECTOR_ELEMENTS as u64
        {
            return self.scalar.swiglu(gate, up, pool);
        }
        let staging = self.staging_pool(pool);
        let gate_values = f32_values(gate, &staging)?;
        let up_values = f32_values(up, &staging)?;
        let mut activated = StagedF32::zeros(gate_values.len(), &staging)?;
        for (value, destination) in gate_values.as_slice().iter().zip(activated.as_mut_slice()) {
            let value = f64::from(*value);
            *destination = f32_from_f64(value / (1.0 + (-value).exp()))?;
        }
        let mut output = StagedF32::zeros(gate_values.len(), &staging)?;
        accelerated_multiply(activated.as_slice(), up_values.as_slice(), output.as_mut_slice())?;
        checked_tensor(output.len(), output.as_slice(), gate.shape().clone(), pool)
    }

    fn rope(
        &self,
        input: &Tensor,
        position: u32,
        theta: f32,
        pool: &MemoryPool,
    ) -> Result<Tensor, EngineError> {
        if !can_accelerate_rope(input, theta, pool)? {
            return self.scalar.rope(input, position, theta, pool);
        }
        let dimension =
            addressable(input.shape().dimensions()[1], "ARM64 RoPE dimension is not addressable")?;
        let staging = self.staging_pool(pool);
        let values = f32_values(input, &staging)?;
        let mut cosine = StagedF32::zeros(values.len(), &staging)?;
        let mut sine = StagedF32::zeros(values.len(), &staging)?;
        for head in
            0..addressable(input.shape().dimensions()[0], "ARM64 RoPE heads are not addressable")?
        {
            let base = head * dimension;
            for pair in (0..dimension).step_by(2) {
                let exponent = f64_from_usize(pair)? / f64_from_usize(dimension)?;
                let angle = f64::from(position) / f64::from(theta).powf(exponent);
                let (sin, cos) = angle.sin_cos();
                let sin = f32_from_f64(sin)?;
                let cos = f32_from_f64(cos)?;
                sine.values[base + pair] = sin;
                sine.values[base + pair + 1] = sin;
                cosine.values[base + pair] = cos;
                cosine.values[base + pair + 1] = cos;
            }
        }
        let mut output = StagedF32::zeros(values.len(), &staging)?;
        accelerated_rope(
            values.as_slice(),
            cosine.as_slice(),
            sine.as_slice(),
            output.as_mut_slice(),
        )?;
        checked_tensor(output.len(), output.as_slice(), input.shape().clone(), pool)
    }

    fn grouped_query_attention(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        causal_position: u32,
        pool: &MemoryPool,
    ) -> Result<Tensor, EngineError> {
        if !can_accelerate_attention(query, key, value, causal_position, pool) {
            return self.scalar.grouped_query_attention(query, key, value, causal_position, pool);
        }
        let query_heads =
            addressable(query.shape().dimensions()[0], "ARM64 query heads are not addressable")?;
        let head_dimension =
            addressable(query.shape().dimensions()[1], "ARM64 head dimension is not addressable")?;
        let kv_heads =
            addressable(key.shape().dimensions()[1], "ARM64 KV heads are not addressable")?;
        let sequence =
            usize::try_from(causal_position).ok().and_then(|v| v.checked_add(1)).ok_or_else(
                || EngineError::resource_exhaustion("ARM64 attention sequence length overflows"),
            )?;
        checked_work(&[query_heads as u64, sequence as u64, head_dimension as u64, 2])?;
        let staging = self.staging_pool(pool);
        let query = f32_values(query, &staging)?;
        let key = f32_values(key, &staging)?;
        let value = f32_values(value, &staging)?;
        let mut output = StagedF32::zeros(query.len(), &staging)?;
        let group_size = query_heads / kv_heads;
        let scale = f64_from_usize(head_dimension)?.sqrt();
        for query_head in 0..query_heads {
            let kv_head = query_head / group_size;
            let query_row =
                &query.as_slice()[query_head * head_dimension..(query_head + 1) * head_dimension];
            let mut scores = StagedF64::zeros(sequence, &staging)?;
            for (position, score) in scores.values.iter_mut().enumerate() {
                let start = (position * kv_heads + kv_head) * head_dimension;
                *score = f64::from(accelerated_dot(
                    query_row,
                    &key.as_slice()[start..start + head_dimension],
                )?) / scale;
            }
            let maximum = scores.values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let mut sum = 0.0;
            for score in &mut scores.values {
                *score = (*score - maximum).exp();
                sum += *score;
            }
            if !sum.is_finite() || sum <= 0.0 {
                return Err(EngineError::backend_failure("ARM64 attention softmax is invalid"));
            }
            let output_row = &mut output.as_mut_slice()
                [query_head * head_dimension..(query_head + 1) * head_dimension];
            for (position, score) in scores.values.iter().copied().enumerate() {
                let start = (position * kv_heads + kv_head) * head_dimension;
                accelerated_weighted_accumulate(
                    &value.as_slice()[start..start + head_dimension],
                    f32_from_f64(score / sum)?,
                    output_row,
                )?;
            }
        }
        checked_tensor(
            output.len(),
            output.as_slice(),
            TensorShape::new([query_heads as u64, head_dimension as u64])?,
            pool,
        )
    }

    /// Executes one validated operation with cooperative cancellation.
    pub fn execute_operation_cancellable(
        &self,
        operation: &Operation,
        inputs: &[&Tensor],
        pool: &MemoryPool,
        cancellation: &CancellationToken,
    ) -> Result<Tensor, EngineError> {
        if cancellation.is_cancelled() {
            return Err(EngineError::cancellation(None));
        }
        match (operation, inputs) {
            (Operation::Convert(dtype), [input]) => self.convert(input, *dtype, pool),
            (Operation::Add, [left, right]) => self.add(left, right, pool),
            (Operation::Multiply, [left, right]) => self.multiply(left, right, pool),
            (Operation::ScalarMultiply(scalar), [input]) => {
                self.scalar_multiply(input, *scalar, pool)
            }
            (Operation::MatrixVector, [matrix, vector]) => self.matrix_vector(matrix, vector, pool),
            (Operation::LogitProjection, [matrix, vector]) => {
                self.logit_projection_cancellable(matrix, vector, pool, &|| {
                    cancellation.is_cancelled()
                })
            }
            (Operation::MatrixMatrix, [left, right]) => {
                self.matrix_matrix(left, right, pool, cancellation)
            }
            (Operation::RmsNorm { epsilon }, [input, weight]) => {
                self.rms_norm(input, weight, *epsilon, pool)
            }
            (Operation::Softmax, [input]) => self.softmax(input, pool),
            (Operation::Silu, [input]) => self.silu(input, pool),
            (Operation::SwiGlu, [gate, up]) => self.swiglu(gate, up, pool),
            (Operation::Rope { position, theta }, [input]) => {
                self.rope(input, *position, *theta, pool)
            }
            (Operation::GroupedQueryAttention { causal_position }, [query, key, value]) => {
                self.grouped_query_attention(query, key, value, *causal_position, pool)
            }
            _ => self.scalar.execute_operation(operation, inputs, pool),
        }
    }

    /// Executes a validated AIPQ8/AIPQ4 matrix-vector operation through SIMD dequantized blocks.
    pub fn quantized_matrix_vector(
        &self,
        weights: QuantizedTensor<'_>,
        vector: &Tensor,
        pool: &MemoryPool,
        cancellation: &CancellationToken,
    ) -> Result<Tensor, EngineError> {
        if !matches!(weights.format(), Quantization::AipQ8V1 | Quantization::AipQ4V1) {
            return Err(EngineError::not_implemented("ARM64 quantized format is unsupported"));
        }
        if weights.shape().rank() != 2
            || vector.dtype() != DType::F32
            || vector.shape().rank() != 1
            || vector.shape().dimensions()[0] != weights.shape().dimensions()[1]
            || !vector.metadata().layout().is_contiguous()
            || !vector.accounted_by(pool)
        {
            return Err(EngineError::invalid_input(
                "ARM64 quantized matrix-vector inputs are invalid",
            ));
        }
        if cancellation.is_cancelled() {
            return Err(EngineError::cancellation(None));
        }
        let rows_u64 = weights.shape().dimensions()[0];
        let columns_u64 = weights.shape().dimensions()[1];
        checked_work(&[rows_u64, columns_u64])?;
        let rows = addressable(rows_u64, "ARM64 quantized row count is not addressable")?;
        let columns = addressable(columns_u64, "ARM64 quantized width is not addressable")?;
        if columns < MIN_VECTOR_ELEMENTS {
            return weights.matrix_vector(vector, pool);
        }
        // AIPQ v1 blocks contain exactly 32 logical values. Decode one block
        // at a time into fixed stack storage and accumulate only its overlap
        // with the current row. Scratch is therefore independent of the
        // weight tensor size; the exact output tensor is the only allocation
        // added to the caller's already-accounted input tensor.
        let mut output = MutableTensor::zeros(DType::F32, TensorShape::new([rows_u64])?, pool)?;
        let mut codes = [0_i8; AIPQ_V1_BLOCK_VALUES];
        let mut decoded = [0.0_f32; AIPQ_V1_BLOCK_VALUES];
        let mut input_block = [0.0_f32; AIPQ_V1_BLOCK_VALUES];
        for row in 0..rows {
            if cancellation.is_cancelled() {
                return Err(EngineError::cancellation(None));
            }
            let row_start = row.checked_mul(columns).ok_or_else(|| {
                EngineError::resource_exhaustion("ARM64 quantized row offset overflows")
            })?;
            let row_end = row_start.checked_add(columns).ok_or_else(|| {
                EngineError::resource_exhaustion("ARM64 quantized row end overflows")
            })?;
            let first_block = row_start / AIPQ_V1_BLOCK_VALUES;
            let last_block = row_end.checked_sub(1).ok_or_else(|| {
                EngineError::internal_invariant("ARM64 quantized row is unexpectedly empty")
            })? / AIPQ_V1_BLOCK_VALUES;
            let mut accumulator = 0.0_f64;
            for block in first_block..=last_block {
                if cancellation.is_cancelled() {
                    return Err(EngineError::cancellation(None));
                }
                let (used, scale) = weights.decode_codes_block(block, &mut codes)?;
                accelerated_dequantize_block(&codes, scale, &mut decoded)?;
                let block_start = block.checked_mul(AIPQ_V1_BLOCK_VALUES).ok_or_else(|| {
                    EngineError::resource_exhaustion("ARM64 quantized block offset overflows")
                })?;
                let block_end = block_start.checked_add(used).ok_or_else(|| {
                    EngineError::resource_exhaustion("ARM64 quantized block end overflows")
                })?;
                let overlap_start = row_start.max(block_start);
                let overlap_end = row_end.min(block_end);
                let overlap = overlap_end
                    .checked_sub(overlap_start)
                    .filter(|count| *count > 0)
                    .ok_or_else(|| {
                        EngineError::internal_invariant(
                            "ARM64 quantized row and decoded block do not overlap",
                        )
                    })?;
                let decoded_start = overlap_start.checked_sub(block_start).ok_or_else(|| {
                    EngineError::internal_invariant("ARM64 decoded block offset underflows")
                })?;
                let decoded_end = decoded_start.checked_add(overlap).ok_or_else(|| {
                    EngineError::resource_exhaustion("ARM64 decoded block range overflows")
                })?;
                let vector_start = overlap_start.checked_sub(row_start).ok_or_else(|| {
                    EngineError::internal_invariant("ARM64 input block offset underflows")
                })?;
                for (offset, destination) in input_block.iter_mut().take(overlap).enumerate() {
                    let column = vector_start.checked_add(offset).ok_or_else(|| {
                        EngineError::resource_exhaustion("ARM64 input block index overflows")
                    })?;
                    *destination = require_finite(vector.f32_value(column)?)?;
                }
                let partial =
                    accelerated_dot(&decoded[decoded_start..decoded_end], &input_block[..overlap])?;
                accumulator += f64::from(require_finite(partial)?);
                if !accumulator.is_finite() {
                    return Err(EngineError::backend_failure(
                        "ARM64 quantized matrix-vector accumulator is non-finite",
                    ));
                }
            }
            output.set_f32_value(row, f32_from_f64(accumulator)?)?;
        }
        Ok(output)
    }
}

impl Backend for Arm64Backend {
    fn device(&self) -> Device {
        self.device()
    }

    fn execute_operation(
        &self,
        operation: &Operation,
        inputs: &[&Tensor],
        pool: &MemoryPool,
    ) -> Result<Tensor, EngineError> {
        self.execute_operation_cancellable(operation, inputs, pool, &CancellationToken::new())
    }
}

fn require_contiguous_f32(tensor: &Tensor, pool: &MemoryPool) -> Result<(), EngineError> {
    require_contiguous_f32_with_limit(
        tensor,
        pool,
        MAX_CPU_REFERENCE_OPERATIONS,
        "ARM64 operation exceeds its work ceiling",
    )
}

fn require_contiguous_f32_with_limit(
    tensor: &Tensor,
    pool: &MemoryPool,
    maximum_elements: u64,
    limit_reason: &'static str,
) -> Result<(), EngineError> {
    if tensor.dtype() != DType::F32 || !tensor.metadata().layout().is_contiguous() {
        return Err(EngineError::invalid_input(
            "ARM64 acceleration requires a contiguous F32 tensor",
        ));
    }
    if !tensor.accounted_by(pool) {
        return Err(EngineError::invalid_input("ARM64 tensor belongs to a different memory pool"));
    }
    if tensor.shape().element_count() > maximum_elements {
        return Err(EngineError::resource_exhaustion(limit_reason));
    }
    Ok(())
}

fn validate_matrix_vector_work(
    rows: u64,
    columns: u64,
    maximum_operations: u64,
    limit_reason: &'static str,
) -> Result<u64, EngineError> {
    let operations = rows.checked_mul(columns).ok_or_else(|| {
        EngineError::resource_exhaustion("ARM64 matrix-vector work count overflows")
    })?;
    if operations > maximum_operations {
        return Err(EngineError::resource_exhaustion(limit_reason));
    }
    Ok(operations)
}

fn contiguous_f32_byte_chunk(
    tensor: &Tensor,
    first_element: usize,
    element_length: usize,
) -> Result<&[u8], EngineError> {
    if element_length == 0 {
        return Err(EngineError::internal_invariant("ARM64 contiguous F32 byte chunk is empty"));
    }
    let end_element = first_element.checked_add(element_length).ok_or_else(|| {
        EngineError::resource_exhaustion("ARM64 contiguous F32 element range overflows")
    })?;
    let element_count = addressable(
        tensor.shape().element_count(),
        "ARM64 contiguous F32 tensor length is not addressable",
    )?;
    if end_element > element_count {
        return Err(EngineError::internal_invariant(
            "ARM64 contiguous F32 element range exceeds tensor storage",
        ));
    }
    let first_element = u64::try_from(first_element).map_err(|_| {
        EngineError::resource_exhaustion("ARM64 contiguous F32 offset is not addressable")
    })?;
    let storage_element_offset = addressable(
        tensor.metadata().layout().storage_offset(first_element)?,
        "ARM64 contiguous F32 storage offset is not addressable",
    )?;
    let byte_offset = storage_element_offset.checked_mul(size_of::<f32>()).ok_or_else(|| {
        EngineError::resource_exhaustion("ARM64 contiguous F32 byte offset overflows")
    })?;
    let byte_length = element_length.checked_mul(size_of::<f32>()).ok_or_else(|| {
        EngineError::resource_exhaustion("ARM64 contiguous F32 byte length overflows")
    })?;
    let byte_end = byte_offset.checked_add(byte_length).ok_or_else(|| {
        EngineError::resource_exhaustion("ARM64 contiguous F32 byte range overflows")
    })?;
    tensor.as_bytes().get(byte_offset..byte_end).ok_or_else(|| {
        EngineError::internal_invariant("ARM64 contiguous F32 byte range exceeds tensor storage")
    })
}

fn same_contiguous_f32(
    left: &Tensor,
    right: &Tensor,
    pool: &MemoryPool,
) -> Result<bool, EngineError> {
    if left.shape() != right.shape() {
        return Ok(false);
    }
    require_contiguous_f32(left, pool)?;
    require_contiguous_f32(right, pool)?;
    Ok(true)
}

#[derive(Debug)]
struct StagedF32 {
    values: Vec<f32>,
    _reservation: MemoryReservation,
}

#[derive(Debug)]
struct StagedF64 {
    values: Vec<f64>,
    _reservation: MemoryReservation,
}

#[derive(Debug)]
struct StagedU16 {
    values: Vec<u16>,
    _reservation: MemoryReservation,
}

impl StagedU16 {
    fn zeros(length: usize, pool: &MemoryPool) -> Result<Self, EngineError> {
        let planned_bytes =
            u64::try_from(length).ok().and_then(|length| length.checked_mul(2)).ok_or_else(
                || EngineError::resource_exhaustion("ARM64 U16 staging byte length overflows"),
            )?;
        let mut reservation = pool.reserve(planned_bytes)?;
        let mut values = Vec::new();
        values
            .try_reserve_exact(length)
            .map_err(|_| EngineError::resource_exhaustion("ARM64 U16 staging allocation failed"))?;
        let actual_bytes = u64::try_from(values.capacity())
            .ok()
            .and_then(|capacity| capacity.checked_mul(2))
            .ok_or_else(|| {
                EngineError::resource_exhaustion("ARM64 U16 staging capacity overflows")
            })?;
        if actual_bytes > planned_bytes {
            pool.grow_reservation(&mut reservation, actual_bytes - planned_bytes)?;
        }
        values.resize(length, 0);
        Ok(Self { values, _reservation: reservation })
    }
}

impl StagedF64 {
    fn zeros(length: usize, pool: &MemoryPool) -> Result<Self, EngineError> {
        let planned_bytes =
            u64::try_from(length).ok().and_then(|length| length.checked_mul(8)).ok_or_else(
                || EngineError::resource_exhaustion("ARM64 F64 staging byte length overflows"),
            )?;
        let mut reservation = pool.reserve(planned_bytes)?;
        let mut values = Vec::new();
        values
            .try_reserve_exact(length)
            .map_err(|_| EngineError::resource_exhaustion("ARM64 F64 staging allocation failed"))?;
        let actual_bytes = u64::try_from(values.capacity())
            .ok()
            .and_then(|capacity| capacity.checked_mul(8))
            .ok_or_else(|| {
                EngineError::resource_exhaustion("ARM64 F64 staging capacity overflows")
            })?;
        if actual_bytes > planned_bytes {
            pool.grow_reservation(&mut reservation, actual_bytes - planned_bytes)?;
        }
        values.resize(length, 0.0);
        Ok(Self { values, _reservation: reservation })
    }
}

#[derive(Debug)]
struct StagedAtomicF32 {
    values: Vec<AtomicU32>,
    _reservation: MemoryReservation,
}

impl StagedAtomicF32 {
    fn zeros(length: usize, pool: &MemoryPool) -> Result<Self, EngineError> {
        let planned_bytes = staged_bytes(length)?;
        let mut reservation = pool.reserve(planned_bytes)?;
        let mut values = Vec::new();
        values.try_reserve_exact(length).map_err(|_| {
            EngineError::resource_exhaustion("ARM64 atomic output allocation failed")
        })?;
        let actual_bytes = staged_bytes(values.capacity())?;
        if actual_bytes > planned_bytes {
            pool.grow_reservation(&mut reservation, actual_bytes - planned_bytes)?;
        }
        values.resize_with(length, || AtomicU32::new(0));
        Ok(Self { values, _reservation: reservation })
    }

    fn store(&self, index: usize, value: f32) -> Result<(), EngineError> {
        self.values
            .get(index)
            .ok_or_else(|| {
                EngineError::internal_invariant("ARM64 output index is outside storage")
            })?
            .store(value.to_bits(), Ordering::Relaxed);
        Ok(())
    }

    fn collect(&self, pool: &MemoryPool) -> Result<StagedF32, EngineError> {
        let mut values = StagedF32::zeros(self.values.len(), pool)?;
        for (destination, value) in values.values.iter_mut().zip(&self.values) {
            *destination = f32::from_bits(value.load(Ordering::Relaxed));
        }
        Ok(values)
    }
}

impl StagedF32 {
    fn zeros(length: usize, pool: &MemoryPool) -> Result<Self, EngineError> {
        let planned_bytes = staged_bytes(length)?;
        let mut reservation = pool.reserve(planned_bytes)?;
        let mut values = Vec::new();
        values
            .try_reserve_exact(length)
            .map_err(|_| EngineError::resource_exhaustion("ARM64 staging allocation failed"))?;
        let actual_bytes = staged_bytes(values.capacity())?;
        if actual_bytes > planned_bytes {
            pool.grow_reservation(&mut reservation, actual_bytes - planned_bytes)?;
        }
        values.resize(length, 0.0);
        Ok(Self { values, _reservation: reservation })
    }

    fn as_slice(&self) -> &[f32] {
        &self.values
    }

    fn as_mut_slice(&mut self) -> &mut [f32] {
        &mut self.values
    }

    fn len(&self) -> usize {
        self.values.len()
    }
}

fn f32_values(tensor: &Tensor, pool: &MemoryPool) -> Result<StagedF32, EngineError> {
    let count = usize::try_from(tensor.shape().element_count())
        .map_err(|_| EngineError::resource_exhaustion("ARM64 tensor is not addressable"))?;
    let mut values = StagedF32::zeros(count, pool)?;
    for index in 0..count {
        values.values[index] = require_finite(tensor.f32_value(index)?)?;
    }
    Ok(values)
}

fn f32_values_allow_negative_infinity(
    tensor: &Tensor,
    pool: &MemoryPool,
) -> Result<StagedF32, EngineError> {
    let count = addressable(tensor.shape().element_count(), "ARM64 tensor is not addressable")?;
    let mut values = StagedF32::zeros(count, pool)?;
    for index in 0..count {
        let value = tensor.f32_value(index)?;
        if value.is_nan() || value == f32::INFINITY {
            return Err(EngineError::backend_failure_for(
                BackendId::new(ARM64_BACKEND_ID)?,
                "ARM64 operation encountered an invalid non-finite value",
            ));
        }
        values.values[index] = value;
    }
    Ok(values)
}

fn can_accelerate_unary(input: &Tensor, pool: &MemoryPool) -> Result<bool, EngineError> {
    if input.dtype() != DType::F32
        || !input.metadata().layout().is_contiguous()
        || !input.accounted_by(pool)
    {
        return Ok(false);
    }
    Ok(addressable(input.shape().element_count(), "ARM64 tensor is not addressable")?
        >= MIN_VECTOR_ELEMENTS)
}

fn can_accelerate_matrix_matrix(left: &Tensor, right: &Tensor, pool: &MemoryPool) -> bool {
    if left.dtype() != DType::F32
        || right.dtype() != DType::F32
        || !left.metadata().layout().is_contiguous()
        || !right.metadata().layout().is_contiguous()
        || !left.accounted_by(pool)
        || !right.accounted_by(pool)
        || left.shape().rank() != 2
        || right.shape().rank() != 2
        || left.shape().dimensions()[1] != right.shape().dimensions()[0]
    {
        return false;
    }
    left.shape().dimensions()[1] >= MIN_VECTOR_ELEMENTS as u64
        && right.shape().dimensions()[1] >= MATMUL_COLUMN_TILE as u64
}

fn same_rms_shape(input: &Tensor, weight: &Tensor, epsilon: f32, pool: &MemoryPool) -> bool {
    if !epsilon.is_finite() || epsilon <= 0.0 {
        return false;
    }
    if input.dtype() != DType::F32
        || weight.dtype() != DType::F32
        || !input.metadata().layout().is_contiguous()
        || !weight.metadata().layout().is_contiguous()
        || !input.accounted_by(pool)
        || !weight.accounted_by(pool)
        || input.shape().rank() == 0
        || weight.shape().rank() != 1
        || input.shape().dimensions().last() != Some(&weight.shape().dimensions()[0])
    {
        return false;
    }
    weight.shape().element_count() >= MIN_VECTOR_ELEMENTS as u64
}

fn can_accelerate_rope(input: &Tensor, theta: f32, pool: &MemoryPool) -> Result<bool, EngineError> {
    if !can_accelerate_unary(input, pool)?
        || input.shape().rank() != 2
        || input.shape().dimensions()[1] % 2 != 0
        || !theta.is_finite()
        || theta <= 1.0
    {
        return Ok(false);
    }
    Ok(true)
}

fn can_accelerate_attention(
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    causal_position: u32,
    pool: &MemoryPool,
) -> bool {
    if query.dtype() != DType::F32
        || key.dtype() != DType::F32
        || value.dtype() != DType::F32
        || !query.metadata().layout().is_contiguous()
        || !key.metadata().layout().is_contiguous()
        || !value.metadata().layout().is_contiguous()
        || !query.accounted_by(pool)
        || !key.accounted_by(pool)
        || !value.accounted_by(pool)
        || query.shape().rank() != 2
        || key.shape().rank() != 3
        || value.shape() != key.shape()
        || query.shape().dimensions()[1] != key.shape().dimensions()[2]
        || query.shape().dimensions()[0] % key.shape().dimensions()[1] != 0
        || u64::from(causal_position) >= key.shape().dimensions()[0]
    {
        return false;
    }
    query.shape().dimensions()[1] >= MIN_VECTOR_ELEMENTS as u64
}

fn robust_rms_inverse(values: &[f32], epsilon: f32) -> Result<f32, EngineError> {
    let mut scale = 0.0_f64;
    let mut sum_scaled_squares = 1.0_f64;
    for value in values {
        let absolute = f64::from(require_finite(*value)?.abs());
        if absolute != 0.0 {
            if scale < absolute {
                let ratio = scale / absolute;
                sum_scaled_squares = 1.0 + sum_scaled_squares * ratio * ratio;
                scale = absolute;
            } else {
                let ratio = absolute / scale;
                sum_scaled_squares += ratio * ratio;
            }
        }
    }
    let width = f64_from_usize(values.len())?;
    let inverse = if scale == 0.0 {
        1.0 / f64::from(epsilon).sqrt()
    } else {
        let normalized_epsilon = f64::from(epsilon) / (scale * scale);
        (1.0 / scale) / (sum_scaled_squares / width + normalized_epsilon).sqrt()
    };
    f32_from_f64(inverse)
}

fn checked_work(dimensions: &[u64]) -> Result<(), EngineError> {
    let operations = dimensions.iter().try_fold(1_u64, |product, dimension| {
        product
            .checked_mul(*dimension)
            .ok_or_else(|| EngineError::resource_exhaustion("ARM64 operation count overflows"))
    })?;
    if operations > MAX_CPU_REFERENCE_OPERATIONS {
        return Err(EngineError::resource_exhaustion("ARM64 operation exceeds its work ceiling"));
    }
    Ok(())
}

fn addressable(value: u64, detail: &'static str) -> Result<usize, EngineError> {
    usize::try_from(value).map_err(|_| EngineError::resource_exhaustion(detail))
}

fn f32_from_f64(value: f64) -> Result<f32, EngineError> {
    if !value.is_finite() || value > f64::from(f32::MAX) || value < f64::from(f32::MIN) {
        return Err(EngineError::backend_failure("ARM64 result is outside finite F32 range"));
    }
    #[allow(clippy::cast_possible_truncation)]
    let value = value as f32;
    Ok(value)
}

#[allow(clippy::cast_precision_loss)]
fn f64_from_usize(value: usize) -> Result<f64, EngineError> {
    let value = value as f64;
    if value.is_finite() {
        Ok(value)
    } else {
        Err(EngineError::resource_exhaustion("ARM64 dimension is outside F64 range"))
    }
}

fn runtime_capabilities(config: Arm64ExecutionConfig) -> Arm64Capabilities {
    #[cfg(target_arch = "aarch64")]
    let (advanced_simd, fp16_arithmetic) = (
        std::arch::is_aarch64_feature_detected!("neon"),
        std::arch::is_aarch64_feature_detected!("fp16"),
    );
    #[cfg(not(target_arch = "aarch64"))]
    let (advanced_simd, fp16_arithmetic) = (false, false);
    let operations = if advanced_simd {
        vec![
            Arm64Operation::VectorAdd,
            Arm64Operation::VectorMultiply,
            Arm64Operation::DotProduct,
            Arm64Operation::MatrixVector,
            Arm64Operation::MatrixMatrix,
            Arm64Operation::RmsNorm,
            Arm64Operation::Softmax,
            Arm64Operation::Silu,
            Arm64Operation::SwiGlu,
            Arm64Operation::Rope,
            Arm64Operation::AttentionScores,
            Arm64Operation::AttentionValues,
            Arm64Operation::F16ToF32,
            Arm64Operation::Bf16ToF32,
            Arm64Operation::AipQ8Matrix,
            Arm64Operation::AipQ4Matrix,
        ]
        .into_boxed_slice()
    } else {
        Box::default()
    };
    Arm64Capabilities {
        advanced_simd,
        fp16_arithmetic,
        operations,
        worker_count: config.worker_count,
        deterministic: config.deterministic,
    }
}

fn staged_bytes(length: usize) -> Result<u64, EngineError> {
    u64::try_from(length)
        .ok()
        .and_then(|length| length.checked_mul(4))
        .ok_or_else(|| EngineError::resource_exhaustion("ARM64 staging byte length overflows"))
}

fn checked_tensor(
    expected: usize,
    values: &[f32],
    shape: TensorShape,
    pool: &MemoryPool,
) -> Result<Tensor, EngineError> {
    if values.len() != expected {
        return Err(EngineError::internal_invariant("ARM64 kernel output length changed"));
    }
    for value in values {
        require_finite(*value)?;
    }
    MutableTensor::from_f32_values(shape, values, pool)
}

fn require_finite(value: f32) -> Result<f32, EngineError> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(EngineError::backend_failure_for(
            BackendId::new(ARM64_BACKEND_ID)?,
            "ARM64 operation encountered a non-finite value",
        ))
    }
}

#[cfg(target_arch = "aarch64")]
#[allow(clippy::unnecessary_wraps)] // Preserves the explicit off-target failure signature.
fn accelerated_dot(left: &[f32], right: &[f32]) -> Result<f32, EngineError> {
    Ok(intrinsics::dot(left, right))
}

#[cfg(not(target_arch = "aarch64"))]
fn accelerated_dot(_left: &[f32], _right: &[f32]) -> Result<f32, EngineError> {
    Err(EngineError::not_implemented("ARM64 kernel requires an aarch64 target"))
}

#[cfg(target_arch = "aarch64")]
fn accelerated_dot_canonical_f32_bytes(left: &[u8], right: &[f32]) -> Result<f32, EngineError> {
    let expected_bytes = right.len().checked_mul(size_of::<f32>()).ok_or_else(|| {
        EngineError::resource_exhaustion("ARM64 canonical F32 dot byte length overflows")
    })?;
    if left.len() != expected_bytes {
        return Err(EngineError::internal_invariant("ARM64 canonical F32 dot lengths differ"));
    }
    match intrinsics::dot_canonical_f32_le_bytes(left, right) {
        Some(value) => Ok(value),
        None => Err(EngineError::backend_failure_for(
            BackendId::new(ARM64_BACKEND_ID)?,
            "ARM64 operation encountered a non-finite value",
        )),
    }
}

#[cfg(not(target_arch = "aarch64"))]
fn accelerated_dot_canonical_f32_bytes(_left: &[u8], _right: &[f32]) -> Result<f32, EngineError> {
    Err(EngineError::not_implemented("ARM64 kernel requires an aarch64 target"))
}

#[cfg(target_arch = "aarch64")]
#[allow(clippy::unnecessary_wraps)] // Preserves the explicit off-target failure signature.
fn accelerated_add(left: &[f32], right: &[f32], output: &mut [f32]) -> Result<(), EngineError> {
    intrinsics::add(left, right, output);
    Ok(())
}

#[cfg(not(target_arch = "aarch64"))]
fn accelerated_add(_left: &[f32], _right: &[f32], _output: &mut [f32]) -> Result<(), EngineError> {
    Err(EngineError::not_implemented("ARM64 kernel requires an aarch64 target"))
}

#[cfg(target_arch = "aarch64")]
#[allow(clippy::unnecessary_wraps)] // Preserves the explicit off-target failure signature.
fn accelerated_multiply(
    left: &[f32],
    right: &[f32],
    output: &mut [f32],
) -> Result<(), EngineError> {
    intrinsics::multiply(left, right, output);
    Ok(())
}

#[cfg(not(target_arch = "aarch64"))]
fn accelerated_multiply(
    _left: &[f32],
    _right: &[f32],
    _output: &mut [f32],
) -> Result<(), EngineError> {
    Err(EngineError::not_implemented("ARM64 kernel requires an aarch64 target"))
}

#[cfg(target_arch = "aarch64")]
#[allow(clippy::unnecessary_wraps)] // Preserves the explicit off-target failure signature.
fn accelerated_scale(input: &[f32], scalar: f32, output: &mut [f32]) -> Result<(), EngineError> {
    intrinsics::scale(input, scalar, output);
    Ok(())
}

#[cfg(not(target_arch = "aarch64"))]
fn accelerated_scale(_input: &[f32], _scalar: f32, _output: &mut [f32]) -> Result<(), EngineError> {
    Err(EngineError::not_implemented("ARM64 kernel requires an aarch64 target"))
}

#[cfg(target_arch = "aarch64")]
#[allow(clippy::unnecessary_wraps)]
fn accelerated_scale_in_place(values: &mut [f32], scalar: f32) -> Result<(), EngineError> {
    intrinsics::scale_in_place(values, scalar);
    Ok(())
}

#[cfg(not(target_arch = "aarch64"))]
fn accelerated_scale_in_place(_values: &mut [f32], _scalar: f32) -> Result<(), EngineError> {
    Err(EngineError::not_implemented("ARM64 kernel requires an aarch64 target"))
}

#[cfg(target_arch = "aarch64")]
#[allow(clippy::unnecessary_wraps)]
fn accelerated_matrix_matrix_block(
    left: &[f32],
    right: &[f32],
    columns: usize,
    first_column: usize,
) -> Result<[f32; 4], EngineError> {
    Ok(intrinsics::matrix_matrix_block(left, right, columns, first_column))
}

#[cfg(not(target_arch = "aarch64"))]
fn accelerated_matrix_matrix_block(
    _left: &[f32],
    _right: &[f32],
    _columns: usize,
    _first_column: usize,
) -> Result<[f32; 4], EngineError> {
    Err(EngineError::not_implemented("ARM64 kernel requires an aarch64 target"))
}

#[cfg(target_arch = "aarch64")]
#[allow(clippy::unnecessary_wraps)]
fn accelerated_rms_apply(
    input: &[f32],
    weight: &[f32],
    inverse: f32,
    output: &mut [f32],
) -> Result<(), EngineError> {
    intrinsics::rms_apply(input, weight, inverse, output);
    Ok(())
}

#[cfg(not(target_arch = "aarch64"))]
fn accelerated_rms_apply(
    _input: &[f32],
    _weight: &[f32],
    _inverse: f32,
    _output: &mut [f32],
) -> Result<(), EngineError> {
    Err(EngineError::not_implemented("ARM64 kernel requires an aarch64 target"))
}

#[cfg(target_arch = "aarch64")]
#[allow(clippy::unnecessary_wraps)]
fn accelerated_rope(
    input: &[f32],
    cosine: &[f32],
    sine: &[f32],
    output: &mut [f32],
) -> Result<(), EngineError> {
    intrinsics::rope(input, cosine, sine, output);
    Ok(())
}

#[cfg(not(target_arch = "aarch64"))]
fn accelerated_rope(
    _input: &[f32],
    _cosine: &[f32],
    _sine: &[f32],
    _output: &mut [f32],
) -> Result<(), EngineError> {
    Err(EngineError::not_implemented("ARM64 kernel requires an aarch64 target"))
}

#[cfg(target_arch = "aarch64")]
#[allow(clippy::unnecessary_wraps)]
fn accelerated_weighted_accumulate(
    input: &[f32],
    weight: f32,
    output: &mut [f32],
) -> Result<(), EngineError> {
    intrinsics::weighted_accumulate(input, weight, output);
    Ok(())
}

#[cfg(not(target_arch = "aarch64"))]
fn accelerated_weighted_accumulate(
    _input: &[f32],
    _weight: f32,
    _output: &mut [f32],
) -> Result<(), EngineError> {
    Err(EngineError::not_implemented("ARM64 kernel requires an aarch64 target"))
}

#[cfg(target_arch = "aarch64")]
fn accelerated_half_to_f32(
    dtype: DType,
    input: &[u16],
    output: &mut [f32],
) -> Result<(), EngineError> {
    match dtype {
        DType::F16 => intrinsics::f16_to_f32(input, output),
        DType::BFloat16 => intrinsics::bf16_to_f32(input, output),
        _ => {
            return Err(EngineError::internal_invariant(
                "ARM64 half conversion received an unsupported dtype",
            ));
        }
    }
    Ok(())
}

#[cfg(target_arch = "aarch64")]
#[allow(clippy::unnecessary_wraps)]
fn accelerated_dequantize_block(
    codes: &[i8; 32],
    scale: f32,
    output: &mut [f32; 32],
) -> Result<(), EngineError> {
    intrinsics::dequantize_block(codes, scale, output);
    Ok(())
}

#[cfg(not(target_arch = "aarch64"))]
fn accelerated_dequantize_block(
    _codes: &[i8; 32],
    _scale: f32,
    _output: &mut [f32; 32],
) -> Result<(), EngineError> {
    Err(EngineError::not_implemented("ARM64 kernel requires an aarch64 target"))
}

#[cfg(not(target_arch = "aarch64"))]
fn accelerated_half_to_f32(
    _dtype: DType,
    _input: &[u16],
    _output: &mut [f32],
) -> Result<(), EngineError> {
    Err(EngineError::not_implemented("ARM64 kernel requires an aarch64 target"))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(target_arch = "aarch64")]
    use personalai_engine_cpu_reference::{
        GraphNode, GraphNodeId, InferenceGraph, TensorSpec, ValueId,
    };
    #[cfg(target_arch = "aarch64")]
    use std::collections::BTreeMap;
    #[cfg(target_arch = "aarch64")]
    use std::sync::atomic::AtomicUsize;

    #[cfg(target_arch = "aarch64")]
    fn tensor(pool: &MemoryPool, shape: impl IntoIterator<Item = u64>, data: &[f32]) -> Tensor {
        MutableTensor::from_f32_values(TensorShape::new(shape).unwrap(), data, pool).unwrap()
    }

    #[cfg(target_arch = "aarch64")]
    fn assert_close(actual: &Tensor, expected: &Tensor, absolute: f32, relative: f32) {
        let actual = values(actual);
        let expected = values(expected);
        assert_eq!(actual.len(), expected.len());
        for (index, (actual, expected)) in actual.into_iter().zip(expected).enumerate() {
            let tolerance = absolute.max(relative * expected.abs());
            assert!(
                (actual - expected).abs() <= tolerance,
                "index {index}: {actual} != {expected}, tolerance {tolerance}"
            );
        }
    }

    #[test]
    fn logit_projection_work_ceiling_is_exact_and_does_not_widen_matrix_vector() {
        assert_eq!(
            validate_matrix_vector_work(
                151_936,
                1_536,
                MAX_ARM64_LOGIT_PROJECTION_OPERATIONS,
                "fixture logit limit",
            )
            .expect("the exact reviewed Qwen projection must fit"),
            MAX_ARM64_LOGIT_PROJECTION_OPERATIONS,
        );
        assert!(matches!(
            validate_matrix_vector_work(
                151_937,
                1_536,
                MAX_ARM64_LOGIT_PROJECTION_OPERATIONS,
                "fixture logit limit",
            ),
            Err(EngineError::ResourceExhaustion(_))
        ));
        assert!(matches!(
            validate_matrix_vector_work(
                MAX_CPU_REFERENCE_OPERATIONS + 1,
                1,
                MAX_CPU_REFERENCE_OPERATIONS,
                "fixture generic limit",
            ),
            Err(EngineError::ResourceExhaustion(_))
        ));
        assert!(matches!(
            validate_matrix_vector_work(u64::MAX, 2, u64::MAX, "fixture overflow"),
            Err(EngineError::ResourceExhaustion(_))
        ));
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn logit_projection_matches_scalar_and_cancels_with_bounded_scratch() {
        let pool = MemoryPool::new(1 << 20).unwrap();
        let backend = Arm64Backend::new().unwrap();
        let matrix_values =
            (0_i16..40).map(|value| f32::from(value % 13) / 13.0 - 0.4).collect::<Vec<_>>();
        let matrix = tensor(&pool, [5, 8], &matrix_values);
        let copied = contiguous_f32_byte_chunk(&matrix, 7, 5).unwrap();
        let expected_bytes =
            matrix_values[7..12].iter().flat_map(|value| value.to_le_bytes()).collect::<Vec<_>>();
        assert_eq!(copied, expected_bytes);
        assert!(contiguous_f32_byte_chunk(&matrix, 38, 5).is_err());
        assert!(contiguous_f32_byte_chunk(&matrix, 0, 0).is_err());
        let nonfinite = tensor(&pool, [1, 8], &[0.0, 1.0, 2.0, f32::NAN, 4.0, 5.0, 6.0, 7.0]);
        assert!(
            accelerated_dot_canonical_f32_bytes(
                contiguous_f32_byte_chunk(&nonfinite, 0, 8).unwrap(),
                &[1.0; 8],
            )
            .is_err()
        );
        let vector = tensor(&pool, [8], &[0.5, -1.0, 0.25, 2.0, -0.75, 1.5, 0.125, -0.5]);
        let actual = backend
            .execute_operation(&Operation::LogitProjection, &[&matrix, &vector], &pool)
            .unwrap();
        let expected = backend.scalar.logit_projection(&matrix, &vector, &pool).unwrap();
        assert_close(&actual, &expected, 2.0e-6, 2.0e-6);

        let cancel_rows = LOGIT_PROJECTION_ROW_CHUNK * 2 + 1;
        let cancel_matrix =
            tensor(&pool, [u64::try_from(cancel_rows).unwrap(), 8], &vec![0.25; cancel_rows * 8]);
        let baseline = pool.used_bytes().unwrap();
        let polls = AtomicUsize::new(0);
        let cancelled = || polls.fetch_add(1, Ordering::AcqRel) > LOGIT_PROJECTION_ROW_CHUNK;
        assert!(matches!(
            backend.logit_projection_cancellable(&cancel_matrix, &vector, &pool, &cancelled,),
            Err(EngineError::Cancellation(_))
        ));
        assert!(polls.load(Ordering::Acquire) > LOGIT_PROJECTION_ROW_CHUNK);
        assert_eq!(pool.used_bytes().unwrap(), baseline);
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn canonical_f32_byte_dot_matches_staged_dot_and_rejects_invalid_lanes() {
        let pool = MemoryPool::new(1 << 20).unwrap();
        let values = [1.0_f32, -2.0, 3.5, 0.25, -0.5, 8.0, -4.0, 2.0, 0.125, -1.5];
        let right = [0.5_f32, 1.0, -0.25, 4.0, 2.0, -0.5, 0.75, 1.5, -2.0, 0.25];
        let finite_tensor = tensor(&pool, [10], &values);
        let bytes = contiguous_f32_byte_chunk(&finite_tensor, 0, values.len()).unwrap();
        let direct = accelerated_dot_canonical_f32_bytes(bytes, &right).unwrap();
        let staged = accelerated_dot(&values, &right).unwrap();
        assert_eq!(direct.to_bits(), staged.to_bits());
        assert!(accelerated_dot_canonical_f32_bytes(&bytes[..bytes.len() - 1], &right).is_err());

        let simd_nonfinite =
            tensor(&pool, [8], &[0.0, 1.0, f32::INFINITY, 3.0, 4.0, 5.0, 6.0, 7.0]);
        assert!(
            accelerated_dot_canonical_f32_bytes(
                contiguous_f32_byte_chunk(&simd_nonfinite, 0, 8).unwrap(),
                &[1.0; 8],
            )
            .is_err()
        );
        let tail_nonfinite =
            tensor(&pool, [10], &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, f32::NEG_INFINITY]);
        assert!(
            accelerated_dot_canonical_f32_bytes(
                contiguous_f32_byte_chunk(&tail_nonfinite, 0, 10).unwrap(),
                &[1.0; 10],
            )
            .is_err()
        );
    }

    #[cfg(target_arch = "aarch64")]
    fn values(tensor: &Tensor) -> Vec<f32> {
        (0..usize::try_from(tensor.shape().element_count()).unwrap())
            .map(|index| tensor.f32_value(index).unwrap())
            .collect()
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn accelerated_operations_match_scalar_reference() {
        let pool = MemoryPool::new(1 << 20).unwrap();
        let backend = Arm64Backend::new().unwrap();
        assert_eq!(backend.device(), Device::Arm64);
        let left = Tensor::from_f32_values(
            TensorShape::new([7]).unwrap(),
            &[1.0, -2.0, 3.5, 4.0, 0.25, -6.0, 2.0],
            &pool,
        )
        .unwrap();
        let right = Tensor::from_f32_values(
            TensorShape::new([7]).unwrap(),
            &[2.0, 3.0, -1.0, 0.5, 8.0, -0.25, 4.0],
            &pool,
        )
        .unwrap();
        for operation in [Operation::Add, Operation::Multiply, Operation::ScalarMultiply(1.25)] {
            let inputs = if matches!(operation, Operation::ScalarMultiply(_)) {
                vec![&left]
            } else {
                vec![&left, &right]
            };
            let arm = backend.execute_operation(&operation, &inputs, &pool).unwrap();
            let scalar = backend.scalar.execute_operation(&operation, &inputs, &pool).unwrap();
            assert_eq!(values(&arm), values(&scalar));
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn matrix_vector_is_deterministic_and_within_oracle_tolerance() {
        let pool = MemoryPool::new(1 << 20).unwrap();
        let backend = Arm64Backend::new().unwrap();
        let matrix_values =
            (0_u16..35).map(|value| f32::from(value) / 17.0 - 1.0).collect::<Vec<_>>();
        let matrix =
            Tensor::from_f32_values(TensorShape::new([5, 7]).unwrap(), &matrix_values, &pool)
                .unwrap();
        let vector = Tensor::from_f32_values(
            TensorShape::new([7]).unwrap(),
            &[0.5, -1.0, 0.25, 2.0, -0.75, 1.5, 0.125],
            &pool,
        )
        .unwrap();
        let first = backend
            .execute_operation(&Operation::MatrixVector, &[&matrix, &vector], &pool)
            .unwrap();
        let second = backend
            .execute_operation(&Operation::MatrixVector, &[&matrix, &vector], &pool)
            .unwrap();
        let scalar = backend
            .scalar
            .execute_operation(&Operation::MatrixVector, &[&matrix, &vector], &pool)
            .unwrap();
        assert_eq!(values(&first), values(&second));
        for (actual, expected) in values(&first).into_iter().zip(values(&scalar)) {
            assert!((actual - expected).abs() <= 2.0e-6);
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn immutable_graph_executes_accelerated_and_fallback_nodes() {
        let value = |id| ValueId::new(id).unwrap();
        let graph = InferenceGraph::new(
            [
                (value(1), TensorSpec::new(DType::F32, TensorShape::new([5]).unwrap()).unwrap()),
                (value(2), TensorSpec::new(DType::F32, TensorShape::new([5]).unwrap()).unwrap()),
            ],
            vec![
                GraphNode::new(
                    GraphNodeId::new(1).unwrap(),
                    Operation::Add,
                    [value(1), value(2)],
                    value(3),
                )
                .unwrap(),
                GraphNode::new(
                    GraphNodeId::new(2).unwrap(),
                    Operation::Softmax,
                    [value(3)],
                    value(4),
                )
                .unwrap(),
            ],
            [value(4)],
        )
        .unwrap();
        let pool = MemoryPool::new(1 << 20).unwrap();
        let left = Tensor::from_f32_values(
            TensorShape::new([5]).unwrap(),
            &[1.0, 2.0, 3.0, 4.0, 5.0],
            &pool,
        )
        .unwrap();
        let right = Tensor::from_f32_values(
            TensorShape::new([5]).unwrap(),
            &[5.0, 4.0, 3.0, 2.0, 1.0],
            &pool,
        )
        .unwrap();
        let output = graph
            .execute(
                &Arm64Backend::new().unwrap(),
                BTreeMap::from([(value(1), left), (value(2), right)]),
                &pool,
            )
            .unwrap();
        for result in values(&output[&value(4)]) {
            assert!((result - 0.2).abs() <= f32::EPSILON);
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn invalid_inputs_fail_and_closed_operations_fall_back() {
        let pool = MemoryPool::new(1 << 20).unwrap();
        let other_pool = MemoryPool::new(1 << 20).unwrap();
        let backend = Arm64Backend::new().unwrap();
        let input = Tensor::from_f32_values(TensorShape::new([1]).unwrap(), &[1.0], &pool).unwrap();
        assert!(backend.scalar_multiply(&input, f32::NAN, &pool).is_err());
        assert!(backend.scalar_multiply(&input, 1.0, &other_pool).is_err());
        let copied = backend.execute_operation(&Operation::Copy, &[&input], &pool).unwrap();
        assert_eq!(values(&copied), vec![1.0]);
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn staging_reservations_are_aggregate_accounted_and_released() {
        let pool = MemoryPool::new(1 << 20).unwrap();
        let backend = Arm64Backend::new().unwrap();
        let input =
            Tensor::from_f32_values(TensorShape::new([17]).unwrap(), &[0.25; 17], &pool).unwrap();
        let baseline = pool.used_bytes().unwrap();
        {
            let output = backend
                .execute_operation(&Operation::ScalarMultiply(2.0), &[&input], &pool)
                .unwrap();
            assert_eq!(pool.used_bytes().unwrap(), baseline + output.reserved_bytes());
        }
        assert_eq!(pool.used_bytes().unwrap(), baseline);
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn capabilities_are_runtime_detected_and_registration_is_closed() {
        let backend = Arm64Backend::new().unwrap();
        assert!(backend.capabilities().advanced_simd());
        assert!(backend.capabilities().operations().contains(&Arm64Operation::MatrixMatrix));
        assert!(backend.capabilities().operations().contains(&Arm64Operation::AipQ4Matrix));
        assert!(backend.capabilities().deterministic());
        assert_eq!(backend.capabilities().worker_count(), 1);
        assert!(Arm64ExecutionConfig::new(2, true).is_err());
        assert!(Arm64ExecutionConfig::new(MAX_ARM64_WORKERS + 1, false).is_err());
        let selected = HostCpuBackend::select(Arm64ExecutionConfig::default()).unwrap();
        assert!(matches!(selected, HostCpuBackend::Arm64(_)));
        assert_eq!(selected.backend_id().as_str(), ARM64_BACKEND_ID);
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn matrix_matrix_irregular_tail_matches_scalar_single_and_multi_thread() {
        let pool = MemoryPool::new(1 << 20).unwrap();
        let left_values =
            (0_i16..33).map(|value| f32::from(value) / 11.0 - 1.0).collect::<Vec<_>>();
        let right_values =
            (0_i16..77).map(|value| f32::from(value % 19) / 9.0 - 1.0).collect::<Vec<_>>();
        let left = tensor(&pool, [3, 11], &left_values);
        let right = tensor(&pool, [11, 7], &right_values);
        let scalar = CpuReferenceBackend::new()
            .unwrap()
            .execute_operation(&Operation::MatrixMatrix, &[&left, &right], &pool)
            .unwrap();
        let single = Arm64Backend::new()
            .unwrap()
            .execute_operation(&Operation::MatrixMatrix, &[&left, &right], &pool)
            .unwrap();
        let multi = Arm64Backend::with_config(Arm64ExecutionConfig::new(3, false).unwrap())
            .unwrap()
            .execute_operation(&Operation::MatrixMatrix, &[&left, &right], &pool)
            .unwrap();
        assert_close(&single, &scalar, 3.0e-6, 3.0e-6);
        assert_eq!(single, multi);
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn rms_softmax_silu_swiglu_and_rope_match_scalar_oracle() {
        let pool = MemoryPool::new(1 << 20).unwrap();
        let backend = Arm64Backend::new().unwrap();
        let scalar = CpuReferenceBackend::new().unwrap();
        let input = tensor(
            &pool,
            [2, 8],
            &[
                -3.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 4.0, 8.0, -8.0, 1.5, -2.5, 3.5, -4.5, 0.25,
                -0.25,
            ],
        );
        let weight = tensor(&pool, [8], &[1.0, 0.5, 1.5, 2.0, 0.75, 1.25, 0.25, 1.0]);
        for (operation, inputs, absolute, relative) in [
            (Operation::RmsNorm { epsilon: 1.0e-6 }, vec![&input, &weight], 2.0e-6, 2.0e-6),
            (Operation::Softmax, vec![&input], 2.0e-7, 2.0e-6),
            (Operation::Silu, vec![&input], 2.0e-6, 2.0e-6),
            (Operation::SwiGlu, vec![&input, &input], 2.0e-6, 2.0e-6),
            (Operation::Rope { position: 17, theta: 10_000.0 }, vec![&input], 2.0e-6, 2.0e-6),
        ] {
            let actual = backend.execute_operation(&operation, &inputs, &pool).unwrap();
            let expected = scalar.execute_operation(&operation, &inputs, &pool).unwrap();
            assert_close(&actual, &expected, absolute, relative);
        }
        let masked = tensor(
            &pool,
            [8],
            &[1.0, 2.0, f32::NEG_INFINITY, 4.0, 3.0, -1.0, f32::NEG_INFINITY, 0.0],
        );
        let actual = backend.execute_operation(&Operation::Softmax, &[&masked], &pool).unwrap();
        let expected = scalar.execute_operation(&Operation::Softmax, &[&masked], &pool).unwrap();
        assert_close(&actual, &expected, 2.0e-7, 2.0e-6);
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn attention_scores_and_value_accumulation_match_f64_oracle() {
        let pool = MemoryPool::new(1 << 20).unwrap();
        let backend = Arm64Backend::new().unwrap();
        let scalar = CpuReferenceBackend::new().unwrap();
        let query = tensor(
            &pool,
            [4, 8],
            &(0_i16..32).map(|v| f32::from(v) / 31.0 - 0.5).collect::<Vec<_>>(),
        );
        let key = tensor(
            &pool,
            [3, 2, 8],
            &(0_i16..48).map(|v| f32::from(v % 17) / 17.0 - 0.4).collect::<Vec<_>>(),
        );
        let value = tensor(
            &pool,
            [3, 2, 8],
            &(0_i16..48).map(|v| f32::from(v % 13) / 13.0 - 0.3).collect::<Vec<_>>(),
        );
        let operation = Operation::GroupedQueryAttention { causal_position: 2 };
        let actual = backend.execute_operation(&operation, &[&query, &key, &value], &pool).unwrap();
        let expected =
            scalar.execute_operation(&operation, &[&query, &key, &value], &pool).unwrap();
        assert_close(&actual, &expected, 3.0e-6, 3.0e-6);
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn half_conversions_cover_all_patterns_and_unaligned_intrinsic_inputs() {
        let pool = MemoryPool::new(2 << 20).unwrap();
        let backend = Arm64Backend::new().unwrap();
        let scalar = CpuReferenceBackend::new().unwrap();
        let patterns = (0_u16..=u16::MAX).collect::<Vec<_>>();
        let f16 = MutableTensor::from_f16_bits(
            TensorShape::new([u64::try_from(patterns.len()).unwrap()]).unwrap(),
            &patterns,
            &pool,
        )
        .unwrap();
        let arm =
            backend.execute_operation(&Operation::Convert(DType::F32), &[&f16], &pool).unwrap();
        let reference =
            scalar.execute_operation(&Operation::Convert(DType::F32), &[&f16], &pool).unwrap();
        for index in 0..patterns.len() {
            let actual = arm.f32_value(index).unwrap();
            let expected = reference.f32_value(index).unwrap();
            if expected.is_nan() {
                assert!(actual.is_nan());
            } else {
                assert_eq!(actual.to_bits(), expected.to_bits(), "F16 pattern {index:#06x}");
            }
        }
        let bf16 = MutableTensor::from_bf16_bits(
            TensorShape::new([16]).unwrap(),
            &[0, 1, 0x3f80, 0xbf80, 0x7f7f, 0x0080, 0x8001, 0x4049, 2, 3, 4, 5, 6, 7, 8, 9],
            &pool,
        )
        .unwrap();
        let arm =
            backend.execute_operation(&Operation::Convert(DType::F32), &[&bf16], &pool).unwrap();
        let reference =
            scalar.execute_operation(&Operation::Convert(DType::F32), &[&bf16], &pool).unwrap();
        assert_eq!(arm, reference);

        let left = [99.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 88.0];
        let right = [77.0, -1.0, 0.5, 2.0, -2.0, 1.0, 3.0, 0.25, 66.0];
        let mut output = [0.0; 7];
        intrinsics::add(&left[1..8], &right[1..8], &mut output);
        assert_eq!(
            output.map(f32::to_bits),
            [0.0, 2.5, 5.0, 2.0, 6.0, 9.0, 7.25].map(f32::to_bits)
        );
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn q8_q4_paths_match_dequantized_reference_and_cancel() {
        use personalai_engine_quantization::quantize;
        let pool = MemoryPool::new(1 << 20).unwrap();
        let backend = Arm64Backend::new().unwrap();
        let shape = TensorShape::new([3, 11]).unwrap();
        let weights = (0_i16..33).map(|v| f32::from(v % 15) / 7.0 - 1.0).collect::<Vec<_>>();
        let vector =
            tensor(&pool, [11], &(0_i16..11).map(|v| f32::from(v) / 5.0 - 1.0).collect::<Vec<_>>());
        for format in [Quantization::AipQ8V1, Quantization::AipQ4V1] {
            let payload = quantize(format, &shape, &weights).unwrap();
            let quantized = QuantizedTensor::parse(format, &shape, payload.bytes()).unwrap();
            let actual = backend
                .quantized_matrix_vector(quantized, &vector, &pool, &CancellationToken::new())
                .unwrap();
            let decoded = quantized.dequantize(&pool).unwrap();
            let expected = CpuReferenceBackend::new()
                .unwrap()
                .matrix_vector(&decoded, &vector, &pool)
                .unwrap();
            assert_close(&actual, &expected, 3.0e-6, 3.0e-6);
            let cancellation = CancellationToken::new();
            cancellation.cancel();
            assert!(matches!(
                backend.quantized_matrix_vector(quantized, &vector, &pool, &cancellation),
                Err(EngineError::Cancellation(_))
            ));
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn q8_q4_matvec_uses_bounded_scratch_and_exact_pool_accounting() {
        use personalai_engine_quantization::quantize;
        const ROWS: usize = 64;
        const COLUMNS: usize = 64;
        let backend = Arm64Backend::new().unwrap();
        let shape = TensorShape::new([ROWS as u64, COLUMNS as u64]).unwrap();
        let weights = (0..ROWS * COLUMNS)
            .map(|index| {
                let centered = i16::try_from(index % 29).unwrap() - 14;
                f32::from(centered) * 0.03125
            })
            .collect::<Vec<_>>();
        let inputs = (0..COLUMNS)
            .map(|index| {
                let centered = i16::try_from(index % 17).unwrap() - 8;
                f32::from(centered) * 0.0625
            })
            .collect::<Vec<_>>();

        for format in [Quantization::AipQ8V1, Quantization::AipQ4V1] {
            let payload = quantize(format, &shape, &weights).unwrap();
            let quantized = QuantizedTensor::parse(format, &shape, payload.bytes()).unwrap();
            let pool = MemoryPool::new(1 << 10).unwrap();
            let vector = tensor(&pool, [COLUMNS as u64], &inputs);
            let baseline = pool.used_bytes().unwrap();
            assert!(matches!(quantized.dequantize(&pool), Err(EngineError::ResourceExhaustion(_))));

            let actual = backend
                .quantized_matrix_vector(quantized, &vector, &pool, &CancellationToken::new())
                .expect("bounded block execution must fit without full-tensor staging");
            assert_eq!(pool.used_bytes().unwrap(), baseline + actual.reserved_bytes());
            let repeated = backend
                .quantized_matrix_vector(quantized, &vector, &pool, &CancellationToken::new())
                .expect("bounded block execution must be deterministic");
            assert_eq!(
                values(&actual).into_iter().map(f32::to_bits).collect::<Vec<_>>(),
                values(&repeated).into_iter().map(f32::to_bits).collect::<Vec<_>>()
            );
            assert_eq!(
                pool.used_bytes().unwrap(),
                baseline + actual.reserved_bytes() + repeated.reserved_bytes()
            );
            assert_eq!(pool.statistics().unwrap().peak_bytes(), pool.used_bytes().unwrap());

            let oracle_pool = MemoryPool::new(1 << 20).unwrap();
            let oracle_vector = tensor(&oracle_pool, [COLUMNS as u64], &inputs);
            let expected = quantized.matrix_vector(&oracle_vector, &oracle_pool).unwrap();
            assert_close(&actual, &expected, 2.0e-4, 2.0e-4);
            drop(actual);
            drop(repeated);
            assert_eq!(pool.used_bytes().unwrap(), baseline);
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn nonfinite_small_shape_fallback_and_cancellation_fail_closed() {
        let pool = MemoryPool::new(1 << 20).unwrap();
        let backend = Arm64Backend::new().unwrap();
        let small = tensor(&pool, [3], &[1.0, 2.0, 3.0]);
        assert_eq!(
            backend.execute_operation(&Operation::Silu, &[&small], &pool).unwrap(),
            CpuReferenceBackend::new().unwrap().silu(&small, &pool).unwrap()
        );
        let nonfinite = tensor(&pool, [8], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, f32::NAN]);
        assert!(backend.execute_operation(&Operation::Silu, &[&nonfinite], &pool).is_err());
        let cancellation = CancellationToken::new();
        cancellation.cancel();
        assert!(matches!(
            backend.execute_operation_cancellable(
                &Operation::Silu,
                &[&small],
                &pool,
                &cancellation
            ),
            Err(EngineError::Cancellation(_))
        ));
    }

    #[cfg(not(target_arch = "aarch64"))]
    #[test]
    fn construction_is_explicitly_unsupported_off_target() {
        assert!(matches!(Arm64Backend::new(), Err(EngineError::NotImplemented(_))));
    }
}
