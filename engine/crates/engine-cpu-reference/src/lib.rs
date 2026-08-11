//! Safe deterministic CPU reference operations.
//!
//! These operations prioritize reviewability and correctness evidence. They are
//! not an optimized transformer implementation and do not make a model
//! executable. Inputs and outputs must share one aggregate memory pool.

use personalai_engine_memory::MemoryPool;
use personalai_engine_tensor::Tensor;
use personalai_engine_types::{BackendId, DType, EngineError, TensorShape};

/// Identifier advertised for the safe CPU reference backend.
pub const CPU_REFERENCE_BACKEND_ID: &str = "cpu-reference";
/// Hard ceiling on multiply-accumulate iterations in one reference operation.
pub const MAX_CPU_REFERENCE_OPERATIONS: u64 = 100_000_000;

/// Stateless safe CPU reference backend.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CpuReferenceBackend {
    id: BackendId,
}

impl CpuReferenceBackend {
    /// Constructs the compiled CPU reference backend identity.
    pub fn new() -> Result<Self, EngineError> {
        Ok(Self { id: BackendId::new(CPU_REFERENCE_BACKEND_ID)? })
    }

    /// Returns the backend identifier.
    pub const fn id(&self) -> &BackendId {
        &self.id
    }

    /// Adds two equal-shaped `f32` tensors with finite-value checks.
    pub fn add_f32(
        &self,
        left: &Tensor,
        right: &Tensor,
        pool: &MemoryPool,
    ) -> Result<Tensor, EngineError> {
        require_f32(left)?;
        require_f32(right)?;
        require_pool(left, pool)?;
        require_pool(right, pool)?;
        if left.shape() != right.shape() {
            return Err(EngineError::invalid_input(
                "elementwise addition requires equal tensor shapes",
            ));
        }
        let element_count = usize::try_from(left.shape().element_count()).map_err(|_| {
            EngineError::resource_exhaustion("tensor element count exceeds addressable memory")
        })?;
        checked_scalar_operation_count(left.shape().element_count())?;
        let mut output = Tensor::zeros(DType::F32, left.shape().clone(), pool)?;
        for index in 0..element_count {
            let left_value = left.f32_value(index)?;
            let right_value = right.f32_value(index)?;
            if !left_value.is_finite() || !right_value.is_finite() {
                return Err(EngineError::backend_failure(
                    "CPU reference operation received a non-finite input",
                ));
            }
            let value = left_value + right_value;
            if !value.is_finite() {
                return Err(EngineError::backend_failure(
                    "CPU reference operation produced a non-finite output",
                ));
            }
            output.set_f32_value(index, value)?;
        }
        Ok(output)
    }

    /// Multiplies two rank-two `f32` tensors using checked row-major indexing.
    pub fn matrix_multiply_f32(
        &self,
        left: &Tensor,
        right: &Tensor,
        pool: &MemoryPool,
    ) -> Result<Tensor, EngineError> {
        require_f32(left)?;
        require_f32(right)?;
        require_pool(left, pool)?;
        require_pool(right, pool)?;
        if left.shape().rank() != 2 || right.shape().rank() != 2 {
            return Err(EngineError::invalid_input(
                "matrix multiplication requires two rank-two tensors",
            ));
        }
        let left_dimensions = left.shape().dimensions();
        let right_dimensions = right.shape().dimensions();
        if left_dimensions[1] != right_dimensions[0] {
            return Err(EngineError::invalid_input(
                "matrix multiplication inner dimensions do not match",
            ));
        }
        checked_operation_count(left_dimensions[0], left_dimensions[1], right_dimensions[1])?;
        let rows = usize_dimension(left_dimensions[0])?;
        let shared = usize_dimension(left_dimensions[1])?;
        let columns = usize_dimension(right_dimensions[1])?;
        let shape = TensorShape::new([left_dimensions[0], right_dimensions[1]])?;
        let mut output = Tensor::zeros(DType::F32, shape, pool)?;

        for row in 0..rows {
            for column in 0..columns {
                let mut accumulator = 0.0_f32;
                for inner in 0..shared {
                    let left_index = checked_index(row, shared, inner)?;
                    let right_index = checked_index(inner, columns, column)?;
                    let left_value = left.f32_value(left_index)?;
                    let right_value = right.f32_value(right_index)?;
                    if !left_value.is_finite() || !right_value.is_finite() {
                        return Err(EngineError::backend_failure(
                            "CPU reference operation received a non-finite input",
                        ));
                    }
                    accumulator = left_value.mul_add(right_value, accumulator);
                }
                if !accumulator.is_finite() {
                    return Err(EngineError::backend_failure(
                        "CPU reference operation produced a non-finite output",
                    ));
                }
                let output_index = checked_index(row, columns, column)?;
                output.set_f32_value(output_index, accumulator)?;
            }
        }
        Ok(output)
    }
}

fn require_f32(tensor: &Tensor) -> Result<(), EngineError> {
    if tensor.dtype() == DType::F32 {
        Ok(())
    } else {
        Err(EngineError::not_implemented(
            "CPU reference operation currently supports only f32 tensors",
        ))
    }
}

fn require_pool(tensor: &Tensor, pool: &MemoryPool) -> Result<(), EngineError> {
    if tensor.accounted_by(pool) {
        Ok(())
    } else {
        Err(EngineError::invalid_input("CPU reference tensors must use the operation memory pool"))
    }
}

fn checked_operation_count(rows: u64, shared: u64, columns: u64) -> Result<u64, EngineError> {
    let operations = rows
        .checked_mul(shared)
        .and_then(|value| value.checked_mul(columns))
        .ok_or_else(|| EngineError::resource_exhaustion("CPU operation count overflows"))?;
    if operations > MAX_CPU_REFERENCE_OPERATIONS {
        return Err(EngineError::resource_exhaustion(
            "CPU reference operation exceeds its hard compute bound",
        ));
    }
    Ok(operations)
}

fn checked_scalar_operation_count(operations: u64) -> Result<(), EngineError> {
    if operations > MAX_CPU_REFERENCE_OPERATIONS {
        return Err(EngineError::resource_exhaustion(
            "CPU reference operation exceeds its hard compute bound",
        ));
    }
    Ok(())
}

fn usize_dimension(dimension: u64) -> Result<usize, EngineError> {
    usize::try_from(dimension)
        .map_err(|_| EngineError::resource_exhaustion("tensor dimension is not addressable"))
}

fn checked_index(row: usize, width: usize, column: usize) -> Result<usize, EngineError> {
    row.checked_mul(width)
        .and_then(|offset| offset.checked_add(column))
        .ok_or_else(|| EngineError::internal_invariant("CPU reference tensor index overflows"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tensor(pool: &MemoryPool, shape: [u64; 2], values: &[f32]) -> Tensor {
        Tensor::from_f32_values(TensorShape::new(shape).expect("shape must validate"), values, pool)
            .expect("tensor must construct")
    }

    fn values(tensor: &Tensor) -> Vec<f32> {
        (0..tensor.shape().element_count())
            .map(|index| {
                tensor
                    .f32_value(usize::try_from(index).expect("test index must fit"))
                    .expect("result must decode")
            })
            .collect()
    }

    #[test]
    fn adds_equal_f32_tensors() {
        let pool = MemoryPool::new(1 << 20).expect("pool must validate");
        let backend = CpuReferenceBackend::new().expect("backend id must validate");
        let left = tensor(&pool, [1, 3], &[1.0, 2.0, 3.0]);
        let right = tensor(&pool, [1, 3], &[4.0, 5.0, 6.0]);
        let result = backend.add_f32(&left, &right, &pool).expect("addition must work");
        assert_eq!(values(&result), [5.0, 7.0, 9.0]);
    }

    #[test]
    fn multiplies_small_matrices() {
        let pool = MemoryPool::new(1 << 20).expect("pool must validate");
        let backend = CpuReferenceBackend::new().expect("backend id must validate");
        let left = tensor(&pool, [2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let right = tensor(&pool, [3, 2], &[7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
        let result = backend
            .matrix_multiply_f32(&left, &right, &pool)
            .expect("compatible matrices must multiply");
        assert_eq!(result.shape().dimensions(), &[2, 2]);
        assert_eq!(values(&result), [58.0, 64.0, 139.0, 154.0]);
    }

    #[test]
    fn aggregate_pool_exhaustion_fails_before_output_allocation() {
        let pool = MemoryPool::new(16).expect("pool must validate");
        let backend = CpuReferenceBackend::new().expect("backend id must validate");
        let left = tensor(&pool, [1, 2], &[1.0, 2.0]);
        let right = tensor(&pool, [1, 2], &[3.0, 4.0]);
        assert!(matches!(
            backend.add_f32(&left, &right, &pool),
            Err(EngineError::ResourceExhaustion(_))
        ));
    }

    #[test]
    fn rejects_shape_pool_and_compute_mismatch() {
        let pool = MemoryPool::new(1 << 20).expect("pool must validate");
        let other_pool = MemoryPool::new(1 << 20).expect("pool must validate");
        let backend = CpuReferenceBackend::new().expect("backend id must validate");
        let left = tensor(&pool, [1, 2], &[1.0, 2.0]);
        let right = tensor(&pool, [1, 2], &[3.0, 4.0]);
        assert!(backend.matrix_multiply_f32(&left, &right, &pool).is_err());
        assert!(backend.add_f32(&left, &right, &other_pool).is_err());
        assert!(checked_operation_count(1, MAX_CPU_REFERENCE_OPERATIONS + 1, 1).is_err());
        assert!(checked_scalar_operation_count(MAX_CPU_REFERENCE_OPERATIONS + 1).is_err());
    }

    #[test]
    fn rejects_non_finite_values() {
        let pool = MemoryPool::new(1 << 20).expect("pool must validate");
        let backend = CpuReferenceBackend::new().expect("backend id must validate");
        let left = tensor(&pool, [1, 2], &[1.0, 2.0]);
        let non_finite = tensor(&pool, [1, 2], &[f32::NAN, 1.0]);
        assert!(backend.add_f32(&left, &non_finite, &pool).is_err());
    }
}
