//! First-party ARM64 graph backend.
//!
//! The backend accelerates bounded contiguous F32 elementwise and matrix-vector
//! operations with isolated ARM64 Advanced SIMD kernels. All remaining closed
//! graph operations use the existing scalar correctness implementation. It has
//! no model, network, filesystem, dynamic-loading, or kernel-selection input.

#[cfg(target_arch = "aarch64")]
mod intrinsics;

use personalai_engine_cpu_reference::{
    Backend, CpuReferenceBackend, MAX_CPU_REFERENCE_OPERATIONS, Operation,
};
use personalai_engine_memory::{MemoryOwner, MemoryPool, MemoryReservation};
use personalai_engine_tensor::{DType, Device, MutableTensor, Tensor, TensorShape};
use personalai_engine_types::{BackendId, EngineError};

/// Exact identifier for this trusted host backend.
pub const ARM64_BACKEND_ID: &str = "arm64";

/// ARM64 graph backend with a scalar correctness fallback for closed operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Arm64Backend {
    id: BackendId,
    scalar: CpuReferenceBackend,
}

impl Arm64Backend {
    /// Constructs the backend only on an ARM64 target.
    pub fn new() -> Result<Self, EngineError> {
        if !cfg!(target_arch = "aarch64") {
            return Err(EngineError::not_implemented("ARM64 backend requires an aarch64 target"));
        }
        Ok(Self { id: BackendId::new(ARM64_BACKEND_ID)?, scalar: CpuReferenceBackend::new()? })
    }

    /// Returns the fixed backend identifier.
    pub const fn id(&self) -> &BackendId {
        &self.id
    }

    /// Returns the trusted host execution device.
    pub const fn device(&self) -> Device {
        Device::Arm64
    }

    fn staging_pool(&self, pool: &MemoryPool) -> MemoryPool {
        pool.scoped(MemoryOwner::Backend(self.id.clone()))
    }

    fn add(&self, left: &Tensor, right: &Tensor, pool: &MemoryPool) -> Result<Tensor, EngineError> {
        if !same_contiguous_f32(left, right, pool)? {
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
        if !same_contiguous_f32(left, right, pool)? {
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
        let operations = rows_u64.checked_mul(columns).ok_or_else(|| {
            EngineError::resource_exhaustion("ARM64 matrix-vector work count overflows")
        })?;
        if operations > MAX_CPU_REFERENCE_OPERATIONS {
            return Err(EngineError::resource_exhaustion(
                "ARM64 matrix-vector exceeds its work ceiling",
            ));
        }
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
        match (operation, inputs) {
            (Operation::Add, [left, right]) => self.add(left, right, pool),
            (Operation::Multiply, [left, right]) => self.multiply(left, right, pool),
            (Operation::ScalarMultiply(scalar), [input]) => {
                self.scalar_multiply(input, *scalar, pool)
            }
            (Operation::MatrixVector, [matrix, vector]) => self.matrix_vector(matrix, vector, pool),
            _ => self.scalar.execute_operation(operation, inputs, pool),
        }
    }
}

fn require_contiguous_f32(tensor: &Tensor, pool: &MemoryPool) -> Result<(), EngineError> {
    if tensor.dtype() != DType::F32 || !tensor.metadata().layout().is_contiguous() {
        return Err(EngineError::invalid_input(
            "ARM64 acceleration requires a contiguous F32 tensor",
        ));
    }
    if !tensor.accounted_by(pool) {
        return Err(EngineError::invalid_input("ARM64 tensor belongs to a different memory pool"));
    }
    if tensor.shape().element_count() > MAX_CPU_REFERENCE_OPERATIONS {
        return Err(EngineError::resource_exhaustion("ARM64 operation exceeds its work ceiling"));
    }
    Ok(())
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

    #[cfg(not(target_arch = "aarch64"))]
    #[test]
    fn construction_is_explicitly_unsupported_off_target() {
        assert!(matches!(Arm64Backend::new(), Err(EngineError::NotImplemented(_))));
    }
}
