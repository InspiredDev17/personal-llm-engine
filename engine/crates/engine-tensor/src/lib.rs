//! Safe, owned, aggregate-accounted tensor storage.
//!
//! Tensor bytes are represented by ordinary Rust allocations. Every owned
//! allocation holds an exact reservation from an engine memory pool. This
//! foundation contains no unsafe code, foreign-memory view, memory map, or
//! device pointer.

use personalai_engine_memory::{MemoryPool, MemoryReservation};
use personalai_engine_types::{DType, EngineError, TensorShape};

/// Safe owned tensor whose byte length exactly matches its type and shape.
#[derive(Debug)]
pub struct Tensor {
    dtype: DType,
    shape: TensorShape,
    bytes: Vec<u8>,
    reservation: MemoryReservation,
}

impl Tensor {
    /// Allocates a zero-filled tensor under aggregate pool accounting.
    pub fn zeros(dtype: DType, shape: TensorShape, pool: &MemoryPool) -> Result<Self, EngineError> {
        let byte_length = shape.byte_len(dtype)?;
        let mut reservation = pool.reserve(byte_length)?;
        let byte_length = usize::try_from(byte_length).map_err(|_| {
            EngineError::resource_exhaustion("tensor allocation exceeds addressable memory")
        })?;
        let mut bytes = Vec::new();
        bytes.try_reserve_exact(byte_length).map_err(|_| {
            EngineError::resource_exhaustion("tensor allocation reservation failed")
        })?;
        bytes.resize(byte_length, 0);
        charge_spare_capacity(pool, &mut reservation, byte_length, bytes.capacity())?;
        Ok(Self { dtype, shape, bytes, reservation })
    }

    /// Takes ownership of exact bytes under aggregate pool accounting.
    pub fn from_bytes(
        dtype: DType,
        shape: TensorShape,
        bytes: Vec<u8>,
        pool: &MemoryPool,
    ) -> Result<Self, EngineError> {
        let expected = shape.byte_len(dtype)?;
        let observed = u64::try_from(bytes.len()).map_err(|_| {
            EngineError::resource_exhaustion("tensor byte length cannot be represented")
        })?;
        if observed != expected {
            return Err(EngineError::invalid_input(
                "tensor byte length does not match its type and shape",
            ));
        }
        let allocated = u64::try_from(bytes.capacity()).map_err(|_| {
            EngineError::resource_exhaustion("tensor allocation capacity cannot be represented")
        })?;
        let reservation = pool.reserve(allocated)?;
        Ok(Self { dtype, shape, bytes, reservation })
    }

    /// Copies exact borrowed bytes after reserving their aggregate allocation.
    pub fn copy_from_bytes(
        dtype: DType,
        shape: TensorShape,
        source: &[u8],
        pool: &MemoryPool,
    ) -> Result<Self, EngineError> {
        let expected = shape.byte_len(dtype)?;
        let observed = u64::try_from(source.len()).map_err(|_| {
            EngineError::resource_exhaustion("tensor source length cannot be represented")
        })?;
        if observed != expected {
            return Err(EngineError::invalid_input(
                "tensor source length does not match its type and shape",
            ));
        }
        let mut tensor = Self::zeros(dtype, shape, pool)?;
        tensor.bytes.copy_from_slice(source);
        Ok(tensor)
    }

    /// Constructs an `f32` tensor using explicit little-endian representation.
    pub fn from_f32_values(
        shape: TensorShape,
        values: &[f32],
        pool: &MemoryPool,
    ) -> Result<Self, EngineError> {
        let expected_elements = usize::try_from(shape.element_count()).map_err(|_| {
            EngineError::resource_exhaustion("tensor element count exceeds addressable memory")
        })?;
        if values.len() != expected_elements {
            return Err(EngineError::invalid_input("f32 value count does not match tensor shape"));
        }
        let mut tensor = Self::zeros(DType::F32, shape, pool)?;
        for (index, value) in values.iter().copied().enumerate() {
            tensor.set_f32_value(index, value)?;
        }
        Ok(tensor)
    }

    /// Returns the element data type.
    pub const fn dtype(&self) -> DType {
        self.dtype
    }

    /// Returns the validated tensor shape.
    pub const fn shape(&self) -> &TensorShape {
        &self.shape
    }

    /// Returns the immutable tensor byte representation.
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Returns whether this tensor is charged to the supplied pool.
    pub fn accounted_by(&self, pool: &MemoryPool) -> bool {
        pool.owns(&self.reservation)
    }

    /// Returns the allocation capacity charged to the memory pool.
    pub const fn reserved_bytes(&self) -> u64 {
        self.reservation.bytes()
    }

    /// Reads one `f32` value without allocating a decoded copy.
    pub fn f32_value(&self, index: usize) -> Result<f32, EngineError> {
        self.require_f32()?;
        let range = self.f32_byte_range(index)?;
        let bytes: [u8; 4] = self.bytes[range]
            .try_into()
            .map_err(|_| EngineError::internal_invariant("validated f32 tensor is truncated"))?;
        Ok(f32::from_le_bytes(bytes))
    }

    /// Writes one `f32` value using explicit little-endian representation.
    pub fn set_f32_value(&mut self, index: usize, value: f32) -> Result<(), EngineError> {
        self.require_f32()?;
        let range = self.f32_byte_range(index)?;
        self.bytes[range].copy_from_slice(&value.to_le_bytes());
        Ok(())
    }

    fn require_f32(&self) -> Result<(), EngineError> {
        if self.dtype == DType::F32 {
            Ok(())
        } else {
            Err(EngineError::invalid_input("tensor data type is not f32"))
        }
    }

    fn f32_byte_range(&self, index: usize) -> Result<std::ops::Range<usize>, EngineError> {
        let start = index
            .checked_mul(4)
            .ok_or_else(|| EngineError::invalid_input("tensor element index overflows"))?;
        let end = start
            .checked_add(4)
            .ok_or_else(|| EngineError::invalid_input("tensor element index overflows"))?;
        if end > self.bytes.len() {
            return Err(EngineError::invalid_input("tensor element index is outside its shape"));
        }
        Ok(start..end)
    }
}

impl PartialEq for Tensor {
    fn eq(&self, other: &Self) -> bool {
        self.dtype == other.dtype && self.shape == other.shape && self.bytes == other.bytes
    }
}

impl Eq for Tensor {}

fn charge_spare_capacity(
    pool: &MemoryPool,
    reservation: &mut MemoryReservation,
    logical_bytes: usize,
    allocated_capacity: usize,
) -> Result<(), EngineError> {
    let spare_capacity = allocated_capacity.checked_sub(logical_bytes).ok_or_else(|| {
        EngineError::internal_invariant("tensor allocation capacity is below its length")
    })?;
    let spare_capacity = u64::try_from(spare_capacity).map_err(|_| {
        EngineError::resource_exhaustion("tensor spare capacity cannot be represented")
    })?;
    pool.grow_reservation(reservation, spare_capacity)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_allocation_matches_checked_shape_bytes() {
        let pool = MemoryPool::new(64).expect("pool must validate");
        let shape = TensorShape::new([2, 3]).expect("shape must validate");
        let tensor =
            Tensor::zeros(DType::F32, shape, &pool).expect("in-budget tensor must allocate");
        assert_eq!(tensor.as_bytes(), &[0_u8; 24]);
        assert!(tensor.accounted_by(&pool));
        assert_eq!(
            pool.used_bytes().expect("accounting must be readable"),
            tensor.reserved_bytes()
        );
        assert!(tensor.reserved_bytes() >= 24);
        drop(tensor);
        assert_eq!(pool.used_bytes().expect("accounting must be readable"), 0);
    }

    #[test]
    fn rejects_wrong_byte_length_and_aggregate_exhaustion() {
        let pool = MemoryPool::new(16).expect("pool must validate");
        let shape = TensorShape::new([2, 2]).expect("shape must validate");
        assert!(Tensor::from_bytes(DType::F32, shape.clone(), vec![0; 15], &pool).is_err());
        let first = Tensor::zeros(DType::F32, shape.clone(), &pool)
            .expect("first tensor must consume the pool");
        assert!(Tensor::zeros(DType::F32, shape, &pool).is_err());
        drop(first);
    }

    #[test]
    fn f32_round_trip_is_explicit_and_safe() {
        let pool = MemoryPool::new(16).expect("pool must validate");
        let shape = TensorShape::new([2, 2]).expect("shape must validate");
        let values = [1.0, -2.5, 3.25, 4.0];
        let tensor =
            Tensor::from_f32_values(shape, &values, &pool).expect("f32 tensor must construct");
        for (index, expected) in values.into_iter().enumerate() {
            let actual = tensor.f32_value(index).expect("value must decode");
            assert!((actual - expected).abs() <= f32::EPSILON);
        }
        assert!(tensor.f32_value(values.len()).is_err());
    }

    #[test]
    fn caller_owned_spare_capacity_is_charged() {
        let pool = MemoryPool::new(16).expect("pool must validate");
        let shape = TensorShape::new([2, 2]).expect("shape must validate");
        let mut bytes = Vec::with_capacity(1_024);
        bytes.resize(16, 0);
        assert!(matches!(
            Tensor::from_bytes(DType::F32, shape, bytes, &pool),
            Err(EngineError::ResourceExhaustion(_))
        ));
        assert_eq!(pool.used_bytes().expect("accounting must be readable"), 0);
    }

    #[test]
    fn borrowed_bytes_are_reserved_before_copying() {
        let pool = MemoryPool::new(16).expect("pool must validate");
        let shape = TensorShape::new([2, 2]).expect("shape must validate");
        let tensor = Tensor::copy_from_bytes(DType::F32, shape, &[1_u8; 16], &pool)
            .expect("borrowed bytes must copy");
        assert_eq!(tensor.as_bytes(), &[1_u8; 16]);
        assert!(tensor.accounted_by(&pool));
    }
}
