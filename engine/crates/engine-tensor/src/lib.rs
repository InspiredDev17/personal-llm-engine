//! Safe tensor metadata, layouts, views, and aggregate-accounted storage.
//!
//! The canonical owned layout is dense row-major with the final dimension
//! contiguous. Explicit views may be transposed or strided, but every logical
//! index is checked through its declared layout. Storage remains ordinary Rust
//! bytes; no typed pointer cast, platform acceleration, or unsafe code exists.

use personalai_engine_memory::{MemoryPool, MemoryReservation};
pub use personalai_engine_types::{DType, TensorShape};
use personalai_engine_types::{EngineError, MAX_MODEL_PACKAGE_BYTES};
use std::ops::Range;

/// Tensor failures use the engine's closed, bounded error taxonomy.
pub type TensorError = EngineError;

/// Maximum logical alignment accepted in tensor metadata.
pub const MAX_TENSOR_ALIGNMENT: u64 = 2 * 1024 * 1024;

/// Execution device selected by trusted engine configuration.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Device {
    /// Safe scalar host execution.
    Cpu,
    /// First-party ARM64 Advanced SIMD host execution.
    Arm64,
    /// First-party Apple Metal GPU execution.
    Metal,
}

/// Validated logical tensor layout expressed in element strides.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorLayout {
    shape: TensorShape,
    strides: Box<[u64]>,
    offset_elements: u64,
}

impl TensorLayout {
    /// Constructs the canonical dense row-major layout.
    pub fn contiguous(shape: TensorShape) -> Result<Self, TensorError> {
        let mut strides = vec![0_u64; shape.rank()];
        let mut stride = 1_u64;
        for index in (0..shape.rank()).rev() {
            strides[index] = stride;
            stride = stride
                .checked_mul(shape.dimensions()[index])
                .ok_or_else(|| EngineError::resource_exhaustion("tensor stride overflows"))?;
        }
        Ok(Self { shape, strides: strides.into_boxed_slice(), offset_elements: 0 })
    }

    /// Constructs an explicit bounded layout against a storage element count.
    pub fn new(
        shape: TensorShape,
        strides: impl Into<Box<[u64]>>,
        offset_elements: u64,
        storage_elements: u64,
    ) -> Result<Self, TensorError> {
        let strides = strides.into();
        if strides.len() != shape.rank() || strides.iter().any(|stride| *stride == 0) {
            return Err(EngineError::invalid_input("tensor strides are invalid"));
        }
        let layout = Self { shape, strides, offset_elements };
        if storage_elements == 0 || layout.last_reachable_element()? >= storage_elements {
            return Err(EngineError::invalid_input("tensor layout exceeds its storage"));
        }
        Ok(layout)
    }

    /// Returns the logical tensor shape.
    pub const fn shape(&self) -> &TensorShape {
        &self.shape
    }

    /// Returns element strides in logical dimension order.
    pub fn strides(&self) -> &[u64] {
        &self.strides
    }

    /// Returns the storage element offset of logical element zero.
    pub const fn offset_elements(&self) -> u64 {
        self.offset_elements
    }

    /// Reports whether this is the canonical dense row-major layout.
    pub fn is_contiguous(&self) -> bool {
        if self.offset_elements != 0 {
            return false;
        }
        let mut expected = 1_u64;
        for index in (0..self.shape.rank()).rev() {
            if self.strides[index] != expected {
                return false;
            }
            let Some(next) = expected.checked_mul(self.shape.dimensions()[index]) else {
                return false;
            };
            expected = next;
        }
        true
    }

    /// Returns a view with two logical axes exchanged.
    pub fn transpose(&self, first: usize, second: usize) -> Result<Self, TensorError> {
        if first >= self.shape.rank() || second >= self.shape.rank() {
            return Err(EngineError::invalid_input("transpose axis is outside tensor rank"));
        }
        let mut dimensions = self.shape.dimensions().to_vec();
        let mut strides = self.strides.to_vec();
        dimensions.swap(first, second);
        strides.swap(first, second);
        Ok(Self {
            shape: TensorShape::new(dimensions)?,
            strides: strides.into_boxed_slice(),
            offset_elements: self.offset_elements,
        })
    }

    /// Reshapes a canonical contiguous layout without moving storage.
    pub fn reshape(&self, shape: TensorShape) -> Result<Self, TensorError> {
        if !self.is_contiguous() || shape.element_count() != self.shape.element_count() {
            return Err(EngineError::invalid_input(
                "reshape requires a contiguous layout with equal element count",
            ));
        }
        Self::contiguous(shape)
    }

    /// Maps one row-major logical index to its storage element offset.
    pub fn storage_offset(&self, logical_index: u64) -> Result<u64, TensorError> {
        if logical_index >= self.shape.element_count() {
            return Err(EngineError::invalid_input("tensor element index is outside its shape"));
        }
        if self.shape.is_scalar() {
            return Ok(self.offset_elements);
        }
        let mut remainder = logical_index;
        let mut offset = self.offset_elements;
        for index in (0..self.shape.rank()).rev() {
            let dimension = self.shape.dimensions()[index];
            let coordinate = remainder % dimension;
            remainder /= dimension;
            offset = coordinate
                .checked_mul(self.strides[index])
                .and_then(|part| offset.checked_add(part))
                .ok_or_else(|| EngineError::resource_exhaustion("tensor index overflows"))?;
        }
        Ok(offset)
    }

    fn last_reachable_element(&self) -> Result<u64, TensorError> {
        self.shape.dimensions().iter().copied().zip(self.strides.iter().copied()).try_fold(
            self.offset_elements,
            |offset, (dimension, stride)| {
                dimension
                    .checked_sub(1)
                    .and_then(|value| value.checked_mul(stride))
                    .and_then(|part| offset.checked_add(part))
                    .ok_or_else(|| EngineError::resource_exhaustion("tensor layout span overflows"))
            },
        )
    }
}

/// Immutable validated description of tensor bytes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorMetadata {
    dtype: DType,
    layout: TensorLayout,
    required_alignment: u64,
}

impl TensorMetadata {
    /// Constructs metadata after validating storage capacity and alignment.
    pub fn new(
        dtype: DType,
        layout: TensorLayout,
        required_alignment: u64,
        storage_bytes: u64,
    ) -> Result<Self, TensorError> {
        if required_alignment == 0
            || !required_alignment.is_power_of_two()
            || required_alignment > MAX_TENSOR_ALIGNMENT
        {
            return Err(EngineError::invalid_input("tensor alignment is invalid"));
        }
        let offset_bytes = layout
            .offset_elements()
            .checked_mul(dtype.byte_width())
            .ok_or_else(|| EngineError::resource_exhaustion("tensor byte offset overflows"))?;
        if offset_bytes % required_alignment != 0 {
            return Err(EngineError::invalid_input("tensor offset violates required alignment"));
        }
        let required_bytes = layout
            .last_reachable_element()?
            .checked_add(1)
            .and_then(|elements| elements.checked_mul(dtype.byte_width()))
            .ok_or_else(|| EngineError::resource_exhaustion("tensor storage span overflows"))?;
        if required_bytes > storage_bytes || storage_bytes > MAX_MODEL_PACKAGE_BYTES {
            return Err(EngineError::invalid_input("tensor storage capacity is invalid"));
        }
        Ok(Self { dtype, layout, required_alignment })
    }

    /// Constructs exact canonical owned metadata.
    pub fn contiguous(dtype: DType, shape: TensorShape) -> Result<Self, TensorError> {
        let storage_bytes = shape.byte_len(dtype)?;
        Self::new(dtype, TensorLayout::contiguous(shape)?, dtype.byte_width(), storage_bytes)
    }

    /// Returns the storage data type.
    pub const fn dtype(&self) -> DType {
        self.dtype
    }

    /// Returns the logical shape.
    pub const fn shape(&self) -> &TensorShape {
        self.layout.shape()
    }

    /// Returns the explicit logical layout.
    pub const fn layout(&self) -> &TensorLayout {
        &self.layout
    }

    /// Returns the logical byte alignment requirement.
    pub const fn required_alignment(&self) -> u64 {
        self.required_alignment
    }

    /// Returns the exact bytes for a canonical owned tensor.
    pub fn canonical_byte_len(&self) -> Result<u64, TensorError> {
        self.shape().byte_len(self.dtype)
    }
}

/// Aggregate-accounted owned byte storage.
#[derive(Debug)]
pub struct TensorStorage {
    bytes: Vec<u8>,
    reservation: MemoryReservation,
}

impl TensorStorage {
    fn zeros(length: usize, pool: &MemoryPool) -> Result<Self, TensorError> {
        let mut reservation = pool.reserve(u64_from_usize(length)?)?;
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(length)
            .map_err(|_| EngineError::resource_exhaustion("tensor allocation failed"))?;
        bytes.resize(length, 0);
        charge_spare_capacity(pool, &mut reservation, length, bytes.capacity())?;
        Ok(Self { bytes, reservation })
    }

    fn from_bytes(bytes: Vec<u8>, pool: &MemoryPool) -> Result<Self, TensorError> {
        let reservation = pool.reserve(u64_from_usize(bytes.capacity())?)?;
        Ok(Self { bytes, reservation })
    }

    /// Returns immutable storage bytes.
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Returns whether this allocation belongs to the supplied pool.
    pub fn accounted_by(&self, pool: &MemoryPool) -> bool {
        pool.owns(&self.reservation)
    }

    /// Returns allocated capacity charged to the pool.
    pub const fn reserved_bytes(&self) -> u64 {
        self.reservation.bytes()
    }

    fn as_bytes_mut(&mut self) -> &mut [u8] {
        &mut self.bytes
    }
}

/// Read access shared by owning tensors and explicit views.
pub trait TensorRead {
    /// Returns validated metadata.
    fn metadata(&self) -> &TensorMetadata;
    /// Returns immutable backing bytes.
    fn as_bytes(&self) -> &[u8];
    /// Reports whether backing storage is charged to the supplied pool.
    fn accounted_by(&self, pool: &MemoryPool) -> bool;

    /// Returns one logical element's exact storage bytes.
    fn element_bytes(&self, index: usize) -> Result<&[u8], TensorError> {
        let offset = self.metadata().layout().storage_offset(u64_from_usize(index)?)?;
        let width = usize_from_u64(self.dtype().byte_width())?;
        let range = byte_range(offset, width, self.as_bytes().len())?;
        Ok(&self.as_bytes()[range])
    }

    /// Returns the storage data type.
    fn dtype(&self) -> DType {
        self.metadata().dtype()
    }

    /// Returns the logical shape.
    fn shape(&self) -> &TensorShape {
        self.metadata().shape()
    }

    /// Reads an execution `f32` without accepting another dtype.
    fn f32_value(&self, index: usize) -> Result<f32, TensorError> {
        if self.dtype() != DType::F32 {
            return Err(EngineError::invalid_input("tensor data type is not f32"));
        }
        let offset = self.metadata().layout().storage_offset(u64_from_usize(index)?)?;
        read_f32(self.as_bytes(), offset)
    }

    /// Reads F32, F16, or BF16 storage as an execution `f32`.
    fn numeric_f32_value(&self, index: usize) -> Result<f32, TensorError> {
        let offset = self.metadata().layout().storage_offset(u64_from_usize(index)?)?;
        match self.dtype() {
            DType::F32 => read_f32(self.as_bytes(), offset),
            DType::F16 => Ok(f16_bits_to_f32(read_u16(self.as_bytes(), offset)?)),
            DType::BFloat16 => Ok(bf16_bits_to_f32(read_u16(self.as_bytes(), offset)?)),
            _ => {
                Err(EngineError::not_implemented("tensor storage type cannot be converted to f32"))
            }
        }
    }

    /// Reads one U32 token/index value.
    fn u32_value(&self, index: usize) -> Result<u32, TensorError> {
        if self.dtype() != DType::U32 {
            return Err(EngineError::invalid_input("tensor data type is not u32"));
        }
        let offset = self.metadata().layout().storage_offset(u64_from_usize(index)?)?;
        let range = byte_range(offset, 4, self.as_bytes().len())?;
        Ok(u32::from_le_bytes(
            self.as_bytes()[range]
                .try_into()
                .map_err(|_| EngineError::internal_invariant("validated u32 tensor truncated"))?,
        ))
    }
}

/// Immutable owning tensor.
#[derive(Debug)]
pub struct ReadOnlyTensor {
    metadata: TensorMetadata,
    storage: TensorStorage,
}

impl ReadOnlyTensor {
    /// Copies exact canonical bytes into immutable accounted storage.
    pub fn copy_from_bytes(
        dtype: DType,
        shape: TensorShape,
        source: &[u8],
        pool: &MemoryPool,
    ) -> Result<Self, TensorError> {
        MutableTensor::copy_from_bytes(dtype, shape, source, pool).map(MutableTensor::freeze)
    }

    /// Returns an explicit immutable view.
    pub fn view(&self) -> TensorView<'_> {
        TensorView { metadata: self.metadata.clone(), storage: &self.storage }
    }

    /// Returns validated metadata.
    pub const fn metadata(&self) -> &TensorMetadata {
        &self.metadata
    }

    /// Returns allocation capacity charged to the pool.
    pub const fn reserved_bytes(&self) -> u64 {
        self.storage.reserved_bytes()
    }

    /// Reads one strict F32 element.
    pub fn f32_value(&self, index: usize) -> Result<f32, TensorError> {
        TensorRead::f32_value(self, index)
    }
}

impl TensorRead for ReadOnlyTensor {
    fn metadata(&self) -> &TensorMetadata {
        &self.metadata
    }

    fn as_bytes(&self) -> &[u8] {
        self.storage.as_bytes()
    }

    fn accounted_by(&self, pool: &MemoryPool) -> bool {
        self.storage.accounted_by(pool)
    }
}

/// Mutable owning tensor used only for engine-controlled destinations/state.
#[derive(Debug)]
pub struct MutableTensor {
    metadata: TensorMetadata,
    storage: TensorStorage,
}

/// Compatibility name for existing engine-owned mutable tensors.
pub type Tensor = MutableTensor;

impl MutableTensor {
    /// Allocates a canonical zero-filled tensor.
    pub fn zeros(dtype: DType, shape: TensorShape, pool: &MemoryPool) -> Result<Self, TensorError> {
        let metadata = TensorMetadata::contiguous(dtype, shape)?;
        let length = usize_from_u64(metadata.canonical_byte_len()?)?;
        let storage = TensorStorage::zeros(length, pool)?;
        Ok(Self { metadata, storage })
    }

    /// Takes ownership of exact canonical bytes.
    pub fn from_bytes(
        dtype: DType,
        shape: TensorShape,
        bytes: Vec<u8>,
        pool: &MemoryPool,
    ) -> Result<Self, TensorError> {
        let metadata = TensorMetadata::contiguous(dtype, shape)?;
        if u64_from_usize(bytes.len())? != metadata.canonical_byte_len()? {
            return Err(EngineError::invalid_input(
                "tensor byte length does not match its type and shape",
            ));
        }
        let storage = TensorStorage::from_bytes(bytes, pool)?;
        Ok(Self { metadata, storage })
    }

    /// Copies exact canonical bytes after reserving destination storage.
    pub fn copy_from_bytes(
        dtype: DType,
        shape: TensorShape,
        source: &[u8],
        pool: &MemoryPool,
    ) -> Result<Self, TensorError> {
        let mut tensor = Self::zeros(dtype, shape, pool)?;
        if tensor.storage.as_bytes().len() != source.len() {
            return Err(EngineError::invalid_input(
                "tensor source length does not match its type and shape",
            ));
        }
        tensor.storage.as_bytes_mut().copy_from_slice(source);
        Ok(tensor)
    }

    /// Constructs an F32 execution tensor from exact values.
    pub fn from_f32_values(
        shape: TensorShape,
        values: &[f32],
        pool: &MemoryPool,
    ) -> Result<Self, TensorError> {
        let mut tensor = Self::zeros(DType::F32, shape, pool)?;
        if usize_from_u64(tensor.shape().element_count())? != values.len() {
            return Err(EngineError::invalid_input("f32 value count does not match tensor shape"));
        }
        for (index, value) in values.iter().copied().enumerate() {
            tensor.set_f32_value(index, value)?;
        }
        Ok(tensor)
    }

    /// Constructs exact F16 storage bits.
    pub fn from_f16_bits(
        shape: TensorShape,
        values: &[u16],
        pool: &MemoryPool,
    ) -> Result<Self, TensorError> {
        Self::from_u16_storage(DType::F16, shape, values, pool)
    }

    /// Constructs exact BF16 storage bits.
    pub fn from_bf16_bits(
        shape: TensorShape,
        values: &[u16],
        pool: &MemoryPool,
    ) -> Result<Self, TensorError> {
        Self::from_u16_storage(DType::BFloat16, shape, values, pool)
    }

    /// Constructs exact U32 token/index storage.
    pub fn from_u32_values(
        shape: TensorShape,
        values: &[u32],
        pool: &MemoryPool,
    ) -> Result<Self, TensorError> {
        if usize_from_u64(shape.element_count())? != values.len() {
            return Err(EngineError::invalid_input("u32 value count does not match tensor shape"));
        }
        let mut tensor = Self::zeros(DType::U32, shape, pool)?;
        for (index, value) in values.iter().copied().enumerate() {
            let range = byte_range(u64_from_usize(index)?, 4, tensor.storage.bytes.len())?;
            tensor.storage.bytes[range].copy_from_slice(&value.to_le_bytes());
        }
        Ok(tensor)
    }

    /// Converts validated F32/F16/BF16 storage into canonical F32 execution storage.
    pub fn convert_storage_to_f32(
        dtype: DType,
        shape: TensorShape,
        source: &[u8],
        pool: &MemoryPool,
    ) -> Result<Self, TensorError> {
        let storage = Self::copy_from_bytes(dtype, shape.clone(), source, pool)?;
        if dtype == DType::F32 {
            return Ok(storage);
        }
        if !matches!(dtype, DType::F16 | DType::BFloat16) {
            return Err(EngineError::not_implemented(
                "model storage type has no CPU f32 conversion",
            ));
        }
        let mut output = Self::zeros(DType::F32, shape, pool)?;
        for index in 0..usize_from_u64(output.shape().element_count())? {
            output.set_f32_value(index, storage.numeric_f32_value(index)?)?;
        }
        Ok(output)
    }

    fn from_u16_storage(
        dtype: DType,
        shape: TensorShape,
        values: &[u16],
        pool: &MemoryPool,
    ) -> Result<Self, TensorError> {
        if usize_from_u64(shape.element_count())? != values.len() {
            return Err(EngineError::invalid_input("u16 value count does not match tensor shape"));
        }
        let mut tensor = Self::zeros(dtype, shape, pool)?;
        for (index, value) in values.iter().copied().enumerate() {
            let range = byte_range(u64_from_usize(index)?, 2, tensor.storage.bytes.len())?;
            tensor.storage.bytes[range].copy_from_slice(&value.to_le_bytes());
        }
        Ok(tensor)
    }

    /// Consumes mutable ownership and exposes immutable tensor storage.
    pub fn freeze(self) -> ReadOnlyTensor {
        ReadOnlyTensor { metadata: self.metadata, storage: self.storage }
    }

    /// Returns an immutable explicit view.
    pub fn view(&self) -> TensorView<'_> {
        TensorView { metadata: self.metadata.clone(), storage: &self.storage }
    }

    /// Returns a mutable explicit view.
    pub fn view_mut(&mut self) -> TensorViewMut<'_> {
        TensorViewMut { metadata: self.metadata.clone(), storage: &mut self.storage }
    }

    /// Returns the storage data type.
    pub const fn dtype(&self) -> DType {
        self.metadata.dtype()
    }

    /// Returns the logical shape.
    pub const fn shape(&self) -> &TensorShape {
        self.metadata.shape()
    }

    /// Returns validated metadata.
    pub const fn metadata(&self) -> &TensorMetadata {
        &self.metadata
    }

    /// Returns immutable storage bytes.
    pub fn as_bytes(&self) -> &[u8] {
        self.storage.as_bytes()
    }

    /// Returns whether storage is charged to the supplied pool.
    pub fn accounted_by(&self, pool: &MemoryPool) -> bool {
        self.storage.accounted_by(pool)
    }

    /// Returns allocation capacity charged to the supplied pool.
    pub const fn reserved_bytes(&self) -> u64 {
        self.storage.reserved_bytes()
    }

    /// Reads one strict F32 element.
    pub fn f32_value(&self, index: usize) -> Result<f32, TensorError> {
        TensorRead::f32_value(self, index)
    }

    /// Reads supported floating storage as F32.
    pub fn numeric_f32_value(&self, index: usize) -> Result<f32, TensorError> {
        TensorRead::numeric_f32_value(self, index)
    }

    /// Writes one F32 destination element.
    pub fn set_f32_value(&mut self, index: usize, value: f32) -> Result<(), TensorError> {
        self.view_mut().set_f32_value(index, value)
    }

    /// Writes one exact storage element through the validated owned layout.
    pub fn set_element_bytes(&mut self, index: usize, value: &[u8]) -> Result<(), TensorError> {
        self.view_mut().set_element_bytes(index, value)
    }

    /// Clears every allocated storage byte before state reuse or release.
    pub fn clear(&mut self) {
        self.storage.as_bytes_mut().fill(0);
    }
}

impl TensorRead for MutableTensor {
    fn metadata(&self) -> &TensorMetadata {
        &self.metadata
    }

    fn as_bytes(&self) -> &[u8] {
        self.storage.as_bytes()
    }

    fn accounted_by(&self, pool: &MemoryPool) -> bool {
        self.storage.accounted_by(pool)
    }
}

impl PartialEq for MutableTensor {
    fn eq(&self, other: &Self) -> bool {
        self.metadata == other.metadata && self.storage.bytes == other.storage.bytes
    }
}

impl Eq for MutableTensor {}

/// Borrowed immutable tensor with explicit layout.
#[derive(Debug)]
pub struct TensorView<'a> {
    metadata: TensorMetadata,
    storage: &'a TensorStorage,
}

impl<'a> TensorView<'a> {
    /// Returns a transposed view without moving bytes.
    pub fn transpose(&self, first: usize, second: usize) -> Result<TensorView<'a>, TensorError> {
        let layout = self.metadata.layout().transpose(first, second)?;
        let metadata = TensorMetadata::new(
            self.dtype(),
            layout,
            self.metadata.required_alignment(),
            u64_from_usize(self.storage.bytes.len())?,
        )?;
        Ok(TensorView { metadata, storage: self.storage })
    }

    /// Returns a differently shaped contiguous view.
    pub fn reshape(&self, shape: TensorShape) -> Result<TensorView<'a>, TensorError> {
        let layout = self.metadata.layout().reshape(shape)?;
        let metadata = TensorMetadata::new(
            self.dtype(),
            layout,
            self.metadata.required_alignment(),
            u64_from_usize(self.storage.bytes.len())?,
        )?;
        Ok(TensorView { metadata, storage: self.storage })
    }
}

impl TensorRead for TensorView<'_> {
    fn metadata(&self) -> &TensorMetadata {
        &self.metadata
    }

    fn as_bytes(&self) -> &[u8] {
        self.storage.as_bytes()
    }

    fn accounted_by(&self, pool: &MemoryPool) -> bool {
        self.storage.accounted_by(pool)
    }
}

/// Borrowed mutable destination with explicit layout.
#[derive(Debug)]
pub struct TensorViewMut<'a> {
    metadata: TensorMetadata,
    storage: &'a mut TensorStorage,
}

impl<'a> TensorViewMut<'a> {
    /// Returns validated metadata.
    pub const fn metadata(&self) -> &TensorMetadata {
        &self.metadata
    }

    /// Consumes this exclusive borrow and returns a transposed mutable view.
    pub fn transpose(self, first: usize, second: usize) -> Result<TensorViewMut<'a>, TensorError> {
        let layout = self.metadata.layout().transpose(first, second)?;
        let metadata = TensorMetadata::new(
            self.metadata.dtype(),
            layout,
            self.metadata.required_alignment(),
            u64_from_usize(self.storage.bytes.len())?,
        )?;
        Ok(TensorViewMut { metadata, storage: self.storage })
    }

    /// Consumes this exclusive borrow and returns a reshaped contiguous view.
    pub fn reshape(self, shape: TensorShape) -> Result<TensorViewMut<'a>, TensorError> {
        let layout = self.metadata.layout().reshape(shape)?;
        let metadata = TensorMetadata::new(
            self.metadata.dtype(),
            layout,
            self.metadata.required_alignment(),
            u64_from_usize(self.storage.bytes.len())?,
        )?;
        Ok(TensorViewMut { metadata, storage: self.storage })
    }

    /// Writes one F32 logical element through explicit strides.
    pub fn set_f32_value(&mut self, index: usize, value: f32) -> Result<(), TensorError> {
        if self.metadata.dtype() != DType::F32 {
            return Err(EngineError::invalid_input("tensor destination type is not f32"));
        }
        let offset = self.metadata.layout().storage_offset(u64_from_usize(index)?)?;
        let range = byte_range(offset, 4, self.storage.bytes.len())?;
        self.storage.bytes[range].copy_from_slice(&value.to_le_bytes());
        Ok(())
    }

    /// Writes one exact storage element after validating its byte width.
    pub fn set_element_bytes(&mut self, index: usize, value: &[u8]) -> Result<(), TensorError> {
        let width = usize_from_u64(self.metadata.dtype().byte_width())?;
        if value.len() != width {
            return Err(EngineError::invalid_input(
                "tensor element source has the wrong byte width",
            ));
        }
        let offset = self.metadata.layout().storage_offset(u64_from_usize(index)?)?;
        let range = byte_range(offset, width, self.storage.bytes.len())?;
        self.storage.bytes[range].copy_from_slice(value);
        Ok(())
    }
}

/// Converts one IEEE-754 binary16 bit pattern exactly into binary32.
pub fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = u32::from(bits & 0x8000) << 16;
    let exponent = (bits >> 10) & 0x1f;
    let fraction = bits & 0x03ff;
    let output = match exponent {
        0 if fraction == 0 => sign,
        0 => {
            let mut fraction = u32::from(fraction);
            let mut exponent = -14_i32;
            while fraction & 0x0400 == 0 {
                fraction <<= 1;
                exponent -= 1;
            }
            fraction &= 0x03ff;
            sign | (u32::try_from(exponent + 127).unwrap_or(0) << 23) | (fraction << 13)
        }
        0x1f => sign | 0x7f80_0000 | (u32::from(fraction) << 13),
        _ => sign | (u32::from(exponent + 112) << 23) | (u32::from(fraction) << 13),
    };
    f32::from_bits(output)
}

/// Converts binary32 to binary16 using round-to-nearest, ties-to-even.
pub fn f32_to_f16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = u16::try_from((bits >> 16) & 0x8000).unwrap_or(0);
    let exponent = i32::try_from((bits >> 23) & 0xff).unwrap_or(0);
    let fraction = bits & 0x007f_ffff;
    if exponent == 0xff {
        if fraction == 0 {
            return sign | 0x7c00;
        }
        let payload = u16::try_from(fraction >> 13).unwrap_or(0) | 0x0200;
        return sign | 0x7c00 | payload;
    }
    let half_exponent = exponent - 127 + 15;
    if half_exponent >= 31 {
        return sign | 0x7c00;
    }
    if half_exponent <= 0 {
        if half_exponent < -10 {
            return sign;
        }
        let significand = fraction | 0x0080_0000;
        let shift = u32::try_from(14 - half_exponent).unwrap_or(24);
        let mut rounded = significand >> shift;
        let mask = (1_u32 << shift) - 1;
        let remainder = significand & mask;
        let halfway = 1_u32 << (shift - 1);
        if remainder > halfway || (remainder == halfway && rounded & 1 != 0) {
            rounded += 1;
        }
        return sign | u16::try_from(rounded).unwrap_or(0x0400);
    }
    let mut rounded = (u32::try_from(half_exponent).unwrap_or(0) << 10) | (fraction >> 13);
    let remainder = fraction & 0x1fff;
    if remainder > 0x1000 || (remainder == 0x1000 && rounded & 1 != 0) {
        rounded += 1;
    }
    sign | u16::try_from(rounded).unwrap_or(0x7c00)
}

/// Converts one BF16 bit pattern exactly into binary32.
pub fn bf16_bits_to_f32(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

/// Converts binary32 to BF16 using round-to-nearest, ties-to-even.
pub fn f32_to_bf16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    if bits & 0x7f80_0000 == 0x7f80_0000 && bits & 0x007f_ffff != 0 {
        return u16::try_from(bits >> 16).unwrap_or(0) | 0x0040;
    }
    let rounding_bias = 0x7fff_u32 + ((bits >> 16) & 1);
    u16::try_from(bits.wrapping_add(rounding_bias) >> 16).unwrap_or(0)
}

fn read_f32(bytes: &[u8], element_offset: u64) -> Result<f32, TensorError> {
    let range = byte_range(element_offset, 4, bytes.len())?;
    Ok(f32::from_le_bytes(
        bytes[range]
            .try_into()
            .map_err(|_| EngineError::internal_invariant("validated f32 tensor is truncated"))?,
    ))
}

fn read_u16(bytes: &[u8], element_offset: u64) -> Result<u16, TensorError> {
    let range = byte_range(element_offset, 2, bytes.len())?;
    Ok(u16::from_le_bytes(
        bytes[range]
            .try_into()
            .map_err(|_| EngineError::internal_invariant("validated u16 tensor is truncated"))?,
    ))
}

fn byte_range(
    element_offset: u64,
    width: usize,
    storage_length: usize,
) -> Result<Range<usize>, TensorError> {
    let width_u64 = u64_from_usize(width)?;
    let start = element_offset
        .checked_mul(width_u64)
        .ok_or_else(|| EngineError::resource_exhaustion("tensor byte index overflows"))?;
    let end = start
        .checked_add(width_u64)
        .ok_or_else(|| EngineError::resource_exhaustion("tensor byte index overflows"))?;
    let start = usize_from_u64(start)?;
    let end = usize_from_u64(end)?;
    if end > storage_length {
        return Err(EngineError::invalid_input("tensor element exceeds storage capacity"));
    }
    Ok(start..end)
}

fn charge_spare_capacity(
    pool: &MemoryPool,
    reservation: &mut MemoryReservation,
    logical_bytes: usize,
    allocated_capacity: usize,
) -> Result<(), TensorError> {
    let spare = allocated_capacity.checked_sub(logical_bytes).ok_or_else(|| {
        EngineError::internal_invariant("tensor capacity is below allocated length")
    })?;
    pool.grow_reservation(reservation, u64_from_usize(spare)?)
}

fn usize_from_u64(value: u64) -> Result<usize, TensorError> {
    usize::try_from(value)
        .map_err(|_| EngineError::resource_exhaustion("tensor value is not addressable"))
}

fn u64_from_usize(value: usize) -> Result<u64, TensorError> {
    u64::try_from(value)
        .map_err(|_| EngineError::resource_exhaustion("tensor value is not representable"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn values(tensor: &impl TensorRead) -> Vec<f32> {
        (0..usize_from_u64(tensor.shape().element_count()).unwrap())
            .map(|index| tensor.numeric_f32_value(index).unwrap())
            .collect()
    }

    #[test]
    fn owned_storage_is_exact_and_aggregate_accounted() {
        let pool = MemoryPool::new(64).unwrap();
        let tensor =
            MutableTensor::zeros(DType::F32, TensorShape::new([2, 3]).unwrap(), &pool).unwrap();
        assert_eq!(tensor.as_bytes(), &[0_u8; 24]);
        assert!(tensor.accounted_by(&pool));
        assert_eq!(pool.used_bytes().unwrap(), tensor.reserved_bytes());
        drop(tensor);
        assert_eq!(pool.used_bytes().unwrap(), 0);
    }

    #[test]
    fn invalid_length_capacity_dtype_and_pool_fail_closed() {
        let pool = MemoryPool::new(16).unwrap();
        let shape = TensorShape::new([2, 2]).unwrap();
        assert!(MutableTensor::from_bytes(DType::F32, shape.clone(), vec![0; 15], &pool).is_err());
        let mut bytes = Vec::with_capacity(1_024);
        bytes.resize(16, 0);
        assert!(MutableTensor::from_bytes(DType::F32, shape, bytes, &pool).is_err());
        let u8_tensor =
            MutableTensor::zeros(DType::U8, TensorShape::new([1]).unwrap(), &pool).unwrap();
        assert!(u8_tensor.f32_value(0).is_err());
    }

    #[test]
    fn explicit_transpose_does_not_claim_contiguity() {
        let pool = MemoryPool::new(128).unwrap();
        let tensor = MutableTensor::from_f32_values(
            TensorShape::new([2, 3]).unwrap(),
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            &pool,
        )
        .unwrap();
        let transposed = tensor.view().transpose(0, 1).unwrap();
        assert_eq!(transposed.shape().dimensions(), &[3, 2]);
        assert!(!transposed.metadata().layout().is_contiguous());
        assert_eq!(values(&transposed), [1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
        assert!(transposed.reshape(TensorShape::new([6]).unwrap()).is_err());
    }

    #[test]
    fn invalid_strides_offsets_alignment_and_axes_are_rejected() {
        let shape = TensorShape::new([2, 2]).unwrap();
        assert!(TensorLayout::new(shape.clone(), vec![2, 0], 0, 4).is_err());
        assert!(TensorLayout::new(shape.clone(), vec![4, 1], 0, 4).is_err());
        let layout = TensorLayout::contiguous(shape).unwrap();
        assert!(TensorMetadata::new(DType::F32, layout.clone(), 3, 16).is_err());
        assert!(layout.transpose(0, 2).is_err());
    }

    #[test]
    fn f16_exhaustive_round_trip_and_classification_are_ieee_correct() {
        for bits in 0_u16..=u16::MAX {
            let value = f16_bits_to_f32(bits);
            let exponent = bits & 0x7c00;
            let fraction = bits & 0x03ff;
            let round_trip = f32_to_f16_bits(value);
            if exponent == 0x7c00 && fraction != 0 {
                assert!(f16_bits_to_f32(round_trip).is_nan());
                assert_eq!(round_trip & 0x8000, bits & 0x8000);
            } else {
                assert_eq!(round_trip, bits);
            }
        }
    }

    #[test]
    fn f16_rounding_boundaries_and_extremes_are_stable() {
        assert_eq!(f32_to_f16_bits(0.0), 0x0000);
        assert_eq!(f32_to_f16_bits(-0.0), 0x8000);
        assert_eq!(f32_to_f16_bits(1.0), 0x3c00);
        assert_eq!(f32_to_f16_bits(65_504.0), 0x7bff);
        assert_eq!(f32_to_f16_bits(f32::INFINITY), 0x7c00);
        assert_eq!(f32_to_f16_bits(f32::NEG_INFINITY), 0xfc00);
        assert_eq!(f32_to_f16_bits(2.0_f32.powi(-24)), 0x0001);
        assert_eq!(f32_to_f16_bits(1.0 + 2.0_f32.powi(-11)), 0x3c00);
    }

    #[test]
    fn bf16_rounding_nan_and_extremes_are_stable() {
        for bits in [0x0000, 0x8000, 0x3f80, 0x7f80, 0xff80, 0x0001, 0x7f7f] {
            assert_eq!(f32_to_bf16_bits(bf16_bits_to_f32(bits)), bits);
        }
        assert!(bf16_bits_to_f32(f32_to_bf16_bits(f32::NAN)).is_nan());
        assert_eq!(f32_to_bf16_bits(f32::from_bits(0x3f80_8000)), 0x3f80);
        assert_eq!(f32_to_bf16_bits(f32::from_bits(0x3f81_8000)), 0x3f82);
    }

    #[test]
    fn f16_and_bf16_storage_convert_to_f32_without_silent_dtype_changes() {
        let pool = MemoryPool::new(1 << 20).unwrap();
        let shape = TensorShape::new([4]).unwrap();
        let f16 =
            MutableTensor::from_f16_bits(shape.clone(), &[0x3c00, 0xc000, 0x0001, 0x7bff], &pool)
                .unwrap();
        assert_eq!(values(&f16), [1.0, -2.0, 2.0_f32.powi(-24), 65_504.0]);
        let bf16 =
            MutableTensor::from_bf16_bits(shape, &[0x3f80, 0xc000, 0x0001, 0x7f7f], &pool).unwrap();
        assert_eq!(values(&bf16)[..2], [1.0, -2.0]);
        assert!(bf16.numeric_f32_value(2).unwrap().is_finite());
        assert!(bf16.numeric_f32_value(3).unwrap().is_finite());
    }

    #[test]
    fn mutable_and_read_only_views_preserve_access_boundaries() {
        let pool = MemoryPool::new(64).unwrap();
        let mut tensor =
            MutableTensor::zeros(DType::F32, TensorShape::new([2]).unwrap(), &pool).unwrap();
        tensor.view_mut().set_f32_value(1, 3.5).unwrap();
        let mut transposed = tensor.view_mut().reshape(TensorShape::new([1, 2]).unwrap()).unwrap();
        transposed = transposed.transpose(0, 1).unwrap();
        transposed.set_f32_value(0, 1.25).unwrap();
        assert!(tensor.set_element_bytes(0, &[0, 1]).is_err());
        let frozen = tensor.freeze();
        assert_eq!(frozen.f32_value(0).unwrap().to_bits(), 1.25_f32.to_bits());
        assert_eq!(frozen.f32_value(1).unwrap().to_bits(), 3.5_f32.to_bits());
        assert!(frozen.view().accounted_by(&pool));
    }
}
