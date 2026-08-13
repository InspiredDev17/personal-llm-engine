//! First-party deterministic AIPQ storage and scalar correctness kernels.
//!
//! AIPQ values are data only. A package cannot select executable code, block
//! geometry, scale representation, rounding, or a backend kernel. Version one
//! uses 32-value symmetric blocks and an explicit little-endian F32 scale.

use personalai_engine_memory::MemoryPool;
use personalai_engine_model_format::{
    AIPQ_BLOCK_VALUES, AIPQ_REQUIRED_ALIGNMENT, AIPQ4_BLOCK_BYTES, AIPQ8_BLOCK_BYTES, Quantization,
};
use personalai_engine_tensor::{DType, MutableTensor, Tensor, TensorRead, TensorShape};
use personalai_engine_types::EngineError;

/// Stable magic for one AIPQ8 v1 tensor payload.
pub const AIPQ8_MAGIC: [u8; 8] = *b"AIPQ8V1\0";
/// Stable magic for one AIPQ4 v1 tensor payload.
pub const AIPQ4_MAGIC: [u8; 8] = *b"AIPQ4V1\0";
/// Format version encoded in every tensor payload.
pub const AIPQ_VERSION: u16 = 1;
/// Scale-type identifier for little-endian IEEE binary32.
pub const AIPQ_SCALE_F32_LE: u8 = 1;
/// Q4 layout identifier: even values in the low nibble, odd values in the high nibble.
pub const AIPQ4_LOW_NIBBLE_FIRST: u8 = 1;

const HEADER_BYTES: usize = 32;
const BLOCK_VALUES: usize = 32;
const Q8_MAX: i16 = 127;
const Q4_MAX: i16 = 7;

/// Explicit offline quantization profile.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QuantizationProfile {
    /// Keep every tensor as finite F32.
    ReferenceF32,
    /// Quantize approved rank-two weights with AIPQ8 v1.
    BalancedQ8,
    /// Quantize approved rank-two weights with AIPQ4 v1.
    CompactQ4,
}

impl QuantizationProfile {
    /// Parses one exact operator-selected profile name.
    pub fn parse(value: &str) -> Result<Self, EngineError> {
        match value {
            "reference-f32" => Ok(Self::ReferenceF32),
            "balanced-q8" => Ok(Self::BalancedQ8),
            "compact-q4" => Ok(Self::CompactQ4),
            _ => Err(EngineError::invalid_input("quantization profile is unsupported")),
        }
    }

    /// Returns the stable profile name recorded in signed provenance.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ReferenceF32 => "reference-f32",
            Self::BalancedQ8 => "balanced-q8",
            Self::CompactQ4 => "compact-q4",
        }
    }

    /// Returns the package-wide quantization declaration.
    pub const fn format(self) -> Quantization {
        match self {
            Self::ReferenceF32 => Quantization::None,
            Self::BalancedQ8 => Quantization::AipQ8V1,
            Self::CompactQ4 => Quantization::AipQ4V1,
        }
    }

    /// Selects storage for one architecture-owned tensor.
    ///
    /// Version one quantizes only rank-two matrix weights. Norm and other
    /// rank-one parameters remain exact F32.
    pub const fn tensor_format(self, rank: usize) -> Quantization {
        if rank == 2 { self.format() } else { Quantization::None }
    }
}

/// Deterministic error evidence for one quantized tensor.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct QuantizationErrorMetrics {
    value_count: u64,
    relative_value_count: u64,
    maximum_absolute_error: f64,
    mean_absolute_error: f64,
    root_mean_square_error: f64,
    maximum_relative_error: f64,
}

impl QuantizationErrorMetrics {
    /// Returns the number of compared values.
    pub const fn value_count(self) -> u64 {
        self.value_count
    }

    /// Returns the count for which relative error was defined (reference was nonzero).
    pub const fn relative_value_count(self) -> u64 {
        self.relative_value_count
    }

    /// Returns the maximum absolute error.
    pub const fn maximum_absolute_error(self) -> f64 {
        self.maximum_absolute_error
    }

    /// Returns the mean absolute error.
    pub const fn mean_absolute_error(self) -> f64 {
        self.mean_absolute_error
    }

    /// Returns the root-mean-square error.
    pub const fn root_mean_square_error(self) -> f64 {
        self.root_mean_square_error
    }

    /// Returns the maximum relative error over nonzero reference values.
    pub const fn maximum_relative_error(self) -> f64 {
        self.maximum_relative_error
    }
}

/// One canonical encoded quantized tensor and its complete error evidence.
#[derive(Clone, Debug, PartialEq)]
pub struct QuantizedPayload {
    format: Quantization,
    bytes: Vec<u8>,
    metrics: QuantizationErrorMetrics,
}

impl QuantizedPayload {
    /// Returns the closed storage format.
    pub const fn format(&self) -> Quantization {
        self.format
    }

    /// Returns exact canonical signed-package bytes.
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Returns measured conversion error.
    pub const fn metrics(&self) -> QuantizationErrorMetrics {
        self.metrics
    }

    /// Consumes the value and returns exact canonical bytes.
    pub fn into_bytes(self) -> Vec<u8> {
        self.bytes
    }
}

/// A validated borrowed quantized tensor payload.
#[derive(Clone, Copy, Debug)]
pub struct QuantizedTensor<'a> {
    format: Quantization,
    shape: &'a TensorShape,
    bytes: &'a [u8],
    block_count: usize,
}

impl<'a> QuantizedTensor<'a> {
    /// Parses and validates one complete payload with no trailing data.
    pub fn parse(
        format: Quantization,
        shape: &'a TensorShape,
        bytes: &'a [u8],
    ) -> Result<Self, EngineError> {
        if !matches!(format, Quantization::AipQ8V1 | Quantization::AipQ4V1) {
            return Err(corrupt("AIPQ parser requires a quantized format"));
        }
        let expected = usize::try_from(format.payload_len(shape.element_count())?)
            .map_err(|_| resource("AIPQ payload length is not addressable"))?;
        if bytes.len() != expected {
            return Err(corrupt("AIPQ payload is truncated or has trailing data"));
        }
        let expected_magic = match format {
            Quantization::AipQ8V1 => AIPQ8_MAGIC,
            Quantization::AipQ4V1 => AIPQ4_MAGIC,
            Quantization::None => return Err(corrupt("AIPQ format is unquantized")),
        };
        if bytes[..8] != expected_magic
            || read_u16(bytes, 8)? != AIPQ_VERSION
            || u64::from(read_u16(bytes, 10)?) != AIPQ_BLOCK_VALUES
            || bytes[12] != AIPQ_SCALE_F32_LE
            || bytes[13] != if format == Quantization::AipQ4V1 { AIPQ4_LOW_NIBBLE_FIRST } else { 0 }
            || bytes[14..16].iter().any(|byte| *byte != 0)
            || read_u64(bytes, 16)? != shape.element_count()
        {
            return Err(corrupt("AIPQ header declaration is invalid"));
        }
        let block_count_u64 = block_count(shape.element_count())?;
        if read_u64(bytes, 24)? != block_count_u64 {
            return Err(corrupt("AIPQ block count is inconsistent"));
        }
        let block_count = usize::try_from(block_count_u64)
            .map_err(|_| resource("AIPQ block count is not addressable"))?;
        let value = Self { format, shape, bytes, block_count };
        value.validate_blocks()?;
        Ok(value)
    }

    /// Returns the closed storage format.
    pub const fn format(self) -> Quantization {
        self.format
    }

    /// Returns the logical F32 shape.
    pub const fn shape(self) -> &'a TensorShape {
        self.shape
    }

    /// Returns the required package alignment.
    pub const fn required_alignment(self) -> u32 {
        AIPQ_REQUIRED_ALIGNMENT
    }

    /// Reads one logical value through the validated block encoding.
    pub fn value(self, index: u64) -> Result<f32, EngineError> {
        if index >= self.shape.element_count() {
            return Err(EngineError::invalid_input("AIPQ element index is outside its shape"));
        }
        let block = usize::try_from(index / AIPQ_BLOCK_VALUES)
            .map_err(|_| resource("AIPQ block index is not addressable"))?;
        let within = usize::try_from(index % AIPQ_BLOCK_VALUES)
            .map_err(|_| resource("AIPQ element offset is not addressable"))?;
        let (_, block_bytes) = self.block_range(block)?;
        let scale = read_scale(block_bytes)?;
        let quantized = match self.format {
            Quantization::AipQ8V1 => i16::from(i8::from_le_bytes([block_bytes[4 + within]])),
            Quantization::AipQ4V1 => {
                let packed = block_bytes[4 + within / 2];
                let nibble = if within % 2 == 0 { packed & 0x0f } else { packed >> 4 };
                i16::from(decode_q4(nibble)?)
            }
            Quantization::None => return Err(corrupt("AIPQ format became unquantized")),
        };
        let product = f32::from(quantized) * scale;
        let result = if product.is_infinite() && quantized != 0 {
            if quantized.is_negative() { -f32::MAX } else { f32::MAX }
        } else {
            product
        };
        if !result.is_finite() {
            return Err(corrupt("AIPQ dequantization produced a non-finite value"));
        }
        Ok(result)
    }

    /// Materializes canonical F32 execution storage under the supplied aggregate pool.
    pub fn dequantize(self, pool: &MemoryPool) -> Result<Tensor, EngineError> {
        let mut output = MutableTensor::zeros(DType::F32, self.shape.clone(), pool)?;
        for index in 0..self.shape.element_count() {
            output.set_f32_value(usize_from_u64(index)?, self.value(index)?)?;
        }
        Ok(output)
    }

    /// Scalar deterministic matrix-vector multiplication without materializing weights.
    pub fn matrix_vector(
        self,
        vector: &impl TensorRead,
        pool: &MemoryPool,
    ) -> Result<Tensor, EngineError> {
        if self.shape.rank() != 2
            || vector.dtype() != DType::F32
            || vector.shape().rank() != 1
            || vector.shape().dimensions()[0] != self.shape.dimensions()[1]
            || !vector.accounted_by(pool)
        {
            return Err(EngineError::invalid_input("quantized matrix-vector inputs are invalid"));
        }
        let rows = self.shape.dimensions()[0];
        let shared = self.shape.dimensions()[1];
        checked_operations(&[rows, shared])?;
        let mut output = MutableTensor::zeros(DType::F32, TensorShape::new([rows])?, pool)?;
        for row in 0..rows {
            let mut sum = 0.0_f32;
            for column in 0..shared {
                let weight = self.value(checked_index(row, shared, column)?)?;
                let input = vector.f32_value(usize_from_u64(column)?)?;
                if !input.is_finite() {
                    return Err(EngineError::invalid_input(
                        "quantized matrix-vector input is non-finite",
                    ));
                }
                sum += weight * input;
                if !sum.is_finite() {
                    return Err(EngineError::backend_failure(
                        "quantized matrix-vector output is non-finite",
                    ));
                }
            }
            output.set_f32_value(usize_from_u64(row)?, sum)?;
        }
        Ok(output)
    }

    /// Scalar deterministic matrix-matrix multiplication without materializing weights.
    pub fn matrix_matrix(
        self,
        right: &impl TensorRead,
        pool: &MemoryPool,
    ) -> Result<Tensor, EngineError> {
        if self.shape.rank() != 2
            || right.dtype() != DType::F32
            || right.shape().rank() != 2
            || right.shape().dimensions()[0] != self.shape.dimensions()[1]
            || !right.accounted_by(pool)
        {
            return Err(EngineError::invalid_input("quantized matrix inputs are invalid"));
        }
        let rows = self.shape.dimensions()[0];
        let shared = self.shape.dimensions()[1];
        let columns = right.shape().dimensions()[1];
        checked_operations(&[rows, shared, columns])?;
        let mut output =
            MutableTensor::zeros(DType::F32, TensorShape::new([rows, columns])?, pool)?;
        for row in 0..rows {
            for column in 0..columns {
                let mut sum = 0.0_f32;
                for inner in 0..shared {
                    let weight = self.value(checked_index(row, shared, inner)?)?;
                    let input =
                        right.f32_value(usize_from_u64(checked_index(inner, columns, column)?)?)?;
                    if !input.is_finite() {
                        return Err(EngineError::invalid_input(
                            "quantized matrix input is non-finite",
                        ));
                    }
                    sum += weight * input;
                    if !sum.is_finite() {
                        return Err(EngineError::backend_failure(
                            "quantized matrix output is non-finite",
                        ));
                    }
                }
                output.set_f32_value(usize_from_u64(checked_index(row, columns, column)?)?, sum)?;
            }
        }
        Ok(output)
    }

    fn validate_blocks(self) -> Result<(), EngineError> {
        let last_values = usize::try_from(self.shape.element_count() % AIPQ_BLOCK_VALUES)
            .map_err(|_| resource("AIPQ tail length is not addressable"))?;
        for block in 0..self.block_count {
            let (_, bytes) = self.block_range(block)?;
            let scale = read_scale(bytes)?;
            let used = if block + 1 == self.block_count && last_values != 0 {
                last_values
            } else {
                BLOCK_VALUES
            };
            let mut maximum = 0_i16;
            for index in 0..BLOCK_VALUES {
                let value = match self.format {
                    Quantization::AipQ8V1 => {
                        let value = i16::from(i8::from_le_bytes([bytes[4 + index]]));
                        if value == -128 {
                            return Err(corrupt("AIPQ8 contains the reserved -128 encoding"));
                        }
                        value
                    }
                    Quantization::AipQ4V1 => {
                        let packed = bytes[4 + index / 2];
                        let nibble = if index % 2 == 0 { packed & 0x0f } else { packed >> 4 };
                        i16::from(decode_q4(nibble)?)
                    }
                    Quantization::None => {
                        return Err(corrupt("AIPQ block format became unquantized"));
                    }
                };
                if index >= used && value != 0 {
                    return Err(corrupt("AIPQ tail padding is nonzero"));
                }
                if index < used {
                    maximum = maximum.max(value.abs());
                }
            }
            if scale == 0.0 {
                if scale.to_bits() != 0 || maximum != 0 {
                    return Err(corrupt("AIPQ zero block is not canonical"));
                }
            } else {
                let required = if self.format == Quantization::AipQ8V1 { Q8_MAX } else { Q4_MAX };
                if maximum != required && !(scale.is_subnormal() && maximum == 1) {
                    return Err(corrupt("AIPQ nonzero block has a noncanonical scale"));
                }
            }
        }
        Ok(())
    }

    fn block_range(self, block: usize) -> Result<(usize, &'a [u8]), EngineError> {
        if block >= self.block_count {
            return Err(EngineError::invalid_input("AIPQ block index is outside its payload"));
        }
        let block_bytes = match self.format {
            Quantization::AipQ8V1 => usize::try_from(AIPQ8_BLOCK_BYTES),
            Quantization::AipQ4V1 => usize::try_from(AIPQ4_BLOCK_BYTES),
            Quantization::None => return Err(corrupt("AIPQ block format is unquantized")),
        }
        .map_err(|_| resource("AIPQ block length is not addressable"))?;
        let start = block
            .checked_mul(block_bytes)
            .and_then(|offset| offset.checked_add(HEADER_BYTES))
            .ok_or_else(|| resource("AIPQ block offset overflows"))?;
        let end =
            start.checked_add(block_bytes).ok_or_else(|| resource("AIPQ block range overflows"))?;
        let bytes = self.bytes.get(start..end).ok_or_else(|| corrupt("AIPQ block is truncated"))?;
        Ok((start, bytes))
    }
}

/// Quantizes finite values into one canonical AIPQ8 or AIPQ4 payload.
pub fn quantize(
    format: Quantization,
    shape: &TensorShape,
    values: &[f32],
) -> Result<QuantizedPayload, EngineError> {
    if !matches!(format, Quantization::AipQ8V1 | Quantization::AipQ4V1) {
        return Err(EngineError::invalid_input("quantization format is not AIPQ"));
    }
    if usize::try_from(shape.element_count()).ok() != Some(values.len()) {
        return Err(EngineError::invalid_input(
            "quantization value count does not match tensor shape",
        ));
    }
    if values.iter().any(|value| !value.is_finite()) {
        return Err(EngineError::invalid_input("AIPQ conversion rejects NaN and infinity"));
    }
    let length = usize::try_from(format.payload_len(shape.element_count())?)
        .map_err(|_| resource("AIPQ output length is not addressable"))?;
    let mut bytes = Vec::new();
    bytes.try_reserve_exact(length).map_err(|_| resource("AIPQ output allocation failed"))?;
    bytes.resize(length, 0);
    bytes[..8].copy_from_slice(&match format {
        Quantization::AipQ8V1 => AIPQ8_MAGIC,
        Quantization::AipQ4V1 => AIPQ4_MAGIC,
        Quantization::None => return Err(EngineError::invalid_input("AIPQ format is absent")),
    });
    write_u16(&mut bytes, 8, AIPQ_VERSION)?;
    write_u16(
        &mut bytes,
        10,
        u16::try_from(AIPQ_BLOCK_VALUES)
            .map_err(|_| resource("AIPQ block size exceeds its header field"))?,
    )?;
    bytes[12] = AIPQ_SCALE_F32_LE;
    bytes[13] = if format == Quantization::AipQ4V1 { AIPQ4_LOW_NIBBLE_FIRST } else { 0 };
    write_u64(&mut bytes, 16, shape.element_count())?;
    write_u64(&mut bytes, 24, block_count(shape.element_count())?)?;

    let maximum_code = if format == Quantization::AipQ8V1 { Q8_MAX } else { Q4_MAX };
    let encoded_block_bytes = if format == Quantization::AipQ8V1 {
        usize::try_from(AIPQ8_BLOCK_BYTES)
    } else {
        usize::try_from(AIPQ4_BLOCK_BYTES)
    }
    .map_err(|_| resource("AIPQ block length is not addressable"))?;
    for (block_index, chunk) in values.chunks(BLOCK_VALUES).enumerate() {
        let start = block_index
            .checked_mul(encoded_block_bytes)
            .and_then(|offset| offset.checked_add(HEADER_BYTES))
            .ok_or_else(|| resource("AIPQ output block offset overflows"))?;
        let maximum = chunk.iter().fold(0.0_f32, |current, value| current.max(value.abs()));
        let scale = if maximum == 0.0 {
            0.0
        } else {
            let candidate = maximum / f32::from(maximum_code);
            if candidate == 0.0 { maximum } else { candidate }
        };
        bytes[start..start + 4].copy_from_slice(&scale.to_le_bytes());
        if scale == 0.0 {
            continue;
        }
        for (index, value) in chunk.iter().copied().enumerate() {
            let rounded = (value / scale).round();
            let bounded = rounded.clamp(-f32::from(maximum_code), f32::from(maximum_code));
            #[allow(clippy::cast_possible_truncation)]
            let quantized = bounded as i8;
            if format == Quantization::AipQ8V1 {
                bytes[start + 4 + index] = quantized.to_le_bytes()[0];
            } else {
                let nibble = encode_q4(quantized)?;
                let packed = &mut bytes[start + 4 + index / 2];
                if index % 2 == 0 {
                    *packed = (*packed & 0xf0) | nibble;
                } else {
                    *packed = (*packed & 0x0f) | (nibble << 4);
                }
            }
        }
    }
    let parsed = QuantizedTensor::parse(format, shape, &bytes)?;
    let metrics = error_metrics(values, parsed)?;
    Ok(QuantizedPayload { format, bytes, metrics })
}

/// Computes complete deterministic error metrics against a validated payload.
pub fn error_metrics(
    reference: &[f32],
    quantized: QuantizedTensor<'_>,
) -> Result<QuantizationErrorMetrics, EngineError> {
    if usize::try_from(quantized.shape.element_count()).ok() != Some(reference.len())
        || reference.is_empty()
        || reference.iter().any(|value| !value.is_finite())
    {
        return Err(EngineError::invalid_input("quantization error reference is invalid"));
    }
    let mut maximum_absolute_error = 0.0_f64;
    let mut sum_absolute_error = 0.0_f64;
    let mut sum_squared_error = 0.0_f64;
    let mut maximum_relative_error = 0.0_f64;
    let mut relative_value_count = 0_u64;
    for (index, expected) in reference.iter().copied().enumerate() {
        let observed = quantized.value(u64_from_usize(index)?)?;
        let error = (f64::from(expected) - f64::from(observed)).abs();
        maximum_absolute_error = maximum_absolute_error.max(error);
        sum_absolute_error += error;
        sum_squared_error += error * error;
        if expected != 0.0 {
            relative_value_count = relative_value_count
                .checked_add(1)
                .ok_or_else(|| resource("relative error count overflows"))?;
            maximum_relative_error = maximum_relative_error.max(error / f64::from(expected).abs());
        }
    }
    let value_count = u64_from_usize(reference.len())?;
    #[allow(clippy::cast_precision_loss)]
    let divisor = value_count as f64;
    Ok(QuantizationErrorMetrics {
        value_count,
        relative_value_count,
        maximum_absolute_error,
        mean_absolute_error: sum_absolute_error / divisor,
        root_mean_square_error: (sum_squared_error / divisor).sqrt(),
        maximum_relative_error,
    })
}

fn read_scale(bytes: &[u8]) -> Result<f32, EngineError> {
    let scale = f32::from_le_bytes(
        bytes
            .get(..4)
            .ok_or_else(|| corrupt("AIPQ scale is truncated"))?
            .try_into()
            .map_err(|_| corrupt("AIPQ scale length is invalid"))?,
    );
    if !scale.is_finite() || scale < 0.0 {
        return Err(corrupt("AIPQ scale is non-finite or negative"));
    }
    Ok(scale)
}

fn block_count(elements: u64) -> Result<u64, EngineError> {
    elements
        .checked_add(AIPQ_BLOCK_VALUES - 1)
        .and_then(|value| value.checked_div(AIPQ_BLOCK_VALUES))
        .filter(|value| *value != 0)
        .ok_or_else(|| resource("AIPQ block count overflows"))
}

fn encode_q4(value: i8) -> Result<u8, EngineError> {
    match value {
        0..=7 => u8::try_from(value).map_err(|_| corrupt("AIPQ4 positive value is not encodable")),
        -7..=-1 => Ok(u8::try_from(i16::from(value) + 15)
            .map_err(|_| corrupt("AIPQ4 signed value is not encodable"))?),
        _ => Err(corrupt("AIPQ4 signed value is outside [-7, 7]")),
    }
}

fn decode_q4(nibble: u8) -> Result<i8, EngineError> {
    match nibble {
        0..=7 => i8::try_from(nibble).map_err(|_| corrupt("AIPQ4 nibble mapping failed")),
        8..=14 => Ok(i8::try_from(i16::from(nibble) - 15)
            .map_err(|_| corrupt("AIPQ4 nibble mapping failed"))?),
        _ => Err(corrupt("AIPQ4 contains the reserved nibble encoding")),
    }
}

fn checked_operations(dimensions: &[u64]) -> Result<(), EngineError> {
    let count = dimensions.iter().copied().try_fold(1_u64, |count, dimension| {
        count.checked_mul(dimension).ok_or_else(|| resource("quantized operation count overflows"))
    })?;
    if count > 100_000_000 {
        return Err(resource("quantized operation exceeds CPU reference work limit"));
    }
    Ok(())
}

fn checked_index(row: u64, width: u64, column: u64) -> Result<u64, EngineError> {
    row.checked_mul(width)
        .and_then(|value| value.checked_add(column))
        .ok_or_else(|| resource("quantized tensor index overflows"))
}

fn read_u16(bytes: &[u8], offset: usize) -> Result<u16, EngineError> {
    let end = offset.checked_add(2).ok_or_else(|| resource("AIPQ field offset overflows"))?;
    Ok(u16::from_le_bytes(
        bytes
            .get(offset..end)
            .ok_or_else(|| corrupt("AIPQ u16 field is truncated"))?
            .try_into()
            .map_err(|_| corrupt("AIPQ u16 field length is invalid"))?,
    ))
}

fn read_u64(bytes: &[u8], offset: usize) -> Result<u64, EngineError> {
    let end = offset.checked_add(8).ok_or_else(|| resource("AIPQ field offset overflows"))?;
    Ok(u64::from_le_bytes(
        bytes
            .get(offset..end)
            .ok_or_else(|| corrupt("AIPQ u64 field is truncated"))?
            .try_into()
            .map_err(|_| corrupt("AIPQ u64 field length is invalid"))?,
    ))
}

fn write_u16(bytes: &mut [u8], offset: usize, value: u16) -> Result<(), EngineError> {
    write_exact(bytes, offset, &value.to_le_bytes())
}

fn write_u64(bytes: &mut [u8], offset: usize, value: u64) -> Result<(), EngineError> {
    write_exact(bytes, offset, &value.to_le_bytes())
}

fn write_exact(bytes: &mut [u8], offset: usize, value: &[u8]) -> Result<(), EngineError> {
    let end =
        offset.checked_add(value.len()).ok_or_else(|| resource("AIPQ write offset overflows"))?;
    bytes
        .get_mut(offset..end)
        .ok_or_else(|| corrupt("AIPQ write range is unavailable"))?
        .copy_from_slice(value);
    Ok(())
}

fn usize_from_u64(value: u64) -> Result<usize, EngineError> {
    usize::try_from(value).map_err(|_| resource("AIPQ value is not addressable"))
}

fn u64_from_usize(value: usize) -> Result<u64, EngineError> {
    u64::try_from(value).map_err(|_| resource("AIPQ size is not representable"))
}

fn corrupt(detail: impl AsRef<str>) -> EngineError {
    EngineError::corrupt_model_package(detail)
}

fn resource(detail: impl AsRef<str>) -> EngineError {
    EngineError::resource_exhaustion(detail)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shape(length: u64) -> TensorShape {
        TensorShape::new([length]).expect("test shape must validate")
    }

    fn approx(left: f32, right: f32, tolerance: f32) {
        assert!((left - right).abs() <= tolerance, "{left} != {right}");
    }

    #[test]
    fn q8_known_block_round_trips_and_is_deterministic() {
        let values = (-16_i16..16).map(|value| f32::from(value) / 4.0).collect::<Vec<_>>();
        let first = quantize(Quantization::AipQ8V1, &shape(32), &values)
            .expect("Q8 conversion must succeed");
        let second = quantize(Quantization::AipQ8V1, &shape(32), &values)
            .expect("repeat Q8 conversion must succeed");
        assert_eq!(first.bytes(), second.bytes());
        assert_eq!(&first.bytes()[..8], &AIPQ8_MAGIC);
        let tensor_shape = shape(32);
        let parsed = QuantizedTensor::parse(Quantization::AipQ8V1, &tensor_shape, first.bytes())
            .expect("Q8 payload must parse");
        for (index, expected) in values.iter().copied().enumerate() {
            approx(parsed.value(index as u64).expect("value must decode"), expected, 0.02);
        }
        assert!(first.metrics().maximum_absolute_error() > 0.0);
    }

    #[test]
    fn q8_zero_constant_tail_and_nonfinite_policies_are_exact() {
        let zeros = vec![0.0; 35];
        let encoded =
            quantize(Quantization::AipQ8V1, &shape(35), &zeros).expect("zero tail must quantize");
        let tensor_shape = shape(35);
        let parsed = QuantizedTensor::parse(Quantization::AipQ8V1, &tensor_shape, encoded.bytes())
            .expect("zero tail must validate");
        assert_eq!(parsed.value(34).expect("tail must decode").to_bits(), 0);
        let constants = vec![-3.5; 33];
        assert!(quantize(Quantization::AipQ8V1, &shape(33), &constants).is_ok());
        assert!(quantize(Quantization::AipQ8V1, &shape(1), &[f32::NAN]).is_err());
        assert!(quantize(Quantization::AipQ8V1, &shape(1), &[f32::INFINITY]).is_err());
        for value in [f32::from_bits(1), f32::MAX, -f32::MAX] {
            let encoded = quantize(Quantization::AipQ8V1, &shape(1), &[value])
                .expect("finite extreme must quantize deterministically");
            let tensor_shape = shape(1);
            let parsed =
                QuantizedTensor::parse(Quantization::AipQ8V1, &tensor_shape, encoded.bytes())
                    .expect("finite extreme payload must parse");
            assert!(parsed.value(0).expect("finite extreme must decode").is_finite());
        }
    }

    #[test]
    fn q8_rejects_corruption_truncation_scale_and_tail() {
        let values = (0_i16..33).map(f32::from).collect::<Vec<_>>();
        let encoded =
            quantize(Quantization::AipQ8V1, &shape(33), &values).expect("fixture must quantize");
        assert!(
            QuantizedTensor::parse(
                Quantization::AipQ8V1,
                &shape(33),
                &encoded.bytes()[..encoded.bytes().len() - 1],
            )
            .is_err()
        );
        for (offset, byte) in [(0, 0_u8), (10, 31), (36, 0x80), (73, 1)] {
            let mut bad = encoded.bytes().to_vec();
            bad[offset] = byte;
            assert!(
                QuantizedTensor::parse(Quantization::AipQ8V1, &shape(33), &bad).is_err(),
                "corruption at {offset} was accepted"
            );
        }
        let mut nan_scale = encoded.bytes().to_vec();
        nan_scale[32..36].copy_from_slice(&f32::NAN.to_le_bytes());
        assert!(QuantizedTensor::parse(Quantization::AipQ8V1, &shape(33), &nan_scale).is_err());
    }

    #[test]
    fn q4_known_nibbles_tail_and_reserved_encoding_are_strict() {
        let values = [-7.0, -1.0, 0.0, 1.0, 7.0];
        let encoded = quantize(Quantization::AipQ4V1, &shape(5), &values)
            .expect("Q4 conversion must succeed");
        let payload = encoded.bytes();
        assert_eq!(payload[36], 0xe8);
        assert_eq!(payload[37], 0x10);
        assert_eq!(payload[38] & 0x0f, 0x07);
        let tensor_shape = shape(5);
        let parsed = QuantizedTensor::parse(Quantization::AipQ4V1, &tensor_shape, payload)
            .expect("Q4 payload must parse");
        for (index, expected) in values.into_iter().enumerate() {
            approx(parsed.value(index as u64).expect("value must decode"), expected, 0.001);
        }
        let mut reserved = payload.to_vec();
        reserved[36] = (reserved[36] & 0xf0) | 0x0f;
        assert!(QuantizedTensor::parse(Quantization::AipQ4V1, &shape(5), &reserved).is_err());
    }

    #[test]
    fn scalar_quantized_kernels_match_dequantized_reference() {
        let matrix_shape = TensorShape::new([2, 3]).expect("matrix shape must validate");
        let values = [1.0, -2.0, 3.0, 4.0, 0.5, -1.0];
        let pool = MemoryPool::new(4096).expect("pool must validate");
        let vector = MutableTensor::from_f32_values(
            TensorShape::new([3]).expect("vector shape must validate"),
            &[2.0, 1.0, -1.0],
            &pool,
        )
        .expect("vector must allocate");
        let right = MutableTensor::from_f32_values(
            TensorShape::new([3, 2]).expect("right shape must validate"),
            &[2.0, 1.0, 1.0, 0.0, -1.0, 3.0],
            &pool,
        )
        .expect("right matrix must allocate");
        for format in [Quantization::AipQ8V1, Quantization::AipQ4V1] {
            let encoded = quantize(format, &matrix_shape, &values).expect("must quantize");
            let quantized =
                QuantizedTensor::parse(format, &matrix_shape, encoded.bytes()).expect("must parse");
            let output = quantized.matrix_vector(&vector, &pool).expect("matvec must execute");
            let expected_first = quantized.value(0).expect("weight") * 2.0
                + quantized.value(1).expect("weight")
                - quantized.value(2).expect("weight");
            let expected_second = quantized.value(3).expect("weight") * 2.0
                + quantized.value(4).expect("weight")
                - quantized.value(5).expect("weight");
            approx(output.f32_value(0).expect("value must exist"), expected_first, 0.000_001);
            approx(output.f32_value(1).expect("value must exist"), expected_second, 0.000_001);
            let output = quantized.matrix_matrix(&right, &pool).expect("matmul must execute");
            assert_eq!(output.shape().dimensions(), &[2, 2]);
            assert!(output.as_bytes().iter().any(|byte| *byte != 0));
        }
    }
}
