//! Scalar reference operation implementations.

use crate::{CpuReferenceBackend, MAX_CPU_REFERENCE_OPERATIONS};
use personalai_engine_memory::MemoryPool;
use personalai_engine_tensor::{
    DType, MutableTensor, Tensor, TensorRead, TensorShape, f32_to_bf16_bits, f32_to_f16_bits,
};
use personalai_engine_types::EngineError;

impl CpuReferenceBackend {
    /// Copies a logical tensor into canonical owned storage.
    pub fn copy(&self, input: &impl TensorRead, pool: &MemoryPool) -> Result<Tensor, EngineError> {
        require_pool(input, pool)?;
        checked_elements(input.shape().element_count())?;
        let count = usize_from_u64(input.shape().element_count())?;
        let mut output = MutableTensor::zeros(input.dtype(), input.shape().clone(), pool)?;
        for index in 0..count {
            output.set_element_bytes(index, input.element_bytes(index)?)?;
        }
        Ok(output)
    }

    /// Converts among F32, F16, and BF16 storage without silent fallback.
    pub fn convert(
        &self,
        input: &impl TensorRead,
        destination_dtype: DType,
        pool: &MemoryPool,
    ) -> Result<Tensor, EngineError> {
        require_pool(input, pool)?;
        if !matches!(input.dtype(), DType::F32 | DType::F16 | DType::BFloat16)
            || !matches!(destination_dtype, DType::F32 | DType::F16 | DType::BFloat16)
        {
            return Err(EngineError::not_implemented(
                "CPU type conversion supports only f32, f16, and bf16",
            ));
        }
        let count = usize_from_u64(input.shape().element_count())?;
        checked_elements(input.shape().element_count())?;
        let mut output = MutableTensor::zeros(destination_dtype, input.shape().clone(), pool)?;
        for index in 0..count {
            let value = input.numeric_f32_value(index)?;
            match destination_dtype {
                DType::F32 => output.set_element_bytes(index, &value.to_le_bytes())?,
                DType::F16 => {
                    output.set_element_bytes(index, &f32_to_f16_bits(value).to_le_bytes())?;
                }
                DType::BFloat16 => {
                    output.set_element_bytes(index, &f32_to_bf16_bits(value).to_le_bytes())?;
                }
                _ => {
                    return Err(EngineError::internal_invariant(
                        "validated conversion type became unsupported",
                    ));
                }
            }
        }
        Ok(output)
    }

    /// Adds F32 tensors under standard trailing-dimension broadcasting.
    pub fn add(
        &self,
        left: &impl TensorRead,
        right: &impl TensorRead,
        pool: &MemoryPool,
    ) -> Result<Tensor, EngineError> {
        elementwise_binary(left, right, pool, |a, b| a + b)
    }

    /// Multiplies F32 tensors under standard trailing-dimension broadcasting.
    pub fn multiply(
        &self,
        left: &impl TensorRead,
        right: &impl TensorRead,
        pool: &MemoryPool,
    ) -> Result<Tensor, EngineError> {
        elementwise_binary(left, right, pool, |a, b| a * b)
    }

    /// Multiplies every F32 element by one finite scalar.
    pub fn scalar_multiply(
        &self,
        input: &impl TensorRead,
        scalar: f32,
        pool: &MemoryPool,
    ) -> Result<Tensor, EngineError> {
        require_f32(input)?;
        require_pool(input, pool)?;
        let scalar = finite_input(scalar)?;
        unary_f32(input, pool, |value| value * scalar)
    }

    /// Multiplies a rank-two F32 matrix by a rank-one F32 vector.
    pub fn matrix_vector(
        &self,
        matrix: &impl TensorRead,
        vector: &impl TensorRead,
        pool: &MemoryPool,
    ) -> Result<Tensor, EngineError> {
        require_f32(matrix)?;
        require_f32(vector)?;
        require_pool(matrix, pool)?;
        require_pool(vector, pool)?;
        if matrix.shape().rank() != 2 || vector.shape().rank() != 1 {
            return Err(EngineError::invalid_input("matrix-vector ranks are invalid"));
        }
        let rows = matrix.shape().dimensions()[0];
        let shared = matrix.shape().dimensions()[1];
        if vector.shape().dimensions()[0] != shared {
            return Err(EngineError::invalid_input("matrix-vector dimensions do not match"));
        }
        checked_product(&[rows, shared])?;
        let mut output = MutableTensor::zeros(DType::F32, TensorShape::new([rows])?, pool)?;
        let rows = usize_from_u64(rows)?;
        let shared = usize_from_u64(shared)?;
        for row in 0..rows {
            let mut accumulator = 0.0_f32;
            for column in 0..shared {
                let left = finite_input(matrix.f32_value(checked_index(row, shared, column)?)?)?;
                let right = finite_input(vector.f32_value(column)?)?;
                accumulator = finite_output(accumulator + left * right)?;
            }
            output.set_f32_value(row, accumulator)?;
        }
        Ok(output)
    }

    /// Multiplies two rank-two F32 matrices in deterministic row-major order.
    pub fn matrix_matrix(
        &self,
        left: &impl TensorRead,
        right: &impl TensorRead,
        pool: &MemoryPool,
    ) -> Result<Tensor, EngineError> {
        require_f32(left)?;
        require_f32(right)?;
        require_pool(left, pool)?;
        require_pool(right, pool)?;
        if left.shape().rank() != 2 || right.shape().rank() != 2 {
            return Err(EngineError::invalid_input("matrix multiplication ranks are invalid"));
        }
        let rows = left.shape().dimensions()[0];
        let shared = left.shape().dimensions()[1];
        let columns = right.shape().dimensions()[1];
        if right.shape().dimensions()[0] != shared {
            return Err(EngineError::invalid_input("matrix inner dimensions do not match"));
        }
        checked_product(&[rows, shared, columns])?;
        let mut output =
            MutableTensor::zeros(DType::F32, TensorShape::new([rows, columns])?, pool)?;
        let rows = usize_from_u64(rows)?;
        let shared = usize_from_u64(shared)?;
        let columns = usize_from_u64(columns)?;
        for row in 0..rows {
            for column in 0..columns {
                let mut accumulator = 0.0_f32;
                for inner in 0..shared {
                    let left_value =
                        finite_input(left.f32_value(checked_index(row, shared, inner)?)?)?;
                    let right_value =
                        finite_input(right.f32_value(checked_index(inner, columns, column)?)?)?;
                    accumulator = finite_output(accumulator + left_value * right_value)?;
                }
                output.set_f32_value(checked_index(row, columns, column)?, accumulator)?;
            }
        }
        Ok(output)
    }

    /// Returns one validated row from a rank-two embedding table.
    pub fn embedding_lookup(
        &self,
        embeddings: &impl TensorRead,
        token: u32,
        pool: &MemoryPool,
    ) -> Result<Tensor, EngineError> {
        require_f32(embeddings)?;
        require_pool(embeddings, pool)?;
        if embeddings.shape().rank() != 2 || u64::from(token) >= embeddings.shape().dimensions()[0]
        {
            return Err(EngineError::invalid_input("embedding lookup index or rank is invalid"));
        }
        let width = embeddings.shape().dimensions()[1];
        checked_elements(width)?;
        let mut output = MutableTensor::zeros(DType::F32, TensorShape::new([width])?, pool)?;
        let width = usize_from_u64(width)?;
        let row = usize::try_from(token)
            .map_err(|_| EngineError::invalid_input("embedding token is not addressable"))?;
        for column in 0..width {
            let value = finite_input(embeddings.f32_value(checked_index(row, width, column)?)?)?;
            output.set_f32_value(column, value)?;
        }
        Ok(output)
    }

    /// Applies stable RMS normalization across the final dimension.
    pub fn rms_norm(
        &self,
        input: &impl TensorRead,
        weight: &impl TensorRead,
        epsilon: f32,
        pool: &MemoryPool,
    ) -> Result<Tensor, EngineError> {
        require_f32(input)?;
        require_f32(weight)?;
        require_pool(input, pool)?;
        require_pool(weight, pool)?;
        if input.shape().rank() == 0
            || weight.shape().rank() != 1
            || weight.shape().dimensions()[0]
                != *input
                    .shape()
                    .dimensions()
                    .last()
                    .ok_or_else(|| EngineError::invalid_input("RMSNorm input rank is invalid"))?
            || !epsilon.is_finite()
            || epsilon <= 0.0
        {
            return Err(EngineError::invalid_input("RMSNorm metadata is invalid"));
        }
        checked_product(&[input.shape().element_count(), 2])?;
        let width = usize_from_u64(weight.shape().element_count())?;
        let rows = usize_from_u64(input.shape().element_count())? / width;
        let mut output = MutableTensor::zeros(DType::F32, input.shape().clone(), pool)?;
        for row in 0..rows {
            let start = row
                .checked_mul(width)
                .ok_or_else(|| EngineError::resource_exhaustion("RMSNorm row overflows"))?;
            let mut scale = 0.0_f64;
            let mut sum_scaled_squares = 1.0_f64;
            for column in 0..width {
                let absolute = f64::from(finite_input(input.f32_value(start + column)?)?.abs());
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
            let inverse = if scale == 0.0 {
                1.0_f64 / f64::from(epsilon).sqrt()
            } else {
                let normalized_epsilon = f64::from(epsilon) / (scale * scale);
                (1.0 / scale)
                    / (sum_scaled_squares / f64_from_usize(width)? + normalized_epsilon).sqrt()
            };
            for column in 0..width {
                let value = f64::from(finite_input(input.f32_value(start + column)?)?);
                let weight_value = f64::from(finite_input(weight.f32_value(column)?)?);
                output.set_f32_value(
                    start + column,
                    finite_output(f32_from_f64(value * inverse * weight_value)?)?,
                )?;
            }
        }
        Ok(output)
    }

    /// Applies stable softmax independently across the final dimension.
    ///
    /// Negative infinity is accepted as a causal mask. NaN, positive infinity,
    /// and an all-masked row are rejected.
    pub fn softmax(
        &self,
        input: &impl TensorRead,
        pool: &MemoryPool,
    ) -> Result<Tensor, EngineError> {
        require_f32(input)?;
        require_pool(input, pool)?;
        if input.shape().rank() == 0 {
            return Err(EngineError::invalid_input("softmax requires a non-scalar tensor"));
        }
        checked_product(&[input.shape().element_count(), 3])?;
        let width = usize_from_u64(*input.shape().dimensions().last().ok_or_else(|| {
            EngineError::internal_invariant("validated softmax rank became empty")
        })?)?;
        let rows = usize_from_u64(input.shape().element_count())? / width;
        let mut output = MutableTensor::zeros(DType::F32, input.shape().clone(), pool)?;
        for row in 0..rows {
            let start = row
                .checked_mul(width)
                .ok_or_else(|| EngineError::resource_exhaustion("softmax row overflows"))?;
            let mut maximum = f32::NEG_INFINITY;
            for column in 0..width {
                let value = input.f32_value(start + column)?;
                if value.is_nan() || value == f32::INFINITY {
                    return Err(EngineError::backend_failure("softmax input is invalid"));
                }
                maximum = maximum.max(value);
            }
            if maximum == f32::NEG_INFINITY {
                return Err(EngineError::backend_failure("softmax row is fully masked"));
            }
            let mut sum = 0.0_f64;
            for column in 0..width {
                let value = input.f32_value(start + column)?;
                let exponential = if value == f32::NEG_INFINITY {
                    0.0_f64
                } else {
                    f64::from(value - maximum).exp()
                };
                sum += exponential;
                output.set_f32_value(start + column, f32_from_f64(exponential)?)?;
            }
            if !sum.is_finite() || sum <= 0.0 {
                return Err(EngineError::backend_failure("softmax normalization is invalid"));
            }
            for column in 0..width {
                let probability = f64::from(output.f32_value(start + column)?) / sum;
                output.set_f32_value(start + column, f32_from_f64(probability)?)?;
            }
        }
        Ok(output)
    }

    /// Applies the SiLU activation element-by-element.
    pub fn silu(&self, input: &impl TensorRead, pool: &MemoryPool) -> Result<Tensor, EngineError> {
        require_f32(input)?;
        require_pool(input, pool)?;
        checked_elements(input.shape().element_count())?;
        let mut output = MutableTensor::zeros(DType::F32, input.shape().clone(), pool)?;
        for index in 0..usize_from_u64(input.shape().element_count())? {
            let value = f64::from(finite_input(input.f32_value(index)?)?);
            let activated = f32_from_f64(value / (1.0 + (-value).exp()))?;
            output.set_f32_value(index, finite_output(activated)?)?;
        }
        Ok(output)
    }

    /// Computes SiLU(gate) multiplied by `up` without fusion.
    pub fn swiglu(
        &self,
        gate: &impl TensorRead,
        up: &impl TensorRead,
        pool: &MemoryPool,
    ) -> Result<Tensor, EngineError> {
        require_f32(gate)?;
        require_f32(up)?;
        require_pool(gate, pool)?;
        require_pool(up, pool)?;
        if gate.shape() != up.shape() {
            return Err(EngineError::invalid_input("SwiGLU input shapes differ"));
        }
        checked_elements(gate.shape().element_count())?;
        let mut output = MutableTensor::zeros(DType::F32, gate.shape().clone(), pool)?;
        for index in 0..usize_from_u64(gate.shape().element_count())? {
            let gate_value = f64::from(finite_input(gate.f32_value(index)?)?);
            let up_value = f64::from(finite_input(up.f32_value(index)?)?);
            let silu = gate_value / (1.0 + (-gate_value).exp());
            output.set_f32_value(index, finite_output(f32_from_f64(silu * up_value)?)?)?;
        }
        Ok(output)
    }

    /// Applies rotary positional encoding to `[heads, head_dimension]`.
    pub fn rope(
        &self,
        input: &impl TensorRead,
        position: u32,
        theta: f32,
        pool: &MemoryPool,
    ) -> Result<Tensor, EngineError> {
        require_f32(input)?;
        require_pool(input, pool)?;
        if input.shape().rank() != 2
            || input.shape().dimensions()[1] % 2 != 0
            || !theta.is_finite()
            || theta <= 1.0
        {
            return Err(EngineError::invalid_input("RoPE metadata is invalid"));
        }
        checked_elements(input.shape().element_count())?;
        let heads = usize_from_u64(input.shape().dimensions()[0])?;
        let dimension = usize_from_u64(input.shape().dimensions()[1])?;
        let mut output = MutableTensor::zeros(DType::F32, input.shape().clone(), pool)?;
        for head in 0..heads {
            let base = head
                .checked_mul(dimension)
                .ok_or_else(|| EngineError::resource_exhaustion("RoPE head offset overflows"))?;
            for pair in (0..dimension).step_by(2) {
                let exponent = f64_from_usize(pair)? / f64_from_usize(dimension)?;
                let angle = f64::from(position) / f64::from(theta).powf(exponent);
                let (sine, cosine) = angle.sin_cos();
                let first = f64::from(finite_input(input.f32_value(base + pair)?)?);
                let second = f64::from(finite_input(input.f32_value(base + pair + 1)?)?);
                output.set_f32_value(
                    base + pair,
                    finite_output(f32_from_f64(first * cosine - second * sine)?)?,
                )?;
                output.set_f32_value(
                    base + pair + 1,
                    finite_output(f32_from_f64(first * sine + second * cosine)?)?,
                )?;
            }
        }
        Ok(output)
    }

    /// Copies a flat hidden vector into explicit `[heads, head_dimension]` shape.
    pub fn head_reshape(
        &self,
        input: &impl TensorRead,
        head_count: u32,
        pool: &MemoryPool,
    ) -> Result<Tensor, EngineError> {
        require_f32(input)?;
        require_pool(input, pool)?;
        if head_count == 0 {
            return Err(EngineError::invalid_input("attention head reshape is invalid"));
        }
        let output_shape = match input.shape().rank() {
            1 if input.shape().element_count() % u64::from(head_count) == 0 => TensorShape::new([
                u64::from(head_count),
                input.shape().element_count() / u64::from(head_count),
            ])?,
            2 if input.shape().dimensions()[0] == u64::from(head_count) => {
                TensorShape::new([input.shape().element_count()])?
            }
            _ => {
                return Err(EngineError::invalid_input("attention head reshape is invalid"));
            }
        };
        let mut output = MutableTensor::zeros(DType::F32, output_shape, pool)?;
        for index in 0..usize_from_u64(input.shape().element_count())? {
            output.set_f32_value(index, finite_input(input.f32_value(index)?)?)?;
        }
        Ok(output)
    }

    /// Computes grouped-query causal attention from explicit Q/K/V heads.
    pub fn grouped_query_attention(
        &self,
        query: &impl TensorRead,
        key: &impl TensorRead,
        value: &impl TensorRead,
        causal_position: u32,
        pool: &MemoryPool,
    ) -> Result<Tensor, EngineError> {
        for tensor in [query as &dyn TensorRead, key as &dyn TensorRead, value as &dyn TensorRead] {
            require_f32(tensor)?;
            require_pool(tensor, pool)?;
        }
        if query.shape().rank() != 2
            || key.shape().rank() != 3
            || value.shape() != key.shape()
            || query.shape().dimensions()[1] != key.shape().dimensions()[2]
            || query.shape().dimensions()[0] % key.shape().dimensions()[1] != 0
            || u64::from(causal_position) >= key.shape().dimensions()[0]
        {
            return Err(EngineError::invalid_input("grouped-query attention shapes are invalid"));
        }
        let query_heads = usize_from_u64(query.shape().dimensions()[0])?;
        let head_dimension = usize_from_u64(query.shape().dimensions()[1])?;
        let kv_heads = usize_from_u64(key.shape().dimensions()[1])?;
        let sequence = usize::try_from(causal_position)
            .ok()
            .and_then(|position| position.checked_add(1))
            .ok_or_else(|| EngineError::resource_exhaustion("attention length overflows"))?;
        checked_product(&[
            u64_from_usize(query_heads)?,
            u64_from_usize(sequence)?,
            u64_from_usize(head_dimension)?,
            2,
        ])?;
        let group_size = query_heads / kv_heads;
        let scale = f64_from_usize(head_dimension)?.sqrt();
        let mut output = MutableTensor::zeros(DType::F32, query.shape().clone(), pool)?;
        for query_head in 0..query_heads {
            let kv_head = query_head / group_size;
            let logical_score_bytes = u64_from_usize(sequence)?
                .checked_mul(8)
                .ok_or_else(|| EngineError::resource_exhaustion("attention scores overflow"))?;
            let mut score_reservation = pool.reserve(logical_score_bytes)?;
            let mut scores = Vec::new();
            scores.try_reserve_exact(sequence).map_err(|_| {
                EngineError::resource_exhaustion("attention score allocation failed")
            })?;
            let allocated_score_bytes = u64_from_usize(scores.capacity())?
                .checked_mul(8)
                .ok_or_else(|| EngineError::resource_exhaustion("attention capacity overflows"))?;
            pool.grow_reservation(
                &mut score_reservation,
                allocated_score_bytes.checked_sub(logical_score_bytes).ok_or_else(|| {
                    EngineError::internal_invariant("attention capacity is below requested length")
                })?,
            )?;
            for position in 0..sequence {
                let mut dot = 0.0_f64;
                for dimension in 0..head_dimension {
                    let query_index = checked_index(query_head, head_dimension, dimension)?;
                    let key_index =
                        checked_index3(position, kv_head, dimension, kv_heads, head_dimension)?;
                    dot += f64::from(finite_input(query.f32_value(query_index)?)?)
                        * f64::from(finite_input(key.f32_value(key_index)?)?);
                }
                scores.push(dot / scale);
            }
            let maximum = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let mut sum = 0.0_f64;
            for score in &mut scores {
                *score = (*score - maximum).exp();
                sum += *score;
            }
            if !sum.is_finite() || sum <= 0.0 {
                return Err(EngineError::backend_failure("attention softmax is invalid"));
            }
            for dimension in 0..head_dimension {
                let mut result = 0.0_f64;
                for (position, score) in scores.iter().copied().enumerate() {
                    let value_index =
                        checked_index3(position, kv_head, dimension, kv_heads, head_dimension)?;
                    result +=
                        (score / sum) * f64::from(finite_input(value.f32_value(value_index)?)?);
                }
                output.set_f32_value(
                    checked_index(query_head, head_dimension, dimension)?,
                    finite_output(f32_from_f64(result)?)?,
                )?;
            }
        }
        Ok(output)
    }

    /// Applies a causal negative-infinity mask to rank-two score rows.
    pub fn causal_mask(
        &self,
        scores: &impl TensorRead,
        first_query_position: u32,
        pool: &MemoryPool,
    ) -> Result<Tensor, EngineError> {
        require_f32(scores)?;
        require_pool(scores, pool)?;
        if scores.shape().rank() != 2 {
            return Err(EngineError::invalid_input("causal mask requires rank-two scores"));
        }
        checked_elements(scores.shape().element_count())?;
        let rows = usize_from_u64(scores.shape().dimensions()[0])?;
        let columns = usize_from_u64(scores.shape().dimensions()[1])?;
        let mut output = MutableTensor::zeros(DType::F32, scores.shape().clone(), pool)?;
        for row in 0..rows {
            let allowed = usize::try_from(first_query_position)
                .ok()
                .and_then(|position| position.checked_add(row))
                .ok_or_else(|| EngineError::resource_exhaustion("causal position overflows"))?;
            for column in 0..columns {
                let index = checked_index(row, columns, column)?;
                let value = finite_input(scores.f32_value(index)?)?;
                output.set_f32_value(
                    index,
                    if column <= allowed { value } else { f32::NEG_INFINITY },
                )?;
            }
        }
        Ok(output)
    }

    /// Adds an exact-shape residual without broadcasting.
    pub fn residual_add(
        &self,
        residual: &impl TensorRead,
        update: &impl TensorRead,
        pool: &MemoryPool,
    ) -> Result<Tensor, EngineError> {
        if residual.shape() != update.shape() {
            return Err(EngineError::invalid_input("residual shapes differ"));
        }
        self.add(residual, update, pool)
    }

    /// Projects one hidden vector into vocabulary logits.
    pub fn logit_projection(
        &self,
        weight: &impl TensorRead,
        hidden: &impl TensorRead,
        pool: &MemoryPool,
    ) -> Result<Tensor, EngineError> {
        self.matrix_vector(weight, hidden, pool)
    }

    /// Computes separate Q, K, and V matrix-vector projections.
    ///
    /// This is a convenience composition, not a fused kernel: each projection
    /// runs through the independently tested matrix-vector implementation.
    pub fn qkv_projections(
        &self,
        input: &impl TensorRead,
        query_weight: &impl TensorRead,
        key_weight: &impl TensorRead,
        value_weight: &impl TensorRead,
        pool: &MemoryPool,
    ) -> Result<(Tensor, Tensor, Tensor), EngineError> {
        Ok((
            self.matrix_vector(query_weight, input, pool)?,
            self.matrix_vector(key_weight, input, pool)?,
            self.matrix_vector(value_weight, input, pool)?,
        ))
    }

    /// Backward-compatible exact-shape F32 addition.
    pub fn add_f32(
        &self,
        left: &Tensor,
        right: &Tensor,
        pool: &MemoryPool,
    ) -> Result<Tensor, EngineError> {
        if left.shape() != right.shape() {
            return Err(EngineError::invalid_input(
                "elementwise addition requires equal tensor shapes",
            ));
        }
        self.add(left, right, pool)
    }

    /// Backward-compatible rank-two F32 matrix multiplication.
    pub fn matrix_multiply_f32(
        &self,
        left: &Tensor,
        right: &Tensor,
        pool: &MemoryPool,
    ) -> Result<Tensor, EngineError> {
        self.matrix_matrix(left, right, pool)
    }
}

pub(crate) fn infer_broadcast_shape(
    left: &TensorShape,
    right: &TensorShape,
) -> Result<TensorShape, EngineError> {
    let rank = left.rank().max(right.rank());
    let mut dimensions = vec![1_u64; rank];
    for (output_axis, output_dimension) in dimensions.iter_mut().enumerate() {
        let left_dimension = trailing_dimension(left, rank, output_axis);
        let right_dimension = trailing_dimension(right, rank, output_axis);
        if left_dimension != right_dimension && left_dimension != 1 && right_dimension != 1 {
            return Err(EngineError::invalid_input("tensor broadcast dimensions are incompatible"));
        }
        *output_dimension = left_dimension.max(right_dimension);
    }
    TensorShape::new(dimensions)
}

fn trailing_dimension(shape: &TensorShape, output_rank: usize, output_axis: usize) -> u64 {
    let leading = output_rank - shape.rank();
    if output_axis < leading { 1 } else { shape.dimensions()[output_axis - leading] }
}

fn elementwise_binary(
    left: &impl TensorRead,
    right: &impl TensorRead,
    pool: &MemoryPool,
    operation: impl Fn(f32, f32) -> f32,
) -> Result<Tensor, EngineError> {
    require_f32(left)?;
    require_f32(right)?;
    require_pool(left, pool)?;
    require_pool(right, pool)?;
    let shape = infer_broadcast_shape(left.shape(), right.shape())?;
    checked_elements(shape.element_count())?;
    let mut output = MutableTensor::zeros(DType::F32, shape.clone(), pool)?;
    for output_index in 0..usize_from_u64(shape.element_count())? {
        let left_index = broadcast_index(u64_from_usize(output_index)?, &shape, left.shape())?;
        let right_index = broadcast_index(u64_from_usize(output_index)?, &shape, right.shape())?;
        let value = operation(
            finite_input(left.f32_value(usize_from_u64(left_index)?)?)?,
            finite_input(right.f32_value(usize_from_u64(right_index)?)?)?,
        );
        output.set_f32_value(output_index, finite_output(value)?)?;
    }
    Ok(output)
}

fn unary_f32(
    input: &impl TensorRead,
    pool: &MemoryPool,
    operation: impl Fn(f32) -> f32,
) -> Result<Tensor, EngineError> {
    checked_elements(input.shape().element_count())?;
    let mut output = MutableTensor::zeros(DType::F32, input.shape().clone(), pool)?;
    for index in 0..usize_from_u64(input.shape().element_count())? {
        output.set_f32_value(
            index,
            finite_output(operation(finite_input(input.f32_value(index)?)?))?,
        )?;
    }
    Ok(output)
}

fn broadcast_index(
    output_index: u64,
    output_shape: &TensorShape,
    input_shape: &TensorShape,
) -> Result<u64, EngineError> {
    let output_rank = output_shape.rank();
    let input_leading = output_rank - input_shape.rank();
    let mut remainder = output_index;
    let mut input_index = 0_u64;
    let mut input_stride = 1_u64;
    for axis in (0..output_rank).rev() {
        let output_dimension = output_shape.dimensions()[axis];
        let coordinate = remainder % output_dimension;
        remainder /= output_dimension;
        if axis >= input_leading {
            let input_dimension = input_shape.dimensions()[axis - input_leading];
            let input_coordinate = if input_dimension == 1 { 0 } else { coordinate };
            input_index = input_coordinate
                .checked_mul(input_stride)
                .and_then(|part| input_index.checked_add(part))
                .ok_or_else(|| EngineError::resource_exhaustion("broadcast index overflows"))?;
            input_stride = input_stride
                .checked_mul(input_dimension)
                .ok_or_else(|| EngineError::resource_exhaustion("broadcast stride overflows"))?;
        }
    }
    Ok(input_index)
}

pub(crate) fn require_f32(tensor: &(impl TensorRead + ?Sized)) -> Result<(), EngineError> {
    if tensor.dtype() == DType::F32 {
        Ok(())
    } else {
        Err(EngineError::not_implemented("CPU reference arithmetic requires f32 execution tensors"))
    }
}

pub(crate) fn require_pool(
    tensor: &(impl TensorRead + ?Sized),
    pool: &MemoryPool,
) -> Result<(), EngineError> {
    if tensor.accounted_by(pool) {
        Ok(())
    } else {
        Err(EngineError::invalid_input("CPU reference tensors must use the operation memory pool"))
    }
}

fn checked_elements(elements: u64) -> Result<(), EngineError> {
    if elements > MAX_CPU_REFERENCE_OPERATIONS {
        Err(EngineError::resource_exhaustion(
            "CPU reference operation exceeds its hard compute bound",
        ))
    } else {
        Ok(())
    }
}

fn checked_product(values: &[u64]) -> Result<u64, EngineError> {
    let operations = values.iter().try_fold(1_u64, |product, value| {
        product
            .checked_mul(*value)
            .ok_or_else(|| EngineError::resource_exhaustion("CPU operation count overflows"))
    })?;
    checked_elements(operations)?;
    Ok(operations)
}

fn finite_input(value: f32) -> Result<f32, EngineError> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(EngineError::backend_failure("CPU reference arithmetic received a non-finite input"))
    }
}

fn finite_output(value: f32) -> Result<f32, EngineError> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(EngineError::backend_failure("CPU reference arithmetic produced a non-finite output"))
    }
}

fn f32_from_f64(value: f64) -> Result<f32, EngineError> {
    if !value.is_finite() || value > f64::from(f32::MAX) || value < f64::from(f32::MIN) {
        return Err(EngineError::backend_failure("CPU reference accumulator is outside f32 range"));
    }
    #[allow(clippy::cast_possible_truncation)]
    let converted = value as f32;
    Ok(converted)
}

#[allow(clippy::cast_precision_loss)]
fn f64_from_usize(value: usize) -> Result<f64, EngineError> {
    let converted = value as f64;
    if converted.is_finite() {
        Ok(converted)
    } else {
        Err(EngineError::resource_exhaustion("CPU dimension is outside f64 range"))
    }
}

fn checked_index(row: usize, width: usize, column: usize) -> Result<usize, EngineError> {
    row.checked_mul(width)
        .and_then(|offset| offset.checked_add(column))
        .ok_or_else(|| EngineError::resource_exhaustion("CPU tensor index overflows"))
}

fn checked_index3(
    first: usize,
    second: usize,
    third: usize,
    second_width: usize,
    third_width: usize,
) -> Result<usize, EngineError> {
    first
        .checked_mul(second_width)
        .and_then(|value| value.checked_add(second))
        .and_then(|value| value.checked_mul(third_width))
        .and_then(|value| value.checked_add(third))
        .ok_or_else(|| EngineError::resource_exhaustion("CPU rank-three index overflows"))
}

fn usize_from_u64(value: u64) -> Result<usize, EngineError> {
    usize::try_from(value)
        .map_err(|_| EngineError::resource_exhaustion("CPU dimension is not addressable"))
}

fn u64_from_usize(value: usize) -> Result<u64, EngineError> {
    u64::try_from(value)
        .map_err(|_| EngineError::resource_exhaustion("CPU index is not representable"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tensor(pool: &MemoryPool, shape: impl IntoIterator<Item = u64>, values: &[f32]) -> Tensor {
        MutableTensor::from_f32_values(TensorShape::new(shape).unwrap(), values, pool).unwrap()
    }

    fn values(tensor: &Tensor) -> Vec<f32> {
        (0..usize_from_u64(tensor.shape().element_count()).unwrap())
            .map(|index| tensor.f32_value(index).unwrap())
            .collect()
    }

    #[test]
    fn copy_convert_and_half_storage_are_exact() {
        let pool = MemoryPool::new(1 << 20).unwrap();
        let backend = CpuReferenceBackend::new().unwrap();
        let input = tensor(&pool, [4], &[1.0, -2.0, 0.0, 65_504.0]);
        let f16 = backend.convert(&input, DType::F16, &pool).unwrap();
        let restored = backend.convert(&f16, DType::F32, &pool).unwrap();
        assert_eq!(values(&restored), values(&input));
        let bf16 = backend.convert(&input, DType::BFloat16, &pool).unwrap();
        assert_eq!(
            values(&backend.convert(&bf16, DType::F32, &pool).unwrap())[..3],
            [1.0, -2.0, 0.0]
        );
        assert_eq!(backend.copy(&input, &pool).unwrap(), input);
        let unsupported =
            MutableTensor::zeros(DType::I8, TensorShape::new([1]).unwrap(), &pool).unwrap();
        assert!(backend.convert(&unsupported, DType::F32, &pool).is_err());
    }

    #[test]
    fn add_multiply_scalar_and_broadcast_examples_are_exact() {
        let pool = MemoryPool::new(1 << 20).unwrap();
        let backend = CpuReferenceBackend::new().unwrap();
        let left = tensor(&pool, [2, 2], &[1.0, 2.0, 3.0, 4.0]);
        let right = tensor(&pool, [2], &[10.0, 20.0]);
        assert_eq!(values(&backend.add(&left, &right, &pool).unwrap()), [11.0, 22.0, 13.0, 24.0]);
        assert_eq!(
            values(&backend.multiply(&left, &right, &pool).unwrap()),
            [10.0, 40.0, 30.0, 80.0]
        );
        assert_eq!(values(&backend.scalar_multiply(&right, 0.5, &pool).unwrap()), [5.0, 10.0]);
        let mismatch = tensor(&pool, [3], &[1.0, 2.0, 3.0]);
        assert!(backend.add(&left, &mismatch, &pool).is_err());
    }

    #[test]
    fn matrix_vector_matrix_matrix_and_transpose_are_hand_calculated() {
        let pool = MemoryPool::new(1 << 20).unwrap();
        let backend = CpuReferenceBackend::new().unwrap();
        let matrix = tensor(&pool, [2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let vector = tensor(&pool, [3], &[1.0, 2.0, 3.0]);
        assert_eq!(values(&backend.matrix_vector(&matrix, &vector, &pool).unwrap()), [14.0, 32.0]);
        let right = tensor(&pool, [3, 2], &[7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
        assert_eq!(
            values(&backend.matrix_matrix(&matrix, &right, &pool).unwrap()),
            [58.0, 64.0, 139.0, 154.0]
        );
        let right_owner = tensor(&pool, [2, 3], &[7.0, 9.0, 11.0, 8.0, 10.0, 12.0]);
        let right_view = right_owner.view();
        let transposed = right_view.transpose(0, 1).unwrap();
        assert_eq!(
            values(&backend.matrix_matrix(&matrix, &transposed, &pool).unwrap()),
            [58.0, 64.0, 139.0, 154.0]
        );
    }

    #[test]
    fn embedding_rmsnorm_softmax_and_extreme_magnitude_are_stable() {
        let pool = MemoryPool::new(1 << 20).unwrap();
        let backend = CpuReferenceBackend::new().unwrap();
        let embedding = tensor(&pool, [2, 2], &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(values(&backend.embedding_lookup(&embedding, 1, &pool).unwrap()), [3.0, 4.0]);
        assert!(backend.embedding_lookup(&embedding, 2, &pool).is_err());
        let input = tensor(&pool, [1, 2], &[3.0, 4.0]);
        let weight = tensor(&pool, [2], &[1.0, 1.0]);
        let normalized = backend.rms_norm(&input, &weight, 1.0e-6, &pool).unwrap();
        assert!((normalized.f32_value(0).unwrap() - 0.848_528).abs() < 1.0e-5);
        let extreme = tensor(&pool, [1, 2], &[f32::MAX, f32::MAX]);
        assert!(backend.rms_norm(&extreme, &weight, 1.0e-6, &pool).is_ok());
        let logits = tensor(&pool, [3], &[1_000.0, 1_001.0, f32::NEG_INFINITY]);
        let probabilities = backend.softmax(&logits, &pool).unwrap();
        let sum: f32 = values(&probabilities).iter().sum();
        assert!((sum - 1.0).abs() < 1.0e-6);
        assert_eq!(probabilities.f32_value(2).unwrap().to_bits(), 0.0_f32.to_bits());
    }

    #[test]
    fn silu_swiglu_rope_heads_and_causal_mask_are_deterministic() {
        let pool = MemoryPool::new(1 << 20).unwrap();
        let backend = CpuReferenceBackend::new().unwrap();
        let gate = tensor(&pool, [2], &[0.0, 1.0]);
        let up = tensor(&pool, [2], &[2.0, 3.0]);
        assert_eq!(
            backend.silu(&gate, &pool).unwrap().f32_value(0).unwrap().to_bits(),
            0.0_f32.to_bits()
        );
        assert!(
            (backend.swiglu(&gate, &up, &pool).unwrap().f32_value(1).unwrap() - 2.193_176).abs()
                < 1.0e-5
        );
        let heads =
            backend.head_reshape(&tensor(&pool, [4], &[1.0, 0.0, 0.0, 1.0]), 2, &pool).unwrap();
        let first = backend.rope(&heads, 1, 10_000.0, &pool).unwrap();
        let second = backend.rope(&heads, 1, 10_000.0, &pool).unwrap();
        assert_eq!(first, second);
        let scores = tensor(&pool, [2, 3], &[1.0; 6]);
        let masked = backend.causal_mask(&scores, 0, &pool).unwrap();
        assert_eq!(
            values(&masked),
            [1.0, f32::NEG_INFINITY, f32::NEG_INFINITY, 1.0, 1.0, f32::NEG_INFINITY]
        );
    }

    #[test]
    fn grouped_query_attention_residual_logits_and_qkv_are_exact() {
        let pool = MemoryPool::new(1 << 20).unwrap();
        let backend = CpuReferenceBackend::new().unwrap();
        let query = tensor(&pool, [2, 2], &[1.0, 0.0, 0.0, 1.0]);
        let key = tensor(&pool, [1, 1, 2], &[1.0, 0.0]);
        let value = tensor(&pool, [1, 1, 2], &[2.0, 3.0]);
        assert_eq!(
            values(&backend.grouped_query_attention(&query, &key, &value, 0, &pool).unwrap()),
            [2.0, 3.0, 2.0, 3.0]
        );
        assert_eq!(
            values(&backend.residual_add(&query, &query, &pool).unwrap()),
            [2.0, 0.0, 0.0, 2.0]
        );
        let weight = tensor(&pool, [2, 2], &[1.0, 0.0, 0.0, 2.0]);
        let hidden = tensor(&pool, [2], &[3.0, 4.0]);
        assert_eq!(values(&backend.logit_projection(&weight, &hidden, &pool).unwrap()), [3.0, 8.0]);
        let (q, k, v) = backend.qkv_projections(&hidden, &weight, &weight, &weight, &pool).unwrap();
        assert_eq!(q, k);
        assert_eq!(k, v);

        let bounded_pool = MemoryPool::new(48).unwrap();
        let bounded_query = tensor(&bounded_pool, [2, 2], &[1.0, 0.0, 0.0, 1.0]);
        let bounded_key = tensor(&bounded_pool, [1, 1, 2], &[1.0, 0.0]);
        let bounded_value = tensor(&bounded_pool, [1, 1, 2], &[2.0, 3.0]);
        assert!(matches!(
            backend.grouped_query_attention(
                &bounded_query,
                &bounded_key,
                &bounded_value,
                0,
                &bounded_pool,
            ),
            Err(EngineError::ResourceExhaustion(_))
        ));
    }

    #[test]
    fn prohibited_dimensions_nonfinite_values_and_wrong_pool_fail_closed() {
        let pool = MemoryPool::new(1 << 20).unwrap();
        let other = MemoryPool::new(1 << 20).unwrap();
        let backend = CpuReferenceBackend::new().unwrap();
        let finite = tensor(&pool, [1], &[1.0]);
        let nan = tensor(&pool, [1], &[f32::NAN]);
        let infinity = tensor(&pool, [1], &[f32::INFINITY]);
        assert!(backend.add(&finite, &nan, &pool).is_err());
        assert!(backend.multiply(&finite, &infinity, &pool).is_err());
        assert!(backend.add(&finite, &finite, &other).is_err());
        assert!(backend.scalar_multiply(&finite, f32::NAN, &pool).is_err());
        let scalar = tensor(&pool, [], &[1.0]);
        assert!(backend.softmax(&scalar, &pool).is_err());
        let all_masked = tensor(&pool, [2], &[f32::NEG_INFINITY; 2]);
        assert!(backend.softmax(&all_masked, &pool).is_err());
    }

    #[test]
    #[allow(clippy::too_many_lines)]
    fn every_required_operation_repeats_bit_exactly() {
        let pool = MemoryPool::new(1 << 22).unwrap();
        let backend = CpuReferenceBackend::new().unwrap();
        let vector = tensor(&pool, [2], &[0.25, -0.5]);
        let other = tensor(&pool, [2], &[0.75, 1.5]);
        let matrix = tensor(&pool, [2, 2], &[1.0, 0.5, -0.25, 2.0]);
        let embedding = tensor(&pool, [2, 2], &[1.0, 2.0, 3.0, 4.0]);
        let weight = tensor(&pool, [2], &[1.0, 1.0]);
        let query = tensor(&pool, [2, 2], &[1.0, 0.0, 0.0, 1.0]);
        let key = tensor(&pool, [1, 1, 2], &[1.0, 0.0]);
        let attention_value = tensor(&pool, [1, 1, 2], &[2.0, 3.0]);
        let scores = tensor(&pool, [1, 2], &[1.0, 2.0]);

        assert_eq!(backend.copy(&vector, &pool).unwrap(), backend.copy(&vector, &pool).unwrap());
        assert_eq!(
            backend.convert(&vector, DType::F16, &pool).unwrap(),
            backend.convert(&vector, DType::F16, &pool).unwrap()
        );
        assert_eq!(
            backend.add(&vector, &other, &pool).unwrap(),
            backend.add(&vector, &other, &pool).unwrap()
        );
        assert_eq!(
            backend.multiply(&vector, &other, &pool).unwrap(),
            backend.multiply(&vector, &other, &pool).unwrap()
        );
        assert_eq!(
            backend.scalar_multiply(&vector, 0.5, &pool).unwrap(),
            backend.scalar_multiply(&vector, 0.5, &pool).unwrap()
        );
        assert_eq!(
            backend.matrix_vector(&matrix, &vector, &pool).unwrap(),
            backend.matrix_vector(&matrix, &vector, &pool).unwrap()
        );
        assert_eq!(
            backend.matrix_matrix(&matrix, &matrix, &pool).unwrap(),
            backend.matrix_matrix(&matrix, &matrix, &pool).unwrap()
        );
        assert_eq!(
            backend.embedding_lookup(&embedding, 1, &pool).unwrap(),
            backend.embedding_lookup(&embedding, 1, &pool).unwrap()
        );
        assert_eq!(
            backend.rms_norm(&vector, &weight, 1.0e-5, &pool).unwrap(),
            backend.rms_norm(&vector, &weight, 1.0e-5, &pool).unwrap()
        );
        assert_eq!(
            backend.softmax(&scores, &pool).unwrap(),
            backend.softmax(&scores, &pool).unwrap()
        );
        assert_eq!(backend.silu(&vector, &pool).unwrap(), backend.silu(&vector, &pool).unwrap());
        assert_eq!(
            backend.swiglu(&vector, &other, &pool).unwrap(),
            backend.swiglu(&vector, &other, &pool).unwrap()
        );
        assert_eq!(
            backend.rope(&query, 1, 10_000.0, &pool).unwrap(),
            backend.rope(&query, 1, 10_000.0, &pool).unwrap()
        );
        assert_eq!(
            backend.head_reshape(&vector, 1, &pool).unwrap(),
            backend.head_reshape(&vector, 1, &pool).unwrap()
        );
        assert_eq!(
            backend.grouped_query_attention(&query, &key, &attention_value, 0, &pool).unwrap(),
            backend.grouped_query_attention(&query, &key, &attention_value, 0, &pool).unwrap()
        );
        assert_eq!(
            backend.causal_mask(&scores, 0, &pool).unwrap(),
            backend.causal_mask(&scores, 0, &pool).unwrap()
        );
        assert_eq!(
            backend.residual_add(&vector, &other, &pool).unwrap(),
            backend.residual_add(&vector, &other, &pool).unwrap()
        );
        assert_eq!(
            backend.logit_projection(&matrix, &vector, &pool).unwrap(),
            backend.logit_projection(&matrix, &vector, &pool).unwrap()
        );
        let first = backend.qkv_projections(&vector, &matrix, &matrix, &matrix, &pool).unwrap();
        let second = backend.qkv_projections(&vector, &matrix, &matrix, &matrix, &pool).unwrap();
        assert_eq!(first, second);
    }
}
