//! Isolated ARM64 Advanced SIMD boundary.
//!
//! Safe callers supply validated initialized slices or fixed arrays. Every
//! intrinsic load and store remains within a complete 4-lane F32/U16, 8-lane
//! I8, or fixed 32-code range proven by the safe facade. ARM64 requires
//! Advanced SIMD, which the facade also verifies at runtime.

#![allow(unsafe_code)]

#[cfg(target_endian = "big")]
use std::arch::aarch64::vrev32q_u8;
use std::arch::aarch64::{
    float32x4_t, vaddq_f32, vaddq_u32, vaddvq_f32, vandq_u32, vceqq_u32, vcvtq_f32_s32,
    vdupq_n_f32, vdupq_n_u32, vget_high_s16, vget_low_s16, vld1_s8, vld1_u16, vld1q_f32, vld1q_u8,
    vmaxvq_u32, vmovl_s8, vmovl_s16, vmovl_u16, vmulq_f32, vorrq_u32, vreinterpretq_f32_u32,
    vreinterpretq_u32_u8, vrev64q_f32, vshlq_n_u32, vst1q_f32,
};
use std::mem::{size_of, size_of_val};

pub(super) fn dot(left: &[f32], right: &[f32]) -> f32 {
    debug_assert_eq!(left.len(), right.len());
    let vector_end = left.len() - left.len() % 4;
    let vector_sum = unsafe {
        let mut lanes: float32x4_t = vdupq_n_f32(0.0);
        let mut index = 0;
        while index < vector_end {
            let left_lanes = vld1q_f32(left.as_ptr().add(index));
            let right_lanes = vld1q_f32(right.as_ptr().add(index));
            lanes = vaddq_f32(lanes, vmulq_f32(left_lanes, right_lanes));
            index += 4;
        }
        vaddvq_f32(lanes)
    };
    left[vector_end..]
        .iter()
        .copied()
        .zip(right[vector_end..].iter().copied())
        .fold(vector_sum, |sum, (left, right)| sum + left * right)
}

/// Dots canonical little-endian F32 bytes directly with one finite F32 slice.
///
/// The safe facade proves the exact byte/value length relationship. This
/// boundary checks every encoded input lane before interpreting it as F32 so a
/// NaN or infinity can never bypass the backend's ordinary finite-input rule.
pub(super) fn dot_canonical_f32_le_bytes(left: &[u8], right: &[f32]) -> Option<f32> {
    debug_assert_eq!(left.len(), size_of_val(right));
    let vector_end = right.len() - right.len() % 4;
    let vector_sum = unsafe {
        let exponent_mask = vdupq_n_u32(0x7f80_0000);
        let mut lanes: float32x4_t = vdupq_n_f32(0.0);
        let mut index = 0;
        while index < vector_end {
            let byte_offset = index * size_of::<f32>();
            let encoded = vld1q_u8(left.as_ptr().add(byte_offset));
            #[cfg(target_endian = "big")]
            let encoded = vrev32q_u8(encoded);
            let bits = vreinterpretq_u32_u8(encoded);
            let nonfinite = vceqq_u32(vandq_u32(bits, exponent_mask), exponent_mask);
            if vmaxvq_u32(nonfinite) != 0 {
                return None;
            }
            lanes = vaddq_f32(
                lanes,
                vmulq_f32(vreinterpretq_f32_u32(bits), vld1q_f32(right.as_ptr().add(index))),
            );
            index += 4;
        }
        vaddvq_f32(lanes)
    };
    let mut sum = vector_sum;
    for (encoded, right) in left[vector_end * size_of::<f32>()..]
        .chunks_exact(size_of::<f32>())
        .zip(&right[vector_end..])
    {
        let value = f32::from_le_bytes([encoded[0], encoded[1], encoded[2], encoded[3]]);
        if !value.is_finite() {
            return None;
        }
        sum += value * right;
    }
    Some(sum)
}

pub(super) fn add(left: &[f32], right: &[f32], output: &mut [f32]) {
    debug_assert_eq!(left.len(), right.len());
    debug_assert_eq!(left.len(), output.len());
    let vector_end = left.len() - left.len() % 4;
    unsafe {
        let mut index = 0;
        while index < vector_end {
            let value = vaddq_f32(
                vld1q_f32(left.as_ptr().add(index)),
                vld1q_f32(right.as_ptr().add(index)),
            );
            vst1q_f32(output.as_mut_ptr().add(index), value);
            index += 4;
        }
    }
    for index in vector_end..left.len() {
        output[index] = left[index] + right[index];
    }
}

pub(super) fn multiply(left: &[f32], right: &[f32], output: &mut [f32]) {
    debug_assert_eq!(left.len(), right.len());
    debug_assert_eq!(left.len(), output.len());
    let vector_end = left.len() - left.len() % 4;
    unsafe {
        let mut index = 0;
        while index < vector_end {
            let value = vmulq_f32(
                vld1q_f32(left.as_ptr().add(index)),
                vld1q_f32(right.as_ptr().add(index)),
            );
            vst1q_f32(output.as_mut_ptr().add(index), value);
            index += 4;
        }
    }
    for index in vector_end..left.len() {
        output[index] = left[index] * right[index];
    }
}

pub(super) fn scale(input: &[f32], scalar: f32, output: &mut [f32]) {
    debug_assert_eq!(input.len(), output.len());
    let vector_end = input.len() - input.len() % 4;
    unsafe {
        let scalar_lanes = vdupq_n_f32(scalar);
        let mut index = 0;
        while index < vector_end {
            let value = vmulq_f32(vld1q_f32(input.as_ptr().add(index)), scalar_lanes);
            vst1q_f32(output.as_mut_ptr().add(index), value);
            index += 4;
        }
    }
    for index in vector_end..input.len() {
        output[index] = input[index] * scalar;
    }
}

/// Scales one initialized slice in place.
pub(super) fn scale_in_place(values: &mut [f32], scalar: f32) {
    let vector_end = values.len() - values.len() % 4;
    unsafe {
        let scalar_lanes = vdupq_n_f32(scalar);
        let mut index = 0;
        while index < vector_end {
            let value = vmulq_f32(vld1q_f32(values.as_ptr().add(index)), scalar_lanes);
            vst1q_f32(values.as_mut_ptr().add(index), value);
            index += 4;
        }
    }
    for value in &mut values[vector_end..] {
        *value *= scalar;
    }
}

/// Computes up to four output columns for one row-major matrix row.
pub(super) fn matrix_matrix_block(
    left: &[f32],
    right: &[f32],
    columns: usize,
    first_column: usize,
) -> [f32; 4] {
    debug_assert_eq!(right.len() % columns, 0);
    debug_assert_eq!(left.len(), right.len() / columns);
    let available = columns.saturating_sub(first_column).min(4);
    let mut output = [0.0_f32; 4];
    if available == 4 {
        unsafe {
            let mut accumulator = vdupq_n_f32(0.0);
            for (inner, left_value) in left.iter().copied().enumerate() {
                let right_offset = inner * columns + first_column;
                accumulator = vaddq_f32(
                    accumulator,
                    vmulq_f32(vdupq_n_f32(left_value), vld1q_f32(right.as_ptr().add(right_offset))),
                );
            }
            vst1q_f32(output.as_mut_ptr(), accumulator);
        }
    } else {
        for (offset, destination) in output.iter_mut().take(available).enumerate() {
            let column = first_column + offset;
            *destination = left
                .iter()
                .copied()
                .enumerate()
                .fold(0.0, |sum, (inner, value)| sum + value * right[inner * columns + column]);
        }
    }
    output
}

/// Applies the final RMSNorm multiply after the safe facade computes its robust scale.
pub(super) fn rms_apply(input: &[f32], weight: &[f32], inverse: f32, output: &mut [f32]) {
    debug_assert_eq!(input.len(), weight.len());
    debug_assert_eq!(input.len(), output.len());
    let vector_end = input.len() - input.len() % 4;
    unsafe {
        let inverse = vdupq_n_f32(inverse);
        let mut index = 0;
        while index < vector_end {
            let value = vmulq_f32(
                vmulq_f32(
                    vld1q_f32(input.as_ptr().add(index)),
                    vld1q_f32(weight.as_ptr().add(index)),
                ),
                inverse,
            );
            vst1q_f32(output.as_mut_ptr().add(index), value);
            index += 4;
        }
    }
    for index in vector_end..input.len() {
        output[index] = input[index] * weight[index] * inverse;
    }
}

/// Applies interleaved complex rotations to complete `(x, y)` pairs.
pub(super) fn rope(input: &[f32], cosine: &[f32], sine: &[f32], output: &mut [f32]) {
    debug_assert_eq!(input.len(), cosine.len());
    debug_assert_eq!(input.len(), sine.len());
    debug_assert_eq!(input.len(), output.len());
    debug_assert_eq!(input.len() % 2, 0);
    let vector_end = input.len() - input.len() % 4;
    unsafe {
        let signs = vld1q_f32([-1.0_f32, 1.0, -1.0, 1.0].as_ptr());
        let mut index = 0;
        while index < vector_end {
            let values = vld1q_f32(input.as_ptr().add(index));
            let rotated = vaddq_f32(
                vmulq_f32(values, vld1q_f32(cosine.as_ptr().add(index))),
                vmulq_f32(
                    vmulq_f32(vrev64q_f32(values), vld1q_f32(sine.as_ptr().add(index))),
                    signs,
                ),
            );
            vst1q_f32(output.as_mut_ptr().add(index), rotated);
            index += 4;
        }
    }
    for index in (vector_end..input.len()).step_by(2) {
        output[index] = input[index] * cosine[index] - input[index + 1] * sine[index];
        output[index + 1] = input[index] * sine[index] + input[index + 1] * cosine[index];
    }
}

/// Adds `weight * input` to an initialized accumulation slice.
pub(super) fn weighted_accumulate(input: &[f32], weight: f32, output: &mut [f32]) {
    debug_assert_eq!(input.len(), output.len());
    let vector_end = input.len() - input.len() % 4;
    unsafe {
        let weight = vdupq_n_f32(weight);
        let mut index = 0;
        while index < vector_end {
            let value = vaddq_f32(
                vld1q_f32(output.as_ptr().add(index)),
                vmulq_f32(vld1q_f32(input.as_ptr().add(index)), weight),
            );
            vst1q_f32(output.as_mut_ptr().add(index), value);
            index += 4;
        }
    }
    for index in vector_end..input.len() {
        output[index] += input[index] * weight;
    }
}

/// Expands normal finite F16 encodings to F32. Special encodings use the scalar tail.
pub(super) fn f16_to_f32(input: &[u16], output: &mut [f32]) {
    debug_assert_eq!(input.len(), output.len());
    let mut index = 0;
    while index + 4 <= input.len() {
        let chunk = &input[index..index + 4];
        if chunk.iter().all(|bits| {
            let exponent = bits & 0x7c00;
            exponent != 0 && exponent != 0x7c00
        }) {
            unsafe {
                let half = vmovl_u16(vld1_u16(input.as_ptr().add(index)));
                let sign = vshlq_n_u32(vandq_u32(half, vdupq_n_u32(0x8000)), 16);
                let exponent = vshlq_n_u32(
                    vaddq_u32(vandq_u32(half, vdupq_n_u32(0x7c00)), vdupq_n_u32(112 << 10)),
                    13,
                );
                let fraction = vshlq_n_u32(vandq_u32(half, vdupq_n_u32(0x03ff)), 13);
                let bits = vorrq_u32(sign, vorrq_u32(exponent, fraction));
                vst1q_f32(output.as_mut_ptr().add(index), vreinterpretq_f32_u32(bits));
            }
        } else {
            for offset in 0..4 {
                output[index + offset] = personalai_engine_tensor::f16_bits_to_f32(chunk[offset]);
            }
        }
        index += 4;
    }
    for offset in index..input.len() {
        output[offset] = personalai_engine_tensor::f16_bits_to_f32(input[offset]);
    }
}

/// Expands BF16 encodings by shifting their exact IEEE payload into F32 lanes.
pub(super) fn bf16_to_f32(input: &[u16], output: &mut [f32]) {
    debug_assert_eq!(input.len(), output.len());
    let vector_end = input.len() - input.len() % 4;
    unsafe {
        let mut index = 0;
        while index < vector_end {
            let widened = vmovl_u16(vld1_u16(input.as_ptr().add(index)));
            let bits = vshlq_n_u32(widened, 16);
            vst1q_f32(output.as_mut_ptr().add(index), vreinterpretq_f32_u32(bits));
            index += 4;
        }
    }
    for index in vector_end..input.len() {
        output[index] = personalai_engine_tensor::bf16_bits_to_f32(input[index]);
    }
}

/// Expands 32 signed AIPQ codes using one explicit F32 scale.
pub(super) fn dequantize_block(codes: &[i8; 32], scale: f32, output: &mut [f32; 32]) {
    unsafe {
        let scale = vdupq_n_f32(scale);
        for chunk in 0..4 {
            let wide16 = vmovl_s8(vld1_s8(codes.as_ptr().add(chunk * 8)));
            let lanes = [vmovl_s16(vget_low_s16(wide16)), vmovl_s16(vget_high_s16(wide16))];
            for (quarter, values) in lanes.into_iter().enumerate() {
                let offset = chunk * 8 + quarter * 4;
                vst1q_f32(output.as_mut_ptr().add(offset), vmulq_f32(vcvtq_f32_s32(values), scale));
            }
        }
    }
}
