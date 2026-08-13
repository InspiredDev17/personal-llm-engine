//! Isolated ARM64 Advanced SIMD boundary.
//!
//! Safe callers supply equally sized, initialized F32 slices. Every intrinsic
//! load and store remains within a four-element chunk proven by the loop bound.
//! ARM64 requires Advanced SIMD, so no optional runtime feature is assumed.

#![allow(unsafe_code)]

use std::arch::aarch64::{
    float32x4_t, vaddq_f32, vaddvq_f32, vdupq_n_f32, vld1q_f32, vmulq_f32, vst1q_f32,
};

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
