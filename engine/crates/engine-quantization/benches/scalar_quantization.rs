//! Bounded scalar reference measurements; these are not production claims.

use personalai_engine_memory::MemoryPool;
use personalai_engine_model_format::Quantization;
use personalai_engine_quantization::{QuantizedTensor, quantize};
use personalai_engine_tensor::{MutableTensor, TensorShape};
use std::hint::black_box;
use std::time::Instant;

const ROWS: u64 = 128;
const COLUMNS: u64 = 128;
const ITERATIONS: u32 = 25;

fn main() {
    let shape = TensorShape::new([ROWS, COLUMNS]).expect("benchmark shape must validate");
    let count = usize::try_from(shape.element_count()).expect("benchmark count must fit");
    let values = (0..count)
        .map(|index| {
            let bounded = i16::try_from(index % 257).expect("bounded benchmark value fits i16");
            f32::from(bounded - 128) / 17.0
        })
        .collect::<Vec<_>>();
    let pool = MemoryPool::new(8 * 1024 * 1024).expect("benchmark pool must validate");
    let vector = MutableTensor::from_f32_values(
        TensorShape::new([COLUMNS]).expect("benchmark vector shape must validate"),
        &vec![0.25; usize::try_from(COLUMNS).expect("columns fit")],
        &pool,
    )
    .expect("benchmark vector must allocate");
    for format in [Quantization::AipQ8V1, Quantization::AipQ4V1] {
        let start = Instant::now();
        for _ in 0..ITERATIONS {
            black_box(quantize(format, &shape, black_box(&values)).expect("quantize must succeed"));
        }
        let quantize_ns = per_iteration(start.elapsed().as_nanos());
        let payload = quantize(format, &shape, &values).expect("payload must quantize");
        let tensor =
            QuantizedTensor::parse(format, &shape, payload.bytes()).expect("payload must validate");
        let start = Instant::now();
        for _ in 0..ITERATIONS {
            black_box(tensor.dequantize(&pool).expect("dequantize must succeed"));
        }
        let dequantize_ns = per_iteration(start.elapsed().as_nanos());
        let start = Instant::now();
        for _ in 0..ITERATIONS {
            black_box(tensor.matrix_vector(&vector, &pool).expect("matvec must succeed"));
        }
        let matvec_ns = per_iteration(start.elapsed().as_nanos());
        println!(
            "format={format:?} shape={ROWS}x{COLUMNS} iterations={ITERATIONS} quantize_ns={quantize_ns} dequantize_ns={dequantize_ns} matvec_ns={matvec_ns}"
        );
    }
}

fn per_iteration(total_nanoseconds: u128) -> u128 {
    total_nanoseconds / u128::from(ITERATIONS)
}
