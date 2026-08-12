//! Manual microbenchmarks for the scalar correctness oracle.

use personalai_engine_cpu_reference::CpuReferenceBackend;
use personalai_engine_memory::MemoryPool;
use personalai_engine_tensor::{MutableTensor, Tensor, TensorShape};
use std::hint::black_box;
use std::time::{Duration, Instant};

const ITERATIONS: u32 = 25;

fn tensor(pool: &MemoryPool, shape: impl IntoIterator<Item = u64>, value: f32) -> Tensor {
    let shape = TensorShape::new(shape).expect("benchmark shape must validate");
    let count = usize::try_from(shape.element_count()).expect("benchmark size must fit");
    MutableTensor::from_f32_values(shape, &vec![value; count], pool)
        .expect("benchmark tensor must construct")
}

fn measure(label: &str, mut operation: impl FnMut()) -> Duration {
    let start = Instant::now();
    for _ in 0..ITERATIONS {
        operation();
    }
    let elapsed = start.elapsed();
    println!(
        "correctness-oracle {label}: {ITERATIONS} iterations in {} microseconds",
        elapsed.as_micros()
    );
    elapsed
}

fn main() {
    let pool = MemoryPool::new(64 * 1024 * 1024).expect("benchmark pool must validate");
    let backend = CpuReferenceBackend::new().expect("backend must construct");
    let matrix = tensor(&pool, [16, 16], 0.031_25);
    let matrix_right = tensor(&pool, [16, 16], 0.062_5);
    let rms_input = tensor(&pool, [4, 64], 0.25);
    let rms_weight = tensor(&pool, [64], 1.0);
    let logits = tensor(&pool, [4, 64], 0.125);
    let rope_input = tensor(&pool, [4, 16], 0.25);

    measure("matrix-matrix-16x16", || {
        black_box(backend.matrix_matrix(&matrix, &matrix_right, &pool).unwrap());
    });
    measure("rmsnorm-4x64", || {
        black_box(backend.rms_norm(&rms_input, &rms_weight, 1.0e-5, &pool).unwrap());
    });
    measure("softmax-4x64", || {
        black_box(backend.softmax(&logits, &pool).unwrap());
    });
    measure("rope-4x16", || {
        black_box(backend.rope(&rope_input, 7, 10_000.0, &pool).unwrap());
    });

    let state = tensor(&pool, [4], 0.125);
    let norm = tensor(&pool, [4], 1.0);
    let projection = tensor(&pool, [4, 4], 0.125);
    let key = tensor(&pool, [1, 1, 2], 0.125);
    let value = tensor(&pool, [1, 1, 2], 0.25);
    let gate_projection = tensor(&pool, [8, 4], 0.125);
    let down_projection = tensor(&pool, [4, 8], 0.125);
    measure("tiny-decoder-block", || {
        let normalized = backend.rms_norm(&state, &norm, 1.0e-5, &pool).unwrap();
        let query = backend.matrix_vector(&projection, &normalized, &pool).unwrap();
        let query_heads = backend.head_reshape(&query, 2, &pool).unwrap();
        let rotated = backend.rope(&query_heads, 0, 10_000.0, &pool).unwrap();
        let attended = backend.grouped_query_attention(&rotated, &key, &value, 0, &pool).unwrap();
        let merged = backend.head_reshape(&attended, 2, &pool).unwrap();
        let update = backend.matrix_vector(&projection, &merged, &pool).unwrap();
        let residual = backend.residual_add(&state, &update, &pool).unwrap();
        let ffn_input = backend.rms_norm(&residual, &norm, 1.0e-5, &pool).unwrap();
        let gate = backend.matrix_vector(&gate_projection, &ffn_input, &pool).unwrap();
        let up = backend.matrix_vector(&gate_projection, &ffn_input, &pool).unwrap();
        let activated = backend.swiglu(&gate, &up, &pool).unwrap();
        black_box(
            backend
                .matrix_vector(&down_projection, &activated, &pool)
                .and_then(|output| backend.residual_add(&residual, &output, &pool))
                .unwrap(),
        );
    });
}
