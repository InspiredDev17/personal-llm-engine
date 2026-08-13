//! Reproducible manual Apple Silicon kernel and graph measurements.

#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

#[cfg(target_arch = "aarch64")]
use personalai_engine_arm64::{Arm64Backend, Arm64ExecutionConfig, CancellationToken};
#[cfg(target_arch = "aarch64")]
use personalai_engine_cpu_reference::{
    Backend, CpuReferenceBackend, GraphNode, GraphNodeId, InferenceGraph, Operation, TensorSpec,
    ValueId,
};
#[cfg(target_arch = "aarch64")]
use personalai_engine_memory::MemoryPool;
#[cfg(target_arch = "aarch64")]
use personalai_engine_model_format::Quantization;
#[cfg(target_arch = "aarch64")]
use personalai_engine_quantization::{QuantizedTensor, quantize};
#[cfg(target_arch = "aarch64")]
use personalai_engine_tensor::{DType, Tensor, TensorShape};
#[cfg(target_arch = "aarch64")]
use std::collections::BTreeMap;
#[cfg(target_arch = "aarch64")]
use std::hint::black_box;
#[cfg(target_arch = "aarch64")]
use std::time::{Duration, Instant};

#[cfg(target_arch = "aarch64")]
const WARMUP_ITERATIONS: u32 = 3;
#[cfg(target_arch = "aarch64")]
const MEASURED_ITERATIONS: u32 = 20;

#[cfg(target_arch = "aarch64")]
fn main() {
    if cfg!(debug_assertions) {
        println!("ARM64 benchmark skipped outside an optimized cargo bench build");
        return;
    }
    println!("suite=arm64-graph-v2");
    println!("target_arch={}", std::env::consts::ARCH);
    println!("build_mode={}", if cfg!(debug_assertions) { "debug" } else { "release" });
    println!("fixture=deterministic-synthetic-f32-v1");
    println!("context_tokens=16");
    println!("prompt_tokens=16");
    println!("generated_tokens=32");
    println!("dtype=f32");
    println!("warmup_iterations={WARMUP_ITERATIONS}");
    println!("measured_iterations={MEASURED_ITERATIONS}");

    let pool = MemoryPool::new(512 << 20).expect("benchmark pool must construct");
    let scalar = CpuReferenceBackend::new().expect("scalar backend must construct");
    let arm64 = Arm64Backend::new().expect("ARM64 backend must construct");

    bench_matrix_vector(&pool, &scalar, &arm64);
    bench_matrix_matrix_by_thread_count(&pool, &scalar);
    bench_operation(&pool, &scalar, &arm64, &Operation::RmsNorm { epsilon: 1.0e-6 }, 32, 256);
    bench_operation(&pool, &scalar, &arm64, &Operation::Softmax, 32, 256);
    bench_operation(&pool, &scalar, &arm64, &Operation::Silu, 32, 256);
    bench_operation(&pool, &scalar, &arm64, &Operation::SwiGlu, 32, 256);
    bench_operation(
        &pool,
        &scalar,
        &arm64,
        &Operation::Rope { position: 127, theta: 10_000.0 },
        32,
        256,
    );
    bench_memory_bandwidth(&pool, &scalar, &arm64);
    bench_attention(&pool, &scalar, &arm64);
    bench_half_conversion(&pool, &scalar, &arm64);
    bench_quantized(&pool, &arm64);
    bench_tiny_transformer_graph(&pool, &scalar, &arm64);

    let stats = pool.statistics().expect("benchmark accounting must remain readable");
    println!("peak_host_memory_bytes={}", stats.peak_bytes());
    println!("final_host_memory_bytes={}", stats.reserved_bytes());
}

#[cfg(target_arch = "aarch64")]
fn bench_matrix_vector(pool: &MemoryPool, scalar: &CpuReferenceBackend, arm64: &Arm64Backend) {
    const ROWS: u64 = 512;
    const COLUMNS: u64 = 512;
    let matrix = tensor(pool, [ROWS, COLUMNS], ROWS as usize * COLUMNS as usize);
    let vector = tensor(pool, [COLUMNS], COLUMNS as usize);
    report_parity(
        "matvec-512x512",
        &scalar.execute_operation(&Operation::MatrixVector, &[&matrix, &vector], pool).unwrap(),
        &arm64.execute_operation(&Operation::MatrixVector, &[&matrix, &vector], pool).unwrap(),
    );
    for (label, backend) in [("scalar", scalar as &dyn Backend), ("arm64", arm64 as &dyn Backend)] {
        let elapsed = measure(|| {
            backend
                .execute_operation(&Operation::MatrixVector, &[&matrix, &vector], pool)
                .expect("matrix-vector benchmark must execute")
        });
        let operations = 2.0 * ROWS as f64 * COLUMNS as f64 * f64::from(MEASURED_ITERATIONS);
        println!(
            "benchmark=matvec-512x512 backend={label} elapsed_ns={} gflops={:.6}",
            elapsed.as_nanos(),
            operations / elapsed.as_secs_f64() / 1.0e9
        );
    }
}

#[cfg(target_arch = "aarch64")]
fn bench_matrix_matrix_by_thread_count(pool: &MemoryPool, scalar: &CpuReferenceBackend) {
    const SIZE: u64 = 128;
    let left = tensor(pool, [SIZE, SIZE], SIZE as usize * SIZE as usize);
    let right = tensor(pool, [SIZE, SIZE], SIZE as usize * SIZE as usize);
    let scalar_result = scalar
        .execute_operation(&Operation::MatrixMatrix, &[&left, &right], pool)
        .expect("scalar matrix parity must execute");
    let scalar_elapsed = measure(|| {
        scalar
            .execute_operation(&Operation::MatrixMatrix, &[&left, &right], pool)
            .expect("scalar matrix benchmark must execute")
    });
    println!(
        "benchmark=matmul-128 backend=scalar threads=1 elapsed_ns={} gflops={:.6}",
        scalar_elapsed.as_nanos(),
        matmul_gflops(SIZE, scalar_elapsed)
    );
    for workers in [1, 2, 4] {
        let backend = Arm64Backend::with_config(
            Arm64ExecutionConfig::new(workers, workers == 1)
                .expect("benchmark worker policy must validate"),
        )
        .expect("ARM64 benchmark backend must construct");
        report_parity(
            &format!("matmul-128-threads-{workers}"),
            &scalar_result,
            &backend.execute_operation(&Operation::MatrixMatrix, &[&left, &right], pool).unwrap(),
        );
        let elapsed = measure(|| {
            backend
                .execute_operation(&Operation::MatrixMatrix, &[&left, &right], pool)
                .expect("ARM64 matrix benchmark must execute")
        });
        println!(
            "benchmark=matmul-128 backend=arm64 threads={workers} elapsed_ns={} gflops={:.6}",
            elapsed.as_nanos(),
            matmul_gflops(SIZE, elapsed)
        );
    }
}

#[cfg(target_arch = "aarch64")]
fn bench_operation(
    pool: &MemoryPool,
    scalar: &CpuReferenceBackend,
    arm64: &Arm64Backend,
    operation: &Operation,
    rows: u64,
    width: u64,
) {
    let input = tensor(pool, [rows, width], rows as usize * width as usize);
    let weight = tensor(pool, [width], width as usize);
    let up = tensor(pool, [rows, width], rows as usize * width as usize);
    let inputs: Vec<&Tensor> = match operation {
        Operation::RmsNorm { .. } => vec![&input, &weight],
        Operation::SwiGlu => vec![&input, &up],
        _ => vec![&input],
    };
    let name = match operation {
        Operation::RmsNorm { .. } => "rmsnorm-32x256",
        Operation::Softmax => "softmax-32x256",
        Operation::Silu => "silu-32x256",
        Operation::SwiGlu => "swiglu-32x256",
        Operation::Rope { .. } => "rope-32x256",
        _ => "unknown",
    };
    report_parity(
        name,
        &scalar.execute_operation(operation, &inputs, pool).unwrap(),
        &arm64.execute_operation(operation, &inputs, pool).unwrap(),
    );
    for (label, backend) in [("scalar", scalar as &dyn Backend), ("arm64", arm64 as &dyn Backend)] {
        let elapsed = measure(|| {
            backend
                .execute_operation(operation, &inputs, pool)
                .expect("operation benchmark must execute")
        });
        println!(
            "benchmark={name} backend={label} elapsed_ns={} latency_us_per_iteration={:.3}",
            elapsed.as_nanos(),
            elapsed.as_secs_f64() * 1.0e6 / f64::from(MEASURED_ITERATIONS)
        );
    }
}

#[cfg(target_arch = "aarch64")]
fn bench_memory_bandwidth(pool: &MemoryPool, scalar: &CpuReferenceBackend, arm64: &Arm64Backend) {
    const ELEMENTS: usize = 1 << 20;
    let left = tensor(pool, [ELEMENTS as u64], ELEMENTS);
    let right = tensor(pool, [ELEMENTS as u64], ELEMENTS);
    for (label, backend) in [("scalar", scalar as &dyn Backend), ("arm64", arm64 as &dyn Backend)] {
        let elapsed = measure(|| {
            backend
                .execute_operation(&Operation::Add, &[&left, &right], pool)
                .expect("bandwidth benchmark must execute")
        });
        let bytes = 3.0 * ELEMENTS as f64 * 4.0 * f64::from(MEASURED_ITERATIONS);
        println!(
            "benchmark=vector-add-1m backend={label} elapsed_ns={} effective_gib_per_second={:.6}",
            elapsed.as_nanos(),
            bytes / elapsed.as_secs_f64() / (1024.0 * 1024.0 * 1024.0)
        );
    }
    for (label, backend) in [("scalar", scalar as &dyn Backend), ("arm64", arm64 as &dyn Backend)] {
        let elapsed = measure(|| {
            backend
                .execute_operation(&Operation::Multiply, &[&left, &right], pool)
                .expect("multiply benchmark must execute")
        });
        println!(
            "benchmark=vector-multiply-1m backend={label} elapsed_ns={} latency_us_per_iteration={:.3}",
            elapsed.as_nanos(),
            elapsed.as_secs_f64() * 1.0e6 / f64::from(MEASURED_ITERATIONS)
        );
    }
}

#[cfg(target_arch = "aarch64")]
fn bench_attention(pool: &MemoryPool, scalar: &CpuReferenceBackend, arm64: &Arm64Backend) {
    let query = tensor(pool, [8, 64], 8 * 64);
    let key = tensor(pool, [16, 2, 64], 16 * 2 * 64);
    let value = tensor_offset(pool, [16, 2, 64], 16 * 2 * 64, 9);
    let operation = Operation::GroupedQueryAttention { causal_position: 15 };
    let inputs = [&query, &key, &value];
    report_parity(
        "attention-8x16x64",
        &scalar.execute_operation(&operation, &inputs, pool).unwrap(),
        &arm64.execute_operation(&operation, &inputs, pool).unwrap(),
    );
    for (label, backend) in [("scalar", scalar as &dyn Backend), ("arm64", arm64 as &dyn Backend)] {
        let elapsed = measure(|| backend.execute_operation(&operation, &inputs, pool).unwrap());
        println!(
            "benchmark=attention-8x16x64 backend={label} elapsed_ns={} latency_us_per_iteration={:.3}",
            elapsed.as_nanos(),
            elapsed.as_secs_f64() * 1.0e6 / f64::from(MEASURED_ITERATIONS)
        );
    }
}

#[cfg(target_arch = "aarch64")]
fn bench_half_conversion(pool: &MemoryPool, scalar: &CpuReferenceBackend, arm64: &Arm64Backend) {
    let patterns = (0_u16..8192).collect::<Vec<_>>();
    let f16 = Tensor::from_f16_bits(TensorShape::new([8192]).unwrap(), &patterns, pool).unwrap();
    let bf16 = Tensor::from_bf16_bits(TensorShape::new([8192]).unwrap(), &patterns, pool).unwrap();
    let operation = Operation::Convert(DType::F32);
    for (dtype, input) in [("f16", &f16), ("bf16", &bf16)] {
        for (label, backend) in
            [("scalar", scalar as &dyn Backend), ("arm64", arm64 as &dyn Backend)]
        {
            let elapsed =
                measure(|| backend.execute_operation(&operation, &[input], pool).unwrap());
            println!(
                "benchmark={dtype}-to-f32-8192 backend={label} elapsed_ns={} latency_us_per_iteration={:.3}",
                elapsed.as_nanos(),
                elapsed.as_secs_f64() * 1.0e6 / f64::from(MEASURED_ITERATIONS)
            );
        }
    }
}

#[cfg(target_arch = "aarch64")]
fn bench_quantized(pool: &MemoryPool, arm64: &Arm64Backend) {
    let shape = TensorShape::new([128, 128]).unwrap();
    let values = (0_u32..128 * 128)
        .map(|index| f32::from(u8::try_from(index % 127).unwrap()) / 127.0 - 0.5)
        .collect::<Vec<_>>();
    let vector = tensor(pool, [128], 128);
    for format in [Quantization::AipQ8V1, Quantization::AipQ4V1] {
        let payload = quantize(format, &shape, &values).unwrap();
        let quantized = QuantizedTensor::parse(format, &shape, payload.bytes()).unwrap();
        let scalar_elapsed = measure(|| quantized.matrix_vector(&vector, pool).unwrap());
        let arm_elapsed = measure(|| {
            arm64
                .quantized_matrix_vector(quantized, &vector, pool, &CancellationToken::new())
                .unwrap()
        });
        println!(
            "benchmark=quantized-matvec-128 format={format:?} backend=scalar elapsed_ns={} latency_us_per_iteration={:.3}",
            scalar_elapsed.as_nanos(),
            scalar_elapsed.as_secs_f64() * 1.0e6 / f64::from(MEASURED_ITERATIONS)
        );
        println!(
            "benchmark=quantized-matvec-128 format={format:?} backend=arm64 elapsed_ns={} latency_us_per_iteration={:.3}",
            arm_elapsed.as_nanos(),
            arm_elapsed.as_secs_f64() * 1.0e6 / f64::from(MEASURED_ITERATIONS)
        );
    }
}

#[cfg(target_arch = "aarch64")]
fn bench_tiny_transformer_graph(
    pool: &MemoryPool,
    scalar: &CpuReferenceBackend,
    arm64: &Arm64Backend,
) {
    const WIDTH: u64 = 256;
    let value = |id| ValueId::new(id).expect("value ID must validate");
    let graph = InferenceGraph::new(
        [
            (value(1), TensorSpec::new(DType::F32, TensorShape::new([WIDTH]).unwrap()).unwrap()),
            (value(2), TensorSpec::new(DType::F32, TensorShape::new([WIDTH]).unwrap()).unwrap()),
            (
                value(3),
                TensorSpec::new(DType::F32, TensorShape::new([WIDTH, WIDTH]).unwrap()).unwrap(),
            ),
        ],
        vec![
            GraphNode::new(
                GraphNodeId::new(1).unwrap(),
                Operation::RmsNorm { epsilon: 1.0e-6 },
                [value(1), value(2)],
                value(4),
            )
            .unwrap(),
            GraphNode::new(
                GraphNodeId::new(2).unwrap(),
                Operation::MatrixVector,
                [value(3), value(4)],
                value(5),
            )
            .unwrap(),
            GraphNode::new(GraphNodeId::new(3).unwrap(), Operation::Silu, [value(5)], value(6))
                .unwrap(),
        ],
        [value(6)],
    )
    .unwrap();
    bench_graph_backend("scalar", scalar, &graph, pool, WIDTH);
    bench_graph_backend("arm64", arm64, &graph, pool, WIDTH);
}

#[cfg(target_arch = "aarch64")]
fn bench_graph_backend(
    label: &str,
    backend: &impl Backend,
    graph: &InferenceGraph,
    pool: &MemoryPool,
    width: u64,
) {
    let prefill = execute_graph_tokens(backend, graph, pool, width, 0, 16);
    let decode = execute_graph_tokens(backend, graph, pool, width, 16, 32);
    let executions = 48;
    let elapsed = prefill + decode;
    println!(
        "benchmark=tiny-transformer-block backend={label} prefill_tokens=16 prefill_elapsed_ns={} decode_tokens=32 decode_elapsed_ns={} total_elapsed_ns={} tokens_per_second={:.6}",
        prefill.as_nanos(),
        decode.as_nanos(),
        elapsed.as_nanos(),
        f64::from(executions) / elapsed.as_secs_f64()
    );
}

#[cfg(target_arch = "aarch64")]
fn execute_graph_tokens(
    backend: &impl Backend,
    graph: &InferenceGraph,
    pool: &MemoryPool,
    width: u64,
    first_token: u32,
    token_count: u32,
) -> Duration {
    let value = |id| ValueId::new(id).expect("value ID must validate");
    let started = Instant::now();
    for token in first_token..first_token + token_count {
        let hidden = tensor_offset(pool, [width], width as usize, token);
        let norm = Tensor::from_f32_values(
            TensorShape::new([width]).unwrap(),
            &vec![1.0; width as usize],
            pool,
        )
        .unwrap();
        let matrix = tensor(pool, [width, width], width as usize * width as usize);
        black_box(
            graph
                .execute(
                    backend,
                    BTreeMap::from([(value(1), hidden), (value(2), norm), (value(3), matrix)]),
                    pool,
                )
                .expect("tiny graph must execute"),
        );
    }
    started.elapsed()
}

#[cfg(target_arch = "aarch64")]
fn tensor(pool: &MemoryPool, shape: impl IntoIterator<Item = u64>, length: usize) -> Tensor {
    tensor_offset(pool, shape, length, 0)
}

#[cfg(target_arch = "aarch64")]
fn tensor_offset(
    pool: &MemoryPool,
    shape: impl IntoIterator<Item = u64>,
    length: usize,
    offset: u32,
) -> Tensor {
    let values = (0..length)
        .map(|index| ((index as u32 + offset) % 127) as f32 / 127.0 - 0.5)
        .collect::<Vec<_>>();
    Tensor::from_f32_values(TensorShape::new(shape).unwrap(), &values, pool).unwrap()
}

#[cfg(target_arch = "aarch64")]
fn measure(mut operation: impl FnMut() -> Tensor) -> Duration {
    for _ in 0..WARMUP_ITERATIONS {
        black_box(operation());
    }
    let started = Instant::now();
    for _ in 0..MEASURED_ITERATIONS {
        black_box(operation());
    }
    started.elapsed()
}

#[cfg(target_arch = "aarch64")]
fn matmul_gflops(size: u64, elapsed: Duration) -> f64 {
    let operations = 2.0 * size as f64 * size as f64 * size as f64 * f64::from(MEASURED_ITERATIONS);
    operations / elapsed.as_secs_f64() / 1.0e9
}

#[cfg(target_arch = "aarch64")]
fn report_parity(name: &str, expected: &Tensor, actual: &Tensor) {
    let mut maximum_absolute = 0.0_f64;
    let mut maximum_relative = 0.0_f64;
    for index in 0..usize::try_from(expected.shape().element_count()).unwrap() {
        let expected = f64::from(expected.f32_value(index).unwrap());
        let actual = f64::from(actual.f32_value(index).unwrap());
        let absolute = (expected - actual).abs();
        maximum_absolute = maximum_absolute.max(absolute);
        if expected != 0.0 {
            maximum_relative = maximum_relative.max(absolute / expected.abs());
        }
    }
    println!(
        "parity={name} maximum_absolute_error={maximum_absolute:.12e} maximum_relative_error={maximum_relative:.12e}"
    );
}

#[cfg(not(target_arch = "aarch64"))]
fn main() {
    println!("ARM64 benchmark skipped: target is not aarch64");
}
