//! Bounded local Metal graph microbenchmark with explicit methodology output.

use personalai_engine_cpu_reference::{Backend, CpuReferenceBackend, Operation};
use personalai_engine_memory::MemoryPool;
use personalai_engine_metal::MetalBackend;
use personalai_engine_tensor::{MutableTensor, TensorShape};
use std::time::Instant;

fn main() {
    let Ok(backend) = MetalBackend::new(0, 64 << 20, 16) else {
        println!("Metal benchmark unavailable on this host");
        return;
    };
    let pool = MemoryPool::new(64 << 20).expect("benchmark pool");
    let scalar = CpuReferenceBackend::new().expect("scalar backend");
    let matrix = MutableTensor::from_f32_values(
        TensorShape::new([128, 128]).expect("shape"),
        &vec![0.01; 128 * 128],
        &pool,
    )
    .expect("matrix");
    let vector = MutableTensor::from_f32_values(
        TensorShape::new([128]).expect("shape"),
        &vec![0.5; 128],
        &pool,
    )
    .expect("vector");
    for _ in 0..3 {
        backend
            .execute_operation(&Operation::MatrixVector, &[&matrix, &vector], &pool)
            .expect("warmup");
    }
    let metal_started = Instant::now();
    for _ in 0..25 {
        backend
            .execute_operation(&Operation::MatrixVector, &[&matrix, &vector], &pool)
            .expect("dispatch");
    }
    let metal_elapsed = metal_started.elapsed();
    let scalar_started = Instant::now();
    for _ in 0..25 {
        scalar
            .execute_operation(&Operation::MatrixVector, &[&matrix, &vector], &pool)
            .expect("scalar dispatch");
    }
    let scalar_elapsed = scalar_started.elapsed();
    let statistics = backend.statistics().expect("Metal statistics");
    println!(
        "metal-f32-matvec-128 rows=128 columns=128 iterations=25 warmup=3 metal_ns={} scalar_ns={} h2d_bytes={} d2h_bytes={} dispatches={} peak_device_bytes={}",
        metal_elapsed.as_nanos(),
        scalar_elapsed.as_nanos(),
        statistics.host_to_device_bytes,
        statistics.device_to_host_bytes,
        statistics.dispatch_count,
        statistics.peak_allocated_bytes,
    );
}
