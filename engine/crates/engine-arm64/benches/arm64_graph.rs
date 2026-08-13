//! Manual ARM64 graph-kernel measurements; not production qualification.

#[cfg(target_arch = "aarch64")]
use personalai_engine_arm64::Arm64Backend;
#[cfg(target_arch = "aarch64")]
use personalai_engine_cpu_reference::{Backend, CpuReferenceBackend, Operation};
#[cfg(target_arch = "aarch64")]
use personalai_engine_memory::MemoryPool;
#[cfg(target_arch = "aarch64")]
use personalai_engine_tensor::{Tensor, TensorShape};
#[cfg(target_arch = "aarch64")]
use std::hint::black_box;
#[cfg(target_arch = "aarch64")]
use std::time::Instant;

#[cfg(target_arch = "aarch64")]
fn main() {
    const ROWS: u64 = 256;
    const COLUMNS: u64 = 256;
    const ITERATIONS: u32 = 20;

    let pool = MemoryPool::new(32 << 20).expect("benchmark pool must construct");
    let matrix_values =
        (0_u16..=u16::MAX).map(|value| f32::from(value % 127) / 127.0 - 0.5).collect::<Vec<_>>();
    let vector_values =
        (0_u16..256).map(|value| f32::from(value % 31) / 31.0 - 0.5).collect::<Vec<_>>();
    let matrix = Tensor::from_f32_values(
        TensorShape::new([ROWS, COLUMNS]).expect("shape must validate"),
        &matrix_values,
        &pool,
    )
    .expect("matrix must construct");
    let vector = Tensor::from_f32_values(
        TensorShape::new([COLUMNS]).expect("shape must validate"),
        &vector_values,
        &pool,
    )
    .expect("vector must construct");
    let arm64 = Arm64Backend::new().expect("ARM64 backend must construct");
    let scalar = CpuReferenceBackend::new().expect("scalar backend must construct");

    for (label, backend) in [("arm64", &arm64 as &dyn Backend), ("scalar", &scalar as &dyn Backend)]
    {
        let started = Instant::now();
        for _ in 0..ITERATIONS {
            black_box(
                backend
                    .execute_operation(&Operation::MatrixVector, &[&matrix, &vector], &pool)
                    .expect("matrix-vector must execute"),
            );
        }
        println!(
            "{label} matrix-vector-256x256: {ITERATIONS} iterations in {} microseconds",
            started.elapsed().as_micros()
        );
    }
}

#[cfg(not(target_arch = "aarch64"))]
fn main() {
    println!("ARM64 benchmark skipped: target is not aarch64");
}
