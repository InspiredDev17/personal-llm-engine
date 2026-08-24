# ADR-0023: Isolated ARM64 graph kernels

Status: accepted and integrated as an owner-selected transformer-service
backend; production model qualification remains external.

## Decision

The `engine-arm64` crate is a first-party, default-off host graph backend. On
an AArch64 target it runtime-detects the mandatory Advanced SIMD facility and
uses fixed intrinsics for contiguous F32 add, multiply, scalar multiply, dot,
matrix-vector, matrix-matrix, RMSNorm application, softmax scaling, SiLU and
SwiGLU multiplication, RoPE pairs, attention scores/value accumulation,
F16/BF16 expansion, and AIPQ8/AIPQ4 dequantization and matrix-vector work. The
closed immutable graph interface delegates unsupported dtype, layout, size, or
operation cases to the scalar correctness backend. `HostCpuBackend::select`
also falls back to the scalar backend when the host is not AArch64 with
Advanced SIMD. No model or client selects an intrinsic, operation
implementation, pointer, offset, thread count, or kernel name.

The safe facade validates dtype, shape, contiguity, pool ownership, finite
inputs and outputs, checked work counts, and exact destination sizes. Temporary
F32/F64/U16/atomic staging allocations are charged to the caller's aggregate
pool under the ARM64 backend owner and released deterministically. Intrinsics
accept unaligned Rust slices and never issue a vector load unless a complete
lane group is in bounds. The operation ceiling remains the CPU oracle's
100-million-work bound.

Matrix multiplication uses a two-row by four-column logical tile. It vectorizes
four contiguous output columns and handles remaining columns scalarly. Shared
dimensions need not be SIMD multiples. Dimensions below the published dispatch
threshold use the scalar backend because staging overhead dominates.

`Arm64ExecutionConfig` fixes one to sixteen workers. Deterministic mode requires
exactly one worker. Multi-worker matrix multiplication partitions disjoint row
tiles, uses a bounded channel with capacity equal to the worker count, checks a
cooperative cancellation token between tiles, catches worker panics, waits for
every submitted job, and joins every persistent worker during pool drop. Safe
atomic output slots avoid shared mutable pointers. Metrics report fixed workers,
live/peak jobs, completions, cancellations, and isolated panics.

All architecture-specific pointer operations are confined to
`engine-arm64/src/intrinsics.rs`. Its thirteen reviewed unsafe blocks process
only complete 4-lane F32, canonical little-endian F32-byte, 4-lane U16,
8-lane I8, or fixed 32-code blocks proven in bounds by safe slice or array
lengths; scalar tails never use pointers.
Inputs need not be 16-byte aligned because the documented AArch64 load/store
intrinsics permit unaligned addresses. AArch64 defines Advanced SIMD as a
baseline facility, but the backend still verifies it at runtime. Optional FP16
arithmetic is reported but not required: F16 normal encodings and BF16 values
use integer-lane expansion, while F16 zero/subnormal/infinity/NaN encodings use
the exhaustive first-party scalar converter.

## Numerical policy

- F32 is the only arithmetic dtype; no silent precision change is permitted.
- Elementwise operations retain F32 semantics. SIMD dot reductions may reorder
  additions, so matvec/attention use absolute and relative differential limits.
- RMSNorm keeps the scalar oracle's robust F64 scale computation, then applies
  the finite F32 scale with SIMD.
- Softmax and SiLU keep scalar F64 exponentials; only bounded final vector work
  is accelerated. RoPE keeps F64 angle/trigonometric computation.
- F16 conversion is tested over all 65,536 bit patterns. NaN payload class and
  sign are preserved under the same policy as the tensor runtime.
- AIPQ code validation remains in `engine-quantization`; the ARM64 backend only
  expands validated signed codes and then uses reviewed F32 matrix kernels.

Differential tests use `3e-6` absolute/relative limits for reductions and
attention and `2e-6` for normalization/activation/rotation. The release
measurement below observed a larger `3.433227539062e-5` absolute matvec delta
on a 512-wide reduction, with relative error `1.275940793679e-5`; this is
recorded rather than hidden and the larger production-shape tolerance remains
an activation issue for real-model evaluation.

## Evidence and limits

Apple M1 tests compare every accelerated class to the scalar oracle, exhaust
F16 encodings, cover unaligned slices, vector/scalar tails, irregular shapes,
single/multi-worker equivalence, cancellation, panic isolation, closed-operation
fallback, quantized paths, non-finite values, and foreign pools. The exact
unsafe file, block count, and dependency DAG are architecture-tested.

## Apple M1 release measurement (2026-08-13)

Method: Apple M1 Mac mini, 8 cores (4 performance/4 efficiency), 16 GiB,
macOS 26.5.2 (25F84), Rust 1.97.1/LLVM 22.1.6, optimized Cargo bench build,
deterministic synthetic F32 fixture, three warmups and twenty measured kernel
iterations. Results are single-run local evidence, not a sustained or
cross-hardware claim.

| Workload | Scalar | ARM64 | Observed speedup |
| --- | ---: | ---: | ---: |
| matvec/dot 512x512 | 0.153 GFLOP/s | 0.278 GFLOP/s | 1.82x |
| matmul 128, 1 worker | 0.156 GFLOP/s | 2.67 GFLOP/s | 17.1x |
| matmul 128, 2 workers | 0.156 GFLOP/s | 2.80 GFLOP/s | 17.9x |
| matmul 128, 4 workers | 0.156 GFLOP/s | 3.23 GFLOP/s | 20.7x |
| RMSNorm 32x256 | 580.4 us | 537.2 us | 1.08x |
| softmax 32x256 | 1130.9 us | 503.0 us | 2.25x |
| SiLU 32x256 | 590.5 us | 540.8 us | 1.09x |
| SwiGLU 32x256 | 657.4 us | 576.9 us | 1.14x |
| RoPE 32x256 | 678.6 us | 606.5 us | 1.12x |
| vector add, 1M | 0.162 GiB/s | 0.167 GiB/s | 1.03x |
| vector multiply, 1M | 73,168 us | 70,306 us | 1.04x |
| GQA scores + values, 8x16x64 | 213.9 us | 71.0 us | 3.01x |
| F16 to F32, 8,192 | 495.2 us | 457.1 us | 1.08x |
| BF16 to F32, 8,192 | 519.8 us | 432.1 us | 1.20x |
| AIPQ8 matvec 128 | 138.0 us | 21.5 us | 6.42x |
| AIPQ4 matvec 128 | 149.1 us | 34.0 us | 4.38x |
| tiny block graph proxy | 235.4 tokens/s | 253.9 tokens/s | 1.08x |

Peak accounted host memory was 25,165,824 bytes and returned to zero. The tiny
block graph is an allocation-inclusive RMSNorm -> matvec -> SiLU proxy with a
16-step prefill (68.3 ms scalar, 63.3 ms ARM64) and 32-step decode (135.6 ms
scalar, 125.7 ms ARM64); it is not a signed full-model throughput claim.
Staging, scalar exponentials/trigonometry, and tensor materialization remain
optimization targets.

The signed decoder now routes linear projections, RMSNorm, RoPE, residual add,
SwiGLU, and grouped-query attention through the closed ARM64 backend interface.
It materializes bounded cache tensors for attention and retains the scalar
fallback already enforced by that interface. A full transformer differential
test covers cached prefill/decode with an explicit `2e-4` absolute plus `2e-4`
relative acceptance bound, while the authenticated UDS service test
loads a signed package, advertises ARM64, streams generation, and reports an
ARM64 descriptor. Selection is owner-only; neither package nor request can
choose the backend. Direct immutable weight views without staging, production-
size differential generation, longer cancellation/soak evidence, and real-
model quality/performance evaluation remain external acceptance work.
