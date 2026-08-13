# ADR-0023: Isolated ARM64 graph kernels

Status: accepted as an accelerator foundation; transformer-service activation
remains disabled.

## Decision

The `engine-arm64` crate is a first-party, default-off host graph backend. On
an AArch64 target it uses fixed Advanced SIMD kernels for contiguous F32 add,
multiply, scalar multiply, dot product, and matrix-vector multiplication. The
closed immutable graph interface delegates every other allowlisted operation
to the scalar correctness backend. No model or client selects an intrinsic,
operation implementation, pointer, offset, or kernel name.

The safe facade validates dtype, shape, contiguity, pool ownership, finite
inputs and outputs, checked work counts, and exact destination sizes. Temporary
aligned F32 staging allocations are charged to the caller's aggregate pool
under the ARM64 backend owner and released deterministically. The operation
ceiling remains the CPU oracle's 100-million-work bound.

All architecture-specific pointer operations are confined to
`engine-arm64/src/intrinsics.rs`. Its four reviewed unsafe blocks process only
four-element chunks proven in bounds by safe slice lengths; scalar tails never
use pointers. AArch64 defines Advanced SIMD as a baseline facility. Off-target
construction returns an explicit not-implemented error.

## Evidence and limits

Apple M1 tests compare accelerated outputs to the scalar oracle, cover vector
and scalar tails, prove deterministic repeated matrix-vector results, exercise
closed-operation fallback, and reject non-finite values and foreign memory
pools. The exact unsafe file and block count are architecture-tested.

This does not yet make `arm64` a complete transformer-service backend. The
current model path still uses scalar tensor storage and decoder orchestration;
the CLI and service must continue reporting ARM64 model execution as
not-implemented. Direct immutable weight layouts, complete decoder integration,
end-to-end differential generation, cancellation/soak evidence, and measured
performance remain before activation.
