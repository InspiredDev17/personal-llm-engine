# Apple Metal backend boundary

Status: **SIGNED SYNTHETIC DECODER IMPLEMENTED / UDS-GO CUTOVER PENDING**.

The `personalai-engine-metal` crate owns the safe Rust integration. This
directory owns one minimal Objective-C++ bridge and first-party MSL kernels.
The build compiles `shaders/all.metal` into an embedded metallib, records its
SHA-384 digest, and refuses an identity mismatch before device creation. There
is no runtime source compilation, shader download, dynamic-library discovery,
model kernel selector, or client-provided kernel name.

The C ABI exposes opaque contexts and buffers with explicit create/destroy,
whole and bounded-range transfers, synchronization, a fixed 19-value kernel
enum, capability discovery, and content-free resource statistics. Objective-C
objects and C++ exceptions never cross the boundary. Rust unsafe code is
isolated to `engine-metal/src/ffi.rs`; the Objective-C++ bridge catches
Objective-C exceptions and compiles with C++ exceptions disabled.

Implemented kernels are copy, F32 add/multiply/scalar multiply, F32 and F16
matvec/matmul, RMSNorm, softmax, SiLU, SwiGLU, RoPE, embedding lookup,
attention scores, causal mask, attention value accumulation, AIPQ8 matvec, and
AIPQ4 matvec. Unsupported validated graph operations, layouts, and data types
fall back to the scalar oracle. Transfers and dispatches are explicit metrics.

Metal KV storage is contiguous, model-owned, sequence-isolated,
generation-checked, context-bounded, and cleared on allocation, reset,
cancellation, destruction, pooling, or model unload. This implementation is a
correctness-first device cache; paged KV and fused attention remain future
performance work.

The generic immutable graph and fixed decoder run on Metal, and hardware
differential tests pass on the recorded Apple M1 host. Validated weights are
uploaded once into model-scoped buffers; signed synthetic packages complete
prefill, cached decode, sampling, and typed streaming. Metal model execution is
advertised only by direct trusted composition. The public UDS daemon and Go
adapter remain CPU-selected pending canary and rollback configuration.
