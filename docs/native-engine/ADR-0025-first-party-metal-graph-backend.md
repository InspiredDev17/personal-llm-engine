# ADR-0025: First-party Metal graph backend

Status: accepted for the immutable graph; production decoder cutover pending.

## Decision

Use a new `personalai-engine-metal` crate as the only safe Rust owner of Apple
GPU execution. A single reviewed build script compiles repository-owned MSL
with the installed Apple Metal toolchain, links a repository-owned minimal
Objective-C++ bridge, hashes the metallib with the existing approved SHA-384
implementation, and embeds both bytes and expected digest. Runtime code can
select only a closed Rust/C enum; it accepts neither shader bytes, names,
libraries, paths, nor source from models or clients.

The bridge exposes opaque explicitly destroyed contexts and buffers. All Rust
pointer operations are confined to `engine-metal/src/ffi.rs`. The bridge owns
Objective-C objects, catches Objective-C exceptions, disables C++ exceptions,
checks nulls, lengths, ranges, alignment, ownership, kernel buffer counts, and
dispatch grids, and returns status codes plus bounded local diagnostic text.

Device allocations are charged to an engine `MemoryPool`. Idle buffers are
bounded by count, cleared before reuse, and reclaimable. Host/device transfers,
allocations, dispatches, failures, and pool residency are measured without
content. Metal KV buffers use exact model and sequence ownership, generation
checked handles, range-checked indexing, context limits, and clearing on every
invalidating lifecycle transition.

## Kernel and numerical policy

Version one contains 19 unfused kernels: copy, add, multiply, scalar multiply,
F32/F16 matvec and matmul, RMSNorm, softmax, SiLU, SwiGLU, RoPE, embedding,
attention scores, causal mask, attention values, AIPQ8 matvec, and AIPQ4
matvec. F32 reductions use deterministic per-output serial accumulation on the
GPU. Differential tolerance is `5e-5` generally and `1e-4` for composed
attention. Unsupported operations, dtypes, layouts, or shapes fall back to the
scalar oracle; fallback never changes model or cache state.

## Supply-chain exception

`engine-metal/build.rs` is the only workspace build script. It uses no Cargo
native-build dependency, network client, shell interpreter, downloaded source,
or runtime discovery. It reads fixed repository paths and the installed Apple
Metal component and writes only Cargo `OUT_DIR`. Architecture tests enforce the
one-file allowlist, exact internal dependency graph, and isolated unsafe file.

## Evidence and remaining gate

On the Apple M1 development host, hardware tests cover every public kernel
class, scalar and ARM64 differential comparison, irregular/tail dimensions,
F16, AIPQ8/AIPQ4, cancellation, allocation pressure, reuse clearing, fallback,
and Metal KV isolation/reset/cancel/destroy/unload. These tests skip clearly
when the process sandbox exposes no Metal device; a separate direct-hardware
run is required for release evidence.

This ADR does not mark Metal model execution complete. The production signed
decoder and local service still select the CPU implementation. Activation
requires a signed tiny AIPack to tokenize, prefill, decode, sample, and stream
through Metal with matching CPU output, followed by service-level cancellation
and model-unload tests and release-mode latency measurements.
