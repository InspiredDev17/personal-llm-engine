# ADR-0024: First-Party AIPQ Block Quantization

## Status

Accepted for the scalar CPU correctness path. AIPQ8 v1 was gated before AIPQ4
v1 acceptance. Production-performance backend integration remains separate.

## Decision

Use first-party, fixed 32-value symmetric blocks with explicit little-endian
F32 scales. Quantized tensors have self-describing fixed headers and canonical
encodings. `balanced-q8` and `compact-q4` quantize only reviewed rank-two model
weights; excluded tensors stay F32. The package admits only one quantization
family plus unquantized exclusions. The importer requires an explicit profile
and signs complete source/destination hashes and error evidence.

The production loader validates every payload before aggregate-accounted F32
materialization. Separate safe scalar direct-AIPQ matrix kernels provide a
correctness target for future optimized backends. No package field selects a
kernel or operation.

## Consequences

- Format, rounding, clamping, zero, non-finite, tail, nibble, and alignment
  behavior are unambiguous and bounded.
- Package and runtime dependencies remain unchanged; all implementation is safe
  first-party Rust.
- Current service execution gains smaller signed artifacts but materializes F32
  weights, so it does not yet claim reduced steady-state host memory.
- AIPQ4 can materially alter generation; its synthetic four-of-four divergence
  remains explicit release evidence, not hidden behind averages.
- Optimized ARM64, Metal, or CUDA kernels require separate differential and
  hardware qualification and cannot be selected by model data.

