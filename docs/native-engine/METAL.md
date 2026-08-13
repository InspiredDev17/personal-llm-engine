# First-party Metal backend evidence

## Recorded host

- Hardware: Apple M1 Mac mini (`Macmini9,1`), 8 CPU cores, 16 GiB unified
  memory.
- Operating system: macOS 26.5.2 build 25F84.
- Xcode: 26.6 build 17F113.
- Metal compiler component: 17.6.109.0 build 17F109.
- Workspace compiler: Rust 1.85.0, LLVM 19.1.7,
  `aarch64-apple-darwin`.
- Build mode: Cargo release benchmark, first-party F32 128-by-128 matrix-vector
  fixture, 3 warmups and 25 measured iterations.

## Measured result

The bounded local run reported:

```text
metal-f32-matvec-128 rows=128 columns=128 iterations=25 warmup=3 metal_ns=11941958 scalar_ns=5779750 h2d_bytes=1849344 d2h_bytes=14336 dispatches=28 peak_device_bytes=66560
```

This is 477,678 ns per Metal invocation versus 231,190 ns per scalar
invocation, or approximately 0.48 times scalar throughput for this deliberately
small unfused case. Metal is therefore slower here; the result is not a GPU
speedup claim. The measurement includes explicit per-operation transfers and
synchronous command completion, which is useful correctness behavior but poor
production scheduling.

The benchmark is a microbenchmark, not prefill, decode, tokens-per-second,
time-to-first-token, thermal, or production-model evidence. Those measurements
remain blocked on signed-decoder integration, persistent device-resident
weights, command-buffer batching, and a legally approved production model.

## Correctness evidence

The direct-hardware test run passed six suites covering device creation,
metallib loading, resource metrics, F32 graph parity against scalar and ARM64,
irregular dimensions, composed grouped-query attention, F16 kernels,
AIPQ8/AIPQ4 kernels, invalid inputs, cancellation, fallback, pool clearing,
and Metal KV sequence isolation/reset/cancel/destroy/unload. The ordinary
sandbox can hide the Metal device; tests detect that condition and skip rather
than emulating a GPU. Release evidence must always include a direct-hardware
run where the six suites execute.

## Numerical policy

Elementwise exact cases require exact equality. F32 reductions use absolute
and relative tolerance `5e-5`; composed grouped-query attention uses `1e-4`.
F16 known-answer cases use exactly representable values. Quantized known-answer
cases follow the reviewed AIPQ8/AIPQ4 nibble and scale definitions. Inputs with
NaN or infinity fail closed; causal-mask negative infinity is the single
intentional output exception.
