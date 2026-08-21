# CAG Benchmark Evidence

## Evidence policy

Performance is not inferred from implementation or synthetic correctness tests. Reported cache
benefit requires the exact hardware, OS, Rust version, release profile, signed model/package hash,
backend, K/V dtype, static-prefix length, continuation length, output length, memory state, and
sample count.

Required comparisons are no cache, exact-prefix cache, and shorter exact-prefix cache over multiple
context lengths. Measure prefill time, time to first token, total generation latency, tokens
recomputed, accounted cache bytes, and variance. CPU and GPU tiers must be reported separately.

## Current evidence

Portable tests prove deterministic identity, bounded storage, longest exact-prefix selection,
request-local fork isolation, and scalar cached/full-prefill logit and generated-token equivalence
within the existing `1e-6` scalar tolerance. They do not prove production speedup.

No production model or hardware benchmark is recorded by this change. CPU-RAM cache metrics record
measured prefill and time-to-first-token nanoseconds for cached requests, plus tokens/bytes reused
and recomputed. The time-saved field remains zero because no paired measured baseline is available.
GPU cache benchmarking is unsupported because GPU reusable-cache migration is not implemented.

One synthetic planner/store smoke run on the development host (`arm64`, macOS 26.6.2, Rust
1.85.0, optimized bench profile) performed 10,000 exact-prefix lookups over a 256-token synthetic
entry in 1,916,417 ns total. This is a single non-statistical run, measures lookup machinery only,
and is not model latency, throughput, or production performance evidence.

## External qualification command shape

Use the repository release benchmark harness on the target host after a signed model is qualified.
Retain raw output and an evidence manifest; do not copy numbers between machines or builds. A pass
requires numerical parity first, then a predeclared latency/memory threshold. Regression rollback is
the prior signed engine release and disabling CAG in Go routing policy.
