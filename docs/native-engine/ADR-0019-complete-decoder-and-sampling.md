# ADR-0019: Complete scalar decoder generation and first-party sampling

## Status

Accepted for the CPU correctness reference path.

## Context

The signed package loader, AIPTok tokenizer, scalar tensor operations, memory
manager, and contiguous KV cache already existed. Generation previously exposed
only greedy selection and the decoder maintained a separate private cache. That
left seeded sampling, an explicit prefill/decode contract, and end-to-end use of
the reviewed cache contract incomplete.

## Decision

The only executable architecture remains
`decoder-rms-rope-swiglu-gqa-v1`. Production construction validates its exact
tensor inventory and shapes before execution. A decoder session exposes
non-empty prefill and exactly-one-token decode, writes K/V through
`ContiguousKvCache`, and publishes a token position only after every layer is
complete. Package data cannot select operations or kernels.

Sampling is first-party and uses this fixed order: token allow/deny constraints,
repetition penalty, temperature, top-k, min-p, then top-p. Temperature zero is
exact greedy decoding with token-ID tie breaking. Stochastic selection uses a
documented xorshift64* generator for reproducibility only; it is not
cryptographic. NaN and positive infinity are errors, negative infinity is an
explicitly excluded candidate, and empty or non-finite distributions fail.
Parameters are validated rather than corrected.

The rich in-process lifecycle distinguishes acceptance, tokenization, prefill,
token output, stopping, completion, cancellation, resource limits, model
errors, and backend errors. It carries bounded identifiers, token IDs, text,
usage, and structured errors only. The existing authenticated Go-facing wire
event set is unchanged. The owner CLI can invoke the rich path only after
registered-ID resolution and signed directory-package verification.

## Evidence and limits

The deterministic signed fixture contains two layers, two query heads, one KV
head, RoPE, SwiGLU, and byte-level tokenization. A checked scalar run with seed
`0x5eed` and the recorded sampler settings produces the golden token sequence
`67, 66, 65, 257`; the expected values come from this implementation, not a
third-party runtime. Prefill and incremental decode logits agree within
`1e-6`; fixed scalar ordering normally makes them bit-identical.

This is a correctness oracle, not a performance backend. It is single-threaded,
scalar F32 execution with F32 KV storage, allocates intermediate vectors, and
has no SIMD, fusion, speculative decoding, beam search, grammar decoding,
distributed execution, ARM64 optimization, Metal, or CUDA. Real-model numeric
qualification and production performance evidence remain mandatory before
cutover.
