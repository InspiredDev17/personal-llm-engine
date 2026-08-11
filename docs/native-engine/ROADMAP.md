# Native Engine Roadmap

- Version: 0.2.0
- Updated: 2026-08-11
- Scope: inference only

## Delivery rule

Every capability stays disabled until its success, corruption, resource,
cancellation, and authority-boundary tests pass. The Go control plane adopts
the native provider additively and can disable it without changing policy,
memory, tool, approval, or Agent Green behavior.

## Current stage summary

| Stage | Status | Current evidence |
| --- | --- | --- |
| 0. Boundaries/workspace | COMPLETE | 16-crate workspace, docs, limits, feature and DAG tests |
| 1. Package/cryptography | COMPLETE for V1 | Canonical flat package, SHA-384, strict Ed25519, revocation/tamper tests |
| 2. Registry/loader | COMPLETE for V1 | ID-only registry, direct-child no-follow opener, inode/size/mode checks, bounded tensor materialization |
| 3. Tokenizer | COMPLETE for V1 | First-party `PAIBPE01`, byte fallback, bounded declarative merges, encode/decode tests |
| 4. Tensor/CPU core | COMPLETE for target architecture | Safe owned tensors, aggregate memory pool, checked scalar work |
| 5. Decoder architecture | COMPLETE for synthetic V1 | Fixed `personalai.decoder-v1`, exact tensor inventory, causal grouped attention and SwiGLU |
| 6. Generation | COMPLETE for greedy CPU | Prefill/decode, real KV tensors, streaming UTF-8 events, usage and cancellation checks |
| 7. Scheduler | COMPLETE for current single-worker runtime | Count/byte admission, FIFO lifecycle and queued cancellation |
| 8. UDS/Go adapter | COMPLETE for local reference path | Ed25519-authenticated requests, nonce checks, UDS runtime, Go `NATIVE_ENGINE` adapter and real socket tests |
| 9. Production package operations | PLANNED | Rust-only converter, offline signer workflow, approved model and provenance |
| 10. ARM64/Metal | HARDWARE GATED | Fixed kernels, differential and device tests required |
| 11. CUDA | HARDWARE GATED | Fixed kernels, differential and device tests required |
| 12. Production cutover | PLANNED | Fuzz/fault/performance/SBOM/vendoring and reversible migration |

## Completed host-independent implementation

- Foundational types and error taxonomy with checked arithmetic and hard bounds.
- Exact, non-recursive, data-only package structures with no arbitrary
  deserialization, compression, plugins, paths, code, or kernel selection.
- Mature audited cryptographic primitives only; production private keys are
  outside Rust.
- Secure local file admission for configuration-owned direct-child packages.
- Deterministic fixture generation entirely in Rust.
- First-party tokenizer, tensor, memory, KV, decoder, greedy sampling,
  scheduler, authenticated protocol, UDS transport, daemon composition, CLI,
  and Go provider translation.
- CPU socket-to-signed-package generation exercised end to end.

## Next software release gate: production model operations

Implement a Rust-only offline converter for one legally approved source model.
The converter must independently parse that source format, map only the fixed
V1 tensor inventory, validate tokenizer semantics, emit deterministic package
bytes, and never run generated/model code or access the network. Signing must
use an offline operator-controlled key and emit provenance, license, source
hash, package hash, key ID, reproducibility evidence, SBOM, and test results.

Exit evidence is a reviewed production-size package that opens through its
registered ID, passes exact signature and tensor validation, and matches
independent CPU logits/generation vectors. No useful production model is
currently included in this repository.

## Accelerator gates

ARM64, Metal, and CUDA remain default-off. Each backend requires fixed
architecture-owned kernels, isolated documented unsafe/FFI if unavoidable,
exact device and allocation bounds, cancellation, differential tests against
the CPU oracle, driver failure tests, soak tests, and hardware-specific
performance evidence. Model metadata cannot select code or kernels.

## Hardening and cutover gate

Before the native runtime becomes mandatory:

- fuzz every package/tokenizer/frame parser and run fault-injection tests;
- add concurrent active cancellation and fair multi-sequence scheduling;
- qualify memory, latency, throughput, thermal behavior, and long-context use;
- vendor release dependencies, generate an SBOM, and pass license/advisory
  review from a fresh offline build;
- complete Agent Green telemetry mapping in Go, never in the model or engine;
- canary the provider behind Go routing and preserve instant rollback;
- retire third-party runtimes only after equivalent production evidence.

Hardware is not required for Stage 9 format/conversion/fuzz work. It is required
for accepting Stages 10 and 11 and for final performance qualification.
