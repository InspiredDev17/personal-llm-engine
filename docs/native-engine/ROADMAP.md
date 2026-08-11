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
| 8. UDS/Go adapter | COMPLETE for local reference path | Separate Ed25519 request/response identities, signed ordered events, active same-stream cancellation/disconnect watcher, durable per-key replay rejection, UDS runtime, Go `NATIVE_ENGINE` adapter and real socket tests |
| 9. Production package operations | PLANNED | Rust-only converter, offline signer workflow, approved model and provenance |
| 10. ARM64/Metal | HARDWARE GATED | Fixed kernels, differential and device tests required |
| 11. CUDA | HARDWARE GATED | Fixed kernels, differential and device tests required |
| 12. Production cutover | PARTIAL | Mutual response trust, exclusively owned durable request replay, fixture mutation/transport faults, vendored offline graph and SPDX complete; coverage fuzz, soak/performance, production key lifecycle and reversible live canary remain |

## Completed host-independent implementation

- Foundational types and error taxonomy with checked arithmetic and hard bounds.
- Exact, non-recursive, data-only package structures with no arbitrary
  deserialization, compression, plugins, paths, code, or kernel selection.
- Mature audited cryptographic primitives only; package/control private keys
  stay outside Rust, while the distinct response-only key is stdin-provisioned
  and zeroized.
- Secure local file admission for configuration-owned direct-child packages.
- Deterministic fixture generation entirely in Rust.
- First-party tokenizer, tensor, memory, KV, decoder, greedy sampling,
  scheduler, mutually authenticated protocol, UDS transport, daemon composition, CLI,
  and Go provider translation.
- CPU socket-to-signed-package generation exercised end to end.
- Bounded per-control-key durable nonce journal rejects cross-connection and
  post-restart replay before execution, excludes a second live process, and
  fails closed on corrupt or contended state.
- Exhaustive canonical-fixture and signed-event truncation/single-byte mutation suites, signed
  active cancellation, a checked-in vendor graph, and lockfile-bound SPDX.

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

- run continuous coverage-guided fuzzing in addition to the deterministic
  package/tokenizer/frame mutation and partial-transport fault suites;
- add fair multi-sequence scheduling (active same-stream cancellation is done);
- qualify memory, latency, throughput, thermal behavior, and long-context use;
- reproduce the checked-in vendor graph and SPDX from a fresh offline build and
  renew license/advisory review;
- complete Agent Green telemetry mapping in Go, never in the model or engine;
- canary the provider behind Go routing and preserve instant rollback;
- retire third-party runtimes only after equivalent production evidence.

Hardware is not required for Stage 9 format/conversion/fuzz work. It is required
for accepting Stages 10 and 11 and for final performance qualification.
