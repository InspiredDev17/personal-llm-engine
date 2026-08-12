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
| 7. Scheduler | COMPLETE for bounded reference runtime | Count/byte admission, FIFO lifecycle, queued cancellation, resumable one-token cursors, round-robin coordination, exact-batch and continuous multi-stream execution, and bounded output isolation |
| 8. UDS/Go adapter | COMPLETE for local reference path | Separate Ed25519 request/response identities, bounded two-key verification overlap, signed ordered events, cancellation/disconnect watchers, durable replay rejection, single-stream CLI plus bounded batch/continuous reference drivers, Go `NATIVE_ENGINE` adapter and real socket tests |
| 9. Production package operations | PLANNED | Rust-only converter, offline signer workflow, approved model and provenance |
| 10. ARM64/Metal | HARDWARE GATED | Fixed kernels, differential and device tests required |
| 11. CUDA | HARDWARE GATED | Fixed kernels, differential and device tests required |
| 12. Production cutover | PARTIAL | Mutual response trust, bounded rotation mechanics, exclusively owned durable request replay, deterministic mutation/transport faults, scheduled fuzzing, signal-drained continuous CLI serving, bounded delayed-reader/restart soak, and its pinned scheduled-workflow definition are complete; hosted cron evidence awaits default-branch integration, while production-size operated soak/performance, key/journal recovery and reversible live canary remain |

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
- Two-key current/previous verification windows support staged control and
  response identity cutover; each accepted control key retains its own locked
  journal and command streams cannot switch identities.
- Exhaustive canonical-fixture and signed-event truncation/single-byte mutation suites, signed
  active cancellation, a checked-in vendor graph, and lockfile-bound SPDX.
- Isolated coverage-guided Rust fuzz targets for package, tokenizer, and frame
  parsers plus bounded Go fuzz targets for unsigned and authenticated response
  events, with deterministic Rust-generated seeds and scheduled CI campaigns.
- Request-local greedy generation cursors yield after at most one decoder token
  step. A bounded pull-based coordinator enforces active/request concurrency
  limits and unique active identities before cursor construction, plus fallible
  queue growth, conservative aggregate active-retention charging, one-turn
  rotation, and request-local failure. The UDS batch adds fixed eight-event
  queues and bounded writer threads; their worst-case retention is charged
  before cursor creation, and saturation cancels only the affected request.
  Deterministic tests prove independent KV state, non-starvation, and
  cancellation isolation. The UDS batch and continuous reference paths now
  prove authenticated multi-stream execution, bounded ownership,
  incomplete-handshake interruption, caller-owned graceful drain, and cleanup
  on every exit. The owner CLI selects that path only through explicit bounded
  limits; isolated `SIGINT`/`SIGTERM` handling drives the same tested graceful
  drain. A non-ignored test runs eight six-client restart cycles, holds one
  reader until five authenticated neighbors finish, validates all terminal
  signatures/order, drains, and proves cleanup. The exact configured CI workload
  of eight repetitions passed locally; GitHub's weekly cron begins only after
  default-branch integration. Production-size operated soak remains required.

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

- retain scheduled coverage-guided fuzzing, triage every reproducible finding,
  and grow production-size corpora without placing production model data in CI;
- reproduce cancellation, signed ordering, cleanup, and round-robin fairness in
  production-size operated multi-client/slow-reader soak of the signal-drained
  continuous CLI; the bounded synthetic scheduled gate is already mandatory;
- qualify memory, latency, throughput, thermal behavior, and long-context use;
- reproduce the checked-in vendor graph and SPDX from a fresh offline build and
  renew license/advisory review;
- ship a target-specific service definition that denies Rust outbound
  networking, then prove CPU and accelerated inference over the protected UDS
  while that denial is active;
- require Go's platform-wide local-only mode to reject every remote primary,
  fallback, verifier, and manual override before sovereign cutover;
- complete Agent Green telemetry mapping in Go, never in the model or engine;
- canary the provider behind Go routing and preserve instant rollback;
- retire third-party runtimes only after equivalent production evidence.

Hardware is not required for Stage 9 format/conversion/fuzz work. It is required
for accepting Stages 10 and 11 and for final performance qualification.
