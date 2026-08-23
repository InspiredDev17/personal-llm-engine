# Native Engine Roadmap

- Version: 0.2.0
- Updated: 2026-08-16
- Scope: inference only

## Delivery rule

Every capability stays disabled until its success, corruption, resource,
cancellation, and authority-boundary tests pass. The Go control plane adopts
the native provider additively and can disable it without changing policy,
memory, tool, approval, or Agent Green behavior.

## Current stage summary

| Stage | Status | Current evidence |
| --- | --- | --- |
| 0. Boundaries/workspace | COMPLETE | 22-crate workspace, docs, limits, feature/DAG tests, and a pinned offline Linux/macOS verification definition |
| 1. Package/cryptography | COMPLETE for legacy v1.0 and experimental Qwen2.5 v1.1 mechanics | Fixed manifest/catalog/records, SHA-384, strict Ed25519, closed files, version/architecture binding and corruption tests; v1.1 is engine 0.2.0 and no production Qwen package approval exists |
| 2. Registry/loader | COMPLETE for legacy v1.0 and default-off Qwen development admission | Registry V2 stays frozen. A separate signed, anti-rollback V3 record plus canonical Models 2.0 descriptor, exact PAIDIR01 identity and owner Runtime-v3 identity admit only the ARM64 development path; CPU, Metal and production remain ineligible. |
| 3. Tokenizer | COMPLETE for executable AIPTok v1 and exact Qwen development construction | Exact pinned Qwen source converts deterministically to canonical `AIPTOK02`; a content-free offline exporter publishes only the fixed bytes to a fresh read-only path. Only the exact V3 ARM64 package-bound path may construct it; generic, CPU, Metal and production paths remain closed. |
| 4. Tensor/CPU core | COMPLETE as correctness oracle | Canonical row-major tensors, explicit checked views, F32 execution plus F16/BF16 conversion, immutable statically validated graph, complete scalar operation set, explicit-owner memory, bounded cleared scratch reuse, real CPU device-memory transfers, aggregate compute bounds, exhaustive F16 tests and benchmarks |
| 5. Decoder architecture | COMPLETE for legacy synthetic V1; Qwen2.5 tiny correctness mechanics only | Legacy fixed decoder remains unchanged; the new exact Qwen2.5 ID adds required Q/K/V biases, tied output, split-half RoPE, exact inventory and deterministic tiny tests. Internal ARM64 selection preserves split-half semantics through the scalar correctness operation and cannot route through legacy interleaved RoPE. Exact 1.5B scalar execution is rejected at 1,543,712,768 estimated operations versus the 100,000,000 ceiling; Foundry package/registry work, optimized backend qualification, and real-model numeric vectors remain open |
| 6. Generation | COMPLETE as scalar CPU reference | Explicit prefill/one-token decode, decoder-integrated contiguous KV with atomic all-layer commits, Sovereign Knowledge 1.1-bound immutable reusable CPU prefix caches with exact compatibility and parity tests, deterministic greedy and seeded temperature/top-k/top-p/min-p/repetition sampling, stop policies, rich typed lifecycle, and signed two-layer GQA golden generation. Knowledge 1.1 remains experimental/unactivated; GPU/SSD cache tiers and selective recomposition remain disabled. |
| 7. Scheduler | COMPLETE for bounded reference runtime | Count/byte admission, FIFO lifecycle, queued cancellation, resumable one-token cursors, round-robin coordination, exact-batch and continuous multi-stream execution, and bounded output isolation |
| 8. UDS/Go adapter | COMPLETE for local reference path | Same-service-user kernel peer admission, separate Ed25519 request/response identities, bounded two-key verification overlap, signed ordered events, cancellation/disconnect watchers, durable replay rejection, single-stream CLI plus bounded batch/continuous reference drivers, Go `NATIVE_ENGINE` adapter and real socket tests |
| 8a. Canonical runtime v3 | PARTIAL / AUTHENTICATED DEVELOPMENT EXECUTION | Contract-pinned strict document codec, owner-configured live identity binding, signed availability/reservation/cancellation/release/execution over the existing peer-checked replay-journaled UDS service, finite continuous-mode execution workers, real selected loaded-model dispatch, exact request-bound CAG attachment, terminal result/resource enforcement, and replay/substitution/theft/double-use/expiry/concurrent-cancellation tests. The simulator is a separate explicit fault fixture. Single-connection execution fails closed; durable reservation recovery and actual qualified-package operated evidence remain unimplemented. |
| 9. Production package operations | COMPLETE for synthetic legacy inputs; Qwen source preflight plus exact tokenizer export; real-model approval pending | Rust-only quarantine safetensors importer, deterministic package mechanics and offline signing remain. Qwen diagnostics validate the 338-tensor BF16 source; the exporter writes only the exact AIPTOK02 to a protected fresh path. Model Foundry owns real Qwen package emission, approval, signing and qualification. |
| 9a. Quantized package storage | COMPLETE for AIPQ8/AIPQ4 synthetic evidence | Explicit profiles, canonical fixed-block payloads, signed source/destination/error evidence, secure loading, scalar kernels, ARM64 bounded block-scratch matvec without full F32 weight staging, and F32/Q8/Q4 transformer divergence regression |
| 9b. Execution placement | PARTIAL; initial sparse-expert portable foundation unactivated | Exact verified-artifact plans are composed into signed dense service loads; owner startup may select fully resident or bounded scalar whole-layer streamed CPU execution, with depth-one prefetch, generation parity and cancellation-safe KV invalidation. Separate sparse-expert mechanics now provide exact base and complete-plan identity, bounded immutable CPU cache and leases, deterministic eviction, one serialized miss, stale-completion rejection, per-cache plus process-wide bounded depth-one prefetch, cancellation-aware hashing, stable top-k routing and synthetic resident/streamed parity. No production MoE manifest, importer, package adapter, decoder/service activation, larger-than-RAM admission, direct quantized expert execution or accelerator evidence exists. Hybrid placement, per-model policy and operated performance remain open, with no performance claim. |
| 10. ARM64/Metal | SOURCE PATHS COMPLETE / OPERATED QUALIFICATION PENDING | ARM64 and Metal have owner-selected signed-directory loading, complete tiny-model prefill/decode/sampling/streaming, cancellation/unload, scalar parity and authenticated backend identity; target-specific production model, pressure, thermal, soak and performance qualification remain |
| 11. CUDA | HARDWARE GATED | Fixed kernels, differential and device tests required |
| 12. Production cutover | PARTIAL | Mutual response trust, bounded rotation mechanics, exclusively owned durable request replay, release-only signed artifact verification, monotonic rollback state, Linux pre-launch binding, root-owned verifier event routing, deterministic mutation/transport faults, scheduled fuzzing, signal-drained continuous CLI serving, bounded delayed-reader/restart soak, and pinned offline Linux/macOS verification are complete in source; trusted OS installation, hosted evidence, production-size operated soak/performance, key/journal recovery and reversible live canary remain |

## Completed host-independent implementation

- Foundational types and error taxonomy with checked arithmetic and hard bounds.
- Separate deterministic engine-release manifests use a release-only Ed25519
  domain and exact read-only direct-child artifacts. The offline pre-launch
  verifier advances a locked monotonic ledger outside engine runtime state and
  rejects signed rollback or same-generation substitution without giving the
  inference service install or self-update authority. The checked-in Linux
  supervisor gate retains bounded content-free failures for Go without giving
  Rust, clients, or models control-plane authority. A protected root-owned
  sequence checkpoint prevents reset of the bounded ring from reusing event
  identity. Go now persists only the last completed supervisor sequence,
  detects restart loss, ring overwrite/reset/rollback, and concurrent
  advancement without writing the root ring, and persists source-bound provider
  quarantine with audited exact-revision owner recovery.
- Exact, non-recursive, data-only package structures with no arbitrary
  deserialization, compression, plugins, paths, code, or kernel selection.
- Mature audited cryptographic primitives only; package/control private keys
  stay outside Rust, while the distinct response-only key is stdin-provisioned
  and zeroized.
- Secure local file admission for configuration-owned direct-child packages.
- Deterministic fixture generation entirely in Rust.
- Strictly separated Rust-only offline importer and production directory loader;
  source parsing/signing authority is absent from the runtime graph.
- Deterministic multi-shard import sorts unique local source filenames,
  enforces one global tensor inventory and aggregate byte ceiling, records every
  source digest, and emits contiguous signed destination shards independently
  of caller argument order. F32, F16, and BF16 source tensors are explicitly
  validated; finite values enter an explicit F32, AIPQ8, or AIPQ4 package
  profile. Signed per-tensor and per-layer error evidence retains poor results;
  the synthetic AIPQ4 fixture currently diverges on all four generated tokens.
- First-party tokenizer, tensor, immutable graph, scalar correctness backend,
  memory, KV, decoder, bounded first-party sampling,
  scheduler, mutually authenticated protocol, UDS transport, daemon composition, CLI,
  and Go provider translation.
- CPU socket-to-signed-package generation exercised end to end.
- A correctness-first contiguous KV-cache and bounded scratch subsystem define
  the backend-neutral ownership/admission contract. The scalar decoder now
  commits every token through that cache and reads only committed or current
  all-layer state using semantic layer/position/head coordinates.
- Every accepted Unix stream is admitted only when its kernel-reported effective
  user ID equals the engine process user. Mismatches are rejected before frame
  parsing and do not stop the continuous service; signature and replay checks
  remain independent mandatory layers.
- A read-only, finite Linux/macOS workflow runs the complete Rust workspace gate
  from the locked vendored graph with Cargo offline. Architecture tests pin its
  triggers, operating systems, toolchain, commands, permissions, and denial of
  privileged or download steps. Hosted success remains commit-specific evidence.
- Bounded per-control-key durable nonce journal rejects cross-connection and
  post-restart replay before execution, excludes a second live process, and
  fails closed on corrupt or contended state.
- Two-key current/previous verification windows support staged control and
  response identity cutover; each accepted control key retains its own locked
  journal and command streams cannot switch identities.
- Exhaustive canonical-fixture and signed-event truncation/single-byte mutation suites, signed
  active cancellation, a checked-in vendor graph, and lockfile-bound SPDX.
- Isolated coverage-guided Rust fuzz targets for package, tokenizer, frame,
  authenticated service envelope/body,
  directory manifest, tensor-index, safetensors-import, and package-path
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
  of eight repetitions passed locally. The weekly workflow is present on the
  default branch, but successful hosted execution is still absent.
  Production-size operated soak remains required.
- CPU service metrics now accumulate checked prefill/decode forward time,
  generated-token throughput, aggregate host reservations, and live contiguous
  KV reservations across single and bounded multi-stream execution. These are
  content-free operational counters, not production performance evidence.
- The protected UDS now also carries a signed, fixed-header service-control
  envelope with the closed lifecycle/tokenize/health/metrics/model-status/
  shutdown catalog. The path-free model-status query returns one signed load,
  capability, health, and scheduler-accounting snapshot for Go resource
  routing. The complete catalog is served by both the one-command reference
  path and the continuously admitting daemon, including while another stream
  generates.
  Owner-configured model budgets remain outside client messages, unload rejects
  scheduler ownership, and normal data-plane clients cannot authorize shutdown.
  Scheduler admission adds deterministic 4:2:1 priority service, bounded agent
  quotas, per-model concurrency, deadlines, and head-of-line avoidance while
  preserving count and aggregate-byte limits. ADR-0020 records exact semantics.

## Next software release gate: production model operations

Select and legally approve one real source model, then run the implemented
Rust-only importer within the reviewed quarantine/signing procedure. Review the
explicit architecture mapping and provenance, reproduce identical package
bytes independently, and retain license, source hash, package hash, key ID,
toolchain/source revision, SBOM, test results, and signer custody evidence.

Exit evidence is a reviewed production-size package that opens through its
registered ID, passes exact signature and tensor validation, and matches
independent CPU logits/generation vectors. No useful production model is
currently included in this repository.

## Accelerator gates

ARM64, Metal, and CUDA remain default-off. ARM64 now has a complete optimized
graph backend with isolated documented intrinsics, runtime dispatch, bounded
workers, scalar differential tests, and Apple M1 release measurements. Its
owner-selected signed decoder/service path now performs authenticated package
loading, prefill, cached decode, sampling, streaming, and exact descriptor
reporting, with scalar operation fallback when required. Production-size ARM64
model qualification and sustained operated evidence remain open. Metal now
has a build-time first-party metallib, closed C ABI, all baseline correctness
kernels, bounded reuse, explicit transfer/dispatch metrics, and contiguous
  sequence-isolated device KV storage with Apple M1 differential evidence. A
  registered signed synthetic package now runs complete prefill, cached decode,
  greedy sampling, and typed streaming through persistent Metal weights;
  cancellation and idle-only unload invalidate device state. Owner-selected
  UDS execution and exact Go descriptor pinning are complete; production-size
  qualification and an operated reversible rollout remain open. Each
activated backend requires fixed
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
- retain the checked-in Linux systemd definition that restricts address
  families to `AF_UNIX`, denies all IP addresses, makes the model store
  read-only, and confines writes to the runtime directory; operated CPU proof
  under that unit remains required; retain the checked-in macOS deny-by-default
  Seatbelt profile and local UDS-versus-loopback proof as transitional
  defense-in-depth while a signed App Sandbox release design remains required;
- require Go's platform-wide local-only mode to reject every remote primary,
  fallback, verifier, and manual override before sovereign cutover;
- operate and qualify the implemented Go Agent Green telemetry mapping,
  durable sender continuity, and duplicate-delivery handling; mapping remains
  outside the model and engine;
- canary the provider behind Go routing and preserve instant rollback;
- retire third-party runtimes only after equivalent production evidence.

Paged KV storage is deliberately deferred. It may follow only after the scalar
decoder uses the contiguous cache end to end and real-model long-context tests
show material pressure. The paged design must preserve zero-on-reassignment,
generation-checked sequence ownership, atomic token visibility, unload
invalidation and exact accounting, and must pass differential tests against
contiguous V1. Model packages and clients will not select page layout.

Hardware is not required for Stage 9 format/conversion/fuzz work. It is required
for accepting Stages 10 and 11 and for final performance qualification.

## Portable completion boundary

The bounded multi-model residency audit closed the prior single-CLI-registration
and all-models-hot control-plane gaps through ADR-0028. ADR-0029 adds the
verified-artifact placement abstraction, exact fully resident admission, and a
bounded cancellation-safe CPU staging/prefetch substrate. Scalar streamed
prefill/decode is bit-exact with the resident oracle on signed fixtures. The
authenticated service defaults to fully resident and may select scalar streamed
execution only through owner startup configuration. True hybrid execution is
not represented as complete.
ADR-0038 adds portable sparse-expert planning, cache, lease, invalidation,
prefetch and scalar routing/parity mechanics without accepting a production MoE
architecture. Exact model selection, manifest/import mapping, pinned loader
source integration, direct quantized expert kernels, model-larger-than-RAM
admission, optimized backends and operated evidence remain explicit gates.
Automatic eviction, a multi-entry signed Foundry registry,
and new modalities/architectures remain deliberately gated on measured or
model-specific requirements rather than represented as working features. Further
claims require external evidence: a legally approved real model, target
accelerator hardware and drivers, production service identities and filesystem
permissions, hosted workflow execution, operated recovery drills,
production-size soak/performance runs, and a reversible live canary through the
Go control plane. These are explicit evidence gates, not permission to add fake
backends, weaken verification, or move control-plane authority into Rust.
