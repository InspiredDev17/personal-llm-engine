# Personal LLM Engine

Status: **CPU reference runtime and synthetic offline model packaging implemented; real-model approval and accelerated backends remain gated.**

This repository is the first-party Rust inference data plane for the sovereign
PersonalAI platform. The existing Go platform remains the control plane and
sole authority for model registration, routing, policy, approvals, tools,
memory, and audit. The engine is inference-only and is not a second AI platform.

The workspace implements a signed directory-backed data-only `.paimodel`
format, a Rust-only quarantine safetensors importer with deterministic
multi-shard conversion, deterministic provenance,
SHA-384 and strict Ed25519 admission, a configuration-only model registry,
an explicit exact-owner-selected Model Foundry registry V2 admission mode and
a separately signed development-only Qwen registry V3 adapter bound to
canonical Sovereign Models 2.0 descriptors,
no-follow direct-file opening, read-only non-executable shard mapping, strict
AIPTok Byte-Level BPE v1 metadata conversion and bounded byte/UTF-8/incremental
tokenization, plus exact pinned-source conversion into a separate `AIPTOK02`
Qwen2.5 table format with a full-hash-bound trusted-loader constructor,
canonical aggregate-accounted tensors with explicit F16/BF16
storage conversion, deterministic first-party AIPQ8/AIPQ4 package profiles and
scalar quantized reference kernels, an immutable architecture-owned graph,
a closed legacy decoder and Qwen2.5 correctness scaffolding with
architecture-first split-half RoPE under internal ARM64 selection,
scalar CPU reference
inference, contiguous sequence-owned KV state, deterministic greedy and seeded
temperature/top-k/top-p/min-p/repetition-penalty sampling, a bounded scheduler,
resumable one-token round-robin coordination, mutually authenticated binary
protocol, protected Unix-domain socket runtime, a bounded authenticated
multi-stream reference batch, a bounded per-control-key durable replay journal,
per-stream bounded output queues, a caller-shutdown continuous reference driver,
an inference-runtime-v3 semantic reservation state machine pinned to the
canonical Sovereign AI Contracts commit, canonical availability/reservation/
cancellation/release/execution documents carried by the existing signed,
replay-journaled UDS service,
an owner CLI, and a separate release-only signer/verifier with a monotonic
installed-version ledger. The CLI also exposes an explicit one-shot `generate`
command that admits a registered signed package by model ID and never accepts a
URL or arbitrary package path. The CLI exposes the continuous driver only
through an explicit bounded mode, requires an explicit owner-pinned, frozen
Model Foundry V2, or experimental Qwen V3 registration mode, and translates
owner-process `SIGINT`/`SIGTERM` into the existing stop-admit, cancel, and drain
path. Qwen V3 additionally requires the exact
`--experimental-qwen-development-execution owner-approved` gate, ARM64, and a
matching Runtime-v3 identity. A normal real-socket test repeats eight
six-client restart cycles while one authenticated reader waits for five peers;
a pinned weekly/manual workflow is configured to repeat that bounded test eight
times once integrated onto the default branch. A separate pinned, read-only
push/pull-request workflow runs the complete locked and offline workspace gate
on Linux and macOS, compiling both kernel peer-credential implementations. The
Go repository contains the corresponding `NATIVE_ENGINE` provider adapter; it
has no HTTP or remote-network fallback.

The executable CPU path is a correctness baseline tested with a deterministic
signed legacy synthetic model and direct deterministic tiny Qwen2.5 scalar
fixtures. It is not yet a useful production model. Engine 0.2.0 accepts the
exact Qwen2.5 PAIDIR01 v1.1 format only through a separately signed,
anti-rollback development registry V3 record, an exact canonical Sovereign
Models 2.0 descriptor, and an additional explicit owner startup approval.
That experimental path is ARM64-only, cancellation-aware, package-bound to
`AIPTOK02`, and remains default-off. Registry V2 is frozen and unchanged;
CPU-reference and Metal cannot execute the Qwen profile. Production activation
is structurally unavailable. Model Foundry must still produce a real signed
development package/record and genuine qualification evidence before local
inference can be claimed. The exact 1.5B profile also remains outside the
100,000,000-operation scalar correctness ceiling. A default-off ARM64 graph
backend provides bounded Advanced SIMD graph kernels with scalar-oracle
differential tests. The default-off Metal backend now completes
the signed synthetic decoder and owner-selected UDS service path, while its
production-size canary remains gated. A legally approved real source
model/package and independent numeric evidence, ARM64 service integration,
CUDA, performance qualification, and
production-size fuzz/soak evidence remain required before replacement of
existing production runtimes. Ordinary CPU and Metal service tokenizer support
remains limited to AIPTok Byte-Level BPE v1. AIPTok v2 can be deterministically
exported to one fresh owner-private read-only file only from the exact pinned
Qwen2.5 source, replayed through a feature-gated content-free offline
conformance diagnostic, and constructed only by the Qwen V3 ARM64 path after
revalidating an authenticated package manifest's complete tokenizer hash. The
generic factory still returns unsupported; SentencePiece,
Unigram, WordPiece, tiktoken, and arbitrary `tokenizer.json` graphs are not
supported. The locked dependency
graph is checked in under `engine/vendor`, Cargo is offline/source-replaced by
default, and
`release/native-engine.spdx` records package versions, checksums, and licenses.

## Build and verify

Run from `engine/`:

```sh
cargo fmt --all --check
cargo check --workspace --all-targets --all-features --locked
cargo build --workspace --all-targets --all-features --locked
cargo clippy --workspace --all-targets --all-features --locked -- -D warnings
cargo test --workspace --all-targets --all-features --locked
cargo doc --workspace --all-features --no-deps --locked
cargo audit --no-fetch
```

The default feature is the safe CPU reference backend. `arm64`, `metal`, and
`cuda` are explicit disabled boundaries and never report implemented execution.

## Security properties

- no model training, Python, TypeScript, TCP listener, outbound connection,
  download, updater, dynamic-library discovery, or model-supplied code;
- release-only signatures bind the exact read-only engine artifact set, and a
  protected ledger outside engine runtime rejects signed rollback and
  same-generation substitution without granting the service self-update;
- clients send a registered `ModelId`, never a path;
- runtime-v3 reservations bind the control-plane logical model, engine registry
  model, artifact/package digest, backend and engine instances, task/request,
  budget, expiry, and idempotency key; exact replay is idempotent while
  substitution, theft, double consumption, overcommit, and outward budget
  escalation fail closed;
- runtime-v3 service composition is owner-enabled and model identities are
  owner-bound to registered package/backend state before signed availability or
  reservation is exposed; canonical execution invokes the exact selected loaded
  model through a finite continuous-mode worker set, revalidates identity,
  deadline, cancellation, and resource ceilings, and returns an exact signed
  terminal result. The simulator remains only for explicit development fault
  tests. Single-connection serving rejects execution because it cannot observe
  concurrent canonical cancellation; durable accepted-reservation recovery
  remains a release gate;
- registry-backed admission verifies the exact owner-selected signed V2 entry
  before package access, derives the approved path from signed identities, and
  never chooses `latest` or falls back to owner-pinned registration;
- registry and package signing identities must be distinct from each other and
  from Go control/engine response identities;
- Go signs control requests; Rust signs every response with a separate
  owner-provisioned response identity whose seed enters only through stdin and
  is zeroized after construction;
- each verifier accepts at most an explicit current/previous public-key pair
  during staged rotation; unknown, duplicate, partial, or model-selected keys
  are rejected and one command stream cannot switch identities;
- response signatures bind the request nonce, exact request ID, strictly
  increasing event ordinal, response key ID, and exact event bytes;
- Go sends signed same-stream cancellation and Rust observes disconnects while
  CPU generation is active;
- the multi-stream reference path caps connection count, cancellation-watcher
  and output-writer stack sizes, per-stream output events, and aggregate active
  request/token/cursor/output retention before work; a saturated writer queue
  cancels only its request;
- the continuous reference path applies one 64-connection ceiling across
  authenticating, queued, active, and draining streams; graceful shutdown stops
  admission, interrupts incomplete handshakes, cancels work, and joins every
  bounded worker before returning; the CLI requires explicit connection and
  active-sequence limits and accepts shutdown only from owner-process signals;
- every authenticated command nonce is appended and synchronized to a bounded,
  mode-0600 per-control-key journal before the command can execute; an exclusive
  lifetime lock prevents a second process from using the same key journal, and
  retiring a key does not delete its replay history;
- production packages are closed directory-backed data, exact-hash pinned,
  signer verified, and mapped read-only without execute permission;
- the UDS is created only in an owner-private runtime directory with mode 0600;
- every accepted UDS stream must have a kernel-reported effective user ID equal
  to the engine service user before any frame is read or allocated; cryptographic
  authentication and durable replay checks remain mandatory afterward;
- the engine cannot execute tools, approve actions, mutate platform memory, or
  connect directly to Agent Green.

See [the charter](docs/native-engine/CHARTER.md),
[architecture](docs/native-engine/ARCHITECTURE.md), and
[roadmap](docs/native-engine/ROADMAP.md). The exact quantization format and
synthetic accuracy limitations are in
[first-party quantization](docs/native-engine/QUANTIZATION.md).
