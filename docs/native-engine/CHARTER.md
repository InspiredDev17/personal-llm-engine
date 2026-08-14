# Native Engine Charter

## Mission

Build and operate a first-party, inference-only Rust data plane for the existing
sovereign local-first PersonalAI platform. The existing Go platform is and
remains the control plane and system authority. This repository does not train
models and is not a second agent platform.

## Non-negotiable authority boundary

The engine:

- cannot execute tools or side effects;
- cannot make approval, policy, routing, or authorization decisions;
- cannot read or modify platform memory or durable audit state;
- cannot access arbitrary files; only owner-registered model IDs resolve to
  configuration-owned direct-child package names;
- cannot download models or open outbound Internet connections;
- cannot update itself;
- cannot load model-supplied executable code, plugins, libraries, graphs, or
  kernel choices;
- cannot accept an arbitrary client model path;
- cannot hold Go request-signing private keys or package-signing private keys;
- may hold only the separate owner-provisioned engine response-signing key,
  which confers no control-plane, model, tool, file, network, or Agent Green
  authority;
- cannot connect directly to Agent Green or interpret its decisions.

The engine returns tokens, usage, failures, capabilities, and sanitized
security telemetry. All returned data is untrusted at the Go boundary.

## Model policy

Production models must use the first-party signed, canonical, data-only
directory-backed `.paimodel` V1 package. Every model must be registered by `ModelId` in local owner
configuration with an exact SHA-384 package hash and active Ed25519 signer
identity. Requests contain only that ID. Packages contain no paths, scripts,
plugins, shared libraries, Metal source, PTX, or executable configuration.
Production activation requires a non-symlink model root, package directory,
and trusted files with no write permission bits.

Package storage may use exact F32/F16/BF16 or the reviewed first-party AIPQ8 v1
and AIPQ4 v1 formats. The offline operator must select a named profile; the
model cannot select conversion behavior or a kernel. Quantized bytes, source
and destination hashes, and unaggregated error evidence are signed package
data. No third-party quantization format or runtime is accepted.

The Rust-only offline importer accepts bounded local safetensors plus strict
`AIPTOK-METADATA-V1` tokenizer source metadata from an owner quarantine,
converts the exact
allowlisted architecture, emits deterministic hashes/provenance, and signs the
closed package file set. It has no network, subprocess, plugin, pickle, Python,
or repository-code execution capability. The production loader cannot import
source formats; it opens only an already converted registered directory.

The sole tokenizer family is AIPTok Byte-Level BPE v1. It preserves exact
input bytes, has mandatory byte fallback and fixed decoder behavior, and
contains no programmable normalization, pre-tokenizer, chat template, role,
tool, prompt-file, policy, or remote-code behavior. Go assembles chat text and
retains every policy boundary. Rust treats prompt text and explicit token data
only as bounded inference input.

The implemented V1 CPU path is validated by a tiny deterministic test package.
That fixture is not a production model. No production model is bundled or
claimed until Rust-only conversion, provenance, licensing, signing, independent
numeric vectors, and production-size resource tests pass.

## Transport policy

The service is local-only over a protected Unix-domain socket. TCP, HTTP,
outbound sockets, peer-selected endpoints, and network fallback are prohibited.
Go authenticates commands with a strict Ed25519 signature and nonzero nonce.
Rust holds that public verification identity only. Rust authenticates every
response with a separate strict Ed25519 identity provisioned through stdin;
the seed is zeroized after construction. Go pins its public key and rejects
tampered, replayed, reordered, wrong-request, wrong-nonce, and wrong-key events.
The Go adapter is consumer-owned and preserves canonical platform
privacy/routing validation.

## Agent Green policy

Agent Green receives allowlisted security events through the Go platform's
existing liaison. It is never controlled by the model. Rust can emit only typed
sanitized telemetry to Go; Go owns validation, audit, authentication, delivery,
and response. Rust holds no Agent Green key/token and has no direct transport.

## Language and dependency policy

No Python is required in production, tests, conversion, fixture generation, or
build-critical paths. No TypeScript or Node runtime is introduced. The Rust
workspace may use the toolchain, standard OS facilities, later reviewed Metal
or CUDA system APIs, and the minimal recorded general-purpose allowlist.

Third-party AI, tensor, tokenizer, model-loader, scheduler, model-server, and
GPU-kernel dependencies are prohibited. Cryptography uses mature reviewed
implementations; custom cryptography is prohibited. Exact versions are locked,
unnecessary default features are disabled at direct dependency declarations,
and releases must be vendored with an SBOM.

Unsafe Rust is denied workspace-wide by default. The current exceptions are the
isolated Unix replay-journal lock FFI module recorded by ADR-0004 and the
process-signal install/restore module recorded by ADR-0011, plus the read-only
peer-credential module recorded by ADR-0013 and the read-only non-executable
tensor-shard mapping module recorded by ADR-0015. Their exact unsafe blocks are
hidden behind safe wrappers and enforced by an
architecture-boundary test. Any other unsafe code requires its own invariant
and review record in an isolated FFI/backend module.

## Engineering rules

- Checked arithmetic precedes dimensions, offsets, lengths, allocations, token
  admission, KV growth, and CPU work.
- Data structures and schemas are flat and bounded; no arbitrary or recursive
  deserialization is accepted.
- Every incomplete capability fails with a typed unsupported/not-implemented
  error; no empty success or fake inference is allowed.
- CPU reference is the default oracle. The optimized ARM64 and Metal graph
  backends are default-off, runtime-checked, and bounded. Metal may advertise
  model execution only after registered signed-package admission and complete
  owner startup configuration; the UDS scheduler now supports that path and Go
  must pin the signed Metal descriptor. ARM64 decoder activation and CUDA remain
  disabled until their remaining acceptance gates pass.
- Inference graphs are immutable engine-owned plans built only by reviewed
  architecture code. A model package or client cannot supply operations,
  operation names, executable graphs, kernels, or backend choices.
- The Go platform never depends on unfinished behavior. Adoption is additive,
  canaried, and reversible.

## Current completion boundary

The software foundation through deterministic one-or-more-shard offline
synthetic-model import,
signed directory-package verification/mapping, mutually authenticated
CPU-reference UDS generation and bounded continuous graceful-drain reference
serving, strict AIPTok conversion/encoding/streaming decode, plus deterministic
delayed-reader/restart soak is implemented and
tested. Remaining release work is selecting and licensing an approved real
source model, producing/reviewing its production package and independent
numeric vectors, production-size fuzz/operated-soak/performance evidence, operated
request/response-key injection, rotation/recovery evidence, repeatable
vendor/SBOM release renewal, trusted OS installation and operated release
rollback evidence, and hardware-specific
accelerator implementation/validation. These limits prohibit claiming complete
production replacement today.
