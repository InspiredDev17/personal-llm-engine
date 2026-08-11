# Personal LLM Engine

Status: **CPU reference runtime implemented; production model conversion and accelerated backends remain gated.**

This repository is the first-party Rust inference data plane for the sovereign
PersonalAI platform. The existing Go platform remains the control plane and
sole authority for model registration, routing, policy, approvals, tools,
memory, and audit. The engine is inference-only and is not a second AI platform.

The workspace implements a signed data-only `.paimodel` format, SHA-384 and
strict Ed25519 admission, a configuration-only model registry, no-follow local
package opening, a first-party bounded byte/BPE tokenizer, safe aggregate-
accounted tensors, one fixed decoder architecture, scalar CPU reference
inference, KV state, deterministic greedy generation, a bounded scheduler,
mutually authenticated binary protocol, protected Unix-domain socket runtime,
a bounded per-control-key durable replay journal, and an owner CLI. The Go
repository contains the corresponding `NATIVE_ENGINE` provider adapter; it has
no HTTP or remote-network fallback.

The executable CPU path is a correctness baseline tested with a deterministic
signed synthetic model. It is not yet a useful production model. Rust-only
conversion/signing operations, an approved production package, stochastic
sampling, optimized ARM64, Metal, CUDA, performance qualification, and
coverage-guided fuzzing remain required before replacement of existing
production runtimes. The locked dependency graph is checked in under
`engine/vendor`, Cargo is offline/source-replaced by default, and
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
- clients send a registered `ModelId`, never a path;
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
- every authenticated command nonce is appended and synchronized to a bounded,
  mode-0600 per-control-key journal before the command can execute; an exclusive
  lifetime lock prevents a second process from using the same key journal, and
  retiring a key does not delete its replay history;
- packages are fixed-format data, exact-hash pinned, and signer verified;
- the UDS is created only in an owner-private runtime directory with mode 0600;
- the engine cannot execute tools, approve actions, mutate platform memory, or
  connect directly to Agent Green.

See [the charter](docs/native-engine/CHARTER.md),
[architecture](docs/native-engine/ARCHITECTURE.md), and
[roadmap](docs/native-engine/ROADMAP.md).
