# Native Engine Dependency Policy

## Default and prohibitions

The default is no external dependency. No third-party AI runtime, tensor,
tokenizer, model format/loader, scheduler, model server, GPU-kernel package,
graph executor, or dynamic-loading crate is allowed. Network clients, remote
code fetch, curl-pipe-shell installation, runtime library discovery, and build
scripts that fetch or generate remote code are prohibited.

No Python, Node, or TypeScript runtime may enter production, tests, fixtures,
conversion, or a build-critical path. Model-supplied code and plugins are never
dependencies.

## Approval record

Before adding a dependency, commit:

- exact crate/version/source and whether direct, transitive, dev, or build;
- narrow purpose and why standard/in-repository code is insufficient;
- enabled/disabled features and build-script/native-link behavior;
- license and compatibility decision;
- maintainer/release/advisory review;
- attack-surface, transitive graph, unsafe, network, filesystem, process,
  dynamic-loading, and code-generation analysis;
- removal/replacement plan and reviewer/date.

Cryptography must use mature reviewed implementations. Custom cryptographic
primitives are forbidden. New crypto algorithms require an ADR and known-answer,
negative, weak-key, domain-separation, revocation, and malformed-input tests.

## Current allowlist

Direct production dependencies are exact `sha2 0.11.0`, `ed25519-dalek 3.0.0`
with default features disabled, and `libc 0.2.189` with default features
disabled on Unix. SHA-384 and strict Ed25519 are package/control-message
integrity mechanisms. `libc` supplies only `O_CLOEXEC` and `O_NOFOLLOW`
constants to safe `std::fs::OpenOptions`; the workspace contains no unsafe code.

The complete resolved graph, purpose, license, source, and attack-surface record
is [DEPENDENCY-INVENTORY.md](DEPENDENCY-INVENTORY.md). `Cargo.lock` pins exact
versions and registry checksums. Architecture tests fail on any unrecorded
external package or internal crate edge.

## Feature and build controls

- Disable unnecessary direct default features.
- No workspace crate may declare native `links` or weaken workspace lints.
- No `build.rs` is permitted in workspace crates. A transitive build script
  requires explicit inventory review; current curve arithmetic build metadata
  is offline and locked.
- No runtime dynamic-library lookup by arbitrary/configured name.
- `cpu-reference` is default; `arm64`, `metal`, and `cuda` are explicit and
  default-off.
- Unsafe Rust is forbidden today. A future FFI/backend exception requires an
  isolated module, `deny(unsafe_op_in_unsafe_fn)`, written invariants, safe
  wrappers, platform tests, and separate approval.

## Required checks

```sh
cargo metadata --format-version 1 --locked
cargo tree --workspace --all-features --locked -e features
cargo audit --no-fetch
cargo fmt --all --check
cargo clippy --workspace --all-targets --all-features --locked -- -D warnings
cargo test --workspace --all-targets --all-features --locked
```

Review source/checksums, licenses, feature unification, build scripts, native
links, unsafe use, network/process/file capabilities, and advisory status.

## Release requirements

Production releases must vendor the reviewed Cargo graph, build offline from a
fresh environment, publish a CycloneDX or SPDX SBOM with checksums/licenses,
record Rust/toolchain/target flags, retain advisory results, and make the
dependency archive reproducible. Vendoring and SBOM publication are not yet
claimed for this development workspace and remain a production stop gate.
