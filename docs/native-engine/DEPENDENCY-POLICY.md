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

The engine may not gain egress indirectly through standard-library Internet
sockets, raw socket FFI, asynchronous network modules, subprocess execution,
or an external telemetry exporter. The protected Unix-domain stream in
`engine-server` is the sole approved communications primitive. Architecture
tests enforce source markers in addition to the exact dependency allowlist;
production deployment independently applies an operating-system outbound
network denial as specified by ADR-0006.

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

Direct production dependencies are exact `sha2 0.11.0`, `ed25519-dalek 3.0.0`,
`zeroize 1.8.2`, and `libc 0.2.189`, all with default features disabled.
SHA-384 and strict Ed25519 are package/control-message/response integrity
mechanisms. The narrowly enabled Ed25519 `zeroize` feature plus the direct
`zeroize` crate clear response-key seed material and signing state; no derive
feature is enabled. `libc` supplies only `O_CLOEXEC` and `O_NOFOLLOW`
constants to safe `std::fs::OpenOptions` for package and durable replay state,
the isolated nonblocking `flock` call recorded in ADR-0004, and the exact
`mmap`/`munmap` constants and calls used by the read-only, non-executable shard
mapping wrapper recorded in ADR-0015. The offline importer reuses the same
reviewed SHA-384, Ed25519, and zeroization dependencies; it adds no dependency.

The complete resolved graph, purpose, license, source, and attack-surface record
is [DEPENDENCY-INVENTORY.md](DEPENDENCY-INVENTORY.md). `Cargo.lock` pins exact
versions and registry checksums. Architecture tests fail on any unrecorded
external package or internal crate edge.

Coverage-guided parser testing is isolated in `engine/fuzz`, which is excluded
from the production workspace and has its own lockfile. Its only new direct
crate is `libfuzzer-sys 0.4.13`; the exact direct and transitive review is
recorded in `engine/fuzz/DEPENDENCIES.md`. The external libFuzzer runtime,
bundled C++, compiler-launching build script, and unsafe implementation are
test-only and are never linked into an engine binary, included in the release
SBOM, or made available at runtime. CI installs the separately pinned
`cargo-fuzz 0.13.2` orchestrator and Rust nightly, runs bounded campaigns, and
does not execute Python, Node, TypeScript, downloaded scripts, or model code.

## Feature and build controls

- Disable unnecessary direct default features.
- No workspace crate may declare native `links` or weaken workspace lints.
- No `build.rs` is permitted in workspace crates. A transitive build script
  requires explicit inventory review; current curve arithmetic build metadata
  is offline and locked.
- No runtime dynamic-library lookup by arbitrary/configured name.
- `cpu-reference` is default; `arm64`, `metal`, and `cuda` are explicit and
  default-off.
- Unsafe Rust is denied by default. The approved exceptions are the exact Unix
  lock module in ADR-0004, process-signal module in ADR-0011, and read-only
  peer-credential module in ADR-0013, plus the read-only shard mapping module
  in ADR-0015. They expose safe wrappers; the mapping owns the descriptor,
  pointer, and exact length and never exposes mutation or execution. Exact unsafe-block counts are enforced by
  architecture tests. Tests reject unsafe code and
  unsafe-lint allowances everywhere else. A future FFI/backend exception
  requires an isolated module, `deny(unsafe_op_in_unsafe_fn)`, written
  invariants, safe wrappers, platform tests, and separate approval.

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
The production checks above run only against `engine/Cargo.lock`; fuzz checks
and their separate lock are specified in [FUZZING.md](FUZZING.md).

The pinned `native-engine-verify` workflow runs the locked production metadata,
feature graph, formatting, build/check, Clippy, tests, docs, and disabled-backend
profile on Linux and macOS with Cargo explicitly offline. It has read-only
repository permission and no production secrets. Advisory review remains a
separate operator-owned `cargo audit --no-fetch` release check because neither
the audit executable nor advisory database belongs to the vendored runtime
dependency graph.

## Release requirements

The reviewed Cargo graph is checked in under `engine/vendor`, and Cargo replaces
crates.io with that directory while offline. `release/native-engine.spdx` is an
SPDX 2.3 inventory of every locked external version, archive checksum, and
license. Boundary tests require the vendor and SBOM package sets to equal the
lockfile and prohibit Python/TypeScript in the release tree. Each production
release must still reproduce a fresh offline build, record
Rust/toolchain/target flags and advisory results, and verify the dependency
archive before publication.
