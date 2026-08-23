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
`zeroize 1.8.2`, `libc 0.2.189`, `unicode-normalization 0.1.24`,
`unicode-general-category 1.1.0`, and Sovereign AI Contracts commit
`af62132e46fc641e7dd74eb3e21eabb6d40947c7`, all with unnecessary default
features disabled. The Contracts dependency enables only the narrow
`models-v2` consumer profile and its runtime-v3 dependency; cognitive, tool, memory, security, and other
control-plane authority modules are excluded from the compiled engine graph.
SHA-384 and strict Ed25519 are package/control-message/response/release integrity
mechanisms. `engine-tokenizer` reuses the existing first-party SHA-384 wrapper
for AIPTok's internal content hash; it adds no external dependency. The
first-party `engine-quantization` crate uses only workspace model-format,
tensor, memory, and types crates; it adds no external dependency or native
code. The
narrowly enabled Ed25519 `zeroize` feature plus the direct
`zeroize` crate clear response-key seed material and signing state; no derive
feature is enabled. `libc` supplies only `O_CLOEXEC` and `O_NOFOLLOW`
constants to safe `std::fs::OpenOptions` for package and durable replay state,
the isolated nonblocking `flock` call recorded in ADR-0004, and the exact
`mmap`/`munmap` constants and calls used by the read-only, non-executable shard
mapping wrapper recorded in ADR-0015. The offline importer and isolated
engine-release signer/verifier reuse the same reviewed SHA-384, Ed25519, and
zeroization dependencies; they add no dependency. Release signatures have a
distinct domain and Rust trust type. Release artifact files use the already
reviewed `O_NOFOLLOW|O_CLOEXEC` constants; the ledger uses only safe
standard-library create-new, synchronization, and rename operations.

AIPTok v2's data-only Qwen2.5 parser and private fixed scanner/BPE/decode core
need exact Unicode NFC and General Category `L`/`N` semantics that Rust `std`
does not provide. The reviewed,
Unicode-version-matched pair is therefore admitted as general-purpose Unicode
infrastructure: `unicode-normalization 0.1.24` and
`unicode-general-category 1.1.0` both declare Unicode 16.0.0 data. They do not
implement a tokenizer, vocabulary, merge algorithm, regex, pre-tokenizer,
decoder, chat template, model loader, or model-selected rule. Only
`engine-tokenizer` may depend on them. A crate-private adapter now consumes the
two version constants, streams NFC scalars into an engine-bounded output, and
maps every General Category `L*`/`N*` value. The fixed contraction matcher uses
an engine-owned Unicode-16 simple-fold table only for mappings that reach ASCII
(ASCII case, U+017F LONG S, and U+212A KELVIN SIGN). The runtime core and
adapter remain nonconstructible through ordinary public APIs. Exact pinned
source conversion and fixed-corpus conformance run only through a feature-gated
offline diagnostic. The exporter writes one exact read-only artifact; it
returns no tokenizer handle and grants no package trust. Only the separately
signed, owner-approved, ARM64 development path can construct that tokenizer,
and production activation remains unavailable.

The private adapter rejects input whose Unicode-16 NFKD stream-safe view would
insert a CGJ after more than 30 consecutive nonstarters; it never forwards the
inserted CGJ or transforms over-limit input. The vendored normalizer is locally
hardened so the four-entry decomposition and recomposition `TinyVec` buffers
have 64 and 32 inline entries respectively, and canonical pending marks use
stable in-place insertion sort rather than an allocator-backed stable sort.
The decomposition bound covers 30 pending nonstarters plus the complete
four-scalar maximum Unicode-16 canonical decomposition. Thus every admitted
NFC run stays inline and has no sort allocation. A conservative fixed 4 KiB
workspace charge remains held in the engine memory pool across NFC, separately
from the bounded normalized output reservation. Architecture tests pin the
reject-only precheck, buffer sizes, insertion-sort implementation, workspace
charge, and retained-file hashes.

Both direct crates have defaults disabled and no features selected. NFC's
iterator API is available through `unicode-normalization`'s `no_std` + `alloc`
path, so its empty `std` feature is intentionally not enabled. The former
resolves `tinyvec 1.12.0` with `alloc` and its
`tinyvec_macros 0.1.1` dependency; `tinyvec`'s activated `default` feature is
empty. No optional serialization, schema, debugger, benchmark, or other
`tinyvec` feature is enabled. The latter has no transitive dependency. All four
registry versions and checksums are exact in `Cargo.lock`.

Admission review on 2026-08-22 found:

- `unicode-normalization 0.1.24` (published 2024-09-17; crates.io owners
  `huonw`, `SimonSapin`, `Manishearth`, `kwantam`, and `sujayakar`) is
  MIT OR Apache-2.0. Its archive SHA-256 is
  `5033c97c4262335cded6d6fc3e5c18ab755e1a3dc96376350f3d8e9f009ad956`.
  Direct defaults and all features are disabled. It has no build script,
  native link, network, filesystem, process, or dynamic
  loading surface at runtime. Its only unsafe code is the reviewed Hangul
  decomposition/composition implementation in `src/normalize.rs`: one unsafe
  function, four unsafe expressions/blocks, and five checked
  `char::from_u32_unchecked` conversions guarded by documented Unicode scalar
  range invariants. The two iterator buffer declarations and canonical pending
  sort carry the reviewed local allocation-bound patch described above; their
  exact patched SHA-256 values are recorded in Cargo's retained-file map. The
  unused executable Python table generator was pruned from the vendored tree
  and removed from that map. The upstream archive checksum, license, table data,
  public API, and package identity remain recorded and unchanged.
- `unicode-general-category 1.1.0` (published 2025-09-16; crates.io owners
  `wezm` and the YesLogic developers team) is Apache-2.0. Its archive SHA-256 is
  `0b993bddc193ae5bd0d623b49ec06ac3e9312875fdae725a975c51db1cc1677f`.
  Its deterministic Rust `build.rs` reads only the bundled Unicode table and
  writes a compiled table under Cargo `OUT_DIR`; it has no dependency, unsafe,
  network, subprocess, dynamic-loading, or native-link surface. Its unused
  external table-generation Makefile was pruned. The reviewed build script
  SHA-256 is
  `a877183b60fdc9846063f7ed5a1dc23f8ef26e1ff4994bc4cfc7418ddb4ec48e`.
- `tinyvec 1.12.0` (published 2026-07-10; crates.io owners `Lokathor` and the
  rust-bus maintainers team) is Zlib OR Apache-2.0 OR MIT, forbids unsafe code,
  has no build script/native link, and has archive SHA-256
  `bb4ebadaa0af04fab11ae01eb5f9fdb5f9c5b875506e210e71c07873528baa7f`.
  `tinyvec_macros 0.1.1` (published 2023-02-02; owner `Soveu`) is MIT OR
  Apache-2.0 OR Zlib, is an ordinary macro library rather than a proc macro,
  forbids unsafe code, has no build script/dependency/native link, and has
  archive SHA-256
  `1f3ccbac311fea05f86f61904b462b55fb3df8837a366dfc601a0161d0532f20`.
- The category and `tinyvec` archives' package-local `Cargo.lock` files are
  ignored upstream and are not runtime, build, or release inputs. They were
  pruned together with their retained-file-map entries. Architecture tests now
  require every remaining vendor checksum-map path to exist in Git-tracked
  content, with a clean `git archive` check proving the no-`.git` release case.
- `cargo audit --no-fetch` scanned this lock against 1,225 locally cached
  advisories at database commit
  `bf5c0d245a92671908518d7e765914d437954ed6` (2026-08-21) and reported no
  vulnerability. This is cached advisory evidence, not a live registry audit.

Removal now requires a reviewed first-party replacement that passes the same
Unicode-16 normalization/category, nonstarter-bound, memory-accounting, and
tokenizer vectors. Delete the two direct `engine-tokenizer` dependency entries,
regenerate the exact lock/vendor/SBOM, and remove these four vendor directories
only as one reviewed slice. Handwritten partial normalization, an approximate
regex, or use of a proc-macro Unicode table as runtime authority is prohibited.

The complete resolved graph, purpose, license, source, and attack-surface record
is [DEPENDENCY-INVENTORY.md](DEPENDENCY-INVENTORY.md). `Cargo.lock` pins exact
versions and registry checksums. Architecture tests fail on any unrecorded
external package or internal crate edge.

The exact Contracts commit is the sole reviewed non-registry source. Cargo's
source replacement binds that exact Git revision to the checked-in vendor tree,
and architecture tests reject revision or source drift. Its Serde/JSON and
SHA-256 dependencies exist only to strictly decode, validate, serialize, and
bind the provider-neutral runtime-v3 documents owned by Sovereign Contracts.
They add no socket, filesystem, process, model-loading, scheduler, or inference
implementation. Replacing this dependency with repository-local wire structs
is prohibited because it would fork the cross-repository security contract.

Coverage-guided parser testing is isolated in `engine/fuzz`, which is excluded
from the production workspace and has its own lockfile. Its only new direct
crate is `libfuzzer-sys 0.4.13`; the exact direct and transitive review is
recorded in `engine/fuzz/DEPENDENCIES.md`. The external libFuzzer runtime,
bundled C++, compiler-launching build script, and unsafe implementation are
test-only and are never linked into an engine binary, included in the release
SBOM, or made available at runtime. CI installs the separately pinned
`cargo-fuzz 0.13.2` orchestrator and Rust nightly, runs bounded campaigns, and
does not execute Python, Node, TypeScript, downloaded scripts, or model code.

The internal `engine-knowledge-contract` crate introduces no external package.
It depends only on `engine-types` and the existing audited hash primitives in
`engine-crypto`. It pins the audited Sovereign-AI-Contracts commit, schema
digest, and lifecycle fixture digests while carrying only closed execution
facts. The broader Contracts `full` Rust feature is intentionally not imported:
it would add unrelated control-plane domains and JSON dependencies to the
inference data plane.

## Feature and build controls

- Disable unnecessary direct default features.
- No workspace crate may declare native `links` or weaken workspace lints.
- Exactly one workspace build script is permitted:
  `engine-metal/build.rs`, recorded in ADR-0025. It invokes only the installed
  Apple `metal`, `metallib`, `clang++`, and `ar` tools against repository-owned
  sources, writes only to `OUT_DIR`, embeds the result, and performs no network
  access. Architecture tests reject every additional workspace build script.
  A transitive build script requires explicit inventory review. Current curve
  arithmetic build metadata is offline and locked. The only newly admitted
  Unicode build script is `unicode-general-category 1.1.0/build.rs`; its exact
  offline, bundled-table-to-`OUT_DIR` behavior and hash are recorded above and
  enforced by architecture tests.
- No runtime dynamic-library lookup by arbitrary/configured name.
- `cpu-reference` is default; `arm64`, `metal`, and `cuda` are explicit and
  default-off.
- Unsafe Rust is denied by default. The approved exceptions are the isolated
  ARM64 Advanced SIMD module recorded in ADR-0023, the Metal C ABI module
  recorded in ADR-0025, and the exact Unix
  lock module in ADR-0004, process-signal module in ADR-0011, and read-only
  peer-credential module in ADR-0013, plus the read-only shard mapping module
  in ADR-0015. They expose safe wrappers; the mapping owns the descriptor,
  pointer, and exact length and never exposes mutation or execution. Exact unsafe-block counts are enforced by
  architecture tests. Tests reject unsafe code and
  unsafe-lint allowances everywhere else. Any further FFI/backend exception
  requires an isolated module, `deny(unsafe_op_in_unsafe_fn)`, written
  invariants, safe wrappers, platform tests, and separate approval.
  Vendored `unicode-normalization 0.1.24` contains a separately reviewed
  upstream Hangul-only unsafe boundary described above; it is not workspace
  code, FFI, model-selected behavior, or authority to add another exception.

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
