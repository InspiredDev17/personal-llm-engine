# Resolved Dependency Inventory

Sources are either the locked crates.io registry checksum in `Cargo.lock` or the
single exact Sovereign AI Contracts Git commit recorded below. No entry supplies
AI, tensor, tokenizer, model-loading, scheduling, serving, networking, dynamic
loading, or GPU kernels.

| Package | License | Purpose and attack-surface justification |
| --- | --- | --- |
| `sha2 0.11.0` | MIT OR Apache-2.0 | Direct mature SHA-384 implementation for packages, messages, AIPTok, and streaming engine-release artifact identity; pure digest surface |
| `ed25519-dalek 3.0.0` | BSD-3-Clause | Direct strict Ed25519 verification and role-separated package/response/release signing; only the `zeroize` feature is enabled |
| `zeroize 1.8.2` | Apache-2.0 OR MIT | Direct audited clearing of owner-provided response and offline release-signing seeds plus Ed25519 signing state; no derive feature |
| `libc 0.2.189` | MIT OR Apache-2.0 | Direct Unix constants `O_NOFOLLOW`/`O_CLOEXEC` for safe package/replay/release opens, isolated checked `flock` for exclusive journal ownership, isolated `sigaction` install/restore for graceful shutdown, isolated read-only effective-user/peer-credential calls for UDS admission, and isolated `PROT_READ|MAP_PRIVATE`/`munmap` tensor-shard mapping; no executable mapping or dynamic link discovery |
| `unicode-normalization 0.1.24` | MIT OR Apache-2.0 | Direct `engine-tokenizer`-only Unicode 16.0 NFC and allocation-free stream-safe primitives behind the private AIPTok v2 adapter; defaults and all direct features are disabled, its decomposition/recomposition buffers are locally hardened to 64/32 inline entries, canonical pending marks use stable in-place insertion sort, and it supplies no tokenizer graph, regex, model rule, I/O, or authority |
| `unicode-general-category 1.1.0` | Apache-2.0 | Direct `engine-tokenizer`-only Unicode 16.0 General Category lookup used by the private fixed `L`/`N` classifier; defaults disabled, no features or dependency, and no model-selected classification |
| `tinyvec 1.12.0` | Zlib OR Apache-2.0 OR MIT | Transitive safe inline storage beneath Unicode normalization; only `alloc` plus its empty default feature resolve, but the private adapter's reject-only 30-nonstarter gate and local inline-64/32 patch prevent heap spill on admitted NFC input |
| `tinyvec_macros 0.1.1` | MIT OR Apache-2.0 OR Zlib | Transitive ordinary declarative macro support selected by `tinyvec/alloc`; not a proc macro and has no dependency or build script |
| `digest 0.11.3` | MIT OR Apache-2.0 | SHA-2 trait/internal support |
| `block-buffer 0.12.1` | MIT OR Apache-2.0 | SHA-2 fixed-block buffering |
| `crypto-common 0.2.2` | MIT OR Apache-2.0 | RustCrypto shared types |
| `hybrid-array 0.4.14` | MIT OR Apache-2.0 | Fixed-size crypto arrays |
| `typenum 1.20.1` | MIT OR Apache-2.0 | Compile-time crypto sizes |
| `cfg-if 1.0.4` | MIT OR Apache-2.0 | Compile-time platform selection |
| `cpufeatures 0.3.0` | MIT OR Apache-2.0 | Fixed CPU feature selection used by SHA-2; no plugin/loading surface |
| `ed25519 3.0.0` | Apache-2.0 OR MIT | Signature type/trait support |
| `signature 3.0.0` | Apache-2.0 OR MIT | Signature signing and verification traits |
| `curve25519-dalek 5.0.0` | BSD-3-Clause | Reviewed curve arithmetic beneath Ed25519 |
| `subtle 2.6.1` | BSD-3-Clause | Constant-time primitives |
| `fiat-crypto 0.3.0` | MIT OR Apache-2.0 OR BSD-1-Clause | Generated verified field arithmetic selected by curve crate |
| `curve25519-dalek-derive 0.1.1` | MIT/Apache-2.0 | Compile-time derive used by curve implementation |
| `proc-macro2 1.0.107` | MIT OR Apache-2.0 | Offline derive compilation support |
| `quote 1.0.47` | MIT OR Apache-2.0 | Offline derive token generation |
| `syn 2.0.119` | MIT OR Apache-2.0 | Offline derive parsing |
| `unicode-ident 1.0.24` | (MIT OR Apache-2.0) AND Unicode-3.0 | Rust identifier parsing beneath proc macros |
| `rustc_version 0.4.1` | MIT OR Apache-2.0 | Locked curve build-time compiler capability selection; no fetch |
| `semver 1.0.28` | MIT OR Apache-2.0 | Version parsing for `rustc_version` |
| `sovereign-ai-contracts 3.0.0` at `f3ea60596c33faca673ad368304f162b30f2ac2f` | Apache-2.0 | Direct canonical Models 2.1 admission, Models 2.0 package identity, and provider-neutral Runtime V3 identity; only the narrow `model-admission-v1` feature and its models-v2/runtime-v3 dependencies are compiled, excluding cognitive/tool/memory/security authority modules |
| `serde 1.0.229` | MIT OR Apache-2.0 | Strict bounded runtime-v3 document serialization support beneath the canonical contract |
| `serde_core 1.0.229` | MIT OR Apache-2.0 | Serde core traits beneath the canonical contract |
| `serde_derive 1.0.229` | MIT OR Apache-2.0 | Offline compile-time derivation for closed canonical contract documents |
| `serde_json 1.0.151` | MIT OR Apache-2.0 | Strict JSON decoding/encoding for canonical runtime-v3 documents; no network or filesystem access |
| `itoa 1.0.18` | MIT OR Apache-2.0 | Integer serialization beneath `serde_json` |
| `memchr 2.8.3` | Unlicense OR MIT | Bounded byte scanning beneath `serde_json` |
| `zmij 1.0.23` | MIT | Numeric formatting beneath the locked `serde_json` implementation |
| `sha2 0.10.9` | MIT OR Apache-2.0 | Canonical runtime-v3 SHA-256 output and idempotency digest validation inside Sovereign Contracts |
| `digest 0.10.7` | MIT OR Apache-2.0 | SHA-2 0.10 trait/internal support beneath the canonical contract |
| `block-buffer 0.10.4` | MIT OR Apache-2.0 | SHA-2 0.10 fixed-block buffering |
| `crypto-common 0.1.7` | MIT OR Apache-2.0 | SHA-2 0.10 shared types |
| `generic-array 0.14.7` | MIT | Fixed-size digest arrays beneath SHA-2 0.10 |
| `cpufeatures 0.2.17` | MIT OR Apache-2.0 | Fixed CPU feature selection beneath SHA-2 0.10; no plugin/loading surface |
| `version_check 0.9.5` | MIT OR Apache-2.0 | Offline compiler capability selection beneath fixed-size crypto arrays |
| `syn 3.0.3` | MIT OR Apache-2.0 | Offline Serde derive parsing |

The proc-macro/build entries execute only while compiling the locked crypto
implementation. They do not fetch remote code, inspect model data, or ship as a
runtime model capability. Any version, feature, source, or package-set change
requires renewal of this record and the automated exact-allowlist test.

The four Unicode entries remain narrowly admitted supply-chain inputs. The
private scanner/BPE/decode core and its adapter add no package to this
inventory and cannot be selected or configured publicly. A narrow wrapper may
construct only the pinned full-hash-bound tokenizer profile. The adapter invokes
bounded NFC, stream-safe classification and General Category lookup. Exact
pinned-source conversion and fixed-corpus replay are feature-gated offline
diagnostics. Qwen2.5 construction is confined to the separately signed,
owner-approved, default-off ARM64 development path; CPU, Metal and production
execution remain explicitly unsupported.
Their resolved graph is
`engine-tokenizer -> unicode-normalization 0.1.24 -> tinyvec 1.12.0 ->
tinyvec_macros 0.1.1` plus the independent
`engine-tokenizer -> unicode-general-category 1.1.0` edge. Only
`tinyvec/alloc`, the upstream-selected empty `tinyvec/default`, and
`tinyvec/tinyvec_macros` are active; neither direct Unicode crate has an active
feature.

Both direct crates declare Unicode 16.0.0. The vendored normalization and
category table SHA-256 values are respectively
`69a0faad67afe03146d80ff5b2d889d1898cbfe8f7677a24f012148f6fd27348`
and `9a06bcac89a552d169aec1411438ab948391a218cbfef8c71bf186cd39ec5268`.
The normalizer's upstream archive identity remains unchanged, while the
retained-file map now binds the reviewed local hardening hashes
`06599747369f758a907d828712bd81ec3d7607c3ed6050283b844ea6700c105c`
for `src/decompose.rs` and
`c1b489f6402b96d3b6b55d7ccd08a22d066fc66b7c14331cf579d4c2d10bb963`
for `src/recompose.rs`.
The normalization archive's unused executable `scripts/unicode.py`, the
category archive's unused external generator `Makefile`, and ignored
package-local `Cargo.lock` files from the category and `tinyvec` archives were
pruned; Cargo's retained-file maps were regenerated while keeping the registry
archive hashes. An architecture test requires every remaining retained-file
path to exist in tracked release content.
No JavaScript source was present. `unicode-general-category/build.rs` is the
only new build-time surface: it deterministically reads bundled `src/tables.rs`
and writes `category.rs` only to `OUT_DIR`, without network, subprocess, native
link, dynamic loading, or unsafe code. `unicode-normalization` has no build
script; its reviewed `src/normalize.rs` Hangul boundary contains one unsafe
function, four unsafe expressions/blocks, and five range-guarded unchecked
scalar conversions. Both tinyvec crates forbid unsafe code. Full maintainer,
license, archive checksum, advisory, removal, and review-date evidence is in
[DEPENDENCY-POLICY.md](DEPENDENCY-POLICY.md).

`zeroize 1.8.2` was reviewed on 2026-08-11 as a RustCrypto-maintained memory
clearing primitive. It has no build script, network, filesystem, process,
dynamic-loading, AI, or model surface in this graph. Default and derive
features are disabled. Its narrow volatile-memory clearing behavior avoids an
unsafe and unreliable in-repository secret-erasure implementation; removal
is appropriate only if the response private key moves wholly into an
OS-backed signing service or hardware key.
