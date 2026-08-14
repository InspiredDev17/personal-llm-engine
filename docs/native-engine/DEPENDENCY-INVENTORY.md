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
| `sovereign-ai-contracts 3.0.0` at `138128520dea7a602c0cf2ea905fa32396917db8` | Apache-2.0 | Direct canonical provider-neutral runtime-v3 data contract; only the minimal `runtime-v3` feature is compiled, excluding cognitive/tool/memory/security authority modules |
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

`zeroize 1.8.2` was reviewed on 2026-08-11 as a RustCrypto-maintained memory
clearing primitive. It has no build script, network, filesystem, process,
dynamic-loading, AI, or model surface in this graph. Default and derive
features are disabled. Its narrow volatile-memory clearing behavior avoids an
unsafe and unreliable in-repository secret-erasure implementation; removal
is appropriate only if the response private key moves wholly into an
OS-backed signing service or hardware key.
