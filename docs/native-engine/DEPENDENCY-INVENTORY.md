# Resolved Dependency Inventory

Source for every entry: locked crates.io registry with checksum in `Cargo.lock`.
No entry supplies AI, tensor, tokenizer, model-loading, scheduling, serving,
networking, dynamic loading, or GPU kernels.

| Package | License | Purpose and attack-surface justification |
| --- | --- | --- |
| `sha2 0.11.0` | MIT OR Apache-2.0 | Direct mature SHA-384 implementation; pure digest surface |
| `ed25519-dalek 3.0.0` | BSD-3-Clause | Direct strict Ed25519 verification and engine-response signing; only the `zeroize` feature is enabled |
| `zeroize 1.8.2` | Apache-2.0 OR MIT | Direct audited clearing of the owner-provided response-key seed and Ed25519 signing state; no derive feature |
| `libc 0.2.189` | MIT OR Apache-2.0 | Direct Unix constants `O_NOFOLLOW`/`O_CLOEXEC`; no direct FFI or unsafe block |
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
