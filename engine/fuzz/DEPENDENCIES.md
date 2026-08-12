# Fuzz-only Dependency Record

- Reviewed: 2026-08-12
- Scope: `engine/fuzz` only; excluded from every production binary, lockfile,
  vendor directory, release package, and SBOM
- Registry source: `https://crates.io`; exact archive checksums are pinned in
  `Cargo.lock`

## Direct runtime

| Package | Purpose | Features | License | Source and attack surface |
| --- | --- | --- | --- | --- |
| `libfuzzer-sys 0.4.13` | Rust binding and bundled LLVM libFuzzer runtime | defaults off; `link_libfuzzer` only | `(MIT OR Apache-2.0) AND NCSA` | `rust-fuzz/libfuzzer`; links only to fuzz binaries, contains external unsafe/native C++, and its build script invokes the configured C++ compiler without network access |

The standard library cannot provide coverage feedback, sanitizer integration,
corpus mutation, or crash minimization. This dependency is therefore approved
only for defensive test binaries and has no production replacement burden; it
is removed by deleting the excluded fuzz workspace.

## New transitive packages

| Package | Purpose | License | Source and attack surface |
| --- | --- | --- | --- |
| `arbitrary 1.4.2` | byte-input support required by `libfuzzer-sys` | MIT OR Apache-2.0 | `rust-fuzz/arbitrary`; pure Rust parsing helpers, no I/O |
| `cc 1.4.2` | compiles bundled libFuzzer C++ during fuzz builds | MIT OR Apache-2.0 | `rust-lang/cc-rs`; build-only process/environment/compiler access |
| `find-msvc-tools 0.1.10` | Windows compiler discovery for `cc` | MIT OR Apache-2.0 | `rust-lang/cc-rs`; build-only tool discovery, inactive on current Unix CI |
| `jobserver 0.1.35` | bounded compiler job coordination for `cc` | MIT OR Apache-2.0 | `rust-lang/jobserver-rs`; build-only environment and file-descriptor access |
| `shlex 2.0.1` | compiler argument parsing for `cc` | MIT OR Apache-2.0 | `comex/rust-shlex`; build-only string parsing |
| `getrandom 0.4.3` | target-specific transitive support retained by the lock | MIT OR Apache-2.0 | `rust-random/getrandom`; OS randomness on applicable targets, not active in current fuzz tree |
| `r-efi 6.0.0` | target-specific EFI definitions retained by the lock | MIT OR Apache-2.0 OR LGPL-2.1-or-later | `r-efi/r-efi`; data definitions only, inactive in current fuzz tree |

All other registry packages in the fuzz lock are exact versions already
reviewed in `docs/native-engine/DEPENDENCY-INVENTORY.md` for the production
path crates exercised by the targets.

The directory-manifest, tensor-index, safetensors-import, package-path, AIPTok
parser, encode, decode, incremental-state, and authenticated service-envelope
targets add only first-party path dependencies plus a direct declaration of
the already production-reviewed `ed25519-dalek 3.0.0` used to re-sign mutated
local envelopes. They add no external package to either lockfile and do not
expand the production graph.

## Orchestrator

CI installs `cargo-fuzz 0.13.2` with its published lockfile using pinned
`nightly-2026-08-01`. It is an MIT OR Apache-2.0 command-line test orchestrator
from `rust-fuzz/cargo-fuzz`, not a crate dependency and not a release artifact.
It can invoke Cargo, the compiler, sanitizers, and fuzz binaries and can write
corpus/artifact files; those powers are confined to the bounded CI job and
local developer invocation. It does not run inside the engine.
