# Native Engine Release Readiness

## Verdict: NO-GO

The 2026-08-13 source hardening gates pass, but this revision is not approved
for production release. A failed requirement is not converted into a
documentation exception.

## Evidence that passed

| Gate | Evidence |
| --- | --- |
| Rust correctness | Rust 1.85.0 workspace tests, all targets/features, Clippy with warnings denied, formatting, documentation tests, and documentation build passed on the Apple Silicon host |
| Go boundary | `go test ./...`, `go test -race ./...`, `go vet ./...`, and `go mod verify` passed for Go integration baseline `3c7c082` |
| Parser smoke fuzzing | All nine requested parsing/validation boundaries ran for five seconds each; no crash, timeout, sanitizer, or resource artifact was produced |
| Metal correctness | Six direct-GPU suites passed with Metal API Validation and GPU Validation enabled |
| Native bridge | Objective-C++ vector-add lifecycle passed with AddressSanitizer and UndefinedBehaviorSanitizer; macOS LeakSanitizer was unavailable |
| Local transport isolation | macOS sandbox proof allowed the configured UDS and denied live loopback TCP |
| Package trust | Production directory packages require exact registered identity, SHA-384 catalog/file/tensor verification, active trusted Ed25519 signer, compatibility checks, and data-only contents |
| Dependency integrity | Exact Cargo lock, checked-in vendor tree, dependency inventory, SPDX SBOM, boundary equality tests, `cargo audit --no-fetch`, and Go module verification passed |

## Release blockers

1. The source tree now has a first-party release-only signed manifest, exact
   installed-artifact verifier, and monotonic rollback ledger. Production is
   still blocked until the verifier itself is OS-authenticated, an installer
   invokes it atomically before launch, release-key custody is reviewed, and an
   operated target proves upgrade, same-version restart, tamper rejection, and
   rollback rejection. Package signing still does not sign the executable.
2. The repository does not declare an owner-selected license for its own Rust
   workspace crates. `cargo-deny` therefore cannot produce a clean complete
   license verdict without either a real license decision or an unsafe ignore.
3. A registered signed synthetic package now runs end-to-end through persistent
   Metal weights, device KV, sampling, and typed streaming with scalar parity.
   Trusted UDS Metal selection and exact Go descriptor pinning are implemented,
   but production routing remains blocked pending a reversible operated canary,
   production-size qualification, and rollback evidence.
4. No legally approved production model, production tokenizer vectors,
   production-size context soak, quality evaluation, or thermal/tokens-per-
   second evidence exists.
5. Linux systemd confinement is statically tested but was not operated on this
   macOS host. macOS `sandbox-exec` is deprecated and remains a development
   proof, not a signed production App Sandbox/launcher qualification.
6. Miri could not run because the pinned nightly lacks the Miri component.
7. The release verifier rejects binary/metallib mismatch and downgrade, but a
   qualified supervisor-to-Go event source for those pre-launch failures and
   model quarantine is not implemented.
8. The audited published Go integration baseline and the separate local Go
   security history must be deliberately reconciled and retested as one exact
   release revision; this audit does not silently merge divergent histories.

## Promotion conditions

A future verdict may become GO only after every blocker above is resolved,
the exact candidate is rebuilt offline from a clean checkout, signed artifacts
are independently verified, all automated gates are rerun, and operated
no-egress/restart/rotation/rollback evidence is attached. Any source, lockfile,
toolchain, model, signer, compiler, OS, or Metal-driver change invalidates the
corresponding evidence.

## Exact audit commands

```sh
cd engine
cargo +1.85.0 fmt --all --check
cargo +1.85.0 test --workspace --all-targets --all-features --locked --offline
cargo +1.85.0 clippy --workspace --all-targets --all-features --locked --offline -- -D warnings
cargo +1.85.0 test --doc --workspace --all-features --locked --offline
cargo +1.85.0 doc --workspace --all-features --no-deps --locked --offline
cargo audit --no-fetch --file Cargo.lock
MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1 \
  cargo +1.85.0 test -p personalai-engine-metal --all-features --locked --offline

cd "$GO_REPOSITORY"
gofmt -l .
go test ./...
go test -race ./...
go vet ./...
go mod verify
```

The fuzz and native sanitizer commands are preserved in
[FUZZING.md](FUZZING.md) and [SECURITY-AUDIT.md](SECURITY-AUDIT.md).
