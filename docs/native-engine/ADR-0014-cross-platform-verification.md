# ADR-0014: Cross-platform offline verification gate

- Status: Accepted
- Date: 2026-08-12

## Context

The workspace has complete local quality commands, scheduled parser fuzzing,
and a scheduled continuous-service soak. It did not have an ordinary push and
pull-request workflow that compiled and tested the entire Rust workspace on
both reviewed Unix peer-credential implementations. Local macOS evidence does
not compile Linux `SO_PEERCRED`, while a Linux-only check does not compile the
Apple/BSD `getpeereid` branch.

An ordinary hosted check must not turn into a dependency-fetching or privileged
integration path. It must use the checked-in vendor graph, carry no repository
write permission, receive no production secrets or model data, and terminate
within a fixed bound.

## Decision

Add `native-engine-verify`, triggered by pushes to `main` and the native-engine
integration branch, pull requests, and explicit dispatch. Run the same job on
fixed `ubuntu-24.04` and `macos-14` runner labels with a 30-minute job bound,
read-only repository permission, concurrency cancellation, an immutable
checkout action revision, and Rust 1.85.0 with only Clippy and rustfmt added.

Every Cargo operation is locked and explicitly offline where supported. The
workflow validates metadata and the resolved feature graph, formatting,
all-target/all-feature check and build, Clippy with warnings denied, all tests,
documentation tests and output, the no-default-feature profile, and a clean
repository diff. The workspace source replacement therefore uses only
`engine/vendor`; the workflow does not install an unreviewed Cargo utility or
query a package registry.

The existing local/release `cargo audit --no-fetch` gate remains separate. Its
advisory database and executable are operator-managed inputs that are not part
of the vendored production dependency graph, so the hosted workflow does not
download either one implicitly.

## Invariants

- The workflow has read-only repository permission and no secrets.
- It does not use `pull_request_target`, elevated commands, general-purpose
  download commands, service containers, or external model data. The exact
  Rust toolchain bootstrap is the only intentional hosted download.
- Both operating-system jobs run the complete workspace tests, including real
  local Unix sockets and kernel peer-credential admission.
- Cargo cannot contact a registry or alter the exact lockfile resolution.
- Each job has a finite timeout and superseded runs are cancelled.
- A command failure fails the job; no step may continue on error.
- Hosted success is claimed only after GitHub reports a completed run for the
  exact commit. Local workflow-policy tests prove configuration, not execution.

## Evidence boundary

This gate provides compiler and deterministic synthetic runtime coverage on
ordinary Linux and macOS hosts. It does not prove production supervisor
configuration, service-user separation, outbound-network denial, secret
injection, production-model behavior, performance, thermal limits, Metal, or
CUDA. The hosted runner image and Rust distribution remain external CI inputs,
not production runtime dependencies. Those operated release gates remain.

## Rejected alternatives

- **Test only on the developer host:** leaves the other credential ABI
  uncompiled and vulnerable to silent drift.
- **Fetch crates during CI:** weakens the checked-in offline release graph and
  makes an external registry part of every verification run.
- **Install `cargo-audit` dynamically:** adds a mutable build-time tool path;
  advisory review remains an explicit offline release operation.
- **Grant write permission:** verification has no legitimate repository
  mutation requirement.
- **Use `pull_request_target`:** untrusted change validation does not require
  privileged base-repository context.
