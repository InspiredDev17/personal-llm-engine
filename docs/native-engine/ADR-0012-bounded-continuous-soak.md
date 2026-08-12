# ADR-0012: Bounded continuous delayed-reader and restart soak

- Status: Accepted
- Date: 2026-08-11

## Context

The continuous server has focused tests for multi-stream completion, caller
shutdown, incomplete authentication, request-local cancellation, queue
saturation, and worker cleanup. Those tests do not repeatedly combine delayed
reader behavior, concurrent authenticated neighbors, graceful drain, runtime
teardown, and a fresh restart.

An unbounded or production-size soak is inappropriate for every developer test
run and hosted CI. A skipped or ignored test would also allow the reference path
to regress silently.

## Decision

Add one normal, non-ignored real-Unix-socket soak test to `engine-server`. It
runs eight sequential cycles. Every cycle creates a fresh runtime, signed
synthetic package, replay journal, and listener; admits six separately signed
generation streams with a three-active-sequence ceiling; and prevents one
client from reading any response until all five fast neighbors have received
verified terminal events.

The delayed client has a fixed five-second release bound so a regression cannot
hang the suite indefinitely. All test clients have fixed read/write timeouts.
After the delayed reader receives its signed terminal event, the coordinator
requests the existing graceful drain. The test requires exactly six accepted,
authenticated, and completed commands, zero failures, ownership never above six,
empty queued/active scheduler state, joined client/coordinator threads, and
complete runtime-directory cleanup in every cycle.

Add a weekly and manually dispatchable GitHub workflow that installs the pinned
Rust 1.85 toolchain and invokes the exact test eight times. One scheduled job is
therefore fixed at 64 fresh-runtime cycles and 384 synthetic generations and is
bounded by a 15-minute job timeout. Cargo remains locked and source-replaced by
the checked-in vendor directory.

The exact eight-pass workflow command must pass locally before commit. GitHub's
cron execution begins only when the workflow is present on the repository's
default branch, so a feature-branch commit proves configuration and local
reproduction, not a completed hosted run.

## Invariants

- The soak uses only deterministic Rust-generated test packages and public test
  keys; it reads no production model, credential, prompt, or journal.
- Every stream uses the existing authenticated protocol, unique nonzero request
  ID and nonce, strict response signature checks, and exact event ordinals.
- The delayed reader cannot block neighbor completion or grant authority.
- Counts, active work, connection ownership, client I/O time, cycle count,
  scheduled repetitions, and overall workflow time are finite constants.
- Every client and coordinator thread is joined, including after a neighbor
  reports failure; no short-circuit can detach remaining test workers.
- Every cycle drains the scheduler and removes the socket, replay journal, and
  private runtime directory before the next restart.
- The test adds no production dependency, unsafe code, protocol field, network
  client, TCP listener, tool path, or control-plane authority.

## Evidence boundary

This gate proves repeatable synthetic reference-runtime behavior on ordinary CI
hosts. It is not production-size or operated-host evidence. It does not measure
real-model latency, memory overhead, throughput, thermal behavior, long-context
pressure, supervisor policy, service-user isolation, key injection/recovery, or
accelerator behavior. Those remain cutover gates.

## Rejected alternatives

- **Ignore the soak by default:** makes the most relevant lifecycle interaction
  optional and easy to miss.
- **Use sleeps as the success criterion:** introduces timing flakiness; the test
  uses channel ordering and timeouts only as failure bounds.
- **Leave failed client handles detached:** can leak sockets across cycles and
  hide cleanup failures; every handle is joined unconditionally.
- **Run without hard repetition/time limits:** turns a reliability gate into an
  availability risk for local and hosted builds.
- **Claim production qualification:** synthetic tiny-model CI cannot substitute
  for operated production-model and hardware evidence.
