# ADR-0004: Exclusive Replay-Journal Ownership

- Status: Accepted
- Date: 2026-08-11

## Context

Durable nonce records reject replay across connections and restarts, but two
live engine processes could otherwise open the same control-key journal and
build independent in-memory indexes. Distinct configured socket names in one
runtime root would then permit both processes to accept the same authenticated
nonce before either observed the other's append.

Rust 1.85 is the workspace MSRV and does not provide a stable standard-library
file-lock API. A create-only lock file would remain stale after a crash; PID
files cannot safely distinguish PID reuse; and adding a locking crate would
expand the dependency graph for one operating-system call.

## Decision

Hold a nonblocking exclusive advisory `flock` on the already-open per-key replay
journal for the entire `DurableReplayJournal` lifetime. Runtime construction
fails closed if another owner holds the lock. Normal descriptor close releases
the lock after orderly exit or process death, so no stale lock artifact or
recovery deletion protocol is required.

The call is isolated in `engine-server/src/os_lock.rs`. Workspace policy denies
unsafe code by default; that module alone permits it. Architecture tests reject
unsafe blocks and unsafe-lint allowances in every other Rust source.

## Safety invariants

1. The wrapper owns a live `std::fs::File` before obtaining its raw descriptor.
2. `flock` receives an integer descriptor only; it receives and retains no
   pointer or Rust reference.
3. The result is checked before the safe wrapper is constructed.
4. The wrapper exposes no raw descriptor, underlying file, clone, or unlock
   operation. Its bounded read/write/seek/sync methods delegate to the owned
   file.
5. `O_CLOEXEC` prevents descriptor inheritance across an unexpected exec, and
   the engine has no process-execution capability.
6. Dropping the sole owned descriptor releases the operating-system lock.

Focused tests start a separate test process, prove that the competitor fails
closed while the first owner lives, then prove the journal reopens after the
owner exits. Existing tests continue to cover no-follow opening, file mode,
corruption, bounds, persistence, and key scoping.

## Consequences

The same control-key journal has exactly one live process owner, closing the
known concurrent-index replay race without a new dependency or wire change.
The lock is advisory, so all supported engine entrypoints must continue to use
`DurableReplayJournal`; arbitrary owner processes remain outside the engine's
threat boundary. Each future operating-system target and any future unsafe FFI
must receive separate review and tests.
