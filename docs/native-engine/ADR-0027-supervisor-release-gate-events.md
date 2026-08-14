# ADR-0027: Supervisor release gate and pre-launch events

- Status: Accepted
- Date: 2026-08-14

## Context

The release verifier existed as an operator command, but the checked-in Linux
service could start without it. A verification failure also occurs before the
engine process exists, so the engine's authenticated UDS event queue cannot
report that failure. Giving the engine write access to supervisor evidence
would let a compromised data plane forge or suppress the signal.

## Decision

Linux uses a separate root `Type=oneshot` verifier unit ordered before and
required by the unprivileged engine unit. It invokes `verify-supervised` with
fixed owner-controlled paths, the release-only public key, a protected
rollback-state root, and a distinct event root. Any verification, trust,
resource, ledger, or rollback error returns nonzero and keeps the engine
stopped. The engine executable is launched only from the verified read-only
release directory. The verifier executable itself remains outside that
directory and must be authenticated by the OS installer/package trust path;
self-verification is deliberately not claimed.

The verifier remains active only as a dependency marker while the engine is
running. `PartOf=personalai-engine.service` propagates every engine stop or
restart to that marker, so a subsequent start reruns verification rather than
reusing stale success.

Before returning a verification failure, the verifier writes one fixed-width
`PAIRSEV1` record to a 64-slot root-owned ring. Records contain only version,
closed event kind, nonzero sequence, and an optional already-verified release
generation plus SHA-384 manifest digest. They contain no path, diagnostic,
prompt, model output, credential, or key. Atomic replacement and directory
synchronization publish a slot. A separate protected create-new lock
serializes writers. If event retention itself fails, launch still fails closed.
A fixed-width `PAIRSEQ1` checkpoint in the disjoint root-owned state directory
retains the highest issued event sequence. The verifier writes the ring record
before advancing that checkpoint, adopts at most one valid interrupted record,
and never reuses its sequence. Clearing or rolling back the bounded ring alone
therefore cannot restart event identity at one. Corrupt, writable, malformed,
or excessively advanced checkpoint/ring state fails closed.

The inference service has write access to neither rollback nor event state and
cannot read the private rollback ledger. Under the current reviewed same-group
deployment it can read the content-free event records, but cannot alter their
root-owned directory or files and gains no launch or acknowledgement authority.
A Go control-plane adapter reads the group-readable, root-owned event ring,
validate owner, mode, identity, grammar, and sequence, and deliver the mapped
category through its existing audit and Agent Green boundary. Go never writes
or removes the ring. The Go control plane now persists only the fixed source ID
and last completed sequence in PostgreSQL. It loads that cursor before a new
scan and advances it by exact compare-and-swap only after Agent Green delivery
and content-free audit. This downstream durability does not give Rust, Go,
Agent Green, clients, or models write authority over supervisor evidence.
The nonzero supervisor sequence also maps to one stable content-free Go event
and audit correlation UUID, so a retry does not acquire a new logical identity.
Go integration `1951a42` additionally persists a source-bound provider
quarantine before exposing each supervisor event, blocks native admission and
calls across restarts, and permits only exact-revision owner recovery after the
durable delivery cursor has acknowledged the trigger.

## Bounds and failure behavior

- record size: 76 bytes;
- retained slots: 64;
- event kinds: seven closed pre-launch categories;
- sequence: nonzero `u64`, checked and never wrapped;
- sequence checkpoint: fixed 20-byte `PAIRSEQ1` state, written durably outside
  the ring and engine runtime;
- unknown release identity: exactly zero generation and all-zero digest;
- verified identity: nonzero generation and nonzero digest;
- unknown files, kinds, versions, flags, duplicates, discontinuities, unsafe
  ownership/modes, symlinks, truncation, and trailing bytes fail closed.

Ring overrun before acknowledgement is reported as a lost sequence and is not
silently skipped, including across Go restarts. Empty/reset state after an
acknowledgement, sequence rollback, a missing exact successor, database failure,
and concurrent cursor advancement all fail visibly. A crash after authenticated
delivery but before cursor commit can repeat one event under the explicit
at-least-once contract. The ring is defensive telemetry, not authorization:
neither Go, Agent Green, the client, nor a model can use it to install, select,
approve, update, or launch a release.

## Consequences

The repository now provides the Linux pre-launch binding, monotonic event
identity checkpoint, and typed event bridge in source. The Go control plane now
provides durable provider quarantine and audited owner recovery in source. This
does not prove OS authentication of the verifier binary, release-key custody,
systemd behavior on the target host, service-account/group provisioning,
owner-authorized ring/cursor disaster recovery, operated Agent Green delivery,
or operated upgrade/tamper/rollback evidence. Those remain release
qualification gates.
