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

The inference service has write access to neither rollback nor event state and
cannot read the private rollback ledger. Under the current reviewed same-group
deployment it can read the content-free event records, but cannot alter their
root-owned directory or files and gains no launch or acknowledgement authority.
A Go control-plane adapter reads the group-readable, root-owned event ring,
validate owner, mode, identity, grammar, and sequence, and deliver the mapped
category through its existing audit and Agent Green boundary. Go never writes
or removes the ring. Its acknowledgement cursor is in memory, so process
restart may replay retained events; Agent Green's authenticated replay controls
remain the durable receiver boundary.

## Bounds and failure behavior

- record size: 76 bytes;
- retained slots: 64;
- event kinds: seven closed pre-launch categories;
- sequence: nonzero `u64`, checked and never wrapped;
- unknown release identity: exactly zero generation and all-zero digest;
- verified identity: nonzero generation and nonzero digest;
- unknown files, kinds, versions, flags, duplicates, discontinuities, unsafe
  ownership/modes, symlinks, truncation, and trailing bytes fail closed.

During one Go process lifetime, ring overrun before acknowledgement is reported
as a lost sequence and is not silently skipped. Because the Go cursor is not
yet durable, more than 64 failures while Go is stopped can overwrite older
records without proving whether they were previously delivered; that remains
an operated retention gate. The ring is defensive telemetry, not authorization:
neither Go, Agent Green, the client, nor a model can use it to install, select,
approve, update, or launch a release.

## Consequences

The repository now provides the Linux pre-launch binding and typed event
bridge in source. It does not prove OS authentication of the verifier binary,
release-key custody, systemd behavior on the target host, service-account/group
provisioning, crash recovery, event delivery while Go is offline, or operated
upgrade/tamper/rollback evidence. Those remain release qualification gates.
