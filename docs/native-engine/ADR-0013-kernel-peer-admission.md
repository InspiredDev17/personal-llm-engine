# ADR-0013: Same-service-user kernel peer admission

- Status: Accepted
- Date: 2026-08-12

## Context

The engine socket is a mode-0600 inode in an owner-private runtime directory,
and every control command is strictly signed and durably replay-checked. The
transport did not independently inspect the kernel identity attached to an
accepted Unix stream. That left the same-service-user deployment assumption
implicit and delayed rejection until frame authentication.

Peer identity cannot come from a request field, model package, environment
claim, pathname, or unsigned message. A credential mismatch must be rejected
before allocating or parsing a peer-controlled frame, and one rejected stream
must not terminate continuous service for authorized neighbors.

## Decision

At listener bind, record the Rust process effective user ID. For every accepted
Unix stream, query the kernel for the peer effective user ID and require exact
equality before configuring frame I/O or reading any bytes. Linux and Android
use `getsockopt(SOL_SOCKET, SO_PEERCRED)` with exact `ucred` storage. Apple and
reviewed BSD targets use `getpeereid`. Other Unix targets fail closed with an
explicit not-implemented error until their credential ABI is reviewed.

The blocking listener discards a mismatched peer and continues waiting for an
authorized peer. Nonblocking continuous admission records the accepted stream
as failed, drops it, and continues polling without mutating the scheduler. The
kernel check is defense in depth, not a replacement for control signatures,
key-ID binding, nonce journaling, request validation, or socket permissions.

The reference deployment profile therefore requires the Go control plane and
Rust engine to run with the same effective service user. A future distinct-user
profile requires an explicit socket group/ACL design and a new ADR; it may not
disable peer inspection or accept a client-asserted identity.

## FFI boundary

All calls live in `engine-server/src/peer_credential.rs`, which locally allows
unsafe code while the workspace denies it elsewhere. Four documented unsafe
blocks cover: the no-argument process `geteuid` call; Linux/BSD credential
queries selected exclusively at compile time; and Linux initialization after
the kernel reports the exact `ucred` size. Descriptors are borrowed from live
`UnixStream` values, output pointers refer to aligned exact ABI storage, no call
retains a pointer, and no raw descriptor or mutable credential is exposed.

Architecture tests require exactly four unsafe blocks and one unsafe-lint
allowance in this module. The existing `libc 0.2.189` dependency is reused with
default features disabled; there is no new package, wire field, listener,
network client, dynamic loading, build script, or authority path.

## Invariants

- The allowed identity is derived only from the engine process at bind.
- A model, client, frame, package, or command-line flag cannot choose it.
- Peer credentials come only from the connected stream's kernel metadata.
- Mismatch or unavailable credentials fail before frame allocation and parsing.
- Continuous rejection does not admit, queue, activate, or cancel any request.
- Socket mode, Ed25519 authentication, replay protection, and all request bounds
  remain mandatory independent controls.
- Go remains the only routing, tool, approval, memory, policy, and Agent Green
  authority.

## Evidence

Real Unix-stream tests prove the current process identity is accepted, signed
frame round trips still work, and an intentionally mismatched listener policy
rejects the connected peer before a frame read. The complete authenticated
runtime, multi-stream, shutdown, and delayed-reader suites exercise the same
admission path. Architecture tests lock the isolated FFI surface.

## Rejected alternatives

- **Trust a UID or PID sent by the client:** it is peer-controlled data, not a
  kernel-authenticated identity.
- **Rely only on mode 0600:** strong but implicit; the credential check makes the
  service-user invariant explicit at the accepted descriptor.
- **Use PID as the stable authorization identity:** ordinary Go restarts change
  PID and would require a mutable out-of-band update without adding protection
  beyond the mandatory signing identity.
- **Permit an arbitrary UID flag now:** mode-0600 socket access and the current
  deployment profile are same-user; advertising a distinct-user switch without
  a reviewed group/ACL design would be misleading.
- **Treat a mismatch as a daemon-fatal error:** lets one rejected local attempt
  interrupt unrelated authorized inference.
