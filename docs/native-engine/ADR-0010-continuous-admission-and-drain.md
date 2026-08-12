# ADR-0010: Bounded continuous admission and graceful drain

- Status: Accepted
- Date: 2026-08-11

## Context

The exact-batch driver proves fair authenticated generation, cancellation,
output isolation, and cleanup, but it waits for a caller-declared connection
count before work begins and cannot admit a new request while a batch executes.
The owner CLI therefore remains on the earlier single-worker loop.

Continuous admission must not create an unbounded thread pool, let incomplete
handshakes stop generation, bypass durable replay, exceed scheduler/output
budgets, or leave workers and scheduler state behind during shutdown or error.

## Decision

Add a library-level `serve_continuous` reference driver with a validated maximum
of 64 simultaneously owned connections and a cloneable caller-owned shutdown
signal. The one ownership ceiling includes every stream in these phases:

1. accepted and authenticating;
2. authenticated and scheduler-queued;
3. actively generating; and
4. terminal with output/cancellation workers still draining.

The listener alone enters nonblocking mode for readiness-based admission. Every
accepted stream is explicitly restored to blocking I/O with the existing finite
read/write deadline. Each incomplete handshake runs in one fallibly created,
256 KiB-stack ingress worker; it cannot block decoder turns or another accept.
Authentication and durable nonce admission complete before the command enters
the scheduler.

Authenticated generation streams reuse ADR-0009's fixed output queue and
bounded writer plus the existing same-stream cancellation watcher. Requests can
remain in the bounded scheduler queue while active cursors advance one turn at a
time. All queued, active, and output retention limits still apply before cursor
construction. Sequence allocation advances once a sequence can appear in a
signed event, including a post-assignment rejection, so identifiers are never
reused.

On caller shutdown, the driver:

1. stops accepting new streams;
2. shuts down reads owned by incomplete ingress workers;
3. removes and emits signed cancellation failures for queued generation;
4. marks every active cursor for request-local cancellation;
5. continues bounded turns until terminal events are produced;
6. closes output queues and joins ingress, cancellation, and writer workers; and
7. verifies empty scheduler/coordinator state before restoring blocking listener
   mode and returning a bounded summary.

A state destructor performs the same stop-and-join cleanup on every error exit.
No worker is detached. No dependency, unsafe code, wire schema, control-plane
authority, filesystem scope, or outbound-network capability changes.

## Invariants

- Total owned connections never exceed the caller limit or the hard ceiling of
  64.
- Thread count and stack size are derived only from that fixed ownership bound.
- A client controls neither listener mode, worker count, queue capacity, nor
  shutdown.
- Authentication and synchronized replay admission precede scheduler mutation.
- New queued requests do not mutate active KV/session state.
- Shutdown and every error path join all owned workers and leave no queued or
  active scheduler request.
- Listener nonblocking state never leaks past the reference call.
- The Go platform remains the only routing, tool, approval, memory, and policy
  authority.

## Consequences

Deterministic Unix-socket tests prove two already-ready authenticated peers are
owned concurrently and complete with signed terminal events, the ownership peak
stays within its exact limit, caller shutdown returns an empty scheduler, and an
incomplete handshake is interrupted rather than waiting for the normal socket
deadline.

ADR-0011 subsequently exposes this driver through an explicit bounded owner-CLI
mode and translates `SIGINT`/`SIGTERM` into the caller-owned shutdown signal.
Peer-credential qualification, operated continuous/slow-reader soak,
production-model latency evidence, and reversible Go-provider canary remain
required before production cutover.

## Rejected alternatives

- **Thread per connection without a shared ownership ceiling:** permits
  unbounded stack, descriptor, and queue growth.
- **Authenticate on the decoder thread:** lets a silent peer hold all generation
  for the socket deadline.
- **Detach workers at shutdown or error:** leaves replay, socket, and scheduler
  lifetime uncertain.
- **Inherit listener nonblocking mode on streams:** turns a temporary
  `WouldBlock` into false disconnect/cancellation.
- **Activate the CLI implicitly:** would replace the established single-worker
  behavior without an explicit owner choice or reviewed shutdown bridge.
