# ADR-0008: Bounded authenticated multi-connection serving

- Status: Accepted
- Date: 2026-08-11

## Context

ADR-0007 introduced request-local generation cursors and an engine-owned
round-robin coordinator, but the Unix service still exercised only one
connection at a time. Activating an endless concurrent accept loop before
connection admission, cancellation, signed response ordering, failure
isolation, and cleanup were proven together would expand the production daemon
boundary prematurely.

An unbounded thread per connection is prohibited. A slow or malformed client
must not gain scheduling authority, duplicate an active identity, bypass
durable replay admission, or corrupt another sequence's scheduler state.

## Decision

Add a bounded reference `serve_batch` path to the Rust runtime. Its caller
declares an exact positive connection count no greater than the configured
scheduler active limit or the independent 64-connection batch ceiling. The
runtime must be idle at entry. It accepts exactly
that many protected Unix streams, then:

1. strictly decodes and authenticates each first command;
2. durably admits its nonce before command execution;
3. rejects unknown models, duplicate identities, exhausted limits, and invalid
   requests per connection;
4. conservatively charges active request copies, worst-case input-token storage,
   and cursor overhead to the configured retained-byte ceiling, then creates
   each cursor only after scheduler and coordinator admission;
5. runs active cursors through the one-token round-robin coordinator;
6. writes and flushes at most one signed, ordinal-bound event before another
   coordinator turn;
7. uses at most one cancellation watcher per admitted active stream, bounded by
   the 64-connection ceiling and an explicit 256 KiB stack per watcher; and
8. removes every terminal request from both the coordinator and scheduler and
   shuts down its inbound watcher before returning ordered per-connection
   results.

The cursor owns a validated copy of its request. The coordinator owns a shared
request-local cancellation handle, so neither depends on unstable container
addresses. Model state remains immutable and shared; decoder sessions and KV
caches remain sequence-local and charged to the existing aggregate model pool.

No wire schema, dependency, listener type, authority boundary, or outbound
capability changes. The CLI continues using the established single-connection
loop. `serve_batch` is an integration and qualification boundary, not the final
continuous daemon.

## Invariants

- Batch and active counts are nonzero and hard bounded.
- Aggregate active request/token/cursor retention is checked before tokenization
  and cannot exceed the configured maximum or the 64 MiB hard ceiling.
- Authentication failure receives no trusted response; an authenticated
  request rejection receives a signed failed event when its stream remains
  writable.
- Per-stream event ordinals begin at one and advance with checked arithmetic.
- A cancellation, decoder failure, malformed peer, disconnect, or event-write
  failure cannot abort a neighboring admitted sequence.
- Durable replay state is shared by control-key identity and admitted before
  either generation or cancellation changes scheduler state.
- Thread creation is bounded and fallible. A partial watcher-start failure
  stops watchers and removes all active scheduler state before returning.
- The batch path grants no routing, model installation, tool, approval, memory,
  Agent Green, filesystem-discovery, or Internet authority.

## Consequences

Deterministic Unix-socket tests now prove two signed streams complete through
the same coordinator, authenticated cancellation fails only its target,
malformed authentication does not block a valid neighbor, disconnect/write
failure is request-local, signatures and ordinals remain per stream, and the
scheduler is empty afterward.

The exact-batch reference still has deliberate production limitations. Accept
waits for the declared count and initial command reads are sequential with
finite timeouts. ADR-0009 adds bounded per-stream output isolation without
changing this decision, and ADR-0010 adds a separate continuous reference
driver with graceful drain. Operated multi-client and slow-reader soak plus a
reviewed external shutdown integration remain required before the CLI switches
from `serve_one`.

## Rejected alternatives

- **Immediately replace the CLI loop:** lacks continuous-ingress, shutdown, and
  slow-consumer qualification.
- **One unbounded worker thread per client:** defeats deterministic scheduling
  and hard resource ceilings.
- **Buffer unbounded events per stream:** transfers memory-denial control to a
  client and obscures backpressure.
- **Share one cancellation flag or decoder session:** permits cross-request
  interference and breaks sequence isolation.
