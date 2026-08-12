# ADR-0009: Bounded per-stream output isolation

- Status: Accepted
- Date: 2026-08-11

## Context

ADR-0008 introduced fair multi-stream CPU generation, but its coordinator wrote
each signed event directly to the corresponding Unix stream. A peer that stopped
reading could therefore hold the coordinator inside one finite socket write and
delay every otherwise independent sequence until the write completed or timed
out.

Unbounded buffering and an unbounded worker per event are prohibited. Output
backpressure must be charged before generation, preserve exact per-stream event
ordering, and fail only the affected request.

## Decision

The bounded reference batch assigns each admitted generation stream:

1. one fixed-capacity queue containing at most eight complete signed event
   frames;
2. one fallibly created output writer with an explicit 256 KiB stack;
3. one request-local writer-failure flag; and
4. a conservative reservation for ten maximum-size generation-event frames:
   eight queued, one held by the writer, and one transient enqueue candidate.
   The reservation is charged to the existing aggregate coordinator retention
   ceiling before the cursor is constructed.

The coordinator signs and enqueues at most one event during a turn. It never
waits for socket output. A full or disconnected queue becomes an explicit
request-local error and sets only that cursor's cancellation flag. The output
writer preserves channel order, uses the transport's finite write deadline, and
drains accepted frames before it exits. Terminal cleanup closes the sender,
stops the cancellation watcher, joins both bounded workers, and replaces an
apparent success with the writer error if delivery failed asynchronously.

Thread handle and input allocations are fallible and occur before any worker is
started. Partial thread-start failure closes every queue, stops every watcher,
joins started workers, and clears scheduler state before returning failure.

No dependency, unsafe code, wire schema, listener type, routing authority,
model capability, filesystem authority, or outbound-network capability changes.

## Invariants

- Queue capacity is exactly eight frames per active stream and cannot be
  selected by a model or client.
- Maximum queued, in-flight, and transient enqueue retention is included in
  admission before tokenization or cursor construction.
- Event nonce, ordinal, signature, and byte order are unchanged.
- One queue or writer failure cannot stop a neighboring decoder cursor or output
  writer.
- No event queue, output thread, cancellation watcher, or scheduler entry
  survives terminal or error cleanup.
- Socket writes remain finite and local to the protected Unix-domain transport.

## Consequences

The exact-batch reference path now has an output-readiness equivalent that keeps
generation scheduling independent of socket write latency. Deterministic tests
prove queue saturation and writer disappearance fail closed, aggregate queue
retention is charged before cursor creation, real signed multi-stream delivery
still completes, and a disconnected peer does not abort its neighbor.

ADR-0010 reuses this output isolation in a bounded continuous reference driver.
The CLI remains single-worker until external shutdown integration and operated
continuous/slow-reader soak qualify that reference path.

## Rejected alternatives

- **Synchronous writes in the coordinator:** preserves bounds but lets one peer
  delay unrelated cursor turns.
- **Unbounded per-stream channels:** gives a non-reading client control over
  resident memory.
- **Dropping events while reporting success:** violates signed stream and usage
  semantics.
- **One thread per event:** has no deterministic thread ceiling and loses
  per-stream ordering.
