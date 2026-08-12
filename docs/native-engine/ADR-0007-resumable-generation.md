# ADR-0007: Resumable token-quantum generation

- Status: Accepted
- Date: 2026-08-11

## Context

The CPU reference runtime currently processes one authenticated connection and
one generation request synchronously. Its original generation loop owns a
decoder session from prompt prefill through completion, so a future scheduler
cannot rotate between active sequences without duplicating generation,
tokenization, UTF-8 streaming, cancellation, and usage-accounting logic.

Unbounded request threads would not provide deterministic fairness and could
bypass the engine's aggregate memory and active-sequence limits. Model metadata
must not select scheduling behavior, batch shape, or execution kernels.

## Decision

Represent one CPU greedy generation as a request-local `GreedyGeneration`
cursor. One `advance` call performs no more than one decoder token step and
returns either bounded internal progress or exactly one typed generation event.
The cursor owns the sequence's decoder session, KV cache, prompt position,
generated-token count, UTF-8 prefix, terminal state, and checked usage. It
borrows only the immutable loaded model and tokenizer and performs no I/O.

The existing synchronous `generate_greedy` API remains as a compatibility
wrapper that advances the cursor until completion. The current UDS daemon
continues to process one sequence at a time; this ADR does not claim that the
multi-connection runtime or batching is complete.

## Invariants

1. Each cursor has one request ID, engine-assigned sequence ID, decoder session,
   KV cache, and resource budget; every advance receives that sequence's
   request-local cancellation flag.
2. One advance commits at most one model token to that sequence's KV cache.
3. Started, token, completed, and failed event ordering remains explicit and
   usage is validated against the originating request.
4. A completed cursor cannot advance again.
5. Scheduling is engine-owned. Model packages and clients cannot set quantum,
   priority, batching, memory location, or kernel choice.
6. Aggregate model memory accounting and hard active-sequence limits remain
   mandatory when the runtime activates multiple cursors.
7. The cursor adds no listener, network path, tool, approval, memory, or Agent
   Green authority.

## Consequences

Deterministic round-robin orchestration can interleave long and short prompts
without rebuilding the generation path, and the synchronous reference behavior
continues through the same implementation. Two-sequence tests prove independent
state and bounded per-advance work.

The next slice must connect multiple authenticated UDS connections to an
engine-owned ready queue, enforce admission and aggregate memory before cursor
creation, preserve request-local cancellation and signed per-stream event
ordinals, and apply bounded backpressure. Until that work passes integration
tests, the daemon remains intentionally single-worker.

## Rejected alternatives

- **One unbounded OS thread per request:** does not enforce deterministic
  fairness and expands memory, cancellation, and shutdown risk.
- **Duplicate synchronous and scheduled generation loops:** invites divergent
  security checks and event semantics.
- **Model-selected batching or quantum:** transfers implementation authority to
  untrusted model data.
- **Wire-protocol changes first:** fairness is an internal execution concern;
  the existing authenticated request and event schemas are sufficient.
