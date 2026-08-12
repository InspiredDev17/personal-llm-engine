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
wrapper that advances the cursor until completion. An engine-owned
`GenerationRoundRobin` admits requests under a configured hard active
ceiling and the most restrictive active request concurrency budget. It creates
the cursor only after those checks, so rejected work cannot trigger prompt
tokenization or session allocation. Each pull
advances the front cursor once, returns at most one unbuffered event, removes a
completed or failed cursor, and rotates an unfinished cursor to the back.
Request or sequence identifiers cannot be active twice.

The CLI's continuous UDS daemon continues to process one sequence at a time.
ADR-0008 adds a bounded authenticated multi-connection reference batch without
claiming that continuous concurrent admission or production batching is
complete.

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
6. Coordinator admission enforces its hard active ceiling and every active
   request's concurrency budget. It conservatively charges activation,
   scheduler-owned, and cursor-owned request copies, worst-case byte-token
   storage, and fixed cursor overhead against an aggregate retained-byte ceiling
   before tokenization. Its queue uses fallible bounded allocation.
7. A cursor error becomes that request's typed failed event and cannot stop a
   neighboring sequence. The coordinator stores no event queue, so the caller
   must deliver or reject the returned event before requesting another turn.
8. Aggregate model and per-session memory remain charged through the existing
   request-local pools; production multi-client memory and soak qualification
   remain mandatory before activation.
9. The cursor and coordinator add no listener, network path, tool, approval,
   memory, or Agent Green authority.

## Consequences

Deterministic round-robin orchestration now interleaves long and short prompts
without rebuilding the generation path, and the synchronous reference behavior
continues through the same implementation. Tests prove independent state,
bounded per-turn work, active and request-budget admission, unique active
identities, request-local cancellation/failure, and completion of an unaffected
neighbor.

The bounded reference integration now connects multiple authenticated UDS
connections while preserving durable request admission, request-local
cancellation, signed per-stream event ordinals, bounded writes, and connection
lifetime cleanup. ADR-0010 subsequently added bounded continuous ingress and
graceful drain; ADR-0011 added explicit CLI selection and owner-process signal
shutdown. Operated slow-consumer soak remains required.

## Rejected alternatives

- **One unbounded OS thread per request:** does not enforce deterministic
  fairness and expands memory, cancellation, and shutdown risk.
- **Duplicate synchronous and scheduled generation loops:** invites divergent
  security checks and event semantics.
- **Model-selected batching or quantum:** transfers implementation authority to
  untrusted model data.
- **Wire-protocol changes first:** fairness is an internal execution concern;
  the existing authenticated request and event schemas are sufficient.
