# ADR-0018: Explicit inference memory and contiguous KV-cache V1

- Status: Accepted
- Date: 2026-08-12

## Context

Tensor storage already used a shared checked byte pool, and the scalar decoder
owned a private contiguous F32 cache. The engine still needed an explicit
ownership vocabulary, safe reusable request scratch, backend-neutral memory
operations, a reusable one-model/multi-sequence cache, and stable pressure
metrics for scheduler admission. Importing another runtime allocator or paged
cache would violate the first-party boundary and would make correctness harder
to establish.

## Decision

All inference allocations carry one explicit owner category. Scoped pools share
one underlying ceiling and counters. Reservations are RAII, checked, movable
between owners only through an accounting-preserving operation, and report
current, peak, per-category, and allocation-count metrics.

CPU memory is initialized ordinary host memory. Host/device transfers cover the
whole equal-sized buffer, and release clears storage. Metal and CUDA return
not-implemented errors. There is no fake device execution.

Scratch storage is held by a manager with an engine-wide ceiling. Each arena is
bound to one `RequestId` and a lower request ceiling. Handles include an arena
generation. Completion, cancellation, error unwind, and drop clear every byte.
Only cleared idle buffers may be reused; idle reservations are attributed back
to the backend and may be reclaimed deterministically.

KV-cache V1 is contiguous and owns one `ModelId`. Each sequence is bound to a
`RequestId`, numeric `SequenceId`, and non-wrapping generation. Its canonical
layout is `[2, layer, context, kv-head, head-dimension]`. The only cache storage
types are F32, F16, and BF16. Callers provide complete layer K/V values, never
offsets or positions. The cache selects the next position. A position is
committed only after all layers are present; abandoning or failing a write
clears and cancels the sequence. Reset, cancel, destroy, and model unload clear
storage. Destroy releases its reservation. Unload is terminal.

Admission APIs expose full sequence bytes, logical bytes per token and prefill,
current availability, model/request fit, and current/peak/used/reclaimable
metrics. Contiguous allocation reserves the configured maximum context at
sequence creation, favoring a simple proof over utilization.

Signed tensor shards remain private read-only, non-executable mappings. The
loader hashes mapped bytes before use and provides event-emitting full
revalidation. Mappings and their file descriptors drop with package ownership.

## Consequences

- Sequence and request identifiers remain distinct Rust types.
- No sequence can name another sequence's offsets or observe uncommitted data.
- Stale handles cannot reach a new sequence reusing the same numeric ID.
- Cancellation and partial backend failure cannot leave readable partial KV.
- Scratch reuse cannot expose prior request bytes.
- Contiguous cache memory grows by maximum context rather than current context.
- The scalar decoder uses the contiguous cache directly. Each decoder session
  owns one generation-checked handle, commits a position only after every layer
  is written, and clears/releases the sequence on drop.

## Paged-cache transition gate

A paged implementation is not authorized by this ADR. A later ADR must define
fixed engine-selected page geometry, checked page tables, fallible page
allocation, zero-on-reassignment, atomic token publication, cancellation and
unload behavior, reclaimability metrics, and backend synchronization. It must
run identical lifecycle/isolation tests and differential value tests against
contiguous V1. Packages and clients must remain unable to select page layout,
allocator behavior, cache persistence, or filesystem serialization.
