# K/V Cache Lifecycle

## Ownership

- Signed model weights, tokenizer tables, and compatibility facts belong to one loaded model.
- `ContiguousKvCache` owns mutable request/sequence K/V storage.
- `ReusableKvCacheStore` owns sealed immutable CPU cache objects and their memory reservations.
- `KvCacheAttachment` is an immutable lease. Attaching copies K/V into a fresh request-local
  sequence; decode never writes the shared sealed object.
- Sampling history and streaming decoder state remain request-local.

## States and transitions

Reusable V1 objects have one usable state: `Sealed`. Creation is transactional: the source
sequence must contain exactly the supplied tokens, final logits must be finite, capacity must be
reserved, committed bytes must copy successfully, and a content hash must be computed before the
entry becomes visible.

The lifecycle is:

`request-local prefill -> seal -> lookup/describe -> attach lease -> forked sequence -> decode ->
drop lease -> invalidate/evict`.

A generation cursor consumes a staged attachment once. Cancellation, client disconnect, cursor
drop, backend failure, or normal completion drops the request-local sequence and lease. Queued and
disconnected cancellation removes the staged binding. Model unload is rejected while scheduler or
staged attachment ownership exists, then invalidates idle reusable objects before removal.

## Memory and eviction

The store shares the loaded model's checked `MemoryPool`. Each object charges compact committed K/V
bytes, token IDs, final logits, and conservative metadata. Store limits require nonzero
full-sequence headroom, so reusable state cannot consume the entire configured inference pool.

TTL expiry and memory pressure use deterministic eviction. Entries are ordered by pin state,
priority, last access, creation time, then cache ID. Pinned entries and entries with live attachment
leases cannot be evicted. If sufficient headroom cannot be reclaimed, admission fails explicitly.

V1 physically copies a compact sealed prefix into a full-capacity contiguous sequence. This favors
correctness and isolation over memory efficiency. Paged K/V storage and device-resident cache
migration remain future work.

## Invalidation

Supported selectors are exact cache ID, exact scope, loaded model, exact source identity, and
source revision within an authorized scope. Active leases fail invalidation rather than becoming
stale. Package/model revision changes naturally produce incompatible keys; model unload removes
all idle objects.
