# ADR-0038: First-party sparse-expert streaming foundation

Status: accepted as portable, synthetic, CPU-reference infrastructure;
production model activation and optimized execution are not accepted

Date: 2026-08-22

## Context

The existing `LayerStreamed` path materializes every tensor in one dense
decoder stage. That is useful when a dense model does not fit the configured
execution-memory envelope, but it is not sparse mixture-of-experts execution.
Sparse MoE models keep dense attention, router, and shared tensors resident and
touch only a small architecture-selected expert set for each token. A
first-party implementation needs a separate cache and identity model without
allowing packages or clients to supply placement algorithms, offsets, kernels,
paths, or expert choices.

No reviewed production MoE architecture, manifest revision, tokenizer,
importer mapping, or signed artifact exists. Adding a model-family guess to the
production loader would create false support. Importing another inference
runtime would violate the charter and dependency policy.

## Decision

`engine-placement` now provides an engine-internal sparse-expert foundation:

- an immutable identity binding model, optional artifact ID, package digest,
  package-format revision, architecture, actual backend, package quantization,
  model generation, and plan generation, plus a canonical SHA-384 digest of the
  complete sparse inventory, mode, envelope, and prefetch policy;
- canonical bounded layer and expert inventories with exact payload hashes,
  quantization identities, storage bytes, and execution-byte ceilings;
- orthogonal `FullyResident` and `ExpertStreamed` expert residency modes;
- checked aggregate cache, execution-staging, and safety-reserve envelopes;
- exact architecture-owned selections containing no client offsets;
- a CPU host cache using durable aggregate `SparseExpertCache(ModelId)` and
  `SparseExpertStaging(ModelId)` capacity guards, fixed fallibly allocated
  slots charged for every possible expert, and bounded child cache/staging
  pools that avoid double-charging actual payload and scratch allocations;
- deterministic unpinned LRU eviction, immutable leases, and one serialized
  cache miss so duplicate reads cannot exceed the I/O bound;
- independent exact-length and SHA-384 verification after a pinned source
  fills an initialized accounted buffer;
- a nonzero cache generation and cache-lifecycle cancellation probe that reject
  a read or execution completing after idempotent invalidation/unload;
- a scoped depth-one prefetch helper whose per-cache worker, process-wide
  four-worker permit, and explicit 256 KiB stack request are bounded; every
  worker is joined and partial leases are released on failure or cancellation;
- cancellation-aware SHA-384 verification in bounded 64 KiB chunks; and
- bounded content-free lookup, hit, miss, load, eviction, invalidation,
  resident-byte, and actual current/peak staging metrics.

`engine-transformer` now contains deterministic scalar diagnostic mechanics:
finite router scores, score-descending/expert-ID tie selection, stable softmax
over exactly the selected set, expert-ID accumulation order, an operation
ceiling, and an exact-plan/layer/cache-bound F32 gate/up/SwiGLU/down fixture
evaluator using admitted staging scratch and cooperative cancellation.
Quantized, F16, and BF16 expert execution fails explicitly. This is a
correctness substrate only. It does not register an architecture ID or make
the service advertise MoE execution.

The local protocol remains unchanged. Go still chooses a qualified model and
the engine still executes only the exact requested artifact. Expert IDs,
router results, cache state, byte ranges, and physical policy do not cross the
wire.

## Consequences

- The dense fully resident and whole-layer streamed paths are unchanged.
- Sparse expert caching is artifact/backend/generation isolated and cannot
  silently substitute a model, package, backend, quantization, or expert.
- Shared cache entries are immutable. A lease pins an entry; pressure fails
  explicitly if every candidate is pinned.
- Invalidation is terminal for a cache instance. A reload must create a new
  cache with a new plan generation, so stale completions cannot repopulate it.
  Repeated/concurrent invalidation is idempotent and cleanup requires no new
  allocation.
- Distinct sparse inventories derived from the same base placement plan have
  distinct plan digests. Router selections, caches, leases, and the scalar
  evaluator reject cross-plan substitution.
- Layer/expert storage may be unequal. Resource admission uses the sum of the
  largest possible selected payloads and, for depth-one prefetch, the largest
  adjacent selected sets.
- There is no claim that expert streaming is faster than residency. It is a
  memory-capacity fallback whose value depends on sparsity, storage, cache hit
  rate, active parameters, and actual hardware.

## Deferred production work

Production activation requires an exact selected model and a new reviewed
data-only manifest revision covering expert count, top-k, router
normalization/scaling, shared-expert behavior, tensor naming, and dense/MoE
layer inventory. The loader then needs a verified source adapter that retains
the signed package handle. Direct AIPQ8/AIPQ4 expert kernels must avoid F32
weight expansion. Package extent/address-space accounting must be separated
from committed resident memory before the engine can honestly admit an
artifact larger than host RAM.

ARM64, Metal, and CUDA sparse execution remain explicit future work. No model
download, raw safetensors production load, HTTP service, Python path, dynamic
library, model-supplied kernel, learned routing override, or persistent router
history is introduced.

## Verification boundary

Portable tests cover malformed/non-contiguous plans, checked ceilings, exact
source identity, payload truncation/digest mismatch, deterministic LRU,
pinned-entry pressure, concurrent-reader load coalescing, cooperative
cancellation, repeat-safe invalidation, stale completion, prefetch/source/work
and token-establishment races, prefetch/source/work panic conversion only in
unwind-enabled test builds, scoped/global prefetch limits,
cancellation-aware hashing, fixed-slot and staging accounting,
cross-plan/cache isolation, deterministic routing, non-finite rejection,
explicit storage-format rejection, scalar operation bounds, and bit-exact
fully resident versus streamed expert outputs on a synthetic signed-package
fixture.

These tests establish mechanics, not a model-quality or performance result.
Production release builds use `panic = "abort"`; trusted source/work callbacks
must not panic, and these tests do not establish production panic isolation.
Acceptance of a real model still requires legal/provenance approval, signed
package conversion, independent numerical vectors, target-hardware pressure
and throughput tests, sustained soak, and owner-controlled activation.
