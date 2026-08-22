# Sparse Expert Streaming

## Purpose

Sparse expert streaming is a first-party capacity mechanism for a future
reviewed mixture-of-experts architecture. Dense attention, router, shared
weights, KV state, and request scratch remain governed by the existing engine.
Only the exact experts selected by trusted router math may be acquired from the
bounded expert cache.

This is not the existing whole-layer streamer and it is not automatically a
performance optimization. A model that fits in appropriate fast memory should
normally remain resident. Dense startup models do not gain sparse activation
from this subsystem.

## Initial portable foundation

The current source provides:

1. `SparseExpertPlanIdentity`, binding exact model, optional Model Foundry
   artifact ID, signed package digest, package-format revision, architecture,
   actual backend, package quantization profile, model generation, and plan
   generation. A separate canonical SHA-384 plan digest binds the complete
   sparse inventory, residency mode, resource envelope, and prefetch policy.
2. `SparseExpertLayerPlan`, containing a contiguous per-layer engine-owned
   expert inventory and one fixed reviewed active-expert count. Sparse layers
   themselves may be strictly ordered and non-contiguous, so a future reviewed
   architecture may interleave dense and MoE layers.
3. `SparseExpertExecutionPlan`, validating complete residency or selected
   expert streaming against hard cache, staging, total, and safety ceilings.
4. `SparseExpertCache`, storing initialized immutable CPU buffers under a
   child quota backed by a durable aggregate cache-capacity reservation.
5. `SparseExpertLease`, pinning one exact entry without exposing mutation.
6. `SparseExpertStagingBuffer`, providing zero-initialized scalar scratch from
   a separate child quota backed by the aggregate staging reservation.
7. `ExpertByteSource`, a path-free internal adapter boundary. The cache checks
   identity, length, SHA-384, cancellation, and cache generation after every
   source read. Hash verification polls cancellation between bounded 64 KiB
   chunks.
8. Deterministic unpinned LRU eviction and one serialized miss. Reads do not
   hold the cache-state lock.
9. A single cache-owned scoped prefetch worker, plus a process-wide ceiling of
   four workers and an explicit 256 KiB stack request per worker. It is joined
   on success, failure, or cancellation and cannot outlive the call. The
   independent work closure receives combined request, cache-lifecycle, and
   dependency cancellation and must remain transactional until the helper
   succeeds.
10. Stable scalar top-k routing and a tiny, exact-plan-bound F32
    gate/up/SwiGLU/down correctness evaluator whose scratch is charged to the
    admitted staging quota. Quantized, F16, and BF16 expert execution returns
    explicit unsupported behavior.

## Hard limits

| Resource | Limit |
| --- | ---: |
| sparse layers | 4,096 |
| experts per layer | 4,096 |
| experts per model plan | 262,144 |
| active experts per token/layer | 64 |
| asynchronous prefetch depth | 1 |
| process-wide prefetch workers | 4 |
| requested prefetch-worker stack | 256 KiB |
| expert hash cancellation interval | 64 KiB |
| scalar diagnostic operations/layer | 100,000,000 |
| aggregate memory | existing engine 1 TiB logical ceiling |

All sums, indexes, byte lengths, matrix sizes, generations, pins, and counters
use checked arithmetic. Allocations are fallible and buffers are initialized.

## Admission and memory

Expert quantization is an artifact property, not a residency mode. Each
descriptor records signed storage bytes and conservative execution bytes.

Before the cache becomes usable, it fallibly allocates an indexed slot table
and charges a conservative 256 bytes for every possible expert slot. This is a
fixed-table reservation, not a charge only for live entries. `FullyResident`
then requires that fixed table plus every expert payload. `ExpertStreamed`
requires the fixed table plus the largest possible selected set; with
depth-one prefetch it uses the largest checked sum of adjacent selected sets.
Execution staging uses the corresponding largest selected execution-byte sum.
The configured cache ceiling plus execution staging plus deliberately
unusable emergency/OS safety margin must fit the aggregate envelope.

The aggregate pool holds durable `SparseExpertCache(ModelId)` and
`SparseExpertStaging(ModelId)` capacity guards. Exact payload and scratch
allocations occur in bounded child pools backed by those guards, so current and
peak staging use is observable without double-charging the parent. The safety
guard is intentionally not consumable scratch. Active KV, request scratch,
tokenizer/model state, dense weights, and backend allocations retain their
existing owners. If aggregate admission or a child quota fails, execution
returns resource exhaustion; it does not exceed the limit or substitute a
model.

## Lifecycle

```text
verified plan + pinned source
          |
          v
  lookup exact ExpertId ---- hit ----> immutable lease
          |
         miss
          v
 serialized load token -> evict unpinned LRU if required
          |
          v
 initialized accounted buffer -> exact source read -> SHA-384 check
          |
          v
 generation recheck -> immutable insertion -> lease
```

Invalidation is idempotent, increments the cache generation once, cancels
cooperative source reads/execution, and removes unpinned entries without a new
allocation. A load or prefetch token becomes an RAII owner before its in-flight
flag can defer terminal release, so invalidation during token establishment
cannot strand the admitted cache, staging, or safety reservations.

Pinned entries remain readable only by their existing leases, then disappear
when the last lease drops. New lookups and reads completing against the old
generation fail closed. A reloaded model uses a new cache instance and plan
generation.

Source, prefetch, and independent-work callbacks are trusted no-panic engine
boundaries. Debug unwind builds convert such panics into typed failures where
possible, but production release builds retain the workspace-wide
`panic = "abort"` policy. Portable tests therefore do not claim production
panic isolation.

## Security boundary

Neither clients nor model packages provide runtime offsets, paths, kernels,
device addresses, routing algorithms, learned cache policy, or executable
placement plans. The model package will eventually select only a reviewed
architecture ID and bounded declarative configuration. Trusted architecture
code constructs the expert inventory and router semantics.

Cache metrics contain only counts and cache/staging byte totals, including
current/peak staging and current staging-allocation count. They contain no
prompts, tokens, completions, weights, paths, keys, or raw memory.

## Not yet supported

- no production MoE architecture or manifest revision;
- no importer mapping for a real MoE family;
- no production `VerifiedDirectoryPackage` expert-source adapter;
- no direct quantized expert execution in the decoder;
- no production decoder/service MoE advertisement;
- no protocol or Go control-plane change;
- no ARM64, Metal, CUDA, SSD, network, or multi-node expert tier;
- no speculative expert prediction or cache-aware router mutation;
- no proof that a model larger than RAM can be admitted while the loader
  charges complete shard mappings as committed bytes; and
- no production-model quality, latency, throughput, thermal, or soak evidence.

ADR-0038 records the acceptance boundary.
