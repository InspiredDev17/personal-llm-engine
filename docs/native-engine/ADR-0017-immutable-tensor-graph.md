# ADR-0017: Immutable tensor graph and scalar correctness oracle

- Status: Accepted
- Date: 2026-08-12

## Decision

The initial runtime uses canonical dense row-major owned tensors, explicit
checked strided views, F32 execution, and explicit first-party F16/BF16 storage
conversion. All owned allocations are charged to one aggregate `MemoryPool`.
Reference operations allocate fresh destinations and are single-threaded scalar
loops with a fixed row-major evaluation order and a 100-million-iteration cap.

Architecture code, not model data, constructs an immutable `InferenceGraph`
from a closed `Operation` enum. Construction rejects duplicate node/value IDs,
missing values, cycles, unsupported operations, invalid shape propagation, and
overflowing temporary-storage bounds. Kahn ordering uses node ID as its stable
tie breaker. The execution plan cannot be mutated after construction.

F16 conversion implements IEEE binary16 classes and round-to-nearest,
ties-to-even; its tests enumerate all 65,536 source bit patterns. BF16 narrowing
also uses ties-to-even. Arithmetic is F32 with stable F64 accumulation where
needed for RMSNorm, softmax, attention scores, and trigonometric evaluation.
Ordinary arithmetic rejects NaN/infinity; negative infinity is accepted only
as a softmax/causal-mask sentinel.

## Consequences

- The CPU backend is a correctness oracle, not a throughput claim.
- Signed packages may contain F32/F16/BF16 storage, but CPU execution receives
  explicitly converted F32 tensors and never silently reinterprets a dtype.
- Non-contiguous views remain honest and are read through checked layout
  indexing; operations that require owned output create canonical storage.
- Input/output overlap is excluded by fresh destinations and Rust borrowing.
- Package/client-controlled graphs, arbitrary operation names, operator fusion,
  BLAS, SIMD, multithreading, GPU execution, and model-defined kernels remain
  outside this decision.
- Future optimized backends must match this oracle within documented numerical
  tolerances and receive independent memory, aliasing, unsafe, and hardware
  review.
