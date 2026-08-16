# ADR-0029: Trusted execution placement and bounded layer staging

Status: accepted for portable planning, fully resident composition, and CPU
staging; streamed decoder, hybrid execution, and operated qualification remain
open

Date: 2026-08-16

## Context

The signed directory loader previously verified and privately mapped complete
packages, then materialized every execution tensor as F32. That is the correct
fully resident reference path, but it provided no trusted representation for a
model that exceeds the configured fastest-memory envelope. A client-provided
physical plan would expose package offsets and backend mechanics as remote
authority. Treating Apple unified memory as discrete VRAM would also create
false resource evidence.

Sovereign-AI-Contracts does not currently define a placement-policy contract.
The wire protocol therefore must not acquire a competing transitional schema.

## Decision

`engine-placement` owns a closed, immutable internal `ExecutionPlacementPlan`.
It derives only from an already verified package, trusted backend capabilities,
an engine-owned memory envelope, and an owner-selected closed policy. The plan
binds the exact model ID, optional registry artifact ID, SHA-384 package
identity, package and model generations, architecture, backend, quantization,
memory topology, trusted tensor ranges, stage order, and checked byte totals.
Clients cannot submit paths, tensor offsets, layer ranges, kernels, backend
plugins, or a physical execution plan.

Three modes are named so unsupported behavior is explicit:

- `FullyResident` preserves the current complete-materialization fast path and
  now becomes part of signed CPU, ARM64, and Metal model composition.
- `LayerStreamed` validates an exact architecture stage inventory and provides
  correctness-first CPU staging with at most one lifecycle-owned prefetch
  worker. The worker is joined before ownership can end; cancellation, read,
  verification, or callback failure drops all request-owned staging state.
  Mapping-identity failures emit the existing bounded package-rejection event.
- `HybridResident` remains a typed unsupported placement. No backend currently
  implements a reviewed cross-device execution path.

Quantization remains an artifact property, not a residency mode. A plan never
silently changes package, backend, model revision, or quantization. The loader
can materialize a unique trusted subset of manifest tensor names and revalidates
all live mapped hashes before exposure. The read-only mapping is explicitly
`Send` and `Sync`: it owns one nonempty `MAP_PRIVATE|PROT_READ` region and live
descriptor, exposes only immutable byte slices, and unmaps only after all
shared owners are gone.

Memory accounting includes mapped package bytes, immutable metadata,
materialized execution bytes, per-staged-tensor metadata reservations, staging,
existing engine reservations, and a safety reserve. Arithmetic and configured
limits are bounded by the existing 1 TiB hard ceiling. Streamed staging uses the
dedicated `PlacementStaging(ModelId)` owner. The initial prefetch depth is zero
or one and the layer ceiling is 4,096.

Apple Metal is described as unified topology: backend ownership remains
logical and no fabricated VRAM-free value is produced. Host-only CPU/ARM64 and
future discrete-device topology are distinct. CUDA remains unimplemented.

## Consequences

- Signed fully resident execution gains exact placement and resource identity
  without changing client authority or degrading its execution path.
- Layer ranges may be unequal and are derived from signed tensor records rather
  than caller offsets.
- The staged loader/prefetch substrate is real and fault-tested, but the
  transformer service does not yet advertise or execute streamed generation.
  No performance claim is made.
- Automatic eviction remains absent. Go may request exact idle unload under
  ADR-0028; pressure otherwise fails explicitly.
- A future canonical placement policy requires a reviewed Sovereign Contracts
  revision and cross-language fixtures before it crosses the protocol.

## Verification and acceptance boundary

Portable tests cover real signed package conversion and verification, exact
resident/streamed plans, unequal stages, tied weights, checked ceilings,
bounded prefetch order, cancellation and injected failure cleanup, mapped-file
mutation after verification, hybrid rejection, and protocol/DAG authority
boundaries. These tests establish planning and staging correctness only.

Production acceptance still requires a licensed qualified model, owner-held
signing keys, actual Apple Silicon/Metal pressure and throughput measurements,
long cancellation/load/unload soak, and any future CUDA hardware/toolchain
qualification. Layer streaming must not be called faster than residency without
that evidence.
