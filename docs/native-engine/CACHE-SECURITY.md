# Reusable Cache Security

## Security invariants

- Models are untrusted data and cannot request caching, choose a namespace, create a path, select
  offsets, or control eviction.
- Only the authenticated Go control plane can issue cache service operations.
- Cache IDs are content-derived identifiers, not authorization tokens. Every describe, list,
  attach, and scoped invalidation checks the exact namespace and bounded owner.
- Exact model package, tokenizer, architecture, backend, causal position semantics, RoPE values,
  K/V dtype, token hash, and token count must match.
- Raw K/V tensors never cross the service protocol.
- Shared sealed bytes are immutable. Each request receives a distinct mutable sequence copy.
- Checked arithmetic guards all compact/capacity ranges and allocations.
- Content hashes are rechecked before attachment; this detects in-process accidental mutation but
  is not a substitute for OS process isolation.
- Metrics and descriptors contain no prompt, completion, token bytes, keys, credentials, or paths.

## Scope model

The closed namespaces are `global-safe-static`, `user`, `project`, `task`, and `agent`. Namespace
plus owner is an exact authorization scope. The engine does not infer relationships among owners
and does not perform semantic access control. Go must authenticate the caller and authorize the
scope before signing the request.

## Failure behavior

Unknown IDs, wrong scope, compatibility mismatch, stale attachment, invalid token count, exhausted
headroom, active-lease eviction, unsupported tiers, malformed payloads, and trailing fields fail
closed with bounded typed errors. No failure causes cloud fallback, alternate model selection,
filesystem lookup, network access, or model-supplied code execution.

## Persistence and device tiers

CPU RAM is the only implemented reusable tier. `GpuMemory` and `LocalSsd` are representable future
locations, but migration returns `NotImplemented`; the engine does not fake residency. Any SSD
format must be separately versioned, signed or independently authenticated, length-bounded,
no-follow opened beneath an owner-configured root, corruption tested, encrypted if policy requires,
and never accept a client path. Metal/CUDA reuse must establish device ownership, synchronization,
unload, cancellation, and parity on real hardware before activation.
