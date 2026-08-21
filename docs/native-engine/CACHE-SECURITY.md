# Reusable Cache Security

## Security invariants

- Models are untrusted data and cannot request caching, choose a namespace, create a path, select
  offsets, or control eviction.
- Only the authenticated Go control plane can issue cache service operations.
- Cache IDs are content-derived identifiers, not authorization tokens. Every describe, list,
  attach, release, metrics query, and invalidation checks the complete canonical authorization
  scope. Cache-ID guessing cannot bypass that check.
- Logical/provider model, artifact ID/digest, tokenizer ID/digests, architecture, backend
  ID/version/instance, engine instance, causal position/RoPE semantics, exact RoPE values, K/V
  dtype, cache format, bundle/revision/digests, optional source revision, sensitivity, token hash,
  and token count remain bound and cannot be substituted.
- Raw K/V tensors never cross the service protocol.
- Shared sealed bytes are immutable. Each request receives a distinct mutable sequence copy.
- Checked arithmetic guards all compact/capacity ranges and allocations.
- Content hashes are rechecked before attachment; this detects in-process accidental mutation but
  is not a substitute for OS process isolation.
- Metrics and descriptors contain no prompt, completion, token bytes, keys, credentials, or paths.

## Scope model

The Knowledge 1.1 closed namespaces are `owner`, `project`, `task`, and `agent`. Their required
owner/project/task/agent identities are shape-validated exactly. The engine does not infer
relationships among identities and does not perform semantic access control. Go must authenticate
the caller and authorize the complete scope and sensitivity before signing the request.

## Failure behavior

Unknown IDs, wrong scope or sensitivity, compatibility mismatch, stale attachment, malformed or
digest-mismatched canonical prefix, invalid token count, exhausted
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
