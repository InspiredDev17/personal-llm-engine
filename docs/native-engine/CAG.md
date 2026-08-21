# Cache-Augmented Generation (CAG)

## Status and authority

The engine implements first-party reusable prefix K/V state for the CPU decoder. CAG is an
inference optimization, not retrieval or durable knowledge. AI-Avenger decides whether a request
may use CAG and supplies the authorization scope. Memory-Vault owns retrieval and durable
knowledge. The engine performs exact compatibility validation and execution only.

V1 supports CPU-RAM-resident, immutable caches for
`decoder-rms-rope-swiglu-gqa-v1`. GPU migration, SSD persistence, semantic matching, document
ranking, cross-document composition, and selective recomputation are unsupported.

## Identity and exact-prefix rule

`KvCacheKey` binds the full narrow projection of Sovereign Knowledge 1.1 `CacheBinding`: logical
model ID, provider model ID, artifact ID and digest, tokenizer ID plus SHA-256 and native SHA-384,
architecture, backend ID/version/instance, engine instance, cache format, zero-based causal/RoPE
semantics, exact RoPE theta/scaling bits, K/V dtype, static bundle ID/revision/digests, optional
source/revision identity, authorization scope, sensitivity, ordered-token SHA-384, and token count.
`KvCacheId` is a deterministic SHA-384 over that complete canonicalized key. Attach compares the
entire immutable binding, so changing any execution, bundle, scope, sensitivity, or runtime fact
fails closed rather than downgrading compatibility.

Lookup is byte-for-byte token-prefix matching. It chooses the longest compatible cached token
sequence. Whitespace, tokenization, document order, model revision, tokenizer, RoPE, backend, K/V
dtype, namespace, or owner changes produce a miss or authorization failure. Semantic similarity is
never a compatibility signal.

## Request flow

1. An authenticated `CreateReusableCache` request names a loaded registered model, complete
   canonical binding, exact Memory Vault prefix bytes, a per-request cache-byte ceiling, and
   bounded retention settings. The prefix must begin with
   `SOVEREIGN-STATIC-KNOWLEDGE-PREFIX-V1\n` and contain canonical decimal byte-length frames.
   NUL-bearing, malformed, truncated, oversized, or SHA-256-mismatched bytes fail before
   tokenization. The engine never trims, normalizes, reconstructs, or reorders those bytes.
2. The engine tokenizes, prefills a request-local sequence, copies only committed K/V bytes, stores
   final-prefix logits, verifies finite data, hashes the sealed contents, and destroys the source
   sequence.
3. `AttachCache` stages a one-shot full binding under an explicit target generation request ID.
4. The generation request with that target ID contains only the uncached continuation text. It is
   encoded without a second BOS token.
5. The engine copies the sealed prefix into a new sequence and continues prefill/decode at the
   cached token position. The sealed object is held through an immutable lease and is not mutated.

The one-shot staged binding expires after 30 seconds if generation never consumes it. This bounds
crash/disconnect residue without mutating or evicting the shared sealed cache object.

The Go prompt assembler must preserve the declared static/dynamic token boundary. A BPE merge that
would cross that boundary can make separately tokenized pieces differ from tokenizing concatenated
text; Go must qualify and use boundaries whose exact token sequence is intended. Direct internal
lookup accepts full token sequences and supports longest exact-prefix selection.

Canonical service creation requires a TTL bounded to 365 days. A longer-lived cache must be
renewed through fresh authenticated policy rather than encoded as an effectively unbounded
expiration. An exact duplicate create reports `reused`, describe distinguishes `missing`, and a
bounded list reports whether it was truncated. Exact invalidation and release are idempotent: a
repeated operation reports zero removed or `already_released` without broadening its selector.

## Protocol operations

PAISVC01/PAISVR01 kinds 27 through 33 are: create, describe, list, attach, release, invalidate, and
metrics. Payloads are fixed-order, bounded binary fields. Clients use `ModelId`, `KvCacheId`, and
closed selectors only; no path, tensor offset, raw K/V bytes, backend plugin, or code is accepted.

Kinds 22 through 26 remain reserved for the canonical inference-runtime-v3 service bridge. Cache
operations must never reuse or reinterpret those message numbers.

The authenticated binary payload is an internal transport adapter behind Sovereign Knowledge
1.1.0 at Contracts commit `f3f6881d1661a088eef5dfbf7b51be4c747f1093`; its canonical schema
SHA-256 is `4976b42a04c35efbbf9f00be71d2e26343399d1c8528811eecf6fa4aa07451f9`.
`engine-knowledge-contract` contains only the execution-safe closed projection and pinned fixture
digests. It is not a competing public contract. The current Contracts Rust crate cannot compile
Knowledge 1.1 under `runtime-v3` alone because its lifecycle module imports the `full`-gated
knowledge module. Pulling `full` would import control-plane domains and JSON dependencies that are
prohibited here, so the engine does not depend directly on that broad feature. AI-Avenger must
perform canonical JSON validation and translate only validated fields into this adapter; activating
the experimental contract remains an explicit downstream decision.

## Unsupported selective recomposition

`CacheReusePolicy::SelectiveCacheRecompositionDisabled` is the explicit extension point for a
future `selective_cache_recomposition` design. Any future implementation must prove causal/RoPE
correctness, identify cross-block attention-sensitive state, recompute required boundaries, and
pass quality evaluation. It remains fail-closed and no LMCache or CacheBlend source is used.
