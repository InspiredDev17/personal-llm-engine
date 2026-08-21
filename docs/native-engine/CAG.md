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

`KvCacheKey` binds the model ID, exact signed package SHA-384, tokenizer SHA-384, architecture,
actual backend, cache format version, zero-based causal-position semantics, exact RoPE theta and
scaling bits, K/V dtype, namespace, bounded owner, optional source hash/revision, ordered token
SHA-384, and token count. `KvCacheId` is a deterministic SHA-384 over the canonical key.

Lookup is byte-for-byte token-prefix matching. It chooses the longest compatible cached token
sequence. Whitespace, tokenization, document order, model revision, tokenizer, RoPE, backend, K/V
dtype, namespace, or owner changes produce a miss or authorization failure. Semantic similarity is
never a compatibility signal.

## Request flow

1. An authenticated `CreateReusableCache` request names a loaded registered model, scope, bounded
   prompt text, optional exact source identity, and retention settings.
2. The engine tokenizes, prefills a request-local sequence, copies only committed K/V bytes, stores
   final-prefix logits, verifies finite data, hashes the sealed contents, and destroys the source
   sequence.
3. `AttachCache` stages a one-shot logical binding under the service envelope request ID.
4. The generation request with that same ID contains only the uncached continuation text. It is
   encoded without a second BOS token.
5. The engine copies the sealed prefix into a new sequence and continues prefill/decode at the
   cached token position. The sealed object is held through an immutable lease and is not mutated.

The Go prompt assembler must preserve the declared static/dynamic token boundary. A BPE merge that
would cross that boundary can make separately tokenized pieces differ from tokenizing concatenated
text; Go must qualify and use boundaries whose exact token sequence is intended. Direct internal
lookup accepts full token sequences and supports longest exact-prefix selection.

## Protocol operations

PAISVC01/PAISVR01 kinds 22 through 28 are: create, describe, list, attach, release, invalidate, and
metrics. Payloads are fixed-order, bounded binary fields. Clients use `ModelId`, `KvCacheId`, and
closed selectors only; no path, tensor offset, raw K/V bytes, backend plugin, or code is accepted.

The protocol extension is repository-local until the corresponding canonical
Sovereign-AI-Contracts revision is published. It must not be treated as a competing source of
cross-repository truth.

## Unsupported selective recomposition

`CacheReusePolicy::SelectiveCacheRecompositionDisabled` is the explicit extension point for a
future `selective_cache_recomposition` design. Any future implementation must prove causal/RoPE
correctness, identify cross-block attention-sensitive state, recompute required boundaries, and
pass quality evaluation. It remains fail-closed and no LMCache or CacheBlend source is used.
