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

For owner-pinned packages, which do not carry a Model Foundry artifact ID, Rust derives the
canonical execution identifier as `art_` plus the first 63 lowercase hexadecimal digits of the
verified package SHA-384. The complete SHA-384 remains a separate mandatory binding, so this
bounded identifier is never treated as the artifact's cryptographic identity. Registry-backed
packages retain their exact registered `art_` identifier.

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

Canonical Runtime-v3 execution uses the same one-shot target request ID. The
continuous execution coordinator consumes that exact pending attachment before
the worker starts; a wrong, expired, invalidated, or substituted attachment is a
terminal failure and never silently becomes uncached generation. Cancellation,
disconnect, or an unsent worker job terminalizes the reservation and removes
request-local attachment state. For Qwen2.5 the static prefix remains the exact
raw Memory Vault token sequence, while only the dynamic ChatML suffix permits
the two engine-owned delimiter token IDs. Synthetic real-decoder tests compare
that exact sequence with uncached `raw prefix + identical suffix` execution.
Actual-package Qwen equivalence remains an opt-in operated gate and is not
established by synthetic fixtures.

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
1.1.0 and the exact-byte handoff defined by Knowledge 1.2.0 at Contracts commit
`2eaa6c23e6334df9b1309dcb28a0a8d15ebb5356`. The 1.1 schema SHA-256 is
`4976b42a04c35efbbf9f00be71d2e26343399d1c8528811eecf6fa4aa07451f9`; the 1.2 schema
SHA-256 is `0f711d87f54c22992887e21a3ef4b2a4c7814f5ff3c47bb95f8d29fbd2ed68b9`.
`engine-knowledge-contract` contains only the execution-safe closed projection and pinned fixture
digests. It is not a competing public contract. The engine does not parse the Knowledge 1.2 JSON
envelope or base64 and has no Memory Vault client: AI-Avenger validates and authorizes the exact
Memory Vault result, then passes only decoded bytes through the authenticated internal adapter.
The engine independently rechecks UTF-8, NUL absence, the 8 MiB canonical ceiling, one through 256
positive canonical decimal frames with at most ten length digits, its stricter request-size bound,
and the ordered-prefix digest before tokenization.

The engine pins Contracts commit
`346b04f5899facce34fe89ea3292d139bcb7de05` and intentionally compiles only its narrow
`models-v2` feature and required `runtime-v3` dependency. That feature boundary excludes Knowledge
lifecycle and control-plane modules by design. CAG therefore remains a reviewed internal binary
projection of the Knowledge 1.1/1.2 execution facts rather than a second canonical JSON consumer.
Activating either experimental contract remains an explicit downstream decision.

Portable interoperability was exercised against Memory-Vault commit
`a25f76ac6ac932ff4dfa7a1ec51441c8020a4a98` and the real AI-Avenger native client at commit
`002366ab6567c3e71752107e693fb095b3a9e91f`. A local test-only signed directory AIPack was loaded
by the CPU reference daemon; the published Go client completed authenticated health, exact model
identity, Knowledge 1.2 prefix tokenization, cache create/describe/list, attached streaming
generation, release, metrics, idempotent exact invalidation, miss confirmation, and unload over a
protected Unix socket. The harness and synthetic model remain outside the repository. This is
portable conformance evidence only: it is not activation, production-model qualification,
signing-key custody, accelerator qualification, soak evidence, or canary authorization.

## Unsupported selective recomposition

`CacheReusePolicy::SelectiveCacheRecompositionDisabled` is the explicit extension point for a
future `selective_cache_recomposition` design. Any future implementation must prove causal/RoPE
correctness, identify cross-block attention-sensitive state, recompute required boundaries, and
pass quality evaluation. It remains fail-closed and no LMCache or CacheBlend source is used.
