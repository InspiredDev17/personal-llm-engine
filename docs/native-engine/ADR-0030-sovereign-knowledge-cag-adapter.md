# ADR-0030: Sovereign Knowledge CAG Adapter

## Status

Accepted for the experimental, unactivated Knowledge 1.1 lifecycle and 1.2
exact-prefix integration.

## Context

Sovereign-AI-Contracts commit
`2eaa6c23e6334df9b1309dcb28a0a8d15ebb5356` defines the canonical CAG cache
lifecycle at version 1.1.0 and exact static-prefix acquisition at version
1.2.0. The engine already used authenticated fixed binary service kinds 27
through 33. Directly enabling the Contracts Rust `full` feature would import
unrelated control-plane domains and general JSON support. The narrower
`runtime-v3` profile still cannot compile Knowledge 1.1 by itself because its
lifecycle module imports the `full`-gated knowledge module.

## Decision

Retain service kinds 27 through 33 as an internal authenticated transport
adapter. Add `engine-knowledge-contract`, with no new external dependency, as a
closed projection of the execution facts the engine must validate. Pin the
Contracts commit, both canonical schema SHA-256 values, and the 1.1 and 1.2
accepted/rejected fixture digests in source and tests.

Owner-pinned packages have no Model Foundry artifact ID. For that mode, derive a bounded canonical
`art_` identifier from the first 63 lowercase hexadecimal digits of the verified package SHA-384
while retaining the complete SHA-384 as the mandatory cryptographic binding. Registry-backed
packages retain the exact registered artifact ID.

Every reusable key binds the complete runtime, artifact, tokenizer,
architecture, backend, engine instance, cache format, bundle, source revision,
scope, sensitivity, prefix, and token-count identity. The engine validates the
Memory Vault prefix framing and digest before tokenization. Attach compares the
entire binding. Release and exact invalidation are request-bound and
idempotent. The engine consumes only already-authorized decoded prefix bytes,
not the 1.2 JSON/base64 acquisition envelope, and independently enforces the
canonical byte grammar, ceiling, and digest. Unknown values and trailing bytes
fail closed.

AI-Avenger remains responsible for validating canonical JSON, authorization,
activation, timestamps, and control-plane policy, then translating only
validated fields into the binary adapter. The engine does not acquire
retrieval, ranking, memory, model selection, tool, network, or activation
authority.

## Consequences

- No JSON or broad Contracts dependency enters the production engine graph.
- The binary bridge remains internal and cannot become canonical protocol
  truth.
- Cache compatibility is stricter and substitution-resistant.
- V1 reusable state remains process-local CPU memory; restart starts empty.
- Knowledge 1.1 and 1.2 remain explicitly unactivated; portable
  interoperability is not production authorization.
