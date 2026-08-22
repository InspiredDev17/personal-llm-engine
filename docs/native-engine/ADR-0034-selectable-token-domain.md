# ADR-0034: Model/tokenizer-bound selectable token domains

## Status

Accepted as portable sampling hardening. Qwen2.5 service activation remains
disabled.

## Context

Some reviewed decoder families may expose more output-logit rows than their
tokenizer defines. Qwen2.5 1.5B has 151,936 logit rows but its pinned tokenizer
defines IDs only through 151,664. Sampling every logit would permit an
undefined ID to win greedy or stochastic selection and fail only later during
decode. A request-side deny list is not a trust boundary: it can be omitted,
stale, or changed by the caller.

## Decision

Every loaded executable model owns one immutable `SelectableTokenDomain`.
Engine-owned load code derives it from the already verified package identity,
tokenizer identity, model logit count, and validated tokenizer-defined ID set.
The current AIPTok v1 path accepts only `EntireModelVocabulary`, preserving its
existing exact vocabulary-equality requirement. The separate
`ContiguousTokenizerPrefix` representation prepares the reviewed Qwen shape:
IDs below the exclusive bound are selectable and the non-empty model tail is
reserved.

The domain is not encoded in generation requests or accepted from the local
protocol. Request allow/deny and stop-token settings may only reference IDs in
the trusted domain. They may narrow it but cannot expand it. Empty domains,
out-of-domain settings or history, mismatched logit counts, duplicate request
sets, and contradictory allow/deny sets fail explicitly.

Domain membership is applied before reading, sorting, normalizing, penalizing,
or filtering logits. Consequently even a reserved row containing the largest
finite value, NaN, or infinity cannot influence selection. NaN and positive
infinity remain errors when they occur on an otherwise eligible selectable
row. Greedy and every stochastic filter share the same candidate construction.

The domain also binds the exact model package SHA-384 and tokenizer SHA-384 so
loaded-model construction and diagnostics cannot confuse otherwise similar
vocabulary sizes. Errors remain bounded and content-free; they do not disclose
individual rejected IDs or model content.

## Consequences

Legacy signed fixtures and production AIPTok v1 packages retain all-token
behavior. No public protocol changes, dependencies, registry widening, or
Qwen activation are introduced. A future AIPTok Qwen profile must independently
prove the exact 151,665-ID tokenizer domain before service load code may select
the prefix representation. It must not accept an arbitrary package-provided
bitmap or client-provided boundary.
