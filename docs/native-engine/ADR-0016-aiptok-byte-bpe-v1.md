# ADR-0016: AIPTok Byte-Level BPE v1

- Status: accepted
- Date: 2026-08-12

## Context

The production engine needs deterministic local tokenization without a
third-party tokenizer runtime, model-defined normalization graph, chat
template, script, plugin, network path, or Python conversion step. Tokenizer
data is untrusted even after package signature verification and needs an
independent bounded parser and content identity.

## Decision

Adopt exactly one family: AIPTok Byte-Level BPE v1. Its wire magic is
`AIPTOK01`; its header and records use fixed little-endian fields. The first 256
contiguous token IDs are exact byte identities. Each later ordinary token is
defined by one explicit contiguous-rank merge. BOS, EOS, and UNK are distinct
required special IDs. Token bytes, merge pairs/results, token IDs, ranks,
special IDs/roles, lengths, offsets, limits, reserved data, trailing data, and
the SHA-384 content hash are validated before immutable tables are exposed.

Normalization is preserve-bytes. Byte fallback guarantees lossless arbitrary
byte round trips and the UNK marker is not emitted for byte input. Text decode
is strict UTF-8; byte decode is exact. Streaming decode retains only an
incomplete UTF-8 suffix. Special text is canonical `<|name|>` data and is
rejected unless a caller explicitly allows its ID/all IDs or requests ordinary
byte treatment. The production prompt path rejects literal special text.

The offline importer converts only ordered `AIPTOK-METADATA-V1`. It requires no
pre-tokenizer, no normalization beyond preservation, no chat template, fixed
decode behavior, and byte fallback. The source filename, format, and SHA-384
enter signed provenance. Lossy approved-package conversion is unavailable.

Chat assembly, roles, templates, system prompts, tool invocation, memory,
policy, and approvals remain in Go. AIPTok accepts bounded text/bytes/tokens
only and has no I/O or execution capability.

## Consequences

The format is intentionally incompatible with arbitrary `tokenizer.json`
graphs, SentencePiece, Unigram, WordPiece, tiktoken, locale normalization, and
model-supplied templates. Supporting another family requires a new reviewed
implementation, format identifier, conversion profile, vectors, fuzzing, and
ADR; package metadata alone cannot enable it.

The tokenizer now depends internally on `engine-crypto` solely for the already
approved SHA-384 wrapper. No external dependency, unsafe Rust, network, Python,
JavaScript, or executable model logic is added.
