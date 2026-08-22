# ADR-0036: Private Qwen2.5 tokenizer runtime core

- Status: accepted as inert runtime mechanics; public execution disabled
- Date: 2026-08-22

## Context

ADR-0035 introduced the separate, frozen `AIPTOK02` data format and deliberately
kept it non-executable. Exact Qwen2.5 tokenization requires two different kinds
of work:

1. model-independent runtime mechanics: immutable table lookup, the one fixed
   pre-tokenizer state machine, byte mapping, ranked BPE, added-token policy,
   decode and incremental decode; and
2. Unicode authority: version-pinned NFC normalization, General Category
   `Letter`/`Number`, and case-insensitive contraction behavior.

Rust `std` does not supply the second category. A separately reviewed supply-
chain slice admitted `unicode-normalization 0.1.24` and
`unicode-general-category 1.1.0`, which both bind Unicode 16.0 data. Those
primitives are not invoked by this private core. Wiring them through the
trusted adapter and fixing the exact contraction case-fold boundary must not
accidentally activate an approximate tokenizer.

## Decision

Add a crate-private Qwen2.5 runtime core. It consumes the immutable tables
retained by the already bounded `AIPTOK02` parser and has no public constructor,
trait implementation, service path, or importer emission path.

The core implements the fixed pre-tokenizer expression as reviewed first-party
state logic, not as package-provided regex or executable configuration:

```text
(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
```

The source-profile validator pins that complete literal rather than accepting a
package-provided expression. Alternation priority, single-number splitting,
trailing-whitespace backtracking, CR/LF attachment, and the Unicode White_Space
scalar set are explicit code.
Unicode classification, normalization, and contraction folding enter only
through a private trusted interface that requires Unicode 16.0. No package or
client can implement, choose, or configure that interface.

Parsed vocabulary, merge, added-token and byte-map tables are immutable. Base
bytes enter a deterministic ranked merge heap; token IDs outside the defined
151,665-token prefix are rejected even though the paired model has 151,936
logit rows. BOS is never inserted implicitly. Special-token text is rejected by
default and may be allowed globally or by a validated special-ID allowlist;
the diagnostic ordinary-byte policy bypasses added-token recognition. Ordinary
added tokens remain decodable data, while BOS/EOS/control roles are skipped by
normal decoding.

Persistent table copies, scanner work, normalization/output bounds, BPE work,
and incremental UTF-8 suffix state use the existing memory pool. Input, output,
token-piece, table-count and hard request ceilings remain checked. The runtime
does not allocate from a model-controlled size without prior validation.

## Evidence and remaining activation gate

Deterministic tests cover the independently derived fixed scanner boundaries
for contractions, Unicode letter/number categories, punctuation/newlines,
trailing whitespace, NUL, mixed scripts, NBSP and ZWSP. Synthetic trusted-input
fixtures cover the reviewed Unicode-16 normalization ordering, decomposed Latin
and Jamo composition, and the Unicode-version boundary scalars U+088F and
U+11DE0. Additional tests cover all 256 bytes, ranked BPE, exact lookup,
no-implicit-BOS behavior, special policies, undefined-tail rejection, strict
and incremental decode, limits, memory release and thread determinism.

The exact activation targets are retained in
[QWEN25-TOKENIZER-VECTORS.md](QWEN25-TOKENIZER-VECTORS.md). These tests prove
the state mechanics; they do not prove a production Unicode
database or the 151,665-entry pinned tokenizer table. Public construction still
parses `AIPTOK02` and returns `UnsupportedTokenizer`. Activation still requires:

1. a separately reviewed adapter using the admitted offline-vendored
   Unicode-16 NFC/category primitives and an exact contraction case-fold rule;
2. complete version-bound Unicode conformance vectors;
3. deterministic conversion of the exact pinned `tokenizer.json` into
   `AIPTOK02` and replay of its independently derived token-ID vectors;
4. engine-version, Registry V2, optimized-backend, signing, Foundry and owner
   activation approvals already required by ADR-0033 and ADR-0035.

## Consequences

- AIPTok v1 and the AIPTok v2 wire format remain unchanged.
- The private core adds no dependency beyond the separately admitted Unicode
  primitives and does not invoke them yet. It adds no general regex engine,
  Python, script, network path or model-supplied executable behavior.
- The remaining Unicode and production-table evidence is explicit instead of
  being approximated by ASCII or whichever Unicode tables happen to compile.
- The private core can be reviewed and fuzzed further without making Qwen
  packages executable.
