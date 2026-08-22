# ADR-0036: Package-bound Qwen2.5 tokenizer runtime core

- Status: accepted as package-bound runtime mechanics; ordinary service execution disabled
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
primitives were initially not invoked by this private core. ADR-0037 wires them
through a private bounded adapter and fixes the finite contraction case-fold
boundary without activating an approximate tokenizer.

## Decision

Add a crate-private Qwen2.5 runtime core. It consumes the immutable tables
retained by the already bounded `AIPTOK02` parser. A narrow public wrapper now
implements the engine tokenizer trait only after revalidating the complete
serialized tokenizer hash supplied from an independently authenticated package
manifest. The generic tokenizer factory, service path, and importer package
emission path remain closed.

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

Persistent table copies, scanner work, bounded normalization output, BPE work,
the first-byte added-token index, and incremental UTF-8 suffix state use the
existing memory pool. Executable construction accepts only the exact pinned
production cardinalities; compact structural tables remain test-only. Input,
output, token-piece, table-count and hard request ceilings remain checked. A
malformed incremental UTF-8 stream terminalizes its decoder, while a rejected
token ID changes no decoder state.

## Evidence and remaining activation gate

Deterministic tests cover the independently derived fixed scanner boundaries
for contractions including U+017F LONG S folding, Unicode letter/number
categories, punctuation/newlines,
trailing whitespace, NUL, mixed scripts, NBSP and ZWSP. Synthetic fixtures
continue to isolate state mechanics; ADR-0037 additionally exercises the
private adapter against both exact dependency version constants, every valid
Unicode scalar's `L*`/`N*` classification, decomposed Latin and Jamo
composition, normalization ordering and bounded NFC expansion. Additional
tests cover all 256 bytes, ranked BPE, exact lookup,
no-implicit-BOS behavior, special policies, undefined-tail rejection, strict
and incremental decode, limits, memory release and thread determinism.

The exact activation targets are retained in
[QWEN25-TOKENIZER-VECTORS.md](QWEN25-TOKENIZER-VECTORS.md). These tests prove
the state mechanics and the admitted Unicode-data boundary. The exact pinned
source is now deterministically converted, and a feature-gated offline
diagnostic binds the complete canonical output identity and replays the fixed
production encode, full-decode and incremental-decode vectors. It returns no
runtime handle and accepts no caller-selected inputs. Trusted loader work may
construct the exact pinned profile only through
`build_package_bound_qwen25_tokenizer`; the constructor rechecks the full
serialization hash and keeps model rows, defined IDs and selectable IDs
distinct. `build_tokenizer` and ordinary service admission still return
`UnsupportedTokenizer` for `AIPTOK02`. Activation still requires:

1. independent review/acceptance of the pinned conversion and conformance
   evidence for package use;
2. engine-version, Registry V2, optimized-backend, signing, Foundry and owner
   activation approvals already required by ADR-0033 and ADR-0035.

## Consequences

- AIPTok v1 and the AIPTok v2 wire format remain unchanged.
- The private core adds no dependency beyond the separately admitted Unicode
  primitives. ADR-0037's private adapter invokes only their NFC iterator,
  General Category lookup, and version constants. It adds no general regex engine,
  Python, script, network path or model-supplied executable behavior.
- Production-table evidence is explicit and content-free rather than being
  approximated by compact tables or exposed through ordinary runtime APIs.
- The private core can be reviewed and fuzzed further without making Qwen
  packages executable.
