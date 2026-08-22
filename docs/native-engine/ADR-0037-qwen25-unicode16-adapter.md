# ADR-0037: Private Qwen2.5 Unicode-16 adapter

- Status: accepted as private conformance infrastructure; public execution disabled
- Date: 2026-08-22

## Context

ADR-0036 isolated Qwen2.5 scanner, BPE and decode mechanics behind a private
Unicode authority. The separately admitted `unicode-normalization 0.1.24` and
`unicode-general-category 1.1.0` both declare Unicode 16.0 data, but the core
did not invoke them. Substituting Rust `std` character predicates or building a
complete normalized `String` before checking its size would respectively make
token boundaries compiler-dependent or permit unaccounted output growth.

The fixed contraction alternative is case-insensitive only for the literal
ASCII suffixes `'s`, `'t`, `'re`, `'ve`, `'m`, `'ll`, and `'d`. Unicode 16
default simple folding adds U+017F LONG S to the relevant `s` equivalence
class. U+212A KELVIN SIGN is the only other non-ASCII simple fold whose result
is ASCII; it folds to `k` and therefore cannot match any allowed suffix. Full
multi-scalar folds such as U+00DF LATIN SMALL LETTER SHARP S are intentionally
not simple-fold matches.

## Decision

Add one crate-private, engine-owned adapter. It is not a public trait
implementation and cannot be supplied or selected by a model, package, client,
or service request.

The adapter:

1. requires both dependency data-version constants to equal `(16, 0, 0)` and
   otherwise reports a version value rejected by the existing core gate;
2. consumes the NFC iterator one scalar at a time and appends each encoded
   scalar through `BoundedNormalizedText`, so checked output growth occurs
   before every write and no complete unbounded normalized output is built;
3. maps `Lu`, `Ll`, `Lt`, `Lm`, and `Lo` to `Letter`, maps `Nd`, `Nl`, and `No`
   to `Number`, and maps every other General Category to `Other`; and
4. uses a finite first-party Unicode-16 default-simple-fold mapping for ASCII
   case, U+017F to `s`, and U+212A to `k`.

Unicode White_Space remains the closed scalar set already encoded by the fixed
scanner. The adapter adds no regex engine, tokenizer runtime, model loader,
normalizer graph, executable configuration, network path, or new dependency.

## Evidence

Tests pin both dependency version constants and cover Latin and Hangul
composition, canonical combining-mark ordering, NFC expansion at U+0344,
output-bound rejection before growth, and deterministic normalized runtime
decode. Classification is checked across every valid Unicode scalar against
the Unicode-16 General Category family abbreviation, with explicit examples
for all five `L*` and all three `N*` categories plus unassigned boundary
scalars. Contraction tests cover ASCII case, U+017F, U+212A, and rejection of
full, Turkic, non-ASCII, and compatibility-only alternatives.

These are portable conformance checks for the private adapter. They are not
evidence for the absent 151,665-entry production tokenizer table, its exact
conversion, or end-to-end token-ID parity.

## Consequences and activation gate

- The `AIPTOK02` format and public tokenizer factory do not change.
- Public construction continues to parse and validate v2, then returns
  `UnsupportedTokenizer`.
- The offline importer still emits no Qwen tokenizer or model package.
- Activation still requires deterministic conversion of the exact pinned
  `tokenizer.json`, replay of the independent production token-ID vectors, the
  already documented runtime/registry/backend/signing/Foundry gates, and owner
  approval.
