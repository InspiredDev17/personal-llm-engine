# ADR-0035: AIPTok v2 Qwen2.5 parser foundation

- Status: accepted as a fail-closed format foundation; execution disabled
- Date: 2026-08-22

## Context

The pinned Qwen2.5 1.5B tokenizer cannot be represented by frozen `AIPTOK01`.
It requires NFC normalization, a fixed Unicode-property pre-tokenizer, GPT-2
byte-to-symbol behavior, no unknown token, no implicit BOS, and 151,665 defined
token IDs inside a 151,936-row model vocabulary. Reinterpreting AIPTok v1 would
silently change already signed package semantics.

Rust's standard library does not implement Unicode NFC or Unicode General
Category `L`/`N`. A handwritten partial Unicode database or approximate regex
would create false compatibility. A follow-up dependency review admitted the
version-matched general-purpose pair `unicode-normalization 0.1.24` and
`unicode-general-category 1.1.0`, both fixed to Unicode 16.0.0, into the locked
offline vendor graph. Defaults and all direct features are disabled. This is a
bounded supply-chain admission: a later private adapter now uses only these
reviewed Unicode primitives, and a feature-gated offline diagnostic replays
independently derived production tokenizer vectors. Neither dependency is
selectable by a package or request.

## Decision

Add a separate `AIPTOK02` data-only profile with a bounded parser and canonical
fixture encoder. AIPTok v1 remains byte-for-byte and behaviorally unchanged.

The v2 header fixes the tokenizer family, NFC declaration, Qwen pre-tokenizer
profile, GPT-2 byte-symbol decoder profile, absence of unknown-token fallback,
absence of implicit BOS/EOS, table counts, required token identities, limits,
offsets, and a SHA-384 table hash. Tables contain contiguous vocabulary IDs,
ranked fixed-width merges, a contiguous added-token suffix with closed roles,
and a 256-entry raw-byte/base-token permutation. Parsing is non-recursive, uses
checked arithmetic, reserves aggregate memory first, rejects trailing data,
validates every merge and token reference, and rejects undefined tail IDs.

Production-cardinality recognition is separate from exact profile identity.
If any of the five pinned production counts is present, all five counts must
match; the parser then requires the exact special count, BOS/EOS IDs, fixed
header behavior flags/profile, and all 22 reviewed added-token literals/roles.
A partial or near-pinned table cannot downgrade itself into generic structural
fixture mode. Tests exercise the fixed-size production identity validator with
all exact values (including the 271 undefined model-output IDs) without
allocating a 151,665-entry fixture, while compact parser-path fixtures prove
the same fail-closed dispatch. The parser exposes only content-free immutable
descriptor facts. It does not expose an executable tokenizer.

The public tokenizer factory recognizes `AIPTOK02`, fully validates it, and
then returns `UnsupportedTokenizer`. This preserves corruption-before-
activation error precedence while ensuring no caller can accidentally obtain
approximate encode, decode, or incremental-decode behavior. The offline Qwen
converter can now emit canonical tokenizer bytes through a dedicated
exact-source conversion API, but continues to emit no package.

## Activation gate

The portable tokenizer mechanics originally listed here are now implemented
behind closed boundaries: bounded Unicode-16 normalization/classification,
the fixed Qwen scanner and GPT-2 byte mapping, deterministic
encode/decode/incremental decode with explicit special-token policy,
independent vectors and hard-bound tests, and exact deterministic conversion
of the pinned `tokenizer.json` into `AIPTOK02`.

Execution and importer package emission remain disabled. Activation still
requires reviewed binding of the exact converted tokenizer to an independently
accepted package, engine 0.2 compatibility, Model Foundry Registry V2
admission, a qualified optimized backend, package signing and promotion, and
explicit owner approval as required by ADR-0033. None of those authorities is
granted by successful conversion or conformance replay.

Until every gate closes, a signed package containing AIPTok v2 remains
unexecutable. Parsing proves structural integrity only; it does not prove
tokenizer equivalence, model quality, package approval, or production safety.

## Consequences

- Frozen AIPTok v1 compatibility is preserved.
- Malformed Qwen tokenizer tables can be tested and fuzzed without enabling
  incomplete inference.
- No third-party tokenizer/runtime library, network feature, Python, or
  model-provided code is added. The admitted Unicode primitives remain general
  Unicode algorithms/data, not tokenizer implementations; their one reviewed
  offline table build script cannot read model input.
- The exact Qwen implementation remains private and feature-gated for offline
  conformance; ordinary public tokenizer construction stays fail-closed until
  the remaining package and activation gates close.
