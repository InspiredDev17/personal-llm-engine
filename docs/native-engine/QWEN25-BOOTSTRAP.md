# Qwen2.5 Bootstrap Boundary

The offline importer contains a closed transition profile for the three pinned
Qwen2.5 1.5B startup sources. It validates the exact `config.json` identities,
the shared exact `tokenizer.json` identity, the complete fixed tokenizer graph,
and the Qwen safetensors name/shape inventory including Q/K/V biases and tied
embedding/output weights.

Each configuration also binds one exact 3,087,467,144-byte
`model.safetensors` SHA-256 identity. This is source identity, not Foundry
approval, qualification, promotion, or package trust. The operator-only
preflight checks the descriptor length before allocation, parses only the
bounded header, and then streams SHA-256 from the same open descriptor.

| Profile | `model.safetensors` SHA-256 |
| --- | --- |
| Instruct | `dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee` |
| Math Instruct | `08faf1e950173bb44c11d26df28c55da7237dc55c77dbc1cbae4a5d7584540d9` |
| Coder Instruct | `c1b9b30e907950516ba3c646bdf570d8084c25a6410a0cdca80cf04b11bc13a8` |

The trusted execution architecture identifier is
`qwen2.5-rms-rope-swiglu-gqa-bias-tied-v1`. Canonical layer tensors add
`q_proj.bias`, `k_proj.bias`, and `v_proj.bias`; `output.weight` is prohibited
because logits use `token_embedding.weight`.

## Fail-closed tokenizer transition

The shared pinned tokenizer is not representable by `PAIBPE01`:

- it requires NFC normalization;
- it requires the exact Unicode-aware Qwen regex split before byte-level BPE;
- it declares no unknown token;
- it inserts no BOS implicitly; and
- it defines token IDs through 151664 while model embeddings contain 151936
  rows.

The importer therefore performs bounded, non-recursive source inspection but
emits no lossy AIPTok conversion. A separate `AIPTOK02` foundation validates
the closed data tables, byte map, merge graph, added-token roles, undefined
model-vocabulary tail and hard limits. A private runtime core retains those
immutable tables and implements the exact fixed scanner state logic,
deterministic byte-map/BPE, added-token policy, decode and incremental decode.
It cannot be constructed publicly. Version-matched Unicode 16 normalization
and category primitives are approved and offline-vendored, but no runtime
adapter invokes them yet and the exact contraction case-fold boundary remains
to be qualified. The public factory therefore still validates and returns
`UnsupportedTokenizer`. Adding a general regex crate or pretending byte-only
tokenization is equivalent is prohibited. Exact source conversion and replay
of independent production-table token-ID vectors are required before
packaging. See ADR-0035 and ADR-0036.

The independently derived activation vectors and their current evidence level
are recorded in [QWEN25-TOKENIZER-VECTORS.md](QWEN25-TOKENIZER-VECTORS.md).

The sampling runtime now has a separate immutable selectable-token-domain
boundary. Deterministic tests model the exact Qwen counts and prove that IDs
151665 through 151935 cannot influence greedy, temperature, top-k, top-p,
min-p, or repetition-penalty selection even when every reserved row has the
largest logit. This is portable safety scaffolding only. Current service load
code still derives only the legacy all-vocabulary domain after exact AIPTok v1
vocabulary equality. The data-only AIPTok v2 parser proves the Qwen prefix
cardinality but deliberately refuses tokenizer construction, so the service
cannot derive or execute that domain until exact Unicode/tokenizer execution,
conversion, and activation are reviewed.

## Package boundary

`PAIDIR01` v1.1 and the scalar Qwen correctness scaffold now exist without
changing v1.0. That does not activate the source profile: v1.1 deliberately
requires engine 0.2.0 while the current engine is 0.1.0, the exact tokenizer
converter/executable runtime remains absent, Registry V2 remains legacy-only,
and the full profile exceeds the scalar oracle ceiling without a qualified optimized
backend. Package emission therefore still returns `NotImplemented`; none of
these source files are approved, signed, activated, or loadable.
Every public package/import entry performs this unsupported-target check before
path inspection, source reads, shard parsing, tensor conversion, or allocation.

The portable preflight now requires one absolute normalized owner-private
ingress root, one safe direct-child artifact name, an owner-private `source/`
directory, and exactly `config.json`, `tokenizer.json`, and
`model.safetensors`. Directories and files must share the ingress owner and
grant no group/other permission. Unix opens use `O_NOFOLLOW`, `O_CLOEXEC`, and
`O_DIRECTORY` where applicable; root, artifact, and source directory
descriptors remain open throughout inspection. File device, inode, length,
mode, owner, and modification timestamp must agree before open, after open, and
after reading, and the current path must still identify the opened object.

Only the at-most-64-MiB safetensors JSON header is allocated. Its flat parser
requires exactly 338 BF16 tensors, the reviewed Qwen names and shapes, checked
contiguous non-overlapping ranges, and complete coverage of the declared file
extent. It rejects gaps, overlap, truncation, trailing data, unsupported dtype,
and arithmetic overflow. The 3,087,467,144-byte payload is never materialized;
SHA-256 is streamed from the same descriptor after header validation. Output
contains only bounded profile, digest, byte-count, header-count, tensor-count,
and explicit preflight-only status fields.

This diagnostic closes the source-inspection mechanics only. It returns no
reusable file handle and does not convert or emit a package. A future emission
path must retain or explicitly transfer ownership of the verified descriptor;
it must not reopen the path and treat this earlier diagnostic as package trust.
Owner/root compromise and mutable-filesystem attacks are fail-detect controls,
not an immutable-storage guarantee.

The three exact config files are compact committed fixtures. A reproducible
content-free diagnostic for an operator-held tokenizer is available as:

```text
cargo run --locked -p personalai-engine-model-importer --example verify_qwen25_sources -- INGRESS_ROOT ARTIFACT_NAME
```

Exact local validation of the shared 7,031,645-byte tokenizer was performed
during implementation; the file itself is intentionally not committed. This is
parser evidence only, not model qualification, tokenizer numerical
conformance, license approval, package signing, or production activation. The
closed JSON cursor uses a deliberately stricter numeric subset: negative zero
and leading integer zeroes are rejected, while standard exponent grammar is
retained because the pinned config spells epsilon as `1e-06`.

The three multi-gigabyte tensor files are intentionally not repository test
fixtures. Synthetic tests cover exact inventory, header-only range validation,
wrong length/hash, symlinks, path replacement, descriptor mutation, truncation,
and overlap. On 2026-08-22 the operator diagnostic inspected all three local
startup artifacts: each reported 3,087,467,144 bytes, a 38,528-byte header,
338 BF16 tensors, the shared pinned tokenizer digest, and its profile-specific
pinned model digest. That is local acquisition evidence only. It is not license
acceptance, quarantine approval, Foundry qualification, package signing,
runtime compatibility, numerical parity, performance, or activation evidence.
