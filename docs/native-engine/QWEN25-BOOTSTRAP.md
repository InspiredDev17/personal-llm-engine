# Qwen2.5 Bootstrap Boundary

The offline importer contains a closed transition profile for the three pinned
Qwen2.5 1.5B startup sources. It validates the exact `config.json` identities,
the shared exact `tokenizer.json` identity, the complete fixed tokenizer graph,
and the Qwen safetensors name/shape inventory including Q/K/V biases and tied
embedding/output weights.

Each configuration also binds one exact 3,087,467,144-byte
`model.safetensors` SHA-256 identity. This is source identity, not Foundry
approval, qualification, promotion, or package trust. Length is checked before
hashing, and identity is checked before safetensors parsing.

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

The importer therefore performs bounded, non-recursive inspection but emits no
lossy AIPTok conversion. The transition converter returns
`UnsupportedTokenizer`. Adding a general regex crate or pretending byte-only
tokenization is equivalent is prohibited. An exact first-party tokenizer
runtime plus independent conformance vectors is required before packaging.

The sampling runtime now has a separate immutable selectable-token-domain
boundary. Deterministic tests model the exact Qwen counts and prove that IDs
151665 through 151935 cannot influence greedy, temperature, top-k, top-p,
min-p, or repetition-penalty selection even when every reserved row has the
largest logit. This is portable safety scaffolding only. Current service load
code still derives only the legacy all-vocabulary domain after exact AIPTok v1
vocabulary equality; it cannot derive the Qwen prefix domain until the exact
Qwen tokenizer format and parser exist.

## Package boundary

`PAIDIR01` v1.1 and the scalar Qwen correctness scaffold now exist without
changing v1.0. That does not activate the source profile: v1.1 deliberately
requires engine 0.2.0 while the current engine is 0.1.0, the exact tokenizer
converter/runtime remains absent, Registry V2 remains legacy-only, and the
full profile exceeds the scalar oracle ceiling without a qualified optimized
backend. Package emission therefore still returns `NotImplemented`; none of
these source files are approved, signed, activated, or loadable.
Every public package/import entry performs this unsupported-target check before
path inspection, source reads, shard parsing, tensor conversion, or allocation.

Before any future emission activation, the import path must additionally
preflight the exact descriptor count, direct-child filename, and byte length
from filesystem metadata before any generic source read or source-sized
allocation. The SHA-256 must then be computed from the same securely opened,
descriptor-bound bytes; path-based verification followed by a second reopen is
not sufficient. Exact tensor descriptor count/name/length validation must also
complete before tensor payload conversion or destination allocation.

The three exact config files are compact committed fixtures. A reproducible
content-free diagnostic for an operator-held tokenizer is available as:

```text
cargo run -p personalai-engine-model-importer --example verify_qwen25_sources -- CONFIG_JSON TOKENIZER_JSON
```

Exact local validation of the shared 7,031,645-byte tokenizer was performed
during implementation; the file itself is intentionally not committed. This is
parser evidence only, not model qualification, tokenizer numerical
conformance, license approval, package signing, or production activation. The
closed JSON cursor uses a deliberately stricter numeric subset: negative zero
and leading integer zeroes are rejected, while standard exponent grammar is
retained because the pinned config spells epsilon as `1e-06`.

The three multi-gigabyte tensor files are intentionally not repository test
fixtures. Synthetic tests cover exact name/length/hash precedence and verify
that malformed bytes cannot reach safetensors parsing. Full-file SHA-256 and
inventory validation remains an operated source-acquisition gate once each
download is complete; no incomplete local download is counted as evidence.
