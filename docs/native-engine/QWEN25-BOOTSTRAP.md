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

## Package boundary

The current `PAIDIR01` version 1.0 manifest names only the original no-bias
decoder. Qwen source mapping must not be mislabeled under that architecture.
Package emission therefore returns `NotImplemented` until reviewed PAIDIR 1.1
and scalar Qwen execution are integrated. Registry V2 remains closed and
unchanged; none of these source files are approved, activated, or loadable.
Every public package/import entry performs this unsupported-target check before
path inspection, source reads, shard parsing, tensor conversion, or allocation.

Exact local validation of the three pinned config identities and the shared
7,031,645-byte tokenizer was performed during implementation. This is parser
evidence only, not model qualification, tokenizer numerical conformance,
license approval, package signing, or production activation.

The three multi-gigabyte tensor files are intentionally not repository test
fixtures. Synthetic tests cover exact name/length/hash precedence and verify
that malformed bytes cannot reach safetensors parsing. Full-file SHA-256 and
inventory validation remains an operated source-acquisition gate once each
download is complete; no incomplete local download is counted as evidence.
