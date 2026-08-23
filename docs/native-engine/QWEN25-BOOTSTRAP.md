# Qwen2.5 Bootstrap Boundary

The offline importer contains a closed transition profile for the three pinned
Qwen2.5 1.5B startup sources. It validates the exact `config.json` identities,
the shared exact `tokenizer.json` identity, the complete fixed tokenizer graph,
and the Qwen safetensors name/shape inventory including Q/K/V biases and tied
embedding/output weights.

Each configuration also binds one exact 3,087,467,144-byte
`model.safetensors` SHA-256 identity. This is source identity, not Foundry
approval, qualification, promotion, or package trust. The operator-only
preflight checks the descriptor length, authenticates the complete file by
streaming SHA-256 from the same open descriptor, and only then parses its
bounded header.

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

The importer therefore performs bounded, non-recursive source inspection and
now converts only the exact 7,031,645-byte, SHA-256-pinned source into canonical
`AIPTOK02`; it still emits no model package. The converter preserves source ID
order, reverses the GPT-2 byte-symbol alphabet, resolves merge results by exact
source-symbol concatenation, retains all 22 trusted added-token roles, and
self-validates through the existing parser. The output is exactly 4,613,135
bytes with SHA-256 `c154a79da4917e0c851130b5989f53f1f582fe5965e44429a038727398f873ef`
and SHA-384
`f1b418104aedb90abc6ee892ef5b71db2b02c2c0c82a9f8e162487d933ffdd636b9ae077eed806f011c251cfea5187cc`.
A separate `AIPTOK02` foundation validates
the closed data tables, byte map, merge graph, added-token roles, undefined
model-vocabulary tail and hard limits. A crate-private runtime core retains those
immutable tables and implements the exact fixed scanner state logic,
deterministic byte-map/BPE, added-token policy, decode and incremental decode.
The narrow package-bound constructor revalidates the complete serialized
tokenizer hash before exposing the exact pinned profile; it is not used by the
ordinary service factory. A private adapter checks the admitted
normalization/category tables are both Unicode 16.0, streams NFC into an
engine-bounded output, maps all General Category `L*`/`N*` values, and pins the
finite default-simple-fold mappings that reach ASCII, including U+017F LONG S.
The ordinary factory still validates and returns `UnsupportedTokenizer`. Adding
a general regex crate or pretending byte-only tokenization is equivalent is
prohibited. A feature-gated offline diagnostic replays 23 fixed production
encode/special-policy checks, all 271 undefined-tail IDs, and full/incremental
decode parity;
it returns no tokenizer handle and accepts no caller-selected text. Packaging
and activation remain separate closed gates. See ADR-0035 through ADR-0037.

The independently derived activation vectors and their current evidence level
are recorded in [QWEN25-TOKENIZER-VECTORS.md](QWEN25-TOKENIZER-VECTORS.md).

The sampling runtime now has a separate immutable selectable-token-domain
boundary. Deterministic tests model the exact Qwen counts and prove that IDs
151665 through 151935 cannot influence greedy, temperature, top-k, top-p,
min-p, or repetition-penalty selection even when every reserved row has the
largest logit. Trusted loader-side domain derivation now selects either the
legacy all-vocabulary domain or the exact tokenizer-defined prefix after
checking model-row equality and the complete tokenizer hash. Only the exact
signed Qwen V3 ARM64 development path may construct the tokenizer; generic,
CPU-reference, Metal and production service paths remain closed.

## Package boundary

`PAIDIR01` v1.1 and the scalar Qwen correctness scaffold exist without changing
v1.0. Engine 0.2.0 accepts the profile only through canonical Sovereign Models
2.0 identity, a separately signed development registry V3 record, exact package
verification and a separate owner-approved ARM64 startup identity. Registry V2
remains legacy-only, CPU/Metal remain ineligible, and production activation is
unavailable. Importer package emission still returns `NotImplemented`; Model
Foundry owns package creation, signing and qualification.
An internal ARM64 backend selection no longer risks legacy interleaved RoPE:
the decoder resolves the Qwen architecture first and uses the exact split-half
correctness operation. That narrow parity guard neither supplies a production
Qwen backend nor changes the package, registry, qualification, signing or
owner-activation gates.
Every public importer package/emission entry performs this unsupported-target check before
path inspection, source reads, shard parsing, tensor conversion, or allocation.

An explicit startup-only metadata-readiness policy now permits trusted owner
code to inspect a registered future package without making it loadable. The
default is disabled and neither the CLI nor authenticated client protocol can
enable it. Inspection verifies the signed catalog, package signer, closed file
set, manifest, tokenizer, provenance, shard identities, protected file
metadata, exact Qwen architecture profile, package-bound `AIPTOK02` runtime
tables, selectable token domain, and owner Runtime-v3 model/package/artifact/
backend/version/instance identities. It deliberately does not hash tensor-shard
contents or reserve/model-load their runtime resources. Its typed result always
reports service ineligibility and remaining qualification, tensor-verification,
resource-admission and owner-activation blockers. It is not the execution gate;
the V3 ARM64 path independently verifies the complete package and identity.

The loader parser also has a content-free cross-repository conformance vector
from the Model Foundry candidate encoder. It contains the complete canonical
338-tensor BF16 name/shape inventory, not placeholder tensors, and binds the
exact tokenizer identity and one 3,087,428,608-byte shard identity without
including that shard. Engine encoding and parsing reproduce the Foundry
manifest length/digest (65,472 bytes, SHA-384
`a8f31f32b09d44fa66e103d004412e8af3787edc6d94f7cf1dd6ad0123151fb22ee2799db00818e904083d943e6d829e`)
and catalog length/digest (528 bytes, SHA-384
`86651018b34498e8639dac2276655d074a3a8ae8eaf976ca242431a569276e456d1626e5209456cf29517ab2c86fd11b`).
This proves deterministic metadata-format parity only. The vector is unsigned,
contains synthetic content digests, and grants no package trust,
qualification, tensor-content evidence, or activation authority.

A second, separately owner-selected startup policy turns that verified
metadata into ARM64 execution preparation. It reuses the exact
package-bound tokenizer, retains the 151,936-row/151,665-selectable token
domain, revalidates the complete Runtime-v3 identity, and derives the bounded
full-context per-sequence KV requirement (1,879,048,192 bytes for this profile)
under the caller's memory, KV and concurrency ceilings. Cancellation is checked
before package work and between preparation stages. The ARM64 compute check
retains checked operation arithmetic but does not apply the scalar
correctness-oracle ceiling to a different backend. Engine 0.2 composes this
preparation only after V3/Models 2.0 validation and the exact CLI
`owner-approved` decision. The load then uses the cancellation-aware
full-content path that hashes
large shard mappings in 1 MiB chunks and drops partial state on cancellation.
Tensor materialization likewise polls snapshot hashing/copying and conversion;
BF16/F16 conversion writes directly to the accounted F32 destination instead
of retaining another owned source tensor, and quantized conversion polls at
fixed block boundaries. Cancellation drops partial package/tokenizer/tensor
state. A real signed package, exact Foundry qualification, independent output
evidence, production ARM64 evidence and a separate production design remain
mandatory.

The portable preflight now requires one absolute normalized owner-private
ingress root, one safe direct-child artifact name, an owner-private `source/`
directory, and exactly `config.json`, `tokenizer.json`, and
`model.safetensors`. Hidden or dot-alias artifact names are rejected.
Directories and files must share the ingress owner; source files must have
exact mode `0400`, with write, execute, and special bits forbidden. Unix opens
use `O_NOFOLLOW`, `O_CLOEXEC`, and
`O_DIRECTORY` where applicable; root, artifact, and source directory
descriptors remain open throughout inspection. File device, inode, length,
mode, owner, and modification timestamp must agree before open, after open, and
after reading, and the current path must still identify the opened object.

Only after complete-file identity authentication is the at-most-64-MiB
safetensors JSON header allocated. Its flat parser enforces the 338-entry
ceiling while decoding, then requires exactly 338 BF16 tensors, the reviewed
Qwen names and shapes, checked
contiguous non-overlapping ranges, and complete coverage of the declared file
extent. It rejects gaps, overlap, truncation, trailing data, unsupported dtype,
and arithmetic overflow. The 3,087,467,144-byte payload is never materialized;
SHA-256 is streamed from the same descriptor before header validation. Output
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

Exact tokenizer conversion and fixed-vector replay use standard input and write
only content-free identities/counts:

```text
cargo run --locked -p personalai-engine-model-importer --example verify_qwen25_tokenizer_conversion < tokenizer.json
```

After verification, an operator may publish the exact bytes for Model Foundry
to one new file inside a same-owner mode-0700 directory:

```text
cargo run --locked -p personalai-engine-model-importer --example export_qwen25_aiptok -- /absolute/pinned/tokenizer.json /absolute/private/new-tokenizer.aiptok
```

The source must be the exact pinned file at mode `0400`. The exporter binds the
owner-private parent, uses no-follow/create-new publication, cannot overwrite,
verifies the 4,613,135 output bytes and both fixed digests through the opened
descriptor, synchronizes the file and directory, and publishes mode `0400`.
Failure cleanup unlinks only the inode it created. Its report is content-free;
the command is offline supply-chain tooling and has no service authority.

The source parser accepts only the exact 7,031,645-byte input, fixed counts and
bounded 256-byte token/512-byte merge strings. The encoded tokenizer is further
bounded by the 64-MiB tokenizer-data ceiling; the diagnostic gives table/runtime
inspection a 128-MiB `MemoryPool`. Lookup keys borrow the ordered source table
and are dropped before encoding. One local diagnostic run observed 79,008 KiB
peak RSS; that is environment evidence, not a cross-platform production memory
claim.

Exact local validation of the shared 7,031,645-byte tokenizer was performed
during implementation; the file itself is intentionally not committed. This is
conversion and fixed-corpus conformance evidence only, not model qualification,
license approval, package signing, or production activation. The
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
