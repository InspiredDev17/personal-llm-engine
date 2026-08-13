# ADR-0021: Deterministic multi-shard offline import

Status: accepted for the directory package V1 importer.

## Context

The signed directory format and production loader already supported up to
1,024 immutable tensor shards, but the offline importer could produce only one
destination shard. Production-sized safetensors releases commonly divide an
otherwise fixed model tensor inventory across several local source files. A
single-source importer would force an external repacking step outside the
reviewed Rust supply-chain boundary.

## Decision

The offline importer accepts one or more repeated `--source` arguments. Every
source is a local, nonempty, bounded `.safetensors` file opened beneath the
quarantine root with the existing no-follow and identity checks. URLs,
traversal, duplicate filenames, more than 1,024 files, and an aggregate source
size above the package ceiling fail before signing.

Caller order has no semantic effect. The importer sorts safe ASCII source
filenames, assigns destination shard indices in that order, and records the
ordered filename and SHA-384 digest list in bounded canonical provenance. Each
source produces exactly one contiguous `tensors-NNNN.aipt` output. Tensor names
are mapped through the fixed first-party architecture mapping and must be
globally unique. The union of all shards must exactly equal the architecture's
required tensor inventory. Dtype, shape, finite-value, alignment, range, and
aggregate output-size checks remain mandatory.

The existing single-source APIs and provenance remain stable. New sharded APIs
are explicit, and the CLI selects them only when more than one `--source` is
provided. Package signing still occurs only after the complete manifest,
provenance, tokenizer, hash catalog, and every destination shard are fixed.

## Evidence and limits

Rust tests split the deterministic two-layer decoder fixture across two
safetensors files, import both caller orders, compare identical signed package
bytes, verify contiguous shard records and sorted provenance, install the
package, and load it through the production registered-ID loader. Separate
tests reject empty source sets, duplicate filenames, and cross-shard duplicate
tensors.

The importer still materializes bounded source and destination shards in host
memory and supports only F32, F16, and BF16 source storage, the exact Llama
safetensors name mapping, and the strict AIPTok metadata profile. F16/BF16 are
deterministically converted to finite F32 destination storage using the
first-party tensor conversion routines. Streaming conversion, quantization,
additional architecture mappings, and real-model numerical evidence remain
separate reviewed gates.
