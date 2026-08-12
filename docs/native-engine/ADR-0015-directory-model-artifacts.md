# ADR-0015: Directory-backed model artifacts and read-only shard mapping

- Status: Accepted
- Date: 2026-08-12

## Context

The production runtime needs a first-party model supply-chain boundary without
embedding a source-model ecosystem, archive decompressor, arbitrary graph,
dynamic loader, Python runtime, or third-party AI/tensor/tokenizer framework.
Large tensor payloads should be independently hashable and eligible for
read-only mapping. Source conversion and production loading have different
authority and must not share one runtime component.

## Decision

Use directory-backed `.paimodel` V1 packages. The signed trust root is the
fixed `hashes.aiph` catalog; local configuration pins its SHA-384 identity and
one active Ed25519 key. Strict Ed25519 signs SHA-384 of the exact catalog under
`PersonalAI.DirectoryModelPackage.Signature.v1\0`. The catalog closes the file
set and hashes the fixed manifest, first-party tokenizer, canonical provenance,
and every contiguous tensor shard. The manifest accepts only
`decoder-rms-rope-swiglu-gqa-v1` and fixed-width bounded records.

`engine-model-importer` is an offline-only component. It accepts only local
bounded safetensors and tokenizer data that resolve under an owner quarantine,
requires an explicit reviewed mapping, writes deterministic package bytes, and
signs with an exact stdin seed that is immediately zeroized. It has no network,
subprocess, dynamic-loading, pickle, script, Python, or model-code path.

`engine-loader` is the production component. It resolves only configuration-
registered ModelIds and safe direct-child directory names. It verifies protected
non-symlink roots/direct files, catalog identity, signer, exact file set, all
hashes, manifest compatibility, architecture inventory/shapes, and aggregate
budgets. It has no importer or unsigned-loading feature. Every failed admission
reports one typed event to a mandatory caller-owned sink.

Tensor shards are mapped by `engine-loader/src/read_only_map.rs`, the only new
unsafe boundary. It has exactly three unsafe blocks: `mmap`, immutable slice
construction, and `munmap`. Its safe invariants are:

- the securely opened regular file and mapping owner remain live together;
- the exact nonzero checked file length is used for map, slice, and unmap;
- `PROT_READ|MAP_PRIVATE` is used, never write or execute permission;
- no mutable slice, raw pointer, descriptor, clone, or manual unmap is exposed;
- one non-cloneable owner unmaps the exact successful mapping once on drop.

Architecture tests pin the module, its one unsafe allowance, exact three unsafe
blocks, internal dependency DAG, and absence of unsafe code elsewhere.

## Consequences

Packages are easy to inspect, hash, replace atomically, and map read-only. The
production parser surface stays fixed and data-only. The file set is not a
general extension mechanism: new trusted files, required flags, architecture
IDs, dtypes, quantization, or optional sections require a format revision and
reviewed engine implementation. The legacy flat fixture format remains test-
only and is not a production load path.

The initial implementation supports one F32 shard, no quantization, and RoPE
scaling 1.0. A real model is not approved merely because it converts; license,
provenance, independent numeric vectors, production-size resource evidence,
signer custody, hardware qualification, and reversible Go activation remain
separate release gates.
