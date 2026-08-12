# Model Supply Chain

## Production rule

The production engine loads only first-party directory-backed `.paimodel` V1
packages registered by local owner configuration under a `ModelId`, exact
SHA-384 catalog identity, active signer key ID/public key, and lower resource
budget. A client submits only the ID. It cannot submit a path, URL, signer,
hash, architecture, tokenizer, source format, optional feature, or kernel.

The offline importer and production loader are separate crates and binaries.
`engine-model-importer` is not a production runtime dependency.
`engine-loader` has no source-model importer and no unsigned loading feature.

## Directory-backed `.paimodel` V1

One package is `<model-id>.paimodel/` with exactly:

```text
manifest.aipm
tokenizer.aiptok
tensors-0000.aipt
provenance.json
hashes.aiph
signature.ed25519
```

Additional shards, when implemented by a future importer profile, must be
contiguous `tensors-NNNN.aipt` direct children and remain under the 1,024-shard
ceiling. The exact-file comparison rejects every unlisted file or directory.
Executables, scripts, dynamic libraries, object files, bytecode, shell/Python/
JavaScript modules, WASM, device kernels, and executable templates therefore
cannot be present in an admitted package.

`manifest.aipm` uses an eight-byte magic, fixed version and endian marker,
fixed 512-byte header, fixed 64-byte shard records, and fixed 192-byte tensor
records. It declares only the reviewed
`decoder-rms-rope-swiglu-gqa-v1` architecture. Records carry canonical tensor
name, dtype, rank/dimensions, shard index, byte range, alignment,
quantization declaration, and SHA-384 content hash. Parser rules include exact
length, reserved-zero fields, checked arithmetic, bounded strings/counts/rank/
dimensions, sorted unique names, nonoverlapping in-bounds ranges, valid UTF-8,
no trailing data, no recursion, no compression, and no ambiguous integer
encoding. Required unknown flags and unsupported optional sections fail closed.

The initial importer emits one 64-byte-aligned F32 shard, quantization `None`,
RoPE scaling `1.0`, and the first-party `PAIBPE01` tokenizer. Those restrictions
are implementation gates, not package-controlled kernel or graph choices.

## Cryptographic profile

`hashes.aiph` is a fixed, sorted, closed catalog containing the exact byte
length and SHA-384 hash of the manifest, tokenizer, provenance, and every shard.
The configured package identity is SHA-384 over the exact catalog bytes. The
signature is strict Ed25519 over SHA-384 of the catalog under domain
`PersonalAI.DirectoryModelPackage.Signature.v1\0`. The envelope key ID must
equal the active local signer configuration. Weak, revoked, mismatched,
malformed, untrusted, or invalid signatures fail before package contents are
exposed. Manifest tokenizer/shard hashes and per-tensor hashes add explicit
cross-checks after the signed catalog verifies.

The signing seed is exactly 32 bytes supplied to the offline importer through
stdin. It is never accepted through argv, stored in the repository/package,
printed, or logged, and is zeroized immediately after signing-key construction.
The production engine holds only the configured public verification key.

## Offline importer

The importer operates on owner-selected local files only. Both source and
tokenizer must canonicalize under one absolute quarantine root. URLs and
symlink escapes are refused. The initial source parser accepts a bounded,
data-only F32 safetensors subset implemented in Rust: eight-byte little-endian
header length, bounded JSON object, flat tensor records, exact `dtype`, `shape`,
and `data_offsets`, nonoverlapping contiguous data, and no trailing bytes.

The importer does not have code paths for pickle, `.bin`, `.pt`, `.pth`, model
repository scripts, tokenizer scripts, `trust_remote_code`, shared libraries,
subprocesses, Python, network sockets, downloads, registries, or dynamic
loading. The required explicit mapping is `llama-safetensors-v1`; all unknown,
missing, extra, duplicate, wrongly shaped, wrongly typed, non-finite, or
out-of-range tensors are rejected against the exact internal architecture.

Provenance records the architecture, license identifier, mapping, model ID,
local source filename/label, source format, source SHA-384, importer identity,
and version. Package bytes, ordering, padding, hashes, provenance, and signature
are deterministic for identical inputs/configuration/key. A staging directory
is durably written, sealed to read-only file/directory modes, and atomically renamed. An existing approved directory is
never overwritten implicitly: replacement requires a bounded audit note, writes
an append-only local audit record, and preserves the previous directory.

## Production admission and mapping

The production model store and package directory must already exist, be
non-symlinks, and have no write permission bits. Registrations accept one safe
direct-child `.paimodel` name. Each trusted file must be a non-symlink regular
file with no write permission bits. The importer can write only while its
separate destination root is writable; activation requires sealing that root.
The loader uses `O_NOFOLLOW|O_CLOEXEC`, compares pre-open and opened device,
inode, type, and length, and applies configured limits before unbounded reads or
mapping.

Admission order is catalog identity, signature, exact file set, catalog hashes,
manifest/engine/model compatibility, fixed architecture inventory and shapes,
tokenizer/provenance validation, shard identities/hashes, then per-tensor
hashes. Tensor shards use private `PROT_READ` mappings without `PROT_EXEC`; only
immutable slices are exposed, and CPU materialization remains charged to the
aggregate memory pool. Every failed admission emits one bounded typed
`ModelPackageRejected` event to a mandatory caller-owned sink. No arbitrary path
or unsigned development bypass exists.

## Verification evidence

Rust tests convert a synthetic source fixture twice and compare identical files,
sign and install it, load it by registered ID, and compare validated tensor
metadata and bytes with the source. Separate tests reject bad magic/version,
truncation, overflow, excessive count/rank/dimension, duplicates, overlap,
out-of-bounds ranges, invalid dtype/architecture/UTF-8, wrong/missing/unexpected
tensors, tokenizer/shard corruption, invalid/untrusted signatures, traversal,
symlink escape, trailing data, unsupported optional features, unaudited
overwrite, and package-budget excess.

Four dedicated bounded fuzz targets cover the directory manifest, one tensor
index record, safetensors import parser, and package-name resolution. A local
10-second smoke campaign per target is only smoke evidence; production-size
corpora and scheduled sustained campaigns remain release gates.

## Activation, update, revocation, and remaining gate

Activation is an owner configuration change in Go/service deployment, never a
model request. Updates receive a new exact hash and review. Signer revocation
blocks new loads. Rollback selects a previously reviewed directory and disables
the newer Go provider binding. There is no auto-download, self-update, remote
registry lookup, implicit latest version, cloud fallback, or telemetry.

No approved real production model is bundled. Before one is activated, retain
its license/use review, provenance and source hashes, reproducible package,
signer status, independent tokenizer/logit/generation vectors, resource and
corruption evidence, target-hardware results, SBOM/advisory records, and a
rollback drill. The synthetic fixture proves mechanics only, not model quality.

Model Foundry registry V2 parsing, exact owner-selected signature verification,
explicit registry-backed registration, canonical path derivation, and terminal
package loading are implemented. The owner must pin the exact sequence,
envelope SHA-384, `ModelId`, `ArtifactId`, both public signer identities, and a
nondecreasing sequence floor. The engine opens only the derived sequence file,
verifies it before package access, and derives the approved package path from
the signed entry. It then rechecks the signed catalog, package signer, context,
tokenizer, provenance, and shard identities before allocation. Failure never
falls back to owner-pinned registration. The owner CLI now requires an explicit
registration mode and accepts the complete exact V2 decision as public startup
configuration; the Go native-engine adapter binds the same selection without
accepting any artifact path. Owner-pinned rollback remains a separate explicit
configuration and cannot be selected after registry failure. Model Foundry flat
V1 is not compatible with this loader and must not be translated or used as
fallback. See ADR-0020.
