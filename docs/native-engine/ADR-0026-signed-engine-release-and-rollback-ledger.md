# ADR-0026: Signed engine release and monotonic rollback ledger

- Status: Accepted
- Date: 2026-08-14

## Context

Model-package signing does not authenticate the executable that interprets a
package. A valid old executable can also be vulnerable even when its signature
is still cryptographically correct. The inference process cannot solve either
problem because the charter forbids self-update and because writable runtime
state is not a trustworthy rollback checkpoint.

## Decision

`engine-release` is a separate operator/supervisor component and is not a
dependency of `engine-server` or `engine-cli`. It defines a canonical bounded
little-endian `PAIRLSV1` manifest, fixed `PAIRSIG1` Ed25519 envelope, exact
closed artifact roles, and fixed `PAIRLDG1` checkpoint. A release requires one
engine binary, SPDX SBOM, and Cargo.lock. One separate metallib and one notices
file are optional. Every declared artifact is a direct-child regular file with
an exact nonzero length and SHA-384 digest. Installed roots, metadata, and
artifacts are non-symlink and read-only; unknown files are rejected.

The release signature uses the distinct
`PersonalAI.Engine.Release.Signature.v1` transcript and a release-only Rust
key type. Package, registry, control-request, and response signers cannot be
passed to this verifier. Offline signing reads exactly 32 seed bytes from
standard input, zeroizes the caller copy, writes new files only, and never
installs or launches a release.

After signature and exact artifact verification, the operator tool may advance
`installed-release-v1.ledger` beneath an owner-selected protected state root.
The root must be disjoint from the engine runtime. A create-new lock serializes
writers; a crash-left lock fails closed for owner recovery. The ledger binds
the highest nonzero generation to one exact manifest digest. Lower generations
and same-generation substitutions are rejected. Advancing uses a private
temporary file, file synchronization, atomic rename, and directory
synchronization.

## Bounds

- manifest: 64 KiB;
- artifacts: 16;
- direct-child filename: 128 bytes;
- one artifact: 2 GiB;
- aggregate artifacts: 4 GiB;
- release ID: 64 bytes;
- target: 128 bytes;
- toolchain ID: 64 bytes.

Hashing is streaming with a fixed 16 KiB buffer. Parsing is flat, fixed-width,
checked, canonical, and rejects unknown versions, flags, roles, invalid UTF-8,
duplicate roles/names, unsafe names, truncation, and trailing data.

## Consequences

The engine still cannot update itself. A trusted OS supervisor or installer
must authenticate the verifier binary, hold the release public key in owner
configuration, place the ledger outside engine-writable state, and invoke
verification before launch. Emergency downgrade is deliberately not present;
it requires a separate owner-authorized recovery-record ADR. Source tests do
not prove OS package signing, installer atomicity, key custody, operated
rollback, or production artifact publication.
