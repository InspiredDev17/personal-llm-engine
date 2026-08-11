# Model Supply Chain

## Production rule

The engine loads only first-party `.paimodel` packages registered by local
owner configuration under a `ModelId`, exact SHA-384 package hash, active signer
key ID/public key, and lower resource budget. A client can submit only the ID.
It cannot submit a path, URL, signer, hash, architecture, tokenizer, or kernel.

## V1 package

V1 is canonical, flat, bounded, and data-only:

- fixed 64-byte package header;
- fixed 256-byte architecture manifest;
- sorted fixed 128-byte tensor records;
- bounded `PAIBPE01` tokenizer bytes;
- aligned exact tensor payload ranges;
- fixed 96-byte Ed25519 signature envelope;
- exact total length, reserved-zero fields, and no trailing data.

It contains no executable code, scripts, plugins, shared-library names, paths,
URLs, recursive objects, compression, general graph, arbitrary operations,
Metal source, CUDA/PTX, or kernel selectors. Architecture and tokenizer IDs are
closed first-party identifiers.

## Cryptographic profile

Complete package identity is SHA-384. Signed-content identity is SHA-384 over
the exact canonical signed range. Ed25519 verification is strict and uses the
domain `PersonalAI.ModelPackage.Signature.v1\0`. The envelope key ID must equal
the active local signer configuration. Revoked, weak, mismatched, malformed, or
invalid signatures fail closed before tensor exposure.

The engine never generates or stores a production package-signing private key.
The deterministic testkit key is non-production and must never appear in an
approved package.

## Local admission

The model root must already exist, be a non-symlink directory, and not be
writable by group/other. V1 registrations are direct-child `.paimodel`
basenames. The file must be a non-symlink regular file not writable by
group/other. The loader uses `O_NOFOLLOW|O_CLOEXEC`, compares pre-open and
opened device/inode/length, bounds length before allocation, reads exactly, and
rejects growth/trailing bytes. Package hash, structure, signer, tokenizer,
tensor ranges/shapes/dtypes, memory, and architecture then validate in order.

## Offline conversion and signing gate

No production converter is implemented yet. It must be Rust-only and offline,
with no Python, third-party tokenizer/model loader, AI runtime, generated/model
code execution, or network access. For one approved source model it must:

1. record upstream artifact identity, license, redistribution/use decision,
   architecture/tokenizer versions, and source hashes;
2. independently parse only the required source data structures;
3. map an exact fixed tensor inventory and reject missing/extra/ambiguous data;
4. reproduce tokenizer and numeric vectors from independent reference evidence;
5. emit deterministic canonical bytes twice with identical hashes;
6. sign offline with an operator key unavailable to the engine;
7. publish package hash, signer/key status, tool source revision, Rust/toolchain,
   conversion parameters, SBOM, licenses, tests, and reviewer approvals.

Converter inputs and outputs must be treated as hostile. Limits apply before
allocation or arithmetic. Conversion must never make a source package
authoritative merely because it parsed.

## Activation, update, revocation, rollback

Activation is an owner configuration change in Go/service deployment, never a
model request. Updates receive a new exact package hash and explicit review.
Signer revocation blocks new loads. Rollback selects a previously reviewed exact
package and disables the newer Go provider/model binding. No auto-download,
self-update, remote registry lookup, or implicit latest-version behavior exists.

## Release evidence

A production model release must retain provenance/license review, source and
package hashes, signer ID/status, deterministic conversion evidence, independent
token/logit/generation vectors, corruption/resource/cancellation tests,
dependency lock/vendor archive, SBOM, advisory scan, target hardware results,
and rollback drill. Until that evidence exists, only the signed synthetic test
fixture is supported and the repository makes no model-quality claim.
