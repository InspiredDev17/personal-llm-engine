# ADR-0020: Model Foundry registry V2 and directory-package boundary

- Status: Accepted; Phase 1 bounded parser implemented
- Date: 2026-08-12
- Native Engine review base: `f458c46cd7cd69e73fddd575669604f99373acce`
- Model Foundry review base: `d6ab7b99fa083a76688ac5ef4b32b2e31480118c`

## Context

The Native Engine review base loads a signed directory-backed `.paimodel` V1.
The package identity is SHA-384 of exact `hashes.aiph` bytes. Strict Ed25519
signs the catalog digest under
`PersonalAI.DirectoryModelPackage.Signature.v1\0`. The package architecture is
`decoder-rms-rope-swiglu-gqa-v1`, its tokenizer is `PAIBPE01`, and production
storage is F32.

Model Foundry V1 currently emits a different flat `.paimodel` file, signs a
different transcript, hashes the complete flat file, and records
`personalai.decoder-v1` in registry schema V1. Sharing a suffix does not make
the formats compatible. The engine does not parse the Model Foundry registry.

## Decision

The cross-repository target retains this engine's directory-backed package V1
as the production artifact contract. Model Foundry must implement a separately
versioned directory writer and terminal verifier before its output can enter
this loader. Flat Model Foundry V1 is neither translated nor treated as a
fallback.

The proposed selection contract is Model Foundry registry schema V2, signed
under `PersonalAI.ModelRegistry.Entry.Signature.v2\0`. It binds SHA-384 of
`hashes.aiph` as `package_catalog_sha384` and the exact profile
`directory-paimodel-v1+decoder-rms-rope-swiglu-gqa-v1+paibpe01+f32-cpu-reference`.
It names distinct `registry_signer` and `package_signer` identities.

The Go control plane and deployment owner remain authoritative for exact
registry sequence/entry selection, both public trust roles, the nondecreasing
external anti-rollback checkpoint, activation, and resource budgets. The engine
will eventually derive the directory path from exact owner-selected `ModelId`
and `ArtifactId`; it will not accept request paths, query a remote registry,
choose `latest`, or fall back after registry verification failure.

This ADR does not implement a registry parser or change the loader. Those are
Phase 1 changes requiring bounded parsing, fuzzing, shared positive/negative
vectors, and a separately reviewed engine configuration schema.

## Frozen evidence

The authoritative cross-repository vectors are at
`Model-Foundry/fixtures/native-engine-registry-v2/vectors.rs` on the review base
descendant. The Phase 0 file SHA-384 is
`91910351ad0229c730635a02ea0debbd001844bc7352a8ad2fa2af969f5f3f04e5b519177c136db6513b772ac4a48da0`.
They contain canonical signed rotation entries, owner trust state, negative
registry envelopes, and the package fingerprint ledger without private seeds
or committed package bytes.

The engine's deterministic importer test independently regenerates the
directory package and pins every file length/hash, catalog identity, and public
test signer. That fixture is synthetic mechanics evidence only; it is not a
model approval or a trusted production identity.

## Consequences

Phase 1 implements both the Foundry directory writer/verifier and this engine's
bounded registry parser/verifier. It does not connect a registry decision to
package path derivation, registration, activation, or loading. A trusted adapter
still cannot repair a legacy flat-package mismatch, and no cross-repository
positive production load is claimed before Phase 2.

Existing owner-pinned directory registration remains the only implemented
production admission mode. Any future coexistence with registry-backed mode is
explicit and downgrade-proof; registry failure never selects legacy mode.
