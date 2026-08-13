# ADR-0020: Model Foundry registry V2 and directory-package boundary

- Status: Accepted; Phases 1 through 3 implemented
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

At the review base, Model Foundry V1 emits a different flat `.paimodel` file, signs a
different transcript, hashes the complete flat file, and records
`personalai.decoder-v1` in registry schema V1. Sharing a suffix does not make
the formats compatible. The review-base engine does not parse the Model Foundry
registry.

## Decision

The cross-repository target retains this engine's directory-backed package V1
as the production artifact contract. Model Foundry must implement a separately
versioned directory writer and terminal verifier before its output can enter
this loader. Flat Model Foundry V1 is neither translated nor treated as a
fallback.

The implemented selection contract is Model Foundry registry schema V2, signed
under `PersonalAI.ModelRegistry.Entry.Signature.v2\0`. It binds SHA-384 of
`hashes.aiph` as `package_catalog_sha384` and the exact profile
`directory-paimodel-v1+decoder-rms-rope-swiglu-gqa-v1+paibpe01+f32-cpu-reference`.
It names distinct `registry_signer` and `package_signer` identities.

The Go control plane and deployment owner remain authoritative for exact
registry sequence/entry selection, both public trust roles, the nondecreasing
external anti-rollback checkpoint, activation, and resource budgets. The engine
derives the directory path from exact owner-selected `ModelId` and `ArtifactId`;
it does not accept request paths, query a remote registry, choose `latest`, or
fall back after registry verification failure.

Phase 1 added bounded parsing, signature verification, fuzzing, and shared
positive/negative vectors. Phase 2 added a separate registry-backed registration
mode and terminally bound the selected registry decision to package loading.
Phase 3 makes the mode and exact selection available through owner startup
configuration and binds the same public decision into the Go adapter.

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
bounded registry parser/verifier. Phase 2 introduces
`RegistrationMode::ModelFoundryRegistryV2`. Owner selection binds the exact
sequence, envelope SHA-384, `ModelId`, `ArtifactId`, distinct trusted registry
and package signers, and a nonzero anti-rollback sequence floor. The engine
securely opens only `registry/<sequence>.registry.json`, verifies it before
package access, and derives
`<approved>/<ArtifactId>/<ModelId>.paimodel` from the verified entry.

Loading rechecks the signed catalog identity, package signature, context,
tokenizer hash, provenance hash, and complete sorted shard-hash set before model
allocation. Existing owner-pinned registration remains a separate explicit
mode; registry failure never selects it as fallback. A trusted adapter still
cannot repair a legacy flat-package mismatch.

Phase 2 tests reject an absent exact sequence when a differently numbered entry
exists, a symlinked registry entry, a symlinked artifact root, wrong
owner-selected identities, signer-role mismatch, signature corruption, and
anti-rollback mismatch. The frozen positive vector proves bounded parsing and
signature verification. Subsequent deterministic importer changes altered the
synthetic package ledger, so the terminal-load integration test now proves that
the previously signed decision is rejected for the current package. A new
positive terminal-load vector requires a fresh offline Model Foundry approval;
the engine cannot manufacture or silently update that independent decision.

Phase 3 requires `--registration-mode` for every serving process. Registry mode
requires the exact model/artifact IDs, sequence, envelope SHA-384, independently
retained minimum sequence, and distinct active registry/package public signer
identities. Owner-pinned package flags are rejected in registry mode and all
registry flags are rejected in owner-pinned mode. The Go `NATIVE_ENGINE` adapter
binds the same public selection, refuses implicit mode, revoked roles, identity
reuse, and rollback below the retained checkpoint, and contains no artifact
path. Restart and bounded rotation tests preserve that decision. Runtime model
requests still carry only `ModelId` and cannot select or mutate registration.
