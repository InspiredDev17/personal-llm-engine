# ADR-0041: Consume canonical model execution admission without activation

- Status: Accepted; authenticated evidence consumer implemented, execution deferred
- Date: 2026-08-26
- Scope: provider-neutral package admission for Native Engine 0.2

## Context

Sovereign-AI-Contracts commit
`f3ea60596c33faca673ad368304f162b30f2ac2f` is now published and defines the
canonical Sovereign Models 2.1 admission and narrow `model-admission-v1` Rust
feature. The engine must consume that definition without treating signed
syntax as package registration, backend support, qualification, owner
activation, or execution authority.

## Decision

The engine pins the published Contracts commit, regenerates the checked-in
vendored package from that exact Git source, and enables only
`model-admission-v1` with its Models 2.0 and Runtime V3 dependencies. The
provider-neutral consumer:

- accepts only bounded recursively key-sorted canonical JSON and the closed
  Contracts schema;
- validates digest, freshness, exact admission/revision/lineage/sequence floor,
  complete Runtime V3 identity, engine profile and minimum engine release;
- binds exact canonical Models 2.0 descriptor bytes and exact package-catalog
  bytes;
- requires distinct active owner-pinned registry and package trust and verifies
  the Contracts-produced Ed25519 transcript;
- returns suspended and revoked documents as restrictive evidence while a
  separate owner gate is still required for development-authorized state.

The authenticated evidence type has no package-registration constructor. Its
execution-adapter check deliberately returns typed `NotImplemented`. No Models
2.1 CLI flag, service route, package fallback, production activation, or GLM
execution claim is exposed. The existing Qwen transition remains its own
development-only legacy admission. GLM-5.3-Flash remains source preflight and
checked non-executable planning only.

## Required follow-up

Define and review a package registration and authenticated runtime-attestation
flow before composing Models 2.1 into service execution. That future slice must
revalidate admission freshness, state, complete Runtime V3 identity,
descriptor/catalog identity, signer roles, rollback floor, package signature,
live backend identity and owner policy immediately before execution.

The admission must remain explicitly development-authorized and
production-closed. A valid admission may select only the exact already-supported
model/package/backend identity. It cannot supply a decoder, kernel, tokenizer,
qualification result, routing authority, or implicit fallback.
