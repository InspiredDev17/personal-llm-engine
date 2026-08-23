# ADR-0039: Qwen2.5 Experimental Development Execution

Status: Accepted for source-complete development use; production activation is
unavailable.

## Context

PAIDIR01 v1.1 and `AIPTOK02` require engine 0.2.0, but the frozen Model Foundry
Registry V2 profile describes only the legacy v1.0 architecture. Mutating V2
would break its signed meaning. Sovereign AI Contracts commit
`346b04f5899facce34fe89ea3292d139bcb7de05` supplies the canonical Models 2.0
execution-package descriptor but does not manufacture qualification or owner
activation.

Development execution is required to generate real local evidence before any
production design can be reviewed. A descriptor or package signature alone
must not turn a model into an executable service.

## Decision

The workspace release is 0.2.0. Registry V2 remains frozen. A separately
versioned engine-private Registry V3 adapter accepts exactly one Qwen2.5
development decision with:

- status `development_fixture_authorized`;
- `development_execution_authorized: true`;
- `production_activation: false`;
- a monotonic nonzero sequence and owner-pinned anti-rollback floor;
- distinct registry and package Ed25519 identities;
- exact provider model, canonical artifact, Models 2.0 descriptor and PAIDIR01
  catalog SHA-384 identities; and
- compatibility
  `personalai.directory_paimodel+qwen2.5+aiptok02+bf16+arm64-v1`.

The strict canonical JSON record is transitional and local. Models 2.0 remains
the cross-repository package/runtime identity authority. V3 cannot authorize
production and cannot widen Models 2.0.

The canonical descriptor must bind the exact six members in order: manifest,
tokenizer, provenance, tensor shard 0000, integrity catalog and integrity
signature. It also fixes the Qwen architecture, split-half RoPE semantics,
unquantized BF16 profile, `AIPTOK02` identity, 151,936 model rows and 151,665
defined/selectable tokenizer rows. Every digest and every model, artifact,
backend, backend-instance and engine-instance identity is compared exactly.

Registration is not execution. Startup must additionally supply:

- `--registration-mode experimental-qwen-registry-v3`;
- ARM64 as the selected backend;
- an exact matching Runtime-v3 identity; and
- `--experimental-qwen-development-execution owner-approved`.

The default is disabled. Client frames and model data cannot supply or change
the approval. CPU-reference and Metal remain ineligible. The ARM64 loader
revalidates the signed descriptor immediately before load, constructs the
tokenizer only from the verified package bytes, enforces resource ceilings,
polls cancellation during full package hashing and tensor materialization, and
drops partial state on failure. The owner CLI installs its bounded signal bridge
before the initial load, propagates `SIGINT`/`SIGTERM` into that cooperative
cancellation flag, joins the watcher and restores the prior handlers before
opening the service listener.

The offline importer may export the exact pinned 4,613,135-byte `AIPTOK02` for
Model Foundry. It accepts only the pinned source, creates one new file in a
same-owner private directory without following links or overwriting, verifies
both fixed output digests through the opened descriptor, synchronizes it and
publishes mode `0400`. The exporter is supply-chain tooling, not service
authority.

## Consequences

The engine can consume a genuine signed development package once Model Foundry
produces it, without pretending that the package is production qualified.
Registry history is preserved and the canonical package contract is not copied
or redefined locally.

This ADR does not prove a real package, license approval, Foundry qualification,
ARM64 numerical parity, memory pressure, performance, soak, production keys or
production activation. Those remain separate evidence and authority gates.
