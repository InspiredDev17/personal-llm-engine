# Portable-Code Completion Audit

Date: 2026-08-14

This audit separates source completion from model, hardware, operating-system,
credential, key-custody, and operated acceptance. It is not a production
approval and does not assign an artificial completion percentage.

## Scope and authority

The Rust repository remains the inference data plane. It validates and executes
registered signed artifacts, but it cannot route cognitive work, approve tools,
persist conversational memory, install models, access credentials, or control
Agent Green. AI-Avenger remains the Go control plane and routing authority.
Model Foundry owns acquisition, provenance, evaluation, qualification, signing,
and promotion. Sovereign-AI-Contracts owns canonical cross-repository wire
truth. Action Broker owns privileged execution.

The audit used the clean integration worktrees for
`personal-llm-engine@agent/native-inference-engine` and
`AI-Avenger-platform@codex/foundation-v0.1`, plus
`Sovereign-AI-Contracts@main`. Dirty owner worktrees were not reset, stashed,
cleaned, or incorporated.

## Classification key

- **A — code now:** implementation and deterministic acceptance are portable.
- **B — code now, final acceptance external:** source and deterministic tests
  are complete, but production acceptance needs a target or qualified artifact.
- **C — external or operated evidence:** no honest source substitute exists.
- **D — intentionally deferred:** prerequisites or measurements do not justify
  implementation yet.
- **E — blocked by model selection:** exact model facts are required.

## Finding register

Each row records current behavior, prior discrepancy, evidence, risk, required
work, acceptance, dependencies, order, and status.

| ID | Repository / location | Current behavior and evidence | Claimed behavior / risk | Class | Implementation and tests | External acceptance | Dependencies / order | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| F-01 | AI-Avenger `internal/model/providers/nativeengine` and PostgreSQL migration | Supervisor cursor is durable, source-bound, monotonic, compare-and-swap protected, and advances only after Agent Green delivery plus audit. Tests cover startup, restart, gaps, rollback, ring wrap/overwrite, database/audit/delivery failure, replay, and concurrent advancement. | An earlier in-memory cursor could lose distinction after a 64-slot overwrite or Go restart, producing false acknowledgement. | A | Durable cursor migration/store and fail-visible recovery were implemented and tested. | Operated PostgreSQL backup/restore and live Agent Green delivery remain F-10/F-11. | Must precede production bridge activation. | Complete in source. |
| F-02 | Rust release supervisor, Go native provider | Rust retains monotonic release-event identity; Go persists provider quarantine and exact-revision audited owner recovery. Model output cannot recover the provider. | Release failure must inhibit routing across restart without creating model, tool, memory, or Agent Green authority. | A | Monotonic sequence, cold-start source, durable quarantine, routing guard, and owner-only recovery tests are present. | Live supervisor crash/restart/recovery drill remains F-11. | F-01 and canonical event contract. | Complete in source. |
| F-03 | Sovereign-AI-Contracts `PAIRSEV1`, Rust producer, Go consumer | Canonical 76-byte content-free V1 record, Go/Rust strict bindings, accepted vectors, single-byte mutations, truncation, unknown-version, sequence and identity vectors are pinned. Go consumes the canonical decoder. | Competing parsers could drift or accept ambiguous records. | A | Canonical fixtures and both language suites pass; Rust independently pins the full canonical commit. | None beyond ordinary release CI reproduction. | Precedes operated bridge. | Complete in source. |
| F-04 | AI-Avenger native composition | One protected engine endpoint now composes a sorted closed population of 1–64 statically configured models. Every model is negotiated, registered-ID checked, loaded, descriptor-pinned, refreshed, and required for readiness. Refresh is deterministic and sequential. Cognitive labels are absent. | Earlier Go startup composed exactly one model, preventing heterogeneous qualified populations even though Rust already supported multiple resident models. | A | Strict bounded JSON, legacy single-model compatibility, duplicate/order/unknown-field rejection, per-model budgets, multi-model construction/readiness tests, race tests, and vet. | Real population capacity and routing quality remain F-09/F-13. | Control-plane registry and Rust multi-model lifecycle. | Complete in source. |
| F-05 | Rust ARM64 transformer/server/CLI; Go backend pin | Owner-selected `arm64` now loads a registered signed package, executes full prefill/cached decode/sampling/streaming through the closed ARM64 backend, reports authenticated backend identity, and retains scalar operation fallback. Go accepts and pins `arm64`. | Kernels previously existed only at immutable-graph level while service truthfully reported model execution unavailable. | B | Full transformer differential prefill/decode and signed authenticated UDS streaming tests run on AArch64. CLI selection is compile/target gated. | Production-size model parity, performance, thermal, pressure, long cancellation/soak, and CPU rollback on each supported Apple SKU. | Scalar oracle and signed package path were prerequisites. | Source complete; external qualification open. |
| F-06 | Rust Metal backend/service; Go backend pin | Complete signed synthetic decoder execution and owner-selected UDS path exist with exact descriptor pin and CPU fallback plan. | Synthetic and microbenchmark evidence cannot establish production model quality or operational stability. | B | Deterministic scalar parity, package, service, cancellation, unload, and memory tests exist. | Production package, driver-pressure/reset, thermal, throughput, soak, and operated CPU rollback. | F-09 qualified model and macOS target. | Source complete; external qualification open. |
| F-07 | Rust release verifier, ledger, systemd unit, operations docs | Source has signed release manifests, exact installed-artifact verification, monotonic rollback ledger, secure-path checks, Linux pre-launch unit binding, explicit recovery/rollback commands, and content-free failure events. No updater exists. | Source cannot authenticate its own installation or prove root ownership, unit semantics, atomic install, or rollback on the deployment host. | B | Static layouts, mutation tests, path/permission checks, rollback-ledger tests, and unit-text assertions exist. | Trusted installer/package, root-owned release/state/event roots, target systemd verification, tamper/upgrade/rollback drill. | Release key custody F-08 and Linux host F-11. | Portable implementation complete. |
| F-08 | Model and engine signing boundaries | Ed25519/SHA-384 roles are separate, bounded, pinned, rotatable with finite overlap, and private material is absent from repository/package/protocol/logging. | Test keys prove mechanics only; source cannot prove production key generation, custody, revocation, recovery, or separation of duties. | C | No additional inference source is required before the custody ceremony. | Offline signing system or hardware token, named custodians, public-key promotion record, backup/recovery drill, revocation drill. | Owner/security authority. | External. |
| F-09 | Model Foundry and model package admission | AIPack, AIPTok, F32/F16/BF16, AIPQ8/AIPQ4, exact decoder architecture, importer, loader, tokenizer, scalar decoder, KV cache, sampling, and signed synthetic end-to-end fixtures are implemented. | Synthetic fixtures do not prove that any production model is licensed, correctly mapped, useful, safe, or quality-preserving after quantization. | E | Do not guess additional tensor mappings, tokenizer families, architectures, or quantization profiles. | Exact model license, provenance, source hash, architecture/tokenizer mapping, conversion transcript, independent logits, task evaluations, quality thresholds, signed promotion. | Model selection and Model Foundry authority. | Blocked by selected model. |
| F-10 | Go PostgreSQL continuity | Durable cursor, quarantine, audit, replay and sequence stores have deterministic transaction/concurrency tests. | Unit/integration code cannot prove production database identity, backup durability, restore ordering, rollback detection, or operator recovery. | C | No schema weakening or in-memory fallback is permitted. | Dedicated PostgreSQL identity, migration record, backup/restore rehearsal, rolled-back-state rejection, concurrent-router drill, database outage recovery. | Platform owner/database operator. | External. |
| F-11 | Linux/macOS operated service and Agent Green | UDS permission, same-user peer credentials, timeouts, replay, drain, disconnect, slow-reader and bounded restart tests exist; Go delivers only content-free allowlisted events. | Developer tests cannot prove service UID/GID, real filesystem ACLs, supervisor behavior, crash dumps/log retention, live Agent Green receipt, or days-long stability. | C | Portable controls and runbooks are present. | Dedicated host, service accounts, protected roots/socket, live sidecar identities, no-egress observation, crash/restart/slow-reader soak, delivery/audit/ack trace. | Host owner, Go owner, Agent Green authority. | External. |
| F-12 | CUDA | No CUDA runtime, client-selected kernels, runtime compilation, or fake GPU success exists. | CUDA implementation without an NVIDIA host/toolchain and after-gate evidence would be untestable and violate the requested later-stage prerequisites. | D | Keep explicit unsupported behavior. Begin a separately reviewed C ABI/kernel slice only after prerequisites pass. | NVIDIA GPU, supported driver/CUDA toolkit, sanitizer/memcheck, single-GPU parity and pressure evidence before multi-GPU. | Stable scalar/ARM64/Metal/security gates plus hardware. | Intentionally deferred. |
| F-13 | Cross-repository model execution descriptor | AIPack and authenticated service descriptors carry the exact facts required by the sole accepted decoder/text profile: package hash, architecture, backend and context. Go pins artifact identity and separately owns cognitive capabilities/routing. The existing Contracts V1 model artifact descriptor predates AIPack and is not silently repurposed. | Adding tokenizer, quantization, modality, memory and backend-requirement fields independently in each repository would create competing truth; promoting package metadata directly could create execution authority. | D | Design a new canonical contract version only when a second execution architecture/modality makes the missing semantics concrete. Preserve closed enums and validation fixtures. | Cross-repository architecture review and migration evidence. | Sovereign-AI-Contracts owner; F-09 model/architecture selection. | Adequate for current profile; extension deferred. |
| F-14 | Additional model families/modalities | Engine rejects every architecture except `decoder-rms-rope-swiglu-gqa-v1` and every tokenizer family except AIPTok Byte-Level BPE V1. No package-defined operations exist. | Vision, audio, embeddings, reranking, encoder-only and encoder-decoder work require real reviewed execution semantics; accepting labels without implementations would be fake support. | E | Implement one closed architecture at a time only after exact model requirements and scalar vectors exist. | Qualified artifact, canonical modality contract, task evaluation and backend parity. | Model Foundry selection and F-13 contract evolution. | Blocked by selected models. |
| F-15 | Paged KV, continuous batching, speculative/beam/grammar/distributed execution | Contiguous KV and deterministic bounded fair scheduling are correct and isolated. Unsupported features fail explicitly. | Premature complexity would increase memory/cancellation/isolation risk without measured need. | D | Measure production workload first; introduce separate ADRs and scalar correctness models if justified. | Production workload traces and acceptance thresholds. | F-09 and operated measurements. | Intentionally deferred. |
| F-16 | Fuzzing, sanitizers and supply chain | All requested local parser boundaries have deterministic corpus/target support and bounded smoke evidence; vendored dependencies, exact lockfile, dependency inventory and offline checks exist. | Five-second smoke runs, local ASan/UBSan, and a cached advisory database are not sustained hosted assurance or a fresh release SBOM/advisory attestation. | B | Targets and CI definitions are source-complete; findings must remain release blocking. | Scheduled long campaigns, supported sanitizers/Miri where available, fresh advisory DB, release SBOM and license scan tied to exact artifact. | Hosted runners/tool availability. | Portable implementation complete. |
| F-17 | Rust workspace licensing | Dependency licenses are inventoried, but the repository's first-party Rust crates have no owner-selected license declaration. | Codex cannot choose the owner's legal distribution terms; inventing a license would create authority and could conflict with Creator OS commercialization plans. | C | No source algorithm gap exists. Add the owner-approved license text and package metadata only after the decision. | Written owner/legal decision covering private use, source distribution and any bundled Creator OS release. | Owner/legal authority. | External decision required. |

## Heterogeneous execution assessment

The engine is heterogeneous at the execution substrate without acquiring
selection authority:

- stable logical model and immutable package identities;
- closed architecture, tokenizer, quantization and backend identifiers;
- per-model signed tensor inventory, context and resource bounds;
- multiple concurrently registered/resident model IDs;
- per-model load/unload, scheduler ownership, KV isolation, cancellation,
  failure containment and authenticated descriptors;
- CPU reference, ARM64 and Metal owner-selected execution with truthful
  capability reporting and fail-closed fallback;
- no cognitive ranking or task-category field in Rust.

Current heterogeneity means multiple qualified instances of the one reviewed
decoder/text execution profile, potentially with different weights, contexts,
quantization and demonstrated Foundry capabilities. It does **not** mean that
vision, audio, embedding, reranking, encoder-only, encoder-decoder, arbitrary
tokenizer graphs, or model-defined operations are implemented.

## External completion register

| Gate | Why code cannot prove it | Required equipment / authority | Runbook or command family | Required evidence and pass/fail | Rollback | Acceptance owner |
| --- | --- | --- | --- | --- | --- | --- |
| Production model qualification | Model license, provenance, mapping and task quality are facts about a selected artifact. | Approved local source, license record, quarantine, Model Foundry signer/evaluators. | Model Foundry acquire → quarantine → import with explicit mapping/profile → evaluate → sign/promote; engine load by registered ID. | Pass only with exact source/package hashes, license/provenance, independent tensor/logit/tokenizer vectors, per-task thresholds, quantization deltas and signed promotion. | Revoke promotion, unload ID, restore prior approved package/registry sequence. | Model Foundry plus owner/legal where required. |
| First-party repository license | Distribution terms are an owner/legal decision, not a technical inference result. | Owner and legal review if the engine will ship with Creator OS. | Select the license; add exact `LICENSE` text and matching Cargo metadata; run the license/SBOM gate. | Pass only when repository policy and every published artifact state the same approved terms. | Do not distribute the engine until resolved; revert an incorrect license only through reviewed history. | Owner/legal authority. |
| Signing-key custody | Test keys and APIs cannot establish physical/organizational custody. | Offline signing host or hardware token; separate model/release/control/response/Agent Green roles. | Follow key generation, public-key enrollment, overlap rotation, revocation and recovery ceremony; never place private keys in command history. | Pass with custodian approvals, public fingerprints, separation proof, backup/recovery and revocation drill. | Revoke affected public identity, stop service, restore last trusted generation and rotate. | Owner/security authority. |
| Linux installation and rollback | Source cannot prove root ownership, mount/ACL semantics or systemd behavior. | Dedicated Linux host, root installer authority, service UID/GID, protected release/state/event/model roots. | `personalai-engine-release verify-supervised ...`; `systemd-analyze verify <unit>` where available; install/enable/start through reviewed package/runbook; execute documented rollback. | Pass with exact manifest/ledger hashes, `stat`/unit evidence, non-writable service roots, protected socket, failed tamper/downgrade launch, successful authorized rollback. | Stop unit, restore previous signed release and configuration, re-run verifier, retain events/audit. | Host owner and release authority. |
| macOS ARM64/Metal qualification | Synthetic tests do not establish production-size parity, thermal or pressure behavior. | Each supported Apple Silicon class, exact macOS/Rust build, qualified package. | Release build; scalar/ARM64/Metal differential suite; documented benchmark harness; long prompt/decode/cancel/unload/restart/pressure soak; Go canary with CPU fallback. | Pass only within model-approved numerical/logit/token tolerances, zero isolation faults/leaks, bounded memory, stable throughput/temperature and successful rollback. | Restart Rust and Go with the pinned `cpu-reference` descriptor and prior package. | Engine owner, Model Foundry for quality, platform owner for canary. |
| PostgreSQL continuity | In-process tests cannot prove actual backup, restore or identity behavior. | Production-like PostgreSQL and service identity. | Apply migrations; stop router; generate more than 64 supervisor events; restore/restart; inject delivery/audit/database failure; run owner-authorized recovery. | Pass with no false ack, visible gap/rollback/reset, monotonic CAS advancement only after delivery+audit, repeated delivery allowed, exact audit trail. | Stop routers, preserve ring/database, restore last accepted database backup, use exact-revision recovery only with owner approval. | Go control-plane owner/database operator. |
| Agent Green liaison | Mock/local sinks do not prove independent-domain authentication or receipt. | Agent Green sidecar/service identity and keys, Go application identity. | Start protected local endpoints; inject each allowlisted content-free event; observe delivery, audit, then source acknowledgement; test outage/replay. | Pass with mutual authentication, category/ID/sequence only, no prompts/output/paths/secrets, at-least-once replay, no model influence. | Disable routing integration while keeping provider quarantined and evidence retained; rotate liaison keys if compromised. | Agent Green authority and Go owner. |
| Sustained soak and release security | Short deterministic tests cannot expose long thermal, allocator, driver or restart behavior. | Hosted runners and deployment-class hosts. | Scheduled fuzz/sanitizer workflows; multi-day UDS slow-reader/cancel/restart/model-load/unload soak; fresh `cargo audit`; release SBOM/license scan. | Pass with no crash, sanitizer finding, unbounded growth, stale KV, stuck drain, replay failure, advisory or undeclared artifact. Any finding fails release. | Stop canary, preserve artifacts/logs/content-free events, return to last signed release/backend. | Release/security authority. |
| CUDA qualification | The current host has no NVIDIA execution environment and later-stage prerequisites govern start. | Supported NVIDIA GPU, driver, CUDA toolkit and memory/sanitizer tools. | After a separate ADR: build fixed kernels, run C ABI ownership and scalar-parity suites, cuda-memcheck/sanitizers, single-GPU soak; only then consider multi-GPU. | Pass only with exact capability detection, bounded allocation, no runtime/model code loading, parity tolerances, clean device-loss/OOM/cancel/unload behavior. | Disable CUDA feature/backend and restart on CPU; preserve quarantined failure evidence. | Engine owner and release/security authority. |
| Live canary | Source tests cannot prove routing, workload or human acceptance in the operated system. | Approved model, host, keys, service identities, PostgreSQL, Go and Agent Green. | Enable one statically pinned model/backend for an allowlisted traffic slice; monitor content-free health/resource/security metrics; exercise cancellation/quarantine/recovery/rollback. | Pass only with correct attribution, no authority boundary violation, stable resources/latency, accepted quality and demonstrated rollback. | Route to prior qualified local provider or disable native provider; unload model; preserve audit. | Platform owner; security recovery remains Agent Green/owner controlled. |

## Portable verification commands

Run from `engine/`:

```text
cargo fmt --all --check
cargo check --workspace --all-targets --all-features --locked
cargo build --workspace --all-targets --all-features --locked
cargo clippy --workspace --all-targets --all-features --locked -- -D warnings
cargo test --workspace --all-targets --all-features --locked
cargo test --doc --workspace --all-features --locked
cargo doc --workspace --all-features --no-deps --locked
cargo check --workspace --all-targets --no-default-features --locked
cargo audit
```

Run from AI-Avenger:

```text
go test ./...
go test -race ./...
go vet ./...
go mod verify
```

Run from Sovereign-AI-Contracts:

```text
make offline
```

The Unix-socket and Metal tests require the local sandbox or host to permit
their system calls. A sandbox denial is an environment failure, not a reason to
weaken the tests.

## Stop line

After the Category A and B source work above passes all gates, the remaining
items are not “nothing left.” They are required acceptance work with different
owners and evidence. Codex should continue to fix any newly discovered
portable defect, but must not fabricate a model mapping, private key, NVIDIA
device result, production identity, OS permission proof, sustained soak, or
canary approval.
