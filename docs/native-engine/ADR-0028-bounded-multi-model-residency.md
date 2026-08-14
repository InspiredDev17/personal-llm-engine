# ADR-0028: Bounded multi-model residency

Status: accepted (portable source behavior); operated population qualification pending

Date: 2026-08-14

## Context

The service runtime already stores registered packages and loaded models in
model-ID-keyed maps, but the production CLI exposed one startup registration
and loaded it immediately. The Go control plane consequently loaded every
configured model and made every package readiness-critical. That was safe for a
single model but imposed unnecessary permanent residency on a heterogeneous
population.

The engine must own execution safety without acquiring cognitive routing
authority. It may load or unload only the exact registered ID requested by the
authenticated Go control plane and must never substitute a package.

## Decision

Owner-pinned serving retains one required primary model for backward-compatible
startup. It may additionally register at most 127 cold models through repeatable
closed eight-field `--additional-owner-model` records. Each record contains
only model ID, direct-child package basename, exact SHA-384 package identity,
trusted Ed25519 signer identity, and an owner-defined resource budget. Duplicate
model IDs and package basenames fail before service construction. Signing roles
must be distinct from each other and from control-plane and response identities.

Only the primary model is loaded at startup. Additional packages receive trusted
load budgets but remain unloaded until an authenticated `LoadModel` request
names the exact registered ID. `UnloadModel` remains idle-only. Registration,
load, unload, scheduler ownership, model handles, tensor/KV state, and metrics
remain keyed by model identity. No client supplies paths, signers, budgets,
backend names, versions, or fallback targets.

The Go control plane assigns execution-only `KEEP_HOT` or `LOAD_ON_DEMAND`
residency policy. It verifies registration for every configured model, loads
and descriptor-pins keep-hot models at startup, and exposes explicit exact-ID
activation/deactivation for cold models. A cold model is unavailable to routing
until its authenticated descriptor and fresh resource state pass. Go remains
the only component that may decide which qualified model to request.

Go invalidates the prior registry resource snapshot before activation or
deactivation. This is a local revocation, not fabricated availability evidence:
only a later authenticated engine status can restore routing. Failed unload
therefore leaves the still-resident model unavailable instead of routable under
its prior five-second snapshot.

## Consequences

- A service can represent a bounded heterogeneous population without holding
  every package resident.
- Failed load and descriptor mismatch cannot silently make another model
  routable.
- All-on-demand Go configuration can prove the authenticated engine is alive
  without claiming that an unloaded model is available.
- Automatic eviction is intentionally absent. A future policy needs measured
  pressure data, deterministic ordering, and its own review.
- The current repeatable cold-registration form is owner-pinned only. A
  multi-entry Model Foundry registry selection requires a canonical signed set
  contract rather than independently inventing one in this CLI.
- The source limit of 128 configured models is a bounded initial population,
  not evidence that 128 production packages fit simultaneously or route well.

## Verification

Tests cover closed parsing, duplicate rejection, registry-mode rejection,
bounded budgets, separate loaded-model residency, idempotent stale-unload
rejection, Go cold startup, exact activation, descriptor pinning, refreshed
routability, pre-transition resource invalidation, failed-unload fail-closed
behavior, exact deactivation, and unknown-ID rejection. Hardware/model
capacity and production eviction policy remain external acceptance work.
