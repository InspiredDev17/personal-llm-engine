# ADR-0029: Canonical inference-runtime-v3 reservation semantics

- Status: Accepted; authenticated admission transport implemented, execution and recovery pending
- Date: 2026-08-14
- Contract: Sovereign-AI-Contracts `inference/runtime v3.0.0`
- Current consumer pin: `138128520dea7a602c0cf2ea905fa32396917db8`

## Context

The existing authenticated binary service safely executes one registered model
identifier, but it predates a canonical cross-repository contract binding the
control plane's logical model identity to the engine registry model, artifact
digest, backend instance, resource admission, task correlation, cancellation,
and terminal result. Mutating that released protocol independently would create
a competing contract and could permit identity or budget substitution.

## Decision

`engine-server` implements a transport-neutral reservation state machine for
the immutable Sovereign AI Contracts inference-runtime-v3 semantics. It binds:

- one logical control-plane model ID and one engine registry model ID;
- one artifact ID and verified SHA-384 package digest;
- backend ID/version/instance and engine instance;
- reservation, request, and task identity;
- provider-neutral input/output/memory/time ceilings;
- a nonzero idempotency key and absolute expiry.

The manager is finite and synchronized. It permits one first consumption,
releases accounting on cancellation/release/expiry, returns the original
decision for an exact replay, and rejects altered replay, authority theft,
identity substitution, output escalation, double consumption, terminal state
regression, and concurrent overcommit. It provides an explicit method to
revalidate engine-owned registry model, package hash, backend, and instance
identity immediately before reservation or execution.

The semantic state module is not a second wire contract. The separately pinned
Contracts adapter validates the externally owned v3 schema and maps it into
these engine types. Cognitive roles, qualification, routing, cloud policy,
tools, memory, owner authorization, and Agent Green authority remain absent.

## Consequences

Host-independent reservation semantics and concurrent resource-accounting
failure modes can be proven before production hardware or models exist. The
state is currently in-memory; process restart does not yet reconcile accepted
reservations. The protected UDS service exposes canonical
availability/reservation/cancellation/release documents, but not canonical v3
execution. Therefore this is authenticated admission integration evidence, not
an operated v3 inference or production admission claim.

Before activation, add exact canonical execution request/result dispatch, Go
transport conformance, a bounded durable journal or explicit supervisor
reconciliation design, scheduler cancellation/deadline mapping, and
restart/expiry/terminal-result recovery tests. These additions must preserve
the canonical contract rather than inventing a repository-local substitute.
