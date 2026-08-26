# ADR-0032: Consume canonical inference runtime v3 without importing authority

## Status

Accepted for authenticated admission integration. Operated inference execution
and deployment evidence remain pending.

## Context

The engine already owned a bounded reservation state machine, while Sovereign
AI Contracts owned the provider-neutral runtime-v3 request, decision, execution,
result, release, and cancellation documents. Reimplementing their JSON grammar,
timestamp normalization, identity rules, or cross-document validation here
would create a second permanent wire contract and allow Go/Rust drift.

The full Contracts library also contains control-plane domains the inference
data plane must not import. A direct default-feature dependency would therefore
blur the authority boundary even if those modules were unused.

## Decision

The engine pins exact Contracts commit
`f3ea60596c33faca673ad368304f162b30f2ac2f` and compiles only its narrow
`model-admission-v1` feature plus its `models-v2` and `runtime-v3` dependencies.
The resulting dependency graph contains common typed IDs, generation data,
Runtime V3 documents, Models 2.0 package identity, and Models 2.1 admission only. Cognitive roles, qualification
authority, tool authority, memory authority, security authority, and component
administration are excluded.

`RuntimeV3ContractAdapter` performs strict bounded document decoding, maps exact
canonical identity into engine-owned types, requires SHA-384 AIPack identity,
revalidates live engine-owned identity before reservation and execution,
enforces single-consumption, cancellation, release, token/resource ceilings,
and exact idempotent result replay. Canonical documents themselves are not an
authentication mechanism. They may be dispatched only inside the existing
signed, replay-journaled, same-user Unix-domain-socket service envelope.

The closed service catalog now carries availability, reservation,
cancellation, release, and execution documents as bounded opaque payloads. The
server decodes them only through the pinned Contracts implementation, binds
owner-configured logical/artifact correlations to live registered package and
backend state, and revalidates that state before reservation. Availability
revisions derive monotonically from the authenticated service clock; a clock
rollback therefore makes cached consumers reject status instead of accepting
an older revision.

The native consumer profile intentionally narrows canonical request IDs to
`req_` plus exactly 32 lowercase hexadecimal characters so they map losslessly
to the engine's nonzero 128-bit request identity. A broader canonical ID is
valid Contracts data but unsupported by this native runtime profile.

## Consequences

- Cross-repository validation and timestamp semantics have one implementation.
- The engine remains unaware of cognitive plans, roles, policy, cloud, tools,
  memory, approvals, and Agent Green authority.
- Serde/JSON and the Contracts SHA-256 stack enlarge the reviewed offline graph;
  exact versions, sources, vendor contents, and SPDX entries are enforced.
- Signed UDS admission and canonical execution dispatch are integration-tested
  with synthetic packages, development keys, the real bounded decoder cursor,
  concurrent canonical cancellation, and a separately gated fault simulator.
  Execution is available only through bounded continuous serving; synchronous
  single-connection execution fails closed. Actual qualified-package
  control-plane proof and durable accepted-reservation recovery remain required
  before this is an operated inference-runtime-v3 service.
- Production model, hardware, latency, throughput, resource-pressure, and soak
  evidence remain separate deployment gates.
