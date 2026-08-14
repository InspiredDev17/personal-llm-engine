# ADR-0030: Consume canonical inference runtime v3 without importing authority

## Status

Accepted for source-code integration. Operated service composition remains
pending an authenticated runtime-v3 service dispatch and deployment evidence.

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
`138128520dea7a602c0cf2ea905fa32396917db8` and compiles only its minimal
`runtime-v3` feature. The resulting dependency graph contains common typed IDs,
generation data, and runtime-v3 documents only. Cognitive roles, qualification
authority, tool authority, memory authority, security authority, and component
administration are excluded.

`RuntimeV3ContractAdapter` performs strict bounded document decoding, maps exact
canonical identity into engine-owned types, requires SHA-384 AIPack identity,
revalidates live engine-owned identity before reservation and execution,
enforces single-consumption, cancellation, release, token/resource ceilings,
and exact idempotent result replay. Canonical documents themselves are not an
authentication mechanism. They may be dispatched only inside the existing
signed, replay-journaled, same-user Unix-domain-socket service envelope.

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
- The adapter and deterministic tests are source-code evidence only. Until the
  signed service dispatcher and a control-plane transport are composed, this is
  not an operated inference-runtime-v3 service.
- Production model, hardware, latency, throughput, resource-pressure, and soak
  evidence remain separate deployment gates.
