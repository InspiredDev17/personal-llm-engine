# ADR-0001: Engine Boundaries

- Status: Accepted
- Date: 2026-08-11

## Decision

Keep the existing Go PersonalAI platform as the authoritative control plane and
implement inference as a separate Rust data plane in this repository. Rust owns
only package admission, tokenization, tensors/memory/KV, backend execution,
sampling, inference scheduling, authenticated local transport, and typed
inference/security events.

Go owns model/provider registration policy, routing, privacy, prompts/context,
memory, tools, approvals, audit, credentials, lifecycle policy, and Agent Green
liaison. No engine result is an authorization. The engine cannot call tools,
mutate memory, download/update, use arbitrary files, open outbound connections,
or connect directly to Agent Green.

Use narrowly scoped crates with an exact tested dependency DAG. CPU reference
is the default correctness oracle. ARM64/Metal/CUDA are explicit default-off
backends. The only service endpoint is a protected UDS with fixed bounded binary
messages and Go request signatures. The existing HTTP-compatible adapter is not
reused.

## Consequences

The architecture minimizes authority and parser/kernel supply-chain coupling,
keeps replacement reversible in Go, and lets the CPU oracle validate later
hardware backends. It also requires first-party package/tokenizer/compute code,
a Go-specific UDS adapter, offline package operations, and more acceptance work
before production performance is available.

## Rejected alternatives

- Embedding inference into the Go control plane: couples failure/resource
  domains and language ownership.
- Reusing the Go secure-tool Rust executor: conflates inference with tool
  authority.
- HTTP/TCP or an Ollama-compatible surface: expands network attack surface and
  makes a third-party protocol a first-party boundary.
- Vendoring an existing AI runtime/tensor/tokenizer stack: violates provenance,
  dependency, and long-term sovereignty goals.
- Model-defined graphs/plugins/kernels: turns data into executable policy.

## Current evidence

The decision is implemented through a signed synthetic package: protected UDS,
strict Go-message authentication, registry-only model selection, no-follow
package open, V1 parser/tokenizer/tensor/decoder/greedy generation, bounded
scheduler, typed events, and the Go `NATIVE_ENGINE` provider. Production
converter/model, accelerated backends, and cutover are separate gates.
