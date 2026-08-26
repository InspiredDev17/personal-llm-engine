# ADR-0040: GLM-5.3-Flash remains an exact, non-executable source profile

- Status: accepted
- Date: 2026-08-26
- Scope: offline source inspection and host-independent resource planning only

## Decision

The engine recognizes one immutable upstream metadata set for
`zai-org/GLM-5.3-Flash` revision
`3f1971b7b5f7a528c9c4ef6212c8785298a8c24a`. Recognition is deliberately
limited to the offline importer and a separate transformer architecture/state
planner. It does not add a package format, registry record, tokenizer runtime,
decoder, service route, protocol capability, backend activation, or model
qualification.

The four large source metadata documents are admitted by exact byte length and
SHA-256 identity. This is stricter than accepting a configurable GLM-like JSON
shape and avoids adding a general third-party tokenizer or model loader:

| Artifact | Bytes | SHA-256 |
| --- | ---: | --- |
| `config.json` | 69,416 | `bb8f01c42cb92a52ca72e65afb4d5bd8d11aef083cd210e8de25dfb904f23e9f` |
| `tokenizer.json` | 20,217,442 | `19e773648cb4e65de8660ea6365e10acca112d42a854923df93db4a6f333a82d` |
| `processor_config.json` | 909 | `aae38374c94b08cc9b0547c6e64f05b951bd9735cea571c6988f5ed552bed3ed` |
| `model.safetensors.index.json` | 8,406,613 | `3c3f40366a53c3fd7974b4eab7881a365a98c2a4329150befebab99fe7c18b05` |

The index describes 62 shards, 76,108 tensors and 328,326,771,576 tensor
payload bytes. Those numbers are source inventory, not a statement that a
package exists or that any host can load the weights.

## Exact architecture facts retained

The content-free plan fixes these reviewed text-graph facts:

- 45 layers: 34 Kimi Delta Attention layers and 11 DeepSeek Sparse Attention
  layers at exact indices;
- three dense MLP layers followed by 42 sparse MoE layers;
- 288 routed experts, eight selected experts, and one shared expert;
- sigmoid `noaux_tc` routing with normalized selected scores and scaling 2.5;
- KDA 64-by-128 heads with convolution width four;
- DSA query rank 1,536, KV rank 512, NoPE 256-dimensional Q/K, 256-dimensional
  values, and a 32-by-128 indexer with top-k 2,048 and pool width four;
- mHC multiplier four, 20 Sinkhorn iterations, and epsilon `1e-6`;
- 4,096 hidden width, 154,880 output rows, and a declared 1,048,576-token
  context ceiling.

The source safetensors parser can represent `F8_E4M3` and account for its
one-byte storage width. That source enum intentionally has no corresponding
runtime tensor dtype. Conversion or execution returns typed `NotImplemented`.

## Resource accounting

The experimental planner performs checked arithmetic for a bounded token and
sequence count. It accounts separately for:

- KDA convolution state;
- FP32 KDA recurrent state;
- DSA compressed MLA state;
- DSA index state;
- lower-bound mHC working state; and
- lower-bound active/shared expert working state.

The plan is not an allocator layout or hardware fit result. It excludes model
weights, backend alignment, allocator overhead, graph workspaces, vision state,
and kernel-specific scratch. The source weight byte count is reported
separately because only a future qualified placement implementation can decide
how much must be resident.

Planning up to the model-declared context is allowed so overflow and capacity
can be evaluated without raising the engine's existing 262,144-token runtime
limit. Any plan above that current runtime limit reports a blocker and remains
non-executable.

## Fail-closed status

The readiness report is finite and content-free. It always reports execution
unavailable while any of these gates remain open:

- signed package profile and decoder;
- exact tokenizer runtime and conformance corpus;
- KDA, DSA, and mHC implementations;
- scale-bound FP8 decoding and numerical evidence;
- multimodal processor and vision execution;
- runtime support for the declared context; and
- independent package/model/backend qualification.

`require_executable` returns `UnsupportedArchitecture`. The client protocol,
ordinary CLI, loader, server, and model package format cannot reference or
activate the profile. Architecture tests preserve that confinement.

## Consequences

This foundation can detect source substitution, compute deterministic minimum
state requirements, and expose concrete implementation blockers before model
download or hardware deployment. It cannot generate text, process images or
video, prove Metal/CUDA compatibility, establish model quality, or authorize a
production model. Those require separate implementation, Foundry evidence,
owner policy, and operated qualification.
