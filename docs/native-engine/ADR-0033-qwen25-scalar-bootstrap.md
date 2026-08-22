# ADR-0033: Closed Qwen2.5 1.5B scalar bootstrap profile

- Status: Accepted for portable scalar mechanics; production activation blocked
- Date: 2026-08-21

## Context

The legacy manifest and decoder identity encodes adjacent-pair rotary layout,
no projection biases, a fixed `1e-5` RMSNorm epsilon, and either tied or untied
output weights. Reinterpreting that identifier for Qwen2.5 would silently
change already-reviewed semantics. The three bootstrap candidates use the
Qwen2.5 decoder operation family, required Q/K/V projection biases, split-half
rotary layout, `1e-6` RMSNorm epsilon, and tied embedding/output weights. The
general and coder candidates declare 32,768 positions with theta 1,000,000;
the math candidate declares 4,096 positions with theta 10,000.

## Decision

Keep manifest v1.0 and `decoder-rms-rope-swiglu-gqa-v1` unchanged. Add manifest
v1.1 bound exclusively to
`qwen2.5-rms-rope-swiglu-gqa-bias-tied-v1`. Header bytes 84 through 87, which
must remain zero in v1.0, carry the explicit little-endian F32 RMSNorm epsilon
in v1.1. V1.1 requires tied output and RoPE scaling exactly 1.0. Unknown
version/architecture combinations fail closed.

V1.1 declares minimum engine revision 0.2.0. The current production runtime
identity remains 0.1.0, so the signed-package loader rejects v1.1 before
architecture or tokenizer activation. This deliberately prevents an older
0.1.0 engine that predates these semantics from appearing compatible. A future
release may bump the runtime identity only after the tokenizer, registry,
optimized backend, release, and qualification gates are reviewed together.

The v1.1 architecture validator is further closed to the exact Qwen2.5 1.5B dimensions:
151,936 vocabulary entries, hidden size 1,536, 28 layers, 12 query heads, two
key/value heads, and feed-forward size 8,960. It accepts only the two reviewed
context/theta combinations above. Every layer must contain the fixed legacy
weight inventory plus `q_proj.bias`, `k_proj.bias`, and `v_proj.bias`; an
explicit `output.weight` is prohibited because the trusted loader derives it
from `token_embedding.weight` for tied output.

The scalar correctness scaffold adds each validated bias after its corresponding
projection and applies Qwen split-half RoPE. Tiny deterministic configurations
exercise that math. The exact reviewed 1.5B configuration is not admitted by
the scalar backend: its position-zero estimate is 1,543,712,768 operations,
including 57,344 Q/K/V bias additions, while the immutable reference ceiling is
100,000,000. Model construction and execution planning reject it before weight
materialization; the limit is not weakened.

Future service activation has the same invariant: after trusted manifest
conversion and before tokenizer allocation or tensor materialization, the CPU
service constructs the checked Qwen transformer plan and applies scalar compute
admission. This Qwen-only hook leaves the legacy load order unchanged. The
current future-engine-version and incompatible-tokenizer gates deliberately run
earlier, so v1.1 remains unreachable in the production service today.

Resident Qwen execution would read the embedding tensor directly for logits,
so the tied output is not separately allocated or counted. Streamed planning
retains a logical `output.weight` alias to that exact source range. ARM64 and
Metal do not implement this new semantic profile and therefore reject it
explicitly. Package placement binds the actual manifest revision rather than
assuming v1.0.

## Evidence and boundaries

Portable Rust tests cover v1.0 preservation, v1.1 version/architecture binding,
the future-version production rejection, invalid epsilon/tied/scaling
declarations, the exact inventory, unreviewed configuration rejection, required
bias shape/presence, tied-output validation and unique resident accounting,
split-half RoPE, bias execution, deterministic tiny two-layer grouped-query
prefill/decode, cached-versus-full parity, and exact 1.5B scalar compute
rejection before weight materialization.

An offline transition slice now performs descriptor-bound inspection of the
three pinned config identities, the shared exact tokenizer graph/identity, and
the closed safetensors name/shape profile. It securely opens an exact
owner-private direct-child source set, streams and authenticates the profile-
specific SHA-256 from the retained descriptor, and only then allocates the
bounded header and validates 338 BF16 tensor descriptors and complete
non-overlapping ranges. It deliberately cannot emit a package or exact Qwen
tokenizer. This decision still adds no
chat-template execution, registry authority, model download, package approval,
or production qualification.
Existing AIPTok remains the only runtime tokenizer. The current authenticated
service first rejects v1.1 as requiring a future runtime; a second explicit
tokenizer gate prevents a future compatible runtime from silently applying
AIPTok's incompatible preserve-bytes/no-pretokenizer/BOS profile. Model Foundry
must finish the exact offline tokenizer conversion, connect future package
emission to an owned verified descriptor rather than reopening a path, issue
new signed provenance and registry artifacts, and produce independent
real-model numeric vectors. A separately
reviewed optimized backend must then demonstrate admission and parity without
weakening the scalar oracle's bound. Licensing, signing-key custody, actual
1.5B resource evidence, long-context checks, and Apple hardware qualification
remain external gates. Until those steps exist, this profile is portable
correctness scaffolding and must not be advertised as executable or qualified.
