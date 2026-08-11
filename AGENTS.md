# Native Engine Engineering Rules

These rules apply to the entire repository.

## Authority boundary

- The external PersonalAI Go platform is the control plane and system authority.
- Rust in this repository is an inference-only data plane.
- Models are untrusted data. They cannot authorize, execute tools, modify
  memory, choose filesystem paths or kernels, download code or data, update the
  engine, or open network connections.
- Agent Green remains a separate repository, process, deployment, and trust
  domain. Engine security events flow outward to the Go control plane; no model
  controls Agent Green.

## Implementation rules

- Do not add training behavior.
- Do not copy, vendor, or depend on third-party AI runtimes, tensor libraries,
  tokenizers, model loaders, schedulers, model servers, or GPU-kernel packages.
- Do not add Python or TypeScript to production, tests, conversion, fixtures, or
  build-critical paths.
- Do not add TCP listeners or network-client dependencies. A protected
  Unix-domain socket is the only planned service transport.
- Keep unsafe Rust forbidden by default. Any future exception belongs in an
  isolated backend or FFI module with an ADR, reviewed invariants, and tests.
- Use checked arithmetic and hard bounds for all untrusted sizes, offsets,
  dimensions, token counts, frames, and allocations.
- Parse first-party data formats explicitly. Reject unknown versions, trailing
  ambiguity, overflows, recursive structures, executable payloads, and
  model-selected plugins, paths, or kernels.
- Unimplemented behavior must return a typed unsupported or not-implemented
  error; never return fake success.

## Dependencies and verification

- The dependency policy and allowlist record in
  `docs/native-engine/DEPENDENCY-POLICY.md` is mandatory.
- Pin resolved versions in `Cargo.lock`, disable unnecessary default features,
  and prohibit build scripts that fetch remote code or runtime dynamic-library
  discovery.
- Before claiming a slice complete, run formatting, workspace build/check,
  Clippy with warnings denied, tests, documentation tests, and source/dependency
  boundary audits. Report unavailable checks precisely.
- Preserve unrelated work and never weaken the Go control plane or Agent Green
  trust boundaries.
