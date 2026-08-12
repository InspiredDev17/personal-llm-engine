# Workspace test organization

Rust workspace-level architecture and supply-chain tests live in
`crates/engine-testkit/tests/` because a virtual Cargo workspace cannot own an
integration-test target directly. Crate-local unit tests remain beside their
implementations. All fixtures are generated deterministically in Rust.

The engine-server suite includes a normal, non-ignored bounded continuous soak.
It runs eight fresh-runtime cycles. Each cycle admits six authenticated Unix
streams with three active slots, prevents one client from reading until five
neighbors have completed, verifies all signed terminal events, requests a
graceful drain, proves empty scheduler state, and removes its runtime fixtures.
The configured `bounded-continuous-soak` workflow repeats the exact test eight
times: 64 lifecycle cycles and 384 synthetic generations, with a 15-minute job
cap. Its GitHub cron begins only after the workflow reaches the repository's
default branch. This is deterministic reference-runtime evidence, not
production-size or operated-hardware qualification.
