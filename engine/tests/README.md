# Workspace test organization

Rust workspace-level architecture and supply-chain tests live in
`crates/engine-testkit/tests/` because a virtual Cargo workspace cannot own an
integration-test target directly. Crate-local unit tests remain beside their
implementations. All fixtures are generated deterministically in Rust.
