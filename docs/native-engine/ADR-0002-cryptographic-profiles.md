# ADR-0002: Cryptographic Profiles

- Status: Accepted
- Date: 2026-08-11

Use SHA-384 for exact package and signed-content identity and strict Ed25519 for
package and Go control-message signatures. Package transcripts use
`PersonalAI.ModelPackage.Signature.v1\0`; control messages use
`PersonalAI.ControlPlane.Request.Signature.v1\0`. Key IDs are exact nonzero
16-byte values. Rust stores public verification keys only.

Use exact, default-feature-minimized RustCrypto `sha2 0.11.0` and
`ed25519-dalek 3.0.0`; do not implement custom cryptography. Separate domains
prevent cross-protocol use. Strict verification, weak/invalid key rejection,
key-ID matching, active/revoked trust state, tamper tests, and locked checksums
are required. Algorithm migration requires a new package/message version and
ADR; silent negotiation or downgrade is prohibited.
