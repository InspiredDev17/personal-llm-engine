# ADR-0003: Authenticated Local Protocol

- Status: Accepted
- Date: 2026-08-11

Use a protected Unix-domain socket only. Frames have a fixed 16-byte header,
closed directional kinds, exact little-endian fields, and an 8 MiB total cap.
Generation/cancellation bodies are fixed schemas authenticated by the Go
control-plane Ed25519 identity and a strictly increasing connection nonce.
Protocol v2 responses are fixed typed generation events correlated to the
request ID and signed by a distinct owner-provisioned engine identity. Each
signature binds the originating request nonce, nonzero strictly increasing
event ordinal, response key ID, and exact event bytes. Go pins the response
public key and verifies authentication before event parsing.
Cancellation uses a higher-nonce signed command on the active full-duplex
stream; disconnect or invalid follow-up control data cancels fail-closed.

Reject TCP, HTTP fallback, arbitrary serialization, recursive messages,
client-supplied paths, model installation, tools, approvals, memory mutation,
and Agent Green commands. The Go adapter owns canonical request translation and
validates every event before exposing it to the platform.
