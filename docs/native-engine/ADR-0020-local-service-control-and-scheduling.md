# ADR-0020: Local service control and bounded scheduling

Status: accepted for the CPU reference service.

## Decision

The native engine remains an inference data plane. The Go platform owns routing,
policy, approvals, prompt assembly, registration configuration, signer trust,
and any Agent Green delivery. Rust accepts only owner-registered `ModelId`
values and never accepts a package path, kernel name, command, plugin, trust
mutation, self-update, or network destination.

The service uses the existing `PAIE` 16-byte little-endian frame and protocol
version 2.0. Generation and cancellation retain their existing authenticated
wire formats. Frame kinds 6 and 7 carry a separate `PAISVC01` request and
`PAISVR01` response envelope. Its fixed 56-byte header contains version,
closed message type, reserved byte, nonzero nonce, 128-bit request ID,
16-byte key ID, and 32-bit payload length. Requests carry a 64-byte Ed25519
signature under the existing control-plane domain; responses carry the engine
response signature. The first-party response decoder additionally requires the
expected request ID, originating nonce, pinned response-key ID, and pinned
public key before exposing a typed body. Exact lengths, direction, flags,
type, version, correlation, identity, and signature are mandatory.

Message bodies are closed first-party Rust types, not caller-defined byte
blobs. The message kind is derived from the body when signing and the body is
decoded only after envelope authentication. Fixed schemas are: two little-
endian `u16` values for hello negotiation; an empty body for queries and
lifecycle acknowledgements; validated `ModelId` text for load/unload; a
`u16` model-ID length plus bounded UTF-8 text for tokenize; a reviewed `u64`
capability mask; at most 1,024 length-prefixed unique model IDs; a `u32` token
count followed by validated `u32` token IDs; one closed byte for health,
errors, and shutdown rejection; and exactly 17 little-endian `u64` metrics.
Unknown capability bits, enum values, token IDs, duplicate models, truncation,
trailing data, invalid UTF-8, NUL prompt text, and kind/body mismatches fail
closed. This tightens parsing without changing the PAISVC01/PAISVR01 wire
representation.

`fixtures/go-service-health-request-v1.hex` is an exact signed health request
emitted by the production Go encoder with deterministic test-only identity
material. The Rust decoder verifies its signature, correlation fields, key ID,
message kind, and empty health body. The reciprocal Rust-generated signed
response fixture is verified in Go. These copied fixtures are transitional
cross-repository evidence until Sovereign Contracts owns a shared canonical
corpus.

The closed service catalog is `Hello`, `HelloAccepted`, `HelloRejected`,
`ListCapabilities`, `ListRegisteredModels`, `LoadModel`, `UnloadModel`,
`Tokenize`, `Generate`, `Cancel`, `GenerationEvent`, `Health`, `Metrics`,
`Error`, `ShutdownRequest`, `ShutdownRejected`, and `ShutdownAccepted`.
Generation, cancellation, and events remain their specialized streams. Normal
data-plane authentication always returns `ShutdownRejected`; only the
owner-controlled process signal may begin graceful drain. `ShutdownAccepted`
is reserved for a future separately authenticated administrative endpoint.

Metrics are fixed-order unsigned counters and never contain prompts, generated
content, paths, keys, or model bytes. Health is `ready`, `degraded` (no loaded
model), or `draining`. Load uses an owner-installed budget for a statically
registered ID. Unload rejects queued or active ownership. Package failures
produce bounded typed security events for Go polling.

The admission scheduler uses three queues and the repeating deterministic
dispatch cycle `interactive, standard, interactive, background, interactive,
standard, interactive`. This 4:2:1 cycle provides bounded service to every
nonempty class under normal load. Count and aggregate-byte ceilings remain
mandatory. Trusted metadata may add a bounded agent ID with queue/active quota,
a monotonic deadline, and a per-model concurrency ceiling. Metadata grants no
authorization. A temporarily ineligible request does not head-of-line block an
eligible request in the same class.

## Local transport profile

The only listener is a Unix-domain socket directly beneath a normalized,
owner-private mode-0700 runtime directory. The socket is mode 0600. Kernel peer
credentials must match the engine effective user. Read/write deadlines default
to 30 seconds and configuration is restricted to 10 milliseconds through 300
seconds. Connection ownership is capped at 64; output is capped at eight queued
events per stream with precharged retention. Existing paths are never followed:
only an owner-owned, mode-0600 socket inode that refuses connection may be
removed, after a second device/inode/owner check. Active sockets, symlinks,
regular files, permissive entries, and uncertain liveness fail closed.

## Consequences and limits

The exact single-command reference path serves the complete service-control
catalog. The continuous path currently admits generation/cancellation streams;
general control requests use short separate connections through the same
authenticated listener. CPU generation remains scalar and one token per
round-robin quantum; continuous batching is not claimed. Prefill/decode timing,
tokens-per-second, device memory, and aggregate active KV bytes are present in
the fixed metrics schema but remain zero until those measurements can be
collected without weakening ownership or determinism.
