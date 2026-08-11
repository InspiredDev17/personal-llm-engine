# Native Engine Architecture

## System boundary

```text
clients -> Go control plane -> signed request -> protected UDS -> Rust data plane
             |                                      |
             | routing, policy, memory, tools       | package verification
             | approvals, audit, Agent Green        | tokenize / infer / sample
             +<----------- typed events ------------+
```

Go is authoritative. Rust performs inference only. Generated text and engine
events are untrusted inputs when they return to Go. The engine has no tool,
approval, platform-memory, audit-policy, download, update, Internet, or Agent
Green authority.

## Workspace ownership

| Crate | Responsibility |
| --- | --- |
| `engine-types` | bounded IDs, shapes, budgets, requests/events, errors and security-event types |
| `engine-crypto` | SHA-384 and strict Ed25519 verification/domain separation |
| `engine-model-format` | exact flat `.paimodel` V1 parser and tensor index |
| `engine-loader` | registered-ID resolution, secure local open, package verification and tensor materialization |
| `engine-tokenizer` | first-party `PAIBPE01` parser and bounded encode/decode |
| `engine-memory` | aggregate fallible reservations |
| `engine-tensor` | safe owned typed storage tied to reservations |
| `engine-cpu-reference` | checked scalar numeric primitives |
| `engine-kv-cache` | bounded sequence KV storage/accounting |
| `engine-transformer` | fixed decoder configuration, tensor inventory and CPU forward path |
| `engine-sampling` | deterministic tensor-direct greedy selection |
| `engine-scheduler` | bounded inference queue and sequence lifecycle |
| `engine-protocol` | directional frames, signed request/cancel envelopes and signed event encoding |
| `engine-server` | service composition, protected UDS and CPU runtime |
| `engine-cli` | owner configuration and daemon entrypoint |
| `engine-testkit` | deterministic non-production signed fixtures and boundary tests |

Internal dependencies are checked against an explicit DAG. The only external
dependency family is the reviewed RustCrypto SHA-384/Ed25519 graph plus `libc`
constants used by the Unix no-follow package and replay-journal openers. There
is no third-party AI, tensor, tokenizer, model-loader, scheduler, server, or
GPU-kernel dependency.

## Package and loading flow

1. Owner CLI configuration supplies a `ModelId`, direct-child `.paimodel`
   basename, exact SHA-384 hash, active package signer public key/key ID, and
   resource budget.
2. The loader rejects absolute, nested, traversal, and non-V1 names.
3. The model root must be a non-symlink directory not writable by group/other.
   The package must be a non-symlink regular file not writable by group/other.
4. `O_NOFOLLOW|O_CLOEXEC`, device/inode equality, exact prebounded length, and
   trailing-byte checks defend the read. No client supplies this path.
5. Complete-package SHA-384, canonical ranges/records, signed-content SHA-384,
   key ID, and strict Ed25519 signature verify before tokenizer or tensor use.
6. Tensor storage and tokenizer structures reserve against one aggregate
   `MemoryPool`; shape, byte, vocabulary, context, and work calculations use
   checked arithmetic.

The supported architecture is `personalai.decoder-v1`, a fixed causal decoder
with architecture-owned RMSNorm, Q/K/V projections, RoPE, grouped attention,
output projection, residual paths, SwiGLU MLP, final norm, and logits. Model
metadata cannot define operations, plugins, paths, executable code, or kernels.

## Local protocol and authentication

The only listener is a Unix-domain socket in a pre-existing owner-private
runtime directory. The socket path is a direct `.sock` child, length bounded,
created mode 0600, and removed only when its device/inode still match the
listener that created it. There is no TCP listener or network client.

Protocol v2 frames use a fixed 16-byte little-endian header, 8 MiB
complete-frame cap, closed directional kinds, and message-specific bounds.
Generation and cancellation commands use fixed schemas and strict Ed25519
signatures over a domain-separated SHA-384 transcript. A connection session
accepts strictly increasing nonzero nonces. Before any authenticated command
executes, Rust appends its nonce to the mode-0600 journal derived from the
configured control key ID and synchronizes the record to disk. The flat journal
has a fixed header, 8-byte records, and a 1,048,576-entry ceiling. Duplicate,
zero, corrupt, truncated, identity-mismatched, exhausted, symlinked, or
write-uncertain state fails closed. Reopening reconstructs the bounded replay
index, so a valid signed
command cannot be replayed on a later connection or after restart. Rust stores
only the configured Go public key.
Every event has a separate domain-separated Ed25519 signature over its exact
bytes plus the originating request nonce, a strictly increasing event ordinal,
and the configured engine response-key ID. Go pins the engine public key and
verifies the binding before parsing or exposing event data.

The current runtime executes one CPU sequence at a time. The Go adapter opens
one connection per generation and signs with the Go-held private key. On
cancellation it sends a higher-nonce signed cancellation on the same full-duplex
stream before closing; a Rust watcher treats that command, disconnect, malformed
follow-up, or mismatched cancellation as request-local cancellation while the
CPU loop is active. The response-signing seed is read as exactly 32 bytes from
stdin before model or socket startup, cleared after signer construction, never
accepted through argv/model/client data, and never logged. Go validates the
signature, nonce, ordinal, request ID, event schema, and usage before exposing
canonical platform responses. It has no HTTP fallback. Fair multi-sequence
scheduling and batching remain production hardening work.

## Resource ceilings

The code centrally bounds package bytes (1 TiB hard maximum with lower owner
budget), tensors (262,144), rank (8), dimensions (`1<<30`), elements (`1<<40`),
token collections (1,048,576), vocabulary (4,194,304), context (262,144),
request (4 MiB), complete frame (8 MiB), tokenizer data (64 MiB), sequences
(4,096), aggregate queued bytes (64 MiB), CPU work (100 million operations per
bounded operation), durable nonces per control key (1,048,576), and portable
socket path (100 bytes).

## Backend features

`cpu-reference` is default and implemented as a correctness path. `arm64`,
`metal`, and `cuda` are explicit, default-off boundaries and report
not-implemented. Native directories contain no kernels. A future backend must
be selected by trusted engine configuration and architecture code, never by
model data, and must pass CPU differential and isolated unsafe/FFI review.

## Agent Green

`SecurityEvent` is a sanitized typed value only. Engine telemetry is delivered
to Go; Go validates, audits, and sends only allowlisted events through its own
authenticated Agent Green liaison. The engine has no Agent Green client or key,
and the model cannot create, route, suppress, or act on Agent Green decisions.
