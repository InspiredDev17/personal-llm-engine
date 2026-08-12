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
constants used by the Unix no-follow package/replay-journal openers and the one
isolated nonblocking `flock` call. There is no third-party AI, tensor,
tokenizer, model-loader, scheduler, server, or GPU-kernel dependency.

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

The Rust data plane is permanently offline; this is not a runtime option that
a request, model, or operator flag can relax. Its exact dependency graph has no
network client, and architecture tests reject Internet socket APIs, raw socket
FFI, asynchronous networking modules, and process-spawning paths that could be
used as indirect production egress. The approved transport module may use only
Unix-domain stream primitives. Sanitized security events leave through that
same authenticated local channel to Go; the engine has no cloud telemetry
exporter.

Code-level absence is one layer, not an operating-system sandbox. A production
service definition must independently deny outbound networking while allowing
the configured Unix socket and bounded local files. The target-specific
service profile and an operated no-egress test are cutover evidence; they do
not belong inside model data or backend selection. Go separately owns the
platform-wide local-only routing decision and must reject remote primary,
fallback, verifier, and manual-override destinations in that mode. ADR-0006
records this split.

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
index, so a valid signed command cannot be replayed on a later connection or
after restart. A nonblocking exclusive advisory lock is held by the journal's
sole owned file descriptor for its complete lifetime; another process using the
same control-key journal fails before it can serve commands. Rust stores only
the configured Go public keys and accepts at most an explicit current/previous
pair. It rejects duplicate identities, binds one stream to one authenticated
key, and keeps a separate locked journal for each accepted key. Removing a key
from configuration does not delete its replay history.
Every event has a separate domain-separated Ed25519 signature over its exact
bytes plus the originating request nonce, a strictly increasing event ordinal,
and the configured engine response-key ID. Go pins at most an explicit
current/previous response public-key pair and verifies the selected binding
before parsing or exposing event data.

The current UDS runtime executes one CPU sequence at a time. Internally, greedy
CPU generation uses a request-local resumable cursor that yields after at most
one decoder token step; the existing synchronous generation API advances that
cursor to completion. A bounded engine-owned round-robin coordinator checks
its active ceiling, the most restrictive request concurrency budget, and
duplicate active request or sequence IDs before it creates a cursor or
tokenizes the prompt. It also conservatively charges activation,
scheduler-owned, and cursor-owned request copies, worst-case byte-token
storage, and cursor overhead to an aggregate
retained-byte ceiling. It advances one cursor once per pull and isolates cursor
failure to that request. Deterministic tests prove that a
shorter prompt can complete without waiting for a longer prompt and that
cancellation does not prevent its neighbor from completing. The owner CLI still
uses the single-worker loop. Separate reference drivers provide both an exact
batch and continuous admission. The batch accepts an exact configured number of
connections no greater than the scheduler active limit or the separate
64-connection ceiling,
authenticates and durably replay-checks each stream, then drives their cursors
through the coordinator with per-stream signing, ordinals, 256 KiB bounded-stack
cancellation watchers, separate 256 KiB bounded-stack output writers, fixed
eight-event per-stream queues, and a ten-frame reservation covering queued,
in-flight, and transient enqueue storage before cursor construction. Generation
never blocks on a socket write: a
full queue or failed writer cancels and fails only that stream, while writer
completion is still bounded by the finite socket deadline. It is covered by
real socket tests. The continuous driver switches only the listener to
nonblocking admission, explicitly returns every accepted stream to blocking
finite-deadline I/O, and gives each handshake its own bounded 256 KiB worker.
Authenticating, scheduler-queued, active, and writer-draining streams share one
64-connection ownership ceiling. New authenticated requests can queue while
other cursors execute. A caller-owned atomic shutdown signal stops admission,
interrupts incomplete reads, rejects queued requests, cancels active cursors,
drains signed terminal output, joins all workers on success or error, and
restores blocking listener mode before return. It is a tested reference path but
the owner CLI selects it only through `serve-continuous`, which requires
explicit active-sequence and total-connection limits. The CLI installs isolated
`SIGINT`/`SIGTERM` handlers whose only signal-context action is an atomic store;
a bounded watcher translates that store into the existing caller-owned shutdown
signal. Handler state is restored after the server drains. The Go adapter
opens one connection per generation and signs with the Go-held private key. On
cancellation it sends a higher-nonce signed cancellation on the same full-duplex
stream before closing; a Rust watcher treats that command, disconnect, malformed
follow-up, or mismatched cancellation as request-local cancellation while the
CPU loop is active. The response-signing seed is read as exactly 32 bytes from
stdin before model or socket startup, cleared after signer construction, never
accepted through argv/model/client data, and never logged. Go validates the
signature, nonce, ordinal, request ID, event schema, and usage before exposing
canonical platform responses. It has no HTTP fallback. Operated
continuous/slow-reader soak, peer-credential qualification, and optional
architecture-owned batching remain production hardening work. ADR-0011 defines
CLI signal integration; ADR-0010 defines continuous admission and graceful drain;
ADR-0009 defines bounded output isolation; ADR-0008 defines the bounded
multi-connection reference path; ADR-0007 defines the resumable generation and
round-robin coordination boundary; ADR-0005
defines the overlap, cutover, drain, rollback, and retirement order for both
identities; production secret injection and journal recovery still require
operated proof.

## Resource ceilings

The code centrally bounds package bytes (1 TiB hard maximum with lower owner
budget), tensors (262,144), rank (8), dimensions (`1<<30`), elements (`1<<40`),
token collections (1,048,576), vocabulary (4,194,304), context (262,144),
request (4 MiB), complete frame (8 MiB), tokenizer data (64 MiB), sequences
(4,096), continuously owned/reference-batch connections (64), ingress,
cancellation-watcher, and output-writer stacks (256 KiB each), pending output
events per active stream (8), aggregate queued bytes (64 MiB), active coordinator/request/output
retention (64 MiB or a lower configured limit), CPU work (100 million
operations per bounded operation), durable nonces per control key (1,048,576),
and portable socket path (100 bytes).

## Backend features

`cpu-reference` is default and implemented as a correctness path. `arm64`,
`metal`, and `cuda` are explicit, default-off boundaries and report
not-implemented. Native directories contain no kernels. A future backend must
be selected by trusted engine configuration and architecture code, never by
model data, and must pass CPU differential and isolated unsafe/FFI review.

## Unsafe and FFI boundary

Workspace policy denies unsafe Rust by default. The approved exceptions are
`engine-server/src/os_lock.rs`, which makes one checked `libc::flock` call using
an owned valid descriptor, and `engine-cli/src/os_signal.rs`, which uses two
documented `sigaction` blocks to install and restore owner-process handlers.
The signal handler retains no pointer and performs only an atomic store.
Architecture tests enforce the exact block and allowance counts and reject an
unsafe block or unsafe-lint allowance in every other first-party Rust source.
ADR-0004 records the lock invariants; ADR-0011 records the signal invariants and
portability scope.

## Agent Green

`SecurityEvent` is a sanitized typed value only. Engine telemetry is delivered
to Go; Go validates, audits, and sends only allowlisted events through its own
authenticated Agent Green liaison. The engine has no Agent Green client or key,
and the model cannot create, route, suppress, or act on Agent Green decisions.
