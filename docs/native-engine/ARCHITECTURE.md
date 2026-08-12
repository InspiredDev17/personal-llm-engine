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
| `engine-model-format` | fixed directory-manifest, hash-catalog, signature-envelope and legacy test-fixture parsers |
| `engine-model-importer` | offline quarantine-only safetensors conversion, provenance, deterministic package writing and signing |
| `engine-loader` | production registered-ID directory resolution, secure local opens, signature/hash verification, read-only shard mapping and tensor materialization |
| `engine-tokenizer` | first-party AIPTok Byte-Level BPE v1 format, immutable tables, bounded encode/decode and incremental UTF-8 decode |
| `engine-memory` | explicit-owner aggregate accounting, request scratch arenas, CPU device memory, admission calculations and pressure metrics |
| `engine-tensor` | validated row-major metadata, owned/read-only/mutable storage, explicit strided views, and F32/F16/BF16 conversion tied to aggregate reservations |
| `engine-cpu-reference` | immutable allowlisted graph, deterministic plan, and checked scalar correctness-oracle operations |
| `engine-kv-cache` | contiguous multi-sequence KV storage, atomic token commits, lifecycle invalidation and admission metrics |
| `engine-transformer` | fixed decoder configuration, tensor inventory and CPU forward path |
| `engine-sampling` | deterministic tensor-direct greedy selection |
| `engine-scheduler` | bounded inference queue and sequence lifecycle |
| `engine-protocol` | directional frames, signed request/cancel envelopes and signed event encoding |
| `engine-server` | service composition, protected UDS and CPU runtime |
| `engine-cli` | owner configuration and daemon entrypoint |
| `engine-testkit` | deterministic non-production signed fixtures and boundary tests |

Internal dependencies are checked against an explicit DAG. The only external
dependency family is the reviewed RustCrypto SHA-384/Ed25519 graph plus `libc`
constants used by Unix no-follow openers, the isolated nonblocking `flock`
call, and the isolated `mmap`/`munmap` read-only tensor mapping. There is no third-party AI, tensor,
tokenizer, model-loader, scheduler, server, or GPU-kernel dependency.

## Package and loading flow

The import and production components are intentionally separate:

1. The offline importer accepts only local `.safetensors` and strict
   `AIPTOK-METADATA-V1` files
   that canonicalize beneath an owner-selected quarantine. It refuses URLs,
   quarantine escapes, pickle/executable formats, unknown tensor names,
   missing/extra tensors, unsupported dtypes, non-finite values, and any
   architecture mapping other than `llama-safetensors-v1`.
2. The importer writes a staging directory and atomically installs
   `<model-id>.paimodel`. Replacement requires a bounded explicit audit note;
   the previous directory is preserved. The signing seed is exact stdin data,
   never stored or logged, and zeroized after signer construction.
3. A production package contains exactly `manifest.aipm`, `tokenizer.aiptok`,
   `provenance.json`, `hashes.aiph`, `signature.ed25519`, and contiguous
   `tensors-NNNN.aipt` shards. Any extra file, directory, script, library,
   object, bytecode, WASM, kernel, or template is rejected by the closed set.
4. Owner configuration supplies only a `ModelId`, direct-child `.paimodel`
   directory name, SHA-384 identity of `hashes.aiph`, trusted signer key ID and
   public key, and a lower resource budget. Clients still submit only ModelId.
5. The production model root, package directory, and every direct-child file
   must be non-symlink and have no write permission bits. Each file uses
   `O_NOFOLLOW|O_CLOEXEC` with pre/open device, inode, type, and length checks.
6. The loader verifies the catalog identity, domain-separated strict Ed25519
   signature, exact file set, every catalog hash, manifest compatibility,
   tokenizer/shard/tensor hashes, architecture inventory/shapes, and aggregate
   package budget before exposure. Each failure emits one typed event to its
   mandatory caller-owned security-event sink.
7. Tensor shards are privately mapped `PROT_READ` without `PROT_EXEC`; immutable
   F32, F16, or BF16 ranges are then converted into aggregate-accounted F32 CPU
   execution tensors. No loader API
   accepts source formats, an arbitrary path, or an unsigned package. The old
   flat package remains only as deterministic unit-test compatibility data.

The supported architecture is `decoder-rms-rope-swiglu-gqa-v1`, a fixed causal decoder
with architecture-owned RMSNorm, Q/K/V projections, RoPE, grouped attention,
output projection, residual paths, SwiGLU MLP, final norm, and logits. Model
metadata cannot define operations, plugins, paths, executable code, or kernels.

## Tensor, graph, and numerical policy

Owned tensors use canonical dense row-major order with the final dimension
contiguous. `TensorLayout` carries explicit element strides and offsets;
transpose creates a checked non-contiguous view and never relabels it as
contiguous. Logical package alignment is a bounded power of two and every
reachable byte, element count, stride, shape, storage length, and view offset
is checked. Engine operations allocate a fresh destination in the same
aggregate `MemoryPool`, so input/output aliasing is structurally excluded;
mutable overlapping borrows remain prohibited by Rust's borrow rules.

F32 is the initial execution type. Signed packages may store unquantized F32,
IEEE binary16, or BF16 tensors; the loader performs explicit first-party
conversion to F32 and rejects every other type. F16 handles normal, subnormal,
signed-zero, infinity, and NaN bit patterns with round-to-nearest-ties-to-even;
BF16 narrowing also uses ties-to-even. Conversion preserves non-finite classes,
while arithmetic rejects NaN and infinity except that softmax accepts negative
infinity only as an explicit mask. RMSNorm uses a scaled F64 sum-of-squares;
softmax subtracts the row maximum and accumulates exponentials in F64. Scalar
loops execute in fixed row-major order with no SIMD, threads, fusion, BLAS, or
platform acceleration and reject non-finite arithmetic results.

`InferenceGraph` is immutable after validation. Nodes use the closed
first-party `Operation` enum, explicit value IDs, static F32 shape propagation,
deterministic node-ID-tiebroken topological order, cycle/missing/duplicate
rejection, and a conservative checked sum of intermediate bytes. The graph
executor accepts no package/client operation strings or code. The reviewed
transformer architecture constructs its own 14-node reference decoder-block
graph; packages select only the allowlisted architecture ID. The scalar backend
implements copy, F32/F16/BF16 conversion, broadcast add/multiply, scalar
multiply, matrix-vector/matrix-matrix, embedding lookup, RMSNorm, stable
softmax, SiLU, SwiGLU, RoPE, head split/merge, grouped-query attention, causal
masking, residual addition, Q/K/V projection composition, and logit projection.
The 100-million-iteration operation ceiling and aggregate memory pool apply
before output allocation. ADR-0017 records these invariants.

## Inference memory and KV-cache ownership

Every accounted allocation has one engine-owned category: model weights,
model metadata, tokenizer tables, immutable model state, request prompt,
sequence KV, request scratch, output logits, sampling state, streaming decoder,
or backend resources. Pools expose current, peak, category and allocation-count
metrics. A scoped pool changes only the owner label; it shares the exact same
hard ceiling and counters, so a subcomponent cannot create capacity by nesting
pools. Loader tensor materialization labels immutable metadata and weights by
the registered `ModelId`.

The loader's signed tensor shards remain owned by a non-cloneable
`VerifiedDirectoryPackage`. Each shard is mapped `MAP_PRIVATE|PROT_READ`, never
`PROT_WRITE` or `PROT_EXEC`; mappings drop with the package. The loader checks
the signed SHA-384 identity after mapping, rechecks a shard before exposing a
tensor range, and offers full mapping revalidation with a mandatory typed
security-event sink. Only a registered model ID reaches this path.

`ScratchMemoryManager` owns one engine-wide CPU pool. Each `ScratchArena` is
bound to one `RequestId` and a lower request ceiling. Buffers are initialized,
exposed only through generation-checked arena-local handles, cleared on normal
finish, cancellation, failure unwind and drop, then may enter an idle cache.
Ownership accounting is transferred back to the engine while idle and to the
next request on reuse. Idle bytes are separately reported and explicitly
reclaimable. No uninitialized bytes are returned.

`ContiguousKvCache` owns exactly one verified `ModelId` and multiple guarded
sequence objects under a hard configured concurrent-sequence ceiling. One sequence allocation has canonical row-major layout
`[K-or-V, layer, position, kv-head, head-dimension]`. Storage is F32, F16, or
BF16 only; the CPU interface accepts finite F32 values and performs the reviewed
first-party conversion. Callers never provide byte offsets. The engine chooses
the next position, a token becomes readable only after every layer K/V pair is
written, and an abandoned or failed partial write clears and cancels the whole
sequence. Generation-tagged handles prevent an old owner from reaching a new
sequence that reuses the numeric `SequenceId`. Reset, cancellation,
destruction, and model unload clear storage; unload permanently invalidates the
cache. Mutex-guarded lifecycle methods provide deterministic isolation under
concurrent scheduler operations.

The scheduler can query exact full-sequence reservation, per-token logical
bytes, prefill bytes, current/peak/reserved/used/reclaimable cache pressure,
scratch pressure, model-fit, request-fit, and extension capacity before work.
CPU host/device allocation and whole-buffer transfers are real initialized
host operations. Metal and CUDA memory requests return explicit
not-implemented errors; this stage contains no GPU emulation.

Contiguous allocation is the V1 correctness baseline. A future paged cache may
be added only as a separate implementation behind this lifecycle/admission
contract. It must retain generation-checked ownership, zero-on-reassignment,
atomic token publication, checked page-table arithmetic, model-unload
invalidation and differential tests against the contiguous cache. No package
field may choose cache implementation or page size. ADR-0018 records the full
decision and migration gate.

## Tokenizer boundary

`tokenizer.aiptok` uses magic `AIPTOK01`, version 1.0, a fixed 128-byte
little-endian header, explicit family/normalization/unknown/decoder IDs,
contiguous vocabulary IDs, length-prefixed vocabulary bytes, fixed 16-byte
ranked merge records, fixed 8-byte special-token records, required BOS/EOS/UNK
IDs, per-package input/output limits, and an internal SHA-384 content hash.
The first 256 IDs are exact one-byte identity tokens. Every later ordinary
token has exactly one defining merge; special tokens use canonical bounded
`<|name|>` UTF-8 text and never participate in merges.

Normalization is exactly preserve-bytes. The UTF-8 API cannot receive malformed
UTF-8 by construction; the byte API accepts all byte strings and preserves
them. Strict text decode rejects malformed UTF-8, while byte decode returns the
exact bytes. Incremental decode buffers only an incomplete UTF-8 suffix and
must finish without pending bytes. Literal special-token text is rejected by
default; callers must explicitly allow all, allow a validated ID list, or
treat it as ordinary bytes. The runtime uses the rejecting policy for assembled
prompt text. Aggregate tokenizer-table memory is reserved before table
allocation; encoding work is separately bounded and reserved. Chat
roles/templates, system-prompt files, tools, and policy remain entirely in Go.

The offline converter accepts one ordered, ASCII, line-oriented metadata
profile. It requires byte preservation, no pre-tokenizer, fixed decoder,
byte fallback, and `chat_template=none`; unsupported families, normalizers,
components, template text, extra fields, lossy mappings, and trailing data fail
closed. The source metadata SHA-384, filename, and format are signed through
package provenance. There is no diagnostic mode that can emit an approved
lossy package.

## Local protocol and authentication

The only listener is a Unix-domain socket in a pre-existing owner-private
runtime directory. The socket path is a direct `.sock` child, length bounded,
created mode 0600, and removed only when its device/inode still match the
listener that created it. The listener records the engine's effective user ID
at bind. Each accepted stream must have the same kernel-reported effective user
ID before timeouts, frame allocation, or parsing. A mismatched blocking peer is
dropped while admission continues; continuous admission counts the failure and
continues serving authorized neighbors. This is an independent prefilter:
Ed25519 control authentication and durable replay admission remain mandatory.
There is no TCP listener or network client.

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
canonical platform responses. It has no HTTP fallback. The current
peer-credential profile requires Go and Rust to share one effective service
user; a future distinct-user deployment requires a separate reviewed socket
group/ACL design and must not bypass kernel identity checks. Operated
continuous/slow-reader soak and optional
architecture-owned batching remain production hardening work. ADR-0011 defines
CLI signal integration. ADR-0012 defines the scheduled bounded delayed-reader
and restart soak; ADR-0013 defines kernel peer admission; ADR-0010 defines
continuous admission and graceful drain;
ADR-0009 defines bounded output isolation; ADR-0008 defines the bounded
multi-connection reference path; ADR-0007 defines the resumable generation and
round-robin coordination boundary; ADR-0005
defines the overlap, cutover, drain, rollback, and retirement order for both
identities; production secret injection and journal recovery still require
operated proof.

## Resource ceilings

The code centrally bounds package bytes (1 TiB hard maximum with lower owner
budget), tensors (262,144), rank (8), dimensions (`1<<30`), elements (`1<<40`),
directory manifest (64 MiB), shards (1,024), tensor name (47 bytes), tensor
alignment (2 MiB), hash-catalog entries (1,027), provenance (1 MiB), and
safetensors import header (64 MiB),
token collections (1,048,576), vocabulary (4,194,304), tokenizer specials
(4,096), token pieces (256 bytes), AIPTok source/package data (64 MiB each),
context (262,144),
request (4 MiB), complete frame (8 MiB), sequences
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
documented `sigaction` blocks to install and restore owner-process handlers,
plus `engine-server/src/peer_credential.rs`, which reads the process and peer
effective user IDs through platform credential APIs. The signal handler retains
no pointer and performs only an atomic store; peer outputs use exact initialized
ABI storage and borrowed live descriptors. `engine-loader/src/read_only_map.rs`
owns exactly one nonempty mapping, exposes only an immutable slice, requests
`PROT_READ|MAP_PRIVATE` without execution, and unmaps the exact pointer/length
once while its file remains live. Architecture tests enforce the exact
block and allowance counts and reject an unsafe block or unsafe-lint allowance
in every other first-party Rust source. ADR-0004 records the lock invariants;
ADR-0011 records signal invariants; ADR-0013 records peer-credential invariants
and portability scope. ADR-0015 records directory-artifact and mapping
invariants. ADR-0014 pins an offline Linux/macOS workflow that
compiles and tests both credential implementations without expanding runtime
capability or dependency authority.

## Agent Green

`SecurityEvent` is a sanitized typed value only. Engine telemetry is delivered
to Go; Go validates, audits, and sends only allowlisted events through its own
authenticated Agent Green liaison. The engine has no Agent Green client or key,
and the model cannot create, route, suppress, or act on Agent Green decisions.
