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
| `engine-release` | offline release-only signing, exact installed-artifact verification and monotonic rollback ledger |
| `engine-model-format` | versioned fixed directory manifests, hash catalog, signature envelope and legacy test-fixture parsers |
| `engine-quantization` | fixed AIPQ8/AIPQ4 validation, deterministic conversion, error measurement and safe scalar reference kernels |
| `engine-model-importer` | offline quarantine-only safetensors conversion, provenance, deterministic package writing and signing |
| `engine-loader` | production registered-ID directory resolution, secure local opens, signature/hash verification, read-only shard mapping and tensor materialization |
| `engine-placement` | immutable verified-artifact placement plans, checked residency accounting and bounded layer staging/prefetch |
| `engine-tokenizer` | executable first-party AIPTok Byte-Level BPE v1 plus a fail-closed `AIPTOK02` Qwen2.5 parser and private fixed scanner/BPE/decode core; reviewed Unicode-16 primitives are admitted but not wired through the missing trusted adapter, so immutable v1 remains the only publicly constructible runtime path |
| `engine-memory` | explicit-owner aggregate accounting, request scratch arenas, CPU device memory, admission calculations and pressure metrics |
| `engine-knowledge-contract` | narrow pinned execution projection of experimental Sovereign Knowledge 1.1 lifecycle semantics and Knowledge 1.2 exact prefix framing |
| `engine-tensor` | validated row-major metadata, owned/read-only/mutable storage, explicit strided views, and F32/F16/BF16 conversion tied to aggregate reservations |
| `engine-cpu-reference` | immutable allowlisted graph, deterministic plan, and checked scalar correctness-oracle operations |
| `engine-arm64` | default-off optimized ARM64 graph backend, bounded worker pool, runtime dispatch, quantized kernels, and scalar-oracle fallback |
| `engine-kv-cache` | contiguous multi-sequence KV storage, atomic token commits, lifecycle invalidation and admission metrics |
| `engine-transformer` | closed legacy decoder execution plus Qwen2.5 tiny scalar correctness scaffolding, exact tensor inventories, resident CPU path and bounded scalar streamed path |
| `engine-sampling` | model/tokenizer-bound selectable token domains plus bounded deterministic greedy, temperature, top-k, top-p, min-p and repetition-penalty selection with a reproducible non-cryptographic PRNG |
| `engine-scheduler` | bounded inference queue and sequence lifecycle |
| `engine-protocol` | directional frames, signed request/cancel envelopes, signed event encoding, bounded canonical Runtime-v3 documents, and the fixed-binary internal Knowledge 1.1 cache adapter |
| `engine-server` | service composition, protected UDS, owner-selected CPU/Metal runtime, reusable cache lifecycle, and the bounded inference-runtime-v3 semantic reservation lifecycle |
| `engine-cli` | owner configuration, explicit backend policy and daemon entrypoint |
| `engine-testkit` | deterministic non-production signed fixtures and boundary tests |

Internal dependencies are checked against an explicit DAG. The only external
dependency family is the reviewed RustCrypto SHA-384/Ed25519 graph plus `libc`
constants used by Unix no-follow openers, the isolated nonblocking `flock`
call, and the isolated `mmap`/`munmap` read-only tensor mapping. There is no third-party AI, tensor,
tokenizer, model-loader, scheduler, server, or GPU-kernel dependency.

The runtime-v3 reservation lifecycle is pinned to Sovereign AI Contracts
commit `2c4a7a6afd5625e7def3afec26f18fedac343d11`. It does not define or parse a
second wire format. A strict bounded adapter consumes only canonical v3
documents decoded by the pinned Contracts implementation, maps them into the
engine state machine, revalidates engine-owned identity before reservation and
consumption, and emits canonical decision/result documents. The in-memory state
machine enforces exact identity/correlation, finite capacity, idempotent replay,
single consumption, cancellation, release, expiry, terminal identity, and
resource ceilings. The existing signed, replay-journaled, same-user UDS service
now carries bounded canonical availability, reservation, cancellation, release,
and execution documents without assigning them authentication authority.
Execution is fail-closed unless a backend is explicitly composed; the only
host-independent implementation in this slice is a compile-time-gated
deterministic development simulator. Production backend dispatch and durable
recovery of accepted reservations remain incomplete, so this is not yet an
operated v3 inference service.

Model unload has one shared fail-closed gate for both the authenticated service
and the internal runtime API. It refuses unload while the scheduler owns the
model, while a reusable-cache attachment is staged, or while an unexpired
Runtime-v3 reservation holds capacity for that exact provider model. Reusable
cache invalidation occurs only after those ownership checks pass.

Cancellation remains authority-specific. The fixed-binary `Cancel` envelope
cleans scheduler ownership and any staged reusable-cache attachment for its
exact request ID; canonical `RuntimeV3Cancel` cleans only the fully correlated
Runtime-v3 reservation. A control-plane request that owns both states must send
both authenticated operations. Neither operation guesses the other authority
from a request ID, and cancellation in one domain cannot cancel an unrelated
neighbor in the other.

An exact Runtime-v3 execution retry after a lost response returns the retained
terminal result before re-admission, so it never invokes the backend twice.
Altered retry content fails closed. Terminal reservation/result receipts remain
available through the original reservation expiry; when the bounded 16,384
record set is full, only already-terminal records whose original authority has
expired are pruned. Active or still-valid replay state is never evicted. This
retention is in-memory and does not replace the still-required durable accepted-
reservation recovery gate.

## Engine release trust flow

Engine releases and model packages use separate signatures and roles. The
offline `personalai-engine-release sign` command records a nonzero monotonic
generation, release/target/toolchain identifiers, and the exact SHA-384 and
length of the engine binary, SBOM, Cargo.lock, and any explicitly declared
metallib/notices. It produces canonical `engine-release.pairm` and
`engine-release.ed25519` files without overwriting an approval.

Before launch, an owner-controlled supervisor invokes
`personalai-engine-release verify-supervised` against a fixed read-only release root. The
verifier accepts no URL and no client/model path, requires the exact closed
direct-child file set, rejects symlinks and writable installed content, verifies
the release-only Ed25519 identity and every artifact, then advances a locked,
durable monotonic ledger outside the engine runtime. The release crate has no
dependency edge into the inference server, so it confers no update, install,
launch, model, tool, or network authority. ADR-0026 defines the signed format;
ADR-0027 binds the Linux unit to this gate. A failed check retains one
content-free fixed-width event in a root-owned 64-slot ring. Go validates and
routes those events through its existing audit/Agent Green boundary without
write access. OS authentication of the verifier and operated evidence remain
supervisor/installer assumptions.

## Package and loading flow

The import and production components are intentionally separate:

1. The offline importer accepts one or more local `.safetensors` shards and a
   strict `AIPTOK-METADATA-V1` file
   that canonicalize beneath an owner-selected quarantine. It refuses URLs,
   quarantine escapes, pickle/executable formats, unknown tensor names,
   missing/extra/duplicate tensors, unsupported dtypes, non-finite values, and
   unapproved architecture mappings. A disabled `qwen2.5-safetensors-v1`
   transition profile pins three exact configurations, one closed tokenizer
   graph/identity, exact single-file identities, and the Qwen tensor
   names/shapes, but it fails before source I/O and cannot emit a package until
   the remaining tokenizer/version/registry/backend activation gates close.
   Shard input order cannot affect the output: unique source filenames are
   sorted, the complete architecture inventory is validated globally, and F32,
   F16, or BF16 source storage is converted explicitly into finite F32 values
   before an explicit `reference-f32`, `balanced-q8`, or `compact-q4` profile is applied.
   Only reviewed rank-two weights are quantized.
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
   F32, F16, BF16, AIPQ8, or AIPQ4 ranges are then validated and converted into
   aggregate-accounted F32 CPU execution tensors. No loader API
   accepts source formats, an arbitrary path, or an unsigned package. The old
   flat package remains only as deterministic unit-test compatibility data.

Two version-bound architecture identifiers are parsed. Manifest v1.0 retains
the original `decoder-rms-rope-swiglu-gqa-v1` semantics unchanged. Manifest
v1.1 selects only `qwen2.5-rms-rope-swiglu-gqa-bias-tied-v1`: the same closed
causal decoder operation set plus required Q/K/V projection biases, split-half
Qwen rotary layout, explicit RMSNorm epsilon, and tied token-embedding/output
weights. The Qwen2.5 v1.1 validator recognizes only the reviewed 1.5B bootstrap
dimensions and either the 32,768-context/1,000,000-theta general/coder profile
or the 4,096-context/10,000-theta math profile once the profile is activated.
V1.1 requires engine 0.2.0, while the current runtime remains 0.1.0 and rejects
it before activation. Tiny scalar configurations provide deterministic math
evidence, but the exact 1.5B profile's 1,543,712,768-operation initial step
exceeds the unchanged 100,000,000-operation scalar ceiling and is rejected
before weights are materialized. The CPU service also applies this Qwen-only
compute admission immediately after trusted configuration conversion and before
future tokenizer allocation or tensor materialization; current version and
tokenizer gates remain earlier, and the legacy load order is unchanged. ARM64
and Metal reject this new architecture rather than invoking their legacy
adjacent-pair RoPE kernels.
Model metadata cannot define operations, plugins, paths, executable code, or
kernels.

This is engine-private package/runtime support, not a production model
approval. The offline importer still produces only the legacy Llama mapping,
although it now contains the fail-closed Qwen source inspection/mapping
transition. `engine-tokenizer` recognizes and fully validates a separate
data-only `AIPTOK02` table profile, but deliberately returns
`UnsupportedTokenizer` rather than constructing an approximate Qwen runtime.
The Model Foundry registry V2 compatibility profile still names manifest v1.0,
and no exact Qwen tokenizer converter or signed Qwen package exists here. The
current service returns the earlier typed future-engine rejection; after an
eventual runtime revision bump it still reaches the closed tokenizer gate.
Exact NFC/category-aware pre-tokenization, encode/decode/incremental decode,
conversion, independent vectors, Model Foundry admission, and a qualified
optimized backend must be implemented and qualified before activation.
The exact Unicode 16 normalization/category primitives are admitted to the
offline graph but deliberately unused until that separately reviewed runtime
and its conformance evidence exist.

## Trusted execution placement

Physical placement is an internal runtime decision, not protocol authority.
Sovereign-AI-Contracts does not yet define a canonical placement policy, so the
local wire protocol intentionally accepts neither physical plans nor tensor
offsets. `engine-placement` derives an immutable plan only from a verified
package, trusted backend capabilities, engine memory state, an owner-selected
closed policy, and the configured resource envelope. It binds the exact model,
artifact and package identities, package/model generations, architecture,
backend, quantization, topology, stage/range inventory, and checked byte totals.

`FullyResident` is the only mode composed into the signed CPU, ARM64, and Metal
service paths. It preserves complete materialization while requiring mapped,
metadata, execution, existing-reservation, and safety-reserve bytes to fit
before model exposure. `LayerStreamed` currently provides a real CPU staging
substrate over immutable unequal-sized architecture stages with at most one
prefetch worker. Each selected mapping identity and signed per-tensor content
hash is revalidated before materialization; the exact bytes used for conversion
are copied into an accounted snapshot and hashed again, closing the
check-versus-copy race. Unrelated layers are not repeatedly hashed. Cancellation or failure joins the worker and releases staged
reservations. A mapping-identity failure uses the existing typed, content-free
package-rejection event. The scalar decoder now performs prefill and one-token
decode directly over these stages and preserves its existing atomic contiguous
KV-cache commit. Cancellation during a layer invalidates partial KV state. The
local service accepts an owner-only startup selection for fully resident or
scalar layer-streamed weights. It defaults to fully resident and rejects
streaming with ARM64 or Metal. Authenticated clients still provide only a
registered `ModelId`; the wire exposes no physical plan or placement override.
`HybridResident` returns an explicit unsupported error.

The placement plan records the parsed package format revision, so a v1.1 Qwen
artifact cannot be mislabeled as v1.0 after verification. Qwen's tied logical
output aliases the signed embedding range for streamed execution but is counted
only once for fully resident execution, matching its materialization behavior.
Quantization is orthogonal and stays bound to the signed artifact. Apple CPU
and Metal placement use unified-memory topology rather than fabricated
host-versus-VRAM capacity. Backend ownership remains logically explicit. A
future discrete CUDA implementation may use distinct host, staging, and device
ceilings, but CUDA has no runtime success path today. ADR-0029 records the
decision and acceptance boundary.

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
IEEE binary16, BF16, AIPQ8 v1, or AIPQ4 v1 tensors; the loader performs explicit
first-party conversion to F32 and rejects every other type. Quantized payloads
use fixed 32-value symmetric blocks, explicit little-endian F32 scales, exact
lengths, canonical codes and zero tail padding. F16 handles normal, subnormal,
signed-zero, infinity, and NaN bit patterns with round-to-nearest-ties-to-even;
BF16 narrowing also uses ties-to-even. Conversion preserves non-finite classes,
while arithmetic rejects NaN and infinity except that softmax accepts negative
infinity only as an explicit mask. RMSNorm uses a scaled F64 sum-of-squares;
softmax subtracts the row maximum and accumulates exponentials in F64. Scalar
loops execute in fixed row-major order with no SIMD, threads, fusion, BLAS, or
platform acceleration and reject non-finite arithmetic results.

The AIPQ scalar kernels directly execute matrix-vector and matrix-matrix
operations over validated payloads without materializing weight tensors. The
current service intentionally materializes F32 weights until those direct
kernels are integrated throughout the immutable transformer path. Exact format
and synthetic accuracy evidence is in [QUANTIZATION.md](QUANTIZATION.md), and
ADR-0024 records the acceptance boundary.

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
placement staging, or backend resources. Pools expose current, peak, category and allocation-count
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

The executable `tokenizer.aiptok` profile uses magic `AIPTOK01`, version 1.0,
a fixed 128-byte little-endian header, explicit
family/normalization/unknown/decoder IDs,
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

`AIPTOK02` is a distinct, publicly non-executable Qwen2.5 profile; it never
reinterprets v1. Its fixed 160-byte header declares the closed NFC,
Unicode-property pre-tokenizer, GPT-2 byte-symbol decoder, no-unknown and
no-implicit-BOS/EOS behavior identities. Contiguous vocabulary records,
fixed-width merge/added-token records, a complete 256-byte permutation, checked
offsets, declared limits, exact special roles, and a SHA-384 table hash are
validated under aggregate memory accounting. For the pinned production counts,
the parser also fixes all 22 added-token literals/roles and distinguishes
151,665 defined token IDs from 151,936 model logits. Undefined tail IDs are
always invalid. A private core now retains these immutable tables and implements
the fixed pre-tokenizer state machine, deterministic byte-map/BPE, explicit
added/special-token policies, strict decode, and bounded incremental decode.
Unicode NFC/category/case-fold behavior is an explicit trusted internal
Unicode-16 interface with no production implementation in this revision. The
public factory therefore still validates the structure and returns
`UnsupportedTokenizer`; no service or importer path can construct the core.
ADR-0035 records the format gate and ADR-0036 records this inert runtime slice.

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

The continuous runtime accounts pre-request failures without trusting a
request ID. Kernel identity or message-authentication rejection enqueues
`UnauthorizedClient`; malformed, oversized, and unknown-version frames map to
their closed event variants; backend failures remain typed. Every eighth
authentication failure emits `RepeatedAuthenticationFailure`, and every eighth
resource rejection emits both `ResourceExhaustionPattern` and
`RepeatedRejectedRequests`. The in-process queue remains capped at 256 events
for Go-owned delivery and contains no prompt, output, path, key, or model bytes.
Authenticated `SecurityEvents` polls are non-destructive. Go delivers each
returned prefix through its separate Agent Green boundary and then sends
`AcknowledgeSecurityEvents`; failed delivery leaves the prefix available for
retry. This is at-least-once delivery, so a lost acknowledgement response can
produce a duplicate and downstream correlation must remain idempotent.
`CrashLoopIndication` is supervisor-owned because a terminated process cannot
truthfully observe its own restart history.

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

The current UDS runtime executes one CPU sequence at a time. Internally, CPU
generation uses a request-local resumable cursor that yields after at most
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
cancellation does not prevent its neighbor from completing. A separate rich
internal lifecycle covers request acceptance, tokenization, prefill, generated
tokens, stops, completion, cancellation, limits, model errors, and backend
errors without changing the authenticated Go wire contract. The owner CLI can
run this complete path for a registered signed package with explicit sampling
and resource limits. Loaded CPU and Metal models privately retain an immutable
selectable-token domain bound to the verified package and tokenizer hashes.
Sampling intersects request filters with that domain before reading candidate
logits, so callers can narrow but cannot expand the tokenizer-defined ID set.
Current AIPTok v1 packages retain exact all-vocabulary behavior; the disabled
Qwen bootstrap can use its reserved-tail representation only after its exact
tokenizer runtime is implemented and admitted. Separate reference drivers
provide both an exact batch and continuous admission. The batch accepts an
exact configured number of connections no greater than the scheduler active
limit or the separate 64-connection ceiling,
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

Service-control requests use the same protected socket and cryptographic
identities but a separate fixed `PAISVC01` envelope. Its header carries the
closed message type, nonzero nonce, request ID, control key ID, and exact
payload length; `PAISVR01` responses are signed by the engine identity. The
catalog is limited to negotiation, capability/model listing, registered-ID
load/unload, tokenize, health, metrics, atomic registered-model status, and
shutdown response types. Normal
data-plane sessions cannot authorize shutdown. Package paths, signer
configuration, model budgets, and administrative authority never cross this
protocol. Each message uses a closed, bounded first-party body codec; signing
derives the message kind from that body, and authenticated decoders reject
unknown values, duplicate model IDs, malformed lengths, invalid token IDs,
truncation, and trailing data before exposing a typed body. The exact
single-command driver and the continuously admitting daemon both serve these
control requests. Continuous controls use short authenticated connections
under the same 64-connection ownership ceiling as handshakes, queued/active
generations, and draining writers. Generation cursors own cloned immutable
model/tokenizer handles, so a separately registered model may load without
invalidating active state; unload remains prohibited while that model has
queued or active scheduler ownership.
The model-status response combines load state, health, capabilities, scheduler
counts/limits, and queued-byte accounting under one signed correlation rather
than making the Go router combine independent observations. ADR-0020 defines
the format and deterministic 4:2:1 scheduler.

The production owner-pinned CLI registers one backward-compatible keep-hot
primary plus at most 127 additional cold packages. Cold records are bounded,
path-relative, hash/signer-pinned, uniquely identified, and assigned fixed load
budgets before the listener opens. They become resident only after an
authenticated exact-ID load command. Go separately declares `KEEP_HOT` or
`LOAD_ON_DEMAND`, verifies every registered identity, and publishes a cold
model to routing only after load, authenticated descriptor pinning, and fresh
resource evidence. Explicit unload never substitutes another package and the
engine rejects it while scheduler or sequence state owns the target. ADR-0028
defines this execution-only residency boundary; cognitive selection and future
eviction policy remain Go-owned.

Target service confinement is independent of Rust configuration. Linux uses
the checked-in systemd unit with `AF_UNIX` and all-IP denial. macOS development
and operated qualification may use the deny-by-default exact-socket Seatbelt
profile and its local UDS-versus-loopback proof. Because Apple deprecates
`sandbox-exec`, ADR-0021 keeps a signed hardened/App Sandbox distribution design
as a mandatory production gate rather than treating the transitional profile
as permanent authority.

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
portable socket path (100 bytes), placement layers (4,096), and placement
prefetch depth (1). Placement/backend memory ceilings cannot exceed the existing
1 TiB resource hard limit.

## Backend features

`cpu-reference` is default and implemented as a correctness path. `arm64`,
`metal`, and `cuda` remain explicit and default-off. The ARM64 feature exposes
a real immutable-graph backend with runtime Advanced SIMD detection, fixed
elementwise, reduction, matvec/matmul, normalization, activation, RoPE,
attention, half-conversion, and AIPQ8/AIPQ4 kernels, plus scalar fallback for
unsupported shapes, layouts, dtypes, and operations. Its first-party worker
pool is fixed at one to sixteen threads, requires one thread in deterministic
mode, bounds ingress to the worker count, isolates panics, and joins cleanly.
The signed decoder now selects the ARM64 backend only through owner startup
configuration, executes its complete prefill/decode path through the closed
backend operation interface, and retains scalar fallback for unsupported
validated cases. The service advertises ARM64 only on an AArch64 build with the
feature enabled and returns the exact backend in its authenticated model
descriptor. The Metal feature exposes a real immutable-graph backend,
an embedded SHA-384-identified first-party metallib, a closed 19-kernel C ABI,
  bounded pooling and transfer metrics, scalar fallback, and sequence-isolated
  contiguous device KV storage. Its fixed decoder uploads verified weights once
  into model-scoped immutable device buffers and executes the complete reviewed
  architecture through closed kernels. The registered production directory
loader can construct it only after signature, hash, tokenizer, and tensor
validation. The UDS daemon selects CPU reference by default, or ARM64/Metal only
through complete owner startup configuration. The same fair cursor,
  cancellation, replay, signed streaming, load/unload and descriptor path is
  used for either backend; requests and packages cannot choose it. Go pins the
  signed descriptor backend and rejects a mismatch. CUDA remains not
  implemented.
Any backend is selected by trusted engine configuration and architecture code,
never by model data, and must pass CPU differential and isolated unsafe/FFI
review.

## Unsafe and FFI boundary

Workspace policy denies unsafe Rust by default. The approved exceptions are
`engine-arm64/src/intrinsics.rs`, whose twelve Advanced SIMD blocks load and
store only complete in-bounds lane groups supplied by safe slices or fixed
arrays,
`engine-metal/src/ffi.rs`, whose safe wrappers contain the opaque pointer
lifetime, null/status checks, exact buffer lengths, fixed kernel enum, and
explicit destruction for the reviewed Objective-C++ bridge,
`engine-server/src/os_lock.rs`, which makes one checked `libc::flock` call using
an owned valid descriptor, and `engine-cli/src/os_signal.rs`, which uses two
documented `sigaction` blocks to install and restore owner-process handlers,
plus `engine-server/src/peer_credential.rs`, which reads the process and peer
effective user IDs through platform credential APIs. The signal handler retains
no pointer and performs only an atomic store; peer outputs use exact initialized
ABI storage and borrowed live descriptors. `engine-loader/src/read_only_map.rs`
owns exactly one nonempty mapping, exposes only an immutable slice, requests
`PROT_READ|MAP_PRIVATE` without execution, and unmaps the exact pointer/length
once while its file remains live. Its reviewed `Send`/`Sync` implementations
are valid only because shared access exposes immutable bytes and the mapping
cannot be unmapped until the last owning package is dropped. Architecture tests enforce the exact
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

## Reusable prefix K/V state

First-party CAG is implemented as a model-local `ReusableKvCacheStore` above the contiguous
request K/V cache. It accepts only exact, compatibility-bound token prefixes and copies sealed CPU
state into a fresh sequence before decode. Retrieval, ranking, document ordering, durable knowledge,
and CAG routing remain Go/Memory-Vault responsibilities. See [CAG.md](CAG.md),
[KV-CACHE-LIFECYCLE.md](KV-CACHE-LIFECYCLE.md), and [CACHE-SECURITY.md](CACHE-SECURITY.md).

CPU RAM is the only implemented reusable tier. Metal/CUDA migration and SSD persistence are
explicitly unsupported. `selective_cache_recomposition` is reserved and disabled pending
mathematical correctness and model-quality qualification.
