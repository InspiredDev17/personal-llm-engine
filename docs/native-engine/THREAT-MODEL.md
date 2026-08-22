# Native Engine Threat Model

## Assets and adversaries

Assets are Go control-plane authority, private signing keys, model/package
integrity, prompts and generated text, host memory/CPU/storage, UDS identity,
platform memory/audit, and the Agent Green boundary. Adversaries include a
malicious client, malformed or replaced package, hostile model output,
unprivileged local process, compromised dependency, and faulty backend/driver.

The owner and Go configuration are trusted to select model IDs, package hashes,
public keys, paths, budgets, and provider enablement. Model bytes, client frames,
token data, generated text, and engine events are untrusted.

## Trust boundaries

1. Owner configuration to CLI/service.
2. Model root/pathname to no-follow package opener.
3. Quarantined external data to the offline importer and signing boundary.
4. Signed directory-package files to format/hash/signature admission.
5. Parsed tokenizer/tensors to memory and CPU execution.
6. Go to protected UDS and authenticated message decoder.
7. Engine events back to Go.
8. Go to its separate authenticated Agent Green liaison.
9. Safe Rust to the isolated Unix lock/read-only-map FFI modules.
10. Safe Rust Metal backend to its reviewed isolated C ABI and future safe Rust
    CUDA backend to isolated FFI.

## Threats and controls

| Threat | Current controls | Residual/next gate |
| --- | --- | --- |
| Authority escalation through output | Rust has no tools, approval, memory, audit-policy, or Agent Green API; outputs are typed untrusted data | Go must retain authorization and injection defenses |
| Compromised ordinary agent | Only the Go control plane holds routing and approval authority; the UDS requires same-user kernel credentials plus a pinned signing identity, and requests remain bounded to registered model IDs | same-user agent compromise can still attempt signed requests if it compromises Go key material; service-account and key isolation require operated evidence |
| Prompt or tool-call injection | Rust treats prompt and generated text only as data and has no tool-call message, Action Broker credential, approval API, memory mutation API, or model-defined operation | Go must continue to mark output untrusted and require registered-tool policy, exact approval binding, and Action Broker enforcement |
| Model attempts to disable security or alter memory | No protocol field changes security telemetry, trust stores, audit, Agent Green, platform memory, or engine policy; model output cannot acknowledge retained events | Go event polling/acknowledgement and memory writes remain independently authorized control-plane operations |
| Arbitrary file access/path traversal | Clients send only `ModelId`; package is one safe direct child; importer canonicalizes source under quarantine; absolute/nested/traversal and URL inputs are rejected | OS sandbox should additionally restrict filesystem visibility |
| Symlink/replacement/non-file package | non-symlink read-only root/package/direct files, `O_NOFOLLOW`, regular-file, device/inode/length equality; importer rejects quarantine escape, seals output, and preserves replaced package | privileged/owner compromise remains out of process scope |
| Package tamper/downgrade | SHA-384 catalog identity and per-file/tensor hashes, closed file set, domain-separated strict Ed25519, key ID and active/revoked trust state | key rotation/rollback operational qualification remains |
| Downgrade or rollback to a vulnerable engine | Release-only signed manifests bind a nonzero generation and exact artifact hashes; a protected external-runtime ledger rejects lower generations and same-generation substitution; the Linux engine unit requires the root pre-launch verifier; no automatic updater exists | verifier/installer OS trust, release-key custody, and operated target rollback evidence remain release blockers |
| Model-supplied code, graph, kernels, architecture, or capability substitution | exact data-only directory contents and version-bound closed architecture IDs; v1.1 Qwen2.5 additionally fixes tied output, required biases, split-half RoPE and exact reviewed 1.5B configuration pairs, requires future engine 0.2.0, and is rejected by current 0.1.0; the exact profile also fails the unchanged scalar compute ceiling before materialization; the engine accepts no package/client graph definitions, plugins, dynamic libraries, scripts, WASM, Metal/PTX, paths, or kernel selectors | each future architecture/operation/backend needs a new reviewed identity, admission proof, and independent tests |
| Hostile source model | offline importer emits only its bounded legacy-Llama safetensors subset and strict `AIPTOK-METADATA-V1`; a disabled Qwen2.5 transition profile separately pins three config identities, one tokenizer graph/identity, one tensor-file identity per profile, exact names/shapes including Q/K/V biases, and tied output; its operator-only preflight requires owner-private direct children, `O_NOFOLLOW`/`O_CLOEXEC`, stable descriptor/path identity, exact length and streamed same-descriptor SHA-256, then validates only a bounded header containing exactly 338 BF16 non-overlapping full-extent tensors; arbitrary normalizers/pre-tokenizers/templates, pickle, `.bin`, `.pt`, `.pth`, scripts and unknown mappings are rejected | exact Qwen tokenizer conversion, a future emission path retaining the verified descriptor, engine 0.2 activation, Registry V2 admission, approved license/provenance, qualified optimized execution, and independent numeric evidence remain; Qwen package emission still fails before source I/O |
| Tokenizer ambiguity/injection | executable AIPTok v1 retains exact magic/version/family/behavior IDs, content hash, contiguous IDs/ranks, canonical byte fallback, required special roles and reject-by-default literal special text. AIPTok v2 has a separate magic/profile, validates fixed no-UNK/no-implicit-BOS Qwen table semantics, exact byte permutation, contiguous merges/added roles, undefined logit tail and production-count added-token literals, and rejects partial pinned cardinalities instead of downgrading them to generic structural mode. Its private runtime mechanics accept Unicode data only through an engine-owned adapter that checks both Unicode-16 data versions, bounds NFC output scalar-by-scalar, maps only General Category `L*`/`N*`, and pins a finite simple-fold table; neither runtime nor adapter can be constructed by package/client/service code | source conversion and production-table token-ID vectors remain mandatory, and parsed tables/private mechanics are not execution approval |
| Undefined or substituted output-token domain | each loaded model stores an immutable domain bound to its verified package hash, tokenizer hash, model logit count and tokenizer-defined count; candidate membership is enforced before logit reads or sampling filters; requests can narrow but never expand it; current AIPTok v1 still requires equality | future Qwen tokenizer conversion must prove its exact 151,665-ID contiguous domain before the disabled service gate can open |
| Tokenizer resource/UTF-8 denial | both formats have package and per-tokenizer input/output ceilings, bounded pieces/merges/specials, checked offsets/allocation/accounting and unit/property/malformed/fuzz coverage; executable v1 retains strict text/exact byte decode and bounded incremental suffix state; private v2 scanner/BPE/decode work is separately reserved and its incremental suffix is fixed-size | v2 has no public text API; production-size parser/runtime throughput, Unicode conformance and corpus campaigns remain required before activation |
| Tensor metadata or aliasing confusion | checked rank/dimensions/strides/offsets/length/alignment, explicit non-contiguous views, canonical owned layout, fresh operation destinations, Rust-exclusive mutable views, and one aggregate pool | allocator overhead and future FFI backends require separate accounting proof |
| Cross-request memory disclosure | explicit allocation owners; request-local generation-checked scratch handles; initialized buffers; clear on completion, cancellation, failure unwind and drop; idle scratch is separately reclaimable | OS crash dumps and privileged process inspection remain operational concerns |
| Cross-sequence KV disclosure | one model-owned cache; request-bound generation-checked sequence handles; internal-only offsets; atomic all-layer token commit; cancelled/reset/destroyed/unloaded storage is cleared; concurrent lifecycle stress tests | future accelerated/paged implementations require differential isolation and zeroing evidence |
| KV context or allocation pressure | checked exact contiguous plan; request and engine ceilings; full-sequence admission; per-token/prefill estimates; current/peak/used/reclaimable metrics; context checked before mutation | contiguous V1 reserves maximum context per sequence and is intentionally memory-inefficient |
| Stale or changed weight mapping | protected non-symlink files, secure descriptor identity, signed hash before exposure, read-only non-executable private mapping, hash recheck before tensor range exposure and explicit event-emitting full revalidation | privileged mutation between checks remains fail-detect rather than impossible without an immutable filesystem |
| Numerical instability or silent dtype fallback | execution is explicit F32; storage conversion accepts only F32/F16/BF16 with ties-to-even tests; exhaustive F16 patterns; stable F64 RMSNorm/softmax accumulation; non-finite arithmetic fails except explicit negative-infinity masks | real-model differential vectors and per-operation tolerances remain release gates |
| Quantized storage ambiguity or hidden quality loss | AIPQ8/AIPQ4 use fixed headers, block geometry, endian, scale, rounding, clamp, canonical code and tail rules; profiles are explicit; source/destination hashes and per-tensor/per-layer errors are signed; synthetic logit and token divergence is retained | real-model layer/logit/generation evaluation and acceptance remain mandatory; current service materializes F32 and claims no steady-state memory reduction |
| Malicious or invalid graph | engine-owned closed operation enum, static shape propagation, duplicate/missing/cycle rejection, deterministic topology, conservative intermediate-byte bound; packages and clients cannot submit graphs | future operation additions require architecture and malformed-graph review |
| Parser/size/overflow attack | fixed schemas, checked arithmetic, exact lengths, central ceilings, flat bounded collections, no arbitrary deserializer, malformed fixtures and bounded manifest/tensor-index/safetensors/path fuzz targets | production-size corpora and sustained campaigns remain required |
| Memory/CPU denial | explicit-owner aggregate reservations, fallible initialized allocation, request/engine scratch ceilings, exact contiguous KV admission/metrics, queue count/bytes, conservative active request/token/cursor/output retained-byte charging, tensor/context/token ceilings, CPU operation caps, finite socket deadlines, one 64-connection ceiling across authenticating/queued/active/draining ownership, eight-event output queues with ten-frame reservation, explicit 256 KiB ingress/cancellation/output stacks, one-token turns, request-local saturation, explicit CLI limits, concurrent cache stress, continuous multi-stream tests, bounded delayed-reader/restart soak, and a pinned finite workflow definition | the scheduled workflow is present on the default branch but successful hosted evidence is absent; allocator overhead plus production-size operated continuous/slow-reader and long-context soak remain |
| GPU-memory exhaustion or backend substitution | Metal startup requires owner-selected device, byte and buffer ceilings; allocation/reuse is accounted and initialized; one backend-neutral fair cursor preserves admission/cancellation; signed descriptors expose the actual loaded backend and Go pins it exactly; packages and clients cannot select backends | production-size GPU pressure, driver reset, process-level memory-pressure behavior and operated CPU rollback lack qualification |
| Placement-plan injection or cross-model staging | physical plans never cross the protocol; the engine derives immutable ranges from a verified package and binds model, artifact, digest, revision, architecture, quantization and backend; owner startup alone can select streamed scalar residency; stage materialization revalidates live mappings, uses a model-scoped owner, bounds prefetch depth to one, joins workers, drops staged data on cancellation/failure, and invalidates partial streamed KV writes | true hybrid placement and per-model canonical residency status are not implemented; allocator overhead and production pressure/throughput require operated evidence |
| UDS impersonation/injection | owner-private runtime directory, mode-0600 inode-owned socket, kernel-reported peer effective UID must equal the engine service UID before frame allocation/parsing, strict frame directions, separate Ed25519 request and response identities, explicit two-key overlap with no discovery; continuous ingress records bounded typed unauthorized/malformed/oversized/version telemetry and exact eight-failure thresholds | production supervisor must run Go and Rust under the reviewed same-user service identity; crash-loop classification and operated rotation remain supervisor qualification |
| Replay/reordering | connection-order checks plus a bounded mode-0600 per-control-key journal that synchronizes each authenticated nonce before execution, holds an exclusive lifetime lock, and rejects duplicate/corrupt/truncated/exhausted/write-uncertain/contended state; current/previous control keys use distinct retained journals; streams cannot switch keys; every signed event binds the request nonce and a strictly increasing ordinal | journal archival/recovery and operated rotation need qualification |
| Event or service-response spoofing/confusion | strict engine signature verified before message-body parse; signed key ID/request nonce/event ordinal; service kinds are derived from closed bodies; Rust rejects unknown values, kind/body mismatch, malformed collections, duplicate models, invalid token IDs and trailing bytes; a bounded harness re-signs mutated request/response bodies so coverage reaches post-authentication parsers; Go validates frame kind, request ID, UTF-8, exact lengths, usage arithmetic and closed errors | grow a shared cross-language production-size corpus without prompts or credentials |
| Response-key exposure | seed enters only as exact stdin bytes, is never argv/config/model data, and is zeroized after signer construction; signing state zeroizes on drop; Go accepts only an explicit two-key verification overlap | production secret injection, process memory, crash-dump and operated rotation policy need qualification |
| Secret leakage in errors/logs | bounded control-free diagnostics; static/sanitized error guidance; no host path/package/prompt in protocol failures | review daemon operational logging before release |
| Cancellation or lifecycle race | Go sends a higher-nonce signed same-stream cancellation before close; Rust watches concurrently; disconnect/malformed follow-up cancels fail-closed; each backend-neutral resumable advance checks its request-local flag; active cursors own immutable model/tokenizer handles; unload rejects scheduler ownership and Metal also invalidates device state; batch and continuous tests isolate cancellation/control traffic and graceful shutdown interrupts incomplete ingress, rejects queued work, cancels active cursors, drains writers, and joins all workers | operated cancellation/load/unload races and long-running production-model latency qualification remain |
| Unauthorized process shutdown | authenticated data-plane `ShutdownRequest` always produces signed `ShutdownRejected`; only OS-authorized local principals can deliver `SIGINT`/`SIGTERM`; handler only enters the bounded graceful-drain path | a future `ShutdownAccepted` path requires a separately reviewed administrative identity; run under a dedicated identity and supervisor policy |
| Dependency compromise | minimal exact lock, checked-in source-replacement vendor graph, SPDX checksums/licenses, exact boundary tests, no AI stack or build-time fetch | reproduce the release archive/offline build and renew advisory review for each release |
| Compromised build dependency or model source | external crate versions/checksums are locked, vendored, inventoried, and SBOM checked; model input stays quarantined until deterministic conversion, validation, provenance, hashing, and signing | repository-owned crates have no owner-selected license declaration; independent source provenance and release-builder attestation remain required |
| Replaced metallib or engine binary | build embeds the metallib bytes and SHA-384 identity; runtime validates the embedded metallib before device use; release-only manifests verify exact read-only engine/metallib artifacts; the Linux unit fails launch and retains a root-owned content-free event on mismatch; source/build boundary tests forbid package-provided kernels | OS trust of the verifier and operated tamper evidence remain unqualified |
| Crash-loop denial of service | bounded restart/backoff policy, typed repeated-ingress thresholds, graceful shutdown, replay-journal locking, and finite continuous soak | supervisor crash-loop event delivery and operated recovery have not been demonstrated |
| Prompt/event egress from Rust | permanent offline data-plane boundary; no network client dependency; architecture tests reject Internet sockets, raw socket FFI, async networking, subprocess egress, and cloud telemetry; authenticated UDS is the only communications path | production service definition must independently deny outbound networking and pass an operated no-egress test on each supported OS |
| Unsafe/FFI memory error | unsafe is denied by default; separate architecture-tested modules own replay `flock`, process `sigaction`, read-only kernel peer credentials, and private read-only non-executable tensor mappings; exact unsafe blocks are counted, lifetimes/descriptors are owned, and no wrapper exposes raw handles | confirm exact hosted results for each release, independently review each target OS, and isolate/document future backend unsafe invariants |
| Agent Green spoofing/control | engine has no Agent Green client/key; model cannot emit authoritative decision; Go maps allowlisted telemetry | live mapping/delivery belongs to Go integration tests |

## Hard limits

Package 1 TiB hard maximum (always lowered by configured memory), 64 MiB
directory manifest, 1,024 shards, 1,027 hash entries, 1 MiB provenance,
64 MiB safetensors header, 47-byte canonical tensor names, 2 MiB alignment, 262,144
tensors, rank 8, dimension `1<<30`, elements `1<<40`, 4,194,304 vocabulary
entries, 1,048,576-token collections, 262,144 context tokens, 4 MiB requests,
8 MiB complete frames, 64 MiB tokenizer data, 4,096 tokenizer special tokens,
256-byte token pieces, 4,096 sequences, 64 MiB queued request/active-retention
bytes, 8 pending events per active output stream, 64
continuously owned/reference-batch connections, 256 KiB per ingress,
cancellation, or output worker stack, 100
million CPU operations per bounded operation, and a 100-byte portable UDS path.
Each control-key replay journal retains at most 1,048,576
nonces before fail-closed key rotation is required.

## Security events and Agent Green

`SecurityEvent` contains bounded sanitized categories, never a model-created
command. Events travel only to Go. Go owns validation, audit, allowlisting, and
authenticated delivery to Agent Green. Agent Green can receive those events but
is not controlled by the model or Rust service.

## Production stop conditions

Do not enable mandatory production routing if coverage-guided parser/protocol
fuzzing is missing; a real package lacks provenance/license/signature evidence;
the vendored/SBOM-recorded release cannot reproduce offline; resource/soak
tests fail; a backend differs from
CPU beyond documented tolerance; active cancellation is required but not
implemented; a client path, TCP/outbound network, dynamic loading, Python,
TypeScript, model code, tool, memory, approval, or Agent Green authority appears;
the target service lacks independent OS egress confinement; an operated
no-egress test fails; or rollback through Go cannot be demonstrated.

## Reusable K/V cache threats

Reusable cache state adds stale-state, cross-scope disclosure, cross-model substitution, mutable
shared-state, guessed-ID, memory-exhaustion, and positional-mismatch risks. V1 mitigates these with
exact compatibility keys, namespace-plus-owner authorization, immutable sealed storage,
request-local copies, content revalidation, deterministic bounded eviction, active-sequence
headroom, one-shot staged attachments, and path-free authenticated protocol operations. A cache ID
is never treated as authorization. Residual risks and unsupported persistence/device tiers are
tracked in [CACHE-SECURITY.md](CACHE-SECURITY.md).
