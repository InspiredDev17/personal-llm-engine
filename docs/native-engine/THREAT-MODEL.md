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
10. Future safe Rust backend to isolated Metal/CUDA FFI.

## Threats and controls

| Threat | Current controls | Residual/next gate |
| --- | --- | --- |
| Authority escalation through output | Rust has no tools, approval, memory, audit-policy, or Agent Green API; outputs are typed untrusted data | Go must retain authorization and injection defenses |
| Arbitrary file access/path traversal | Clients send only `ModelId`; package is one safe direct child; importer canonicalizes source under quarantine; absolute/nested/traversal and URL inputs are rejected | OS sandbox should additionally restrict filesystem visibility |
| Symlink/replacement/non-file package | non-symlink read-only root/package/direct files, `O_NOFOLLOW`, regular-file, device/inode/length equality; importer rejects quarantine escape, seals output, and preserves replaced package | privileged/owner compromise remains out of process scope |
| Package tamper/downgrade | SHA-384 catalog identity and per-file/tensor hashes, closed file set, domain-separated strict Ed25519, key ID and active/revoked trust state | key rotation/rollback operational qualification remains |
| Model-supplied code, graph, or kernels | exact data-only directory contents and fixed architecture; the engine constructs an immutable closed-enum graph; no package/client graph definitions, plugins, dynamic libraries, scripts, WASM, Metal/PTX, paths, or kernel selectors | each future operation/backend needs separate code review |
| Hostile source model | offline importer accepts only its bounded safetensors subset and strict `AIPTOK-METADATA-V1`; arbitrary normalizers/pre-tokenizers/templates, pickle, `.bin`, `.pt`, `.pth`, scripts and unknown mappings are not parser inputs | approved real-model license/provenance and independent numeric evidence remain |
| Tokenizer ambiguity/injection | exact AIPTok magic/version/family/behavior IDs, content hash, contiguous IDs/ranks, unique token bytes/pairs/results, canonical byte fallback, required special roles, no programmable normalization/templates, and reject-by-default literal special text | each real-model conversion still needs independent tokenizer vectors and provenance review |
| Tokenizer resource/UTF-8 denial | package and per-tokenizer input/output ceilings, bounded pieces/merges/specials, checked allocation/accounting, strict text decode, exact byte decode, bounded incremental suffix state, unit/property/malformed/fuzz coverage | production-size tokenizer throughput and corpus campaigns remain required |
| Tensor metadata or aliasing confusion | checked rank/dimensions/strides/offsets/length/alignment, explicit non-contiguous views, canonical owned layout, fresh operation destinations, Rust-exclusive mutable views, and one aggregate pool | allocator overhead and future FFI backends require separate accounting proof |
| Numerical instability or silent dtype fallback | execution is explicit F32; storage conversion accepts only F32/F16/BF16 with ties-to-even tests; exhaustive F16 patterns; stable F64 RMSNorm/softmax accumulation; non-finite arithmetic fails except explicit negative-infinity masks | real-model differential vectors and per-operation tolerances remain release gates |
| Malicious or invalid graph | engine-owned closed operation enum, static shape propagation, duplicate/missing/cycle rejection, deterministic topology, conservative intermediate-byte bound; packages and clients cannot submit graphs | future operation additions require architecture and malformed-graph review |
| Parser/size/overflow attack | fixed schemas, checked arithmetic, exact lengths, central ceilings, flat bounded collections, no arbitrary deserializer, malformed fixtures and bounded manifest/tensor-index/safetensors/path fuzz targets | production-size corpora and sustained campaigns remain required |
| Memory/CPU denial | aggregate tensor reservations, fallible allocation, queue count/bytes, conservative active request/token/cursor/output retained-byte charging, tensor/context/token ceilings, CPU operation caps, finite socket deadlines, one 64-connection ceiling across authenticating/queued/active/draining ownership, eight-event output queues with ten-frame reservation, explicit 256 KiB ingress/cancellation/output stacks, one-token turns, request-local saturation, explicit CLI limits, continuous multi-stream tests, bounded delayed-reader/restart soak, and a pinned finite workflow definition | hosted cron evidence awaits default-branch integration; allocator-overhead measurement plus production-size operated continuous/slow-reader and long-context soak remain |
| UDS impersonation/injection | owner-private runtime directory, mode-0600 inode-owned socket, kernel-reported peer effective UID must equal the engine service UID before frame allocation/parsing, strict frame directions, separate Ed25519 request and response identities, explicit two-key overlap with no discovery | production supervisor must run Go and Rust under the reviewed same-user service identity; operated rotation still needs qualification |
| Replay/reordering | connection-order checks plus a bounded mode-0600 per-control-key journal that synchronizes each authenticated nonce before execution, holds an exclusive lifetime lock, and rejects duplicate/corrupt/truncated/exhausted/write-uncertain/contended state; current/previous control keys use distinct retained journals; streams cannot switch keys; every signed event binds the request nonce and a strictly increasing ordinal | journal archival/recovery and operated rotation need qualification |
| Event spoofing/confusion | strict engine signature verified before parse; signed key ID/request nonce/event ordinal; Go validates frame kind, request ID, UTF-8, exact lengths, usage arithmetic and closed errors | add continuous fuzz corpus shared across languages |
| Response-key exposure | seed enters only as exact stdin bytes, is never argv/config/model data, and is zeroized after signer construction; signing state zeroizes on drop; Go accepts only an explicit two-key verification overlap | production secret injection, process memory, crash-dump and operated rotation policy need qualification |
| Secret leakage in errors/logs | bounded control-free diagnostics; static/sanitized error guidance; no host path/package/prompt in protocol failures | review daemon operational logging before release |
| Cancellation failure | Go sends a higher-nonce signed same-stream cancellation before close; Rust watches concurrently; disconnect/malformed follow-up cancels fail-closed; each resumable CPU advance checks its request-local flag; batch and continuous tests isolate cancellation and graceful shutdown interrupts incomplete ingress, rejects queued work, cancels active cursors, drains writers, and joins all workers | operated cancellation races and long-running production-model latency qualification remain |
| Unauthorized process shutdown | no protocol/model shutdown command; only OS-authorized local principals can deliver `SIGINT`/`SIGTERM`; handler only enters the bounded graceful-drain path | run the service under a dedicated identity and target-specific supervisor/sandbox policy |
| Dependency compromise | minimal exact lock, checked-in source-replacement vendor graph, SPDX checksums/licenses, exact boundary tests, no AI stack or build-time fetch | reproduce the release archive/offline build and renew advisory review for each release |
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
