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
3. Package bytes to format/hash/signature admission.
4. Parsed tokenizer/tensors to memory and CPU execution.
5. Go to protected UDS and authenticated message decoder.
6. Engine events back to Go.
7. Go to its separate authenticated Agent Green liaison.
8. Safe Rust replay journal to the isolated Unix file-lock FFI.
9. Future safe Rust backend to isolated Metal/CUDA FFI.

## Threats and controls

| Threat | Current controls | Residual/next gate |
| --- | --- | --- |
| Authority escalation through output | Rust has no tools, approval, memory, audit-policy, or Agent Green API; outputs are typed untrusted data | Go must retain authorization and injection defenses |
| Arbitrary file access/path traversal | Clients send only `ModelId`; owner package is one direct normal child; absolute/nested/traversal rejected | OS sandbox should additionally restrict filesystem visibility |
| Symlink/replacement/non-file package | non-symlink root/file, no group/other write, `O_NOFOLLOW`, regular-file, device/inode/length equality, exact read | privileged/owner compromise remains out of process scope |
| Package tamper/downgrade | exact SHA-384 identity, canonical V1 ranges/records, domain-separated strict Ed25519, key ID and active/revoked trust state | key rotation/rollback operational tooling is Stage 9 |
| Model-supplied code or kernels | flat data-only fixed architecture; no plugins, graph executor, dynamic libraries, Metal/PTX, paths, or kernel selectors | each future backend needs separate code review |
| Parser/size/overflow attack | fixed schemas, checked arithmetic, exact lengths, central ceilings, flat bounded collections, no arbitrary deserializer, exhaustive fixture truncation/single-byte mutation tests | coverage-guided fuzzing and production-size corpora remain required |
| Memory/CPU denial | aggregate memory reservations, fallible allocation, queue count/bytes, tensor/context/token ceilings, CPU operation caps, finite socket deadlines | production long-context/soak and multi-client fairness remain |
| UDS impersonation/injection | owner-private runtime directory, mode-0600 inode-owned socket, strict frame directions, separate pinned Ed25519 request and response identities | bind peer-credential policy and key rotation need operational qualification |
| Replay/reordering | connection-order checks plus a bounded mode-0600 per-control-key journal that synchronizes each authenticated nonce before execution, holds an exclusive lifetime lock, and rejects duplicate/corrupt/truncated/exhausted/write-uncertain/contended state; every signed event binds the request nonce and a strictly increasing ordinal | production key rotation and journal lifecycle/recovery need operational qualification |
| Event spoofing/confusion | strict engine signature verified before parse; signed key ID/request nonce/event ordinal; Go validates frame kind, request ID, UTF-8, exact lengths, usage arithmetic and closed errors | add continuous fuzz corpus shared across languages |
| Response-key exposure | seed enters only as exact stdin bytes, is never argv/config/model data, and is zeroized after signer construction; signing state zeroizes on drop | production secret injection, rotation, process memory and crash-dump policy need operational qualification |
| Secret leakage in errors/logs | bounded control-free diagnostics; static/sanitized error guidance; no host path/package/prompt in protocol failures | review daemon operational logging before release |
| Cancellation failure | Go sends a higher-nonce signed same-stream cancellation before close; Rust watches concurrently; disconnect/malformed follow-up cancels fail-closed; CPU loop checks the request-local flag | multi-worker cancellation and long-running production-model latency qualification remain |
| Dependency compromise | minimal exact lock, checked-in source-replacement vendor graph, SPDX checksums/licenses, exact boundary tests, no AI stack or build-time fetch | reproduce the release archive/offline build and renew advisory review for each release |
| Unsafe/FFI memory error | unsafe is denied by default; one architecture-tested Unix module calls `flock` with a live owned descriptor, checks the result, exposes no raw descriptor/clone, and contains no pointer operation | independently review each target OS; isolate and document any future backend unsafe invariants separately |
| Agent Green spoofing/control | engine has no Agent Green client/key; model cannot emit authoritative decision; Go maps allowlisted telemetry | live mapping/delivery belongs to Go integration tests |

## Hard limits

Package 1 TiB hard maximum (always lowered by configured memory), 262,144
tensors, rank 8, dimension `1<<30`, elements `1<<40`, 4,194,304 vocabulary
entries, 1,048,576-token collections, 262,144 context tokens, 4 MiB requests,
8 MiB complete frames, 64 MiB tokenizer data, 4,096 sequences, 64 MiB queued
request bytes, 100 million CPU operations per bounded operation, and 100-byte
portable UDS path. Each control-key replay journal retains at most 1,048,576
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
or rollback through Go cannot be demonstrated.
