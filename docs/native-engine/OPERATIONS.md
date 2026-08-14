# Native Engine Operations

## Authority and identity

Run the Rust engine as a dedicated unprivileged identity. Go remains the sole
control plane. The engine identity receives no shell, cloud credentials,
tool/Action Broker credentials, Agent Green key, home directory, model signing
key, or write access to the model store. Go and Rust use pinned, separate
request/response Ed25519 identities plus kernel peer credentials over the one
owner-private Unix socket.

## Filesystem and process layout

| Path/class | Required access |
| --- | --- |
| engine binary and supporting release files | root/owner installed, read-only to service |
| approved model store | read-only; direct-child registered packages only |
| runtime directory | service-owned mode 0700 |
| Unix socket | service-owned mode 0600 |
| replay journals | service-owned mode 0600 beneath runtime root |
| response signing seed | exact 32-byte supervisor credential on stdin; never environment/argv/repository |

The Linux unit denies IP networking, permits only `AF_UNIX`, denies privilege
gain/capabilities, protects home/system/kernel/proc namespaces, allows writes
only to the runtime directory, disables core dumps and swap, and caps open
files (1,024), processes/tasks (128), and memory (16 GiB). Operators must lower
the memory ceiling where the approved package and hardware permit; raising it
requires recorded capacity review.

## Preflight

1. Verify the exact source revision, Rust/Go/Apple compiler versions, target,
   release flags, lockfiles, vendor tree, SPDX SBOM, and advisory snapshot.
2. Verify release signatures once a release-signing system exists. Until then,
   deployment is prohibited by [RELEASE-READINESS.md](RELEASE-READINESS.md).
3. Verify the configured package ID/hash/signer/minimum engine version and
   read-only, no-symlink model/runtime parent properties.
4. Verify the Go provider has cloud fallback disabled for the local route and
   preserves approvals, Action Broker, memory, audit, and Agent Green boundaries.
5. Run direct-host Metal validation if Metal will be enabled; a skipped GPU
   test is not release evidence.
   Start Metal only with `--execution-backend metal` plus explicit device index,
   device-memory ceiling and buffer-count ceiling. Omission selects CPU
   reference. Confirm Go pins `PERSONALAI_NATIVE_ENGINE_BACKEND_ID=metal` and
   rejects any signed descriptor mismatch before registering the provider.
6. On Linux, run `systemd-analyze security` and an operated UDS-success plus
   IPv4/IPv6-denial test. On macOS, run `verify-sandbox.sh`; treat it only as a
   transitional development proof.

## Start, health, and shutdown

Use the reviewed supervisor unit/launcher and fixed configuration. Never accept
runtime model paths or trust keys from a client. Read health, capability, model
status, metrics, and security events only through authenticated protocol
messages. Ordinary data-plane shutdown requests are always rejected; stop with
the supervisor's OS-authorized signal and wait for bounded drain/join cleanup.

## Monitoring and privacy

Alert on authentication failures, malformed/oversized/version-invalid frames,
resource exhaustion patterns, package signature/hash rejection, backend
failure, replay-journal contention/corruption, repeated rejection, and crash
loops. Metrics and logs must never contain prompts, token text, generated text,
keys, host package paths, or raw tensor bytes. Disable and securely contain core
dumps. Collect only bounded structured diagnostics.

## Key rotation and revocation

- Model/control/response keys have distinct roles and IDs. Never reuse a seed.
- Install the new public key as the bounded previous/current overlap before
  switching the signer. Verify both directions and replay journals, then revoke
  and remove the old private material.
- Revoked model/control keys remain identifiable but never resolve as active.
- If any private key may be exposed, stop admission, preserve content-free
  audit evidence, revoke the key, quarantine affected packages/releases, rotate
  to a fresh key, and require independent revalidation before restart.
- Response-key seed rotation requires supervisor credential replacement and a
  coordinated Go verification overlap. Never pass the seed via environment,
  command line, logs, model package, or source control.

## Rollback

There is currently no enforceable installed-engine rollback ledger; therefore
production deployment is blocked. The required future policy is monotonic:
store the highest approved release generation outside the writable engine
runtime, reject lower generations even when correctly signed, and require a
separate owner-authorized emergency recovery record for any downgrade. Model
registry activation already selects an exact signed package entry and must not
fall back to older or unregistered content.

Backend rollback is configuration-only and does not weaken package identity:
drain and stop Rust, restart it with `--execution-backend cpu-reference`, set
Go's pinned backend identity to `cpu-reference`, then restart Go and require its
full signed health, capability, registration, load and descriptor preflight.
There is no in-process automatic Metal-to-CPU model switch and no client
override. This reversible restart procedure is source-complete but still needs
an operated canary/rollback record on the target deployment.
