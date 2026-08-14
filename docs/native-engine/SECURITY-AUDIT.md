# Native Engine Security Audit

## Scope and method

This defensive review covered the Go authority boundary, Rust engine, AIPack,
AIPTok, importer, cryptography, loader/mapping, tensors, transformer, sampling,
memory/KV, scheduler, UDS service, ARM64, Objective-C++, Metal, routing,
permissions, security telemetry, Agent Green mapping, and build/release files.
The reviewed engine baseline was `866e713`; the reviewed published Go
integration baseline was `3c7c082`. The hardening delta adds fuzz boundaries,
process limits, threat coverage, and these evidence records.

## Findings

| ID | Severity | Status | Finding |
| --- | --- | --- | --- |
| R-01 | P0 release | Reduced | Release-only signed manifests, exact read-only artifact measurement, and a monotonic external-runtime ledger are source-complete; trusted installer/supervisor integration, key custody, and operated rollback evidence remain production blockers |
| R-02 | P1 release | Open | Repository-owned Rust crates have no declared owner license, so complete machine license policy cannot pass honestly |
| R-03 | P1 qualification | Reduced | Registered signed synthetic packages now complete Metal prefill/decode/sampling/streaming with scalar parity and persistent weights; owner-selected UDS execution and exact Go descriptor pinning are implemented, while operated canary/rollback and production-size qualification remain |
| R-04 | P1 operations | Open | Linux confinement and crash-loop telemetry lack operated-host proof; macOS Seatbelt proof is transitional |
| R-05 | P2 tooling | Open | Miri component is unavailable for the pinned nightly; pure-Rust Miri evidence is absent |
| R-06 | P2 telemetry | Reduced | The offline verifier detects installed binary/metallib mismatch and downgrade; a qualified supervisor-to-Go event bridge and model-quarantine source remain absent |
| R-07 | P2 release | Open | Published Go integration and separate local Go security commits are divergent and need deliberate reconciliation |
| F-01 | P2 tooling | Fixed | Missing explicit fuzz targets for configuration import, authenticated generation requests, and sampling parameters |
| F-02 | P2 operations | Fixed | Linux service profile lacked explicit core, descriptor, process, task, memory/swap, home, namespace, and common secret-environment limits |

No unresolved source-level P0 memory-safety, parser-bypass, authentication-
bypass, public-listener, outbound-engine-network, arbitrary-code-loading, or
Go-approval-bypass defect was demonstrated. R-01 is nevertheless a P0 release
blocker because source controls have not yet been bound to a trusted installer,
supervisor identity, release key, and operated target evidence.

## Boundary conclusions

- Clients provide registered model IDs, never package paths, libraries, kernel
  names, commands, trust keys, or security controls.
- The engine cannot execute tools, approve actions, mutate platform memory,
  contact Agent Green, or make outbound Internet connections.
- Prompt and generated text are untrusted data. The protocol contains no
  direct tool invocation. Go retains model routing, cloud authorization,
  context assembly, approval, audit, memory, and Action Broker enforcement.
- AIPack/AIPTok and safetensors source parsing are flat, bounded, checked, and
  data-only. Production packages are registered, hashed, signed, compatible,
  closed-file-set packages opened without symlink following.
- Request/KV/scratch ownership is explicit and cleared or invalidated on
  cancellation, release, reuse, and unload. Tests cover cross-sequence and
  delayed-client isolation.
- Engine security events are retained outside model output and polled/acked by
  authenticated Go messages. Ordinary output cannot suppress them. Current
  event coverage is listed in [KNOWN-LIMITATIONS.md](KNOWN-LIMITATIONS.md).

## Fuzz and sanitizer evidence

Nine separate five-second libFuzzer campaigns plus one corrected generation-
request rerun completed successfully on 2026-08-13 (60 aggregate wall-clock
seconds as reported by the targets). No crash artifact was produced. This is
bounded smoke evidence only.

The new engine-release manifest parser additionally completed a bounded
1,000-run macOS ARM64 libFuzzer smoke campaign on 2026-08-14: 47 coverage
counters, 55 features, less than 40 MiB RSS, and no crash, timeout, sanitizer,
or resource-limit artifact. This does not establish parser exhaustion.

The Objective-C++ bridge was compiled and run with Apple clang 21.0.0:

```sh
xcrun --sdk macosx clang++ -std=c++17 -fobjc-arc -fno-exceptions \
  -fno-omit-frame-pointer -fsanitize=address,undefined -Wall -Wextra -Werror \
  -I native/apple/metal/include native/apple/metal/src/native_metal_bridge.mm \
  /private/tmp/pai_metal_sanitizer_harness.mm -framework Metal \
  -framework Foundation -o /private/tmp/pai-metal-sanitizer-harness
ASAN_OPTIONS=detect_leaks=0:halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1 \
  /private/tmp/pai-metal-sanitizer-harness personalai.metallib
```

The vector-add buffer lifecycle exited zero with no ASan or UBSan finding.
`detect_leaks=1` aborted because LeakSanitizer is unsupported on this platform;
that is a tool limitation, not a passing leak check.

## Analysis-tool evidence

- `cargo audit --no-fetch --file Cargo.lock`: exit zero; 43 dependency entries
  scanned against 1,216 locally cached advisories, no vulnerability reported.
- `cargo deny check`: advisories and sources passed, but overall failed because
  repository crates lack a license and workspace path dependencies inherit
  versions. No ignore/allow weakening was accepted. External licenses remain
  exactly checked by the inventory/SPDX boundary test.
- `cargo +nightly-2026-08-01 miri test -p personalai-engine-types --lib`:
  unavailable because `cargo-miri` is not installed for the pinned nightly.
- `staticcheck`, `govulncheck`, `syft`, and `cosign`: not installed locally.
- `systemd-analyze`: unavailable on macOS; static unit tests passed.

## Red-team mapping

Existing malformed and integration suites reject model-root escape, symlinks,
extra executable files, tensor overlap/overflow/out-of-bounds ranges, excessive
allocations, oversized/partial/wrong-direction frames, replay, cross-sequence
IDs, stale KV state, unauthorized/unregistered load, protocol shutdown, trust-
store mutation fields, cloud fallback, direct tool execution, approval bypass,
and security-event suppression. These are repository-owned defensive tests;
they do not target external systems.
