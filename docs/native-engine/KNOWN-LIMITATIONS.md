# Native Engine Known Limitations

## Release and operations

- This revision is NO-GO for production. Source now provides release-only
  manifest signing, exact installed-artifact measurement, and a monotonic
  rollback ledger. No target installer/supervisor invocation, OS-authenticated
  verifier installation, release-key custody, or operated rollback evidence
  exists yet.
- Model-package and engine-release signatures are separate roles. Never treat a
  signed model as proof that the engine binary is trusted, or vice versa.
- No automatic updater exists. Installation, rollback, and revocation are
  operator actions and must remain fail-closed.
- The Rust workspace's own license has not been selected by the owner. External
  dependency licenses are inventoried and SBOM checked.
- Linux systemd controls have static test evidence only. The macOS Seatbelt
  profile passed UDS/no-loopback proof but uses deprecated `sandbox-exec` and
  grants broad read access needed by the transitional launcher.

## Models and numerical evidence

- Only `decoder-rms-rope-swiglu-gqa-v1` is accepted.
- Only synthetic signed tiny packages are qualified. No production model,
  license/provenance review, tokenizer conformance corpus, long-context quality,
  or generated-output acceptance evidence exists.
- Scalar CPU is the correctness oracle. ARM64 and Metal operation tests pass
  documented synthetic tolerances. Metal now has persistent device-resident
  weights and complete signed synthetic-model generation. The UDS daemon now
  has explicit owner-selected Metal execution and Go already pins the signed
  descriptor, but production-size canary and operated CPU rollback evidence do
  not exist.
- The measured 128x128 Metal matvec is a microbenchmark, not production token
  throughput. Prior recorded methodology and numbers remain in [METAL.md](METAL.md).
- AIPQ8/AIPQ4 synthetic error evidence does not establish real-model quality.

## Resource and service behavior

- Contiguous KV cache reserves maximum context and is memory-inefficient; paged
  KV is not implemented.
- Continuous batching, distributed execution, speculative decoding, beam
  search, grammar decoding, and remote serving are intentionally absent.
- The correctness-first Metal decoder synchronizes and transfers intermediates
  between operations; it is not command-buffer batched or fused and is much
  slower than CPU on the tiny fixture. Device KV history currently crosses
  bounded host tensors for attention dispatch. Metal driver reset, GPU exhaustion under production-size loads, prolonged
  thermal behavior, and days-long restart/cancellation soak are unqualified.
- Security telemetry is bounded and content-free. The offline verifier now
  rejects binary/metallib mismatch and downgrade, but no qualified supervisor
  event bridge yet converts those pre-launch failures into Go/Agent Green
  events. Supervisor model quarantine also remains absent.

## Tooling gaps

- The local pinned nightly lacks Miri.
- LeakSanitizer is unsupported on this macOS platform; ASan and UBSan passed the
  bridge round trip with leak detection disabled.
- `staticcheck`, `govulncheck`, `syft`, and `cosign` were unavailable locally.
- Five-second fuzz runs are smoke campaigns, not sustained coverage assurance.
