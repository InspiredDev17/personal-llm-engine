# Native Engine Known Limitations

## Release and operations

- This revision is NO-GO for production. It has no signed engine binary or
  signed release-manifest verification, installed binary/metallib measurement,
  monotonic version ledger, or rollback enforcement.
- AIPack signing is implemented; release-artifact signing is a separate missing
  control. Never treat a signed model as proof that the engine binary is trusted.
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
  documented synthetic tolerances, but production generation remains CPU-
  backed and persistent device-resident model execution is absent.
- The measured 128x128 Metal matvec is a microbenchmark, not production token
  throughput. Prior recorded methodology and numbers remain in [METAL.md](METAL.md).
- AIPQ8/AIPQ4 synthetic error evidence does not establish real-model quality.

## Resource and service behavior

- Contiguous KV cache reserves maximum context and is memory-inefficient; paged
  KV is not implemented.
- Continuous batching, distributed execution, speculative decoding, beam
  search, grammar decoding, and remote serving are intentionally absent.
- Metal driver reset, GPU exhaustion under production-size loads, prolonged
  thermal behavior, and days-long restart/cancellation soak are unqualified.
- Security telemetry is bounded and content-free. It cannot currently report
  installed engine binary mismatch, installed metallib mismatch, downgrade
  attempt, or supervisor model quarantine because the release verifier and
  supervisor event source do not exist.

## Tooling gaps

- The local pinned nightly lacks Miri.
- LeakSanitizer is unsupported on this macOS platform; ASan and UBSan passed the
  bridge round trip with leak detection disabled.
- `staticcheck`, `govulncheck`, `syft`, and `cosign` were unavailable locally.
- Five-second fuzz runs are smoke campaigns, not sustained coverage assurance.
