# Native Engine Known Limitations

## Release and operations

- This revision is NO-GO for production. Source now provides release-only
  manifest signing, exact installed-artifact measurement, a monotonic rollback
  ledger, a Linux pre-launch unit binding, and a root-owned event bridge. No
  OS-authenticated verifier installation, release-key custody, or operated
  upgrade/tamper/rollback evidence exists yet.
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

- Manifest v1.0 accepts `decoder-rms-rope-swiglu-gqa-v1`. Engine-private
  manifest v1.1 parses the exact
  `qwen2.5-rms-rope-swiglu-gqa-bias-tied-v1` 1.5B bootstrap profiles as future
  0.2.0 artifacts. The current 0.1.0 loader rejects them. Tiny configurations
  provide scalar math evidence only; the exact 1.5B position-zero estimate is
  1,543,712,768 operations and fails the 100,000,000-operation scalar ceiling
  before model materialization. A bounded data-only `AIPTOK02` Qwen table
  parser now exists, but exact NFC/category pre-tokenization,
  encode/decode/incremental decode and importer conversion do not. Offline
  importer emission, the Model Foundry registry compatibility profile, an
  optimized backend, signed package, independent logits, and production
  qualification remain absent. ARM64 and Metal reject the identifier. A second
  authenticated-service
  tokenizer gate prevents AIPTok v1 from being mislabeled after any future
  engine-version activation.
- Only synthetic signed tiny packages are qualified. No production model,
  license/provenance review, tokenizer conformance corpus, long-context quality,
  or generated-output acceptance evidence exists.
- Scalar CPU is the correctness oracle. ARM64 and Metal operation tests pass
  documented synthetic tolerances. Both now run complete signed synthetic-model
  generation through explicit owner-selected UDS execution, and Go pins the
  signed descriptor. ARM64 currently materializes bounded intermediates and KV
  history around the kernel interface. Production-size canaries and operated
  CPU rollback evidence do not exist for either accelerated path.
- The measured 128x128 Metal matvec is a microbenchmark, not production token
  throughput. Prior recorded methodology and numbers remain in [METAL.md](METAL.md).
- AIPQ8/AIPQ4 synthetic error evidence does not establish real-model quality.

## Resource and service behavior

- Signed CPU, ARM64, and Metal loads now derive exact fully resident placement
  plans. Scalar CPU streamed prefill/decode runs through bounded layer staging
  and depth-one prefetch with signed-fixture parity. Owner startup may select it
  only for the scalar CPU service; fully resident remains the default and
  clients cannot override it. The V1 descriptor does not report residency mode.
  Hybrid CPU/GPU placement is explicitly unsupported. There is no evidence that
  streaming is faster than residency.
- Placement accounting uses conservative logical bytes for mapped content,
  immutable metadata, execution tensors, staged metadata, existing reservations,
  and safety reserve. General allocator/page-table overhead remains part of the
  operated memory-pressure qualification gate.
- Automatic pressure eviction remains absent. The Go control plane may request
  exact idle unload; without that authorization the engine rejects pressure
  rather than selecting a cognitively different model.
- Contiguous KV cache reserves maximum context and is memory-inefficient; paged
  KV is not implemented.
- Continuous batching, distributed execution, speculative decoding, beam
  search, grammar decoding, and remote serving are intentionally absent.
- The correctness-first Metal decoder synchronizes and transfers intermediates
  between operations; it is not command-buffer batched or fused and is much
  slower than CPU on the tiny fixture. Device KV history currently crosses
  bounded host tensors for attention dispatch. Metal driver reset, GPU exhaustion under production-size loads, prolonged
  thermal behavior, and days-long restart/cancellation soak are unqualified.
- Security telemetry is bounded and content-free. The verifier rejects
  binary/metallib mismatch and downgrade and retains bounded pre-launch
  categories for Go/Agent Green delivery. A root-owned monotonic sequence
  checkpoint prevents ring reset from reusing event identity. Go integration
  `1951a42` now persists provider quarantine and exact-revision audited owner
  recovery, but the source bridge, quarantine, and recovery workflow are not
  yet operated or qualified on the production host.

## Tooling gaps

- The local pinned nightly lacks Miri.
- LeakSanitizer is unsupported on this macOS platform; ASan and UBSan passed the
  bridge round trip with leak detection disabled.
- `staticcheck`, `govulncheck`, `syft`, and `cosign` were unavailable locally.
- Five-second fuzz runs are smoke campaigns, not sustained coverage assurance.
