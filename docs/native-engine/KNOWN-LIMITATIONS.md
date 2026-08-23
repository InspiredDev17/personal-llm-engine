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

- Manifest v1.0 accepts `decoder-rms-rope-swiglu-gqa-v1`. Engine 0.2.0 parses
  the exact engine-private manifest v1.1
  `qwen2.5-rms-rope-swiglu-gqa-bias-tied-v1` 1.5B bootstrap profiles. Tiny configurations
  provide scalar math evidence only; the exact 1.5B position-zero estimate is
  1,543,712,768 operations and fails the 100,000,000-operation scalar ceiling
  before model materialization. A bounded `AIPTOK02` Qwen table parser and a
  private fixed scanner/BPE/decode/incremental-decode core now exist. A narrow
  wrapper can construct the exact pinned profile only after revalidating its
  complete package tokenizer hash. A startup-only, default-disabled metadata readiness surface can bind
  a signed future package to that tokenizer, its selectable domain and an exact
  Runtime-v3 identity, but it deliberately does not verify tensor contents,
  perform runtime resource admission, or expose an executable result. The core
  accepts Unicode behavior only through a private trusted
  Unicode-16 interface. A version-checked private adapter supplies
  bounded NFC, every `L*`/`N*` General Category, and the closed contraction
  simple-fold mapping. Exact pinned-source conversion and fixed production
  encode/decode/incremental-decode replay now pass through an offline,
  feature-gated content-free diagnostic. An offline exporter can publish only
  that exact AIPTOK02 to a fresh owner-private read-only path. Registry V2
  remains frozen; a separate signed V3 development record plus canonical
  Models 2.0 descriptor and explicit owner startup approval can admit only the
  ARM64 path. A real signed package, independent logits, and production
  qualification remain absent. ARM64 uses the exact split-half scalar
  correctness operation instead of its legacy interleaved RoPE when the Qwen
  architecture is selected; that is not optimized-backend qualification.
  CPU-reference and Metal Qwen service remain closed. A second
  authenticated-service tokenizer gate prevents AIPTok v1 from being
  mislabeled.
- A separate default-disabled ARM64 preparation policy now derives the exact
  selectable domain and bounded per-sequence KV plan from verified metadata.
  ARM64 session/step admission retains checked operation arithmetic without
  applying the scalar correctness-oracle ceiling. This is model-independent
  resource preparation is composed only by the explicit Qwen V3 ARM64
  development load. It remains blocked without exact package verification,
  production qualification, optimized Qwen evidence, and explicit owner
  activation. Client requests cannot enable it.
- The package loader can now cooperatively cancel full shard hashing and tensor
  materialization, and BF16/F16 conversion no longer retains a second owned
  source tensor. This closes portable cancellation and peak-copy prerequisites
  only. It does not authorize an internal Qwen load by itself: the signed V3
  record, Models 2.0 descriptor, exact PAIDIR01 v1.1 package, owner-approved
  ARM64 identity and qualification gates remain independently mandatory.
- Only synthetic signed tiny packages are qualified. No production model,
  license/provenance review, tokenizer conformance corpus, long-context quality,
  or generated-output acceptance evidence exists.
- Scalar CPU is the correctness oracle. ARM64 and Metal operation tests pass
  documented synthetic tolerances. Both now run complete signed synthetic-model
  generation through explicit owner-selected UDS execution, and Go pins the
  signed descriptor. ARM64 currently materializes bounded intermediates and KV
  history around the kernel interface. Its AIPQ8/AIPQ4 matvec decodes fixed
  blocks without materializing a full F32 weight tensor, but the ordinary
  service loader still materializes quantized model weights. Qwen split-half
  RoPE remains on an exact scalar correctness path when ARM64 is selected; this
  is not a qualified optimized Qwen backend. Production-size canaries and operated
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
- A first-party sparse-expert plan/cache/router correctness foundation now
  exists, including exact identity, bounded memory, immutable leases,
  deterministic eviction, cancellation-safe depth-one prefetch, and synthetic
  resident-versus-streamed parity. It is deliberately not wired to a production
  architecture, package revision, importer, decoder, service capability, or
  accelerator. Complete shard mappings are still conservatively charged as
  committed bytes, so model-larger-than-RAM admission is not yet claimed.
  Prefetch is globally capped at four explicitly sized worker stacks, but the
  stacks remain OS-managed rather than `MemoryPool` allocations. Production
  release builds use panic-abort, so trusted source/work callbacks are no-panic
  boundaries rather than process-isolated plugins.
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
