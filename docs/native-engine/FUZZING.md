# Defensive Fuzzing

## Scope and boundary

The fuzz suite tests only repository-owned, data-only parsing boundaries. It
does not scan a host or network, execute a model, generate an exploit, access a
credential, open a listener, or weaken authentication. A finding means that a
local parser needs defensive correction; it is not by itself evidence of an
exploitable vulnerability.

`engine/fuzz` is an excluded Cargo workspace. Nothing in it is linked into the
production engine, included in `engine/Cargo.lock`, or represented as a release
dependency. Its independent lockfile exists only to make fuzz tooling
reproducible.

## Rust targets

| Target | Boundary | Per-input ceiling |
| --- | --- | ---: |
| `model_package` | complete canonical `ParsedModelPackage` parser | 8 MiB campaign input |
| `tokenizer_data` | retained first-party AIPTok parser compatibility target | 8 MiB input and 128 MiB accounting pool |
| `aiptok_parser` | AIPTok Byte-Level BPE v1 parser | 8 MiB input and 128 MiB accounting pool |
| `aiptok_encode` | arbitrary byte-oriented encode input | 64 KiB input |
| `aiptok_decode` | arbitrary little-endian token-ID sequence | 256 KiB input |
| `aiptok_incremental` | arbitrary incremental decoder token state | 256 KiB input |
| `protocol_frame` | inbound and outbound directional frame decoders | engine 8 MiB frame ceiling |
| `service_envelope` | correctly signed request/response envelopes plus closed message-body decoders | 64 KiB body plus two harness selector bytes |
| `manifest_parser` | directory-backed `.paimodel` manifest parser | 8 MiB campaign input; parser hard cap 64 MiB |
| `tensor_index_parser` | one fixed directory tensor-index record | 192 bytes |
| `safetensors_import` | offline importer's bounded safetensors parser | 8 MiB campaign input |
| `package_path_resolution` | safe direct-child `.paimodel` name parser | 1,024 bytes |
| `model_registry_v2` | canonical signed Model Foundry registry V2 parser | 64 KiB |
| `configuration_importer` | strict offline `AIPTOK-METADATA-V1` conversion | 256 KiB campaign input; parser hard cap 64 MiB |
| `generation_request` | correctly signed generation envelope and request fields | 256 KiB campaign prompt |
| `sampling_parameters` | sampler parameter validation and bounded token selection | 4 KiB campaign input; 128-token harness vocabulary |

The input ceiling is a campaign resource bound, not a reduction of the parser's
documented format limit. Deterministic unit and mutation tests continue to
exercise declared hard-limit behavior.

The corpus generator is Rust-only and additionally produces valid directory
manifest/tensor-record/F32-F16-BF16 safetensors/package-name seeds plus
malformed boundary seeds. It retains the decoder package, tokenizer, and frame
seeds. The decoder seed is generated from the current two-layer
grouped-query-attention fixture, so corpus verification catches stale fixture
bytes:

```sh
cd /tmp
cargo +1.85.0 run \
  --manifest-path "$REPOSITORY/engine/fuzz/Cargo.toml" \
  --locked --bin generate-corpus
```

Run a bounded local smoke campaign from outside `engine` so the production
workspace's intentionally offline Cargo configuration remains authoritative:

```sh
cd /tmp
cargo +nightly-2026-08-01 fuzz run \
  --fuzz-dir "$REPOSITORY/engine/fuzz" model_package \
  "$REPOSITORY/engine/fuzz/corpus/model_package" -- \
  -runs=1000 -max_len=8388608 -timeout=10 -rss_limit_mb=1024
```

Replace the target and corpus directory for any table entry and use its exact
table ceiling (`max_len` is 192 for `tensor_index_parser`, 1,024 for
`package_path_resolution`, 65,536 for `aiptok_encode`, or 262,144 for either
AIPTok token-sequence target; `service_envelope` uses 65,538 and
`model_registry_v2` uses 65,536). Copy a
checked-in corpus to a temporary directory when the run must not retain newly
discovered coverage inputs.

The `service_envelope` target received a bounded local macOS ARM64 smoke run on
2026-08-12: an initial 1,000-run campaign and a separate 10-second campaign.
The timed campaign completed 32,451 inputs in 11 wall-clock seconds, reached
1,248 coverage counters and 1,874 features, stayed below 123 MiB RSS, and
produced no crash, sanitizer, timeout, or resource-limit artifact. Local `atos`
symbolizer startup warnings did not stop the campaign. This is smoke evidence,
not a claim that scheduled or production-size fuzzing is complete.

After adding the closed `ModelStatus` body, the deterministic corpus gained one
valid request and one valid response seed. A separate five-second macOS ARM64
campaign on 2026-08-13 completed 16,092 inputs in six wall-clock seconds,
reached 1,269 coverage counters and 1,777 features, stayed below 82 MiB RSS,
and produced no crash, sanitizer, timeout, or resource-limit artifact. The same
non-fatal local `atos` symbolizer warning occurred.

The new `model_registry_v2` target received a separate bounded 1,000-run macOS
ARM64 smoke campaign on 2026-08-13 using a temporary copy of its deterministic
corpus. It reached 393 coverage counters and 731 features, stayed below 43 MiB
RSS, and produced no crash, sanitizer, timeout, or resource-limit artifact.
Non-fatal local `atos` symbolizer warnings did not stop the campaign. This is
smoke evidence only, not a completeness claim.

After adding valid F16 and BF16 source coverage, the `safetensors_import`
target received a separate bounded 1,000-run macOS ARM64 smoke campaign on
2026-08-13 using a temporary corpus copy. It reached 329 coverage counters and
460 features, stayed at or below 42 MiB RSS, and produced no crash, sanitizer,
timeout, or resource-limit artifact. Non-fatal local `atos` symbolizer warnings
did not stop the campaign. This is smoke evidence only, not a completeness
claim.

On 2026-08-13, all nine release-requested boundaries (`manifest_parser`,
`tensor_index_parser`, `aiptok_parser`, `safetensors_import`,
`configuration_importer`, `protocol_frame`, `generation_request`,
`sampling_parameters`, and `model_registry_v2`) received separate five-second
macOS ARM64 campaigns from deterministic valid and malformed seeds. Each target
reported `Done` after six wall-clock seconds; the combined command exited zero
with no crash, timeout, sanitizer, or resource-limit artifact. Local `atos`
symbolizer startup warnings were non-fatal. The command was:

```sh
for target in manifest_parser tensor_index_parser aiptok_parser \
  safetensors_import configuration_importer protocol_frame \
  generation_request sampling_parameters model_registry_v2; do
  cargo +nightly-2026-08-01 fuzz run --fuzz-dir engine/fuzz "$target" \
    "engine/fuzz/corpus/$target" -- \
    -max_total_time=5 -timeout=5 -rss_limit_mb=1024 || exit 1
done
```

The generation harness was then corrected to carry a valid model ID so valid
requests reach prompt parsing, and rerun against a temporary corpus: 11,658
inputs, 845 coverage counters, 923 features, 76 MiB RSS, and no crash or
resource artifact in six seconds. The resulting 60 seconds of aggregate target
runtime is smoke evidence, not evidence of parser exhaustion or long-duration
coverage stability.

## Continuous campaign

`.github/workflows/fuzz.yml` runs weekly and on explicit dispatch. Each target
receives 60 seconds, its table-specific maximum input, a 10-second single-input
timeout, a 1 GiB libFuzzer RSS limit, and a 20-minute job limit. The workflow verifies
the isolated lockfile, regenerates the seeds, rejects corpus drift, uses a
temporary mutable corpus, and retains crash artifacts for 14 days.

Any crash, sanitizer finding, timeout, or resource-limit finding must be
reproduced locally from the exact artifact before a fix. Minimized regression
bytes belong in the narrow owning crate's deterministic tests unless they
contain sensitive or production model data. Production packages, prompts,
generated text, credentials, host paths, and secrets must never enter a corpus
or CI artifact.
