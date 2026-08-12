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
| `tokenizer_data` | first-party `PAIBPE01` parser | 8 MiB input and 128 MiB accounting pool |
| `protocol_frame` | inbound and outbound directional frame decoders | engine 8 MiB frame ceiling |
| `manifest_parser` | directory-backed `.paimodel` manifest parser | 8 MiB campaign input; parser hard cap 64 MiB |
| `tensor_index_parser` | one fixed directory tensor-index record | 192 bytes |
| `safetensors_import` | offline importer's bounded safetensors parser | 8 MiB campaign input |
| `package_path_resolution` | safe direct-child `.paimodel` name parser | 1,024 bytes |

The input ceiling is a campaign resource bound, not a reduction of the parser's
documented format limit. Deterministic unit and mutation tests continue to
exercise declared hard-limit behavior.

The corpus generator is Rust-only and additionally produces valid directory
manifest/tensor-record/safetensors/package-name seeds plus malformed boundary
seeds. It retains the original decoder package, tokenizer, and frame seeds:

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

Replace the target and corpus directory for any table entry and lower `max_len`
to 192 for `tensor_index_parser` or 1,024 for `package_path_resolution`. Copy a checked-in corpus to a temporary directory when the
run must not retain newly discovered coverage inputs.

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
