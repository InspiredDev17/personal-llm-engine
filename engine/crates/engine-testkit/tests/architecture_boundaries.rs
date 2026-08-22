//! Workspace architecture and supply-chain boundary tests.

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

#[test]
#[allow(
    clippy::too_many_lines,
    reason = "the complete exact external dependency and source allowlist is intentionally visible"
)]
fn workspace_external_dependencies_match_the_reviewed_allowlist() {
    let root = engine_root();
    let lock = fs::read_to_string(root.join("Cargo.lock")).expect("Cargo.lock must exist");
    let expected = [
        "block-buffer@0.12.1",
        "block-buffer@0.10.4",
        "cfg-if@1.0.4",
        "cpufeatures@0.2.17",
        "cpufeatures@0.3.0",
        "crypto-common@0.1.7",
        "crypto-common@0.2.2",
        "curve25519-dalek-derive@0.1.1",
        "curve25519-dalek@5.0.0",
        "digest@0.10.7",
        "digest@0.11.3",
        "ed25519-dalek@3.0.0",
        "ed25519@3.0.0",
        "fiat-crypto@0.3.0",
        "generic-array@0.14.7",
        "hybrid-array@0.4.14",
        "itoa@1.0.18",
        "libc@0.2.189",
        "memchr@2.8.3",
        "proc-macro2@1.0.107",
        "quote@1.0.47",
        "rustc_version@0.4.1",
        "semver@1.0.28",
        "serde@1.0.229",
        "serde_core@1.0.229",
        "serde_derive@1.0.229",
        "serde_json@1.0.151",
        "sha2@0.10.9",
        "sha2@0.11.0",
        "signature@3.0.0",
        "sovereign-ai-contracts@3.0.0",
        "subtle@2.6.1",
        "syn@2.0.119",
        "syn@3.0.3",
        "tinyvec@1.12.0",
        "tinyvec_macros@0.1.1",
        "typenum@1.20.1",
        "unicode-general-category@1.1.0",
        "unicode-ident@1.0.24",
        "unicode-normalization@0.1.24",
        "version_check@0.9.5",
        "zeroize@1.8.2",
        "zmij@1.0.23",
    ]
    .into_iter()
    .map(str::to_owned)
    .collect::<BTreeSet<_>>();
    let mut observed = BTreeSet::new();
    for package in lock.split("[[package]]").skip(1) {
        let name = lock_value(package, "name").expect("locked package must have a name");
        let version = lock_value(package, "version").expect("locked package must have a version");
        if let Some(source) = lock_value(package, "source") {
            if name == "sovereign-ai-contracts" {
                assert_eq!(
                    source,
                    "git+https://github.com/InspiredDev17/Sovereign-AI-Contracts.git?rev=2c4a7a6afd5625e7def3afec26f18fedac343d11#2c4a7a6afd5625e7def3afec26f18fedac343d11",
                    "canonical runtime contract dependency must use its exact reviewed commit"
                );
                assert!(
                    lock_value(package, "checksum").is_none(),
                    "git contract source is content-pinned by commit, not a registry checksum"
                );
            } else {
                assert_eq!(
                    source, "registry+https://github.com/rust-lang/crates.io-index",
                    "external dependency must use the approved registry"
                );
                assert!(
                    lock_value(package, "checksum").is_some(),
                    "external dependency must have a locked checksum: {name}"
                );
            }
            observed.insert(format!("{name}@{version}"));
        }
    }
    assert_eq!(observed, expected, "resolved external dependency allowlist changed");

    let workspace_manifest =
        fs::read_to_string(root.join("Cargo.toml")).expect("workspace manifest must exist");
    assert!(
        workspace_manifest.contains(
            "sovereign-ai-contracts = { git = \"https://github.com/InspiredDev17/Sovereign-AI-Contracts.git\", rev = \"2c4a7a6afd5625e7def3afec26f18fedac343d11\", default-features = false, features = [\"runtime-v3\"] }"
        ),
        "canonical contracts must remain exact-commit pinned with only runtime-v3 enabled"
    );

    let prohibited = [
        "ollama",
        "llama",
        "mlx",
        "vllm",
        "ggml",
        "pytorch",
        "tensorflow",
        "jax",
        "onnx",
        "candle",
        "burn",
        "ndarray",
        "tokenizers",
        "libloading",
        "reqwest",
        "hyper",
        "curl",
    ];
    for path in collect_files(&root, "Cargo.toml") {
        let manifest = fs::read_to_string(&path).expect("manifest must be readable");
        let lowercase = manifest.to_ascii_lowercase();
        for dependency in prohibited {
            assert!(
                !lowercase.contains(dependency),
                "prohibited dependency marker {dependency:?} found in {}",
                path.display()
            );
        }
    }
}

#[test]
fn unicode_16_dependencies_are_exact_and_confined_to_the_private_qwen_adapter() {
    let root = engine_root();
    let workspace_manifest =
        fs::read_to_string(root.join("Cargo.toml")).expect("workspace manifest must exist");
    for required in [
        "unicode-general-category = { version = \"=1.1.0\", default-features = false }",
        "unicode-normalization = { version = \"=0.1.24\", default-features = false }",
    ] {
        assert!(
            workspace_manifest.contains(required),
            "Unicode dependency declaration changed: {required}"
        );
    }

    let crates = root.join("crates");
    for manifest_path in collect_files(&crates, "Cargo.toml") {
        let manifest = fs::read_to_string(&manifest_path).expect("manifest must be readable");
        let is_tokenizer = manifest_path == crates.join("engine-tokenizer/Cargo.toml");
        for dependency in ["unicode-general-category", "unicode-normalization"] {
            assert_eq!(
                manifest.contains(&format!("{dependency}.workspace = true")),
                is_tokenizer,
                "only engine-tokenizer may directly consume {dependency}: {}",
                manifest_path.display()
            );
        }
    }

    let adapter_path = crates.join("engine-tokenizer/src/qwen_runtime.rs");
    for crate_entry in fs::read_dir(&crates).expect("crate directory must be readable") {
        let source_root = crate_entry.expect("crate entry").path().join("src");
        for source_path in collect_files(&source_root, "rs") {
            let source = fs::read_to_string(&source_path).expect("source must be readable");
            let is_private_adapter = source_path == adapter_path;
            for marker in ["unicode_general_category", "unicode_normalization"] {
                assert_eq!(
                    source.contains(marker),
                    is_private_adapter,
                    "Unicode runtime authority must remain confined to the private Qwen adapter: {}",
                    source_path.display()
                );
            }
        }
    }

    let adapter = fs::read_to_string(&adapter_path).expect("private adapter source must exist");
    for required in [
        "use unicode_general_category::{GeneralCategory, get_general_category};",
        "use unicode_normalization::UnicodeNormalization;",
        "unicode_normalization::UNICODE_VERSION == REQUIRED_UNICODE_VERSION",
        "unicode_general_category::UNICODE_VERSION == required_category_version",
        "const MAX_NFC_COMPATIBILITY_NONSTARTERS: usize = 30;",
        "const MAX_NFC_CANONICAL_DECOMPOSITION_SCALARS: usize = 4;",
        "const NFC_DECOMPOSITION_INLINE_SCALARS: usize = 64;",
        "const NFC_RECOMPOSITION_INLINE_SCALARS: usize = 32;",
        "const NFC_ITERATOR_AND_SORT_WORKSPACE_BYTES: u64 = 4 * 1024;",
        "input.chars().stream_safe().eq(input.chars())",
        ".reserve(normalization_reservation_bytes(normalized_limit)?)?;",
    ] {
        assert!(adapter.contains(required), "private adapter boundary changed: {required}");
    }

    let tokenizer_root = fs::read_to_string(crates.join("engine-tokenizer/src/lib.rs"))
        .expect("tokenizer root must be readable");
    assert!(tokenizer_root.contains("mod qwen_runtime;"));
    assert!(!tokenizer_root.contains("pub mod qwen_runtime"));
    assert!(!tokenizer_root.contains("pub use qwen_runtime"));
    assert!(tokenizer_root.contains("let _ = inspect_qwen25_aiptok(data, pool)?;"));
    assert!(tokenizer_root.contains(
        "AIPTok v2/Qwen2.5 execution remains disabled until exact source conversion and production-table token-ID conformance are qualified"
    ));

    let fuzz_lock = fs::read_to_string(root.join("fuzz/Cargo.lock")).expect("fuzz lock must exist");
    let tokenizer_package = fuzz_lock
        .split("[[package]]")
        .find(|package| {
            lock_value(package, "name").is_some_and(|name| name == "personalai-engine-tokenizer")
        })
        .expect("fuzz lock must contain engine-tokenizer");
    for dependency in ["\"unicode-general-category\"", "\"unicode-normalization\""] {
        assert!(
            tokenizer_package.contains(dependency),
            "fuzz lock tokenizer edge omitted {dependency}"
        );
    }
    for required in [
        "name = \"tinyvec\"\nversion = \"1.12.0\"",
        "name = \"tinyvec_macros\"\nversion = \"0.1.1\"",
        "name = \"unicode-general-category\"\nversion = \"1.1.0\"",
        "name = \"unicode-normalization\"\nversion = \"0.1.24\"",
    ] {
        assert!(fuzz_lock.contains(required), "fuzz lock omitted {required}");
    }
}

#[test]
fn reviewed_unicode_16_vendor_boundaries_remain_pinned() {
    let vendor = engine_root().join("vendor");
    let normalization = vendor.join("unicode-normalization-0.1.24");
    let category = vendor.join("unicode-general-category-1.1.0");
    let tinyvec = vendor.join("tinyvec-1.12.0");
    let macros = vendor.join("tinyvec_macros-0.1.1");

    let normalization_tables =
        fs::read_to_string(normalization.join("src/tables.rs")).expect("NFC tables must exist");
    assert!(normalization_tables.contains("UNICODE_VERSION: (u8, u8, u8) = (16, 0, 0)"));
    let category_source =
        fs::read_to_string(category.join("src/lib.rs")).expect("category source must exist");
    assert!(category_source.contains("UNICODE_VERSION: (u64, u64, u64) = (16, 0, 0)"));

    let normalize_source =
        fs::read_to_string(normalization.join("src/normalize.rs")).expect("normalizer source");
    assert_eq!(normalize_source.matches("#[allow(unsafe_code").count(), 3);
    assert_eq!(normalize_source.matches("unsafe fn").count(), 1);
    assert_eq!(normalize_source.matches("unsafe {").count(), 4);
    assert_eq!(normalize_source.matches("from_u32_unchecked").count(), 5);
    assert!(normalize_source.contains("is_hangul_syllable(c)"));

    let decomposition_source =
        fs::read_to_string(normalization.join("src/decompose.rs")).expect("decomposer source");
    assert!(decomposition_source.contains("buffer: TinyVec<[(u8, char); 64]>,"));
    assert!(decomposition_source.contains("while cursor > 0"));
    assert!(decomposition_source.contains("pending.swap(cursor - 1, cursor)"));
    assert!(!decomposition_source.contains("sort_by_key"));
    let recomposition_source =
        fs::read_to_string(normalization.join("src/recompose.rs")).expect("recomposer source");
    assert!(recomposition_source.contains("buffer: TinyVec<[char; 32]>,"));
    let stream_safe_source =
        fs::read_to_string(normalization.join("src/stream_safe.rs")).expect("stream-safe source");
    assert!(stream_safe_source.contains("pub(crate) const MAX_NONSTARTERS: usize = 30;"));
    let normalization_checksums = fs::read_to_string(normalization.join(".cargo-checksum.json"))
        .expect("normalizer retained-file checksums must exist");
    for required in [
        "\"src/decompose.rs\": \"06599747369f758a907d828712bd81ec3d7607c3ed6050283b844ea6700c105c\"",
        "\"src/recompose.rs\": \"c1b489f6402b96d3b6b55d7ccd08a22d066fc66b7c14331cf579d4c2d10bb963\"",
    ] {
        assert!(
            normalization_checksums.contains(required),
            "normalizer hardening checksum drifted: {required}"
        );
    }

    let normalization_manifest =
        fs::read_to_string(normalization.join("Cargo.toml")).expect("normalizer manifest");
    assert!(normalization_manifest.contains("default = [\"std\"]"));
    assert!(normalization_manifest.contains("features = [\"alloc\"]"));
    let tinyvec_manifest =
        fs::read_to_string(tinyvec.join("Cargo.toml")).expect("tinyvec manifest");
    assert!(tinyvec_manifest.contains("alloc = [\"tinyvec_macros\"]"));
    assert!(tinyvec_manifest.contains("default = []"));

    for safe_crate in [&tinyvec, &macros] {
        let lib = fs::read_to_string(safe_crate.join("src/lib.rs")).expect("safe crate source");
        assert!(lib.contains("#![forbid(unsafe_code)]"));
        for source_path in collect_files(&safe_crate.join("src"), "rs") {
            let source = fs::read_to_string(&source_path).expect("safe crate source");
            assert!(!source.contains("unsafe {"));
            assert!(!source.contains("unsafe fn"));
        }
    }

    let build = fs::read_to_string(category.join("build.rs")).expect("reviewed build script");
    for required in ["env::var(\"OUT_DIR\")", "compile_table()", "File::create(path)"] {
        assert!(build.contains(required), "category build script drifted: {required}");
    }
    for prohibited in [
        "std::process",
        "Command::",
        "std::net",
        "TcpStream",
        "UdpSocket",
        "curl",
        "wget",
        "unsafe",
    ] {
        assert!(
            !build.contains(prohibited),
            "category build script gained prohibited capability {prohibited:?}"
        );
    }

    assert!(!normalization.join("scripts/unicode.py").try_exists().expect("generator path"));
    assert!(!category.join("Makefile").try_exists().expect("generator path"));
    let normalization_checksums = fs::read_to_string(normalization.join(".cargo-checksum.json"))
        .expect("normalizer checksum map");
    assert!(!normalization_checksums.contains("scripts/unicode.py"));
    assert!(
        normalization_checksums
            .contains("69a0faad67afe03146d80ff5b2d889d1898cbfe8f7677a24f012148f6fd27348")
    );
    let category_checksums =
        fs::read_to_string(category.join(".cargo-checksum.json")).expect("category checksum map");
    assert!(!category_checksums.contains("Makefile"));
    assert!(
        category_checksums
            .contains("a877183b60fdc9846063f7ed5a1dc23f8ef26e1ff4994bc4cfc7418ddb4ec48e")
    );
}

#[test]
fn fuzz_tooling_is_isolated_from_the_production_release_graph() {
    let root = engine_root();
    let workspace_manifest =
        fs::read_to_string(root.join("Cargo.toml")).expect("workspace manifest must exist");
    assert!(
        workspace_manifest.contains("exclude = [\"fuzz\"]"),
        "the fuzz project must remain outside the production workspace"
    );

    let production_lock =
        fs::read_to_string(root.join("Cargo.lock")).expect("production lock must exist");
    assert!(
        !production_lock.contains("name = \"libfuzzer-sys\""),
        "the fuzz runtime must never enter the production lockfile"
    );

    let fuzz_root = root.join("fuzz");
    let fuzz_manifest =
        fs::read_to_string(fuzz_root.join("Cargo.toml")).expect("fuzz manifest must exist");
    assert!(fuzz_manifest.contains("[workspace]"));
    assert!(fuzz_manifest.contains("cargo-fuzz = true"));
    assert!(fuzz_manifest.contains("version = \"=0.4.13\""));

    let fuzz_lock = fs::read_to_string(fuzz_root.join("Cargo.lock")).expect("fuzz lock must exist");
    let production_packages = locked_external_packages(&production_lock);
    let fuzz_only = locked_external_packages(&fuzz_lock)
        .difference(&production_packages)
        .cloned()
        .collect::<BTreeSet<_>>();
    let expected = [
        "arbitrary@1.4.2",
        "cc@1.4.2",
        "find-msvc-tools@0.1.10",
        "getrandom@0.4.3",
        "jobserver@0.1.35",
        "libfuzzer-sys@0.4.13",
        "r-efi@6.0.0",
        "shlex@2.0.1",
    ]
    .into_iter()
    .map(str::to_owned)
    .collect::<BTreeSet<_>>();
    assert_eq!(fuzz_only, expected, "the reviewed fuzz-only dependency set changed");
}

#[test]
fn defensive_fuzz_workflow_covers_owned_parser_boundaries() {
    let root = engine_root();
    let repository = root.parent().expect("engine must have repository parent");
    let workflow = fs::read_to_string(repository.join(".github/workflows/fuzz.yml"))
        .expect("defensive fuzz workflow must exist");
    for required in [
        "permissions:\n  contents: read",
        "timeout-minutes: 20",
        "nightly-2026-08-01",
        "cargo-fuzz --version 0.13.2 --locked",
        "manifest_parser",
        "tensor_index_parser",
        "safetensors_import",
        "package_path_resolution",
        "aiptok_parser",
        "aiptok_encode",
        "aiptok_decode",
        "aiptok_incremental",
        "service_envelope",
        "model_registry_v2",
        "configuration_importer",
        "generation_request",
        "sampling_parameters",
        "release_manifest",
        "-max_total_time=60",
        "-max_len=192",
        "-max_len=1024",
        "-max_len=65536",
        "-max_len=65538",
        "-max_len=262144",
        "-max_len=4096",
        "-timeout=10",
        "-rss_limit_mb=1024",
        "git -C \"$GITHUB_WORKSPACE\" diff --exit-code -- engine/fuzz/corpus",
    ] {
        assert!(workflow.contains(required), "fuzz workflow omitted invariant {required:?}");
    }
    for prohibited in ["continue-on-error", "pull_request_target", "sudo ", "curl ", "wget "] {
        assert!(
            !workflow.contains(prohibited),
            "fuzz workflow contains prohibited capability {prohibited:?}"
        );
    }
}

#[test]
fn continuous_soak_workflow_is_pinned_bounded_and_fail_closed() {
    let root = engine_root();
    let repository = root.parent().expect("engine must have repository parent");
    let workflow = fs::read_to_string(repository.join(".github/workflows/continuous-soak.yml"))
        .expect("bounded continuous soak workflow must exist");
    for required in [
        "schedule:",
        "workflow_dispatch:",
        "permissions:\n  contents: read",
        "timeout-minutes: 15",
        "actions/checkout@08eba0b27e820071cde6df949e0beb9ba4906955",
        "rustup toolchain install 1.85.0 --profile minimal",
        "for pass in 1 2 3 4 5 6 7 8; do",
        "cargo +1.85.0 test --locked --all-features",
        "tests::bounded_continuous_soak_isolates_delayed_readers_across_restarts",
        "-- --exact",
    ] {
        assert!(workflow.contains(required), "soak workflow omitted invariant {required:?}");
    }
    for prohibited in ["continue-on-error", "pull_request_target", "sudo ", "curl ", "wget "] {
        assert!(
            !workflow.contains(prohibited),
            "soak workflow contains prohibited capability {prohibited:?}"
        );
    }
}

#[test]
fn linux_service_profile_denies_ip_network_and_limits_filesystem_authority() {
    let root = engine_root();
    let repository = root.parent().expect("engine must have repository parent");
    let unit =
        fs::read_to_string(repository.join("ops/native-engine/systemd/personalai-engine.service"))
            .expect("Linux native-engine service unit must exist");
    for required in [
        "User=personalai",
        "Requires=personalai-engine-release-verify.service",
        "After=personalai-engine-release-verify.service",
        "UMask=0077",
        "StandardInput=file:/run/credentials/personalai-engine.service/response.seed",
        "RuntimeDirectoryMode=0700",
        "NoNewPrivileges=yes",
        "ProtectSystem=strict",
        "PrivateDevices=yes",
        "MemoryDenyWriteExecute=yes",
        "CapabilityBoundingSet=\n",
        "RestrictAddressFamilies=AF_UNIX",
        "IPAddressDeny=any",
        "ReadOnlyPaths=/var/lib/personalai-engine/models",
        "ReadOnlyPaths=/usr/lib/personalai-engine/release",
        "ReadWritePaths=/run/personalai-engine",
        "Environment=HOME=/nonexistent",
        "UnsetEnvironment=AWS_ACCESS_KEY_ID",
        "LimitCORE=0",
        "LimitNOFILE=1024",
        "LimitNPROC=128",
        "TasksMax=128",
        "MemoryMax=16G",
        "MemorySwapMax=0",
        "RestrictNamespaces=yes",
    ] {
        assert!(unit.contains(required), "service unit omitted invariant {required:?}");
    }
    for prohibited in ["AF_INET", "http://", "https://", "curl ", "wget ", "/bin/sh"] {
        assert!(
            !unit.contains(prohibited),
            "service unit contains prohibited capability {prohibited:?}"
        );
    }

    assert!(
        unit.contains("ExecStart=/usr/lib/personalai-engine/release/personalai-engine "),
        "service must launch the exact verified release artifact"
    );

    let verifier = fs::read_to_string(
        repository.join("ops/native-engine/systemd/personalai-engine-release-verify.service"),
    )
    .expect("Linux release-verifier service unit must exist");
    for required in [
        "Type=oneshot",
        "RemainAfterExit=yes",
        "User=root",
        "Group=personalai",
        "UMask=0027",
        "Before=personalai-engine.service",
        "PartOf=personalai-engine.service",
        "verify-supervised",
        "--release-root /usr/lib/personalai-engine/release",
        "--ledger-root /var/lib/personalai-engine/release-state",
        "--engine-runtime-root /run/personalai-engine",
        "--security-event-root /var/lib/personalai-engine/release-events",
        "StateDirectoryMode=0750",
        "NoNewPrivileges=yes",
        "PrivateNetwork=yes",
        "RestrictAddressFamilies=none",
        "IPAddressDeny=any",
        "CapabilityBoundingSet=\n",
        "ReadOnlyPaths=/usr/lib/personalai-engine/release",
        "ReadWritePaths=/var/lib/personalai-engine/release-state /var/lib/personalai-engine/release-events",
        "MemoryDenyWriteExecute=yes",
        "MemorySwapMax=0",
    ] {
        assert!(verifier.contains(required), "release verifier omitted invariant {required:?}");
    }
    for prohibited in ["AF_INET", "http://", "https://", "curl ", "wget ", "/bin/sh"] {
        assert!(
            !verifier.contains(prohibited),
            "release verifier contains prohibited capability {prohibited:?}"
        );
    }
}

#[test]
fn macos_service_profile_allows_only_the_configured_unix_socket() {
    let root = engine_root();
    let repository = root.parent().expect("engine must have repository parent");
    let profile =
        fs::read_to_string(repository.join("ops/native-engine/macos/personalai-engine.sb"))
            .expect("macOS native-engine sandbox profile must exist");
    for required in [
        "(deny default)",
        "(define model-root (param \"MODEL_ROOT\"))",
        "(define runtime-root (param \"RUNTIME_ROOT\"))",
        "(define socket-path (param \"SOCKET_PATH\"))",
        "(allow file-write*\n    (subpath runtime-root))",
        "(allow network-bind\n    (local unix-socket",
        "(path-literal \"/private/var/run/personalai-engine/engine.sock\")))",
        "(allow network-outbound\n    (remote unix-socket",
    ] {
        assert!(profile.contains(required), "macOS sandbox omitted invariant {required:?}");
    }
    for prohibited in ["remote ip", "local ip", "network-inbound", "network*"] {
        assert!(
            !profile.contains(prohibited),
            "macOS sandbox contains prohibited network rule {prohibited:?}"
        );
    }
}

#[test]
fn cross_platform_verification_is_pinned_offline_and_fail_closed() {
    let root = engine_root();
    let repository = root.parent().expect("engine must have repository parent");
    let workflow = fs::read_to_string(repository.join(".github/workflows/verify.yml"))
        .expect("cross-platform verification workflow must exist");
    for required in [
        "push:\n    branches:\n      - main\n      - agent/native-inference-engine",
        "pull_request:",
        "workflow_dispatch:",
        "permissions:\n  contents: read",
        "fail-fast: false",
        "timeout-minutes: 30",
        "ubuntu-24.04",
        "macos-14",
        "actions/checkout@08eba0b27e820071cde6df949e0beb9ba4906955",
        "rustup toolchain install 1.85.0 --profile minimal",
        "cargo +1.85.0 fmt --all --check",
        "cargo +1.85.0 metadata --format-version 1 --locked --offline --no-deps",
        "cargo +1.85.0 tree --workspace --all-features --locked --offline -e features",
        "cargo +1.85.0 check --workspace --all-targets --all-features --locked --offline",
        "cargo +1.85.0 build --workspace --all-targets --all-features --locked --offline",
        "cargo +1.85.0 clippy --workspace --all-targets --all-features",
        "--locked --offline -- -D warnings",
        "cargo +1.85.0 test --workspace --all-targets --all-features --locked --offline",
        "cargo +1.85.0 test --workspace --doc --all-features --locked --offline",
        "cargo +1.85.0 doc --workspace --all-features --no-deps --locked --offline",
        "cargo +1.85.0 check --workspace --all-targets --no-default-features",
        "git diff --exit-code",
    ] {
        assert!(workflow.contains(required), "verification workflow omitted {required:?}");
    }
    for prohibited in [
        "continue-on-error",
        "pull_request_target",
        "permissions: write",
        "sudo ",
        "curl ",
        "wget ",
        "cargo install",
    ] {
        assert!(
            !workflow.contains(prohibited),
            "verification workflow contains prohibited capability {prohibited:?}"
        );
    }
    assert_eq!(workflow.matches("uses:").count(), 1, "only pinned checkout is allowed");
}

#[test]
fn vendored_release_graph_matches_lock_and_contains_no_prohibited_languages() {
    let root = engine_root();
    let lock = fs::read_to_string(root.join("Cargo.lock")).expect("Cargo.lock must exist");
    let vendor = root.join("vendor");
    let config = fs::read_to_string(root.join(".cargo/config.toml"))
        .expect("Cargo source configuration must exist");
    assert!(config.contains("replace-with = \"vendored-sources\""));
    assert!(config.contains("directory = \"vendor\""));

    let mut expected = BTreeSet::new();
    for package in lock.split("[[package]]").skip(1) {
        if lock_value(package, "source").is_none() {
            continue;
        }
        let name = lock_value(package, "name").expect("locked package must have a name");
        let version = lock_value(package, "version").expect("locked package must have a version");
        let directory = format!("{name}-{version}");
        expected.insert(directory.clone());
        let vendored_checksum =
            fs::read_to_string(vendor.join(&directory).join(".cargo-checksum.json"))
                .expect("vendored package checksum metadata must exist");
        if let Some(checksum) = lock_value(package, "checksum") {
            assert!(
                vendored_checksum.contains(&format!("\"package\": \"{checksum}\""))
                    || vendored_checksum.contains(&format!("\"package\":\"{checksum}\"")),
                "vendored archive checksum differs for {directory}"
            );
        } else {
            assert!(
                vendored_checksum.contains("\"package\":null")
                    || vendored_checksum.contains("\"package\": null"),
                "vendored git dependency has unexpected package checksum for {directory}"
            );
        }
    }
    let observed = fs::read_dir(&vendor)
        .expect("vendor directory must exist")
        .map(|entry| {
            entry.expect("vendor entry must be readable").file_name().to_string_lossy().into_owned()
        })
        .collect::<BTreeSet<_>>();
    assert_eq!(observed, expected, "vendored package set differs from Cargo.lock");

    for path in collect_all_files(&vendor) {
        let extension = path.extension().and_then(|value| value.to_str()).unwrap_or_default();
        assert!(
            !matches!(
                extension,
                "py" | "pyi" | "sage" | "js" | "jsx" | "mjs" | "cjs" | "ts" | "tsx"
            ),
            "vendored release graph contains a prohibited language file: {}",
            path.display()
        );
    }
}

#[test]
fn every_vendor_checksum_path_exists_in_tracked_release_content() {
    let root = engine_root();
    let repository = root.parent().expect("engine must have repository parent");
    let vendor = root.join("vendor");
    let tracked = tracked_vendor_files(repository);
    let checksum_maps = collect_files(&vendor, ".cargo-checksum.json");
    assert!(!checksum_maps.is_empty(), "vendor checksum maps must exist");

    for checksum_map in checksum_maps {
        let package = checksum_map.parent().expect("checksum map must have package parent");
        let package_relative =
            package.strip_prefix(repository).expect("vendored package must be repository relative");
        let document = fs::read_to_string(&checksum_map).expect("checksum map must be readable");
        for retained_path in cargo_checksum_file_paths(&document) {
            let retained_path = Path::new(retained_path);
            assert!(
                !retained_path.as_os_str().is_empty()
                    && retained_path
                        .components()
                        .all(|component| matches!(component, std::path::Component::Normal(_))),
                "vendor checksum map contains a non-local path: {} in {}",
                retained_path.display(),
                checksum_map.display()
            );
            let retained_file = package.join(retained_path);
            assert!(
                retained_file.is_file(),
                "vendor checksum map references a missing release file: {}",
                retained_file.display()
            );
            if let Some(tracked) = &tracked {
                let repository_relative = package_relative.join(retained_path);
                assert!(
                    tracked.contains(&repository_relative),
                    "vendor checksum map references a file excluded from Git release content: {}",
                    repository_relative.display()
                );
            }
        }
    }
}

#[test]
fn spdx_sbom_covers_every_locked_external_package_and_checksum() {
    let root = engine_root();
    let lock = fs::read_to_string(root.join("Cargo.lock")).expect("Cargo.lock must exist");
    let sbom = fs::read_to_string(
        root.parent()
            .expect("engine must have repository parent")
            .join("release/native-engine.spdx"),
    )
    .expect("release SPDX document must exist");
    assert!(sbom.starts_with("SPDXVersion: SPDX-2.3\n"));
    let observed_names = sbom
        .split("\n\n")
        .filter_map(|block| {
            let name = block.lines().find_map(|line| line.strip_prefix("PackageName: "))?;
            if name == "personalai-native-engine" {
                return None;
            }
            let version = block.lines().find_map(|line| line.strip_prefix("PackageVersion: "))?;
            Some(format!("{name}@{version}"))
        })
        .collect::<BTreeSet<_>>();
    let mut expected_names = BTreeSet::new();
    for package in lock.split("[[package]]").skip(1) {
        if lock_value(package, "source").is_none() {
            continue;
        }
        let name = lock_value(package, "name").expect("locked package must have a name");
        let version = lock_value(package, "version").expect("locked package must have a version");
        expected_names.insert(format!("{name}@{version}"));
        if let Some(checksum) = lock_value(package, "checksum") {
            assert!(
                sbom.contains(&format!("PackageChecksum: SHA256: {checksum}")),
                "SPDX document omits checksum for {name}@{version}"
            );
        } else {
            assert!(
                sbom.contains("PackageDownloadLocation: git+https://github.com/InspiredDev17/Sovereign-AI-Contracts.git@2c4a7a6afd5625e7def3afec26f18fedac343d11"),
                "SPDX document omits the exact canonical contract commit"
            );
        }
    }
    assert_eq!(observed_names, expected_names, "SPDX package set differs from Cargo.lock");
}

#[test]
fn workspace_contains_no_python_or_typescript() {
    let root = engine_root();
    let repository = root.parent().expect("engine must have repository parent");
    for path in collect_all_files(repository) {
        let extension = path.extension().and_then(|value| value.to_str()).unwrap_or_default();
        assert!(
            !matches!(
                extension,
                "py" | "pyi" | "sage" | "js" | "jsx" | "mjs" | "cjs" | "ts" | "tsx"
            ),
            "prohibited language file found: {}",
            path.display()
        );
    }
}

#[test]
#[allow(clippy::too_many_lines)] // The explicit 23-crate DAG is intentionally visible in one test.
fn internal_crate_dependencies_match_the_approved_dag() {
    let allowed: [(&str, &[&str]); 23] = [
        ("engine-types", &[]),
        ("engine-crypto", &["personalai-engine-types"]),
        ("engine-release", &["personalai-engine-crypto", "personalai-engine-types"]),
        ("engine-model-format", &["personalai-engine-types"]),
        (
            "engine-quantization",
            &[
                "personalai-engine-memory",
                "personalai-engine-model-format",
                "personalai-engine-tensor",
                "personalai-engine-types",
            ],
        ),
        (
            "engine-loader",
            &[
                "personalai-engine-crypto",
                "personalai-engine-memory",
                "personalai-engine-model-format",
                "personalai-engine-quantization",
                "personalai-engine-tensor",
                "personalai-engine-types",
            ],
        ),
        (
            "engine-placement",
            &[
                "personalai-engine-crypto",
                "personalai-engine-loader",
                "personalai-engine-memory",
                "personalai-engine-model-format",
                "personalai-engine-types",
            ],
        ),
        (
            "engine-model-importer",
            &[
                "personalai-engine-crypto",
                "personalai-engine-memory",
                "personalai-engine-model-format",
                "personalai-engine-quantization",
                "personalai-engine-tensor",
                "personalai-engine-tokenizer",
                "personalai-engine-types",
            ],
        ),
        (
            "engine-tokenizer",
            &["personalai-engine-crypto", "personalai-engine-memory", "personalai-engine-types"],
        ),
        ("engine-tensor", &["personalai-engine-memory", "personalai-engine-types"]),
        (
            "engine-cpu-reference",
            &["personalai-engine-memory", "personalai-engine-tensor", "personalai-engine-types"],
        ),
        (
            "engine-arm64",
            &[
                "personalai-engine-cpu-reference",
                "personalai-engine-memory",
                "personalai-engine-model-format",
                "personalai-engine-quantization",
                "personalai-engine-tensor",
                "personalai-engine-types",
            ],
        ),
        (
            "engine-metal",
            &[
                "personalai-engine-cpu-reference",
                "personalai-engine-crypto",
                "personalai-engine-kv-cache",
                "personalai-engine-memory",
                "personalai-engine-tensor",
                "personalai-engine-types",
            ],
        ),
        ("engine-memory", &["personalai-engine-types"]),
        ("engine-knowledge-contract", &["personalai-engine-crypto", "personalai-engine-types"]),
        (
            "engine-kv-cache",
            &[
                "personalai-engine-crypto",
                "personalai-engine-knowledge-contract",
                "personalai-engine-memory",
                "personalai-engine-tensor",
                "personalai-engine-types",
            ],
        ),
        (
            "engine-transformer",
            &[
                "personalai-engine-arm64",
                "personalai-engine-cpu-reference",
                "personalai-engine-kv-cache",
                "personalai-engine-knowledge-contract",
                "personalai-engine-loader",
                "personalai-engine-metal",
                "personalai-engine-memory",
                "personalai-engine-model-format",
                "personalai-engine-placement",
                "personalai-engine-tensor",
                "personalai-engine-types",
            ],
        ),
        (
            "engine-sampling",
            &["personalai-engine-memory", "personalai-engine-tensor", "personalai-engine-types"],
        ),
        ("engine-scheduler", &["personalai-engine-types"]),
        (
            "engine-protocol",
            &[
                "personalai-engine-crypto",
                "personalai-engine-knowledge-contract",
                "personalai-engine-types",
            ],
        ),
        (
            "engine-server",
            &[
                "personalai-engine-crypto",
                "personalai-engine-kv-cache",
                "personalai-engine-knowledge-contract",
                "personalai-engine-loader",
                "personalai-engine-memory",
                "personalai-engine-placement",
                "personalai-engine-protocol",
                "personalai-engine-sampling",
                "personalai-engine-scheduler",
                "personalai-engine-tensor",
                "personalai-engine-tokenizer",
                "personalai-engine-transformer",
                "personalai-engine-types",
            ],
        ),
        (
            "engine-cli",
            &[
                "personalai-engine-crypto",
                "personalai-engine-placement",
                "personalai-engine-sampling",
                "personalai-engine-scheduler",
                "personalai-engine-server",
                "personalai-engine-types",
            ],
        ),
        (
            "engine-testkit",
            &[
                "personalai-engine-cpu-reference",
                "personalai-engine-crypto",
                "personalai-engine-memory",
                "personalai-engine-model-format",
                "personalai-engine-tensor",
                "personalai-engine-types",
            ],
        ),
    ];
    let crates = engine_root().join("crates");
    for (crate_directory, expected) in allowed {
        let manifest_path = crates.join(crate_directory).join("Cargo.toml");
        let manifest = fs::read_to_string(&manifest_path).expect("manifest must be readable");
        let observed = internal_dependencies(&manifest);
        let expected = expected.iter().copied().map(str::to_owned).collect::<BTreeSet<_>>();
        assert_eq!(
            observed,
            expected,
            "internal dependency boundary changed for {}",
            manifest_path.display()
        );
    }
}

#[test]
fn placement_authority_remains_internal_to_the_trusted_runtime() {
    let root = engine_root();
    let protocol_manifest = fs::read_to_string(root.join("crates/engine-protocol/Cargo.toml"))
        .expect("protocol manifest");
    let protocol_sources = collect_files(&root.join("crates/engine-protocol/src"), "rs");

    assert!(
        !protocol_manifest.contains("personalai-engine-placement"),
        "the wire protocol must not depend on internal physical placement mechanics"
    );
    for prohibited in [
        "ExecutionPlacementPlan",
        "WeightResidencyMode",
        "SparseExpertExecutionPlan",
        "SparseExpertId",
        "SparseExpertSelection",
        "expert_id",
        "expert_ids",
        "expert_offset",
        "expert_offsets",
        "router_selection",
        "router_scores",
        "byte_offset",
        "package_path",
    ] {
        for source in &protocol_sources {
            let contents = fs::read_to_string(source).expect("protocol source must be readable");
            assert!(
                !contents.contains(prohibited),
                "the client protocol must not accept internal placement authority {prohibited:?}: {}",
                source.display()
            );
        }
    }
}

#[test]
fn selectable_token_domain_is_not_client_protocol_authority() {
    let root = engine_root();
    let protocol_root = root.join("crates/engine-protocol");
    let protocol_manifest =
        fs::read_to_string(protocol_root.join("Cargo.toml")).expect("protocol manifest");
    assert!(
        !protocol_manifest.contains("personalai-engine-sampling"),
        "the client protocol must not construct sampling-domain authority"
    );
    for source in collect_files(&protocol_root.join("src"), "rs") {
        let contents = fs::read_to_string(&source).expect("protocol source must be readable");
        for prohibited in ["SelectableTokenDomain", "selectable_token_count", "reserved_token_tail"]
        {
            assert!(
                !contents.contains(prohibited),
                "client protocol source must not expose trusted token-domain field {prohibited:?}: {}",
                source.display()
            );
        }
    }
}

#[test]
fn crate_manifests_cannot_add_native_links_or_weaken_workspace_lints() {
    let crates = engine_root().join("crates");
    for manifest_path in collect_files(&crates, "Cargo.toml") {
        let manifest = fs::read_to_string(&manifest_path).expect("manifest must be readable");
        assert!(
            !manifest.lines().any(|line| line.trim_start().starts_with("links =")),
            "native link discovery is prohibited in {}",
            manifest_path.display()
        );
        assert!(
            manifest.contains("[lints]\nworkspace = true"),
            "crate must inherit the workspace unsafe and warning policy: {}",
            manifest_path.display()
        );
        assert!(
            !manifest.contains("unsafe_code"),
            "crate manifest cannot override the workspace unsafe policy: {}",
            manifest_path.display()
        );
    }
}

#[test]
#[allow(clippy::too_many_lines)]
fn rust_sources_have_only_approved_listener_and_unsafe_ffi_boundaries() {
    let root = engine_root();
    let listener_marker = ["Tcp", "Listener"].concat();
    let unix_listener_marker = ["Unix", "Listener"].concat();
    let unix_datagram_marker = ["Unix", "Datagram"].concat();
    let network_module_marker = ["std::", "net"].concat();
    let unsafe_block_marker = ["unsafe", " {"].concat();
    let unsafe_impl_marker = ["unsafe", " impl"].concat();
    let unsafe_allow_marker = ["allow", "(unsafe_code)"].concat();
    let dynamic_open_marker = ["dl", "open"].concat();
    let source_path = PathBuf::from(file!());
    let source_path = if source_path.is_absolute() { source_path } else { root.join(source_path) };
    let this_file = source_path.canonicalize().expect("test path must canonicalize");
    let approved_unix_transport = root
        .join("crates/engine-server/src/transport.rs")
        .canonicalize()
        .expect("approved Unix transport must exist");
    let approved_unix_lock = root
        .join("crates/engine-server/src/os_lock.rs")
        .canonicalize()
        .expect("approved Unix lock FFI must exist");
    let approved_process_signal = root
        .join("crates/engine-cli/src/os_signal.rs")
        .canonicalize()
        .expect("approved process signal FFI must exist");
    let approved_peer_credential = root
        .join("crates/engine-server/src/peer_credential.rs")
        .canonicalize()
        .expect("approved peer credential FFI must exist");
    let approved_read_only_mapping = root
        .join("crates/engine-loader/src/read_only_map.rs")
        .canonicalize()
        .expect("approved read-only mapping FFI must exist");
    let approved_arm64_intrinsics = root
        .join("crates/engine-arm64/src/intrinsics.rs")
        .canonicalize()
        .expect("approved ARM64 intrinsic boundary must exist");
    let approved_metal_ffi = root
        .join("crates/engine-metal/src/ffi.rs")
        .canonicalize()
        .expect("approved Metal FFI boundary must exist");
    let approved_metal_build = root
        .join("crates/engine-metal/build.rs")
        .canonicalize()
        .expect("approved Metal build script must exist");

    for path in collect_files(&root, ".rs") {
        let canonical = path.canonicalize().expect("source path must canonicalize");
        if canonical == this_file {
            continue;
        }
        let source = fs::read_to_string(&path).expect("Rust source must be readable");
        let production_source = source
            .split_once("#[cfg(test)]")
            .map_or(source.as_str(), |(production, _tests)| production);
        assert_exact_unsafe_boundary(
            &canonical,
            &approved_unix_lock,
            &source,
            &unsafe_block_marker,
            &unsafe_allow_marker,
            1,
            "Unix lock FFI",
        );
        assert_exact_unsafe_boundary(
            &canonical,
            &approved_process_signal,
            &source,
            &unsafe_block_marker,
            &unsafe_allow_marker,
            2,
            "process signal FFI",
        );
        assert_exact_unsafe_boundary(
            &canonical,
            &approved_peer_credential,
            &source,
            &unsafe_block_marker,
            &unsafe_allow_marker,
            4,
            "peer credential FFI",
        );
        assert_exact_unsafe_boundary(
            &canonical,
            &approved_read_only_mapping,
            &source,
            &unsafe_block_marker,
            &unsafe_allow_marker,
            3,
            "read-only model mapping FFI",
        );
        if canonical == approved_read_only_mapping {
            assert_eq!(
                source.matches(&unsafe_impl_marker).count(),
                2,
                "read-only mapping must have exactly the reviewed Send and Sync implementations"
            );
        }
        assert_exact_unsafe_boundary(
            &canonical,
            &approved_arm64_intrinsics,
            &source,
            &unsafe_block_marker,
            &unsafe_allow_marker,
            12,
            "ARM64 Advanced SIMD",
        );
        assert_exact_unsafe_boundary(
            &canonical,
            &approved_metal_ffi,
            &source,
            &unsafe_block_marker,
            &unsafe_allow_marker,
            18,
            "Metal C ABI",
        );
        for marker in [
            listener_marker.as_str(),
            unix_listener_marker.as_str(),
            unix_datagram_marker.as_str(),
            network_module_marker.as_str(),
            unsafe_block_marker.as_str(),
            unsafe_impl_marker.as_str(),
            unsafe_allow_marker.as_str(),
            dynamic_open_marker.as_str(),
        ] {
            if canonical == approved_unix_transport
                && (marker == unix_listener_marker || marker == network_module_marker)
            {
                continue;
            }
            if canonical == approved_unix_lock
                && (marker == unsafe_block_marker || marker == unsafe_allow_marker)
            {
                continue;
            }
            if canonical == approved_process_signal
                && (marker == unsafe_block_marker || marker == unsafe_allow_marker)
            {
                continue;
            }
            if canonical == approved_peer_credential
                && (marker == unsafe_block_marker || marker == unsafe_allow_marker)
            {
                continue;
            }
            if canonical == approved_read_only_mapping
                && (marker == unsafe_block_marker
                    || marker == unsafe_impl_marker
                    || marker == unsafe_allow_marker)
            {
                continue;
            }
            if canonical == approved_arm64_intrinsics
                && (marker == unsafe_block_marker || marker == unsafe_allow_marker)
            {
                continue;
            }
            if canonical == approved_metal_ffi
                && (marker == unsafe_block_marker || marker == unsafe_allow_marker)
            {
                continue;
            }
            assert!(
                !source.contains(marker),
                "prohibited Rust source marker {marker:?} found in {}",
                path.display()
            );
        }
        let production_source =
            if canonical == approved_metal_build { "" } else { production_source };
        assert_no_outbound_or_process_egress(&source, production_source, &path);
    }
}

#[test]
fn workspace_has_only_the_reviewed_local_metal_build_script() {
    assert_eq!(
        collect_files(&engine_root(), "build.rs"),
        [engine_root().join("crates/engine-metal/build.rs")]
    );
}

fn assert_exact_unsafe_boundary(
    source_path: &Path,
    approved_path: &Path,
    source: &str,
    unsafe_block_marker: &str,
    unsafe_allow_marker: &str,
    expected_blocks: usize,
    label: &str,
) {
    if source_path != approved_path {
        return;
    }
    assert_eq!(
        source.matches(unsafe_block_marker).count(),
        expected_blocks,
        "approved {label} has an unexpected unsafe block count"
    );
    assert_eq!(
        source.matches(unsafe_allow_marker).count(),
        1,
        "approved {label} must contain exactly one unsafe allowance"
    );
}

fn assert_no_outbound_or_process_egress(source: &str, production_source: &str, path: &Path) {
    let network_markers = [
        ["Tcp", "Stream"].concat(),
        ["Udp", "Socket"].concat(),
        ["Socket", "Addr"].concat(),
        ["To", "Socket", "Addrs"].concat(),
        ["tokio::", "net"].concat(),
        ["async_std::", "net"].concat(),
        ["smol::", "net"].concat(),
        ["mio::", "net"].concat(),
        ["libc::", "socket"].concat(),
        ["libc::", "connect"].concat(),
        ["AF_", "INET"].concat(),
        ["SOCK_", "RAW"].concat(),
        ["SYS_", "socket"].concat(),
    ];
    for marker in &network_markers {
        assert!(
            !source.contains(marker),
            "prohibited outbound egress marker {marker:?} found in {}",
            path.display()
        );
    }

    let process_markers = [
        ["process::", "Command"].concat(),
        ["Command", "::new"].concat(),
        ["libc::", "exec"].concat(),
        ["posix_", "spawn"].concat(),
    ];
    for marker in &process_markers {
        assert!(
            !production_source.contains(marker),
            "prohibited production process egress marker {marker:?} found in {}",
            path.display()
        );
    }
}

fn internal_dependencies(manifest: &str) -> BTreeSet<String> {
    let mut dependencies = BTreeSet::new();
    let mut in_dependencies = false;
    for line in manifest.lines() {
        let line = line.trim();
        if line.starts_with('[') {
            in_dependencies = line == "[dependencies]";
            continue;
        }
        if !in_dependencies || line.is_empty() || line.starts_with('#') {
            continue;
        }
        let Some((name, _value)) = line.split_once('=') else {
            continue;
        };
        let name = name.trim().strip_suffix(".workspace").unwrap_or(name.trim());
        if name.starts_with("personalai-engine-") {
            dependencies.insert(name.to_owned());
        }
    }
    dependencies
}

fn lock_value<'a>(package: &'a str, key: &str) -> Option<&'a str> {
    package.lines().find_map(|line| {
        let (candidate, value) = line.split_once('=')?;
        if candidate.trim() != key {
            return None;
        }
        Some(value.trim().trim_matches('"'))
    })
}

fn locked_external_packages(lock: &str) -> BTreeSet<String> {
    lock.split("[[package]]")
        .skip(1)
        .filter(|package| lock_value(package, "source").is_some())
        .map(|package| {
            let name = lock_value(package, "name").expect("locked package must have a name");
            let version =
                lock_value(package, "version").expect("locked package must have a version");
            let source = lock_value(package, "source").expect("external package source");
            assert!(
                lock_value(package, "checksum").is_some()
                    || (name == "sovereign-ai-contracts"
                        && source
                            == "git+https://github.com/InspiredDev17/Sovereign-AI-Contracts.git?rev=2c4a7a6afd5625e7def3afec26f18fedac343d11#2c4a7a6afd5625e7def3afec26f18fedac343d11"),
                "external dependency must have a locked checksum or exact reviewed commit: {name}"
            );
            format!("{name}@{version}")
        })
        .collect()
}

#[test]
fn backend_features_are_explicit_and_cpu_is_default() {
    let manifest = fs::read_to_string(engine_root().join("crates/engine-transformer/Cargo.toml"))
        .expect("transformer manifest must be readable");
    assert!(manifest.contains("default = [\"cpu-reference\"]"));
    for feature in ["cpu-reference", "arm64", "metal", "cuda"] {
        assert!(
            manifest.lines().any(|line| line.trim_start().starts_with(feature)),
            "missing backend feature {feature}"
        );
    }
}

fn engine_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("testkit must be nested under engine/crates")
        .to_path_buf()
}

fn collect_files(root: &Path, suffix: &str) -> Vec<PathBuf> {
    collect_all_files(root)
        .into_iter()
        .filter(|path| {
            path.file_name().and_then(|value| value.to_str()) == Some(suffix)
                || path.to_string_lossy().ends_with(suffix)
        })
        .collect()
}

fn collect_all_files(root: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();
    let mut pending = vec![root.to_path_buf()];
    while let Some(directory) = pending.pop() {
        for entry in fs::read_dir(&directory).expect("workspace directory must be readable") {
            let entry = entry.expect("workspace entry must be readable");
            let path = entry.path();
            let name = entry.file_name();
            if entry.file_type().expect("entry type must be readable").is_dir() {
                if name != "target" && name != ".git" && name != "vendor" {
                    pending.push(path);
                }
            } else {
                files.push(path);
            }
        }
    }
    files.sort();
    files
}

fn tracked_vendor_files(repository: &Path) -> Option<BTreeSet<PathBuf>> {
    if !repository.join(".git").try_exists().expect("Git marker must be inspectable") {
        // A source archive contains only tracked Git content, so existence is
        // the equivalent invariant when repository metadata is intentionally absent.
        return None;
    }
    let output = Command::new("git")
        .arg("-C")
        .arg(repository)
        .args(["ls-files", "-z", "--", "engine/vendor"])
        .output()
        .expect("git ls-files must run for the source worktree");
    assert!(output.status.success(), "git ls-files failed for vendored release content");
    let output = String::from_utf8(output.stdout).expect("tracked paths must be UTF-8");
    Some(output.split('\0').filter(|path| !path.is_empty()).map(PathBuf::from).collect())
}

fn cargo_checksum_file_paths(document: &str) -> Vec<&str> {
    let files_marker = document.find("\"files\"").expect("checksum map must contain files");
    let object_offset = document[files_marker..]
        .find('{')
        .map(|offset| files_marker + offset)
        .expect("checksum files value must be an object");
    let bytes = document.as_bytes();
    let mut cursor = object_offset + 1;
    let mut first = true;
    let mut paths = Vec::new();

    loop {
        skip_ascii_whitespace(bytes, &mut cursor);
        assert!(cursor < bytes.len(), "checksum files object must terminate");
        if bytes[cursor] == b'}' {
            break;
        }
        if !first {
            assert_eq!(bytes[cursor], b',', "checksum file entries must be comma separated");
            cursor += 1;
            skip_ascii_whitespace(bytes, &mut cursor);
        }
        first = false;
        assert_eq!(bytes.get(cursor), Some(&b'\"'), "checksum path must be a JSON string");
        cursor += 1;
        let path_start = cursor;
        while bytes.get(cursor).is_some_and(|byte| *byte != b'\"') {
            assert_ne!(bytes[cursor], b'\\', "escaped vendor paths are not admitted");
            cursor += 1;
        }
        assert!(cursor < bytes.len(), "checksum path string must terminate");
        let path = &document[path_start..cursor];
        cursor += 1;
        skip_ascii_whitespace(bytes, &mut cursor);
        assert_eq!(bytes.get(cursor), Some(&b':'), "checksum path must have a value");
        cursor += 1;
        skip_ascii_whitespace(bytes, &mut cursor);
        assert_eq!(bytes.get(cursor), Some(&b'\"'), "file checksum must be a JSON string");
        cursor += 1;
        let hash_start = cursor;
        while bytes.get(cursor).is_some_and(|byte| *byte != b'\"') {
            assert!(bytes[cursor].is_ascii_hexdigit(), "file checksum must be hexadecimal");
            cursor += 1;
        }
        assert_eq!(cursor - hash_start, 64, "file checksum must be SHA-256");
        assert!(cursor < bytes.len(), "file checksum string must terminate");
        cursor += 1;
        paths.push(path);
    }
    paths
}

fn skip_ascii_whitespace(bytes: &[u8], cursor: &mut usize) {
    while bytes.get(*cursor).is_some_and(u8::is_ascii_whitespace) {
        *cursor += 1;
    }
}
