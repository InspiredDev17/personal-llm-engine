//! Workspace architecture and supply-chain boundary tests.

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

#[test]
fn workspace_external_dependencies_match_the_cryptography_allowlist() {
    let root = engine_root();
    let lock = fs::read_to_string(root.join("Cargo.lock")).expect("Cargo.lock must exist");
    let expected = [
        "block-buffer@0.12.1",
        "cfg-if@1.0.4",
        "cpufeatures@0.3.0",
        "crypto-common@0.2.2",
        "curve25519-dalek-derive@0.1.1",
        "curve25519-dalek@5.0.0",
        "digest@0.11.3",
        "ed25519-dalek@3.0.0",
        "ed25519@3.0.0",
        "fiat-crypto@0.3.0",
        "hybrid-array@0.4.14",
        "libc@0.2.189",
        "proc-macro2@1.0.107",
        "quote@1.0.47",
        "rustc_version@0.4.1",
        "semver@1.0.28",
        "sha2@0.11.0",
        "signature@3.0.0",
        "subtle@2.6.1",
        "syn@2.0.119",
        "typenum@1.20.1",
        "unicode-ident@1.0.24",
        "zeroize@1.8.2",
    ]
    .into_iter()
    .map(str::to_owned)
    .collect::<BTreeSet<_>>();
    let mut observed = BTreeSet::new();
    for package in lock.split("[[package]]").skip(1) {
        let name = lock_value(package, "name").expect("locked package must have a name");
        let version = lock_value(package, "version").expect("locked package must have a version");
        if let Some(source) = lock_value(package, "source") {
            assert_eq!(
                source, "registry+https://github.com/rust-lang/crates.io-index",
                "external dependency must use the approved registry"
            );
            assert!(
                lock_value(package, "checksum").is_some(),
                "external dependency must have a locked checksum: {name}"
            );
            observed.insert(format!("{name}@{version}"));
        }
    }
    assert_eq!(observed, expected, "resolved external dependency allowlist changed");

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
        let checksum = lock_value(package, "checksum").expect("external package needs checksum");
        expected.insert(name.to_owned());
        let vendored_checksum = fs::read_to_string(vendor.join(name).join(".cargo-checksum.json"))
            .expect("vendored package checksum metadata must exist");
        assert!(
            vendored_checksum.contains(&format!("\"package\": \"{checksum}\""))
                || vendored_checksum.contains(&format!("\"package\":\"{checksum}\"")),
            "vendored archive checksum differs for {name}"
        );
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
            !matches!(extension, "py" | "pyi" | "sage" | "ts" | "tsx"),
            "vendored release graph contains a prohibited language file: {}",
            path.display()
        );
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
        .lines()
        .filter_map(|line| line.strip_prefix("PackageName: "))
        .filter(|name| *name != "personalai-native-engine")
        .map(str::to_owned)
        .collect::<BTreeSet<_>>();
    let mut expected_names = BTreeSet::new();
    for package in lock.split("[[package]]").skip(1) {
        if lock_value(package, "source").is_none() {
            continue;
        }
        let name = lock_value(package, "name").expect("locked package must have a name");
        let checksum = lock_value(package, "checksum").expect("external package needs checksum");
        expected_names.insert(name.to_owned());
        assert!(
            sbom.contains(&format!("PackageChecksum: SHA256: {checksum}")),
            "SPDX document omits checksum for {name}"
        );
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
            !matches!(extension, "py" | "pyi" | "sage" | "ts" | "tsx"),
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
    let protocol_source = fs::read_to_string(root.join("crates/engine-protocol/src/lib.rs"))
        .expect("protocol source");

    assert!(
        !protocol_manifest.contains("personalai-engine-placement"),
        "the wire protocol must not depend on internal physical placement mechanics"
    );
    for prohibited in
        ["ExecutionPlacementPlan", "WeightResidencyMode", "byte_offset", "package_path"]
    {
        assert!(
            !protocol_source.contains(prohibited),
            "the client protocol must not accept internal placement authority {prohibited:?}"
        );
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
            assert!(
                lock_value(package, "checksum").is_some(),
                "external dependency must have a locked checksum: {name}"
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
