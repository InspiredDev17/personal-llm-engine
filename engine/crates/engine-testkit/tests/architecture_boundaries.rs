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
fn internal_crate_dependencies_match_the_approved_dag() {
    let allowed: [(&str, &[&str]); 16] = [
        ("engine-types", &[]),
        ("engine-crypto", &["personalai-engine-types"]),
        ("engine-model-format", &["personalai-engine-types"]),
        (
            "engine-loader",
            &[
                "personalai-engine-crypto",
                "personalai-engine-memory",
                "personalai-engine-model-format",
                "personalai-engine-tensor",
                "personalai-engine-types",
            ],
        ),
        ("engine-tokenizer", &["personalai-engine-memory", "personalai-engine-types"]),
        ("engine-tensor", &["personalai-engine-memory", "personalai-engine-types"]),
        (
            "engine-cpu-reference",
            &["personalai-engine-memory", "personalai-engine-tensor", "personalai-engine-types"],
        ),
        ("engine-memory", &["personalai-engine-types"]),
        ("engine-kv-cache", &["personalai-engine-memory", "personalai-engine-types"]),
        (
            "engine-transformer",
            &[
                "personalai-engine-cpu-reference",
                "personalai-engine-kv-cache",
                "personalai-engine-memory",
                "personalai-engine-model-format",
                "personalai-engine-tensor",
                "personalai-engine-types",
            ],
        ),
        ("engine-sampling", &["personalai-engine-tensor", "personalai-engine-types"]),
        ("engine-scheduler", &["personalai-engine-types"]),
        ("engine-protocol", &["personalai-engine-crypto", "personalai-engine-types"]),
        (
            "engine-server",
            &[
                "personalai-engine-crypto",
                "personalai-engine-loader",
                "personalai-engine-memory",
                "personalai-engine-protocol",
                "personalai-engine-sampling",
                "personalai-engine-scheduler",
                "personalai-engine-tokenizer",
                "personalai-engine-transformer",
                "personalai-engine-types",
            ],
        ),
        (
            "engine-cli",
            &[
                "personalai-engine-crypto",
                "personalai-engine-scheduler",
                "personalai-engine-server",
                "personalai-engine-types",
            ],
        ),
        (
            "engine-testkit",
            &[
                "personalai-engine-crypto",
                "personalai-engine-model-format",
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
fn rust_sources_have_only_approved_listener_and_unsafe_ffi_boundaries() {
    let root = engine_root();
    let listener_marker = ["Tcp", "Listener"].concat();
    let unix_listener_marker = ["Unix", "Listener"].concat();
    let unix_datagram_marker = ["Unix", "Datagram"].concat();
    let network_module_marker = ["std::", "net"].concat();
    let unsafe_block_marker = ["unsafe", " {"].concat();
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

    for path in collect_files(&root, ".rs") {
        let canonical = path.canonicalize().expect("source path must canonicalize");
        if canonical == this_file {
            continue;
        }
        let source = fs::read_to_string(&path).expect("Rust source must be readable");
        let production_source = source
            .split_once("#[cfg(test)]")
            .map_or(source.as_str(), |(production, _tests)| production);
        if canonical == approved_unix_lock {
            assert_eq!(
                source.matches(unsafe_block_marker.as_str()).count(),
                1,
                "approved Unix lock FFI must contain exactly one unsafe block"
            );
            assert_eq!(
                source.matches(unsafe_allow_marker.as_str()).count(),
                1,
                "approved Unix lock FFI must contain exactly one unsafe allowance"
            );
        }
        for marker in [
            listener_marker.as_str(),
            unix_listener_marker.as_str(),
            unix_datagram_marker.as_str(),
            network_module_marker.as_str(),
            unsafe_block_marker.as_str(),
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
            assert!(
                !source.contains(marker),
                "prohibited Rust source marker {marker:?} found in {}",
                path.display()
            );
        }
        assert_no_outbound_or_process_egress(&source, production_source, &path);
    }
    assert!(collect_files(&root, "build.rs").is_empty());
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
