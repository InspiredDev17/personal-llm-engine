//! Deterministic local native bridge and embedded metallib build.

use sha2::{Digest, Sha384};
use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rustc-check-cfg=cfg(personalai_metal_native)");
    let manifest = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").expect("manifest directory"));
    let native = manifest.join("../../native/apple/metal");
    for path in [
        "include/native_metal_bridge.h",
        "src/native_metal_bridge.mm",
        "shaders/all.metal",
        "shaders/common.metal",
        "shaders/elementwise.metal",
        "shaders/matmul.metal",
        "shaders/norm.metal",
        "shaders/softmax.metal",
        "shaders/activation.metal",
        "shaders/rope.metal",
        "shaders/attention.metal",
        "shaders/quantized.metal",
    ] {
        println!("cargo:rerun-if-changed={}", native.join(path).display());
    }
    if env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("macos") {
        return;
    }
    let output = PathBuf::from(env::var_os("OUT_DIR").expect("output directory"));
    let toolchain = metal_toolchain();
    let metal = toolchain.join("Metal.xctoolchain/usr/bin/metal");
    let metallib = toolchain.join("Metal.xctoolchain/usr/bin/metallib");
    let air = output.join("personalai.air");
    let library = output.join("personalai.metallib");
    let module_cache = output.join("module-cache");
    fs::create_dir_all(&module_cache).expect("native compiler module cache must create");
    run(
        Command::new(&metal)
            .arg("-c")
            .arg("-std=metal3.0")
            .arg(format!("-fmodules-cache-path={}", module_cache.display()))
            .arg("-I")
            .arg(native.join("shaders"))
            .arg(native.join("shaders/all.metal"))
            .arg("-o")
            .arg(&air),
        "Metal shader compilation",
    );
    run(Command::new(&metallib).arg(&air).arg("-o").arg(&library), "metallib linking");
    let bytes = fs::read(&library).expect("compiled metallib must be readable");
    fs::write(output.join("personalai.metallib.sha384"), Sha384::digest(&bytes))
        .expect("metallib digest must write");

    let object = output.join("native_metal_bridge.o");
    run(
        Command::new("xcrun")
            .args(["--sdk", "macosx", "clang++", "-std=c++17", "-fobjc-arc", "-fno-exceptions"])
            .args(["-Wall", "-Wextra", "-Werror"])
            .arg("-mmacosx-version-min=11.0")
            .arg(format!("-fmodules-cache-path={}", module_cache.display()))
            .arg("-I")
            .arg(native.join("include"))
            .arg("-c")
            .arg(native.join("src/native_metal_bridge.mm"))
            .arg("-o")
            .arg(&object),
        "Objective-C++ bridge compilation",
    );
    let archive = output.join("libpersonalai_metal_native.a");
    run(
        Command::new("xcrun").args(["ar", "crs"]).arg(&archive).arg(&object),
        "native bridge archiving",
    );
    println!("cargo:rustc-cfg=personalai_metal_native");
    println!("cargo:rustc-link-search=native={}", output.display());
    println!("cargo:rustc-link-lib=static=personalai_metal_native");
    println!("cargo:rustc-link-lib=c++");
    println!("cargo:rustc-link-lib=framework=Metal");
    println!("cargo:rustc-link-lib=framework=Foundation");
}

fn metal_toolchain() -> PathBuf {
    if let Some(path) = env::var_os("PERSONALAI_METAL_TOOLCHAIN") {
        return PathBuf::from(path);
    }
    if let Ok(entries) = fs::read_dir("/private/var/run/com.apple.security.cryptexd/mnt") {
        for entry in entries.flatten() {
            let path = entry.path();
            if entry
                .file_name()
                .to_string_lossy()
                .starts_with("com.apple.MobileAsset.MetalToolchain-")
                && path.join("Metal.xctoolchain/usr/bin/metal").is_file()
                && path.join("Metal.xctoolchain/usr/bin/metallib").is_file()
            {
                return path;
            }
        }
    }
    let output = Command::new("xcodebuild")
        .args(["-showComponent", "MetalToolchain", "-json"])
        .output()
        .expect("xcodebuild must report the installed Metal component");
    assert!(output.status.success(), "installed Metal component query failed");
    let text = String::from_utf8(output.stdout).expect("Metal component report must be UTF-8");
    let line = text
        .lines()
        .find(|line| line.contains("toolchainSearchPath"))
        .expect("Metal component report lacks toolchainSearchPath");
    let value =
        line.split('"').nth(3).expect("Metal component path must be quoted").replace("\\/", "/");
    let path = PathBuf::from(value);
    assert!(
        path.is_absolute() && path.join("Metal.xctoolchain/usr/bin/metal").is_file(),
        "Metal component path is invalid"
    );
    path
}

fn run(command: &mut Command, label: &str) {
    let status =
        command.status().unwrap_or_else(|error| panic!("{label} could not start: {error}"));
    assert!(status.success(), "{label} failed with {status}");
}
