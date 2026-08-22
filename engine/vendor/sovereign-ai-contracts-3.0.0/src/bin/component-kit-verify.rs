use sovereign_ai_contracts::component_kit::verify_component_kit;
use std::path::PathBuf;

fn main() {
    let arguments: Vec<_> = std::env::args_os().skip(1).collect();
    if arguments.len() != 4 || arguments[0] != "--kit" || arguments[2] != "--profile" {
        eprintln!("usage: component-kit-verify --kit PATH --profile FILE");
        std::process::exit(2);
    }
    let kit = PathBuf::from(&arguments[1]);
    let profile = PathBuf::from(&arguments[3]);
    let result = verify_component_kit(&kit, &profile).unwrap_or_else(|error| {
        eprintln!("component-kit-verify: {error}");
        std::process::exit(1);
    });
    println!(
        "verified component kit for {} at commit {}: {} schemas, manifest sha256 {}",
        result.component, result.source_commit, result.schema_count, result.manifest_sha256
    );
}
