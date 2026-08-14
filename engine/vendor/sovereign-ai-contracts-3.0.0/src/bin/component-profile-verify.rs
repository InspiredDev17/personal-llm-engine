use sovereign_ai_contracts::component_profile::verify_component_support_profile;
use std::path::PathBuf;

fn main() {
    let arguments: Vec<_> = std::env::args_os().skip(1).collect();
    if arguments.len() != 4 || arguments[0] != "--schema-root" || arguments[2] != "--profile" {
        eprintln!("usage: component-profile-verify --schema-root PATH --profile FILE");
        std::process::exit(2);
    }
    let schema_root = PathBuf::from(&arguments[1]);
    let profile_path = PathBuf::from(&arguments[3]);
    let profile =
        verify_component_support_profile(&schema_root, &profile_path).unwrap_or_else(|error| {
            eprintln!("component-profile-verify: {error}");
            std::process::exit(1);
        });
    match serde_json::to_string(&profile) {
        Ok(encoded) => println!("{encoded}"),
        Err(error) => {
            eprintln!("component-profile-verify: {error}");
            std::process::exit(1);
        }
    }
}
