use sovereign_ai_contracts::component_profile::component_support_profiles;
use std::path::PathBuf;

fn main() {
    let arguments: Vec<_> = std::env::args_os().skip(1).collect();
    if arguments.len() != 2 || arguments[0] != "--schema-root" {
        eprintln!("usage: component-profile --schema-root PATH");
        std::process::exit(2);
    }
    let root = PathBuf::from(&arguments[1]);
    let profiles = component_support_profiles(&root).unwrap_or_else(|error| {
        eprintln!("component-profile: {error}");
        std::process::exit(1);
    });
    for profile in profiles {
        match serde_json::to_string(&profile) {
            Ok(encoded) => println!("{encoded}"),
            Err(error) => {
                eprintln!("component-profile: {error}");
                std::process::exit(1);
            }
        }
    }
}
