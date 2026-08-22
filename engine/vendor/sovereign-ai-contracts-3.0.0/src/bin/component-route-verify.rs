use sovereign_ai_contracts::component_matrix::verify_component_route_pin;
use std::path::PathBuf;

fn main() {
    let arguments: Vec<_> = std::env::args_os().skip(1).collect();
    if arguments.len() != 4 || arguments[0] != "--schema-root" || arguments[2] != "--route" {
        eprintln!("usage: component-route-verify --schema-root PATH --route FILE");
        std::process::exit(2);
    }
    let schema_root = PathBuf::from(&arguments[1]);
    let route_path = PathBuf::from(&arguments[3]);
    let route = verify_component_route_pin(&schema_root, &route_path).unwrap_or_else(|error| {
        eprintln!("component-route-verify: {error}");
        std::process::exit(1);
    });
    match serde_json::to_string(&route) {
        Ok(encoded) => println!("{encoded}"),
        Err(error) => {
            eprintln!("component-route-verify: {error}");
            std::process::exit(1);
        }
    }
}
