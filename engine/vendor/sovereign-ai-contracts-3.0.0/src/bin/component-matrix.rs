use sovereign_ai_contracts::component_matrix::component_route_matrix;
use std::path::PathBuf;

fn main() {
    let arguments: Vec<_> = std::env::args_os().skip(1).collect();
    if arguments.len() != 2 || arguments[0] != "--schema-root" {
        eprintln!("usage: component-matrix --schema-root PATH");
        std::process::exit(2);
    }
    let root = PathBuf::from(&arguments[1]);
    let matrix = component_route_matrix(&root).unwrap_or_else(|error| {
        eprintln!("component-matrix: {error}");
        std::process::exit(1);
    });
    for pair in matrix {
        match serde_json::to_string(&pair) {
            Ok(encoded) => println!("{encoded}"),
            Err(error) => {
                eprintln!("component-matrix: {error}");
                std::process::exit(1);
            }
        }
    }
}
