use sovereign_ai_contracts::component_registry::component_contract_registry;

fn main() {
    if std::env::args_os().len() != 1 {
        eprintln!("usage: component-registry");
        std::process::exit(2);
    }
    for entry in component_contract_registry() {
        match serde_json::to_string(&entry) {
            Ok(encoded) => println!("{encoded}"),
            Err(error) => {
                eprintln!("component-registry: {error}");
                std::process::exit(1);
            }
        }
    }
}
