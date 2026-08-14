use sovereign_ai_contracts::type_registry::public_type_registry;

fn main() {
    if std::env::args_os().len() != 1 {
        eprintln!("usage: type-registry");
        std::process::exit(2);
    }
    for entry in public_type_registry() {
        match serde_json::to_string(&entry) {
            Ok(encoded) => println!("{encoded}"),
            Err(error) => {
                eprintln!("type-registry: {error}");
                std::process::exit(1);
            }
        }
    }
}
