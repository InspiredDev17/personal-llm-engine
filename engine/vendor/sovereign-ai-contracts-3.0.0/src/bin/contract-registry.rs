use sovereign_ai_contracts::document::document_registry;

fn main() {
    if std::env::args_os().len() != 1 {
        eprintln!("usage: contract-registry");
        std::process::exit(2);
    }
    for entry in document_registry() {
        match serde_json::to_string(entry) {
            Ok(encoded) => println!("{encoded}"),
            Err(error) => {
                eprintln!("contract-registry: {error}");
                std::process::exit(1);
            }
        }
    }
}
