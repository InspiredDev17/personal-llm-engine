//! Offline operator command for one exact, non-overwritable AIPTOK02 export.

use personalai_engine_model_importer::export_qwen25_aiptok;
use std::path::PathBuf;
use std::process::ExitCode;

const USAGE: &str = "usage: cargo run -p personalai-engine-model-importer --example export_qwen25_aiptok -- /absolute/pinned/tokenizer.json /absolute/new/tokenizer.aiptok";

fn main() -> ExitCode {
    let mut args = std::env::args_os().skip(1);
    let Some(source) = args.next() else {
        eprintln!("AIPTOK02 export rejected: source argument is absent\n{USAGE}");
        return ExitCode::from(2);
    };
    let Some(output) = args.next() else {
        eprintln!("AIPTOK02 export rejected: output argument is absent\n{USAGE}");
        return ExitCode::from(2);
    };
    if args.next().is_some() {
        eprintln!("AIPTOK02 export rejected: unexpected argument\n{USAGE}");
        return ExitCode::from(2);
    }
    match export_qwen25_aiptok(&PathBuf::from(source), &PathBuf::from(output)) {
        Ok(report) => {
            println!(
                "source_sha256={} aiptok_sha256={} aiptok_sha384={} aiptok_bytes={} mode=0400 status=offline-development-artifact-unqualified-unactivated",
                report.source_sha256(),
                report.output_sha256(),
                report.output_sha384(),
                report.output_bytes(),
            );
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("AIPTOK02 export rejected: {error}\n{USAGE}");
            ExitCode::from(2)
        }
    }
}
