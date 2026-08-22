//! Reproducible, descriptor-bound diagnostic for exact local Qwen2.5 sources.
//!
//! This does not create, sign, qualify, or activate a model package.

use personalai_engine_model_importer::preflight_qwen25_ingress;
use personalai_engine_types::EngineError;
use std::ffi::OsString;
use std::path::PathBuf;
use std::process::ExitCode;

const USAGE: &str = "usage: cargo run -p personalai-engine-model-importer --example verify_qwen25_sources -- INGRESS_ROOT ARTIFACT_NAME";

fn main() -> ExitCode {
    match run(std::env::args_os().skip(1)) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("Qwen2.5 source inspection rejected: {error}\n{USAGE}");
            ExitCode::from(2)
        }
    }
}

fn run(mut arguments: impl Iterator<Item = OsString>) -> Result<(), EngineError> {
    let ingress_root = PathBuf::from(
        arguments.next().ok_or_else(|| EngineError::invalid_input("ingress root is required"))?,
    );
    let artifact_name = arguments
        .next()
        .ok_or_else(|| EngineError::invalid_input("artifact name is required"))?
        .into_string()
        .map_err(|_| EngineError::invalid_input("artifact name is invalid UTF-8"))?;
    if arguments.next().is_some() {
        return Err(EngineError::invalid_input("unexpected diagnostic argument"));
    }
    let report = preflight_qwen25_ingress(&ingress_root, &artifact_name)?;
    println!(
        "profile={:?} config_sha256={} tokenizer_sha256={} model_sha256={} model_bytes={} safetensors_header_bytes={} tensors={} status=preflight-only-unqualified",
        report.profile(),
        report.config_sha256(),
        report.tokenizer_sha256(),
        report.model_sha256(),
        report.model_bytes(),
        report.safetensors_header_bytes(),
        report.tensor_count(),
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn diagnostic_requires_exact_ingress_and_artifact_arguments() {
        assert!(run(std::iter::empty()).is_err());
        assert!(run([OsString::from("one")].into_iter()).is_err());
        assert!(
            run([OsString::from("one"), OsString::from("two"), OsString::from("three")].into_iter())
                .is_err()
        );
    }
}
