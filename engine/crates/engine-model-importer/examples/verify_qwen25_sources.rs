//! Reproducible, content-free diagnostic for exact local Qwen2.5 source metadata.
//!
//! This does not create, sign, qualify, or activate a model package.

use personalai_engine_model_importer::{
    ImportSpec, QWEN25_TOKENIZER_BYTES, inspect_qwen25_tokenizer_json,
};
use personalai_engine_quantization::QuantizationProfile;
use personalai_engine_types::{EngineError, ModelId};
use std::ffi::OsString;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

const USAGE: &str = "usage: cargo run -p personalai-engine-model-importer --example verify_qwen25_sources -- CONFIG_JSON TOKENIZER_JSON";

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
    let config_path = PathBuf::from(
        arguments
            .next()
            .ok_or_else(|| EngineError::invalid_input("configuration path is required"))?,
    );
    let tokenizer_path = PathBuf::from(
        arguments.next().ok_or_else(|| EngineError::invalid_input("tokenizer path is required"))?,
    );
    if arguments.next().is_some() {
        return Err(EngineError::invalid_input("unexpected diagnostic argument"));
    }
    let config = read_exact_local(&config_path, &[656, 660], "configuration")?;
    let tokenizer = read_exact_local(&tokenizer_path, &[QWEN25_TOKENIZER_BYTES], "tokenizer")?;
    let descriptor = inspect_qwen25_tokenizer_json(&tokenizer)?;
    let spec = ImportSpec::new_qwen25_1_5b(
        ModelId::new("qwen25-source-inspection")?,
        &config,
        &tokenizer,
        "operator-local-diagnostic",
        "unqualified-source",
        QuantizationProfile::ReferenceF32,
    )?;
    let config = spec
        .qwen25_config()
        .ok_or_else(|| EngineError::internal_invariant("Qwen config evidence was not retained"))?;
    println!(
        "profile={:?} config_sha256={} tokenizer_sha256={} tokenizer_ids={} merges={} model_source_bytes={}",
        config.profile(),
        config.source_sha256(),
        descriptor.source_sha256(),
        descriptor.defined_token_count(),
        descriptor.merge_count(),
        config.model_source_bytes(),
    );
    Ok(())
}

fn read_exact_local(
    path: &Path,
    allowed_lengths: &[usize],
    label: &'static str,
) -> Result<Vec<u8>, EngineError> {
    let metadata = fs::symlink_metadata(path)
        .map_err(|_| EngineError::invalid_input(format!("{label} source is unavailable")))?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err(EngineError::invalid_input(format!(
            "{label} source must be a regular non-symlink file"
        )));
    }
    let length = usize::try_from(metadata.len()).map_err(|_| {
        EngineError::resource_exhaustion(format!("{label} length is not addressable"))
    })?;
    if !allowed_lengths.contains(&length) {
        return Err(EngineError::corrupt_model_package(format!(
            "{label} byte length is not pinned"
        )));
    }
    let bytes = fs::read(path)
        .map_err(|_| EngineError::authentication_failure(format!("{label} source read failed")))?;
    if bytes.len() != length {
        return Err(EngineError::authentication_failure(format!(
            "{label} source changed during diagnostic read"
        )));
    }
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn diagnostic_requires_exact_two_local_sources() {
        assert!(run(std::iter::empty()).is_err());
        assert!(run([OsString::from("one")].into_iter()).is_err());
        assert!(
            run([OsString::from("one"), OsString::from("two"), OsString::from("three")].into_iter())
                .is_err()
        );
    }
}
