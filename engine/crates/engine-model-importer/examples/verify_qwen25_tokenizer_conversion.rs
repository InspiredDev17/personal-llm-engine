//! Content-free diagnostic for the pinned Qwen2.5 tokenizer conversion.
//!
//! This reads the exact operator-held `tokenizer.json` from standard input and
//! prints only bounded identities and counts. It never writes source or
//! `AIPTOK02` bytes and does not create, sign, qualify, package, or activate a
//! model.

use personalai_engine_crypto::{hash_package, hash_sha256};
use personalai_engine_memory::MemoryPool;
use personalai_engine_model_importer::{
    QWEN25_TOKENIZER_BYTES, QWEN25_TOKENIZER_SHA256, convert_qwen25_tokenizer_json,
};
use personalai_engine_tokenizer::{inspect_qwen25_aiptok, verify_pinned_qwen25_production_vectors};
use personalai_engine_types::EngineError;
use std::io::{Read, Write};
use std::process::ExitCode;

const INSPECTION_POOL_BYTES: u64 = 128 * 1024 * 1024;
const USAGE: &str = "usage: cargo run -p personalai-engine-model-importer --example verify_qwen25_tokenizer_conversion < tokenizer.json";

fn main() -> ExitCode {
    if std::env::args_os().nth(1).is_some() {
        eprintln!("Qwen2.5 tokenizer conversion rejected: unexpected argument\n{USAGE}");
        return ExitCode::from(2);
    }
    match run(std::io::stdin().lock(), std::io::stdout().lock()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("Qwen2.5 tokenizer conversion rejected: {error}\n{USAGE}");
            ExitCode::from(2)
        }
    }
}

fn run(input: impl Read, mut output: impl Write) -> Result<(), EngineError> {
    let input_bound = QWEN25_TOKENIZER_BYTES
        .checked_add(1)
        .ok_or_else(|| EngineError::resource_exhaustion("tokenizer input bound overflows"))?;
    let mut source = Vec::new();
    source
        .try_reserve_exact(input_bound)
        .map_err(|_| EngineError::resource_exhaustion("tokenizer input allocation failed"))?;
    input
        .take(
            u64::try_from(input_bound).map_err(|_| {
                EngineError::resource_exhaustion("tokenizer input is not addressable")
            })?,
        )
        .read_to_end(&mut source)
        .map_err(|_| EngineError::invalid_input("tokenizer input could not be read"))?;
    if source.len() != QWEN25_TOKENIZER_BYTES {
        return Err(EngineError::corrupt_model_package(
            "Qwen2.5 tokenizer source length is not pinned",
        ));
    }

    let encoded = convert_qwen25_tokenizer_json(&source)?;
    let pool = MemoryPool::new(INSPECTION_POOL_BYTES)?;
    let descriptor = inspect_qwen25_aiptok(&encoded, &pool)?;
    let conformance_vectors = verify_pinned_qwen25_production_vectors(&encoded, &pool)?;
    writeln!(
        output,
        "source_sha256={} aiptok_sha256={} aiptok_sha384={} aiptok_table_sha384={} aiptok_bytes={} model_vocabulary={} defined_tokens={} bpe_vocabulary={} merges={} added_tokens={} conformance_vectors={} status=conversion-only-unqualified-unactivated",
        QWEN25_TOKENIZER_SHA256,
        hash_sha256(&encoded),
        hash_package(&encoded),
        descriptor.content_hash(),
        encoded.len(),
        descriptor.model_vocabulary_size(),
        descriptor.defined_token_count(),
        descriptor.bpe_vocabulary_size(),
        descriptor.merge_count(),
        descriptor.added_token_count(),
        conformance_vectors,
    )
    .map_err(|_| EngineError::backend_failure("diagnostic output could not be written"))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn diagnostic_rejects_unpinned_input_without_writing_output() {
        let mut output = Vec::new();
        assert!(run(&b"not-the-pinned-tokenizer"[..], &mut output).is_err());
        assert!(output.is_empty());
    }
}
