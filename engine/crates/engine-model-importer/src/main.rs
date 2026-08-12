//! Owner-facing offline importer executable.

use personalai_engine_model_format::DIRECTORY_DECODER_ARCHITECTURE_ID;
use personalai_engine_model_importer::{
    AIPTOK_METADATA_SOURCE_FORMAT, ImportSpec, SourceMapping, import_local,
};
use personalai_engine_types::{ContextLimit, EngineError, ModelId, SignerKeyId};
use std::collections::BTreeMap;
use std::ffi::OsString;
use std::io::{self, Read};
use std::path::PathBuf;
use std::process::ExitCode;
use zeroize::Zeroize;

const USAGE: &str = "usage: personalai-engine-model-importer import \
  --quarantine-root PATH --source PATH --tokenizer PATH --output-root PATH \
  --model-id ID --architecture decoder-rms-rope-swiglu-gqa-v1 \
  --mapping llama-safetensors-v1 --tokenizer-source-format aiptok-metadata-v1 \
  --context-limit U32 --vocabulary-size U32 \
  --hidden-dimension U32 --layer-count U32 --attention-head-count U32 \
  --kv-head-count U32 --feed-forward-dimension U32 --rope-theta F32 \
  --rope-scaling F32 --tied-output true|false --source-label TEXT \
  --license-id TEXT --signer-key-id HEX32 [--force-audit-note TEXT]\n\
reads exactly 32 binary Ed25519 signing-seed bytes from stdin; local paths only";

fn main() -> ExitCode {
    match run(std::env::args_os().skip(1), &mut io::stdin().lock()) {
        Ok(receipt) => {
            println!(
                "package={} package_sha384={} signer_key_id={} signer_public_key={}",
                receipt.package_path().display(),
                receipt.package_hash(),
                receipt.signer_key_id(),
                encode_hex(&receipt.signer_public_key()),
            );
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("import rejected: {error}\n{USAGE}");
            ExitCode::from(2)
        }
    }
}

fn run(
    mut arguments: impl Iterator<Item = OsString>,
    signing_input: &mut impl Read,
) -> Result<personalai_engine_model_importer::ImportReceipt, EngineError> {
    if arguments.next().and_then(|value| value.into_string().ok()).as_deref() != Some("import") {
        return Err(EngineError::invalid_input("import command is required"));
    }
    let mut values = BTreeMap::new();
    while let Some(flag) = arguments.next() {
        let flag = flag
            .into_string()
            .map_err(|_| EngineError::invalid_input("import flag must be valid UTF-8"))?;
        if !flag.starts_with("--") {
            return Err(EngineError::invalid_input("import argument must be a named flag"));
        }
        let value = arguments
            .next()
            .ok_or_else(|| EngineError::invalid_input("import flag requires a value"))?;
        if values.insert(flag, value).is_some() {
            return Err(EngineError::invalid_input("import flag may appear only once"));
        }
    }
    let architecture = text(take(&mut values, "--architecture")?)?;
    if architecture != DIRECTORY_DECODER_ARCHITECTURE_ID {
        return Err(EngineError::invalid_input("explicit architecture mapping is unsupported"));
    }
    let model_id = ModelId::new(text(take(&mut values, "--model-id")?)?)?;
    let mapping = SourceMapping::parse(&text(take(&mut values, "--mapping")?)?)?;
    if text(take(&mut values, "--tokenizer-source-format")?)? != AIPTOK_METADATA_SOURCE_FORMAT {
        return Err(EngineError::unsupported_tokenizer("offline tokenizer source format"));
    }
    let context_limit = ContextLimit::new(number(&mut values, "--context-limit")?)?;
    let spec = ImportSpec::new(
        model_id,
        mapping,
        context_limit,
        number(&mut values, "--vocabulary-size")?,
        number(&mut values, "--hidden-dimension")?,
        number(&mut values, "--layer-count")?,
        number(&mut values, "--attention-head-count")?,
        number(&mut values, "--kv-head-count")?,
        number(&mut values, "--feed-forward-dimension")?,
        float(&mut values, "--rope-theta")?,
        float(&mut values, "--rope-scaling")?,
        boolean(&mut values, "--tied-output")?,
        text(take(&mut values, "--source-label")?)?,
        text(take(&mut values, "--license-id")?)?,
    )?;
    let signer_key_id =
        SignerKeyId::from_bytes(parse_hex(&text(take(&mut values, "--signer-key-id")?)?)?)?;
    let quarantine_root = PathBuf::from(take(&mut values, "--quarantine-root")?);
    let source = PathBuf::from(take(&mut values, "--source")?);
    let tokenizer = PathBuf::from(take(&mut values, "--tokenizer")?);
    let output_root = PathBuf::from(take(&mut values, "--output-root")?);
    let force_note = values.remove("--force-audit-note").map(text).transpose()?;
    if !values.is_empty() {
        return Err(EngineError::invalid_input("unknown import flag"));
    }
    let mut seed = [0_u8; 32];
    signing_input
        .read_exact(&mut seed)
        .map_err(|_| EngineError::authentication_failure("signing seed input is truncated"))?;
    let mut extra = [0_u8; 1];
    if signing_input
        .read(&mut extra)
        .map_err(|_| EngineError::authentication_failure("signing seed trailing check failed"))?
        != 0
    {
        seed.zeroize();
        return Err(EngineError::authentication_failure(
            "signing seed input must contain exactly 32 bytes",
        ));
    }
    import_local(
        &quarantine_root,
        &source,
        &tokenizer,
        &output_root,
        &spec,
        signer_key_id,
        &mut seed,
        force_note.as_deref(),
    )
}

fn take(values: &mut BTreeMap<String, OsString>, key: &str) -> Result<OsString, EngineError> {
    values.remove(key).ok_or_else(|| EngineError::invalid_input("required import flag is missing"))
}

fn text(value: OsString) -> Result<String, EngineError> {
    value.into_string().map_err(|_| EngineError::invalid_input("import value must be valid UTF-8"))
}

fn number(values: &mut BTreeMap<String, OsString>, key: &str) -> Result<u32, EngineError> {
    text(take(values, key)?)?
        .parse()
        .map_err(|_| EngineError::invalid_input("import integer is invalid"))
}

fn float(values: &mut BTreeMap<String, OsString>, key: &str) -> Result<f32, EngineError> {
    text(take(values, key)?)?
        .parse()
        .map_err(|_| EngineError::invalid_input("import floating-point value is invalid"))
}

fn boolean(values: &mut BTreeMap<String, OsString>, key: &str) -> Result<bool, EngineError> {
    match text(take(values, key)?)?.as_str() {
        "true" => Ok(true),
        "false" => Ok(false),
        _ => Err(EngineError::invalid_input("import boolean must be true or false")),
    }
}

fn parse_hex<const N: usize>(value: &str) -> Result<[u8; N], EngineError> {
    if value.len() != N * 2 {
        return Err(EngineError::invalid_input("hex value has the wrong length"));
    }
    let mut bytes = [0_u8; N];
    for (index, pair) in value.as_bytes().chunks_exact(2).enumerate() {
        let high = hex_nibble(pair[0])?;
        let low = hex_nibble(pair[1])?;
        bytes[index] = (high << 4) | low;
    }
    Ok(bytes)
}

fn hex_nibble(value: u8) -> Result<u8, EngineError> {
    match value {
        b'0'..=b'9' => Ok(value - b'0'),
        b'a'..=b'f' => Ok(value - b'a' + 10),
        b'A'..=b'F' => Ok(value - b'A' + 10),
        _ => Err(EngineError::invalid_input("hex value contains an invalid character")),
    }
}

fn encode_hex(bytes: &[u8]) -> String {
    use std::fmt::Write;
    let mut encoded = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        let _ = write!(encoded, "{byte:02x}");
    }
    encoded
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cli_refuses_urls_unknown_architecture_and_short_secret() {
        let args = ["import", "--architecture", "wrong"].into_iter().map(OsString::from);
        assert!(run(args, &mut &[][..]).is_err());
        assert!(parse_hex::<16>("00").is_err());
    }
}
