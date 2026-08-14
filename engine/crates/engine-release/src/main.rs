//! Owner-facing offline release signer and pre-launch verifier.

use personalai_engine_crypto::{EngineReleaseSigner, TrustedReleaseSigner};
use personalai_engine_release::{
    MonotonicReleaseLedger, ReleaseArtifactRole, ReleaseManifest, inspect_release_artifact,
    verify_installed_release, write_signed_manifest,
};
use personalai_engine_types::{EngineError, SignerKeyId};
use std::collections::BTreeMap;
use std::ffi::OsString;
use std::io::{self, Read};
use std::path::PathBuf;
use std::process::ExitCode;
use zeroize::Zeroize;

const USAGE: &str = "usage:\n  personalai-engine-release sign \
--release-root PATH --generation U64 --release-id ID --target TARGET \
--rust-toolchain ID --signer-key-id HEX32 \
--engine-binary NAME --sbom NAME --cargo-lock NAME \
[--metal-library NAME] [--notices NAME]\n  \
personalai-engine-release verify \
--release-root PATH --ledger-root PATH --engine-runtime-root PATH \
--signer-key-id HEX32 --signer-public-key HEX64\n\n\
sign reads exactly 32 binary Ed25519 release-signing seed bytes from stdin; \
verify never reads private key material; all paths are owner configuration";

fn main() -> ExitCode {
    match run(std::env::args_os().skip(1), &mut io::stdin().lock()) {
        Ok(message) => {
            println!("{message}");
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("release operation rejected: {error}\n{USAGE}");
            ExitCode::from(2)
        }
    }
}

fn run(
    mut arguments: impl Iterator<Item = OsString>,
    signing_input: &mut impl Read,
) -> Result<String, EngineError> {
    let command = arguments
        .next()
        .ok_or_else(|| EngineError::invalid_input("release command is required"))?
        .into_string()
        .map_err(|_| EngineError::invalid_input("release command must be valid UTF-8"))?;
    let values = parse_named_arguments(arguments)?;
    match command.as_str() {
        "sign" => sign(values, signing_input),
        "verify" => verify(values),
        _ => Err(EngineError::invalid_input("release command is unsupported")),
    }
}

fn sign(
    mut values: BTreeMap<String, OsString>,
    signing_input: &mut impl Read,
) -> Result<String, EngineError> {
    let root = PathBuf::from(take(&mut values, "--release-root")?);
    let generation = number(&mut values, "--generation")?;
    let release_id = text(take(&mut values, "--release-id")?)?;
    let target = text(take(&mut values, "--target")?)?;
    let toolchain = text(take(&mut values, "--rust-toolchain")?)?;
    let key_id =
        SignerKeyId::from_bytes(parse_hex(&text(take(&mut values, "--signer-key-id")?)?)?)?;
    let required = [
        (ReleaseArtifactRole::EngineBinary, "--engine-binary"),
        (ReleaseArtifactRole::Sbom, "--sbom"),
        (ReleaseArtifactRole::CargoLock, "--cargo-lock"),
    ];
    let mut artifacts = Vec::new();
    for (role, flag) in required {
        let name = text(take(&mut values, flag)?)?;
        artifacts.push(inspect_release_artifact(&root, role, &name)?);
    }
    for (role, flag) in [
        (ReleaseArtifactRole::MetalLibrary, "--metal-library"),
        (ReleaseArtifactRole::Notices, "--notices"),
    ] {
        if let Some(value) = values.remove(flag) {
            let name = text(value)?;
            artifacts.push(inspect_release_artifact(&root, role, &name)?);
        }
    }
    if !values.is_empty() {
        return Err(EngineError::invalid_input("unknown release sign flag"));
    }
    let manifest = ReleaseManifest::new(generation, release_id, target, toolchain, artifacts)?;
    let mut seed = read_seed(signing_input)?;
    let signer = EngineReleaseSigner::from_seed(key_id, &mut seed)?;
    let (digest, public_key) = write_signed_manifest(&root, &manifest, &signer)?;
    Ok(format!(
        "signed_release_generation={} manifest_sha384={} signer_key_id={} signer_public_key={}",
        generation,
        encode_hex(&digest),
        key_id,
        encode_hex(&public_key)
    ))
}

fn verify(mut values: BTreeMap<String, OsString>) -> Result<String, EngineError> {
    let release_root = PathBuf::from(take(&mut values, "--release-root")?);
    let ledger_root = PathBuf::from(take(&mut values, "--ledger-root")?);
    let runtime_root = PathBuf::from(take(&mut values, "--engine-runtime-root")?);
    let key_id =
        SignerKeyId::from_bytes(parse_hex(&text(take(&mut values, "--signer-key-id")?)?)?)?;
    let public_key = parse_hex(&text(take(&mut values, "--signer-public-key")?)?)?;
    if !values.is_empty() {
        return Err(EngineError::invalid_input("unknown release verify flag"));
    }
    let trusted = TrustedReleaseSigner::active(key_id, public_key)?;
    let release = verify_installed_release(&release_root, &trusted)?;
    let ledger = MonotonicReleaseLedger::open(&ledger_root, &runtime_root)?;
    ledger.checkpoint(&release)?;
    Ok(format!(
        "verified_release_generation={} release_id={} target={} manifest_sha384={}",
        release.manifest().generation(),
        release.manifest().release_id(),
        release.manifest().target(),
        encode_hex(release.manifest_digest())
    ))
}

fn parse_named_arguments(
    mut arguments: impl Iterator<Item = OsString>,
) -> Result<BTreeMap<String, OsString>, EngineError> {
    let mut values = BTreeMap::new();
    while let Some(flag) = arguments.next() {
        let flag = flag
            .into_string()
            .map_err(|_| EngineError::invalid_input("release flag must be valid UTF-8"))?;
        if !flag.starts_with("--") {
            return Err(EngineError::invalid_input("release argument must be a named flag"));
        }
        let value = arguments
            .next()
            .ok_or_else(|| EngineError::invalid_input("release flag requires a value"))?;
        if values.insert(flag, value).is_some() {
            return Err(EngineError::invalid_input("release flag may appear only once"));
        }
    }
    Ok(values)
}

fn take(values: &mut BTreeMap<String, OsString>, key: &str) -> Result<OsString, EngineError> {
    values.remove(key).ok_or_else(|| EngineError::invalid_input("required release flag is missing"))
}

fn text(value: OsString) -> Result<String, EngineError> {
    value.into_string().map_err(|_| EngineError::invalid_input("release value must be valid UTF-8"))
}

fn number(values: &mut BTreeMap<String, OsString>, key: &str) -> Result<u64, EngineError> {
    text(take(values, key)?)?
        .parse()
        .map_err(|_| EngineError::invalid_input("release integer is invalid"))
}

fn read_seed(reader: &mut impl Read) -> Result<[u8; 32], EngineError> {
    let mut seed = [0_u8; 32];
    reader
        .read_exact(&mut seed)
        .map_err(|_| EngineError::authentication_failure("release signing seed is truncated"))?;
    let mut trailing = [0_u8; 1];
    if reader
        .read(&mut trailing)
        .map_err(|_| EngineError::authentication_failure("release signing seed read failed"))?
        != 0
    {
        seed.zeroize();
        return Err(EngineError::authentication_failure(
            "release signing seed must contain exactly 32 bytes",
        ));
    }
    Ok(seed)
}

fn parse_hex<const N: usize>(value: &str) -> Result<[u8; N], EngineError> {
    if value.len() != N * 2 {
        return Err(EngineError::invalid_input("release hex value has the wrong length"));
    }
    let mut bytes = [0_u8; N];
    for (index, pair) in value.as_bytes().chunks_exact(2).enumerate() {
        bytes[index] = (hex_nibble(pair[0])? << 4) | hex_nibble(pair[1])?;
    }
    Ok(bytes)
}

fn hex_nibble(value: u8) -> Result<u8, EngineError> {
    match value {
        b'0'..=b'9' => Ok(value - b'0'),
        b'a'..=b'f' => Ok(value - b'a' + 10),
        b'A'..=b'F' => Ok(value - b'A' + 10),
        _ => Err(EngineError::invalid_input("release hex value is invalid")),
    }
}

fn encode_hex(bytes: &[u8]) -> String {
    use std::fmt::Write;
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        let _ = write!(output, "{byte:02x}");
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parser_rejects_missing_duplicate_unknown_and_bad_hex_arguments() {
        assert!(run(std::iter::empty(), &mut &[][..]).is_err());
        assert!(parse_hex::<16>("00").is_err());
        assert!(parse_hex::<1>("zz").is_err());
        let duplicate = ["verify", "--release-root", "a", "--release-root", "b"]
            .into_iter()
            .map(OsString::from);
        assert!(run(duplicate, &mut &[][..]).is_err());
    }

    #[test]
    fn release_seed_reader_requires_exact_length_and_clears_trailing_case() {
        assert!(read_seed(&mut &[1_u8; 31][..]).is_err());
        assert!(read_seed(&mut &[1_u8; 33][..]).is_err());
        assert_eq!(read_seed(&mut &[1_u8; 32][..]), Ok([1_u8; 32]));
    }
}
