//! Owner-facing offline release signer and pre-launch verifier.

use personalai_engine_crypto::{EngineReleaseSigner, TrustedReleaseSigner};
use personalai_engine_release::{
    MonotonicReleaseLedger, ReleaseArtifactRole, ReleaseManifest, SupervisorSecurityEventJournal,
    SupervisorSecurityEventKind, inspect_release_artifact, verify_installed_release,
    write_signed_manifest,
};
use personalai_engine_types::{EngineError, EngineErrorKind, SignerKeyId};
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
--signer-key-id HEX32 --signer-public-key HEX64\n  \
personalai-engine-release verify-supervised \
--release-root PATH --ledger-root PATH --engine-runtime-root PATH \
--security-event-root PATH --signer-key-id HEX32 --signer-public-key HEX64\n\n\
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
        "verify-supervised" => verify_supervised(values),
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
    let options = verify_options(&mut values)?;
    if !values.is_empty() {
        return Err(EngineError::invalid_input("unknown release verify flag"));
    }
    let trusted = TrustedReleaseSigner::active(options.key_id, options.public_key)?;
    let release = verify_installed_release(&options.release_root, &trusted)?;
    let ledger = MonotonicReleaseLedger::open(&options.ledger_root, &options.runtime_root)?;
    ledger.checkpoint(&release)?;
    Ok(verified_message(&release))
}

fn verify_supervised(mut values: BTreeMap<String, OsString>) -> Result<String, EngineError> {
    let event_root = PathBuf::from(take(&mut values, "--security-event-root")?);
    let options = verify_options(&mut values)?;
    if !values.is_empty() {
        return Err(EngineError::invalid_input("unknown supervised release verify flag"));
    }
    let events = SupervisorSecurityEventJournal::open(
        &options.ledger_root,
        &event_root,
        &options.release_root,
        &options.runtime_root,
    )?;
    let trusted = match TrustedReleaseSigner::active(options.key_id, options.public_key) {
        Ok(trusted) => trusted,
        Err(error) => {
            events.append(SupervisorSecurityEventKind::ReleaseTrustFailure, 0, [0; 48])?;
            return Err(error);
        }
    };
    let release = match verify_installed_release(&options.release_root, &trusted) {
        Ok(release) => release,
        Err(error) => {
            events.append(supervisor_verification_kind(error.kind()), 0, [0; 48])?;
            return Err(error);
        }
    };
    let ledger = match MonotonicReleaseLedger::open(&options.ledger_root, &options.runtime_root) {
        Ok(ledger) => ledger,
        Err(error) => {
            events.append(
                SupervisorSecurityEventKind::ReleaseLedgerFailure,
                release.manifest().generation(),
                *release.manifest_digest(),
            )?;
            return Err(error);
        }
    };
    if let Err(error) = ledger.checkpoint(&release) {
        let kind = if error.kind() == EngineErrorKind::AuthenticationFailure {
            SupervisorSecurityEventKind::ReleaseRollbackAttempt
        } else {
            SupervisorSecurityEventKind::ReleaseLedgerFailure
        };
        events.append(kind, release.manifest().generation(), *release.manifest_digest())?;
        return Err(error);
    }
    Ok(verified_message(&release))
}

struct VerifyOptions {
    release_root: PathBuf,
    ledger_root: PathBuf,
    runtime_root: PathBuf,
    key_id: SignerKeyId,
    public_key: [u8; 32],
}

fn verify_options(values: &mut BTreeMap<String, OsString>) -> Result<VerifyOptions, EngineError> {
    let release_root = PathBuf::from(take(values, "--release-root")?);
    let ledger_root = PathBuf::from(take(values, "--ledger-root")?);
    let runtime_root = PathBuf::from(take(values, "--engine-runtime-root")?);
    let key_id = SignerKeyId::from_bytes(parse_hex(&text(take(values, "--signer-key-id")?)?)?)?;
    let public_key = parse_hex(&text(take(values, "--signer-public-key")?)?)?;
    Ok(VerifyOptions { release_root, ledger_root, runtime_root, key_id, public_key })
}

fn verified_message(release: &personalai_engine_release::VerifiedRelease) -> String {
    format!(
        "verified_release_generation={} release_id={} target={} manifest_sha384={}",
        release.manifest().generation(),
        release.manifest().release_id(),
        release.manifest().target(),
        encode_hex(release.manifest_digest())
    )
}

const fn supervisor_verification_kind(kind: EngineErrorKind) -> SupervisorSecurityEventKind {
    match kind {
        EngineErrorKind::SignatureFailure => SupervisorSecurityEventKind::ReleaseIntegrityFailure,
        EngineErrorKind::AuthenticationFailure => SupervisorSecurityEventKind::ReleaseTrustFailure,
        EngineErrorKind::ResourceExhaustion => {
            SupervisorSecurityEventKind::ReleaseResourceExhaustion
        }
        EngineErrorKind::InvalidInput
        | EngineErrorKind::UnsupportedArchitecture
        | EngineErrorKind::UnsupportedTokenizer
        | EngineErrorKind::CorruptModelPackage
        | EngineErrorKind::ProtocolViolation => {
            SupervisorSecurityEventKind::ReleaseManifestRejected
        }
        _ => SupervisorSecurityEventKind::ReleaseVerifierFailure,
    }
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
    use personalai_engine_crypto::EngineReleaseSigner;
    use personalai_engine_release::SupervisorSecurityEvent;
    use std::fs;
    use std::os::unix::fs::PermissionsExt;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};

    static NEXT_TEST_ID: AtomicU64 = AtomicU64::new(1);

    fn temporary_root(label: &str) -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock must follow epoch")
            .as_nanos();
        let sequence = NEXT_TEST_ID.fetch_add(1, Ordering::Relaxed);
        let root = std::env::temp_dir()
            .join(format!("pair-release-cli-{label}-{}-{nonce}-{sequence}", std::process::id()));
        fs::create_dir(&root).expect("temporary root must create");
        fs::set_permissions(&root, fs::Permissions::from_mode(0o700))
            .expect("temporary root mode must set");
        root
    }

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

    #[test]
    fn supervised_verification_failure_emits_one_content_free_event() {
        let release_root = temporary_root("release");
        let ledger_root = temporary_root("ledger");
        let event_root = temporary_root("events");
        let runtime_parent = temporary_root("runtime-parent");
        fs::set_permissions(&release_root, fs::Permissions::from_mode(0o500))
            .expect("release root must be read only");
        let key_id = SignerKeyId::from_bytes([0x31; 16]).expect("key ID must validate");
        let mut seed = [0x32; 32];
        let signer = EngineReleaseSigner::from_seed(key_id, &mut seed).expect("signer");
        let arguments = [
            "verify-supervised".to_owned(),
            "--release-root".to_owned(),
            release_root.to_string_lossy().into_owned(),
            "--ledger-root".to_owned(),
            ledger_root.to_string_lossy().into_owned(),
            "--engine-runtime-root".to_owned(),
            runtime_parent.join("engine").to_string_lossy().into_owned(),
            "--security-event-root".to_owned(),
            event_root.to_string_lossy().into_owned(),
            "--signer-key-id".to_owned(),
            encode_hex(key_id.as_bytes()),
            "--signer-public-key".to_owned(),
            encode_hex(&signer.public_key()),
        ]
        .into_iter()
        .map(OsString::from);
        assert!(run(arguments, &mut &[][..]).is_err());
        let entries = fs::read_dir(&event_root)
            .expect("event directory must list")
            .collect::<Result<Vec<_>, _>>()
            .expect("event entries must read");
        assert_eq!(entries.len(), 1);
        let event = SupervisorSecurityEvent::parse(
            &fs::read(entries[0].path()).expect("event bytes must read"),
        )
        .expect("event must parse");
        assert_eq!(event.kind(), SupervisorSecurityEventKind::ReleaseManifestRejected);
        assert_eq!(event.release_generation(), 0);
        assert!(event.manifest_digest().iter().all(|byte| *byte == 0));

        fs::set_permissions(&release_root, fs::Permissions::from_mode(0o700))
            .expect("cleanup mode");
        for root in [release_root, ledger_root, event_root, runtime_parent] {
            fs::remove_dir_all(root).expect("temporary root must remove");
        }
    }
}
