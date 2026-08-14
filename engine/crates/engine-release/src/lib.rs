//! Offline engine-release signing, installed-artifact verification, and
//! monotonic rollback protection.
//!
//! This crate is intentionally absent from the inference server dependency
//! graph. It does not install, launch, update, download, or select an engine.
//! An owner-controlled supervisor invokes it before launch against a fixed
//! release directory and a protected ledger outside the engine runtime.

use personalai_engine_crypto::{
    ED25519_SIGNATURE_BYTES, EngineReleaseSigner, TrustedReleaseSigner, hash_bounded_reader,
    hash_signed_content, verify_engine_release_signature,
};
use personalai_engine_types::{EngineError, ModelPackageHash, SignerKeyId};
use std::collections::BTreeSet;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::os::unix::fs::{MetadataExt, OpenOptionsExt, PermissionsExt};
use std::path::{Component, Path, PathBuf};

/// Exact release-manifest magic.
pub const RELEASE_MANIFEST_MAGIC: [u8; 8] = *b"PAIRLSV1";
/// Exact release-signature-envelope magic.
pub const RELEASE_SIGNATURE_MAGIC: [u8; 8] = *b"PAIRSIG1";
/// Exact rollback-ledger magic.
pub const ROLLBACK_LEDGER_MAGIC: [u8; 8] = *b"PAIRLDG1";
/// Supported format version.
pub const RELEASE_FORMAT_VERSION: u16 = 1;
/// Maximum encoded manifest bytes.
pub const MAX_RELEASE_MANIFEST_BYTES: usize = 64 * 1024;
/// Maximum exact release artifacts.
pub const MAX_RELEASE_ARTIFACTS: usize = 16;
/// Maximum one release artifact size.
pub const MAX_RELEASE_ARTIFACT_BYTES: u64 = 2 * 1024 * 1024 * 1024;
/// Maximum total release artifact bytes.
pub const MAX_RELEASE_TOTAL_BYTES: u64 = 4 * 1024 * 1024 * 1024;
/// Maximum release identifier length.
pub const MAX_RELEASE_ID_BYTES: usize = 64;
/// Maximum target identifier length.
pub const MAX_TARGET_BYTES: usize = 128;
/// Maximum toolchain identifier length.
pub const MAX_TOOLCHAIN_BYTES: usize = 64;
/// Maximum direct-child artifact filename length.
pub const MAX_ARTIFACT_NAME_BYTES: usize = 128;

const SIGNATURE_ENVELOPE_BYTES: usize = 8 + 2 + 2 + 16 + ED25519_SIGNATURE_BYTES;
const LEDGER_BYTES: usize = 8 + 2 + 2 + 8 + ModelPackageHash::BYTE_LEN;
const SIGNATURE_ALGORITHM_ED25519: u16 = 1;
const MANIFEST_FLAGS: u16 = 0;

/// Closed artifact-role vocabulary for an engine release.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
pub enum ReleaseArtifactRole {
    /// Native engine executable.
    EngineBinary = 1,
    /// SPDX release dependency inventory.
    Sbom = 2,
    /// Exact Cargo lockfile used for the build.
    CargoLock = 3,
    /// Optional separately shipped first-party Metal library.
    MetalLibrary = 4,
    /// Optional release notices/license material.
    Notices = 5,
}

impl ReleaseArtifactRole {
    fn parse(value: u8) -> Result<Self, EngineError> {
        match value {
            1 => Ok(Self::EngineBinary),
            2 => Ok(Self::Sbom),
            3 => Ok(Self::CargoLock),
            4 => Ok(Self::MetalLibrary),
            5 => Ok(Self::Notices),
            _ => Err(EngineError::invalid_input("release artifact role is unknown")),
        }
    }
}

/// Exact immutable identity of one release artifact.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReleaseArtifact {
    role: ReleaseArtifactRole,
    name: String,
    byte_length: u64,
    sha384: [u8; ModelPackageHash::BYTE_LEN],
}

impl ReleaseArtifact {
    /// Constructs one exact direct-child artifact record.
    pub fn new(
        role: ReleaseArtifactRole,
        name: impl Into<String>,
        byte_length: u64,
        sha384: [u8; ModelPackageHash::BYTE_LEN],
    ) -> Result<Self, EngineError> {
        let name = name.into();
        validate_identifier("artifact name", &name, MAX_ARTIFACT_NAME_BYTES)?;
        validate_direct_child_name(&name)?;
        if byte_length == 0 || byte_length > MAX_RELEASE_ARTIFACT_BYTES {
            return Err(EngineError::invalid_input("release artifact length is outside bounds"));
        }
        Ok(Self { role, name, byte_length, sha384 })
    }

    /// Returns the closed artifact role.
    pub const fn role(&self) -> ReleaseArtifactRole {
        self.role
    }

    /// Returns the direct-child artifact filename.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns the exact artifact length.
    pub const fn byte_length(&self) -> u64 {
        self.byte_length
    }

    /// Returns the exact SHA-384 artifact identity.
    pub const fn sha384(&self) -> &[u8; ModelPackageHash::BYTE_LEN] {
        &self.sha384
    }
}

/// Canonical signed engine release manifest V1.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReleaseManifest {
    generation: u64,
    release_id: String,
    target: String,
    rust_toolchain: String,
    artifacts: Vec<ReleaseArtifact>,
}

impl ReleaseManifest {
    /// Constructs and validates a canonical manifest.
    pub fn new(
        generation: u64,
        release_id: impl Into<String>,
        target: impl Into<String>,
        rust_toolchain: impl Into<String>,
        mut artifacts: Vec<ReleaseArtifact>,
    ) -> Result<Self, EngineError> {
        if generation == 0 {
            return Err(EngineError::invalid_input("release generation must be nonzero"));
        }
        let release_id = release_id.into();
        let target = target.into();
        let rust_toolchain = rust_toolchain.into();
        validate_identifier("release ID", &release_id, MAX_RELEASE_ID_BYTES)?;
        validate_identifier("target", &target, MAX_TARGET_BYTES)?;
        validate_identifier("toolchain", &rust_toolchain, MAX_TOOLCHAIN_BYTES)?;
        if artifacts.len() > MAX_RELEASE_ARTIFACTS {
            return Err(EngineError::invalid_input("release artifact count exceeds bound"));
        }
        artifacts.sort_by(|left, right| (left.role, &left.name).cmp(&(right.role, &right.name)));
        validate_artifacts(&artifacts)?;
        Ok(Self { generation, release_id, target, rust_toolchain, artifacts })
    }

    /// Returns the monotonic owner-assigned release generation.
    pub const fn generation(&self) -> u64 {
        self.generation
    }

    /// Returns the bounded release identifier.
    pub fn release_id(&self) -> &str {
        &self.release_id
    }

    /// Returns the exact build target identifier.
    pub fn target(&self) -> &str {
        &self.target
    }

    /// Returns the exact Rust toolchain identifier.
    pub fn rust_toolchain(&self) -> &str {
        &self.rust_toolchain
    }

    /// Returns canonical artifact records.
    pub fn artifacts(&self) -> &[ReleaseArtifact] {
        &self.artifacts
    }

    /// Encodes the exact canonical little-endian V1 representation.
    pub fn encode(&self) -> Result<Vec<u8>, EngineError> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&RELEASE_MANIFEST_MAGIC);
        bytes.extend_from_slice(&RELEASE_FORMAT_VERSION.to_le_bytes());
        bytes.extend_from_slice(&MANIFEST_FLAGS.to_le_bytes());
        bytes.extend_from_slice(&self.generation.to_le_bytes());
        push_text(&mut bytes, &self.release_id)?;
        push_text(&mut bytes, &self.target)?;
        push_text(&mut bytes, &self.rust_toolchain)?;
        let count = u16::try_from(self.artifacts.len())
            .map_err(|_| EngineError::internal_invariant("release artifact count overflows"))?;
        bytes.extend_from_slice(&count.to_le_bytes());
        for artifact in &self.artifacts {
            bytes.push(artifact.role as u8);
            bytes.push(0);
            push_text(&mut bytes, &artifact.name)?;
            bytes.extend_from_slice(&artifact.byte_length.to_le_bytes());
            bytes.extend_from_slice(&artifact.sha384);
        }
        if bytes.len() > MAX_RELEASE_MANIFEST_BYTES {
            return Err(EngineError::resource_exhaustion("release manifest exceeds bound"));
        }
        Ok(bytes)
    }

    /// Parses one exact canonical V1 representation.
    pub fn parse(bytes: &[u8]) -> Result<Self, EngineError> {
        if bytes.len() > MAX_RELEASE_MANIFEST_BYTES {
            return Err(EngineError::invalid_input("release manifest exceeds bound"));
        }
        let mut parser = Parser::new(bytes);
        if parser.take(8)? != RELEASE_MANIFEST_MAGIC {
            return Err(EngineError::invalid_input("release manifest magic is invalid"));
        }
        if parser.u16()? != RELEASE_FORMAT_VERSION {
            return Err(EngineError::invalid_input("release manifest version is unsupported"));
        }
        if parser.u16()? != MANIFEST_FLAGS {
            return Err(EngineError::invalid_input("release manifest required flags are unknown"));
        }
        let generation = parser.u64()?;
        let release_id = parser.text(MAX_RELEASE_ID_BYTES)?;
        let target = parser.text(MAX_TARGET_BYTES)?;
        let rust_toolchain = parser.text(MAX_TOOLCHAIN_BYTES)?;
        let count = usize::from(parser.u16()?);
        if count > MAX_RELEASE_ARTIFACTS {
            return Err(EngineError::invalid_input("release artifact count exceeds bound"));
        }
        let mut artifacts = Vec::new();
        artifacts
            .try_reserve_exact(count)
            .map_err(|_| EngineError::resource_exhaustion("release artifact allocation failed"))?;
        for _ in 0..count {
            let role = ReleaseArtifactRole::parse(parser.u8()?)?;
            if parser.u8()? != 0 {
                return Err(EngineError::invalid_input("release artifact flags are unknown"));
            }
            let name = parser.text(MAX_ARTIFACT_NAME_BYTES)?;
            let byte_length = parser.u64()?;
            let mut sha384 = [0_u8; ModelPackageHash::BYTE_LEN];
            sha384.copy_from_slice(parser.take(ModelPackageHash::BYTE_LEN)?);
            artifacts.push(ReleaseArtifact::new(role, name, byte_length, sha384)?);
        }
        if !parser.is_empty() {
            return Err(EngineError::invalid_input("release manifest has trailing data"));
        }
        let manifest = Self::new(generation, release_id, target, rust_toolchain, artifacts)?;
        if manifest.encode()?.as_slice() != bytes {
            return Err(EngineError::invalid_input("release manifest is not canonical"));
        }
        Ok(manifest)
    }
}

/// Signed release-manifest envelope V1.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReleaseSignature {
    key_id: SignerKeyId,
    signature: [u8; ED25519_SIGNATURE_BYTES],
}

impl ReleaseSignature {
    /// Creates an envelope from a release-only signer.
    pub fn sign(signer: &EngineReleaseSigner, manifest_bytes: &[u8]) -> Result<Self, EngineError> {
        let digest = hash_signed_content(manifest_bytes);
        Ok(Self { key_id: signer.key_id(), signature: signer.sign_manifest_digest(&digest)? })
    }

    /// Returns the release signer key identifier.
    pub const fn key_id(&self) -> SignerKeyId {
        self.key_id
    }

    /// Encodes the exact fixed-size signature envelope.
    pub fn encode(&self) -> [u8; SIGNATURE_ENVELOPE_BYTES] {
        let mut bytes = [0_u8; SIGNATURE_ENVELOPE_BYTES];
        bytes[0..8].copy_from_slice(&RELEASE_SIGNATURE_MAGIC);
        bytes[8..10].copy_from_slice(&RELEASE_FORMAT_VERSION.to_le_bytes());
        bytes[10..12].copy_from_slice(&SIGNATURE_ALGORITHM_ED25519.to_le_bytes());
        bytes[12..28].copy_from_slice(self.key_id.as_bytes());
        bytes[28..].copy_from_slice(&self.signature);
        bytes
    }

    /// Parses the exact fixed-size signature envelope.
    pub fn parse(bytes: &[u8]) -> Result<Self, EngineError> {
        if bytes.len() != SIGNATURE_ENVELOPE_BYTES {
            return Err(EngineError::signature_failure("release signature length is invalid"));
        }
        if bytes[0..8] != RELEASE_SIGNATURE_MAGIC {
            return Err(EngineError::signature_failure("release signature magic is invalid"));
        }
        if u16::from_le_bytes([bytes[8], bytes[9]]) != RELEASE_FORMAT_VERSION {
            return Err(EngineError::signature_failure("release signature version is unsupported"));
        }
        if u16::from_le_bytes([bytes[10], bytes[11]]) != SIGNATURE_ALGORITHM_ED25519 {
            return Err(EngineError::signature_failure(
                "release signature algorithm is unsupported",
            ));
        }
        let mut key_id = [0_u8; 16];
        key_id.copy_from_slice(&bytes[12..28]);
        let mut signature = [0_u8; ED25519_SIGNATURE_BYTES];
        signature.copy_from_slice(&bytes[28..]);
        Ok(Self { key_id: SignerKeyId::from_bytes(key_id)?, signature })
    }
}

/// Manifest proven to have an active trusted signature and exact artifact set.
#[derive(Clone, Debug)]
pub struct VerifiedRelease {
    manifest: ReleaseManifest,
    manifest_digest: [u8; ModelPackageHash::BYTE_LEN],
}

impl VerifiedRelease {
    /// Returns the verified manifest.
    pub const fn manifest(&self) -> &ReleaseManifest {
        &self.manifest
    }

    /// Returns the verified manifest SHA-384 identity.
    pub const fn manifest_digest(&self) -> &[u8; ModelPackageHash::BYTE_LEN] {
        &self.manifest_digest
    }
}

/// Verifies a signed manifest and every exact artifact beneath one fixed root.
pub fn verify_release_directory(
    release_root: &Path,
    manifest_bytes: &[u8],
    signature_bytes: &[u8],
    trusted_signer: &TrustedReleaseSigner,
) -> Result<VerifiedRelease, EngineError> {
    validate_secure_directory(release_root, false)?;
    let root_before = fs::symlink_metadata(release_root)
        .map_err(|_| EngineError::invalid_input("installed release root is unavailable"))?;
    let signature = ReleaseSignature::parse(signature_bytes)?;
    let manifest_digest = hash_signed_content(manifest_bytes);
    verify_engine_release_signature(
        trusted_signer,
        signature.key_id,
        &manifest_digest,
        &signature.signature,
    )?;
    let manifest = ReleaseManifest::parse(manifest_bytes)?;
    let mut total_bytes = 0_u64;
    for artifact in manifest.artifacts() {
        total_bytes = total_bytes
            .checked_add(artifact.byte_length())
            .ok_or_else(|| EngineError::resource_exhaustion("release byte total overflows"))?;
        if total_bytes > MAX_RELEASE_TOTAL_BYTES {
            return Err(EngineError::resource_exhaustion("release byte total exceeds bound"));
        }
        verify_artifact(release_root, artifact)?;
    }
    let root_after = fs::symlink_metadata(release_root)
        .map_err(|_| EngineError::signature_failure("installed release root disappeared"))?;
    if root_before.dev() != root_after.dev()
        || root_before.ino() != root_after.ino()
        || root_after.file_type().is_symlink()
    {
        return Err(EngineError::signature_failure(
            "installed release root changed during verification",
        ));
    }
    Ok(VerifiedRelease { manifest, manifest_digest })
}

/// Safely opens the fixed manifest/signature children and verifies the release.
pub fn verify_installed_release(
    release_root: &Path,
    trusted_signer: &TrustedReleaseSigner,
) -> Result<VerifiedRelease, EngineError> {
    validate_secure_directory(release_root, false)?;
    let manifest =
        read_small_direct_child(release_root, "engine-release.pairm", MAX_RELEASE_MANIFEST_BYTES)?;
    let signature =
        read_small_direct_child(release_root, "engine-release.ed25519", SIGNATURE_ENVELOPE_BYTES)?;
    let verified = verify_release_directory(release_root, &manifest, &signature, trusted_signer)?;
    validate_closed_release_set(release_root, verified.manifest())?;
    Ok(verified)
}

/// Builds deterministic artifact metadata by reading an exact direct child.
pub fn inspect_release_artifact(
    release_root: &Path,
    role: ReleaseArtifactRole,
    name: &str,
) -> Result<ReleaseArtifact, EngineError> {
    validate_secure_directory(release_root, true)?;
    validate_direct_child_name(name)?;
    let path = release_root.join(name);
    let mut file = open_readonly_regular_no_follow(&path)?;
    let metadata = file
        .metadata()
        .map_err(|_| EngineError::invalid_input("release artifact metadata is unavailable"))?;
    let byte_length = metadata.len();
    if byte_length == 0 || byte_length > MAX_RELEASE_ARTIFACT_BYTES {
        return Err(EngineError::invalid_input("release artifact length is outside bounds"));
    }
    let (sha384, actual_length) = hash_bounded_reader(&mut file, MAX_RELEASE_ARTIFACT_BYTES)?;
    if actual_length != byte_length {
        return Err(EngineError::signature_failure("release artifact length changed"));
    }
    ReleaseArtifact::new(role, name, byte_length, sha384)
}

/// Writes a signed canonical manifest without overwriting existing approval.
pub fn write_signed_manifest(
    release_root: &Path,
    manifest: &ReleaseManifest,
    signer: &EngineReleaseSigner,
) -> Result<([u8; ModelPackageHash::BYTE_LEN], [u8; 32]), EngineError> {
    validate_secure_directory(release_root, true)?;
    let manifest_bytes = manifest.encode()?;
    let signature = ReleaseSignature::sign(signer, &manifest_bytes)?.encode();
    let manifest_path = release_root.join("engine-release.pairm");
    let signature_path = release_root.join("engine-release.ed25519");
    if manifest_path.exists() || signature_path.exists() {
        return Err(EngineError::invalid_input("signed release outputs already exist"));
    }
    write_new_readonly(&manifest_path, &manifest_bytes)?;
    if let Err(error) = write_new_readonly(&signature_path, &signature) {
        let _ = fs::remove_file(&manifest_path);
        return Err(error);
    }
    Ok((hash_signed_content(&manifest_bytes), signer.public_key()))
}

/// Durable monotonic installed-release checkpoint outside engine runtime state.
pub struct MonotonicReleaseLedger {
    root: PathBuf,
}

impl MonotonicReleaseLedger {
    /// Opens a protected ledger root and proves it is separate from runtime.
    pub fn open(root: &Path, engine_runtime_root: &Path) -> Result<Self, EngineError> {
        validate_secure_directory(root, true)?;
        validate_secure_directory(engine_runtime_root, true)?;
        let root = fs::canonicalize(root)
            .map_err(|_| EngineError::invalid_input("release ledger root cannot be resolved"))?;
        let runtime = fs::canonicalize(engine_runtime_root)
            .map_err(|_| EngineError::invalid_input("engine runtime root cannot be resolved"))?;
        if root == runtime || root.starts_with(&runtime) || runtime.starts_with(&root) {
            return Err(EngineError::invalid_input(
                "release ledger root must be disjoint from engine runtime",
            ));
        }
        Ok(Self { root })
    }

    /// Returns the retained highest generation, when one exists.
    pub fn current(&self) -> Result<Option<(u64, [u8; ModelPackageHash::BYTE_LEN])>, EngineError> {
        read_ledger(&self.root.join("installed-release-v1.ledger"))
    }

    /// Advances or idempotently confirms the verified release checkpoint.
    pub fn checkpoint(&self, release: &VerifiedRelease) -> Result<(), EngineError> {
        let _lock = LedgerLock::acquire(&self.root.join("installed-release-v1.lock"))?;
        let ledger_path = self.root.join("installed-release-v1.ledger");
        if let Some((generation, digest)) = read_ledger(&ledger_path)? {
            if release.manifest.generation < generation {
                return Err(EngineError::authentication_failure(
                    "signed engine release rollback was rejected",
                ));
            }
            if release.manifest.generation == generation {
                if release.manifest_digest != digest {
                    return Err(EngineError::authentication_failure(
                        "release generation is already bound to another manifest",
                    ));
                }
                return Ok(());
            }
        }
        let bytes = encode_ledger(release.manifest.generation, &release.manifest_digest);
        let temporary = self.root.join("installed-release-v1.ledger.new");
        let mut file = open_new_private(&temporary)?;
        if let Err(error) = (|| {
            file.write_all(&bytes)
                .map_err(|_| EngineError::internal_invariant("release ledger write failed"))?;
            file.sync_all()
                .map_err(|_| EngineError::internal_invariant("release ledger sync failed"))?;
            fs::rename(&temporary, &ledger_path)
                .map_err(|_| EngineError::internal_invariant("release ledger activation failed"))?;
            sync_directory(&self.root)
        })() {
            let _ = fs::remove_file(&temporary);
            return Err(error);
        }
        Ok(())
    }
}

fn validate_artifacts(artifacts: &[ReleaseArtifact]) -> Result<(), EngineError> {
    if artifacts.is_empty() {
        return Err(EngineError::invalid_input("release artifact set is empty"));
    }
    let mut roles = BTreeSet::new();
    let mut names = BTreeSet::new();
    let mut total = 0_u64;
    for artifact in artifacts {
        if !roles.insert(artifact.role) {
            return Err(EngineError::invalid_input("release artifact role is duplicated"));
        }
        if !names.insert(artifact.name.as_str()) {
            return Err(EngineError::invalid_input("release artifact name is duplicated"));
        }
        total = total
            .checked_add(artifact.byte_length)
            .ok_or_else(|| EngineError::resource_exhaustion("release byte total overflows"))?;
    }
    if !roles.contains(&ReleaseArtifactRole::EngineBinary)
        || !roles.contains(&ReleaseArtifactRole::Sbom)
        || !roles.contains(&ReleaseArtifactRole::CargoLock)
    {
        return Err(EngineError::invalid_input(
            "release requires engine binary, SBOM, and Cargo.lock artifacts",
        ));
    }
    if total > MAX_RELEASE_TOTAL_BYTES {
        return Err(EngineError::resource_exhaustion("release byte total exceeds bound"));
    }
    Ok(())
}

fn validate_closed_release_set(root: &Path, manifest: &ReleaseManifest) -> Result<(), EngineError> {
    let mut expected = manifest
        .artifacts()
        .iter()
        .map(|artifact| artifact.name().to_owned())
        .collect::<BTreeSet<_>>();
    expected.insert("engine-release.pairm".to_owned());
    expected.insert("engine-release.ed25519".to_owned());
    let mut observed = BTreeSet::new();
    let entries = fs::read_dir(root)
        .map_err(|_| EngineError::invalid_input("installed release directory cannot be listed"))?;
    for entry in entries {
        let entry = entry
            .map_err(|_| EngineError::invalid_input("installed release directory entry failed"))?;
        let name = entry
            .file_name()
            .into_string()
            .map_err(|_| EngineError::invalid_input("release filename is not valid UTF-8"))?;
        if !observed.insert(name) || observed.len() > MAX_RELEASE_ARTIFACTS + 2 {
            return Err(EngineError::invalid_input("installed release file set exceeds bounds"));
        }
    }
    if observed != expected {
        return Err(EngineError::signature_failure("installed release file set is not exact"));
    }
    Ok(())
}

fn validate_identifier(
    field: &'static str,
    value: &str,
    maximum: usize,
) -> Result<(), EngineError> {
    if value.is_empty() || value.len() > maximum || value.chars().any(char::is_control) {
        return Err(EngineError::invalid_field(
            field,
            "is empty, oversized, or contains control text",
        ));
    }
    Ok(())
}

fn validate_direct_child_name(name: &str) -> Result<(), EngineError> {
    let mut components = Path::new(name).components();
    let valid =
        matches!(components.next(), Some(Component::Normal(_))) && components.next().is_none();
    if !valid || name == "." || name == ".." {
        return Err(EngineError::invalid_input(
            "release artifact must be one direct-child filename",
        ));
    }
    Ok(())
}

fn push_text(bytes: &mut Vec<u8>, value: &str) -> Result<(), EngineError> {
    let length = u16::try_from(value.len())
        .map_err(|_| EngineError::internal_invariant("release text length overflows"))?;
    bytes.extend_from_slice(&length.to_le_bytes());
    bytes.extend_from_slice(value.as_bytes());
    Ok(())
}

struct Parser<'a> {
    remaining: &'a [u8],
}

impl<'a> Parser<'a> {
    const fn new(bytes: &'a [u8]) -> Self {
        Self { remaining: bytes }
    }

    fn take(&mut self, length: usize) -> Result<&'a [u8], EngineError> {
        if length > self.remaining.len() {
            return Err(EngineError::invalid_input("release manifest is truncated"));
        }
        let (value, rest) = self.remaining.split_at(length);
        self.remaining = rest;
        Ok(value)
    }

    fn u8(&mut self) -> Result<u8, EngineError> {
        Ok(self.take(1)?[0])
    }

    fn u16(&mut self) -> Result<u16, EngineError> {
        let bytes = self.take(2)?;
        Ok(u16::from_le_bytes([bytes[0], bytes[1]]))
    }

    fn u64(&mut self) -> Result<u64, EngineError> {
        let bytes = self.take(8)?;
        Ok(u64::from_le_bytes(
            bytes
                .try_into()
                .map_err(|_| EngineError::internal_invariant("release integer width changed"))?,
        ))
    }

    fn text(&mut self, maximum: usize) -> Result<String, EngineError> {
        let length = usize::from(self.u16()?);
        if length == 0 || length > maximum {
            return Err(EngineError::invalid_input("release text length is outside bounds"));
        }
        let value = std::str::from_utf8(self.take(length)?)
            .map_err(|_| EngineError::invalid_input("release text is not valid UTF-8"))?;
        if value.chars().any(char::is_control) {
            return Err(EngineError::invalid_input("release text contains control characters"));
        }
        Ok(value.to_owned())
    }

    const fn is_empty(&self) -> bool {
        self.remaining.is_empty()
    }
}

fn verify_artifact(root: &Path, artifact: &ReleaseArtifact) -> Result<(), EngineError> {
    let path = root.join(artifact.name());
    let mut file = open_readonly_regular_no_follow(&path)?;
    let metadata = file
        .metadata()
        .map_err(|_| EngineError::invalid_input("release artifact metadata is unavailable"))?;
    if metadata.permissions().mode() & 0o222 != 0 {
        return Err(EngineError::invalid_input("installed release artifact must be read-only"));
    }
    if metadata.len() != artifact.byte_length() {
        return Err(EngineError::signature_failure("release artifact length changed"));
    }
    let (digest, actual_length) = hash_bounded_reader(&mut file, MAX_RELEASE_ARTIFACT_BYTES)?;
    if actual_length != metadata.len() {
        return Err(EngineError::signature_failure("release artifact length changed"));
    }
    if digest != artifact.sha384 {
        return Err(EngineError::signature_failure("release artifact hash changed"));
    }
    let after = fs::symlink_metadata(&path)
        .map_err(|_| EngineError::signature_failure("release artifact disappeared"))?;
    if metadata.dev() != after.dev()
        || metadata.ino() != after.ino()
        || metadata.len() != after.len()
    {
        return Err(EngineError::signature_failure("release artifact changed during verification"));
    }
    Ok(())
}

fn open_readonly_regular_no_follow(path: &Path) -> Result<File, EngineError> {
    let before = fs::symlink_metadata(path)
        .map_err(|_| EngineError::invalid_input("release artifact is unavailable"))?;
    if before.file_type().is_symlink() || !before.file_type().is_file() {
        return Err(EngineError::invalid_input("release artifact must be a regular non-symlink"));
    }
    let file = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_CLOEXEC | libc::O_NOFOLLOW)
        .open(path)
        .map_err(|_| EngineError::invalid_input("release artifact cannot be opened safely"))?;
    let opened = file
        .metadata()
        .map_err(|_| EngineError::invalid_input("release artifact metadata is unavailable"))?;
    if before.dev() != opened.dev() || before.ino() != opened.ino() || !opened.is_file() {
        return Err(EngineError::invalid_input("release artifact identity changed while opening"));
    }
    Ok(file)
}

fn read_small_direct_child(
    root: &Path,
    name: &str,
    maximum: usize,
) -> Result<Vec<u8>, EngineError> {
    validate_direct_child_name(name)?;
    let file = open_readonly_regular_no_follow(&root.join(name))?;
    let metadata = file
        .metadata()
        .map_err(|_| EngineError::invalid_input("release metadata is unavailable"))?;
    if metadata.permissions().mode() & 0o222 != 0 {
        return Err(EngineError::invalid_input("installed release metadata must be read-only"));
    }
    let length = metadata.len();
    if length == 0 || length > u64::try_from(maximum).unwrap_or(u64::MAX) {
        return Err(EngineError::invalid_input("release metadata length is outside bounds"));
    }
    let capacity = usize::try_from(length).map_err(|_| {
        EngineError::resource_exhaustion("release metadata cannot fit address space")
    })?;
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(capacity)
        .map_err(|_| EngineError::resource_exhaustion("release metadata allocation failed"))?;
    file.take(length.saturating_add(1))
        .read_to_end(&mut bytes)
        .map_err(|_| EngineError::invalid_input("release metadata read failed"))?;
    if bytes.len() != capacity {
        return Err(EngineError::invalid_input("release metadata length changed"));
    }
    Ok(bytes)
}

fn validate_secure_directory(path: &Path, writable: bool) -> Result<(), EngineError> {
    let metadata = fs::symlink_metadata(path)
        .map_err(|_| EngineError::invalid_input("configured release directory is unavailable"))?;
    if metadata.file_type().is_symlink() || !metadata.is_dir() {
        return Err(EngineError::invalid_input(
            "configured release directory must be a non-symlink directory",
        ));
    }
    let mode = metadata.permissions().mode();
    if mode & 0o022 != 0 {
        return Err(EngineError::invalid_input(
            "configured release directory must not be group or world writable",
        ));
    }
    if !writable && mode & 0o222 != 0 {
        return Err(EngineError::invalid_input("installed release directory must be read-only"));
    }
    Ok(())
}

fn write_new_readonly(path: &Path, bytes: &[u8]) -> Result<(), EngineError> {
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o400)
        .custom_flags(libc::O_CLOEXEC | libc::O_NOFOLLOW)
        .open(path)
        .map_err(|_| EngineError::invalid_input("release output already exists or is unsafe"))?;
    file.write_all(bytes)
        .map_err(|_| EngineError::internal_invariant("release output write failed"))?;
    file.sync_all().map_err(|_| EngineError::internal_invariant("release output sync failed"))
}

fn open_new_private(path: &Path) -> Result<File, EngineError> {
    OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .custom_flags(libc::O_CLOEXEC | libc::O_NOFOLLOW)
        .open(path)
        .map_err(|_| {
            EngineError::authentication_failure("release ledger state is locked or unsafe")
        })
}

fn encode_ledger(generation: u64, digest: &[u8; ModelPackageHash::BYTE_LEN]) -> [u8; LEDGER_BYTES] {
    let mut bytes = [0_u8; LEDGER_BYTES];
    bytes[0..8].copy_from_slice(&ROLLBACK_LEDGER_MAGIC);
    bytes[8..10].copy_from_slice(&RELEASE_FORMAT_VERSION.to_le_bytes());
    bytes[10..12].copy_from_slice(&0_u16.to_le_bytes());
    bytes[12..20].copy_from_slice(&generation.to_le_bytes());
    bytes[20..].copy_from_slice(digest);
    bytes
}

fn read_ledger(
    path: &Path,
) -> Result<Option<(u64, [u8; ModelPackageHash::BYTE_LEN])>, EngineError> {
    match fs::symlink_metadata(path) {
        Ok(metadata) if metadata.file_type().is_symlink() || !metadata.is_file() => {
            return Err(EngineError::authentication_failure(
                "release ledger is not a regular non-symlink file",
            ));
        }
        Ok(_) => {}
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(_) => {
            return Err(EngineError::authentication_failure(
                "release ledger metadata is unavailable",
            ));
        }
    }
    let mut file = open_readonly_regular_no_follow(path)?;
    let mut bytes = [0_u8; LEDGER_BYTES];
    file.read_exact(&mut bytes)
        .map_err(|_| EngineError::authentication_failure("release ledger is truncated"))?;
    let mut trailing = [0_u8; 1];
    if file
        .read(&mut trailing)
        .map_err(|_| EngineError::authentication_failure("release ledger read failed"))?
        != 0
    {
        return Err(EngineError::authentication_failure("release ledger has trailing data"));
    }
    if bytes[0..8] != ROLLBACK_LEDGER_MAGIC
        || u16::from_le_bytes([bytes[8], bytes[9]]) != RELEASE_FORMAT_VERSION
        || u16::from_le_bytes([bytes[10], bytes[11]]) != 0
    {
        return Err(EngineError::authentication_failure("release ledger header is invalid"));
    }
    let generation =
        u64::from_le_bytes(bytes[12..20].try_into().map_err(|_| {
            EngineError::internal_invariant("release ledger generation width changed")
        })?);
    if generation == 0 {
        return Err(EngineError::authentication_failure("release ledger generation is zero"));
    }
    let mut digest = [0_u8; ModelPackageHash::BYTE_LEN];
    digest.copy_from_slice(&bytes[20..]);
    Ok(Some((generation, digest)))
}

struct LedgerLock {
    path: PathBuf,
    _file: File,
}

impl LedgerLock {
    fn acquire(path: &Path) -> Result<Self, EngineError> {
        let file = open_new_private(path)?;
        Ok(Self { path: path.to_owned(), _file: file })
    }
}

impl Drop for LedgerLock {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

fn sync_directory(path: &Path) -> Result<(), EngineError> {
    File::open(path)
        .and_then(|directory| directory.sync_all())
        .map_err(|_| EngineError::internal_invariant("release ledger directory sync failed"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use personalai_engine_crypto::EngineReleaseSigner;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};

    static NEXT_TEMPORARY_ID: AtomicU64 = AtomicU64::new(1);

    fn temporary_root(label: &str) -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock must follow epoch")
            .as_nanos();
        let sequence = NEXT_TEMPORARY_ID.fetch_add(1, Ordering::Relaxed);
        let root = std::env::temp_dir()
            .join(format!("pair-release-{label}-{}-{nonce}-{sequence}", std::process::id()));
        fs::create_dir(&root).expect("temporary root must create");
        fs::set_permissions(&root, fs::Permissions::from_mode(0o700))
            .expect("temporary root mode must set");
        root
    }

    fn write_artifact(root: &Path, name: &str, bytes: &[u8]) {
        fs::write(root.join(name), bytes).expect("artifact must write");
        fs::set_permissions(root.join(name), fs::Permissions::from_mode(0o400))
            .expect("artifact mode must set");
    }

    fn fixture(
        generation: u64,
    ) -> (PathBuf, ReleaseManifest, EngineReleaseSigner, TrustedReleaseSigner) {
        let root = temporary_root("fixture");
        write_artifact(&root, "personalai-engine", b"engine-v1");
        write_artifact(&root, "native-engine.spdx", b"SPDXVersion: SPDX-2.3\n");
        write_artifact(&root, "Cargo.lock", b"version = 4\n");
        let artifacts = [
            (ReleaseArtifactRole::EngineBinary, "personalai-engine"),
            (ReleaseArtifactRole::Sbom, "native-engine.spdx"),
            (ReleaseArtifactRole::CargoLock, "Cargo.lock"),
        ]
        .into_iter()
        .map(|(role, name)| {
            inspect_release_artifact(&root, role, name).expect("artifact must inspect")
        })
        .collect();
        let manifest = ReleaseManifest::new(
            generation,
            format!("native-engine-0.1.0-g{generation}"),
            "aarch64-apple-darwin",
            "rustc-1.85.0",
            artifacts,
        )
        .expect("manifest must validate");
        let key_id = SignerKeyId::from_bytes([0x51; 16]).expect("key ID must validate");
        let mut seed = [0x52; 32];
        let signer =
            EngineReleaseSigner::from_seed(key_id, &mut seed).expect("signer must validate");
        assert_eq!(seed, [0; 32]);
        let trusted = TrustedReleaseSigner::active(key_id, signer.public_key())
            .expect("trusted signer must validate");
        (root, manifest, signer, trusted)
    }

    fn signed_bytes(
        manifest: &ReleaseManifest,
        signer: &EngineReleaseSigner,
    ) -> (Vec<u8>, [u8; SIGNATURE_ENVELOPE_BYTES]) {
        let bytes = manifest.encode().expect("manifest must encode");
        let signature =
            ReleaseSignature::sign(signer, &bytes).expect("manifest must sign").encode();
        (bytes, signature)
    }

    #[test]
    fn canonical_round_trip_and_exact_artifact_verification_pass() {
        let (root, manifest, signer, trusted) = fixture(1);
        let (manifest_bytes, signature) = signed_bytes(&manifest, &signer);
        fs::set_permissions(&root, fs::Permissions::from_mode(0o500))
            .expect("release root must become read only");
        let verified = verify_release_directory(&root, &manifest_bytes, &signature, &trusted)
            .expect("signed release must verify");
        assert_eq!(verified.manifest(), &manifest);
        assert_eq!(ReleaseManifest::parse(&manifest_bytes), Ok(manifest));
        fs::set_permissions(&root, fs::Permissions::from_mode(0o700)).expect("cleanup mode");
        fs::remove_dir_all(root).expect("fixture must remove");
    }

    #[test]
    fn parser_rejects_bad_magic_version_truncation_trailing_and_unknown_fields() {
        let (root, manifest, _signer, _trusted) = fixture(1);
        let bytes = manifest.encode().expect("manifest must encode");
        for mutation in [0_usize, 8, 10] {
            let mut bad = bytes.clone();
            bad[mutation] ^= 1;
            assert!(ReleaseManifest::parse(&bad).is_err());
        }
        assert!(ReleaseManifest::parse(&bytes[..bytes.len() - 1]).is_err());
        let mut trailing = bytes;
        trailing.push(0);
        assert!(ReleaseManifest::parse(&trailing).is_err());
        fs::remove_dir_all(root).expect("fixture must remove");
    }

    #[test]
    fn manifest_rejects_missing_duplicate_or_unsafe_artifacts() {
        let hash = [1_u8; ModelPackageHash::BYTE_LEN];
        assert!(
            ReleaseArtifact::new(ReleaseArtifactRole::EngineBinary, "../engine", 1, hash).is_err()
        );
        assert!(
            ReleaseArtifact::new(ReleaseArtifactRole::EngineBinary, "/engine", 1, hash).is_err()
        );
        let engine = ReleaseArtifact::new(ReleaseArtifactRole::EngineBinary, "engine", 1, hash)
            .expect("record must validate");
        assert!(ReleaseManifest::new(1, "r", "t", "c", vec![engine.clone()]).is_err());
        let sbom = ReleaseArtifact::new(ReleaseArtifactRole::Sbom, "sbom", 1, hash)
            .expect("record must validate");
        let lock = ReleaseArtifact::new(ReleaseArtifactRole::CargoLock, "lock", 1, hash)
            .expect("record must validate");
        assert!(
            ReleaseManifest::new(1, "r", "t", "c", vec![engine.clone(), engine, sbom, lock])
                .is_err()
        );
    }

    #[test]
    fn signature_rejects_tamper_wrong_revoked_and_untrusted_signers() {
        let (root, manifest, signer, trusted) = fixture(1);
        let (mut bytes, signature) = signed_bytes(&manifest, &signer);
        fs::set_permissions(&root, fs::Permissions::from_mode(0o500)).expect("read-only root");
        bytes[20] ^= 1;
        assert!(verify_release_directory(&root, &bytes, &signature, &trusted).is_err());
        let wrong_id = SignerKeyId::from_bytes([0x61; 16]).expect("key ID");
        let mut wrong_seed = [0x62; 32];
        let wrong = EngineReleaseSigner::from_seed(wrong_id, &mut wrong_seed).expect("signer");
        let wrong_trust =
            TrustedReleaseSigner::active(wrong_id, wrong.public_key()).expect("trust");
        let (bytes, _) = signed_bytes(&manifest, &signer);
        assert!(verify_release_directory(&root, &bytes, &signature, &wrong_trust).is_err());
        let revoked =
            TrustedReleaseSigner::revoked(trusted.key_id(), signer.public_key()).expect("revoked");
        assert!(verify_release_directory(&root, &bytes, &signature, &revoked).is_err());
        fs::set_permissions(&root, fs::Permissions::from_mode(0o700)).expect("cleanup mode");
        fs::remove_dir_all(root).expect("fixture must remove");
    }

    #[test]
    fn changed_length_hash_and_symlink_artifacts_fail_closed() {
        let (root, manifest, signer, trusted) = fixture(1);
        let (bytes, signature) = signed_bytes(&manifest, &signer);
        fs::set_permissions(root.join("personalai-engine"), fs::Permissions::from_mode(0o600))
            .expect("artifact must become writable for tamper fixture");
        fs::write(root.join("personalai-engine"), b"tampered!").expect("tamper must write");
        fs::set_permissions(&root, fs::Permissions::from_mode(0o500)).expect("read-only root");
        assert!(verify_release_directory(&root, &bytes, &signature, &trusted).is_err());
        fs::set_permissions(&root, fs::Permissions::from_mode(0o700)).expect("writable root");
        fs::remove_file(root.join("personalai-engine")).expect("artifact remove");
        std::os::unix::fs::symlink("Cargo.lock", root.join("personalai-engine"))
            .expect("symlink must create");
        fs::set_permissions(&root, fs::Permissions::from_mode(0o500)).expect("read-only root");
        assert!(verify_release_directory(&root, &bytes, &signature, &trusted).is_err());
        fs::set_permissions(&root, fs::Permissions::from_mode(0o700)).expect("cleanup mode");
        fs::remove_dir_all(root).expect("fixture must remove");
    }

    #[test]
    fn signed_outputs_are_deterministic_and_never_overwritten() {
        let (root, manifest, signer, _trusted) = fixture(1);
        let expected = signed_bytes(&manifest, &signer);
        let (digest, public_key) =
            write_signed_manifest(&root, &manifest, &signer).expect("signed outputs must write");
        assert_eq!(digest, hash_signed_content(&expected.0));
        assert_eq!(public_key, signer.public_key());
        assert_eq!(fs::read(root.join("engine-release.pairm")).expect("manifest read"), expected.0);
        assert_eq!(
            fs::read(root.join("engine-release.ed25519")).expect("signature read"),
            expected.1
        );
        assert!(write_signed_manifest(&root, &manifest, &signer).is_err());
        fs::remove_dir_all(root).expect("fixture must remove");
    }

    #[test]
    fn installed_verifier_requires_the_exact_closed_read_only_file_set() {
        let (root, manifest, signer, trusted) = fixture(1);
        write_signed_manifest(&root, &manifest, &signer).expect("signed outputs must write");
        fs::set_permissions(&root, fs::Permissions::from_mode(0o500)).expect("read-only root");
        verify_installed_release(&root, &trusted).expect("closed release must verify");
        fs::set_permissions(root.join("personalai-engine"), fs::Permissions::from_mode(0o600))
            .expect("artifact mode changes for negative test");
        assert!(verify_installed_release(&root, &trusted).is_err());
        fs::set_permissions(root.join("personalai-engine"), fs::Permissions::from_mode(0o400))
            .expect("artifact mode restores");
        fs::set_permissions(&root, fs::Permissions::from_mode(0o700)).expect("writable root");
        write_artifact(&root, "unexpected", b"extra");
        fs::set_permissions(&root, fs::Permissions::from_mode(0o500)).expect("read-only root");
        assert!(verify_installed_release(&root, &trusted).is_err());
        fs::set_permissions(&root, fs::Permissions::from_mode(0o700)).expect("cleanup mode");
        fs::remove_dir_all(root).expect("fixture must remove");
    }

    #[test]
    fn monotonic_ledger_accepts_advance_and_idempotence_but_rejects_rollback() {
        let ledger_root = temporary_root("ledger");
        let runtime_root = temporary_root("runtime");
        let ledger =
            MonotonicReleaseLedger::open(&ledger_root, &runtime_root).expect("ledger opens");

        let (release_root_one, manifest_one, signer_one, trusted_one) = fixture(1);
        let (bytes_one, signature_one) = signed_bytes(&manifest_one, &signer_one);
        fs::set_permissions(&release_root_one, fs::Permissions::from_mode(0o500))
            .expect("read only");
        let first =
            verify_release_directory(&release_root_one, &bytes_one, &signature_one, &trusted_one)
                .expect("first release verifies");
        ledger.checkpoint(&first).expect("first checkpoint");
        ledger.checkpoint(&first).expect("idempotent checkpoint");

        let (release_root_two, manifest_two, signer_two, trusted_two) = fixture(2);
        let (bytes_two, signature_two) = signed_bytes(&manifest_two, &signer_two);
        fs::set_permissions(&release_root_two, fs::Permissions::from_mode(0o500))
            .expect("read only");
        let second =
            verify_release_directory(&release_root_two, &bytes_two, &signature_two, &trusted_two)
                .expect("second release verifies");
        ledger.checkpoint(&second).expect("second checkpoint");
        assert_eq!(ledger.current().expect("ledger reads").map(|value| value.0), Some(2));
        assert!(ledger.checkpoint(&first).is_err());

        for root in [&release_root_one, &release_root_two] {
            fs::set_permissions(root, fs::Permissions::from_mode(0o700)).expect("cleanup mode");
            fs::remove_dir_all(root).expect("release root remove");
        }
        fs::remove_dir_all(ledger_root).expect("ledger root remove");
        fs::remove_dir_all(runtime_root).expect("runtime root remove");
    }

    #[test]
    fn ledger_rejects_same_generation_substitution_corruption_lock_and_runtime_overlap() {
        let ledger_root = temporary_root("ledger-negative");
        assert!(MonotonicReleaseLedger::open(&ledger_root, &ledger_root).is_err());
        let runtime_root = temporary_root("runtime-negative");
        let ledger =
            MonotonicReleaseLedger::open(&ledger_root, &runtime_root).expect("ledger opens");
        fs::write(ledger_root.join("installed-release-v1.ledger"), b"bad").expect("corrupt ledger");
        assert!(ledger.current().is_err());
        fs::remove_file(ledger_root.join("installed-release-v1.ledger")).expect("corrupt remove");
        let lock = open_new_private(&ledger_root.join("installed-release-v1.lock")).expect("lock");
        let (release_root, manifest, signer, trusted) = fixture(1);
        let (bytes, signature) = signed_bytes(&manifest, &signer);
        fs::set_permissions(&release_root, fs::Permissions::from_mode(0o500)).expect("read only");
        let release = verify_release_directory(&release_root, &bytes, &signature, &trusted)
            .expect("release verifies");
        assert!(ledger.checkpoint(&release).is_err());
        drop(lock);
        fs::remove_file(ledger_root.join("installed-release-v1.lock")).expect("lock remove");
        ledger.checkpoint(&release).expect("first checkpoint succeeds");
        let substituted = VerifiedRelease {
            manifest: release.manifest.clone(),
            manifest_digest: [0x99; ModelPackageHash::BYTE_LEN],
        };
        assert!(ledger.checkpoint(&substituted).is_err());
        fs::set_permissions(&release_root, fs::Permissions::from_mode(0o700))
            .expect("cleanup mode");
        fs::remove_dir_all(release_root).expect("release root remove");
        fs::remove_dir_all(ledger_root).expect("ledger root remove");
        fs::remove_dir_all(runtime_root).expect("runtime root remove");
    }
}
