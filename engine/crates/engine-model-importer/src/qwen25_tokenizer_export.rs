//! Offline-only publication of the exact pinned Qwen2.5 `AIPTOK02` artifact.
//!
//! This operator surface is intentionally outside the inference service. It
//! accepts one exact source file and creates one new, non-overwritable output.

use crate::{
    QWEN25_AIPTOK_BYTES, QWEN25_AIPTOK_SHA256, QWEN25_AIPTOK_SHA384, QWEN25_TOKENIZER_BYTES,
    QWEN25_TOKENIZER_SHA256, convert_qwen25_tokenizer_json,
};
use personalai_engine_crypto::{hash_package, hash_sha256};
use personalai_engine_types::{EngineError, ModelHashAlgorithm, ModelPackageHash, Sha256Digest};
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::os::unix::fs::{MetadataExt, OpenOptionsExt, PermissionsExt};
use std::path::{Component, Path};
use zeroize::Zeroize;

/// Content-free proof of one exact offline export.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Qwen25AipTokExportReport {
    source_sha256: Sha256Digest,
    output_sha256: Sha256Digest,
    output_sha384: ModelPackageHash,
    output_bytes: u64,
}

impl Qwen25AipTokExportReport {
    /// Exact pinned source identity.
    pub const fn source_sha256(self) -> Sha256Digest {
        self.source_sha256
    }

    /// Exact canonical output SHA-256.
    pub const fn output_sha256(self) -> Sha256Digest {
        self.output_sha256
    }

    /// Exact canonical output SHA-384.
    pub const fn output_sha384(self) -> ModelPackageHash {
        self.output_sha384
    }

    /// Exact canonical output length.
    pub const fn output_bytes(self) -> u64 {
        self.output_bytes
    }
}

/// Converts one exact pinned source and publishes one fresh read-only file.
///
/// The output is created with no permissions, verified through its already
/// opened descriptor, synchronized, and only then made owner-readable. A
/// pre-existing path is never opened or replaced.
pub fn export_qwen25_aiptok(
    source_path: &Path,
    output_path: &Path,
) -> Result<Qwen25AipTokExportReport, EngineError> {
    validate_absolute_normal_file_path(source_path, "tokenizer source")?;
    validate_absolute_normal_file_path(output_path, "AIPTOK02 output")?;
    if source_path == output_path {
        return Err(EngineError::invalid_input("source and output paths must differ"));
    }

    let (mut source, owner_uid) = read_exact_pinned_source(source_path)?;
    let source_sha256 = hash_sha256(&source);
    let mut encoded = match convert_qwen25_tokenizer_json(&source) {
        Ok(encoded) => encoded,
        Err(error) => {
            source.zeroize();
            return Err(error);
        }
    };
    source.zeroize();

    let expected_sha256 = Sha256Digest::from_hex(QWEN25_AIPTOK_SHA256)
        .map_err(|_| EngineError::internal_invariant("pinned AIPTOK02 SHA-256 is invalid"))?;
    let expected_sha384 = ModelPackageHash::from_hex(
        ModelHashAlgorithm::Sha384,
        QWEN25_AIPTOK_SHA384
            .strip_prefix("sha384:")
            .ok_or_else(|| EngineError::internal_invariant("pinned AIPTOK02 SHA-384 is invalid"))?,
    )?;
    if encoded.len() != QWEN25_AIPTOK_BYTES
        || !encoded.starts_with(b"AIPTOK02")
        || hash_sha256(&encoded) != expected_sha256
        || hash_package(&encoded) != expected_sha384
    {
        encoded.zeroize();
        return Err(EngineError::internal_invariant(
            "canonical AIPTOK02 conversion identity changed",
        ));
    }

    let result =
        publish_new_read_only(output_path, owner_uid, &encoded, expected_sha256, expected_sha384);
    encoded.zeroize();
    result.map(|()| Qwen25AipTokExportReport {
        source_sha256,
        output_sha256: expected_sha256,
        output_sha384: expected_sha384,
        output_bytes: QWEN25_AIPTOK_BYTES as u64,
    })
}

fn read_exact_pinned_source(path: &Path) -> Result<(Vec<u8>, u32), EngineError> {
    let before = fs::symlink_metadata(path)
        .map_err(|_| EngineError::invalid_input("pinned tokenizer source is unavailable"))?;
    if !before.is_file()
        || before.file_type().is_symlink()
        || before.mode() & 0o7777 != 0o400
        || before.len() != QWEN25_TOKENIZER_BYTES as u64
    {
        return Err(EngineError::authentication_failure(
            "pinned tokenizer source metadata is invalid",
        ));
    }
    let mut file = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_CLOEXEC | libc::O_NOFOLLOW)
        .open(path)
        .map_err(|_| EngineError::authentication_failure("pinned tokenizer secure open failed"))?;
    let opened = file
        .metadata()
        .map_err(|_| EngineError::authentication_failure("pinned tokenizer metadata failed"))?;
    if !opened.is_file()
        || opened.dev() != before.dev()
        || opened.ino() != before.ino()
        || opened.len() != before.len()
    {
        return Err(EngineError::authentication_failure(
            "pinned tokenizer identity changed during secure open",
        ));
    }
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(QWEN25_TOKENIZER_BYTES)
        .map_err(|_| EngineError::resource_exhaustion("tokenizer source allocation failed"))?;
    bytes.resize(QWEN25_TOKENIZER_BYTES, 0);
    if file.read_exact(&mut bytes).is_err() {
        bytes.zeroize();
        return Err(EngineError::corrupt_model_package("pinned tokenizer source is truncated"));
    }
    let mut trailing = [0_u8; 1];
    if file
        .read(&mut trailing)
        .map_err(|_| EngineError::authentication_failure("tokenizer trailing check failed"))?
        != 0
    {
        bytes.zeroize();
        return Err(EngineError::authentication_failure(
            "pinned tokenizer source changed during read",
        ));
    }
    if hash_sha256(&bytes).to_string() != QWEN25_TOKENIZER_SHA256 {
        bytes.zeroize();
        return Err(EngineError::signature_failure("pinned tokenizer source identity differs"));
    }
    Ok((bytes, before.uid()))
}

fn publish_new_read_only(
    path: &Path,
    owner_uid: u32,
    bytes: &[u8],
    expected_sha256: Sha256Digest,
    expected_sha384: ModelPackageHash,
) -> Result<(), EngineError> {
    let parent = path
        .parent()
        .ok_or_else(|| EngineError::invalid_input("AIPTOK02 output parent is absent"))?;
    let parent_metadata = fs::symlink_metadata(parent)
        .map_err(|_| EngineError::invalid_input("AIPTOK02 output parent is unavailable"))?;
    if !parent_metadata.is_dir()
        || parent_metadata.file_type().is_symlink()
        || parent_metadata.uid() != owner_uid
        || parent_metadata.mode() & 0o077 != 0
        || parent_metadata.mode() & 0o200 == 0
    {
        return Err(EngineError::authentication_failure(
            "AIPTOK02 output parent must be one owner-private writable non-symlink directory",
        ));
    }
    let parent_identity = FileIdentity::from_metadata(&parent_metadata);
    let parent_file = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_CLOEXEC | libc::O_NOFOLLOW | libc::O_DIRECTORY)
        .open(parent)
        .map_err(|_| EngineError::authentication_failure("AIPTOK02 output parent open failed"))?;
    if FileIdentity::from_metadata(&parent_file.metadata().map_err(|_| {
        EngineError::authentication_failure("AIPTOK02 output parent metadata failed")
    })?) != parent_identity
    {
        return Err(EngineError::authentication_failure(
            "AIPTOK02 output parent changed during secure open",
        ));
    }
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .create_new(true)
        .mode(0o000)
        .custom_flags(libc::O_CLOEXEC | libc::O_NOFOLLOW)
        .open(path)
        .map_err(|_| EngineError::authentication_failure("AIPTOK02 output must be a fresh path"))?;
    let output_identity = FileIdentity::from_metadata(
        &file
            .metadata()
            .map_err(|_| EngineError::authentication_failure("AIPTOK02 output metadata failed"))?,
    );
    let mut guard = UnpublishedOutput {
        path,
        parent,
        parent_file: Some(parent_file),
        parent_identity,
        file: Some(file),
        output_identity,
        committed: false,
    };
    guard.verify_current()?;
    let file = guard
        .file
        .as_mut()
        .ok_or_else(|| EngineError::internal_invariant("AIPTOK02 output descriptor disappeared"))?;
    file.write_all(bytes)
        .and_then(|()| file.flush())
        .and_then(|()| file.sync_all())
        .map_err(|_| EngineError::backend_failure("AIPTOK02 output write failed"))?;
    file.seek(SeekFrom::Start(0))
        .map_err(|_| EngineError::backend_failure("AIPTOK02 output verification seek failed"))?;
    let mut verified = Vec::new();
    verified
        .try_reserve_exact(QWEN25_AIPTOK_BYTES)
        .map_err(|_| EngineError::resource_exhaustion("AIPTOK02 verification allocation failed"))?;
    file.read_to_end(&mut verified)
        .map_err(|_| EngineError::backend_failure("AIPTOK02 output verification read failed"))?;
    let valid = verified.len() == QWEN25_AIPTOK_BYTES
        && verified.starts_with(b"AIPTOK02")
        && hash_sha256(&verified) == expected_sha256
        && hash_package(&verified) == expected_sha384;
    verified.zeroize();
    if !valid {
        return Err(EngineError::signature_failure("AIPTOK02 output verification failed"));
    }
    guard.verify_current()?;
    let file = guard
        .file
        .as_mut()
        .ok_or_else(|| EngineError::internal_invariant("AIPTOK02 output descriptor disappeared"))?;
    file.set_permissions(fs::Permissions::from_mode(0o400))
        .and_then(|()| file.sync_all())
        .map_err(|_| EngineError::backend_failure("AIPTOK02 read-only publication failed"))?;
    guard.verify_current()?;
    guard
        .parent_file
        .as_ref()
        .ok_or_else(|| EngineError::internal_invariant("AIPTOK02 parent descriptor disappeared"))?
        .sync_all()
        .map_err(|_| EngineError::backend_failure("AIPTOK02 output parent sync failed"))?;
    guard.committed = true;
    Ok(())
}

struct UnpublishedOutput<'a> {
    path: &'a Path,
    parent: &'a Path,
    parent_file: Option<File>,
    parent_identity: FileIdentity,
    file: Option<File>,
    output_identity: FileIdentity,
    committed: bool,
}

impl UnpublishedOutput<'_> {
    fn verify_current(&self) -> Result<(), EngineError> {
        let parent_opened = self
            .parent_file
            .as_ref()
            .ok_or_else(|| {
                EngineError::internal_invariant("AIPTOK02 parent descriptor disappeared")
            })?
            .metadata()
            .map_err(|_| {
                EngineError::authentication_failure("AIPTOK02 parent descriptor changed")
            })?;
        let parent_path = fs::symlink_metadata(self.parent)
            .map_err(|_| EngineError::authentication_failure("AIPTOK02 parent path changed"))?;
        let output_opened = self
            .file
            .as_ref()
            .ok_or_else(|| {
                EngineError::internal_invariant("AIPTOK02 output descriptor disappeared")
            })?
            .metadata()
            .map_err(|_| {
                EngineError::authentication_failure("AIPTOK02 output descriptor changed")
            })?;
        let output_path = fs::symlink_metadata(self.path)
            .map_err(|_| EngineError::authentication_failure("AIPTOK02 output path changed"))?;
        if !parent_opened.is_dir()
            || !parent_path.is_dir()
            || parent_path.file_type().is_symlink()
            || FileIdentity::from_metadata(&parent_opened) != self.parent_identity
            || FileIdentity::from_metadata(&parent_path) != self.parent_identity
            || !output_opened.is_file()
            || !output_path.is_file()
            || output_path.file_type().is_symlink()
            || FileIdentity::from_metadata(&output_opened) != self.output_identity
            || FileIdentity::from_metadata(&output_path) != self.output_identity
        {
            return Err(EngineError::authentication_failure(
                "AIPTOK02 publication identity changed",
            ));
        }
        Ok(())
    }
}

impl Drop for UnpublishedOutput<'_> {
    fn drop(&mut self) {
        let _ = self.file.take();
        if !self.committed && self.path_is_original_output() {
            let _ = fs::remove_file(self.path);
        }
        let _ = self.parent_file.take();
    }
}

impl UnpublishedOutput<'_> {
    fn path_is_original_output(&self) -> bool {
        let Ok(parent) = fs::symlink_metadata(self.parent) else {
            return false;
        };
        let Ok(output) = fs::symlink_metadata(self.path) else {
            return false;
        };
        parent.is_dir()
            && !parent.file_type().is_symlink()
            && FileIdentity::from_metadata(&parent) == self.parent_identity
            && output.is_file()
            && !output.file_type().is_symlink()
            && FileIdentity::from_metadata(&output) == self.output_identity
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct FileIdentity {
    device: u64,
    inode: u64,
    uid: u32,
}

impl FileIdentity {
    fn from_metadata(metadata: &fs::Metadata) -> Self {
        Self { device: metadata.dev(), inode: metadata.ino(), uid: metadata.uid() }
    }
}

fn validate_absolute_normal_file_path(path: &Path, label: &str) -> Result<(), EngineError> {
    if !path.is_absolute()
        || path.file_name().is_none()
        || path.components().any(|component| {
            matches!(component, Component::Prefix(_) | Component::CurDir | Component::ParentDir)
        })
    {
        return Err(EngineError::invalid_input(format!(
            "{label} must be one absolute normalized file path"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_dir(label: &str) -> std::path::PathBuf {
        let unique =
            SystemTime::now().duration_since(UNIX_EPOCH).expect("clock follows epoch").as_nanos();
        let path = std::env::temp_dir()
            .join(format!("personalai-aiptok-export-{label}-{}-{unique}", std::process::id()));
        fs::create_dir(&path).expect("temporary directory creates");
        fs::set_permissions(&path, fs::Permissions::from_mode(0o700))
            .expect("temporary directory seals");
        path
    }

    #[test]
    fn output_creation_is_non_overwriting_and_cleans_failed_publication() {
        let root = temp_dir("fresh");
        let owner_uid = fs::symlink_metadata(&root).expect("root metadata").uid();
        let output = root.join("tokenizer.aiptok");
        let existing = b"owner-data";
        fs::write(&output, existing).expect("fixture writes");
        assert!(
            publish_new_read_only(
                &output,
                owner_uid,
                b"AIPTOK02wrong",
                Sha256Digest::from_bytes([1; 32]),
                ModelPackageHash::from_sha384_bytes([2; 48]),
            )
            .is_err()
        );
        assert_eq!(fs::read(&output).expect("owner data reads"), existing);

        let fresh = root.join("fresh.aiptok");
        assert!(
            publish_new_read_only(
                &fresh,
                owner_uid,
                b"AIPTOK02wrong",
                Sha256Digest::from_bytes([1; 32]),
                ModelPackageHash::from_sha384_bytes([2; 48]),
            )
            .is_err()
        );
        assert!(!fresh.exists());
        fs::remove_dir_all(root).expect("fixture removes");
    }

    #[test]
    fn paths_must_be_absolute_normalized_and_distinct() {
        assert!(validate_absolute_normal_file_path(Path::new("relative"), "source").is_err());
        let absolute = Path::new("/tmp/source");
        assert!(export_qwen25_aiptok(absolute, absolute).is_err());
    }
}
