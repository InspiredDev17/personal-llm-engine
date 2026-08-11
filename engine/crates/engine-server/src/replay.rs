//! Durable bounded replay rejection for authenticated control commands.

use crate::os_lock::ExclusiveFile;
use personalai_engine_types::{EngineError, SignerKeyId};
use std::collections::HashSet;
use std::fs::{self, File, OpenOptions};
use std::io::{ErrorKind, SeekFrom};
use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};
use std::path::{Path, PathBuf};

const REPLAY_MAGIC: [u8; 8] = *b"PAIRPL01";
const REPLAY_VERSION: u16 = 1;
const REPLAY_HEADER_BYTES: usize = 32;
const REPLAY_RECORD_BYTES: usize = 8;
/// Maximum authenticated commands retained for one control-key identity.
pub const MAX_DURABLE_REPLAY_ENTRIES: usize = 1_048_576;

/// Configuration-derived, append-only replay state for one control identity.
pub struct DurableReplayJournal {
    file: ExclusiveFile,
    path: PathBuf,
    seen: HashSet<u64>,
    maximum_entries: usize,
    usable: bool,
}

impl DurableReplayJournal {
    /// Opens or creates the journal inside an existing owner-private runtime root.
    pub fn open(runtime_root: &Path, key_id: SignerKeyId) -> Result<Self, EngineError> {
        Self::open_with_limit(runtime_root, key_id, MAX_DURABLE_REPLAY_ENTRIES)
    }

    fn open_with_limit(
        runtime_root: &Path,
        key_id: SignerKeyId,
        maximum_entries: usize,
    ) -> Result<Self, EngineError> {
        validate_runtime_root(runtime_root)?;
        if maximum_entries == 0 || maximum_entries > MAX_DURABLE_REPLAY_ENTRIES {
            return Err(EngineError::invalid_input("durable replay limit is invalid"));
        }
        let path = runtime_root.join(format!("control-replay-{key_id}.bin"));
        let mut file = match open_existing(&path) {
            Ok(file) => file,
            Err(error) if error.kind() == ErrorKind::NotFound => {
                match create_new(runtime_root, &path, key_id) {
                    Ok(file) => file,
                    Err(create_error) if create_error.kind() == ErrorKind::AlreadyExists => {
                        open_existing(&path).map_err(|_| {
                            EngineError::authentication_failure(
                                "durable replay journal cannot be opened safely",
                            )
                        })?
                    }
                    Err(_) => {
                        return Err(EngineError::backend_failure(
                            "durable replay journal creation failed",
                        ));
                    }
                }
            }
            Err(_) => {
                return Err(EngineError::authentication_failure(
                    "durable replay journal cannot be opened safely",
                ));
            }
        };
        let metadata = file.metadata().map_err(|_| {
            EngineError::backend_failure("durable replay journal metadata is unavailable")
        })?;
        if !metadata.is_file() || metadata.permissions().mode() & 0o777 != 0o600 {
            return Err(EngineError::authentication_failure(
                "durable replay journal must be a mode-0600 regular file",
            ));
        }
        let maximum_bytes = REPLAY_HEADER_BYTES
            .checked_add(maximum_entries.checked_mul(REPLAY_RECORD_BYTES).ok_or_else(|| {
                EngineError::internal_invariant("durable replay maximum length overflows")
            })?)
            .ok_or_else(|| {
                EngineError::internal_invariant("durable replay maximum length overflows")
            })?;
        let length = usize::try_from(metadata.len()).map_err(|_| {
            EngineError::authentication_failure("durable replay journal is not addressable")
        })?;
        if length < REPLAY_HEADER_BYTES
            || length > maximum_bytes
            || (length - REPLAY_HEADER_BYTES) % REPLAY_RECORD_BYTES != 0
        {
            return Err(EngineError::authentication_failure(
                "durable replay journal length is invalid",
            ));
        }
        file.seek(SeekFrom::Start(0))
            .map_err(|_| EngineError::backend_failure("durable replay journal seek failed"))?;
        let mut header = [0_u8; REPLAY_HEADER_BYTES];
        file.read_exact(&mut header).map_err(|_| {
            EngineError::authentication_failure("durable replay journal header is truncated")
        })?;
        if header[..8] != REPLAY_MAGIC
            || u16::from_le_bytes(
                header[8..10].try_into().map_err(|_| {
                    EngineError::internal_invariant("replay version is unavailable")
                })?,
            ) != REPLAY_VERSION
            || header[10..16] != [0_u8; 6]
            || header[16..32] != *key_id.as_bytes()
        {
            return Err(EngineError::authentication_failure(
                "durable replay journal identity is invalid",
            ));
        }
        let record_count = (length - REPLAY_HEADER_BYTES) / REPLAY_RECORD_BYTES;
        let mut seen = HashSet::new();
        seen.try_reserve(record_count).map_err(|_| {
            EngineError::resource_exhaustion("durable replay index allocation failed")
        })?;
        for _ in 0..record_count {
            let mut record = [0_u8; REPLAY_RECORD_BYTES];
            file.read_exact(&mut record).map_err(|_| {
                EngineError::authentication_failure("durable replay record is truncated")
            })?;
            let nonce = u64::from_le_bytes(record);
            if nonce == 0 || !seen.insert(nonce) {
                return Err(EngineError::authentication_failure(
                    "durable replay journal contains an invalid record",
                ));
            }
        }
        file.seek(SeekFrom::End(0))
            .map_err(|_| EngineError::backend_failure("durable replay journal seek failed"))?;
        Ok(Self { file, path, seen, maximum_entries, usable: true })
    }

    /// Durably records one authenticated nonce before command execution.
    pub fn admit(&mut self, nonce: u64) -> Result<(), EngineError> {
        if !self.usable {
            return Err(EngineError::authentication_failure(
                "durable replay journal is unavailable after a write failure",
            ));
        }
        if nonce == 0 {
            return Err(EngineError::protocol_violation(
                "authenticated command nonce must be nonzero",
            ));
        }
        if self.seen.contains(&nonce) {
            return Err(EngineError::authentication_failure(
                "authenticated command nonce was replayed",
            ));
        }
        if self.seen.len() >= self.maximum_entries {
            return Err(EngineError::resource_exhaustion(
                "durable replay journal is exhausted; rotate the control identity",
            ));
        }
        self.seen.try_reserve(1).map_err(|_| {
            EngineError::resource_exhaustion("durable replay index allocation failed")
        })?;
        self.usable = false;
        self.file
            .write_all(&nonce.to_le_bytes())
            .and_then(|()| self.file.sync_data())
            .map_err(|_| EngineError::backend_failure("durable replay append failed"))?;
        if !self.seen.insert(nonce) {
            return Err(EngineError::internal_invariant(
                "durable replay index changed during append",
            ));
        }
        self.usable = true;
        Ok(())
    }

    /// Returns the configuration-derived journal path.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Returns the number of durably admitted commands.
    pub fn entry_count(&self) -> usize {
        self.seen.len()
    }
}

fn validate_runtime_root(runtime_root: &Path) -> Result<(), EngineError> {
    let metadata = fs::symlink_metadata(runtime_root)
        .map_err(|_| EngineError::invalid_input("runtime root is unavailable"))?;
    if !metadata.is_dir()
        || metadata.file_type().is_symlink()
        || metadata.permissions().mode() & 0o077 != 0
    {
        return Err(EngineError::authentication_failure(
            "runtime root must be a non-symlink owner-private directory",
        ));
    }
    Ok(())
}

fn open_existing(path: &Path) -> std::io::Result<ExclusiveFile> {
    let file = OpenOptions::new()
        .read(true)
        .append(true)
        .custom_flags(libc::O_CLOEXEC | libc::O_NOFOLLOW)
        .open(path)?;
    ExclusiveFile::try_acquire(file)
}

fn create_new(
    runtime_root: &Path,
    path: &Path,
    key_id: SignerKeyId,
) -> std::io::Result<ExclusiveFile> {
    let file = OpenOptions::new()
        .read(true)
        .append(true)
        .create_new(true)
        .mode(0o600)
        .custom_flags(libc::O_CLOEXEC | libc::O_NOFOLLOW)
        .open(path)?;
    let mut file = ExclusiveFile::try_acquire(file)?;
    let mut header = [0_u8; REPLAY_HEADER_BYTES];
    header[..8].copy_from_slice(&REPLAY_MAGIC);
    header[8..10].copy_from_slice(&REPLAY_VERSION.to_le_bytes());
    header[16..32].copy_from_slice(key_id.as_bytes());
    file.write_all(&header)?;
    file.sync_all()?;
    File::open(runtime_root)?.sync_all()?;
    Ok(file)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_DIRECTORY: AtomicU64 = AtomicU64::new(1);

    #[test]
    fn journal_rejects_replay_across_reopen_and_is_key_scoped() {
        let root = temporary_directory();
        let key_id = SignerKeyId::from_bytes([0x71; 16]).expect("key ID must validate");
        let path;
        {
            let mut journal =
                DurableReplayJournal::open(&root, key_id).expect("journal must create");
            path = journal.path().to_path_buf();
            assert_eq!(journal.entry_count(), 0);
            journal.admit(11).expect("fresh nonce must persist");
            assert!(matches!(journal.admit(11), Err(EngineError::AuthenticationFailure(_))));
        }
        let metadata = fs::symlink_metadata(&path).expect("journal metadata must exist");
        assert!(metadata.is_file());
        assert_eq!(metadata.permissions().mode() & 0o777, 0o600);
        let mut reopened = DurableReplayJournal::open(&root, key_id).expect("journal must reopen");
        assert_eq!(reopened.entry_count(), 1);
        assert!(matches!(reopened.admit(11), Err(EngineError::AuthenticationFailure(_))));
        reopened.admit(12).expect("second nonce must persist");
        drop(reopened);
        let other_key_id = SignerKeyId::from_bytes([0x75; 16]).expect("key ID must validate");
        let mut other =
            DurableReplayJournal::open(&root, other_key_id).expect("other key journal must create");
        let other_path = other.path().to_path_buf();
        other.admit(11).expect("nonce namespace must be key scoped");
        drop(other);
        fs::remove_file(path).expect("journal must remove");
        fs::remove_file(other_path).expect("other key journal must remove");
        fs::remove_dir(root).expect("runtime root must remove");
    }

    #[test]
    fn corruption_partial_records_and_exhaustion_fail_closed() {
        let root = temporary_directory();
        let key_id = SignerKeyId::from_bytes([0x72; 16]).expect("key ID must validate");
        let mut journal =
            DurableReplayJournal::open_with_limit(&root, key_id, 2).expect("journal must create");
        let path = journal.path().to_path_buf();
        journal.admit(1).expect("first nonce must persist");
        journal.admit(2).expect("second nonce must persist");
        assert!(matches!(journal.admit(3), Err(EngineError::ResourceExhaustion(_))));
        drop(journal);
        OpenOptions::new()
            .append(true)
            .open(&path)
            .expect("journal must open for corruption fixture")
            .write_all(&[0xff])
            .expect("partial fixture must append");
        assert!(matches!(
            DurableReplayJournal::open_with_limit(&root, key_id, 3),
            Err(EngineError::AuthenticationFailure(_))
        ));
        fs::remove_file(path).expect("journal must remove");
        fs::remove_dir(root).expect("runtime root must remove");
    }

    #[test]
    fn one_control_key_journal_has_exactly_one_live_process_owner() {
        const CHILD_ROOT: &str = "PERSONALAI_TEST_REPLAY_LOCK_ROOT";
        let key_id = SignerKeyId::from_bytes([0x77; 16]).expect("key ID must validate");
        if let Some(root) = std::env::var_os(CHILD_ROOT) {
            assert!(matches!(
                DurableReplayJournal::open(Path::new(&root), key_id),
                Err(EngineError::AuthenticationFailure(_))
            ));
            return;
        }

        let root = temporary_directory();
        let journal = DurableReplayJournal::open(&root, key_id).expect("journal must create");
        let path = journal.path().to_path_buf();
        let child = std::process::Command::new(
            std::env::current_exe().expect("test executable must be available"),
        )
        .arg("replay::tests::one_control_key_journal_has_exactly_one_live_process_owner")
        .arg("--exact")
        .env(CHILD_ROOT, &root)
        .status()
        .expect("lock competitor must run");
        assert!(child.success(), "competitor must observe fail-closed exclusion");
        drop(journal);
        let reopened =
            DurableReplayJournal::open(&root, key_id).expect("released lock must be reusable");
        drop(reopened);
        fs::remove_file(path).expect("journal must remove");
        fs::remove_dir(root).expect("runtime root must remove");
    }

    #[test]
    fn permissive_runtime_roots_are_rejected() {
        let root = temporary_directory();
        fs::set_permissions(&root, fs::Permissions::from_mode(0o755))
            .expect("permissions must change");
        let key_id = SignerKeyId::from_bytes([0x73; 16]).expect("key ID must validate");
        assert!(matches!(
            DurableReplayJournal::open(&root, key_id),
            Err(EngineError::AuthenticationFailure(_))
        ));
        fs::remove_dir(root).expect("runtime root must remove");
    }

    #[test]
    fn symlink_runtime_roots_are_rejected() {
        use std::os::unix::fs::symlink;

        let root = temporary_directory();
        let link = root.with_extension("link");
        symlink(&root, &link).expect("runtime-root symlink must create");
        let key_id = SignerKeyId::from_bytes([0x74; 16]).expect("key ID must validate");
        assert!(matches!(
            DurableReplayJournal::open(&link, key_id),
            Err(EngineError::AuthenticationFailure(_))
        ));
        fs::remove_file(link).expect("runtime-root symlink must remove");
        fs::remove_dir(root).expect("runtime root must remove");
    }

    #[test]
    fn symlink_journal_files_are_rejected() {
        use std::os::unix::fs::symlink;

        let root = temporary_directory();
        let key_id = SignerKeyId::from_bytes([0x76; 16]).expect("key ID must validate");
        let path = root.join(format!("control-replay-{key_id}.bin"));
        symlink(root.join("missing-target"), &path).expect("journal symlink must create");
        assert!(matches!(
            DurableReplayJournal::open(&root, key_id),
            Err(EngineError::AuthenticationFailure(_))
        ));
        fs::remove_file(path).expect("journal symlink must remove");
        fs::remove_dir(root).expect("runtime root must remove");
    }

    fn temporary_directory() -> PathBuf {
        let sequence = NEXT_DIRECTORY.fetch_add(1, Ordering::Relaxed);
        let root = PathBuf::from("/tmp")
            .join(format!("personalai-replay-{}-{sequence}", std::process::id()));
        fs::create_dir(&root).expect("runtime root must create");
        fs::set_permissions(&root, fs::Permissions::from_mode(0o700))
            .expect("runtime permissions must set");
        root
    }
}
