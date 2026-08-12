//! Exact owner-selected Model Foundry registry V2 verification.

use personalai_engine_crypto::{
    TrustedSigner, hash_package, hash_signed_content, verify_model_foundry_registry_signature,
};
#[cfg(unix)]
use personalai_engine_model_format::MODEL_REGISTRY_MAX_BYTES;
use personalai_engine_model_format::{ModelRegistryEntryV2, SignedModelRegistryEntryV2};
use personalai_engine_types::{EngineError, ModelId, ModelPackageHash};
#[cfg(unix)]
use std::fs::{self, OpenOptions};
#[cfg(unix)]
use std::io::Read;
#[cfg(unix)]
use std::os::unix::fs::{MetadataExt, OpenOptionsExt, PermissionsExt};
#[cfg(unix)]
use std::path::Path;

/// Immutable owner configuration for one exact signed registry decision.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RegistrySelectionV2 {
    expected_model_id: ModelId,
    expected_artifact_id: String,
    expected_entry_hash: ModelPackageHash,
    expected_sequence: u64,
    minimum_sequence: u64,
    registry_signer: TrustedSigner,
    package_signer: TrustedSigner,
}

impl RegistrySelectionV2 {
    /// Constructs a bounded exact selection with separate public trust roles.
    pub fn new(
        expected_model_id: ModelId,
        expected_artifact_id: impl Into<String>,
        expected_entry_hash: ModelPackageHash,
        expected_sequence: u64,
        minimum_sequence: u64,
        registry_signer: TrustedSigner,
        package_signer: TrustedSigner,
    ) -> Result<Self, EngineError> {
        let expected_artifact_id = expected_artifact_id.into();
        validate_artifact_id(&expected_artifact_id)?;
        if minimum_sequence == 0 || expected_sequence < minimum_sequence {
            return Err(EngineError::invalid_input(
                "registry sequence and anti-rollback checkpoint are invalid",
            ));
        }
        if !registry_signer.is_active() || !package_signer.is_active() {
            return Err(EngineError::authentication_failure(
                "registry selection requires active public trust identities",
            ));
        }
        Ok(Self {
            expected_model_id,
            expected_artifact_id,
            expected_entry_hash,
            expected_sequence,
            minimum_sequence,
            registry_signer,
            package_signer,
        })
    }

    /// Returns the independently retained exact entry identity.
    pub const fn expected_entry_hash(&self) -> ModelPackageHash {
        self.expected_entry_hash
    }

    /// Returns the exact owner-selected registry sequence.
    pub const fn expected_sequence(&self) -> u64 {
        self.expected_sequence
    }

    /// Returns the exact owner-selected model identity.
    pub const fn expected_model_id(&self) -> &ModelId {
        &self.expected_model_id
    }

    /// Returns the exact owner-selected artifact identity.
    pub fn expected_artifact_id(&self) -> &str {
        &self.expected_artifact_id
    }

    /// Returns the independently retained nondecreasing sequence floor.
    pub const fn minimum_sequence(&self) -> u64 {
        self.minimum_sequence
    }
}

/// Cryptographically verified registry decision suitable for registration.
///
/// Its fields cannot be constructed from unsigned entry data. The type is
/// produced only after exact owner selection, anti-rollback, role separation,
/// and strict signature verification succeed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VerifiedRegistrySelectionV2 {
    entry: ModelRegistryEntryV2,
    entry_hash: ModelPackageHash,
    package_signer: TrustedSigner,
}

impl VerifiedRegistrySelectionV2 {
    /// Returns the verified signed entry.
    pub const fn entry(&self) -> &ModelRegistryEntryV2 {
        &self.entry
    }

    /// Returns the exact independently selected envelope identity.
    pub const fn entry_hash(&self) -> ModelPackageHash {
        self.entry_hash
    }

    /// Returns the active package signer bound by owner trust and the entry.
    pub const fn package_signer(&self) -> &TrustedSigner {
        &self.package_signer
    }
}

/// Parses and verifies one exact owner-selected registry decision.
///
/// This function does not activate, locate, or load a package. It establishes
/// only the signed selection data needed by a later reviewed registration path.
pub fn verify_registry_selection_v2(
    envelope: &[u8],
    selection: &RegistrySelectionV2,
) -> Result<VerifiedRegistrySelectionV2, EngineError> {
    if hash_package(envelope) != selection.expected_entry_hash {
        return Err(EngineError::authentication_failure(
            "model registry entry identity differs from owner configuration",
        ));
    }
    let signed = SignedModelRegistryEntryV2::parse(envelope)?;
    let entry = signed.entry();
    if entry.registry_sequence() != selection.expected_sequence
        || entry.registry_sequence() < selection.minimum_sequence
    {
        return Err(EngineError::authentication_failure(
            "model registry sequence differs from owner selection or anti-rollback checkpoint",
        ));
    }
    if entry.model_id() != selection.expected_model_id.as_str()
        || entry.artifact_id() != selection.expected_artifact_id
    {
        return Err(EngineError::authentication_failure(
            "model registry identities differ from owner selection",
        ));
    }
    if entry.registry_signer() != selection.registry_signer.key_id() {
        return Err(EngineError::signature_failure(
            "signed registry identity differs from owner trust",
        ));
    }
    if entry.package_signer() != selection.package_signer.key_id() {
        return Err(EngineError::signature_failure(
            "signed package identity differs from owner trust",
        ));
    }
    let digest = hash_signed_content(signed.entry_bytes(envelope)?);
    verify_model_foundry_registry_signature(
        &selection.registry_signer,
        entry.registry_signer(),
        &digest,
        signed.signature(),
    )?;
    Ok(VerifiedRegistrySelectionV2 {
        entry: entry.clone(),
        entry_hash: selection.expected_entry_hash,
        package_signer: selection.package_signer.clone(),
    })
}

/// Securely reads and verifies the exact sequence-derived registry entry.
#[cfg(unix)]
pub(crate) fn read_verified_registry_selection_v2(
    approved_root: &Path,
    selection: &RegistrySelectionV2,
) -> Result<VerifiedRegistrySelectionV2, EngineError> {
    let approved_before = fs::symlink_metadata(approved_root)
        .map_err(|_| EngineError::invalid_input("Model Foundry approved root is unavailable"))?;
    if !approved_before.is_dir()
        || approved_before.file_type().is_symlink()
        || approved_before.permissions().mode() & 0o222 != 0
    {
        return Err(EngineError::authentication_failure(
            "Model Foundry approved root must be a protected non-symlink directory",
        ));
    }
    let registry_root = approved_root.join("registry");
    let directory_before = fs::symlink_metadata(&registry_root)
        .map_err(|_| EngineError::invalid_input("Model Foundry registry root is unavailable"))?;
    if !directory_before.is_dir()
        || directory_before.file_type().is_symlink()
        || directory_before.permissions().mode() & 0o222 != 0
    {
        return Err(EngineError::authentication_failure(
            "Model Foundry registry root must be a protected non-symlink directory",
        ));
    }
    let name = format!("{:020}.registry.json", selection.expected_sequence);
    let path = registry_root.join(name);
    let before = fs::symlink_metadata(&path).map_err(|_| {
        EngineError::invalid_input("selected Model Foundry registry entry is absent")
    })?;
    if !before.is_file()
        || before.file_type().is_symlink()
        || before.permissions().mode() & 0o222 != 0
        || before.len() == 0
        || before.len() > u64::try_from(MODEL_REGISTRY_MAX_BYTES).unwrap_or(u64::MAX)
    {
        return Err(EngineError::authentication_failure(
            "selected Model Foundry registry entry metadata is invalid",
        ));
    }
    let mut file = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_CLOEXEC | libc::O_NOFOLLOW)
        .open(path)
        .map_err(|_| EngineError::authentication_failure("registry entry secure open failed"))?;
    let opened = file.metadata().map_err(|_| {
        EngineError::authentication_failure("opened registry entry metadata failed")
    })?;
    if !opened.is_file()
        || opened.dev() != before.dev()
        || opened.ino() != before.ino()
        || opened.len() != before.len()
    {
        return Err(EngineError::authentication_failure(
            "selected registry entry changed during secure open",
        ));
    }
    let length = usize::try_from(opened.len())
        .map_err(|_| EngineError::resource_exhaustion("registry entry is not addressable"))?;
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(length)
        .map_err(|_| EngineError::resource_exhaustion("registry entry allocation failed"))?;
    bytes.resize(length, 0);
    file.read_exact(&mut bytes)
        .map_err(|_| EngineError::corrupt_model_package("registry entry is truncated"))?;
    let mut trailing = [0_u8; 1];
    if file
        .read(&mut trailing)
        .map_err(|_| EngineError::corrupt_model_package("registry entry trailing check failed"))?
        != 0
    {
        return Err(EngineError::corrupt_model_package("registry entry grew during secure read"));
    }
    let directory_after = fs::symlink_metadata(&registry_root)
        .map_err(|_| EngineError::authentication_failure("registry root changed during read"))?;
    if directory_after.dev() != directory_before.dev()
        || directory_after.ino() != directory_before.ino()
    {
        return Err(EngineError::authentication_failure("registry root changed during read"));
    }
    let approved_after = fs::symlink_metadata(approved_root)
        .map_err(|_| EngineError::authentication_failure("approved root changed during read"))?;
    if approved_after.dev() != approved_before.dev()
        || approved_after.ino() != approved_before.ino()
    {
        return Err(EngineError::authentication_failure("approved root changed during read"));
    }
    verify_registry_selection_v2(&bytes, selection)
}

fn validate_artifact_id(value: &str) -> Result<(), EngineError> {
    let bytes = value.as_bytes();
    if bytes.is_empty()
        || bytes.len() > 128
        || !matches!(bytes[0], b'a'..=b'z' | b'0'..=b'9')
        || bytes.iter().any(|byte| !matches!(byte, b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.'))
        || value.contains("..")
    {
        return Err(EngineError::invalid_input(
            "registry artifact ID is outside the safe canonical grammar",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use personalai_engine_types::SignerKeyId;

    const POSITIVE_FILE: &[u8] = include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/model-foundry-registry-v2/positive-0002.registry.json"
    ));

    fn positive() -> &'static [u8] {
        POSITIVE_FILE.strip_suffix(b"\n").unwrap_or(POSITIVE_FILE)
    }

    #[test]
    fn exact_active_selection_verifies_without_loading_a_package() {
        let selection = selection(2, hash_package(positive()));
        let entry = verify_registry_selection_v2(positive(), &selection)
            .expect("exact registry decision must verify");
        assert_eq!(entry.entry().model_id(), "synthetic-import");
        assert_eq!(entry.entry().artifact_id(), "phase0-synthetic-import");
        assert_eq!(
            entry.entry().package_catalog_hash(),
            ModelPackageHash::from_sha384_hex(
                "5b97582279ce8e9dc1ed114b0b46b243fd1f6655f9ed694def28c1e52cbfeffd\
                 f1eb5e27f19fab5c7529b91d6471b329",
            )
            .expect("catalog hash must parse"),
        );
    }

    #[test]
    fn mutation_rollback_wrong_roles_and_revocation_fail_closed() {
        let exact_hash = hash_package(positive());
        let exact_selection = selection(2, exact_hash);
        let mut corrupt = positive().to_vec();
        let marker = b"\"signature\":\"";
        let signature = corrupt
            .windows(marker.len())
            .position(|window| window == marker)
            .expect("signature marker")
            + marker.len();
        corrupt[signature] ^= 1;
        assert!(verify_registry_selection_v2(&corrupt, &exact_selection).is_err());
        assert!(
            RegistrySelectionV2::new(
                ModelId::new("synthetic-import").expect("model ID"),
                "phase0-synthetic-import",
                exact_hash,
                2,
                3,
                active(0x63, registry_public()),
                active(0x51, package_public()),
            )
            .is_err()
        );

        let wrong_registry = RegistrySelectionV2::new(
            ModelId::new("synthetic-import").expect("model ID"),
            "phase0-synthetic-import",
            exact_hash,
            2,
            2,
            active(0x65, registry_public()),
            active(0x51, package_public()),
        )
        .expect("selection constructs");
        assert!(verify_registry_selection_v2(positive(), &wrong_registry).is_err());
        assert!(
            RegistrySelectionV2::new(
                ModelId::new("synthetic-import").expect("model ID"),
                "phase0-synthetic-import",
                exact_hash,
                2,
                2,
                TrustedSigner::revoked(key(0x63), registry_public()).expect("revoked key"),
                active(0x51, package_public()),
            )
            .is_err()
        );
    }

    fn selection(sequence: u64, hash: ModelPackageHash) -> RegistrySelectionV2 {
        RegistrySelectionV2::new(
            ModelId::new("synthetic-import").expect("model ID"),
            "phase0-synthetic-import",
            hash,
            2,
            sequence,
            active(0x63, registry_public()),
            active(0x51, package_public()),
        )
        .expect("selection must construct")
    }

    fn active(id: u8, public_key: [u8; 32]) -> TrustedSigner {
        TrustedSigner::active(key(id), public_key).expect("trusted key must construct")
    }

    fn key(value: u8) -> SignerKeyId {
        SignerKeyId::from_bytes([value; 16]).expect("key ID must construct")
    }

    fn registry_public() -> [u8; 32] {
        decode("2bc2800b3316e009209ffd757dab19ccf0ae84bc7ae90654e1e81712d270f653")
    }

    fn package_public() -> [u8; 32] {
        decode("2012cb90ca60e8e5d8daf66e2272d2233e0486d557e8c66141ed8920177d7eb7")
    }

    fn decode(value: &str) -> [u8; 32] {
        let mut bytes = [0_u8; 32];
        for (index, pair) in value.as_bytes().chunks_exact(2).enumerate() {
            let digit = |byte: u8| match byte {
                b'0'..=b'9' => byte - b'0',
                b'a'..=b'f' => byte - b'a' + 10,
                _ => panic!("fixture hex is invalid"),
            };
            bytes[index] = (digit(pair[0]) << 4) | digit(pair[1]);
        }
        bytes
    }
}
