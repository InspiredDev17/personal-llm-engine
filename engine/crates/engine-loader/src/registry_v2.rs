//! Exact owner-selected Model Foundry registry V2 verification.

use personalai_engine_crypto::{
    TrustedSigner, hash_package, hash_signed_content, verify_model_foundry_registry_signature,
};
use personalai_engine_model_format::{ModelRegistryEntryV2, SignedModelRegistryEntryV2};
use personalai_engine_types::{EngineError, ModelPackageHash};

/// Immutable owner configuration for one exact signed registry decision.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RegistrySelectionV2 {
    expected_entry_hash: ModelPackageHash,
    minimum_sequence: u64,
    registry_signer: TrustedSigner,
    package_signer: TrustedSigner,
}

impl RegistrySelectionV2 {
    /// Constructs a bounded exact selection with separate public trust roles.
    pub fn new(
        expected_entry_hash: ModelPackageHash,
        minimum_sequence: u64,
        registry_signer: TrustedSigner,
        package_signer: TrustedSigner,
    ) -> Result<Self, EngineError> {
        if minimum_sequence == 0 {
            return Err(EngineError::invalid_input("registry minimum sequence must be nonzero"));
        }
        if !registry_signer.is_active() || !package_signer.is_active() {
            return Err(EngineError::authentication_failure(
                "registry selection requires active public trust identities",
            ));
        }
        Ok(Self { expected_entry_hash, minimum_sequence, registry_signer, package_signer })
    }

    /// Returns the independently retained exact entry identity.
    pub const fn expected_entry_hash(&self) -> ModelPackageHash {
        self.expected_entry_hash
    }

    /// Returns the independently retained nondecreasing sequence floor.
    pub const fn minimum_sequence(&self) -> u64 {
        self.minimum_sequence
    }
}

/// Parses and verifies one exact owner-selected registry decision.
///
/// This function does not activate, locate, or load a package. It establishes
/// only the signed selection data needed by a later reviewed registration path.
pub fn verify_registry_selection_v2(
    envelope: &[u8],
    selection: &RegistrySelectionV2,
) -> Result<ModelRegistryEntryV2, EngineError> {
    if hash_package(envelope) != selection.expected_entry_hash {
        return Err(EngineError::authentication_failure(
            "model registry entry identity differs from owner configuration",
        ));
    }
    let signed = SignedModelRegistryEntryV2::parse(envelope)?;
    let entry = signed.entry();
    if entry.registry_sequence() < selection.minimum_sequence {
        return Err(EngineError::authentication_failure(
            "model registry sequence is below the retained anti-rollback checkpoint",
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
    Ok(entry.clone())
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
        assert_eq!(entry.model_id(), "synthetic-import");
        assert_eq!(entry.artifact_id(), "phase0-synthetic-import");
        assert_eq!(
            entry.package_catalog_hash(),
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
        assert!(verify_registry_selection_v2(positive(), &selection(3, exact_hash)).is_err());

        let wrong_registry = RegistrySelectionV2::new(
            exact_hash,
            2,
            active(0x65, registry_public()),
            active(0x51, package_public()),
        )
        .expect("selection constructs");
        assert!(verify_registry_selection_v2(positive(), &wrong_registry).is_err());
        assert!(
            RegistrySelectionV2::new(
                exact_hash,
                2,
                TrustedSigner::revoked(key(0x63), registry_public()).expect("revoked key"),
                active(0x51, package_public()),
            )
            .is_err()
        );
    }

    fn selection(sequence: u64, hash: ModelPackageHash) -> RegistrySelectionV2 {
        RegistrySelectionV2::new(
            hash,
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
