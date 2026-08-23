//! Transitional, engine-private parser for development-only Qwen registry V3.
//!
//! Sovereign Models 2.0 owns the execution-package descriptor. This record is
//! deliberately narrower: it binds one separately signed Model Foundry
//! development decision to that canonical descriptor and to one signed
//! `PAIDIR01` catalog. It is not a canonical cross-repository contract and it
//! can never authorize production activation.

use crate::registry::{Parser, corrupt, require_canonical_timestamp};
use personalai_engine_types::{EngineError, ModelPackageHash, SignerKeyId};

/// Maximum accepted signed Qwen registry-envelope bytes.
pub const QWEN_REGISTRY_V3_MAX_BYTES: usize = 16 * 1024;
/// Exact local transitional registry profile.
pub const QWEN_REGISTRY_SCHEMA_V3: u64 = 3;
/// Exact development-only execution compatibility represented by V3.
pub const QWEN_REGISTRY_ENGINE_COMPATIBILITY: &str =
    "personalai.directory_paimodel+qwen2.5+aiptok02+bf16+arm64-v1";
/// Exact minimum engine release encoded by the Qwen V3 profile.
pub const QWEN_REGISTRY_MINIMUM_ENGINE_VERSION: &str = "0.2.0";
/// Exact non-production lifecycle status.
pub const QWEN_REGISTRY_DEVELOPMENT_STATUS: &str = "development_fixture_authorized";

/// One strictly parsed development-only Qwen registry decision.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct QwenRegistryEntryV3 {
    approval_timestamp: String,
    artifact_id: String,
    descriptor_hash: ModelPackageHash,
    model_id: String,
    package_catalog_hash: ModelPackageHash,
    package_signer: SignerKeyId,
    registry_sequence: u64,
    registry_signer: SignerKeyId,
}

impl QwenRegistryEntryV3 {
    /// Returns the canonical approval timestamp.
    pub fn approval_timestamp(&self) -> &str {
        &self.approval_timestamp
    }

    /// Returns the canonical artifact identity.
    pub fn artifact_id(&self) -> &str {
        &self.artifact_id
    }

    /// Returns SHA-384 of the exact canonical Models 2.0 descriptor bytes.
    pub const fn descriptor_hash(&self) -> ModelPackageHash {
        self.descriptor_hash
    }

    /// Returns the provider/execution model identity.
    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    /// Returns SHA-384 of exact `hashes.aiph` bytes.
    pub const fn package_catalog_hash(&self) -> ModelPackageHash {
        self.package_catalog_hash
    }

    /// Returns the selected signed-package trust identity.
    pub const fn package_signer(&self) -> SignerKeyId {
        self.package_signer
    }

    /// Returns the monotonic development registry sequence.
    pub const fn registry_sequence(&self) -> u64 {
        self.registry_sequence
    }

    /// Returns the selected registry-signing identity.
    pub const fn registry_signer(&self) -> SignerKeyId {
        self.registry_signer
    }

    /// Always reports true for a valid V3 record. A separate owner startup
    /// gate is still mandatory before execution.
    pub const fn development_execution_authorized(&self) -> bool {
        true
    }

    /// Always reports false. V3 has no production activation semantics.
    pub const fn production_activation(&self) -> bool {
        false
    }
}

/// One exact canonical signed Qwen registry envelope.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SignedQwenRegistryEntryV3 {
    entry: QwenRegistryEntryV3,
    entry_range: std::ops::Range<usize>,
    signature: [u8; 64],
}

impl SignedQwenRegistryEntryV3 {
    /// Parses one byte-exact, closed V3 envelope.
    pub fn parse(bytes: &[u8]) -> Result<Self, EngineError> {
        if bytes.is_empty() || bytes.len() > QWEN_REGISTRY_V3_MAX_BYTES {
            return Err(corrupt("Qwen registry envelope length is outside hard bounds"));
        }
        let mut parser = Parser::new(bytes);
        parser.expect(b"{\"entry\":")?;
        let entry_start = parser.position();
        let entry = parse_entry(&mut parser)?;
        let entry_end = parser.position();
        parser.expect(b",\"signature\":\"")?;
        let signature = parser.hex_array::<64>("Qwen registry signature")?;
        parser.expect(b"\"}")?;
        if !parser.is_finished() {
            return Err(corrupt("Qwen registry envelope contains trailing bytes"));
        }
        Ok(Self { entry, entry_range: entry_start..entry_end, signature })
    }

    /// Returns the validated unsigned entry.
    pub const fn entry(&self) -> &QwenRegistryEntryV3 {
        &self.entry
    }

    /// Returns exact canonical unsigned entry bytes from the envelope.
    pub fn entry_bytes<'a>(&self, envelope: &'a [u8]) -> Result<&'a [u8], EngineError> {
        envelope
            .get(self.entry_range.clone())
            .ok_or_else(|| corrupt("Qwen registry entry range is unavailable"))
    }

    /// Returns exact Ed25519 signature bytes.
    pub const fn signature(&self) -> &[u8; 64] {
        &self.signature
    }
}

fn parse_entry(parser: &mut Parser<'_>) -> Result<QwenRegistryEntryV3, EngineError> {
    parser.expect(b"{\"approval_timestamp\":\"")?;
    let approval_timestamp = parser.ascii_string(20, "approval timestamp")?;
    require_canonical_timestamp(&approval_timestamp)?;
    parser.expect(b"\",\"artifact_id\":\"")?;
    let artifact_id = parser.identifier(128, "artifact ID")?;
    parser.expect(b"\",\"descriptor_sha384\":\"")?;
    let descriptor_hash = parser.nonzero_hash("execution descriptor")?;
    parser.expect(b"\",\"development_execution_authorized\":true")?;
    parser.expect(b",\"engine_compatibility\":\"")?;
    parser.exact_string(QWEN_REGISTRY_ENGINE_COMPATIBILITY, "engine compatibility")?;
    parser.expect(b"\",\"minimum_engine_version\":\"")?;
    parser.exact_string(QWEN_REGISTRY_MINIMUM_ENGINE_VERSION, "minimum engine version")?;
    parser.expect(b"\",\"model_id\":\"")?;
    let model_id = parser.identifier(128, "model ID")?;
    parser.expect(b"\",\"package_catalog_sha384\":\"")?;
    let package_catalog_hash = parser.nonzero_hash("package catalog")?;
    parser.expect(b"\",\"package_signer\":\"")?;
    let package_signer = parser.key_id("package signer")?;
    parser.expect(b"\",\"production_activation\":false")?;
    parser.expect(b",\"registry_sequence\":")?;
    let registry_sequence = parser.positive_u64("registry sequence")?;
    parser.expect(b",\"registry_signer\":\"")?;
    let registry_signer = parser.key_id("registry signer")?;
    parser.expect(b"\",\"schema_version\":")?;
    if parser.positive_u64("registry schema")? != QWEN_REGISTRY_SCHEMA_V3 {
        return Err(corrupt("Qwen registry schema version is unsupported"));
    }
    parser.expect(b",\"status\":\"")?;
    parser.exact_string(QWEN_REGISTRY_DEVELOPMENT_STATUS, "Qwen registry status")?;
    parser.expect(b"\"}")?;
    Ok(QwenRegistryEntryV3 {
        approval_timestamp,
        artifact_id,
        descriptor_hash,
        model_id,
        package_catalog_hash,
        package_signer,
        registry_sequence,
        registry_signer,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hash(byte: u8) -> String {
        format!("{byte:02x}").repeat(48)
    }

    fn fixture() -> Vec<u8> {
        format!(
            "{{\"entry\":{{\"approval_timestamp\":\"2026-08-23T12:00:00Z\",\"artifact_id\":\"art_qwen25_general_1_5b_989aa798\",\"descriptor_sha384\":\"{}\",\"development_execution_authorized\":true,\"engine_compatibility\":\"{}\",\"minimum_engine_version\":\"{}\",\"model_id\":\"qwen2.5-1.5b-instruct-989aa798\",\"package_catalog_sha384\":\"{}\",\"package_signer\":\"{}\",\"production_activation\":false,\"registry_sequence\":1,\"registry_signer\":\"{}\",\"schema_version\":3,\"status\":\"{}\"}},\"signature\":\"{}\"}}",
            hash(0x11),
            QWEN_REGISTRY_ENGINE_COMPATIBILITY,
            QWEN_REGISTRY_MINIMUM_ENGINE_VERSION,
            hash(0x22),
            "33".repeat(16),
            "44".repeat(16),
            QWEN_REGISTRY_DEVELOPMENT_STATUS,
            "55".repeat(64),
        )
        .into_bytes()
    }

    #[test]
    fn exact_development_fixture_parses_without_production_authority() {
        let bytes = fixture();
        let parsed = SignedQwenRegistryEntryV3::parse(&bytes).expect("fixture parses");
        assert!(parsed.entry().development_execution_authorized());
        assert!(!parsed.entry().production_activation());
        assert_eq!(parsed.entry().registry_sequence(), 1);
        assert_eq!(
            parsed.entry().descriptor_hash(),
            ModelPackageHash::from_sha384_hex(&hash(0x11)).unwrap()
        );
        assert_eq!(
            parsed.entry().package_catalog_hash(),
            ModelPackageHash::from_sha384_hex(&hash(0x22)).unwrap()
        );
        assert_eq!(parsed.entry_bytes(&bytes).unwrap()[0], b'{');
    }

    #[test]
    fn mutation_truncation_and_authority_widening_fail_closed() {
        let bytes = fixture();
        for end in 0..bytes.len() {
            assert!(SignedQwenRegistryEntryV3::parse(&bytes[..end]).is_err());
        }
        let production = String::from_utf8(bytes.clone())
            .unwrap()
            .replace("production_activation\":false", "production_activation\":true");
        assert!(SignedQwenRegistryEntryV3::parse(production.as_bytes()).is_err());
        let disabled = String::from_utf8(bytes).unwrap().replace(
            "development_execution_authorized\":true",
            "development_execution_authorized\":false",
        );
        assert!(SignedQwenRegistryEntryV3::parse(disabled.as_bytes()).is_err());
    }
}
