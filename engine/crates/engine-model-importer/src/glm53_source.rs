//! Exact, non-executable source preflight for the reviewed GLM-5.3-Flash release.
//!
//! This module accepts only the byte-identical metadata files from one pinned
//! upstream revision. Exact digests make the large tokenizer and tensor index
//! data-only inputs without adding a permissive third-party JSON/tokenizer
//! loader. Successful preflight is content-free evidence; it neither converts
//! weights nor creates, signs, qualifies, registers, or activates a package.

use personalai_engine_crypto::hash_sha256;
use personalai_engine_types::{ArchitectureId, EngineError, Sha256Digest};

/// Reviewed upstream repository identity.
pub const GLM53_FLASH_SOURCE_REPOSITORY: &str = "zai-org/GLM-5.3-Flash";
/// Immutable reviewed upstream revision.
pub const GLM53_FLASH_SOURCE_REVISION: &str = "3f1971b7b5f7a528c9c4ef6212c8785298a8c24a";
/// Internal, experimental architecture identifier. No package/runtime decoder implements it.
pub const GLM53_FLASH_ARCHITECTURE_ID: &str = "glm5-next-flash-experimental-v1";

/// Exact upstream configuration length.
pub const GLM53_CONFIG_BYTES: usize = 69_416;
/// Exact upstream tokenizer length.
pub const GLM53_TOKENIZER_BYTES: usize = 20_217_442;
/// Exact upstream processor configuration length.
pub const GLM53_PROCESSOR_CONFIG_BYTES: usize = 909;
/// Exact upstream sharded safetensors index length.
pub const GLM53_MODEL_INDEX_BYTES: usize = 8_406_613;
/// Exact number of safetensors shards described by the reviewed source.
pub const GLM53_MODEL_SHARD_COUNT: u32 = 62;
/// Exact source payload size reported by the reviewed safetensors index.
pub const GLM53_MODEL_TENSOR_BYTES: u64 = 328_326_771_576;
/// Exact number of tensors described by the reviewed safetensors index.
pub const GLM53_MODEL_TENSOR_COUNT: u32 = 76_108;

/// Text vocabulary rows expected by the model graph.
pub const GLM53_MODEL_VOCABULARY_SIZE: u32 = 154_880;
/// Base BPE vocabulary entries in the pinned tokenizer.
pub const GLM53_TOKENIZER_BPE_VOCABULARY_SIZE: u32 = 154_820;
/// Added-token entries in the pinned tokenizer.
pub const GLM53_TOKENIZER_ADDED_TOKEN_COUNT: u32 = 36;
/// Merge rules in the pinned tokenizer.
pub const GLM53_TOKENIZER_MERGE_COUNT: u32 = 321_649;
/// Stable identity for the exact no-normalizer, regex-split ByteLevel BPE graph.
pub const GLM53_TOKENIZER_PROFILE_ID: &str = "glm5-next-bytelevel-bpe-regex-v1";
/// Exact image/vision patch width and height.
pub const GLM53_IMAGE_PATCH_SIZE: u32 = 14;
/// Exact temporal patch size.
pub const GLM53_TEMPORAL_PATCH_SIZE: u32 = 2;
/// Exact spatial merge size.
pub const GLM53_SPATIAL_MERGE_SIZE: u32 = 2;
/// Exact minimum image token budget declared by the processor.
pub const GLM53_MIN_IMAGE_TOKENS: u32 = 16;
/// Exact maximum image token budget declared by the processor.
pub const GLM53_MAX_IMAGE_TOKENS: u32 = 8_000;
/// Exact maximum video token budget declared by the processor.
pub const GLM53_MAX_VIDEO_TOKENS: u32 = 240_000;
/// Exact default video sampling rate declared by the processor.
pub const GLM53_VIDEO_FRAMES_PER_SECOND: u32 = 2;
/// Exact FP8 dynamic quantization block rows.
pub const GLM53_FP8_BLOCK_ROWS: u32 = 128;
/// Exact FP8 dynamic quantization block columns.
pub const GLM53_FP8_BLOCK_COLUMNS: u32 = 128;

const CONFIG_SHA256: &str = "bb8f01c42cb92a52ca72e65afb4d5bd8d11aef083cd210e8de25dfb904f23e9f";
const TOKENIZER_SHA256: &str = "19e773648cb4e65de8660ea6365e10acca112d42a854923df93db4a6f333a82d";
const PROCESSOR_CONFIG_SHA256: &str =
    "aae38374c94b08cc9b0547c6e64f05b951bd9735cea571c6988f5ed552bed3ed";
const MODEL_INDEX_SHA256: &str = "3c3f40366a53c3fd7974b4eab7881a365a98c2a4329150befebab99fe7c18b05";

/// Exact metadata set required for offline source inspection.
#[derive(Clone, Copy)]
pub struct Glm53MetadataFiles<'a> {
    /// Exact `config.json` bytes.
    pub config_json: &'a [u8],
    /// Exact `tokenizer.json` bytes.
    pub tokenizer_json: &'a [u8],
    /// Exact `processor_config.json` bytes.
    pub processor_config_json: &'a [u8],
    /// Exact `model.safetensors.index.json` bytes.
    pub model_index_json: &'a [u8],
}

/// Content-free evidence that the pinned metadata set was observed exactly.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Glm53SourcePreflightReport {
    architecture_id: ArchitectureId,
    config_sha256: Sha256Digest,
    tokenizer_sha256: Sha256Digest,
    processor_config_sha256: Sha256Digest,
    model_index_sha256: Sha256Digest,
}

impl Glm53SourcePreflightReport {
    /// Returns the non-executable experimental architecture identity.
    pub const fn architecture_id(&self) -> &ArchitectureId {
        &self.architecture_id
    }

    /// Returns the exact reviewed configuration digest.
    pub const fn config_sha256(&self) -> Sha256Digest {
        self.config_sha256
    }

    /// Returns the exact reviewed tokenizer digest.
    pub const fn tokenizer_sha256(&self) -> Sha256Digest {
        self.tokenizer_sha256
    }

    /// Returns the exact reviewed multimodal processor digest.
    pub const fn processor_config_sha256(&self) -> Sha256Digest {
        self.processor_config_sha256
    }

    /// Returns the exact reviewed shard-index digest.
    pub const fn model_index_sha256(&self) -> Sha256Digest {
        self.model_index_sha256
    }

    /// Returns the immutable reviewed upstream revision.
    pub const fn source_revision(&self) -> &'static str {
        GLM53_FLASH_SOURCE_REVISION
    }

    /// Returns the exact reviewed source shard count.
    pub const fn model_shard_count(&self) -> u32 {
        GLM53_MODEL_SHARD_COUNT
    }

    /// Returns the exact indexed source tensor count.
    pub const fn model_tensor_count(&self) -> u32 {
        GLM53_MODEL_TENSOR_COUNT
    }

    /// Returns the exact indexed source tensor payload size.
    pub const fn model_tensor_bytes(&self) -> u64 {
        GLM53_MODEL_TENSOR_BYTES
    }

    /// Always fails closed because no first-party decoder implements this profile.
    pub fn require_executable(&self) -> Result<(), EngineError> {
        Err(EngineError::unsupported_architecture(self.architecture_id.clone()))
    }
}

/// Verifies the four exact reviewed metadata artifacts with bounded work.
///
/// The cancellation probe is checked before and after every bounded digest.
/// No source bytes or source paths are retained in the returned report.
pub fn preflight_glm53_flash_metadata(
    files: Glm53MetadataFiles<'_>,
    is_cancelled: &impl Fn() -> bool,
) -> Result<Glm53SourcePreflightReport, EngineError> {
    check_cancelled(is_cancelled)?;
    let config_sha256 = verify_exact_artifact(
        "GLM-5.3-Flash configuration",
        files.config_json,
        GLM53_CONFIG_BYTES,
        CONFIG_SHA256,
    )?;
    check_cancelled(is_cancelled)?;
    let tokenizer_sha256 = verify_exact_artifact(
        "GLM-5.3-Flash tokenizer",
        files.tokenizer_json,
        GLM53_TOKENIZER_BYTES,
        TOKENIZER_SHA256,
    )?;
    check_cancelled(is_cancelled)?;
    let processor_config_sha256 = verify_exact_artifact(
        "GLM-5.3-Flash processor configuration",
        files.processor_config_json,
        GLM53_PROCESSOR_CONFIG_BYTES,
        PROCESSOR_CONFIG_SHA256,
    )?;
    check_cancelled(is_cancelled)?;
    let model_index_sha256 = verify_exact_artifact(
        "GLM-5.3-Flash model index",
        files.model_index_json,
        GLM53_MODEL_INDEX_BYTES,
        MODEL_INDEX_SHA256,
    )?;
    check_cancelled(is_cancelled)?;

    Ok(Glm53SourcePreflightReport {
        architecture_id: ArchitectureId::new(GLM53_FLASH_ARCHITECTURE_ID).map_err(|_| {
            EngineError::internal_invariant("GLM-5.3-Flash architecture identifier is invalid")
        })?,
        config_sha256,
        tokenizer_sha256,
        processor_config_sha256,
        model_index_sha256,
    })
}

fn verify_exact_artifact(
    label: &'static str,
    bytes: &[u8],
    expected_length: usize,
    expected_sha256: &'static str,
) -> Result<Sha256Digest, EngineError> {
    if bytes.len() != expected_length {
        return Err(EngineError::corrupt_model_package(format!(
            "{label} byte length differs from the pinned source"
        )));
    }
    let expected = pinned_digest(expected_sha256)?;
    let observed = hash_sha256(bytes);
    if observed != expected {
        return Err(EngineError::corrupt_model_package(format!(
            "{label} SHA-256 differs from the pinned source"
        )));
    }
    Ok(observed)
}

fn pinned_digest(value: &'static str) -> Result<Sha256Digest, EngineError> {
    Sha256Digest::from_hex(value)
        .map_err(|_| EngineError::internal_invariant("pinned GLM-5.3-Flash digest is invalid"))
}

fn check_cancelled(is_cancelled: &impl Fn() -> bool) -> Result<(), EngineError> {
    if is_cancelled() { Err(EngineError::cancellation(None)) } else { Ok(()) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pinned_source_identities_and_declared_inventory_are_exact() {
        for (label, length, digest) in [
            ("configuration", GLM53_CONFIG_BYTES, CONFIG_SHA256),
            ("tokenizer", GLM53_TOKENIZER_BYTES, TOKENIZER_SHA256),
            ("processor", GLM53_PROCESSOR_CONFIG_BYTES, PROCESSOR_CONFIG_SHA256),
            ("model index", GLM53_MODEL_INDEX_BYTES, MODEL_INDEX_SHA256),
        ] {
            assert!(length > 0, "{label} must be nonempty");
            assert_eq!(pinned_digest(digest).unwrap().to_string(), digest);
        }
        assert_eq!(GLM53_MODEL_SHARD_COUNT, 62);
        assert_eq!(GLM53_MODEL_TENSOR_COUNT, 76_108);
        assert_eq!(GLM53_MODEL_TENSOR_BYTES, 328_326_771_576);
        assert_eq!(
            GLM53_TOKENIZER_BPE_VOCABULARY_SIZE + GLM53_TOKENIZER_ADDED_TOKEN_COUNT,
            154_856
        );
        assert_eq!(GLM53_MODEL_VOCABULARY_SIZE - 154_856, 24);
        assert_eq!(GLM53_TOKENIZER_PROFILE_ID, "glm5-next-bytelevel-bpe-regex-v1");
        assert_eq!((GLM53_FP8_BLOCK_ROWS, GLM53_FP8_BLOCK_COLUMNS), (128, 128));
        assert_eq!((GLM53_IMAGE_PATCH_SIZE, GLM53_TEMPORAL_PATCH_SIZE), (14, 2));
        assert_eq!(GLM53_MAX_VIDEO_TOKENS, 240_000);
    }

    #[test]
    fn malformed_or_substituted_metadata_fails_before_profile_creation() {
        assert!(matches!(
            verify_exact_artifact("configuration", b"{}", GLM53_CONFIG_BYTES, CONFIG_SHA256),
            Err(EngineError::CorruptModelPackage(_))
        ));
        let wrong = [0_u8; 4];
        let digest = hash_sha256(&wrong).to_string();
        assert_ne!(digest, CONFIG_SHA256);
        assert!(matches!(
            verify_exact_artifact("configuration", &wrong, wrong.len(), CONFIG_SHA256),
            Err(EngineError::CorruptModelPackage(_))
        ));
    }

    #[test]
    fn cancellation_precedes_large_metadata_work() {
        let empty = Glm53MetadataFiles {
            config_json: &[],
            tokenizer_json: &[],
            processor_config_json: &[],
            model_index_json: &[],
        };
        assert!(matches!(
            preflight_glm53_flash_metadata(empty, &|| true),
            Err(EngineError::Cancellation(_))
        ));
    }

    #[test]
    fn executable_claim_remains_typed_unsupported() {
        let report = Glm53SourcePreflightReport {
            architecture_id: ArchitectureId::new(GLM53_FLASH_ARCHITECTURE_ID).unwrap(),
            config_sha256: pinned_digest(CONFIG_SHA256).unwrap(),
            tokenizer_sha256: pinned_digest(TOKENIZER_SHA256).unwrap(),
            processor_config_sha256: pinned_digest(PROCESSOR_CONFIG_SHA256).unwrap(),
            model_index_sha256: pinned_digest(MODEL_INDEX_SHA256).unwrap(),
        };
        assert!(matches!(
            report.require_executable(),
            Err(EngineError::UnsupportedArchitecture(_))
        ));
    }
}
