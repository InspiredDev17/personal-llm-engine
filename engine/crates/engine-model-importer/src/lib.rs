//! Offline-only model import and package signing.
//!
//! This crate has filesystem read/write authority for an owner-selected local
//! quarantine and output store. Production inference crates do not depend on
//! it. It contains no socket, subprocess, dynamic loading, Python, or model-code
//! execution capability.

mod aiptok_metadata;
mod bounded_json;
mod glm53_source;
mod qwen25_config_json;
#[cfg(unix)]
mod qwen25_source_preflight;
#[cfg(unix)]
mod qwen25_tokenizer_export;
mod qwen25_tokenizer_json;
mod safetensors;

pub use aiptok_metadata::{AIPTOK_METADATA_SOURCE_FORMAT, convert as convert_aiptok_metadata};
pub use glm53_source::{
    GLM53_CONFIG_BYTES, GLM53_FLASH_ARCHITECTURE_ID, GLM53_FLASH_SOURCE_REPOSITORY,
    GLM53_FLASH_SOURCE_REVISION, GLM53_FP8_BLOCK_COLUMNS, GLM53_FP8_BLOCK_ROWS,
    GLM53_IMAGE_PATCH_SIZE, GLM53_MAX_IMAGE_TOKENS, GLM53_MAX_VIDEO_TOKENS, GLM53_MIN_IMAGE_TOKENS,
    GLM53_MODEL_INDEX_BYTES, GLM53_MODEL_SHARD_COUNT, GLM53_MODEL_TENSOR_BYTES,
    GLM53_MODEL_TENSOR_COUNT, GLM53_MODEL_VOCABULARY_SIZE, GLM53_PROCESSOR_CONFIG_BYTES,
    GLM53_SPATIAL_MERGE_SIZE, GLM53_TEMPORAL_PATCH_SIZE, GLM53_TOKENIZER_ADDED_TOKEN_COUNT,
    GLM53_TOKENIZER_BPE_VOCABULARY_SIZE, GLM53_TOKENIZER_BYTES, GLM53_TOKENIZER_MERGE_COUNT,
    GLM53_TOKENIZER_PROFILE_ID, GLM53_VIDEO_FRAMES_PER_SECOND, Glm53MetadataFiles,
    Glm53SourcePreflightReport, preflight_glm53_flash_metadata,
};
pub use qwen25_config_json::{
    QWEN25_CONFIG_JSON_SOURCE_FORMAT, QWEN25_MODEL_SAFETENSORS_BYTES, Qwen25Config,
    Qwen25ConfigProfile,
};
#[cfg(unix)]
pub use qwen25_source_preflight::{Qwen25SourcePreflightReport, preflight_qwen25_ingress};
#[cfg(unix)]
pub use qwen25_tokenizer_export::{Qwen25AipTokExportReport, export_qwen25_aiptok};
pub use qwen25_tokenizer_json::{
    QWEN25_ADDED_TOKEN_COUNT, QWEN25_AIPTOK_BYTES, QWEN25_AIPTOK_SHA256, QWEN25_AIPTOK_SHA384,
    QWEN25_BOS_TOKEN_ID, QWEN25_BPE_VOCABULARY_SIZE, QWEN25_DEFINED_TOKEN_COUNT,
    QWEN25_EOS_TOKEN_ID, QWEN25_MERGE_COUNT, QWEN25_MODEL_VOCABULARY_SIZE, QWEN25_TOKENIZER_BYTES,
    QWEN25_TOKENIZER_JSON_SOURCE_FORMAT, QWEN25_TOKENIZER_SHA256, Qwen25TokenizerDescriptor,
    convert as convert_qwen25_tokenizer_json, inspect as inspect_qwen25_tokenizer_json,
};

pub use safetensors::{
    MAX_SAFETENSORS_HEADER_BYTES, MAX_SOURCE_TENSOR_NAME_BYTES, ParsedSafetensors,
    ParsedSafetensorsHeader, SafeTensorRecord, SafeTensorSourceDType,
};

use ed25519_dalek::{Signer, SigningKey};
use personalai_engine_crypto::{
    directory_package_signature_message, hash_package, hash_sha256, hash_signed_content,
};
use personalai_engine_memory::MemoryPool;
use personalai_engine_model_format::{
    DIRECTORY_DECODER_ARCHITECTURE_ID, DIRECTORY_QWEN2_DECODER_ARCHITECTURE_ID,
    DirectoryManifestBuilder, DirectoryShardRecord, DirectoryTensorRecord, HashCatalog,
    HashCatalogEntry, MAX_DIRECTORY_SHARDS, PackageFileKind, Quantization, SignatureEnvelope,
};
use personalai_engine_quantization::{QuantizationErrorMetrics, QuantizationProfile, quantize};
use personalai_engine_tensor::{bf16_bits_to_f32, f16_bits_to_f32};
use personalai_engine_tokenizer::{FirstPartyBpeTokenizer, Tokenizer};
use personalai_engine_types::{
    ContextLimit, DType, EngineError, MAX_MODEL_PACKAGE_BYTES, ModelId, ModelPackageHash,
    Sha256Digest, SignerKeyId, TensorShape,
};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
#[cfg(unix)]
use std::os::unix::fs::{MetadataExt, OpenOptionsExt, PermissionsExt};
use std::path::{Component, Path, PathBuf};
use zeroize::Zeroize;

/// Exact supported source-name mapping.
pub const LLAMA_SAFETENSORS_MAPPING_V1: &str = "llama-safetensors-v1";
/// Exact Qwen2.5 source-name mapping, including required Q/K/V projection biases.
pub const QWEN25_SAFETENSORS_MAPPING_V1: &str = "qwen2.5-safetensors-v1";
/// Maximum provenance document length.
pub const MAX_PROVENANCE_BYTES: usize = 1024 * 1024;
/// Package shard alignment used by the initial importer.
pub const IMPORT_TENSOR_ALIGNMENT: u32 = 64;

/// One immutable local safetensors source shard supplied to deterministic conversion.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SafetensorsSource<'a> {
    name: &'a str,
    bytes: &'a [u8],
}

impl<'a> SafetensorsSource<'a> {
    /// Validates one direct-child source filename and its bounded, nonempty bytes.
    pub fn new(name: &'a str, bytes: &'a [u8]) -> Result<Self, EngineError> {
        validate_source_name(name)?;
        if bytes.is_empty()
            || u64::try_from(bytes.len()).unwrap_or(u64::MAX) > MAX_MODEL_PACKAGE_BYTES
        {
            return Err(EngineError::resource_exhaustion(
                "safetensors source shard size is outside hard bounds",
            ));
        }
        Ok(Self { name, bytes })
    }

    /// Returns the provenance filename.
    pub const fn name(self) -> &'a str {
        self.name
    }

    /// Returns the exact source bytes.
    pub const fn bytes(self) -> &'a [u8] {
        self.bytes
    }
}

/// Closed source-to-canonical tensor mapping.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SourceMapping {
    /// Explicit Llama-family safetensors names mapped to the fixed decoder.
    LlamaSafetensorsV1,
    /// Pinned Qwen2.5 names mapped to the reviewed bias-and-tied decoder profile.
    Qwen25SafetensorsV1,
}

impl SourceMapping {
    /// Parses the exact owner-selected mapping identifier.
    pub fn parse(value: &str) -> Result<Self, EngineError> {
        match value {
            LLAMA_SAFETENSORS_MAPPING_V1 => Ok(Self::LlamaSafetensorsV1),
            QWEN25_SAFETENSORS_MAPPING_V1 => Ok(Self::Qwen25SafetensorsV1),
            _ => Err(EngineError::invalid_input("source tensor mapping is unsupported")),
        }
    }

    /// Returns the canonical mapping identifier.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::LlamaSafetensorsV1 => LLAMA_SAFETENSORS_MAPPING_V1,
            Self::Qwen25SafetensorsV1 => QWEN25_SAFETENSORS_MAPPING_V1,
        }
    }
}

/// Trusted operator configuration for one deterministic offline conversion.
#[derive(Clone, Debug, PartialEq)]
pub struct ImportSpec {
    model_id: ModelId,
    mapping: SourceMapping,
    context_limit: ContextLimit,
    vocabulary_size: u32,
    hidden_dimension: u32,
    layer_count: u32,
    attention_head_count: u32,
    kv_head_count: u32,
    feed_forward_dimension: u32,
    rope_theta: f32,
    rope_scaling: f32,
    tied_output: bool,
    source_label: String,
    license_id: String,
    quantization_profile: QuantizationProfile,
    qwen25_config: Option<Qwen25Config>,
    pinned_source: Option<PinnedSourceIdentity>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct PinnedSourceIdentity {
    name: &'static str,
    byte_length: u64,
    sha256: Sha256Digest,
}

impl ImportSpec {
    /// Creates one explicit fixed-architecture import specification.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        model_id: ModelId,
        mapping: SourceMapping,
        context_limit: ContextLimit,
        vocabulary_size: u32,
        hidden_dimension: u32,
        layer_count: u32,
        attention_head_count: u32,
        kv_head_count: u32,
        feed_forward_dimension: u32,
        rope_theta: f32,
        rope_scaling: f32,
        tied_output: bool,
        source_label: impl Into<String>,
        license_id: impl Into<String>,
        quantization_profile: QuantizationProfile,
    ) -> Result<Self, EngineError> {
        if mapping != SourceMapping::LlamaSafetensorsV1 {
            return Err(EngineError::invalid_input(
                "Qwen2.5 imports require pinned configuration and tokenizer sources",
            ));
        }
        let source_label = source_label.into();
        let license_id = license_id.into();
        validate_provenance_text(&source_label, 256)?;
        validate_provenance_text(&license_id, 128)?;
        if vocabulary_size < 256
            || hidden_dimension == 0
            || layer_count == 0
            || attention_head_count == 0
            || kv_head_count == 0
            || kv_head_count > attention_head_count
            || attention_head_count % kv_head_count != 0
            || hidden_dimension % attention_head_count != 0
            || feed_forward_dimension == 0
            || !rope_theta.is_finite()
            || rope_theta <= 1.0
            || rope_scaling.to_bits() != 1.0_f32.to_bits()
        {
            return Err(EngineError::invalid_input("import architecture dimensions are invalid"));
        }
        Ok(Self {
            model_id,
            mapping,
            context_limit,
            vocabulary_size,
            hidden_dimension,
            layer_count,
            attention_head_count,
            kv_head_count,
            feed_forward_dimension,
            rope_theta,
            rope_scaling,
            tied_output,
            source_label,
            license_id,
            quantization_profile,
            qwen25_config: None,
            pinned_source: None,
        })
    }

    /// Creates the exact Qwen2.5 1.5B source profile from pinned local data.
    pub fn new_qwen25_1_5b(
        model_id: ModelId,
        config_json: &[u8],
        tokenizer_json: &[u8],
        source_label: impl Into<String>,
        license_id: impl Into<String>,
        quantization_profile: QuantizationProfile,
    ) -> Result<Self, EngineError> {
        let config = Qwen25Config::parse(config_json)?;
        let tokenizer = inspect_qwen25_tokenizer_json(tokenizer_json)?;
        Self::new_qwen25_1_5b_validated(
            model_id,
            config,
            tokenizer,
            source_label,
            license_id,
            quantization_profile,
        )
    }

    /// Creates the exact Qwen2.5 1.5B profile from closed validated evidence.
    ///
    /// The argument types have no public constructors; production callers can
    /// obtain them only through the exact pinned configuration and tokenizer
    /// inspectors. This overload lets a quarantine workflow validate each
    /// bounded source once without weakening source identity.
    pub fn new_qwen25_1_5b_validated(
        model_id: ModelId,
        config: Qwen25Config,
        tokenizer: Qwen25TokenizerDescriptor,
        source_label: impl Into<String>,
        license_id: impl Into<String>,
        quantization_profile: QuantizationProfile,
    ) -> Result<Self, EngineError> {
        if tokenizer.model_vocabulary_size() != config.vocabulary_size()
            || tokenizer.unknown_token().is_some()
            || tokenizer.inserts_bos()
        {
            return Err(corrupt("Qwen2.5 tokenizer/configuration semantics differ"));
        }
        let mut spec = Self::new(
            model_id,
            SourceMapping::LlamaSafetensorsV1,
            ContextLimit::new(config.context_limit())?,
            config.vocabulary_size(),
            config.hidden_dimension(),
            config.layer_count(),
            config.attention_head_count(),
            config.kv_head_count(),
            config.feed_forward_dimension(),
            config.rope_theta(),
            1.0,
            true,
            source_label,
            license_id,
            quantization_profile,
        )?;
        spec.mapping = SourceMapping::Qwen25SafetensorsV1;
        spec.qwen25_config = Some(config);
        spec.pinned_source = Some(PinnedSourceIdentity {
            name: "model.safetensors",
            byte_length: config.model_source_bytes(),
            sha256: config.model_source_sha256(),
        });
        Ok(spec)
    }

    /// Returns exact pinned Qwen2.5 configuration evidence when applicable.
    pub const fn qwen25_config(&self) -> Option<Qwen25Config> {
        self.qwen25_config
    }

    /// Returns the reviewed Qwen source byte length and SHA-256 identity.
    pub const fn qwen25_model_source_identity(&self) -> Option<(u64, Sha256Digest)> {
        match &self.pinned_source {
            Some(identity) => Some((identity.byte_length, identity.sha256)),
            None => None,
        }
    }

    /// Returns the exact source descriptor required before Qwen source I/O.
    pub const fn qwen25_model_source_descriptor(
        &self,
    ) -> Option<(&'static str, u64, Sha256Digest)> {
        match &self.pinned_source {
            Some(identity) => Some((identity.name, identity.byte_length, identity.sha256)),
            None => None,
        }
    }

    /// Returns the destination model ID.
    pub const fn model_id(&self) -> &ModelId {
        &self.model_id
    }

    /// Returns the explicit storage profile selected by the offline operator.
    pub const fn quantization_profile(&self) -> QuantizationProfile {
        self.quantization_profile
    }
}

#[derive(Clone, Debug)]
struct TensorQuantizationEvidence {
    name: String,
    format: Quantization,
    source_hash: ModelPackageHash,
    destination_hash: ModelPackageHash,
    metrics: Option<QuantizationErrorMetrics>,
}

struct BuiltShards {
    shards: Vec<Vec<u8>>,
    records: Vec<DirectoryTensorRecord>,
    evidence: Vec<TensorQuantizationEvidence>,
}

/// Deterministic signed package files before their atomic local-store install.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ImportedPackage {
    files: BTreeMap<String, Vec<u8>>,
    package_hash: ModelPackageHash,
    signer_key_id: SignerKeyId,
    signer_public_key: [u8; 32],
}

impl ImportedPackage {
    /// Returns all exact direct-child package files.
    pub const fn files(&self) -> &BTreeMap<String, Vec<u8>> {
        &self.files
    }

    /// Returns the SHA-384 identity of `hashes.aiph`.
    pub const fn package_hash(&self) -> ModelPackageHash {
        self.package_hash
    }

    /// Returns the offline signer key ID.
    pub const fn signer_key_id(&self) -> SignerKeyId {
        self.signer_key_id
    }

    /// Returns the offline signer public key.
    pub const fn signer_public_key(&self) -> [u8; 32] {
        self.signer_public_key
    }
}

/// Result of installing one package beneath an owner-selected local store.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ImportReceipt {
    package_path: PathBuf,
    package_hash: ModelPackageHash,
    signer_key_id: SignerKeyId,
    signer_public_key: [u8; 32],
}

#[derive(Clone, Copy)]
struct TokenizerSourceEvidence<'a> {
    name: &'a str,
    bytes: &'a [u8],
    format: &'a str,
}

impl ImportReceipt {
    /// Returns the installed package path.
    pub fn package_path(&self) -> &Path {
        &self.package_path
    }

    /// Returns the exact configured package identity.
    pub const fn package_hash(&self) -> ModelPackageHash {
        self.package_hash
    }

    /// Returns the package signer ID.
    pub const fn signer_key_id(&self) -> SignerKeyId {
        self.signer_key_id
    }

    /// Returns the package signer public key.
    pub const fn signer_public_key(&self) -> [u8; 32] {
        self.signer_public_key
    }
}

/// Builds one deterministic signed package entirely in memory.
pub fn build_package(
    spec: &ImportSpec,
    source_name: &str,
    source_bytes: &[u8],
    tokenizer_bytes: &[u8],
    signer_key_id: SignerKeyId,
    signing_seed: &mut [u8; 32],
) -> Result<ImportedPackage, EngineError> {
    let result = with_package_emission_gate(spec, || {
        build_package_inner(
            spec,
            source_name,
            source_bytes,
            TokenizerSourceEvidence {
                name: "embedded-tokenizer.aiptok",
                bytes: tokenizer_bytes,
                format: "aiptok-byte-level-bpe-v1",
            },
            tokenizer_bytes,
            signer_key_id,
            signing_seed,
        )
    });
    signing_seed.zeroize();
    result
}

/// Builds one deterministic signed package from one or more safetensors shards.
///
/// Input ordering is deliberately ignored: shards are assigned canonical output
/// indices by ascending source filename. Tensor identities remain globally
/// unique across every shard.
pub fn build_package_sharded(
    spec: &ImportSpec,
    sources: &[SafetensorsSource<'_>],
    tokenizer_bytes: &[u8],
    signer_key_id: SignerKeyId,
    signing_seed: &mut [u8; 32],
) -> Result<ImportedPackage, EngineError> {
    let result = with_package_emission_gate(spec, || {
        build_package_sharded_inner(
            spec,
            sources,
            TokenizerSourceEvidence {
                name: "embedded-tokenizer.aiptok",
                bytes: tokenizer_bytes,
                format: "aiptok-byte-level-bpe-v1",
            },
            tokenizer_bytes,
            signer_key_id,
            signing_seed,
        )
    });
    signing_seed.zeroize();
    result
}

fn build_package_inner(
    spec: &ImportSpec,
    source_name: &str,
    source_bytes: &[u8],
    tokenizer_source: TokenizerSourceEvidence<'_>,
    tokenizer_bytes: &[u8],
    signer_key_id: SignerKeyId,
    signing_seed: &mut [u8; 32],
) -> Result<ImportedPackage, EngineError> {
    validate_pinned_source_identity(spec, source_name, source_bytes)?;
    let source = SafetensorsSource::new(source_name, source_bytes)?;
    let built = build_shards(spec, &[source])?;
    let provenance = provenance(
        spec,
        source_name,
        source_bytes,
        tokenizer_source.name,
        tokenizer_source.bytes,
        tokenizer_source.format,
        &built.evidence,
    )?;
    finish_package(
        spec,
        built.shards,
        built.records,
        provenance,
        tokenizer_bytes,
        signer_key_id,
        signing_seed,
    )
}

fn build_package_sharded_inner(
    spec: &ImportSpec,
    sources: &[SafetensorsSource<'_>],
    tokenizer_source: TokenizerSourceEvidence<'_>,
    tokenizer_bytes: &[u8],
    signer_key_id: SignerKeyId,
    signing_seed: &mut [u8; 32],
) -> Result<ImportedPackage, EngineError> {
    if spec.mapping == SourceMapping::Qwen25SafetensorsV1 {
        let source = sources
            .first()
            .filter(|_| sources.len() == 1)
            .ok_or_else(|| corrupt("Qwen2.5 requires exactly one pinned source shard"))?;
        validate_pinned_source_identity(spec, source.name(), source.bytes())?;
    }
    let ordered = validate_and_order_sources(sources)?;
    let built = build_shards(spec, &ordered)?;
    let provenance = sharded_provenance(
        spec,
        &ordered,
        tokenizer_source.name,
        tokenizer_source.bytes,
        tokenizer_source.format,
        &built.evidence,
    )?;
    finish_package(
        spec,
        built.shards,
        built.records,
        provenance,
        tokenizer_bytes,
        signer_key_id,
        signing_seed,
    )
}

fn finish_package(
    spec: &ImportSpec,
    shards: Vec<Vec<u8>>,
    tensor_records: Vec<DirectoryTensorRecord>,
    provenance: Vec<u8>,
    tokenizer_bytes: &[u8],
    signer_key_id: SignerKeyId,
    signing_seed: &mut [u8; 32],
) -> Result<ImportedPackage, EngineError> {
    ensure_package_target_supported(spec)?;
    let tokenizer_pool = MemoryPool::new(128 * 1024 * 1024)?;
    let tokenizer = FirstPartyBpeTokenizer::parse(tokenizer_bytes, &tokenizer_pool)?;
    if tokenizer.spec().vocabulary_size() != spec.vocabulary_size {
        return Err(corrupt("tokenizer vocabulary differs from the import specification"));
    }
    let mut shard_records = Vec::new();
    let mut total_shard_bytes = 0_u64;
    shard_records
        .try_reserve_exact(shards.len())
        .map_err(|_| EngineError::resource_exhaustion("shard metadata allocation failed"))?;
    for (index, shard) in shards.iter().enumerate() {
        let byte_length = usize_to_u64(shard.len())?;
        total_shard_bytes = total_shard_bytes
            .checked_add(byte_length)
            .filter(|total| *total <= MAX_MODEL_PACKAGE_BYTES)
            .ok_or_else(|| {
                EngineError::resource_exhaustion("aggregate output shard bytes exceed hard limit")
            })?;
        shard_records.push(DirectoryShardRecord::new(
            u32::try_from(index)
                .map_err(|_| EngineError::resource_exhaustion("shard index exceeds u32"))?,
            byte_length,
            hash_package(shard),
        )?);
    }
    let tokenizer_hash = hash_package(tokenizer_bytes);
    let manifest = DirectoryManifestBuilder::new(
        spec.model_id.clone(),
        (0, 1, 0),
        spec.context_limit,
        spec.vocabulary_size,
        spec.hidden_dimension,
        spec.layer_count,
        spec.attention_head_count,
        spec.kv_head_count,
        spec.feed_forward_dimension,
        spec.rope_theta,
        spec.rope_scaling,
        DType::F32,
        spec.quantization_profile.format(),
        spec.tied_output,
        tokenizer_hash,
        shard_records,
        tensor_records,
    )?
    .build_bytes()?;
    let mut catalog_entries = vec![
        catalog_entry(PackageFileKind::Manifest, 0, "manifest.aipm", &manifest)?,
        catalog_entry(PackageFileKind::Tokenizer, 0, "tokenizer.aiptok", tokenizer_bytes)?,
        catalog_entry(PackageFileKind::Provenance, 0, "provenance.json", &provenance)?,
    ];
    catalog_entries
        .try_reserve_exact(shards.len())
        .map_err(|_| EngineError::resource_exhaustion("catalog allocation failed"))?;
    for (index, shard) in shards.iter().enumerate() {
        let index = u32::try_from(index)
            .map_err(|_| EngineError::resource_exhaustion("shard index exceeds u32"))?;
        catalog_entries.push(catalog_entry(
            PackageFileKind::TensorShard,
            index,
            &format!("tensors-{index:04}.aipt"),
            shard,
        )?);
    }
    let catalog = HashCatalog::new(catalog_entries)?;
    let catalog_bytes = catalog.encode()?;
    let catalog_digest = hash_signed_content(&catalog_bytes);
    let message = directory_package_signature_message(&catalog_digest)?;
    let signing_key = SigningKey::from_bytes(signing_seed);
    signing_seed.zeroize();
    let signer_public_key = signing_key.verifying_key().to_bytes();
    if signing_key.verifying_key().is_weak() {
        return Err(EngineError::invalid_input("package signing key is weak"));
    }
    let signature = signing_key.sign(&message).to_bytes();
    let signature = SignatureEnvelope::encode_ed25519(signer_key_id, signature)?.to_vec();
    let package_hash = hash_package(&catalog_bytes);
    let mut files = BTreeMap::from([
        ("hashes.aiph".to_owned(), catalog_bytes),
        ("manifest.aipm".to_owned(), manifest),
        ("provenance.json".to_owned(), provenance),
        ("signature.ed25519".to_owned(), signature),
        ("tokenizer.aiptok".to_owned(), tokenizer_bytes.to_vec()),
    ]);
    for (index, shard) in shards.into_iter().enumerate() {
        files.insert(format!("tensors-{index:04}.aipt"), shard);
    }
    files.values().try_fold(0_u64, |total, bytes| {
        total
            .checked_add(usize_to_u64(bytes.len())?)
            .filter(|total| *total <= MAX_MODEL_PACKAGE_BYTES)
            .ok_or_else(|| {
                EngineError::resource_exhaustion("completed package bytes exceed hard limit")
            })
    })?;
    Ok(ImportedPackage { files, package_hash, signer_key_id, signer_public_key })
}

fn ensure_package_target_supported(spec: &ImportSpec) -> Result<(), EngineError> {
    if spec.mapping == SourceMapping::Qwen25SafetensorsV1 {
        Err(EngineError::not_implemented(
            "Qwen2.5 package emission remains disabled pending Model Foundry package construction, independent package acceptance, signed development registry admission, and optimized-backend qualification",
        ))
    } else {
        Ok(())
    }
}

/// Runs an import operation only after its package target is eligible.
///
/// Keeping the gate outside the operation closure is intentional: Qwen source
/// paths, descriptors, bytes, parsers, conversions, and allocations must not
/// be observed while package emission remains disabled.
fn with_package_emission_gate<T>(
    spec: &ImportSpec,
    operation: impl FnOnce() -> Result<T, EngineError>,
) -> Result<T, EngineError> {
    ensure_package_target_supported(spec)?;
    operation()
}

fn validate_pinned_source_identity(
    spec: &ImportSpec,
    source_name: &str,
    source_bytes: &[u8],
) -> Result<(), EngineError> {
    if spec.mapping != SourceMapping::Qwen25SafetensorsV1 {
        return Ok(());
    }
    let identity = spec
        .pinned_source
        .as_ref()
        .ok_or_else(|| corrupt("Qwen2.5 pinned source identity is missing"))?;
    let observed_length = u64::try_from(source_bytes.len())
        .map_err(|_| EngineError::resource_exhaustion("source length is not representable"))?;
    if source_name != identity.name || observed_length != identity.byte_length {
        return Err(corrupt("Qwen2.5 source name or byte length differs from the pinned identity"));
    }
    if hash_sha256(source_bytes) != identity.sha256 {
        return Err(corrupt("Qwen2.5 source SHA-256 differs from the pinned identity"));
    }
    Ok(())
}

/// Reads quarantined inputs, converts them, and installs a new local package.
#[allow(clippy::too_many_arguments)]
pub fn import_local(
    quarantine_root: &Path,
    source_path: &Path,
    tokenizer_path: &Path,
    output_root: &Path,
    spec: &ImportSpec,
    signer_key_id: SignerKeyId,
    signing_seed: &mut [u8; 32],
    force_audit_note: Option<&str>,
) -> Result<ImportReceipt, EngineError> {
    let result = with_package_emission_gate(spec, || {
        validate_local_path_text(source_path)?;
        validate_local_path_text(tokenizer_path)?;
        let source = read_quarantined_file(quarantine_root, source_path, MAX_MODEL_PACKAGE_BYTES)?;
        let tokenizer_source =
            read_quarantined_file(quarantine_root, tokenizer_path, 64 * 1024 * 1024)?;
        let tokenizer = convert_aiptok_metadata(&tokenizer_source)?;
        let source_name = source_path
            .file_name()
            .and_then(|name| name.to_str())
            .ok_or_else(|| EngineError::invalid_input("source filename is invalid UTF-8"))?;
        let tokenizer_source_name = tokenizer_path
            .file_name()
            .and_then(|name| name.to_str())
            .ok_or_else(|| EngineError::invalid_input("tokenizer filename is invalid UTF-8"))?;
        validate_tokenizer_source_name(tokenizer_source_name)?;
        let package = build_package_inner(
            spec,
            source_name,
            &source,
            TokenizerSourceEvidence {
                name: tokenizer_source_name,
                bytes: &tokenizer_source,
                format: AIPTOK_METADATA_SOURCE_FORMAT,
            },
            &tokenizer,
            signer_key_id,
            signing_seed,
        )?;
        install_package(output_root, spec, &package, force_audit_note)?;
        Ok(ImportReceipt {
            package_path: output_root.join(format!("{}.paimodel", spec.model_id.as_str())),
            package_hash: package.package_hash,
            signer_key_id,
            signer_public_key: package.signer_public_key,
        })
    });
    signing_seed.zeroize();
    result
}

/// Reads multiple quarantined safetensors shards and installs one signed package.
#[allow(clippy::too_many_arguments)]
pub fn import_local_sharded(
    quarantine_root: &Path,
    source_paths: &[PathBuf],
    tokenizer_path: &Path,
    output_root: &Path,
    spec: &ImportSpec,
    signer_key_id: SignerKeyId,
    signing_seed: &mut [u8; 32],
    force_audit_note: Option<&str>,
) -> Result<ImportReceipt, EngineError> {
    let result = with_package_emission_gate(spec, || {
        if source_paths.is_empty()
            || source_paths.len() > usize::try_from(MAX_DIRECTORY_SHARDS).unwrap_or(usize::MAX)
        {
            return Err(EngineError::invalid_input("source shard count is outside hard bounds"));
        }
        validate_local_path_text(tokenizer_path)?;
        let mut owned_sources = Vec::new();
        owned_sources
            .try_reserve_exact(source_paths.len())
            .map_err(|_| EngineError::resource_exhaustion("source shard allocation failed"))?;
        let mut aggregate_source_bytes = 0_u64;
        for source_path in source_paths {
            validate_local_path_text(source_path)?;
            let bytes =
                read_quarantined_file(quarantine_root, source_path, MAX_MODEL_PACKAGE_BYTES)?;
            aggregate_source_bytes = aggregate_source_bytes
                .checked_add(usize_to_u64(bytes.len())?)
                .filter(|total| *total <= MAX_MODEL_PACKAGE_BYTES)
                .ok_or_else(|| {
                    EngineError::resource_exhaustion(
                        "aggregate safetensors source bytes exceed hard limit",
                    )
                })?;
            let name = source_path
                .file_name()
                .and_then(|name| name.to_str())
                .ok_or_else(|| EngineError::invalid_input("source filename is invalid UTF-8"))?
                .to_owned();
            owned_sources.push((name, bytes));
        }
        let tokenizer_source =
            read_quarantined_file(quarantine_root, tokenizer_path, 64 * 1024 * 1024)?;
        let tokenizer = convert_aiptok_metadata(&tokenizer_source)?;
        let tokenizer_source_name = tokenizer_path
            .file_name()
            .and_then(|name| name.to_str())
            .ok_or_else(|| EngineError::invalid_input("tokenizer filename is invalid UTF-8"))?;
        validate_tokenizer_source_name(tokenizer_source_name)?;
        let borrowed_sources = owned_sources
            .iter()
            .map(|(name, bytes)| SafetensorsSource::new(name, bytes))
            .collect::<Result<Vec<_>, _>>()?;
        let package = build_package_sharded_inner(
            spec,
            &borrowed_sources,
            TokenizerSourceEvidence {
                name: tokenizer_source_name,
                bytes: &tokenizer_source,
                format: AIPTOK_METADATA_SOURCE_FORMAT,
            },
            &tokenizer,
            signer_key_id,
            signing_seed,
        )?;
        install_package(output_root, spec, &package, force_audit_note)?;
        Ok(ImportReceipt {
            package_path: output_root.join(format!("{}.paimodel", spec.model_id.as_str())),
            package_hash: package.package_hash,
            signer_key_id,
            signer_public_key: package.signer_public_key,
        })
    });
    signing_seed.zeroize();
    result
}

fn validate_and_order_sources<'a>(
    sources: &[SafetensorsSource<'a>],
) -> Result<Vec<SafetensorsSource<'a>>, EngineError> {
    if sources.is_empty()
        || sources.len() > usize::try_from(MAX_DIRECTORY_SHARDS).unwrap_or(usize::MAX)
    {
        return Err(EngineError::invalid_input("source shard count is outside hard bounds"));
    }
    let mut ordered = Vec::new();
    ordered
        .try_reserve_exact(sources.len())
        .map_err(|_| EngineError::resource_exhaustion("source shard ordering failed"))?;
    ordered.extend_from_slice(sources);
    ordered.sort_by(|left, right| left.name.cmp(right.name));
    let mut aggregate = 0_u64;
    for (index, source) in ordered.iter().copied().enumerate() {
        validate_source_name(source.name)?;
        if source.bytes.is_empty() {
            return Err(EngineError::resource_exhaustion("safetensors source shard is empty"));
        }
        aggregate = aggregate
            .checked_add(usize_to_u64(source.bytes.len())?)
            .filter(|total| *total <= MAX_MODEL_PACKAGE_BYTES)
            .ok_or_else(|| {
                EngineError::resource_exhaustion(
                    "aggregate safetensors source bytes exceed hard limit",
                )
            })?;
        if index > 0 && ordered[index - 1].name == source.name {
            return Err(EngineError::invalid_input("safetensors source filenames must be unique"));
        }
    }
    Ok(ordered)
}

#[allow(clippy::too_many_lines)]
fn build_shards(
    spec: &ImportSpec,
    sources: &[SafetensorsSource<'_>],
) -> Result<BuiltShards, EngineError> {
    let expected = expected_shapes(spec)?;
    let expected_names = expected.keys().cloned().collect::<BTreeSet<_>>();
    let mut seen_names = BTreeSet::new();
    let mut shards = Vec::new();
    let mut records = Vec::new();
    let mut evidence = Vec::new();
    shards
        .try_reserve_exact(sources.len())
        .map_err(|_| EngineError::resource_exhaustion("output shard allocation failed"))?;
    records.try_reserve_exact(expected.len()).map_err(|_| {
        EngineError::resource_exhaustion("import tensor metadata allocation failed")
    })?;
    evidence
        .try_reserve_exact(expected.len())
        .map_err(|_| EngineError::resource_exhaustion("quantization evidence allocation failed"))?;
    let mut completed_output_bytes = 0_u64;

    for (shard_index, source) in sources.iter().copied().enumerate() {
        let parsed = ParsedSafetensors::parse(source.bytes)?;
        let mut mapped = BTreeMap::new();
        for tensor in parsed.tensors() {
            let canonical = canonical_name(spec.mapping, tensor.name(), spec.layer_count)?;
            if !seen_names.insert(canonical.clone()) {
                return Err(corrupt("source shards produce a duplicate canonical tensor"));
            }
            if mapped.insert(canonical, tensor).is_some() {
                return Err(corrupt("source mapping produced a duplicate canonical tensor"));
            }
        }
        if mapped.is_empty() {
            return Err(corrupt("safetensors source shard contains no tensors"));
        }
        let mut shard = Vec::new();
        let shard_index = u32::try_from(shard_index)
            .map_err(|_| EngineError::resource_exhaustion("shard index exceeds u32"))?;
        for (name, tensor) in mapped {
            let shape = expected
                .get(&name)
                .ok_or_else(|| corrupt("source tensor inventory contains an unexpected tensor"))?;
            let runtime_dtype = tensor.runtime_dtype()?;
            if tensor.shape() != shape {
                return Err(corrupt("source tensor dtype or shape does not match architecture"));
            }
            let source_bytes = parsed.tensor_bytes(tensor)?;
            let aligned =
                align_up(shard.len(), usize::try_from(IMPORT_TENSOR_ALIGNMENT).unwrap_or(64))?;
            let converted = convert_source_tensor_to_f32(runtime_dtype, source_bytes)?;
            let values = f32_values(&converted)?;
            let quantization = spec.quantization_profile.tensor_format(tensor.shape().rank());
            let (destination, metrics) = match quantization {
                Quantization::None => (converted, None),
                Quantization::AipQ8V1 | Quantization::AipQ4V1 => {
                    let payload = quantize(quantization, tensor.shape(), &values)?;
                    let metrics = payload.metrics();
                    (payload.into_bytes(), Some(metrics))
                }
            };
            let destination_length = destination.len();
            let planned_shard_end = aligned.checked_add(destination_length).ok_or_else(|| {
                EngineError::resource_exhaustion("converted output shard length overflows")
            })?;
            completed_output_bytes
                .checked_add(usize_to_u64(planned_shard_end)?)
                .filter(|total| *total <= MAX_MODEL_PACKAGE_BYTES)
                .ok_or_else(|| {
                    EngineError::resource_exhaustion(
                        "aggregate converted output bytes exceed hard limit",
                    )
                })?;
            shard
                .try_reserve_exact(planned_shard_end - shard.len())
                .map_err(|_| EngineError::resource_exhaustion("import shard allocation failed"))?;
            if aligned > shard.len() {
                shard.resize(aligned, 0);
            }
            shard.extend_from_slice(&destination);
            let destination_hash = hash_package(&destination);
            records.push(DirectoryTensorRecord::new(
                name.clone(),
                DType::F32,
                tensor.shape().clone(),
                shard_index,
                usize_to_u64(aligned)?,
                usize_to_u64(destination.len())?,
                IMPORT_TENSOR_ALIGNMENT,
                quantization,
                destination_hash,
            )?);
            evidence.push(TensorQuantizationEvidence {
                name,
                format: quantization,
                source_hash: hash_package(source_bytes),
                destination_hash,
                metrics,
            });
        }
        completed_output_bytes =
            completed_output_bytes.checked_add(usize_to_u64(shard.len())?).ok_or_else(|| {
                EngineError::resource_exhaustion("aggregate converted output bytes overflow")
            })?;
        shards.push(shard);
    }
    if seen_names != expected_names {
        return Err(corrupt("source tensor inventory has missing tensors"));
    }
    records.sort_by(|left, right| left.name().cmp(right.name()));
    evidence.sort_by(|left, right| left.name.cmp(&right.name));
    Ok(BuiltShards { shards, records, evidence })
}

fn expected_shapes(spec: &ImportSpec) -> Result<BTreeMap<String, TensorShape>, EngineError> {
    if spec.mapping == SourceMapping::Qwen25SafetensorsV1 && !spec.tied_output {
        return Err(corrupt("Qwen2.5 requires tied embedding/output weights"));
    }
    let hidden = u64::from(spec.hidden_dimension);
    let vocabulary = u64::from(spec.vocabulary_size);
    let intermediate = u64::from(spec.feed_forward_dimension);
    let kv_dimension = u64::from(spec.kv_head_count)
        .checked_mul(hidden / u64::from(spec.attention_head_count))
        .ok_or_else(|| corrupt("architecture KV dimension overflows"))?;
    let mut expected = BTreeMap::from([
        ("token_embedding.weight".to_owned(), TensorShape::new([vocabulary, hidden])?),
        ("final_norm.weight".to_owned(), TensorShape::new([hidden])?),
    ]);
    if !spec.tied_output {
        expected.insert("output.weight".to_owned(), TensorShape::new([vocabulary, hidden])?);
    }
    for layer in 0..spec.layer_count {
        let prefix = format!("layers.{layer:03}");
        for suffix in ["attn_norm.weight", "ffn_norm.weight"] {
            expected.insert(format!("{prefix}.{suffix}"), TensorShape::new([hidden])?);
        }
        for suffix in ["q_proj.weight", "o_proj.weight"] {
            expected.insert(format!("{prefix}.{suffix}"), TensorShape::new([hidden, hidden])?);
        }
        for suffix in ["k_proj.weight", "v_proj.weight"] {
            expected
                .insert(format!("{prefix}.{suffix}"), TensorShape::new([kv_dimension, hidden])?);
        }
        if spec.mapping == SourceMapping::Qwen25SafetensorsV1 {
            expected.insert(format!("{prefix}.q_proj.bias"), TensorShape::new([hidden])?);
            for suffix in ["k_proj.bias", "v_proj.bias"] {
                expected.insert(format!("{prefix}.{suffix}"), TensorShape::new([kv_dimension])?);
            }
        }
        for suffix in ["gate_proj.weight", "up_proj.weight"] {
            expected
                .insert(format!("{prefix}.{suffix}"), TensorShape::new([intermediate, hidden])?);
        }
        expected.insert(
            format!("{prefix}.down_proj.weight"),
            TensorShape::new([hidden, intermediate])?,
        );
    }
    Ok(expected)
}

fn canonical_name(
    mapping: SourceMapping,
    source: &str,
    layer_count: u32,
) -> Result<String, EngineError> {
    match mapping {
        SourceMapping::LlamaSafetensorsV1 => map_llama_name(source, layer_count),
        SourceMapping::Qwen25SafetensorsV1 => map_qwen25_name(source, layer_count),
    }
}

fn map_qwen25_name(source: &str, layer_count: u32) -> Result<String, EngineError> {
    let canonical = map_llama_name(source, layer_count);
    if canonical.is_ok() {
        return canonical;
    }
    let rest = source
        .strip_prefix("model.layers.")
        .ok_or_else(|| corrupt("Qwen2.5 tensor name has no approved canonical mapping"))?;
    let (layer_text, suffix) =
        rest.split_once('.').ok_or_else(|| corrupt("Qwen2.5 layer tensor name is malformed"))?;
    if layer_text.len() > 1 && layer_text.starts_with('0') {
        return Err(corrupt("Qwen2.5 layer index has ambiguous leading zeroes"));
    }
    let layer: u32 = layer_text.parse().map_err(|_| corrupt("Qwen2.5 layer index is invalid"))?;
    if layer >= layer_count {
        return Err(corrupt("Qwen2.5 layer index exceeds configured architecture"));
    }
    let suffix = match suffix {
        "self_attn.q_proj.bias" => "q_proj.bias",
        "self_attn.k_proj.bias" => "k_proj.bias",
        "self_attn.v_proj.bias" => "v_proj.bias",
        _ => return Err(corrupt("Qwen2.5 tensor name has no approved canonical mapping")),
    };
    Ok(format!("layers.{layer:03}.{suffix}"))
}

fn map_llama_name(source: &str, layer_count: u32) -> Result<String, EngineError> {
    match source {
        "model.embed_tokens.weight" => return Ok("token_embedding.weight".to_owned()),
        "model.norm.weight" => return Ok("final_norm.weight".to_owned()),
        "lm_head.weight" => return Ok("output.weight".to_owned()),
        _ => {}
    }
    let rest = source
        .strip_prefix("model.layers.")
        .ok_or_else(|| corrupt("source tensor name has no approved canonical mapping"))?;
    let (layer_text, suffix) =
        rest.split_once('.').ok_or_else(|| corrupt("source layer tensor name is malformed"))?;
    if layer_text.len() > 1 && layer_text.starts_with('0') {
        return Err(corrupt("source layer index has ambiguous leading zeroes"));
    }
    let layer: u32 = layer_text.parse().map_err(|_| corrupt("source layer index is invalid"))?;
    if layer >= layer_count {
        return Err(corrupt("source layer index exceeds configured architecture"));
    }
    let canonical_suffix = match suffix {
        "input_layernorm.weight" => "attn_norm.weight",
        "post_attention_layernorm.weight" => "ffn_norm.weight",
        "self_attn.q_proj.weight" => "q_proj.weight",
        "self_attn.k_proj.weight" => "k_proj.weight",
        "self_attn.v_proj.weight" => "v_proj.weight",
        "self_attn.o_proj.weight" => "o_proj.weight",
        "mlp.gate_proj.weight" => "gate_proj.weight",
        "mlp.up_proj.weight" => "up_proj.weight",
        "mlp.down_proj.weight" => "down_proj.weight",
        _ => return Err(corrupt("source layer tensor has no approved canonical mapping")),
    };
    Ok(format!("layers.{layer:03}.{canonical_suffix}"))
}

fn render_quantization_evidence(
    evidence: &[TensorQuantizationEvidence],
) -> Result<String, EngineError> {
    use std::fmt::Write as _;

    let mut records = String::new();
    for (index, tensor) in evidence.iter().enumerate() {
        let (maximum, mean, rms, relative, relative_count) =
            tensor.metrics.map_or((0.0, 0.0, 0.0, 0.0, 0), |metrics| {
                (
                    metrics.maximum_absolute_error(),
                    metrics.mean_absolute_error(),
                    metrics.root_mean_square_error(),
                    metrics.maximum_relative_error(),
                    metrics.relative_value_count(),
                )
            });
        writeln!(
            &mut records,
            concat!(
                "    {{\"destination_sha384\": \"{}\", \"format\": \"{}\", ",
                "\"maximum_absolute_error\": {:.9e}, \"maximum_relative_error\": {:.9e}, ",
                "\"mean_absolute_error\": {:.9e}, \"name\": \"{}\", ",
                "\"relative_value_count\": {}, \"root_mean_square_error\": {:.9e}, ",
                "\"source_sha384\": \"{}\"}}{}"
            ),
            tensor.destination_hash,
            quantization_name(tensor.format),
            maximum,
            relative,
            mean,
            tensor.name,
            relative_count,
            rms,
            tensor.source_hash,
            if index + 1 == evidence.len() { "" } else { "," },
        )
        .map_err(|_| EngineError::internal_invariant("quantization evidence write failed"))?;
        if records.len() > MAX_PROVENANCE_BYTES {
            return Err(EngineError::resource_exhaustion(
                "quantization evidence exceeds provenance limit",
            ));
        }
    }
    Ok(records)
}

fn render_layer_error_evidence(
    evidence: &[TensorQuantizationEvidence],
) -> Result<String, EngineError> {
    use std::fmt::Write as _;

    let mut layers = BTreeMap::<String, (u64, f64, f64, f64, f64)>::new();
    for tensor in evidence {
        let layer = tensor
            .name
            .strip_prefix("layers.")
            .and_then(|suffix| suffix.get(..3))
            .map_or("global".to_owned(), |number| format!("layer-{number}"));
        let entry = layers.entry(layer).or_insert((0, 0.0, 0.0, 0.0, 0.0));
        entry.0 = entry
            .0
            .checked_add(1)
            .ok_or_else(|| EngineError::resource_exhaustion("layer evidence count overflows"))?;
        if let Some(metrics) = tensor.metrics {
            entry.1 = entry.1.max(metrics.maximum_absolute_error());
            entry.2 = entry.2.max(metrics.mean_absolute_error());
            entry.3 = entry.3.max(metrics.root_mean_square_error());
            entry.4 = entry.4.max(metrics.maximum_relative_error());
        }
    }
    let mut records = String::new();
    let count = layers.len();
    for (index, (layer, (tensor_count, maximum, mean, rms, relative))) in
        layers.into_iter().enumerate()
    {
        writeln!(
            &mut records,
            concat!(
                "    {{\"layer\": \"{}\", \"maximum_absolute_error\": {:.9e}, ",
                "\"maximum_mean_absolute_error\": {:.9e}, ",
                "\"maximum_relative_error\": {:.9e}, ",
                "\"maximum_root_mean_square_error\": {:.9e}, \"tensor_count\": {}}}{}"
            ),
            layer,
            maximum,
            mean,
            relative,
            rms,
            tensor_count,
            if index + 1 == count { "" } else { "," },
        )
        .map_err(|_| EngineError::internal_invariant("layer error evidence write failed"))?;
    }
    Ok(records)
}

const fn quantization_name(format: Quantization) -> &'static str {
    match format {
        Quantization::None => "f32",
        Quantization::AipQ8V1 => "aipq8-v1",
        Quantization::AipQ4V1 => "aipq4-v1",
    }
}

fn f32_values(bytes: &[u8]) -> Result<Vec<f32>, EngineError> {
    if bytes.is_empty() || bytes.len() % 4 != 0 {
        return Err(EngineError::internal_invariant("converted F32 tensor byte length is invalid"));
    }
    let mut values = Vec::new();
    values
        .try_reserve_exact(bytes.len() / 4)
        .map_err(|_| EngineError::resource_exhaustion("F32 conversion allocation failed"))?;
    for scalar in bytes.chunks_exact(4) {
        let value = f32::from_le_bytes(
            scalar
                .try_into()
                .map_err(|_| EngineError::internal_invariant("F32 scalar length is invalid"))?,
        );
        if !value.is_finite() {
            return Err(corrupt("source tensor contains non-finite F32 data"));
        }
        values.push(value);
    }
    Ok(values)
}

fn provenance(
    spec: &ImportSpec,
    source_name: &str,
    source_bytes: &[u8],
    tokenizer_source_name: &str,
    tokenizer_source_bytes: &[u8],
    tokenizer_source_format: &str,
    quantization_evidence: &[TensorQuantizationEvidence],
) -> Result<Vec<u8>, EngineError> {
    validate_provenance_text(source_name, 256)?;
    validate_provenance_text(tokenizer_source_name, 256)?;
    validate_provenance_text(tokenizer_source_format, 64)?;
    let quantization_records = render_quantization_evidence(quantization_evidence)?;
    let layer_error_records = render_layer_error_evidence(quantization_evidence)?;
    let document = format!(
        concat!(
            "{{\n",
            "  \"architecture\": \"{}\",\n",
            "  \"license_id\": \"{}\",\n",
            "  \"mapping\": \"{}\",\n",
            "  \"model_id\": \"{}\",\n",
            "  \"quantization_layer_errors\": [\n",
            "{}",
            "  ],\n",
            "  \"quantization_profile\": \"{}\",\n",
            "  \"quantization_tensors\": [\n",
            "{}",
            "  ],\n",
            "  \"source_file\": \"{}\",\n",
            "  \"source_format\": \"safetensors\",\n",
            "  \"source_label\": \"{}\",\n",
            "  \"source_sha384\": \"{}\",\n",
            "  \"tokenizer_source_file\": \"{}\",\n",
            "  \"tokenizer_source_format\": \"{}\",\n",
            "  \"tokenizer_source_sha384\": \"{}\",\n",
            "  \"tool\": \"personalai-engine-model-importer\",\n",
            "  \"tool_version\": \"{}\"\n",
            "}}\n"
        ),
        destination_architecture_id(spec),
        spec.license_id,
        spec.mapping.as_str(),
        spec.model_id,
        layer_error_records,
        spec.quantization_profile.as_str(),
        quantization_records,
        source_name,
        spec.source_label,
        hash_package(source_bytes),
        tokenizer_source_name,
        tokenizer_source_format,
        hash_package(tokenizer_source_bytes),
        env!("CARGO_PKG_VERSION"),
    );
    if document.len() > MAX_PROVENANCE_BYTES {
        return Err(EngineError::resource_exhaustion("provenance exceeds its hard limit"));
    }
    Ok(document.into_bytes())
}

fn sharded_provenance(
    spec: &ImportSpec,
    sources: &[SafetensorsSource<'_>],
    tokenizer_source_name: &str,
    tokenizer_source_bytes: &[u8],
    tokenizer_source_format: &str,
    quantization_evidence: &[TensorQuantizationEvidence],
) -> Result<Vec<u8>, EngineError> {
    use std::fmt::Write as _;

    validate_provenance_text(tokenizer_source_name, 256)?;
    validate_provenance_text(tokenizer_source_format, 64)?;
    let quantization_records = render_quantization_evidence(quantization_evidence)?;
    let layer_error_records = render_layer_error_evidence(quantization_evidence)?;
    let mut source_records = String::new();
    for (index, source) in sources.iter().copied().enumerate() {
        validate_provenance_text(source.name, 256)?;
        writeln!(
            &mut source_records,
            "    {{\"file\": \"{}\", \"sha384\": \"{}\"}}{}",
            source.name,
            hash_package(source.bytes),
            if index + 1 == sources.len() { "" } else { "," },
        )
        .map_err(|_| EngineError::internal_invariant("provenance string write failed"))?;
        if source_records.len() > MAX_PROVENANCE_BYTES {
            return Err(EngineError::resource_exhaustion(
                "provenance source list exceeds its hard limit",
            ));
        }
    }
    let document = format!(
        concat!(
            "{{\n",
            "  \"architecture\": \"{}\",\n",
            "  \"license_id\": \"{}\",\n",
            "  \"mapping\": \"{}\",\n",
            "  \"model_id\": \"{}\",\n",
            "  \"quantization_layer_errors\": [\n",
            "{}",
            "  ],\n",
            "  \"quantization_profile\": \"{}\",\n",
            "  \"quantization_tensors\": [\n",
            "{}",
            "  ],\n",
            "  \"source_files\": [\n",
            "{}",
            "  ],\n",
            "  \"source_format\": \"safetensors-shards-v1\",\n",
            "  \"source_label\": \"{}\",\n",
            "  \"tokenizer_source_file\": \"{}\",\n",
            "  \"tokenizer_source_format\": \"{}\",\n",
            "  \"tokenizer_source_sha384\": \"{}\",\n",
            "  \"tool\": \"personalai-engine-model-importer\",\n",
            "  \"tool_version\": \"{}\"\n",
            "}}\n"
        ),
        destination_architecture_id(spec),
        spec.license_id,
        spec.mapping.as_str(),
        spec.model_id,
        layer_error_records,
        spec.quantization_profile.as_str(),
        quantization_records,
        source_records,
        spec.source_label,
        tokenizer_source_name,
        tokenizer_source_format,
        hash_package(tokenizer_source_bytes),
        env!("CARGO_PKG_VERSION"),
    );
    if document.len() > MAX_PROVENANCE_BYTES {
        return Err(EngineError::resource_exhaustion("provenance exceeds its hard limit"));
    }
    Ok(document.into_bytes())
}

fn destination_architecture_id(spec: &ImportSpec) -> &'static str {
    match spec.mapping {
        SourceMapping::LlamaSafetensorsV1 => DIRECTORY_DECODER_ARCHITECTURE_ID,
        SourceMapping::Qwen25SafetensorsV1 => DIRECTORY_QWEN2_DECODER_ARCHITECTURE_ID,
    }
}

fn catalog_entry(
    kind: PackageFileKind,
    index: u32,
    name: &str,
    bytes: &[u8],
) -> Result<HashCatalogEntry, EngineError> {
    HashCatalogEntry::new(kind, index, name, usize_to_u64(bytes.len())?, hash_package(bytes))
}

fn install_package(
    output_root: &Path,
    spec: &ImportSpec,
    package: &ImportedPackage,
    force_audit_note: Option<&str>,
) -> Result<(), EngineError> {
    validate_store_root(output_root)?;
    let destination = output_root.join(format!("{}.paimodel", spec.model_id.as_str()));
    let staging = output_root.join(format!(".{}.paimodel.importing", spec.model_id.as_str()));
    let previous = output_root.join(format!("{}.paimodel.previous", spec.model_id.as_str()));
    if staging.exists() || previous.exists() {
        return Err(EngineError::invalid_input("staging or preserved package path already exists"));
    }
    if destination.exists() {
        let note = force_audit_note.ok_or_else(|| {
            EngineError::invalid_input("existing package requires force audit note")
        })?;
        validate_provenance_text(note, 512)?;
        append_audit(output_root, spec, package, note)?;
        fs::rename(&destination, &previous).map_err(|_| {
            EngineError::authentication_failure("existing package preservation failed")
        })?;
    } else if force_audit_note.is_some() {
        return Err(EngineError::invalid_input(
            "force audit note is invalid without an existing package",
        ));
    }
    create_private_directory(&staging)?;
    write_package_files(&staging, package)?;
    fs::rename(&staging, &destination)
        .map_err(|_| EngineError::backend_failure("package atomic install failed"))?;
    Ok(())
}

fn write_package_files(directory: &Path, package: &ImportedPackage) -> Result<(), EngineError> {
    for (name, bytes) in &package.files {
        let path = directory.join(name);
        let mut options = OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(unix)]
        options.mode(0o600);
        let mut file = options
            .open(&path)
            .map_err(|_| EngineError::backend_failure("package file creation failed"))?;
        file.write_all(bytes)
            .and_then(|()| file.sync_all())
            .map_err(|_| EngineError::backend_failure("package file durable write failed"))?;
        #[cfg(unix)]
        fs::set_permissions(&path, fs::Permissions::from_mode(0o400))
            .map_err(|_| EngineError::backend_failure("package file sealing failed"))?;
    }
    File::open(directory)
        .and_then(|file| file.sync_all())
        .map_err(|_| EngineError::backend_failure("package directory sync failed"))?;
    #[cfg(unix)]
    fs::set_permissions(directory, fs::Permissions::from_mode(0o500))
        .map_err(|_| EngineError::backend_failure("package directory sealing failed"))?;
    Ok(())
}

fn append_audit(
    output_root: &Path,
    spec: &ImportSpec,
    package: &ImportedPackage,
    note: &str,
) -> Result<(), EngineError> {
    let path = output_root.join("import-audit.log");
    let mut options = OpenOptions::new();
    options.append(true).create(true);
    #[cfg(unix)]
    options.mode(0o600);
    let mut file = options
        .open(path)
        .map_err(|_| EngineError::backend_failure("import audit record open failed"))?;
    writeln!(
        file,
        "model_id={} replacement_hash={} signer_key_id={} note={}",
        spec.model_id, package.package_hash, package.signer_key_id, note
    )
    .and_then(|()| file.sync_all())
    .map_err(|_| EngineError::backend_failure("import audit record write failed"))
}

fn read_quarantined_file(
    quarantine_root: &Path,
    source: &Path,
    maximum: u64,
) -> Result<Vec<u8>, EngineError> {
    validate_absolute_normal_path(quarantine_root, "quarantine root")?;
    let root = fs::canonicalize(quarantine_root)
        .map_err(|_| EngineError::invalid_input("quarantine root is unavailable"))?;
    let path = if source.is_absolute() { source.to_path_buf() } else { root.join(source) };
    let canonical = fs::canonicalize(&path)
        .map_err(|_| EngineError::invalid_input("quarantined source is unavailable"))?;
    if !canonical.starts_with(&root) {
        return Err(EngineError::authentication_failure("source symlink escapes quarantine root"));
    }
    let metadata = fs::symlink_metadata(&canonical)
        .map_err(|_| EngineError::invalid_input("quarantined source metadata is unavailable"))?;
    if !metadata.is_file()
        || metadata.file_type().is_symlink()
        || metadata.len() == 0
        || metadata.len() > maximum
    {
        return Err(EngineError::resource_exhaustion("quarantined source size is outside bounds"));
    }
    let length = usize::try_from(metadata.len())
        .map_err(|_| EngineError::resource_exhaustion("quarantined source is not addressable"))?;
    let mut options = OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    options.custom_flags(libc::O_CLOEXEC | libc::O_NOFOLLOW);
    let mut file = options
        .open(&canonical)
        .map_err(|_| EngineError::authentication_failure("quarantined source open failed"))?;
    #[cfg(unix)]
    {
        let opened = file
            .metadata()
            .map_err(|_| EngineError::authentication_failure("opened source metadata failed"))?;
        if opened.dev() != metadata.dev()
            || opened.ino() != metadata.ino()
            || opened.len() != metadata.len()
            || !opened.is_file()
        {
            return Err(EngineError::authentication_failure(
                "quarantined source changed during secure open",
            ));
        }
    }
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(length)
        .map_err(|_| EngineError::resource_exhaustion("quarantined source allocation failed"))?;
    bytes.resize(length, 0);
    file.read_exact(&mut bytes).map_err(|_| corrupt("quarantined source is truncated"))?;
    let mut extra = [0_u8; 1];
    if file.read(&mut extra).map_err(|_| corrupt("quarantined source trailing check failed"))? != 0
    {
        return Err(corrupt("quarantined source grew during secure read"));
    }
    Ok(bytes)
}

fn validate_local_path_text(path: &Path) -> Result<(), EngineError> {
    let text = path
        .to_str()
        .ok_or_else(|| EngineError::invalid_input("local source path is invalid UTF-8"))?;
    if text.contains("://") || text.starts_with("file:") {
        return Err(EngineError::invalid_input("URLs are prohibited import sources"));
    }
    Ok(())
}

fn validate_store_root(root: &Path) -> Result<(), EngineError> {
    validate_absolute_normal_path(root, "output root")?;
    let metadata = fs::symlink_metadata(root)
        .map_err(|_| EngineError::invalid_input("output root is unavailable"))?;
    if !metadata.is_dir() || metadata.file_type().is_symlink() {
        return Err(EngineError::authentication_failure(
            "output root must be a non-symlink directory",
        ));
    }
    Ok(())
}

fn validate_absolute_normal_path(path: &Path, label: &str) -> Result<(), EngineError> {
    if !path.is_absolute()
        || path.components().any(|component| {
            matches!(component, Component::Prefix(_) | Component::CurDir | Component::ParentDir)
        })
    {
        return Err(EngineError::invalid_input(format!("{label} must be absolute and normalized")));
    }
    Ok(())
}

fn validate_provenance_text(value: &str, maximum: usize) -> Result<(), EngineError> {
    if value.is_empty()
        || value.len() > maximum
        || !value.is_ascii()
        || value.chars().any(char::is_control)
        || value.contains('"')
        || value.contains('\\')
    {
        return Err(EngineError::invalid_input("provenance text is outside safe bounds"));
    }
    Ok(())
}

fn validate_source_name(name: &str) -> Result<(), EngineError> {
    if !name.ends_with(".safetensors")
        || name.contains('/')
        || name.contains('\\')
        || name.contains("..")
        || name.len() > 256
        || !name.is_ascii()
        || name.chars().any(char::is_control)
    {
        return Err(EngineError::invalid_input(
            "source must be one safe local safetensors filename",
        ));
    }
    Ok(())
}

fn validate_tokenizer_source_name(name: &str) -> Result<(), EngineError> {
    if !name.ends_with(".aiptokmeta")
        || name.contains('/')
        || name.contains('\\')
        || name.contains("..")
        || name.len() > 256
        || !name.is_ascii()
        || name.chars().any(char::is_control)
    {
        return Err(EngineError::invalid_input(
            "tokenizer source must be one safe local .aiptokmeta filename",
        ));
    }
    Ok(())
}

fn validate_finite_f32(bytes: &[u8]) -> Result<(), EngineError> {
    if bytes.len() % 4 != 0 {
        return Err(corrupt("F32 tensor byte length is not scalar-aligned"));
    }
    for chunk in bytes.chunks_exact(4) {
        let value = f32::from_le_bytes(
            chunk.try_into().map_err(|_| corrupt("F32 tensor scalar is truncated"))?,
        );
        if !value.is_finite() {
            return Err(corrupt("source tensor contains a non-finite scalar"));
        }
    }
    Ok(())
}

fn convert_source_tensor_to_f32(dtype: DType, bytes: &[u8]) -> Result<Vec<u8>, EngineError> {
    let scalar_width = dtype.byte_width();
    let scalar_count = u64::try_from(bytes.len())
        .ok()
        .filter(|length| *length % scalar_width == 0)
        .map(|length| length / scalar_width)
        .ok_or_else(|| corrupt("source tensor byte length is not scalar-aligned"))?;
    let output_length = scalar_count
        .checked_mul(DType::F32.byte_width())
        .and_then(|length| usize::try_from(length).ok())
        .ok_or_else(|| EngineError::resource_exhaustion("converted tensor length overflows"))?;
    let mut output = Vec::new();
    output
        .try_reserve_exact(output_length)
        .map_err(|_| EngineError::resource_exhaustion("converted tensor allocation failed"))?;
    match dtype {
        DType::F32 => output.extend_from_slice(bytes),
        DType::F16 | DType::BFloat16 => {
            for encoded in bytes.chunks_exact(2) {
                let bits = u16::from_le_bytes(
                    encoded.try_into().map_err(|_| corrupt("source tensor scalar is truncated"))?,
                );
                let value = if dtype == DType::F16 {
                    f16_bits_to_f32(bits)
                } else {
                    bf16_bits_to_f32(bits)
                };
                output.extend_from_slice(&value.to_le_bytes());
            }
        }
        _ => return Err(EngineError::not_implemented("safetensors source dtype is unsupported")),
    }
    if output.len() != output_length {
        return Err(EngineError::internal_invariant(
            "converted tensor length differs from its checked plan",
        ));
    }
    validate_finite_f32(&output)?;
    Ok(output)
}

fn align_up(value: usize, alignment: usize) -> Result<usize, EngineError> {
    let remainder = value % alignment;
    if remainder == 0 {
        Ok(value)
    } else {
        value
            .checked_add(alignment - remainder)
            .ok_or_else(|| EngineError::resource_exhaustion("import shard alignment overflows"))
    }
}

fn create_private_directory(path: &Path) -> Result<(), EngineError> {
    fs::create_dir(path)
        .map_err(|_| EngineError::backend_failure("package staging create failed"))?;
    #[cfg(unix)]
    fs::set_permissions(path, fs::Permissions::from_mode(0o700))
        .map_err(|_| EngineError::backend_failure("package staging permissions failed"))?;
    Ok(())
}

fn usize_to_u64(value: usize) -> Result<u64, EngineError> {
    u64::try_from(value)
        .map_err(|_| EngineError::resource_exhaustion("import byte length is not representable"))
}

fn corrupt(detail: impl AsRef<str>) -> EngineError {
    EngineError::corrupt_model_package(detail)
}

#[cfg(test)]
mod tests {
    use super::*;
    use personalai_engine_crypto::TrustedSigner;
    use personalai_engine_loader::{
        DirectoryLoadCancellationProbe, ModelRegistry, RegistrationMode, RegistrySelectionV2,
        VerifiedDirectoryPackage,
    };
    use personalai_engine_model_format::{
        DIRECTORY_MANIFEST_HEADER_BYTES, DIRECTORY_QWEN2_MANIFEST_VERSION,
        DIRECTORY_QWEN2_MINIMUM_ENGINE_VERSION, DirectoryManifest, DirectoryManifestBuilder,
        ParsedModelPackage,
    };
    use personalai_engine_placement::{PlacementPolicy, WeightResidencyMode};
    use personalai_engine_scheduler::SchedulerLimits;
    use personalai_engine_server::{EngineService, ServiceConfig};
    use personalai_engine_testkit::DecoderPackageFixtureBuilder;
    use personalai_engine_types::{
        GenerationEvent, GenerationRequest, ModelPackageRejection, RequestId, ResourceBudget,
        SecurityEvent, SequenceId,
    };
    use std::sync::atomic::{AtomicU64, Ordering};

    fn spec() -> ImportSpec {
        spec_with_profile(QuantizationProfile::ReferenceF32)
    }

    fn spec_with_profile(profile: QuantizationProfile) -> ImportSpec {
        ImportSpec::new(
            ModelId::new("synthetic-import").expect("model ID must validate"),
            SourceMapping::LlamaSafetensorsV1,
            ContextLimit::new(8).expect("context must validate"),
            259,
            4,
            2,
            2,
            1,
            8,
            10_000.0,
            1.0,
            false,
            "synthetic-local-fixture",
            "test-only",
            profile,
        )
        .expect("spec must validate")
    }

    fn exact_qwen_spec(config: &[u8]) -> ImportSpec {
        let config = Qwen25Config::parse(config).expect("pinned config must validate");
        ImportSpec::new_qwen25_1_5b_validated(
            ModelId::new("qwen25-import-test").expect("model ID must validate"),
            config,
            qwen25_tokenizer_json::pinned_test_descriptor(),
            "pinned-local-fixture",
            "test-only",
            QuantizationProfile::ReferenceF32,
        )
        .expect("validated Qwen evidence must construct the closed import spec")
    }

    #[test]
    fn qwen25_public_import_spec_constructors_are_closed_and_profile_exact() {
        let fixtures: [&[u8]; 3] = [
            include_bytes!("../tests/fixtures/qwen25-1.5b-instruct-config.json"),
            include_bytes!("../tests/fixtures/qwen25-math-1.5b-instruct-config.json"),
            include_bytes!("../tests/fixtures/qwen25-coder-1.5b-instruct-config.json"),
        ];
        for fixture in fixtures {
            let spec = exact_qwen_spec(fixture);
            assert_eq!(spec.mapping, SourceMapping::Qwen25SafetensorsV1);
            assert!(spec.tied_output);
            assert_eq!(spec.qwen25_model_source_identity().unwrap().0, 3_087_467_144);
            assert_eq!(spec.qwen25_model_source_descriptor().unwrap().0, "model.safetensors");
            assert_eq!(destination_architecture_id(&spec), DIRECTORY_QWEN2_DECODER_ARCHITECTURE_ID);

            let result = ImportSpec::new_qwen25_1_5b(
                ModelId::new("qwen25-source-constructor").unwrap(),
                fixture,
                b"not-the-pinned-tokenizer",
                "pinned-local-fixture",
                "test-only",
                QuantizationProfile::ReferenceF32,
            );
            assert!(
                format!("{}", result.expect_err("unpinned tokenizer must fail"))
                    .contains("tokenizer source identity is not pinned")
            );
        }
    }

    #[test]
    fn qwen25_mapping_is_closed_bias_aware_tied_and_not_emitted_as_manifest_v1() {
        assert_eq!(
            SourceMapping::parse(QWEN25_SAFETENSORS_MAPPING_V1).unwrap(),
            SourceMapping::Qwen25SafetensorsV1
        );
        assert_eq!(
            canonical_name(
                SourceMapping::Qwen25SafetensorsV1,
                "model.layers.0.self_attn.q_proj.bias",
                2,
            )
            .unwrap(),
            "layers.000.q_proj.bias"
        );
        assert!(
            canonical_name(
                SourceMapping::Qwen25SafetensorsV1,
                "model.layers.2.self_attn.k_proj.bias",
                2,
            )
            .is_err()
        );

        let qwen =
            exact_qwen_spec(include_bytes!("../tests/fixtures/qwen25-1.5b-instruct-config.json"));
        let shapes = expected_shapes(&qwen).unwrap();
        assert_eq!(shapes.len(), 338);
        for layer in 0..28 {
            let prefix = format!("layers.{layer:03}");
            assert_eq!(shapes[&format!("{prefix}.q_proj.bias")].dimensions(), &[1_536]);
            assert_eq!(shapes[&format!("{prefix}.k_proj.bias")].dimensions(), &[256]);
            assert_eq!(shapes[&format!("{prefix}.v_proj.bias")].dimensions(), &[256]);
            for projection in ["q", "k", "v"] {
                assert_eq!(
                    canonical_name(
                        SourceMapping::Qwen25SafetensorsV1,
                        &format!("model.layers.{layer}.self_attn.{projection}_proj.bias"),
                        28,
                    )
                    .unwrap(),
                    format!("{prefix}.{projection}_proj.bias")
                );
            }
        }
        assert!(!shapes.contains_key("output.weight"));

        let mut untied = qwen.clone();
        untied.tied_output = false;
        assert!(
            format!("{}", expected_shapes(&untied).unwrap_err())
                .contains("requires tied embedding/output weights")
        );

        let mut seed = [7_u8; 32];
        let result = finish_package(
            &qwen,
            vec![vec![0; 4]],
            Vec::new(),
            b"{}\n".to_vec(),
            b"invalid-tokenizer",
            SignerKeyId::from_bytes([1; 16]).unwrap(),
            &mut seed,
        );
        assert!(matches!(result, Err(EngineError::NotImplemented(_))));
    }

    #[test]
    fn qwen25_public_emission_gates_precede_invalid_sources_and_zeroize_seeds() {
        let mut qwen = spec();
        qwen.mapping = SourceMapping::Qwen25SafetensorsV1;
        qwen.tied_output = true;

        let mut seed = [9_u8; 32];
        let result = build_package(
            &qwen,
            "not-model.bin",
            &[],
            &[],
            SignerKeyId::from_bytes([1; 16]).unwrap(),
            &mut seed,
        );
        assert!(matches!(result, Err(EngineError::NotImplemented(_))));
        assert_eq!(seed, [0; 32]);

        let mut seed = [9_u8; 32];
        let result = build_package_sharded(
            &qwen,
            &[],
            &[],
            SignerKeyId::from_bytes([1; 16]).unwrap(),
            &mut seed,
        );
        assert!(matches!(result, Err(EngineError::NotImplemented(_))));
        assert_eq!(seed, [0; 32]);

        let observed = AtomicU64::new(0);
        let result = with_package_emission_gate(&qwen, || {
            observed.fetch_add(1, Ordering::SeqCst);
            Ok(())
        });
        assert!(matches!(result, Err(EngineError::NotImplemented(_))));
        assert_eq!(observed.load(Ordering::SeqCst), 0);

        let mut seed = [9_u8; 32];
        let result = import_local(
            Path::new("/definitely-not-readable-qwen-quarantine"),
            Path::new("https://untrusted.invalid/model.safetensors"),
            Path::new("/missing-tokenizer.json"),
            Path::new("/missing-output"),
            &qwen,
            SignerKeyId::from_bytes([1; 16]).unwrap(),
            &mut seed,
            None,
        );
        assert!(matches!(result, Err(EngineError::NotImplemented(_))));
        assert_eq!(seed, [0; 32]);

        let mut seed = [9_u8; 32];
        let result = import_local_sharded(
            Path::new("/definitely-not-readable-qwen-quarantine"),
            &[],
            Path::new("/missing-tokenizer.json"),
            Path::new("/missing-output"),
            &qwen,
            SignerKeyId::from_bytes([1; 16]).unwrap(),
            &mut seed,
            None,
        );
        assert!(matches!(result, Err(EngineError::NotImplemented(_))));
        assert_eq!(seed, [0; 32]);
    }

    #[test]
    fn qwen25_source_identity_is_checked_before_safetensors_parsing() {
        let exact = b"synthetic-pinned-source";
        let mut qwen = spec();
        qwen.mapping = SourceMapping::Qwen25SafetensorsV1;
        qwen.pinned_source = Some(PinnedSourceIdentity {
            name: "model.safetensors",
            byte_length: u64::try_from(exact.len()).unwrap(),
            sha256: hash_sha256(exact),
        });
        assert!(validate_pinned_source_identity(&qwen, "model.safetensors", exact).is_ok());
        assert!(validate_pinned_source_identity(&qwen, "other.safetensors", exact).is_err());
        assert!(validate_pinned_source_identity(&qwen, "model.safetensors", b"short").is_err());
        let mut changed = exact.to_vec();
        changed[0] ^= 1;
        assert!(validate_pinned_source_identity(&qwen, "model.safetensors", &changed).is_err());

        let tokenizer = TokenizerSourceEvidence {
            name: "tokenizer.json",
            bytes: b"invalid",
            format: QWEN25_TOKENIZER_JSON_SOURCE_FORMAT,
        };
        let signer = SignerKeyId::from_bytes([1; 16]).unwrap();
        let mut seed = [4; 32];
        let wrong_name = build_package_inner(
            &qwen,
            "wrong.safetensors",
            exact,
            tokenizer,
            b"invalid",
            signer,
            &mut seed,
        )
        .expect_err("source identity must precede parser validation");
        assert!(format!("{wrong_name}").contains("name or byte length"));

        let mut seed = [4; 32];
        let wrong_hash = build_package_inner(
            &qwen,
            "model.safetensors",
            &changed,
            tokenizer,
            b"invalid",
            signer,
            &mut seed,
        )
        .expect_err("source digest must precede parser validation");
        assert!(format!("{wrong_hash}").contains("SHA-256"));

        let mut seed = [4; 32];
        let parser = build_package_inner(
            &qwen,
            "model.safetensors",
            exact,
            tokenizer,
            b"invalid",
            signer,
            &mut seed,
        )
        .expect_err("identity-valid malformed source must reach the parser");
        assert!(!format!("{parser}").contains("pinned identity"));
    }

    fn source_and_tokenizer() -> (Vec<u8>, Vec<u8>) {
        let fixture = DecoderPackageFixtureBuilder::new().build().expect("fixture must build");
        let parsed = ParsedModelPackage::parse(fixture.bytes()).expect("fixture must parse");
        let tokenizer =
            parsed.tokenizer_data(fixture.bytes()).expect("tokenizer must exist").to_vec();
        let mut records = Vec::new();
        let mut data = Vec::new();
        for tensor in parsed.tensors() {
            let source_name = external_name(tensor.name().as_str());
            let tensor_bytes =
                parsed.tensor_data(fixture.bytes(), tensor).expect("tensor bytes must exist");
            let start = data.len();
            data.extend_from_slice(tensor_bytes);
            let dimensions = tensor
                .shape()
                .dimensions()
                .iter()
                .map(u64::to_string)
                .collect::<Vec<_>>()
                .join(",");
            records.push(format!(
                "\"{source_name}\":{{\"dtype\":\"F32\",\"shape\":[{dimensions}],\"data_offsets\":[{start},{}]}}",
                data.len()
            ));
        }
        let header = format!("{{{}}}", records.join(","));
        let mut source = Vec::new();
        source.extend_from_slice(
            &u64::try_from(header.len()).expect("length must fit").to_le_bytes(),
        );
        source.extend_from_slice(header.as_bytes());
        source.extend_from_slice(&data);
        (source, tokenizer)
    }

    fn split_safetensors_source(source: &[u8]) -> (Vec<u8>, Vec<u8>) {
        let parsed = ParsedSafetensors::parse(source).expect("source must parse before splitting");
        let build = |parity: usize| {
            let selected = parsed
                .tensors()
                .iter()
                .enumerate()
                .filter(|(index, _)| index % 2 == parity)
                .map(|(_, tensor)| tensor)
                .collect::<Vec<_>>();
            let mut data = Vec::new();
            let mut records = Vec::new();
            for tensor in selected {
                let start = data.len();
                data.extend_from_slice(
                    parsed.tensor_bytes(tensor).expect("split tensor bytes must exist"),
                );
                let dimensions = tensor
                    .shape()
                    .dimensions()
                    .iter()
                    .map(u64::to_string)
                    .collect::<Vec<_>>()
                    .join(",");
                records.push(format!(
                    "\"{}\":{{\"dtype\":\"F32\",\"shape\":[{dimensions}],\"data_offsets\":[{start},{}]}}",
                    tensor.name(),
                    data.len(),
                ));
            }
            let header = format!("{{{}}}", records.join(","));
            let mut encoded = Vec::new();
            encoded.extend_from_slice(
                &u64::try_from(header.len()).expect("header length must fit").to_le_bytes(),
            );
            encoded.extend_from_slice(header.as_bytes());
            encoded.extend_from_slice(&data);
            encoded
        };
        (build(0), build(1))
    }

    fn convert_fixture_source_dtype(source: &[u8], dtype: DType) -> Vec<u8> {
        let parsed = ParsedSafetensors::parse(source).expect("source must parse before conversion");
        let mut data = Vec::new();
        let mut records = Vec::new();
        for tensor in parsed.tensors() {
            let start = data.len();
            for scalar in
                parsed.tensor_bytes(tensor).expect("source tensor bytes must exist").chunks_exact(4)
            {
                let value = f32::from_le_bytes(scalar.try_into().expect("f32 scalar"));
                let bits = if dtype == DType::F16 {
                    personalai_engine_tensor::f32_to_f16_bits(value)
                } else {
                    personalai_engine_tensor::f32_to_bf16_bits(value)
                };
                data.extend_from_slice(&bits.to_le_bytes());
            }
            let dimensions = tensor
                .shape()
                .dimensions()
                .iter()
                .map(u64::to_string)
                .collect::<Vec<_>>()
                .join(",");
            let dtype_name = if dtype == DType::F16 { "F16" } else { "BF16" };
            records.push(format!(
                "\"{}\":{{\"dtype\":\"{dtype_name}\",\"shape\":[{dimensions}],\"data_offsets\":[{start},{}]}}",
                tensor.name(),
                data.len(),
            ));
        }
        let header = format!("{{{}}}", records.join(","));
        let mut encoded = Vec::new();
        encoded.extend_from_slice(
            &u64::try_from(header.len()).expect("header length must fit").to_le_bytes(),
        );
        encoded.extend_from_slice(header.as_bytes());
        encoded.extend_from_slice(&data);
        encoded
    }

    fn source_tokenizer_metadata() -> Vec<u8> {
        use std::fmt::Write as _;
        let mut text = concat!(
            "AIPTOK-METADATA-V1\n",
            "family=byte-level-bpe-v1\n",
            "normalizer=preserve-bytes\n",
            "pre_tokenizer=none\n",
            "decoder=concatenate-bytes-skip-special\n",
            "unknown_policy=byte-fallback\n",
            "chat_template=none\n",
            "vocabulary_size=259\n",
            "bos_token_id=256\n",
            "eos_token_id=257\n",
            "unknown_token_id=258\n",
            "maximum_input_bytes=4194304\n",
            "maximum_produced_tokens=1048576\n",
        )
        .to_owned();
        for token in 0_u16..=255 {
            writeln!(&mut text, "token {token} ordinary {token:02x}")
                .expect("metadata write must work");
        }
        text.push_str(
            "token 256 bos 3c7c626f737c3e\n\
             token 257 eos 3c7c656f737c3e\n\
             token 258 unknown 3c7c756e6b7c3e\n\
             merge_count=0\n",
        );
        text.into_bytes()
    }

    fn assert_tokenizer_provenance(
        loaded: &personalai_engine_loader::VerifiedDirectoryPackage,
        metadata: &[u8],
    ) {
        let provenance = std::str::from_utf8(loaded.provenance_bytes())
            .expect("provenance must be canonical UTF-8");
        assert!(provenance.contains("\"tokenizer_source_format\": \"aiptok-metadata-v1\""));
        assert!(
            provenance
                .contains(&format!("\"tokenizer_source_sha384\": \"{}\"", hash_package(metadata)))
        );
    }

    fn assert_signed_cpu_generation(
        cpu_model: &personalai_engine_server::LoadedCpuModel,
        budget: ResourceBudget,
    ) {
        let request = GenerationRequest::new(
            RequestId::new(1).expect("request ID must validate"),
            spec().model_id().clone(),
            "A",
            4,
            ContextLimit::new(8).expect("context must validate"),
            budget,
        )
        .expect("generation request must validate");
        let mut events = Vec::new();
        let usage = cpu_model
            .generate_greedy(
                &request,
                SequenceId::new(1).expect("sequence ID must validate"),
                &std::sync::atomic::AtomicBool::new(false),
                |event| {
                    events.push(event);
                    Ok(())
                },
            )
            .expect("signed directory package must generate");
        assert!(usage.output_tokens() > 0);
        assert!(events.iter().any(|event| matches!(event, GenerationEvent::Token { .. })));
        assert!(matches!(events.last(), Some(GenerationEvent::Completed { .. })));
    }

    #[cfg(all(unix, target_os = "macos"))]
    fn assert_signed_metal_generation(
        service: &EngineService,
        budget: ResourceBudget,
        service_events: &mut Vec<SecurityEvent>,
    ) {
        if !personalai_engine_server::MetalBackend::devices()
            .is_ok_and(|devices| !devices.is_empty())
        {
            return;
        }
        let backend = personalai_engine_server::MetalBackend::new(0, 1 << 28, 16)
            .expect("Metal backend must construct");
        let metal_model = service
            .load_registered_metal_model(
                spec().model_id(),
                &budget,
                backend.clone(),
                &mut |event| service_events.push(event),
            )
            .expect("signed imported directory package must load for Metal");
        let request = GenerationRequest::new(
            RequestId::new(91).expect("request ID must validate"),
            spec().model_id().clone(),
            "A",
            2,
            ContextLimit::new(8).expect("context must validate"),
            budget,
        )
        .expect("Metal request must validate");
        let usage = metal_model
            .generate_greedy(
                &request,
                SequenceId::new(91).expect("sequence ID must validate"),
                &std::sync::atomic::AtomicBool::new(false),
                |_| Ok(()),
            )
            .expect("signed imported package must generate through Metal");
        assert_eq!(usage.output_tokens(), 2);
        assert!(backend.statistics().expect("Metal metrics must read").dispatch_count > 0);
        assert!(service_events.is_empty());
    }

    fn write_import_sources(quarantine: &Path, source: &[u8]) -> Vec<u8> {
        fs::write(quarantine.join("model.safetensors"), source).expect("source must write");
        let metadata = source_tokenizer_metadata();
        fs::write(quarantine.join("tokenizer.aiptokmeta"), &metadata)
            .expect("tokenizer metadata must write");
        metadata
    }

    fn temporary_root(label: &str) -> PathBuf {
        static NEXT: AtomicU64 = AtomicU64::new(1);
        PathBuf::from("/tmp").join(format!(
            "personalai-importer-{label}-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ))
    }

    struct CancelAfterChecks {
        checks: AtomicU64,
        cancel_at: u64,
    }

    impl CancelAfterChecks {
        const fn new(cancel_at: u64) -> Self {
            Self { checks: AtomicU64::new(0), cancel_at }
        }
    }

    impl DirectoryLoadCancellationProbe for CancelAfterChecks {
        fn is_cancelled(&self) -> bool {
            self.checks.fetch_add(1, Ordering::AcqRel) + 1 >= self.cancel_at
        }
    }

    #[cfg(unix)]
    fn load_cpu_profile(profile: QuantizationProfile) -> personalai_engine_server::LoadedCpuModel {
        let (model, root, store, package_path) =
            load_cpu_profile_fixture(profile, PlacementPolicy::fully_resident());
        remove_sealed_store(&root, &store, &package_path);
        model
    }

    #[cfg(unix)]
    fn load_cpu_profile_fixture(
        profile: QuantizationProfile,
        placement_policy: PlacementPolicy,
    ) -> (personalai_engine_server::LoadedCpuModel, PathBuf, PathBuf, PathBuf) {
        let root = temporary_root(&format!("cpu-{}", profile.as_str()));
        let store = root.join("models");
        let runtime = root.join("runtime");
        let package_path = store.join("synthetic-import.paimodel");
        fs::create_dir(&root).expect("profile root must create");
        fs::create_dir(&store).expect("profile store must create");
        fs::create_dir(&runtime).expect("profile runtime must create");
        create_private_directory(&package_path).expect("profile package must create");
        for directory in [&root, &store, &runtime] {
            fs::set_permissions(directory, fs::Permissions::from_mode(0o700))
                .expect("profile directory mode must set");
        }
        let (source, tokenizer) = source_and_tokenizer();
        let spec = spec_with_profile(profile);
        let key_id = SignerKeyId::from_bytes([0x45; 16]).expect("profile key ID must validate");
        let mut seed = [0x46; 32];
        let package =
            build_package(&spec, "model.safetensors", &source, &tokenizer, key_id, &mut seed)
                .expect("profile package must build");
        write_package_files(&package_path, &package).expect("profile package must write");
        fs::set_permissions(&store, fs::Permissions::from_mode(0o500))
            .expect("profile store must seal");
        let trusted = TrustedSigner::active(key_id, package.signer_public_key())
            .expect("profile signer must validate");
        let config = ServiceConfig::new(&store, &runtime, runtime.join("engine.sock"))
            .expect("profile service config must validate");
        let mut service = EngineService::new(
            config,
            SchedulerLimits::new(8, 1, 1 << 20).expect("profile scheduler must validate"),
        )
        .expect("profile service must construct");
        service
            .register_configured_model(
                spec.model_id().clone(),
                "synthetic-import.paimodel",
                package.package_hash(),
                trusted,
            )
            .expect("profile package must register");
        let budget = ResourceBudget::new(1 << 30, 1 << 28, 1).expect("budget must validate");
        let model = service
            .load_registered_cpu_model_with_placement(
                spec.model_id(),
                &budget,
                placement_policy,
                &mut |_| {},
            )
            .expect("profile CPU model must load");
        (model, root, store, package_path)
    }

    #[cfg(unix)]
    fn profile_logits(
        model: &personalai_engine_server::LoadedCpuModel,
        request_value: u128,
        sequence_value: u64,
    ) -> Vec<f32> {
        let budget = ResourceBudget::new(1 << 30, 1 << 28, 1).expect("budget must validate");
        let mut session = model
            .model()
            .start_session(
                &budget,
                RequestId::new(request_value).expect("request ID must validate"),
                SequenceId::new(sequence_value).expect("sequence ID must validate"),
            )
            .expect("profile session must start");
        let logits = session.prefill(&[65, 66]).expect("profile prefill must execute");
        (0..usize::try_from(logits.shape().element_count()).expect("logits must be addressable"))
            .map(|index| logits.f32_value(index).expect("logit must read"))
            .collect()
    }

    #[cfg(unix)]
    fn profile_tokens(
        model: &personalai_engine_server::LoadedCpuModel,
        request_value: u128,
        sequence_value: u64,
    ) -> Vec<u32> {
        let budget = ResourceBudget::new(1 << 30, 1 << 28, 1).expect("budget must validate");
        let request = GenerationRequest::new(
            RequestId::new(request_value).expect("request ID must validate"),
            spec().model_id().clone(),
            "AB",
            4,
            ContextLimit::new(8).expect("context must validate"),
            budget,
        )
        .expect("profile generation request must validate");
        let mut tokens = Vec::new();
        model
            .generate_greedy(
                &request,
                SequenceId::new(sequence_value).expect("sequence ID must validate"),
                &std::sync::atomic::AtomicBool::new(false),
                |event| {
                    if let GenerationEvent::Token { token_id, .. } = event {
                        tokens.push(token_id.get());
                    }
                    Ok(())
                },
            )
            .expect("profile generation must complete");
        tokens
    }

    #[cfg(unix)]
    #[test]
    fn owner_selected_streamed_service_generation_matches_resident_scalar_tokens() {
        let resident = load_cpu_profile(QuantizationProfile::ReferenceF32);
        let (streamed, root, store, package_path) = load_cpu_profile_fixture(
            QuantizationProfile::ReferenceF32,
            PlacementPolicy::layer_streamed(),
        );
        assert_eq!(
            resident.placement().expect("resident plan").mode(),
            WeightResidencyMode::FullyResident,
        );
        assert_eq!(
            streamed.placement().expect("streamed plan").mode(),
            WeightResidencyMode::LayerStreamed,
        );
        assert_eq!(profile_tokens(&resident, 701, 701), profile_tokens(&streamed, 702, 702));
        drop(streamed);
        remove_sealed_store(&root, &store, &package_path);
    }

    const REGISTRY_V2: &[u8] = include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/model-foundry-registry-v2/positive-0002.registry.json"
    ));

    fn registry_v2_bytes() -> &'static [u8] {
        REGISTRY_V2.strip_suffix(b"\n").unwrap_or(REGISTRY_V2)
    }

    fn registry_v2_selection() -> RegistrySelectionV2 {
        RegistrySelectionV2::new(
            ModelId::new("synthetic-import").expect("model ID"),
            "phase0-synthetic-import",
            hash_package(registry_v2_bytes()),
            2,
            2,
            registry_v2_signer(),
            registry_v2_package_signer(),
        )
        .expect("selection")
    }

    fn registry_v2_signer() -> TrustedSigner {
        TrustedSigner::active(
            SignerKeyId::from_bytes([0x63; 16]).expect("registry key ID"),
            decode_public_key("2bc2800b3316e009209ffd757dab19ccf0ae84bc7ae90654e1e81712d270f653"),
        )
        .expect("registry trust")
    }

    fn registry_v2_package_signer() -> TrustedSigner {
        TrustedSigner::active(
            SignerKeyId::from_bytes([0x51; 16]).expect("package key ID"),
            decode_public_key("2012cb90ca60e8e5d8daf66e2272d2233e0486d557e8c66141ed8920177d7eb7"),
        )
        .expect("package trust")
    }

    fn decode_public_key(value: &str) -> [u8; 32] {
        let mut bytes = [0_u8; 32];
        for (index, pair) in value.as_bytes().chunks_exact(2).enumerate() {
            let digit = |byte: u8| match byte {
                b'0'..=b'9' => byte - b'0',
                b'a'..=b'f' => byte - b'a' + 10,
                _ => panic!("fixture public key is invalid"),
            };
            bytes[index] = (digit(pair[0]) << 4) | digit(pair[1]);
        }
        bytes
    }

    #[cfg(unix)]
    fn with_corrupted_read_only_file(path: &Path, mutate_last_byte: bool, inspect: impl FnOnce()) {
        let original = fs::read(path).expect("file must read");
        let mut corrupt_bytes = original.clone();
        let mutation = if mutate_last_byte { corrupt_bytes.len() - 1 } else { 0 };
        corrupt_bytes[mutation] ^= 1;
        fs::set_permissions(path, fs::Permissions::from_mode(0o600))
            .expect("file must become writable");
        fs::write(path, corrupt_bytes).expect("corruption must write");
        fs::set_permissions(path, fs::Permissions::from_mode(0o400))
            .expect("file must become read-only");
        inspect();
        fs::set_permissions(path, fs::Permissions::from_mode(0o600))
            .expect("file must become writable");
        fs::write(path, original).expect("file must restore");
        fs::set_permissions(path, fs::Permissions::from_mode(0o400))
            .expect("file must become read-only");
    }

    #[cfg(unix)]
    fn assert_live_mapping_change_is_reported(
        loaded: &VerifiedDirectoryPackage,
        shard_path: &Path,
    ) {
        let mut events = Vec::new();
        with_corrupted_read_only_file(shard_path, false, || {
            assert!(loaded.revalidate_mappings(&mut |event| events.push(event)).is_err());
        });
        assert_eq!(events.len(), 1);
    }

    #[cfg(unix)]
    fn remove_sealed_store(root: &Path, store: &Path, package: &Path) {
        fs::set_permissions(store, fs::Permissions::from_mode(0o700))
            .expect("temporary store must unseal for cleanup");
        fs::set_permissions(package, fs::Permissions::from_mode(0o700))
            .expect("temporary package must unseal for cleanup");
        fs::remove_dir_all(root).expect("temporary fixture must clean up");
    }

    fn resign_manifest(
        package: &ImportedPackage,
        mutate: impl FnOnce(&mut [u8]),
        signer_key_id: SignerKeyId,
        signing_seed: [u8; 32],
    ) -> ImportedPackage {
        let mut files = package.files.clone();
        let manifest = files.get_mut("manifest.aipm").expect("manifest must exist");
        mutate(manifest);
        let old_catalog =
            HashCatalog::parse(files.get("hashes.aiph").expect("hash catalog must exist"))
                .expect("hash catalog must parse");
        let entries = old_catalog
            .entries()
            .iter()
            .map(|entry| {
                let bytes = files.get(entry.name()).expect("catalog file must exist");
                HashCatalogEntry::new(
                    entry.kind(),
                    entry.index(),
                    entry.name(),
                    u64::try_from(bytes.len()).expect("fixture length must fit"),
                    hash_package(bytes),
                )
                .expect("replacement entry must validate")
            })
            .collect::<Vec<_>>();
        let catalog = HashCatalog::new(entries)
            .expect("replacement catalog must validate")
            .encode()
            .expect("replacement catalog must encode");
        let digest = hash_signed_content(&catalog);
        let message =
            directory_package_signature_message(&digest).expect("signature message must construct");
        let signing_key = SigningKey::from_bytes(&signing_seed);
        let signature =
            SignatureEnvelope::encode_ed25519(signer_key_id, signing_key.sign(&message).to_bytes())
                .expect("signature envelope must encode")
                .to_vec();
        let package_hash = hash_package(&catalog);
        files.insert("hashes.aiph".to_owned(), catalog);
        files.insert("signature.ed25519".to_owned(), signature);
        ImportedPackage {
            files,
            package_hash,
            signer_key_id,
            signer_public_key: signing_key.verifying_key().to_bytes(),
        }
    }

    fn resign_as_bounded_qwen_v11(
        package: &ImportedPackage,
        signer_key_id: SignerKeyId,
        signing_seed: [u8; 32],
    ) -> ImportedPackage {
        let legacy = DirectoryManifest::parse(
            package.files.get("manifest.aipm").expect("manifest must exist"),
        )
        .expect("legacy manifest must parse");
        let qwen = DirectoryManifestBuilder::new_qwen2(
            legacy.model_id().clone(),
            DIRECTORY_QWEN2_MINIMUM_ENGINE_VERSION,
            legacy.context_limit(),
            legacy.vocabulary_size(),
            legacy.hidden_dimension(),
            legacy.layer_count(),
            legacy.attention_head_count(),
            legacy.kv_head_count(),
            legacy.feed_forward_dimension(),
            1.0e-6,
            1_000_000.0,
            legacy.storage_dtype(),
            legacy.quantization(),
            legacy.tokenizer_hash(),
            legacy.shards().to_vec(),
            legacy.tensors().to_vec(),
        )
        .expect("bounded Qwen v1.1 manifest must build")
        .build_bytes()
        .expect("bounded Qwen v1.1 manifest must encode");
        assert_eq!(qwen.len(), package.files["manifest.aipm"].len());
        resign_manifest(
            package,
            |manifest| manifest.copy_from_slice(&qwen),
            signer_key_id,
            signing_seed,
        )
    }

    fn external_name(canonical: &str) -> String {
        match canonical {
            "token_embedding.weight" => "model.embed_tokens.weight".to_owned(),
            "final_norm.weight" => "model.norm.weight".to_owned(),
            "output.weight" => "lm_head.weight".to_owned(),
            name => {
                let rest = name.strip_prefix("layers.").expect("canonical layer prefix");
                let (layer, suffix) = rest.split_once('.').expect("canonical layer suffix");
                let layer = layer.parse::<u32>().expect("canonical layer number");
                format!("model.layers.{layer}.{suffix}")
                    .replace("attn_norm.weight", "input_layernorm.weight")
                    .replace("ffn_norm.weight", "post_attention_layernorm.weight")
                    .replace("q_proj.weight", "self_attn.q_proj.weight")
                    .replace("k_proj.weight", "self_attn.k_proj.weight")
                    .replace("v_proj.weight", "self_attn.v_proj.weight")
                    .replace("o_proj.weight", "self_attn.o_proj.weight")
                    .replace("gate_proj.weight", "mlp.gate_proj.weight")
                    .replace("up_proj.weight", "mlp.up_proj.weight")
                    .replace("down_proj.weight", "mlp.down_proj.weight")
            }
        }
    }

    #[test]
    fn deterministic_signed_import_builds_exact_files() {
        let (source, tokenizer) = source_and_tokenizer();
        let key_id = SignerKeyId::from_bytes([0x51; 16]).expect("key ID must validate");
        let mut first_seed = [0x52; 32];
        let mut second_seed = first_seed;
        let first = build_package(
            &spec(),
            "model.safetensors",
            &source,
            &tokenizer,
            key_id,
            &mut first_seed,
        )
        .expect("first import must succeed");
        let second = build_package(
            &spec(),
            "model.safetensors",
            &source,
            &tokenizer,
            key_id,
            &mut second_seed,
        )
        .expect("second import must succeed");
        assert_eq!(first, second);
        assert_eq!(
            first.package_hash(),
            ModelPackageHash::from_sha384_hex(
                "979b5ffe376f078a28f4fae69fc5e92a68aea290287cb8a4876136889bf7a508\
                 bee6238c9fffc8e683229aa56b79c5f7",
            )
            .expect("fixture package hash must parse"),
        );
        assert_eq!(
            first.signer_public_key(),
            [
                0x20, 0x12, 0xcb, 0x90, 0xca, 0x60, 0xe8, 0xe5, 0xd8, 0xda, 0xf6, 0x6e, 0x22, 0x72,
                0xd2, 0x23, 0x3e, 0x04, 0x86, 0xd5, 0x57, 0xe8, 0xc6, 0x61, 0x41, 0xed, 0x89, 0x20,
                0x17, 0x7d, 0x7e, 0xb7,
            ],
        );
        let unchanged_data_files = [
            (
                "manifest.aipm",
                4_608,
                "6a2a5349c18916a515b3afac4253ae5240e45d0e3fb808bf5a8093d7c45cc469\
                 b45fbdf6ad95f88e94f7d93a1d62941d",
            ),
            (
                "tensors-0000.aipt",
                9_904,
                "b0ca20c39137666448ca362acc76abfdbc1f79fdde96729f852bda4a85d0cf83\
                 14b9791e49ce48d83c90fb510447a4cd",
            ),
            (
                "tokenizer.aiptok",
                2_501,
                "c366ab2a065461ce28b6456c9d8391a9e83e69fb9938f82ec5dcf6f9246c30f\
                 84f0928d55d375e82e4c25be3163abce4",
            ),
        ];
        for (name, expected_length, expected_hash) in unchanged_data_files {
            let bytes = first.files().get(name).expect("fixture file must exist");
            assert_eq!(bytes.len(), expected_length);
            assert_eq!(
                hash_package(bytes),
                ModelPackageHash::from_sha384_hex(expected_hash)
                    .expect("fixture file hash must parse"),
            );
        }
        assert_eq!(first_seed, [0; 32]);
        assert_eq!(second_seed, [0; 32]);
        assert_eq!(
            first.files.keys().map(String::as_str).collect::<Vec<_>>(),
            [
                "hashes.aiph",
                "manifest.aipm",
                "provenance.json",
                "signature.ed25519",
                "tensors-0000.aipt",
                "tokenizer.aiptok",
            ]
        );
    }

    #[cfg(unix)]
    #[test]
    #[allow(
        clippy::too_many_lines,
        reason = "one supply-chain regression keeps package, loader, service, and event denial correlated"
    )]
    fn owner_pinned_qwen_v11_is_rejected_by_loader_and_service_identity() {
        let root = temporary_root("qwen-v11-future-runtime");
        let store = root.join("models");
        let runtime = root.join("runtime");
        let package_path = store.join("synthetic-import.paimodel");
        for directory in [&root, &store, &runtime] {
            fs::create_dir(directory).expect("fixture directory must create");
            fs::set_permissions(directory, fs::Permissions::from_mode(0o700))
                .expect("fixture directory mode must set");
        }
        create_private_directory(&package_path).expect("package directory must create");

        let (source, tokenizer) = source_and_tokenizer();
        let key_id = SignerKeyId::from_bytes([0x53; 16]).expect("key ID must validate");
        let signing_seed = [0x54; 32];
        let mut seed = signing_seed;
        let legacy =
            build_package(&spec(), "model.safetensors", &source, &tokenizer, key_id, &mut seed)
                .expect("legacy source fixture must build");
        let package = resign_as_bounded_qwen_v11(&legacy, key_id, signing_seed);
        let manifest = DirectoryManifest::parse(
            package.files().get("manifest.aipm").expect("manifest must exist"),
        )
        .expect("signed v1.1 manifest must parse");
        assert_eq!(manifest.format_version(), DIRECTORY_QWEN2_MANIFEST_VERSION);
        assert_eq!(manifest.required_engine_version(), DIRECTORY_QWEN2_MINIMUM_ENGINE_VERSION);
        write_package_files(&package_path, &package).expect("package must write and seal");
        fs::set_permissions(&store, fs::Permissions::from_mode(0o500))
            .expect("model store must seal");

        let trusted = TrustedSigner::active(key_id, package.signer_public_key())
            .expect("trusted signer must validate");
        let budget = ResourceBudget::new(1 << 30, 1 << 28, 1).expect("budget must validate");
        let mut registry = ModelRegistry::new(&store).expect("registry must construct");
        registry
            .register(
                spec().model_id().clone(),
                "synthetic-import.paimodel",
                package.package_hash(),
                trusted.clone(),
            )
            .expect("signed package must register");
        let mut loader_events = Vec::new();
        let error = registry
            .load_registered_directory(spec().model_id(), &budget, &mut |event| {
                loader_events.push(event);
            })
            .expect_err("owner-pinned Qwen package must remain ineligible");
        assert!(
            matches!(
                error,
                EngineError::AuthenticationFailure(_) | EngineError::CorruptModelPackage(_)
            ),
            "{error:?}"
        );
        assert_eq!(
            loader_events,
            vec![SecurityEvent::ModelPackageRejected {
                model_id: spec().model_id().clone(),
                reason: ModelPackageRejection::Corrupt,
            }]
        );
        let mut cancellable_events = Vec::new();
        let error = registry
            .load_registered_directory_cancellable(
                spec().model_id(),
                &budget,
                &std::sync::atomic::AtomicBool::new(false),
                &mut |event| cancellable_events.push(event),
            )
            .expect_err("cancellable owner-pinned Qwen package must remain ineligible");
        assert!(
            matches!(
                error,
                EngineError::AuthenticationFailure(_) | EngineError::CorruptModelPackage(_)
            ),
            "{error:?}"
        );
        assert_eq!(cancellable_events, loader_events);

        let config = ServiceConfig::new(&store, &runtime, runtime.join("engine.sock"))
            .expect("service config must validate");
        let mut service = EngineService::new(
            config,
            SchedulerLimits::new(8, 1, 1 << 20).expect("scheduler limits must validate"),
        )
        .expect("service must construct");
        service
            .register_configured_model(
                spec().model_id().clone(),
                "synthetic-import.paimodel",
                package.package_hash(),
                trusted,
            )
            .expect("service package must register");
        let mut service_events = Vec::new();
        let error =
            match service.load_registered_cpu_model(spec().model_id(), &budget, &mut |event| {
                service_events.push(event);
            }) {
                Ok(_) => panic!("owner-pinned Qwen service load must remain ineligible"),
                Err(error) => error,
            };
        assert!(
            matches!(
                error,
                EngineError::AuthenticationFailure(_) | EngineError::CorruptModelPackage(_)
            ),
            "{error:?}"
        );
        assert_eq!(
            service_events,
            vec![SecurityEvent::ModelPackageRejected {
                model_id: spec().model_id().clone(),
                reason: ModelPackageRejection::Corrupt,
            }]
        );

        drop(service);
        drop(registry);
        remove_sealed_store(&root, &store, &package_path);
    }

    #[cfg(unix)]
    #[test]
    #[allow(clippy::too_many_lines)]
    fn quantized_packages_are_deterministic_signed_loadable_measured_and_accounted() {
        for (profile, expected_format) in [
            (QuantizationProfile::BalancedQ8, Quantization::AipQ8V1),
            (QuantizationProfile::CompactQ4, Quantization::AipQ4V1),
        ] {
            let root = temporary_root(profile.as_str());
            let store = root.join("models");
            let runtime = root.join("runtime");
            let package_path = store.join("synthetic-import.paimodel");
            fs::create_dir(&root).expect("root must create");
            fs::create_dir(&store).expect("store must create");
            fs::create_dir(&runtime).expect("runtime must create");
            create_private_directory(&package_path).expect("package directory must create");
            for directory in [&root, &store, &runtime] {
                fs::set_permissions(directory, fs::Permissions::from_mode(0o700))
                    .expect("private directory mode must set");
            }
            let (source, tokenizer) = source_and_tokenizer();
            let spec = spec_with_profile(profile);
            let key_id = SignerKeyId::from_bytes([0x35; 16]).expect("key ID must validate");
            let mut first_seed = [0x36; 32];
            let mut second_seed = first_seed;
            let first = build_package(
                &spec,
                "model.safetensors",
                &source,
                &tokenizer,
                key_id,
                &mut first_seed,
            )
            .expect("quantized package must build");
            let second = build_package(
                &spec,
                "model.safetensors",
                &source,
                &tokenizer,
                key_id,
                &mut second_seed,
            )
            .expect("repeat quantized package must build");
            assert_eq!(first, second);
            let manifest = DirectoryManifest::parse(
                first.files().get("manifest.aipm").expect("manifest must exist"),
            )
            .expect("quantized manifest must parse");
            assert_eq!(manifest.quantization(), expected_format);
            assert!(manifest.tensors().iter().any(|tensor| {
                tensor.shape().rank() == 2 && tensor.quantization() == expected_format
            }));
            assert!(manifest.tensors().iter().all(|tensor| {
                tensor.shape().rank() == 2 || tensor.quantization() == Quantization::None
            }));
            let provenance = std::str::from_utf8(
                first.files().get("provenance.json").expect("provenance must exist"),
            )
            .expect("provenance must be UTF-8");
            assert!(
                provenance.contains(&format!("\"quantization_profile\": \"{}\"", profile.as_str()))
            );
            assert!(provenance.contains("\"quantization_layer_errors\""));
            assert!(provenance.contains("\"maximum_absolute_error\""));
            write_package_files(&package_path, &first).expect("quantized package files must write");
            fs::set_permissions(&store, fs::Permissions::from_mode(0o500))
                .expect("model store must seal");

            let trusted = TrustedSigner::active(key_id, first.signer_public_key())
                .expect("trusted signer must validate");
            let mut registry = ModelRegistry::new(&store).expect("registry must construct");
            registry
                .register(
                    spec.model_id().clone(),
                    "synthetic-import.paimodel",
                    first.package_hash(),
                    trusted.clone(),
                )
                .expect("quantized package must register");
            let budget = ResourceBudget::new(1 << 30, 1 << 28, 1).expect("budget must validate");
            let mut cancellation_events = Vec::new();
            let cancellation = CancelAfterChecks::new(6);
            assert!(matches!(
                registry.load_registered_directory_cancellable(
                    spec.model_id(),
                    &budget,
                    &cancellation,
                    &mut |event| cancellation_events.push(event),
                ),
                Err(EngineError::Cancellation(_))
            ));
            assert!(cancellation_events.is_empty());
            let loaded = registry
                .load_registered_directory(spec.model_id(), &budget, &mut |_| {})
                .expect("signed quantized package must load");
            let pool = MemoryPool::new(1 << 20).expect("model pool must validate");
            let materialization_cancellation = CancelAfterChecks::new(9);
            assert!(matches!(
                loaded.materialize_tensors_cancellable(&pool, &materialization_cancellation),
                Err(EngineError::Cancellation(_))
            ));
            assert_eq!(pool.statistics().expect("metrics must read").reserved_bytes(), 0);
            let tensors =
                loaded.materialize_tensors(&pool).expect("quantized tensors must materialize");
            assert!(pool.statistics().expect("metrics must read").reserved_bytes() > 0);
            drop(tensors);
            assert_eq!(pool.statistics().expect("metrics must read").reserved_bytes(), 0);

            let config = ServiceConfig::new(&store, &runtime, runtime.join("engine.sock"))
                .expect("service config must validate");
            let mut service = EngineService::new(
                config,
                SchedulerLimits::new(8, 1, 1 << 20).expect("scheduler limits must validate"),
            )
            .expect("service must construct");
            service
                .register_configured_model(
                    spec.model_id().clone(),
                    "synthetic-import.paimodel",
                    first.package_hash(),
                    trusted,
                )
                .expect("service package must register");
            let cpu_model = service
                .load_registered_cpu_model(spec.model_id(), &budget, &mut |_| {})
                .expect("quantized CPU model must load");
            let request = GenerationRequest::new(
                RequestId::new(9).expect("request ID must validate"),
                spec.model_id().clone(),
                "A",
                2,
                ContextLimit::new(8).expect("context must validate"),
                budget,
            )
            .expect("request must validate");
            let cancelled = std::sync::atomic::AtomicBool::new(true);
            assert!(matches!(
                cpu_model.generate_greedy(
                    &request,
                    SequenceId::new(9).expect("sequence ID must validate"),
                    &cancelled,
                    |_| Ok(()),
                ),
                Err(EngineError::Cancellation(_))
            ));
            drop(cpu_model);
            drop(loaded);
            remove_sealed_store(&root, &store, &package_path);
        }
    }

    #[cfg(unix)]
    #[test]
    fn quantized_transformer_logit_and_generation_divergence_is_measured() {
        let reference = load_cpu_profile(QuantizationProfile::ReferenceF32);
        let q8 = load_cpu_profile(QuantizationProfile::BalancedQ8);
        let q4 = load_cpu_profile(QuantizationProfile::CompactQ4);
        let reference_logits = profile_logits(&reference, 101, 101);
        let q8_logits = profile_logits(&q8, 102, 102);
        let q4_logits = profile_logits(&q4, 103, 103);
        let measure = |observed: &[f32]| {
            assert_eq!(reference_logits.len(), observed.len());
            let mut maximum = 0.0_f64;
            let mut absolute_sum = 0.0_f64;
            let mut squared_sum = 0.0_f64;
            for (expected, actual) in reference_logits.iter().zip(observed) {
                assert!(expected.is_finite() && actual.is_finite());
                let error = (f64::from(*expected) - f64::from(*actual)).abs();
                maximum = maximum.max(error);
                absolute_sum += error;
                squared_sum += error * error;
            }
            let count = u32::try_from(observed.len()).expect("synthetic logits fit u32");
            let divisor = f64::from(count);
            (maximum, absolute_sum / divisor, (squared_sum / divisor).sqrt())
        };
        let q8_error = measure(&q8_logits);
        let q4_error = measure(&q4_logits);
        let reference_tokens = profile_tokens(&reference, 201, 201);
        let q8_tokens = profile_tokens(&q8, 202, 202);
        let q4_tokens = profile_tokens(&q4, 203, 203);
        let divergence = |observed: &[u32]| {
            reference_tokens
                .iter()
                .zip(observed)
                .filter(|(expected, actual)| expected != actual)
                .count()
                + reference_tokens.len().abs_diff(observed.len())
        };
        let q8_divergence = divergence(&q8_tokens);
        let q4_divergence = divergence(&q4_tokens);
        eprintln!(
            "synthetic quantization evidence: q8 max={:.9e} mean={:.9e} rms={:.9e} token_divergence={}/{}; q4 max={:.9e} mean={:.9e} rms={:.9e} token_divergence={}/{}",
            q8_error.0,
            q8_error.1,
            q8_error.2,
            q8_divergence,
            reference_tokens.len().max(q8_tokens.len()),
            q4_error.0,
            q4_error.1,
            q4_error.2,
            q4_divergence,
            reference_tokens.len().max(q4_tokens.len()),
        );
        assert!(q8_error.0 <= 0.003 && q8_error.2 <= 0.001);
        assert!(q4_error.0 <= 0.04 && q4_error.2 <= 0.02);
        assert_eq!(q8_divergence, 0);
        assert_eq!(q4_divergence, 4);
    }

    #[cfg(unix)]
    #[test]
    fn deterministic_multishard_import_is_order_independent_and_loader_verified() {
        let root = temporary_root("multishard");
        let quarantine = root.join("quarantine");
        let store = root.join("models");
        fs::create_dir(&root).expect("root must create");
        fs::create_dir(&quarantine).expect("quarantine must create");
        fs::create_dir(&store).expect("store must create");
        for directory in [&root, &quarantine, &store] {
            fs::set_permissions(directory, fs::Permissions::from_mode(0o700))
                .expect("private directory mode must set");
        }

        let (source, tokenizer) = source_and_tokenizer();
        let (first_source, second_source) = split_safetensors_source(&source);
        let first = SafetensorsSource::new("model-00001-of-00002.safetensors", &first_source)
            .expect("first source must validate");
        let second = SafetensorsSource::new("model-00002-of-00002.safetensors", &second_source)
            .expect("second source must validate");
        let key_id = SignerKeyId::from_bytes([0x41; 16]).expect("key ID must validate");
        let mut forward_seed = [0x42; 32];
        let mut reverse_seed = forward_seed;
        let forward =
            build_package_sharded(&spec(), &[first, second], &tokenizer, key_id, &mut forward_seed)
                .expect("forward shard order must import");
        let reverse =
            build_package_sharded(&spec(), &[second, first], &tokenizer, key_id, &mut reverse_seed)
                .expect("reverse shard order must import identically");
        assert_eq!(forward, reverse);
        assert_eq!(forward_seed, [0; 32]);
        assert_eq!(reverse_seed, [0; 32]);
        assert!(forward.files().contains_key("tensors-0000.aipt"));
        assert!(forward.files().contains_key("tensors-0001.aipt"));
        let manifest = DirectoryManifest::parse(
            forward.files().get("manifest.aipm").expect("manifest must exist"),
        )
        .expect("multishard manifest must parse");
        assert_eq!(manifest.shards().len(), 2);
        assert!(manifest.tensors().iter().any(|tensor| tensor.shard_index() == 0));
        assert!(manifest.tensors().iter().any(|tensor| tensor.shard_index() == 1));
        let provenance = std::str::from_utf8(
            forward.files().get("provenance.json").expect("provenance must exist"),
        )
        .expect("provenance must be UTF-8");
        let first_position = provenance
            .find("model-00001-of-00002.safetensors")
            .expect("first source must be recorded");
        let second_position = provenance
            .find("model-00002-of-00002.safetensors")
            .expect("second source must be recorded");
        assert!(first_position < second_position);

        fs::write(quarantine.join(first.name()), first.bytes()).expect("first shard must write");
        fs::write(quarantine.join(second.name()), second.bytes()).expect("second shard must write");
        let tokenizer_metadata = source_tokenizer_metadata();
        fs::write(quarantine.join("tokenizer.aiptokmeta"), &tokenizer_metadata)
            .expect("tokenizer metadata must write");
        let mut local_seed = [0x42; 32];
        let receipt = import_local_sharded(
            &quarantine,
            &[PathBuf::from(second.name()), PathBuf::from(first.name())],
            Path::new("tokenizer.aiptokmeta"),
            &store,
            &spec(),
            key_id,
            &mut local_seed,
            None,
        )
        .expect("local multishard import must succeed");
        assert_eq!(local_seed, [0; 32]);
        fs::set_permissions(&store, fs::Permissions::from_mode(0o500))
            .expect("production store must seal read-only");
        let trusted = TrustedSigner::active(key_id, receipt.signer_public_key())
            .expect("trusted signer must validate");
        let mut registry = ModelRegistry::new(&store).expect("registry must construct");
        registry
            .register(
                spec().model_id().clone(),
                "synthetic-import.paimodel",
                receipt.package_hash(),
                trusted,
            )
            .expect("multishard package must register");
        let budget = ResourceBudget::new(1 << 30, 1 << 28, 1).expect("budget must validate");
        let loaded = registry
            .load_registered_directory(spec().model_id(), &budget, &mut |_| {})
            .expect("signed multishard package must load");
        assert_eq!(loaded.manifest().shards().len(), 2);
        assert_tokenizer_provenance(&loaded, &tokenizer_metadata);

        remove_sealed_store(&root, &store, receipt.package_path());
    }

    #[test]
    fn multishard_import_rejects_empty_duplicate_and_duplicate_tensor_sources() {
        let (source, tokenizer) = source_and_tokenizer();
        let key_id = SignerKeyId::from_bytes([0x43; 16]).expect("key ID must validate");
        let mut empty_seed = [0x44; 32];
        assert!(build_package_sharded(&spec(), &[], &tokenizer, key_id, &mut empty_seed).is_err());
        assert_eq!(empty_seed, [0; 32]);

        let first = SafetensorsSource::new("model-00001.safetensors", &source)
            .expect("source must validate");
        let duplicate_name = SafetensorsSource::new("model-00001.safetensors", &source)
            .expect("duplicate source name is individually valid");
        let mut duplicate_name_seed = [0x45; 32];
        assert!(
            build_package_sharded(
                &spec(),
                &[first, duplicate_name],
                &tokenizer,
                key_id,
                &mut duplicate_name_seed,
            )
            .is_err()
        );
        assert_eq!(duplicate_name_seed, [0; 32]);

        let duplicate_tensor = SafetensorsSource::new("model-00002.safetensors", &source)
            .expect("second source must validate");
        let mut duplicate_tensor_seed = [0x46; 32];
        assert!(
            build_package_sharded(
                &spec(),
                &[first, duplicate_tensor],
                &tokenizer,
                key_id,
                &mut duplicate_tensor_seed,
            )
            .is_err()
        );
        assert_eq!(duplicate_tensor_seed, [0; 32]);
    }

    #[test]
    fn f16_and_bf16_sources_convert_to_finite_f32_package_storage() {
        let (source, tokenizer) = source_and_tokenizer();
        let key_id = SignerKeyId::from_bytes([0x47; 16]).expect("key ID must validate");
        for (dtype, seed_byte) in [(DType::F16, 0x48), (DType::BFloat16, 0x49)] {
            let converted_source = convert_fixture_source_dtype(&source, dtype);
            let mut seed = [seed_byte; 32];
            let package = build_package(
                &spec(),
                "model.safetensors",
                &converted_source,
                &tokenizer,
                key_id,
                &mut seed,
            )
            .expect("supported compact source dtype must convert");
            assert_eq!(seed, [0; 32]);
            let manifest = DirectoryManifest::parse(
                package.files().get("manifest.aipm").expect("manifest must exist"),
            )
            .expect("manifest must parse");
            assert!(manifest.tensors().iter().all(|tensor| tensor.dtype() == DType::F32));
            let shard = package.files().get("tensors-0000.aipt").expect("shard must exist");
            for tensor in manifest.tensors() {
                let start = usize::try_from(tensor.byte_offset()).expect("offset must fit");
                let end = start
                    .checked_add(usize::try_from(tensor.byte_length()).expect("length must fit"))
                    .expect("range must fit");
                validate_finite_f32(&shard[start..end]).expect("converted tensor must be finite");
            }
        }
    }

    #[cfg(unix)]
    #[test]
    fn registry_v2_rejects_a_package_after_deterministic_importer_contract_drift() {
        let root = temporary_root("registry-v2");
        let artifact = root.join("phase0-synthetic-import");
        let package_path = artifact.join("synthetic-import.paimodel");
        let registry_root = root.join("registry");
        fs::create_dir(&root).expect("approved root");
        fs::create_dir(&artifact).expect("artifact root");
        fs::create_dir(&package_path).expect("package root");
        fs::create_dir(&registry_root).expect("registry root");

        let (source, tokenizer) = source_and_tokenizer();
        let package_key = SignerKeyId::from_bytes([0x51; 16]).expect("package key ID");
        let mut seed = [0x52; 32];
        let package = build_package(
            &spec(),
            "model.safetensors",
            &source,
            &tokenizer,
            package_key,
            &mut seed,
        )
        .expect("package must build");
        for (name, bytes) in package.files() {
            let path = package_path.join(name);
            fs::write(&path, bytes).expect("package file must write");
            fs::set_permissions(path, fs::Permissions::from_mode(0o400))
                .expect("package file must seal");
        }
        let registry_path = registry_root.join("00000000000000000002.registry.json");
        fs::write(&registry_path, registry_v2_bytes()).expect("registry entry must write");
        fs::set_permissions(&registry_path, fs::Permissions::from_mode(0o400))
            .expect("registry entry must seal");
        for directory in [&package_path, &artifact, &registry_root, &root] {
            fs::set_permissions(directory, fs::Permissions::from_mode(0o500))
                .expect("approved directory must seal");
        }

        let mut wrong_owner = ModelRegistry::new(&root).expect("registry constructs");
        let wrong_selection = RegistrySelectionV2::new(
            ModelId::new("other-model").expect("model ID"),
            "phase0-synthetic-import",
            hash_package(registry_v2_bytes()),
            2,
            2,
            registry_v2_signer(),
            registry_v2_package_signer(),
        );
        assert!(wrong_selection.is_ok());
        assert!(wrong_owner.register_registry_v2(&wrong_selection.expect("selection")).is_err());
        assert!(wrong_owner.is_empty());

        let selection = registry_v2_selection();
        let mut registry = ModelRegistry::new(&root).expect("registry constructs");
        registry.register_registry_v2(&selection).expect("exact registry selection must register");
        let registration =
            registry.resolve(spec().model_id()).expect("registered model must resolve");
        assert_eq!(registration.mode(), RegistrationMode::ModelFoundryRegistryV2);
        assert_eq!(registration.package_path(), package_path);
        assert_eq!(registration.registry_artifact_id(), Some("phase0-synthetic-import"));
        assert_eq!(registration.registry_sequence(), Some(2));
        assert_eq!(registration.registry_entry_hash(), Some(hash_package(registry_v2_bytes())));

        let budget = ResourceBudget::new(1 << 30, 1 << 28, 1).expect("budget");
        let mut events = Vec::new();
        let result = registry.load_registered_directory(spec().model_id(), &budget, &mut |event| {
            events.push(event);
        });
        assert!(matches!(result, Err(EngineError::SignatureFailure(_))));
        assert_eq!(events.len(), 1);

        for directory in [&root, &registry_root, &artifact, &package_path] {
            fs::set_permissions(directory, fs::Permissions::from_mode(0o700))
                .expect("test directory must unseal");
        }
        fs::remove_dir_all(root).expect("test root must clean up");
    }

    #[cfg(unix)]
    #[test]
    fn registry_v2_rejects_latest_fallback_registry_symlinks_and_artifact_symlinks() {
        use std::os::unix::fs::symlink;

        let root = temporary_root("registry-v2-boundaries");
        let registry_root = root.join("registry");
        let selected_path = registry_root.join("00000000000000000002.registry.json");
        let newer_path = registry_root.join("00000000000000000003.registry.json");
        let external_entry = temporary_root("external-registry-v2");
        let external_artifact = temporary_root("external-artifact-v2");
        let artifact = root.join("phase0-synthetic-import");
        fs::create_dir(&root).expect("approved root");
        fs::create_dir(&registry_root).expect("registry root");
        fs::write(&newer_path, registry_v2_bytes()).expect("newer registry entry must write");
        fs::set_permissions(&newer_path, fs::Permissions::from_mode(0o400))
            .expect("newer entry must seal");
        fs::set_permissions(&registry_root, fs::Permissions::from_mode(0o500))
            .expect("registry root must seal");
        fs::set_permissions(&root, fs::Permissions::from_mode(0o500))
            .expect("approved root must seal");

        let mut latest_fallback = ModelRegistry::new(&root).expect("registry constructs");
        assert!(latest_fallback.register_registry_v2(&registry_v2_selection()).is_err());
        assert!(latest_fallback.is_empty());

        fs::set_permissions(&root, fs::Permissions::from_mode(0o700))
            .expect("approved root must unseal");
        fs::set_permissions(&registry_root, fs::Permissions::from_mode(0o700))
            .expect("registry root must unseal");
        fs::write(&external_entry, registry_v2_bytes()).expect("external entry must write");
        fs::set_permissions(&external_entry, fs::Permissions::from_mode(0o400))
            .expect("external entry must seal");
        symlink(&external_entry, &selected_path).expect("registry symlink must create");
        fs::set_permissions(&registry_root, fs::Permissions::from_mode(0o500))
            .expect("registry root must reseal");
        fs::set_permissions(&root, fs::Permissions::from_mode(0o500))
            .expect("approved root must reseal");

        let mut symlinked_registry = ModelRegistry::new(&root).expect("registry constructs");
        assert!(symlinked_registry.register_registry_v2(&registry_v2_selection()).is_err());
        assert!(symlinked_registry.is_empty());

        fs::set_permissions(&root, fs::Permissions::from_mode(0o700))
            .expect("approved root must unseal");
        fs::set_permissions(&registry_root, fs::Permissions::from_mode(0o700))
            .expect("registry root must unseal");
        fs::remove_file(&selected_path).expect("registry symlink must remove");
        fs::write(&selected_path, registry_v2_bytes()).expect("selected entry must write");
        fs::set_permissions(&selected_path, fs::Permissions::from_mode(0o400))
            .expect("selected entry must seal");
        fs::create_dir(&external_artifact).expect("external artifact must create");
        symlink(&external_artifact, &artifact).expect("artifact symlink must create");
        fs::set_permissions(&registry_root, fs::Permissions::from_mode(0o500))
            .expect("registry root must reseal");
        fs::set_permissions(&root, fs::Permissions::from_mode(0o500))
            .expect("approved root must reseal");

        let mut artifact_escape = ModelRegistry::new(&root).expect("registry constructs");
        artifact_escape
            .register_registry_v2(&registry_v2_selection())
            .expect("exact registry entry must register before package access");
        let budget = ResourceBudget::new(1 << 30, 1 << 28, 1).expect("budget");
        let mut events = Vec::new();
        assert!(
            artifact_escape
                .load_registered_directory(spec().model_id(), &budget, &mut |event| {
                    events.push(event);
                })
                .is_err()
        );
        assert_eq!(events.len(), 1);

        fs::set_permissions(&root, fs::Permissions::from_mode(0o700))
            .expect("approved root must unseal");
        fs::set_permissions(&registry_root, fs::Permissions::from_mode(0o700))
            .expect("registry root must unseal");
        fs::remove_dir_all(&root).expect("test root must clean up");
        fs::set_permissions(&external_entry, fs::Permissions::from_mode(0o600))
            .expect("external entry must unseal");
        fs::remove_file(external_entry).expect("external entry must clean up");
        fs::remove_dir(external_artifact).expect("external artifact must clean up");
    }

    #[test]
    fn import_rejects_wrong_shape_missing_unexpected_and_non_finite_data() {
        let (source, tokenizer) = source_and_tokenizer();
        let key_id = SignerKeyId::from_bytes([0x53; 16]).expect("key ID must validate");

        let mut wrong_shape_spec = spec();
        wrong_shape_spec.hidden_dimension = 8;
        let mut seed = [0x50; 32];
        assert!(
            build_package(
                &wrong_shape_spec,
                "model.safetensors",
                &source,
                &tokenizer,
                key_id,
                &mut seed,
            )
            .is_err()
        );
        assert_eq!(seed, [0; 32]);

        let mut wrong_vocabulary_spec = spec();
        wrong_vocabulary_spec.vocabulary_size = 258;
        let mut seed = [0x58; 32];
        assert!(
            build_package(
                &wrong_vocabulary_spec,
                "model.safetensors",
                &source,
                &tokenizer,
                key_id,
                &mut seed,
            )
            .is_err()
        );
        assert_eq!(seed, [0; 32]);

        let mut missing = source.clone();
        let needle = b"lm_head.weight";
        let start = missing
            .windows(needle.len())
            .position(|window| window == needle)
            .expect("fixture output tensor must exist");
        missing[start..start + needle.len()].copy_from_slice(b"unknown.weight");
        let mut seed = [0x51; 32];
        assert!(
            build_package(&spec(), "model.safetensors", &missing, &tokenizer, key_id, &mut seed,)
                .is_err()
        );

        let mut truncated = source.clone();
        truncated.pop();
        let mut seed = [0x54; 32];
        assert!(
            build_package(&spec(), "model.safetensors", &truncated, &tokenizer, key_id, &mut seed,)
                .is_err()
        );
        let mut nonfinite = source;
        let last = nonfinite.len() - 4;
        nonfinite[last..].copy_from_slice(&f32::NAN.to_le_bytes());
        let mut seed = [0x55; 32];
        assert!(
            build_package(&spec(), "model.safetensors", &nonfinite, &tokenizer, key_id, &mut seed,)
                .is_err()
        );
    }

    #[test]
    fn importer_has_no_executable_or_pickle_source_format_path() {
        let (source, tokenizer) = source_and_tokenizer();
        let key_id = SignerKeyId::from_bytes([0x56; 16]).expect("key ID must validate");
        for name in [
            "model.bin",
            "model.pt",
            "model.pth",
            "model.py",
            "model.js",
            "model.wasm",
            "model.so",
            "model.dylib",
            "model.dll",
        ] {
            let mut seed = [0x57; 32];
            assert!(
                build_package(&spec(), name, &source, &tokenizer, key_id, &mut seed).is_err(),
                "prohibited source format {name} was accepted"
            );
            assert_eq!(seed, [0; 32]);
        }
    }

    #[cfg(unix)]
    #[test]
    #[allow(clippy::too_many_lines)]
    fn importer_to_directory_loader_round_trip_is_signed_and_byte_exact() {
        let root = temporary_root("roundtrip");
        let quarantine = root.join("quarantine");
        let store = root.join("models");
        fs::create_dir(&root).expect("root must create");
        fs::create_dir(&quarantine).expect("quarantine must create");
        fs::create_dir(&store).expect("store must create");
        for directory in [&root, &quarantine, &store] {
            fs::set_permissions(directory, fs::Permissions::from_mode(0o700))
                .expect("private directory mode must set");
        }
        let (source, tokenizer) = source_and_tokenizer();
        let tokenizer_metadata = write_import_sources(&quarantine, &source);
        let key_id = SignerKeyId::from_bytes([0x61; 16]).expect("key ID must validate");
        let mut seed = [0x62; 32];
        let receipt = import_local(
            &quarantine,
            Path::new("model.safetensors"),
            Path::new("tokenizer.aiptokmeta"),
            &store,
            &spec(),
            key_id,
            &mut seed,
            None,
        )
        .expect("offline import must succeed");
        assert_eq!(seed, [0; 32]);
        fs::set_permissions(&store, fs::Permissions::from_mode(0o500))
            .expect("production store must seal read-only");

        let trusted = TrustedSigner::active(key_id, receipt.signer_public_key())
            .expect("trusted signer must validate");
        let mut registry = ModelRegistry::new(&store).expect("registry must construct");
        registry
            .register(
                spec().model_id().clone(),
                "synthetic-import.paimodel",
                receipt.package_hash(),
                trusted.clone(),
            )
            .expect("package must register");
        let budget = ResourceBudget::new(1 << 30, 1 << 28, 1).expect("budget must validate");
        let mut events = Vec::new();
        let loaded = registry
            .load_registered_directory(spec().model_id(), &budget, &mut |event| {
                events.push(event);
            })
            .expect("signed package must load");
        assert!(events.is_empty());
        assert_eq!(loaded.manifest().model_id(), spec().model_id());
        assert_eq!(loaded.tokenizer_bytes(), tokenizer);
        assert_tokenizer_provenance(&loaded, &tokenizer_metadata);

        let parsed_source = ParsedSafetensors::parse(&source).expect("source must parse");
        let source_embedding = parsed_source
            .tensors()
            .iter()
            .find(|tensor| tensor.name() == "model.embed_tokens.weight")
            .expect("source embedding must exist");
        let destination_embedding = loaded
            .manifest()
            .tensors()
            .iter()
            .find(|tensor| tensor.name() == "token_embedding.weight")
            .expect("destination embedding must exist");
        assert_eq!(
            loaded.tensor_bytes(destination_embedding).expect("mapped bytes must exist"),
            parsed_source.tensor_bytes(source_embedding).expect("source bytes must exist")
        );

        let runtime_root = root.join("runtime");
        fs::create_dir(&runtime_root).expect("runtime root must create");
        fs::set_permissions(&runtime_root, fs::Permissions::from_mode(0o700))
            .expect("runtime root mode must set");
        let config = ServiceConfig::new(&store, &runtime_root, runtime_root.join("engine.sock"))
            .expect("service config must validate");
        let mut service = EngineService::new(
            config,
            SchedulerLimits::new(8, 1, 1 << 20).expect("scheduler limits must validate"),
        )
        .expect("service must construct");
        service
            .register_configured_model(
                spec().model_id().clone(),
                "synthetic-import.paimodel",
                receipt.package_hash(),
                trusted,
            )
            .expect("service package must register");
        let mut service_events = Vec::new();
        let cpu_model = service
            .load_registered_cpu_model(spec().model_id(), &budget, &mut |event| {
                service_events.push(event);
            })
            .expect("production CPU composition must load the directory package");
        assert_eq!(cpu_model.model_id(), spec().model_id());
        assert!(service_events.is_empty());
        assert_signed_cpu_generation(&cpu_model, budget);

        #[cfg(target_os = "macos")]
        assert_signed_metal_generation(&service, budget, &mut service_events);

        assert_live_mapping_change_is_reported(
            &loaded,
            &receipt.package_path().join("tensors-0000.aipt"),
        );

        remove_sealed_store(&root, &store, receipt.package_path());
    }

    #[cfg(unix)]
    #[test]
    fn production_loader_rejects_corruption_untrusted_signers_symlinks_and_budget() {
        let root = temporary_root("rejections");
        let store = root.join("models");
        let package_path = store.join("synthetic-import.paimodel");
        fs::create_dir(&root).expect("root must create");
        fs::create_dir(&store).expect("store must create");
        fs::create_dir(&package_path).expect("package must create");
        for directory in [&root, &store, &package_path] {
            fs::set_permissions(directory, fs::Permissions::from_mode(0o700))
                .expect("private directory mode must set");
        }
        let (source, tokenizer) = source_and_tokenizer();
        let key_id = SignerKeyId::from_bytes([0x71; 16]).expect("key ID must validate");
        let mut seed = [0x72; 32];
        let package =
            build_package(&spec(), "model.safetensors", &source, &tokenizer, key_id, &mut seed)
                .expect("package must build");
        for (name, bytes) in package.files() {
            let path = package_path.join(name);
            fs::write(&path, bytes).expect("package file must write");
            fs::set_permissions(path, fs::Permissions::from_mode(0o400))
                .expect("package file mode must set");
        }
        fs::set_permissions(&package_path, fs::Permissions::from_mode(0o500))
            .expect("package directory must seal");
        fs::set_permissions(&store, fs::Permissions::from_mode(0o500))
            .expect("model store must seal");
        let signer = TrustedSigner::active(key_id, package.signer_public_key())
            .expect("trusted signer must validate");
        let budget = ResourceBudget::new(1 << 30, 1 << 28, 1).expect("budget must validate");

        let load = |trusted: TrustedSigner,
                    expected_hash: ModelPackageHash,
                    budget: &ResourceBudget|
         -> (
            Result<personalai_engine_loader::VerifiedDirectoryPackage, EngineError>,
            usize,
        ) {
            let mut registry = ModelRegistry::new(&store).expect("registry must construct");
            registry
                .register(
                    spec().model_id().clone(),
                    "synthetic-import.paimodel",
                    expected_hash,
                    trusted,
                )
                .expect("package must register");
            let mut event_count = 0;
            let result =
                registry.load_registered_directory(spec().model_id(), budget, &mut |_event| {
                    event_count += 1;
                });
            (result, event_count)
        };

        let small_budget = ResourceBudget::new(1, 1, 1).expect("small budget must validate");
        let (result, event_count) = load(signer.clone(), package.package_hash(), &small_budget);
        assert!(matches!(result, Err(EngineError::ResourceExhaustion(_))));
        assert_eq!(event_count, 1);

        let wrong_signer = TrustedSigner::active(
            key_id,
            SigningKey::from_bytes(&[0x73; 32]).verifying_key().to_bytes(),
        )
        .expect("alternate signer must validate");
        let (result, event_count) = load(wrong_signer, package.package_hash(), &budget);
        assert!(matches!(result, Err(EngineError::SignatureFailure(_))));
        assert_eq!(event_count, 1);

        for filename in ["tokenizer.aiptok", "tensors-0000.aipt", "signature.ed25519"] {
            let path = package_path.join(filename);
            with_corrupted_read_only_file(&path, filename == "signature.ed25519", || {
                let (result, event_count) = load(signer.clone(), package.package_hash(), &budget);
                assert!(result.is_err(), "corrupt {filename} was accepted");
                assert_eq!(event_count, 1);
            });
        }

        let outside = root.join("outside.aipt");
        fs::write(&outside, b"outside").expect("outside fixture must write");
        let shard = package_path.join("tensors-0000.aipt");
        fs::set_permissions(&store, fs::Permissions::from_mode(0o700))
            .expect("store must unseal for symlink fixture");
        fs::set_permissions(&package_path, fs::Permissions::from_mode(0o700))
            .expect("package must unseal for symlink fixture");
        fs::remove_file(&shard).expect("shard must remove");
        std::os::unix::fs::symlink(&outside, &shard).expect("symlink must create");
        fs::set_permissions(&package_path, fs::Permissions::from_mode(0o500))
            .expect("package must reseal");
        fs::set_permissions(&store, fs::Permissions::from_mode(0o500)).expect("store must reseal");
        let (result, event_count) = load(signer, package.package_hash(), &budget);
        assert!(matches!(result, Err(EngineError::AuthenticationFailure(_))));
        assert_eq!(event_count, 1);

        fs::set_permissions(&store, fs::Permissions::from_mode(0o700))
            .expect("temporary store must unseal for cleanup");
        fs::set_permissions(&package_path, fs::Permissions::from_mode(0o700))
            .expect("temporary package must unseal for cleanup");
        fs::remove_dir_all(&root).expect("temporary fixture must clean up");
    }

    #[cfg(unix)]
    #[test]
    fn loader_rejects_resigned_manifest_hash_inconsistency() {
        let root = temporary_root("cross-hashes");
        let store = root.join("models");
        let package_path = store.join("synthetic-import.paimodel");
        fs::create_dir(&root).expect("root must create");
        fs::create_dir(&store).expect("store must create");
        fs::create_dir(&package_path).expect("package must create");
        for directory in [&root, &store, &package_path] {
            fs::set_permissions(directory, fs::Permissions::from_mode(0o700))
                .expect("private directory mode must set");
        }
        let (source, tokenizer) = source_and_tokenizer();
        let key_id = SignerKeyId::from_bytes([0x78; 16]).expect("key ID must validate");
        let original_seed = [0x79; 32];
        let mut seed = original_seed;
        let original =
            build_package(&spec(), "model.safetensors", &source, &tokenizer, key_id, &mut seed)
                .expect("package must build");
        let variants = [
            resign_manifest(&original, |manifest| manifest[96] ^= 1, key_id, original_seed),
            resign_manifest(
                &original,
                |manifest| manifest[DIRECTORY_MANIFEST_HEADER_BYTES + 16] ^= 1,
                key_id,
                original_seed,
            ),
        ];
        let signer = TrustedSigner::active(key_id, original.signer_public_key())
            .expect("trusted signer must validate");
        let budget = ResourceBudget::new(1 << 30, 1 << 28, 1).expect("budget must validate");
        for variant in variants {
            fs::set_permissions(&store, fs::Permissions::from_mode(0o700))
                .expect("store must unseal for fixture write");
            fs::set_permissions(&package_path, fs::Permissions::from_mode(0o700))
                .expect("package must unseal for fixture write");
            for (name, bytes) in variant.files() {
                let path = package_path.join(name);
                if path.exists() {
                    fs::set_permissions(&path, fs::Permissions::from_mode(0o600))
                        .expect("variant file must become writable");
                }
                fs::write(&path, bytes).expect("variant file must write");
                fs::set_permissions(path, fs::Permissions::from_mode(0o400))
                    .expect("variant file mode must set");
            }
            fs::set_permissions(&package_path, fs::Permissions::from_mode(0o500))
                .expect("variant package must seal");
            fs::set_permissions(&store, fs::Permissions::from_mode(0o500))
                .expect("variant store must seal");
            let mut registry = ModelRegistry::new(&store).expect("registry must construct");
            registry
                .register(
                    spec().model_id().clone(),
                    "synthetic-import.paimodel",
                    variant.package_hash(),
                    signer.clone(),
                )
                .expect("variant must register");
            let mut events = Vec::new();
            assert!(
                registry
                    .load_registered_directory(spec().model_id(), &budget, &mut |event| {
                        events.push(event);
                    })
                    .is_err()
            );
            assert_eq!(events.len(), 1);
        }
        fs::set_permissions(&store, fs::Permissions::from_mode(0o700))
            .expect("temporary store must unseal for cleanup");
        fs::set_permissions(&package_path, fs::Permissions::from_mode(0o700))
            .expect("temporary package must unseal for cleanup");
        fs::remove_dir_all(&root).expect("temporary fixture must clean up");
    }

    #[cfg(unix)]
    #[test]
    fn local_import_refuses_urls_escape_and_unaudited_overwrite() {
        let root = temporary_root("import-policy");
        let quarantine = root.join("quarantine");
        let store = root.join("models");
        fs::create_dir(&root).expect("root must create");
        fs::create_dir(&quarantine).expect("quarantine must create");
        fs::create_dir(&store).expect("store must create");
        for directory in [&root, &quarantine, &store] {
            fs::set_permissions(directory, fs::Permissions::from_mode(0o700))
                .expect("private directory mode must set");
        }
        let (source, _tokenizer) = source_and_tokenizer();
        fs::write(quarantine.join("model.safetensors"), &source).expect("source must write");
        fs::write(quarantine.join("tokenizer.aiptokmeta"), source_tokenizer_metadata())
            .expect("tokenizer metadata must write");
        let outside = root.join("outside.safetensors");
        fs::write(&outside, &source).expect("outside source must write");
        std::os::unix::fs::symlink(&outside, quarantine.join("escape.safetensors"))
            .expect("escape symlink must create");
        let key_id = SignerKeyId::from_bytes([0x74; 16]).expect("key ID must validate");

        let mut url_seed = [0x75; 32];
        assert!(
            import_local(
                &quarantine,
                Path::new("https://invalid/model.safetensors"),
                Path::new("tokenizer.aiptokmeta"),
                &store,
                &spec(),
                key_id,
                &mut url_seed,
                None,
            )
            .is_err()
        );
        assert_eq!(url_seed, [0; 32]);
        let mut escape_seed = [0x76; 32];
        assert!(matches!(
            import_local(
                &quarantine,
                Path::new("escape.safetensors"),
                Path::new("tokenizer.aiptokmeta"),
                &store,
                &spec(),
                key_id,
                &mut escape_seed,
                None,
            ),
            Err(EngineError::AuthenticationFailure(_))
        ));

        let mut first_seed = [0x77; 32];
        import_local(
            &quarantine,
            Path::new("model.safetensors"),
            Path::new("tokenizer.aiptokmeta"),
            &store,
            &spec(),
            key_id,
            &mut first_seed,
            None,
        )
        .expect("first import must succeed");
        let mut refused_seed = [0x77; 32];
        assert!(
            import_local(
                &quarantine,
                Path::new("model.safetensors"),
                Path::new("tokenizer.aiptokmeta"),
                &store,
                &spec(),
                key_id,
                &mut refused_seed,
                None,
            )
            .is_err()
        );
        let mut force_seed = [0x77; 32];
        import_local(
            &quarantine,
            Path::new("model.safetensors"),
            Path::new("tokenizer.aiptokmeta"),
            &store,
            &spec(),
            key_id,
            &mut force_seed,
            Some("operator-approved deterministic replacement"),
        )
        .expect("audited replacement must succeed");
        assert!(store.join("synthetic-import.paimodel.previous").is_dir());
        assert!(store.join("import-audit.log").is_file());

        for package in [
            store.join("synthetic-import.paimodel"),
            store.join("synthetic-import.paimodel.previous"),
        ] {
            fs::set_permissions(package, fs::Permissions::from_mode(0o700))
                .expect("temporary package must unseal for cleanup");
        }
        fs::remove_dir_all(&root).expect("temporary fixture must clean up");
    }
}
