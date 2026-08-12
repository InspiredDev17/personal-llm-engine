//! Offline-only model import and package signing.
//!
//! This crate has filesystem read/write authority for an owner-selected local
//! quarantine and output store. Production inference crates do not depend on
//! it. It contains no socket, subprocess, dynamic loading, Python, or model-code
//! execution capability.

mod safetensors;

pub use safetensors::{
    MAX_SAFETENSORS_HEADER_BYTES, MAX_SOURCE_TENSOR_NAME_BYTES, ParsedSafetensors, SafeTensorRecord,
};

use ed25519_dalek::{Signer, SigningKey};
use personalai_engine_crypto::{
    directory_package_signature_message, hash_package, hash_signed_content,
};
use personalai_engine_memory::MemoryPool;
use personalai_engine_model_format::{
    DIRECTORY_DECODER_ARCHITECTURE_ID, DirectoryManifestBuilder, DirectoryShardRecord,
    DirectoryTensorRecord, HashCatalog, HashCatalogEntry, PackageFileKind, Quantization,
    SignatureEnvelope,
};
use personalai_engine_tokenizer::{FirstPartyBpeTokenizer, Tokenizer};
use personalai_engine_types::{
    ContextLimit, DType, EngineError, MAX_MODEL_PACKAGE_BYTES, ModelId, ModelPackageHash,
    SignerKeyId, TensorShape,
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
/// Maximum provenance document length.
pub const MAX_PROVENANCE_BYTES: usize = 1024 * 1024;
/// Package shard alignment used by the initial importer.
pub const IMPORT_TENSOR_ALIGNMENT: u32 = 64;

/// Closed source-to-canonical tensor mapping.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SourceMapping {
    /// Explicit Llama-family safetensors names mapped to the fixed decoder.
    LlamaSafetensorsV1,
}

impl SourceMapping {
    /// Parses the exact owner-selected mapping identifier.
    pub fn parse(value: &str) -> Result<Self, EngineError> {
        match value {
            LLAMA_SAFETENSORS_MAPPING_V1 => Ok(Self::LlamaSafetensorsV1),
            _ => Err(EngineError::invalid_input("source tensor mapping is unsupported")),
        }
    }

    /// Returns the canonical mapping identifier.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::LlamaSafetensorsV1 => LLAMA_SAFETENSORS_MAPPING_V1,
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
    ) -> Result<Self, EngineError> {
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
        })
    }

    /// Returns the destination model ID.
    pub const fn model_id(&self) -> &ModelId {
        &self.model_id
    }
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
    let result = build_package_inner(
        spec,
        source_name,
        source_bytes,
        tokenizer_bytes,
        signer_key_id,
        signing_seed,
    );
    signing_seed.zeroize();
    result
}

fn build_package_inner(
    spec: &ImportSpec,
    source_name: &str,
    source_bytes: &[u8],
    tokenizer_bytes: &[u8],
    signer_key_id: SignerKeyId,
    signing_seed: &mut [u8; 32],
) -> Result<ImportedPackage, EngineError> {
    if !source_name.ends_with(".safetensors")
        || source_name.contains('/')
        || source_name.contains('\\')
    {
        return Err(EngineError::invalid_input("source must be one local safetensors filename"));
    }
    let parsed = ParsedSafetensors::parse(source_bytes)?;
    let tokenizer_pool = MemoryPool::new(128 * 1024 * 1024)?;
    let tokenizer = FirstPartyBpeTokenizer::parse(tokenizer_bytes, &tokenizer_pool)?;
    if tokenizer.spec().vocabulary_size() != spec.vocabulary_size {
        return Err(corrupt("tokenizer vocabulary differs from the import specification"));
    }
    let (shard, tensor_records) = build_shard(spec, &parsed)?;
    let shard_hash = hash_package(&shard);
    let tokenizer_hash = hash_package(tokenizer_bytes);
    let shard_record = DirectoryShardRecord::new(0, usize_to_u64(shard.len())?, shard_hash)?;
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
        Quantization::None,
        spec.tied_output,
        tokenizer_hash,
        vec![shard_record],
        tensor_records,
    )?
    .build_bytes()?;
    let provenance = provenance(spec, source_name, source_bytes)?;
    let catalog = HashCatalog::new(vec![
        catalog_entry(PackageFileKind::Manifest, 0, "manifest.aipm", &manifest)?,
        catalog_entry(PackageFileKind::Tokenizer, 0, "tokenizer.aiptok", tokenizer_bytes)?,
        catalog_entry(PackageFileKind::Provenance, 0, "provenance.json", &provenance)?,
        catalog_entry(PackageFileKind::TensorShard, 0, "tensors-0000.aipt", &shard)?,
    ])?;
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
    let files = BTreeMap::from([
        ("hashes.aiph".to_owned(), catalog_bytes),
        ("manifest.aipm".to_owned(), manifest),
        ("provenance.json".to_owned(), provenance),
        ("signature.ed25519".to_owned(), signature),
        ("tensors-0000.aipt".to_owned(), shard),
        ("tokenizer.aiptok".to_owned(), tokenizer_bytes.to_vec()),
    ]);
    Ok(ImportedPackage { files, package_hash, signer_key_id, signer_public_key })
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
    let result = (|| {
        validate_local_path_text(source_path)?;
        validate_local_path_text(tokenizer_path)?;
        let source = read_quarantined_file(quarantine_root, source_path, MAX_MODEL_PACKAGE_BYTES)?;
        let tokenizer = read_quarantined_file(quarantine_root, tokenizer_path, 64 * 1024 * 1024)?;
        let source_name = source_path
            .file_name()
            .and_then(|name| name.to_str())
            .ok_or_else(|| EngineError::invalid_input("source filename is invalid UTF-8"))?;
        let package =
            build_package(spec, source_name, &source, &tokenizer, signer_key_id, signing_seed)?;
        install_package(output_root, spec, &package, force_audit_note)?;
        Ok(ImportReceipt {
            package_path: output_root.join(format!("{}.paimodel", spec.model_id.as_str())),
            package_hash: package.package_hash,
            signer_key_id,
            signer_public_key: package.signer_public_key,
        })
    })();
    signing_seed.zeroize();
    result
}

fn build_shard(
    spec: &ImportSpec,
    source: &ParsedSafetensors<'_>,
) -> Result<(Vec<u8>, Vec<DirectoryTensorRecord>), EngineError> {
    let mut mapped = BTreeMap::new();
    for tensor in source.tensors() {
        let canonical = canonical_name(spec.mapping, tensor.name(), spec.layer_count)?;
        if mapped.insert(canonical, tensor).is_some() {
            return Err(corrupt("source mapping produced a duplicate canonical tensor"));
        }
    }
    let expected = expected_shapes(spec)?;
    let mapped_names = mapped.keys().cloned().collect::<BTreeSet<_>>();
    let expected_names = expected.keys().cloned().collect::<BTreeSet<_>>();
    if mapped_names != expected_names {
        return Err(corrupt("source tensor inventory has missing or unexpected tensors"));
    }
    let mut shard = Vec::new();
    let mut records = Vec::new();
    records.try_reserve_exact(mapped.len()).map_err(|_| {
        EngineError::resource_exhaustion("import tensor metadata allocation failed")
    })?;
    for (name, tensor) in mapped {
        if tensor.dtype() != DType::F32 || tensor.shape() != &expected[&name] {
            return Err(corrupt("source tensor dtype or shape does not match architecture"));
        }
        let source_bytes = source.tensor_bytes(tensor)?;
        validate_finite_f32(source_bytes)?;
        let aligned =
            align_up(shard.len(), usize::try_from(IMPORT_TENSOR_ALIGNMENT).unwrap_or(64))?;
        if aligned > shard.len() {
            shard.resize(aligned, 0);
        }
        shard
            .try_reserve_exact(source_bytes.len())
            .map_err(|_| EngineError::resource_exhaustion("import shard allocation failed"))?;
        shard.extend_from_slice(source_bytes);
        records.push(DirectoryTensorRecord::new(
            name,
            DType::F32,
            tensor.shape().clone(),
            0,
            usize_to_u64(aligned)?,
            usize_to_u64(source_bytes.len())?,
            IMPORT_TENSOR_ALIGNMENT,
            Quantization::None,
            hash_package(source_bytes),
        )?);
    }
    Ok((shard, records))
}

fn expected_shapes(spec: &ImportSpec) -> Result<BTreeMap<String, TensorShape>, EngineError> {
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
    }
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

fn provenance(
    spec: &ImportSpec,
    source_name: &str,
    source_bytes: &[u8],
) -> Result<Vec<u8>, EngineError> {
    validate_provenance_text(source_name, 256)?;
    let document = format!(
        concat!(
            "{{\n",
            "  \"architecture\": \"{}\",\n",
            "  \"license_id\": \"{}\",\n",
            "  \"mapping\": \"{}\",\n",
            "  \"model_id\": \"{}\",\n",
            "  \"source_file\": \"{}\",\n",
            "  \"source_format\": \"safetensors\",\n",
            "  \"source_label\": \"{}\",\n",
            "  \"source_sha384\": \"{}\",\n",
            "  \"tool\": \"personalai-engine-model-importer\",\n",
            "  \"tool_version\": \"{}\"\n",
            "}}\n"
        ),
        DIRECTORY_DECODER_ARCHITECTURE_ID,
        spec.license_id,
        spec.mapping.as_str(),
        spec.model_id,
        source_name,
        spec.source_label,
        hash_package(source_bytes),
        env!("CARGO_PKG_VERSION"),
    );
    if document.len() > MAX_PROVENANCE_BYTES {
        return Err(EngineError::resource_exhaustion("provenance exceeds its hard limit"));
    }
    Ok(document.into_bytes())
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
    use personalai_engine_loader::{ModelRegistry, RegistrationMode, RegistrySelectionV2};
    use personalai_engine_model_format::{DIRECTORY_MANIFEST_HEADER_BYTES, ParsedModelPackage};
    use personalai_engine_scheduler::SchedulerLimits;
    use personalai_engine_server::{EngineService, ServiceConfig};
    use personalai_engine_testkit::DecoderPackageFixtureBuilder;
    use personalai_engine_types::ResourceBudget;
    use std::sync::atomic::{AtomicU64, Ordering};

    fn spec() -> ImportSpec {
        ImportSpec::new(
            ModelId::new("synthetic-import").expect("model ID must validate"),
            SourceMapping::LlamaSafetensorsV1,
            ContextLimit::new(8).expect("context must validate"),
            258,
            4,
            1,
            1,
            1,
            8,
            10_000.0,
            1.0,
            false,
            "synthetic-local-fixture",
            "test-only",
        )
        .expect("spec must validate")
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

    fn temporary_root(label: &str) -> PathBuf {
        static NEXT: AtomicU64 = AtomicU64::new(1);
        PathBuf::from("/tmp").join(format!(
            "personalai-importer-{label}-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ))
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

    fn external_name(canonical: &str) -> String {
        match canonical {
            "token_embedding.weight" => "model.embed_tokens.weight".to_owned(),
            "final_norm.weight" => "model.norm.weight".to_owned(),
            "output.weight" => "lm_head.weight".to_owned(),
            name => name
                .replace("layers.000.", "model.layers.0.")
                .replace("attn_norm.weight", "input_layernorm.weight")
                .replace("ffn_norm.weight", "post_attention_layernorm.weight")
                .replace("q_proj.weight", "self_attn.q_proj.weight")
                .replace("k_proj.weight", "self_attn.k_proj.weight")
                .replace("v_proj.weight", "self_attn.v_proj.weight")
                .replace("o_proj.weight", "self_attn.o_proj.weight")
                .replace("gate_proj.weight", "mlp.gate_proj.weight")
                .replace("up_proj.weight", "mlp.up_proj.weight")
                .replace("down_proj.weight", "mlp.down_proj.weight"),
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
                "5b97582279ce8e9dc1ed114b0b46b243fd1f6655f9ed694def28c1e52cbfeffd\
                 f1eb5e27f19fab5c7529b91d6471b329",
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
        let expected_files = [
            (
                "hashes.aiph",
                528,
                "5b97582279ce8e9dc1ed114b0b46b243fd1f6655f9ed694def28c1e52cbfeffd\
                 f1eb5e27f19fab5c7529b91d6471b329",
            ),
            (
                "manifest.aipm",
                2_880,
                "6e417bdfd2682c1e0c06a324670797b462089b2e0bff06f20615b543497f664d\
                 8509e0a8cb4e7f5ed7049cd8b3c42115",
            ),
            (
                "provenance.json",
                471,
                "b10362b19f93ea9ecfbc721af0635835ecc319ca71134608eca7dcff047c480d3\
                 7e2349b53681b317654f3130cf4bdd5",
            ),
            (
                "signature.ed25519",
                96,
                "1fd392c088ac1c2a22d18307f0f909743b76a27595cbdb9e12e8f99a58dbe31\
                 f76e883d726ec8a6347f44704621fb3b0",
            ),
            (
                "tensors-0000.aipt",
                9_120,
                "3f283f50f5c5937ad8625da6c692c31bdf5833733589520842ea946921c17eb2\
                 842aa31c2b09c0310baec907d3ff1d72",
            ),
            (
                "tokenizer.aiptok",
                1_328,
                "1c6de2b8090bc7b45683292075d18ea846361269e57f98d2d158eb38731eb8c2\
                 a342e58eeb15d6632520fd01ac40d194",
            ),
        ];
        for (name, expected_length, expected_hash) in expected_files {
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
    fn registry_v2_derives_and_loads_only_the_exact_owner_selected_package() {
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
        let loaded = registry
            .load_registered_directory(spec().model_id(), &budget, &mut |event| {
                events.push(event);
            })
            .expect("registry-backed package must load");
        assert!(events.is_empty());
        assert_eq!(loaded.manifest().model_id(), spec().model_id());
        assert_eq!(loaded.tokenizer_bytes(), tokenizer);

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
        fs::write(quarantine.join("model.safetensors"), &source).expect("source must write");
        fs::write(quarantine.join("tokenizer.aiptok"), &tokenizer).expect("tokenizer must write");
        let key_id = SignerKeyId::from_bytes([0x61; 16]).expect("key ID must validate");
        let mut seed = [0x62; 32];
        let receipt = import_local(
            &quarantine,
            Path::new("model.safetensors"),
            Path::new("tokenizer.aiptok"),
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

        fs::set_permissions(&store, fs::Permissions::from_mode(0o700))
            .expect("temporary store must unseal for cleanup");
        fs::set_permissions(receipt.package_path(), fs::Permissions::from_mode(0o700))
            .expect("temporary package must unseal for cleanup");
        fs::remove_dir_all(&root).expect("temporary fixture must clean up");
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
        let (source, tokenizer) = source_and_tokenizer();
        fs::write(quarantine.join("model.safetensors"), &source).expect("source must write");
        fs::write(quarantine.join("tokenizer.aiptok"), &tokenizer).expect("tokenizer must write");
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
                Path::new("tokenizer.aiptok"),
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
                Path::new("tokenizer.aiptok"),
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
            Path::new("tokenizer.aiptok"),
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
                Path::new("tokenizer.aiptok"),
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
            Path::new("tokenizer.aiptok"),
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
