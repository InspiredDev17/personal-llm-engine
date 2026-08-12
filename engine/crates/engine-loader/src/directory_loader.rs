//! Production-only admission for signed directory-backed `.paimodel` packages.

use crate::{ModelRegistry, RegisteredModel, read_only_map::ReadOnlyFileMap};
use personalai_engine_crypto::{
    hash_package, hash_signed_content, verify_directory_package_signature,
};
use personalai_engine_memory::{MemoryPool, MemoryReservation};
use personalai_engine_model_format::{
    DIRECTORY_DECODER_ARCHITECTURE_ID, DirectoryManifest, DirectoryTensorRecord, HashCatalog,
    PackageFileKind, SignatureEnvelope,
};
use personalai_engine_tensor::Tensor;
use personalai_engine_types::{
    DType, EngineError, MAX_MODEL_PACKAGE_BYTES, ModelId, ModelPackageRejection, ResourceBudget,
    SecurityEvent, TensorShape,
};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File, OpenOptions};
use std::io::Read;
use std::os::unix::fs::{MetadataExt, OpenOptionsExt, PermissionsExt};
use std::path::Path;

const HASH_CATALOG_FILE: &str = "hashes.aiph";
const SIGNATURE_FILE: &str = "signature.ed25519";
const MAX_HASH_CATALOG_BYTES: u64 = 256 * 1024;
const MAX_PROVENANCE_BYTES: u64 = 1024 * 1024;
const CURRENT_ENGINE_VERSION: (u16, u16, u16) = (0, 1, 0);

/// Mandatory typed event destination for production package validation.
pub trait DirectorySecurityEventSink {
    /// Receives one sanitized model-package rejection event.
    fn emit(&mut self, event: SecurityEvent);
}

impl<F> DirectorySecurityEventSink for F
where
    F: FnMut(SecurityEvent),
{
    fn emit(&mut self, event: SecurityEvent) {
        self(event);
    }
}

/// One immutable mapped shard admitted through every package gate.
#[derive(Debug)]
struct VerifiedShard {
    map: ReadOnlyFileMap,
}

/// Signed, hash-verified, architecture-validated directory package.
#[derive(Debug)]
pub struct VerifiedDirectoryPackage {
    registration: RegisteredModel,
    manifest: DirectoryManifest,
    tokenizer: Vec<u8>,
    provenance: Vec<u8>,
    shards: Vec<VerifiedShard>,
}

impl VerifiedDirectoryPackage {
    /// Returns the trusted local registration.
    pub const fn registration(&self) -> &RegisteredModel {
        &self.registration
    }

    /// Returns the parsed canonical manifest.
    pub const fn manifest(&self) -> &DirectoryManifest {
        &self.manifest
    }

    /// Returns immutable verified tokenizer bytes.
    pub fn tokenizer_bytes(&self) -> &[u8] {
        &self.tokenizer
    }

    /// Returns immutable verified provenance bytes.
    pub fn provenance_bytes(&self) -> &[u8] {
        &self.provenance
    }

    /// Returns one immutable verified mapped tensor range.
    pub fn tensor_bytes(&self, tensor: &DirectoryTensorRecord) -> Result<&[u8], EngineError> {
        let shard = self
            .shards
            .get(usize::try_from(tensor.shard_index()).map_err(|_| {
                EngineError::resource_exhaustion("tensor shard index is not addressable")
            })?)
            .ok_or_else(|| EngineError::internal_invariant("validated tensor shard is missing"))?;
        let start = usize::try_from(tensor.byte_offset())
            .map_err(|_| EngineError::resource_exhaustion("tensor offset is not addressable"))?;
        let length = usize::try_from(tensor.byte_length())
            .map_err(|_| EngineError::resource_exhaustion("tensor length is not addressable"))?;
        let end = start
            .checked_add(length)
            .ok_or_else(|| EngineError::resource_exhaustion("tensor range overflows"))?;
        shard
            .map
            .bytes()
            .get(start..end)
            .ok_or_else(|| EngineError::internal_invariant("validated tensor range is unavailable"))
    }

    /// Copies mapped tensor bytes into aggregate-accounted CPU storage.
    pub fn materialize_tensors(
        &self,
        pool: &MemoryPool,
    ) -> Result<LoadedDirectoryTensors, EngineError> {
        let metadata_bytes = u64::try_from(self.manifest.tensors().len())
            .ok()
            .and_then(|count| count.checked_mul(192))
            .ok_or_else(|| EngineError::resource_exhaustion("tensor metadata bound overflows"))?;
        let metadata_reservation = pool.reserve(metadata_bytes)?;
        let mut tensors = BTreeMap::new();
        for record in self.manifest.tensors() {
            let tensor = Tensor::copy_from_bytes(
                record.dtype(),
                record.shape().clone(),
                self.tensor_bytes(record)?,
                pool,
            )?;
            if tensors.insert(record.name().to_owned(), tensor).is_some() {
                return Err(EngineError::internal_invariant(
                    "validated directory manifest contains duplicate tensor name",
                ));
            }
        }
        if self.manifest.tied_output() {
            let embedding = self
                .manifest
                .tensors()
                .iter()
                .find(|record| record.name() == "token_embedding.weight")
                .ok_or_else(|| EngineError::internal_invariant("tied embedding is missing"))?;
            let output = Tensor::copy_from_bytes(
                embedding.dtype(),
                embedding.shape().clone(),
                self.tensor_bytes(embedding)?,
                pool,
            )?;
            if tensors.insert("output.weight".to_owned(), output).is_some() {
                return Err(EngineError::internal_invariant(
                    "tied output conflicts with explicit output tensor",
                ));
            }
        }
        Ok(LoadedDirectoryTensors { tensors, _metadata_reservation: metadata_reservation })
    }
}

/// Aggregate-accounted CPU tensors copied from verified read-only mappings.
#[derive(Debug)]
pub struct LoadedDirectoryTensors {
    tensors: BTreeMap<String, Tensor>,
    _metadata_reservation: MemoryReservation,
}

impl LoadedDirectoryTensors {
    /// Transfers the complete tensor map to the architecture implementation.
    pub fn into_tensors(self) -> BTreeMap<String, Tensor> {
        self.tensors
    }
}

pub(crate) fn load(
    registry: &ModelRegistry,
    model_id: &ModelId,
    budget: &ResourceBudget,
    events: &mut impl DirectorySecurityEventSink,
) -> Result<VerifiedDirectoryPackage, EngineError> {
    let result = load_inner(registry, model_id, budget);
    if let Err(error) = &result {
        events.emit(SecurityEvent::ModelPackageRejected {
            model_id: model_id.clone(),
            reason: rejection_reason(error),
        });
    }
    result
}

#[allow(clippy::too_many_lines)]
fn load_inner(
    registry: &ModelRegistry,
    model_id: &ModelId,
    budget: &ResourceBudget,
) -> Result<VerifiedDirectoryPackage, EngineError> {
    let registration = registry.resolve(model_id)?.clone();
    validate_root(registry.model_root())?;
    let artifact_metadata = validate_registry_artifact_parent(registry, &registration)?;
    let package_metadata = fs::symlink_metadata(registration.package_path())
        .map_err(|_| EngineError::invalid_input("registered directory package is unavailable"))?;
    if !package_metadata.is_dir()
        || package_metadata.file_type().is_symlink()
        || package_metadata.permissions().mode() & 0o222 != 0
    {
        return Err(EngineError::authentication_failure(
            "registered package must be a protected non-symlink directory",
        ));
    }

    let catalog_bytes = read_direct_file(
        registration.package_path(),
        HASH_CATALOG_FILE,
        MAX_HASH_CATALOG_BYTES.min(budget.max_memory_bytes()),
    )?;
    if hash_package(&catalog_bytes) != *registration.expected_hash() {
        return Err(EngineError::signature_failure(
            "directory package identity does not match local configuration",
        ));
    }
    let catalog = HashCatalog::parse(&catalog_bytes)?;
    let catalog_length = usize_to_u64(catalog_bytes.len())?;
    ensure_entry_budget(0, catalog_length, budget)?;
    let signature_maximum = budget.max_memory_bytes().saturating_sub(catalog_length).min(96);
    let signature_bytes =
        read_direct_file(registration.package_path(), SIGNATURE_FILE, signature_maximum)?;
    let signature = SignatureEnvelope::parse(&signature_bytes)?;
    verify_directory_package_signature(
        registration.trusted_signer(),
        signature.key_id(),
        &hash_signed_content(&catalog_bytes),
        signature.signature(),
    )?;
    validate_exact_file_set(registration.package_path(), &catalog)?;

    let mut total_bytes = catalog_length
        .checked_add(usize_to_u64(signature_bytes.len())?)
        .ok_or_else(|| EngineError::resource_exhaustion("directory package size overflows"))?;
    let manifest_entry = required_entry(&catalog, "manifest.aipm")?;
    ensure_entry_budget(total_bytes, manifest_entry.byte_length(), budget)?;
    let manifest_bytes = read_catalog_file(registration.package_path(), manifest_entry)?;
    let manifest = DirectoryManifest::parse(&manifest_bytes)?;
    if manifest.model_id() != model_id {
        return Err(EngineError::authentication_failure(
            "directory manifest model ID differs from local registration",
        ));
    }
    if manifest.required_engine_version() > CURRENT_ENGINE_VERSION {
        return Err(EngineError::not_implemented(
            "directory package requires a newer engine version",
        ));
    }
    validate_architecture(&manifest)?;
    if let Some(binding) = registration.registry_binding() {
        if manifest.context_limit().get() != binding.context_capacity {
            return Err(EngineError::authentication_failure(
                "directory manifest context differs from the signed registry decision",
            ));
        }
    }
    total_bytes = checked_total(total_bytes, manifest_bytes.len(), budget)?;

    let tokenizer_entry = required_entry(&catalog, "tokenizer.aiptok")?;
    ensure_entry_budget(total_bytes, tokenizer_entry.byte_length(), budget)?;
    let tokenizer = read_catalog_file(registration.package_path(), tokenizer_entry)?;
    if hash_package(&tokenizer) != manifest.tokenizer_hash() {
        return Err(corrupt("directory tokenizer hash differs from manifest"));
    }
    if let Some(binding) = registration.registry_binding() {
        if manifest.tokenizer_hash() != binding.tokenizer_hash {
            return Err(EngineError::authentication_failure(
                "directory tokenizer identity differs from the signed registry decision",
            ));
        }
    }
    total_bytes = checked_total(total_bytes, tokenizer.len(), budget)?;
    let provenance_entry = required_entry(&catalog, "provenance.json")?;
    ensure_entry_budget(total_bytes, provenance_entry.byte_length(), budget)?;
    let provenance = read_catalog_file(registration.package_path(), provenance_entry)?;
    validate_provenance(&provenance)?;
    if let Some(binding) = registration.registry_binding() {
        if hash_package(&provenance) != binding.provenance_hash {
            return Err(EngineError::authentication_failure(
                "directory provenance identity differs from the signed registry decision",
            ));
        }
    }
    total_bytes = checked_total(total_bytes, provenance.len(), budget)?;

    if manifest.shards().len()
        != catalog
            .entries()
            .iter()
            .filter(|entry| entry.kind() == PackageFileKind::TensorShard)
            .count()
    {
        return Err(corrupt("manifest and hash catalog shard counts differ"));
    }
    let mut shards = Vec::new();
    shards
        .try_reserve_exact(manifest.shards().len())
        .map_err(|_| EngineError::resource_exhaustion("mapped shard allocation failed"))?;
    for shard in manifest.shards() {
        let name = shard.file_name();
        let catalog_entry = required_entry(&catalog, &name)?;
        if catalog_entry.kind() != PackageFileKind::TensorShard
            || catalog_entry.index() != shard.index()
            || catalog_entry.byte_length() != shard.byte_length()
            || catalog_entry.content_hash() != shard.content_hash()
        {
            return Err(corrupt("manifest and hash catalog shard identities differ"));
        }
        ensure_entry_budget(total_bytes, shard.byte_length(), budget)?;
        total_bytes = checked_total(
            total_bytes,
            usize::try_from(shard.byte_length())
                .map_err(|_| EngineError::resource_exhaustion("shard size is not addressable"))?,
            budget,
        )?;
        let map = map_catalog_file(registration.package_path(), catalog_entry)?;
        if hash_package(map.bytes()) != shard.content_hash() {
            return Err(EngineError::signature_failure("tensor shard hash verification failed"));
        }
        shards.push(VerifiedShard { map });
    }
    if let Some(binding) = registration.registry_binding() {
        let mut observed = manifest
            .shards()
            .iter()
            .map(personalai_engine_model_format::DirectoryShardRecord::content_hash)
            .collect::<Vec<_>>();
        observed.sort_unstable();
        if observed != binding.tensor_shard_hashes {
            return Err(EngineError::authentication_failure(
                "directory tensor-shard identities differ from the signed registry decision",
            ));
        }
    }
    let package =
        VerifiedDirectoryPackage { registration, manifest, tokenizer, provenance, shards };
    validate_tensor_hashes(&package)?;
    let after = fs::symlink_metadata(package.registration.package_path())
        .map_err(|_| EngineError::authentication_failure("package changed during verification"))?;
    if after.dev() != package_metadata.dev() || after.ino() != package_metadata.ino() {
        return Err(EngineError::authentication_failure("package directory changed during load"));
    }
    if let Some(before) = artifact_metadata {
        let artifact =
            package.registration.package_path().parent().ok_or_else(|| {
                EngineError::internal_invariant("registry artifact parent is absent")
            })?;
        let after = fs::symlink_metadata(artifact)
            .map_err(|_| EngineError::authentication_failure("artifact directory changed"))?;
        if after.dev() != before.dev() || after.ino() != before.ino() {
            return Err(EngineError::authentication_failure(
                "artifact directory changed during package load",
            ));
        }
    }
    let _ = total_bytes;
    Ok(package)
}

fn validate_registry_artifact_parent(
    registry: &ModelRegistry,
    registration: &RegisteredModel,
) -> Result<Option<fs::Metadata>, EngineError> {
    let Some(binding) = registration.registry_binding() else {
        return Ok(None);
    };
    let artifact = registry.model_root().join(&binding.artifact_id);
    let expected_package = artifact.join(format!(
        "{}.{}",
        registration.model_id().as_str(),
        crate::MODEL_PACKAGE_EXTENSION
    ));
    if registration.package_path() != expected_package {
        return Err(EngineError::internal_invariant(
            "registry-backed package path is not canonically derived",
        ));
    }
    let metadata = fs::symlink_metadata(&artifact)
        .map_err(|_| EngineError::invalid_input("registry artifact directory is unavailable"))?;
    if !metadata.is_dir()
        || metadata.file_type().is_symlink()
        || metadata.permissions().mode() & 0o222 != 0
    {
        return Err(EngineError::authentication_failure(
            "registry artifact must be a protected non-symlink directory",
        ));
    }
    Ok(Some(metadata))
}

fn validate_root(root: &Path) -> Result<(), EngineError> {
    let metadata = fs::symlink_metadata(root)
        .map_err(|_| EngineError::invalid_input("configured model root is unavailable"))?;
    if !metadata.is_dir()
        || metadata.file_type().is_symlink()
        || metadata.permissions().mode() & 0o222 != 0
    {
        return Err(EngineError::authentication_failure(
            "model root must be a protected non-symlink directory",
        ));
    }
    Ok(())
}

fn validate_exact_file_set(directory: &Path, catalog: &HashCatalog) -> Result<(), EngineError> {
    let mut expected =
        catalog.entries().iter().map(|entry| entry.name().to_owned()).collect::<BTreeSet<_>>();
    expected.insert(HASH_CATALOG_FILE.to_owned());
    expected.insert(SIGNATURE_FILE.to_owned());
    let mut observed = BTreeSet::new();
    for entry in fs::read_dir(directory)
        .map_err(|_| EngineError::authentication_failure("package directory read failed"))?
    {
        let entry = entry
            .map_err(|_| EngineError::authentication_failure("package directory entry failed"))?;
        let name = entry
            .file_name()
            .into_string()
            .map_err(|_| corrupt("package filename is invalid UTF-8"))?;
        let metadata = fs::symlink_metadata(entry.path())
            .map_err(|_| EngineError::authentication_failure("package entry metadata failed"))?;
        if !metadata.is_file() || metadata.file_type().is_symlink() {
            return Err(EngineError::authentication_failure(
                "package entries must be non-symlink regular files",
            ));
        }
        observed.insert(name);
    }
    if observed != expected {
        return Err(corrupt("package contains a missing or unexpected trusted file"));
    }
    Ok(())
}

fn required_entry<'a>(
    catalog: &'a HashCatalog,
    name: &str,
) -> Result<&'a personalai_engine_model_format::HashCatalogEntry, EngineError> {
    catalog.entry(name).ok_or_else(|| corrupt("hash catalog omits required file"))
}

fn read_catalog_file(
    directory: &Path,
    entry: &personalai_engine_model_format::HashCatalogEntry,
) -> Result<Vec<u8>, EngineError> {
    let bytes = read_direct_file(directory, entry.name(), entry.byte_length())?;
    if usize_to_u64(bytes.len())? != entry.byte_length()
        || hash_package(&bytes) != entry.content_hash()
    {
        return Err(EngineError::signature_failure("package file hash verification failed"));
    }
    Ok(bytes)
}

fn read_direct_file(directory: &Path, name: &str, maximum: u64) -> Result<Vec<u8>, EngineError> {
    let (mut file, length) = open_direct_file(directory, name, maximum)?;
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(length)
        .map_err(|_| EngineError::resource_exhaustion("package file allocation failed"))?;
    bytes.resize(length, 0);
    file.read_exact(&mut bytes).map_err(|_| corrupt("package file is truncated"))?;
    let mut extra = [0_u8; 1];
    if file.read(&mut extra).map_err(|_| corrupt("package file trailing check failed"))? != 0 {
        return Err(corrupt("package file grew during read"));
    }
    Ok(bytes)
}

fn map_catalog_file(
    directory: &Path,
    entry: &personalai_engine_model_format::HashCatalogEntry,
) -> Result<ReadOnlyFileMap, EngineError> {
    let (file, length) = open_direct_file(directory, entry.name(), entry.byte_length())?;
    if usize_to_u64(length)? != entry.byte_length() {
        return Err(corrupt("mapped shard length differs from hash catalog"));
    }
    ReadOnlyFileMap::new(file, length)
}

fn open_direct_file(
    directory: &Path,
    name: &str,
    maximum: u64,
) -> Result<(File, usize), EngineError> {
    if name.is_empty() || name.contains('/') || name.contains('\\') || name.contains("..") {
        return Err(corrupt("package direct-child filename is invalid"));
    }
    let path = directory.join(name);
    let before = fs::symlink_metadata(&path)
        .map_err(|_| EngineError::invalid_input("registered package file is unavailable"))?;
    if !before.is_file()
        || before.file_type().is_symlink()
        || before.permissions().mode() & 0o222 != 0
    {
        return Err(EngineError::authentication_failure(
            "package file metadata or permissions are invalid",
        ));
    }
    if before.len() == 0 || before.len() > maximum {
        return Err(EngineError::resource_exhaustion(
            "package file exceeds its configured byte limit",
        ));
    }
    let file = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_CLOEXEC | libc::O_NOFOLLOW)
        .open(path)
        .map_err(|_| EngineError::authentication_failure("package file secure open failed"))?;
    let opened = file
        .metadata()
        .map_err(|_| EngineError::authentication_failure("opened package metadata failed"))?;
    if opened.dev() != before.dev()
        || opened.ino() != before.ino()
        || opened.len() != before.len()
        || !opened.is_file()
    {
        return Err(EngineError::authentication_failure("package file changed during secure open"));
    }
    let length = usize::try_from(opened.len())
        .map_err(|_| EngineError::resource_exhaustion("package file is not addressable"))?;
    Ok((file, length))
}

fn validate_architecture(manifest: &DirectoryManifest) -> Result<(), EngineError> {
    if manifest.architecture_id().as_str() != DIRECTORY_DECODER_ARCHITECTURE_ID
        || manifest.storage_dtype() != DType::F32
    {
        return Err(EngineError::unsupported_architecture(manifest.architecture_id().clone()));
    }
    let expected = expected_shapes(manifest)?;
    let observed = manifest
        .tensors()
        .iter()
        .map(|tensor| (tensor.name(), tensor.shape()))
        .collect::<BTreeMap<_, _>>();
    if observed.len() != expected.len()
        || expected.iter().any(|(name, shape)| observed.get(name.as_str()) != Some(&shape))
    {
        return Err(corrupt("directory tensor inventory or shape is invalid"));
    }
    Ok(())
}

fn expected_shapes(
    manifest: &DirectoryManifest,
) -> Result<BTreeMap<String, TensorShape>, EngineError> {
    let hidden = u64::from(manifest.hidden_dimension());
    let vocabulary = u64::from(manifest.vocabulary_size());
    let intermediate = u64::from(manifest.feed_forward_dimension());
    let head_dimension = hidden
        .checked_div(u64::from(manifest.attention_head_count()))
        .filter(|value| *value != 0)
        .ok_or_else(|| corrupt("attention dimensions are inconsistent"))?;
    let kv = u64::from(manifest.kv_head_count())
        .checked_mul(head_dimension)
        .ok_or_else(|| corrupt("KV dimension overflows"))?;
    let mut expected = BTreeMap::from([
        ("token_embedding.weight".to_owned(), TensorShape::new([vocabulary, hidden])?),
        ("final_norm.weight".to_owned(), TensorShape::new([hidden])?),
    ]);
    if !manifest.tied_output() {
        expected.insert("output.weight".to_owned(), TensorShape::new([vocabulary, hidden])?);
    }
    for layer in 0..manifest.layer_count() {
        let prefix = format!("layers.{layer:03}");
        for suffix in ["attn_norm.weight", "ffn_norm.weight"] {
            expected.insert(format!("{prefix}.{suffix}"), TensorShape::new([hidden])?);
        }
        for suffix in ["q_proj.weight", "o_proj.weight"] {
            expected.insert(format!("{prefix}.{suffix}"), TensorShape::new([hidden, hidden])?);
        }
        for suffix in ["k_proj.weight", "v_proj.weight"] {
            expected.insert(format!("{prefix}.{suffix}"), TensorShape::new([kv, hidden])?);
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

fn validate_tensor_hashes(package: &VerifiedDirectoryPackage) -> Result<(), EngineError> {
    for tensor in package.manifest.tensors() {
        if hash_package(package.tensor_bytes(tensor)?) != tensor.content_hash() {
            return Err(EngineError::signature_failure("tensor content hash verification failed"));
        }
    }
    Ok(())
}

fn validate_provenance(bytes: &[u8]) -> Result<(), EngineError> {
    if bytes.is_empty()
        || bytes.len() > usize::try_from(MAX_PROVENANCE_BYTES).unwrap_or(usize::MAX)
        || bytes.first() != Some(&b'{')
        || !bytes.ends_with(b"}\n")
    {
        return Err(corrupt("provenance document is not canonical"));
    }
    let text = std::str::from_utf8(bytes).map_err(|_| corrupt("provenance is invalid UTF-8"))?;
    if text
        .chars()
        .any(|character| character.is_control() && !matches!(character, '\n' | '\r' | '\t'))
    {
        return Err(corrupt("provenance contains prohibited control data"));
    }
    Ok(())
}

fn checked_total(
    total: u64,
    additional: usize,
    budget: &ResourceBudget,
) -> Result<u64, EngineError> {
    let total = total
        .checked_add(usize_to_u64(additional)?)
        .ok_or_else(|| EngineError::resource_exhaustion("directory package size overflows"))?;
    if total > MAX_MODEL_PACKAGE_BYTES || total > budget.max_memory_bytes() {
        return Err(EngineError::resource_exhaustion(
            "directory package exceeds its configured byte limit",
        ));
    }
    Ok(total)
}

fn ensure_entry_budget(
    total: u64,
    additional: u64,
    budget: &ResourceBudget,
) -> Result<(), EngineError> {
    let projected = total
        .checked_add(additional)
        .ok_or_else(|| EngineError::resource_exhaustion("directory package size overflows"))?;
    if projected > MAX_MODEL_PACKAGE_BYTES || projected > budget.max_memory_bytes() {
        return Err(EngineError::resource_exhaustion(
            "directory package exceeds its configured byte limit",
        ));
    }
    Ok(())
}

fn rejection_reason(error: &EngineError) -> ModelPackageRejection {
    match error {
        EngineError::SignatureFailure(_) | EngineError::AuthenticationFailure(_) => {
            ModelPackageRejection::SignatureFailure
        }
        EngineError::UnsupportedArchitecture(_) => ModelPackageRejection::UnsupportedArchitecture,
        EngineError::UnsupportedTokenizer(_) => ModelPackageRejection::UnsupportedTokenizer,
        EngineError::ResourceExhaustion(_) => ModelPackageRejection::SizeLimit,
        _ => ModelPackageRejection::Corrupt,
    }
}

fn usize_to_u64(value: usize) -> Result<u64, EngineError> {
    u64::try_from(value)
        .map_err(|_| EngineError::resource_exhaustion("package size is not representable"))
}

fn corrupt(detail: impl AsRef<str>) -> EngineError {
    EngineError::corrupt_model_package(detail)
}
