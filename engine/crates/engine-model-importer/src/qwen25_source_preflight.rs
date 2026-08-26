//! Descriptor-bound, operator-only preflight for pinned Qwen2.5 source files.
//!
//! The preflight reads only one exact direct-child ingress artifact containing
//! `source/config.json`, `source/tokenizer.json`, and
//! `source/model.safetensors`. Large tensor payloads are streamed through one
//! already-open descriptor and are never allocated. This module does not emit,
//! sign, qualify, approve, or activate a model package.

use crate::safetensors::ParsedSafetensorsHeader;
use crate::{
    ImportSpec, QWEN25_MODEL_SAFETENSORS_BYTES, QWEN25_TOKENIZER_BYTES, Qwen25ConfigProfile,
    SafeTensorSourceDType, SourceMapping, canonical_name, corrupt, expected_shapes,
};
use personalai_engine_crypto::hash_sha256_bounded_reader;
use personalai_engine_quantization::QuantizationProfile;
#[cfg(test)]
use personalai_engine_types::DType;
use personalai_engine_types::{EngineError, ModelId, Sha256Digest};
use std::collections::BTreeSet;
use std::fs::{self, File, Metadata, OpenOptions};
use std::io::{Read, Seek, SeekFrom};
use std::os::unix::fs::{MetadataExt, OpenOptionsExt};
use std::path::{Component, Path, PathBuf};

const CONFIG_NAME: &str = "config.json";
const TOKENIZER_NAME: &str = "tokenizer.json";
const MODEL_NAME: &str = "model.safetensors";
const SOURCE_DIRECTORY_NAME: &str = "source";
const MAX_ARTIFACT_NAME_BYTES: usize = 160;
const QWEN25_TENSOR_COUNT: usize = 338;

/// Content-free evidence from one successful pinned Qwen2.5 source preflight.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Qwen25SourcePreflightReport {
    profile: Qwen25ConfigProfile,
    config_sha256: Sha256Digest,
    tokenizer_sha256: Sha256Digest,
    model_sha256: Sha256Digest,
    model_bytes: u64,
    safetensors_header_bytes: u64,
    tensor_count: u32,
}

impl Qwen25SourcePreflightReport {
    /// Returns the exact pinned source profile.
    pub const fn profile(self) -> Qwen25ConfigProfile {
        self.profile
    }

    /// Returns the exact configuration source identity.
    pub const fn config_sha256(self) -> Sha256Digest {
        self.config_sha256
    }

    /// Returns the exact tokenizer source identity.
    pub const fn tokenizer_sha256(self) -> Sha256Digest {
        self.tokenizer_sha256
    }

    /// Returns the exact streamed tensor-file identity.
    pub const fn model_sha256(self) -> Sha256Digest {
        self.model_sha256
    }

    /// Returns the descriptor-bound tensor-file byte length.
    pub const fn model_bytes(self) -> u64 {
        self.model_bytes
    }

    /// Returns the bounded safetensors JSON-header byte length.
    pub const fn safetensors_header_bytes(self) -> u64 {
        self.safetensors_header_bytes
    }

    /// Returns the validated tensor count.
    pub const fn tensor_count(self) -> u32 {
        self.tensor_count
    }
}

/// Preflights one exact, local Qwen2.5 ingress artifact without package output.
///
/// `artifact_name` must be one safe direct-child name beneath the trusted,
/// owner-private ingress root. Requests and the production service must never
/// expose this operator-only filesystem API.
pub fn preflight_qwen25_ingress(
    ingress_root: &Path,
    artifact_name: &str,
) -> Result<Qwen25SourcePreflightReport, EngineError> {
    validate_absolute_normal_root(ingress_root)?;
    validate_artifact_name(artifact_name)?;

    let root_guard = DirectoryGuard::open(ingress_root, None, "ingress root")?;
    let artifact_path = ingress_root.join(artifact_name);
    let artifact_guard =
        DirectoryGuard::open(&artifact_path, Some(root_guard.identity.uid), "artifact directory")?;
    let source_path = artifact_path.join(SOURCE_DIRECTORY_NAME);
    let source_guard =
        DirectoryGuard::open(&source_path, Some(root_guard.identity.uid), "source directory")?;
    validate_exact_source_entries(&source_path)?;

    let config_path = source_path.join(CONFIG_NAME);
    let mut config_file = PinnedSourceFile::open(
        &config_path,
        root_guard.identity.uid,
        &[656, 660],
        "configuration",
    )?;
    let config_bytes = config_file.read_complete_bounded(660)?;

    let tokenizer_path = source_path.join(TOKENIZER_NAME);
    let mut tokenizer_file = PinnedSourceFile::open(
        &tokenizer_path,
        root_guard.identity.uid,
        &[u64::try_from(QWEN25_TOKENIZER_BYTES).map_err(|_| {
            EngineError::internal_invariant("pinned tokenizer length is not representable")
        })?],
        "tokenizer",
    )?;
    let tokenizer_bytes =
        tokenizer_file.read_complete_bounded(u64::try_from(QWEN25_TOKENIZER_BYTES).map_err(
            |_| EngineError::internal_invariant("pinned tokenizer length is not representable"),
        )?)?;

    let spec = ImportSpec::new_qwen25_1_5b(
        ModelId::new("qwen25-operator-preflight")?,
        &config_bytes,
        &tokenizer_bytes,
        "operator-local-preflight",
        "unqualified-source",
        QuantizationProfile::ReferenceF32,
    )?;
    let config = spec
        .qwen25_config()
        .ok_or_else(|| EngineError::internal_invariant("Qwen preflight lost configuration"))?;
    let (_, expected_model_bytes, expected_model_sha256) = spec
        .qwen25_model_source_descriptor()
        .ok_or_else(|| EngineError::internal_invariant("Qwen preflight lost source identity"))?;

    let model_path = source_path.join(MODEL_NAME);
    let mut model_file = PinnedSourceFile::open(
        &model_path,
        root_guard.identity.uid,
        &[expected_model_bytes],
        "model tensor",
    )?;
    let (observed_model_sha256, metadata) = authenticate_and_parse_model_source(
        &mut model_file,
        expected_model_bytes,
        expected_model_sha256,
    )?;
    validate_qwen25_tensor_inventory(&spec, &metadata)?;
    model_file.verify_unchanged()?;
    source_guard.verify_current()?;
    artifact_guard.verify_current()?;
    root_guard.verify_current()?;

    Ok(Qwen25SourcePreflightReport {
        profile: config.profile(),
        config_sha256: config.source_sha256(),
        tokenizer_sha256: personalai_engine_crypto::hash_sha256(&tokenizer_bytes),
        model_sha256: observed_model_sha256,
        model_bytes: expected_model_bytes,
        safetensors_header_bytes: u64::try_from(metadata.header_length()).map_err(|_| {
            EngineError::internal_invariant("safetensors header length is not representable")
        })?,
        tensor_count: u32::try_from(metadata.tensors().len()).map_err(|_| {
            EngineError::internal_invariant("safetensors tensor count is not representable")
        })?,
    })
}

fn authenticate_and_parse_model_source(
    source: &mut PinnedSourceFile,
    expected_bytes: u64,
    expected_sha256: Sha256Digest,
) -> Result<(Sha256Digest, ParsedSafetensorsHeader), EngineError> {
    let observed_sha256 = verify_streamed_sha256(source, expected_bytes, expected_sha256)?;
    let metadata = read_safetensors_header(source, expected_bytes)?;
    source.verify_unchanged()?;
    Ok((observed_sha256, metadata))
}

fn verify_streamed_sha256(
    source: &mut PinnedSourceFile,
    expected_bytes: u64,
    expected_sha256: Sha256Digest,
) -> Result<Sha256Digest, EngineError> {
    source
        .file
        .seek(SeekFrom::Start(0))
        .map_err(|_| EngineError::authentication_failure("model tensor descriptor seek failed"))?;
    let (observed_sha256, observed_bytes) =
        hash_sha256_bounded_reader(&mut source.file, expected_bytes)?;
    if observed_bytes != expected_bytes {
        return Err(corrupt("Qwen2.5 model tensor source is truncated"));
    }
    if observed_sha256 != expected_sha256 {
        return Err(corrupt("Qwen2.5 model tensor source SHA-256 differs"));
    }
    source.verify_unchanged()?;
    Ok(observed_sha256)
}

fn read_safetensors_header(
    source: &mut PinnedSourceFile,
    file_length: u64,
) -> Result<ParsedSafetensorsHeader, EngineError> {
    source
        .file
        .seek(SeekFrom::Start(0))
        .map_err(|_| EngineError::authentication_failure("model tensor descriptor seek failed"))?;
    let mut length = [0_u8; 8];
    source
        .file
        .read_exact(&mut length)
        .map_err(|_| corrupt("safetensors header length is truncated"))?;
    let header_length = usize::try_from(u64::from_le_bytes(length))
        .map_err(|_| EngineError::resource_exhaustion("safetensors header is not addressable"))?;
    if !(2..=crate::MAX_SAFETENSORS_HEADER_BYTES).contains(&header_length) {
        return Err(corrupt("safetensors header length is outside hard bounds"));
    }
    let mut header = Vec::new();
    header
        .try_reserve_exact(header_length)
        .map_err(|_| EngineError::resource_exhaustion("safetensors header allocation failed"))?;
    header.resize(header_length, 0);
    source
        .file
        .read_exact(&mut header)
        .map_err(|_| corrupt("safetensors JSON header is truncated"))?;
    ParsedSafetensorsHeader::parse_with_tensor_limit(&header, file_length, QWEN25_TENSOR_COUNT)
}

fn validate_qwen25_tensor_inventory(
    spec: &ImportSpec,
    metadata: &ParsedSafetensorsHeader,
) -> Result<(), EngineError> {
    if metadata.file_length() != QWEN25_MODEL_SAFETENSORS_BYTES {
        return Err(corrupt("Qwen2.5 model tensor file length differs"));
    }
    let expected = expected_shapes(spec)?;
    if metadata.tensors().len() != expected.len() || expected.len() != QWEN25_TENSOR_COUNT {
        return Err(corrupt("Qwen2.5 tensor count differs from the reviewed inventory"));
    }
    let mut seen = BTreeSet::new();
    for tensor in metadata.tensors() {
        if tensor.dtype() != SafeTensorSourceDType::BFloat16 {
            return Err(corrupt("Qwen2.5 source tensor dtype must be BF16"));
        }
        let canonical = canonical_name(SourceMapping::Qwen25SafetensorsV1, tensor.name(), 28)?;
        if !seen.insert(canonical.clone()) {
            return Err(corrupt("Qwen2.5 source produces duplicate canonical tensors"));
        }
        let expected_shape = expected
            .get(&canonical)
            .ok_or_else(|| corrupt("Qwen2.5 source contains an unexpected tensor"))?;
        if tensor.shape() != expected_shape {
            return Err(corrupt("Qwen2.5 source tensor shape differs"));
        }
    }
    if seen != expected.keys().cloned().collect() {
        return Err(corrupt("Qwen2.5 source tensor inventory is incomplete"));
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct FileIdentity {
    device: u64,
    inode: u64,
    length: u64,
    modified_seconds: i64,
    modified_nanoseconds: i64,
    uid: u32,
    mode: u32,
}

impl FileIdentity {
    fn from_metadata(metadata: &Metadata) -> Self {
        Self {
            device: metadata.dev(),
            inode: metadata.ino(),
            length: metadata.len(),
            modified_seconds: metadata.mtime(),
            modified_nanoseconds: metadata.mtime_nsec(),
            uid: metadata.uid(),
            mode: metadata.mode(),
        }
    }
}

struct DirectoryGuard {
    path: PathBuf,
    file: File,
    identity: FileIdentity,
}

impl DirectoryGuard {
    fn open(path: &Path, owner_uid: Option<u32>, label: &'static str) -> Result<Self, EngineError> {
        let before = fs::symlink_metadata(path)
            .map_err(|_| EngineError::invalid_input(format!("{label} is unavailable")))?;
        if before.file_type().is_symlink()
            || !before.is_dir()
            || before.mode() & 0o077 != 0
            || owner_uid.is_some_and(|uid| before.uid() != uid)
        {
            return Err(EngineError::authentication_failure(format!(
                "{label} must be one owner-private non-symlink directory"
            )));
        }
        let expected = FileIdentity::from_metadata(&before);
        let mut options = OpenOptions::new();
        options.read(true).custom_flags(libc::O_CLOEXEC | libc::O_NOFOLLOW | libc::O_DIRECTORY);
        let file = options.open(path).map_err(|_| {
            EngineError::authentication_failure(format!("{label} secure open failed"))
        })?;
        let opened = file
            .metadata()
            .map_err(|_| EngineError::authentication_failure(format!("{label} metadata failed")))?;
        if !opened.is_dir() || FileIdentity::from_metadata(&opened) != expected {
            return Err(EngineError::authentication_failure(format!(
                "{label} changed during secure open"
            )));
        }
        Ok(Self { path: path.to_path_buf(), file, identity: expected })
    }

    fn verify_current(&self) -> Result<(), EngineError> {
        let opened = self.file.metadata().map_err(|_| {
            EngineError::authentication_failure("source directory descriptor metadata failed")
        })?;
        let current = fs::symlink_metadata(&self.path).map_err(|_| {
            EngineError::authentication_failure("source directory path became unavailable")
        })?;
        if !opened.is_dir()
            || current.file_type().is_symlink()
            || !current.is_dir()
            || FileIdentity::from_metadata(&opened) != self.identity
            || FileIdentity::from_metadata(&current) != self.identity
        {
            return Err(EngineError::authentication_failure(
                "source directory identity changed during preflight",
            ));
        }
        Ok(())
    }
}

struct PinnedSourceFile {
    path: PathBuf,
    file: File,
    identity: FileIdentity,
    label: &'static str,
}

impl PinnedSourceFile {
    fn open(
        path: &Path,
        owner_uid: u32,
        allowed_lengths: &[u64],
        label: &'static str,
    ) -> Result<Self, EngineError> {
        let before = fs::symlink_metadata(path)
            .map_err(|_| EngineError::invalid_input(format!("{label} source is unavailable")))?;
        if before.file_type().is_symlink()
            || !before.is_file()
            || before.uid() != owner_uid
            || before.mode() & 0o7777 != 0o400
        {
            return Err(EngineError::authentication_failure(format!(
                "{label} source must be one owner-read-only regular non-symlink file"
            )));
        }
        if !allowed_lengths.contains(&before.len()) {
            return Err(corrupt(format!("{label} source byte length is not pinned")));
        }
        let expected = FileIdentity::from_metadata(&before);
        let mut options = OpenOptions::new();
        options.read(true).custom_flags(libc::O_CLOEXEC | libc::O_NOFOLLOW);
        let file = options.open(path).map_err(|_| {
            EngineError::authentication_failure(format!("{label} secure open failed"))
        })?;
        let opened = file.metadata().map_err(|_| {
            EngineError::authentication_failure(format!("{label} descriptor metadata failed"))
        })?;
        if !opened.is_file() || FileIdentity::from_metadata(&opened) != expected {
            return Err(EngineError::authentication_failure(format!(
                "{label} source changed during secure open"
            )));
        }
        Ok(Self { path: path.to_path_buf(), file, identity: expected, label })
    }

    fn read_complete_bounded(&mut self, maximum: u64) -> Result<Vec<u8>, EngineError> {
        if self.identity.length > maximum {
            return Err(EngineError::resource_exhaustion(format!(
                "{} source exceeds its read ceiling",
                self.label
            )));
        }
        let length = usize::try_from(self.identity.length).map_err(|_| {
            EngineError::resource_exhaustion(format!(
                "{} source length is not addressable",
                self.label
            ))
        })?;
        self.file.seek(SeekFrom::Start(0)).map_err(|_| {
            EngineError::authentication_failure(format!("{} descriptor seek failed", self.label))
        })?;
        let mut bytes = Vec::new();
        bytes.try_reserve_exact(length).map_err(|_| {
            EngineError::resource_exhaustion(format!("{} source allocation failed", self.label))
        })?;
        bytes.resize(length, 0);
        self.file
            .read_exact(&mut bytes)
            .map_err(|_| corrupt(format!("{} source is truncated", self.label)))?;
        let mut extra = [0_u8; 1];
        if self.file.read(&mut extra).map_err(|_| {
            EngineError::authentication_failure(format!(
                "{} source trailing check failed",
                self.label
            ))
        })? != 0
        {
            return Err(corrupt(format!("{} source grew during read", self.label)));
        }
        self.verify_unchanged()?;
        Ok(bytes)
    }

    fn verify_unchanged(&self) -> Result<(), EngineError> {
        let opened = self.file.metadata().map_err(|_| {
            EngineError::authentication_failure(format!(
                "{} descriptor metadata failed",
                self.label
            ))
        })?;
        let current = fs::symlink_metadata(&self.path).map_err(|_| {
            EngineError::authentication_failure(format!(
                "{} source path became unavailable",
                self.label
            ))
        })?;
        if !opened.is_file()
            || current.file_type().is_symlink()
            || !current.is_file()
            || FileIdentity::from_metadata(&opened) != self.identity
            || FileIdentity::from_metadata(&current) != self.identity
        {
            return Err(EngineError::authentication_failure(format!(
                "{} source identity changed during preflight",
                self.label
            )));
        }
        Ok(())
    }
}

fn validate_absolute_normal_root(root: &Path) -> Result<(), EngineError> {
    if !root.is_absolute()
        || root.components().any(|component| {
            matches!(component, Component::Prefix(_) | Component::CurDir | Component::ParentDir)
        })
    {
        return Err(EngineError::invalid_input(
            "Qwen ingress root must be absolute and normalized",
        ));
    }
    Ok(())
}

fn validate_artifact_name(name: &str) -> Result<(), EngineError> {
    if name.is_empty()
        || name.len() > MAX_ARTIFACT_NAME_BYTES
        || !name.is_ascii()
        || name.starts_with('.')
        || name.contains("..")
        || name.chars().any(char::is_control)
        || !name
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.'))
    {
        return Err(EngineError::invalid_input(
            "Qwen ingress artifact name must be one safe direct child",
        ));
    }
    Ok(())
}

fn validate_exact_source_entries(source: &Path) -> Result<(), EngineError> {
    let mut observed = BTreeSet::new();
    for entry in fs::read_dir(source)
        .map_err(|_| EngineError::authentication_failure("source directory enumeration failed"))?
    {
        let entry = entry.map_err(|_| {
            EngineError::authentication_failure("source directory entry could not be read")
        })?;
        let name = entry.file_name().into_string().map_err(|_| {
            EngineError::authentication_failure("source directory entry name is invalid UTF-8")
        })?;
        if !observed.insert(name) {
            return Err(EngineError::authentication_failure(
                "source directory contains duplicate entries",
            ));
        }
    }
    let expected = [CONFIG_NAME, MODEL_NAME, TOKENIZER_NAME]
        .into_iter()
        .map(str::to_owned)
        .collect::<BTreeSet<_>>();
    if observed != expected {
        return Err(EngineError::authentication_failure(
            "source directory does not contain the exact approved file set",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use personalai_engine_crypto::hash_sha256;
    use std::io::Write;
    use std::os::unix::fs::{PermissionsExt, symlink};
    use std::time::{SystemTime, UNIX_EPOCH};

    fn private_temp(label: &str) -> PathBuf {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock must follow epoch")
            .as_nanos();
        let path = std::env::temp_dir()
            .join(format!("personalai-qwen-preflight-{label}-{}-{unique}", std::process::id()));
        fs::create_dir(&path).expect("temporary directory must create");
        fs::set_permissions(&path, fs::Permissions::from_mode(0o700))
            .expect("temporary permissions must set");
        path
    }

    fn private_file(path: &Path, bytes: &[u8]) {
        let mut options = OpenOptions::new();
        options.write(true).create_new(true).mode(0o600);
        let mut file = options.open(path).expect("test file must create");
        file.write_all(bytes).expect("test file must write");
        fs::set_permissions(path, fs::Permissions::from_mode(0o400)).expect("test file must seal");
    }

    fn instruct_spec() -> ImportSpec {
        let config = include_bytes!("../tests/fixtures/qwen25-1.5b-instruct-config.json");
        let config = crate::Qwen25Config::parse(config).expect("config must validate");
        ImportSpec::new_qwen25_1_5b_validated(
            ModelId::new("qwen-preflight-test").expect("model ID must validate"),
            config,
            crate::qwen25_tokenizer_json::pinned_test_descriptor(),
            "preflight-test",
            "test-only",
            QuantizationProfile::ReferenceF32,
        )
        .expect("spec must validate")
    }

    fn qwen_source_name(canonical: &str) -> String {
        match canonical {
            "token_embedding.weight" => "model.embed_tokens.weight".to_owned(),
            "final_norm.weight" => "model.norm.weight".to_owned(),
            name => {
                let rest = name.strip_prefix("layers.").expect("layer prefix");
                let (layer, suffix) = rest.split_once('.').expect("layer suffix");
                let layer = layer.parse::<u32>().expect("layer number");
                format!("model.layers.{layer}.{suffix}")
                    .replace("attn_norm.weight", "input_layernorm.weight")
                    .replace("ffn_norm.weight", "post_attention_layernorm.weight")
                    .replace("q_proj.weight", "self_attn.q_proj.weight")
                    .replace("k_proj.weight", "self_attn.k_proj.weight")
                    .replace("v_proj.weight", "self_attn.v_proj.weight")
                    .replace("o_proj.weight", "self_attn.o_proj.weight")
                    .replace("q_proj.bias", "self_attn.q_proj.bias")
                    .replace("k_proj.bias", "self_attn.k_proj.bias")
                    .replace("v_proj.bias", "self_attn.v_proj.bias")
                    .replace("gate_proj.weight", "mlp.gate_proj.weight")
                    .replace("up_proj.weight", "mlp.up_proj.weight")
                    .replace("down_proj.weight", "mlp.down_proj.weight")
            }
        }
    }

    fn exact_qwen_header() -> Vec<u8> {
        let expected = expected_shapes(&instruct_spec()).expect("shapes must build");
        let mut records = Vec::new();
        let mut offset = 0_u64;
        for (canonical, shape) in expected {
            let length = shape.byte_len(DType::BFloat16).expect("tensor byte length must fit");
            let end = offset.checked_add(length).expect("tensor end must fit");
            let dimensions =
                shape.dimensions().iter().map(u64::to_string).collect::<Vec<_>>().join(",");
            records.push(format!(
                "\"{}\":{{\"dtype\":\"BF16\",\"shape\":[{dimensions}],\"data_offsets\":[{offset},{end}]}}",
                qwen_source_name(&canonical)
            ));
            offset = end;
        }
        let mut header = format!("{{{}}}", records.join(",")).into_bytes();
        let target = QWEN25_MODEL_SAFETENSORS_BYTES
            .checked_sub(8)
            .and_then(|length| length.checked_sub(offset))
            .and_then(|length| usize::try_from(length).ok())
            .expect("pinned header length must fit");
        assert!(header.len() <= target);
        header.resize(target, b' ');
        header
    }

    #[test]
    fn exact_qwen_header_validates_all_338_bf16_tensors_without_payload() {
        let header = exact_qwen_header();
        let parsed = ParsedSafetensorsHeader::parse(&header, QWEN25_MODEL_SAFETENSORS_BYTES)
            .expect("header must parse");
        validate_qwen25_tensor_inventory(&instruct_spec(), &parsed)
            .expect("inventory must validate");
        assert_eq!(parsed.tensors().len(), 338);
        assert_eq!(parsed.data_start(), 8 + header.len());
    }

    #[test]
    fn qwen_inventory_rejects_wrong_dtype_shape_and_trailing_extent() {
        const ORIGINAL_SHAPE: &str = "[151936,1536]";
        const WRONG_SHAPE: &str = "[1536,151936]";

        let header = exact_qwen_header();
        let mut wrong_dtype = String::from_utf8(header.clone()).expect("header is UTF-8");
        let dtype_start = wrong_dtype.find("\"BF16\"").expect("dtype must exist");
        wrong_dtype.replace_range(dtype_start..dtype_start + 6, "\"F16\"");
        wrong_dtype.push(' ');
        let parsed_dtype =
            ParsedSafetensorsHeader::parse(wrong_dtype.as_bytes(), QWEN25_MODEL_SAFETENSORS_BYTES)
                .expect("F16 is structurally supported");
        assert!(validate_qwen25_tensor_inventory(&instruct_spec(), &parsed_dtype).is_err());

        let mut wrong_shape = String::from_utf8(header.clone()).expect("header is UTF-8");
        assert_eq!(ORIGINAL_SHAPE.len(), WRONG_SHAPE.len());
        let shape_start = wrong_shape.find(ORIGINAL_SHAPE).expect("embedding shape must exist");
        wrong_shape.replace_range(shape_start..shape_start + ORIGINAL_SHAPE.len(), WRONG_SHAPE);
        let parsed_shape =
            ParsedSafetensorsHeader::parse(wrong_shape.as_bytes(), QWEN25_MODEL_SAFETENSORS_BYTES)
                .expect("same-element-count shape is structurally valid");
        assert!(validate_qwen25_tensor_inventory(&instruct_spec(), &parsed_shape).is_err());

        assert!(
            ParsedSafetensorsHeader::parse(&header, QWEN25_MODEL_SAFETENSORS_BYTES + 1).is_err()
        );
    }

    #[test]
    fn secure_open_rejects_symlink_nonregular_wrong_size_and_hash() {
        let root = private_temp("open-rejections");
        let target = root.join("target");
        private_file(&target, b"exact");
        let link = root.join("link");
        symlink(&target, &link).expect("symlink must create");
        let uid = fs::metadata(&root).expect("root metadata").uid();
        assert!(PinnedSourceFile::open(&link, uid, &[5], "test").is_err());
        assert!(PinnedSourceFile::open(&root, uid, &[5], "test").is_err());
        assert!(PinnedSourceFile::open(&target, uid, &[4], "test").is_err());

        let mut pinned =
            PinnedSourceFile::open(&target, uid, &[5], "test").expect("regular file must open");
        let expected = hash_sha256(b"exact");
        assert_eq!(
            verify_streamed_sha256(&mut pinned, 5, expected)
                .expect("exact descriptor hash must verify"),
            expected
        );
        assert!(verify_streamed_sha256(&mut pinned, 5, hash_sha256(b"other")).is_err());
        fs::remove_dir_all(root).expect("temporary tree must remove");
    }

    #[test]
    fn model_hash_authentication_precedes_header_parsing() {
        let root = private_temp("hash-before-header");
        let path = root.join("model.safetensors");
        let mut malformed = Vec::from(64_u64.to_le_bytes());
        malformed.extend_from_slice(b"{}");
        private_file(&path, &malformed);
        let uid = fs::metadata(&root).expect("root metadata").uid();
        let length = u64::try_from(malformed.len()).expect("fixture length fits");
        let mut pinned =
            PinnedSourceFile::open(&path, uid, &[length], "test").expect("source securely opens");

        let wrong_hash_error =
            authenticate_and_parse_model_source(&mut pinned, length, hash_sha256(b"different"))
                .expect_err("wrong hash must fail before malformed header parsing");
        assert!(wrong_hash_error.to_string().contains("SHA-256 differs"));

        let header_error =
            authenticate_and_parse_model_source(&mut pinned, length, hash_sha256(&malformed))
                .expect_err("authenticated malformed header must then fail parsing");
        assert!(header_error.to_string().contains("header"));
        fs::remove_dir_all(root).expect("temporary tree must remove");
    }

    #[test]
    fn header_reader_rejects_truncation_and_parser_rejects_overlap() {
        let root = private_temp("header-truncation");
        let path = root.join("model.safetensors");
        let mut truncated = Vec::from(64_u64.to_le_bytes());
        truncated.extend_from_slice(b"{}");
        private_file(&path, &truncated);
        let uid = fs::metadata(&root).expect("root metadata").uid();
        let mut pinned = PinnedSourceFile::open(
            &path,
            uid,
            &[u64::try_from(truncated.len()).expect("length must fit")],
            "test",
        )
        .expect("truncated file must securely open before parsing");
        assert!(
            read_safetensors_header(
                &mut pinned,
                u64::try_from(truncated.len()).expect("length must fit")
            )
            .is_err()
        );
        fs::remove_dir_all(root).expect("temporary tree must remove");

        let overlap = concat!(
            "{\"a\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]},",
            "\"b\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[2,6]}}"
        );
        let file_length = 8_u64
            .checked_add(u64::try_from(overlap.len()).expect("header length must fit"))
            .and_then(|length| length.checked_add(6))
            .expect("file length must fit");
        assert!(ParsedSafetensorsHeader::parse(overlap.as_bytes(), file_length).is_err());
    }

    #[test]
    fn pinned_descriptor_detects_path_replacement_and_length_mutation() {
        let root = private_temp("identity-mutation");
        let path = root.join("model.safetensors");
        private_file(&path, b"first");
        let uid = fs::metadata(&root).expect("root metadata").uid();
        let pinned =
            PinnedSourceFile::open(&path, uid, &[5], "test").expect("source must securely open");

        let replacement = root.join("replacement");
        private_file(&replacement, b"other");
        fs::rename(&replacement, &path).expect("replacement must be atomic");
        assert!(pinned.verify_unchanged().is_err());
        fs::remove_dir_all(&root).expect("temporary tree must remove");

        let root = private_temp("length-mutation");
        let path = root.join("model.safetensors");
        private_file(&path, b"first");
        let uid = fs::metadata(&root).expect("root metadata").uid();
        let pinned =
            PinnedSourceFile::open(&path, uid, &[5], "test").expect("source must securely open");
        fs::set_permissions(&path, fs::Permissions::from_mode(0o600))
            .expect("test mutation permissions must set");
        let writable = OpenOptions::new().write(true).open(&path).expect("test mutation opens");
        writable.set_len(4).expect("test mutation truncates");
        assert!(pinned.verify_unchanged().is_err());
        fs::remove_dir_all(root).expect("temporary tree must remove");
    }

    #[test]
    fn root_and_artifact_names_fail_closed() {
        assert!(validate_absolute_normal_root(Path::new("relative")).is_err());
        for name in ["", ".", ".hidden", "../escape", "a/b", "https://example.invalid"] {
            assert!(validate_artifact_name(name).is_err());
        }

        let root = private_temp("directory-symlink");
        let linked = root.with_extension("link");
        symlink(&root, &linked).expect("directory symlink must create");
        assert!(DirectoryGuard::open(&linked, None, "test root").is_err());
        fs::remove_file(linked).expect("symlink must remove");
        fs::remove_dir_all(root).expect("temporary tree must remove");
    }

    #[test]
    fn source_files_require_exact_read_only_data_permissions() {
        let root = private_temp("source-file-modes");
        let path = root.join("model.safetensors");
        private_file(&path, b"x");
        let uid = fs::metadata(&root).expect("root metadata").uid();
        assert!(PinnedSourceFile::open(&path, uid, &[1], "test").is_ok());

        for mode in [0o600, 0o500, 0o700, 0o4500, 0o040] {
            fs::set_permissions(&path, fs::Permissions::from_mode(mode))
                .expect("test mode must set");
            assert!(PinnedSourceFile::open(&path, uid, &[1], "test").is_err());
        }
        fs::remove_dir_all(root).expect("temporary tree must remove");
    }
}
