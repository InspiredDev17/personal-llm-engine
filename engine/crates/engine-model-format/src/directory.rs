//! Canonical binary manifest for directory-backed `.paimodel` packages.

use personalai_engine_types::{
    ArchitectureId, ContextLimit, DType, EngineError, MAX_DIMENSION_SIZE, MAX_MODEL_PACKAGE_BYTES,
    MAX_TENSOR_COUNT, MAX_TENSOR_RANK, MAX_VOCABULARY_SIZE, ModelId, ModelPackageHash, TensorShape,
};
use std::collections::{BTreeMap, BTreeSet};

/// Exact architecture implemented by the first production package profile.
pub const DIRECTORY_DECODER_ARCHITECTURE_ID: &str = "decoder-rms-rope-swiglu-gqa-v1";
/// Exact Qwen2.5 decoder profile with scalar correctness scaffolding.
pub const DIRECTORY_QWEN2_DECODER_ARCHITECTURE_ID: &str =
    "qwen2.5-rms-rope-swiglu-gqa-bias-tied-v1";
/// Directory-manifest magic.
pub const DIRECTORY_MANIFEST_MAGIC: [u8; 8] = *b"PAIDIR01";
/// Original directory-manifest version for the legacy decoder profile.
pub const DIRECTORY_MANIFEST_VERSION: (u16, u16) = (1, 0);
/// Directory-manifest version for the closed Qwen2.5 decoder profile.
pub const DIRECTORY_QWEN2_MANIFEST_VERSION: (u16, u16) = (1, 1);
/// Minimum engine revision eligible to activate the Qwen2.5 manifest profile.
///
/// The current production runtime remains 0.1.0. Keeping this profile at a
/// future revision makes v1.1 packages unambiguously non-activatable until the
/// tokenizer, registry, backend, and release gates are completed together.
pub const DIRECTORY_QWEN2_MINIMUM_ENGINE_VERSION: (u16, u16, u16) = (0, 2, 0);
/// Fixed directory-manifest header length.
pub const DIRECTORY_MANIFEST_HEADER_BYTES: usize = 512;
/// Fixed shard-record length.
pub const DIRECTORY_SHARD_RECORD_BYTES: usize = 64;
/// Fixed tensor-record length.
pub const DIRECTORY_TENSOR_RECORD_BYTES: usize = 192;
/// Largest accepted directory manifest: 64 MiB.
pub const DIRECTORY_MANIFEST_MAX_BYTES: usize = 64 * 1024 * 1024;
/// Largest number of immutable tensor shards.
pub const MAX_DIRECTORY_SHARDS: u32 = 1_024;
/// Largest required tensor alignment.
pub const MAX_TENSOR_ALIGNMENT: u32 = 2 * 1024 * 1024;
/// Maximum canonical tensor-name bytes.
pub const MAX_DIRECTORY_TENSOR_NAME_BYTES: usize = 47;
/// Number of logical values in every AIPQ block.
pub const AIPQ_BLOCK_VALUES: u64 = 32;
/// Fixed self-describing header length for one quantized tensor payload.
pub const AIPQ_TENSOR_HEADER_BYTES: u64 = 32;
/// Fixed AIPQ8 block length: one little-endian F32 scale and 32 signed bytes.
pub const AIPQ8_BLOCK_BYTES: u64 = 36;
/// Fixed AIPQ4 block length: one little-endian F32 scale and 16 packed bytes.
pub const AIPQ4_BLOCK_BYTES: u64 = 20;
/// Required package alignment for all AIPQ tensor payloads.
pub const AIPQ_REQUIRED_ALIGNMENT: u32 = 64;

const ENDIAN_MARKER: u32 = 0x0102_0304;
const FLAG_TIED_OUTPUT: u32 = 1;
const KNOWN_FLAGS: u32 = FLAG_TIED_OUTPUT;
const LEGACY_RMS_NORM_EPSILON: f32 = 1.0e-5;

/// Closed quantization declaration for directory package version one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Quantization {
    /// Exact unquantized tensor storage.
    None,
    /// First-party symmetric signed eight-bit block storage.
    AipQ8V1,
    /// First-party symmetric packed signed four-bit block storage.
    AipQ4V1,
}

impl Quantization {
    /// Returns the stable package-format identifier.
    pub const fn id(self) -> u8 {
        match self {
            Self::None => 0,
            Self::AipQ8V1 => 1,
            Self::AipQ4V1 => 2,
        }
    }

    /// Returns the exact encoded payload length for a logical element count.
    pub fn payload_len(self, element_count: u64) -> Result<u64, EngineError> {
        if element_count == 0 {
            return Err(corrupt("quantized tensor element count is zero"));
        }
        match self {
            Self::None => Err(corrupt("unquantized storage has no AIPQ payload length")),
            Self::AipQ8V1 | Self::AipQ4V1 => {
                let blocks = element_count
                    .checked_add(AIPQ_BLOCK_VALUES - 1)
                    .and_then(|value| value.checked_div(AIPQ_BLOCK_VALUES))
                    .ok_or_else(|| corrupt("quantized tensor block count overflows"))?;
                let block_bytes = match self {
                    Self::AipQ8V1 => AIPQ8_BLOCK_BYTES,
                    Self::AipQ4V1 => AIPQ4_BLOCK_BYTES,
                    Self::None => return Err(corrupt("unquantized storage has no AIPQ blocks")),
                };
                blocks
                    .checked_mul(block_bytes)
                    .and_then(|bytes| bytes.checked_add(AIPQ_TENSOR_HEADER_BYTES))
                    .filter(|bytes| *bytes <= MAX_MODEL_PACKAGE_BYTES)
                    .ok_or_else(|| corrupt("quantized tensor payload length overflows"))
            }
        }
    }

    fn decode(value: u8) -> Result<Self, EngineError> {
        match value {
            0 => Ok(Self::None),
            1 => Ok(Self::AipQ8V1),
            2 => Ok(Self::AipQ4V1),
            _ => Err(corrupt("directory manifest quantization is unsupported")),
        }
    }
}

/// One immutable shard declared by a directory manifest.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DirectoryShardRecord {
    index: u32,
    byte_length: u64,
    content_hash: ModelPackageHash,
}

impl DirectoryShardRecord {
    /// Constructs a bounded shard record.
    pub fn new(
        index: u32,
        byte_length: u64,
        content_hash: ModelPackageHash,
    ) -> Result<Self, EngineError> {
        if index >= MAX_DIRECTORY_SHARDS
            || byte_length == 0
            || byte_length > MAX_MODEL_PACKAGE_BYTES
        {
            return Err(corrupt("directory shard metadata is outside hard bounds"));
        }
        Ok(Self { index, byte_length, content_hash })
    }

    /// Returns the zero-based shard index.
    pub const fn index(&self) -> u32 {
        self.index
    }

    /// Returns the exact shard byte length.
    pub const fn byte_length(&self) -> u64 {
        self.byte_length
    }

    /// Returns the exact SHA-384 shard digest.
    pub const fn content_hash(&self) -> ModelPackageHash {
        self.content_hash
    }

    /// Returns the canonical package filename for this shard.
    pub fn file_name(&self) -> String {
        format!("tensors-{:04}.aipt", self.index)
    }
}

/// One canonical tensor range in an immutable shard.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DirectoryTensorRecord {
    name: String,
    dtype: DType,
    shape: TensorShape,
    shard_index: u32,
    byte_offset: u64,
    byte_length: u64,
    required_alignment: u32,
    quantization: Quantization,
    content_hash: ModelPackageHash,
}

impl DirectoryTensorRecord {
    /// Parses one exact fixed-width tensor-index record.
    ///
    /// Package-level shard existence, range overlap, and bounds remain the
    /// responsibility of [`DirectoryManifest::parse`]. This entry point exists
    /// for isolated parser testing and bounded fuzzing.
    pub fn parse_exact(bytes: &[u8]) -> Result<Self, EngineError> {
        if bytes.len() != DIRECTORY_TENSOR_RECORD_BYTES {
            return Err(corrupt("directory tensor record length is invalid"));
        }
        parse_tensor_record(bytes)
    }

    /// Constructs and validates one tensor record.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        name: impl Into<String>,
        dtype: DType,
        shape: TensorShape,
        shard_index: u32,
        byte_offset: u64,
        byte_length: u64,
        required_alignment: u32,
        quantization: Quantization,
        content_hash: ModelPackageHash,
    ) -> Result<Self, EngineError> {
        let name = name.into();
        validate_tensor_name(&name)?;
        if shard_index >= MAX_DIRECTORY_SHARDS {
            return Err(corrupt("tensor shard index is outside hard bounds"));
        }
        if required_alignment == 0
            || !required_alignment.is_power_of_two()
            || required_alignment > MAX_TENSOR_ALIGNMENT
            || byte_offset % u64::from(required_alignment) != 0
        {
            return Err(corrupt("tensor alignment is invalid"));
        }
        let expected_length = match quantization {
            Quantization::None => shape.byte_len(dtype)?,
            Quantization::AipQ8V1 | Quantization::AipQ4V1 => {
                if dtype != DType::F32 || required_alignment < AIPQ_REQUIRED_ALIGNMENT {
                    return Err(corrupt("quantized tensor dtype or required alignment is invalid"));
                }
                quantization.payload_len(shape.element_count())?
            }
        };
        if byte_length == 0 || byte_length != expected_length {
            return Err(corrupt("tensor byte length does not match storage declaration"));
        }
        byte_offset
            .checked_add(byte_length)
            .ok_or_else(|| corrupt("tensor byte range overflows"))?;
        Ok(Self {
            name,
            dtype,
            shape,
            shard_index,
            byte_offset,
            byte_length,
            required_alignment,
            quantization,
            content_hash,
        })
    }

    /// Returns the canonical architecture-owned tensor name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns the exact tensor dtype.
    pub const fn dtype(&self) -> DType {
        self.dtype
    }

    /// Returns the checked tensor shape.
    pub const fn shape(&self) -> &TensorShape {
        &self.shape
    }

    /// Returns the containing shard index.
    pub const fn shard_index(&self) -> u32 {
        self.shard_index
    }

    /// Returns the byte offset inside the shard.
    pub const fn byte_offset(&self) -> u64 {
        self.byte_offset
    }

    /// Returns the exact tensor byte length.
    pub const fn byte_length(&self) -> u64 {
        self.byte_length
    }

    /// Returns the required power-of-two alignment.
    pub const fn required_alignment(&self) -> u32 {
        self.required_alignment
    }

    /// Returns the closed quantization declaration.
    pub const fn quantization(&self) -> Quantization {
        self.quantization
    }

    /// Returns the exact SHA-384 tensor digest.
    pub const fn content_hash(&self) -> ModelPackageHash {
        self.content_hash
    }
}

/// Parsed, fully validated directory-package manifest.
#[derive(Clone, Debug, PartialEq)]
pub struct DirectoryManifest {
    format_version: (u16, u16),
    model_id: ModelId,
    architecture_id: ArchitectureId,
    required_engine_version: (u16, u16, u16),
    context_limit: ContextLimit,
    vocabulary_size: u32,
    hidden_dimension: u32,
    layer_count: u32,
    attention_head_count: u32,
    kv_head_count: u32,
    feed_forward_dimension: u32,
    rms_norm_epsilon: f32,
    rope_theta: f32,
    rope_scaling: f32,
    storage_dtype: DType,
    quantization: Quantization,
    tied_output: bool,
    tokenizer_hash: ModelPackageHash,
    shards: Vec<DirectoryShardRecord>,
    tensors: Vec<DirectoryTensorRecord>,
}

impl DirectoryManifest {
    /// Parses one complete manifest with no accepted trailing bytes.
    #[allow(clippy::too_many_lines)]
    pub fn parse(bytes: &[u8]) -> Result<Self, EngineError> {
        if bytes.len() < DIRECTORY_MANIFEST_HEADER_BYTES {
            return Err(corrupt("directory manifest header is truncated"));
        }
        if bytes.len() > DIRECTORY_MANIFEST_MAX_BYTES {
            return Err(EngineError::resource_exhaustion(
                "directory manifest exceeds its hard byte limit",
            ));
        }
        if bytes[..8] != DIRECTORY_MANIFEST_MAGIC {
            return Err(corrupt("directory manifest magic is invalid"));
        }
        let format_version = (read_u16(bytes, 8)?, read_u16(bytes, 10)?);
        if !matches!(format_version, DIRECTORY_MANIFEST_VERSION | DIRECTORY_QWEN2_MANIFEST_VERSION)
        {
            return Err(corrupt("directory manifest version is unsupported"));
        }
        if read_u32(bytes, 12)? != ENDIAN_MARKER {
            return Err(corrupt("directory manifest byte order is unsupported"));
        }
        if read_u32(bytes, 16)? != usize_to_u32(DIRECTORY_MANIFEST_HEADER_BYTES)? {
            return Err(corrupt("directory manifest header length is unsupported"));
        }
        let flags = read_u32(bytes, 20)?;
        if flags & !KNOWN_FLAGS != 0 {
            return Err(EngineError::not_implemented(
                "directory manifest required feature is unsupported",
            ));
        }
        let required_engine_version =
            (read_u16(bytes, 24)?, read_u16(bytes, 26)?, read_u16(bytes, 28)?);
        if read_u16(bytes, 30)? != 0 {
            return Err(corrupt("directory manifest reserved header field is nonzero"));
        }
        let architecture_length = usize::from(read_u16(bytes, 32)?);
        let model_id_length = usize::from(read_u16(bytes, 34)?);
        let tensor_count = read_u32(bytes, 36)?;
        let shard_count = read_u32(bytes, 40)?;
        let storage_dtype = decode_dtype(bytes[44])?;
        let quantization = Quantization::decode(bytes[45])?;
        if read_u16(bytes, 46)? != 0 {
            return Err(corrupt("directory manifest reserved dtype field is nonzero"));
        }
        let context_limit = ContextLimit::new(read_u32(bytes, 48)?)
            .map_err(|_| corrupt("directory manifest context is invalid"))?;
        let vocabulary_size = read_u32(bytes, 52)?;
        let hidden_dimension = read_nonzero_u32(bytes, 56, "hidden dimension")?;
        let layer_count = read_nonzero_u32(bytes, 60, "layer count")?;
        let attention_head_count = read_nonzero_u32(bytes, 64, "attention head count")?;
        let kv_head_count = read_nonzero_u32(bytes, 68, "key/value head count")?;
        let feed_forward_dimension = read_nonzero_u32(bytes, 72, "feed-forward dimension")?;
        let rope_theta = f32::from_bits(read_u32(bytes, 76)?);
        let rope_scaling = f32::from_bits(read_u32(bytes, 80)?);
        let rms_norm_epsilon = match format_version {
            DIRECTORY_MANIFEST_VERSION => {
                if read_u32(bytes, 84)? != 0 {
                    return Err(EngineError::not_implemented(
                        "directory manifest optional section is unsupported",
                    ));
                }
                LEGACY_RMS_NORM_EPSILON
            }
            DIRECTORY_QWEN2_MANIFEST_VERSION => f32::from_bits(read_u32(bytes, 84)?),
            _ => return Err(corrupt("directory manifest version is unsupported")),
        };
        let declared_length = read_u64(bytes, 88)?;
        if declared_length != usize_to_u64(bytes.len())? {
            return Err(corrupt("directory manifest length or trailing data is invalid"));
        }
        if tensor_count == 0 || tensor_count > MAX_TENSOR_COUNT {
            return Err(corrupt("directory manifest tensor count is outside hard bounds"));
        }
        if shard_count == 0 || shard_count > MAX_DIRECTORY_SHARDS {
            return Err(corrupt("directory manifest shard count is outside hard bounds"));
        }
        if vocabulary_size == 0 || vocabulary_size > MAX_VOCABULARY_SIZE {
            return Err(corrupt("directory manifest vocabulary is outside hard bounds"));
        }
        if !rms_norm_epsilon.is_finite()
            || rms_norm_epsilon <= 0.0
            || rms_norm_epsilon > 1.0
            || !rope_theta.is_finite()
            || rope_theta <= 1.0
            || !rope_scaling.is_finite()
            || rope_scaling <= 0.0
        {
            return Err(corrupt("directory manifest RoPE parameters are invalid"));
        }
        let expected_length = manifest_length(shard_count, tensor_count)?;
        if bytes.len() != expected_length {
            return Err(corrupt("directory manifest record length is not canonical"));
        }
        let architecture = parse_padded_text(bytes, 144, 64, architecture_length, "architecture")?;
        let architecture_id = ArchitectureId::new(architecture)
            .map_err(|_| corrupt("directory manifest architecture identifier is invalid"))?;
        let expected_architecture = match format_version {
            DIRECTORY_MANIFEST_VERSION => DIRECTORY_DECODER_ARCHITECTURE_ID,
            DIRECTORY_QWEN2_MANIFEST_VERSION => DIRECTORY_QWEN2_DECODER_ARCHITECTURE_ID,
            _ => return Err(corrupt("directory manifest version is unsupported")),
        };
        if architecture_id.as_str() != expected_architecture {
            if matches!(
                architecture_id.as_str(),
                DIRECTORY_DECODER_ARCHITECTURE_ID | DIRECTORY_QWEN2_DECODER_ARCHITECTURE_ID
            ) {
                return Err(corrupt(
                    "directory manifest architecture and format version are inconsistent",
                ));
            }
            return Err(EngineError::unsupported_architecture(architecture_id));
        }
        if format_version == DIRECTORY_QWEN2_MANIFEST_VERSION
            && (flags & FLAG_TIED_OUTPUT == 0 || rope_scaling.to_bits() != 1.0_f32.to_bits())
        {
            return Err(corrupt(
                "Qwen2.5 manifest requires tied output and unscaled rotary positions",
            ));
        }
        if format_version == DIRECTORY_QWEN2_MANIFEST_VERSION
            && required_engine_version < DIRECTORY_QWEN2_MINIMUM_ENGINE_VERSION
        {
            return Err(corrupt(
                "Qwen2.5 manifest declares an engine version predating the profile",
            ));
        }
        let model_id_text = parse_padded_text(bytes, 208, 128, model_id_length, "model ID")?;
        let model_id = ModelId::new(model_id_text)
            .map_err(|_| corrupt("directory manifest model identifier is invalid"))?;
        if bytes[336..DIRECTORY_MANIFEST_HEADER_BYTES].iter().any(|byte| *byte != 0) {
            return Err(corrupt("directory manifest reserved bytes are nonzero"));
        }
        let mut tokenizer_hash = [0_u8; ModelPackageHash::BYTE_LEN];
        tokenizer_hash.copy_from_slice(&bytes[96..144]);
        let tokenizer_hash = ModelPackageHash::from_sha384_bytes(tokenizer_hash);

        let mut shards = Vec::new();
        shards
            .try_reserve_exact(usize::try_from(shard_count).map_err(|_| {
                EngineError::resource_exhaustion("directory shard count is not addressable")
            })?)
            .map_err(|_| EngineError::resource_exhaustion("directory shard allocation failed"))?;
        let mut cursor = DIRECTORY_MANIFEST_HEADER_BYTES;
        for expected_index in 0..shard_count {
            let record = parse_shard_record(record(bytes, cursor, DIRECTORY_SHARD_RECORD_BYTES)?)?;
            if record.index() != expected_index {
                return Err(corrupt("directory shard indices are not canonical"));
            }
            shards.push(record);
            cursor = checked_add_usize(cursor, DIRECTORY_SHARD_RECORD_BYTES)?;
        }

        let mut tensors = Vec::new();
        tensors
            .try_reserve_exact(usize::try_from(tensor_count).map_err(|_| {
                EngineError::resource_exhaustion("directory tensor count is not addressable")
            })?)
            .map_err(|_| EngineError::resource_exhaustion("directory tensor allocation failed"))?;
        let mut names = BTreeSet::new();
        for _ in 0..tensor_count {
            let tensor =
                parse_tensor_record(record(bytes, cursor, DIRECTORY_TENSOR_RECORD_BYTES)?)?;
            if !names.insert(tensor.name().to_owned()) {
                return Err(corrupt("directory manifest contains a duplicate tensor name"));
            }
            if tensor.shard_index() >= shard_count {
                return Err(corrupt("tensor references an unavailable shard"));
            }
            tensors.push(tensor);
            cursor = checked_add_usize(cursor, DIRECTORY_TENSOR_RECORD_BYTES)?;
        }
        if cursor != bytes.len() {
            return Err(corrupt("directory manifest has trailing unrecognized data"));
        }
        if !tensors.windows(2).all(|pair| pair[0].name() < pair[1].name()) {
            return Err(corrupt("directory tensor names are not in canonical order"));
        }
        validate_tensor_ranges(&shards, &tensors)?;
        validate_quantization_profile(storage_dtype, quantization, &tensors)?;

        Ok(Self {
            format_version,
            model_id,
            architecture_id,
            required_engine_version,
            context_limit,
            vocabulary_size,
            hidden_dimension,
            layer_count,
            attention_head_count,
            kv_head_count,
            feed_forward_dimension,
            rms_norm_epsilon,
            rope_theta,
            rope_scaling,
            storage_dtype,
            quantization,
            tied_output: flags & FLAG_TIED_OUTPUT != 0,
            tokenizer_hash,
            shards,
            tensors,
        })
    }

    /// Encodes the canonical manifest exactly.
    pub fn encode(&self) -> Result<Vec<u8>, EngineError> {
        DirectoryManifestBuilder::from_manifest(self).build_bytes()
    }

    /// Returns the exact directory-manifest format revision.
    pub const fn format_version(&self) -> (u16, u16) {
        self.format_version
    }

    /// Returns the package model identifier.
    pub const fn model_id(&self) -> &ModelId {
        &self.model_id
    }

    /// Returns the closed architecture identifier.
    pub const fn architecture_id(&self) -> &ArchitectureId {
        &self.architecture_id
    }

    /// Returns the minimum compatible engine semantic version.
    pub const fn required_engine_version(&self) -> (u16, u16, u16) {
        self.required_engine_version
    }

    /// Returns the model context limit.
    pub const fn context_limit(&self) -> ContextLimit {
        self.context_limit
    }

    /// Returns the tokenizer vocabulary size.
    pub const fn vocabulary_size(&self) -> u32 {
        self.vocabulary_size
    }

    /// Returns the hidden dimension.
    pub const fn hidden_dimension(&self) -> u32 {
        self.hidden_dimension
    }

    /// Returns the decoder layer count.
    pub const fn layer_count(&self) -> u32 {
        self.layer_count
    }

    /// Returns the attention head count.
    pub const fn attention_head_count(&self) -> u32 {
        self.attention_head_count
    }

    /// Returns the grouped-query key/value head count.
    pub const fn kv_head_count(&self) -> u32 {
        self.kv_head_count
    }

    /// Returns the feed-forward intermediate dimension.
    pub const fn feed_forward_dimension(&self) -> u32 {
        self.feed_forward_dimension
    }

    /// Returns the architecture-bound RMS normalization epsilon.
    pub const fn rms_norm_epsilon(&self) -> f32 {
        self.rms_norm_epsilon
    }

    /// Returns the rotary base theta.
    pub const fn rope_theta(&self) -> f32 {
        self.rope_theta
    }

    /// Returns the rotary scaling factor.
    pub const fn rope_scaling(&self) -> f32 {
        self.rope_scaling
    }

    /// Returns the package-wide storage dtype.
    pub const fn storage_dtype(&self) -> DType {
        self.storage_dtype
    }

    /// Returns the closed quantization declaration.
    pub const fn quantization(&self) -> Quantization {
        self.quantization
    }

    /// Returns whether output weights are tied to embeddings.
    pub const fn tied_output(&self) -> bool {
        self.tied_output
    }

    /// Returns the tokenizer SHA-384 digest.
    pub const fn tokenizer_hash(&self) -> ModelPackageHash {
        self.tokenizer_hash
    }

    /// Returns all canonical shard records.
    pub fn shards(&self) -> &[DirectoryShardRecord] {
        &self.shards
    }

    /// Returns all canonical tensor records.
    pub fn tensors(&self) -> &[DirectoryTensorRecord] {
        &self.tensors
    }
}

/// Explicit builder used by the offline importer and deterministic fixtures.
#[derive(Clone, Debug)]
pub struct DirectoryManifestBuilder {
    manifest: DirectoryManifest,
}

impl DirectoryManifestBuilder {
    /// Creates a manifest builder after validating every cross-record invariant.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        model_id: ModelId,
        required_engine_version: (u16, u16, u16),
        context_limit: ContextLimit,
        vocabulary_size: u32,
        hidden_dimension: u32,
        layer_count: u32,
        attention_head_count: u32,
        kv_head_count: u32,
        feed_forward_dimension: u32,
        rope_theta: f32,
        rope_scaling: f32,
        storage_dtype: DType,
        quantization: Quantization,
        tied_output: bool,
        tokenizer_hash: ModelPackageHash,
        shards: Vec<DirectoryShardRecord>,
        tensors: Vec<DirectoryTensorRecord>,
    ) -> Result<Self, EngineError> {
        Self::new_profile(
            DIRECTORY_MANIFEST_VERSION,
            DIRECTORY_DECODER_ARCHITECTURE_ID,
            model_id,
            required_engine_version,
            context_limit,
            vocabulary_size,
            hidden_dimension,
            layer_count,
            attention_head_count,
            kv_head_count,
            feed_forward_dimension,
            LEGACY_RMS_NORM_EPSILON,
            rope_theta,
            rope_scaling,
            storage_dtype,
            quantization,
            tied_output,
            tokenizer_hash,
            shards,
            tensors,
        )
    }

    /// Creates a versioned manifest for the closed Qwen2.5 scalar profile.
    ///
    /// The architecture fixes split-half rotary positions, Q/K/V projection
    /// biases, and tied token-embedding/output weights. Source conversion
    /// remains an offline Model Foundry responsibility.
    #[allow(clippy::too_many_arguments)]
    pub fn new_qwen2(
        model_id: ModelId,
        required_engine_version: (u16, u16, u16),
        context_limit: ContextLimit,
        vocabulary_size: u32,
        hidden_dimension: u32,
        layer_count: u32,
        attention_head_count: u32,
        kv_head_count: u32,
        feed_forward_dimension: u32,
        rms_norm_epsilon: f32,
        rope_theta: f32,
        storage_dtype: DType,
        quantization: Quantization,
        tokenizer_hash: ModelPackageHash,
        shards: Vec<DirectoryShardRecord>,
        tensors: Vec<DirectoryTensorRecord>,
    ) -> Result<Self, EngineError> {
        Self::new_profile(
            DIRECTORY_QWEN2_MANIFEST_VERSION,
            DIRECTORY_QWEN2_DECODER_ARCHITECTURE_ID,
            model_id,
            required_engine_version,
            context_limit,
            vocabulary_size,
            hidden_dimension,
            layer_count,
            attention_head_count,
            kv_head_count,
            feed_forward_dimension,
            rms_norm_epsilon,
            rope_theta,
            1.0,
            storage_dtype,
            quantization,
            true,
            tokenizer_hash,
            shards,
            tensors,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn new_profile(
        format_version: (u16, u16),
        architecture_id: &str,
        model_id: ModelId,
        required_engine_version: (u16, u16, u16),
        context_limit: ContextLimit,
        vocabulary_size: u32,
        hidden_dimension: u32,
        layer_count: u32,
        attention_head_count: u32,
        kv_head_count: u32,
        feed_forward_dimension: u32,
        rms_norm_epsilon: f32,
        rope_theta: f32,
        rope_scaling: f32,
        storage_dtype: DType,
        quantization: Quantization,
        tied_output: bool,
        tokenizer_hash: ModelPackageHash,
        shards: Vec<DirectoryShardRecord>,
        mut tensors: Vec<DirectoryTensorRecord>,
    ) -> Result<Self, EngineError> {
        tensors.sort_by(|left, right| left.name().cmp(right.name()));
        let manifest = DirectoryManifest {
            format_version,
            model_id,
            architecture_id: ArchitectureId::new(architecture_id)?,
            required_engine_version,
            context_limit,
            vocabulary_size,
            hidden_dimension,
            layer_count,
            attention_head_count,
            kv_head_count,
            feed_forward_dimension,
            rms_norm_epsilon,
            rope_theta,
            rope_scaling,
            storage_dtype,
            quantization,
            tied_output,
            tokenizer_hash,
            shards,
            tensors,
        };
        let encoded = Self { manifest: manifest.clone() }.build_bytes()?;
        DirectoryManifest::parse(&encoded)?;
        Ok(Self { manifest })
    }

    fn from_manifest(manifest: &DirectoryManifest) -> Self {
        Self { manifest: manifest.clone() }
    }

    /// Returns the validated manifest value.
    pub const fn manifest(&self) -> &DirectoryManifest {
        &self.manifest
    }

    /// Builds deterministic canonical manifest bytes.
    pub fn build_bytes(&self) -> Result<Vec<u8>, EngineError> {
        let shard_count = u32::try_from(self.manifest.shards.len())
            .map_err(|_| corrupt("directory shard count exceeds its field"))?;
        let tensor_count = u32::try_from(self.manifest.tensors.len())
            .map_err(|_| corrupt("directory tensor count exceeds its field"))?;
        let length = manifest_length(shard_count, tensor_count)?;
        let mut bytes = zeroed(length)?;
        bytes[..8].copy_from_slice(&DIRECTORY_MANIFEST_MAGIC);
        write_u16(&mut bytes, 8, self.manifest.format_version.0)?;
        write_u16(&mut bytes, 10, self.manifest.format_version.1)?;
        write_u32(&mut bytes, 12, ENDIAN_MARKER)?;
        write_u32(&mut bytes, 16, usize_to_u32(DIRECTORY_MANIFEST_HEADER_BYTES)?)?;
        write_u32(&mut bytes, 20, u32::from(self.manifest.tied_output))?;
        write_u16(&mut bytes, 24, self.manifest.required_engine_version.0)?;
        write_u16(&mut bytes, 26, self.manifest.required_engine_version.1)?;
        write_u16(&mut bytes, 28, self.manifest.required_engine_version.2)?;
        write_u16(
            &mut bytes,
            32,
            u16::try_from(self.manifest.architecture_id.as_str().len())
                .map_err(|_| corrupt("architecture identifier exceeds its field"))?,
        )?;
        write_u16(
            &mut bytes,
            34,
            u16::try_from(self.manifest.model_id.as_str().len())
                .map_err(|_| corrupt("model identifier exceeds its field"))?,
        )?;
        write_u32(&mut bytes, 36, tensor_count)?;
        write_u32(&mut bytes, 40, shard_count)?;
        bytes[44] = encode_dtype(self.manifest.storage_dtype)?;
        bytes[45] = self.manifest.quantization.id();
        write_u32(&mut bytes, 48, self.manifest.context_limit.get())?;
        write_u32(&mut bytes, 52, self.manifest.vocabulary_size)?;
        write_u32(&mut bytes, 56, self.manifest.hidden_dimension)?;
        write_u32(&mut bytes, 60, self.manifest.layer_count)?;
        write_u32(&mut bytes, 64, self.manifest.attention_head_count)?;
        write_u32(&mut bytes, 68, self.manifest.kv_head_count)?;
        write_u32(&mut bytes, 72, self.manifest.feed_forward_dimension)?;
        write_u32(&mut bytes, 76, self.manifest.rope_theta.to_bits())?;
        write_u32(&mut bytes, 80, self.manifest.rope_scaling.to_bits())?;
        if self.manifest.format_version == DIRECTORY_QWEN2_MANIFEST_VERSION {
            write_u32(&mut bytes, 84, self.manifest.rms_norm_epsilon.to_bits())?;
        }
        write_u64(&mut bytes, 88, usize_to_u64(length)?)?;
        bytes[96..144].copy_from_slice(self.manifest.tokenizer_hash.as_bytes());
        write_padded_text(&mut bytes, 144, 64, self.manifest.architecture_id.as_str())?;
        write_padded_text(&mut bytes, 208, 128, self.manifest.model_id.as_str())?;

        let mut cursor = DIRECTORY_MANIFEST_HEADER_BYTES;
        for shard in &self.manifest.shards {
            write_u32(&mut bytes, cursor, shard.index)?;
            write_u64(&mut bytes, cursor + 8, shard.byte_length)?;
            write_exact(&mut bytes, cursor + 16, shard.content_hash.as_bytes())?;
            cursor = checked_add_usize(cursor, DIRECTORY_SHARD_RECORD_BYTES)?;
        }
        for tensor in &self.manifest.tensors {
            write_u16(
                &mut bytes,
                cursor,
                u16::try_from(tensor.name.len())
                    .map_err(|_| corrupt("tensor name exceeds its field"))?,
            )?;
            bytes[cursor + 2] = encode_dtype(tensor.dtype)?;
            bytes[cursor + 3] = u8::try_from(tensor.shape.rank())
                .map_err(|_| corrupt("tensor rank exceeds its field"))?;
            write_u32(&mut bytes, cursor + 4, tensor.shard_index)?;
            write_u32(&mut bytes, cursor + 8, tensor.required_alignment)?;
            bytes[cursor + 12] = tensor.quantization.id();
            for (index, dimension) in tensor.shape.dimensions().iter().enumerate() {
                write_u64(&mut bytes, cursor + 16 + index * 8, *dimension)?;
            }
            write_u64(&mut bytes, cursor + 80, tensor.byte_offset)?;
            write_u64(&mut bytes, cursor + 88, tensor.byte_length)?;
            write_exact(&mut bytes, cursor + 96, tensor.content_hash.as_bytes())?;
            write_padded_text(&mut bytes, cursor + 144, 48, &tensor.name)?;
            cursor = checked_add_usize(cursor, DIRECTORY_TENSOR_RECORD_BYTES)?;
        }
        Ok(bytes)
    }
}

fn parse_shard_record(bytes: &[u8]) -> Result<DirectoryShardRecord, EngineError> {
    if bytes[4..8].iter().any(|byte| *byte != 0) {
        return Err(corrupt("directory shard reserved bytes are nonzero"));
    }
    let index = read_u32(bytes, 0)?;
    let byte_length = read_u64(bytes, 8)?;
    let mut hash = [0_u8; ModelPackageHash::BYTE_LEN];
    hash.copy_from_slice(&bytes[16..64]);
    DirectoryShardRecord::new(index, byte_length, ModelPackageHash::from_sha384_bytes(hash))
}

fn parse_tensor_record(bytes: &[u8]) -> Result<DirectoryTensorRecord, EngineError> {
    let name_length = usize::from(read_u16(bytes, 0)?);
    let dtype = decode_dtype(bytes[2])?;
    let rank = usize::from(bytes[3]);
    if rank > MAX_TENSOR_RANK {
        return Err(corrupt("directory tensor rank exceeds its hard limit"));
    }
    let shard_index = read_u32(bytes, 4)?;
    let required_alignment = read_u32(bytes, 8)?;
    let quantization = Quantization::decode(bytes[12])?;
    if bytes[13..16].iter().any(|byte| *byte != 0) {
        return Err(corrupt("directory tensor reserved bytes are nonzero"));
    }
    let mut dimensions = Vec::new();
    dimensions
        .try_reserve_exact(rank)
        .map_err(|_| EngineError::resource_exhaustion("tensor shape allocation failed"))?;
    for index in 0..MAX_TENSOR_RANK {
        let dimension = read_u64(bytes, 16 + index * 8)?;
        if index < rank {
            if dimension == 0 || dimension > MAX_DIMENSION_SIZE {
                return Err(corrupt("directory tensor dimension is outside hard bounds"));
            }
            dimensions.push(dimension);
        } else if dimension != 0 {
            return Err(corrupt("unused directory tensor dimensions are nonzero"));
        }
    }
    let byte_offset = read_u64(bytes, 80)?;
    let byte_length = read_u64(bytes, 88)?;
    let mut hash = [0_u8; ModelPackageHash::BYTE_LEN];
    hash.copy_from_slice(&bytes[96..144]);
    let name = parse_padded_text(bytes, 144, 48, name_length, "tensor name")?;
    DirectoryTensorRecord::new(
        name,
        dtype,
        TensorShape::new(dimensions).map_err(|_| corrupt("directory tensor shape is invalid"))?,
        shard_index,
        byte_offset,
        byte_length,
        required_alignment,
        quantization,
        ModelPackageHash::from_sha384_bytes(hash),
    )
}

fn validate_tensor_ranges(
    shards: &[DirectoryShardRecord],
    tensors: &[DirectoryTensorRecord],
) -> Result<(), EngineError> {
    let mut ranges = BTreeMap::<u32, Vec<(u64, u64)>>::new();
    for tensor in tensors {
        let end = tensor
            .byte_offset
            .checked_add(tensor.byte_length)
            .ok_or_else(|| corrupt("directory tensor range overflows"))?;
        let shard = shards
            .get(usize::try_from(tensor.shard_index).map_err(|_| {
                EngineError::resource_exhaustion("tensor shard index is not addressable")
            })?)
            .ok_or_else(|| corrupt("tensor references an unavailable shard"))?;
        if end > shard.byte_length {
            return Err(corrupt("directory tensor range exceeds its shard"));
        }
        ranges.entry(tensor.shard_index).or_default().push((tensor.byte_offset, end));
    }
    for shard_ranges in ranges.values_mut() {
        shard_ranges.sort_unstable();
        if shard_ranges.windows(2).any(|pair| pair[0].1 > pair[1].0) {
            return Err(corrupt("directory tensor ranges overlap"));
        }
    }
    Ok(())
}

fn validate_quantization_profile(
    storage_dtype: DType,
    profile: Quantization,
    tensors: &[DirectoryTensorRecord],
) -> Result<(), EngineError> {
    match profile {
        Quantization::None => {
            if tensors.iter().any(|tensor| tensor.quantization() != Quantization::None) {
                return Err(corrupt(
                    "unquantized manifest contains a quantized tensor declaration",
                ));
            }
        }
        Quantization::AipQ8V1 | Quantization::AipQ4V1 => {
            if storage_dtype != DType::F32 {
                return Err(corrupt("quantized manifest logical dtype must be F32"));
            }
            let mut observed_quantized = false;
            for tensor in tensors {
                match tensor.quantization() {
                    Quantization::None => {}
                    value if value == profile => observed_quantized = true,
                    _ => {
                        return Err(corrupt(
                            "manifest contains unsupported mixed quantization formats",
                        ));
                    }
                }
            }
            if !observed_quantized {
                return Err(corrupt(
                    "quantized manifest contains no quantized tensor declarations",
                ));
            }
        }
    }
    Ok(())
}

fn manifest_length(shard_count: u32, tensor_count: u32) -> Result<usize, EngineError> {
    let shard_bytes = usize::try_from(shard_count)
        .ok()
        .and_then(|count| count.checked_mul(DIRECTORY_SHARD_RECORD_BYTES))
        .ok_or_else(|| EngineError::resource_exhaustion("shard table length overflows"))?;
    let tensor_bytes = usize::try_from(tensor_count)
        .ok()
        .and_then(|count| count.checked_mul(DIRECTORY_TENSOR_RECORD_BYTES))
        .ok_or_else(|| EngineError::resource_exhaustion("tensor table length overflows"))?;
    DIRECTORY_MANIFEST_HEADER_BYTES
        .checked_add(shard_bytes)
        .and_then(|value| value.checked_add(tensor_bytes))
        .filter(|length| *length <= DIRECTORY_MANIFEST_MAX_BYTES)
        .ok_or_else(|| EngineError::resource_exhaustion("directory manifest length exceeds limit"))
}

fn validate_tensor_name(name: &str) -> Result<(), EngineError> {
    if name.is_empty()
        || name.len() > MAX_DIRECTORY_TENSOR_NAME_BYTES
        || !name.is_ascii()
        || name.contains("..")
        || !name
            .as_bytes()
            .iter()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_'))
    {
        return Err(corrupt("canonical tensor name is invalid"));
    }
    Ok(())
}

fn parse_padded_text(
    bytes: &[u8],
    offset: usize,
    capacity: usize,
    length: usize,
    label: &str,
) -> Result<String, EngineError> {
    if length == 0 || length > capacity {
        return Err(corrupt("directory manifest text length is invalid"));
    }
    let field = record(bytes, offset, capacity)?;
    if field[length..].iter().any(|byte| *byte != 0) {
        return Err(corrupt("directory manifest text padding is nonzero"));
    }
    let text = std::str::from_utf8(&field[..length])
        .map_err(|_| corrupt("directory manifest text is not valid UTF-8"))?;
    if text.chars().any(char::is_control) {
        return Err(corrupt("directory manifest text contains control data"));
    }
    let _ = label;
    Ok(text.to_owned())
}

fn write_padded_text(
    bytes: &mut [u8],
    offset: usize,
    capacity: usize,
    text: &str,
) -> Result<(), EngineError> {
    if text.is_empty() || text.len() > capacity {
        return Err(corrupt("directory manifest text exceeds fixed field"));
    }
    write_exact(bytes, offset, text.as_bytes())
}

fn encode_dtype(dtype: DType) -> Result<u8, EngineError> {
    match dtype {
        DType::F32 => Ok(1),
        DType::F16 => Ok(2),
        DType::BFloat16 => Ok(3),
        DType::I8 => Ok(4),
        DType::U8 => Ok(5),
        DType::I16 => Ok(6),
        DType::U16 => Ok(7),
        DType::I32 => Ok(8),
        DType::U32 => Ok(9),
        DType::I64 => Ok(10),
        DType::U64 => Ok(11),
        DType::Bool => Ok(12),
        _ => Err(corrupt("directory tensor dtype is unsupported")),
    }
}

fn decode_dtype(value: u8) -> Result<DType, EngineError> {
    match value {
        1 => Ok(DType::F32),
        2 => Ok(DType::F16),
        3 => Ok(DType::BFloat16),
        4 => Ok(DType::I8),
        5 => Ok(DType::U8),
        6 => Ok(DType::I16),
        7 => Ok(DType::U16),
        8 => Ok(DType::I32),
        9 => Ok(DType::U32),
        10 => Ok(DType::I64),
        11 => Ok(DType::U64),
        12 => Ok(DType::Bool),
        _ => Err(corrupt("directory tensor dtype is invalid")),
    }
}

fn read_nonzero_u32(bytes: &[u8], offset: usize, label: &str) -> Result<u32, EngineError> {
    let value = read_u32(bytes, offset)?;
    if value == 0 {
        return Err(corrupt(label));
    }
    Ok(value)
}

fn read_u16(bytes: &[u8], offset: usize) -> Result<u16, EngineError> {
    Ok(u16::from_le_bytes(
        record(bytes, offset, 2)?.try_into().map_err(|_| corrupt("u16 field is truncated"))?,
    ))
}

fn read_u32(bytes: &[u8], offset: usize) -> Result<u32, EngineError> {
    Ok(u32::from_le_bytes(
        record(bytes, offset, 4)?.try_into().map_err(|_| corrupt("u32 field is truncated"))?,
    ))
}

fn read_u64(bytes: &[u8], offset: usize) -> Result<u64, EngineError> {
    Ok(u64::from_le_bytes(
        record(bytes, offset, 8)?.try_into().map_err(|_| corrupt("u64 field is truncated"))?,
    ))
}

fn write_u16(bytes: &mut [u8], offset: usize, value: u16) -> Result<(), EngineError> {
    write_exact(bytes, offset, &value.to_le_bytes())
}

fn write_u32(bytes: &mut [u8], offset: usize, value: u32) -> Result<(), EngineError> {
    write_exact(bytes, offset, &value.to_le_bytes())
}

fn write_u64(bytes: &mut [u8], offset: usize, value: u64) -> Result<(), EngineError> {
    write_exact(bytes, offset, &value.to_le_bytes())
}

fn write_exact(bytes: &mut [u8], offset: usize, value: &[u8]) -> Result<(), EngineError> {
    let end = checked_add_usize(offset, value.len())?;
    bytes
        .get_mut(offset..end)
        .ok_or_else(|| corrupt("directory manifest write range is unavailable"))?
        .copy_from_slice(value);
    Ok(())
}

fn record(bytes: &[u8], offset: usize, length: usize) -> Result<&[u8], EngineError> {
    let end = checked_add_usize(offset, length)?;
    bytes.get(offset..end).ok_or_else(|| corrupt("directory manifest field is truncated"))
}

fn checked_add_usize(left: usize, right: usize) -> Result<usize, EngineError> {
    left.checked_add(right)
        .ok_or_else(|| EngineError::resource_exhaustion("directory manifest offset overflows"))
}

fn usize_to_u32(value: usize) -> Result<u32, EngineError> {
    u32::try_from(value)
        .map_err(|_| EngineError::resource_exhaustion("directory size exceeds its field"))
}

fn usize_to_u64(value: usize) -> Result<u64, EngineError> {
    u64::try_from(value)
        .map_err(|_| EngineError::resource_exhaustion("directory size is not representable"))
}

fn zeroed(length: usize) -> Result<Vec<u8>, EngineError> {
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(length)
        .map_err(|_| EngineError::resource_exhaustion("directory manifest allocation failed"))?;
    bytes.resize(length, 0);
    Ok(bytes)
}

fn corrupt(detail: impl AsRef<str>) -> EngineError {
    EngineError::corrupt_model_package(detail)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hash(byte: u8) -> ModelPackageHash {
        ModelPackageHash::from_sha384_bytes([byte; ModelPackageHash::BYTE_LEN])
    }

    fn fixture() -> DirectoryManifest {
        let tensor = DirectoryTensorRecord::new(
            "token_embedding.weight",
            DType::F32,
            TensorShape::new([4, 4]).expect("shape must validate"),
            0,
            0,
            64,
            64,
            Quantization::None,
            hash(3),
        )
        .expect("tensor must validate");
        DirectoryManifestBuilder::new(
            ModelId::new("synthetic-model").expect("model ID must validate"),
            (0, 1, 0),
            ContextLimit::new(128).expect("context must validate"),
            256,
            4,
            1,
            1,
            1,
            8,
            10_000.0,
            1.0,
            DType::F32,
            Quantization::None,
            false,
            hash(1),
            vec![DirectoryShardRecord::new(0, 64, hash(2)).expect("shard must validate")],
            vec![tensor],
        )
        .expect("builder must validate")
        .manifest()
        .clone()
    }

    fn qwen2_fixture() -> DirectoryManifest {
        let tensor = DirectoryTensorRecord::new(
            "token_embedding.weight",
            DType::F32,
            TensorShape::new([4, 4]).expect("shape must validate"),
            0,
            0,
            64,
            64,
            Quantization::None,
            hash(3),
        )
        .expect("tensor must validate");
        DirectoryManifestBuilder::new_qwen2(
            ModelId::new("synthetic-qwen2").expect("model ID must validate"),
            DIRECTORY_QWEN2_MINIMUM_ENGINE_VERSION,
            ContextLimit::new(128).expect("context must validate"),
            256,
            4,
            1,
            1,
            1,
            8,
            1.0e-6,
            1_000_000.0,
            DType::F32,
            Quantization::None,
            hash(1),
            vec![DirectoryShardRecord::new(0, 64, hash(2)).expect("shard must validate")],
            vec![tensor],
        )
        .expect("Qwen2.5 builder must validate")
        .manifest()
        .clone()
    }

    #[test]
    fn directory_manifest_round_trip_is_canonical() {
        let manifest = fixture();
        let encoded = manifest.encode().expect("manifest must encode");
        assert_eq!(DirectoryManifest::parse(&encoded).expect("manifest must parse"), manifest);
        assert_eq!(manifest.encode().expect("repeat must encode"), encoded);
        assert_eq!(manifest.format_version(), DIRECTORY_MANIFEST_VERSION);
        assert_eq!(read_u32(&encoded, 84).expect("reserved field must read"), 0);
        assert_eq!(manifest.rms_norm_epsilon().to_bits(), LEGACY_RMS_NORM_EPSILON.to_bits());
    }

    #[test]
    fn qwen2_manifest_profile_is_versioned_and_fail_closed() {
        let manifest = qwen2_fixture();
        let encoded = manifest.encode().expect("Qwen2.5 manifest must encode");
        assert_eq!(manifest.format_version(), DIRECTORY_QWEN2_MANIFEST_VERSION);
        assert_eq!(manifest.architecture_id().as_str(), DIRECTORY_QWEN2_DECODER_ARCHITECTURE_ID);
        assert!(manifest.tied_output());
        assert_eq!(manifest.rms_norm_epsilon().to_bits(), 1.0e-6_f32.to_bits());
        assert_eq!(manifest.rope_scaling().to_bits(), 1.0_f32.to_bits());
        assert_eq!(
            DirectoryManifest::parse(&encoded).expect("Qwen2.5 manifest must parse"),
            manifest
        );

        let mut legacy_version = encoded.clone();
        legacy_version[8..10].copy_from_slice(&DIRECTORY_MANIFEST_VERSION.0.to_le_bytes());
        legacy_version[10..12].copy_from_slice(&DIRECTORY_MANIFEST_VERSION.1.to_le_bytes());
        assert!(DirectoryManifest::parse(&legacy_version).is_err());

        let mut untied = encoded.clone();
        untied[20..24].copy_from_slice(&0_u32.to_le_bytes());
        assert!(DirectoryManifest::parse(&untied).is_err());

        let mut predating_engine = encoded.clone();
        predating_engine[24..26].copy_from_slice(&0_u16.to_le_bytes());
        predating_engine[26..28].copy_from_slice(&1_u16.to_le_bytes());
        predating_engine[28..30].copy_from_slice(&0_u16.to_le_bytes());
        assert!(DirectoryManifest::parse(&predating_engine).is_err());

        let mut scaled = encoded.clone();
        scaled[80..84].copy_from_slice(&2.0_f32.to_bits().to_le_bytes());
        assert!(DirectoryManifest::parse(&scaled).is_err());

        for invalid_epsilon in [0.0_f32, f32::NAN, f32::INFINITY] {
            let mut invalid = encoded.clone();
            invalid[84..88].copy_from_slice(&invalid_epsilon.to_bits().to_le_bytes());
            assert!(DirectoryManifest::parse(&invalid).is_err());
        }
    }

    #[test]
    fn quantized_manifest_accepts_only_canonical_profile_and_lengths() {
        let matrix_shape = TensorShape::new([4, 4]).expect("matrix shape must validate");
        let matrix = DirectoryTensorRecord::new(
            "matrix.weight",
            DType::F32,
            matrix_shape.clone(),
            0,
            0,
            Quantization::AipQ8V1
                .payload_len(matrix_shape.element_count())
                .expect("Q8 length must calculate"),
            AIPQ_REQUIRED_ALIGNMENT,
            Quantization::AipQ8V1,
            hash(3),
        )
        .expect("Q8 tensor must validate");
        let norm = DirectoryTensorRecord::new(
            "norm.weight",
            DType::F32,
            TensorShape::new([4]).expect("norm shape must validate"),
            0,
            128,
            16,
            64,
            Quantization::None,
            hash(4),
        )
        .expect("F32 tensor must validate");
        let manifest = DirectoryManifestBuilder::new(
            ModelId::new("synthetic-quantized").expect("model ID must validate"),
            (0, 1, 0),
            ContextLimit::new(128).expect("context must validate"),
            256,
            4,
            1,
            1,
            1,
            8,
            10_000.0,
            1.0,
            DType::F32,
            Quantization::AipQ8V1,
            false,
            hash(1),
            vec![DirectoryShardRecord::new(0, 192, hash(2)).expect("shard must validate")],
            vec![matrix, norm],
        )
        .expect("Q8 manifest must validate")
        .build_bytes()
        .expect("Q8 manifest must encode");
        let parsed = DirectoryManifest::parse(&manifest).expect("Q8 manifest must parse");
        assert_eq!(parsed.quantization(), Quantization::AipQ8V1);

        let mut mixed = manifest.clone();
        mixed[45] = Quantization::AipQ4V1.id();
        assert!(DirectoryManifest::parse(&mixed).is_err());
        assert!(
            DirectoryTensorRecord::new(
                "bad.weight",
                DType::F32,
                matrix_shape,
                0,
                0,
                64,
                64,
                Quantization::AipQ8V1,
                hash(5),
            )
            .is_err()
        );
    }

    #[test]
    fn directory_manifest_rejects_core_malformed_cases() {
        let encoded = fixture().encode().expect("manifest must encode");
        for end in 0..DIRECTORY_MANIFEST_HEADER_BYTES {
            assert!(DirectoryManifest::parse(&encoded[..end]).is_err());
        }
        for (offset, value) in [(0, 0_u8), (8, 2), (12, 0), (44, 0xff), (144, 0xff)] {
            let mut bad = encoded.clone();
            bad[offset] = value;
            assert!(DirectoryManifest::parse(&bad).is_err(), "offset {offset} was accepted");
        }
        let mut trailing = encoded.clone();
        trailing.push(0);
        assert!(DirectoryManifest::parse(&trailing).is_err());
    }

    #[test]
    fn directory_manifest_rejects_each_bounded_parser_failure_class() {
        let encoded = fixture().encode().expect("manifest must encode");
        let tensor = DIRECTORY_MANIFEST_HEADER_BYTES + DIRECTORY_SHARD_RECORD_BYTES;

        let mut unsupported_version = encoded.clone();
        unsupported_version[8..10].copy_from_slice(&2_u16.to_le_bytes());
        assert!(DirectoryManifest::parse(&unsupported_version).is_err());

        let mut excessive_count = encoded.clone();
        excessive_count[36..40].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(DirectoryManifest::parse(&excessive_count).is_err());

        let mut excessive_rank = encoded.clone();
        excessive_rank[tensor + 3] = u8::try_from(MAX_TENSOR_RANK + 1).expect("rank must fit");
        assert!(DirectoryManifest::parse(&excessive_rank).is_err());

        let mut excessive_dimension = encoded.clone();
        excessive_dimension[tensor + 16..tensor + 24]
            .copy_from_slice(&(MAX_DIMENSION_SIZE + 1).to_le_bytes());
        assert!(DirectoryManifest::parse(&excessive_dimension).is_err());

        let mut overflowing_range = encoded.clone();
        overflowing_range[tensor + 80..tensor + 88].copy_from_slice(&(u64::MAX - 63).to_le_bytes());
        assert!(DirectoryManifest::parse(&overflowing_range).is_err());

        let mut out_of_bounds = encoded.clone();
        out_of_bounds[DIRECTORY_MANIFEST_HEADER_BYTES + 8..DIRECTORY_MANIFEST_HEADER_BYTES + 16]
            .copy_from_slice(&63_u64.to_le_bytes());
        assert!(DirectoryManifest::parse(&out_of_bounds).is_err());

        let mut invalid_dtype = encoded.clone();
        invalid_dtype[tensor + 2] = u8::MAX;
        assert!(DirectoryManifest::parse(&invalid_dtype).is_err());

        let mut invalid_architecture = encoded.clone();
        invalid_architecture[144] = b'x';
        assert!(matches!(
            DirectoryManifest::parse(&invalid_architecture),
            Err(EngineError::UnsupportedArchitecture(_))
        ));

        let mut invalid_utf8 = encoded.clone();
        invalid_utf8[144] = u8::MAX;
        assert!(DirectoryManifest::parse(&invalid_utf8).is_err());

        let mut unsupported_optional = encoded.clone();
        unsupported_optional[84..88].copy_from_slice(&1_u32.to_le_bytes());
        assert!(matches!(
            DirectoryManifest::parse(&unsupported_optional),
            Err(EngineError::NotImplemented(_))
        ));

        let mut trailing = encoded;
        trailing.push(0);
        assert!(DirectoryManifest::parse(&trailing).is_err());
    }

    #[test]
    fn directory_manifest_rejects_duplicate_names_and_overlapping_ranges_on_wire() {
        let encoded = fixture().encode().expect("manifest must encode");
        let tensor = DIRECTORY_MANIFEST_HEADER_BYTES + DIRECTORY_SHARD_RECORD_BYTES;
        let mut duplicate = encoded.clone();
        duplicate[36..40].copy_from_slice(&2_u32.to_le_bytes());
        let new_length = duplicate.len() + DIRECTORY_TENSOR_RECORD_BYTES;
        duplicate[88..96]
            .copy_from_slice(&u64::try_from(new_length).expect("length must fit").to_le_bytes());
        duplicate.extend_from_slice(&encoded[tensor..tensor + DIRECTORY_TENSOR_RECORD_BYTES]);
        assert!(DirectoryManifest::parse(&duplicate).is_err());

        let mut overlapping = duplicate;
        let second = tensor + DIRECTORY_TENSOR_RECORD_BYTES;
        let replacement = b"token_embedding.weighu";
        assert_eq!(replacement.len(), "token_embedding.weight".len());
        overlapping[second + 144..second + 144 + replacement.len()].copy_from_slice(replacement);
        assert!(DirectoryManifest::parse(&overlapping).is_err());
    }

    #[test]
    fn directory_manifest_rejects_duplicate_and_overlapping_tensors() {
        let manifest = fixture();
        let tensor = manifest.tensors()[0].clone();
        assert!(
            DirectoryManifestBuilder::new(
                manifest.model_id().clone(),
                manifest.required_engine_version(),
                manifest.context_limit(),
                manifest.vocabulary_size(),
                manifest.hidden_dimension(),
                manifest.layer_count(),
                manifest.attention_head_count(),
                manifest.kv_head_count(),
                manifest.feed_forward_dimension(),
                manifest.rope_theta(),
                manifest.rope_scaling(),
                manifest.storage_dtype(),
                manifest.quantization(),
                manifest.tied_output(),
                manifest.tokenizer_hash(),
                manifest.shards().to_vec(),
                vec![tensor.clone(), tensor],
            )
            .is_err()
        );
    }
}
