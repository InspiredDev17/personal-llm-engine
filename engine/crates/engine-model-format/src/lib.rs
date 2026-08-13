//! Canonical parser for the first-party signed, data-only model package.
//!
//! The format is fixed-width and non-recursive. It contains a 64-byte package
//! header, one 256-byte manifest, fixed 128-byte tensor records, bounded
//! tokenizer and tensor data, and one fixed 96-byte Ed25519 signature envelope.
//! No package field can name code, a library, a kernel, or a filesystem path.

mod directory;
mod hash_catalog;
mod registry;

pub use directory::{
    DIRECTORY_DECODER_ARCHITECTURE_ID, DIRECTORY_MANIFEST_HEADER_BYTES, DIRECTORY_MANIFEST_MAGIC,
    DIRECTORY_MANIFEST_MAX_BYTES, DIRECTORY_MANIFEST_VERSION, DIRECTORY_SHARD_RECORD_BYTES,
    DIRECTORY_TENSOR_RECORD_BYTES, DirectoryManifest, DirectoryManifestBuilder,
    DirectoryShardRecord, DirectoryTensorRecord, Quantization,
};
pub use hash_catalog::{
    HASH_CATALOG_MAGIC, HASH_CATALOG_MAX_ENTRIES, HASH_CATALOG_RECORD_BYTES, HashCatalog,
    HashCatalogEntry, PackageFileKind,
};
pub use registry::{
    DIRECTORY_ENGINE_COMPATIBILITY, MODEL_REGISTRY_MAX_BYTES,
    MODEL_REGISTRY_MINIMUM_ENGINE_VERSION, MODEL_REGISTRY_SCHEMA_V2, ModelRegistryEntryV2,
    SignedModelRegistryEntryV2,
};

use personalai_engine_types::{
    ArchitectureId, ContextLimit, DType, EngineError, MAX_MODEL_PACKAGE_BYTES, MAX_TENSOR_COUNT,
    MAX_TENSOR_RANK, MAX_TOKENIZER_DATA_BYTES, ModelPackageHash, SignerKeyId, TensorShape,
};
use std::ops::Range;

/// Magic bytes at the start of every first-party model package.
pub const MODEL_PACKAGE_MAGIC: [u8; 8] = *b"PAIMDL\0\x01";
/// Magic bytes at the start of the canonical manifest.
pub const MODEL_MANIFEST_MAGIC: [u8; 8] = *b"PAIMAN01";
/// Magic bytes at the start of the signature envelope.
pub const SIGNATURE_ENVELOPE_MAGIC: [u8; 8] = *b"PAISIG01";
/// Number of bytes in the version-one fixed header.
pub const MODEL_PACKAGE_HEADER_BYTES: usize = 64;
/// Exact version-one manifest byte length.
pub const MODEL_MANIFEST_BYTES: usize = 256;
/// Fixed size of one tensor-index record.
pub const TENSOR_INDEX_ENTRY_BYTES: u64 = 128;
/// Exact version-one signature-envelope length.
pub const SIGNATURE_ENVELOPE_BYTES: usize = 96;
/// Compatibility alias for the exact manifest ceiling.
pub const MAX_MANIFEST_BYTES: u64 = 256;
/// Compatibility alias for the exact signature-envelope ceiling.
pub const MAX_SIGNATURE_BYTES: u32 = 96;
/// Required tensor-data alignment.
pub const TENSOR_DATA_ALIGNMENT: u64 = 64;
/// Maximum encoded tensor name, excluding zero padding.
pub const MAX_TENSOR_NAME_BYTES: usize = 31;

const CAPABILITY_TEXT_GENERATION: u32 = 1;
const CAPABILITY_STREAMING: u32 = 1 << 1;
const KNOWN_CAPABILITIES: u32 = CAPABILITY_TEXT_GENERATION | CAPABILITY_STREAMING;

/// Supported first-party package format version.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PackageFormatVersion {
    major: u16,
    minor: u16,
}

impl PackageFormatVersion {
    /// Version supported by this parser.
    pub const V1: Self = Self { major: 1, minor: 0 };

    /// Returns the major version.
    pub const fn major(self) -> u16 {
        self.major
    }

    /// Returns the minor version.
    pub const fn minor(self) -> u16 {
        self.minor
    }
}

/// Validated layout metadata from a first-party model package.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelPackageHeader {
    version: PackageFormatVersion,
    package_length: u64,
    manifest_offset: u64,
    manifest_length: u64,
    tensor_table_offset: u64,
    tensor_count: u32,
    signature_offset: u64,
    signature_length: u32,
}

impl ModelPackageHeader {
    /// Parses and validates one fixed-width package header.
    pub fn parse(input: &[u8]) -> Result<Self, EngineError> {
        if input.len() < MODEL_PACKAGE_HEADER_BYTES {
            return Err(corrupt("model package header is truncated"));
        }
        if input[..MODEL_PACKAGE_MAGIC.len()] != MODEL_PACKAGE_MAGIC {
            return Err(corrupt("model package magic is invalid"));
        }
        let version =
            PackageFormatVersion { major: read_u16(input, 8)?, minor: read_u16(input, 10)? };
        if version != PackageFormatVersion::V1 {
            return Err(corrupt("model package format version is unsupported"));
        }
        if read_u32(input, 12)? != usize_to_u32(MODEL_PACKAGE_HEADER_BYTES)? {
            return Err(corrupt("model package header length is unsupported"));
        }
        let header = Self {
            version,
            package_length: read_u64(input, 16)?,
            manifest_offset: read_u64(input, 24)?,
            manifest_length: read_u64(input, 32)?,
            tensor_table_offset: read_u64(input, 40)?,
            tensor_count: read_u32(input, 48)?,
            signature_length: read_u32(input, 52)?,
            signature_offset: read_u64(input, 56)?,
        };
        header.validate()?;
        Ok(header)
    }

    /// Returns the package format version.
    pub const fn version(&self) -> PackageFormatVersion {
        self.version
    }

    /// Returns the exact declared package byte length.
    pub const fn package_length(&self) -> u64 {
        self.package_length
    }

    /// Returns the manifest byte range as `(offset, length)`.
    pub const fn manifest_range(&self) -> (u64, u64) {
        (self.manifest_offset, self.manifest_length)
    }

    /// Returns the tensor-table offset.
    pub const fn tensor_table_offset(&self) -> u64 {
        self.tensor_table_offset
    }

    /// Returns the declared tensor count.
    pub const fn tensor_count(&self) -> u32 {
        self.tensor_count
    }

    /// Returns the signature-envelope byte range as `(offset, length)`.
    pub const fn signature_range(&self) -> (u64, u32) {
        (self.signature_offset, self.signature_length)
    }

    /// Returns the byte range covered by the content signature.
    pub fn signed_content_range(&self) -> Result<Range<usize>, EngineError> {
        Ok(0..u64_to_usize(self.signature_offset)?)
    }

    fn validate(&self) -> Result<(), EngineError> {
        let header_bytes = usize_to_u64(MODEL_PACKAGE_HEADER_BYTES)?;
        let manifest_bytes = usize_to_u64(MODEL_MANIFEST_BYTES)?;
        if self.package_length < header_bytes || self.package_length > MAX_MODEL_PACKAGE_BYTES {
            return Err(corrupt("declared model package length is outside hard bounds"));
        }
        if self.manifest_offset != header_bytes || self.manifest_length != manifest_bytes {
            return Err(corrupt("manifest range is not canonical"));
        }
        if self.tensor_count == 0 || self.tensor_count > MAX_TENSOR_COUNT {
            return Err(corrupt("tensor count is outside hard bounds"));
        }
        let manifest_end = checked_end(self.manifest_offset, self.manifest_length)?;
        if self.tensor_table_offset != manifest_end {
            return Err(corrupt("tensor table does not immediately follow the manifest"));
        }
        let table_length = u64::from(self.tensor_count)
            .checked_mul(TENSOR_INDEX_ENTRY_BYTES)
            .ok_or_else(|| corrupt("tensor table byte length overflows"))?;
        let table_end = checked_end(self.tensor_table_offset, table_length)?;
        if self.signature_offset < table_end {
            return Err(corrupt("signature envelope overlaps package metadata"));
        }
        if self.signature_length != MAX_SIGNATURE_BYTES {
            return Err(corrupt("signature envelope length is not canonical"));
        }
        if checked_end(self.signature_offset, u64::from(self.signature_length))?
            != self.package_length
        {
            return Err(corrupt("signature envelope must terminate the package"));
        }
        Ok(())
    }
}

/// Declarative tokenizer-data representation selected by the manifest.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenizerDataFormat {
    /// First-party AIPTok Byte-Level BPE v1 data.
    AipTokByteLevelBpeV1,
}

/// Validated canonical model manifest.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelManifest {
    architecture_id: ArchitectureId,
    layer_count: u32,
    hidden_dimension: u32,
    attention_head_count: u32,
    kv_head_count: u32,
    vocabulary_size: u32,
    context_limit: ContextLimit,
    tokenizer_format: TokenizerDataFormat,
    storage_dtype: DType,
    tensor_count: u32,
    tokenizer_offset: u64,
    tokenizer_length: u64,
    weights_offset: u64,
    weights_length: u64,
    source_hash: [u8; ModelPackageHash::BYTE_LEN],
    streaming: bool,
    intermediate_dimension: u32,
    rms_norm_epsilon: f32,
    rope_theta: f32,
}

impl ModelManifest {
    fn parse(input: &[u8], header: &ModelPackageHeader) -> Result<Self, EngineError> {
        if input.len() != MODEL_MANIFEST_BYTES || input[..8] != MODEL_MANIFEST_MAGIC {
            return Err(corrupt("model manifest magic or length is invalid"));
        }
        if read_u16(input, 8)? != 1 || read_u16(input, 10)? != 0 {
            return Err(corrupt("model manifest version is unsupported"));
        }
        let capabilities = read_u32(input, 12)?;
        if capabilities & !KNOWN_CAPABILITIES != 0 || capabilities & CAPABILITY_TEXT_GENERATION == 0
        {
            return Err(corrupt("model manifest capabilities are invalid"));
        }
        let architecture_id = parse_architecture_id(&input[16..80])?;
        let layer_count = read_nonzero_u32(input, 80, "layer count")?;
        let hidden_dimension = read_nonzero_u32(input, 84, "hidden dimension")?;
        let attention_head_count = read_nonzero_u32(input, 88, "attention head count")?;
        let kv_head_count = read_nonzero_u32(input, 92, "key/value head count")?;
        let vocabulary_size = read_nonzero_u32(input, 96, "vocabulary size")?;
        let context_limit = ContextLimit::new(read_u32(input, 100)?)
            .map_err(|_| corrupt("model manifest context limit is invalid"))?;
        let tokenizer_format = match read_u16(input, 104)? {
            1 => TokenizerDataFormat::AipTokByteLevelBpeV1,
            _ => return Err(corrupt("model manifest tokenizer format is unsupported")),
        };
        let storage_dtype = decode_dtype(input[106])?;
        if input[107] != 0 {
            return Err(corrupt("model manifest reserved byte is nonzero"));
        }
        let tensor_count = read_u32(input, 108)?;
        if tensor_count != header.tensor_count() {
            return Err(corrupt("manifest and header tensor counts differ"));
        }
        let tokenizer_offset = read_u64(input, 112)?;
        let tokenizer_length = read_u64(input, 120)?;
        let weights_offset = read_u64(input, 128)?;
        let weights_length = read_u64(input, 136)?;
        if tokenizer_length == 0 || tokenizer_length > MAX_TOKENIZER_DATA_BYTES {
            return Err(corrupt("tokenizer data length is outside hard bounds"));
        }
        let table_length = u64::from(tensor_count)
            .checked_mul(TENSOR_INDEX_ENTRY_BYTES)
            .ok_or_else(|| corrupt("tensor table length overflows"))?;
        let table_end = checked_end(header.tensor_table_offset(), table_length)?;
        if tokenizer_offset != table_end {
            return Err(corrupt("tokenizer data does not immediately follow the tensor table"));
        }
        let tokenizer_end = checked_end(tokenizer_offset, tokenizer_length)?;
        if weights_offset != align_up(tokenizer_end, TENSOR_DATA_ALIGNMENT)? {
            return Err(corrupt("weights offset is not the canonical aligned value"));
        }
        if weights_length == 0
            || checked_end(weights_offset, weights_length)? != header.signature_range().0
        {
            return Err(corrupt("weights range does not terminate at the signature envelope"));
        }
        let mut source_hash = [0_u8; ModelPackageHash::BYTE_LEN];
        source_hash.copy_from_slice(&input[144..192]);
        if source_hash == [0_u8; ModelPackageHash::BYTE_LEN] {
            return Err(corrupt("model source identity hash must not be all zero"));
        }
        let intermediate_dimension = read_nonzero_u32(input, 192, "intermediate dimension")?;
        let rms_norm_epsilon = f32::from_bits(read_u32(input, 196)?);
        let rope_theta = f32::from_bits(read_u32(input, 200)?);
        if !rms_norm_epsilon.is_finite()
            || rms_norm_epsilon <= 0.0
            || !rope_theta.is_finite()
            || rope_theta <= 1.0
        {
            return Err(corrupt("model numeric hyperparameters are invalid"));
        }
        if input[204..].iter().any(|byte| *byte != 0) {
            return Err(corrupt("model manifest reserved bytes are nonzero"));
        }
        Ok(Self {
            architecture_id,
            layer_count,
            hidden_dimension,
            attention_head_count,
            kv_head_count,
            vocabulary_size,
            context_limit,
            tokenizer_format,
            storage_dtype,
            tensor_count,
            tokenizer_offset,
            tokenizer_length,
            weights_offset,
            weights_length,
            source_hash,
            streaming: capabilities & CAPABILITY_STREAMING != 0,
            intermediate_dimension,
            rms_norm_epsilon,
            rope_theta,
        })
    }

    /// Returns the exact supported architecture identifier.
    pub const fn architecture_id(&self) -> &ArchitectureId {
        &self.architecture_id
    }

    /// Returns the transformer layer count.
    pub const fn layer_count(&self) -> u32 {
        self.layer_count
    }

    /// Returns the hidden dimension.
    pub const fn hidden_dimension(&self) -> u32 {
        self.hidden_dimension
    }

    /// Returns the attention head count.
    pub const fn attention_head_count(&self) -> u32 {
        self.attention_head_count
    }

    /// Returns the key/value head count.
    pub const fn kv_head_count(&self) -> u32 {
        self.kv_head_count
    }

    /// Returns the tokenizer vocabulary size.
    pub const fn vocabulary_size(&self) -> u32 {
        self.vocabulary_size
    }

    /// Returns the model context limit.
    pub const fn context_limit(&self) -> ContextLimit {
        self.context_limit
    }

    /// Returns the tokenizer data format.
    pub const fn tokenizer_format(&self) -> TokenizerDataFormat {
        self.tokenizer_format
    }

    /// Returns the declared tensor storage type.
    pub const fn storage_dtype(&self) -> DType {
        self.storage_dtype
    }

    /// Returns the tensor count cross-checked against the package header.
    pub const fn tensor_count(&self) -> u32 {
        self.tensor_count
    }

    /// Returns whether streaming generation is declared.
    pub const fn supports_streaming(&self) -> bool {
        self.streaming
    }

    /// Returns the feed-forward intermediate dimension.
    pub const fn intermediate_dimension(&self) -> u32 {
        self.intermediate_dimension
    }

    /// Returns the RMS normalization epsilon.
    pub const fn rms_norm_epsilon(&self) -> f32 {
        self.rms_norm_epsilon
    }

    /// Returns the rotary-position base theta.
    pub const fn rope_theta(&self) -> f32 {
        self.rope_theta
    }

    /// Returns the tokenizer-data byte range.
    pub fn tokenizer_range(&self) -> Result<Range<usize>, EngineError> {
        usize_range(self.tokenizer_offset, self.tokenizer_length)
    }

    /// Returns the tensor-weight byte range.
    pub fn weights_range(&self) -> Result<Range<usize>, EngineError> {
        usize_range(self.weights_offset, self.weights_length)
    }

    /// Returns the declared source-artifact SHA-384 digest.
    pub const fn source_hash(&self) -> &[u8; ModelPackageHash::BYTE_LEN] {
        &self.source_hash
    }
}

/// Validated path-free tensor name.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TensorName(String);

impl TensorName {
    /// Returns the canonical tensor name.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// One validated tensor index entry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorRecord {
    name: TensorName,
    dtype: DType,
    shape: TensorShape,
    data_offset: u64,
    data_length: u64,
}

impl TensorRecord {
    fn parse(input: &[u8], weights: &Range<usize>) -> Result<Self, EngineError> {
        if input.len() != usize::try_from(TENSOR_INDEX_ENTRY_BYTES).unwrap_or(usize::MAX) {
            return Err(corrupt("tensor index record length is invalid"));
        }
        let name = parse_tensor_name(&input[..32])?;
        let dtype = decode_dtype(input[32])?;
        let rank = usize::from(input[33]);
        if rank > MAX_TENSOR_RANK || input[34..40].iter().any(|byte| *byte != 0) {
            return Err(corrupt("tensor rank or reserved bytes are invalid"));
        }
        let mut dimensions = Vec::new();
        dimensions
            .try_reserve_exact(rank)
            .map_err(|_| EngineError::resource_exhaustion("tensor shape allocation failed"))?;
        for index in 0..MAX_TENSOR_RANK {
            let dimension = read_u64(input, 40 + index * 8)?;
            if index < rank {
                dimensions.push(dimension);
            } else if dimension != 0 {
                return Err(corrupt("unused tensor dimensions must be zero"));
            }
        }
        let shape = TensorShape::new(dimensions)
            .map_err(|_| corrupt("tensor index shape is outside hard bounds"))?;
        let data_offset = read_u64(input, 104)?;
        let data_length = read_u64(input, 112)?;
        if input[120..128].iter().any(|byte| *byte != 0) {
            return Err(corrupt("tensor index reserved bytes are nonzero"));
        }
        if data_offset % TENSOR_DATA_ALIGNMENT != 0 {
            return Err(corrupt("tensor data offset is not aligned"));
        }
        if data_length != shape.byte_len(dtype)? {
            return Err(corrupt("tensor data length does not match dtype and shape"));
        }
        let data_range = usize_range(data_offset, data_length)?;
        if data_range.start < weights.start || data_range.end > weights.end {
            return Err(corrupt("tensor data range is outside the weights section"));
        }
        Ok(Self { name, dtype, shape, data_offset, data_length })
    }

    /// Returns the canonical tensor name.
    pub const fn name(&self) -> &TensorName {
        &self.name
    }

    /// Returns the tensor storage type.
    pub const fn dtype(&self) -> DType {
        self.dtype
    }

    /// Returns the validated tensor shape.
    pub const fn shape(&self) -> &TensorShape {
        &self.shape
    }

    /// Returns the exact tensor data range.
    pub fn data_range(&self) -> Result<Range<usize>, EngineError> {
        usize_range(self.data_offset, self.data_length)
    }
}

/// Version-one package signature algorithm.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PackageSignatureAlgorithm {
    /// Strict Ed25519 verification.
    Ed25519,
}

/// Parsed fixed signature envelope.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SignatureEnvelope {
    algorithm: PackageSignatureAlgorithm,
    key_id: SignerKeyId,
    signature: [u8; 64],
}

impl SignatureEnvelope {
    /// Parses one exact canonical signature envelope.
    pub fn parse(input: &[u8]) -> Result<Self, EngineError> {
        if input.len() != SIGNATURE_ENVELOPE_BYTES || input[..8] != SIGNATURE_ENVELOPE_MAGIC {
            return Err(corrupt("signature envelope magic or length is invalid"));
        }
        if read_u16(input, 8)? != 1 {
            return Err(corrupt("signature envelope version is unsupported"));
        }
        let algorithm = match read_u16(input, 10)? {
            1 => PackageSignatureAlgorithm::Ed25519,
            _ => return Err(corrupt("signature algorithm is unsupported")),
        };
        let mut key_id = [0_u8; SignerKeyId::BYTE_LEN];
        key_id.copy_from_slice(&input[12..28]);
        let key_id = SignerKeyId::from_bytes(key_id)
            .map_err(|_| corrupt("signature key identifier is invalid"))?;
        let mut signature = [0_u8; 64];
        signature.copy_from_slice(&input[28..92]);
        if input[92..96].iter().any(|byte| *byte != 0) {
            return Err(corrupt("signature envelope reserved bytes are nonzero"));
        }
        Ok(Self { algorithm, key_id, signature })
    }

    /// Encodes one exact canonical Ed25519 signature envelope.
    pub fn encode_ed25519(
        key_id: SignerKeyId,
        signature: [u8; 64],
    ) -> Result<[u8; SIGNATURE_ENVELOPE_BYTES], EngineError> {
        let mut bytes = [0_u8; SIGNATURE_ENVELOPE_BYTES];
        bytes[..8].copy_from_slice(&SIGNATURE_ENVELOPE_MAGIC);
        bytes[8..10].copy_from_slice(&1_u16.to_le_bytes());
        bytes[10..12].copy_from_slice(&1_u16.to_le_bytes());
        bytes[12..28].copy_from_slice(key_id.as_bytes());
        bytes[28..92].copy_from_slice(&signature);
        Self::parse(&bytes)?;
        Ok(bytes)
    }

    /// Returns the signature algorithm.
    pub const fn algorithm(&self) -> PackageSignatureAlgorithm {
        self.algorithm
    }

    /// Returns the declared signer key identifier.
    pub const fn key_id(&self) -> SignerKeyId {
        self.key_id
    }

    /// Returns the exact Ed25519 signature bytes.
    pub const fn signature(&self) -> &[u8; 64] {
        &self.signature
    }
}

/// Fully parsed structural view of one canonical package.
#[derive(Debug, Clone, PartialEq)]
pub struct ParsedModelPackage {
    header: ModelPackageHeader,
    manifest: ModelManifest,
    tensors: Vec<TensorRecord>,
    signature: SignatureEnvelope,
}

impl ParsedModelPackage {
    /// Parses every canonical metadata structure and validates all data ranges.
    pub fn parse(package_bytes: &[u8]) -> Result<Self, EngineError> {
        let header = ModelPackageHeader::parse(package_bytes)?;
        if u64::try_from(package_bytes.len()).map_err(|_| corrupt("package length overflows"))?
            != header.package_length()
        {
            return Err(corrupt("actual package length differs from its header"));
        }
        let manifest_range = usize_range(header.manifest_offset, header.manifest_length)?;
        let manifest = ModelManifest::parse(&package_bytes[manifest_range], &header)?;
        let weights_range = manifest.weights_range()?;
        let table_start = u64_to_usize(header.tensor_table_offset)?;
        let record_bytes = u64_to_usize(TENSOR_INDEX_ENTRY_BYTES)?;
        let tensor_count = usize::try_from(header.tensor_count())
            .map_err(|_| corrupt("tensor count is not addressable"))?;
        let mut tensors = Vec::new();
        tensors
            .try_reserve_exact(tensor_count)
            .map_err(|_| EngineError::resource_exhaustion("tensor index allocation failed"))?;
        for index in 0..tensor_count {
            let offset = index
                .checked_mul(record_bytes)
                .and_then(|value| value.checked_add(table_start))
                .ok_or_else(|| corrupt("tensor record offset overflows"))?;
            let end = offset
                .checked_add(record_bytes)
                .ok_or_else(|| corrupt("tensor record end overflows"))?;
            let record = TensorRecord::parse(
                package_bytes
                    .get(offset..end)
                    .ok_or_else(|| corrupt("tensor table is truncated"))?,
                &weights_range,
            )?;
            tensors.push(record);
        }
        validate_canonical_tensor_order(package_bytes, &tensors, &weights_range)?;
        let tokenizer_range = manifest.tokenizer_range()?;
        let padding = tokenizer_range.end..weights_range.start;
        if package_bytes[padding].iter().any(|byte| *byte != 0) {
            return Err(corrupt("tokenizer alignment padding is nonzero"));
        }
        let signature_range =
            usize_range(header.signature_range().0, u64::from(header.signature_range().1))?;
        let signature = SignatureEnvelope::parse(&package_bytes[signature_range])?;
        Ok(Self { header, manifest, tensors, signature })
    }

    /// Returns the validated package header.
    pub const fn header(&self) -> &ModelPackageHeader {
        &self.header
    }

    /// Returns the validated manifest.
    pub const fn manifest(&self) -> &ModelManifest {
        &self.manifest
    }

    /// Returns the sorted, non-overlapping tensor index.
    pub fn tensors(&self) -> &[TensorRecord] {
        &self.tensors
    }

    /// Returns the parsed signature envelope.
    pub const fn signature(&self) -> &SignatureEnvelope {
        &self.signature
    }

    /// Returns the package tokenizer data without interpreting it.
    pub fn tokenizer_data<'a>(&self, package_bytes: &'a [u8]) -> Result<&'a [u8], EngineError> {
        package_bytes
            .get(self.manifest.tokenizer_range()?)
            .ok_or_else(|| corrupt("validated tokenizer range is unavailable"))
    }

    /// Returns one tensor's exact immutable data bytes.
    pub fn tensor_data<'a>(
        &self,
        package_bytes: &'a [u8],
        tensor: &TensorRecord,
    ) -> Result<&'a [u8], EngineError> {
        package_bytes
            .get(tensor.data_range()?)
            .ok_or_else(|| corrupt("validated tensor range is unavailable"))
    }
}

fn validate_canonical_tensor_order(
    package_bytes: &[u8],
    tensors: &[TensorRecord],
    weights: &Range<usize>,
) -> Result<(), EngineError> {
    let mut expected_offset = weights.start;
    let mut previous_name: Option<&TensorName> = None;
    for tensor in tensors {
        if previous_name.is_some_and(|name| name >= tensor.name()) {
            return Err(corrupt("tensor names are duplicate or not canonically sorted"));
        }
        let aligned_offset =
            u64_to_usize(align_up(usize_to_u64(expected_offset)?, TENSOR_DATA_ALIGNMENT)?)?;
        if package_bytes[expected_offset..aligned_offset].iter().any(|byte| *byte != 0) {
            return Err(corrupt("tensor alignment padding is nonzero"));
        }
        let range = tensor.data_range()?;
        if range.start != aligned_offset {
            return Err(corrupt("tensor data offsets are not canonical"));
        }
        expected_offset = range.end;
        previous_name = Some(tensor.name());
    }
    if expected_offset != weights.end {
        return Err(corrupt("tensor data does not exactly fill the weights section"));
    }
    Ok(())
}

fn parse_architecture_id(bytes: &[u8]) -> Result<ArchitectureId, EngineError> {
    let value = parse_zero_padded_ascii(bytes, "architecture identifier")?;
    ArchitectureId::new(value).map_err(|_| corrupt("architecture identifier is invalid"))
}

fn parse_tensor_name(bytes: &[u8]) -> Result<TensorName, EngineError> {
    let value = parse_zero_padded_ascii(bytes, "tensor name")?;
    if value.len() > MAX_TENSOR_NAME_BYTES
        || !value
            .as_bytes()
            .iter()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-'))
        || value.contains("..")
    {
        return Err(corrupt("tensor name is invalid"));
    }
    Ok(TensorName(value.to_owned()))
}

fn parse_zero_padded_ascii<'a>(bytes: &'a [u8], label: &str) -> Result<&'a str, EngineError> {
    let end = bytes.iter().position(|byte| *byte == 0).unwrap_or(bytes.len());
    if end == 0 || bytes[end..].iter().any(|byte| *byte != 0) || !bytes[..end].is_ascii() {
        return Err(corrupt(label));
    }
    std::str::from_utf8(&bytes[..end]).map_err(|_| corrupt(label))
}

/// Encodes a supported dtype for canonical fixture/conversion tooling.
pub fn encode_dtype(dtype: DType) -> Result<u8, EngineError> {
    let encoded = match dtype {
        DType::F32 => 1,
        DType::F16 => 2,
        DType::BFloat16 => 3,
        DType::I8 => 4,
        DType::U8 => 5,
        DType::I16 => 6,
        DType::U16 => 7,
        DType::I32 => 8,
        DType::U32 => 9,
        DType::I64 => 10,
        DType::U64 => 11,
        DType::Bool => 12,
        _ => return Err(EngineError::not_implemented("dtype encoding is not implemented")),
    };
    Ok(encoded)
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
        _ => Err(corrupt("tensor dtype is unsupported")),
    }
}

fn read_nonzero_u32(input: &[u8], offset: usize, label: &str) -> Result<u32, EngineError> {
    let value = read_u32(input, offset)?;
    if value == 0 {
        return Err(corrupt(label));
    }
    Ok(value)
}

fn checked_end(offset: u64, length: u64) -> Result<u64, EngineError> {
    offset
        .checked_add(length)
        .filter(|end| *end <= MAX_MODEL_PACKAGE_BYTES)
        .ok_or_else(|| corrupt("model package range overflows"))
}

fn align_up(value: u64, alignment: u64) -> Result<u64, EngineError> {
    let remainder = value % alignment;
    if remainder == 0 {
        Ok(value)
    } else {
        value
            .checked_add(alignment - remainder)
            .ok_or_else(|| corrupt("aligned package offset overflows"))
    }
}

fn usize_range(offset: u64, length: u64) -> Result<Range<usize>, EngineError> {
    let end = checked_end(offset, length)?;
    Ok(u64_to_usize(offset)?..u64_to_usize(end)?)
}

fn read_u16(input: &[u8], offset: usize) -> Result<u16, EngineError> {
    Ok(u16::from_le_bytes(read_array(input, offset)?))
}

fn read_u32(input: &[u8], offset: usize) -> Result<u32, EngineError> {
    Ok(u32::from_le_bytes(read_array(input, offset)?))
}

fn read_u64(input: &[u8], offset: usize) -> Result<u64, EngineError> {
    Ok(u64::from_le_bytes(read_array(input, offset)?))
}

fn read_array<const N: usize>(input: &[u8], offset: usize) -> Result<[u8; N], EngineError> {
    let end = offset.checked_add(N).ok_or_else(|| corrupt("package field offset overflows"))?;
    input
        .get(offset..end)
        .and_then(|bytes| bytes.try_into().ok())
        .ok_or_else(|| corrupt("package structure is truncated"))
}

fn u64_to_usize(value: u64) -> Result<usize, EngineError> {
    usize::try_from(value).map_err(|_| corrupt("package offset is not addressable"))
}

fn usize_to_u64(value: usize) -> Result<u64, EngineError> {
    u64::try_from(value).map_err(|_| corrupt("package size is not representable"))
}

fn usize_to_u32(value: usize) -> Result<u32, EngineError> {
    u32::try_from(value).map_err(|_| corrupt("package field is not representable"))
}

fn corrupt(reason: impl AsRef<str>) -> EngineError {
    EngineError::corrupt_model_package(reason)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_rejects_noncanonical_ranges_and_actual_length_mismatch() {
        let mut bytes = minimal_header();
        let header = ModelPackageHeader::parse(&bytes).expect("canonical header must parse");
        assert_eq!(header.manifest_range(), (64, 256));
        bytes[40..48].copy_from_slice(&321_u64.to_le_bytes());
        assert!(ModelPackageHeader::parse(&bytes).is_err());
    }

    #[test]
    fn rejects_truncation_unknown_versions_and_bad_magic() {
        let bytes = minimal_header();
        assert!(ModelPackageHeader::parse(&bytes[..63]).is_err());
        let mut unknown = bytes.clone();
        unknown[8..10].copy_from_slice(&2_u16.to_le_bytes());
        assert!(ModelPackageHeader::parse(&unknown).is_err());
        let mut bad_magic = bytes;
        bad_magic[0] ^= 0xff;
        assert!(ModelPackageHeader::parse(&bad_magic).is_err());
    }

    fn minimal_header() -> Vec<u8> {
        let tensor_table_offset = 64_u64 + 256;
        let signature_offset = tensor_table_offset + TENSOR_INDEX_ENTRY_BYTES;
        let package_length = signature_offset + u64::from(MAX_SIGNATURE_BYTES);
        let mut bytes = vec![0_u8; MODEL_PACKAGE_HEADER_BYTES];
        bytes[..8].copy_from_slice(&MODEL_PACKAGE_MAGIC);
        bytes[8..10].copy_from_slice(&1_u16.to_le_bytes());
        bytes[10..12].copy_from_slice(&0_u16.to_le_bytes());
        bytes[12..16].copy_from_slice(&64_u32.to_le_bytes());
        bytes[16..24].copy_from_slice(&package_length.to_le_bytes());
        bytes[24..32].copy_from_slice(&64_u64.to_le_bytes());
        bytes[32..40].copy_from_slice(&256_u64.to_le_bytes());
        bytes[40..48].copy_from_slice(&tensor_table_offset.to_le_bytes());
        bytes[48..52].copy_from_slice(&1_u32.to_le_bytes());
        bytes[52..56].copy_from_slice(&MAX_SIGNATURE_BYTES.to_le_bytes());
        bytes[56..64].copy_from_slice(&signature_offset.to_le_bytes());
        bytes
    }
}
