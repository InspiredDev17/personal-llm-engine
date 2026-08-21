//! Foundational, validated types shared across the native inference engine.
//!
//! This crate intentionally contains no inference implementation. It establishes
//! finite input bounds and vocabulary shared by package parsing, loading,
//! tokenization, scheduling, backends, and the local protocol.

#![forbid(unsafe_code)]

use std::error::Error;
use std::fmt;
use std::str::FromStr;

/// Largest accepted first-party model package: 1 TiB.
pub const MAX_MODEL_PACKAGE_BYTES: u64 = 1_u64 << 40;
/// Largest number of tensors described by one model package.
pub const MAX_TENSOR_COUNT: u32 = 262_144;
/// Largest supported tensor rank.
pub const MAX_TENSOR_RANK: usize = 8;
/// Largest value accepted for one tensor dimension.
pub const MAX_DIMENSION_SIZE: u64 = 1_u64 << 30;
/// Largest element count accepted for one tensor.
pub const MAX_TENSOR_ELEMENTS: u64 = 1_u64 << 40;
/// Largest bounded token collection processed at once.
pub const MAX_TOKEN_COUNT: u32 = 1_048_576;
/// Largest first-party tokenizer vocabulary.
pub const MAX_VOCABULARY_SIZE: u32 = 4_194_304;
/// Largest first-party tokenizer data section: 64 MiB.
pub const MAX_TOKENIZER_DATA_BYTES: u64 = 64 * 1024 * 1024;
/// Largest context window accepted by the engine.
pub const MAX_CONTEXT_TOKENS: u32 = 262_144;
/// Largest logical request, including bounded metadata: 4 MiB.
pub const MAX_REQUEST_BYTES: usize = 4 * 1024 * 1024;
/// Largest local protocol frame: 8 MiB.
pub const MAX_FRAME_BYTES: usize = 8 * 1024 * 1024;
/// Largest number of sequences permitted by one resource budget.
pub const MAX_CONCURRENT_SEQUENCES: u32 = 4_096;
/// Longest reusable-cache TTL accepted from control-plane policy: 365 days.
pub const MAX_REUSABLE_CACHE_TTL_SECONDS: u64 = 365 * 24 * 60 * 60;
/// Largest resident-memory budget accepted by the engine: 1 TiB.
pub const MAX_RESOURCE_MEMORY_BYTES: u64 = 1_u64 << 40;
/// Largest diagnostic retained in an engine error.
pub const MAX_ERROR_DETAIL_BYTES: usize = 1_024;
/// Largest UTF-8 text chunk emitted in one generation event.
pub const MAX_GENERATION_CHUNK_BYTES: usize = 64 * 1024;

const MAX_MODEL_ID_BYTES: usize = 128;
const MAX_COMPONENT_ID_BYTES: usize = 64;
const REQUEST_METADATA_OVERHEAD_BYTES: usize = 256;
const MAX_TENSOR_RANK_U64: u64 = 8;

const _: () = {
    assert!(MAX_FRAME_BYTES >= MAX_REQUEST_BYTES);
    assert!(MAX_CONTEXT_TOKENS <= MAX_TOKEN_COUNT);
    assert!(MAX_TOKEN_COUNT <= MAX_VOCABULARY_SIZE);
    assert!(MAX_TENSOR_RANK == 8);
    assert!(MAX_TENSOR_RANK_U64 == 8);
    assert!(MAX_GENERATION_CHUNK_BYTES < MAX_FRAME_BYTES);
};

fn validate_identifier(
    field: &'static str,
    value: String,
    max_bytes: usize,
) -> Result<String, EngineError> {
    if value.is_empty() {
        return Err(EngineError::invalid_field(field, "must not be empty"));
    }
    if value.len() > max_bytes {
        return Err(EngineError::invalid_field(field, "exceeds its maximum encoded length"));
    }
    if !value.is_ascii() {
        return Err(EngineError::invalid_field(field, "must be ASCII"));
    }

    let bytes = value.as_bytes();
    if !bytes[0].is_ascii_alphanumeric() || !bytes[bytes.len() - 1].is_ascii_alphanumeric() {
        return Err(EngineError::invalid_field(
            field,
            "must begin and end with an ASCII letter or digit",
        ));
    }
    if !bytes.iter().all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.'))
    {
        return Err(EngineError::invalid_field(field, "contains a prohibited character"));
    }
    if value.contains("..") {
        return Err(EngineError::invalid_field(
            field,
            "must not contain a path-traversal sequence",
        ));
    }

    Ok(value)
}

macro_rules! identifier_type {
    ($(#[$meta:meta])* $name:ident, $field:literal, $max_bytes:expr) => {
        $(#[$meta])*
        #[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
        pub struct $name(String);

        impl $name {
            /// Validates and constructs the identifier.
            pub fn new(value: impl Into<String>) -> Result<Self, EngineError> {
                validate_identifier($field, value.into(), $max_bytes).map(Self)
            }

            /// Returns the validated identifier text.
            pub fn as_str(&self) -> &str {
                &self.0
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str(&self.0)
            }
        }

        impl FromStr for $name {
            type Err = EngineError;

            fn from_str(value: &str) -> Result<Self, Self::Err> {
                Self::new(value)
            }
        }

        impl TryFrom<String> for $name {
            type Error = EngineError;

            fn try_from(value: String) -> Result<Self, Self::Error> {
                Self::new(value)
            }
        }

        impl TryFrom<&str> for $name {
            type Error = EngineError;

            fn try_from(value: &str) -> Result<Self, Self::Error> {
                Self::new(value)
            }
        }
    };
}

identifier_type!(
    /// Locally registered, path-free model identifier.
    ModelId,
    "model_id",
    MAX_MODEL_ID_BYTES
);
identifier_type!(
    /// Stable identifier for a supported model architecture.
    ArchitectureId,
    "architecture_id",
    MAX_COMPONENT_ID_BYTES
);
identifier_type!(
    /// Stable identifier for an engine backend.
    BackendId,
    "backend_id",
    MAX_COMPONENT_ID_BYTES
);
identifier_type!(
    /// Stable identifier for a locally configured device.
    DeviceId,
    "device_id",
    MAX_COMPONENT_ID_BYTES
);

/// Fixed local identifier for one trusted model-package signing key.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SignerKeyId([u8; Self::BYTE_LEN]);

impl SignerKeyId {
    /// Encoded key-identifier length.
    pub const BYTE_LEN: usize = 16;

    /// Constructs a nonzero key identifier from exact bytes.
    pub fn from_bytes(bytes: [u8; Self::BYTE_LEN]) -> Result<Self, EngineError> {
        if bytes == [0_u8; Self::BYTE_LEN] {
            return Err(EngineError::invalid_field("signer_key_id", "must not be all zero"));
        }
        Ok(Self(bytes))
    }

    /// Returns the exact identifier bytes.
    pub const fn as_bytes(&self) -> &[u8; Self::BYTE_LEN] {
        &self.0
    }
}

impl fmt::Display for SignerKeyId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        for byte in self.0 {
            write!(formatter, "{byte:02x}")?;
        }
        Ok(())
    }
}

/// Explicit algorithm identifier for a model-package digest.
///
/// This identifies an externally computed digest. It does not provide a hash
/// implementation or establish the separate package-signature profile.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[non_exhaustive]
pub enum ModelHashAlgorithm {
    /// SHA-384, represented by exactly 48 digest bytes.
    Sha384,
}

impl fmt::Display for ModelHashAlgorithm {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Sha384 => formatter.write_str("sha384"),
        }
    }
}

/// Algorithm-tagged digest of a first-party model package.
///
/// The digest is data only. This type neither computes hashes nor verifies
/// signatures.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ModelPackageHash {
    algorithm: ModelHashAlgorithm,
    digest: [u8; Self::BYTE_LEN],
}

/// Exact SHA-256 digest used by canonical cross-component content identities.
///
/// This is a value type only. Digest computation remains in the reviewed
/// cryptography crate, and possession of a digest grants no trust or authority.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Sha256Digest([u8; Self::BYTE_LEN]);

impl Sha256Digest {
    /// Digest length in bytes.
    pub const BYTE_LEN: usize = 32;
    /// Lowercase hexadecimal digest length.
    pub const HEX_LEN: usize = Self::BYTE_LEN * 2;

    /// Constructs a digest from exactly 32 bytes.
    pub const fn from_bytes(bytes: [u8; Self::BYTE_LEN]) -> Self {
        Self(bytes)
    }

    /// Parses exactly 64 hexadecimal characters.
    pub fn from_hex(encoded: &str) -> Result<Self, EngineError> {
        if encoded.len() != Self::HEX_LEN
            || !encoded.bytes().all(|byte| byte.is_ascii_digit() || matches!(byte, b'a'..=b'f'))
        {
            return Err(EngineError::invalid_field(
                "sha256_digest",
                "must contain exactly 64 lowercase hexadecimal characters",
            ));
        }
        let mut bytes = [0_u8; Self::BYTE_LEN];
        for (index, pair) in encoded.as_bytes().chunks_exact(2).enumerate() {
            let high = decode_hex_nibble(pair[0]).ok_or_else(|| {
                EngineError::invalid_field("sha256_digest", "contains non-hexadecimal text")
            })?;
            let low = decode_hex_nibble(pair[1]).ok_or_else(|| {
                EngineError::invalid_field("sha256_digest", "contains non-hexadecimal text")
            })?;
            bytes[index] = (high << 4) | low;
        }
        Ok(Self(bytes))
    }

    /// Returns the exact digest bytes.
    pub const fn as_bytes(&self) -> &[u8; Self::BYTE_LEN] {
        &self.0
    }
}

impl fmt::Display for Sha256Digest {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        for byte in self.0 {
            write!(formatter, "{byte:02x}")?;
        }
        Ok(())
    }
}

impl fmt::Debug for Sha256Digest {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "Sha256Digest({self})")
    }
}

impl ModelPackageHash {
    /// Digest length in bytes.
    pub const BYTE_LEN: usize = 48;
    /// Lowercase hexadecimal digest length.
    pub const HEX_LEN: usize = Self::BYTE_LEN * 2;

    /// Constructs a digest from its exact bytes.
    pub const fn from_bytes(algorithm: ModelHashAlgorithm, digest: [u8; Self::BYTE_LEN]) -> Self {
        Self { algorithm, digest }
    }

    /// Constructs a SHA-384-tagged digest from exactly 48 bytes.
    pub const fn from_sha384_bytes(digest: [u8; Self::BYTE_LEN]) -> Self {
        Self::from_bytes(ModelHashAlgorithm::Sha384, digest)
    }

    /// Parses the exact hexadecimal digest length for an explicit algorithm.
    pub fn from_hex(algorithm: ModelHashAlgorithm, encoded: &str) -> Result<Self, EngineError> {
        if encoded.len() != Self::HEX_LEN {
            return Err(EngineError::invalid_field(
                "model_package_hash",
                "must contain exactly 96 hexadecimal characters",
            ));
        }

        let mut bytes = [0_u8; Self::BYTE_LEN];
        for (index, pair) in encoded.as_bytes().chunks_exact(2).enumerate() {
            let high = decode_hex_nibble(pair[0]).ok_or_else(|| {
                EngineError::invalid_field("model_package_hash", "contains non-hexadecimal text")
            })?;
            let low = decode_hex_nibble(pair[1]).ok_or_else(|| {
                EngineError::invalid_field("model_package_hash", "contains non-hexadecimal text")
            })?;
            bytes[index] = (high << 4) | low;
        }
        Ok(Self::from_bytes(algorithm, bytes))
    }

    /// Parses exactly 96 hexadecimal characters and tags them as SHA-384.
    pub fn from_sha384_hex(encoded: &str) -> Result<Self, EngineError> {
        Self::from_hex(ModelHashAlgorithm::Sha384, encoded)
    }

    /// Returns the explicit digest algorithm.
    pub const fn algorithm(self) -> ModelHashAlgorithm {
        self.algorithm
    }

    /// Returns the digest bytes.
    pub const fn as_bytes(&self) -> &[u8; Self::BYTE_LEN] {
        &self.digest
    }
}

impl fmt::Display for ModelPackageHash {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}:", self.algorithm)?;
        for byte in self.digest {
            write!(formatter, "{byte:02x}")?;
        }
        Ok(())
    }
}

impl fmt::Debug for ModelPackageHash {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "ModelPackageHash({self})")
    }
}

impl FromStr for ModelPackageHash {
    type Err = EngineError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        let encoded = value.strip_prefix("sha384:").ok_or_else(|| {
            EngineError::invalid_field(
                "model_package_hash",
                "must use the explicit sha384 algorithm tag",
            )
        })?;
        Self::from_sha384_hex(encoded)
    }
}

const fn decode_hex_nibble(value: u8) -> Option<u8> {
    match value {
        b'0'..=b'9' => Some(value - b'0'),
        b'a'..=b'f' => Some(value - b'a' + 10),
        b'A'..=b'F' => Some(value - b'A' + 10),
        _ => None,
    }
}

/// Nonzero request correlation identifier.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RequestId(u128);

impl RequestId {
    /// Validates and constructs a request identifier.
    pub fn new(value: u128) -> Result<Self, EngineError> {
        if value == 0 {
            return Err(EngineError::invalid_field("request_id", "must be nonzero"));
        }
        Ok(Self(value))
    }

    /// Parses a request identifier from 16 big-endian bytes.
    pub fn from_bytes(bytes: [u8; 16]) -> Result<Self, EngineError> {
        Self::new(u128::from_be_bytes(bytes))
    }

    /// Returns the numeric identifier.
    pub const fn get(self) -> u128 {
        self.0
    }

    /// Returns the 16-byte big-endian representation.
    pub const fn to_bytes(self) -> [u8; 16] {
        self.0.to_be_bytes()
    }
}

impl fmt::Display for RequestId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{:032x}", self.0)
    }
}

/// Nonzero sequence identifier within the scheduler.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SequenceId(u64);

impl SequenceId {
    /// Validates and constructs a sequence identifier.
    pub fn new(value: u64) -> Result<Self, EngineError> {
        if value == 0 {
            return Err(EngineError::invalid_field("sequence_id", "must be nonzero"));
        }
        Ok(Self(value))
    }

    /// Returns the numeric identifier.
    pub const fn get(self) -> u64 {
        self.0
    }

    /// Advances the sequence identifier without wrapping.
    pub fn checked_next(self) -> Result<Self, EngineError> {
        self.0.checked_add(1).map(Self).ok_or_else(|| {
            EngineError::resource_exhaustion_unknown(ResourceKind::SequenceIdentifiers, u64::MAX)
        })
    }
}

impl fmt::Display for SequenceId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}", self.0)
    }
}

/// Fixed-width scalar storage types supported by the foundational tensor contract.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum DType {
    /// IEEE 754 binary32.
    F32,
    /// IEEE 754 binary16.
    F16,
    /// Brain floating point 16-bit.
    BFloat16,
    /// Signed 8-bit integer.
    I8,
    /// Unsigned 8-bit integer.
    U8,
    /// Signed 16-bit integer.
    I16,
    /// Unsigned 16-bit integer.
    U16,
    /// Signed 32-bit integer.
    I32,
    /// Unsigned 32-bit integer.
    U32,
    /// Signed 64-bit integer.
    I64,
    /// Unsigned 64-bit integer.
    U64,
    /// Boolean encoded as one byte.
    Bool,
}

impl DType {
    /// Returns the number of storage bytes per element.
    pub const fn byte_width(self) -> u64 {
        match self {
            Self::F32 | Self::I32 | Self::U32 => 4,
            Self::F16 | Self::BFloat16 | Self::I16 | Self::U16 => 2,
            Self::I8 | Self::U8 | Self::Bool => 1,
            Self::I64 | Self::U64 => 8,
        }
    }

    /// Reports whether this is a floating-point storage type.
    pub const fn is_floating_point(self) -> bool {
        matches!(self, Self::F32 | Self::F16 | Self::BFloat16)
    }
}

/// Validated, finite tensor dimensions and their checked element count.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct TensorShape {
    dimensions: Vec<u64>,
    element_count: u64,
}

impl TensorShape {
    /// Constructs a shape while bounding rank, dimensions, and their product.
    ///
    /// An empty dimension list represents a scalar with one element.
    pub fn new<I>(dimensions: I) -> Result<Self, EngineError>
    where
        I: IntoIterator<Item = u64>,
    {
        let mut validated = Vec::with_capacity(MAX_TENSOR_RANK);
        let mut element_count = 1_u64;

        for dimension in dimensions {
            if validated.len() >= MAX_TENSOR_RANK {
                return Err(EngineError::resource_limit(
                    ResourceKind::TensorRank,
                    MAX_TENSOR_RANK_U64 + 1,
                    MAX_TENSOR_RANK_U64,
                ));
            }
            if dimension == 0 {
                return Err(EngineError::invalid_field(
                    "tensor_shape",
                    "dimensions must be nonzero",
                ));
            }
            if dimension > MAX_DIMENSION_SIZE {
                return Err(EngineError::resource_limit(
                    ResourceKind::TensorDimension,
                    dimension,
                    MAX_DIMENSION_SIZE,
                ));
            }

            element_count = element_count.checked_mul(dimension).ok_or_else(|| {
                EngineError::resource_exhaustion_unknown(
                    ResourceKind::TensorElements,
                    MAX_TENSOR_ELEMENTS,
                )
            })?;
            if element_count > MAX_TENSOR_ELEMENTS {
                return Err(EngineError::resource_limit(
                    ResourceKind::TensorElements,
                    element_count,
                    MAX_TENSOR_ELEMENTS,
                ));
            }
            validated.push(dimension);
        }

        Ok(Self { dimensions: validated, element_count })
    }

    /// Returns tensor dimensions in row-major order.
    pub fn dimensions(&self) -> &[u64] {
        &self.dimensions
    }

    /// Returns the tensor rank.
    pub fn rank(&self) -> usize {
        self.dimensions.len()
    }

    /// Returns the previously checked element count.
    pub const fn element_count(&self) -> u64 {
        self.element_count
    }

    /// Computes the bounded storage length for a scalar type.
    pub fn byte_len(&self, dtype: DType) -> Result<u64, EngineError> {
        let byte_len = self.element_count.checked_mul(dtype.byte_width()).ok_or_else(|| {
            EngineError::resource_exhaustion_unknown(
                ResourceKind::TensorBytes,
                MAX_MODEL_PACKAGE_BYTES,
            )
        })?;
        if byte_len > MAX_MODEL_PACKAGE_BYTES {
            return Err(EngineError::resource_limit(
                ResourceKind::TensorBytes,
                byte_len,
                MAX_MODEL_PACKAGE_BYTES,
            ));
        }
        Ok(byte_len)
    }

    /// Reports whether the shape represents a scalar.
    pub fn is_scalar(&self) -> bool {
        self.dimensions.is_empty()
    }
}

/// Validated maximum number of tokens in one inference context.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ContextLimit(u32);

impl ContextLimit {
    /// Constructs a nonzero context limit within the engine hard bound.
    pub fn new(tokens: u32) -> Result<Self, EngineError> {
        if tokens == 0 {
            return Err(EngineError::invalid_field("context_limit", "must be nonzero"));
        }
        if tokens > MAX_CONTEXT_TOKENS {
            return Err(EngineError::resource_limit(
                ResourceKind::ContextTokens,
                u64::from(tokens),
                u64::from(MAX_CONTEXT_TOKENS),
            ));
        }
        Ok(Self(tokens))
    }

    /// Returns the token limit.
    pub const fn get(self) -> u32 {
        self.0
    }
}

/// Per-request resource ceilings enforced by scheduling and memory accounting.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ResourceBudget {
    max_memory_bytes: u64,
    max_kv_cache_bytes: u64,
    max_concurrent_sequences: u32,
}

impl ResourceBudget {
    /// Constructs a bounded resource budget.
    pub fn new(
        max_memory_bytes: u64,
        max_kv_cache_bytes: u64,
        max_concurrent_sequences: u32,
    ) -> Result<Self, EngineError> {
        validate_nonzero_memory("max_memory_bytes", max_memory_bytes)?;
        validate_nonzero_memory("max_kv_cache_bytes", max_kv_cache_bytes)?;
        if max_kv_cache_bytes > max_memory_bytes {
            return Err(EngineError::invalid_field(
                "max_kv_cache_bytes",
                "must not exceed max_memory_bytes",
            ));
        }
        if max_concurrent_sequences == 0 {
            return Err(EngineError::invalid_field("max_concurrent_sequences", "must be nonzero"));
        }
        if max_concurrent_sequences > MAX_CONCURRENT_SEQUENCES {
            return Err(EngineError::resource_limit(
                ResourceKind::ConcurrentSequences,
                u64::from(max_concurrent_sequences),
                u64::from(MAX_CONCURRENT_SEQUENCES),
            ));
        }

        Ok(Self { max_memory_bytes, max_kv_cache_bytes, max_concurrent_sequences })
    }

    /// Returns the maximum total resident bytes.
    pub const fn max_memory_bytes(self) -> u64 {
        self.max_memory_bytes
    }

    /// Returns the maximum bytes available to KV cache state.
    pub const fn max_kv_cache_bytes(self) -> u64 {
        self.max_kv_cache_bytes
    }

    /// Returns the maximum active sequence count.
    pub const fn max_concurrent_sequences(self) -> u32 {
        self.max_concurrent_sequences
    }
}

fn validate_nonzero_memory(field: &'static str, bytes: u64) -> Result<(), EngineError> {
    if bytes == 0 {
        return Err(EngineError::invalid_field(field, "must be nonzero"));
    }
    if bytes > MAX_RESOURCE_MEMORY_BYTES {
        return Err(EngineError::resource_limit(
            ResourceKind::MemoryBytes,
            bytes,
            MAX_RESOURCE_MEMORY_BYTES,
        ));
    }
    Ok(())
}

/// Resource category associated with a finite engine limit.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ResourceKind {
    /// Encoded model package bytes.
    ModelPackageBytes,
    /// Tensor records in a package.
    TensorCount,
    /// Tensor rank.
    TensorRank,
    /// One tensor dimension.
    TensorDimension,
    /// Tensor element count.
    TensorElements,
    /// Tensor storage bytes.
    TensorBytes,
    /// Bounded token collection size.
    TokenCount,
    /// Tokenizer vocabulary entries.
    VocabularySize,
    /// Context tokens.
    ContextTokens,
    /// Encoded request bytes.
    RequestBytes,
    /// Protocol frame bytes.
    FrameBytes,
    /// Runtime resident bytes.
    MemoryBytes,
    /// Concurrent scheduler sequences.
    ConcurrentSequences,
    /// Sequence identifier space.
    SequenceIdentifiers,
}

impl fmt::Display for ResourceKind {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let value = match self {
            Self::ModelPackageBytes => "model package bytes",
            Self::TensorCount => "tensor count",
            Self::TensorRank => "tensor rank",
            Self::TensorDimension => "tensor dimension",
            Self::TensorElements => "tensor elements",
            Self::TensorBytes => "tensor bytes",
            Self::TokenCount => "token count",
            Self::VocabularySize => "vocabulary size",
            Self::ContextTokens => "context tokens",
            Self::RequestBytes => "request bytes",
            Self::FrameBytes => "frame bytes",
            Self::MemoryBytes => "memory bytes",
            Self::ConcurrentSequences => "concurrent sequences",
            Self::SequenceIdentifiers => "sequence identifiers",
        };
        formatter.write_str(value)
    }
}

/// A bounded, nonempty, single-line-safe diagnostic message.
///
/// Callers must provide only static or explicitly sanitized diagnostics. Error
/// detail must never contain prompt or generated text, model/package content,
/// secrets, credentials, tokens, or host filesystem paths.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ErrorDetail(String);

impl ErrorDetail {
    /// Validates a diagnostic without truncating it.
    ///
    /// All Unicode control characters are rejected. The same confidentiality
    /// contract as [`ErrorDetail`] applies to caller-provided text.
    pub fn new(value: impl Into<String>) -> Result<Self, EngineError> {
        let value = value.into();
        if value.is_empty() {
            return Err(EngineError::invalid_field("error_detail", "must not be empty"));
        }
        if value.len() > MAX_ERROR_DETAIL_BYTES {
            return Err(EngineError::invalid_field(
                "error_detail",
                "exceeds its maximum encoded length",
            ));
        }
        if value.chars().any(char::is_control) {
            return Err(EngineError::invalid_field(
                "error_detail",
                "must not contain Unicode control characters",
            ));
        }
        Ok(Self(value))
    }

    /// Returns the bounded diagnostic text.
    pub fn as_str(&self) -> &str {
        &self.0
    }

    fn bounded(value: impl AsRef<str>) -> Self {
        let mut bounded = String::with_capacity(MAX_ERROR_DETAIL_BYTES);
        push_bounded_text(&mut bounded, value.as_ref());
        if bounded.is_empty() {
            bounded.push_str("unspecified");
        }
        Self(bounded)
    }
}

fn push_bounded_text(target: &mut String, value: &str) {
    for character in value.chars() {
        if character.is_control() {
            continue;
        }
        let Some(next_len) = target.len().checked_add(character.len_utf8()) else {
            break;
        };
        if next_len > MAX_ERROR_DETAIL_BYTES {
            break;
        }
        target.push(character);
    }
}

impl fmt::Display for ErrorDetail {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

/// Bounded detail for a resource-exhaustion failure.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResourceExhaustionDetail {
    reason: Option<ErrorDetail>,
    resource: Option<ResourceKind>,
    requested: Option<u64>,
    limit: Option<u64>,
}

impl ResourceExhaustionDetail {
    fn message(reason: impl AsRef<str>) -> Self {
        Self {
            reason: Some(ErrorDetail::bounded(reason)),
            resource: None,
            requested: None,
            limit: None,
        }
    }

    const fn limit(resource: ResourceKind, requested: Option<u64>, limit: u64) -> Self {
        Self { reason: None, resource: Some(resource), requested, limit: Some(limit) }
    }

    /// Returns the bounded human-readable reason, when this was message-based.
    pub fn reason(&self) -> Option<&str> {
        self.reason.as_ref().map(ErrorDetail::as_str)
    }

    /// Returns the structured resource category, when known.
    pub const fn resource(&self) -> Option<ResourceKind> {
        self.resource
    }

    /// Returns the requested amount, when it was representable.
    pub const fn requested(&self) -> Option<u64> {
        self.requested
    }

    /// Returns the enforced limit, when structured limit data is available.
    pub const fn limit_value(&self) -> Option<u64> {
        self.limit
    }
}

/// Bounded detail for an isolated-backend failure.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BackendFailureDetail {
    backend: Option<BackendId>,
    reason: ErrorDetail,
}

impl BackendFailureDetail {
    /// Returns the selected backend, when one had been selected.
    pub const fn backend(&self) -> Option<&BackendId> {
        self.backend.as_ref()
    }

    /// Returns the bounded backend failure reason.
    pub fn reason(&self) -> &str {
        self.reason.as_str()
    }
}

/// Stable category for an [`EngineError`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum EngineErrorKind {
    /// Invalid input.
    InvalidInput,
    /// Unsupported model architecture.
    UnsupportedArchitecture,
    /// Unsupported tokenizer.
    UnsupportedTokenizer,
    /// Corrupt model package.
    CorruptModelPackage,
    /// Signature validation failure.
    SignatureFailure,
    /// A finite resource was exhausted.
    ResourceExhaustion,
    /// Work was cancelled.
    Cancellation,
    /// Backend execution failure.
    BackendFailure,
    /// Internal invariant violation.
    InternalInvariantViolation,
    /// Local authentication failure.
    AuthenticationFailure,
    /// Local protocol violation.
    ProtocolViolation,
    /// Deliberately unimplemented behavior.
    NotImplemented,
}

/// Fail-closed error vocabulary shared by all engine components.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum EngineError {
    /// A caller supplied invalid input.
    InvalidInput(ErrorDetail),
    /// The model architecture has no first-party implementation.
    UnsupportedArchitecture(ArchitectureId),
    /// The tokenizer has no first-party implementation.
    UnsupportedTokenizer(ErrorDetail),
    /// A data-only model package failed structural validation.
    CorruptModelPackage(ErrorDetail),
    /// A model-package signature did not validate.
    SignatureFailure(ErrorDetail),
    /// A finite resource limit prevented the operation.
    ResourceExhaustion(ResourceExhaustionDetail),
    /// An in-flight operation was cancelled.
    Cancellation(Option<RequestId>),
    /// An isolated compute backend failed.
    BackendFailure(BackendFailureDetail),
    /// A programmer or verified-format invariant was violated.
    InternalInvariantViolation(ErrorDetail),
    /// A peer failed local authentication.
    AuthenticationFailure(ErrorDetail),
    /// A peer violated the bounded local protocol.
    ProtocolViolation(ErrorDetail),
    /// A known capability has not been implemented.
    NotImplemented(ErrorDetail),
}

impl EngineError {
    /// Constructs an invalid-input error with a bounded reason.
    pub fn invalid_input(reason: impl AsRef<str>) -> Self {
        Self::InvalidInput(ErrorDetail::bounded(reason))
    }

    /// Constructs a field-specific invalid-input error with bounded text.
    pub fn invalid_field(field: &'static str, reason: impl AsRef<str>) -> Self {
        let mut combined = String::with_capacity(MAX_ERROR_DETAIL_BYTES);
        push_bounded_text(&mut combined, field);
        push_bounded_text(&mut combined, ": ");
        push_bounded_text(&mut combined, reason.as_ref());
        Self::InvalidInput(ErrorDetail::bounded(combined))
    }

    /// Constructs an unsupported-architecture error.
    pub const fn unsupported_architecture(architecture: ArchitectureId) -> Self {
        Self::UnsupportedArchitecture(architecture)
    }

    /// Constructs an unsupported-tokenizer error with bounded text.
    pub fn unsupported_tokenizer(tokenizer: impl AsRef<str>) -> Self {
        Self::UnsupportedTokenizer(ErrorDetail::bounded(tokenizer))
    }

    /// Constructs a corrupt-model-package error with bounded text.
    pub fn corrupt_model_package(reason: impl AsRef<str>) -> Self {
        Self::CorruptModelPackage(ErrorDetail::bounded(reason))
    }

    /// Constructs a signature-failure error with bounded text.
    pub fn signature_failure(reason: impl AsRef<str>) -> Self {
        Self::SignatureFailure(ErrorDetail::bounded(reason))
    }

    /// Constructs a resource-exhaustion error with a bounded reason.
    pub fn resource_exhaustion(reason: impl AsRef<str>) -> Self {
        Self::ResourceExhaustion(ResourceExhaustionDetail::message(reason))
    }

    /// Constructs resource exhaustion with a known requested amount and limit.
    pub const fn resource_limit(resource: ResourceKind, requested: u64, limit: u64) -> Self {
        Self::ResourceExhaustion(ResourceExhaustionDetail::limit(resource, Some(requested), limit))
    }

    /// Constructs a resource-exhaustion error after arithmetic overflow.
    pub const fn resource_exhaustion_unknown(resource: ResourceKind, limit: u64) -> Self {
        Self::ResourceExhaustion(ResourceExhaustionDetail::limit(resource, None, limit))
    }

    /// Constructs a cancellation error.
    pub const fn cancellation(request_id: Option<RequestId>) -> Self {
        Self::Cancellation(request_id)
    }

    /// Constructs a backend-failure error with bounded text.
    pub fn backend_failure(reason: impl AsRef<str>) -> Self {
        Self::BackendFailure(BackendFailureDetail {
            backend: None,
            reason: ErrorDetail::bounded(reason),
        })
    }

    /// Constructs a backend-failure error correlated to a selected backend.
    pub fn backend_failure_for(backend: BackendId, reason: impl AsRef<str>) -> Self {
        Self::BackendFailure(BackendFailureDetail {
            backend: Some(backend),
            reason: ErrorDetail::bounded(reason),
        })
    }

    /// Constructs an internal-invariant error with bounded text.
    pub fn internal_invariant(invariant: impl AsRef<str>) -> Self {
        Self::InternalInvariantViolation(ErrorDetail::bounded(invariant))
    }

    /// Alias with the full error-category name.
    pub fn internal_invariant_violation(invariant: impl AsRef<str>) -> Self {
        Self::internal_invariant(invariant)
    }

    /// Constructs an authentication-failure error with bounded text.
    pub fn authentication_failure(reason: impl AsRef<str>) -> Self {
        Self::AuthenticationFailure(ErrorDetail::bounded(reason))
    }

    /// Constructs a protocol-violation error with bounded text.
    pub fn protocol_violation(reason: impl AsRef<str>) -> Self {
        Self::ProtocolViolation(ErrorDetail::bounded(reason))
    }

    /// Constructs an explicit not-implemented error with bounded text.
    pub fn not_implemented(capability: impl AsRef<str>) -> Self {
        Self::NotImplemented(ErrorDetail::bounded(capability))
    }

    /// Returns the stable error category.
    pub const fn kind(&self) -> EngineErrorKind {
        match self {
            Self::InvalidInput(_) => EngineErrorKind::InvalidInput,
            Self::UnsupportedArchitecture(_) => EngineErrorKind::UnsupportedArchitecture,
            Self::UnsupportedTokenizer(_) => EngineErrorKind::UnsupportedTokenizer,
            Self::CorruptModelPackage(_) => EngineErrorKind::CorruptModelPackage,
            Self::SignatureFailure(_) => EngineErrorKind::SignatureFailure,
            Self::ResourceExhaustion(_) => EngineErrorKind::ResourceExhaustion,
            Self::Cancellation(_) => EngineErrorKind::Cancellation,
            Self::BackendFailure(_) => EngineErrorKind::BackendFailure,
            Self::InternalInvariantViolation(_) => EngineErrorKind::InternalInvariantViolation,
            Self::AuthenticationFailure(_) => EngineErrorKind::AuthenticationFailure,
            Self::ProtocolViolation(_) => EngineErrorKind::ProtocolViolation,
            Self::NotImplemented(_) => EngineErrorKind::NotImplemented,
        }
    }
}

impl fmt::Display for EngineError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidInput(reason) => write!(formatter, "invalid input: {reason}"),
            Self::UnsupportedArchitecture(architecture) => {
                write!(formatter, "unsupported architecture: {architecture}")
            }
            Self::UnsupportedTokenizer(tokenizer) => {
                write!(formatter, "unsupported tokenizer: {tokenizer}")
            }
            Self::CorruptModelPackage(reason) => {
                write!(formatter, "corrupt model package: {reason}")
            }
            Self::SignatureFailure(reason) => {
                write!(formatter, "model package signature failure: {reason}")
            }
            Self::ResourceExhaustion(detail) => {
                match (detail.resource, detail.requested, detail.limit, detail.reason()) {
                    (Some(resource), Some(requested), Some(limit), _) => write!(
                        formatter,
                        "resource exhausted ({resource}): requested {requested}, limit {limit}"
                    ),
                    (Some(resource), None, Some(limit), _) => write!(
                        formatter,
                        "resource exhausted ({resource}): requested amount overflowed, limit {limit}"
                    ),
                    (_, _, _, Some(reason)) => write!(formatter, "resource exhausted: {reason}"),
                    _ => formatter.write_str("resource exhausted"),
                }
            }
            Self::Cancellation(Some(request_id)) => {
                write!(formatter, "request {request_id} cancelled")
            }
            Self::Cancellation(None) => formatter.write_str("operation cancelled"),
            Self::BackendFailure(detail) => match detail.backend() {
                Some(backend) => write!(formatter, "backend {backend} failed: {}", detail.reason),
                None => write!(formatter, "backend failed: {}", detail.reason),
            },
            Self::InternalInvariantViolation(invariant) => {
                write!(formatter, "internal invariant violation: {invariant}")
            }
            Self::AuthenticationFailure(reason) => {
                write!(formatter, "authentication failure: {reason}")
            }
            Self::ProtocolViolation(reason) => {
                write!(formatter, "protocol violation: {reason}")
            }
            Self::NotImplemented(capability) => {
                write!(formatter, "not implemented: {capability}")
            }
        }
    }
}

impl Error for EngineError {}

/// Native engine feature that callers may query.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum EngineCapability {
    /// Deterministic safe-Rust CPU reference backend.
    CpuReference,
    /// Optimized ARM64 backend.
    Arm64,
    /// Apple Metal backend.
    Metal,
    /// NVIDIA CUDA backend.
    Cuda,
    /// Incremental generation events.
    StreamingGeneration,
    /// Local Unix-domain-socket transport, once implemented and enabled.
    UnixDomainSocket,
}

/// Capability declared by a validated data-only model package.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ModelCapability {
    /// Autoregressive text generation.
    TextGeneration,
    /// Incremental token streaming.
    StreamingGeneration,
    /// A storage type present in the model package.
    StorageType(DType),
    /// Maximum model context supported by validated metadata.
    Context(ContextLimit),
}

/// Token identifier validated against the hard vocabulary bound.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TokenId(u32);

impl TokenId {
    /// Constructs an identifier smaller than [`MAX_VOCABULARY_SIZE`].
    pub fn new(value: u32) -> Result<Self, EngineError> {
        if value >= MAX_VOCABULARY_SIZE {
            return Err(EngineError::resource_limit(
                ResourceKind::VocabularySize,
                u64::from(value) + 1,
                u64::from(MAX_VOCABULARY_SIZE),
            ));
        }
        Ok(Self(value))
    }

    /// Returns the token identifier.
    pub const fn get(self) -> u32 {
        self.0
    }
}

/// Bounded UTF-8 text emitted with one token event.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct TextChunk(String);

impl TextChunk {
    /// Constructs a text chunk no larger than the event bound.
    pub fn new(value: impl Into<String>) -> Result<Self, EngineError> {
        let value = value.into();
        if value.len() > MAX_GENERATION_CHUNK_BYTES {
            return Err(EngineError::resource_limit(
                ResourceKind::FrameBytes,
                usize_to_u64(value.len()),
                usize_to_u64(MAX_GENERATION_CHUNK_BYTES),
            ));
        }
        Ok(Self(value))
    }

    /// Returns the UTF-8 chunk.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// A validated request to perform text generation only.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GenerationRequest {
    request_id: RequestId,
    model_id: ModelId,
    input_text: String,
    max_output_tokens: u32,
    context_limit: ContextLimit,
    resource_budget: ResourceBudget,
    estimated_size_bytes: usize,
}

impl GenerationRequest {
    /// Constructs a bounded text-generation request.
    ///
    /// Input remains UTF-8 text at this boundary so tokenization stays inside
    /// the Rust inference data plane.
    pub fn new(
        request_id: RequestId,
        model_id: ModelId,
        input_text: impl Into<String>,
        max_output_tokens: u32,
        context_limit: ContextLimit,
        resource_budget: ResourceBudget,
    ) -> Result<Self, EngineError> {
        let input_text = input_text.into();
        if input_text.is_empty() {
            return Err(EngineError::invalid_field("input_text", "must not be empty"));
        }
        if input_text.contains('\0') {
            return Err(EngineError::invalid_field("input_text", "must not contain NUL"));
        }
        if max_output_tokens == 0 {
            return Err(EngineError::invalid_field("max_output_tokens", "must be nonzero"));
        }
        if max_output_tokens > context_limit.get() {
            return Err(EngineError::resource_limit(
                ResourceKind::ContextTokens,
                u64::from(max_output_tokens),
                u64::from(context_limit.get()),
            ));
        }

        let estimated_size_bytes = REQUEST_METADATA_OVERHEAD_BYTES
            .checked_add(model_id.as_str().len())
            .and_then(|size| size.checked_add(input_text.len()))
            .ok_or_else(|| {
                EngineError::resource_exhaustion_unknown(
                    ResourceKind::RequestBytes,
                    usize_to_u64(MAX_REQUEST_BYTES),
                )
            })?;
        if estimated_size_bytes > MAX_REQUEST_BYTES {
            return Err(EngineError::resource_limit(
                ResourceKind::RequestBytes,
                usize_to_u64(estimated_size_bytes),
                usize_to_u64(MAX_REQUEST_BYTES),
            ));
        }

        Ok(Self {
            request_id,
            model_id,
            input_text,
            max_output_tokens,
            context_limit,
            resource_budget,
            estimated_size_bytes,
        })
    }

    /// Returns the request identifier.
    pub const fn request_id(&self) -> RequestId {
        self.request_id
    }

    /// Returns the registered model identifier.
    pub const fn model_id(&self) -> &ModelId {
        &self.model_id
    }

    /// Returns the original bounded UTF-8 input.
    pub fn input_text(&self) -> &str {
        &self.input_text
    }

    /// Returns the requested maximum number of generated tokens.
    pub const fn max_output_tokens(&self) -> u32 {
        self.max_output_tokens
    }

    /// Returns the request context limit.
    pub const fn context_limit(&self) -> ContextLimit {
        self.context_limit
    }

    /// Returns the request resource budget.
    pub const fn resource_budget(&self) -> ResourceBudget {
        self.resource_budget
    }

    /// Returns the checked conservative encoded-size estimate.
    pub const fn estimated_size_bytes(&self) -> usize {
        self.estimated_size_bytes
    }

    /// Validates the tokenized input before scheduler or model admission.
    ///
    /// This check remains inside the Rust data plane: the request carries text,
    /// then the engine-owned tokenizer reports the resulting token count here.
    /// Admission reserves the request's full output allowance, so the input and
    /// `max_output_tokens` must fit together within this request's context limit.
    pub fn validate_tokenized_input(&self, input_tokens: u32) -> Result<(), EngineError> {
        if input_tokens > MAX_TOKEN_COUNT {
            return Err(EngineError::resource_limit(
                ResourceKind::TokenCount,
                u64::from(input_tokens),
                u64::from(MAX_TOKEN_COUNT),
            ));
        }
        let reserved_context_tokens =
            input_tokens.checked_add(self.max_output_tokens).ok_or_else(|| {
                EngineError::resource_exhaustion_unknown(
                    ResourceKind::ContextTokens,
                    u64::from(self.context_limit.get()),
                )
            })?;
        if reserved_context_tokens > self.context_limit.get() {
            return Err(EngineError::resource_limit(
                ResourceKind::ContextTokens,
                u64::from(reserved_context_tokens),
                u64::from(self.context_limit.get()),
            ));
        }
        Ok(())
    }
}

/// Checked token accounting for a completed generation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct GenerationUsage {
    input_tokens: u32,
    output_tokens: u32,
    total_tokens: u32,
}

impl GenerationUsage {
    /// Constructs usage after checked addition and context-limit validation.
    pub fn new(input_tokens: u32, output_tokens: u32) -> Result<Self, EngineError> {
        if input_tokens > MAX_TOKEN_COUNT || output_tokens > MAX_TOKEN_COUNT {
            let requested = u64::from(input_tokens.max(output_tokens));
            return Err(EngineError::resource_limit(
                ResourceKind::TokenCount,
                requested,
                u64::from(MAX_TOKEN_COUNT),
            ));
        }
        let total_tokens = input_tokens.checked_add(output_tokens).ok_or_else(|| {
            EngineError::resource_exhaustion_unknown(
                ResourceKind::ContextTokens,
                u64::from(MAX_CONTEXT_TOKENS),
            )
        })?;
        if total_tokens > MAX_CONTEXT_TOKENS {
            return Err(EngineError::resource_limit(
                ResourceKind::ContextTokens,
                u64::from(total_tokens),
                u64::from(MAX_CONTEXT_TOKENS),
            ));
        }
        Ok(Self { input_tokens, output_tokens, total_tokens })
    }

    /// Constructs usage that is valid for one originating generation request.
    ///
    /// The input must have passed request admission with the entire requested
    /// output allowance reserved, and actual output cannot exceed that allowance.
    pub fn for_request(
        request: &GenerationRequest,
        input_tokens: u32,
        output_tokens: u32,
    ) -> Result<Self, EngineError> {
        request.validate_tokenized_input(input_tokens)?;
        if output_tokens > request.max_output_tokens() {
            return Err(EngineError::resource_limit(
                ResourceKind::TokenCount,
                u64::from(output_tokens),
                u64::from(request.max_output_tokens()),
            ));
        }

        let usage = Self::new(input_tokens, output_tokens)?;
        if usage.total_tokens() > request.context_limit().get() {
            return Err(EngineError::resource_limit(
                ResourceKind::ContextTokens,
                u64::from(usage.total_tokens()),
                u64::from(request.context_limit().get()),
            ));
        }
        Ok(usage)
    }

    /// Returns consumed input tokens.
    pub const fn input_tokens(self) -> u32 {
        self.input_tokens
    }

    /// Returns generated output tokens.
    pub const fn output_tokens(self) -> u32 {
        self.output_tokens
    }

    /// Returns the checked sum of input and output tokens.
    pub const fn total_tokens(self) -> u32 {
        self.total_tokens
    }
}

/// Bounded lifecycle event for one text-generation request.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum GenerationEvent {
    /// A scheduler sequence was assigned.
    Started {
        /// Request correlation identifier.
        request_id: RequestId,
        /// Assigned sequence identifier.
        sequence_id: SequenceId,
    },
    /// One validated token and UTF-8 text chunk was produced.
    Token {
        /// Request correlation identifier.
        request_id: RequestId,
        /// Assigned sequence identifier.
        sequence_id: SequenceId,
        /// Generated token identifier.
        token_id: TokenId,
        /// Detokenized text chunk.
        text: TextChunk,
    },
    /// Generation completed with checked usage.
    Completed {
        /// Request correlation identifier.
        request_id: RequestId,
        /// Assigned sequence identifier.
        sequence_id: SequenceId,
        /// Final token accounting.
        usage: GenerationUsage,
    },
    /// Generation failed explicitly; failure is never represented as success.
    Failed {
        /// Request correlation identifier.
        request_id: RequestId,
        /// Assigned sequence, if scheduling reached that stage.
        sequence_id: Option<SequenceId>,
        /// Structured failure.
        error: EngineError,
    },
}

impl GenerationEvent {
    /// Returns the request correlation identifier.
    pub const fn request_id(&self) -> RequestId {
        match self {
            Self::Started { request_id, .. }
            | Self::Token { request_id, .. }
            | Self::Completed { request_id, .. }
            | Self::Failed { request_id, .. } => *request_id,
        }
    }

    /// Returns the scheduler sequence when assigned.
    pub const fn sequence_id(&self) -> Option<SequenceId> {
        match self {
            Self::Started { sequence_id, .. }
            | Self::Token { sequence_id, .. }
            | Self::Completed { sequence_id, .. } => Some(*sequence_id),
            Self::Failed { sequence_id, .. } => *sequence_id,
        }
    }
}

/// Prohibited authority boundary that triggered a security event.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ForbiddenOperation {
    /// Tool execution.
    ToolExecution,
    /// Approval or policy decision.
    ApprovalDecision,
    /// Platform-memory mutation.
    MemoryMutation,
    /// Arbitrary filesystem access.
    ArbitraryFileAccess,
    /// Model download.
    ModelDownload,
    /// Engine self-update.
    SelfUpdate,
    /// Outbound network access.
    OutboundNetwork,
    /// Model-supplied executable code.
    ModelExecutableCode,
    /// Model-selected implementation kernel.
    ModelDefinedKernel,
}

/// Structured reason a model package was rejected.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ModelPackageRejection {
    /// Package structure or bounds were invalid.
    Corrupt,
    /// Package signature failed validation.
    SignatureFailure,
    /// Package architecture is unsupported.
    UnsupportedArchitecture,
    /// Package tokenizer is unsupported.
    UnsupportedTokenizer,
    /// Model identifier is not present in local configuration.
    UnregisteredModel,
    /// Package exceeded a hard size limit.
    SizeLimit,
}

/// Security-relevant event intended only for delivery to the Go control plane.
///
/// Events contain structured, bounded fields rather than model-controlled free
/// text. This foundation defines the type but has no event transport. Once
/// integrated, Go alone owns any allowlisted delivery to Agent Green; the model
/// never controls delivery, and the engine has no direct Agent Green path.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum SecurityEvent {
    /// Kernel peer credentials or request authentication denied a client.
    UnauthorizedClient,
    /// A bounded authentication-failure threshold was reached.
    RepeatedAuthenticationFailure,
    /// A frame could not satisfy the fixed protocol grammar.
    MalformedFrame,
    /// A frame declared or carried more than the hard byte ceiling.
    OversizedFrame,
    /// A peer offered an unknown protocol version.
    UnknownProtocolVersion,
    /// A request supplied a syntactically invalid model identifier.
    InvalidModelId {
        /// Correlation identifier, when safely decoded.
        request_id: Option<RequestId>,
    },
    /// A registered package signature failed verification.
    ModelSignatureFailure {
        /// Configuration-owned model identifier.
        model_id: ModelId,
    },
    /// A registered package hash failed verification.
    ModelHashFailure {
        /// Configuration-owned model identifier.
        model_id: ModelId,
    },
    /// Descriptor-safe package resolution detected a path escape attempt.
    ModelPathEscapeAttempt {
        /// Configuration-owned model identifier, when resolution reached one.
        model_id: Option<ModelId>,
    },
    /// Repeated bounded resource exhaustion crossed the alert threshold.
    ResourceExhaustionPattern,
    /// The owner supervisor reported a bounded crash-loop threshold.
    CrashLoopIndication,
    /// Repeated request rejection crossed the alert threshold.
    RepeatedRejectedRequests,
    /// A peer failed local authentication.
    AuthenticationFailure {
        /// Request correlation identifier, when safely decoded.
        request_id: Option<RequestId>,
    },
    /// A peer violated the bounded local protocol.
    ProtocolViolation {
        /// Request correlation identifier, when safely decoded.
        request_id: Option<RequestId>,
    },
    /// A request named a well-formed but unregistered model.
    UnregisteredModelRequested {
        /// Request correlation identifier.
        request_id: RequestId,
        /// Locally safe identifier that was not registered.
        model_id: ModelId,
    },
    /// A configured data-only model package was rejected.
    ModelPackageRejected {
        /// Locally registered model identifier.
        model_id: ModelId,
        /// Structured rejection category.
        reason: ModelPackageRejection,
    },
    /// A finite resource limit was exceeded.
    ResourceLimitExceeded {
        /// Request correlation identifier, when applicable.
        request_id: Option<RequestId>,
        /// Resource category.
        resource: ResourceKind,
        /// Requested amount, or `None` following arithmetic overflow.
        requested: Option<u64>,
        /// Enforced maximum.
        limit: u64,
    },
    /// A backend failed within the inference boundary.
    BackendFailure {
        /// Request correlation identifier, when applicable.
        request_id: Option<RequestId>,
        /// Selected backend, when available.
        backend: Option<BackendId>,
    },
    /// A forbidden authority boundary was reached.
    ForbiddenOperationAttempt {
        /// Request correlation identifier, when applicable.
        request_id: Option<RequestId>,
        /// Prohibited operation category.
        operation: ForbiddenOperation,
    },
    /// A verified internal invariant failed closed.
    InternalInvariantViolation {
        /// Request correlation identifier, when applicable.
        request_id: Option<RequestId>,
    },
}

fn usize_to_u64(value: usize) -> u64 {
    u64::try_from(value).unwrap_or(u64::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request_id() -> RequestId {
        RequestId::new(1).expect("one is a valid request identifier")
    }

    fn model_id() -> ModelId {
        ModelId::new("local-model-v1").expect("fixture model identifier is valid")
    }

    fn context_limit() -> ContextLimit {
        ContextLimit::new(4_096).expect("fixture context is valid")
    }

    fn resource_budget() -> ResourceBudget {
        ResourceBudget::new(8 * 1024 * 1024, 4 * 1024 * 1024, 1).expect("fixture budget is valid")
    }

    #[test]
    fn registered_identifiers_reject_paths_and_ambiguous_text() {
        assert!(ModelId::new("model-1.alpha").is_ok());
        assert!(ModelId::new("").is_err());
        assert!(ModelId::new("../model").is_err());
        assert!(ModelId::new("models/model").is_err());
        assert!(ModelId::new("model\\model").is_err());
        assert!(ModelId::new("-model").is_err());
        assert!(ModelId::new("model-").is_err());
        assert!(ModelId::new("mödel").is_err());
    }

    #[test]
    fn registered_identifier_length_is_bounded() {
        assert!(ModelId::new("a".repeat(MAX_MODEL_ID_BYTES)).is_ok());
        let error = ModelId::new("a".repeat(MAX_MODEL_ID_BYTES + 1))
            .expect_err("oversized identifier must fail");
        assert_eq!(error.kind(), EngineErrorKind::InvalidInput);
    }

    #[test]
    fn package_hash_hex_round_trips() {
        let encoded = "0123456789abcdef".repeat(6);
        let digest = ModelPackageHash::from_sha384_hex(&encoded).expect("valid digest");
        let tagged = format!("sha384:{encoded}");
        assert_eq!(digest.to_string(), tagged);
        assert_eq!(format!("{digest:?}"), format!("ModelPackageHash({tagged})"));
        assert_eq!(tagged.parse(), Ok(digest));
        assert_eq!(digest.algorithm(), ModelHashAlgorithm::Sha384);
        assert_eq!(digest.as_bytes()[0], 0x01);
        assert_eq!(digest.as_bytes()[31], 0xef);
        assert_eq!(digest.as_bytes()[47], 0xef);
    }

    #[test]
    fn canonical_sha256_digest_is_exact_and_lowercase() {
        let encoded = "0123456789abcdef".repeat(4);
        let digest = Sha256Digest::from_hex(&encoded).expect("valid canonical SHA-256");
        assert_eq!(digest.to_string(), encoded);
        assert_eq!(digest.as_bytes()[0], 0x01);
        assert!(Sha256Digest::from_hex("00").is_err());
        assert!(Sha256Digest::from_hex(&"A".repeat(Sha256Digest::HEX_LEN)).is_err());
    }

    #[test]
    fn package_hash_rejects_bad_length_and_alphabet() {
        assert!(ModelPackageHash::from_sha384_hex("00").is_err());
        assert!(ModelPackageHash::from_sha384_hex(&"g".repeat(ModelPackageHash::HEX_LEN)).is_err());
        assert!("00".parse::<ModelPackageHash>().is_err());
    }

    #[test]
    fn request_and_sequence_ids_are_nonzero_and_checked() {
        assert!(RequestId::new(0).is_err());
        let request = RequestId::new(0xfeed).expect("nonzero request id");
        assert_eq!(RequestId::from_bytes(request.to_bytes()), Ok(request));
        assert_eq!(request.to_string().len(), 32);

        assert!(SequenceId::new(0).is_err());
        assert_eq!(SequenceId::new(1).and_then(SequenceId::checked_next), SequenceId::new(2));
        let error = SequenceId::new(u64::MAX)
            .and_then(SequenceId::checked_next)
            .expect_err("sequence identifier must not wrap");
        assert_eq!(error.kind(), EngineErrorKind::ResourceExhaustion);
    }

    #[test]
    fn dtype_widths_are_explicit() {
        assert_eq!(DType::F32.byte_width(), 4);
        assert_eq!(DType::F16.byte_width(), 2);
        assert_eq!(DType::BFloat16.byte_width(), 2);
        assert_eq!(DType::I8.byte_width(), 1);
        assert_eq!(DType::U64.byte_width(), 8);
        assert!(DType::F32.is_floating_point());
        assert!(!DType::I32.is_floating_point());
    }

    #[test]
    fn tensor_shape_checks_rank_dimensions_product_and_bytes() {
        let scalar = TensorShape::new([]).expect("scalar shape is valid");
        assert!(scalar.is_scalar());
        assert_eq!(scalar.element_count(), 1);

        let matrix = TensorShape::new([2, 3, 4]).expect("bounded shape is valid");
        assert_eq!(matrix.rank(), 3);
        assert_eq!(matrix.dimensions(), &[2, 3, 4]);
        assert_eq!(matrix.element_count(), 24);
        assert_eq!(matrix.byte_len(DType::F32), Ok(96));

        assert!(TensorShape::new([0]).is_err());
        assert!(TensorShape::new([MAX_DIMENSION_SIZE + 1]).is_err());
        assert!(TensorShape::new([1; MAX_TENSOR_RANK + 1]).is_err());
        assert!(TensorShape::new([MAX_DIMENSION_SIZE, 2_048]).is_err());

        let maximum_elements =
            TensorShape::new([MAX_DIMENSION_SIZE, 1_024]).expect("element cap is accepted");
        assert_eq!(maximum_elements.element_count(), MAX_TENSOR_ELEMENTS);
        assert!(maximum_elements.byte_len(DType::F32).is_err());
    }

    #[test]
    fn context_and_resource_budgets_enforce_bounds() {
        assert!(ContextLimit::new(0).is_err());
        assert!(ContextLimit::new(MAX_CONTEXT_TOKENS).is_ok());
        assert!(ContextLimit::new(MAX_CONTEXT_TOKENS + 1).is_err());

        assert!(ResourceBudget::new(0, 1, 1).is_err());
        assert!(ResourceBudget::new(1, 0, 1).is_err());
        assert!(ResourceBudget::new(1, 2, 1).is_err());
        assert!(ResourceBudget::new(1, 1, 0).is_err());
        assert!(ResourceBudget::new(1, 1, MAX_CONCURRENT_SEQUENCES + 1).is_err());
        assert!(ResourceBudget::new(MAX_RESOURCE_MEMORY_BYTES + 1, 1, 1).is_err());

        let budget = ResourceBudget::new(8_192, 4_096, 2).expect("bounded budget");
        assert_eq!(budget.max_memory_bytes(), 8_192);
        assert_eq!(budget.max_kv_cache_bytes(), 4_096);
        assert_eq!(budget.max_concurrent_sequences(), 2);
    }

    #[test]
    fn generation_request_keeps_tokenization_inside_engine() {
        let request = GenerationRequest::new(
            request_id(),
            model_id(),
            "hello, sovereign engine",
            128,
            context_limit(),
            resource_budget(),
        )
        .expect("bounded generation request");

        assert_eq!(request.input_text(), "hello, sovereign engine");
        assert_eq!(request.request_id(), request_id());
        assert_eq!(request.model_id(), &model_id());
        assert_eq!(request.max_output_tokens(), 128);
        assert_eq!(request.context_limit(), context_limit());
        assert_eq!(request.resource_budget(), resource_budget());
        assert!(request.estimated_size_bytes() <= MAX_REQUEST_BYTES);
    }

    #[test]
    fn generation_request_rejects_invalid_text_and_limits() {
        let build = |text: String, output_tokens| {
            GenerationRequest::new(
                request_id(),
                model_id(),
                text,
                output_tokens,
                context_limit(),
                resource_budget(),
            )
        };

        assert!(build(String::new(), 1).is_err());
        assert!(build("bad\0input".to_owned(), 1).is_err());
        assert!(build("text".to_owned(), 0).is_err());
        assert!(build("text".to_owned(), context_limit().get() + 1).is_err());
        assert!(build("x".repeat(MAX_REQUEST_BYTES), 1).is_err());
    }

    #[test]
    fn generation_request_validates_post_tokenization_context_admission() {
        let request = GenerationRequest::new(
            request_id(),
            model_id(),
            "bounded input",
            128,
            ContextLimit::new(256).expect("context must validate"),
            resource_budget(),
        )
        .expect("request must validate");

        assert_eq!(request.validate_tokenized_input(128), Ok(()));
        let error = request
            .validate_tokenized_input(129)
            .expect_err("input plus reserved output must not exceed request context");
        assert_eq!(error.kind(), EngineErrorKind::ResourceExhaustion);
        assert!(request.validate_tokenized_input(MAX_TOKEN_COUNT + 1).is_err());
    }

    #[test]
    fn generation_usage_uses_checked_context_accounting() {
        let usage = GenerationUsage::new(100, 25).expect("bounded usage");
        assert_eq!(usage.input_tokens(), 100);
        assert_eq!(usage.output_tokens(), 25);
        assert_eq!(usage.total_tokens(), 125);
        assert!(GenerationUsage::new(MAX_CONTEXT_TOKENS, 1).is_err());
        assert!(GenerationUsage::new(MAX_TOKEN_COUNT, MAX_TOKEN_COUNT).is_err());
    }

    #[test]
    fn generation_usage_enforces_originating_request_limits() {
        let request = GenerationRequest::new(
            request_id(),
            model_id(),
            "bounded input",
            32,
            ContextLimit::new(128).expect("context must validate"),
            resource_budget(),
        )
        .expect("request must validate");

        let usage = GenerationUsage::for_request(&request, 96, 32)
            .expect("usage at both request limits must validate");
        assert_eq!(usage.total_tokens(), 128);
        assert!(GenerationUsage::for_request(&request, 97, 1).is_err());
        assert!(GenerationUsage::for_request(&request, 96, 33).is_err());
    }

    #[test]
    fn token_and_text_chunks_are_bounded() {
        assert_eq!(TokenId::new(0).map(TokenId::get), Ok(0));
        assert_eq!(
            TokenId::new(MAX_VOCABULARY_SIZE - 1).map(TokenId::get),
            Ok(MAX_VOCABULARY_SIZE - 1)
        );
        assert!(TokenId::new(MAX_VOCABULARY_SIZE).is_err());
        assert!(TextChunk::new("x".repeat(MAX_GENERATION_CHUNK_BYTES)).is_ok());
        assert!(TextChunk::new("x".repeat(MAX_GENERATION_CHUNK_BYTES + 1)).is_err());
    }

    #[test]
    fn error_details_and_convenience_constructors_remain_bounded() {
        assert!(ErrorDetail::new("").is_err());
        assert!(ErrorDetail::new("bad\0detail").is_err());
        assert!(ErrorDetail::new("bad\rdetail").is_err());
        assert!(ErrorDetail::new("bad\ndetail").is_err());
        assert!(ErrorDetail::new("bad\u{1b}detail").is_err());
        assert!(ErrorDetail::new("x".repeat(MAX_ERROR_DETAIL_BYTES + 1)).is_err());

        let oversized = "🦀".repeat(MAX_ERROR_DETAIL_BYTES);
        let error = EngineError::not_implemented(oversized);
        let EngineError::NotImplemented(capability) = error else {
            panic!("expected not-implemented category");
        };
        assert!(capability.as_str().len() <= MAX_ERROR_DETAIL_BYTES);
        assert!(capability.as_str().is_char_boundary(capability.as_str().len()));

        let sanitized = EngineError::invalid_input("line one\r\n\u{1b}line two");
        let EngineError::InvalidInput(detail) = sanitized else {
            panic!("expected invalid-input category");
        };
        assert_eq!(detail.as_str(), "line oneline two");
    }

    #[test]
    fn every_required_error_category_is_distinct() {
        let architecture = ArchitectureId::new("decoder-v1").expect("valid architecture");
        let backend = BackendId::new("cpu-reference").expect("valid backend");
        let cases = [
            EngineError::invalid_field("field", "reason"),
            EngineError::unsupported_architecture(architecture),
            EngineError::unsupported_tokenizer("tokenizer-v1"),
            EngineError::corrupt_model_package("bad offset"),
            EngineError::signature_failure("untrusted signer"),
            EngineError::resource_limit(ResourceKind::FrameBytes, 2, 1),
            EngineError::cancellation(Some(request_id())),
            EngineError::backend_failure_for(backend, "operation failed"),
            EngineError::internal_invariant_violation("shape was prevalidated"),
            EngineError::authentication_failure("peer credentials rejected"),
            EngineError::protocol_violation("invalid message ordering"),
            EngineError::not_implemented("metal backend"),
        ];
        let kinds: Vec<_> = cases.iter().map(EngineError::kind).collect();
        assert_eq!(kinds.len(), 12);
        for (index, kind) in kinds.iter().enumerate() {
            assert!(!kinds[..index].contains(kind));
        }
        assert!(cases.iter().all(|error| !error.to_string().is_empty()));
    }

    #[test]
    fn generation_events_preserve_correlation() {
        let event = GenerationEvent::Started {
            request_id: request_id(),
            sequence_id: SequenceId::new(7).expect("valid sequence"),
        };
        assert_eq!(event.request_id(), request_id());
        assert_eq!(event.sequence_id().map(SequenceId::get), Some(7));

        let failure = GenerationEvent::Failed {
            request_id: request_id(),
            sequence_id: None,
            error: EngineError::not_implemented("inference"),
        };
        assert_eq!(failure.sequence_id(), None);
    }

    #[test]
    fn security_events_use_only_structured_bounded_fields() {
        let event = SecurityEvent::ForbiddenOperationAttempt {
            request_id: Some(request_id()),
            operation: ForbiddenOperation::ToolExecution,
        };
        assert_eq!(
            event,
            SecurityEvent::ForbiddenOperationAttempt {
                request_id: Some(request_id()),
                operation: ForbiddenOperation::ToolExecution,
            }
        );
    }
}
