//! Deterministic Rust-only fixtures for native-engine tests.
//!
//! Fixtures are synthetic, canonically encoded, and signed only by the fixed
//! test key in this crate. The key is public test material and must never be
//! trusted by production configuration or used to sign a production package.

use ed25519_dalek::{Signer, SigningKey};
use personalai_engine_cpu_reference::{
    Backend, CpuReferenceBackend, InferenceGraph, Operation, ValueId,
};
use personalai_engine_crypto::{
    TrustedSigner, hash_package, hash_signed_content, signature_message,
};
use personalai_engine_memory::MemoryPool;
use personalai_engine_model_format::{
    MODEL_MANIFEST_BYTES, MODEL_MANIFEST_MAGIC, MODEL_PACKAGE_HEADER_BYTES, MODEL_PACKAGE_MAGIC,
    ParsedModelPackage, SIGNATURE_ENVELOPE_BYTES, SIGNATURE_ENVELOPE_MAGIC, TENSOR_DATA_ALIGNMENT,
    TENSOR_INDEX_ENTRY_BYTES, encode_dtype,
};
use personalai_engine_tensor::{MutableTensor, Tensor, TensorRead};
use personalai_engine_types::{
    DType, EngineError, MAX_MODEL_PACKAGE_BYTES, MAX_TENSOR_COUNT, ModelPackageHash, SignerKeyId,
    TensorShape,
};
use std::collections::BTreeMap;
use std::fmt;

const TEST_SIGNING_SECRET: [u8; 32] = [0x5a; 32];
const TEST_SIGNER_KEY_ID_BYTES: [u8; 16] = [0x71; 16];
const FIXTURE_ARCHITECTURE: &[u8] = b"personalai.synthetic-transformer.v1";
const TOKENIZER_MAGIC: &[u8; 8] = b"AIPTOK01";
const TOKENIZER_HEADER_BYTES: usize = 128;
const FIXTURE_VOCABULARY_SIZE: u32 = 259;
const FIXTURE_BOS_TOKEN: u32 = 256;
const FIXTURE_EOS_TOKEN: u32 = 257;
const FIXTURE_UNKNOWN_TOKEN: u32 = 258;
const FIXTURE_TENSOR_ELEMENTS: u64 = 4;
const DECODER_ARCHITECTURE: &[u8] = b"decoder-rms-rope-swiglu-gqa-v1";

/// Tiny deterministic pseudo-random source for test fixtures only.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DeterministicRng {
    state: u64,
}

impl DeterministicRng {
    /// Creates a deterministic generator; a zero seed maps to a fixed nonzero state.
    pub const fn new(seed: u64) -> Self {
        Self { state: if seed == 0 { 0x9e37_79b9_7f4a_7c15 } else { seed } }
    }

    /// Returns the next deterministic 64-bit value using xorshift64*.
    pub fn next_u64(&mut self) -> u64 {
        let mut value = self.state;
        value ^= value >> 12;
        value ^= value << 25;
        value ^= value >> 27;
        self.state = value;
        value.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }

    /// Returns a finite deterministic F32 in the closed interval `[-1, 1]`.
    pub fn next_f32(&mut self) -> f32 {
        let fraction_bits = u32::try_from(self.next_u64() >> 41).unwrap_or(0);
        let unit = f32::from_bits(0x3f80_0000 | fraction_bits) - 1.0;
        unit * 2.0 - 1.0
    }
}

/// Generates one small deterministic F32 tensor from a fixed seed.
pub fn deterministic_tensor(
    shape: TensorShape,
    seed: u64,
    pool: &MemoryPool,
) -> Result<Tensor, EngineError> {
    let count = usize::try_from(shape.element_count())
        .map_err(|_| EngineError::resource_exhaustion("test tensor size is not addressable"))?;
    let mut values = Vec::new();
    values
        .try_reserve_exact(count)
        .map_err(|_| EngineError::resource_exhaustion("test tensor allocation failed"))?;
    let mut random = DeterministicRng::new(seed);
    for _ in 0..count {
        values.push(random.next_f32());
    }
    MutableTensor::from_f32_values(shape, &values, pool)
}

/// Detailed finite-tensor comparison failure.
#[derive(Clone, Debug, PartialEq)]
pub struct TensorMismatch {
    index: Option<usize>,
    detail: &'static str,
    expected: Option<f32>,
    actual: Option<f32>,
}

impl fmt::Display for TensorMismatch {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match (self.index, self.expected, self.actual) {
            (Some(index), Some(expected), Some(actual)) => write!(
                formatter,
                "{} at element {index}: expected {expected:?}, actual {actual:?}",
                self.detail
            ),
            _ => formatter.write_str(self.detail),
        }
    }
}

impl std::error::Error for TensorMismatch {}

/// Compares finite F32 tensors with absolute and relative tolerances.
pub fn compare_tensors(
    actual: &impl TensorRead,
    expected: &impl TensorRead,
    absolute_tolerance: f32,
    relative_tolerance: f32,
) -> Result<(), TensorMismatch> {
    if actual.dtype() != DType::F32
        || expected.dtype() != DType::F32
        || actual.shape() != expected.shape()
        || !absolute_tolerance.is_finite()
        || absolute_tolerance < 0.0
        || !relative_tolerance.is_finite()
        || relative_tolerance < 0.0
    {
        return Err(TensorMismatch {
            index: None,
            detail: "tensor types, shapes, or tolerances differ",
            expected: None,
            actual: None,
        });
    }
    let count = usize::try_from(actual.shape().element_count()).map_err(|_| TensorMismatch {
        index: None,
        detail: "tensor element count is not addressable",
        expected: None,
        actual: None,
    })?;
    for index in 0..count {
        let actual_value = actual.f32_value(index).map_err(|_| TensorMismatch {
            index: Some(index),
            detail: "actual tensor could not be decoded",
            expected: None,
            actual: None,
        })?;
        let expected_value = expected.f32_value(index).map_err(|_| TensorMismatch {
            index: Some(index),
            detail: "expected tensor could not be decoded",
            expected: None,
            actual: None,
        })?;
        if !actual_value.is_finite() || !expected_value.is_finite() {
            return Err(TensorMismatch {
                index: Some(index),
                detail: "tensor contains NaN or infinity",
                expected: Some(expected_value),
                actual: Some(actual_value),
            });
        }
        let difference = (actual_value - expected_value).abs();
        let scale = actual_value.abs().max(expected_value.abs());
        if difference > absolute_tolerance.max(relative_tolerance * scale) {
            return Err(TensorMismatch {
                index: Some(index),
                detail: "tensor values exceed tolerance",
                expected: Some(expected_value),
                actual: Some(actual_value),
            });
        }
    }
    Ok(())
}

/// Executes one allowlisted graph operation using the scalar reference backend.
pub fn execute_reference_node(
    operation: &Operation,
    inputs: &[&Tensor],
    pool: &MemoryPool,
) -> Result<Tensor, EngineError> {
    CpuReferenceBackend::new()?.execute_operation(operation, inputs, pool)
}

/// Executes one complete validated graph using the scalar reference backend.
pub fn execute_reference_graph(
    graph: &InferenceGraph,
    inputs: BTreeMap<ValueId, Tensor>,
    pool: &MemoryPool,
) -> Result<BTreeMap<ValueId, Tensor>, EngineError> {
    graph.execute(&CpuReferenceBackend::new()?, inputs, pool)
}

#[derive(Clone, Copy)]
struct ManifestFixture {
    architecture: &'static [u8],
    layer_count: u32,
    hidden_dimension: u32,
    attention_head_count: u32,
    kv_head_count: u32,
    context_limit: u32,
    intermediate_dimension: u32,
}

struct TensorFixture {
    name: String,
    dimensions: Vec<u64>,
    bytes: Vec<u8>,
}

/// One complete package plus the local trust material needed to verify it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SignedPackageFixture {
    bytes: Vec<u8>,
    package_hash: ModelPackageHash,
    trusted_signer: TrustedSigner,
}

impl SignedPackageFixture {
    /// Returns the exact canonical package bytes.
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Returns a separately owned copy suitable for mutation tests.
    pub fn to_bytes(&self) -> Vec<u8> {
        self.bytes.clone()
    }

    /// Returns the SHA-384 identity of the complete signed package.
    pub const fn package_hash(&self) -> ModelPackageHash {
        self.package_hash
    }

    /// Returns the explicitly configured test-only trusted signer.
    pub const fn trusted_signer(&self) -> &TrustedSigner {
        &self.trusted_signer
    }
}

/// Deterministic builder for a fully parseable and strictly signed package.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PackageFixtureBuilder {
    tensor_count: u32,
}

impl PackageFixtureBuilder {
    /// Creates a small deterministic signed fixture builder.
    pub const fn new() -> Self {
        Self { tensor_count: 2 }
    }

    /// Sets a positive hard-bounded synthetic tensor count.
    pub fn with_tensor_count(mut self, tensor_count: u32) -> Result<Self, EngineError> {
        if tensor_count == 0 || tensor_count > MAX_TENSOR_COUNT {
            return Err(EngineError::invalid_input("fixture tensor count is outside hard bounds"));
        }
        self.tensor_count = tensor_count;
        Ok(self)
    }

    /// Generates identical package bytes and trust state for identical settings.
    pub fn build(self) -> Result<SignedPackageFixture, EngineError> {
        let count = usize::try_from(self.tensor_count).map_err(|_| {
            EngineError::resource_exhaustion("fixture tensor count is not addressable")
        })?;
        let mut tensors = Vec::new();
        tensors
            .try_reserve_exact(count)
            .map_err(|_| EngineError::resource_exhaustion("fixture tensor allocation failed"))?;
        for index in 0..self.tensor_count {
            tensors.push(f32_tensor(
                format!("tensor.{index:06}"),
                &[FIXTURE_TENSOR_ELEMENTS],
                &vec![
                    exact_f32_from_u32(index)?;
                    usize::try_from(FIXTURE_TENSOR_ELEMENTS).unwrap_or(0)
                ],
            )?);
        }
        build_signed_package(
            ManifestFixture {
                architecture: FIXTURE_ARCHITECTURE,
                layer_count: 1,
                hidden_dimension: 4,
                attention_head_count: 1,
                kv_head_count: 1,
                context_limit: 128,
                intermediate_dimension: 8,
            },
            tensors,
        )
    }
}

/// Builder for a tiny package implementing the exact first-party decoder schema.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DecoderPackageFixtureBuilder {
    context_limit: u32,
    printable_output: bool,
}

impl DecoderPackageFixtureBuilder {
    /// Creates the deterministic tiny decoder fixture builder.
    pub const fn new() -> Self {
        Self { context_limit: 8, printable_output: false }
    }

    /// Sets the synthetic decoder context limit without changing its weights.
    pub fn with_context_limit(mut self, context_limit: u32) -> Result<Self, EngineError> {
        if context_limit == 0 || context_limit > personalai_engine_types::MAX_CONTEXT_TOKENS {
            return Err(EngineError::invalid_input("fixture context limit is outside hard bounds"));
        }
        self.context_limit = context_limit;
        Ok(self)
    }

    /// Forces the synthetic decoder to select one of eight printable byte
    /// tokens without predetermining which one its real hidden state favors.
    #[must_use]
    pub const fn with_printable_output(mut self) -> Self {
        self.printable_output = true;
        self
    }

    /// Builds a signed two-layer GQA f32 decoder package with byte tokenizer data.
    pub fn build(self) -> Result<SignedPackageFixture, EngineError> {
        build_signed_package(
            decoder_manifest(self.context_limit),
            decoder_tensors(self.printable_output)?,
        )
    }
}

impl Default for DecoderPackageFixtureBuilder {
    fn default() -> Self {
        Self::new()
    }
}

fn build_signed_package(
    manifest: ManifestFixture,
    mut tensors: Vec<TensorFixture>,
) -> Result<SignedPackageFixture, EngineError> {
    tensors.sort_by(|left, right| left.name.cmp(&right.name));
    let tensor_count = u32::try_from(tensors.len())
        .map_err(|_| EngineError::resource_exhaustion("fixture tensor count exceeds its field"))?;
    if tensor_count == 0 || tensor_count > MAX_TENSOR_COUNT {
        return Err(EngineError::invalid_input("fixture tensor count is outside hard bounds"));
    }
    let tokenizer = build_byte_tokenizer_data()?;
    let header_bytes = usize_to_u64(MODEL_PACKAGE_HEADER_BYTES)?;
    let manifest_bytes = usize_to_u64(MODEL_MANIFEST_BYTES)?;
    let tensor_table_bytes = u64::from(tensor_count)
        .checked_mul(TENSOR_INDEX_ENTRY_BYTES)
        .ok_or_else(|| EngineError::resource_exhaustion("fixture tensor table overflows"))?;
    let tensor_table_offset = checked_add(header_bytes, manifest_bytes)?;
    let tokenizer_offset = checked_add(tensor_table_offset, tensor_table_bytes)?;
    let tokenizer_length = usize_to_u64(tokenizer.len())?;
    let weights_offset =
        align_up(checked_add(tokenizer_offset, tokenizer_length)?, TENSOR_DATA_ALIGNMENT)?;

    let mut offsets = Vec::new();
    offsets
        .try_reserve_exact(tensors.len())
        .map_err(|_| EngineError::resource_exhaustion("fixture offset allocation failed"))?;
    let mut cursor = weights_offset;
    for tensor in &tensors {
        cursor = align_up(cursor, TENSOR_DATA_ALIGNMENT)?;
        offsets.push(cursor);
        cursor = checked_add(cursor, usize_to_u64(tensor.bytes.len())?)?;
    }
    let signature_offset = cursor;
    let weights_length = signature_offset
        .checked_sub(weights_offset)
        .ok_or_else(|| EngineError::internal_invariant("fixture weights range underflows"))?;
    let package_length = checked_add(signature_offset, usize_to_u64(SIGNATURE_ENVELOPE_BYTES)?)?;
    if package_length > MAX_MODEL_PACKAGE_BYTES {
        return Err(EngineError::resource_exhaustion(
            "fixture package exceeds the package hard limit",
        ));
    }

    let mut bytes = zeroed_bytes(package_length)?;
    write_header(&mut bytes, package_length, tensor_table_offset, tensor_count, signature_offset)?;
    write_manifest(
        &mut bytes,
        &manifest,
        tensor_count,
        tokenizer_offset,
        tokenizer_length,
        weights_offset,
        weights_length,
    )?;
    write_tensor_table(&mut bytes, tensor_table_offset, &offsets, &tensors)?;
    copy_at(&mut bytes, tokenizer_offset, &tokenizer)?;
    write_tensor_data(&mut bytes, &offsets, &tensors)?;

    let key_id = SignerKeyId::from_bytes(TEST_SIGNER_KEY_ID_BYTES)?;
    let signing_key = SigningKey::from_bytes(&TEST_SIGNING_SECRET);
    let trusted_signer = TrustedSigner::active(key_id, signing_key.verifying_key().to_bytes())?;
    let signed_end = u64_to_usize(signature_offset)?;
    let digest = hash_signed_content(&bytes[..signed_end]);
    let message = signature_message(&digest)?;
    let signature = signing_key.sign(&message).to_bytes();
    write_signature(&mut bytes, signature_offset, key_id, &signature)?;

    let package_hash = hash_package(&bytes);
    ParsedModelPackage::parse(&bytes)?;
    Ok(SignedPackageFixture { bytes, package_hash, trusted_signer })
}

impl Default for PackageFixtureBuilder {
    fn default() -> Self {
        Self::new()
    }
}

fn decoder_manifest(context_limit: u32) -> ManifestFixture {
    ManifestFixture {
        architecture: DECODER_ARCHITECTURE,
        layer_count: 2,
        hidden_dimension: 4,
        attention_head_count: 2,
        kv_head_count: 1,
        context_limit,
        intermediate_dimension: 8,
    }
}

fn decoder_tensors(printable_output: bool) -> Result<Vec<TensorFixture>, EngineError> {
    let mut output = repeated(1_036, 0.0003);
    if printable_output {
        for dimension in 0..4_usize {
            let positive = (65_usize + dimension * 2) * 4 + dimension;
            let negative = positive + 4;
            output[positive] = 100.0;
            output[negative] = -100.0;
        }
    }
    let mut tensors = vec![
        f32_tensor("token_embedding.weight", &[259, 4], &repeated(1_036, 0.0005))?,
        f32_tensor("final_norm.weight", &[4], &[1.0; 4])?,
        f32_tensor("output.weight", &[259, 4], &output)?,
    ];
    for layer in 0..2 {
        let prefix = format!("layers.{layer:03}");
        tensors.extend([
            f32_tensor(format!("{prefix}.attn_norm.weight"), &[4], &[1.0; 4])?,
            f32_tensor(format!("{prefix}.q_proj.weight"), &[4, 4], &identity(4))?,
            f32_tensor(format!("{prefix}.k_proj.weight"), &[2, 4], &identity_rows(2, 4))?,
            f32_tensor(format!("{prefix}.v_proj.weight"), &[2, 4], &identity_rows(2, 4))?,
            f32_tensor(format!("{prefix}.o_proj.weight"), &[4, 4], &identity(4))?,
            f32_tensor(format!("{prefix}.ffn_norm.weight"), &[4], &[1.0; 4])?,
            f32_tensor(format!("{prefix}.gate_proj.weight"), &[8, 4], &repeated(32, 0.002))?,
            f32_tensor(format!("{prefix}.up_proj.weight"), &[8, 4], &repeated(32, 0.001))?,
            f32_tensor(format!("{prefix}.down_proj.weight"), &[4, 8], &repeated(32, 0.001))?,
        ]);
    }
    tensors.sort_by(|left, right| left.name.cmp(&right.name));
    Ok(tensors)
}

fn f32_tensor(
    name: impl Into<String>,
    dimensions: &[u64],
    values: &[f32],
) -> Result<TensorFixture, EngineError> {
    let element_count = dimensions.iter().try_fold(1_u64, |count, dimension| {
        count.checked_mul(*dimension).ok_or_else(|| {
            EngineError::resource_exhaustion("fixture tensor element count overflows")
        })
    })?;
    if usize_to_u64(values.len())? != element_count {
        return Err(EngineError::internal_invariant(
            "fixture tensor values do not match its dimensions",
        ));
    }
    let byte_length = values
        .len()
        .checked_mul(4)
        .ok_or_else(|| EngineError::resource_exhaustion("fixture tensor bytes overflow"))?;
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(byte_length)
        .map_err(|_| EngineError::resource_exhaustion("fixture tensor allocation failed"))?;
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    Ok(TensorFixture { name: name.into(), dimensions: dimensions.to_vec(), bytes })
}

fn identity(dimension: usize) -> Vec<f32> {
    let mut values = vec![0.0; dimension * dimension];
    for index in 0..dimension {
        values[index * dimension + index] = 1.0;
    }
    values
}

fn identity_rows(rows: usize, columns: usize) -> Vec<f32> {
    let mut values = vec![0.0; rows * columns];
    for index in 0..rows.min(columns) {
        values[index * columns + index] = 1.0;
    }
    values
}

fn repeated(length: usize, scale: f32) -> Vec<f32> {
    (0..length)
        .map(|index| {
            let bounded = u16::try_from(index).unwrap_or(u16::MAX);
            (f32::from(bounded) + 1.0) * scale
        })
        .collect()
}

#[allow(clippy::cast_precision_loss)]
fn exact_f32_from_u32(value: u32) -> Result<f32, EngineError> {
    if value > 1_u32 << 24 {
        return Err(EngineError::invalid_input("fixture integer exceeds the exact f32 range"));
    }
    Ok(value as f32)
}

fn build_byte_tokenizer_data() -> Result<Vec<u8>, EngineError> {
    let byte_entry_bytes = 9_usize;
    let special_pieces: [&[u8]; 3] = [b"<|bos|>", b"<|eos|>", b"<|unk|>"];
    let special_entry_bytes = special_pieces.iter().try_fold(0_usize, |total, piece| {
        total
            .checked_add(8 + piece.len())
            .ok_or_else(|| EngineError::resource_exhaustion("fixture tokenizer length overflows"))
    })?;
    let entries_bytes = 256_usize
        .checked_mul(byte_entry_bytes)
        .and_then(|value| value.checked_add(special_entry_bytes))
        .ok_or_else(|| EngineError::resource_exhaustion("fixture tokenizer length overflows"))?;
    let total_bytes = TOKENIZER_HEADER_BYTES
        .checked_add(entries_bytes)
        .and_then(|value| value.checked_add(3 * 8))
        .ok_or_else(|| EngineError::resource_exhaustion("fixture tokenizer length overflows"))?;
    let mut bytes = zeroed_bytes(usize_to_u64(total_bytes)?)?;
    bytes[..8].copy_from_slice(TOKENIZER_MAGIC);
    write_u16(&mut bytes, 8, 1)?;
    write_u16(&mut bytes, 10, 0)?;
    write_u32(&mut bytes, 12, usize_to_u32(TOKENIZER_HEADER_BYTES)?)?;
    write_u16(&mut bytes, 16, 1)?;
    write_u16(&mut bytes, 18, 1)?;
    write_u16(&mut bytes, 20, 1)?;
    write_u16(&mut bytes, 22, 1)?;
    write_u32(&mut bytes, 24, FIXTURE_VOCABULARY_SIZE)?;
    write_u32(&mut bytes, 28, 0)?;
    write_u32(&mut bytes, 32, 3)?;
    write_u32(&mut bytes, 36, FIXTURE_BOS_TOKEN)?;
    write_u32(&mut bytes, 40, FIXTURE_EOS_TOKEN)?;
    write_u32(&mut bytes, 44, FIXTURE_UNKNOWN_TOKEN)?;
    write_u16(&mut bytes, 48, 7)?;
    write_u16(&mut bytes, 50, 0)?;
    write_u32(&mut bytes, 52, 4 * 1024 * 1024)?;
    write_u32(&mut bytes, 56, 1_048_576)?;
    write_u32(&mut bytes, 60, usize_to_u32(TOKENIZER_HEADER_BYTES)?)?;
    let mut cursor = TOKENIZER_HEADER_BYTES;
    for byte in 0_u16..=255 {
        write_u32(&mut bytes, cursor, u32::from(byte))?;
        write_u16(&mut bytes, cursor + 4, 0)?;
        write_u16(&mut bytes, cursor + 6, 1)?;
        bytes[cursor + 8] = u8::try_from(byte)
            .map_err(|_| EngineError::internal_invariant("fixture byte token overflows"))?;
        cursor = cursor
            .checked_add(byte_entry_bytes)
            .ok_or_else(|| EngineError::internal_invariant("fixture tokenizer cursor overflows"))?;
    }
    for (offset, piece) in special_pieces.iter().enumerate() {
        let token_id = FIXTURE_BOS_TOKEN
            .checked_add(
                u32::try_from(offset).map_err(|_| {
                    EngineError::internal_invariant("fixture special index overflows")
                })?,
            )
            .ok_or_else(|| EngineError::internal_invariant("fixture special token overflows"))?;
        write_u32(&mut bytes, cursor, token_id)?;
        write_u16(&mut bytes, cursor + 4, 1)?;
        write_u16(
            &mut bytes,
            cursor + 6,
            u16::try_from(piece.len())
                .map_err(|_| EngineError::internal_invariant("fixture special length overflows"))?,
        )?;
        copy_at(&mut bytes, usize_to_u64(cursor + 8)?, piece)?;
        cursor = cursor
            .checked_add(8 + piece.len())
            .ok_or_else(|| EngineError::internal_invariant("fixture tokenizer cursor overflows"))?;
    }
    write_u32(&mut bytes, 64, usize_to_u32(cursor)?)?;
    write_u32(&mut bytes, 68, usize_to_u32(cursor)?)?;
    for (token_id, role) in
        [(FIXTURE_BOS_TOKEN, 1_u32), (FIXTURE_EOS_TOKEN, 2_u32), (FIXTURE_UNKNOWN_TOKEN, 4_u32)]
    {
        write_u32(&mut bytes, cursor, token_id)?;
        write_u32(&mut bytes, cursor + 4, role)?;
        cursor = cursor
            .checked_add(8)
            .ok_or_else(|| EngineError::internal_invariant("fixture tokenizer cursor overflows"))?;
    }
    write_u32(&mut bytes, 72, usize_to_u32(total_bytes)?)?;
    let content_hash = hash_package(&bytes[TOKENIZER_HEADER_BYTES..]);
    bytes[80..128].copy_from_slice(content_hash.as_bytes());
    if cursor != total_bytes {
        return Err(EngineError::internal_invariant("fixture tokenizer length invariant failed"));
    }
    Ok(bytes)
}

fn write_header(
    bytes: &mut [u8],
    package_length: u64,
    tensor_table_offset: u64,
    tensor_count: u32,
    signature_offset: u64,
) -> Result<(), EngineError> {
    bytes[..8].copy_from_slice(&MODEL_PACKAGE_MAGIC);
    write_u16(bytes, 8, 1)?;
    write_u16(bytes, 10, 0)?;
    write_u32(bytes, 12, usize_to_u32(MODEL_PACKAGE_HEADER_BYTES)?)?;
    write_u64(bytes, 16, package_length)?;
    write_u64(bytes, 24, usize_to_u64(MODEL_PACKAGE_HEADER_BYTES)?)?;
    write_u64(bytes, 32, usize_to_u64(MODEL_MANIFEST_BYTES)?)?;
    write_u64(bytes, 40, tensor_table_offset)?;
    write_u32(bytes, 48, tensor_count)?;
    write_u32(bytes, 52, usize_to_u32(SIGNATURE_ENVELOPE_BYTES)?)?;
    write_u64(bytes, 56, signature_offset)
}

fn write_manifest(
    bytes: &mut [u8],
    fixture: &ManifestFixture,
    tensor_count: u32,
    tokenizer_offset: u64,
    tokenizer_length: u64,
    weights_offset: u64,
    weights_length: u64,
) -> Result<(), EngineError> {
    let start = MODEL_PACKAGE_HEADER_BYTES;
    let end = start
        .checked_add(MODEL_MANIFEST_BYTES)
        .ok_or_else(|| EngineError::internal_invariant("fixture manifest end overflows"))?;
    let manifest = bytes
        .get_mut(start..end)
        .ok_or_else(|| EngineError::internal_invariant("fixture manifest range is unavailable"))?;
    manifest[..8].copy_from_slice(&MODEL_MANIFEST_MAGIC);
    write_u16(manifest, 8, 1)?;
    write_u16(manifest, 10, 0)?;
    write_u32(manifest, 12, 3)?;
    manifest[16..16 + fixture.architecture.len()].copy_from_slice(fixture.architecture);
    write_u32(manifest, 80, fixture.layer_count)?;
    write_u32(manifest, 84, fixture.hidden_dimension)?;
    write_u32(manifest, 88, fixture.attention_head_count)?;
    write_u32(manifest, 92, fixture.kv_head_count)?;
    write_u32(manifest, 96, FIXTURE_VOCABULARY_SIZE)?;
    write_u32(manifest, 100, fixture.context_limit)?;
    write_u16(manifest, 104, 1)?;
    manifest[106] = encode_dtype(DType::F32)?;
    write_u32(manifest, 108, tensor_count)?;
    write_u64(manifest, 112, tokenizer_offset)?;
    write_u64(manifest, 120, tokenizer_length)?;
    write_u64(manifest, 128, weights_offset)?;
    write_u64(manifest, 136, weights_length)?;
    let source_hash = hash_signed_content(b"personalai deterministic synthetic fixture source v1");
    manifest[144..192].copy_from_slice(&source_hash);
    write_u32(manifest, 192, fixture.intermediate_dimension)?;
    write_u32(manifest, 196, 1.0e-5_f32.to_bits())?;
    write_u32(manifest, 200, 10_000.0_f32.to_bits())?;
    Ok(())
}

fn write_tensor_table(
    bytes: &mut [u8],
    table_offset: u64,
    data_offsets: &[u64],
    tensors: &[TensorFixture],
) -> Result<(), EngineError> {
    let table_start = u64_to_usize(table_offset)?;
    let record_bytes = u64_to_usize(TENSOR_INDEX_ENTRY_BYTES)?;
    for (index, (data_offset, tensor)) in data_offsets.iter().zip(tensors).enumerate() {
        let start = index
            .checked_mul(record_bytes)
            .and_then(|value| value.checked_add(table_start))
            .ok_or_else(|| EngineError::internal_invariant("fixture tensor record overflows"))?;
        let end = start.checked_add(record_bytes).ok_or_else(|| {
            EngineError::internal_invariant("fixture tensor record end overflows")
        })?;
        let record = bytes.get_mut(start..end).ok_or_else(|| {
            EngineError::internal_invariant("fixture tensor record range is unavailable")
        })?;
        record[..tensor.name.len()].copy_from_slice(tensor.name.as_bytes());
        record[32] = encode_dtype(DType::F32)?;
        record[33] = u8::try_from(tensor.dimensions.len()).map_err(|_| {
            EngineError::internal_invariant("fixture tensor rank exceeds its field")
        })?;
        for (dimension_index, dimension) in tensor.dimensions.iter().enumerate() {
            let offset = 40_usize
                .checked_add(dimension_index.checked_mul(8).ok_or_else(|| {
                    EngineError::internal_invariant("fixture dimension offset overflows")
                })?)
                .ok_or_else(|| {
                    EngineError::internal_invariant("fixture dimension offset overflows")
                })?;
            write_u64(record, offset, *dimension)?;
        }
        write_u64(record, 104, *data_offset)?;
        write_u64(record, 112, usize_to_u64(tensor.bytes.len())?)?;
    }
    Ok(())
}

fn write_tensor_data(
    bytes: &mut [u8],
    data_offsets: &[u64],
    tensors: &[TensorFixture],
) -> Result<(), EngineError> {
    for (data_offset, tensor) in data_offsets.iter().zip(tensors) {
        let start = u64_to_usize(*data_offset)?;
        let end = start
            .checked_add(tensor.bytes.len())
            .ok_or_else(|| EngineError::internal_invariant("fixture tensor data end overflows"))?;
        bytes
            .get_mut(start..end)
            .ok_or_else(|| {
                EngineError::internal_invariant("fixture tensor data range is unavailable")
            })?
            .copy_from_slice(&tensor.bytes);
    }
    Ok(())
}

fn write_signature(
    bytes: &mut [u8],
    signature_offset: u64,
    key_id: SignerKeyId,
    signature: &[u8; 64],
) -> Result<(), EngineError> {
    let start = u64_to_usize(signature_offset)?;
    let end = start
        .checked_add(SIGNATURE_ENVELOPE_BYTES)
        .ok_or_else(|| EngineError::internal_invariant("fixture signature end overflows"))?;
    let envelope = bytes
        .get_mut(start..end)
        .ok_or_else(|| EngineError::internal_invariant("fixture signature range is unavailable"))?;
    envelope[..8].copy_from_slice(&SIGNATURE_ENVELOPE_MAGIC);
    write_u16(envelope, 8, 1)?;
    write_u16(envelope, 10, 1)?;
    envelope[12..28].copy_from_slice(key_id.as_bytes());
    envelope[28..92].copy_from_slice(signature);
    Ok(())
}

fn copy_at(destination: &mut [u8], offset: u64, source: &[u8]) -> Result<(), EngineError> {
    let start = u64_to_usize(offset)?;
    let end = start
        .checked_add(source.len())
        .ok_or_else(|| EngineError::internal_invariant("fixture copy range overflows"))?;
    destination
        .get_mut(start..end)
        .ok_or_else(|| EngineError::internal_invariant("fixture copy range is unavailable"))?
        .copy_from_slice(source);
    Ok(())
}

fn zeroed_bytes(length: u64) -> Result<Vec<u8>, EngineError> {
    let length = u64_to_usize(length)?;
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(length)
        .map_err(|_| EngineError::resource_exhaustion("fixture allocation failed"))?;
    bytes.resize(length, 0);
    Ok(bytes)
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
    let end = offset
        .checked_add(value.len())
        .ok_or_else(|| EngineError::internal_invariant("fixture write range overflows"))?;
    bytes
        .get_mut(offset..end)
        .ok_or_else(|| EngineError::internal_invariant("fixture write range is unavailable"))?
        .copy_from_slice(value);
    Ok(())
}

fn checked_add(left: u64, right: u64) -> Result<u64, EngineError> {
    left.checked_add(right)
        .ok_or_else(|| EngineError::resource_exhaustion("fixture package range overflows"))
}

fn align_up(value: u64, alignment: u64) -> Result<u64, EngineError> {
    let remainder = value % alignment;
    if remainder == 0 { Ok(value) } else { checked_add(value, alignment - remainder) }
}

fn usize_to_u64(value: usize) -> Result<u64, EngineError> {
    u64::try_from(value)
        .map_err(|_| EngineError::resource_exhaustion("fixture size is not representable"))
}

fn usize_to_u32(value: usize) -> Result<u32, EngineError> {
    u32::try_from(value)
        .map_err(|_| EngineError::resource_exhaustion("fixture size exceeds its field"))
}

fn u64_to_usize(value: u64) -> Result<usize, EngineError> {
    usize::try_from(value)
        .map_err(|_| EngineError::resource_exhaustion("fixture offset is not addressable"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use personalai_engine_crypto::{hash_signed_content, verify_package_signature};

    #[test]
    fn fixture_is_reproducible_parseable_and_strictly_signed() {
        let first = PackageFixtureBuilder::new().build().expect("fixture must build");
        let second = PackageFixtureBuilder::new().build().expect("fixture must build");
        assert_eq!(first, second);
        let parsed = ParsedModelPackage::parse(first.bytes()).expect("fixture must parse");
        assert_eq!(parsed.tensors().len(), 2);
        assert_eq!(first.package_hash(), hash_package(first.bytes()));
        let signed_range = parsed.header().signed_content_range().expect("range must validate");
        let digest = hash_signed_content(&first.bytes()[signed_range]);
        verify_package_signature(
            first.trusted_signer(),
            parsed.signature().key_id(),
            &digest,
            parsed.signature().signature(),
        )
        .expect("fixture signature must verify");
    }

    #[test]
    fn fixture_settings_remain_bounded_and_tampering_is_detected() {
        assert!(PackageFixtureBuilder::new().with_tensor_count(0).is_err());
        let fixture = PackageFixtureBuilder::new()
            .with_tensor_count(3)
            .expect("count must validate")
            .build()
            .expect("fixture must build");
        let parsed = ParsedModelPackage::parse(fixture.bytes()).expect("fixture must parse");
        assert_eq!(parsed.tensors().len(), 3);
        let mut tampered = fixture.to_bytes();
        let tokenizer_start =
            parsed.manifest().tokenizer_range().expect("tokenizer range must validate").start;
        tampered[tokenizer_start] ^= 1;
        assert_ne!(fixture.package_hash(), hash_package(&tampered));
    }

    #[test]
    fn decoder_fixture_has_the_exact_first_party_weight_schema() {
        let fixture = DecoderPackageFixtureBuilder::new().build().expect("decoder must build");
        let parsed = ParsedModelPackage::parse(fixture.bytes()).expect("decoder must parse");
        assert_eq!(parsed.manifest().architecture_id().as_str(), "decoder-rms-rope-swiglu-gqa-v1");
        assert_eq!(parsed.manifest().vocabulary_size(), 259);
        assert_eq!(parsed.manifest().layer_count(), 2);
        assert_eq!(parsed.manifest().attention_head_count(), 2);
        assert_eq!(parsed.manifest().kv_head_count(), 1);
        assert_eq!(parsed.tensors().len(), 21);
        assert!(parsed.tensors().iter().any(|tensor| tensor.name().as_str() == "output.weight"));
        assert!(
            parsed
                .tensors()
                .iter()
                .any(|tensor| tensor.name().as_str() == "layers.001.down_proj.weight")
        );
    }

    #[test]
    fn deterministic_tensor_and_rng_repeat_exactly() {
        let first_pool = MemoryPool::new(4096).unwrap();
        let second_pool = MemoryPool::new(4096).unwrap();
        let shape = TensorShape::new([2, 3]).unwrap();
        let first = deterministic_tensor(shape.clone(), 42, &first_pool).unwrap();
        let second = deterministic_tensor(shape, 42, &second_pool).unwrap();
        compare_tensors(&first, &second, 0.0, 0.0).unwrap();
        let mut first_rng = DeterministicRng::new(7);
        let mut second_rng = DeterministicRng::new(7);
        for _ in 0..100 {
            assert_eq!(first_rng.next_u64(), second_rng.next_u64());
        }
    }

    #[test]
    fn tensor_comparison_reports_mismatch_and_nonfinite_values() {
        let pool = MemoryPool::new(4096).unwrap();
        let shape = TensorShape::new([2]).unwrap();
        let expected = MutableTensor::from_f32_values(shape.clone(), &[1.0, 2.0], &pool).unwrap();
        let close = MutableTensor::from_f32_values(shape.clone(), &[1.0, 2.000_01], &pool).unwrap();
        compare_tensors(&close, &expected, 1.0e-4, 1.0e-4).unwrap();
        let far = MutableTensor::from_f32_values(shape.clone(), &[1.0, 3.0], &pool).unwrap();
        assert!(compare_tensors(&far, &expected, 1.0e-4, 1.0e-4).is_err());
        let nan = MutableTensor::from_f32_values(shape, &[1.0, f32::NAN], &pool).unwrap();
        assert!(compare_tensors(&nan, &expected, 0.0, 0.0).is_err());
    }

    #[test]
    fn reference_node_helper_executes_allowlisted_operation() {
        let pool = MemoryPool::new(4096).unwrap();
        let shape = TensorShape::new([2]).unwrap();
        let left = MutableTensor::from_f32_values(shape.clone(), &[1.0, 2.0], &pool).unwrap();
        let right = MutableTensor::from_f32_values(shape, &[3.0, 4.0], &pool).unwrap();
        let output = execute_reference_node(&Operation::Add, &[&left, &right], &pool).unwrap();
        assert_eq!(output.f32_value(0).unwrap().to_bits(), 4.0_f32.to_bits());
        assert_eq!(output.f32_value(1).unwrap().to_bits(), 6.0_f32.to_bits());
    }
}
