//! Bounded canonical parser for Model Foundry registry V2 envelopes.

use personalai_engine_types::{ContextLimit, EngineError, ModelPackageHash, SignerKeyId};

/// Maximum accepted signed registry-envelope bytes.
pub const MODEL_REGISTRY_MAX_BYTES: usize = 64 * 1024;
/// Exact schema version implemented by this parser.
pub const MODEL_REGISTRY_SCHEMA_V2: u64 = 2;
/// Exact directory-package compatibility selected by registry V2.
pub const DIRECTORY_ENGINE_COMPATIBILITY: &str =
    "directory-paimodel-v1+decoder-rms-rope-swiglu-gqa-v1+paibpe01+f32-cpu-reference";
/// Exact minimum engine version encoded by the frozen V2 contract.
pub const MODEL_REGISTRY_MINIMUM_ENGINE_VERSION: &str = "0.1.0";
/// Maximum bounded capability tags.
const MAX_CAPABILITY_TAGS: usize = 32;
/// Maximum bounded tensor shards.
const MAX_TENSOR_SHARDS: usize = 1_024;

/// One strictly parsed unsigned registry decision.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModelRegistryEntryV2 {
    approval_timestamp: String,
    artifact_id: String,
    audit_head: ModelPackageHash,
    audit_record_count: u64,
    capability_tags: Vec<String>,
    context_capacity: u32,
    model_id: String,
    package_catalog_hash: ModelPackageHash,
    package_signer: SignerKeyId,
    provenance_hash: ModelPackageHash,
    registry_sequence: u64,
    registry_signer: SignerKeyId,
    tensor_shard_hashes: Vec<ModelPackageHash>,
    tokenizer_hash: ModelPackageHash,
}

impl ModelRegistryEntryV2 {
    /// Returns the owner-selected registry sequence.
    pub const fn registry_sequence(&self) -> u64 {
        self.registry_sequence
    }

    /// Returns the stable model ID.
    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    /// Returns the stable lifecycle artifact ID.
    pub fn artifact_id(&self) -> &str {
        &self.artifact_id
    }

    /// Returns SHA-384 of exact `hashes.aiph` bytes.
    pub const fn package_catalog_hash(&self) -> ModelPackageHash {
        self.package_catalog_hash
    }

    /// Returns the package signer selected by the signed entry.
    pub const fn package_signer(&self) -> SignerKeyId {
        self.package_signer
    }

    /// Returns the registry signer selected by the signed entry.
    pub const fn registry_signer(&self) -> SignerKeyId {
        self.registry_signer
    }

    /// Returns the package context capacity.
    pub const fn context_capacity(&self) -> u32 {
        self.context_capacity
    }

    /// Returns the approval timestamp as bounded ASCII data.
    pub fn approval_timestamp(&self) -> &str {
        &self.approval_timestamp
    }

    /// Returns the signed audit record count.
    pub const fn audit_record_count(&self) -> u64 {
        self.audit_record_count
    }

    /// Returns the signed audit-head digest.
    pub const fn audit_head(&self) -> ModelPackageHash {
        self.audit_head
    }

    /// Returns the signed provenance digest.
    pub const fn provenance_hash(&self) -> ModelPackageHash {
        self.provenance_hash
    }

    /// Returns the signed tokenizer digest.
    pub const fn tokenizer_hash(&self) -> ModelPackageHash {
        self.tokenizer_hash
    }

    /// Returns sorted capability tags.
    pub fn capability_tags(&self) -> &[String] {
        &self.capability_tags
    }

    /// Returns sorted tensor-shard digests.
    pub fn tensor_shard_hashes(&self) -> &[ModelPackageHash] {
        &self.tensor_shard_hashes
    }
}

/// One exact canonical signed registry envelope.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SignedModelRegistryEntryV2 {
    entry: ModelRegistryEntryV2,
    entry_range: std::ops::Range<usize>,
    signature: [u8; 64],
}

impl SignedModelRegistryEntryV2 {
    /// Parses one exact canonical envelope without allocating from external sizes.
    pub fn parse(bytes: &[u8]) -> Result<Self, EngineError> {
        if bytes.is_empty() || bytes.len() > MODEL_REGISTRY_MAX_BYTES {
            return Err(corrupt("registry envelope length is outside hard bounds"));
        }
        let mut parser = Parser::new(bytes);
        parser.expect(b"{\"entry\":")?;
        let entry_start = parser.position();
        let entry = parse_entry(&mut parser)?;
        let entry_end = parser.position();
        parser.expect(b",\"signature\":\"")?;
        let signature = parser.hex_array::<64>("registry signature")?;
        parser.expect(b"\"}")?;
        if !parser.is_finished() {
            return Err(corrupt("registry envelope contains trailing bytes"));
        }
        Ok(Self { entry, entry_range: entry_start..entry_end, signature })
    }

    /// Returns the validated unsigned entry.
    pub const fn entry(&self) -> &ModelRegistryEntryV2 {
        &self.entry
    }

    /// Returns exact canonical unsigned entry bytes from the envelope.
    pub fn entry_bytes<'a>(&self, envelope: &'a [u8]) -> Result<&'a [u8], EngineError> {
        envelope
            .get(self.entry_range.clone())
            .ok_or_else(|| corrupt("registry entry range is unavailable"))
    }

    /// Returns exact Ed25519 signature bytes.
    pub const fn signature(&self) -> &[u8; 64] {
        &self.signature
    }
}

fn parse_entry(parser: &mut Parser<'_>) -> Result<ModelRegistryEntryV2, EngineError> {
    parser.expect(b"{\"approval_timestamp\":\"")?;
    let approval_timestamp = parser.ascii_string(20, "approval timestamp")?;
    require_canonical_timestamp(&approval_timestamp)?;
    parser.expect(b"\",\"architecture\":\"")?;
    parser.exact_string("decoder-rms-rope-swiglu-gqa-v1", "architecture")?;
    parser.expect(b"\",\"artifact_id\":\"")?;
    let artifact_id = parser.identifier(128, "artifact ID")?;
    parser.expect(b"\",\"audit_head_sha384\":\"")?;
    let audit_head = parser.nonzero_hash("audit head")?;
    parser.expect(b"\",\"audit_record_count\":")?;
    let audit_record_count = parser.positive_u64("audit record count")?;
    parser.expect(b",\"capability_tags\":[")?;
    let capability_tags = parser.string_array(MAX_CAPABILITY_TAGS, "capability tag")?;
    require_sorted_unique_text(&capability_tags, "capability tags")?;
    parser.expect(b"],\"context_capacity\":")?;
    let context_capacity = parser.positive_u32("context capacity")?;
    ContextLimit::new(context_capacity)
        .map_err(|_| corrupt("context capacity is outside engine bounds"))?;
    parser.expect(b",\"engine_compatibility\":\"")?;
    parser.exact_string(DIRECTORY_ENGINE_COMPATIBILITY, "engine compatibility")?;
    parser.expect(b"\",\"minimum_engine_version\":\"")?;
    parser.exact_string(MODEL_REGISTRY_MINIMUM_ENGINE_VERSION, "minimum engine version")?;
    parser.expect(b"\",\"model_id\":\"")?;
    let model_id = parser.identifier(128, "model ID")?;
    parser.expect(b"\",\"package_catalog_sha384\":\"")?;
    let package_catalog_hash = parser.nonzero_hash("package catalog")?;
    parser.expect(b"\",\"package_signer\":\"")?;
    let package_signer = parser.key_id("package signer")?;
    parser.expect(b"\",\"provenance_sha384\":\"")?;
    let provenance_hash = parser.nonzero_hash("provenance")?;
    parser.expect(b"\",\"quantization\":\"")?;
    parser.exact_string("f32", "quantization")?;
    parser.expect(b"\",\"registry_sequence\":")?;
    let registry_sequence = parser.positive_u64("registry sequence")?;
    parser.expect(b",\"registry_signer\":\"")?;
    let registry_signer = parser.key_id("registry signer")?;
    parser.expect(b"\",\"schema_version\":")?;
    let schema_version = parser.positive_u64("registry schema")?;
    if schema_version != MODEL_REGISTRY_SCHEMA_V2 {
        return Err(corrupt("registry schema version is unsupported"));
    }
    parser.expect(b",\"tensor_shard_sha384\":[")?;
    let tensor_shard_hashes = parser.hash_array(MAX_TENSOR_SHARDS, "tensor shard")?;
    require_sorted_unique_hashes(&tensor_shard_hashes)?;
    parser.expect(b"],\"tokenizer_sha384\":\"")?;
    let tokenizer_hash = parser.nonzero_hash("tokenizer")?;
    parser.expect(b"\"}")?;
    Ok(ModelRegistryEntryV2 {
        approval_timestamp,
        artifact_id,
        audit_head,
        audit_record_count,
        capability_tags,
        context_capacity,
        model_id,
        package_catalog_hash,
        package_signer,
        provenance_hash,
        registry_sequence,
        registry_signer,
        tensor_shard_hashes,
        tokenizer_hash,
    })
}

pub(crate) struct Parser<'a> {
    bytes: &'a [u8],
    cursor: usize,
}

impl<'a> Parser<'a> {
    pub(crate) const fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, cursor: 0 }
    }

    pub(crate) const fn position(&self) -> usize {
        self.cursor
    }

    pub(crate) fn is_finished(&self) -> bool {
        self.cursor == self.bytes.len()
    }

    pub(crate) fn expect(&mut self, expected: &[u8]) -> Result<(), EngineError> {
        let end = self
            .cursor
            .checked_add(expected.len())
            .ok_or_else(|| corrupt("registry parser offset overflows"))?;
        if self.bytes.get(self.cursor..end) != Some(expected) {
            return Err(corrupt("registry JSON is noncanonical or has an unexpected field"));
        }
        self.cursor = end;
        Ok(())
    }

    fn take_until_quote(&mut self, maximum: usize, label: &str) -> Result<&'a [u8], EngineError> {
        let remaining = self
            .bytes
            .get(self.cursor..)
            .ok_or_else(|| corrupt("registry string offset is unavailable"))?;
        let length = remaining
            .iter()
            .position(|byte| *byte == b'\"')
            .ok_or_else(|| corrupt(format!("{label} is unterminated")))?;
        if length == 0 || length > maximum {
            return Err(corrupt(format!("{label} length is outside hard bounds")));
        }
        let end = self
            .cursor
            .checked_add(length)
            .ok_or_else(|| corrupt("registry string range overflows"))?;
        let value = self
            .bytes
            .get(self.cursor..end)
            .ok_or_else(|| corrupt("registry string range is unavailable"))?;
        if value.iter().any(|byte| !byte.is_ascii_graphic() || matches!(byte, b'\"' | b'\\')) {
            return Err(corrupt(format!("{label} contains noncanonical text")));
        }
        self.cursor = end;
        Ok(value)
    }

    pub(crate) fn ascii_string(
        &mut self,
        maximum: usize,
        label: &str,
    ) -> Result<String, EngineError> {
        let value = self.take_until_quote(maximum, label)?;
        String::from_utf8(value.to_vec()).map_err(|_| corrupt(format!("{label} is invalid UTF-8")))
    }

    pub(crate) fn exact_string(&mut self, expected: &str, label: &str) -> Result<(), EngineError> {
        let observed = self.take_until_quote(expected.len(), label)?;
        if observed != expected.as_bytes() {
            return Err(corrupt(format!("{label} is unsupported")));
        }
        Ok(())
    }

    pub(crate) fn identifier(
        &mut self,
        maximum: usize,
        label: &str,
    ) -> Result<String, EngineError> {
        let value = self.take_until_quote(maximum, label)?;
        if !matches!(value[0], b'a'..=b'z' | b'0'..=b'9')
            || value
                .iter()
                .any(|byte| !matches!(byte, b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.'))
            || value.windows(2).any(|window| window == b"..")
        {
            return Err(corrupt(format!("{label} is not a safe canonical identifier")));
        }
        String::from_utf8(value.to_vec()).map_err(|_| corrupt(format!("{label} is invalid UTF-8")))
    }

    pub(crate) fn positive_u64(&mut self, label: &str) -> Result<u64, EngineError> {
        let start = self.cursor;
        while self.bytes.get(self.cursor).is_some_and(u8::is_ascii_digit) {
            self.cursor = self
                .cursor
                .checked_add(1)
                .ok_or_else(|| corrupt("registry number offset overflows"))?;
        }
        let digits = self
            .bytes
            .get(start..self.cursor)
            .ok_or_else(|| corrupt("registry number range is unavailable"))?;
        if digits.is_empty() || digits[0] == b'0' {
            return Err(corrupt(format!("{label} is not a canonical positive integer")));
        }
        let text =
            std::str::from_utf8(digits).map_err(|_| corrupt("registry integer is invalid"))?;
        text.parse::<u64>().map_err(|_| corrupt(format!("{label} overflows")))
    }

    fn positive_u32(&mut self, label: &str) -> Result<u32, EngineError> {
        u32::try_from(self.positive_u64(label)?).map_err(|_| corrupt(format!("{label} overflows")))
    }

    pub(crate) fn key_id(&mut self, label: &str) -> Result<SignerKeyId, EngineError> {
        let bytes = self.hex_array::<16>(label)?;
        SignerKeyId::from_bytes(bytes).map_err(|_| corrupt(format!("{label} is invalid")))
    }

    fn hash(&mut self, label: &str) -> Result<ModelPackageHash, EngineError> {
        Ok(ModelPackageHash::from_sha384_bytes(self.hex_array::<48>(label)?))
    }

    pub(crate) fn nonzero_hash(&mut self, label: &str) -> Result<ModelPackageHash, EngineError> {
        let hash = self.hash(label)?;
        if hash.as_bytes().iter().all(|byte| *byte == 0) {
            return Err(corrupt(format!("{label} hash must not be zero")));
        }
        Ok(hash)
    }

    pub(crate) fn hex_array<const N: usize>(
        &mut self,
        label: &str,
    ) -> Result<[u8; N], EngineError> {
        let length =
            N.checked_mul(2).ok_or_else(|| corrupt("registry hexadecimal length overflows"))?;
        let end = self
            .cursor
            .checked_add(length)
            .ok_or_else(|| corrupt("registry hexadecimal range overflows"))?;
        let encoded = self
            .bytes
            .get(self.cursor..end)
            .ok_or_else(|| corrupt(format!("{label} hexadecimal field is truncated")))?;
        let mut output = [0_u8; N];
        for (index, pair) in encoded.chunks_exact(2).enumerate() {
            let high = nibble(pair[0])
                .ok_or_else(|| corrupt(format!("{label} is not lowercase hexadecimal")))?;
            let low = nibble(pair[1])
                .ok_or_else(|| corrupt(format!("{label} is not lowercase hexadecimal")))?;
            output[index] = (high << 4) | low;
        }
        self.cursor = end;
        Ok(output)
    }

    fn string_array(&mut self, maximum: usize, label: &str) -> Result<Vec<String>, EngineError> {
        let mut values = Vec::new();
        loop {
            if self.bytes.get(self.cursor) == Some(&b']') {
                break;
            }
            if !values.is_empty() {
                self.expect(b",")?;
            }
            self.expect(b"\"")?;
            values.push(self.identifier(64, label)?);
            self.expect(b"\"")?;
            if values.len() > maximum {
                return Err(corrupt(format!("{label} count exceeds its hard bound")));
            }
        }
        if values.is_empty() {
            return Err(corrupt(format!("{label} array must not be empty")));
        }
        Ok(values)
    }

    fn hash_array(
        &mut self,
        maximum: usize,
        label: &str,
    ) -> Result<Vec<ModelPackageHash>, EngineError> {
        let mut values = Vec::new();
        loop {
            if self.bytes.get(self.cursor) == Some(&b']') {
                break;
            }
            if !values.is_empty() {
                self.expect(b",")?;
            }
            self.expect(b"\"")?;
            values.push(self.nonzero_hash(label)?);
            self.expect(b"\"")?;
            if values.len() > maximum {
                return Err(corrupt(format!("{label} count exceeds its hard bound")));
            }
        }
        if values.is_empty() {
            return Err(corrupt(format!("{label} array must not be empty")));
        }
        Ok(values)
    }
}

pub(crate) fn require_canonical_timestamp(value: &str) -> Result<(), EngineError> {
    let bytes = value.as_bytes();
    if bytes.len() != 20
        || bytes[4] != b'-'
        || bytes[7] != b'-'
        || bytes[10] != b'T'
        || bytes[13] != b':'
        || bytes[16] != b':'
        || bytes[19] != b'Z'
        || bytes.iter().enumerate().any(|(index, byte)| {
            !matches!(index, 4 | 7 | 10 | 13 | 16 | 19) && !byte.is_ascii_digit()
        })
    {
        return Err(corrupt("approval timestamp is not canonical UTC"));
    }
    Ok(())
}

fn require_sorted_unique_text(values: &[String], label: &str) -> Result<(), EngineError> {
    if values.windows(2).any(|pair| pair[0] >= pair[1]) {
        return Err(corrupt(format!("{label} are not sorted and unique")));
    }
    Ok(())
}

fn require_sorted_unique_hashes(values: &[ModelPackageHash]) -> Result<(), EngineError> {
    if values.windows(2).any(|pair| pair[0] >= pair[1]) {
        return Err(corrupt("tensor shard hashes are not sorted and unique"));
    }
    Ok(())
}

const fn nibble(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        _ => None,
    }
}

pub(crate) fn corrupt(detail: impl AsRef<str>) -> EngineError {
    EngineError::corrupt_model_package(detail)
}

#[cfg(test)]
mod tests {
    use super::*;

    const POSITIVE_FILE: &[u8] = include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/model-foundry-registry-v2/positive-0002.registry.json"
    ));

    fn positive() -> &'static [u8] {
        POSITIVE_FILE.strip_suffix(b"\n").unwrap_or(POSITIVE_FILE)
    }

    #[test]
    fn canonical_fixture_parses_and_exposes_exact_selection() {
        let fixture = positive();
        let signed = SignedModelRegistryEntryV2::parse(fixture).expect("fixture must parse");
        assert_eq!(signed.entry().registry_sequence(), 2);
        assert_eq!(signed.entry().model_id(), "synthetic-import");
        assert_eq!(signed.entry().artifact_id(), "phase0-synthetic-import");
        assert_eq!(signed.entry().context_capacity(), 8);
        assert_eq!(signed.entry().package_signer().to_string(), "51".repeat(16));
        assert_eq!(signed.entry().registry_signer().to_string(), "63".repeat(16));
        assert_eq!(signed.entry_bytes(fixture).expect("entry range")[0], b'{');
    }

    #[test]
    fn every_byte_mutation_and_truncation_fails_or_changes_signed_entry() {
        let fixture = positive();
        for end in 0..fixture.len() {
            assert!(SignedModelRegistryEntryV2::parse(&fixture[..end]).is_err());
        }
        for index in 0..fixture.len() {
            let mut mutation = fixture.to_vec();
            mutation[index] ^= 1;
            if let Ok(parsed) = SignedModelRegistryEntryV2::parse(&mutation) {
                assert!(
                    parsed.entry_bytes(&mutation).expect("entry range")
                        != parsed.entry_bytes(fixture).expect("entry range")
                        || parsed.signature()
                            != SignedModelRegistryEntryV2::parse(fixture)
                                .expect("fixture")
                                .signature()
                );
            }
        }
    }

    #[test]
    fn registry_v2_remains_closed_to_the_legacy_package_profile() {
        let fixture = std::str::from_utf8(positive()).expect("fixture must be UTF-8");
        let qwen = fixture.replace(
            DIRECTORY_ENGINE_COMPATIBILITY,
            "directory-paimodel-v1.1+qwen2.5-rms-rope-swiglu-gqa-bias-tied-v1+aiptok01+f32-cpu-reference",
        );
        assert_ne!(qwen.as_bytes(), positive());
        assert!(SignedModelRegistryEntryV2::parse(qwen.as_bytes()).is_err());
    }
}
