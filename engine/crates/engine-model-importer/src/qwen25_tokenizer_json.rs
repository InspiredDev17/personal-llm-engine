//! Closed inspection gate for the pinned Qwen2.5 byte-level BPE source tokenizer.
//!
//! The source graph requires exact NFC normalization plus a Unicode-aware fixed
//! pre-tokenizer. Frozen `AIPTOK01` deliberately supports neither, requires an
//! unknown-token ID, and assumes tokenizer/model vocabulary equality. The
//! separate `AIPTOK02` parser validates data tables but is not yet executable.
//! Consequently this module validates the pinned source without emitting an
//! approved tokenizer package. [`convert`] always fails closed after validation.

use crate::bounded_json::JsonCursor;
use personalai_engine_crypto::hash_sha256;
use personalai_engine_tokenizer::{
    AIPTOK_QWEN25_PINNED_ADDED_TOKEN_COUNT, AIPTOK_QWEN25_PINNED_BOS_TOKEN_ID,
    AIPTOK_QWEN25_PINNED_BPE_VOCABULARY_SIZE, AIPTOK_QWEN25_PINNED_DEFINED_TOKEN_COUNT,
    AIPTOK_QWEN25_PINNED_EOS_TOKEN_ID, AIPTOK_QWEN25_PINNED_MERGE_COUNT,
    AIPTOK_QWEN25_PINNED_MODEL_VOCABULARY_SIZE,
};
use personalai_engine_types::{EngineError, Sha256Digest};
use std::collections::{HashMap, HashSet};

/// Exact explicit source-format selector for the pinned Qwen2.5 tokenizer JSON.
pub const QWEN25_TOKENIZER_JSON_SOURCE_FORMAT: &str = "qwen2.5-tokenizer-json-v1";
/// SHA-256 of the tokenizer shared by the three pinned Qwen2.5 1.5B startup artifacts.
pub const QWEN25_TOKENIZER_SHA256: &str =
    "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539";
/// Exact source tokenizer byte length.
pub const QWEN25_TOKENIZER_BYTES: usize = 7_031_645;
/// BPE-model vocabulary entries in the pinned tokenizer source.
pub const QWEN25_BPE_VOCABULARY_SIZE: u32 = AIPTOK_QWEN25_PINNED_BPE_VOCABULARY_SIZE;
/// Added-token entries in the pinned tokenizer source.
pub const QWEN25_ADDED_TOKEN_COUNT: u32 = AIPTOK_QWEN25_PINNED_ADDED_TOKEN_COUNT;
/// Total token IDs explicitly defined by the tokenizer source.
pub const QWEN25_DEFINED_TOKEN_COUNT: u32 = AIPTOK_QWEN25_PINNED_DEFINED_TOKEN_COUNT;
/// Embedding rows declared by the paired Qwen2.5 model configuration.
pub const QWEN25_MODEL_VOCABULARY_SIZE: u32 = AIPTOK_QWEN25_PINNED_MODEL_VOCABULARY_SIZE;
/// Exact BPE merge count in the pinned tokenizer source.
pub const QWEN25_MERGE_COUNT: u32 = AIPTOK_QWEN25_PINNED_MERGE_COUNT;
/// Qwen2.5 end-of-text/BOS-compatible token ID. It is not inserted implicitly.
pub const QWEN25_BOS_TOKEN_ID: u32 = AIPTOK_QWEN25_PINNED_BOS_TOKEN_ID;
/// Qwen2.5 instruction end token ID.
pub const QWEN25_EOS_TOKEN_ID: u32 = AIPTOK_QWEN25_PINNED_EOS_TOKEN_ID;

const QWEN25_PRETOKENIZER_REGEX: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
const MAX_SOURCE_TOKEN_BYTES: usize = 256;
const MAX_SOURCE_MERGE_BYTES: usize = 512;

const ADDED_TOKENS: [(&str, bool); 22] = [
    ("<|endoftext|>", true),
    ("<|im_start|>", true),
    ("<|im_end|>", true),
    ("<|object_ref_start|>", true),
    ("<|object_ref_end|>", true),
    ("<|box_start|>", true),
    ("<|box_end|>", true),
    ("<|quad_start|>", true),
    ("<|quad_end|>", true),
    ("<|vision_start|>", true),
    ("<|vision_end|>", true),
    ("<|vision_pad|>", true),
    ("<|image_pad|>", true),
    ("<|video_pad|>", true),
    ("<tool_call>", false),
    ("</tool_call>", false),
    ("<|fim_prefix|>", false),
    ("<|fim_middle|>", false),
    ("<|fim_suffix|>", false),
    ("<|fim_pad|>", false),
    ("<|repo_name|>", false),
    ("<|file_sep|>", false),
];

/// Content-free facts proven by the closed source tokenizer parser.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Qwen25TokenizerDescriptor {
    source_sha256: Sha256Digest,
    bpe_vocabulary_size: u32,
    defined_token_count: u32,
    model_vocabulary_size: u32,
    merge_count: u32,
}

impl Qwen25TokenizerDescriptor {
    /// Returns the exact pinned source identity.
    pub const fn source_sha256(self) -> Sha256Digest {
        self.source_sha256
    }

    /// Returns the BPE model vocabulary count.
    pub const fn bpe_vocabulary_size(self) -> u32 {
        self.bpe_vocabulary_size
    }

    /// Returns the count of token IDs explicitly defined by tokenizer data.
    pub const fn defined_token_count(self) -> u32 {
        self.defined_token_count
    }

    /// Returns the paired model embedding-row count.
    pub const fn model_vocabulary_size(self) -> u32 {
        self.model_vocabulary_size
    }

    /// Returns the exact BPE merge count.
    pub const fn merge_count(self) -> u32 {
        self.merge_count
    }

    /// Qwen2.5 has no unknown token in this source profile.
    pub const fn unknown_token(self) -> Option<u32> {
        None
    }

    /// Qwen2.5 source tokenization does not prepend BOS implicitly.
    pub const fn inserts_bos(self) -> bool {
        false
    }
}

#[cfg(test)]
pub(crate) fn pinned_test_descriptor() -> Qwen25TokenizerDescriptor {
    Qwen25TokenizerDescriptor {
        source_sha256: Sha256Digest::from_hex(QWEN25_TOKENIZER_SHA256)
            .expect("pinned tokenizer digest must remain valid"),
        bpe_vocabulary_size: QWEN25_BPE_VOCABULARY_SIZE,
        defined_token_count: QWEN25_DEFINED_TOKEN_COUNT,
        model_vocabulary_size: QWEN25_MODEL_VOCABULARY_SIZE,
        merge_count: QWEN25_MERGE_COUNT,
    }
}

/// Parses and validates the exact pinned Qwen2.5 tokenizer source graph.
pub fn inspect(bytes: &[u8]) -> Result<Qwen25TokenizerDescriptor, EngineError> {
    if bytes.len() != QWEN25_TOKENIZER_BYTES
        || hash_sha256(bytes).to_string() != QWEN25_TOKENIZER_SHA256
    {
        return Err(corrupt("Qwen2.5 tokenizer source identity is not pinned"));
    }
    let expected = Expectations {
        vocabulary_size: QWEN25_BPE_VOCABULARY_SIZE,
        merge_count: QWEN25_MERGE_COUNT,
        byte_base_count: 256,
        added_tokens: &ADDED_TOKENS,
        first_added_id: QWEN25_BPE_VOCABULARY_SIZE,
    };
    parse_graph(bytes, expected)?;
    Ok(Qwen25TokenizerDescriptor {
        source_sha256: hash_sha256(bytes),
        bpe_vocabulary_size: QWEN25_BPE_VOCABULARY_SIZE,
        defined_token_count: QWEN25_DEFINED_TOKEN_COUNT,
        model_vocabulary_size: QWEN25_MODEL_VOCABULARY_SIZE,
        merge_count: QWEN25_MERGE_COUNT,
    })
}

/// Transitional conversion gate.
///
/// The complete source is inspected before the current runtime incompatibility
/// is reported. No partial or lossy `tokenizer.aiptok` is ever returned.
pub fn convert(bytes: &[u8]) -> Result<Vec<u8>, EngineError> {
    let _ = inspect(bytes)?;
    Err(EngineError::unsupported_tokenizer(
        "AIPTok v2 table parsing exists, but Qwen2.5 conversion remains disabled until exact NFC/category pre-tokenization and executable encode/decode semantics are qualified",
    ))
}

#[derive(Clone, Copy)]
struct Expectations<'a> {
    vocabulary_size: u32,
    merge_count: u32,
    byte_base_count: u32,
    added_tokens: &'a [(&'a str, bool)],
    first_added_id: u32,
}

fn parse_graph(bytes: &[u8], expected: Expectations<'_>) -> Result<(), EngineError> {
    let mut cursor = JsonCursor::new(bytes)?;
    cursor.expect(b'{')?;
    exact_key_string(&mut cursor, "version", "1.0")?;
    comma(&mut cursor)?;
    exact_key_null(&mut cursor, "truncation")?;
    comma(&mut cursor)?;
    exact_key_null(&mut cursor, "padding")?;
    comma(&mut cursor)?;
    exact_key(&mut cursor, "added_tokens")?;
    parse_added_tokens(&mut cursor, expected)?;
    comma(&mut cursor)?;
    parse_normalizer(&mut cursor)?;
    comma(&mut cursor)?;
    parse_pre_tokenizer(&mut cursor)?;
    comma(&mut cursor)?;
    parse_byte_level_object(&mut cursor, "post_processor")?;
    comma(&mut cursor)?;
    parse_byte_level_object(&mut cursor, "decoder")?;
    comma(&mut cursor)?;
    exact_key(&mut cursor, "model")?;
    parse_model(&mut cursor, expected)?;
    cursor.expect(b'}')?;
    cursor.finish()
}

fn parse_added_tokens(
    cursor: &mut JsonCursor<'_>,
    expected: Expectations<'_>,
) -> Result<(), EngineError> {
    cursor.expect(b'[')?;
    for (index, (content, special)) in expected.added_tokens.iter().enumerate() {
        if index != 0 {
            comma(cursor)?;
        }
        cursor.expect(b'{')?;
        let mut id = None;
        let mut observed_content = None;
        let mut single_word = None;
        let mut left_strip = None;
        let mut right_strip = None;
        let mut normalized = None;
        let mut observed_special = None;
        loop {
            let key = cursor.key(32)?;
            match key.as_str() {
                "id" if id.is_none() => id = Some(cursor.u32()?),
                "content" if observed_content.is_none() => {
                    observed_content = Some(cursor.string(MAX_SOURCE_TOKEN_BYTES)?);
                }
                "single_word" if single_word.is_none() => single_word = Some(cursor.boolean()?),
                "lstrip" if left_strip.is_none() => left_strip = Some(cursor.boolean()?),
                "rstrip" if right_strip.is_none() => right_strip = Some(cursor.boolean()?),
                "normalized" if normalized.is_none() => normalized = Some(cursor.boolean()?),
                "special" if observed_special.is_none() => {
                    observed_special = Some(cursor.boolean()?);
                }
                _ => return Err(corrupt("Qwen2.5 added-token field is unknown or duplicated")),
            }
            if cursor.consume(b'}') {
                break;
            }
            comma(cursor)?;
        }
        let expected_id = expected
            .first_added_id
            .checked_add(u32::try_from(index).map_err(|_| corrupt("added-token index overflows"))?)
            .ok_or_else(|| corrupt("added-token ID overflows"))?;
        if id != Some(expected_id)
            || observed_content.as_deref() != Some(*content)
            || single_word != Some(false)
            || left_strip != Some(false)
            || right_strip != Some(false)
            || normalized != Some(false)
            || observed_special != Some(*special)
        {
            return Err(corrupt("Qwen2.5 added-token semantics do not match the closed profile"));
        }
    }
    cursor.expect(b']')
}

fn parse_normalizer(cursor: &mut JsonCursor<'_>) -> Result<(), EngineError> {
    exact_key(cursor, "normalizer")?;
    cursor.expect(b'{')?;
    exact_key_string(cursor, "type", "NFC")?;
    cursor.expect(b'}')
}

fn parse_pre_tokenizer(cursor: &mut JsonCursor<'_>) -> Result<(), EngineError> {
    exact_key(cursor, "pre_tokenizer")?;
    cursor.expect(b'{')?;
    exact_key_string(cursor, "type", "Sequence")?;
    comma(cursor)?;
    exact_key(cursor, "pretokenizers")?;
    cursor.expect(b'[')?;
    cursor.expect(b'{')?;
    exact_key_string(cursor, "type", "Split")?;
    comma(cursor)?;
    exact_key(cursor, "pattern")?;
    cursor.expect(b'{')?;
    exact_key_string(cursor, "Regex", QWEN25_PRETOKENIZER_REGEX)?;
    cursor.expect(b'}')?;
    comma(cursor)?;
    exact_key_string(cursor, "behavior", "Isolated")?;
    comma(cursor)?;
    exact_key_bool(cursor, "invert", false)?;
    cursor.expect(b'}')?;
    comma(cursor)?;
    parse_byte_level_fields(cursor)?;
    cursor.expect(b']')?;
    cursor.expect(b'}')
}

fn parse_byte_level_object(
    cursor: &mut JsonCursor<'_>,
    outer_key: &str,
) -> Result<(), EngineError> {
    exact_key(cursor, outer_key)?;
    parse_byte_level_fields(cursor)
}

fn parse_byte_level_fields(cursor: &mut JsonCursor<'_>) -> Result<(), EngineError> {
    cursor.expect(b'{')?;
    exact_key_string(cursor, "type", "ByteLevel")?;
    comma(cursor)?;
    exact_key_bool(cursor, "add_prefix_space", false)?;
    comma(cursor)?;
    exact_key_bool(cursor, "trim_offsets", false)?;
    comma(cursor)?;
    exact_key_bool(cursor, "use_regex", false)?;
    cursor.expect(b'}')
}

fn parse_model(cursor: &mut JsonCursor<'_>, expected: Expectations<'_>) -> Result<(), EngineError> {
    cursor.expect(b'{')?;
    exact_key_string(cursor, "type", "BPE")?;
    comma(cursor)?;
    exact_key_null(cursor, "dropout")?;
    comma(cursor)?;
    exact_key_null(cursor, "unk_token")?;
    comma(cursor)?;
    exact_key_string(cursor, "continuing_subword_prefix", "")?;
    comma(cursor)?;
    exact_key_string(cursor, "end_of_word_suffix", "")?;
    comma(cursor)?;
    exact_key_bool(cursor, "fuse_unk", false)?;
    comma(cursor)?;
    exact_key_bool(cursor, "byte_fallback", false)?;
    comma(cursor)?;
    exact_key(cursor, "vocab")?;
    let vocabulary = parse_vocabulary(cursor, expected)?;
    comma(cursor)?;
    exact_key(cursor, "merges")?;
    parse_merges(cursor, expected, &vocabulary)?;
    cursor.expect(b'}')
}

fn parse_vocabulary(
    cursor: &mut JsonCursor<'_>,
    expected: Expectations<'_>,
) -> Result<HashMap<String, u32>, EngineError> {
    cursor.expect(b'{')?;
    let capacity = usize::try_from(expected.vocabulary_size)
        .map_err(|_| EngineError::resource_exhaustion("Qwen2.5 vocabulary is not addressable"))?;
    let mut vocabulary = HashMap::new();
    vocabulary
        .try_reserve(capacity)
        .map_err(|_| EngineError::resource_exhaustion("Qwen2.5 vocabulary allocation failed"))?;
    for expected_id in 0..expected.vocabulary_size {
        if expected_id != 0 {
            comma(cursor)?;
        }
        let token = cursor.key(MAX_SOURCE_TOKEN_BYTES)?;
        let token_id = cursor.u32()?;
        if token_id != expected_id || vocabulary.insert(token.clone(), token_id).is_some() {
            return Err(corrupt("Qwen2.5 vocabulary IDs or token bytes are noncanonical"));
        }
        if expected_id < expected.byte_base_count && token != byte_symbol_at_index(expected_id)? {
            return Err(corrupt("Qwen2.5 deterministic byte-symbol mapping is invalid"));
        }
    }
    if vocabulary.len() != capacity {
        return Err(corrupt("Qwen2.5 vocabulary count is invalid"));
    }
    cursor.expect(b'}')?;
    Ok(vocabulary)
}

fn parse_merges(
    cursor: &mut JsonCursor<'_>,
    expected: Expectations<'_>,
    vocabulary: &HashMap<String, u32>,
) -> Result<(), EngineError> {
    cursor.expect(b'[')?;
    let mut pairs = HashSet::new();
    pairs
        .try_reserve(usize::try_from(expected.merge_count).map_err(|_| {
            EngineError::resource_exhaustion("Qwen2.5 merge count is not addressable")
        })?)
        .map_err(|_| EngineError::resource_exhaustion("Qwen2.5 merge allocation failed"))?;
    for rank in 0..expected.merge_count {
        if rank != 0 {
            comma(cursor)?;
        }
        let merge = cursor.string(MAX_SOURCE_MERGE_BYTES)?;
        let (left, right) = merge
            .split_once(' ')
            .filter(|(left, right)| !left.is_empty() && !right.is_empty() && !right.contains(' '))
            .ok_or_else(|| corrupt("Qwen2.5 merge text is noncanonical"))?;
        let left_id = *vocabulary
            .get(left)
            .ok_or_else(|| corrupt("Qwen2.5 merge left token is unavailable"))?;
        let right_id = *vocabulary
            .get(right)
            .ok_or_else(|| corrupt("Qwen2.5 merge right token is unavailable"))?;
        if !pairs.insert((left_id, right_id)) {
            return Err(corrupt("Qwen2.5 merge pair is duplicated"));
        }
        let result = left
            .len()
            .checked_add(right.len())
            .filter(|length| *length <= MAX_SOURCE_TOKEN_BYTES)
            .ok_or_else(|| corrupt("Qwen2.5 merge result exceeds token bounds"))?;
        let mut combined = String::new();
        combined
            .try_reserve_exact(result)
            .map_err(|_| EngineError::resource_exhaustion("merge result allocation failed"))?;
        combined.push_str(left);
        combined.push_str(right);
        if !vocabulary.contains_key(&combined) {
            return Err(corrupt("Qwen2.5 merge result token is unavailable"));
        }
    }
    cursor.expect(b']')
}

fn byte_symbol(byte: u8) -> String {
    let scalar = if matches!(byte, b'!'..=b'~' | 0xa1..=0xac | 0xae..=0xff) {
        u32::from(byte)
    } else {
        let mut extra = 0_u32;
        for candidate in 0_u16..=u16::from(byte) {
            let candidate = u8::try_from(candidate).unwrap_or(u8::MAX);
            if !matches!(candidate, b'!'..=b'~' | 0xa1..=0xac | 0xae..=0xff) {
                extra += 1;
            }
        }
        0xff + extra
    };
    char::from_u32(scalar).unwrap_or(char::REPLACEMENT_CHARACTER).to_string()
}

fn byte_symbol_at_index(index: u32) -> Result<String, EngineError> {
    let mut observed = 0_u32;
    for allowed in [true, false] {
        for byte in 0_u16..=255 {
            let byte = u8::try_from(byte)
                .map_err(|_| corrupt("Qwen2.5 byte-symbol source is not representable"))?;
            let is_allowed = matches!(byte, b'!'..=b'~' | 0xa1..=0xac | 0xae..=0xff);
            if is_allowed == allowed {
                if observed == index {
                    return Ok(byte_symbol(byte));
                }
                observed = observed
                    .checked_add(1)
                    .ok_or_else(|| corrupt("Qwen2.5 byte-symbol index overflows"))?;
            }
        }
    }
    Err(corrupt("Qwen2.5 byte-symbol index is outside bounds"))
}

fn exact_key(cursor: &mut JsonCursor<'_>, expected: &str) -> Result<(), EngineError> {
    if cursor.key(64)? == expected {
        Ok(())
    } else {
        Err(corrupt("Qwen2.5 tokenizer field is unsupported or out of order"))
    }
}

fn exact_key_string(
    cursor: &mut JsonCursor<'_>,
    key: &str,
    expected: &str,
) -> Result<(), EngineError> {
    exact_key(cursor, key)?;
    if cursor.string(1_024)? == expected {
        Ok(())
    } else {
        Err(corrupt("Qwen2.5 tokenizer string value is unsupported"))
    }
}

fn exact_key_bool(
    cursor: &mut JsonCursor<'_>,
    key: &str,
    expected: bool,
) -> Result<(), EngineError> {
    exact_key(cursor, key)?;
    if cursor.boolean()? == expected {
        Ok(())
    } else {
        Err(corrupt("Qwen2.5 tokenizer boolean value is unsupported"))
    }
}

fn exact_key_null(cursor: &mut JsonCursor<'_>, key: &str) -> Result<(), EngineError> {
    exact_key(cursor, key)?;
    cursor.null()
}

fn comma(cursor: &mut JsonCursor<'_>) -> Result<(), EngineError> {
    cursor.expect(b',')
}

fn corrupt(detail: impl AsRef<str>) -> EngineError {
    EngineError::corrupt_model_package(detail)
}

#[cfg(test)]
mod tests {
    use super::*;

    const SMALL_ADDED: [(&str, bool); 1] = [("<|end|>", true)];

    fn fixture(regex: &str, unknown: &str) -> Vec<u8> {
        format!(
            concat!(
                "{{\"version\":\"1.0\",\"truncation\":null,\"padding\":null,",
                "\"added_tokens\":[{{\"id\":3,\"content\":\"<|end|>\",",
                "\"single_word\":false,\"lstrip\":false,\"rstrip\":false,",
                "\"normalized\":false,\"special\":true}}],",
                "\"normalizer\":{{\"type\":\"NFC\"}},",
                "\"pre_tokenizer\":{{\"type\":\"Sequence\",\"pretokenizers\":[",
                "{{\"type\":\"Split\",\"pattern\":{{\"Regex\":\"{}\"}},",
                "\"behavior\":\"Isolated\",\"invert\":false}},",
                "{{\"type\":\"ByteLevel\",\"add_prefix_space\":false,",
                "\"trim_offsets\":false,\"use_regex\":false}}]}},",
                "\"post_processor\":{{\"type\":\"ByteLevel\",\"add_prefix_space\":false,",
                "\"trim_offsets\":false,\"use_regex\":false}},",
                "\"decoder\":{{\"type\":\"ByteLevel\",\"add_prefix_space\":false,",
                "\"trim_offsets\":false,\"use_regex\":false}},",
                "\"model\":{{\"type\":\"BPE\",\"dropout\":null,\"unk_token\":{},",
                "\"continuing_subword_prefix\":\"\",\"end_of_word_suffix\":\"\",",
                "\"fuse_unk\":false,\"byte_fallback\":false,",
                "\"vocab\":{{\"a\":0,\"b\":1,\"ab\":2}},\"merges\":[\"a b\"]}}}}"
            ),
            regex.replace('\\', "\\\\"),
            unknown,
        )
        .into_bytes()
    }

    fn expectations() -> Expectations<'static> {
        Expectations {
            vocabulary_size: 3,
            merge_count: 1,
            byte_base_count: 0,
            added_tokens: &SMALL_ADDED,
            first_added_id: 3,
        }
    }

    #[test]
    fn closed_graph_accepts_only_exact_components_and_no_unknown_token() {
        let valid = fixture(QWEN25_PRETOKENIZER_REGEX, "null");
        let parsed = parse_graph(&valid, expectations());
        assert!(parsed.is_ok(), "{parsed:?}");
        assert!(parse_graph(&fixture(".+", "null"), expectations()).is_err());
        assert!(
            parse_graph(&fixture(QWEN25_PRETOKENIZER_REGEX, "\"<unk>\""), expectations()).is_err()
        );
    }

    #[test]
    fn production_entry_point_rejects_unpinned_and_converter_never_emits_lossy_data() {
        let fixture = fixture(QWEN25_PRETOKENIZER_REGEX, "null");
        assert!(inspect(&fixture).is_err());
        assert!(convert(&fixture).is_err());
    }

    #[test]
    fn deterministic_byte_symbol_map_is_total_and_unique() {
        let symbols = (0_u16..=255)
            .map(|byte| byte_symbol(u8::try_from(byte).unwrap()))
            .collect::<HashSet<_>>();
        assert_eq!(symbols.len(), 256);
        assert_eq!(byte_symbol(b' '), "Ġ");
        assert_eq!(byte_symbol(b'A'), "A");
    }
}
