//! Closed source converter for the pinned Qwen2.5 byte-level BPE tokenizer.
//!
//! The source graph requires exact NFC normalization plus a Unicode-aware fixed
//! pre-tokenizer. Frozen `AIPTOK01` deliberately supports neither. This module
//! retains the complete, pinned source tables and emits the separate canonical
//! data-only `AIPTOK02` representation. Public tokenizer construction, model
//! package emission, service selection, and runtime activation remain closed.

use crate::bounded_json::JsonCursor;
use personalai_engine_crypto::{hash_package, hash_sha256};
use personalai_engine_tokenizer::{
    AIPTOK_QWEN25_PINNED_ADDED_TOKEN_COUNT, AIPTOK_QWEN25_PINNED_BOS_TOKEN_ID,
    AIPTOK_QWEN25_PINNED_BPE_VOCABULARY_SIZE, AIPTOK_QWEN25_PINNED_DEFINED_TOKEN_COUNT,
    AIPTOK_QWEN25_PINNED_EOS_TOKEN_ID, AIPTOK_QWEN25_PINNED_MERGE_COUNT,
    AIPTOK_QWEN25_PINNED_MODEL_VOCABULARY_SIZE, AipTokQwen25AddedRole, AipTokQwen25AddedToken,
    AipTokQwen25Definition, AipTokQwen25Merge, AipTokQwen25VocabularyEntry,
};
use personalai_engine_types::{EngineError, MAX_REQUEST_BYTES, MAX_TOKEN_COUNT, Sha256Digest};
use std::collections::{HashMap, HashSet};

/// Exact explicit source-format selector for the pinned Qwen2.5 tokenizer JSON.
pub const QWEN25_TOKENIZER_JSON_SOURCE_FORMAT: &str = "qwen2.5-tokenizer-json-v1";
/// SHA-256 of the tokenizer shared by the three pinned Qwen2.5 1.5B startup artifacts.
pub const QWEN25_TOKENIZER_SHA256: &str =
    "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539";
/// Exact source tokenizer byte length.
pub const QWEN25_TOKENIZER_BYTES: usize = 7_031_645;
/// Exact canonical `AIPTOK02` byte length emitted from the pinned source.
pub const QWEN25_AIPTOK_BYTES: usize = 4_613_135;
/// SHA-256 of the complete canonical `AIPTOK02` bytes.
pub const QWEN25_AIPTOK_SHA256: &str =
    "c154a79da4917e0c851130b5989f53f1f582fe5965e44429a038727398f873ef";
/// SHA-384 of the complete canonical `AIPTOK02` bytes.
pub const QWEN25_AIPTOK_SHA384: &str = "sha384:f1b418104aedb90abc6ee892ef5b71db2b02c2c0c82a9f8e162487d933ffdd636b9ae077eed806f011c251cfea5187cc";
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

const ADDED_TOKENS: [ExpectedAddedToken<'static>; 22] = [
    ExpectedAddedToken::new("<|endoftext|>", true, AipTokQwen25AddedRole::BeginningOfSequence),
    ExpectedAddedToken::new("<|im_start|>", true, AipTokQwen25AddedRole::Special),
    ExpectedAddedToken::new("<|im_end|>", true, AipTokQwen25AddedRole::EndOfSequence),
    ExpectedAddedToken::new("<|object_ref_start|>", true, AipTokQwen25AddedRole::Special),
    ExpectedAddedToken::new("<|object_ref_end|>", true, AipTokQwen25AddedRole::Special),
    ExpectedAddedToken::new("<|box_start|>", true, AipTokQwen25AddedRole::Special),
    ExpectedAddedToken::new("<|box_end|>", true, AipTokQwen25AddedRole::Special),
    ExpectedAddedToken::new("<|quad_start|>", true, AipTokQwen25AddedRole::Special),
    ExpectedAddedToken::new("<|quad_end|>", true, AipTokQwen25AddedRole::Special),
    ExpectedAddedToken::new("<|vision_start|>", true, AipTokQwen25AddedRole::Special),
    ExpectedAddedToken::new("<|vision_end|>", true, AipTokQwen25AddedRole::Special),
    ExpectedAddedToken::new("<|vision_pad|>", true, AipTokQwen25AddedRole::Special),
    ExpectedAddedToken::new("<|image_pad|>", true, AipTokQwen25AddedRole::Special),
    ExpectedAddedToken::new("<|video_pad|>", true, AipTokQwen25AddedRole::Special),
    ExpectedAddedToken::new("<tool_call>", false, AipTokQwen25AddedRole::Ordinary),
    ExpectedAddedToken::new("</tool_call>", false, AipTokQwen25AddedRole::Ordinary),
    ExpectedAddedToken::new("<|fim_prefix|>", false, AipTokQwen25AddedRole::Ordinary),
    ExpectedAddedToken::new("<|fim_middle|>", false, AipTokQwen25AddedRole::Ordinary),
    ExpectedAddedToken::new("<|fim_suffix|>", false, AipTokQwen25AddedRole::Ordinary),
    ExpectedAddedToken::new("<|fim_pad|>", false, AipTokQwen25AddedRole::Ordinary),
    ExpectedAddedToken::new("<|repo_name|>", false, AipTokQwen25AddedRole::Ordinary),
    ExpectedAddedToken::new("<|file_sep|>", false, AipTokQwen25AddedRole::Ordinary),
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
    validate_source_identity(bytes)?;
    let _ = parse_graph(bytes, pinned_expectations(), ParseMode::ValidateOnly)?;
    Ok(Qwen25TokenizerDescriptor {
        source_sha256: hash_sha256(bytes),
        bpe_vocabulary_size: QWEN25_BPE_VOCABULARY_SIZE,
        defined_token_count: QWEN25_DEFINED_TOKEN_COUNT,
        model_vocabulary_size: QWEN25_MODEL_VOCABULARY_SIZE,
        merge_count: QWEN25_MERGE_COUNT,
    })
}

/// Converts the exact pinned source to the canonical data-only `AIPTOK02` form.
///
/// This function does not create a model package, sign or qualify an artifact,
/// construct an executable tokenizer, or activate the Qwen2.5 runtime profile.
pub fn convert(bytes: &[u8]) -> Result<Vec<u8>, EngineError> {
    validate_source_identity(bytes)?;
    let graph =
        parse_graph(bytes, pinned_expectations(), ParseMode::RetainTables)?.ok_or_else(|| {
            EngineError::internal_invariant("Qwen2.5 conversion tables were discarded")
        })?;
    let encoded = encode_graph(
        graph,
        QWEN25_MODEL_VOCABULARY_SIZE,
        QWEN25_BOS_TOKEN_ID,
        QWEN25_EOS_TOKEN_ID,
    )?;
    if encoded.len() != QWEN25_AIPTOK_BYTES
        || hash_sha256(&encoded).to_string() != QWEN25_AIPTOK_SHA256
        || hash_package(&encoded).to_string() != QWEN25_AIPTOK_SHA384
    {
        return Err(EngineError::internal_invariant(
            "pinned Qwen2.5 tokenizer conversion identity changed",
        ));
    }
    Ok(encoded)
}

fn validate_source_identity(bytes: &[u8]) -> Result<(), EngineError> {
    if bytes.len() != QWEN25_TOKENIZER_BYTES
        || hash_sha256(bytes).to_string() != QWEN25_TOKENIZER_SHA256
    {
        return Err(corrupt("Qwen2.5 tokenizer source identity is not pinned"));
    }
    Ok(())
}

const fn pinned_expectations() -> Expectations<'static> {
    Expectations {
        vocabulary_size: QWEN25_BPE_VOCABULARY_SIZE,
        merge_count: QWEN25_MERGE_COUNT,
        byte_base_count: 256,
        added_tokens: &ADDED_TOKENS,
        first_added_id: QWEN25_BPE_VOCABULARY_SIZE,
    }
}

#[derive(Clone, Copy)]
struct Expectations<'a> {
    vocabulary_size: u32,
    merge_count: u32,
    byte_base_count: u32,
    added_tokens: &'a [ExpectedAddedToken<'a>],
    first_added_id: u32,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ParseMode {
    ValidateOnly,
    RetainTables,
}

#[derive(Clone, Copy)]
struct ExpectedAddedToken<'a> {
    content: &'a str,
    source_special: bool,
    role: AipTokQwen25AddedRole,
}

impl<'a> ExpectedAddedToken<'a> {
    const fn new(content: &'a str, source_special: bool, role: AipTokQwen25AddedRole) -> Self {
        Self { content, source_special, role }
    }
}

#[derive(Debug)]
struct ParsedGraph {
    vocabulary: SourceVocabulary,
    merges: Vec<AipTokQwen25Merge>,
    added_tokens: Vec<SourceAddedToken>,
}

#[derive(Debug)]
struct SourceVocabulary {
    ordered_symbols: Vec<String>,
}

#[derive(Debug)]
struct SourceAddedToken {
    token_id: u32,
    content: String,
    role: AipTokQwen25AddedRole,
}

fn parse_graph(
    bytes: &[u8],
    expected: Expectations<'_>,
    mode: ParseMode,
) -> Result<Option<ParsedGraph>, EngineError> {
    let mut cursor = JsonCursor::new(bytes)?;
    cursor.expect(b'{')?;
    exact_key_string(&mut cursor, "version", "1.0")?;
    comma(&mut cursor)?;
    exact_key_null(&mut cursor, "truncation")?;
    comma(&mut cursor)?;
    exact_key_null(&mut cursor, "padding")?;
    comma(&mut cursor)?;
    exact_key(&mut cursor, "added_tokens")?;
    let added_tokens = parse_added_tokens(&mut cursor, expected, mode)?;
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
    let (vocabulary, merges) = parse_model(&mut cursor, expected, mode)?;
    cursor.expect(b'}')?;
    cursor.finish()?;
    if mode == ParseMode::RetainTables {
        Ok(Some(ParsedGraph { vocabulary, merges, added_tokens }))
    } else {
        Ok(None)
    }
}

fn parse_added_tokens(
    cursor: &mut JsonCursor<'_>,
    expected: Expectations<'_>,
    mode: ParseMode,
) -> Result<Vec<SourceAddedToken>, EngineError> {
    cursor.expect(b'[')?;
    let mut added_tokens = Vec::new();
    if mode == ParseMode::RetainTables {
        added_tokens
            .try_reserve_exact(expected.added_tokens.len())
            .map_err(|_| EngineError::resource_exhaustion("added-token allocation failed"))?;
    }
    for (index, expected_token) in expected.added_tokens.iter().enumerate() {
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
            || observed_content.as_deref() != Some(expected_token.content)
            || single_word != Some(false)
            || left_strip != Some(false)
            || right_strip != Some(false)
            || normalized != Some(false)
            || observed_special != Some(expected_token.source_special)
        {
            return Err(corrupt("Qwen2.5 added-token semantics do not match the closed profile"));
        }
        if mode == ParseMode::RetainTables {
            added_tokens.push(SourceAddedToken {
                token_id: expected_id,
                content: observed_content
                    .ok_or_else(|| corrupt("added-token content is absent"))?,
                role: expected_token.role,
            });
        }
    }
    cursor.expect(b']')?;
    Ok(added_tokens)
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

fn parse_model(
    cursor: &mut JsonCursor<'_>,
    expected: Expectations<'_>,
    mode: ParseMode,
) -> Result<(SourceVocabulary, Vec<AipTokQwen25Merge>), EngineError> {
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
    let capacity = vocabulary.ordered_symbols.len();
    let mut by_symbol = HashMap::new();
    by_symbol.try_reserve(capacity).map_err(|_| {
        EngineError::resource_exhaustion("Qwen2.5 vocabulary lookup allocation failed")
    })?;
    for (token_id, symbol) in vocabulary.ordered_symbols.iter().enumerate() {
        if by_symbol
            .insert(
                symbol.as_str(),
                u32::try_from(token_id)
                    .map_err(|_| corrupt("Qwen2.5 vocabulary ID is not representable"))?,
            )
            .is_some()
        {
            return Err(corrupt("Qwen2.5 vocabulary token bytes are duplicated"));
        }
    }
    comma(cursor)?;
    exact_key(cursor, "merges")?;
    let merges = parse_merges(cursor, expected, &by_symbol, mode)?;
    cursor.expect(b'}')?;
    Ok((vocabulary, merges))
}

fn parse_vocabulary(
    cursor: &mut JsonCursor<'_>,
    expected: Expectations<'_>,
) -> Result<SourceVocabulary, EngineError> {
    cursor.expect(b'{')?;
    let capacity = usize::try_from(expected.vocabulary_size)
        .map_err(|_| EngineError::resource_exhaustion("Qwen2.5 vocabulary is not addressable"))?;
    let mut ordered_symbols = Vec::new();
    ordered_symbols
        .try_reserve_exact(capacity)
        .map_err(|_| EngineError::resource_exhaustion("Qwen2.5 vocabulary allocation failed"))?;
    for expected_id in 0..expected.vocabulary_size {
        if expected_id != 0 {
            comma(cursor)?;
        }
        let token = cursor.key(MAX_SOURCE_TOKEN_BYTES)?;
        let token_id = cursor.u32()?;
        if token_id != expected_id {
            return Err(corrupt("Qwen2.5 vocabulary IDs or token bytes are noncanonical"));
        }
        if expected_id < expected.byte_base_count && token != byte_symbol_at_index(expected_id)? {
            return Err(corrupt("Qwen2.5 deterministic byte-symbol mapping is invalid"));
        }
        ordered_symbols.push(token);
    }
    if ordered_symbols.len() != capacity {
        return Err(corrupt("Qwen2.5 vocabulary count is invalid"));
    }
    cursor.expect(b'}')?;
    Ok(SourceVocabulary { ordered_symbols })
}

fn parse_merges(
    cursor: &mut JsonCursor<'_>,
    expected: Expectations<'_>,
    vocabulary: &HashMap<&str, u32>,
    mode: ParseMode,
) -> Result<Vec<AipTokQwen25Merge>, EngineError> {
    cursor.expect(b'[')?;
    let capacity = usize::try_from(expected.merge_count)
        .map_err(|_| EngineError::resource_exhaustion("Qwen2.5 merge count is not addressable"))?;
    let mut merges = Vec::new();
    if mode == ParseMode::RetainTables {
        merges
            .try_reserve_exact(capacity)
            .map_err(|_| EngineError::resource_exhaustion("Qwen2.5 merge allocation failed"))?;
    }
    let mut pairs = HashSet::new();
    pairs
        .try_reserve(capacity)
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
        let result_id = *vocabulary
            .get(combined.as_str())
            .ok_or_else(|| corrupt("Qwen2.5 merge result token is unavailable"))?;
        if mode == ParseMode::RetainTables {
            merges.push(AipTokQwen25Merge {
                rank,
                left: left_id,
                right: right_id,
                result: result_id,
            });
        }
    }
    cursor.expect(b']')?;
    Ok(merges)
}

fn encode_graph(
    graph: ParsedGraph,
    model_vocabulary_size: u32,
    beginning_of_sequence_token: u32,
    end_of_sequence_token: u32,
) -> Result<Vec<u8>, EngineError> {
    let ParsedGraph { vocabulary: source_vocabulary, merges, added_tokens: source_added_tokens } =
        graph;
    let SourceVocabulary { ordered_symbols } = source_vocabulary;
    let bpe_vocabulary_size = u32::try_from(ordered_symbols.len())
        .map_err(|_| corrupt("Qwen2.5 vocabulary count is not representable"))?;
    let reverse_bytes = reverse_byte_symbols()?;
    let mut byte_token_ids = [u32::MAX; 256];
    for (token_id, symbol) in ordered_symbols.iter().take(256).enumerate() {
        let bytes = decode_source_symbol(symbol, &reverse_bytes)?;
        if bytes.len() != 1 {
            return Err(corrupt("Qwen2.5 byte-base token is not one byte"));
        }
        let slot = &mut byte_token_ids[usize::from(bytes[0])];
        if *slot != u32::MAX {
            return Err(corrupt("Qwen2.5 byte-base token is duplicated"));
        }
        *slot = u32::try_from(token_id)
            .map_err(|_| corrupt("Qwen2.5 byte token ID is not representable"))?;
    }
    if byte_token_ids.contains(&u32::MAX) {
        return Err(corrupt("Qwen2.5 byte-base mapping is incomplete"));
    }

    let mut vocabulary = Vec::new();
    vocabulary
        .try_reserve_exact(
            ordered_symbols
                .len()
                .checked_add(source_added_tokens.len())
                .ok_or_else(|| corrupt("Qwen2.5 defined-token count overflows"))?,
        )
        .map_err(|_| EngineError::resource_exhaustion("AIPTok v2 vocabulary allocation failed"))?;
    for (token_id, symbol) in ordered_symbols.into_iter().enumerate() {
        vocabulary.push(AipTokQwen25VocabularyEntry {
            token_id: u32::try_from(token_id)
                .map_err(|_| corrupt("Qwen2.5 vocabulary ID is not representable"))?,
            bytes: decode_source_symbol(&symbol, &reverse_bytes)?,
        });
    }
    drop(reverse_bytes);

    let mut added_tokens = Vec::new();
    added_tokens
        .try_reserve_exact(source_added_tokens.len())
        .map_err(|_| EngineError::resource_exhaustion("AIPTok v2 added-token allocation failed"))?;
    for token in source_added_tokens {
        if (token.token_id == beginning_of_sequence_token)
            != (token.role == AipTokQwen25AddedRole::BeginningOfSequence)
            || (token.token_id == end_of_sequence_token)
                != (token.role == AipTokQwen25AddedRole::EndOfSequence)
        {
            return Err(corrupt("Qwen2.5 added-token role identity is inconsistent"));
        }
        vocabulary.push(AipTokQwen25VocabularyEntry {
            token_id: token.token_id,
            bytes: token.content.into_bytes(),
        });
        added_tokens.push(AipTokQwen25AddedToken { token_id: token.token_id, role: token.role });
    }

    AipTokQwen25Definition {
        vocabulary,
        bpe_vocabulary_size,
        model_vocabulary_size,
        merges,
        added_tokens,
        byte_token_ids,
        beginning_of_sequence_token,
        end_of_sequence_token,
        maximum_input_bytes: u32::try_from(MAX_REQUEST_BYTES)
            .map_err(|_| corrupt("Qwen2.5 input bound is not representable"))?,
        maximum_produced_tokens: MAX_TOKEN_COUNT,
    }
    .encode()
}

fn reverse_byte_symbols() -> Result<HashMap<char, u8>, EngineError> {
    let mut reverse = HashMap::new();
    reverse
        .try_reserve(256)
        .map_err(|_| EngineError::resource_exhaustion("byte-symbol map allocation failed"))?;
    for byte in 0_u16..=255 {
        let byte = u8::try_from(byte)
            .map_err(|_| corrupt("Qwen2.5 byte identity is not representable"))?;
        let symbol = byte_symbol(byte)?;
        let mut characters = symbol.chars();
        let character = characters
            .next()
            .filter(|_| characters.next().is_none())
            .ok_or_else(|| corrupt("Qwen2.5 byte symbol is not one scalar"))?;
        if reverse.insert(character, byte).is_some() {
            return Err(corrupt("Qwen2.5 byte symbol is duplicated"));
        }
    }
    Ok(reverse)
}

fn decode_source_symbol(
    symbol: &str,
    reverse_bytes: &HashMap<char, u8>,
) -> Result<Vec<u8>, EngineError> {
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(symbol.chars().count())
        .map_err(|_| EngineError::resource_exhaustion("token piece allocation failed"))?;
    for character in symbol.chars() {
        bytes.push(
            *reverse_bytes
                .get(&character)
                .ok_or_else(|| corrupt("Qwen2.5 vocabulary contains an unknown byte symbol"))?,
        );
    }
    if bytes.is_empty() || bytes.len() > MAX_SOURCE_TOKEN_BYTES {
        return Err(corrupt("Qwen2.5 decoded token piece is outside bounds"));
    }
    Ok(bytes)
}

fn byte_symbol(byte: u8) -> Result<String, EngineError> {
    let scalar = if matches!(byte, b'!'..=b'~' | 0xa1..=0xac | 0xae..=0xff) {
        u32::from(byte)
    } else {
        let mut extra = 0_u32;
        for candidate in 0_u16..=u16::from(byte) {
            let candidate = u8::try_from(candidate)
                .map_err(|_| corrupt("Qwen2.5 byte identity is not representable"))?;
            if !matches!(candidate, b'!'..=b'~' | 0xa1..=0xac | 0xae..=0xff) {
                extra += 1;
            }
        }
        0xff + extra
    };
    Ok(char::from_u32(scalar)
        .ok_or_else(|| corrupt("Qwen2.5 byte symbol is not a Unicode scalar"))?
        .to_string())
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
                    return byte_symbol(byte);
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
    use personalai_engine_memory::MemoryPool;
    use personalai_engine_tokenizer::{AIPTOK_QWEN25_MAGIC, inspect_qwen25_aiptok};

    const SMALL_ADDED: [ExpectedAddedToken<'static>; 1] =
        [ExpectedAddedToken::new("<|end|>", true, AipTokQwen25AddedRole::Special)];
    const CONVERSION_ADDED: [ExpectedAddedToken<'static>; 2] = [
        ExpectedAddedToken::new("<|bos|>", true, AipTokQwen25AddedRole::BeginningOfSequence),
        ExpectedAddedToken::new("<|eos|>", true, AipTokQwen25AddedRole::EndOfSequence),
    ];

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

    fn conversion_expectations() -> Expectations<'static> {
        Expectations {
            vocabulary_size: 258,
            merge_count: 2,
            byte_base_count: 256,
            added_tokens: &CONVERSION_ADDED,
            first_added_id: 258,
        }
    }

    fn json_string(value: &str) -> String {
        let mut encoded = String::from("\"");
        for character in value.chars() {
            match character {
                '"' => encoded.push_str("\\\""),
                '\\' => encoded.push_str("\\\\"),
                '\u{08}' => encoded.push_str("\\b"),
                '\u{0c}' => encoded.push_str("\\f"),
                '\n' => encoded.push_str("\\n"),
                '\r' => encoded.push_str("\\r"),
                '\t' => encoded.push_str("\\t"),
                character if character.is_control() => {
                    encoded.push_str(&format!("\\u{:04x}", u32::from(character)));
                }
                character => encoded.push(character),
            }
        }
        encoded.push('"');
        encoded
    }

    fn conversion_fixture() -> Vec<u8> {
        let mut vocabulary = Vec::new();
        for token_id in 0..256_u32 {
            let symbol = byte_symbol_at_index(token_id).expect("byte symbol exists");
            vocabulary.push(format!("{}:{token_id}", json_string(&symbol)));
        }
        vocabulary.push(format!("{}:256", json_string("abc")));
        vocabulary.push(format!("{}:257", json_string("ab")));
        format!(
            concat!(
                "{{\"version\":\"1.0\",\"truncation\":null,\"padding\":null,",
                "\"added_tokens\":[",
                "{{\"id\":258,\"content\":\"<|bos|>\",\"single_word\":false,",
                "\"lstrip\":false,\"rstrip\":false,\"normalized\":false,",
                "\"special\":true}},",
                "{{\"id\":259,\"content\":\"<|eos|>\",\"single_word\":false,",
                "\"lstrip\":false,\"rstrip\":false,\"normalized\":false,",
                "\"special\":true}}],",
                "\"normalizer\":{{\"type\":\"NFC\"}},",
                "\"pre_tokenizer\":{{\"type\":\"Sequence\",\"pretokenizers\":[",
                "{{\"type\":\"Split\",\"pattern\":{{\"Regex\":{}}},",
                "\"behavior\":\"Isolated\",\"invert\":false}},",
                "{{\"type\":\"ByteLevel\",\"add_prefix_space\":false,",
                "\"trim_offsets\":false,\"use_regex\":false}}]}},",
                "\"post_processor\":{{\"type\":\"ByteLevel\",",
                "\"add_prefix_space\":false,\"trim_offsets\":false,",
                "\"use_regex\":false}},",
                "\"decoder\":{{\"type\":\"ByteLevel\",\"add_prefix_space\":false,",
                "\"trim_offsets\":false,\"use_regex\":false}},",
                "\"model\":{{\"type\":\"BPE\",\"dropout\":null,",
                "\"unk_token\":null,\"continuing_subword_prefix\":\"\",",
                "\"end_of_word_suffix\":\"\",\"fuse_unk\":false,",
                "\"byte_fallback\":false,\"vocab\":{{{}}},",
                "\"merges\":[\"a b\",\"ab c\"]}}}}"
            ),
            json_string(QWEN25_PRETOKENIZER_REGEX),
            vocabulary.join(","),
        )
        .into_bytes()
    }

    #[test]
    fn closed_graph_accepts_only_exact_components_and_no_unknown_token() {
        let valid = fixture(QWEN25_PRETOKENIZER_REGEX, "null");
        let parsed = parse_graph(&valid, expectations(), ParseMode::ValidateOnly);
        assert!(parsed.is_ok(), "{parsed:?}");
        assert!(
            parse_graph(&fixture(".+", "null"), expectations(), ParseMode::ValidateOnly).is_err()
        );
        assert!(
            parse_graph(
                &fixture(QWEN25_PRETOKENIZER_REGEX, "\"<unk>\""),
                expectations(),
                ParseMode::ValidateOnly,
            )
            .is_err()
        );
    }

    #[test]
    fn production_entry_points_reject_unpinned_source_data() {
        let fixture = fixture(QWEN25_PRETOKENIZER_REGEX, "null");
        assert!(inspect(&fixture).is_err());
        assert!(convert(&fixture).is_err());
    }

    #[test]
    fn retained_source_graph_encodes_deterministic_self_validating_aiptok02() {
        let source = conversion_fixture();
        let first_graph = parse_graph(&source, conversion_expectations(), ParseMode::RetainTables)
            .expect("graph parses")
            .expect("conversion tables retained");
        assert_eq!(first_graph.vocabulary.ordered_symbols.len(), 258);
        assert_eq!(first_graph.merges.len(), 2);
        assert_eq!(first_graph.merges[0].result, 257);
        assert_eq!(first_graph.merges[1].result, 256);
        assert_eq!(first_graph.added_tokens.len(), 2);
        let first = encode_graph(first_graph, 261, 258, 259).expect("graph encodes");
        let second = encode_graph(
            parse_graph(&source, conversion_expectations(), ParseMode::RetainTables)
                .expect("graph parses repeatedly")
                .expect("conversion tables retained"),
            261,
            258,
            259,
        )
        .expect("graph encodes repeatedly");
        assert_eq!(first, second);
        assert_eq!(&first[..AIPTOK_QWEN25_MAGIC.len()], &AIPTOK_QWEN25_MAGIC);

        let pool = MemoryPool::new(1024 * 1024).expect("pool validates");
        let descriptor = inspect_qwen25_aiptok(&first, &pool).expect("output self-validates");
        assert_eq!(descriptor.model_vocabulary_size(), 261);
        assert_eq!(descriptor.defined_token_count(), 260);
        assert_eq!(descriptor.bpe_vocabulary_size(), 258);
        assert_eq!(descriptor.merge_count(), 2);
        assert_eq!(descriptor.added_token_count(), 2);
        assert_eq!(descriptor.special_token_count(), 2);
        assert_eq!(descriptor.beginning_of_sequence_token(), 258);
        assert_eq!(descriptor.end_of_sequence_token(), 259);
        assert_eq!(
            descriptor.maximum_input_bytes(),
            u32::try_from(MAX_REQUEST_BYTES).expect("request bound fits the wire field")
        );
        assert_eq!(descriptor.maximum_produced_tokens(), MAX_TOKEN_COUNT);
    }

    #[test]
    fn conversion_rejects_mutated_roles_merges_and_byte_symbols() {
        let source = conversion_fixture();
        let wrong_role = String::from_utf8(source.clone())
            .expect("fixture is UTF-8")
            .replacen("\"special\":true", "\"special\":false", 1)
            .into_bytes();
        assert!(
            parse_graph(&wrong_role, conversion_expectations(), ParseMode::RetainTables,).is_err()
        );

        let wrong_merge = String::from_utf8(source)
            .expect("fixture is UTF-8")
            .replace("\"merges\":[\"a b\",\"ab c\"]", "\"merges\":[\"a c\",\"ab c\"]")
            .into_bytes();
        assert!(
            parse_graph(&wrong_merge, conversion_expectations(), ParseMode::RetainTables,).is_err()
        );

        let reverse = reverse_byte_symbols().expect("reverse map builds");
        assert!(decode_source_symbol("🙂", &reverse).is_err());
        assert_eq!(decode_source_symbol("ĠA", &reverse).unwrap(), b" A");
    }

    #[test]
    fn emitted_tables_reject_rank_result_role_and_defined_tail_mutations() {
        let source = conversion_fixture();
        let graph = || {
            parse_graph(&source, conversion_expectations(), ParseMode::RetainTables)
                .expect("graph parses")
                .expect("tables retained")
        };

        let mut wrong_rank = graph();
        wrong_rank.merges[0].rank = 1;
        assert!(encode_graph(wrong_rank, 261, 258, 259).is_err());

        let mut wrong_result = graph();
        wrong_result.merges[0].result = 0;
        assert!(encode_graph(wrong_result, 261, 258, 259).is_err());

        let mut wrong_result_bytes = graph();
        wrong_result_bytes.vocabulary.ordered_symbols[257] = "ac".to_owned();
        assert!(encode_graph(wrong_result_bytes, 261, 258, 259).is_err());

        let mut wrong_role = graph();
        wrong_role.added_tokens[0].role = AipTokQwen25AddedRole::Special;
        assert!(encode_graph(wrong_role, 261, 258, 259).is_err());

        let encoded = encode_graph(graph(), 261, 258, 259).expect("valid graph encodes");
        let pool = MemoryPool::new(1024 * 1024).expect("pool validates");
        let descriptor = inspect_qwen25_aiptok(&encoded, &pool).expect("output validates");
        descriptor.validate_token(259).expect("last defined ID works");
        assert!(descriptor.validate_token(260).is_err());
    }

    #[test]
    fn pinned_added_token_roles_are_explicit_for_all_twenty_two_entries() {
        assert_eq!(ADDED_TOKENS.len(), 22);
        for (index, token) in ADDED_TOKENS.iter().enumerate() {
            let expected = match index {
                0 => AipTokQwen25AddedRole::BeginningOfSequence,
                2 => AipTokQwen25AddedRole::EndOfSequence,
                1 | 3..=13 => AipTokQwen25AddedRole::Special,
                14..=21 => AipTokQwen25AddedRole::Ordinary,
                _ => unreachable!("the fixed table has exactly twenty-two entries"),
            };
            assert_eq!(token.role, expected, "role at added-token index {index}");
            assert_eq!(token.source_special, index <= 13);
        }
    }

    #[test]
    fn deterministic_byte_symbol_map_is_total_and_unique() {
        let symbols = (0_u16..=255)
            .map(|byte| byte_symbol(u8::try_from(byte).unwrap()).unwrap())
            .collect::<HashSet<_>>();
        assert_eq!(symbols.len(), 256);
        assert_eq!(byte_symbol(b' ').unwrap(), "Ġ");
        assert_eq!(byte_symbol(b'A').unwrap(), "A");
        for byte in 0_u16..=255 {
            let byte = u8::try_from(byte).expect("byte fits");
            let token_id = match byte {
                33..=126 => u32::from(byte - 33),
                161..=172 => 94 + u32::from(byte - 161),
                174..=255 => 106 + u32::from(byte - 174),
                0..=32 => 188 + u32::from(byte),
                127..=160 => 221 + u32::from(byte - 127),
                173 => 255,
            };
            assert_eq!(byte_symbol_at_index(token_id).unwrap(), byte_symbol(byte).unwrap());
        }
    }
}
