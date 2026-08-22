//! Bounded data-only AIPTok v2 parser foundation for the pinned Qwen2.5 profile.
//!
//! This module defines and validates the immutable tables needed by the exact
//! Qwen2.5 byte-level BPE profile. It intentionally does not implement encode,
//! decode, normalization, or pre-tokenization. The public tokenizer factory
//! validates this format and then returns `UnsupportedTokenizer` until exact
//! NFC and Unicode property semantics are independently qualified.

use super::{
    MAX_BPE_MERGE_RULES, MAX_SPECIAL_TOKENS, MAX_TOKEN_PIECE_BYTES, MAX_VOCABULARY_SIZE,
    checked_add, corrupt, read_u16, read_u32, usize_to_u32, write_at, write_u16_at, write_u16_push,
    write_u32_at, write_u32_push,
};
use personalai_engine_crypto::hash_package;
use personalai_engine_memory::MemoryPool;
use personalai_engine_types::{
    EngineError, MAX_REQUEST_BYTES, MAX_TOKEN_COUNT, MAX_TOKENIZER_DATA_BYTES, ModelPackageHash,
};
use std::collections::{BTreeMap, BTreeSet};
use std::mem::size_of;

/// Exact magic for the separate data-only AIPTok v2/Qwen2.5 profile.
pub const AIPTOK_QWEN25_MAGIC: [u8; 8] = *b"AIPTOK02";
/// Fixed AIPTok v2/Qwen2.5 header length.
pub const AIPTOK_QWEN25_HEADER_BYTES: usize = 160;
/// Exact paired-model vocabulary/logit count for the pinned Qwen2.5 profile.
pub const AIPTOK_QWEN25_PINNED_MODEL_VOCABULARY_SIZE: u32 = 151_936;
/// Exact defined tokenizer-ID count for the pinned Qwen2.5 profile.
pub const AIPTOK_QWEN25_PINNED_DEFINED_TOKEN_COUNT: u32 = 151_665;
/// Exact BPE vocabulary count for the pinned Qwen2.5 profile.
pub const AIPTOK_QWEN25_PINNED_BPE_VOCABULARY_SIZE: u32 = 151_643;
/// Exact merge count for the pinned Qwen2.5 profile.
pub const AIPTOK_QWEN25_PINNED_MERGE_COUNT: u32 = 151_387;
/// Exact added-token count for the pinned Qwen2.5 profile.
pub const AIPTOK_QWEN25_PINNED_ADDED_TOKEN_COUNT: u32 = 22;
/// Exact explicit BOS identity for the pinned Qwen2.5 profile.
pub const AIPTOK_QWEN25_PINNED_BOS_TOKEN_ID: u32 = 151_643;
/// Exact explicit EOS identity for the pinned Qwen2.5 profile.
pub const AIPTOK_QWEN25_PINNED_EOS_TOKEN_ID: u32 = 151_645;

const FAMILY_QWEN25_BYTE_BPE: u16 = 2;
const NORMALIZATION_NFC: u16 = 2;
const PRETOKENIZER_QWEN25_FIXED_V1: u16 = 2;
const DECODER_GPT2_BYTE_SYMBOLS: u16 = 2;
const UNKNOWN_NONE: u16 = 2;
const FLAG_NO_IMPLICIT_BOS: u16 = 1;
const FLAG_NO_IMPLICIT_EOS: u16 = 1 << 1;
const REQUIRED_FLAGS: u16 = FLAG_NO_IMPLICIT_BOS | FLAG_NO_IMPLICIT_EOS;
const UNICODE_PROFILE_QWEN25_FIXED_V1: u32 = 1;
const VOCABULARY_ENTRY_HEADER_BYTES: usize = 8;
const MERGE_RECORD_BYTES: usize = 16;
const ADDED_RECORD_BYTES: usize = 8;
const BYTE_MAP_ENTRIES: usize = 256;
const BYTE_MAP_BYTES: usize = BYTE_MAP_ENTRIES * size_of::<u32>();
const VOCABULARY_FLAG_ADDED: u16 = 1;
const VOCABULARY_FLAG_SPECIAL: u16 = 1 << 1;
const KNOWN_VOCABULARY_FLAGS: u16 = VOCABULARY_FLAG_ADDED | VOCABULARY_FLAG_SPECIAL;
const MAX_ADDED_TOKENS: u32 = MAX_SPECIAL_TOKENS;
const PERSISTENT_BYTES_PER_VOCABULARY_ENTRY: u64 = 96;
const PERSISTENT_BYTES_PER_MERGE: u64 = 128;
const PINNED_QWEN25_SPECIAL_TOKEN_COUNT: u32 = 14;
const PINNED_QWEN25_ADDED_TOKENS: [(&str, AipTokQwen25AddedRole); 22] = [
    ("<|endoftext|>", AipTokQwen25AddedRole::BeginningOfSequence),
    ("<|im_start|>", AipTokQwen25AddedRole::Special),
    ("<|im_end|>", AipTokQwen25AddedRole::EndOfSequence),
    ("<|object_ref_start|>", AipTokQwen25AddedRole::Special),
    ("<|object_ref_end|>", AipTokQwen25AddedRole::Special),
    ("<|box_start|>", AipTokQwen25AddedRole::Special),
    ("<|box_end|>", AipTokQwen25AddedRole::Special),
    ("<|quad_start|>", AipTokQwen25AddedRole::Special),
    ("<|quad_end|>", AipTokQwen25AddedRole::Special),
    ("<|vision_start|>", AipTokQwen25AddedRole::Special),
    ("<|vision_end|>", AipTokQwen25AddedRole::Special),
    ("<|vision_pad|>", AipTokQwen25AddedRole::Special),
    ("<|image_pad|>", AipTokQwen25AddedRole::Special),
    ("<|video_pad|>", AipTokQwen25AddedRole::Special),
    ("<tool_call>", AipTokQwen25AddedRole::Ordinary),
    ("</tool_call>", AipTokQwen25AddedRole::Ordinary),
    ("<|fim_prefix|>", AipTokQwen25AddedRole::Ordinary),
    ("<|fim_middle|>", AipTokQwen25AddedRole::Ordinary),
    ("<|fim_suffix|>", AipTokQwen25AddedRole::Ordinary),
    ("<|fim_pad|>", AipTokQwen25AddedRole::Ordinary),
    ("<|repo_name|>", AipTokQwen25AddedRole::Ordinary),
    ("<|file_sep|>", AipTokQwen25AddedRole::Ordinary),
];

/// Role of one declarative Qwen2.5 added token.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AipTokQwen25AddedRole {
    /// Explicit beginning-of-sequence identity; never inserted implicitly.
    BeginningOfSequence,
    /// Explicit end-of-sequence identity; never inserted implicitly.
    EndOfSequence,
    /// Added token omitted by skip-special decoding.
    Special,
    /// Added token retained by ordinary decoding.
    Ordinary,
}

impl AipTokQwen25AddedRole {
    const fn wire(self) -> u16 {
        match self {
            Self::BeginningOfSequence => 1,
            Self::EndOfSequence => 2,
            Self::Special => 3,
            Self::Ordinary => 4,
        }
    }

    fn from_wire(value: u16) -> Result<Self, EngineError> {
        match value {
            1 => Ok(Self::BeginningOfSequence),
            2 => Ok(Self::EndOfSequence),
            3 => Ok(Self::Special),
            4 => Ok(Self::Ordinary),
            _ => Err(corrupt("AIPTok v2 added-token role is invalid")),
        }
    }

    const fn is_special(self) -> bool {
        !matches!(self, Self::Ordinary)
    }
}

/// One contiguous vocabulary entry in an offline AIPTok v2 definition.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AipTokQwen25VocabularyEntry {
    /// Contiguous token identifier.
    pub token_id: u32,
    /// Canonical decoded bytes for the token.
    pub bytes: Vec<u8>,
}

/// One ranked Qwen2.5 BPE merge.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AipTokQwen25Merge {
    /// Contiguous merge rank.
    pub rank: u32,
    /// Left input token ID.
    pub left: u32,
    /// Right input token ID.
    pub right: u32,
    /// Result token ID.
    pub result: u32,
}

/// One declarative added-token record.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AipTokQwen25AddedToken {
    /// Added token ID.
    pub token_id: u32,
    /// Fixed decode/control role.
    pub role: AipTokQwen25AddedRole,
}

/// Canonical data-only AIPTok v2 definition used by offline fixture tooling.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AipTokQwen25Definition {
    /// Every explicitly defined token, in contiguous ID order.
    pub vocabulary: Vec<AipTokQwen25VocabularyEntry>,
    /// Count of BPE vocabulary entries before the contiguous added-token suffix.
    pub bpe_vocabulary_size: u32,
    /// Logit/embedding vocabulary size; undefined tail IDs remain invalid tokens.
    pub model_vocabulary_size: u32,
    /// Ranked declarative merge table.
    pub merges: Vec<AipTokQwen25Merge>,
    /// Contiguous added-token suffix and exact roles.
    pub added_tokens: Vec<AipTokQwen25AddedToken>,
    /// Raw byte to base-token ID permutation.
    pub byte_token_ids: [u32; BYTE_MAP_ENTRIES],
    /// Explicit BOS identity; the profile never inserts it implicitly.
    pub beginning_of_sequence_token: u32,
    /// Explicit EOS identity; the profile never inserts it implicitly.
    pub end_of_sequence_token: u32,
    /// Maximum accepted UTF-8 input bytes.
    pub maximum_input_bytes: u32,
    /// Maximum produced token count.
    pub maximum_produced_tokens: u32,
}

impl AipTokQwen25Definition {
    /// Encodes and self-validates the canonical AIPTok v2 wire representation.
    #[allow(
        clippy::too_many_lines,
        reason = "the complete fixed wire schema is intentionally visible in one encoder"
    )]
    pub fn encode(&self) -> Result<Vec<u8>, EngineError> {
        let defined_token_count = u32::try_from(self.vocabulary.len())
            .map_err(|_| corrupt("AIPTok v2 vocabulary count is not representable"))?;
        let merge_count = u32::try_from(self.merges.len())
            .map_err(|_| corrupt("AIPTok v2 merge count is not representable"))?;
        let added_count = u32::try_from(self.added_tokens.len())
            .map_err(|_| corrupt("AIPTok v2 added-token count is not representable"))?;
        let special_count =
            u32::try_from(self.added_tokens.iter().filter(|token| token.role.is_special()).count())
                .map_err(|_| corrupt("AIPTok v2 special-token count is not representable"))?;
        let maximum_piece = self
            .vocabulary
            .iter()
            .map(|entry| entry.bytes.len())
            .max()
            .ok_or_else(|| corrupt("AIPTok v2 vocabulary is empty"))?;
        let maximum_piece = u16::try_from(maximum_piece)
            .map_err(|_| corrupt("AIPTok v2 token piece exceeds its field"))?;

        let mut roles = BTreeMap::new();
        for added in &self.added_tokens {
            if roles.insert(added.token_id, added.role).is_some() {
                return Err(corrupt("AIPTok v2 added-token ID is duplicated"));
            }
        }
        let vocabulary_bytes = self.vocabulary.iter().try_fold(0_usize, |total, entry| {
            checked_add(total, checked_add(VOCABULARY_ENTRY_HEADER_BYTES, entry.bytes.len())?)
        })?;
        let merge_bytes = self
            .merges
            .len()
            .checked_mul(MERGE_RECORD_BYTES)
            .ok_or_else(|| corrupt("AIPTok v2 merge bytes overflow"))?;
        let added_bytes = self
            .added_tokens
            .len()
            .checked_mul(ADDED_RECORD_BYTES)
            .ok_or_else(|| corrupt("AIPTok v2 added-token bytes overflow"))?;
        let total_bytes = checked_add(
            checked_add(AIPTOK_QWEN25_HEADER_BYTES, vocabulary_bytes)?,
            checked_add(checked_add(merge_bytes, added_bytes)?, BYTE_MAP_BYTES)?,
        )?;
        if u64::try_from(total_bytes).unwrap_or(u64::MAX) > MAX_TOKENIZER_DATA_BYTES {
            return Err(EngineError::resource_exhaustion("AIPTok v2 data exceeds its hard limit"));
        }

        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(total_bytes)
            .map_err(|_| EngineError::resource_exhaustion("AIPTok v2 allocation failed"))?;
        bytes.resize(AIPTOK_QWEN25_HEADER_BYTES, 0);
        for entry in &self.vocabulary {
            write_u32_push(&mut bytes, entry.token_id);
            let role = roles.get(&entry.token_id).copied();
            let mut flags = 0_u16;
            if role.is_some() {
                flags |= VOCABULARY_FLAG_ADDED;
            }
            if role.is_some_and(AipTokQwen25AddedRole::is_special) {
                flags |= VOCABULARY_FLAG_SPECIAL;
            }
            write_u16_push(&mut bytes, flags);
            write_u16_push(
                &mut bytes,
                u16::try_from(entry.bytes.len())
                    .map_err(|_| corrupt("AIPTok v2 token piece exceeds its field"))?,
            );
            bytes.extend_from_slice(&entry.bytes);
        }
        let merge_offset = usize_to_u32(bytes.len())?;
        for merge in &self.merges {
            write_u32_push(&mut bytes, merge.rank);
            write_u32_push(&mut bytes, merge.left);
            write_u32_push(&mut bytes, merge.right);
            write_u32_push(&mut bytes, merge.result);
        }
        let added_offset = usize_to_u32(bytes.len())?;
        for added in &self.added_tokens {
            write_u32_push(&mut bytes, added.token_id);
            write_u16_push(&mut bytes, added.role.wire());
            write_u16_push(&mut bytes, 0);
        }
        let byte_map_offset = usize_to_u32(bytes.len())?;
        for token in self.byte_token_ids {
            write_u32_push(&mut bytes, token);
        }
        if bytes.len() != total_bytes {
            return Err(EngineError::internal_invariant("AIPTok v2 encoded length changed"));
        }

        bytes[..8].copy_from_slice(&AIPTOK_QWEN25_MAGIC);
        write_u16_at(&mut bytes, 8, 2)?;
        write_u16_at(&mut bytes, 10, 0)?;
        write_u32_at(&mut bytes, 12, usize_to_u32(AIPTOK_QWEN25_HEADER_BYTES)?)?;
        write_u16_at(&mut bytes, 16, FAMILY_QWEN25_BYTE_BPE)?;
        write_u16_at(&mut bytes, 18, NORMALIZATION_NFC)?;
        write_u16_at(&mut bytes, 20, PRETOKENIZER_QWEN25_FIXED_V1)?;
        write_u16_at(&mut bytes, 22, DECODER_GPT2_BYTE_SYMBOLS)?;
        write_u16_at(&mut bytes, 24, UNKNOWN_NONE)?;
        write_u16_at(&mut bytes, 26, REQUIRED_FLAGS)?;
        write_u32_at(&mut bytes, 28, self.model_vocabulary_size)?;
        write_u32_at(&mut bytes, 32, defined_token_count)?;
        write_u32_at(&mut bytes, 36, self.bpe_vocabulary_size)?;
        write_u32_at(&mut bytes, 40, merge_count)?;
        write_u32_at(&mut bytes, 44, added_count)?;
        write_u32_at(&mut bytes, 48, special_count)?;
        write_u32_at(&mut bytes, 52, self.beginning_of_sequence_token)?;
        write_u32_at(&mut bytes, 56, self.end_of_sequence_token)?;
        write_u32_at(&mut bytes, 60, self.maximum_input_bytes)?;
        write_u32_at(&mut bytes, 64, self.maximum_produced_tokens)?;
        write_u16_at(&mut bytes, 68, maximum_piece)?;
        write_u16_at(&mut bytes, 70, 0)?;
        write_u32_at(&mut bytes, 72, usize_to_u32(AIPTOK_QWEN25_HEADER_BYTES)?)?;
        write_u32_at(&mut bytes, 76, merge_offset)?;
        write_u32_at(&mut bytes, 80, added_offset)?;
        write_u32_at(&mut bytes, 84, byte_map_offset)?;
        let total_length = usize_to_u32(bytes.len())?;
        write_u32_at(&mut bytes, 88, total_length)?;
        write_u32_at(&mut bytes, 92, UNICODE_PROFILE_QWEN25_FIXED_V1)?;
        let digest = hash_package(&bytes[AIPTOK_QWEN25_HEADER_BYTES..]);
        write_at(&mut bytes, 112, digest.as_bytes())?;
        let _ = parse_data(&bytes)?;
        Ok(bytes)
    }
}

/// Content-free metadata proven by the bounded AIPTok v2 parser.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Qwen25AipTokDescriptor {
    model_vocabulary_size: u32,
    defined_token_count: u32,
    bpe_vocabulary_size: u32,
    merge_count: u32,
    added_token_count: u32,
    special_token_count: u32,
    beginning_of_sequence_token: u32,
    end_of_sequence_token: u32,
    maximum_input_bytes: u32,
    maximum_produced_tokens: u32,
    content_hash: ModelPackageHash,
}

impl Qwen25AipTokDescriptor {
    /// Returns the model/logit vocabulary size, including any undefined tail.
    pub const fn model_vocabulary_size(self) -> u32 {
        self.model_vocabulary_size
    }

    /// Returns the number of token IDs backed by immutable tokenizer data.
    pub const fn defined_token_count(self) -> u32 {
        self.defined_token_count
    }

    /// Returns the BPE vocabulary count before the added-token suffix.
    pub const fn bpe_vocabulary_size(self) -> u32 {
        self.bpe_vocabulary_size
    }

    /// Returns the exact merge count.
    pub const fn merge_count(self) -> u32 {
        self.merge_count
    }

    /// Returns the added-token count.
    pub const fn added_token_count(self) -> u32 {
        self.added_token_count
    }

    /// Returns the added tokens marked special/control data.
    pub const fn special_token_count(self) -> u32 {
        self.special_token_count
    }

    /// Returns the explicit BOS identity; it is never inserted implicitly.
    pub const fn beginning_of_sequence_token(self) -> u32 {
        self.beginning_of_sequence_token
    }

    /// Returns the explicit EOS identity; it is never inserted implicitly.
    pub const fn end_of_sequence_token(self) -> u32 {
        self.end_of_sequence_token
    }

    /// Confirms that the profile contains no unknown-token identity.
    pub const fn unknown_token(self) -> Option<u32> {
        None
    }

    /// Confirms that the profile never inserts BOS without explicit caller input.
    pub const fn inserts_bos(self) -> bool {
        false
    }

    /// Returns the package-declared input byte ceiling.
    pub const fn maximum_input_bytes(self) -> u32 {
        self.maximum_input_bytes
    }

    /// Returns the package-declared output-token ceiling.
    pub const fn maximum_produced_tokens(self) -> u32 {
        self.maximum_produced_tokens
    }

    /// Returns SHA-384 over all canonical tables following the header.
    pub const fn content_hash(self) -> ModelPackageHash {
        self.content_hash
    }

    /// Returns whether counts match the exact pinned Qwen2.5 1.5B tokenizer.
    pub const fn matches_pinned_qwen25_1_5b_counts(self) -> bool {
        self.model_vocabulary_size == AIPTOK_QWEN25_PINNED_MODEL_VOCABULARY_SIZE
            && self.defined_token_count == AIPTOK_QWEN25_PINNED_DEFINED_TOKEN_COUNT
            && self.bpe_vocabulary_size == AIPTOK_QWEN25_PINNED_BPE_VOCABULARY_SIZE
            && self.merge_count == AIPTOK_QWEN25_PINNED_MERGE_COUNT
            && self.added_token_count == AIPTOK_QWEN25_PINNED_ADDED_TOKEN_COUNT
            && self.special_token_count == PINNED_QWEN25_SPECIAL_TOKEN_COUNT
            && self.beginning_of_sequence_token == AIPTOK_QWEN25_PINNED_BOS_TOKEN_ID
            && self.end_of_sequence_token == AIPTOK_QWEN25_PINNED_EOS_TOKEN_ID
    }

    /// Validates one token ID against defined tokenizer data, not merely logits.
    pub fn validate_token(self, token: u32) -> Result<(), EngineError> {
        if token >= self.defined_token_count {
            return Err(EngineError::invalid_input(
                "token ID is undefined by the AIPTok v2 tokenizer",
            ));
        }
        Ok(())
    }
}

struct Header {
    descriptor: Qwen25AipTokDescriptor,
    maximum_piece_bytes: u16,
    vocabulary_offset: usize,
    merge_offset: usize,
    added_offset: usize,
    byte_map_offset: usize,
}

struct Vocabulary {
    pieces: Vec<Box<[u8]>>,
    flags: Vec<u16>,
}

/// Parses the fixed AIPTok v2/Qwen2.5 tables under an aggregate memory bound.
pub fn inspect_qwen25_aiptok(
    data: &[u8],
    pool: &MemoryPool,
) -> Result<Qwen25AipTokDescriptor, EngineError> {
    let header = parse_header(data)?;
    let reservation_bytes = u64::try_from(data.len())
        .ok()
        .and_then(|bytes| {
            u64::from(header.descriptor.defined_token_count)
                .checked_mul(PERSISTENT_BYTES_PER_VOCABULARY_ENTRY)
                .and_then(|table| bytes.checked_add(table))
        })
        .and_then(|bytes| {
            u64::from(header.descriptor.merge_count)
                .checked_mul(PERSISTENT_BYTES_PER_MERGE)
                .and_then(|table| bytes.checked_add(table))
        })
        .ok_or_else(|| EngineError::resource_exhaustion("AIPTok v2 memory bound overflows"))?;
    let _reservation = pool.reserve(reservation_bytes)?;
    parse_data_with_header(data, &header)
}

fn parse_data(data: &[u8]) -> Result<Qwen25AipTokDescriptor, EngineError> {
    let header = parse_header(data)?;
    parse_data_with_header(data, &header)
}

fn parse_data_with_header(
    data: &[u8],
    header: &Header,
) -> Result<Qwen25AipTokDescriptor, EngineError> {
    let vocabulary = parse_vocabulary(data, header)?;
    parse_merges(data, header, &vocabulary)?;
    parse_added_tokens(data, header, &vocabulary)?;
    parse_byte_map(data, header, &vocabulary)?;
    Ok(header.descriptor)
}

#[allow(
    clippy::too_many_lines,
    reason = "the complete fixed header admission schema is intentionally reviewed together"
)]
fn parse_header(data: &[u8]) -> Result<Header, EngineError> {
    if data.len() < AIPTOK_QWEN25_HEADER_BYTES
        || u64::try_from(data.len()).unwrap_or(u64::MAX) > MAX_TOKENIZER_DATA_BYTES
        || data[..8] != AIPTOK_QWEN25_MAGIC
    {
        return Err(corrupt("AIPTok v2 magic or length is invalid"));
    }
    if read_u16(data, 8)? != 2
        || read_u16(data, 10)? != 0
        || read_u32(data, 12)? != usize_to_u32(AIPTOK_QWEN25_HEADER_BYTES)?
    {
        return Err(corrupt("AIPTok v2 version or header length is unsupported"));
    }
    if read_u16(data, 16)? != FAMILY_QWEN25_BYTE_BPE
        || read_u16(data, 18)? != NORMALIZATION_NFC
        || read_u16(data, 20)? != PRETOKENIZER_QWEN25_FIXED_V1
        || read_u16(data, 22)? != DECODER_GPT2_BYTE_SYMBOLS
        || read_u16(data, 24)? != UNKNOWN_NONE
        || read_u16(data, 26)? != REQUIRED_FLAGS
        || read_u32(data, 92)? != UNICODE_PROFILE_QWEN25_FIXED_V1
    {
        return Err(EngineError::unsupported_tokenizer("AIPTok v2 behavior profile"));
    }
    if data[96..112].iter().any(|byte| *byte != 0) {
        return Err(corrupt("AIPTok v2 reserved header data is nonzero"));
    }

    let model_vocabulary_size = read_u32(data, 28)?;
    let defined_token_count = read_u32(data, 32)?;
    let bpe_vocabulary_size = read_u32(data, 36)?;
    let merge_count = read_u32(data, 40)?;
    let added_token_count = read_u32(data, 44)?;
    let special_token_count = read_u32(data, 48)?;
    if !(256..=MAX_VOCABULARY_SIZE).contains(&bpe_vocabulary_size)
        || defined_token_count < bpe_vocabulary_size
        || defined_token_count > MAX_VOCABULARY_SIZE
        || model_vocabulary_size < defined_token_count
        || model_vocabulary_size > MAX_VOCABULARY_SIZE
        || merge_count > MAX_BPE_MERGE_RULES
        || !(2..=MAX_ADDED_TOKENS).contains(&added_token_count)
        || !(2..=added_token_count).contains(&special_token_count)
    {
        return Err(corrupt("AIPTok v2 table count is outside hard bounds"));
    }
    if 256_u32.checked_add(merge_count) != Some(bpe_vocabulary_size)
        || bpe_vocabulary_size.checked_add(added_token_count) != Some(defined_token_count)
    {
        return Err(corrupt("AIPTok v2 vocabulary composition is not canonical"));
    }
    let beginning_of_sequence_token = read_u32(data, 52)?;
    let end_of_sequence_token = read_u32(data, 56)?;
    if beginning_of_sequence_token == end_of_sequence_token
        || beginning_of_sequence_token < bpe_vocabulary_size
        || end_of_sequence_token < bpe_vocabulary_size
        || beginning_of_sequence_token >= defined_token_count
        || end_of_sequence_token >= defined_token_count
    {
        return Err(corrupt("AIPTok v2 required token IDs are invalid"));
    }
    let maximum_input_bytes = read_u32(data, 60)?;
    let maximum_produced_tokens = read_u32(data, 64)?;
    if maximum_input_bytes == 0
        || usize::try_from(maximum_input_bytes).unwrap_or(usize::MAX) > MAX_REQUEST_BYTES
        || maximum_produced_tokens == 0
        || maximum_produced_tokens > MAX_TOKEN_COUNT
    {
        return Err(corrupt("AIPTok v2 input or output limit is invalid"));
    }
    let maximum_piece_bytes = read_u16(data, 68)?;
    if maximum_piece_bytes == 0
        || maximum_piece_bytes > MAX_TOKEN_PIECE_BYTES
        || read_u16(data, 70)? != 0
    {
        return Err(corrupt("AIPTok v2 piece limit or reserved field is invalid"));
    }
    let vocabulary_offset = u32_to_usize(read_u32(data, 72)?)?;
    let merge_offset = u32_to_usize(read_u32(data, 76)?)?;
    let added_offset = u32_to_usize(read_u32(data, 80)?)?;
    let byte_map_offset = u32_to_usize(read_u32(data, 84)?)?;
    if vocabulary_offset != AIPTOK_QWEN25_HEADER_BYTES
        || vocabulary_offset > merge_offset
        || merge_offset > added_offset
        || added_offset > byte_map_offset
        || read_u32(data, 88)? != usize_to_u32(data.len())?
        || checked_add(byte_map_offset, BYTE_MAP_BYTES)? != data.len()
    {
        return Err(corrupt("AIPTok v2 offsets or total length are invalid"));
    }
    let mut expected_hash = [0_u8; ModelPackageHash::BYTE_LEN];
    expected_hash.copy_from_slice(&data[112..160]);
    let content_hash = ModelPackageHash::from_sha384_bytes(expected_hash);
    if hash_package(&data[AIPTOK_QWEN25_HEADER_BYTES..]) != content_hash {
        return Err(corrupt("AIPTok v2 content hash does not match canonical tables"));
    }
    Ok(Header {
        descriptor: Qwen25AipTokDescriptor {
            model_vocabulary_size,
            defined_token_count,
            bpe_vocabulary_size,
            merge_count,
            added_token_count,
            special_token_count,
            beginning_of_sequence_token,
            end_of_sequence_token,
            maximum_input_bytes,
            maximum_produced_tokens,
            content_hash,
        },
        maximum_piece_bytes,
        vocabulary_offset,
        merge_offset,
        added_offset,
        byte_map_offset,
    })
}

fn parse_vocabulary(data: &[u8], header: &Header) -> Result<Vocabulary, EngineError> {
    let count = u32_to_usize(header.descriptor.defined_token_count)?;
    let bpe_count = header.descriptor.bpe_vocabulary_size;
    let mut pieces = Vec::new();
    let mut flags = Vec::new();
    let mut unique = BTreeSet::new();
    pieces
        .try_reserve_exact(count)
        .map_err(|_| EngineError::resource_exhaustion("AIPTok v2 vocabulary allocation failed"))?;
    flags
        .try_reserve_exact(count)
        .map_err(|_| EngineError::resource_exhaustion("AIPTok v2 flag allocation failed"))?;
    let mut cursor = header.vocabulary_offset;
    let mut actual_maximum = 0_usize;
    let mut observed_specials = 0_u32;
    for expected_id in 0..header.descriptor.defined_token_count {
        let token_id = read_u32(data, cursor)?;
        let token_flags = read_u16(data, checked_add(cursor, 4)?)?;
        let length = usize::from(read_u16(data, checked_add(cursor, 6)?)?);
        cursor = checked_add(cursor, VOCABULARY_ENTRY_HEADER_BYTES)?;
        if token_id != expected_id || token_flags & !KNOWN_VOCABULARY_FLAGS != 0 {
            return Err(corrupt("AIPTok v2 token IDs or flags are noncanonical"));
        }
        let is_added = token_flags & VOCABULARY_FLAG_ADDED != 0;
        if is_added != (expected_id >= bpe_count)
            || token_flags & VOCABULARY_FLAG_SPECIAL != 0 && !is_added
        {
            return Err(corrupt("AIPTok v2 added-token suffix or flags are invalid"));
        }
        if length == 0 || length > usize::from(header.maximum_piece_bytes) {
            return Err(corrupt("AIPTok v2 token piece length is invalid"));
        }
        let end = checked_add(cursor, length)?;
        let piece = data
            .get(cursor..end)
            .ok_or_else(|| corrupt("AIPTok v2 vocabulary piece is truncated"))?;
        if is_added {
            let text = std::str::from_utf8(piece)
                .map_err(|_| corrupt("AIPTok v2 added-token text is invalid UTF-8"))?;
            if piece.contains(&0) || text.chars().any(char::is_control) {
                return Err(corrupt("AIPTok v2 added-token text is not data-only text"));
            }
        }
        if !unique.insert(piece) {
            return Err(corrupt("AIPTok v2 vocabulary contains duplicate decoded bytes"));
        }
        if token_flags & VOCABULARY_FLAG_SPECIAL != 0 {
            observed_specials = observed_specials
                .checked_add(1)
                .ok_or_else(|| corrupt("AIPTok v2 special count overflows"))?;
        }
        pieces.push(piece.to_vec().into_boxed_slice());
        flags.push(token_flags);
        actual_maximum = actual_maximum.max(length);
        cursor = end;
    }
    if cursor != header.merge_offset
        || observed_specials != header.descriptor.special_token_count
        || actual_maximum != usize::from(header.maximum_piece_bytes)
    {
        return Err(corrupt("AIPTok v2 vocabulary metadata is noncanonical"));
    }
    Ok(Vocabulary { pieces, flags })
}

fn parse_merges(data: &[u8], header: &Header, vocabulary: &Vocabulary) -> Result<(), EngineError> {
    let merge_bytes = u32_to_usize(header.descriptor.merge_count)?
        .checked_mul(MERGE_RECORD_BYTES)
        .ok_or_else(|| corrupt("AIPTok v2 merge table length overflows"))?;
    if checked_add(header.merge_offset, merge_bytes)? != header.added_offset {
        return Err(corrupt("AIPTok v2 merge table length is noncanonical"));
    }
    let mut cursor = header.merge_offset;
    let mut pairs = BTreeSet::new();
    let mut results = BTreeSet::new();
    for expected_rank in 0..header.descriptor.merge_count {
        let rank = read_u32(data, cursor)?;
        let left = read_u32(data, checked_add(cursor, 4)?)?;
        let right = read_u32(data, checked_add(cursor, 8)?)?;
        let result = read_u32(data, checked_add(cursor, 12)?)?;
        cursor = checked_add(cursor, MERGE_RECORD_BYTES)?;
        if rank != expected_rank
            || left >= header.descriptor.bpe_vocabulary_size
            || right >= header.descriptor.bpe_vocabulary_size
            || !(256..header.descriptor.bpe_vocabulary_size).contains(&result)
            || result == left
            || result == right
        {
            return Err(corrupt("AIPTok v2 merge rank or reference is invalid"));
        }
        let left_piece = &vocabulary.pieces[u32_to_usize(left)?];
        let right_piece = &vocabulary.pieces[u32_to_usize(right)?];
        let result_piece = &vocabulary.pieces[u32_to_usize(result)?];
        let merged_length = left_piece
            .len()
            .checked_add(right_piece.len())
            .ok_or_else(|| corrupt("AIPTok v2 merge piece length overflows"))?;
        if result_piece.len() != merged_length
            || result_piece.get(..left_piece.len()) != Some(left_piece)
            || result_piece.get(left_piece.len()..) != Some(right_piece)
        {
            return Err(corrupt("AIPTok v2 merge result bytes are invalid"));
        }
        if !pairs.insert((left, right)) || !results.insert(result) {
            return Err(corrupt("AIPTok v2 merge pair or result is duplicated"));
        }
    }
    if (256..header.descriptor.bpe_vocabulary_size).any(|token| !results.contains(&token)) {
        return Err(corrupt("AIPTok v2 BPE token lacks one defining merge"));
    }
    Ok(())
}

fn parse_added_tokens(
    data: &[u8],
    header: &Header,
    vocabulary: &Vocabulary,
) -> Result<(), EngineError> {
    let added_bytes = u32_to_usize(header.descriptor.added_token_count)?
        .checked_mul(ADDED_RECORD_BYTES)
        .ok_or_else(|| corrupt("AIPTok v2 added-token length overflows"))?;
    if checked_add(header.added_offset, added_bytes)? != header.byte_map_offset {
        return Err(corrupt("AIPTok v2 added-token table length is noncanonical"));
    }
    let mut cursor = header.added_offset;
    let mut literals = Vec::new();
    let mut bos_count = 0_u32;
    let mut eos_count = 0_u32;
    for index in 0..header.descriptor.added_token_count {
        let token = read_u32(data, cursor)?;
        let role = AipTokQwen25AddedRole::from_wire(read_u16(data, checked_add(cursor, 4)?)?)?;
        if read_u16(data, checked_add(cursor, 6)?)? != 0 {
            return Err(corrupt("AIPTok v2 added-token reserved field is nonzero"));
        }
        cursor = checked_add(cursor, ADDED_RECORD_BYTES)?;
        let expected = header
            .descriptor
            .bpe_vocabulary_size
            .checked_add(index)
            .ok_or_else(|| corrupt("AIPTok v2 added-token ID overflows"))?;
        if token != expected {
            return Err(corrupt("AIPTok v2 added-token IDs are noncanonical"));
        }
        let flags = vocabulary.flags[u32_to_usize(token)?];
        if (flags & VOCABULARY_FLAG_SPECIAL != 0) != role.is_special() {
            return Err(corrupt("AIPTok v2 added-token role and flags differ"));
        }
        match role {
            AipTokQwen25AddedRole::BeginningOfSequence => {
                bos_count = bos_count
                    .checked_add(1)
                    .ok_or_else(|| corrupt("AIPTok v2 BOS role count overflows"))?;
                if token != header.descriptor.beginning_of_sequence_token {
                    return Err(corrupt("AIPTok v2 BOS role uses the wrong token"));
                }
            }
            AipTokQwen25AddedRole::EndOfSequence => {
                eos_count = eos_count
                    .checked_add(1)
                    .ok_or_else(|| corrupt("AIPTok v2 EOS role count overflows"))?;
                if token != header.descriptor.end_of_sequence_token {
                    return Err(corrupt("AIPTok v2 EOS role uses the wrong token"));
                }
            }
            AipTokQwen25AddedRole::Special | AipTokQwen25AddedRole::Ordinary => {}
        }
        literals.push(&vocabulary.pieces[u32_to_usize(token)?]);
    }
    if bos_count != 1 || eos_count != 1 {
        return Err(corrupt("AIPTok v2 required added-token roles are missing or duplicated"));
    }
    if header.descriptor.matches_pinned_qwen25_1_5b_counts() {
        for (index, (expected_literal, expected_role)) in
            PINNED_QWEN25_ADDED_TOKENS.iter().enumerate()
        {
            let token = header
                .descriptor
                .bpe_vocabulary_size
                .checked_add(
                    u32::try_from(index)
                        .map_err(|_| corrupt("pinned Qwen2.5 added-token index overflows"))?,
                )
                .ok_or_else(|| corrupt("pinned Qwen2.5 added-token ID overflows"))?;
            let record_offset = header
                .added_offset
                .checked_add(
                    index
                        .checked_mul(ADDED_RECORD_BYTES)
                        .ok_or_else(|| corrupt("pinned Qwen2.5 record offset overflows"))?,
                )
                .ok_or_else(|| corrupt("pinned Qwen2.5 record offset overflows"))?;
            let observed_role =
                AipTokQwen25AddedRole::from_wire(read_u16(data, checked_add(record_offset, 4)?)?)?;
            if observed_role != *expected_role
                || vocabulary.pieces[u32_to_usize(token)?].as_ref() != expected_literal.as_bytes()
            {
                return Err(corrupt("AIPTok v2 pinned Qwen2.5 added-token profile differs"));
            }
        }
    }
    for (index, literal) in literals.iter().enumerate() {
        if literals.iter().enumerate().any(|(other_index, other)| {
            index != other_index && (literal.starts_with(other) || other.starts_with(literal))
        }) {
            return Err(corrupt("AIPTok v2 added-token literals have a prefix collision"));
        }
    }
    Ok(())
}

fn parse_byte_map(
    data: &[u8],
    header: &Header,
    vocabulary: &Vocabulary,
) -> Result<(), EngineError> {
    let mut observed = BTreeSet::new();
    let mut cursor = header.byte_map_offset;
    for byte in 0_u16..=255 {
        let token = read_u32(data, cursor)?;
        cursor = checked_add(cursor, size_of::<u32>())?;
        if token >= 256 || !observed.insert(token) {
            return Err(corrupt("AIPTok v2 byte-token map is not a base-token permutation"));
        }
        let expected_byte = u8::try_from(byte)
            .map_err(|_| corrupt("AIPTok v2 byte-map index is not representable"))?;
        if vocabulary.pieces[u32_to_usize(token)?].as_ref() != [expected_byte] {
            return Err(corrupt("AIPTok v2 byte-token map does not match decoded bytes"));
        }
    }
    if cursor != data.len() {
        return Err(corrupt("AIPTok v2 byte-token map has trailing data"));
    }
    Ok(())
}

fn u32_to_usize(value: u32) -> Result<usize, EngineError> {
    usize::try_from(value).map_err(|_| corrupt("AIPTok v2 value is not addressable"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> AipTokQwen25Definition {
        let mut vocabulary = (0_u32..256)
            .map(|token_id| AipTokQwen25VocabularyEntry {
                token_id,
                bytes: vec![255_u8 - u8::try_from(token_id).expect("base ID fits")],
            })
            .collect::<Vec<_>>();
        vocabulary.extend([
            AipTokQwen25VocabularyEntry { token_id: 256, bytes: b"ab".to_vec() },
            AipTokQwen25VocabularyEntry { token_id: 257, bytes: b"abc".to_vec() },
            AipTokQwen25VocabularyEntry { token_id: 258, bytes: b"<|endoftext|>".to_vec() },
            AipTokQwen25VocabularyEntry { token_id: 259, bytes: b"<|im_start|>".to_vec() },
            AipTokQwen25VocabularyEntry { token_id: 260, bytes: b"<|im_end|>".to_vec() },
            AipTokQwen25VocabularyEntry { token_id: 261, bytes: b"<tool_call>".to_vec() },
        ]);
        let byte_token_ids = std::array::from_fn(|byte| {
            u32::try_from(255_usize - byte).expect("byte map index fits")
        });
        AipTokQwen25Definition {
            vocabulary,
            bpe_vocabulary_size: 258,
            model_vocabulary_size: 270,
            merges: vec![
                AipTokQwen25Merge { rank: 0, left: 158, right: 157, result: 256 },
                AipTokQwen25Merge { rank: 1, left: 256, right: 156, result: 257 },
            ],
            added_tokens: vec![
                AipTokQwen25AddedToken {
                    token_id: 258,
                    role: AipTokQwen25AddedRole::BeginningOfSequence,
                },
                AipTokQwen25AddedToken { token_id: 259, role: AipTokQwen25AddedRole::Special },
                AipTokQwen25AddedToken {
                    token_id: 260,
                    role: AipTokQwen25AddedRole::EndOfSequence,
                },
                AipTokQwen25AddedToken { token_id: 261, role: AipTokQwen25AddedRole::Ordinary },
            ],
            byte_token_ids,
            beginning_of_sequence_token: 258,
            end_of_sequence_token: 260,
            maximum_input_bytes: 4096,
            maximum_produced_tokens: 4096,
        }
    }

    fn rehash(bytes: &mut [u8]) {
        let digest = hash_package(&bytes[AIPTOK_QWEN25_HEADER_BYTES..]);
        bytes[112..160].copy_from_slice(digest.as_bytes());
    }

    #[test]
    fn compact_tables_are_deterministic_bounded_and_preserve_undefined_tail() {
        let first = fixture().encode().expect("fixture must encode");
        let second = fixture().encode().expect("fixture must encode again");
        assert_eq!(first, second);
        let pool = MemoryPool::new(16 * 1024 * 1024).expect("pool validates");
        let descriptor = inspect_qwen25_aiptok(&first, &pool).expect("fixture must inspect");
        assert_eq!(descriptor.model_vocabulary_size(), 270);
        assert_eq!(descriptor.defined_token_count(), 262);
        assert_eq!(descriptor.unknown_token(), None);
        assert!(!descriptor.inserts_bos());
        descriptor.validate_token(261).expect("last defined ID works");
        assert!(descriptor.validate_token(262).is_err());
        assert!(!descriptor.matches_pinned_qwen25_1_5b_counts());

        let shared = std::sync::Arc::new(first);
        let threads = (0..8)
            .map(|_| {
                let bytes = std::sync::Arc::clone(&shared);
                std::thread::spawn(move || {
                    let pool = MemoryPool::new(16 * 1024 * 1024).expect("pool validates");
                    inspect_qwen25_aiptok(&bytes, &pool)
                })
            })
            .collect::<Vec<_>>();
        for thread in threads {
            assert_eq!(
                thread.join().expect("thread joins").expect("parser remains deterministic"),
                descriptor
            );
        }
    }

    #[test]
    fn byte_map_covers_every_byte_exactly() {
        let bytes = fixture().encode().expect("fixture must encode");
        let header = parse_header(&bytes).expect("header must parse");
        let vocabulary = parse_vocabulary(&bytes, &header).expect("vocabulary must parse");
        parse_byte_map(&bytes, &header, &vocabulary).expect("byte map must validate");
        let observed =
            vocabulary.pieces[..256].iter().map(|piece| piece[0]).collect::<BTreeSet<_>>();
        assert_eq!(observed.len(), 256);
    }

    #[test]
    fn parser_rejects_truncation_unknown_profiles_and_malformed_tables() {
        let valid = fixture().encode().expect("fixture must encode");
        let pool = MemoryPool::new(16 * 1024 * 1024).expect("pool validates");
        for end in 0..valid.len() {
            assert!(inspect_qwen25_aiptok(&valid[..end], &pool).is_err());
        }
        let mut cases = Vec::new();
        let mut unknown_profile = valid.clone();
        unknown_profile[92..96].copy_from_slice(&2_u32.to_le_bytes());
        cases.push(unknown_profile);
        let mut undefined_flags = valid.clone();
        undefined_flags[AIPTOK_QWEN25_HEADER_BYTES + 4..AIPTOK_QWEN25_HEADER_BYTES + 6]
            .copy_from_slice(&4_u16.to_le_bytes());
        rehash(&mut undefined_flags);
        cases.push(undefined_flags);
        let byte_map = usize::try_from(u32::from_le_bytes(valid[84..88].try_into().unwrap()))
            .expect("offset fits");
        let mut duplicate_byte_map = valid.clone();
        let first_mapping: [u8; 4] = duplicate_byte_map[byte_map..byte_map + 4]
            .try_into()
            .expect("mapping record length is fixed");
        duplicate_byte_map[byte_map + 4..byte_map + 8].copy_from_slice(&first_mapping);
        rehash(&mut duplicate_byte_map);
        cases.push(duplicate_byte_map);
        let merge_offset = usize::try_from(u32::from_le_bytes(valid[76..80].try_into().unwrap()))
            .expect("offset fits");
        let mut invalid_merge = valid.clone();
        invalid_merge[merge_offset..merge_offset + 4].copy_from_slice(&1_u32.to_le_bytes());
        rehash(&mut invalid_merge);
        cases.push(invalid_merge);
        let mut trailing = valid.clone();
        trailing.push(0);
        cases.push(trailing);
        let mut excessive_tokens = valid.clone();
        excessive_tokens[64..68].copy_from_slice(&(MAX_TOKEN_COUNT + 1).to_le_bytes());
        cases.push(excessive_tokens);
        for case in cases {
            assert!(inspect_qwen25_aiptok(&case, &pool).is_err());
        }
    }

    #[test]
    fn parser_respects_aggregate_memory_and_declared_input_limits() {
        let valid = fixture().encode().expect("fixture must encode");
        let tiny_pool = MemoryPool::new(1).expect("pool limit validates");
        assert!(matches!(
            inspect_qwen25_aiptok(&valid, &tiny_pool),
            Err(EngineError::ResourceExhaustion(_))
        ));

        let mut excessive_input = valid;
        let over_limit = u32::try_from(
            MAX_REQUEST_BYTES.checked_add(1).expect("request limit is not usize::MAX"),
        )
        .expect("request limit fits the wire field");
        excessive_input[60..64].copy_from_slice(&over_limit.to_le_bytes());
        let pool = MemoryPool::new(16 * 1024 * 1024).expect("pool validates");
        assert!(matches!(
            inspect_qwen25_aiptok(&excessive_input, &pool),
            Err(EngineError::CorruptModelPackage(_))
        ));
    }

    #[test]
    fn every_table_single_byte_mutation_is_rejected_by_content_hash() {
        let valid = fixture().encode().expect("fixture must encode");
        let pool = MemoryPool::new(16 * 1024 * 1024).expect("pool validates");
        for index in AIPTOK_QWEN25_HEADER_BYTES..valid.len() {
            let mut mutated = valid.clone();
            mutated[index] ^= 0x80;
            assert!(
                inspect_qwen25_aiptok(&mutated, &pool).is_err(),
                "mutation at byte {index} was accepted"
            );
        }
    }

    #[test]
    fn added_token_prefix_collision_and_invalid_text_fail_closed() {
        let mut collision = fixture();
        collision.vocabulary[259].bytes = b"<|endoftext|>suffix".to_vec();
        assert!(collision.encode().is_err());

        let mut nul = fixture();
        nul.vocabulary[261].bytes = b"tool\0call".to_vec();
        assert!(nul.encode().is_err());
    }

    #[test]
    fn runtime_factory_validates_then_refuses_unqualified_v2_execution() {
        let bytes = fixture().encode().expect("fixture must encode");
        let pool = MemoryPool::new(16 * 1024 * 1024).expect("pool validates");
        assert!(matches!(
            super::super::FirstPartyBpeTokenizer::parse(&bytes, &pool),
            Err(EngineError::UnsupportedTokenizer(_))
        ));
        let mut malformed = bytes;
        malformed[112] ^= 1;
        assert!(matches!(
            super::super::FirstPartyBpeTokenizer::parse(&malformed, &pool),
            Err(EngineError::CorruptModelPackage(_))
        ));
    }

    #[test]
    fn pinned_added_token_profile_is_closed_and_has_exact_roles() {
        assert_eq!(PINNED_QWEN25_ADDED_TOKENS.len(), 22);
        assert_eq!(
            PINNED_QWEN25_ADDED_TOKENS.iter().filter(|(_, role)| role.is_special()).count(),
            usize::try_from(PINNED_QWEN25_SPECIAL_TOKEN_COUNT).expect("count fits")
        );
        assert_eq!(
            PINNED_QWEN25_ADDED_TOKENS[0],
            ("<|endoftext|>", AipTokQwen25AddedRole::BeginningOfSequence)
        );
        assert_eq!(
            PINNED_QWEN25_ADDED_TOKENS[2],
            ("<|im_end|>", AipTokQwen25AddedRole::EndOfSequence)
        );
    }
}
