//! First-party, bounded AIPTok Byte-Level BPE v1 tokenizer.
//!
//! AIPTok preserves input bytes exactly. It performs no Unicode or locale
//! normalization, contains no scripts/templates/plugins, and selects no code.
//! The only accepted family is a fixed byte-fallback BPE with declarative merge
//! ranks and an explicit, caller-selected special-token policy.

use personalai_engine_crypto::hash_package;
use personalai_engine_memory::{MemoryPool, MemoryReservation};
use personalai_engine_types::{
    EngineError, MAX_REQUEST_BYTES, MAX_TOKEN_COUNT, MAX_TOKENIZER_DATA_BYTES, ModelPackageHash,
};
use std::cmp::Reverse;
use std::collections::{BTreeMap, BTreeSet, BinaryHeap};
use std::mem::size_of;

/// Maximum vocabulary size accepted by the engine.
pub use personalai_engine_types::MAX_VOCABULARY_SIZE;

/// Exact AIPTok Byte-Level BPE v1 magic.
pub const AIPTOK_MAGIC: [u8; 8] = *b"AIPTOK01";
/// Fixed AIPTok v1 header length.
pub const AIPTOK_HEADER_BYTES: usize = 128;
/// Maximum bytes represented by one vocabulary entry.
pub const MAX_TOKEN_PIECE_BYTES: u16 = 256;
/// Maximum merge records.
pub const MAX_BPE_MERGE_RULES: u32 = MAX_VOCABULARY_SIZE - 256;
/// Maximum declared special tokens.
pub const MAX_SPECIAL_TOKENS: u32 = 4_096;

const AIPTOK_FAMILY_BYTE_BPE: u16 = 1;
const NORMALIZATION_PRESERVE_BYTES: u16 = 1;
const DECODER_CONCATENATE_SKIP_SPECIAL: u16 = 1;
const UNKNOWN_BYTE_FALLBACK: u16 = 1;
const VOCABULARY_ENTRY_HEADER_BYTES: usize = 8;
const MERGE_RULE_BYTES: usize = 16;
const SPECIAL_RECORD_BYTES: usize = 8;
const SPECIAL_VOCABULARY_FLAG: u16 = 1;
const SPECIAL_ROLE_BOS: u32 = 1;
const SPECIAL_ROLE_EOS: u32 = 2;
const SPECIAL_ROLE_UNKNOWN: u32 = 4;
const SPECIAL_ROLE_OTHER: u32 = 8;
const TEMPORARY_BYTES_PER_INPUT_TOKEN: u64 = 256;
const PERSISTENT_BYTES_PER_VOCABULARY_ENTRY: u64 = 96;
const PERSISTENT_BYTES_PER_MERGE_RULE: u64 = 128;

type SpecialLiteral = (Box<[u8]>, u32);

/// Engine-owned tokenizer data format identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenizerFormat {
    /// AIPTok Byte-Level BPE v1.
    AipTokByteLevelBpeV1,
}

/// Deliberately narrow input normalization policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NormalizationPolicy {
    /// Preserve the exact caller-supplied bytes.
    PreserveBytes,
}

/// Behavior when a byte has no learned vocabulary entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnknownTokenPolicy {
    /// Use the mandatory identity token for that byte; never emit UNK for bytes.
    ByteFallback,
}

/// Fixed decode behavior for the initial family.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecoderBehavior {
    /// Concatenate ordinary token bytes and omit special/control tokens.
    ConcatenateBytesSkipSpecial,
}

/// Role assigned to one declared special token.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpecialTokenRole {
    /// Required beginning-of-sequence token.
    BeginningOfSequence,
    /// Required end-of-sequence token.
    EndOfSequence,
    /// Required unknown marker; byte fallback prevents normal emission.
    Unknown,
    /// Additional data-only special marker.
    Other,
}

impl SpecialTokenRole {
    const fn wire(self) -> u32 {
        match self {
            Self::BeginningOfSequence => SPECIAL_ROLE_BOS,
            Self::EndOfSequence => SPECIAL_ROLE_EOS,
            Self::Unknown => SPECIAL_ROLE_UNKNOWN,
            Self::Other => SPECIAL_ROLE_OTHER,
        }
    }

    fn from_wire(value: u32) -> Result<Self, EngineError> {
        match value {
            SPECIAL_ROLE_BOS => Ok(Self::BeginningOfSequence),
            SPECIAL_ROLE_EOS => Ok(Self::EndOfSequence),
            SPECIAL_ROLE_UNKNOWN => Ok(Self::Unknown),
            SPECIAL_ROLE_OTHER => Ok(Self::Other),
            _ => Err(corrupt("AIPTok special-token role is invalid")),
        }
    }
}

/// Caller-selected treatment of literal special-token text.
#[derive(Debug, Clone, Copy)]
pub enum SpecialTokenPolicy<'a> {
    /// Reject input containing any declared special-token text.
    Reject,
    /// Treat special-token text as ordinary bytes and apply normal BPE.
    TreatAsOrdinaryBytes,
    /// Allow every declared special token to be recognized atomically.
    AllowAll,
    /// Allow only the listed declared special-token identifiers.
    AllowListed(&'a [u32]),
}

/// Bounded encoding controls.
#[derive(Debug, Clone, Copy)]
pub struct EncodeOptions<'a> {
    add_beginning_of_sequence: bool,
    add_end_of_sequence: bool,
    special_tokens: SpecialTokenPolicy<'a>,
}

impl<'a> EncodeOptions<'a> {
    /// Constructs explicit encoding controls.
    pub const fn new(
        add_beginning_of_sequence: bool,
        add_end_of_sequence: bool,
        special_tokens: SpecialTokenPolicy<'a>,
    ) -> Self {
        Self { add_beginning_of_sequence, add_end_of_sequence, special_tokens }
    }

    /// Exact byte-round-trip controls: no implicit control tokens or recognition.
    pub const fn byte_round_trip() -> Self {
        Self::new(false, false, SpecialTokenPolicy::TreatAsOrdinaryBytes)
    }
}

impl Default for EncodeOptions<'_> {
    fn default() -> Self {
        Self::new(true, false, SpecialTokenPolicy::Reject)
    }
}

/// Validated immutable AIPTok metadata from a signed package.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenizerSpec {
    format: TokenizerFormat,
    normalization: NormalizationPolicy,
    unknown_policy: UnknownTokenPolicy,
    decoder_behavior: DecoderBehavior,
    vocabulary_size: u32,
    beginning_of_sequence_token: u32,
    end_of_sequence_token: u32,
    unknown_token: u32,
    maximum_piece_bytes: u16,
    maximum_input_bytes: u32,
    maximum_produced_tokens: u32,
    content_hash: ModelPackageHash,
}

impl TokenizerSpec {
    /// Returns the only supported tokenizer format.
    pub const fn format(&self) -> TokenizerFormat {
        self.format
    }
    /// Returns the exact no-normalization policy.
    pub const fn normalization(&self) -> NormalizationPolicy {
        self.normalization
    }
    /// Returns the byte-fallback unknown policy.
    pub const fn unknown_policy(&self) -> UnknownTokenPolicy {
        self.unknown_policy
    }
    /// Returns the fixed decoder policy.
    pub const fn decoder_behavior(&self) -> DecoderBehavior {
        self.decoder_behavior
    }
    /// Returns the declared contiguous vocabulary size.
    pub const fn vocabulary_size(&self) -> u32 {
        self.vocabulary_size
    }
    /// Returns the required BOS token.
    pub const fn beginning_of_sequence_token(&self) -> u32 {
        self.beginning_of_sequence_token
    }
    /// Returns the required EOS token.
    pub const fn end_of_sequence_token(&self) -> u32 {
        self.end_of_sequence_token
    }
    /// Returns the required unknown marker.
    pub const fn unknown_token(&self) -> u32 {
        self.unknown_token
    }
    /// Returns the maximum bytes in one token piece.
    pub const fn maximum_piece_bytes(&self) -> u16 {
        self.maximum_piece_bytes
    }
    /// Returns the package-declared input byte ceiling.
    pub const fn maximum_input_bytes(&self) -> u32 {
        self.maximum_input_bytes
    }
    /// Returns the package-declared produced-token ceiling.
    pub const fn maximum_produced_tokens(&self) -> u32 {
        self.maximum_produced_tokens
    }
    /// Returns SHA-384 over the canonical content following the header.
    pub const fn content_hash(&self) -> ModelPackageHash {
        self.content_hash
    }

    /// Validates a bounded sequence of token identifiers.
    pub fn validate_tokens(&self, tokens: &[u32]) -> Result<(), EngineError> {
        let maximum = usize::try_from(self.maximum_produced_tokens)
            .map_err(|_| EngineError::resource_exhaustion("token ceiling is not addressable"))?;
        if tokens.len() > maximum
            || tokens.len() > usize::try_from(MAX_TOKEN_COUNT).unwrap_or(usize::MAX)
        {
            return Err(EngineError::invalid_input("token sequence exceeds the tokenizer limit"));
        }
        if tokens.iter().any(|token| *token >= self.vocabulary_size) {
            return Err(EngineError::invalid_input(
                "token sequence contains an out-of-vocabulary identifier",
            ));
        }
        Ok(())
    }
}

/// Contract implemented only by reviewed engine-owned tokenizer code.
pub trait Tokenizer: Send + Sync {
    /// Returns immutable tokenizer metadata.
    fn spec(&self) -> &TokenizerSpec;
    /// Encodes UTF-8 with BOS enabled and literal special text rejected.
    fn encode(&self, input: &str) -> Result<Vec<u32>, EngineError>;
    /// Encodes UTF-8 with explicit special/control behavior.
    fn encode_with_options(
        &self,
        input: &str,
        options: EncodeOptions<'_>,
    ) -> Result<Vec<u32>, EngineError>;
    /// Encodes arbitrary bytes. Invalid UTF-8 is preserved, never repaired.
    fn encode_bytes_with_options(
        &self,
        input: &[u8],
        options: EncodeOptions<'_>,
    ) -> Result<Vec<u32>, EngineError>;
    /// Decodes ordinary token bytes as strict UTF-8; invalid output is rejected.
    fn decode(&self, tokens: &[u32]) -> Result<String, EngineError>;
    /// Decodes arbitrary token bytes exactly, omitting declared special tokens.
    fn decode_bytes(&self, tokens: &[u32]) -> Result<Vec<u8>, EngineError>;
    /// Returns the immutable bytes for one token.
    fn token_piece(&self, token: u32) -> Result<&[u8], EngineError>;
    /// Returns a token identifier for exact vocabulary bytes.
    fn vocabulary_lookup(&self, piece: &[u8]) -> Option<u32>;
    /// Returns whether a validated token is special/control data.
    fn is_special_token(&self, token: u32) -> Result<bool, EngineError>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct MergeRule {
    rank: u32,
    result: u32,
}

#[derive(Debug)]
struct Node {
    token: u32,
    previous: Option<usize>,
    next: Option<usize>,
    alive: bool,
}

/// Parsed immutable AIPTok Byte-Level BPE v1 tokenizer.
#[derive(Debug)]
pub struct FirstPartyBpeTokenizer {
    spec: TokenizerSpec,
    pieces: Vec<Box<[u8]>>,
    special: Vec<bool>,
    vocabulary: BTreeMap<Box<[u8]>, u32>,
    special_literals: Vec<SpecialLiteral>,
    merges: BTreeMap<(u32, u32), MergeRule>,
    pool: MemoryPool,
    _reservation: MemoryReservation,
}

impl FirstPartyBpeTokenizer {
    /// Parses canonical AIPTok data under an aggregate memory pool.
    pub fn parse(data: &[u8], pool: &MemoryPool) -> Result<Self, EngineError> {
        let admission = parse_header(data)?;
        let reservation_bytes = persistent_reservation_bytes(
            data.len(),
            admission.spec.vocabulary_size,
            admission.merge_count,
        )?;
        let reservation = pool.reserve(reservation_bytes)?;
        let parsed = ParsedTokenizerData::parse(data)?;
        Ok(Self {
            spec: parsed.spec,
            pieces: parsed.pieces,
            special: parsed.special,
            vocabulary: parsed.vocabulary,
            special_literals: parsed.special_literals,
            merges: parsed.merges,
            pool: pool.clone(),
            _reservation: reservation,
        })
    }

    fn push_candidate(
        &self,
        nodes: &[Node],
        position: usize,
        heap: &mut BinaryHeap<Reverse<(u32, usize)>>,
    ) {
        let Some(left) = nodes.get(position).filter(|node| node.alive) else {
            return;
        };
        let Some(next) = left.next else {
            return;
        };
        let Some(right) = nodes.get(next).filter(|node| node.alive) else {
            return;
        };
        if let Some(rule) = self.merges.get(&(left.token, right.token)) {
            heap.push(Reverse((rule.rank, position)));
        }
    }

    fn apply_merges(&self, bytes: &[u8]) -> Result<Vec<u32>, EngineError> {
        if bytes.is_empty() {
            return Ok(Vec::new());
        }
        let count = bytes.len();
        let temporary_bytes = u64::try_from(count)
            .map_err(|_| EngineError::resource_exhaustion("token count is not representable"))?
            .checked_mul(TEMPORARY_BYTES_PER_INPUT_TOKEN)
            .ok_or_else(|| EngineError::resource_exhaustion("tokenizer work memory overflows"))?;
        let _temporary = self.pool.reserve(temporary_bytes)?;
        let mut nodes = Vec::new();
        nodes
            .try_reserve_exact(count)
            .map_err(|_| EngineError::resource_exhaustion("tokenizer node allocation failed"))?;
        for (index, byte) in bytes.iter().enumerate() {
            nodes.push(Node {
                token: u32::from(*byte),
                previous: index.checked_sub(1),
                next: (index + 1 < count).then_some(index + 1),
                alive: true,
            });
        }
        let heap_capacity = count
            .checked_mul(3)
            .ok_or_else(|| EngineError::resource_exhaustion("merge heap bound overflows"))?;
        let mut heap = BinaryHeap::new();
        heap.try_reserve(heap_capacity)
            .map_err(|_| EngineError::resource_exhaustion("merge heap allocation failed"))?;
        for position in 0..count {
            self.push_candidate(&nodes, position, &mut heap);
        }
        while let Some(Reverse((rank, position))) = heap.pop() {
            let Some(left) = nodes.get(position).filter(|node| node.alive) else {
                continue;
            };
            let Some(right_position) = left.next else {
                continue;
            };
            let Some(right) = nodes.get(right_position).filter(|node| node.alive) else {
                continue;
            };
            let Some(rule) = self.merges.get(&(left.token, right.token)) else {
                continue;
            };
            if rule.rank != rank {
                continue;
            }
            let previous = left.previous;
            let following = right.next;
            nodes[position].token = rule.result;
            nodes[position].next = following;
            nodes[right_position].alive = false;
            nodes[right_position].previous = None;
            nodes[right_position].next = None;
            if let Some(following_position) = following {
                nodes[following_position].previous = Some(position);
            }
            if let Some(previous_position) = previous {
                self.push_candidate(&nodes, previous_position, &mut heap);
            }
            self.push_candidate(&nodes, position, &mut heap);
        }
        let mut tokens = Vec::new();
        tokens
            .try_reserve_exact(count)
            .map_err(|_| EngineError::resource_exhaustion("token output allocation failed"))?;
        let mut position = Some(0);
        while let Some(index) = position {
            let node = nodes
                .get(index)
                .ok_or_else(|| EngineError::internal_invariant("token node is unavailable"))?;
            if !node.alive {
                return Err(EngineError::internal_invariant("token chain contains a dead node"));
            }
            tokens.push(node.token);
            position = node.next;
        }
        Ok(tokens)
    }

    fn append_segment(&self, segment: &[u8], output: &mut Vec<u32>) -> Result<(), EngineError> {
        let segment_tokens = self.apply_merges(segment)?;
        let projected = output
            .len()
            .checked_add(segment_tokens.len())
            .ok_or_else(|| EngineError::resource_exhaustion("encoded token count overflows"))?;
        if projected > usize::try_from(self.spec.maximum_produced_tokens).unwrap_or(usize::MAX) {
            return Err(EngineError::resource_exhaustion(
                "encoded token count exceeds the tokenizer limit",
            ));
        }
        output.extend(segment_tokens);
        Ok(())
    }

    fn matching_special(&self, input: &[u8]) -> Option<(u32, usize)> {
        if !input.starts_with(b"<|") {
            return None;
        }
        let maximum = input.len().min(usize::from(MAX_TOKEN_PIECE_BYTES));
        let end = (2..maximum.saturating_sub(1))
            .find(|index| input.get(*index..index + 2) == Some(b"|>"))?
            .checked_add(2)?;
        self.special_literals
            .iter()
            .find(|(literal, _)| literal.as_ref() == &input[..end])
            .map(|(_, token)| (*token, end))
    }

    fn encode_bytes_inner(
        &self,
        input: &[u8],
        options: EncodeOptions<'_>,
    ) -> Result<Vec<u32>, EngineError> {
        if input.len() > usize::try_from(self.spec.maximum_input_bytes).unwrap_or(usize::MAX)
            || input.len() > MAX_REQUEST_BYTES
        {
            return Err(EngineError::invalid_input("tokenizer input exceeds its byte limit"));
        }
        if let SpecialTokenPolicy::AllowListed(allowed) = options.special_tokens {
            if allowed.len() > usize::try_from(MAX_SPECIAL_TOKENS).unwrap_or(usize::MAX) {
                return Err(EngineError::invalid_input("allowed special-token list is too large"));
            }
            let mut unique = BTreeSet::new();
            for token in allowed {
                if !unique.insert(*token) || !self.is_special_token(*token)? {
                    return Err(EngineError::invalid_input(
                        "allowed special-token list is invalid",
                    ));
                }
            }
        }
        let mut output = Vec::new();
        output
            .try_reserve(
                input
                    .len()
                    .min(usize::try_from(self.spec.maximum_produced_tokens).unwrap_or(usize::MAX)),
            )
            .map_err(|_| EngineError::resource_exhaustion("encoded token allocation failed"))?;
        if options.add_beginning_of_sequence {
            output.push(self.spec.beginning_of_sequence_token);
        }
        match options.special_tokens {
            SpecialTokenPolicy::TreatAsOrdinaryBytes => self.append_segment(input, &mut output)?,
            policy => {
                let mut segment_start = 0;
                let mut position = 0;
                while position < input.len() {
                    let Some((token, length)) = self.matching_special(&input[position..]) else {
                        position = position.checked_add(1).ok_or_else(|| {
                            EngineError::resource_exhaustion("special-token scan overflows")
                        })?;
                        continue;
                    };
                    self.append_segment(&input[segment_start..position], &mut output)?;
                    let allowed = match policy {
                        SpecialTokenPolicy::Reject => false,
                        SpecialTokenPolicy::AllowAll => true,
                        SpecialTokenPolicy::AllowListed(tokens) => tokens.contains(&token),
                        SpecialTokenPolicy::TreatAsOrdinaryBytes => unreachable!(),
                    };
                    if !allowed {
                        return Err(EngineError::invalid_input(
                            "input contains special-token text that was not explicitly allowed",
                        ));
                    }
                    output.push(token);
                    if output.len()
                        > usize::try_from(self.spec.maximum_produced_tokens).unwrap_or(usize::MAX)
                    {
                        return Err(EngineError::resource_exhaustion(
                            "encoded token count exceeds the tokenizer limit",
                        ));
                    }
                    position = position.checked_add(length).ok_or_else(|| {
                        EngineError::resource_exhaustion("special-token scan overflows")
                    })?;
                    segment_start = position;
                }
                self.append_segment(&input[segment_start..], &mut output)?;
            }
        }
        if options.add_end_of_sequence {
            output.push(self.spec.end_of_sequence_token);
        }
        self.spec.validate_tokens(&output)?;
        Ok(output)
    }

    fn token_index(&self, token: u32) -> Result<usize, EngineError> {
        if token >= self.spec.vocabulary_size {
            return Err(EngineError::invalid_input(
                "token identifier is outside the tokenizer vocabulary",
            ));
        }
        usize::try_from(token)
            .map_err(|_| EngineError::invalid_input("token identifier is not addressable"))
    }
}

impl Tokenizer for FirstPartyBpeTokenizer {
    fn spec(&self) -> &TokenizerSpec {
        &self.spec
    }

    fn encode(&self, input: &str) -> Result<Vec<u32>, EngineError> {
        self.encode_with_options(input, EncodeOptions::default())
    }

    fn encode_with_options(
        &self,
        input: &str,
        options: EncodeOptions<'_>,
    ) -> Result<Vec<u32>, EngineError> {
        self.encode_bytes_inner(input.as_bytes(), options)
    }

    fn encode_bytes_with_options(
        &self,
        input: &[u8],
        options: EncodeOptions<'_>,
    ) -> Result<Vec<u32>, EngineError> {
        self.encode_bytes_inner(input, options)
    }

    fn decode(&self, tokens: &[u32]) -> Result<String, EngineError> {
        String::from_utf8(self.decode_bytes(tokens)?)
            .map_err(|_| EngineError::invalid_input("token sequence does not decode to UTF-8"))
    }

    fn decode_bytes(&self, tokens: &[u32]) -> Result<Vec<u8>, EngineError> {
        self.spec.validate_tokens(tokens)?;
        let mut decoded_bytes = 0_usize;
        for token in tokens {
            let index = self.token_index(*token)?;
            if !self.special[index] {
                decoded_bytes = decoded_bytes
                    .checked_add(self.pieces[index].len())
                    .filter(|bytes| *bytes <= MAX_REQUEST_BYTES)
                    .ok_or_else(|| {
                        EngineError::resource_exhaustion("decoded bytes exceed the hard limit")
                    })?;
            }
        }
        let _temporary = if decoded_bytes == 0 {
            None
        } else {
            Some(self.pool.reserve(u64::try_from(decoded_bytes).map_err(|_| {
                EngineError::resource_exhaustion("decoded size is not representable")
            })?)?)
        };
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(decoded_bytes)
            .map_err(|_| EngineError::resource_exhaustion("decoded byte allocation failed"))?;
        for token in tokens {
            let index = self.token_index(*token)?;
            if !self.special[index] {
                bytes.extend_from_slice(&self.pieces[index]);
            }
        }
        Ok(bytes)
    }

    fn token_piece(&self, token: u32) -> Result<&[u8], EngineError> {
        Ok(&self.pieces[self.token_index(token)?])
    }

    fn vocabulary_lookup(&self, piece: &[u8]) -> Option<u32> {
        self.vocabulary.get(piece).copied()
    }

    fn is_special_token(&self, token: u32) -> Result<bool, EngineError> {
        Ok(self.special[self.token_index(token)?])
    }
}

/// Stateful strict UTF-8 decoder for token-at-a-time generation.
pub struct IncrementalDecoder<'a> {
    tokenizer: &'a dyn Tokenizer,
    pending: Vec<u8>,
    decoded_bytes: usize,
}

impl<'a> IncrementalDecoder<'a> {
    /// Starts an empty decoder over immutable tokenizer tables.
    pub fn new(tokenizer: &'a dyn Tokenizer) -> Self {
        Self { tokenizer, pending: Vec::with_capacity(4), decoded_bytes: 0 }
    }

    /// Pushes one token and returns the currently complete UTF-8 prefix.
    pub fn push(&mut self, token: u32) -> Result<String, EngineError> {
        if self.tokenizer.is_special_token(token)? {
            return Ok(String::new());
        }
        let piece = self.tokenizer.token_piece(token)?;
        self.decoded_bytes = self
            .decoded_bytes
            .checked_add(piece.len())
            .filter(|bytes| *bytes <= MAX_REQUEST_BYTES)
            .ok_or_else(|| EngineError::resource_exhaustion("stream decode exceeds byte limit"))?;
        self.pending
            .try_reserve(piece.len())
            .map_err(|_| EngineError::resource_exhaustion("stream decode allocation failed"))?;
        self.pending.extend_from_slice(piece);
        match std::str::from_utf8(&self.pending) {
            Ok(text) => {
                let complete = text.to_owned();
                self.pending.clear();
                Ok(complete)
            }
            Err(error) if error.error_len().is_none() => {
                let complete_length = error.valid_up_to();
                let complete = std::str::from_utf8(&self.pending[..complete_length])
                    .map_err(|_| EngineError::internal_invariant("UTF-8 prefix was not valid"))?
                    .to_owned();
                self.pending.copy_within(complete_length.., 0);
                self.pending.truncate(self.pending.len() - complete_length);
                Ok(complete)
            }
            Err(_) => {
                Err(EngineError::invalid_input("generated token bytes contain malformed UTF-8"))
            }
        }
    }

    /// Finishes the stream, rejecting an incomplete UTF-8 suffix.
    pub fn finish(self) -> Result<(), EngineError> {
        if self.pending.is_empty() {
            Ok(())
        } else {
            Err(EngineError::invalid_input("generated token stream ends with incomplete UTF-8"))
        }
    }

    /// Returns whether a partial UTF-8 sequence is buffered.
    pub fn has_pending_bytes(&self) -> bool {
        !self.pending.is_empty()
    }
}

/// One canonical vocabulary entry used by the offline converter.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AipTokVocabularyEntry {
    /// Contiguous token identifier.
    pub token_id: u32,
    /// Exact token bytes; special-token bytes must be valid UTF-8 text.
    pub bytes: Vec<u8>,
    /// Optional special role.
    pub special_role: Option<SpecialTokenRole>,
}

/// One explicit ranked BPE merge used by the offline converter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AipTokMerge {
    /// Explicit contiguous rank.
    pub rank: u32,
    /// Left token identifier.
    pub left: u32,
    /// Right token identifier.
    pub right: u32,
    /// Result token identifier.
    pub result: u32,
}

/// Canonical data-only definition serialized by the offline importer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AipTokDefinition {
    /// Contiguous vocabulary entries.
    pub vocabulary: Vec<AipTokVocabularyEntry>,
    /// Ranked declarative BPE merges.
    pub merges: Vec<AipTokMerge>,
    /// Required BOS token ID.
    pub beginning_of_sequence_token: u32,
    /// Required EOS token ID.
    pub end_of_sequence_token: u32,
    /// Required unknown marker ID.
    pub unknown_token: u32,
    /// Maximum accepted input bytes.
    pub maximum_input_bytes: u32,
    /// Maximum produced token identifiers.
    pub maximum_produced_tokens: u32,
}

impl AipTokDefinition {
    /// Serializes the exact canonical AIPTok v1 representation.
    pub fn encode(&self) -> Result<Vec<u8>, EngineError> {
        let vocabulary_size = u32::try_from(self.vocabulary.len())
            .map_err(|_| corrupt("AIPTok vocabulary count is not representable"))?;
        let merge_count = u32::try_from(self.merges.len())
            .map_err(|_| corrupt("AIPTok merge count is not representable"))?;
        let special_count = u32::try_from(
            self.vocabulary.iter().filter(|entry| entry.special_role.is_some()).count(),
        )
        .map_err(|_| corrupt("AIPTok special count is not representable"))?;
        let maximum_piece = self
            .vocabulary
            .iter()
            .map(|entry| entry.bytes.len())
            .max()
            .ok_or_else(|| corrupt("AIPTok vocabulary is empty"))?;
        let maximum_piece = u16::try_from(maximum_piece)
            .map_err(|_| corrupt("AIPTok token piece exceeds its field"))?;
        let vocabulary_bytes = self.vocabulary.iter().try_fold(0_usize, |total, entry| {
            checked_add(total, checked_add(VOCABULARY_ENTRY_HEADER_BYTES, entry.bytes.len())?)
        })?;
        let merge_bytes = self
            .merges
            .len()
            .checked_mul(MERGE_RULE_BYTES)
            .ok_or_else(|| corrupt("AIPTok merge bytes overflow"))?;
        let special_bytes = usize::try_from(special_count)
            .ok()
            .and_then(|count| count.checked_mul(SPECIAL_RECORD_BYTES))
            .ok_or_else(|| corrupt("AIPTok special bytes overflow"))?;
        let total_bytes = checked_add(
            checked_add(AIPTOK_HEADER_BYTES, vocabulary_bytes)?,
            checked_add(merge_bytes, special_bytes)?,
        )?;
        if u64::try_from(total_bytes).unwrap_or(u64::MAX) > MAX_TOKENIZER_DATA_BYTES {
            return Err(EngineError::resource_exhaustion("AIPTok data exceeds its hard limit"));
        }
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(total_bytes)
            .map_err(|_| EngineError::resource_exhaustion("AIPTok output allocation failed"))?;
        bytes.resize(AIPTOK_HEADER_BYTES, 0);
        for entry in &self.vocabulary {
            write_u32_push(&mut bytes, entry.token_id);
            let flags = if entry.special_role.is_some() { SPECIAL_VOCABULARY_FLAG } else { 0 };
            write_u16_push(&mut bytes, flags);
            write_u16_push(
                &mut bytes,
                u16::try_from(entry.bytes.len())
                    .map_err(|_| corrupt("AIPTok token piece exceeds its field"))?,
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
        let special_offset = usize_to_u32(bytes.len())?;
        for entry in self.vocabulary.iter().filter(|entry| entry.special_role.is_some()) {
            write_u32_push(&mut bytes, entry.token_id);
            write_u32_push(
                &mut bytes,
                entry
                    .special_role
                    .ok_or_else(|| corrupt("AIPTok special role disappeared"))?
                    .wire(),
            );
        }
        if bytes.len() != total_bytes {
            return Err(EngineError::internal_invariant("AIPTok output length changed"));
        }
        bytes[..8].copy_from_slice(&AIPTOK_MAGIC);
        write_u16_at(&mut bytes, 8, 1)?;
        write_u16_at(&mut bytes, 10, 0)?;
        write_u32_at(&mut bytes, 12, usize_to_u32(AIPTOK_HEADER_BYTES)?)?;
        write_u16_at(&mut bytes, 16, AIPTOK_FAMILY_BYTE_BPE)?;
        write_u16_at(&mut bytes, 18, NORMALIZATION_PRESERVE_BYTES)?;
        write_u16_at(&mut bytes, 20, DECODER_CONCATENATE_SKIP_SPECIAL)?;
        write_u16_at(&mut bytes, 22, UNKNOWN_BYTE_FALLBACK)?;
        write_u32_at(&mut bytes, 24, vocabulary_size)?;
        write_u32_at(&mut bytes, 28, merge_count)?;
        write_u32_at(&mut bytes, 32, special_count)?;
        write_u32_at(&mut bytes, 36, self.beginning_of_sequence_token)?;
        write_u32_at(&mut bytes, 40, self.end_of_sequence_token)?;
        write_u32_at(&mut bytes, 44, self.unknown_token)?;
        write_u16_at(&mut bytes, 48, maximum_piece)?;
        write_u16_at(&mut bytes, 50, 0)?;
        write_u32_at(&mut bytes, 52, self.maximum_input_bytes)?;
        write_u32_at(&mut bytes, 56, self.maximum_produced_tokens)?;
        write_u32_at(&mut bytes, 60, usize_to_u32(AIPTOK_HEADER_BYTES)?)?;
        write_u32_at(&mut bytes, 64, merge_offset)?;
        write_u32_at(&mut bytes, 68, special_offset)?;
        let total_length = usize_to_u32(bytes.len())?;
        write_u32_at(&mut bytes, 72, total_length)?;
        write_u32_at(&mut bytes, 76, 0)?;
        let content_hash = hash_package(&bytes[AIPTOK_HEADER_BYTES..]);
        bytes[80..128].copy_from_slice(content_hash.as_bytes());
        ParsedTokenizerData::parse(&bytes)?;
        Ok(bytes)
    }
}

/// Constructs the sole supported tokenizer implementation.
pub fn build_tokenizer(data: &[u8], pool: &MemoryPool) -> Result<Box<dyn Tokenizer>, EngineError> {
    Ok(Box::new(FirstPartyBpeTokenizer::parse(data, pool)?))
}

struct ParsedTokenizerData {
    spec: TokenizerSpec,
    pieces: Vec<Box<[u8]>>,
    special: Vec<bool>,
    vocabulary: BTreeMap<Box<[u8]>, u32>,
    special_literals: Vec<SpecialLiteral>,
    merges: BTreeMap<(u32, u32), MergeRule>,
}

struct ParsedHeader {
    spec: TokenizerSpec,
    merge_count: u32,
    special_count: u32,
    vocabulary_offset: usize,
    merge_offset: usize,
    special_offset: usize,
}

struct ParsedVocabulary {
    pieces: Vec<Box<[u8]>>,
    special: Vec<bool>,
    vocabulary: BTreeMap<Box<[u8]>, u32>,
    end_offset: usize,
}

impl ParsedTokenizerData {
    fn parse(data: &[u8]) -> Result<Self, EngineError> {
        let header = parse_header(data)?;
        let vocabulary = parse_vocabulary(data, &header)?;
        if vocabulary.end_offset != header.merge_offset {
            return Err(corrupt("AIPTok vocabulary length is not canonical"));
        }
        let merges = parse_merges(data, &header, &vocabulary)?;
        let special_literals = parse_specials(data, &header, &vocabulary)?;
        Ok(Self {
            spec: header.spec,
            pieces: vocabulary.pieces,
            special: vocabulary.special,
            vocabulary: vocabulary.vocabulary,
            special_literals,
            merges,
        })
    }
}

fn parse_header(data: &[u8]) -> Result<ParsedHeader, EngineError> {
    if data.len() < AIPTOK_HEADER_BYTES
        || u64::try_from(data.len()).unwrap_or(u64::MAX) > MAX_TOKENIZER_DATA_BYTES
        || data[..8] != AIPTOK_MAGIC
    {
        return Err(corrupt("AIPTok magic or length is invalid"));
    }
    if read_u16(data, 8)? != 1
        || read_u16(data, 10)? != 0
        || read_u32(data, 12)? != usize_to_u32(AIPTOK_HEADER_BYTES)?
    {
        return Err(corrupt("AIPTok version or header length is unsupported"));
    }
    if read_u16(data, 16)? != AIPTOK_FAMILY_BYTE_BPE {
        return Err(EngineError::unsupported_tokenizer("AIPTok tokenizer family"));
    }
    if read_u16(data, 18)? != NORMALIZATION_PRESERVE_BYTES
        || read_u16(data, 20)? != DECODER_CONCATENATE_SKIP_SPECIAL
        || read_u16(data, 22)? != UNKNOWN_BYTE_FALLBACK
    {
        return Err(EngineError::unsupported_tokenizer("AIPTok behavior profile"));
    }
    let vocabulary_size = read_u32(data, 24)?;
    let merge_count = read_u32(data, 28)?;
    let special_count = read_u32(data, 32)?;
    if !(259..=MAX_VOCABULARY_SIZE).contains(&vocabulary_size)
        || merge_count > MAX_BPE_MERGE_RULES
        || !(3..=MAX_SPECIAL_TOKENS).contains(&special_count)
    {
        return Err(corrupt("AIPTok table count is outside hard bounds"));
    }
    let expected_vocabulary = 256_u32
        .checked_add(merge_count)
        .and_then(|value| value.checked_add(special_count))
        .ok_or_else(|| corrupt("AIPTok vocabulary count overflows"))?;
    if expected_vocabulary != vocabulary_size {
        return Err(corrupt("AIPTok vocabulary composition is not canonical"));
    }
    let bos = read_u32(data, 36)?;
    let eos = read_u32(data, 40)?;
    let unknown = read_u32(data, 44)?;
    if [bos, eos, unknown].iter().any(|token| *token >= vocabulary_size)
        || bos == eos
        || bos == unknown
        || eos == unknown
    {
        return Err(corrupt("AIPTok required token IDs are invalid"));
    }
    let maximum_piece_bytes = read_u16(data, 48)?;
    if maximum_piece_bytes == 0
        || maximum_piece_bytes > MAX_TOKEN_PIECE_BYTES
        || read_u16(data, 50)? != 0
    {
        return Err(corrupt("AIPTok piece limit or reserved field is invalid"));
    }
    let maximum_input_bytes = read_u32(data, 52)?;
    let maximum_produced_tokens = read_u32(data, 56)?;
    if maximum_input_bytes == 0
        || usize::try_from(maximum_input_bytes).unwrap_or(usize::MAX) > MAX_REQUEST_BYTES
        || maximum_produced_tokens == 0
        || maximum_produced_tokens > MAX_TOKEN_COUNT
    {
        return Err(corrupt("AIPTok input or output limit is invalid"));
    }
    let vocabulary_offset = u32_to_usize(read_u32(data, 60)?)?;
    let merge_offset = u32_to_usize(read_u32(data, 64)?)?;
    let special_offset = u32_to_usize(read_u32(data, 68)?)?;
    if vocabulary_offset != AIPTOK_HEADER_BYTES
        || vocabulary_offset > merge_offset
        || merge_offset > special_offset
        || read_u32(data, 72)? != usize_to_u32(data.len())?
        || read_u32(data, 76)? != 0
    {
        return Err(corrupt("AIPTok offsets, total length, or reserved data are invalid"));
    }
    let expected_hash = parse_content_hash(data)?;
    Ok(ParsedHeader {
        spec: TokenizerSpec {
            format: TokenizerFormat::AipTokByteLevelBpeV1,
            normalization: NormalizationPolicy::PreserveBytes,
            unknown_policy: UnknownTokenPolicy::ByteFallback,
            decoder_behavior: DecoderBehavior::ConcatenateBytesSkipSpecial,
            vocabulary_size,
            beginning_of_sequence_token: bos,
            end_of_sequence_token: eos,
            unknown_token: unknown,
            maximum_piece_bytes,
            maximum_input_bytes,
            maximum_produced_tokens,
            content_hash: expected_hash,
        },
        merge_count,
        special_count,
        vocabulary_offset,
        merge_offset,
        special_offset,
    })
}

fn parse_content_hash(data: &[u8]) -> Result<ModelPackageHash, EngineError> {
    let mut expected_hash = [0_u8; ModelPackageHash::BYTE_LEN];
    expected_hash.copy_from_slice(&data[80..128]);
    let expected_hash = ModelPackageHash::from_sha384_bytes(expected_hash);
    if hash_package(&data[AIPTOK_HEADER_BYTES..]) != expected_hash {
        return Err(corrupt("AIPTok content hash does not match canonical content"));
    }
    Ok(expected_hash)
}

fn parse_vocabulary(data: &[u8], header: &ParsedHeader) -> Result<ParsedVocabulary, EngineError> {
    let count = u32_to_usize(header.spec.vocabulary_size)?;
    let mut pieces = Vec::new();
    let mut special = Vec::new();
    let mut vocabulary = BTreeMap::new();
    pieces
        .try_reserve_exact(count)
        .map_err(|_| EngineError::resource_exhaustion("AIPTok vocabulary allocation failed"))?;
    special
        .try_reserve_exact(count)
        .map_err(|_| EngineError::resource_exhaustion("AIPTok flag allocation failed"))?;
    let mut cursor = header.vocabulary_offset;
    let mut actual_maximum = 0_usize;
    let mut observed_specials = 0_u32;
    for expected_id in 0..header.spec.vocabulary_size {
        let token_id = read_u32(data, cursor)?;
        let flags = read_u16(data, checked_add(cursor, 4)?)?;
        let length = usize::from(read_u16(data, checked_add(cursor, 6)?)?);
        cursor = checked_add(cursor, VOCABULARY_ENTRY_HEADER_BYTES)?;
        if token_id != expected_id || flags & !SPECIAL_VOCABULARY_FLAG != 0 {
            return Err(corrupt("AIPTok token IDs or flags are not canonical"));
        }
        let is_special = flags == SPECIAL_VOCABULARY_FLAG;
        if length == 0 || length > usize::from(header.spec.maximum_piece_bytes) {
            return Err(corrupt("AIPTok vocabulary piece length is invalid"));
        }
        let end = checked_add(cursor, length)?;
        let piece =
            data.get(cursor..end).ok_or_else(|| corrupt("AIPTok vocabulary piece is truncated"))?;
        if expected_id < 256
            && (is_special
                || piece.len() != 1
                || piece[0]
                    != u8::try_from(expected_id)
                        .map_err(|_| corrupt("AIPTok byte fallback token overflows"))?)
        {
            return Err(corrupt("AIPTok byte fallback vocabulary is not canonical"));
        }
        if is_special {
            observed_specials = observed_specials
                .checked_add(1)
                .ok_or_else(|| corrupt("AIPTok special count overflows"))?;
            let text = std::str::from_utf8(piece)
                .map_err(|_| corrupt("AIPTok special-token text is invalid UTF-8"))?;
            if piece.len() < 5
                || !piece.starts_with(b"<|")
                || !piece.ends_with(b"|>")
                || !piece[2..piece.len() - 2]
                    .iter()
                    .all(|byte| byte.is_ascii_alphanumeric() || b"_.:-".contains(byte))
                || text.chars().any(char::is_control)
            {
                return Err(corrupt("AIPTok special-token text is not canonical"));
            }
        }
        if vocabulary.insert(piece.to_vec().into_boxed_slice(), token_id).is_some() {
            return Err(corrupt("AIPTok vocabulary contains duplicate token bytes"));
        }
        actual_maximum = actual_maximum.max(length);
        pieces.push(piece.to_vec().into_boxed_slice());
        special.push(is_special);
        cursor = end;
    }
    if observed_specials != header.special_count
        || actual_maximum != usize::from(header.spec.maximum_piece_bytes)
    {
        return Err(corrupt("AIPTok vocabulary metadata is not canonical"));
    }
    Ok(ParsedVocabulary { pieces, special, vocabulary, end_offset: cursor })
}

fn parse_merges(
    data: &[u8],
    header: &ParsedHeader,
    vocabulary: &ParsedVocabulary,
) -> Result<BTreeMap<(u32, u32), MergeRule>, EngineError> {
    let merge_bytes = u32_to_usize(header.merge_count)?
        .checked_mul(MERGE_RULE_BYTES)
        .ok_or_else(|| corrupt("AIPTok merge table length overflows"))?;
    if checked_add(header.merge_offset, merge_bytes)? != header.special_offset {
        return Err(corrupt("AIPTok merge table length is not canonical"));
    }
    let mut cursor = header.merge_offset;
    let mut merges = BTreeMap::new();
    let mut results = BTreeSet::new();
    for expected_rank in 0..header.merge_count {
        let rank = read_u32(data, cursor)?;
        let left = read_u32(data, checked_add(cursor, 4)?)?;
        let right = read_u32(data, checked_add(cursor, 8)?)?;
        let result = read_u32(data, checked_add(cursor, 12)?)?;
        cursor = checked_add(cursor, MERGE_RULE_BYTES)?;
        if rank != expected_rank
            || [left, right, result].iter().any(|token| *token >= header.spec.vocabulary_size)
            || result < 256
            || result == left
            || result == right
        {
            return Err(corrupt("AIPTok merge rank or token reference is invalid"));
        }
        let left_index = u32_to_usize(left)?;
        let right_index = u32_to_usize(right)?;
        let result_index = u32_to_usize(result)?;
        if vocabulary.special[left_index]
            || vocabulary.special[right_index]
            || vocabulary.special[result_index]
        {
            return Err(corrupt("AIPTok merges cannot reference special tokens"));
        }
        if !piece_is_concatenation(
            &vocabulary.pieces[result_index],
            &vocabulary.pieces[left_index],
            &vocabulary.pieces[right_index],
        ) {
            return Err(corrupt("AIPTok merge result does not match token bytes"));
        }
        if merges.insert((left, right), MergeRule { rank, result }).is_some()
            || !results.insert(result)
        {
            return Err(corrupt("AIPTok merge pair or result is duplicated"));
        }
    }
    for token in 256..header.spec.vocabulary_size {
        let index = u32_to_usize(token)?;
        if !vocabulary.special[index] && !results.contains(&token) {
            return Err(corrupt("AIPTok ordinary vocabulary token lacks one defining merge"));
        }
    }
    Ok(merges)
}

fn parse_specials(
    data: &[u8],
    header: &ParsedHeader,
    vocabulary: &ParsedVocabulary,
) -> Result<Vec<SpecialLiteral>, EngineError> {
    let special_bytes = u32_to_usize(header.special_count)?
        .checked_mul(SPECIAL_RECORD_BYTES)
        .ok_or_else(|| corrupt("AIPTok special table length overflows"))?;
    if checked_add(header.special_offset, special_bytes)? != data.len() {
        return Err(corrupt("AIPTok special table has trailing or truncated data"));
    }
    let mut cursor = header.special_offset;
    let mut previous = None;
    let mut roles = BTreeMap::new();
    let mut literals = Vec::new();
    for _ in 0..header.special_count {
        let token = read_u32(data, cursor)?;
        let role = SpecialTokenRole::from_wire(read_u32(data, checked_add(cursor, 4)?)?)?;
        cursor = checked_add(cursor, SPECIAL_RECORD_BYTES)?;
        if token >= header.spec.vocabulary_size
            || previous.is_some_and(|value| token <= value)
            || !vocabulary.special[u32_to_usize(token)?]
            || roles.insert(token, role).is_some()
        {
            return Err(corrupt("AIPTok special-token table is invalid"));
        }
        previous = Some(token);
        literals.push((vocabulary.pieces[u32_to_usize(token)?].clone(), token));
    }
    for (required, role) in [
        (header.spec.beginning_of_sequence_token, SpecialTokenRole::BeginningOfSequence),
        (header.spec.end_of_sequence_token, SpecialTokenRole::EndOfSequence),
        (header.spec.unknown_token, SpecialTokenRole::Unknown),
    ] {
        if roles.get(&required) != Some(&role) {
            return Err(corrupt("AIPTok required special-token role is missing"));
        }
    }
    if roles.iter().any(|(token, role)| {
        *role != SpecialTokenRole::Other
            && ![
                header.spec.beginning_of_sequence_token,
                header.spec.end_of_sequence_token,
                header.spec.unknown_token,
            ]
            .contains(token)
    }) {
        return Err(corrupt("AIPTok required special-token role is duplicated"));
    }
    literals.sort_by(|(left, left_id), (right, right_id)| {
        right.len().cmp(&left.len()).then_with(|| left_id.cmp(right_id))
    });
    for (index, (literal, _)) in literals.iter().enumerate() {
        if literals
            .iter()
            .enumerate()
            .any(|(other_index, (other, _))| index != other_index && other.starts_with(literal))
        {
            return Err(corrupt("AIPTok special-token text has a prefix collision"));
        }
    }
    Ok(literals)
}

fn piece_is_concatenation(result: &[u8], left: &[u8], right: &[u8]) -> bool {
    result.len() == left.len() + right.len()
        && result.get(..left.len()) == Some(left)
        && result.get(left.len()..) == Some(right)
}

fn persistent_reservation_bytes(
    data_length: usize,
    vocabulary_size: u32,
    merge_count: u32,
) -> Result<u64, EngineError> {
    u64::try_from(data_length)
        .ok()
        .and_then(|data| {
            u64::from(vocabulary_size)
                .checked_mul(PERSISTENT_BYTES_PER_VOCABULARY_ENTRY)
                .and_then(|vocabulary| data.checked_add(vocabulary))
        })
        .and_then(|value| {
            u64::from(merge_count)
                .checked_mul(PERSISTENT_BYTES_PER_MERGE_RULE)
                .and_then(|merges| value.checked_add(merges))
        })
        .ok_or_else(|| EngineError::resource_exhaustion("tokenizer memory bound overflows"))
}

fn read_u16(input: &[u8], offset: usize) -> Result<u16, EngineError> {
    Ok(u16::from_le_bytes(read_exact(input, offset)?))
}

fn read_u32(input: &[u8], offset: usize) -> Result<u32, EngineError> {
    Ok(u32::from_le_bytes(read_exact(input, offset)?))
}

fn read_exact<const N: usize>(input: &[u8], offset: usize) -> Result<[u8; N], EngineError> {
    let end = checked_add(offset, N)?;
    input
        .get(offset..end)
        .ok_or_else(|| corrupt("AIPTok data is truncated"))?
        .try_into()
        .map_err(|_| corrupt("AIPTok field length is invalid"))
}

fn checked_add(left: usize, right: usize) -> Result<usize, EngineError> {
    left.checked_add(right).ok_or_else(|| corrupt("AIPTok offset overflows"))
}

fn usize_to_u32(value: usize) -> Result<u32, EngineError> {
    u32::try_from(value).map_err(|_| corrupt("AIPTok length exceeds its field"))
}

fn u32_to_usize(value: u32) -> Result<usize, EngineError> {
    usize::try_from(value).map_err(|_| corrupt("AIPTok value is not addressable"))
}

fn write_u16_push(output: &mut Vec<u8>, value: u16) {
    output.extend_from_slice(&value.to_le_bytes());
}

fn write_u32_push(output: &mut Vec<u8>, value: u32) {
    output.extend_from_slice(&value.to_le_bytes());
}

fn write_u16_at(output: &mut [u8], offset: usize, value: u16) -> Result<(), EngineError> {
    write_at(output, offset, &value.to_le_bytes())
}

fn write_u32_at(output: &mut [u8], offset: usize, value: u32) -> Result<(), EngineError> {
    write_at(output, offset, &value.to_le_bytes())
}

fn write_at(output: &mut [u8], offset: usize, value: &[u8]) -> Result<(), EngineError> {
    let end = checked_add(offset, value.len())?;
    output
        .get_mut(offset..end)
        .ok_or_else(|| EngineError::internal_invariant("AIPTok encoder offset is invalid"))?
        .copy_from_slice(value);
    Ok(())
}

fn corrupt(detail: impl AsRef<str>) -> EngineError {
    EngineError::corrupt_model_package(detail)
}

const _: () = assert!(size_of::<Node>() <= 256);

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    fn definition(maximum_tokens: u32) -> AipTokDefinition {
        let mut vocabulary = (0_u32..=255)
            .map(|token_id| AipTokVocabularyEntry {
                token_id,
                bytes: vec![u8::try_from(token_id).expect("byte token must fit")],
                special_role: None,
            })
            .collect::<Vec<_>>();
        vocabulary.extend([
            AipTokVocabularyEntry { token_id: 256, bytes: b"ab".to_vec(), special_role: None },
            AipTokVocabularyEntry {
                token_id: 257,
                bytes: b"<|bos|>".to_vec(),
                special_role: Some(SpecialTokenRole::BeginningOfSequence),
            },
            AipTokVocabularyEntry {
                token_id: 258,
                bytes: b"<|eos|>".to_vec(),
                special_role: Some(SpecialTokenRole::EndOfSequence),
            },
            AipTokVocabularyEntry {
                token_id: 259,
                bytes: b"<|unk|>".to_vec(),
                special_role: Some(SpecialTokenRole::Unknown),
            },
        ]);
        AipTokDefinition {
            vocabulary,
            merges: vec![AipTokMerge { rank: 0, left: 97, right: 98, result: 256 }],
            beginning_of_sequence_token: 257,
            end_of_sequence_token: 258,
            unknown_token: 259,
            maximum_input_bytes: u32::try_from(MAX_REQUEST_BYTES).expect("request limit fits"),
            maximum_produced_tokens: maximum_tokens,
        }
    }

    fn tokenizer() -> (Vec<u8>, FirstPartyBpeTokenizer) {
        let bytes = definition(MAX_TOKEN_COUNT).encode().expect("definition must encode");
        let pool = MemoryPool::new(128 * 1024 * 1024).expect("pool must validate");
        let tokenizer = FirstPartyBpeTokenizer::parse(&bytes, &pool).expect("data must parse");
        (bytes, tokenizer)
    }

    fn two_merge_definition() -> AipTokDefinition {
        let mut value = definition(MAX_TOKEN_COUNT);
        for entry in value.vocabulary.iter_mut().skip(257) {
            entry.token_id += 1;
        }
        value.vocabulary.insert(
            257,
            AipTokVocabularyEntry { token_id: 257, bytes: b"abc".to_vec(), special_role: None },
        );
        value.merges.push(AipTokMerge { rank: 1, left: 256, right: 99, result: 257 });
        value.beginning_of_sequence_token = 258;
        value.end_of_sequence_token = 259;
        value.unknown_token = 260;
        value
    }

    fn rehash(bytes: &mut [u8]) {
        let digest = hash_package(&bytes[AIPTOK_HEADER_BYTES..]);
        bytes[80..128].copy_from_slice(digest.as_bytes());
    }

    #[test]
    fn empty_ascii_unicode_emoji_and_combining_text_are_exact() {
        let (_, tokenizer) = tokenizer();
        for text in ["", "plain ASCII", "世界", "🦀🚀", "e\u{301}"] {
            let tokens = tokenizer
                .encode_with_options(text, EncodeOptions::byte_round_trip())
                .expect("text must encode");
            assert_eq!(tokenizer.decode(&tokens).expect("tokens must decode"), text);
        }
        assert_eq!(tokenizer.encode("ab").expect("merge input must encode"), vec![257, 256]);
        assert_eq!(tokenizer.vocabulary_lookup(b"ab"), Some(256));
        assert_eq!(tokenizer.spec().unknown_policy(), UnknownTokenPolicy::ByteFallback);
    }

    #[test]
    fn every_byte_nulls_and_pseudorandom_inputs_round_trip() {
        let (_, tokenizer) = tokenizer();
        let all = (0_u16..=255)
            .map(|value| u8::try_from(value).expect("byte must fit"))
            .collect::<Vec<_>>();
        let tokens = tokenizer
            .encode_bytes_with_options(&all, EncodeOptions::byte_round_trip())
            .expect("all bytes must encode");
        assert_eq!(tokenizer.decode_bytes(&tokens).expect("bytes must decode"), all);
        assert_eq!(tokenizer.token_piece(0).expect("null token must exist"), b"\0");

        let mut state = 0x9e37_79b9_u32;
        for length in 0..512_usize {
            let mut input = Vec::with_capacity(length);
            for _ in 0..length {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                input.push(state.to_le_bytes()[3]);
            }
            let encoded = tokenizer
                .encode_bytes_with_options(&input, EncodeOptions::byte_round_trip())
                .expect("arbitrary bytes must encode");
            assert_eq!(tokenizer.decode_bytes(&encoded).expect("bytes must decode"), input);
        }
    }

    #[test]
    fn special_text_requires_explicit_policy_and_collisions_are_rejected() {
        let (_, tokenizer) = tokenizer();
        assert!(tokenizer.encode("x<|eos|>y").is_err());
        let ordinary = tokenizer
            .encode_with_options(
                "x<|eos|>y",
                EncodeOptions::new(false, false, SpecialTokenPolicy::TreatAsOrdinaryBytes),
            )
            .expect("ordinary bytes must encode");
        assert_eq!(tokenizer.decode(&ordinary).expect("ordinary bytes must decode"), "x<|eos|>y");
        let allowed = tokenizer
            .encode_with_options(
                "x<|eos|>y",
                EncodeOptions::new(false, false, SpecialTokenPolicy::AllowListed(&[258])),
            )
            .expect("listed special must encode");
        assert!(allowed.contains(&258));
        assert_eq!(tokenizer.decode(&allowed).expect("special is skipped"), "xy");
        assert!(
            tokenizer
                .encode_with_options(
                    "<|eos|>",
                    EncodeOptions::new(false, false, SpecialTokenPolicy::AllowListed(&[257])),
                )
                .is_err()
        );

        let mut collision = definition(MAX_TOKEN_COUNT);
        collision.vocabulary.push(AipTokVocabularyEntry {
            token_id: 260,
            bytes: b"<|eos|>suffix".to_vec(),
            special_role: Some(SpecialTokenRole::Other),
        });
        assert!(collision.encode().is_err());
    }

    #[test]
    fn invalid_utf8_is_explicit_and_streaming_matches_full_decode() {
        let (_, tokenizer) = tokenizer();
        let invalid = tokenizer
            .encode_bytes_with_options(&[0xff, 0x00], EncodeOptions::byte_round_trip())
            .expect("bytes must encode");
        assert!(tokenizer.decode(&invalid).is_err());
        assert_eq!(tokenizer.decode_bytes(&invalid).expect("raw bytes must decode"), [0xff, 0]);

        let text = "A世界🦀e\u{301}";
        let tokens = tokenizer
            .encode_with_options(text, EncodeOptions::byte_round_trip())
            .expect("text must encode");
        let mut stream = IncrementalDecoder::new(&tokenizer);
        let mut streamed = String::new();
        for token in tokens.iter().copied() {
            streamed.push_str(&stream.push(token).expect("stream token must decode"));
        }
        stream.finish().expect("stream must terminate cleanly");
        assert_eq!(streamed, tokenizer.decode(&tokens).expect("full decode must work"));

        let mut incomplete = IncrementalDecoder::new(&tokenizer);
        assert_eq!(incomplete.push(0xe2).expect("prefix must buffer"), "");
        assert!(incomplete.has_pending_bytes());
        assert!(incomplete.finish().is_err());
    }

    #[test]
    fn maximum_input_token_and_invalid_id_limits_fail_closed() {
        let bytes = definition(8).encode().expect("definition must encode");
        let insufficient_pool = MemoryPool::new(1).expect("tiny pool must validate");
        assert!(FirstPartyBpeTokenizer::parse(&bytes, &insufficient_pool).is_err());
        let pool = MemoryPool::new(128 * 1024 * 1024).expect("pool must validate");
        let tokenizer = FirstPartyBpeTokenizer::parse(&bytes, &pool).expect("data must parse");
        assert!(
            tokenizer
                .encode_bytes_with_options(b"123456789", EncodeOptions::byte_round_trip())
                .is_err()
        );
        assert!(tokenizer.decode_bytes(&[260]).is_err());
        assert!(tokenizer.token_piece(260).is_err());
    }

    #[test]
    fn deterministic_across_runs_and_thread_counts() {
        let (bytes, tokenizer) = tokenizer();
        assert_eq!(bytes, definition(MAX_TOKEN_COUNT).encode().expect("repeat must encode"));
        let tokenizer = Arc::new(tokenizer);
        let expected = tokenizer.encode("abab 世界").expect("baseline must encode");
        let threads = (0..8)
            .map(|_| {
                let tokenizer = Arc::clone(&tokenizer);
                std::thread::spawn(move || tokenizer.encode("abab 世界"))
            })
            .collect::<Vec<_>>();
        for thread in threads {
            assert_eq!(
                thread.join().expect("thread must join").expect("encode must work"),
                expected
            );
        }
    }

    #[test]
    fn malformed_headers_vocabulary_merges_and_specials_are_rejected() {
        let (valid, _) = tokenizer();
        for end in 0..valid.len() {
            let pool = MemoryPool::new(128 * 1024 * 1024).expect("pool must validate");
            assert!(FirstPartyBpeTokenizer::parse(&valid[..end], &pool).is_err());
        }
        let pool = MemoryPool::new(128 * 1024 * 1024).expect("pool must validate");
        let mut cases = Vec::new();
        let mut bad_magic = valid.clone();
        bad_magic[0] ^= 1;
        cases.push(bad_magic);
        let mut unsupported_version = valid.clone();
        unsupported_version[8..10].copy_from_slice(&2_u16.to_le_bytes());
        cases.push(unsupported_version);
        let mut trailing_data = valid.clone();
        trailing_data.push(0);
        cases.push(trailing_data);
        let mut invalid_content_hash = valid.clone();
        invalid_content_hash[80] ^= 1;
        cases.push(invalid_content_hash);
        let mut oversized_vocabulary = valid.clone();
        oversized_vocabulary[24..28].copy_from_slice(&(MAX_VOCABULARY_SIZE + 1).to_le_bytes());
        cases.push(oversized_vocabulary);
        let mut duplicate_id = valid.clone();
        let second_entry = AIPTOK_HEADER_BYTES + VOCABULARY_ENTRY_HEADER_BYTES + 1;
        duplicate_id[second_entry..second_entry + 4].copy_from_slice(&0_u32.to_le_bytes());
        rehash(&mut duplicate_id);
        cases.push(duplicate_id);
        let mut duplicate_piece = valid.clone();
        let second_piece = second_entry + VOCABULARY_ENTRY_HEADER_BYTES;
        duplicate_piece[second_piece] = 0;
        rehash(&mut duplicate_piece);
        cases.push(duplicate_piece);
        let mut oversized_piece = valid.clone();
        oversized_piece[second_entry + 6..second_entry + 8]
            .copy_from_slice(&(MAX_TOKEN_PIECE_BYTES + 1).to_le_bytes());
        rehash(&mut oversized_piece);
        cases.push(oversized_piece);
        let merge_offset = usize::try_from(u32::from_le_bytes(valid[64..68].try_into().unwrap()))
            .expect("offset must fit");
        let mut invalid_rank = valid.clone();
        invalid_rank[merge_offset..merge_offset + 4].copy_from_slice(&1_u32.to_le_bytes());
        rehash(&mut invalid_rank);
        cases.push(invalid_rank);
        let mut duplicate_merge = two_merge_definition().encode().expect("fixture must encode");
        let duplicate_merge_offset =
            usize::try_from(u32::from_le_bytes(duplicate_merge[64..68].try_into().unwrap()))
                .expect("offset must fit");
        let second_merge = duplicate_merge_offset + MERGE_RULE_BYTES;
        duplicate_merge[second_merge + 4..second_merge + 8].copy_from_slice(&97_u32.to_le_bytes());
        duplicate_merge[second_merge + 8..second_merge + 12].copy_from_slice(&98_u32.to_le_bytes());
        duplicate_merge[second_merge + 12..second_merge + 16]
            .copy_from_slice(&256_u32.to_le_bytes());
        rehash(&mut duplicate_merge);
        cases.push(duplicate_merge);
        let special_offset = usize::try_from(u32::from_le_bytes(valid[68..72].try_into().unwrap()))
            .expect("offset must fit");
        let mut missing_required_special = valid.clone();
        missing_required_special[special_offset + 4..special_offset + 8]
            .copy_from_slice(&SpecialTokenRole::Other.wire().to_le_bytes());
        rehash(&mut missing_required_special);
        cases.push(missing_required_special);
        let mut invalid_utf8 = valid.clone();
        let special_piece = valid
            .windows(b"<|bos|>".len())
            .position(|window| window == b"<|bos|>")
            .expect("special text must exist");
        invalid_utf8[special_piece] = 0xff;
        rehash(&mut invalid_utf8);
        cases.push(invalid_utf8);
        for case in cases {
            assert!(FirstPartyBpeTokenizer::parse(&case, &pool).is_err());
        }
    }

    #[test]
    fn long_repeated_input_is_bounded_and_deterministic() {
        let (_, tokenizer) = tokenizer();
        let input = "ab".repeat(100_000);
        let first = tokenizer
            .encode_with_options(&input, EncodeOptions::byte_round_trip())
            .expect("bounded document must encode");
        let second = tokenizer
            .encode_with_options(&input, EncodeOptions::byte_round_trip())
            .expect("bounded document must encode again");
        assert_eq!(first, second);
        assert_eq!(first.len(), 100_000);
    }
}
