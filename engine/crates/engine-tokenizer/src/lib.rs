//! First-party bounded byte-level BPE tokenizer.
//!
//! The tokenizer consumes one fixed, non-recursive data format. Every byte has
//! an identity fallback token, merge rules are declarative data, and no model
//! field selects code, plugins, kernels, or filesystem locations.

use personalai_engine_memory::{MemoryPool, MemoryReservation};
use personalai_engine_types::{
    EngineError, MAX_REQUEST_BYTES, MAX_TOKEN_COUNT, MAX_TOKENIZER_DATA_BYTES,
};
use std::cmp::Reverse;
use std::collections::{BTreeMap, BinaryHeap};
use std::mem::size_of;

/// Maximum vocabulary size accepted by the engine.
pub use personalai_engine_types::MAX_VOCABULARY_SIZE;
/// Maximum UTF-8 byte sequence represented by one tokenizer token.
pub const MAX_TOKEN_PIECE_BYTES: u16 = 256;
/// Maximum merge rules in one tokenizer package.
pub const MAX_BPE_MERGE_RULES: u32 = MAX_VOCABULARY_SIZE;
/// Fixed first-party tokenizer header length.
pub const BPE_HEADER_BYTES: usize = 40;
/// First-party tokenizer magic.
pub const BPE_MAGIC: [u8; 8] = *b"PAIBPE01";

const SPECIAL_TOKEN_FLAG: u16 = 1;
const NO_TOKEN: u32 = u32::MAX;
const MERGE_RULE_BYTES: usize = 12;
const TEMPORARY_BYTES_PER_INPUT_TOKEN: u64 = 256;
const PERSISTENT_BYTES_PER_VOCABULARY_ENTRY: u64 = 64;
const PERSISTENT_BYTES_PER_MERGE_RULE: u64 = 128;

/// Engine-owned tokenizer data format identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenizerFormat {
    /// First-party bounded byte-pair-encoding data format.
    FirstPartyBpeV1,
}

/// Validated, data-only tokenizer metadata from a signed package.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenizerSpec {
    format: TokenizerFormat,
    vocabulary_size: u32,
    beginning_of_sequence_token: Option<u32>,
    end_of_sequence_token: u32,
    maximum_piece_bytes: u16,
}

impl TokenizerSpec {
    /// Validates tokenizer metadata without constructing an implementation.
    pub fn new(
        format: TokenizerFormat,
        vocabulary_size: u32,
        beginning_of_sequence_token: Option<u32>,
        end_of_sequence_token: u32,
        maximum_piece_bytes: u16,
    ) -> Result<Self, EngineError> {
        if !(256..=MAX_VOCABULARY_SIZE).contains(&vocabulary_size) {
            return Err(corrupt("tokenizer vocabulary size is outside hard bounds"));
        }
        if end_of_sequence_token >= vocabulary_size
            || beginning_of_sequence_token.is_some_and(|token| token >= vocabulary_size)
            || beginning_of_sequence_token == Some(end_of_sequence_token)
        {
            return Err(corrupt("tokenizer special token is outside the vocabulary"));
        }
        if maximum_piece_bytes == 0 || maximum_piece_bytes > MAX_TOKEN_PIECE_BYTES {
            return Err(corrupt("tokenizer piece size is outside hard bounds"));
        }
        Ok(Self {
            format,
            vocabulary_size,
            beginning_of_sequence_token,
            end_of_sequence_token,
            maximum_piece_bytes,
        })
    }

    /// Returns the engine-owned tokenizer format.
    pub const fn format(&self) -> TokenizerFormat {
        self.format
    }

    /// Returns the declared vocabulary size.
    pub const fn vocabulary_size(&self) -> u32 {
        self.vocabulary_size
    }

    /// Returns the optional beginning-of-sequence token.
    pub const fn beginning_of_sequence_token(&self) -> Option<u32> {
        self.beginning_of_sequence_token
    }

    /// Returns the end-of-sequence token.
    pub const fn end_of_sequence_token(&self) -> u32 {
        self.end_of_sequence_token
    }

    /// Returns the maximum bytes represented by one token piece.
    pub const fn maximum_piece_bytes(&self) -> u16 {
        self.maximum_piece_bytes
    }

    /// Validates a bounded sequence of token identifiers against this spec.
    pub fn validate_tokens(&self, tokens: &[u32]) -> Result<(), EngineError> {
        if tokens.len() > usize::try_from(MAX_TOKEN_COUNT).unwrap_or(usize::MAX) {
            return Err(EngineError::invalid_input("token sequence exceeds the hard limit"));
        }
        if tokens.iter().any(|token| *token >= self.vocabulary_size) {
            return Err(EngineError::invalid_input(
                "token sequence contains an out-of-vocabulary identifier",
            ));
        }
        Ok(())
    }
}

/// Contract implemented only by reviewed, engine-owned tokenizer code.
pub trait Tokenizer: Send + Sync {
    /// Returns immutable tokenizer metadata.
    fn spec(&self) -> &TokenizerSpec;

    /// Encodes bounded UTF-8 text into token identifiers.
    fn encode(&self, input: &str) -> Result<Vec<u32>, EngineError>;

    /// Decodes bounded token identifiers into UTF-8 text.
    fn decode(&self, tokens: &[u32]) -> Result<String, EngineError>;

    /// Returns the immutable byte piece for one validated token.
    fn token_piece(&self, token: u32) -> Result<&[u8], EngineError>;

    /// Returns whether a validated token is non-textual control data.
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

/// Parsed first-party byte-level BPE tokenizer.
#[derive(Debug)]
pub struct FirstPartyBpeTokenizer {
    spec: TokenizerSpec,
    pieces: Vec<Box<[u8]>>,
    special: Vec<bool>,
    merges: BTreeMap<(u32, u32), MergeRule>,
    pool: MemoryPool,
    _reservation: MemoryReservation,
}

impl FirstPartyBpeTokenizer {
    /// Parses canonical signed-package tokenizer bytes under an aggregate pool.
    pub fn parse(data: &[u8], pool: &MemoryPool) -> Result<Self, EngineError> {
        let parsed = ParsedTokenizerData::parse(data)?;
        let reservation_bytes = persistent_reservation_bytes(
            data.len(),
            parsed.spec.vocabulary_size(),
            u32::try_from(parsed.merges.len()).map_err(|_| {
                EngineError::resource_exhaustion("merge count is not representable")
            })?,
        )?;
        let reservation = pool.reserve(reservation_bytes)?;
        Ok(Self {
            spec: parsed.spec,
            pieces: parsed.pieces,
            special: parsed.special,
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

    fn apply_merges(&self, byte_tokens: &[u8]) -> Result<Vec<u32>, EngineError> {
        if byte_tokens.is_empty() {
            return Ok(Vec::new());
        }
        let count = byte_tokens.len();
        let temporary_bytes = u64::try_from(count)
            .map_err(|_| EngineError::resource_exhaustion("token count is not representable"))?
            .checked_mul(TEMPORARY_BYTES_PER_INPUT_TOKEN)
            .ok_or_else(|| EngineError::resource_exhaustion("tokenizer work memory overflows"))?;
        let _temporary = self.pool.reserve(temporary_bytes)?;

        let mut nodes = Vec::new();
        nodes
            .try_reserve_exact(count)
            .map_err(|_| EngineError::resource_exhaustion("tokenizer node allocation failed"))?;
        for (index, byte) in byte_tokens.iter().enumerate() {
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
}

impl Tokenizer for FirstPartyBpeTokenizer {
    fn spec(&self) -> &TokenizerSpec {
        &self.spec
    }

    fn encode(&self, input: &str) -> Result<Vec<u32>, EngineError> {
        if input.len() > MAX_REQUEST_BYTES {
            return Err(EngineError::invalid_input("tokenizer input exceeds the request bound"));
        }
        let mut encoded = self.apply_merges(input.as_bytes())?;
        if let Some(beginning) = self.spec.beginning_of_sequence_token() {
            if encoded.len() >= usize::try_from(MAX_TOKEN_COUNT).unwrap_or(usize::MAX) {
                return Err(EngineError::invalid_input(
                    "encoded token count exceeds the hard bound",
                ));
            }
            encoded.insert(0, beginning);
        }
        self.spec.validate_tokens(&encoded)?;
        Ok(encoded)
    }

    fn decode(&self, tokens: &[u32]) -> Result<String, EngineError> {
        self.spec.validate_tokens(tokens)?;
        let mut decoded_bytes = 0_usize;
        for token in tokens {
            let index = usize::try_from(*token)
                .map_err(|_| EngineError::invalid_input("token identifier is not addressable"))?;
            if !self.special[index] {
                decoded_bytes = decoded_bytes
                    .checked_add(self.pieces[index].len())
                    .filter(|bytes| *bytes <= MAX_REQUEST_BYTES)
                    .ok_or_else(|| {
                        EngineError::resource_exhaustion("decoded text exceeds the request bound")
                    })?;
            }
        }
        let temporary_bytes = u64::try_from(decoded_bytes)
            .map_err(|_| EngineError::resource_exhaustion("decoded size is not representable"))?;
        let _temporary =
            if temporary_bytes == 0 { None } else { Some(self.pool.reserve(temporary_bytes)?) };
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(decoded_bytes)
            .map_err(|_| EngineError::resource_exhaustion("decoded text allocation failed"))?;
        for token in tokens {
            let index = usize::try_from(*token)
                .map_err(|_| EngineError::invalid_input("token identifier is not addressable"))?;
            if self.special[index] {
                continue;
            }
            bytes.extend_from_slice(&self.pieces[index]);
        }
        String::from_utf8(bytes)
            .map_err(|_| EngineError::invalid_input("token sequence does not decode to UTF-8"))
    }

    fn token_piece(&self, token: u32) -> Result<&[u8], EngineError> {
        let index = self.token_index(token)?;
        Ok(&self.pieces[index])
    }

    fn is_special_token(&self, token: u32) -> Result<bool, EngineError> {
        let index = self.token_index(token)?;
        Ok(self.special[index])
    }
}

impl FirstPartyBpeTokenizer {
    fn token_index(&self, token: u32) -> Result<usize, EngineError> {
        if token >= self.spec.vocabulary_size() {
            return Err(EngineError::invalid_input(
                "token identifier is outside the tokenizer vocabulary",
            ));
        }
        usize::try_from(token)
            .map_err(|_| EngineError::invalid_input("token identifier is not addressable"))
    }
}

/// Constructs the only supported tokenizer implementation.
pub fn build_tokenizer(data: &[u8], pool: &MemoryPool) -> Result<Box<dyn Tokenizer>, EngineError> {
    Ok(Box::new(FirstPartyBpeTokenizer::parse(data, pool)?))
}

struct ParsedTokenizerData {
    spec: TokenizerSpec,
    pieces: Vec<Box<[u8]>>,
    special: Vec<bool>,
    merges: BTreeMap<(u32, u32), MergeRule>,
}

struct ParsedTokenizerHeader {
    spec: TokenizerSpec,
    merge_count: u32,
}

struct ParsedVocabulary {
    pieces: Vec<Box<[u8]>>,
    special: Vec<bool>,
    end_offset: usize,
}

impl ParsedTokenizerData {
    fn parse(data: &[u8]) -> Result<Self, EngineError> {
        let header = parse_header(data)?;
        let vocabulary = parse_vocabulary(data, &header.spec)?;
        let merges = parse_merges(
            data,
            vocabulary.end_offset,
            header.merge_count,
            &header.spec,
            &vocabulary.pieces,
            &vocabulary.special,
        )?;
        Ok(Self {
            spec: header.spec,
            pieces: vocabulary.pieces,
            special: vocabulary.special,
            merges,
        })
    }
}

fn parse_header(data: &[u8]) -> Result<ParsedTokenizerHeader, EngineError> {
    if data.len() < BPE_HEADER_BYTES
        || u64::try_from(data.len()).unwrap_or(u64::MAX) > MAX_TOKENIZER_DATA_BYTES
        || data[..8] != BPE_MAGIC
    {
        return Err(corrupt("tokenizer data magic or length is invalid"));
    }
    if read_u16(data, 8)? != 1
        || read_u16(data, 10)? != 0
        || read_u32(data, 12)? != usize_to_u32(BPE_HEADER_BYTES)?
    {
        return Err(corrupt("tokenizer data version or header length is unsupported"));
    }
    let vocabulary_size = read_u32(data, 16)?;
    let merge_count = read_u32(data, 20)?;
    if merge_count > MAX_BPE_MERGE_RULES {
        return Err(corrupt("tokenizer merge count exceeds the hard bound"));
    }
    let beginning = match read_u32(data, 24)? {
        NO_TOKEN => None,
        token => Some(token),
    };
    if read_u16(data, 34)? != 0 || read_u32(data, 36)? != usize_to_u32(data.len())? {
        return Err(corrupt("tokenizer reserved flags or total length are invalid"));
    }
    let spec = TokenizerSpec::new(
        TokenizerFormat::FirstPartyBpeV1,
        vocabulary_size,
        beginning,
        read_u32(data, 28)?,
        read_u16(data, 32)?,
    )?;
    Ok(ParsedTokenizerHeader { spec, merge_count })
}

fn parse_vocabulary(data: &[u8], spec: &TokenizerSpec) -> Result<ParsedVocabulary, EngineError> {
    let vocabulary_len = usize::try_from(spec.vocabulary_size())
        .map_err(|_| corrupt("tokenizer vocabulary is not addressable"))?;
    let mut pieces = Vec::new();
    let mut special = Vec::new();
    pieces
        .try_reserve_exact(vocabulary_len)
        .map_err(|_| EngineError::resource_exhaustion("tokenizer vocabulary allocation failed"))?;
    special
        .try_reserve_exact(vocabulary_len)
        .map_err(|_| EngineError::resource_exhaustion("tokenizer flags allocation failed"))?;
    let mut cursor = BPE_HEADER_BYTES;
    let mut actual_maximum = 0_usize;
    for token in 0..vocabulary_len {
        let length = usize::from(read_u16(data, cursor)?);
        let flags = read_u16(data, checked_add_usize(cursor, 2)?)?;
        cursor = checked_add_usize(cursor, 4)?;
        if flags & !SPECIAL_TOKEN_FLAG != 0 {
            return Err(corrupt("tokenizer token flags are invalid"));
        }
        let is_special = flags == SPECIAL_TOKEN_FLAG;
        if (is_special && length != 0)
            || (!is_special && (length == 0 || length > usize::from(spec.maximum_piece_bytes())))
        {
            return Err(corrupt("tokenizer token piece length is invalid"));
        }
        let end = checked_add_usize(cursor, length)?;
        let piece =
            data.get(cursor..end).ok_or_else(|| corrupt("tokenizer token piece is truncated"))?;
        if token < 256
            && (is_special
                || piece.len() != 1
                || piece[0]
                    != u8::try_from(token)
                        .map_err(|_| corrupt("byte fallback token is invalid"))?)
        {
            return Err(corrupt("tokenizer byte fallback vocabulary is not canonical"));
        }
        actual_maximum = actual_maximum.max(length);
        pieces.push(piece.to_vec().into_boxed_slice());
        special.push(is_special);
        cursor = end;
    }
    if actual_maximum != usize::from(spec.maximum_piece_bytes()) {
        return Err(corrupt("tokenizer maximum piece length is not canonical"));
    }
    Ok(ParsedVocabulary { pieces, special, end_offset: cursor })
}

fn parse_merges(
    data: &[u8],
    mut cursor: usize,
    merge_count: u32,
    spec: &TokenizerSpec,
    pieces: &[Box<[u8]>],
    special: &[bool],
) -> Result<BTreeMap<(u32, u32), MergeRule>, EngineError> {
    let merge_bytes = usize::try_from(merge_count)
        .map_err(|_| corrupt("tokenizer merge count is not addressable"))?
        .checked_mul(MERGE_RULE_BYTES)
        .ok_or_else(|| corrupt("tokenizer merge table length overflows"))?;
    if checked_add_usize(cursor, merge_bytes)? != data.len() {
        return Err(corrupt("tokenizer merge table does not terminate the data"));
    }
    let mut merges = BTreeMap::new();
    for rank in 0..merge_count {
        let left = read_u32(data, cursor)?;
        let right = read_u32(data, checked_add_usize(cursor, 4)?)?;
        let result = read_u32(data, checked_add_usize(cursor, 8)?)?;
        cursor = checked_add_usize(cursor, MERGE_RULE_BYTES)?;
        if [left, right, result].iter().any(|token| *token >= spec.vocabulary_size())
            || result == left
            || result == right
        {
            return Err(corrupt("tokenizer merge references are invalid"));
        }
        let left_index = usize::try_from(left).map_err(|_| corrupt("merge token overflows"))?;
        let right_index = usize::try_from(right).map_err(|_| corrupt("merge token overflows"))?;
        let result_index = usize::try_from(result).map_err(|_| corrupt("merge token overflows"))?;
        if special[left_index] || special[right_index] || special[result_index] {
            return Err(corrupt("tokenizer merges cannot reference special tokens"));
        }
        let combined_length = pieces[left_index]
            .len()
            .checked_add(pieces[right_index].len())
            .ok_or_else(|| corrupt("merged token piece length overflows"))?;
        if combined_length > usize::from(spec.maximum_piece_bytes())
            || !piece_is_concatenation(
                &pieces[result_index],
                &pieces[left_index],
                &pieces[right_index],
            )
        {
            return Err(corrupt("tokenizer merge result does not match its pieces"));
        }
        if merges.insert((left, right), MergeRule { rank, result }).is_some() {
            return Err(corrupt("tokenizer merge pair is duplicated"));
        }
    }
    Ok(merges)
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
    let data_bytes = u64::try_from(data_length)
        .map_err(|_| EngineError::resource_exhaustion("tokenizer size is not representable"))?;
    let vocabulary_bytes = u64::from(vocabulary_size)
        .checked_mul(PERSISTENT_BYTES_PER_VOCABULARY_ENTRY)
        .ok_or_else(|| EngineError::resource_exhaustion("vocabulary memory bound overflows"))?;
    let merge_bytes = u64::from(merge_count)
        .checked_mul(PERSISTENT_BYTES_PER_MERGE_RULE)
        .ok_or_else(|| EngineError::resource_exhaustion("merge memory bound overflows"))?;
    data_bytes
        .checked_add(vocabulary_bytes)
        .and_then(|value| value.checked_add(merge_bytes))
        .ok_or_else(|| EngineError::resource_exhaustion("tokenizer memory bound overflows"))
}

fn read_u16(input: &[u8], offset: usize) -> Result<u16, EngineError> {
    let bytes = read_exact::<2>(input, offset)?;
    Ok(u16::from_le_bytes(bytes))
}

fn read_u32(input: &[u8], offset: usize) -> Result<u32, EngineError> {
    let bytes = read_exact::<4>(input, offset)?;
    Ok(u32::from_le_bytes(bytes))
}

fn read_exact<const N: usize>(input: &[u8], offset: usize) -> Result<[u8; N], EngineError> {
    let end = offset.checked_add(N).ok_or_else(|| corrupt("tokenizer field offset overflows"))?;
    input
        .get(offset..end)
        .ok_or_else(|| corrupt("tokenizer data is truncated"))?
        .try_into()
        .map_err(|_| corrupt("tokenizer field length is invalid"))
}

fn checked_add_usize(left: usize, right: usize) -> Result<usize, EngineError> {
    left.checked_add(right).ok_or_else(|| corrupt("tokenizer offset overflows"))
}

fn usize_to_u32(value: usize) -> Result<u32, EngineError> {
    u32::try_from(value).map_err(|_| corrupt("tokenizer length exceeds its field"))
}

fn corrupt(detail: impl AsRef<str>) -> EngineError {
    EngineError::corrupt_model_package(detail)
}

const _: () = {
    assert!(size_of::<Node>() <= 256);
};

#[cfg(test)]
mod tests {
    use super::*;
    use personalai_engine_model_format::ParsedModelPackage;
    use personalai_engine_testkit::PackageFixtureBuilder;

    #[test]
    fn signed_fixture_tokenizer_round_trips_utf8_and_special_tokens() {
        let fixture = PackageFixtureBuilder::new().build().expect("fixture must build");
        let parsed = ParsedModelPackage::parse(fixture.bytes()).expect("fixture must parse");
        let data = parsed.tokenizer_data(fixture.bytes()).expect("tokenizer data must exist");
        let pool = MemoryPool::new(1 << 20).expect("pool must validate");
        let tokenizer = build_tokenizer(data, &pool).expect("tokenizer must build");
        let tokens = tokenizer.encode("Hello, 世界").expect("text must encode");
        assert_eq!(tokens.first().copied(), Some(256));
        assert_eq!(tokenizer.decode(&tokens).expect("tokens must decode"), "Hello, 世界");
        assert_eq!(tokenizer.spec().end_of_sequence_token(), 257);
    }

    #[test]
    fn corrupt_data_and_insufficient_memory_fail_closed() {
        let fixture = PackageFixtureBuilder::new().build().expect("fixture must build");
        let parsed = ParsedModelPackage::parse(fixture.bytes()).expect("fixture must parse");
        let data = parsed.tokenizer_data(fixture.bytes()).expect("tokenizer data must exist");
        let mut corrupt_data = data.to_vec();
        corrupt_data[0] ^= 1;
        let pool = MemoryPool::new(1 << 20).expect("pool must validate");
        assert!(matches!(
            FirstPartyBpeTokenizer::parse(&corrupt_data, &pool),
            Err(EngineError::CorruptModelPackage(_))
        ));
        let tiny_pool = MemoryPool::new(1).expect("pool must validate");
        assert!(matches!(
            FirstPartyBpeTokenizer::parse(data, &tiny_pool),
            Err(EngineError::ResourceExhaustion(_))
        ));

        for end in 0..data.len() {
            assert!(
                FirstPartyBpeTokenizer::parse(&data[..end], &pool).is_err(),
                "tokenizer truncation at offset {end} was accepted"
            );
        }
        for index in 0..data.len() {
            let mut mutation = data.to_vec();
            mutation[index] ^= 1;
            let _bounded_result = FirstPartyBpeTokenizer::parse(&mutation, &pool);
        }
    }

    #[test]
    fn metadata_and_token_bounds_are_enforced() {
        assert!(TokenizerSpec::new(TokenizerFormat::FirstPartyBpeV1, 0, None, 0, 1).is_err());
        assert!(TokenizerSpec::new(TokenizerFormat::FirstPartyBpeV1, 256, Some(8), 8, 1).is_err());
        let spec = TokenizerSpec::new(TokenizerFormat::FirstPartyBpeV1, 256, None, 1, 1)
            .expect("metadata must validate");
        assert!(spec.validate_tokens(&[256]).is_err());
    }
}
