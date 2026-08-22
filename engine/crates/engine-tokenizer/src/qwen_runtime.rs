//! Inert Qwen2.5 tokenizer runtime core behind a trusted Unicode-16 boundary.
//!
//! This module is intentionally private. It implements deterministic fixed
//! pre-tokenizer, BPE, added-token, and decoding mechanics over already parsed
//! AIPTok v2 tables. The public tokenizer factory remains fail-closed until the
//! admitted Unicode-16 primitives are wired through a reviewed adapter.

use super::qwen_v2::{
    AipTokQwen25AddedRole, Qwen25LoadedTables, Qwen25MergeRule, load_qwen25_tables,
};
use super::{MAX_SPECIAL_TOKENS, SpecialTokenPolicy, TEMPORARY_BYTES_PER_INPUT_TOKEN, checked_add};
use personalai_engine_memory::{MemoryPool, MemoryReservation};
use personalai_engine_types::{EngineError, MAX_REQUEST_BYTES, MAX_TOKEN_COUNT};
use std::cmp::Reverse;
use std::collections::{BTreeSet, BinaryHeap};
use std::mem::size_of;
use std::ops::Range;

const REQUIRED_UNICODE_VERSION: (u8, u8, u8) = (16, 0, 0);
const SCANNER_BYTES_PER_SCALAR: u64 = 64;
const PENDING_UTF8_BYTES: u64 = 4;
const CONTRACTION_SUFFIXES: [&[u8]; 7] = [b"s", b"t", b"re", b"ve", b"m", b"ll", b"d"];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GeneralClass {
    Letter,
    Number,
    Other,
}

/// Engine-internal Unicode authority. No package or client can implement or
/// select this interface; the production implementation is intentionally
/// absent until the admitted Unicode-16 primitives are safely adapted and
/// independently qualified.
trait TrustedQwen25Unicode16: Send + Sync {
    fn unicode_version(&self) -> (u8, u8, u8);
    fn normalize_nfc(
        &self,
        input: &str,
        output: &mut BoundedNormalizedText,
    ) -> Result<(), EngineError>;
    fn general_class(&self, scalar: char) -> GeneralClass;
    fn contraction_ascii_fold(&self, scalar: char) -> Option<u8>;
}

struct BoundedNormalizedText {
    value: String,
    maximum_bytes: usize,
}

impl BoundedNormalizedText {
    fn new(maximum_bytes: usize) -> Result<Self, EngineError> {
        let mut value = String::new();
        value.try_reserve_exact(maximum_bytes).map_err(|_| {
            EngineError::resource_exhaustion("Qwen normalization allocation failed")
        })?;
        Ok(Self { value, maximum_bytes })
    }

    fn push_str(&mut self, value: &str) -> Result<(), EngineError> {
        let new_length = self.value.len().checked_add(value.len()).ok_or_else(|| {
            EngineError::resource_exhaustion("Qwen normalized input length overflows")
        })?;
        if new_length > self.maximum_bytes {
            return Err(EngineError::resource_exhaustion(
                "Qwen normalized input exceeds the tokenizer limit",
            ));
        }
        self.value.push_str(value);
        Ok(())
    }

    fn as_str(&self) -> &str {
        &self.value
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ScalarClass {
    Letter,
    Number,
    WhiteSpace,
    Other,
}

#[derive(Debug, Clone, Copy)]
struct Scalar {
    scalar: char,
    start: usize,
    end: usize,
    class: ScalarClass,
}

struct PretokenRanges {
    ranges: Vec<Range<usize>>,
    _reservation: Option<MemoryReservation>,
}

fn is_fixed_white_space(scalar: char) -> bool {
    matches!(
        scalar,
        '\u{0009}'..='\u{000D}'
            | '\u{0020}'
            | '\u{0085}'
            | '\u{00A0}'
            | '\u{1680}'
            | '\u{2000}'..='\u{200A}'
            | '\u{2028}'
            | '\u{2029}'
            | '\u{202F}'
            | '\u{205F}'
            | '\u{3000}'
    )
}

fn scalar_class(profile: &dyn TrustedQwen25Unicode16, scalar: char) -> ScalarClass {
    if is_fixed_white_space(scalar) {
        ScalarClass::WhiteSpace
    } else {
        match profile.general_class(scalar) {
            GeneralClass::Letter => ScalarClass::Letter,
            GeneralClass::Number => ScalarClass::Number,
            GeneralClass::Other => ScalarClass::Other,
        }
    }
}

fn contraction_end(
    scalars: &[Scalar],
    start: usize,
    profile: &dyn TrustedQwen25Unicode16,
) -> Option<usize> {
    if scalars.get(start)?.scalar != '\'' {
        return None;
    }
    for suffix in CONTRACTION_SUFFIXES {
        let end = start.checked_add(1)?.checked_add(suffix.len())?;
        let Some(candidate) = scalars.get(start + 1..end) else {
            continue;
        };
        if candidate.iter().zip(suffix).all(|(scalar, expected)| {
            profile.contraction_ascii_fold(scalar.scalar) == Some(*expected)
        }) {
            return Some(end);
        }
    }
    None
}

fn letter_end(scalars: &[Scalar], start: usize) -> Option<usize> {
    let first = scalars.get(start)?;
    let mut cursor = if first.class == ScalarClass::Letter {
        start
    } else if first.scalar != '\r'
        && first.scalar != '\n'
        && first.class != ScalarClass::Number
        && scalars.get(start + 1).is_some_and(|next| next.class == ScalarClass::Letter)
    {
        start + 1
    } else {
        return None;
    };
    while scalars.get(cursor).is_some_and(|scalar| scalar.class == ScalarClass::Letter) {
        cursor += 1;
    }
    Some(cursor)
}

fn punctuation_end(scalars: &[Scalar], start: usize) -> Option<usize> {
    let mut cursor = start;
    if scalars.get(cursor)?.scalar == ' '
        && scalars.get(cursor + 1).is_some_and(|next| next.class == ScalarClass::Other)
    {
        cursor += 1;
    }
    let punctuation_start = cursor;
    while scalars.get(cursor).is_some_and(|scalar| scalar.class == ScalarClass::Other) {
        cursor += 1;
    }
    if cursor == punctuation_start {
        return None;
    }
    while scalars.get(cursor).is_some_and(|scalar| matches!(scalar.scalar, '\r' | '\n')) {
        cursor += 1;
    }
    Some(cursor)
}

fn newline_end(scalars: &[Scalar], start: usize) -> Option<usize> {
    if scalars.get(start)?.class != ScalarClass::WhiteSpace {
        return None;
    }
    let mut cursor = start;
    let mut last_newline = None;
    while let Some(scalar) =
        scalars.get(cursor).filter(|scalar| scalar.class == ScalarClass::WhiteSpace)
    {
        if matches!(scalar.scalar, '\r' | '\n') {
            last_newline = Some(cursor + 1);
        }
        cursor += 1;
    }
    last_newline
}

fn trailing_white_space_end(scalars: &[Scalar], start: usize) -> Option<usize> {
    if scalars.get(start)?.class != ScalarClass::WhiteSpace {
        return None;
    }
    let mut end = start;
    while scalars.get(end).is_some_and(|scalar| scalar.class == ScalarClass::WhiteSpace) {
        end += 1;
    }
    if end == scalars.len() {
        Some(end)
    } else if end.saturating_sub(start) >= 2 {
        Some(end - 1)
    } else {
        None
    }
}

fn white_space_end(scalars: &[Scalar], start: usize) -> Option<usize> {
    if scalars.get(start)?.class != ScalarClass::WhiteSpace {
        return None;
    }
    let mut end = start + 1;
    while scalars.get(end).is_some_and(|scalar| scalar.class == ScalarClass::WhiteSpace) {
        end += 1;
    }
    Some(end)
}

fn scan_pretokens(
    normalized: &str,
    profile: &dyn TrustedQwen25Unicode16,
    pool: &MemoryPool,
) -> Result<PretokenRanges, EngineError> {
    if profile.unicode_version() != REQUIRED_UNICODE_VERSION {
        return Err(EngineError::unsupported_tokenizer(
            "Qwen2.5 tokenizer requires the reviewed Unicode 16.0 profile",
        ));
    }
    if normalized.is_empty() {
        return Ok(PretokenRanges { ranges: Vec::new(), _reservation: None });
    }
    let scalar_count = normalized.chars().count();
    let work_bytes = u64::try_from(scalar_count)
        .ok()
        .and_then(|count| count.checked_mul(SCANNER_BYTES_PER_SCALAR))
        .ok_or_else(|| EngineError::resource_exhaustion("Qwen scanner memory bound overflows"))?;
    let reservation = pool.reserve(work_bytes)?;
    let mut scalars = Vec::new();
    scalars
        .try_reserve_exact(scalar_count)
        .map_err(|_| EngineError::resource_exhaustion("Qwen scanner scalar allocation failed"))?;
    for (start, scalar) in normalized.char_indices() {
        let end = checked_add(start, scalar.len_utf8())?;
        scalars.push(Scalar { scalar, start, end, class: scalar_class(profile, scalar) });
    }
    let mut ranges = Vec::new();
    ranges
        .try_reserve_exact(scalar_count)
        .map_err(|_| EngineError::resource_exhaustion("Qwen scanner range allocation failed"))?;
    let mut start = 0_usize;
    while start < scalars.len() {
        let end = contraction_end(&scalars, start, profile)
            .or_else(|| letter_end(&scalars, start))
            .or_else(|| (scalars[start].class == ScalarClass::Number).then_some(start + 1))
            .or_else(|| punctuation_end(&scalars, start))
            .or_else(|| newline_end(&scalars, start))
            .or_else(|| trailing_white_space_end(&scalars, start))
            .or_else(|| white_space_end(&scalars, start))
            .ok_or_else(|| EngineError::internal_invariant("Qwen scanner made no progress"))?;
        if end <= start || end > scalars.len() {
            return Err(EngineError::internal_invariant("Qwen scanner range is invalid"));
        }
        ranges.push(scalars[start].start..scalars[end - 1].end);
        start = end;
    }
    Ok(PretokenRanges { ranges, _reservation: Some(reservation) })
}

#[derive(Debug)]
struct BpeNode {
    token: u32,
    previous: Option<usize>,
    next: Option<usize>,
    alive: bool,
}

#[derive(Debug)]
struct Qwen25RuntimeCore {
    tables: Qwen25LoadedTables,
    added_by_first_byte: [Vec<usize>; 256],
    _added_index_reservation: Option<MemoryReservation>,
}

impl Qwen25RuntimeCore {
    fn load(data: &[u8], pool: &MemoryPool) -> Result<Self, EngineError> {
        let tables = load_qwen25_tables(data, pool)?;
        if !tables.descriptor.matches_pinned_qwen25_1_5b_counts() {
            return Err(EngineError::unsupported_tokenizer(
                "Qwen runtime construction requires the pinned production cardinality profile",
            ));
        }
        Self::from_tables(tables)
    }

    #[cfg(test)]
    fn load_test_fixture(data: &[u8], pool: &MemoryPool) -> Result<Self, EngineError> {
        Self::from_tables(load_qwen25_tables(data, pool)?)
    }

    fn from_tables(tables: Qwen25LoadedTables) -> Result<Self, EngineError> {
        let index_bytes = tables
            .added_tokens
            .len()
            .checked_mul(size_of::<usize>())
            .and_then(|bytes| u64::try_from(bytes).ok())
            .ok_or_else(|| {
                EngineError::resource_exhaustion("Qwen added-token index bound overflows")
            })?;
        let index_reservation =
            (index_bytes != 0).then(|| tables.pool.reserve(index_bytes)).transpose()?;
        let mut added_by_first_byte: [Vec<usize>; 256] = std::array::from_fn(|_| Vec::new());
        let mut bucket_counts = [0_usize; 256];
        for added in &tables.added_tokens {
            let token_index = usize::try_from(added.token_id).map_err(|_| {
                EngineError::corrupt_model_package("Qwen added-token ID is not addressable")
            })?;
            let first_byte =
                *tables.pieces.get(token_index).and_then(|piece| piece.first()).ok_or_else(
                    || EngineError::corrupt_model_package("Qwen added-token literal is empty"),
                )?;
            let count = &mut bucket_counts[usize::from(first_byte)];
            *count = count.checked_add(1).ok_or_else(|| {
                EngineError::resource_exhaustion("Qwen added-token bucket count overflows")
            })?;
        }
        for (bucket, count) in added_by_first_byte.iter_mut().zip(bucket_counts) {
            bucket.try_reserve_exact(count).map_err(|_| {
                EngineError::resource_exhaustion("Qwen added-token index allocation failed")
            })?;
        }
        for (index, added) in tables.added_tokens.iter().enumerate() {
            let token_index = usize::try_from(added.token_id).map_err(|_| {
                EngineError::corrupt_model_package("Qwen added-token ID is not addressable")
            })?;
            let first_byte =
                *tables.pieces.get(token_index).and_then(|piece| piece.first()).ok_or_else(
                    || EngineError::corrupt_model_package("Qwen added-token literal is empty"),
                )?;
            added_by_first_byte[usize::from(first_byte)].push(index);
        }
        Ok(Self { tables, added_by_first_byte, _added_index_reservation: index_reservation })
    }

    fn token_index(&self, token: u32) -> Result<usize, EngineError> {
        self.tables.descriptor.validate_token(token)?;
        usize::try_from(token)
            .map_err(|_| EngineError::invalid_input("Qwen token ID is not addressable"))
    }

    fn added_role(&self, token: u32) -> Result<Option<AipTokQwen25AddedRole>, EngineError> {
        let index = self.token_index(token)?;
        let bpe_count = usize::try_from(self.tables.descriptor.bpe_vocabulary_size())
            .map_err(|_| EngineError::internal_invariant("Qwen BPE count is not addressable"))?;
        if index < bpe_count {
            return Ok(None);
        }
        let added_index = index
            .checked_sub(bpe_count)
            .ok_or_else(|| EngineError::internal_invariant("Qwen added-token index underflows"))?;
        let added = self.tables.added_tokens.get(added_index).ok_or_else(|| {
            EngineError::internal_invariant("Qwen added-token record is unavailable")
        })?;
        if added.token_id != token {
            return Err(EngineError::internal_invariant("Qwen added-token identity changed"));
        }
        Ok(Some(added.role))
    }

    fn token_piece(&self, token: u32) -> Result<&[u8], EngineError> {
        self.tables
            .pieces
            .get(self.token_index(token)?)
            .map(Box::as_ref)
            .ok_or_else(|| EngineError::internal_invariant("Qwen token piece is unavailable"))
    }

    fn vocabulary_lookup(&self, piece: &[u8]) -> Option<u32> {
        self.tables.vocabulary.get(piece).copied()
    }

    fn is_special_token(&self, token: u32) -> Result<bool, EngineError> {
        Ok(self.added_role(token)?.is_some_and(AipTokQwen25AddedRole::is_special))
    }

    fn push_candidate(
        &self,
        nodes: &[BpeNode],
        position: usize,
        heap: &mut BinaryHeap<Reverse<(u32, usize)>>,
    ) {
        let Some(left) = nodes.get(position).filter(|node| node.alive) else {
            return;
        };
        let Some(right_position) = left.next else {
            return;
        };
        let Some(right) = nodes.get(right_position).filter(|node| node.alive) else {
            return;
        };
        if let Some(rule) = self.tables.merges.get(&(left.token, right.token)) {
            heap.push(Reverse((rule.rank, position)));
        }
    }

    fn append_bpe_bytes(&self, bytes: &[u8], output: &mut Vec<u32>) -> Result<(), EngineError> {
        if bytes.is_empty() {
            return Ok(());
        }
        let count = bytes.len();
        let temporary_bytes = u64::try_from(count)
            .ok()
            .and_then(|count| count.checked_mul(TEMPORARY_BYTES_PER_INPUT_TOKEN))
            .ok_or_else(|| EngineError::resource_exhaustion("Qwen BPE memory bound overflows"))?;
        let _temporary = self.tables.pool.reserve(temporary_bytes)?;
        let mut nodes = Vec::new();
        nodes
            .try_reserve_exact(count)
            .map_err(|_| EngineError::resource_exhaustion("Qwen BPE node allocation failed"))?;
        for (index, byte) in bytes.iter().enumerate() {
            nodes.push(BpeNode {
                token: self.tables.byte_token_ids[usize::from(*byte)],
                previous: index.checked_sub(1),
                next: (index + 1 < count).then_some(index + 1),
                alive: true,
            });
        }
        let heap_capacity = count
            .checked_mul(3)
            .ok_or_else(|| EngineError::resource_exhaustion("Qwen BPE heap bound overflows"))?;
        let mut heap = BinaryHeap::new();
        heap.try_reserve(heap_capacity)
            .map_err(|_| EngineError::resource_exhaustion("Qwen BPE heap allocation failed"))?;
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
            let Some(Qwen25MergeRule { rank: current_rank, result }) =
                self.tables.merges.get(&(left.token, right.token)).copied()
            else {
                continue;
            };
            if current_rank != rank {
                continue;
            }
            let previous = left.previous;
            let following = right.next;
            nodes[position].token = result;
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
        let mut position = Some(0_usize);
        while let Some(index) = position {
            let node = nodes
                .get(index)
                .filter(|node| node.alive)
                .ok_or_else(|| EngineError::internal_invariant("Qwen BPE chain is invalid"))?;
            self.append_token(node.token, output)?;
            position = node.next;
        }
        Ok(())
    }

    fn append_token(&self, token: u32, output: &mut Vec<u32>) -> Result<(), EngineError> {
        if output.len()
            >= usize::try_from(self.tables.descriptor.maximum_produced_tokens())
                .unwrap_or(usize::MAX)
        {
            return Err(EngineError::resource_exhaustion(
                "Qwen encoded token count exceeds the tokenizer limit",
            ));
        }
        output.push(token);
        Ok(())
    }

    fn append_pretokenized(
        &self,
        normalized: &str,
        profile: &dyn TrustedQwen25Unicode16,
        output: &mut Vec<u32>,
    ) -> Result<(), EngineError> {
        let scanned = scan_pretokens(normalized, profile, &self.tables.pool)?;
        for range in scanned.ranges {
            self.append_bpe_bytes(
                normalized
                    .get(range)
                    .ok_or_else(|| {
                        EngineError::internal_invariant(
                            "Qwen scanner split invalid UTF-8 boundaries",
                        )
                    })?
                    .as_bytes(),
                output,
            )?;
        }
        Ok(())
    }

    fn matching_added(&self, input: &[u8]) -> Option<(u32, AipTokQwen25AddedRole, usize)> {
        let candidates = self.added_by_first_byte.get(usize::from(*input.first()?))?;
        candidates.iter().find_map(|index| {
            let added = self.tables.added_tokens.get(*index)?;
            let piece = self.tables.pieces.get(usize::try_from(added.token_id).ok()?)?;
            input.starts_with(piece).then_some((added.token_id, added.role, piece.len()))
        })
    }

    fn validate_special_policy(&self, policy: SpecialTokenPolicy<'_>) -> Result<(), EngineError> {
        let SpecialTokenPolicy::AllowListed(allowed) = policy else {
            return Ok(());
        };
        if allowed.len() > usize::try_from(MAX_SPECIAL_TOKENS).unwrap_or(usize::MAX) {
            return Err(EngineError::invalid_input("Qwen allowed-special list is too large"));
        }
        let mut unique = BTreeSet::new();
        for token in allowed {
            if !unique.insert(*token) || !self.is_special_token(*token)? {
                return Err(EngineError::invalid_input("Qwen allowed-special list is invalid"));
            }
        }
        Ok(())
    }

    fn encode(
        &self,
        input: &str,
        policy: SpecialTokenPolicy<'_>,
        profile: &dyn TrustedQwen25Unicode16,
    ) -> Result<Vec<u32>, EngineError> {
        let maximum_input = usize::try_from(self.tables.descriptor.maximum_input_bytes())
            .map_err(|_| EngineError::resource_exhaustion("Qwen input bound is not addressable"))?;
        if input.len() > maximum_input || input.len() > MAX_REQUEST_BYTES {
            return Err(EngineError::invalid_input("Qwen tokenizer input exceeds its byte limit"));
        }
        if profile.unicode_version() != REQUIRED_UNICODE_VERSION {
            return Err(EngineError::unsupported_tokenizer(
                "Qwen2.5 tokenizer requires the reviewed Unicode 16.0 profile",
            ));
        }
        self.validate_special_policy(policy)?;
        let normalized_limit = maximum_input.min(MAX_REQUEST_BYTES);
        let _normalization =
            self.tables.pool.reserve(u64::try_from(normalized_limit).map_err(|_| {
                EngineError::resource_exhaustion("Qwen normalization bound is not representable")
            })?)?;
        let mut normalized_storage = BoundedNormalizedText::new(normalized_limit)?;
        profile.normalize_nfc(input, &mut normalized_storage)?;
        let normalized = normalized_storage.as_str();
        let token_capacity = normalized.len().min(
            usize::try_from(self.tables.descriptor.maximum_produced_tokens()).unwrap_or(usize::MAX),
        );
        let _output_reservation = if token_capacity == 0 {
            None
        } else {
            let bytes = token_capacity
                .checked_mul(size_of::<u32>())
                .and_then(|bytes| u64::try_from(bytes).ok())
                .ok_or_else(|| EngineError::resource_exhaustion("Qwen output bound overflows"))?;
            Some(self.tables.pool.reserve(bytes)?)
        };
        let mut output = Vec::new();
        output
            .try_reserve_exact(token_capacity)
            .map_err(|_| EngineError::resource_exhaustion("Qwen token output allocation failed"))?;
        if matches!(policy, SpecialTokenPolicy::TreatAsOrdinaryBytes) {
            self.append_pretokenized(normalized, profile, &mut output)?;
            return Ok(output);
        }
        let mut segment_start = 0_usize;
        let mut position = 0_usize;
        while position < normalized.len() {
            let Some((token, role, length)) =
                self.matching_added(&normalized.as_bytes()[position..])
            else {
                position = position.checked_add(1).ok_or_else(|| {
                    EngineError::resource_exhaustion("Qwen added-token scan overflows")
                })?;
                continue;
            };
            self.append_pretokenized(&normalized[segment_start..position], profile, &mut output)?;
            if role.is_special() {
                let allowed = match policy {
                    SpecialTokenPolicy::Reject | SpecialTokenPolicy::TreatAsOrdinaryBytes => false,
                    SpecialTokenPolicy::AllowAll => true,
                    SpecialTokenPolicy::AllowListed(tokens) => tokens.contains(&token),
                };
                if !allowed {
                    return Err(EngineError::invalid_input(
                        "Qwen input contains special-token text that was not explicitly allowed",
                    ));
                }
            }
            self.append_token(token, &mut output)?;
            position = position.checked_add(length).ok_or_else(|| {
                EngineError::resource_exhaustion("Qwen added-token scan overflows")
            })?;
            segment_start = position;
        }
        self.append_pretokenized(&normalized[segment_start..], profile, &mut output)?;
        Ok(output)
    }

    fn encode_pretoken_bytes(&self, bytes: &[u8]) -> Result<Vec<u32>, EngineError> {
        if bytes.len()
            > usize::try_from(self.tables.descriptor.maximum_input_bytes()).unwrap_or(usize::MAX)
        {
            return Err(EngineError::invalid_input("Qwen diagnostic bytes exceed input limit"));
        }
        let mut output = Vec::new();
        output.try_reserve_exact(bytes.len()).map_err(|_| {
            EngineError::resource_exhaustion("Qwen diagnostic output allocation failed")
        })?;
        self.append_bpe_bytes(bytes, &mut output)?;
        Ok(output)
    }

    fn validate_tokens(&self, tokens: &[u32]) -> Result<(), EngineError> {
        if tokens.len()
            > usize::try_from(self.tables.descriptor.maximum_produced_tokens())
                .unwrap_or(usize::MAX)
            || tokens.len() > usize::try_from(MAX_TOKEN_COUNT).unwrap_or(usize::MAX)
        {
            return Err(EngineError::invalid_input("Qwen token sequence exceeds its limit"));
        }
        for token in tokens {
            self.tables.descriptor.validate_token(*token)?;
        }
        Ok(())
    }

    fn decode_bytes(&self, tokens: &[u32]) -> Result<Vec<u8>, EngineError> {
        self.validate_tokens(tokens)?;
        let mut length = 0_usize;
        for token in tokens {
            if !self.is_special_token(*token)? {
                length = length
                    .checked_add(self.token_piece(*token)?.len())
                    .filter(|length| *length <= MAX_REQUEST_BYTES)
                    .ok_or_else(|| {
                        EngineError::resource_exhaustion("Qwen decoded bytes exceed hard limit")
                    })?;
            }
        }
        let _output_reservation = if length == 0 {
            None
        } else {
            Some(self.tables.pool.reserve(u64::try_from(length).map_err(|_| {
                EngineError::resource_exhaustion("Qwen decoded length is not representable")
            })?)?)
        };
        let mut output = Vec::new();
        output.try_reserve_exact(length).map_err(|_| {
            EngineError::resource_exhaustion("Qwen decoded output allocation failed")
        })?;
        for token in tokens {
            if !self.is_special_token(*token)? {
                output.extend_from_slice(self.token_piece(*token)?);
            }
        }
        Ok(output)
    }

    fn decode(&self, tokens: &[u32]) -> Result<String, EngineError> {
        String::from_utf8(self.decode_bytes(tokens)?)
            .map_err(|_| EngineError::invalid_input("Qwen token sequence is not valid UTF-8"))
    }
}

struct Qwen25IncrementalDecoder<'a> {
    core: &'a Qwen25RuntimeCore,
    pending: Vec<u8>,
    decoded_tokens: usize,
    decoded_bytes: usize,
    failed: bool,
    _pending_reservation: MemoryReservation,
}

impl<'a> Qwen25IncrementalDecoder<'a> {
    fn new(core: &'a Qwen25RuntimeCore) -> Result<Self, EngineError> {
        Ok(Self {
            core,
            pending: Vec::with_capacity(usize::try_from(PENDING_UTF8_BYTES).unwrap_or(4)),
            decoded_tokens: 0,
            decoded_bytes: 0,
            failed: false,
            _pending_reservation: core.tables.pool.reserve(PENDING_UTF8_BYTES)?,
        })
    }

    fn push(&mut self, token: u32) -> Result<String, EngineError> {
        if self.failed {
            return Err(EngineError::invalid_input(
                "Qwen incremental decoder is terminal after malformed UTF-8",
            ));
        }
        let next_decoded_tokens = self
            .decoded_tokens
            .checked_add(1)
            .filter(|count| {
                *count
                    <= usize::try_from(self.core.tables.descriptor.maximum_produced_tokens())
                        .unwrap_or(usize::MAX)
                    && *count <= usize::try_from(MAX_TOKEN_COUNT).unwrap_or(usize::MAX)
            })
            .ok_or_else(|| {
                EngineError::resource_exhaustion("Qwen stream exceeds its token limit")
            })?;
        if self.core.is_special_token(token)? {
            self.decoded_tokens = next_decoded_tokens;
            return Ok(String::new());
        }
        let piece = self.core.token_piece(token)?;
        let next_decoded_bytes = self
            .decoded_bytes
            .checked_add(piece.len())
            .filter(|bytes| *bytes <= MAX_REQUEST_BYTES)
            .ok_or_else(|| EngineError::resource_exhaustion("Qwen stream exceeds byte limit"))?;
        let combined_length = self
            .pending
            .len()
            .checked_add(piece.len())
            .ok_or_else(|| EngineError::resource_exhaustion("Qwen stream buffer overflows"))?;
        let _temporary = self.tables_pool_reservation(combined_length)?;
        let mut combined = Vec::new();
        combined.try_reserve_exact(combined_length).map_err(|_| {
            EngineError::resource_exhaustion("Qwen stream temporary allocation failed")
        })?;
        combined.extend_from_slice(&self.pending);
        combined.extend_from_slice(piece);
        match std::str::from_utf8(&combined) {
            Ok(text) => {
                self.pending.clear();
                self.decoded_tokens = next_decoded_tokens;
                self.decoded_bytes = next_decoded_bytes;
                Ok(text.to_owned())
            }
            Err(error) if error.error_len().is_none() => {
                let valid = error.valid_up_to();
                let output = std::str::from_utf8(&combined[..valid])
                    .map_err(|_| EngineError::internal_invariant("Qwen UTF-8 prefix changed"))?
                    .to_owned();
                self.pending.clear();
                self.pending.extend_from_slice(&combined[valid..]);
                if self.pending.len() > usize::try_from(PENDING_UTF8_BYTES).unwrap_or(4) {
                    self.pending.clear();
                    self.failed = true;
                    return Err(EngineError::invalid_input(
                        "Qwen generated token ends with an oversized UTF-8 suffix",
                    ));
                }
                self.decoded_tokens = next_decoded_tokens;
                self.decoded_bytes = next_decoded_bytes;
                Ok(output)
            }
            Err(_) => {
                self.pending.clear();
                self.failed = true;
                Err(EngineError::invalid_input(
                    "Qwen generated token bytes contain malformed UTF-8",
                ))
            }
        }
    }

    fn tables_pool_reservation(&self, bytes: usize) -> Result<MemoryReservation, EngineError> {
        self.core.tables.pool.reserve(u64::try_from(bytes).map_err(|_| {
            EngineError::resource_exhaustion("Qwen stream size is not representable")
        })?)
    }

    fn finish(self) -> Result<(), EngineError> {
        if self.failed {
            Err(EngineError::invalid_input(
                "Qwen incremental decoder is terminal after malformed UTF-8",
            ))
        } else if self.pending.is_empty() {
            Ok(())
        } else {
            Err(EngineError::invalid_input(
                "Qwen generated token stream ends with incomplete UTF-8",
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::qwen_v2::{
        AipTokQwen25AddedToken, AipTokQwen25Definition, AipTokQwen25Merge,
        AipTokQwen25VocabularyEntry,
    };
    use std::sync::Arc;

    #[derive(Debug, Clone, Copy)]
    struct SyntheticUnicode16;

    struct WrongUnicodeVersion;

    struct ExpandingUnicode16;

    impl TrustedQwen25Unicode16 for SyntheticUnicode16 {
        fn unicode_version(&self) -> (u8, u8, u8) {
            REQUIRED_UNICODE_VERSION
        }

        fn normalize_nfc(
            &self,
            input: &str,
            output: &mut BoundedNormalizedText,
        ) -> Result<(), EngineError> {
            let normalized = match input {
                "Cafe\u{0301}" => "Café",
                "\u{1100}\u{1161}" => "가",
                "a\u{1ACF}\u{0315}\u{0300}\u{05AE}b" => "a\u{1ACF}\u{05AE}\u{0300}\u{0315}b",
                _ => input,
            };
            output.push_str(normalized)
        }

        fn general_class(&self, scalar: char) -> GeneralClass {
            if scalar.is_ascii_alphabetic()
                || matches!(
                    scalar,
                    '\u{00E9}'
                        | '\u{01C5}'
                        | '\u{02B0}'
                        | '\u{0416}'
                        | '\u{1100}'
                        | '\u{1161}'
                        | '\u{4E2D}'
                        | '\u{AC00}'
                )
            {
                GeneralClass::Letter
            } else if scalar.is_ascii_digit()
                || matches!(scalar, '\u{0661}' | '\u{2167}' | '\u{00BD}')
            {
                GeneralClass::Number
            } else {
                GeneralClass::Other
            }
        }

        fn contraction_ascii_fold(&self, scalar: char) -> Option<u8> {
            if scalar == '\u{017F}' {
                Some(b's')
            } else {
                scalar.is_ascii_alphabetic().then(|| scalar.to_ascii_lowercase() as u8)
            }
        }
    }

    impl TrustedQwen25Unicode16 for WrongUnicodeVersion {
        fn unicode_version(&self) -> (u8, u8, u8) {
            (17, 0, 0)
        }

        fn normalize_nfc(
            &self,
            input: &str,
            output: &mut BoundedNormalizedText,
        ) -> Result<(), EngineError> {
            output.push_str(input)
        }

        fn general_class(&self, _: char) -> GeneralClass {
            GeneralClass::Other
        }

        fn contraction_ascii_fold(&self, _: char) -> Option<u8> {
            None
        }
    }

    impl TrustedQwen25Unicode16 for ExpandingUnicode16 {
        fn unicode_version(&self) -> (u8, u8, u8) {
            REQUIRED_UNICODE_VERSION
        }

        fn normalize_nfc(
            &self,
            _: &str,
            output: &mut BoundedNormalizedText,
        ) -> Result<(), EngineError> {
            output.push_str("expanded")
        }

        fn general_class(&self, _: char) -> GeneralClass {
            GeneralClass::Other
        }

        fn contraction_ascii_fold(&self, _: char) -> Option<u8> {
            None
        }
    }

    fn fixture(maximum_input_bytes: u32, maximum_produced_tokens: u32) -> Vec<u8> {
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
            byte_token_ids: std::array::from_fn(|byte| {
                u32::try_from(255_usize - byte).expect("byte map index fits")
            }),
            beginning_of_sequence_token: 258,
            end_of_sequence_token: 260,
            maximum_input_bytes,
            maximum_produced_tokens,
        }
        .encode()
        .expect("runtime fixture must encode")
    }

    fn core() -> (Qwen25RuntimeCore, MemoryPool) {
        let pool = MemoryPool::new(64 * 1024 * 1024).expect("pool validates");
        let core = Qwen25RuntimeCore::load_test_fixture(&fixture(4096, 4096), &pool)
            .expect("runtime tables must load");
        (core, pool)
    }

    fn chunks(input: &str) -> Vec<String> {
        let pool = MemoryPool::new(1 << 20).expect("pool validates");
        let profile = SyntheticUnicode16;
        let mut normalized = BoundedNormalizedText::new(input.len().saturating_mul(4).max(1))
            .expect("synthetic normalization buffer validates");
        profile.normalize_nfc(input, &mut normalized).expect("synthetic NFC vector works");
        let normalized = normalized.as_str();
        let scanned = scan_pretokens(normalized, &profile, &pool).expect("scanner works");
        scanned.ranges.into_iter().map(|range| normalized[range].to_owned()).collect()
    }

    fn strings(values: &[&str]) -> Vec<String> {
        values.iter().map(|value| (*value).to_owned()).collect()
    }

    #[test]
    fn independent_chunk_vectors_cover_priority_categories_and_whitespace() {
        assert_eq!(chunks("Hello world"), strings(&["Hello", " world"]));
        assert_eq!(chunks("I'm WE'RE he'd"), strings(&["I", "'m", " WE", "'RE", " he", "'d"]));
        assert_eq!(chunks("'ſfoo"), strings(&["'ſ", "foo"]));
        assert_eq!(chunks("abc123Ⅷ½"), strings(&["abc", "1", "2", "3", "Ⅷ", "½"]));
        assert_eq!(chunks("Hello, world!!!\n"), strings(&["Hello", ",", " world", "!!!\n"]));
        assert_eq!(chunks("A\r\nB\n\n"), strings(&["A", "\r\n", "B", "\n\n"]));
        assert_eq!(chunks("A  B "), strings(&["A", " ", " B", " "]));
        assert_eq!(chunks("\t\r\n"), strings(&["\t\r\n"]));
        assert_eq!(chunks("\u{0301}b"), strings(&["\u{0301}b"]));
        assert_eq!(chunks("\0A"), strings(&["\0A"]));
        assert_eq!(chunks("A中Ж"), strings(&["A中Ж"]));
        assert_eq!(chunks(" !?"), strings(&[" !?"]));
        assert_eq!(chunks(" 😀\n"), strings(&[" 😀\n"]));
        assert_eq!(chunks("\u{00A0}!"), strings(&["\u{00A0}", "!"]));
        assert_eq!(chunks("\u{200B}!"), strings(&["\u{200B}!"]));
        assert_eq!(chunks("<|im_start|>"), strings(&["<|", "im", "_start", "|>"]));
        assert_eq!(chunks("<tool_call>"), strings(&["<tool", "_call", ">"]));
    }

    #[test]
    fn synthetic_nfc_vectors_are_explicit_and_do_not_claim_general_normalization() {
        assert_eq!(chunks("Cafe\u{0301}"), strings(&["Café"]));
        assert_eq!(chunks("\u{1100}\u{1161}"), strings(&["가"]));
        assert_eq!(
            chunks("a\u{1ACF}\u{0315}\u{0300}\u{05AE}b"),
            strings(&["a", "\u{1ACF}\u{05AE}\u{0300}\u{0315}", "b"])
        );
        assert_eq!(SyntheticUnicode16.general_class('\u{088F}'), GeneralClass::Other);
        assert_eq!(SyntheticUnicode16.general_class('\u{11DE0}'), GeneralClass::Other);
    }

    #[test]
    fn fixed_white_space_and_category_classes_are_closed() {
        for scalar in ['A', 'a', 'ǅ', 'ʰ', '中'] {
            assert_eq!(SyntheticUnicode16.general_class(scalar), GeneralClass::Letter);
        }
        for scalar in ['1', '١', 'Ⅷ', '½'] {
            assert_eq!(SyntheticUnicode16.general_class(scalar), GeneralClass::Number);
        }
        assert!(is_fixed_white_space('\u{00A0}'));
        assert!(is_fixed_white_space('\u{3000}'));
        assert!(!is_fixed_white_space('\u{200B}'));
    }

    #[test]
    fn bpe_byte_map_all_bytes_decode_and_lookup_are_deterministic() {
        let (core, pool) = core();
        let baseline = pool.used_bytes().expect("metrics work");
        let all_bytes =
            (0_u16..=255).map(|byte| u8::try_from(byte).expect("byte fits")).collect::<Vec<_>>();
        let first = core.encode_pretoken_bytes(&all_bytes).expect("all bytes encode");
        let second = core.encode_pretoken_bytes(&all_bytes).expect("all bytes encode again");
        assert_eq!(first, second);
        assert_eq!(core.decode_bytes(&first).expect("all bytes decode"), all_bytes);
        assert!(core.decode(&first).is_err());
        assert_eq!(core.vocabulary_lookup(b"abc"), Some(257));
        assert_eq!(core.token_piece(257).expect("piece exists"), b"abc");
        assert_eq!(pool.used_bytes().expect("metrics work"), baseline);
    }

    #[test]
    fn pretoken_boundaries_bpe_and_no_implicit_bos_are_exact() {
        let (core, _) = core();
        let profile = SyntheticUnicode16;
        let tokens =
            core.encode("abcabc", SpecialTokenPolicy::Reject, &profile).expect("letters encode");
        assert_eq!(tokens, vec![257, 257]);
        assert_ne!(tokens.first().copied(), Some(258));
        assert_eq!(core.decode(&tokens).expect("tokens decode"), "abcabc");
    }

    #[test]
    fn special_and_ordinary_added_token_policies_are_distinct() {
        let (core, _) = core();
        let profile = SyntheticUnicode16;
        assert!(core.encode("<|im_start|>", SpecialTokenPolicy::Reject, &profile).is_err());
        assert_eq!(
            core.encode("<|im_start|>", SpecialTokenPolicy::AllowAll, &profile)
                .expect("special allow works"),
            vec![259]
        );
        assert_eq!(
            core.encode("<|im_start|>", SpecialTokenPolicy::AllowListed(&[259]), &profile)
                .expect("listed special works"),
            vec![259]
        );
        let ordinary_special = core
            .encode("<|im_start|>", SpecialTokenPolicy::TreatAsOrdinaryBytes, &profile)
            .expect("diagnostic ordinary path works");
        assert_ne!(ordinary_special, vec![259]);
        assert_eq!(core.decode(&ordinary_special).expect("ordinary bytes decode"), "<|im_start|>");

        assert_eq!(
            core.encode("<tool_call>", SpecialTokenPolicy::Reject, &profile)
                .expect("ordinary added token is automatic"),
            vec![261]
        );
        let bypass = core
            .encode("<tool_call>", SpecialTokenPolicy::TreatAsOrdinaryBytes, &profile)
            .expect("ordinary added token can be bypassed diagnostically");
        assert_ne!(bypass, vec![261]);
        assert_eq!(core.decode(&[259, 261]).expect("decode roles are exact"), "<tool_call>");
        assert!(
            core.encode("<|im_start|>", SpecialTokenPolicy::AllowListed(&[261]), &profile).is_err()
        );
        assert!(
            core.encode("<|im_start|>", SpecialTokenPolicy::AllowListed(&[259, 259]), &profile,)
                .is_err()
        );
        assert_eq!(
            core.encode("😀<|im_start|>", SpecialTokenPolicy::AllowAll, &profile)
                .expect("a special after multibyte UTF-8 remains boundary-safe")
                .last(),
            Some(&259)
        );
    }

    #[test]
    fn undefined_tail_limits_and_unicode_version_fail_closed() {
        let (core, _) = core();
        let profile = SyntheticUnicode16;
        assert!(core.token_piece(262).is_err());
        assert!(core.token_piece(269).is_err());
        assert!(core.decode_bytes(&[262]).is_err());
        assert!(core.encode(&"a".repeat(4097), SpecialTokenPolicy::Reject, &profile).is_err());
        let pool = MemoryPool::new(16 * 1024 * 1024).expect("pool validates");
        let limited =
            Qwen25RuntimeCore::load_test_fixture(&fixture(4096, 1), &pool).expect("tables load");
        assert!(limited.encode("def", SpecialTokenPolicy::Reject, &profile).is_err());

        assert!(matches!(
            core.encode("a", SpecialTokenPolicy::Reject, &WrongUnicodeVersion),
            Err(EngineError::UnsupportedTokenizer(_))
        ));
    }

    #[test]
    fn normalization_cannot_grow_beyond_reserved_output() {
        let pool = MemoryPool::new(16 * 1024 * 1024).expect("pool validates");
        let core =
            Qwen25RuntimeCore::load_test_fixture(&fixture(4, 4096), &pool).expect("tables load");
        let baseline = pool.used_bytes().expect("metrics work");
        assert!(matches!(
            core.encode("a", SpecialTokenPolicy::Reject, &ExpandingUnicode16),
            Err(EngineError::ResourceExhaustion(_))
        ));
        assert_eq!(pool.used_bytes().expect("metrics work"), baseline);
    }

    #[test]
    fn incremental_decode_matches_full_decode_and_rejects_incomplete_utf8() {
        let (core, _) = core();
        let profile = SyntheticUnicode16;
        let tokens =
            core.encode("😀", SpecialTokenPolicy::Reject, &profile).expect("emoji encodes");
        let expected = core.decode(&tokens).expect("full decode works");
        let mut decoder = Qwen25IncrementalDecoder::new(&core).expect("decoder allocates");
        let mut observed = String::new();
        for token in &tokens {
            observed.push_str(&decoder.push(*token).expect("stream token decodes"));
        }
        decoder.finish().expect("stream is complete");
        assert_eq!(observed, expected);

        let first_byte = core.tables.byte_token_ids[0xF0];
        let incomplete = Qwen25IncrementalDecoder::new(&core).expect("decoder allocates");
        let mut incomplete = incomplete;
        assert_eq!(incomplete.push(first_byte).expect("prefix buffers"), "");
        assert!(incomplete.finish().is_err());

        let pool = MemoryPool::new(16 * 1024 * 1024).expect("pool validates");
        let limited =
            Qwen25RuntimeCore::load_test_fixture(&fixture(4096, 1), &pool).expect("tables load");
        let mut limited_decoder =
            Qwen25IncrementalDecoder::new(&limited).expect("decoder allocates");
        assert_eq!(limited_decoder.push(259).expect("one special is in bounds"), "");
        assert!(limited_decoder.push(259).is_err());

        let mut validation_decoder =
            Qwen25IncrementalDecoder::new(&core).expect("decoder allocates");
        assert!(validation_decoder.push(262).is_err());
        let ascii_a = core.tables.byte_token_ids[usize::from(b'a')];
        assert_eq!(validation_decoder.push(ascii_a).expect("rejected ID changes no state"), "a");

        let mut malformed_decoder =
            Qwen25IncrementalDecoder::new(&core).expect("decoder allocates");
        let malformed = core.tables.byte_token_ids[usize::from(0xFF_u8)];
        assert!(malformed_decoder.push(malformed).is_err());
        assert!(malformed_decoder.push(ascii_a).is_err());
        assert!(malformed_decoder.finish().is_err());
    }

    #[test]
    fn shared_tables_are_thread_deterministic_and_temporary_memory_is_released() {
        let (core, pool) = core();
        let core = Arc::new(core);
        let baseline = pool.used_bytes().expect("metrics work");
        let threads = (0..8)
            .map(|_| {
                let core = Arc::clone(&core);
                std::thread::spawn(move || {
                    core.encode("Hello abc 123!\n", SpecialTokenPolicy::Reject, &SyntheticUnicode16)
                })
            })
            .collect::<Vec<_>>();
        let mut expected = None;
        for thread in threads {
            let tokens = thread.join().expect("thread joins").expect("encoding works");
            if let Some(expected) = &expected {
                assert_eq!(&tokens, expected);
            } else {
                expected = Some(tokens);
            }
        }
        assert_eq!(pool.used_bytes().expect("metrics work"), baseline);
    }
}
