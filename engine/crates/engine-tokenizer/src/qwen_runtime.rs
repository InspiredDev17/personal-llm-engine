//! Inert Qwen2.5 tokenizer runtime core behind a trusted Unicode-16 boundary.
//!
//! This module is intentionally private. It implements deterministic fixed
//! pre-tokenizer, BPE, added-token, and decoding mechanics over already parsed
//! AIPTok v2 tables. The exact runtime is constructible only through the
//! package-hash-bound tokenizer API. The ordinary service factory remains
//! fail-closed pending independent qualification and explicit owner activation.

use super::qwen_v2::{
    AipTokQwen25AddedRole, Qwen25LoadedTables, Qwen25MergeRule, load_qwen25_tables,
};
use super::{MAX_SPECIAL_TOKENS, SpecialTokenPolicy, TEMPORARY_BYTES_PER_INPUT_TOKEN, checked_add};
#[cfg(feature = "qwen25-conformance-diagnostic")]
use personalai_engine_crypto::hash_package;
use personalai_engine_memory::{MemoryPool, MemoryReservation};
use personalai_engine_types::{EngineError, MAX_REQUEST_BYTES, MAX_TOKEN_COUNT};
use std::cmp::Reverse;
use std::collections::{BTreeSet, BinaryHeap};
use std::mem::size_of;
use std::ops::Range;
use unicode_general_category::{GeneralCategory, get_general_category};
use unicode_normalization::UnicodeNormalization;

const REQUIRED_UNICODE_VERSION: (u8, u8, u8) = (16, 0, 0);
const SCANNER_BYTES_PER_SCALAR: u64 = 64;
const PENDING_UTF8_BYTES: u64 = 4;
// UAX15-D4 counts compatibility-decomposed nonstarters. The adapter uses its
// allocation-free stream-safe iterator only as a reject-only precheck: it
// never forwards an inserted CGJ or transforms input to satisfy the bound.
const MAX_NFC_COMPATIBILITY_NONSTARTERS: usize = 30;
// Unicode 16's longest recursive canonical decomposition is four scalars. The
// locally audited normalization patch therefore keeps 64 decomposition entries
// and 32 recomposition entries inline: 30 pending nonstarters plus one complete
// four-scalar source decomposition cannot spill. It also replaces stable sort
// with in-place insertion sort. This fixed charge conservatively accounts for
// both iterator buffers and their bounded stack workspace while the separate
// output reservation accounts for the normalized String.
const MAX_NFC_CANONICAL_DECOMPOSITION_SCALARS: usize = 4;
const NFC_DECOMPOSITION_INLINE_SCALARS: usize = 64;
const NFC_RECOMPOSITION_INLINE_SCALARS: usize = 32;
const NFC_ITERATOR_AND_SORT_WORKSPACE_BYTES: u64 = 4 * 1024;
const CONTRACTION_SUFFIXES: [&[u8]; 7] = [b"s", b"t", b"re", b"ve", b"m", b"ll", b"d"];
#[cfg(feature = "qwen25-conformance-diagnostic")]
const PINNED_QWEN25_AIPTOK_SHA384: &str = "sha384:f1b418104aedb90abc6ee892ef5b71db2b02c2c0c82a9f8e162487d933ffdd636b9ae077eed806f011c251cfea5187cc";
#[cfg(feature = "qwen25-conformance-diagnostic")]
const PINNED_QWEN25_TABLE_SHA384: &str = "sha384:2c343d7ddc0952d346fb215fe2d2c8723deafffe7412ff19a252ae33a0eefd35dca3d2f41bd6a354ad5beb0755a54baa";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GeneralClass {
    Letter,
    Number,
    Other,
}

/// Engine-internal Unicode authority. No package or client can implement or
/// select this interface.
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

/// Fixed engine-owned adapter over the separately admitted Unicode 16.0
/// normalization and General Category tables.
///
/// The type remains private with the rest of the Qwen runtime core. It cannot
/// be selected by a package or client and its existence does not make
/// `AIPTOK02` publicly executable.
#[derive(Debug, Clone, Copy)]
struct Qwen25Unicode16Adapter;

impl Qwen25Unicode16Adapter {
    fn versions_match() -> bool {
        let required_category_version = (
            u64::from(REQUIRED_UNICODE_VERSION.0),
            u64::from(REQUIRED_UNICODE_VERSION.1),
            u64::from(REQUIRED_UNICODE_VERSION.2),
        );
        unicode_normalization::UNICODE_VERSION == REQUIRED_UNICODE_VERSION
            && unicode_general_category::UNICODE_VERSION == required_category_version
    }
}

impl TrustedQwen25Unicode16 for Qwen25Unicode16Adapter {
    fn unicode_version(&self) -> (u8, u8, u8) {
        if Self::versions_match() {
            REQUIRED_UNICODE_VERSION
        } else {
            // A future dependency-data drift must fail the existing version
            // check rather than silently changing token boundaries.
            (0, 0, 0)
        }
    }

    fn normalize_nfc(
        &self,
        input: &str,
        output: &mut BoundedNormalizedText,
    ) -> Result<(), EngineError> {
        validate_nfc_nonstarter_bound(input)?;
        for scalar in input.nfc() {
            let mut encoded = [0_u8; 4];
            output.push_str(scalar.encode_utf8(&mut encoded))?;
        }
        Ok(())
    }

    fn general_class(&self, scalar: char) -> GeneralClass {
        match get_general_category(scalar) {
            GeneralCategory::LowercaseLetter
            | GeneralCategory::ModifierLetter
            | GeneralCategory::OtherLetter
            | GeneralCategory::TitlecaseLetter
            | GeneralCategory::UppercaseLetter => GeneralClass::Letter,
            GeneralCategory::DecimalNumber
            | GeneralCategory::LetterNumber
            | GeneralCategory::OtherNumber => GeneralClass::Number,
            _ => GeneralClass::Other,
        }
    }

    fn contraction_ascii_fold(&self, scalar: char) -> Option<u8> {
        match scalar {
            'A'..='Z' => Some((scalar as u8) + (b'a' - b'A')),
            'a'..='z' => Some(scalar as u8),
            // Unicode 16.0 default simple case-fold equivalences that reach an
            // ASCII scalar. LONG S matters to the pinned contraction suffix;
            // KELVIN SIGN is retained so this finite mapping is complete.
            '\u{017F}' => Some(b's'),
            '\u{212A}' => Some(b'k'),
            _ => None,
        }
    }
}

fn validate_nfc_nonstarter_bound(input: &str) -> Result<(), EngineError> {
    if input.chars().stream_safe().eq(input.chars()) {
        Ok(())
    } else {
        Err(EngineError::resource_exhaustion(
            "Qwen NFC compatibility-decomposed nonstarter sequence exceeds the fixed limit",
        ))
    }
}

fn normalization_reservation_bytes(output_bytes: usize) -> Result<u64, EngineError> {
    let output_bytes = u64::try_from(output_bytes).map_err(|_| {
        EngineError::resource_exhaustion("Qwen normalization output size is not representable")
    })?;
    output_bytes
        .checked_add(NFC_ITERATOR_AND_SORT_WORKSPACE_BYTES)
        .ok_or_else(|| EngineError::resource_exhaustion("Qwen normalization reservation overflows"))
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
pub(super) struct Qwen25RuntimeCore {
    tables: Qwen25LoadedTables,
    added_by_first_byte: [Vec<usize>; 256],
    _added_index_reservation: Option<MemoryReservation>,
}

impl Qwen25RuntimeCore {
    pub(super) fn load(data: &[u8], pool: &MemoryPool) -> Result<Self, EngineError> {
        let tables = load_qwen25_tables(data, pool)?;
        if !tables.descriptor.matches_pinned_qwen25_1_5b_counts() {
            return Err(EngineError::unsupported_tokenizer(
                "Qwen runtime construction requires the pinned production cardinality profile",
            ));
        }
        Self::from_tables(tables)
    }

    #[cfg(test)]
    pub(super) fn load_test_fixture(data: &[u8], pool: &MemoryPool) -> Result<Self, EngineError> {
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

    pub(super) fn descriptor(&self) -> super::qwen_v2::Qwen25AipTokDescriptor {
        self.tables.descriptor
    }

    pub(super) fn maximum_piece_bytes(&self) -> Result<u16, EngineError> {
        self.tables
            .pieces
            .iter()
            .map(|piece| piece.len())
            .max()
            .and_then(|length| u16::try_from(length).ok())
            .ok_or_else(|| {
                EngineError::internal_invariant("Qwen maximum token piece is unavailable")
            })
    }

    pub(super) fn token_piece(&self, token: u32) -> Result<&[u8], EngineError> {
        self.tables
            .pieces
            .get(self.token_index(token)?)
            .map(Box::as_ref)
            .ok_or_else(|| EngineError::internal_invariant("Qwen token piece is unavailable"))
    }

    pub(super) fn vocabulary_lookup(&self, piece: &[u8]) -> Option<u32> {
        self.tables.vocabulary.get(piece).copied()
    }

    pub(super) fn is_special_token(&self, token: u32) -> Result<bool, EngineError> {
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
        let _normalization_reservation =
            self.tables.pool.reserve(normalization_reservation_bytes(normalized_limit)?)?;
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

    pub(super) fn encode_reviewed(
        &self,
        input: &str,
        policy: SpecialTokenPolicy<'_>,
    ) -> Result<Vec<u32>, EngineError> {
        self.encode(input, policy, &Qwen25Unicode16Adapter)
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

    pub(super) fn decode_bytes(&self, tokens: &[u32]) -> Result<Vec<u8>, EngineError> {
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

    pub(super) fn decode(&self, tokens: &[u32]) -> Result<String, EngineError> {
        String::from_utf8(self.decode_bytes(tokens)?)
            .map_err(|_| EngineError::invalid_input("Qwen token sequence is not valid UTF-8"))
    }
}

/// Replays the fixed, independently derived production token-ID corpus without
/// exposing a constructible tokenizer or accepting caller-selected inputs.
#[cfg(feature = "qwen25-conformance-diagnostic")]
pub(super) fn verify_pinned_qwen25_production_vectors(
    data: &[u8],
    pool: &MemoryPool,
) -> Result<u32, EngineError> {
    if hash_package(data).to_string() != PINNED_QWEN25_AIPTOK_SHA384 {
        return Err(EngineError::corrupt_model_package(
            "Qwen2.5 AIPTok conversion identity is not pinned",
        ));
    }
    let core = Qwen25RuntimeCore::load(data, pool)?;
    if core.tables.descriptor.content_hash().to_string() != PINNED_QWEN25_TABLE_SHA384 {
        return Err(EngineError::corrupt_model_package(
            "Qwen2.5 tokenizer table identity is not pinned",
        ));
    }
    let profile = Qwen25Unicode16Adapter;
    let vectors: [(&str, &[u32], &str); 17] = [
        ("Hello world", &[9707, 1879], "Hello world"),
        ("I'm WE'RE he'd", &[40, 2776, 19677, 94153, 566, 4172], "I'm WE'RE he'd"),
        ("abc123Ⅷ½", &[13683, 16, 17, 18, 70467, 100, 26062], "abc123Ⅷ½"),
        ("Hello, world!!!\n", &[9707, 11, 1879, 79295], "Hello, world!!!\n"),
        ("A\r\nB\n\n", &[32, 319, 33, 271], "A\r\nB\n\n"),
        ("A  B ", &[32, 220, 425, 220], "A  B "),
        ("\t\r\n", &[4474], "\t\r\n"),
        ("Cafe\u{0301}", &[34, 2577, 963], "Café"),
        ("\u{1100}\u{1161}", &[19969], "가"),
        ("\u{0301}b", &[53839, 65], "\u{0301}b"),
        ("\0A", &[188, 32], "\0A"),
        ("A中Ж", &[32, 15946, 132_042], "A中Ж"),
        (" !?", &[753, 30], " !?"),
        (" 😀\n", &[90316, 198], " 😀\n"),
        ("\u{00A0}!", &[4102, 0], "\u{00A0}!"),
        ("\u{200B}!", &[15692, 0], "\u{200B}!"),
        (
            "a\u{1ACF}\u{0315}\u{0300}\u{05AE}b",
            &[64, 157, 104, 237, 146, 106, 96549, 136, 243, 65],
            "a\u{1ACF}\u{05AE}\u{0300}\u{0315}b",
        ),
    ];
    for (input, expected, decoded) in vectors {
        let observed = core.encode(input, SpecialTokenPolicy::Reject, &profile)?;
        if observed != expected {
            return Err(EngineError::internal_invariant(
                "pinned Qwen2.5 production token-ID vector changed",
            ));
        }
        verify_pinned_qwen25_decode(&core, &observed, decoded)?;
    }

    if core.encode("<|im_start|>", SpecialTokenPolicy::Reject, &profile).is_ok() {
        return Err(EngineError::internal_invariant(
            "pinned Qwen2.5 special-token rejection changed",
        ));
    }
    for (input, policy, expected, decoded) in [
        ("Hello", SpecialTokenPolicy::Reject, &[9707_u32][..], "Hello"),
        ("<|im_start|>", SpecialTokenPolicy::AllowAll, &[151_644][..], ""),
        (
            "<|im_start|>",
            SpecialTokenPolicy::TreatAsOrdinaryBytes,
            &[27, 91, 318, 4906, 91, 29][..],
            "<|im_start|>",
        ),
        ("<tool_call>", SpecialTokenPolicy::Reject, &[151_657][..], "<tool_call>"),
        (
            "<tool_call>",
            SpecialTokenPolicy::TreatAsOrdinaryBytes,
            &[27, 14172, 13429, 29][..],
            "<tool_call>",
        ),
    ] {
        let observed = core.encode(input, policy, &profile)?;
        if observed != expected {
            return Err(EngineError::internal_invariant(
                "pinned Qwen2.5 added-token vector changed",
            ));
        }
        verify_pinned_qwen25_decode(&core, &observed, decoded)?;
    }
    for token in
        core.tables.descriptor.defined_token_count()..core.tables.descriptor.model_vocabulary_size()
    {
        if core.decode_bytes(&[token]).is_ok() {
            return Err(EngineError::internal_invariant(
                "pinned Qwen2.5 undefined tokenizer tail became decodable",
            ));
        }
    }
    u32::try_from(vectors.len() + 6)
        .map_err(|_| EngineError::internal_invariant("Qwen2.5 vector count overflows"))
}

#[cfg(feature = "qwen25-conformance-diagnostic")]
fn verify_pinned_qwen25_decode(
    core: &Qwen25RuntimeCore,
    tokens: &[u32],
    expected: &str,
) -> Result<(), EngineError> {
    if core.decode(tokens)? != expected {
        return Err(EngineError::internal_invariant(
            "pinned Qwen2.5 production decode vector changed",
        ));
    }
    let mut decoder = Qwen25IncrementalDecoder::new(core)?;
    let mut observed = String::new();
    observed.try_reserve_exact(expected.len()).map_err(|_| {
        EngineError::resource_exhaustion("Qwen conformance output allocation failed")
    })?;
    for token in tokens {
        observed.push_str(&decoder.push(*token)?);
    }
    decoder.finish()?;
    if observed != expected {
        return Err(EngineError::internal_invariant(
            "pinned Qwen2.5 incremental decode vector changed",
        ));
    }
    Ok(())
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
    use std::panic::{AssertUnwindSafe, catch_unwind};
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

    fn unicode16_chunks(input: &str) -> Vec<String> {
        let pool = MemoryPool::new(1 << 20).expect("pool validates");
        let profile = Qwen25Unicode16Adapter;
        let mut normalized = BoundedNormalizedText::new(input.len().saturating_mul(4).max(1))
            .expect("Unicode-16 normalization buffer validates");
        profile.normalize_nfc(input, &mut normalized).expect("Unicode-16 NFC works");
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
    fn admitted_unicode_tables_are_exactly_version_16() {
        assert_eq!(unicode_normalization::UNICODE_VERSION, REQUIRED_UNICODE_VERSION);
        assert_eq!(unicode_general_category::UNICODE_VERSION, (16, 0, 0));
        assert!(Qwen25Unicode16Adapter::versions_match());
        assert_eq!(Qwen25Unicode16Adapter.unicode_version(), REQUIRED_UNICODE_VERSION);
    }

    #[test]
    fn fixed_normalization_charge_covers_the_pinned_bounded_iterators() {
        assert_eq!(MAX_NFC_COMPATIBILITY_NONSTARTERS, 30);
        let mut observed_maximum_decomposition = 0_usize;
        for value in 0_u32..=0x10_FFFF {
            let Some(scalar) = char::from_u32(value) else {
                continue;
            };
            let mut decomposed_scalars = 0_usize;
            unicode_normalization::char::decompose_canonical(scalar, |_| {
                decomposed_scalars += 1;
            });
            observed_maximum_decomposition = observed_maximum_decomposition.max(decomposed_scalars);
        }
        assert_eq!(observed_maximum_decomposition, MAX_NFC_CANONICAL_DECOMPOSITION_SCALARS);
        assert!(
            NFC_DECOMPOSITION_INLINE_SCALARS
                >= MAX_NFC_COMPATIBILITY_NONSTARTERS + observed_maximum_decomposition
        );

        let tuple_buffer_bytes = NFC_DECOMPOSITION_INLINE_SCALARS
            .checked_mul(size_of::<(u8, char)>())
            .expect("test bound fits");
        let scalar_buffer_bytes = NFC_RECOMPOSITION_INLINE_SCALARS
            .checked_mul(size_of::<char>())
            .expect("test bound fits");
        let conservative_total = tuple_buffer_bytes
            .checked_add(scalar_buffer_bytes)
            .and_then(|bytes| bytes.checked_mul(4))
            .expect("test bound fits");
        assert!(
            usize::try_from(NFC_ITERATOR_AND_SORT_WORKSPACE_BYTES)
                .is_ok_and(|reserved| reserved >= conservative_total)
        );
        assert_eq!(
            normalization_reservation_bytes(4096).expect("reservation fits"),
            4096 + NFC_ITERATOR_AND_SORT_WORKSPACE_BYTES
        );
        if usize::BITS == 64 {
            assert!(normalization_reservation_bytes(usize::MAX).is_err());
        }
    }

    #[test]
    fn stream_safe_precheck_accepts_thirty_and_rejects_thirty_one_without_mutation() {
        let profile = Qwen25Unicode16Adapter;
        let mark = "\u{0315}";
        let admitted = format!("a{}b{}", mark.repeat(30), mark.repeat(30));
        validate_nfc_nonstarter_bound(&admitted).expect("starter resets the compatibility run");
        let expected = admitted.nfc().collect::<String>();
        let mut output = BoundedNormalizedText::new(expected.len()).expect("output allocates");
        profile.normalize_nfc(&admitted, &mut output).expect("thirty nonstarters pass");
        assert_eq!(output.as_str(), expected);

        // U+1F82 has Unicode 16's maximum four-scalar recursive canonical
        // decomposition. It may arrive while all 30 admitted nonstarters are
        // still pending, so this vector exercises the complete 34-entry
        // decomposition segment that motivated the 64-entry inline buffer.
        let maximum_segment =
            format!("a{}\u{1F82}", mark.repeat(MAX_NFC_COMPATIBILITY_NONSTARTERS));
        validate_nfc_nonstarter_bound(&maximum_segment)
            .expect("the decomposed starter begins a new stream-safe run");
        let maximum_segment_expected = maximum_segment.nfc().collect::<String>();
        let mut maximum_segment_output = BoundedNormalizedText::new(maximum_segment_expected.len())
            .expect("maximum segment output allocates");
        profile
            .normalize_nfc(&maximum_segment, &mut maximum_segment_output)
            .expect("complete maximum decomposition stays inline");
        assert_eq!(maximum_segment_output.as_str(), maximum_segment_expected);

        let rejected = format!("a{}", mark.repeat(31));
        let mut untouched = BoundedNormalizedText::new(rejected.len()).expect("output allocates");
        assert!(matches!(
            profile.normalize_nfc(&rejected, &mut untouched),
            Err(EngineError::ResourceExhaustion(_))
        ));
        assert!(untouched.as_str().is_empty());

        // Each U+0344 decomposes to two leading nonstarters under the UAX15-D4
        // NFKD count, so the guard is not a raw input-scalar count.
        validate_nfc_nonstarter_bound(&"\u{0344}".repeat(15))
            .expect("thirty decomposed nonstarters pass");
        assert!(validate_nfc_nonstarter_bound(&"\u{0344}".repeat(16)).is_err());

        let with_explicit_cgj = format!(
            "{}\u{034F}{}",
            mark.repeat(MAX_NFC_COMPATIBILITY_NONSTARTERS),
            mark.repeat(MAX_NFC_COMPATIBILITY_NONSTARTERS)
        );
        validate_nfc_nonstarter_bound(&with_explicit_cgj)
            .expect("an explicit CGJ resets and remains caller data");
        let mut preserved =
            BoundedNormalizedText::new(with_explicit_cgj.len()).expect("CGJ output allocates");
        profile
            .normalize_nfc(&with_explicit_cgj, &mut preserved)
            .expect("explicit CGJ input normalizes");
        assert_eq!(preserved.as_str(), with_explicit_cgj);
    }

    #[test]
    fn unicode16_adapter_performs_bounded_nfc_without_intermediate_output() {
        let profile = Qwen25Unicode16Adapter;
        for (source, expected) in [
            ("Cafe\u{0301}", "Café"),
            ("\u{1100}\u{1161}", "가"),
            ("a\u{1ACF}\u{0315}\u{0300}\u{05AE}b", "a\u{1ACF}\u{05AE}\u{0300}\u{0315}b"),
        ] {
            let mut output = BoundedNormalizedText::new(expected.len())
                .expect("exact normalization bound allocates");
            profile.normalize_nfc(source, &mut output).expect("vector normalizes");
            assert_eq!(output.as_str(), expected);
        }

        // U+0344 has a two-scalar canonical decomposition that remains
        // decomposed in NFC. Its four output bytes cannot enter a two-byte
        // buffer, proving growth is rejected scalar-by-scalar.
        let mut bounded = BoundedNormalizedText::new("\u{0344}".len())
            .expect("input-sized normalization bound allocates");
        assert!(matches!(
            profile.normalize_nfc("\u{0344}", &mut bounded),
            Err(EngineError::ResourceExhaustion(_))
        ));
        assert!(bounded.as_str().len() <= "\u{0344}".len());
    }

    #[test]
    fn unicode16_adapter_maps_every_letter_and_number_general_category() {
        let profile = Qwen25Unicode16Adapter;
        for scalar in ['A', 'a', 'ǅ', 'ʰ', '中'] {
            assert_eq!(profile.general_class(scalar), GeneralClass::Letter, "{scalar:?}");
        }
        for scalar in ['1', '١', 'Ⅷ', '½'] {
            assert_eq!(profile.general_class(scalar), GeneralClass::Number, "{scalar:?}");
        }
        for scalar in ['\u{0301}', '_', ' ', '😀', '\0', '\u{E000}', '\u{088F}', '\u{11DE0}'] {
            assert_eq!(profile.general_class(scalar), GeneralClass::Other, "{scalar:?}");
        }

        assert_eq!(get_general_category('A'), GeneralCategory::UppercaseLetter);
        assert_eq!(get_general_category('a'), GeneralCategory::LowercaseLetter);
        assert_eq!(get_general_category('ǅ'), GeneralCategory::TitlecaseLetter);
        assert_eq!(get_general_category('ʰ'), GeneralCategory::ModifierLetter);
        assert_eq!(get_general_category('中'), GeneralCategory::OtherLetter);
        assert_eq!(get_general_category('1'), GeneralCategory::DecimalNumber);
        assert_eq!(get_general_category('Ⅷ'), GeneralCategory::LetterNumber);
        assert_eq!(get_general_category('½'), GeneralCategory::OtherNumber);
        assert_eq!(profile.general_class('\u{1C89}'), GeneralClass::Letter);

        for value in 0_u32..=0x10_FFFF {
            let Some(scalar) = char::from_u32(value) else {
                continue;
            };
            let expected = match get_general_category(scalar).abbreviation().as_bytes()[0] {
                b'L' => GeneralClass::Letter,
                b'N' => GeneralClass::Number,
                _ => GeneralClass::Other,
            };
            assert_eq!(profile.general_class(scalar), expected, "U+{value:04X}");
        }
    }

    #[test]
    fn contraction_fold_is_a_closed_unicode16_simple_fold_to_ascii() {
        let profile = Qwen25Unicode16Adapter;
        for (scalar, folded) in [
            ('S', b's'),
            ('s', b's'),
            ('T', b't'),
            ('R', b'r'),
            ('E', b'e'),
            ('V', b'v'),
            ('M', b'm'),
            ('L', b'l'),
            ('D', b'd'),
            ('\u{017F}', b's'),
            ('\u{212A}', b'k'),
        ] {
            assert_eq!(profile.contraction_ascii_fold(scalar), Some(folded), "{scalar:?}");
        }
        for scalar in ['ß', '\u{0130}', '\u{0131}', 'Σ', 'Ａ'] {
            assert_eq!(profile.contraction_ascii_fold(scalar), None, "{scalar:?}");
        }

        assert_eq!(
            unicode16_chunks("I'm WE'RE he'ſ"),
            strings(&["I", "'m", " WE", "'RE", " he", "'ſ"])
        );
        assert_eq!(unicode16_chunks("he'ß"), strings(&["he", "'ß"]));
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
    fn private_unicode16_adapter_drives_normalized_runtime_mechanics() {
        let (core, _) = core();
        let decomposed = "Cafe\u{0301}";
        let tokens = core
            .encode(decomposed, SpecialTokenPolicy::Reject, &Qwen25Unicode16Adapter)
            .expect("private Unicode-16 adapter encodes");
        assert_eq!(core.decode(&tokens).expect("normalized tokens decode"), "Café");
        assert_ne!(core.decode(&tokens).expect("decode is deterministic"), decomposed);
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
    fn real_unicode16_nfc_growth_cannot_exceed_the_memory_reservation() {
        let pool = MemoryPool::new(16 * 1024 * 1024).expect("pool validates");
        let core =
            Qwen25RuntimeCore::load_test_fixture(&fixture(2, 4096), &pool).expect("tables load");
        let baseline = pool.used_bytes().expect("metrics work");
        assert!(matches!(
            core.encode("\u{0344}", SpecialTokenPolicy::Reject, &Qwen25Unicode16Adapter),
            Err(EngineError::ResourceExhaustion(_))
        ));
        assert_eq!(pool.used_bytes().expect("metrics work"), baseline);
    }

    #[test]
    fn normalization_workspace_admission_fails_before_iteration_and_releases_accounting() {
        let data = fixture(4096, 4096);
        let probe_pool = MemoryPool::new(16 * 1024 * 1024).expect("probe pool validates");
        let _probe = Qwen25RuntimeCore::load_test_fixture(&data, &probe_pool).expect("tables load");
        let persistent = probe_pool.used_bytes().expect("metrics work");
        let normalization = normalization_reservation_bytes(4096).expect("bound fits");
        let capacity = persistent
            .checked_add(normalization)
            .and_then(|bytes| bytes.checked_sub(1))
            .expect("test capacity fits");
        let pool = MemoryPool::new(capacity).expect("limited pool validates");
        let core = Qwen25RuntimeCore::load_test_fixture(&data, &pool).expect("tables load");
        let baseline = pool.used_bytes().expect("metrics work");
        let result = catch_unwind(AssertUnwindSafe(|| {
            core.encode("a", SpecialTokenPolicy::Reject, &Qwen25Unicode16Adapter)
        }));
        assert!(result.is_ok(), "pool rejection must not panic");
        assert!(matches!(result.expect("no panic"), Err(EngineError::ResourceExhaustion(_))));
        assert_eq!(pool.used_bytes().expect("metrics work"), baseline);
    }

    #[test]
    fn hostile_long_nonstarter_input_fails_without_panic_or_pool_leak() {
        let pool = MemoryPool::new(64 * 1024 * 1024).expect("pool validates");
        let core = Qwen25RuntimeCore::load_test_fixture(&fixture(32 * 1024, 32 * 1024), &pool)
            .expect("tables load");
        let baseline = pool.used_bytes().expect("metrics work");
        let hostile = "\u{0315}".repeat(10_000);
        let result = catch_unwind(AssertUnwindSafe(|| {
            core.encode(&hostile, SpecialTokenPolicy::Reject, &Qwen25Unicode16Adapter)
        }));
        assert!(result.is_ok(), "hostile combining input must not panic");
        assert!(matches!(result.expect("no panic"), Err(EngineError::ResourceExhaustion(_))));
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
