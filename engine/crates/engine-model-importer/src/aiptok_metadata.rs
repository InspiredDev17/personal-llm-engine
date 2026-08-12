//! Strict, non-JSON tokenizer metadata conversion for the offline importer.

use personalai_engine_tokenizer::{
    AipTokDefinition, AipTokMerge, AipTokVocabularyEntry, MAX_BPE_MERGE_RULES, MAX_SPECIAL_TOKENS,
    MAX_TOKEN_PIECE_BYTES, MAX_VOCABULARY_SIZE, SpecialTokenRole,
};
use personalai_engine_types::{EngineError, MAX_REQUEST_BYTES, MAX_TOKEN_COUNT};

/// Exact explicit source-format selector accepted by the importer CLI.
pub const AIPTOK_METADATA_SOURCE_FORMAT: &str = "aiptok-metadata-v1";
const MAGIC: &str = "AIPTOK-METADATA-V1";

/// Converts one deliberately narrow local metadata document into canonical AIPTok.
pub fn convert(bytes: &[u8]) -> Result<Vec<u8>, EngineError> {
    let text = metadata_text(bytes)?;
    let mut lines = text.lines();
    convert_lines(&mut lines)
}

fn metadata_text(bytes: &[u8]) -> Result<&str, EngineError> {
    if bytes.is_empty()
        || bytes.len() > 64 * 1024 * 1024
        || !bytes.ends_with(b"\n")
        || bytes.iter().any(|byte| *byte == b'\r' || (*byte < 0x20 && *byte != b'\n'))
    {
        return Err(corrupt("tokenizer source metadata bytes are not canonical"));
    }
    let text = std::str::from_utf8(bytes)
        .map_err(|_| corrupt("tokenizer source metadata is invalid UTF-8"))?;
    if !text.is_ascii() {
        return Err(corrupt("tokenizer source metadata must be canonical ASCII"));
    }
    Ok(text)
}

fn convert_lines<'a>(lines: &mut impl Iterator<Item = &'a str>) -> Result<Vec<u8>, EngineError> {
    expect_line(lines, MAGIC)?;
    expect_line(lines, "family=byte-level-bpe-v1")?;
    expect_line(lines, "normalizer=preserve-bytes")?;
    expect_line(lines, "pre_tokenizer=none")?;
    expect_line(lines, "decoder=concatenate-bytes-skip-special")?;
    expect_line(lines, "unknown_policy=byte-fallback")?;
    expect_line(lines, "chat_template=none")?;
    let vocabulary_size = value_u32(next_line(lines)?, "vocabulary_size=")?;
    let bos = value_u32(next_line(lines)?, "bos_token_id=")?;
    let eos = value_u32(next_line(lines)?, "eos_token_id=")?;
    let unknown = value_u32(next_line(lines)?, "unknown_token_id=")?;
    let maximum_input_bytes = value_u32(next_line(lines)?, "maximum_input_bytes=")?;
    let maximum_produced_tokens = value_u32(next_line(lines)?, "maximum_produced_tokens=")?;
    if !(259..=MAX_VOCABULARY_SIZE).contains(&vocabulary_size)
        || maximum_input_bytes == 0
        || usize::try_from(maximum_input_bytes).unwrap_or(usize::MAX) > MAX_REQUEST_BYTES
        || maximum_produced_tokens == 0
        || maximum_produced_tokens > MAX_TOKEN_COUNT
    {
        return Err(corrupt("tokenizer source metadata limits are invalid"));
    }

    let mut vocabulary = Vec::new();
    vocabulary
        .try_reserve_exact(usize::try_from(vocabulary_size).map_err(|_| {
            EngineError::resource_exhaustion("source vocabulary is not addressable")
        })?)
        .map_err(|_| EngineError::resource_exhaustion("source vocabulary allocation failed"))?;
    let mut special_count = 0_u32;
    for expected_id in 0..vocabulary_size {
        let line = next_line(lines)?;
        let mut fields = line.split(' ');
        if fields.next() != Some("token") {
            return Err(corrupt("tokenizer source vocabulary record is missing"));
        }
        let token_id = parse_u32(fields.next())?;
        let role = parse_role(fields.next())?;
        let token_bytes = parse_hex(fields.next())?;
        if fields.next().is_some() || token_id != expected_id || token_bytes.is_empty() {
            return Err(corrupt("tokenizer source vocabulary record is not canonical"));
        }
        if token_bytes.len() > usize::from(MAX_TOKEN_PIECE_BYTES) {
            return Err(corrupt("tokenizer source token is too large"));
        }
        if role.is_some() {
            special_count = special_count
                .checked_add(1)
                .ok_or_else(|| corrupt("source special-token count overflows"))?;
        }
        vocabulary.push(AipTokVocabularyEntry { token_id, bytes: token_bytes, special_role: role });
    }
    if !(3..=MAX_SPECIAL_TOKENS).contains(&special_count) {
        return Err(corrupt("source special-token count is outside bounds"));
    }

    let merge_count = value_u32(next_line(lines)?, "merge_count=")?;
    if merge_count > MAX_BPE_MERGE_RULES {
        return Err(corrupt("tokenizer source merge count exceeds its hard limit"));
    }
    let mut merges = Vec::new();
    merges
        .try_reserve_exact(usize::try_from(merge_count).map_err(|_| {
            EngineError::resource_exhaustion("source merge count is not addressable")
        })?)
        .map_err(|_| EngineError::resource_exhaustion("source merge allocation failed"))?;
    for expected_rank in 0..merge_count {
        let mut fields = next_line(lines)?.split(' ');
        if fields.next() != Some("merge") {
            return Err(corrupt("tokenizer source merge record is missing"));
        }
        let rank = parse_u32(fields.next())?;
        let left = parse_u32(fields.next())?;
        let right = parse_u32(fields.next())?;
        let result = parse_u32(fields.next())?;
        if fields.next().is_some() || rank != expected_rank {
            return Err(corrupt("tokenizer source merge rank is not canonical"));
        }
        merges.push(AipTokMerge { rank, left, right, result });
    }
    if lines.next().is_some() {
        return Err(corrupt(
            "tokenizer source contains unsupported fields, components, or trailing data",
        ));
    }
    AipTokDefinition {
        vocabulary,
        merges,
        beginning_of_sequence_token: bos,
        end_of_sequence_token: eos,
        unknown_token: unknown,
        maximum_input_bytes,
        maximum_produced_tokens,
    }
    .encode()
}

fn expect_line<'a>(
    lines: &mut impl Iterator<Item = &'a str>,
    expected: &str,
) -> Result<(), EngineError> {
    if next_line(lines)? == expected {
        Ok(())
    } else {
        Err(EngineError::unsupported_tokenizer("tokenizer source family or behavior component"))
    }
}

fn next_line<'a>(lines: &mut impl Iterator<Item = &'a str>) -> Result<&'a str, EngineError> {
    lines.next().ok_or_else(|| corrupt("tokenizer source metadata is truncated"))
}

fn value_u32(line: &str, prefix: &str) -> Result<u32, EngineError> {
    line.strip_prefix(prefix)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| corrupt("tokenizer source required field is invalid"))?
        .parse()
        .map_err(|_| corrupt("tokenizer source integer is invalid"))
}

fn parse_u32(value: Option<&str>) -> Result<u32, EngineError> {
    value
        .ok_or_else(|| corrupt("tokenizer source integer is missing"))?
        .parse()
        .map_err(|_| corrupt("tokenizer source integer is invalid"))
}

fn parse_role(value: Option<&str>) -> Result<Option<SpecialTokenRole>, EngineError> {
    match value {
        Some("ordinary") => Ok(None),
        Some("bos") => Ok(Some(SpecialTokenRole::BeginningOfSequence)),
        Some("eos") => Ok(Some(SpecialTokenRole::EndOfSequence)),
        Some("unknown") => Ok(Some(SpecialTokenRole::Unknown)),
        Some("special") => Ok(Some(SpecialTokenRole::Other)),
        _ => Err(corrupt("tokenizer source token role is unsupported")),
    }
}

fn parse_hex(value: Option<&str>) -> Result<Vec<u8>, EngineError> {
    let value = value.ok_or_else(|| corrupt("tokenizer source token bytes are missing"))?;
    if value.is_empty()
        || value.len() % 2 != 0
        || value.len() > usize::from(MAX_TOKEN_PIECE_BYTES) * 2
    {
        return Err(corrupt("tokenizer source token hex length is invalid"));
    }
    let mut output = Vec::new();
    output
        .try_reserve_exact(value.len() / 2)
        .map_err(|_| EngineError::resource_exhaustion("token hex allocation failed"))?;
    for pair in value.as_bytes().chunks_exact(2) {
        output.push((nibble(pair[0])? << 4) | nibble(pair[1])?);
    }
    Ok(output)
}

fn nibble(value: u8) -> Result<u8, EngineError> {
    match value {
        b'0'..=b'9' => Ok(value - b'0'),
        b'a'..=b'f' => Ok(value - b'a' + 10),
        _ => Err(corrupt("tokenizer source token hex is not canonical lowercase")),
    }
}

fn corrupt(detail: impl AsRef<str>) -> EngineError {
    EngineError::corrupt_model_package(detail)
}

#[cfg(test)]
mod tests {
    use super::*;
    use personalai_engine_memory::MemoryPool;
    use personalai_engine_tokenizer::{EncodeOptions, FirstPartyBpeTokenizer, Tokenizer};

    fn metadata() -> Vec<u8> {
        let mut text = concat!(
            "AIPTOK-METADATA-V1\n",
            "family=byte-level-bpe-v1\n",
            "normalizer=preserve-bytes\n",
            "pre_tokenizer=none\n",
            "decoder=concatenate-bytes-skip-special\n",
            "unknown_policy=byte-fallback\n",
            "chat_template=none\n",
            "vocabulary_size=259\n",
            "bos_token_id=256\n",
            "eos_token_id=257\n",
            "unknown_token_id=258\n",
            "maximum_input_bytes=4194304\n",
            "maximum_produced_tokens=1048576\n",
        )
        .to_owned();
        for token in 0_u16..=255 {
            use std::fmt::Write;
            writeln!(&mut text, "token {token} ordinary {token:02x}")
                .expect("metadata write must work");
        }
        text.push_str(
            "token 256 bos 3c7c626f737c3e\n\
             token 257 eos 3c7c656f737c3e\n\
             token 258 unknown 3c7c756e6b7c3e\n\
             merge_count=0\n",
        );
        text.into_bytes()
    }

    #[test]
    fn strict_metadata_converts_deterministically_and_round_trips_bytes() {
        let source = metadata();
        let first = convert(&source).expect("metadata must convert");
        let second = convert(&source).expect("metadata must convert twice");
        assert_eq!(first, second);
        let pool = MemoryPool::new(128 * 1024 * 1024).expect("pool must validate");
        let tokenizer = FirstPartyBpeTokenizer::parse(&first, &pool).expect("AIPTok must parse");
        let input = [0, 0xff, b'a'];
        let tokens = tokenizer
            .encode_bytes_with_options(&input, EncodeOptions::byte_round_trip())
            .expect("bytes must encode");
        assert_eq!(tokenizer.decode_bytes(&tokens).expect("bytes must decode"), input);
    }

    #[test]
    fn unsupported_components_templates_families_and_lossy_metadata_are_rejected() {
        let valid = String::from_utf8(metadata()).expect("fixture must be UTF-8");
        for (needle, replacement) in [
            ("family=byte-level-bpe-v1", "family=wordpiece"),
            ("normalizer=preserve-bytes", "normalizer=nfkc"),
            ("pre_tokenizer=none", "pre_tokenizer=byte-level-regex"),
            ("chat_template=none", "chat_template={{ messages }}"),
            ("decoder=concatenate-bytes-skip-special", "decoder=plugin"),
        ] {
            assert!(convert(valid.replacen(needle, replacement, 1).as_bytes()).is_err());
        }
        let mut trailing = valid;
        trailing.push_str("script=tokenizer.py\n");
        assert!(convert(trailing.as_bytes()).is_err());
    }
}
