//! Consumer pin for the canonical Sovereign-AI-Contracts PAIRSEV1 vectors.
//!
//! Source: InspiredDev17/Sovereign-AI-Contracts commit
//! `1b28edaebb6ff1ab1b84b7e5b8aa694d21476ec8`,
//! `fixtures/v1/pairsev1`. The engine workspace intentionally does not add a
//! runtime contract-library dependency; these exact bytes make drift fail in
//! the producer repository.

use personalai_engine_release::SupervisorSecurityEvent;

const CANONICAL_CONTRACT_COMMIT: &str = "1b28edaebb6ff1ab1b84b7e5b8aa694d21476ec8";

const ACCEPT: &[(&str, &str)] = &[
    (
        "valid-release-identity.hex",
        "50414952534556310100010001000000000000000700000000000000111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111",
    ),
    (
        "valid-unknown-release.hex",
        "50414952534556310100070002000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
    ),
    (
        "valid-maximum-sequence.hex",
        "504149525345563101000500ffffffffffffffff0900000000000000222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222",
    ),
];

const REJECT: &[(&str, &str)] = &[
    (
        "bad-magic.hex",
        "00414952534556310100010001000000000000000700000000000000111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111",
    ),
    (
        "unknown-version.hex",
        "50414952534556310200010001000000000000000700000000000000111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111",
    ),
    (
        "unknown-kind.hex",
        "50414952534556310100080001000000000000000700000000000000111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111",
    ),
    (
        "nonzero-flags.hex",
        "50414952534556310100010101000000000000000700000000000000111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111",
    ),
    (
        "zero-sequence.hex",
        "50414952534556310100010000000000000000000700000000000000111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111",
    ),
    (
        "partial-identity-zero-generation.hex",
        "50414952534556310100010001000000000000000000000000000000111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111",
    ),
    (
        "partial-identity-zero-digest.hex",
        "50414952534556310100010001000000000000000700000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
    ),
    (
        "truncated.hex",
        "504149525345563101000100010000000000000007000000000000001111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111",
    ),
    (
        "trailing.hex",
        "5041495253455631010001000100000000000000070000000000000011111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111100",
    ),
];

#[test]
fn canonical_pairsev1_vectors_are_exactly_accepted_or_rejected() {
    assert_eq!(CANONICAL_CONTRACT_COMMIT.len(), 40);
    for (name, source) in ACCEPT {
        let bytes = decode_lower_hex(source).expect("canonical accepted hex is valid");
        let event = SupervisorSecurityEvent::parse(&bytes)
            .unwrap_or_else(|error| panic!("{name} should parse: {error}"));
        assert_eq!(event.encode().as_slice(), bytes, "{name} must be byte stable");
    }
    for (name, source) in REJECT {
        let bytes = decode_lower_hex(source).expect("canonical rejected hex is valid");
        assert!(SupervisorSecurityEvent::parse(&bytes).is_err(), "{name} must be rejected");
    }
}

fn decode_lower_hex(source: &str) -> Result<Vec<u8>, &'static str> {
    if source.is_empty()
        || source.len() % 2 != 0
        || !source.bytes().all(|byte| byte.is_ascii_digit() || matches!(byte, b'a'..=b'f'))
    {
        return Err("malformed lowercase hexadecimal");
    }
    source
        .as_bytes()
        .chunks_exact(2)
        .map(|pair| Ok((nibble(pair[0])? << 4) | nibble(pair[1])?))
        .collect()
}

const fn nibble(byte: u8) -> Result<u8, &'static str> {
    match byte {
        b'0'..=b'9' => Ok(byte - b'0'),
        b'a'..=b'f' => Ok(byte - b'a' + 10),
        _ => Err("invalid lowercase hexadecimal"),
    }
}
