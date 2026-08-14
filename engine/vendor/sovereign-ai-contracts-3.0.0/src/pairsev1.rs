//! Canonical fixed-width PAIRSEV1 supervisor release-event record.

use crate::error::{ContractError, Result};

/// Exact V1 magic.
pub const MAGIC: [u8; 8] = *b"PAIRSEV1";
/// Exact V1 record width.
pub const RECORD_LEN: usize = 76;
/// Exact supported version.
pub const VERSION: u16 = 1;

/// Closed content-free release-supervisor event vocabulary.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum Kind {
    /// Release artifact or signature integrity failed.
    ReleaseIntegrityFailure = 1,
    /// Release manifest or closed package layout was rejected.
    ReleaseManifestRejected = 2,
    /// Release signer trust failed.
    ReleaseTrustFailure = 3,
    /// A verifier resource ceiling was exceeded.
    ReleaseResourceExhaustion = 4,
    /// Monotonic rollback protection rejected the release.
    ReleaseRollbackAttempt = 5,
    /// Durable rollback state failed closed.
    ReleaseLedgerFailure = 6,
    /// The verifier encountered an internal fail-closed error.
    ReleaseVerifierFailure = 7,
}

impl TryFrom<u8> for Kind {
    type Error = ContractError;

    fn try_from(value: u8) -> Result<Self> {
        match value {
            1 => Ok(Self::ReleaseIntegrityFailure),
            2 => Ok(Self::ReleaseManifestRejected),
            3 => Ok(Self::ReleaseTrustFailure),
            4 => Ok(Self::ReleaseResourceExhaustion),
            5 => Ok(Self::ReleaseRollbackAttempt),
            6 => Ok(Self::ReleaseLedgerFailure),
            7 => Ok(Self::ReleaseVerifierFailure),
            _ => Err(ContractError::Frame("unknown PAIRSEV1 event kind")),
        }
    }
}

/// Exact content-free event. It grants no acknowledgement or authority.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Record {
    /// Closed event kind.
    pub kind: Kind,
    /// Nonzero monotonic producer identity.
    pub sequence: u64,
    /// Release generation, or zero when the release identity is unknown.
    pub release_generation: u64,
    /// Manifest SHA-384, or all zeroes when the release identity is unknown.
    pub manifest_sha384: [u8; 48],
}

impl Record {
    /// Constructs one validated record.
    ///
    /// # Errors
    ///
    /// Returns an error for zero sequence or a partial release identity.
    pub fn new(
        kind: Kind,
        sequence: u64,
        release_generation: u64,
        manifest_sha384: [u8; 48],
    ) -> Result<Self> {
        if sequence == 0 {
            return Err(ContractError::Frame("PAIRSEV1 sequence must be nonzero"));
        }
        let zero_digest = manifest_sha384.iter().all(|value| *value == 0);
        if (release_generation == 0) != zero_digest {
            return Err(ContractError::Frame(
                "PAIRSEV1 release identity must be entirely present or absent",
            ));
        }
        Ok(Self {
            kind,
            sequence,
            release_generation,
            manifest_sha384,
        })
    }

    /// Encodes the exact V1 little-endian record.
    #[must_use]
    pub fn encode(&self) -> [u8; RECORD_LEN] {
        let mut bytes = [0_u8; RECORD_LEN];
        bytes[..8].copy_from_slice(&MAGIC);
        bytes[8..10].copy_from_slice(&VERSION.to_le_bytes());
        bytes[10] = self.kind as u8;
        bytes[12..20].copy_from_slice(&self.sequence.to_le_bytes());
        bytes[20..28].copy_from_slice(&self.release_generation.to_le_bytes());
        bytes[28..].copy_from_slice(&self.manifest_sha384);
        bytes
    }

    /// Strictly decodes one complete V1 record.
    ///
    /// # Errors
    ///
    /// Returns an error for malformed, unsupported, truncated, or trailing data.
    pub fn decode(bytes: &[u8]) -> Result<Self> {
        if bytes.len() != RECORD_LEN {
            return Err(ContractError::Frame("PAIRSEV1 record length is invalid"));
        }
        if bytes[..8] != MAGIC {
            return Err(ContractError::Frame("PAIRSEV1 magic is invalid"));
        }
        if u16::from_le_bytes([bytes[8], bytes[9]]) != VERSION {
            return Err(ContractError::Frame("PAIRSEV1 version is unsupported"));
        }
        if bytes[11] != 0 {
            return Err(ContractError::Frame("PAIRSEV1 flags are unsupported"));
        }
        let kind = Kind::try_from(bytes[10])?;
        let sequence = u64::from_le_bytes(
            bytes[12..20]
                .try_into()
                .map_err(|_| ContractError::Frame("PAIRSEV1 sequence is truncated"))?,
        );
        let release_generation = u64::from_le_bytes(
            bytes[20..28]
                .try_into()
                .map_err(|_| ContractError::Frame("PAIRSEV1 generation is truncated"))?,
        );
        let mut manifest_sha384 = [0_u8; 48];
        manifest_sha384.copy_from_slice(&bytes[28..]);
        Self::new(kind, sequence, release_generation, manifest_sha384)
    }
}

#[cfg(test)]
mod tests {
    use super::{Record, RECORD_LEN};
    use serde::Deserialize;
    use std::fs;
    use std::path::PathBuf;

    #[derive(Deserialize)]
    struct Manifest {
        protocol: String,
        contract_version: String,
        encoding: String,
        record_bytes: usize,
        fixtures: Vec<Fixture>,
    }

    #[derive(Deserialize)]
    struct Fixture {
        path: String,
        expected: String,
        reason_code: Option<String>,
    }

    #[test]
    fn consumes_all_published_pairsev1_golden_fixtures() {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../fixtures/v1/pairsev1");
        let manifest: Manifest = serde_json::from_slice(
            &fs::read(root.join("manifest.json")).expect("read PAIRSEV1 manifest"),
        )
        .expect("decode PAIRSEV1 manifest");
        assert_eq!(manifest.protocol, "PAIRSEV1");
        assert_eq!(manifest.contract_version, "1.0.0");
        assert_eq!(manifest.encoding, "lowercase_hex");
        assert_eq!(manifest.record_bytes, RECORD_LEN);

        for fixture in manifest.fixtures {
            let source = fs::read_to_string(root.join(&fixture.path)).expect("read hex fixture");
            let bytes = decode_lower_hex(source.trim()).expect("fixture uses lowercase hex");
            let decoded = Record::decode(&bytes);
            match fixture.expected.as_str() {
                "accept" => {
                    let record = decoded
                        .unwrap_or_else(|error| panic!("{} should decode: {error}", fixture.path));
                    assert_eq!(
                        record.encode().as_slice(),
                        bytes,
                        "{} is byte stable",
                        fixture.path
                    );
                    assert!(fixture.reason_code.is_none());
                }
                "reject" => {
                    assert!(decoded.is_err(), "{} should be rejected", fixture.path);
                    assert!(fixture.reason_code.is_some());
                }
                other => panic!("unexpected fixture outcome: {other}"),
            }
        }
    }

    fn decode_lower_hex(source: &str) -> std::result::Result<Vec<u8>, &'static str> {
        if source.is_empty()
            || !source.len().is_multiple_of(2)
            || !source
                .bytes()
                .all(|byte| byte.is_ascii_digit() || matches!(byte, b'a'..=b'f'))
        {
            return Err("malformed lowercase hex");
        }
        source
            .as_bytes()
            .chunks_exact(2)
            .map(|pair| {
                let high = nibble(pair[0])?;
                let low = nibble(pair[1])?;
                Ok((high << 4) | low)
            })
            .collect()
    }

    const fn nibble(byte: u8) -> std::result::Result<u8, &'static str> {
        match byte {
            b'0'..=b'9' => Ok(byte - b'0'),
            b'a'..=b'f' => Ok(byte - b'a' + 10),
            _ => Err("invalid lowercase hexadecimal"),
        }
    }
}
