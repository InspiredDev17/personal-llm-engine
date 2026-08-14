use crate::error::{ContractError, Result};
use crate::inference::GenerationEvent;
use crate::Validate;

pub const MAGIC: [u8; 4] = *b"SAE1";
pub const MAJOR_VERSION: u8 = 1;
pub const HEADER_LEN: usize = 12;
pub const MAX_PAYLOAD_LEN: usize = 1024 * 1024;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum FrameKind {
    GenerationEvent = 1,
}

impl TryFrom<u8> for FrameKind {
    type Error = ContractError;

    fn try_from(value: u8) -> Result<Self> {
        match value {
            1 => Ok(Self::GenerationEvent),
            _ => Err(ContractError::Frame("unknown frame kind")),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Frame {
    pub kind: FrameKind,
    pub payload: Vec<u8>,
}

impl Frame {
    /// Creates a bounded generation-event frame.
    ///
    /// # Errors
    ///
    /// Returns an error when the payload is empty or exceeds the v1 1 MiB limit.
    pub fn new(kind: FrameKind, payload: Vec<u8>) -> Result<Self> {
        if payload.is_empty() {
            return Err(ContractError::Frame("payload must not be empty"));
        }
        if payload.len() > MAX_PAYLOAD_LEN {
            return Err(ContractError::Frame("payload exceeds 1 MiB"));
        }
        Ok(Self { kind, payload })
    }

    /// Encodes this frame using the SAE1 v1 header and big-endian length.
    ///
    /// # Errors
    ///
    /// Returns an error when the payload is empty or exceeds the v1 1 MiB limit.
    pub fn encode(&self) -> Result<Vec<u8>> {
        if self.payload.is_empty() {
            return Err(ContractError::Frame("payload must not be empty"));
        }
        if self.payload.len() > MAX_PAYLOAD_LEN {
            return Err(ContractError::Frame("payload exceeds 1 MiB"));
        }
        let payload_length = u32::try_from(self.payload.len())
            .map_err(|_| ContractError::Frame("payload length cannot be encoded"))?;
        let mut encoded = Vec::with_capacity(HEADER_LEN + self.payload.len());
        encoded.extend_from_slice(&MAGIC);
        encoded.push(MAJOR_VERSION);
        encoded.push(self.kind as u8);
        encoded.push(0); // flags: none are defined in v1
        encoded.push(0); // reserved: MUST be zero
        encoded.extend_from_slice(&payload_length.to_be_bytes());
        encoded.extend_from_slice(&self.payload);
        Ok(encoded)
    }

    /// Strictly decodes one complete SAE1 v1 frame.
    ///
    /// # Errors
    ///
    /// Returns an error for malformed, unsupported, oversized, truncated, or
    /// trailing frame data.
    pub fn decode(encoded: &[u8]) -> Result<Self> {
        if encoded.len() < HEADER_LEN {
            return Err(ContractError::Frame("truncated header"));
        }
        if encoded[..4] != MAGIC {
            return Err(ContractError::Frame("bad magic"));
        }
        if encoded[4] != MAJOR_VERSION {
            return Err(ContractError::Frame("unsupported major version"));
        }
        let kind = FrameKind::try_from(encoded[5])?;
        if encoded[6] != 0 {
            return Err(ContractError::Frame("nonzero flags"));
        }
        if encoded[7] != 0 {
            return Err(ContractError::Frame("nonzero reserved byte"));
        }
        let payload_length = u32::from_be_bytes(
            encoded[8..12]
                .try_into()
                .map_err(|_| ContractError::Frame("truncated payload length"))?,
        ) as usize;
        if payload_length == 0 {
            return Err(ContractError::Frame("payload must not be empty"));
        }
        if payload_length > MAX_PAYLOAD_LEN {
            return Err(ContractError::Frame("payload exceeds 1 MiB"));
        }
        let expected_length = HEADER_LEN
            .checked_add(payload_length)
            .ok_or(ContractError::Frame("frame length overflow"))?;
        if encoded.len() < expected_length {
            return Err(ContractError::Frame("truncated payload"));
        }
        if encoded.len() > expected_length {
            return Err(ContractError::Frame("trailing bytes"));
        }
        Self::new(kind, encoded[HEADER_LEN..].to_vec())
    }

    /// Encodes one validated generation event as a complete SAE1 frame.
    ///
    /// # Errors
    ///
    /// Returns an error when event validation or JSON/framing fails.
    pub fn encode_generation_event(event: &GenerationEvent) -> Result<Vec<u8>> {
        event.validate()?;
        Self::new(FrameKind::GenerationEvent, serde_json::to_vec(event)?)?.encode()
    }

    /// Decodes and validates one generation event from a complete SAE1 frame.
    ///
    /// # Errors
    ///
    /// Returns an error for framing, UTF-8, JSON, or contract validation failure.
    pub fn decode_generation_event(encoded: &[u8]) -> Result<GenerationEvent> {
        let frame = Self::decode(encoded)?;
        if frame.kind != FrameKind::GenerationEvent {
            return Err(ContractError::Frame("not a generation-event frame"));
        }
        std::str::from_utf8(&frame.payload)
            .map_err(|_| ContractError::Frame("payload is not valid UTF-8"))?;
        let event: GenerationEvent = serde_json::from_slice(&frame.payload)?;
        event.validate()?;
        Ok(event)
    }
}

#[cfg(test)]
mod tests {
    use super::{Frame, FrameKind, HEADER_LEN, MAGIC, MAX_PAYLOAD_LEN};
    use serde::Deserialize;
    use std::fs;
    use std::path::PathBuf;

    #[test]
    fn round_trips_generation_event_frame() {
        let frame = Frame::new(FrameKind::GenerationEvent, br#"{"ok":true}"#.to_vec())
            .expect("valid frame");
        let encoded = frame.encode().expect("encode");
        assert_eq!(&encoded[..4], &MAGIC);
        assert_eq!(Frame::decode(&encoded).expect("decode"), frame);
    }

    #[test]
    fn rejects_malformed_headers() {
        let valid = Frame::new(FrameKind::GenerationEvent, b"{}".to_vec())
            .expect("valid")
            .encode()
            .expect("encode");
        for index in [4_usize, 5, 6, 7] {
            let mut malformed = valid.clone();
            malformed[index] = 2;
            assert!(Frame::decode(&malformed).is_err(), "index {index}");
        }
        assert!(Frame::decode(&valid[..HEADER_LEN - 1]).is_err());
    }

    #[test]
    fn rejects_length_mismatch_and_oversize() {
        let mut truncated = Vec::from(MAGIC);
        truncated.extend_from_slice(&[1, 1, 0, 0]);
        truncated.extend_from_slice(&1_u32.to_be_bytes());
        assert!(Frame::decode(&truncated).is_err());

        let mut oversized = Vec::from(MAGIC);
        oversized.extend_from_slice(&[1, 1, 0, 0]);
        oversized.extend_from_slice(&u32::try_from(MAX_PAYLOAD_LEN + 1).unwrap().to_be_bytes());
        assert!(Frame::decode(&oversized).is_err());
        assert!(Frame::new(FrameKind::GenerationEvent, vec![0; MAX_PAYLOAD_LEN + 1]).is_err());
    }

    #[test]
    fn rejects_trailing_bytes() {
        let mut encoded = Frame::new(FrameKind::GenerationEvent, b"{}".to_vec())
            .expect("valid")
            .encode()
            .expect("encode");
        encoded.push(0);
        assert!(Frame::decode(&encoded).is_err());
    }

    #[derive(Deserialize)]
    struct BinaryManifest {
        protocol: String,
        contract_version: String,
        encoding: String,
        fixtures: Vec<BinaryFixture>,
    }

    #[derive(Deserialize)]
    struct BinaryFixture {
        path: String,
        expected: String,
        reason_code: Option<String>,
    }

    #[test]
    fn consumes_all_published_sae1_golden_fixtures() {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../fixtures/v1/binary");
        let manifest: BinaryManifest = serde_json::from_slice(
            &fs::read(root.join("manifest.json")).expect("read binary manifest"),
        )
        .expect("decode binary manifest");
        assert_eq!(manifest.protocol, "SAE1");
        assert_eq!(manifest.contract_version, "1.0.0");
        assert_eq!(manifest.encoding, "lowercase_hex");

        for fixture in manifest.fixtures {
            let hex = fs::read_to_string(root.join(&fixture.path)).expect("read hex fixture");
            let bytes = decode_lower_hex(hex.trim()).expect("fixture uses lowercase hex");
            let decoded = Frame::decode_generation_event(&bytes);
            match fixture.expected.as_str() {
                "accept" => {
                    let event = decoded
                        .unwrap_or_else(|error| panic!("{} should decode: {error}", fixture.path));
                    let reencoded =
                        Frame::encode_generation_event(&event).expect("valid event must re-encode");
                    assert_eq!(reencoded, bytes, "{} must be byte stable", fixture.path);
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
            || source.len() % 2 != 0
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
                let high = hex_nibble(pair[0])?;
                let low = hex_nibble(pair[1])?;
                Ok((high << 4) | low)
            })
            .collect()
    }

    fn hex_nibble(byte: u8) -> std::result::Result<u8, &'static str> {
        match byte {
            b'0'..=b'9' => Ok(byte - b'0'),
            b'a'..=b'f' => Ok(byte - b'a' + 10),
            _ => Err("invalid hexadecimal digit"),
        }
    }
}
