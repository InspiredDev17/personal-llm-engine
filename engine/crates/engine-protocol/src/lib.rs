//! Explicit bounded framing for the future protected Unix-domain socket.
//!
//! The framing layer performs no arbitrary deserialization. It validates a
//! fixed header and carries one bounded payload for strict message-specific
//! decoders. No general-purpose deserializer is used.

mod messages;
mod service;
mod service_payload;

pub use messages::{
    CANCELLATION_UNSIGNED_BYTES, CancellationEnvelope, GENERATION_EVENT_AUTH_BYTES,
    GENERATION_REQUEST_FIXED_BYTES, GenerationRequestEnvelope, attach_cancellation_signature,
    attach_generation_signature, encode_signed_generation_event,
};
pub use service::{
    AuthenticatedServiceRequest, AuthenticatedServiceResponse, SERVICE_REQUEST_FIXED_BYTES,
    ServiceMessageKind, attach_service_signature, encode_signed_service_response,
};
pub use service_payload::{
    MAX_RUNTIME_V3_DOCUMENT_BYTES, MAX_SERVICE_REGISTERED_MODELS, MAX_SERVICE_SECURITY_EVENTS,
    RuntimeV3Document, SERVICE_METRIC_COUNT, SERVICE_MODEL_STATUS_BYTES, ServiceCapabilitySet,
    ServiceErrorCode, ServiceHealthStatus, ServiceMetrics, ServiceModelDescriptor,
    ServiceModelStatus, ServiceRequestBody, ServiceResponseBody, ServiceSecurityEvent,
    ServiceSecurityEventKind, ShutdownRejectionReason,
};

use personalai_engine_types::{EngineError, MAX_FRAME_BYTES, MAX_REQUEST_BYTES};

/// Protocol magic for native-engine frames.
pub const FRAME_MAGIC: [u8; 4] = *b"PAIE";
/// Supported protocol major version.
pub const PROTOCOL_MAJOR: u16 = 2;
/// Supported protocol minor version.
pub const PROTOCOL_MINOR: u16 = 0;
/// Fixed header byte length.
pub const FRAME_HEADER_BYTES: usize = 16;
/// Largest payload that keeps the complete encoded frame within its hard cap.
pub const MAX_FRAME_PAYLOAD_BYTES: usize = MAX_FRAME_BYTES - FRAME_HEADER_BYTES;

/// Closed set of native-engine message categories.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum FrameKind {
    /// Go control-plane generation request.
    GenerationRequest = 1,
    /// Engine generation lifecycle event.
    GenerationEvent = 2,
    /// Go control-plane cancellation request.
    Cancellation = 3,
    /// Read-only engine capability query or response.
    Capability = 4,
    /// Sanitized non-authoritative engine security telemetry.
    SecurityEvent = 5,
    /// Authenticated non-generation control-plane request.
    ServiceRequest = 6,
    /// Signed non-generation service response.
    ServiceResponse = 7,
}

impl FrameKind {
    fn from_byte(value: u8) -> Result<Self, EngineError> {
        match value {
            1 => Ok(Self::GenerationRequest),
            2 => Ok(Self::GenerationEvent),
            3 => Ok(Self::Cancellation),
            4 => Ok(Self::Capability),
            5 => Ok(Self::SecurityEvent),
            6 => Ok(Self::ServiceRequest),
            7 => Ok(Self::ServiceResponse),
            _ => Err(EngineError::protocol_violation("frame kind is unsupported")),
        }
    }

    const fn accepted_inbound(self) -> bool {
        matches!(
            self,
            Self::GenerationRequest | Self::Cancellation | Self::Capability | Self::ServiceRequest
        )
    }

    const fn accepted_outbound(self) -> bool {
        matches!(
            self,
            Self::GenerationEvent | Self::Capability | Self::SecurityEvent | Self::ServiceResponse
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FrameDirection {
    Inbound,
    Outbound,
}

/// One validated protocol frame with a bounded payload.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Frame {
    kind: FrameKind,
    payload: Vec<u8>,
}

impl Frame {
    /// Creates a frame after message-specific payload bounds validation.
    pub fn new(kind: FrameKind, payload: Vec<u8>) -> Result<Self, EngineError> {
        validate_payload(kind, &payload)?;
        Ok(Self { kind, payload })
    }

    /// Returns the closed message category.
    pub const fn kind(&self) -> FrameKind {
        self.kind
    }

    /// Returns the raw bounded payload.
    pub fn payload(&self) -> &[u8] {
        &self.payload
    }

    /// Interprets a textual message payload as UTF-8.
    pub fn payload_text(&self) -> Result<&str, EngineError> {
        std::str::from_utf8(&self.payload)
            .map_err(|_| EngineError::invalid_input("frame payload is not UTF-8 text"))
    }

    /// Encodes one frame using a fixed little-endian header.
    pub fn encode(&self) -> Result<Vec<u8>, EngineError> {
        let payload_length = u32::try_from(self.payload.len())
            .map_err(|_| EngineError::resource_exhaustion("frame payload length exceeds u32"))?;
        let total_length = FRAME_HEADER_BYTES.checked_add(self.payload.len()).ok_or_else(|| {
            EngineError::resource_exhaustion("encoded frame byte length overflows")
        })?;
        let mut encoded = Vec::new();
        encoded
            .try_reserve_exact(total_length)
            .map_err(|_| EngineError::resource_exhaustion("frame encoding allocation failed"))?;
        encoded.extend_from_slice(&FRAME_MAGIC);
        encoded.extend_from_slice(&PROTOCOL_MAJOR.to_le_bytes());
        encoded.extend_from_slice(&PROTOCOL_MINOR.to_le_bytes());
        encoded.push(self.kind as u8);
        encoded.push(0);
        encoded.extend_from_slice(&0_u16.to_le_bytes());
        encoded.extend_from_slice(&payload_length.to_le_bytes());
        encoded.extend_from_slice(&self.payload);
        Ok(encoded)
    }

    /// Decodes exactly one Go-to-engine frame.
    ///
    /// Engine-originated generation and security events are rejected so a
    /// client cannot inject outbound-only telemetry into the service boundary.
    pub fn decode_inbound_exact(encoded: &[u8]) -> Result<Self, EngineError> {
        Self::decode_directional_exact(encoded, FrameDirection::Inbound)
    }

    /// Decodes exactly one engine-to-Go frame.
    ///
    /// Client request and cancellation categories are rejected on the outbound
    /// side of the protocol.
    pub fn decode_outbound_exact(encoded: &[u8]) -> Result<Self, EngineError> {
        Self::decode_directional_exact(encoded, FrameDirection::Outbound)
    }

    fn decode_directional_exact(
        encoded: &[u8],
        direction: FrameDirection,
    ) -> Result<Self, EngineError> {
        if encoded.len() < FRAME_HEADER_BYTES {
            return Err(EngineError::protocol_violation("frame header is truncated"));
        }
        if encoded[..4] != FRAME_MAGIC {
            return Err(EngineError::protocol_violation("frame magic is invalid"));
        }
        let major = read_u16(encoded, 4)?;
        let minor = read_u16(encoded, 6)?;
        if major != PROTOCOL_MAJOR || minor != PROTOCOL_MINOR {
            return Err(EngineError::protocol_violation("frame protocol version is unsupported"));
        }
        let kind = FrameKind::from_byte(encoded[8])?;
        let direction_allowed = match direction {
            FrameDirection::Inbound => kind.accepted_inbound(),
            FrameDirection::Outbound => kind.accepted_outbound(),
        };
        if !direction_allowed {
            return Err(EngineError::protocol_violation(
                "frame kind is not valid in this protocol direction",
            ));
        }
        if encoded[9] != 0 || read_u16(encoded, 10)? != 0 {
            return Err(EngineError::protocol_violation("frame reserved header bits are nonzero"));
        }
        let payload_length = usize::try_from(read_u32(encoded, 12)?).map_err(|_| {
            EngineError::protocol_violation("frame payload length is not addressable")
        })?;
        if payload_length == 0 || payload_length > MAX_FRAME_PAYLOAD_BYTES {
            return Err(EngineError::protocol_violation(
                "frame payload length is outside hard bounds",
            ));
        }
        let expected_length = FRAME_HEADER_BYTES
            .checked_add(payload_length)
            .ok_or_else(|| EngineError::protocol_violation("frame total byte length overflows"))?;
        if encoded.len() != expected_length {
            return Err(EngineError::protocol_violation(
                "frame length does not match its fixed header",
            ));
        }
        let source_payload = &encoded[FRAME_HEADER_BYTES..];
        validate_payload(kind, source_payload).map_err(|_| {
            EngineError::protocol_violation("frame payload violates message-specific bounds")
        })?;
        let mut payload = Vec::new();
        payload
            .try_reserve_exact(payload_length)
            .map_err(|_| EngineError::resource_exhaustion("frame decoding allocation failed"))?;
        payload.extend_from_slice(source_payload);
        Ok(Self { kind, payload })
    }
}

fn validate_payload(kind: FrameKind, payload: &[u8]) -> Result<(), EngineError> {
    if payload.is_empty() || payload.len() > MAX_FRAME_PAYLOAD_BYTES {
        return Err(EngineError::invalid_input("frame payload length is outside hard bounds"));
    }
    if matches!(kind, FrameKind::GenerationRequest | FrameKind::ServiceRequest)
        && payload.len() > MAX_REQUEST_BYTES
    {
        return Err(EngineError::invalid_input(
            "generation request payload exceeds the request hard bound",
        ));
    }
    Ok(())
}

fn read_u16(input: &[u8], offset: usize) -> Result<u16, EngineError> {
    let end = offset
        .checked_add(2)
        .ok_or_else(|| EngineError::protocol_violation("frame header offset overflows"))?;
    let bytes: [u8; 2] = input
        .get(offset..end)
        .and_then(|bytes| bytes.try_into().ok())
        .ok_or_else(|| EngineError::protocol_violation("frame header is truncated"))?;
    Ok(u16::from_le_bytes(bytes))
}

fn read_u32(input: &[u8], offset: usize) -> Result<u32, EngineError> {
    let end = offset
        .checked_add(4)
        .ok_or_else(|| EngineError::protocol_violation("frame header offset overflows"))?;
    let bytes: [u8; 4] = input
        .get(offset..end)
        .and_then(|bytes| bytes.try_into().ok())
        .ok_or_else(|| EngineError::protocol_violation("frame header is truncated"))?;
    Ok(u32::from_le_bytes(bytes))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_round_trip_is_exact() {
        let frame = Frame::new(FrameKind::Capability, br#"{"query":"capabilities"}"#.to_vec())
            .expect("payload must validate");
        let encoded = frame.encode().expect("frame must encode");
        assert_eq!(
            Frame::decode_inbound_exact(&encoded).expect("inbound frame must decode"),
            frame
        );
        assert_eq!(
            Frame::decode_outbound_exact(&encoded).expect("outbound frame must decode"),
            frame
        );
    }

    #[test]
    fn rejects_truncation_trailing_data_and_unknown_kind() {
        let frame =
            Frame::new(FrameKind::Cancellation, b"cancel".to_vec()).expect("payload must validate");
        let encoded = frame.encode().expect("frame must encode");
        assert!(Frame::decode_inbound_exact(&encoded[..15]).is_err());
        let mut trailing = encoded.clone();
        trailing.push(0);
        assert!(Frame::decode_inbound_exact(&trailing).is_err());
        let mut unknown = encoded.clone();
        unknown[8] = 99;
        assert!(Frame::decode_inbound_exact(&unknown).is_err());

        for end in 0..encoded.len() {
            assert!(
                Frame::decode_inbound_exact(&encoded[..end]).is_err(),
                "frame truncation at offset {end} was accepted"
            );
        }
        for index in 0..encoded.len() {
            let mut mutation = encoded.clone();
            mutation[index] ^= 1;
            let _bounded_result = Frame::decode_inbound_exact(&mutation);
        }
    }

    #[test]
    fn rejects_reserved_bits_but_allows_strict_binary_message_payloads() {
        let frame =
            Frame::new(FrameKind::SecurityEvent, b"event".to_vec()).expect("payload must validate");
        let mut encoded = frame.encode().expect("frame must encode");
        encoded[9] = 1;
        assert!(Frame::decode_outbound_exact(&encoded).is_err());
        assert!(Frame::new(FrameKind::Capability, vec![0xff]).is_ok());
    }

    #[test]
    fn binary_payload_is_preserved_exactly_by_directional_decode() {
        let mut encoded = Frame::new(FrameKind::Capability, b"valid".to_vec())
            .expect("payload must validate")
            .encode()
            .expect("frame must encode");
        let payload_offset = FRAME_HEADER_BYTES;
        encoded[payload_offset] = 0;
        let decoded = Frame::decode_inbound_exact(&encoded).expect("binary frame must decode");
        assert_eq!(decoded.payload()[0], 0);
    }

    #[test]
    fn rejects_message_categories_in_the_wrong_direction() {
        for kind in [FrameKind::GenerationEvent, FrameKind::SecurityEvent] {
            let encoded = Frame::new(kind, b"engine-event".to_vec())
                .expect("payload must validate")
                .encode()
                .expect("frame must encode");
            assert!(matches!(
                Frame::decode_inbound_exact(&encoded),
                Err(EngineError::ProtocolViolation(_))
            ));
        }

        for kind in [FrameKind::GenerationRequest, FrameKind::Cancellation] {
            let encoded = Frame::new(kind, b"client-request".to_vec())
                .expect("payload must validate")
                .encode()
                .expect("frame must encode");
            assert!(matches!(
                Frame::decode_outbound_exact(&encoded),
                Err(EngineError::ProtocolViolation(_))
            ));
        }
    }

    #[test]
    fn total_frame_and_generation_request_caps_are_distinct() {
        let maximum = Frame::new(FrameKind::GenerationEvent, vec![b'a'; MAX_FRAME_PAYLOAD_BYTES])
            .expect("maximum payload must validate");
        assert_eq!(maximum.encode().expect("maximum frame must encode").len(), MAX_FRAME_BYTES);
        assert!(
            Frame::new(FrameKind::GenerationRequest, vec![b'a'; MAX_REQUEST_BYTES + 1],).is_err()
        );
    }
}
