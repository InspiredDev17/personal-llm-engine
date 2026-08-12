//! Closed, authenticated service-control messages carried inside PAIE frames.
//!
//! The service envelope is deliberately not a general serialization format.
//! Every field has a fixed byte order and width; message-specific payloads are
//! decoded by their owning service handler. Unknown message numbers, flags,
//! trailing bytes, and unsupported versions fail closed.

use personalai_engine_crypto::{
    ED25519_SIGNATURE_BYTES, EngineResponseSigner, TrustedControlPlaneKeys,
    verify_control_plane_signature_with_keys,
};
use personalai_engine_types::{EngineError, MAX_REQUEST_BYTES, RequestId, SignerKeyId};

const SERVICE_REQUEST_MAGIC: [u8; 8] = *b"PAISVC01";
const SERVICE_RESPONSE_MAGIC: [u8; 8] = *b"PAISVR01";
const SERVICE_VERSION: u16 = 1;
/// Bytes in a service envelope before its bounded payload and signature.
pub const SERVICE_REQUEST_FIXED_BYTES: usize = 56;

/// Exact first-party service message catalog.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum ServiceMessageKind {
    /// Begin a version-negotiated authenticated session.
    Hello = 1,
    /// Version negotiation succeeded.
    HelloAccepted = 2,
    /// Version negotiation failed.
    HelloRejected = 3,
    /// Query compiled and implemented engine capabilities.
    ListCapabilities = 4,
    /// Query configuration-owned model identifiers.
    ListRegisteredModels = 5,
    /// Load one statically registered model identifier.
    LoadModel = 6,
    /// Unload one idle model identifier.
    UnloadModel = 7,
    /// Tokenize bounded prompt text with one loaded model.
    Tokenize = 8,
    /// Existing authenticated generation stream message category.
    Generate = 9,
    /// Existing authenticated cancellation message category.
    Cancel = 10,
    /// Existing signed generation event category.
    GenerationEvent = 11,
    /// Query local health state.
    Health = 12,
    /// Query content-free local metrics.
    Metrics = 13,
    /// Explicit structured failure.
    Error = 14,
    /// Request service shutdown through the administrative path.
    ShutdownRequest = 15,
    /// Shutdown was refused by the data-plane authorization policy.
    ShutdownRejected = 16,
    /// Shutdown was accepted by the separate administrative policy.
    ShutdownAccepted = 17,
}

impl ServiceMessageKind {
    fn from_byte(value: u8) -> Result<Self, EngineError> {
        match value {
            1 => Ok(Self::Hello),
            2 => Ok(Self::HelloAccepted),
            3 => Ok(Self::HelloRejected),
            4 => Ok(Self::ListCapabilities),
            5 => Ok(Self::ListRegisteredModels),
            6 => Ok(Self::LoadModel),
            7 => Ok(Self::UnloadModel),
            8 => Ok(Self::Tokenize),
            9 => Ok(Self::Generate),
            10 => Ok(Self::Cancel),
            11 => Ok(Self::GenerationEvent),
            12 => Ok(Self::Health),
            13 => Ok(Self::Metrics),
            14 => Ok(Self::Error),
            15 => Ok(Self::ShutdownRequest),
            16 => Ok(Self::ShutdownRejected),
            17 => Ok(Self::ShutdownAccepted),
            _ => Err(EngineError::protocol_violation("service message type is unsupported")),
        }
    }

    fn accepted_request(self) -> bool {
        matches!(
            self,
            Self::Hello
                | Self::ListCapabilities
                | Self::ListRegisteredModels
                | Self::LoadModel
                | Self::UnloadModel
                | Self::Tokenize
                | Self::Health
                | Self::Metrics
                | Self::ShutdownRequest
        )
    }

    fn accepted_response(self) -> bool {
        matches!(
            self,
            Self::HelloAccepted
                | Self::HelloRejected
                | Self::ListCapabilities
                | Self::ListRegisteredModels
                | Self::LoadModel
                | Self::UnloadModel
                | Self::Tokenize
                | Self::Health
                | Self::Metrics
                | Self::Error
                | Self::ShutdownRejected
                | Self::ShutdownAccepted
        )
    }
}

/// Strictly authenticated non-generation request.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AuthenticatedServiceRequest {
    kind: ServiceMessageKind,
    request_id: RequestId,
    nonce: u64,
    control_key_id: SignerKeyId,
    payload: Vec<u8>,
}

impl AuthenticatedServiceRequest {
    /// Constructs the exact request bytes that the Go control plane signs.
    pub fn unsigned_bytes(
        kind: ServiceMessageKind,
        request_id: RequestId,
        nonce: u64,
        control_key_id: SignerKeyId,
        payload: &[u8],
    ) -> Result<Vec<u8>, EngineError> {
        if !kind.accepted_request() {
            return Err(EngineError::invalid_input("message type is not a service request"));
        }
        if nonce == 0 {
            return Err(EngineError::invalid_input("service request nonce must be nonzero"));
        }
        let total = SERVICE_REQUEST_FIXED_BYTES
            .checked_add(payload.len())
            .filter(|total| *total <= MAX_REQUEST_BYTES - ED25519_SIGNATURE_BYTES)
            .ok_or_else(|| EngineError::resource_exhaustion("service request is too large"))?;
        let payload_length = u32::try_from(payload.len())
            .map_err(|_| EngineError::resource_exhaustion("service payload exceeds u32"))?;
        let mut bytes = vec![0_u8; total];
        bytes[..8].copy_from_slice(&SERVICE_REQUEST_MAGIC);
        bytes[8..10].copy_from_slice(&SERVICE_VERSION.to_le_bytes());
        bytes[10] = kind as u8;
        bytes[12..20].copy_from_slice(&nonce.to_le_bytes());
        bytes[20..36].copy_from_slice(&request_id.to_bytes());
        bytes[36..52].copy_from_slice(control_key_id.as_bytes());
        bytes[52..56].copy_from_slice(&payload_length.to_le_bytes());
        bytes[SERVICE_REQUEST_FIXED_BYTES..].copy_from_slice(payload);
        Ok(bytes)
    }

    /// Verifies and decodes one exact request envelope.
    pub fn decode(
        bytes: &[u8],
        trusted_keys: &TrustedControlPlaneKeys,
    ) -> Result<Self, EngineError> {
        let minimum = SERVICE_REQUEST_FIXED_BYTES + ED25519_SIGNATURE_BYTES;
        if bytes.len() < minimum || bytes.len() > MAX_REQUEST_BYTES {
            return Err(EngineError::protocol_violation(
                "service request length is outside hard bounds",
            ));
        }
        if bytes[..8] != SERVICE_REQUEST_MAGIC
            || read_u16(bytes, 8)? != SERVICE_VERSION
            || bytes[11] != 0
        {
            return Err(EngineError::protocol_violation(
                "service request magic, version, or flags are invalid",
            ));
        }
        let kind = ServiceMessageKind::from_byte(bytes[10])?;
        if !kind.accepted_request() {
            return Err(EngineError::protocol_violation(
                "service response type is invalid on request direction",
            ));
        }
        let nonce = read_u64(bytes, 12)?;
        if nonce == 0 {
            return Err(EngineError::protocol_violation("service request nonce is zero"));
        }
        let request_id = RequestId::from_bytes(read_exact::<16>(bytes, 20)?)
            .map_err(|_| EngineError::protocol_violation("service request ID is invalid"))?;
        let control_key_id = SignerKeyId::from_bytes(read_exact::<16>(bytes, 36)?)
            .map_err(|_| EngineError::protocol_violation("service key ID is invalid"))?;
        let signed_length = bytes.len() - ED25519_SIGNATURE_BYTES;
        let payload_length = usize::try_from(read_u32(bytes, 52)?)
            .map_err(|_| EngineError::protocol_violation("service payload is not addressable"))?;
        let expected = SERVICE_REQUEST_FIXED_BYTES
            .checked_add(payload_length)
            .filter(|expected| *expected == signed_length)
            .ok_or_else(|| EngineError::protocol_violation("service payload length is invalid"))?;
        let signature = read_exact::<ED25519_SIGNATURE_BYTES>(bytes, expected)?;
        verify_control_plane_signature_with_keys(
            trusted_keys,
            control_key_id,
            &bytes[..signed_length],
            &signature,
        )?;
        let payload = bytes[SERVICE_REQUEST_FIXED_BYTES..signed_length].to_vec();
        Ok(Self { kind, request_id, nonce, control_key_id, payload })
    }

    /// Returns the closed message type.
    pub const fn kind(&self) -> ServiceMessageKind {
        self.kind
    }

    /// Returns the correlation identifier from the fixed message header.
    pub const fn request_id(&self) -> RequestId {
        self.request_id
    }

    /// Returns the strictly increasing connection nonce.
    pub const fn nonce(&self) -> u64 {
        self.nonce
    }

    /// Returns the authenticated control-plane key identity.
    pub const fn control_key_id(&self) -> SignerKeyId {
        self.control_key_id
    }

    /// Returns the bounded message-specific bytes.
    pub fn payload(&self) -> &[u8] {
        &self.payload
    }
}

/// Appends a detached Ed25519 signature to exact unsigned service bytes.
pub fn attach_service_signature(
    unsigned: &[u8],
    signature: &[u8; ED25519_SIGNATURE_BYTES],
) -> Result<Vec<u8>, EngineError> {
    if unsigned.len() < SERVICE_REQUEST_FIXED_BYTES
        || unsigned.len() > MAX_REQUEST_BYTES - ED25519_SIGNATURE_BYTES
    {
        return Err(EngineError::invalid_input("unsigned service request length is invalid"));
    }
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(unsigned.len() + signature.len())
        .map_err(|_| EngineError::resource_exhaustion("service signature allocation failed"))?;
    bytes.extend_from_slice(unsigned);
    bytes.extend_from_slice(signature);
    Ok(bytes)
}

/// Encodes and signs one exact service response.
pub fn encode_signed_service_response(
    kind: ServiceMessageKind,
    request_id: RequestId,
    request_nonce: u64,
    payload: &[u8],
    signer: &EngineResponseSigner,
) -> Result<Vec<u8>, EngineError> {
    if !kind.accepted_response() {
        return Err(EngineError::invalid_input("message type is not a service response"));
    }
    if request_nonce == 0 {
        return Err(EngineError::invalid_input("service response nonce must be nonzero"));
    }
    let total = SERVICE_REQUEST_FIXED_BYTES
        .checked_add(payload.len())
        .ok_or_else(|| EngineError::resource_exhaustion("service response length overflows"))?;
    let payload_length = u32::try_from(payload.len())
        .map_err(|_| EngineError::resource_exhaustion("service response payload exceeds u32"))?;
    let mut bytes = vec![0_u8; total];
    bytes[..8].copy_from_slice(&SERVICE_RESPONSE_MAGIC);
    bytes[8..10].copy_from_slice(&SERVICE_VERSION.to_le_bytes());
    bytes[10] = kind as u8;
    bytes[12..20].copy_from_slice(&request_nonce.to_le_bytes());
    bytes[20..36].copy_from_slice(&request_id.to_bytes());
    bytes[36..52].copy_from_slice(signer.key_id().as_bytes());
    bytes[52..56].copy_from_slice(&payload_length.to_le_bytes());
    bytes[SERVICE_REQUEST_FIXED_BYTES..].copy_from_slice(payload);
    let signature = signer.sign_response(&bytes)?;
    bytes.extend_from_slice(&signature);
    Ok(bytes)
}

fn read_u16(bytes: &[u8], offset: usize) -> Result<u16, EngineError> {
    Ok(u16::from_le_bytes(read_exact(bytes, offset)?))
}

fn read_u32(bytes: &[u8], offset: usize) -> Result<u32, EngineError> {
    Ok(u32::from_le_bytes(read_exact(bytes, offset)?))
}

fn read_u64(bytes: &[u8], offset: usize) -> Result<u64, EngineError> {
    Ok(u64::from_le_bytes(read_exact(bytes, offset)?))
}

fn read_exact<const N: usize>(bytes: &[u8], offset: usize) -> Result<[u8; N], EngineError> {
    let end = offset
        .checked_add(N)
        .ok_or_else(|| EngineError::protocol_violation("service field offset overflows"))?;
    bytes
        .get(offset..end)
        .and_then(|field| field.try_into().ok())
        .ok_or_else(|| EngineError::protocol_violation("service field is truncated"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ed25519_dalek::{Signer, SigningKey};
    use personalai_engine_crypto::{TrustedControlPlaneKey, control_plane_signature_message};

    #[test]
    fn authenticated_request_is_exact_and_rejects_tampering() {
        let signing = SigningKey::from_bytes(&[0x31; 32]);
        let key_id = SignerKeyId::from_bytes([0x32; 16]).expect("key ID");
        let trusted = TrustedControlPlaneKeys::single(
            TrustedControlPlaneKey::active(key_id, signing.verifying_key().to_bytes())
                .expect("trusted key"),
        )
        .expect("key ring");
        let request_id = RequestId::new(7).expect("request ID");
        let unsigned = AuthenticatedServiceRequest::unsigned_bytes(
            ServiceMessageKind::Health,
            request_id,
            9,
            key_id,
            &[],
        )
        .expect("unsigned request");
        let transcript = control_plane_signature_message(&unsigned).expect("transcript");
        let signed = attach_service_signature(&unsigned, &signing.sign(&transcript).to_bytes())
            .expect("signed request");
        let decoded =
            AuthenticatedServiceRequest::decode(&signed, &trusted).expect("verified request");
        assert_eq!(decoded.kind(), ServiceMessageKind::Health);
        assert_eq!(decoded.request_id(), request_id);
        assert_eq!(decoded.nonce(), 9);
        assert!(decoded.payload().is_empty());

        let mut tampered = signed;
        tampered[10] = ServiceMessageKind::Metrics as u8;
        assert!(AuthenticatedServiceRequest::decode(&tampered, &trusted).is_err());
    }

    #[test]
    fn request_direction_and_lengths_fail_closed() {
        let request_id = RequestId::new(1).expect("request ID");
        let key_id = SignerKeyId::from_bytes([1; 16]).expect("key ID");
        assert!(
            AuthenticatedServiceRequest::unsigned_bytes(
                ServiceMessageKind::HelloAccepted,
                request_id,
                1,
                key_id,
                &[],
            )
            .is_err()
        );
        assert!(
            AuthenticatedServiceRequest::unsigned_bytes(
                ServiceMessageKind::Health,
                request_id,
                0,
                key_id,
                &[],
            )
            .is_err()
        );
    }
}
