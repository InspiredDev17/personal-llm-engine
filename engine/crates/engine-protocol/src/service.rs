//! Closed, authenticated service-control messages carried inside PAIE frames.
//!
//! The service envelope is deliberately not a general serialization format.
//! Every field has a fixed byte order and width; message-specific payloads are
//! decoded by their owning service handler. Unknown message numbers, flags,
//! trailing bytes, and unsupported versions fail closed.

use personalai_engine_crypto::{
    ED25519_PUBLIC_KEY_BYTES, ED25519_SIGNATURE_BYTES, EngineResponseSigner,
    TrustedControlPlaneKeys, verify_control_plane_signature_with_keys,
    verify_engine_response_signature,
};
use personalai_engine_types::{EngineError, MAX_REQUEST_BYTES, RequestId, SignerKeyId};

use crate::{ServiceRequestBody, ServiceResponseBody};

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
    /// Query one registered model's atomic load and scheduler status.
    ModelStatus = 18,
    /// Poll a bounded batch of sanitized security events for the Go control plane.
    SecurityEvents = 19,
    /// Query authenticated, path-free identity for one loaded model.
    ModelDescriptor = 20,
    /// Acknowledge delivery of a previously polled security-event prefix.
    AcknowledgeSecurityEvents = 21,
    /// Create one sealed reusable prefix cache from a loaded registered model.
    CreateReusableCache = 22,
    /// Describe one scope-authorized reusable cache.
    DescribeCache = 23,
    /// List bounded scope-authorized reusable caches.
    ListCaches = 24,
    /// Stage one cache attachment for the correlated generation request.
    AttachCache = 25,
    /// Release a staged cache attachment.
    ReleaseCache = 26,
    /// Invalidate reusable cache state through a closed selector.
    InvalidateCache = 27,
    /// Query content-free reusable-cache metrics for one loaded model.
    CacheMetrics = 28,
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
            18 => Ok(Self::ModelStatus),
            19 => Ok(Self::SecurityEvents),
            20 => Ok(Self::ModelDescriptor),
            21 => Ok(Self::AcknowledgeSecurityEvents),
            22 => Ok(Self::CreateReusableCache),
            23 => Ok(Self::DescribeCache),
            24 => Ok(Self::ListCaches),
            25 => Ok(Self::AttachCache),
            26 => Ok(Self::ReleaseCache),
            27 => Ok(Self::InvalidateCache),
            28 => Ok(Self::CacheMetrics),
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
                | Self::ModelStatus
                | Self::SecurityEvents
                | Self::ModelDescriptor
                | Self::AcknowledgeSecurityEvents
                | Self::CreateReusableCache
                | Self::DescribeCache
                | Self::ListCaches
                | Self::AttachCache
                | Self::ReleaseCache
                | Self::InvalidateCache
                | Self::CacheMetrics
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
                | Self::ModelStatus
                | Self::SecurityEvents
                | Self::ModelDescriptor
                | Self::AcknowledgeSecurityEvents
                | Self::CreateReusableCache
                | Self::DescribeCache
                | Self::ListCaches
                | Self::AttachCache
                | Self::ReleaseCache
                | Self::InvalidateCache
                | Self::CacheMetrics
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
    body: ServiceRequestBody,
}

impl AuthenticatedServiceRequest {
    /// Constructs the exact request bytes that the Go control plane signs.
    pub fn unsigned_bytes(
        request_id: RequestId,
        nonce: u64,
        control_key_id: SignerKeyId,
        body: &ServiceRequestBody,
    ) -> Result<Vec<u8>, EngineError> {
        if nonce == 0 {
            return Err(EngineError::invalid_input("service request nonce must be nonzero"));
        }
        let kind = body.kind();
        let payload = body.encode()?;
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
        bytes[SERVICE_REQUEST_FIXED_BYTES..].copy_from_slice(&payload);
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
        let body =
            ServiceRequestBody::decode(kind, &bytes[SERVICE_REQUEST_FIXED_BYTES..signed_length])?;
        Ok(Self { kind, request_id, nonce, control_key_id, body })
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

    /// Returns the fully validated, closed message-specific body.
    pub const fn body(&self) -> &ServiceRequestBody {
        &self.body
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
    request_id: RequestId,
    request_nonce: u64,
    body: &ServiceResponseBody,
    signer: &EngineResponseSigner,
) -> Result<Vec<u8>, EngineError> {
    if request_nonce == 0 {
        return Err(EngineError::invalid_input("service response nonce must be nonzero"));
    }
    let kind = body.kind();
    let payload = body.encode()?;
    let total = SERVICE_REQUEST_FIXED_BYTES
        .checked_add(payload.len())
        .filter(|total| *total <= MAX_REQUEST_BYTES - ED25519_SIGNATURE_BYTES)
        .ok_or_else(|| EngineError::resource_exhaustion("service response is too large"))?;
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
    bytes[SERVICE_REQUEST_FIXED_BYTES..].copy_from_slice(&payload);
    let signature = signer.sign_response(&bytes)?;
    bytes.extend_from_slice(&signature);
    Ok(bytes)
}

/// Strictly verified engine service response bound to one originating request.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AuthenticatedServiceResponse {
    kind: ServiceMessageKind,
    request_id: RequestId,
    request_nonce: u64,
    engine_key_id: SignerKeyId,
    body: ServiceResponseBody,
}

impl AuthenticatedServiceResponse {
    /// Verifies structure, correlation, pinned identity, and exact signature.
    pub fn decode(
        bytes: &[u8],
        expected_request_id: RequestId,
        expected_request_nonce: u64,
        expected_engine_key_id: SignerKeyId,
        expected_engine_public_key: &[u8; ED25519_PUBLIC_KEY_BYTES],
    ) -> Result<Self, EngineError> {
        let minimum = SERVICE_REQUEST_FIXED_BYTES + ED25519_SIGNATURE_BYTES;
        if bytes.len() < minimum || bytes.len() > MAX_REQUEST_BYTES {
            return Err(EngineError::protocol_violation(
                "service response length is outside hard bounds",
            ));
        }
        if bytes[..8] != SERVICE_RESPONSE_MAGIC
            || read_u16(bytes, 8)? != SERVICE_VERSION
            || bytes[11] != 0
        {
            return Err(EngineError::protocol_violation(
                "service response magic, version, or flags are invalid",
            ));
        }
        let kind = ServiceMessageKind::from_byte(bytes[10])?;
        if !kind.accepted_response() {
            return Err(EngineError::protocol_violation(
                "service request type is invalid on response direction",
            ));
        }
        let request_nonce = read_u64(bytes, 12)?;
        if request_nonce == 0 || request_nonce != expected_request_nonce {
            return Err(EngineError::authentication_failure(
                "service response nonce does not match its request",
            ));
        }
        let request_id = RequestId::from_bytes(read_exact::<16>(bytes, 20)?)
            .map_err(|_| EngineError::protocol_violation("service response ID is invalid"))?;
        if request_id != expected_request_id {
            return Err(EngineError::authentication_failure(
                "service response ID does not match its request",
            ));
        }
        let engine_key_id = SignerKeyId::from_bytes(read_exact::<16>(bytes, 36)?)
            .map_err(|_| EngineError::protocol_violation("engine response key ID is invalid"))?;
        let signed_length = bytes.len() - ED25519_SIGNATURE_BYTES;
        let payload_length = usize::try_from(read_u32(bytes, 52)?).map_err(|_| {
            EngineError::protocol_violation("service response payload is not addressable")
        })?;
        let expected_length = SERVICE_REQUEST_FIXED_BYTES
            .checked_add(payload_length)
            .filter(|expected| *expected == signed_length)
            .ok_or_else(|| {
                EngineError::protocol_violation("service response payload length is invalid")
            })?;
        let signature = read_exact::<ED25519_SIGNATURE_BYTES>(bytes, expected_length)?;
        verify_engine_response_signature(
            expected_engine_key_id,
            expected_engine_public_key,
            engine_key_id,
            &bytes[..signed_length],
            &signature,
        )?;
        let body =
            ServiceResponseBody::decode(kind, &bytes[SERVICE_REQUEST_FIXED_BYTES..signed_length])?;
        Ok(Self { kind, request_id, request_nonce, engine_key_id, body })
    }

    /// Returns the closed response category.
    pub const fn kind(&self) -> ServiceMessageKind {
        self.kind
    }

    /// Returns the authenticated request correlation identifier.
    pub const fn request_id(&self) -> RequestId {
        self.request_id
    }

    /// Returns the authenticated originating request nonce.
    pub const fn request_nonce(&self) -> u64 {
        self.request_nonce
    }

    /// Returns the pinned engine response identity.
    pub const fn engine_key_id(&self) -> SignerKeyId {
        self.engine_key_id
    }

    /// Returns the fully validated, closed message-specific response body.
    pub const fn body(&self) -> &ServiceResponseBody {
        &self.body
    }
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
    use crate::{SERVICE_METRIC_COUNT, ServiceHealthStatus, ServiceMetrics};
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
            request_id,
            9,
            key_id,
            &ServiceRequestBody::Health,
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
        assert_eq!(decoded.body(), &ServiceRequestBody::Health);

        let mut tampered = signed;
        tampered[10] = ServiceMessageKind::Metrics as u8;
        assert!(AuthenticatedServiceRequest::decode(&tampered, &trusted).is_err());

        let invalid_body = ServiceRequestBody::Tokenize {
            model_id: personalai_engine_types::ModelId::new("tiny").expect("model ID"),
            text: "x".to_owned(),
        };
        let mut unsigned =
            AuthenticatedServiceRequest::unsigned_bytes(request_id, 10, key_id, &invalid_body)
                .expect("unsigned tokenize request");
        *unsigned.last_mut().expect("text byte") = 0;
        let transcript = control_plane_signature_message(&unsigned).expect("transcript");
        let signed = attach_service_signature(&unsigned, &signing.sign(&transcript).to_bytes())
            .expect("signed malformed request");
        assert!(AuthenticatedServiceRequest::decode(&signed, &trusted).is_err());
    }

    #[test]
    fn go_health_request_conformance_fixture_verifies_and_decodes() {
        let fixture = decode_hex(include_str!(
            "../../../../docs/native-engine/fixtures/go-service-health-request-v1.hex"
        ));
        let key_id_bytes = [0x32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        let key_id = SignerKeyId::from_bytes(key_id_bytes).expect("Go fixture key ID");
        let public_key = [
            0xeb, 0xe8, 0x0b, 0x7c, 0xad, 0xea, 0x27, 0x7a, 0xc0, 0x5f, 0xb8, 0x5c, 0x71, 0x64,
            0xfe, 0x15, 0xeb, 0xd6, 0x87, 0x3c, 0x4a, 0x74, 0xb3, 0x29, 0x6a, 0x46, 0x2a, 0x10,
            0x26, 0xfd, 0x9b, 0x0f,
        ];
        let trusted = TrustedControlPlaneKeys::single(
            TrustedControlPlaneKey::active(key_id, public_key).expect("Go fixture key"),
        )
        .expect("Go fixture key ring");
        let request =
            AuthenticatedServiceRequest::decode(&fixture, &trusted).expect("Go health request");

        assert_eq!(request.kind(), ServiceMessageKind::Health);
        assert_eq!(request.nonce(), 7);
        assert_eq!(request.control_key_id(), key_id);
        assert_eq!(
            request.request_id().to_bytes(),
            [
                0xcf, 0x43, 0xd8, 0x8d, 0x78, 0xf2, 0x8f, 0x04, 0xf7, 0xd9, 0x06, 0x03, 0x23, 0x3e,
                0x55, 0xf9,
            ]
        );
        assert_eq!(request.body(), &ServiceRequestBody::Health);
    }

    #[test]
    fn go_model_status_request_conformance_fixture_verifies_and_decodes() {
        let fixture = decode_hex(include_str!(
            "../../../../docs/native-engine/fixtures/go-service-model-status-request-v1.hex"
        ));
        let key_id_bytes = [0x32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        let key_id = SignerKeyId::from_bytes(key_id_bytes).expect("Go fixture key ID");
        let mut signing_seed = [0_u8; 32];
        signing_seed[0] = 0x31;
        let signing = SigningKey::from_bytes(&signing_seed);
        let trusted = TrustedControlPlaneKeys::single(
            TrustedControlPlaneKey::active(key_id, signing.verifying_key().to_bytes())
                .expect("Go fixture key"),
        )
        .expect("Go fixture key ring");
        let request = AuthenticatedServiceRequest::decode(&fixture, &trusted)
            .expect("Go model status request");

        assert_eq!(request.kind(), ServiceMessageKind::ModelStatus);
        assert_eq!(request.nonce(), 79);
        assert_eq!(request.control_key_id(), key_id);
        assert_eq!(
            request.request_id().to_bytes(),
            [
                0x92, 0xb4, 0x66, 0x04, 0x77, 0x70, 0x40, 0xb6, 0xfe, 0x0a, 0x32, 0xb5, 0x7c, 0x3b,
                0x67, 0xb4,
            ]
        );
        assert_eq!(
            request.body(),
            &ServiceRequestBody::ModelStatus(
                personalai_engine_types::ModelId::new("tiny.decoder").expect("model ID")
            )
        );
    }

    #[test]
    fn request_direction_and_lengths_fail_closed() {
        let request_id = RequestId::new(1).expect("request ID");
        let key_id = SignerKeyId::from_bytes([1; 16]).expect("key ID");
        assert!(
            AuthenticatedServiceRequest::unsigned_bytes(
                request_id,
                0,
                key_id,
                &ServiceRequestBody::Health,
            )
            .is_err()
        );
    }

    #[test]
    fn signed_response_verifies_exact_identity_correlation_and_payload() {
        let request_id = RequestId::new(0x77).expect("request ID");
        let engine_key_id = SignerKeyId::from_bytes([0x78; 16]).expect("engine key ID");
        let mut seed = [0x79; 32];
        let signer = EngineResponseSigner::from_seed(engine_key_id, &mut seed).expect("signer");
        let public_key = signer.public_key();
        let encoded = encode_signed_service_response(
            request_id,
            11,
            &ServiceResponseBody::Health(ServiceHealthStatus::Ready),
            &signer,
        )
        .expect("response must encode");
        let response = AuthenticatedServiceResponse::decode(
            &encoded,
            request_id,
            11,
            engine_key_id,
            &public_key,
        )
        .expect("response must verify");
        assert_eq!(response.kind(), ServiceMessageKind::Health);
        assert_eq!(response.request_id(), request_id);
        assert_eq!(response.request_nonce(), 11);
        assert_eq!(response.engine_key_id(), engine_key_id);
        assert_eq!(response.body(), &ServiceResponseBody::Health(ServiceHealthStatus::Ready));

        assert!(
            AuthenticatedServiceResponse::decode(
                &encoded,
                RequestId::new(0x88).expect("other request ID"),
                11,
                engine_key_id,
                &public_key,
            )
            .is_err()
        );
        assert!(
            AuthenticatedServiceResponse::decode(
                &encoded,
                request_id,
                12,
                engine_key_id,
                &public_key,
            )
            .is_err()
        );
        let mut tampered = encoded;
        tampered[SERVICE_REQUEST_FIXED_BYTES] ^= 1;
        assert!(
            AuthenticatedServiceResponse::decode(
                &tampered,
                request_id,
                11,
                engine_key_id,
                &public_key,
            )
            .is_err()
        );
    }

    #[test]
    fn response_parser_rejects_wrong_direction_version_length_and_key() {
        let request_id = RequestId::new(5).expect("request ID");
        let engine_key_id = SignerKeyId::from_bytes([6; 16]).expect("engine key ID");
        let mut seed = [7; 32];
        let signer = EngineResponseSigner::from_seed(engine_key_id, &mut seed).expect("signer");
        let public_key = signer.public_key();
        let encoded = encode_signed_service_response(
            request_id,
            3,
            &ServiceResponseBody::Metrics(ServiceMetrics::new([0; SERVICE_METRIC_COUNT])),
            &signer,
        )
        .expect("response must encode");
        for length in [0, 8, SERVICE_REQUEST_FIXED_BYTES, encoded.len() - 1] {
            assert!(
                AuthenticatedServiceResponse::decode(
                    &encoded[..length],
                    request_id,
                    3,
                    engine_key_id,
                    &public_key,
                )
                .is_err()
            );
        }
        let mut wrong_kind = encoded.clone();
        wrong_kind[10] = ServiceMessageKind::Hello as u8;
        assert!(
            AuthenticatedServiceResponse::decode(
                &wrong_kind,
                request_id,
                3,
                engine_key_id,
                &public_key,
            )
            .is_err()
        );
        let wrong_key = SignerKeyId::from_bytes([8; 16]).expect("wrong key ID");
        assert!(
            AuthenticatedServiceResponse::decode(&encoded, request_id, 3, wrong_key, &public_key,)
                .is_err()
        );
        let mut trailing = encoded;
        trailing.push(0);
        assert!(
            AuthenticatedServiceResponse::decode(
                &trailing,
                request_id,
                3,
                engine_key_id,
                &public_key,
            )
            .is_err()
        );

        let mut invalid_body = encode_signed_service_response(
            request_id,
            4,
            &ServiceResponseBody::Health(ServiceHealthStatus::Ready),
            &signer,
        )
        .expect("health response");
        let signed_length = invalid_body.len() - ED25519_SIGNATURE_BYTES;
        invalid_body[SERVICE_REQUEST_FIXED_BYTES] = 9;
        let signature =
            signer.sign_response(&invalid_body[..signed_length]).expect("replacement signature");
        invalid_body[signed_length..].copy_from_slice(&signature);
        assert!(
            AuthenticatedServiceResponse::decode(
                &invalid_body,
                request_id,
                4,
                engine_key_id,
                &public_key,
            )
            .is_err()
        );
    }

    fn decode_hex(value: &str) -> Vec<u8> {
        let value = value.trim();
        assert_eq!(value.len() % 2, 0, "hex fixture length");
        value
            .as_bytes()
            .chunks_exact(2)
            .map(|pair| {
                let high = hex_nibble(pair[0]);
                let low = hex_nibble(pair[1]);
                (high << 4) | low
            })
            .collect()
    }

    fn hex_nibble(value: u8) -> u8 {
        match value {
            b'0'..=b'9' => value - b'0',
            b'a'..=b'f' => value - b'a' + 10,
            _ => panic!("fixture must use lowercase hexadecimal"),
        }
    }
}
