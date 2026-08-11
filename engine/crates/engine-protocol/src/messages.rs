//! Fixed binary generation-request message and authentication envelope.

use personalai_engine_crypto::{
    ED25519_SIGNATURE_BYTES, TrustedControlPlaneKey, verify_control_plane_signature,
};
use personalai_engine_types::{
    ContextLimit, EngineError, EngineErrorKind, GenerationEvent, GenerationRequest,
    MAX_REQUEST_BYTES, ModelId, RequestId, ResourceBudget, SignerKeyId,
};

const GENERATION_REQUEST_MAGIC: [u8; 8] = *b"PAIGEN01";
const GENERATION_REQUEST_VERSION: u16 = 1;
/// Fixed bytes before the variable model ID and prompt.
pub const GENERATION_REQUEST_FIXED_BYTES: usize = 88;
const CANCELLATION_MAGIC: [u8; 8] = *b"PAICAN01";
/// Exact cancellation bytes before its signature.
pub const CANCELLATION_UNSIGNED_BYTES: usize = 52;
const GENERATION_EVENT_MAGIC: [u8; 8] = *b"PAIEVT01";

/// Authenticated generation request plus its strictly increasing connection nonce.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GenerationRequestEnvelope {
    nonce: u64,
    control_key_id: SignerKeyId,
    request: GenerationRequest,
}

impl GenerationRequestEnvelope {
    /// Decodes and strictly authenticates one exact payload.
    pub fn decode(
        payload: &[u8],
        trusted_key: &TrustedControlPlaneKey,
    ) -> Result<Self, EngineError> {
        let minimum = GENERATION_REQUEST_FIXED_BYTES
            .checked_add(ED25519_SIGNATURE_BYTES)
            .ok_or_else(|| EngineError::internal_invariant("request minimum length overflows"))?;
        if payload.len() < minimum || payload.len() > MAX_REQUEST_BYTES {
            return Err(EngineError::protocol_violation(
                "generation request message length is outside hard bounds",
            ));
        }
        if payload[..8] != GENERATION_REQUEST_MAGIC
            || read_u16(payload, 8)? != GENERATION_REQUEST_VERSION
            || read_u16(payload, 10)? != 0
            || read_u16(payload, 82)? != 0
        {
            return Err(EngineError::protocol_violation(
                "generation request magic, version, or reserved bits are invalid",
            ));
        }
        let nonce = read_u64(payload, 12)?;
        if nonce == 0 {
            return Err(EngineError::protocol_violation(
                "generation request nonce must be nonzero",
            ));
        }
        let key_id = SignerKeyId::from_bytes(read_exact::<16>(payload, 36)?)
            .map_err(|_| EngineError::protocol_violation("control-plane key ID is invalid"))?;
        let signed_length = payload
            .len()
            .checked_sub(ED25519_SIGNATURE_BYTES)
            .ok_or_else(|| EngineError::protocol_violation("request signature is truncated"))?;
        let signature = read_exact::<ED25519_SIGNATURE_BYTES>(payload, signed_length)?;
        verify_control_plane_signature(trusted_key, key_id, &payload[..signed_length], &signature)?;

        let model_length = usize::from(read_u16(payload, 80)?);
        let input_length = usize::try_from(read_u32(payload, 84)?)
            .map_err(|_| EngineError::protocol_violation("prompt length is not addressable"))?;
        let dynamic_length = model_length
            .checked_add(input_length)
            .ok_or_else(|| EngineError::protocol_violation("request dynamic length overflows"))?;
        if GENERATION_REQUEST_FIXED_BYTES
            .checked_add(dynamic_length)
            .filter(|end| *end == signed_length)
            .is_none()
        {
            return Err(EngineError::protocol_violation(
                "generation request lengths do not match the payload",
            ));
        }
        let model_end = GENERATION_REQUEST_FIXED_BYTES
            .checked_add(model_length)
            .ok_or_else(|| EngineError::protocol_violation("model ID range overflows"))?;
        let model_text = std::str::from_utf8(&payload[GENERATION_REQUEST_FIXED_BYTES..model_end])
            .map_err(|_| EngineError::protocol_violation("model ID is not UTF-8"))?;
        let input_text = std::str::from_utf8(&payload[model_end..signed_length])
            .map_err(|_| EngineError::protocol_violation("prompt is not UTF-8"))?;
        let request_id = RequestId::from_bytes(read_exact::<16>(payload, 20)?)
            .map_err(|_| EngineError::protocol_violation("request ID is invalid"))?;
        let model_id = ModelId::new(model_text)
            .map_err(|_| EngineError::protocol_violation("model ID is invalid"))?;
        let context_limit = ContextLimit::new(read_u32(payload, 56)?)
            .map_err(|_| EngineError::protocol_violation("context limit is invalid"))?;
        let budget = ResourceBudget::new(
            read_u64(payload, 60)?,
            read_u64(payload, 68)?,
            read_u32(payload, 76)?,
        )
        .map_err(|_| EngineError::protocol_violation("resource budget is invalid"))?;
        let request = GenerationRequest::new(
            request_id,
            model_id,
            input_text,
            read_u32(payload, 52)?,
            context_limit,
            budget,
        )
        .map_err(|_| EngineError::protocol_violation("generation request fields are invalid"))?;
        Ok(Self { nonce, control_key_id: key_id, request })
    }

    /// Builds the exact bytes that the Go control plane must sign.
    pub fn unsigned_bytes(
        request: &GenerationRequest,
        nonce: u64,
        control_key_id: SignerKeyId,
    ) -> Result<Vec<u8>, EngineError> {
        if nonce == 0 {
            return Err(EngineError::invalid_input("generation request nonce must be nonzero"));
        }
        let model = request.model_id().as_str().as_bytes();
        let input = request.input_text().as_bytes();
        let length = GENERATION_REQUEST_FIXED_BYTES
            .checked_add(model.len())
            .and_then(|value| value.checked_add(input.len()))
            .filter(|value| *value <= MAX_REQUEST_BYTES - ED25519_SIGNATURE_BYTES)
            .ok_or_else(|| EngineError::resource_exhaustion("generation message is too large"))?;
        let mut bytes = zeroed(length)?;
        bytes[..8].copy_from_slice(&GENERATION_REQUEST_MAGIC);
        write_u16(&mut bytes, 8, GENERATION_REQUEST_VERSION)?;
        write_u64(&mut bytes, 12, nonce)?;
        write_exact(&mut bytes, 20, &request.request_id().to_bytes())?;
        write_exact(&mut bytes, 36, control_key_id.as_bytes())?;
        write_u32(&mut bytes, 52, request.max_output_tokens())?;
        write_u32(&mut bytes, 56, request.context_limit().get())?;
        let budget = request.resource_budget();
        write_u64(&mut bytes, 60, budget.max_memory_bytes())?;
        write_u64(&mut bytes, 68, budget.max_kv_cache_bytes())?;
        write_u32(&mut bytes, 76, budget.max_concurrent_sequences())?;
        write_u16(
            &mut bytes,
            80,
            u16::try_from(model.len())
                .map_err(|_| EngineError::invalid_input("model ID exceeds its message field"))?,
        )?;
        write_u32(
            &mut bytes,
            84,
            u32::try_from(input.len())
                .map_err(|_| EngineError::invalid_input("prompt exceeds its message field"))?,
        )?;
        let model_end = GENERATION_REQUEST_FIXED_BYTES + model.len();
        bytes[GENERATION_REQUEST_FIXED_BYTES..model_end].copy_from_slice(model);
        bytes[model_end..].copy_from_slice(input);
        Ok(bytes)
    }

    /// Returns the anti-replay nonce.
    pub const fn nonce(&self) -> u64 {
        self.nonce
    }

    /// Returns the authenticated control-plane key ID.
    pub const fn control_key_id(&self) -> SignerKeyId {
        self.control_key_id
    }

    /// Returns the validated generation request.
    pub const fn request(&self) -> &GenerationRequest {
        &self.request
    }

    /// Consumes the envelope and returns its generation request.
    pub fn into_request(self) -> GenerationRequest {
        self.request
    }
}

/// Appends an exact Ed25519 signature to previously constructed unsigned bytes.
pub fn attach_generation_signature(
    unsigned_bytes: &[u8],
    signature: &[u8; ED25519_SIGNATURE_BYTES],
) -> Result<Vec<u8>, EngineError> {
    if unsigned_bytes.len() < GENERATION_REQUEST_FIXED_BYTES
        || unsigned_bytes.len() > MAX_REQUEST_BYTES - ED25519_SIGNATURE_BYTES
    {
        return Err(EngineError::invalid_input(
            "unsigned generation message length is outside hard bounds",
        ));
    }
    let length = unsigned_bytes
        .len()
        .checked_add(signature.len())
        .ok_or_else(|| EngineError::resource_exhaustion("signed message length overflows"))?;
    let mut payload = Vec::new();
    payload
        .try_reserve_exact(length)
        .map_err(|_| EngineError::resource_exhaustion("signed message allocation failed"))?;
    payload.extend_from_slice(unsigned_bytes);
    payload.extend_from_slice(signature);
    Ok(payload)
}

/// Authenticated cancellation plus its connection anti-replay nonce.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CancellationEnvelope {
    nonce: u64,
    control_key_id: SignerKeyId,
    request_id: RequestId,
}

impl CancellationEnvelope {
    /// Decodes and strictly authenticates one cancellation payload.
    pub fn decode(
        payload: &[u8],
        trusted_key: &TrustedControlPlaneKey,
    ) -> Result<Self, EngineError> {
        let exact_length = CANCELLATION_UNSIGNED_BYTES
            .checked_add(ED25519_SIGNATURE_BYTES)
            .ok_or_else(|| EngineError::internal_invariant("cancellation length overflows"))?;
        if payload.len() != exact_length
            || payload[..8] != CANCELLATION_MAGIC
            || read_u16(payload, 8)? != 1
            || read_u16(payload, 10)? != 0
        {
            return Err(EngineError::protocol_violation(
                "cancellation message structure is invalid",
            ));
        }
        let nonce = read_u64(payload, 12)?;
        if nonce == 0 {
            return Err(EngineError::protocol_violation("cancellation nonce must be nonzero"));
        }
        let request_id = RequestId::from_bytes(read_exact::<16>(payload, 20)?)
            .map_err(|_| EngineError::protocol_violation("cancellation request ID is invalid"))?;
        let key_id = SignerKeyId::from_bytes(read_exact::<16>(payload, 36)?)
            .map_err(|_| EngineError::protocol_violation("control-plane key ID is invalid"))?;
        let signature =
            read_exact::<ED25519_SIGNATURE_BYTES>(payload, CANCELLATION_UNSIGNED_BYTES)?;
        verify_control_plane_signature(
            trusted_key,
            key_id,
            &payload[..CANCELLATION_UNSIGNED_BYTES],
            &signature,
        )?;
        Ok(Self { nonce, control_key_id: key_id, request_id })
    }

    /// Builds the exact cancellation bytes that the Go control plane signs.
    pub fn unsigned_bytes(
        request_id: RequestId,
        nonce: u64,
        control_key_id: SignerKeyId,
    ) -> Result<[u8; CANCELLATION_UNSIGNED_BYTES], EngineError> {
        if nonce == 0 {
            return Err(EngineError::invalid_input("cancellation nonce must be nonzero"));
        }
        let mut bytes = [0_u8; CANCELLATION_UNSIGNED_BYTES];
        bytes[..8].copy_from_slice(&CANCELLATION_MAGIC);
        write_u16(&mut bytes, 8, 1)?;
        write_u64(&mut bytes, 12, nonce)?;
        write_exact(&mut bytes, 20, &request_id.to_bytes())?;
        write_exact(&mut bytes, 36, control_key_id.as_bytes())?;
        Ok(bytes)
    }

    /// Returns the anti-replay nonce.
    pub const fn nonce(self) -> u64 {
        self.nonce
    }

    /// Returns the authenticated control-plane key identifier.
    pub const fn control_key_id(self) -> SignerKeyId {
        self.control_key_id
    }

    /// Returns the request to cancel.
    pub const fn request_id(self) -> RequestId {
        self.request_id
    }
}

/// Appends an exact signature to a cancellation message.
pub fn attach_cancellation_signature(
    unsigned_bytes: &[u8; CANCELLATION_UNSIGNED_BYTES],
    signature: &[u8; ED25519_SIGNATURE_BYTES],
) -> Result<Vec<u8>, EngineError> {
    let length = CANCELLATION_UNSIGNED_BYTES
        .checked_add(ED25519_SIGNATURE_BYTES)
        .ok_or_else(|| EngineError::internal_invariant("cancellation length overflows"))?;
    let mut payload = Vec::new();
    payload
        .try_reserve_exact(length)
        .map_err(|_| EngineError::resource_exhaustion("cancellation allocation failed"))?;
    payload.extend_from_slice(unsigned_bytes);
    payload.extend_from_slice(signature);
    Ok(payload)
}

/// Encodes one engine event into the fixed binary protocol schema.
pub fn encode_generation_event(event: &GenerationEvent) -> Result<Vec<u8>, EngineError> {
    let (kind, sequence, dynamic_length) = match event {
        GenerationEvent::Started { sequence_id, .. } => (1_u8, Some(*sequence_id), 0_usize),
        GenerationEvent::Token { sequence_id, text, .. } => {
            (2, Some(*sequence_id), text.as_str().len())
        }
        GenerationEvent::Completed { sequence_id, .. } => (3, Some(*sequence_id), 0),
        GenerationEvent::Failed { sequence_id, .. } => (4, *sequence_id, 0),
        _ => {
            return Err(EngineError::not_implemented(
                "generation event encoding is not implemented",
            ));
        }
    };
    let fixed_length: usize = match kind {
        1 => 36,
        2 => 44,
        3 => 48,
        4 => 40,
        _ => return Err(EngineError::internal_invariant("event kind is unreachable")),
    };
    let length = fixed_length
        .checked_add(dynamic_length)
        .ok_or_else(|| EngineError::resource_exhaustion("event length overflows"))?;
    let mut bytes = zeroed(length)?;
    bytes[..8].copy_from_slice(&GENERATION_EVENT_MAGIC);
    write_u16(&mut bytes, 8, 1)?;
    bytes[10] = kind;
    bytes[11] = u8::from(sequence.is_some());
    write_exact(&mut bytes, 12, &event.request_id().to_bytes())?;
    write_u64(&mut bytes, 28, sequence.map_or(0, personalai_engine_types::SequenceId::get))?;
    match event {
        GenerationEvent::Started { .. } => {}
        GenerationEvent::Token { token_id, text, .. } => {
            write_u32(&mut bytes, 36, token_id.get())?;
            write_u32(
                &mut bytes,
                40,
                u32::try_from(text.as_str().len())
                    .map_err(|_| EngineError::resource_exhaustion("event text exceeds u32"))?,
            )?;
            bytes[44..].copy_from_slice(text.as_str().as_bytes());
        }
        GenerationEvent::Completed { usage, .. } => {
            write_u32(&mut bytes, 36, usage.input_tokens())?;
            write_u32(&mut bytes, 40, usage.output_tokens())?;
            write_u32(&mut bytes, 44, usage.total_tokens())?;
        }
        GenerationEvent::Failed { error, .. } => {
            write_u16(&mut bytes, 36, error_kind_code(error.kind())?)?;
        }
        _ => {
            return Err(EngineError::not_implemented(
                "generation event encoding is not implemented",
            ));
        }
    }
    Ok(bytes)
}

fn error_kind_code(kind: EngineErrorKind) -> Result<u16, EngineError> {
    let code = match kind {
        EngineErrorKind::InvalidInput => 1,
        EngineErrorKind::UnsupportedArchitecture => 2,
        EngineErrorKind::UnsupportedTokenizer => 3,
        EngineErrorKind::CorruptModelPackage => 4,
        EngineErrorKind::SignatureFailure => 5,
        EngineErrorKind::ResourceExhaustion => 6,
        EngineErrorKind::Cancellation => 7,
        EngineErrorKind::BackendFailure => 8,
        EngineErrorKind::InternalInvariantViolation => 9,
        EngineErrorKind::AuthenticationFailure => 10,
        EngineErrorKind::ProtocolViolation => 11,
        EngineErrorKind::NotImplemented => 12,
        _ => return Err(EngineError::not_implemented("error kind encoding is not implemented")),
    };
    Ok(code)
}

fn read_u16(input: &[u8], offset: usize) -> Result<u16, EngineError> {
    Ok(u16::from_le_bytes(read_exact::<2>(input, offset)?))
}

fn read_u32(input: &[u8], offset: usize) -> Result<u32, EngineError> {
    Ok(u32::from_le_bytes(read_exact::<4>(input, offset)?))
}

fn read_u64(input: &[u8], offset: usize) -> Result<u64, EngineError> {
    Ok(u64::from_le_bytes(read_exact::<8>(input, offset)?))
}

fn read_exact<const N: usize>(input: &[u8], offset: usize) -> Result<[u8; N], EngineError> {
    let end = offset
        .checked_add(N)
        .ok_or_else(|| EngineError::protocol_violation("message field offset overflows"))?;
    input
        .get(offset..end)
        .ok_or_else(|| EngineError::protocol_violation("message field is truncated"))?
        .try_into()
        .map_err(|_| EngineError::protocol_violation("message field length is invalid"))
}

fn zeroed(length: usize) -> Result<Vec<u8>, EngineError> {
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(length)
        .map_err(|_| EngineError::resource_exhaustion("message allocation failed"))?;
    bytes.resize(length, 0);
    Ok(bytes)
}

fn write_u16(bytes: &mut [u8], offset: usize, value: u16) -> Result<(), EngineError> {
    write_exact(bytes, offset, &value.to_le_bytes())
}

fn write_u32(bytes: &mut [u8], offset: usize, value: u32) -> Result<(), EngineError> {
    write_exact(bytes, offset, &value.to_le_bytes())
}

fn write_u64(bytes: &mut [u8], offset: usize, value: u64) -> Result<(), EngineError> {
    write_exact(bytes, offset, &value.to_le_bytes())
}

fn write_exact(bytes: &mut [u8], offset: usize, value: &[u8]) -> Result<(), EngineError> {
    let end = offset
        .checked_add(value.len())
        .ok_or_else(|| EngineError::internal_invariant("message field range overflows"))?;
    bytes
        .get_mut(offset..end)
        .ok_or_else(|| EngineError::internal_invariant("message field range is unavailable"))?
        .copy_from_slice(value);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ed25519_dalek::{Signer, SigningKey};
    use personalai_engine_crypto::control_plane_signature_message;

    #[test]
    fn exact_authenticated_request_round_trips_and_tampering_fails() {
        let signing_key = SigningKey::from_bytes(&[0x33; 32]);
        let key_id = SignerKeyId::from_bytes([0x44; 16]).expect("key id must validate");
        let trusted =
            TrustedControlPlaneKey::active(key_id, signing_key.verifying_key().to_bytes())
                .expect("control key must validate");
        let request = request();
        let unsigned = GenerationRequestEnvelope::unsigned_bytes(&request, 1, key_id)
            .expect("unsigned message must encode");
        let message = control_plane_signature_message(&unsigned).expect("transcript must build");
        let signature = signing_key.sign(&message).to_bytes();
        let payload =
            attach_generation_signature(&unsigned, &signature).expect("signature must attach");
        let decoded = GenerationRequestEnvelope::decode(&payload, &trusted)
            .expect("signed request must verify");
        assert_eq!(decoded.nonce(), 1);
        assert_eq!(decoded.request(), &request);

        let mut tampered = payload;
        tampered[GENERATION_REQUEST_FIXED_BYTES] ^= 1;
        assert!(matches!(
            GenerationRequestEnvelope::decode(&tampered, &trusted),
            Err(EngineError::AuthenticationFailure(_))
        ));
    }

    #[test]
    fn cancellation_authentication_and_event_encoding_are_exact() {
        let signing_key = SigningKey::from_bytes(&[0x33; 32]);
        let key_id = SignerKeyId::from_bytes([0x44; 16]).expect("key id must validate");
        let trusted =
            TrustedControlPlaneKey::active(key_id, signing_key.verifying_key().to_bytes())
                .expect("control key must validate");
        let request_id = RequestId::new(9).expect("request id must validate");
        let unsigned = CancellationEnvelope::unsigned_bytes(request_id, 2, key_id)
            .expect("cancellation must encode");
        let message = control_plane_signature_message(&unsigned).expect("transcript must build");
        let signature = signing_key.sign(&message).to_bytes();
        let payload =
            attach_cancellation_signature(&unsigned, &signature).expect("signature must attach");
        let decoded = CancellationEnvelope::decode(&payload, &trusted)
            .expect("cancellation must authenticate");
        assert_eq!(decoded.nonce(), 2);
        assert_eq!(decoded.request_id(), request_id);

        let event = GenerationEvent::Started {
            request_id,
            sequence_id: personalai_engine_types::SequenceId::new(3)
                .expect("sequence ID must validate"),
        };
        let encoded = encode_generation_event(&event).expect("event must encode");
        assert_eq!(&encoded[..8], b"PAIEVT01");
        assert_eq!(encoded.len(), 36);
    }

    fn request() -> GenerationRequest {
        GenerationRequest::new(
            RequestId::new(1).expect("request id must validate"),
            ModelId::new("decoder-test").expect("model id must validate"),
            "hello",
            4,
            ContextLimit::new(16).expect("context must validate"),
            ResourceBudget::new(1 << 20, 1 << 18, 1).expect("budget must validate"),
        )
        .expect("request must validate")
    }
}
