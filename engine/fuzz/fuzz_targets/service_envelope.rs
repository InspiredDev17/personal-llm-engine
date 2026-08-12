#![no_main]

use ed25519_dalek::{Signer, SigningKey};
use libfuzzer_sys::fuzz_target;
use personalai_engine_crypto::{
    ED25519_SIGNATURE_BYTES, EngineResponseSigner, TrustedControlPlaneKey, TrustedControlPlaneKeys,
    control_plane_signature_message,
};
use personalai_engine_protocol::{
    AuthenticatedServiceRequest, AuthenticatedServiceResponse, SERVICE_REQUEST_FIXED_BYTES,
    attach_service_signature,
};
use personalai_engine_types::{MAX_REQUEST_BYTES, RequestId, SignerKeyId};
use std::sync::OnceLock;

const MAX_CAMPAIGN_BODY_BYTES: usize = 64 * 1024;
const REQUEST_DIRECTION: u8 = 0;
const REQUEST_MAGIC: [u8; 8] = *b"PAISVC01";
const RESPONSE_MAGIC: [u8; 8] = *b"PAISVR01";
const SERVICE_VERSION: u16 = 1;
const NONCE: u64 = 7;
const REQUEST_ID_VALUE: u128 = 9;

// These deterministic keys belong only to the excluded fuzz harness. They are
// not read from configuration, stored in the corpus, or trusted by production.
struct AuthenticationFixture {
    control_signing: SigningKey,
    control_key_id: SignerKeyId,
    trusted_control: TrustedControlPlaneKeys,
    response_signer: EngineResponseSigner,
    response_key_id: SignerKeyId,
    response_public_key: [u8; 32],
}

fn fixture() -> &'static AuthenticationFixture {
    static FIXTURE: OnceLock<AuthenticationFixture> = OnceLock::new();
    FIXTURE.get_or_init(|| {
        let control_signing = SigningKey::from_bytes(&[0x41; 32]);
        let control_key_id = SignerKeyId::from_bytes([0x42; 16]).expect("fixed control key ID");
        let trusted_control = TrustedControlPlaneKeys::single(
            TrustedControlPlaneKey::active(
                control_key_id,
                control_signing.verifying_key().to_bytes(),
            )
            .expect("fixed trusted control key"),
        )
        .expect("fixed trusted control key ring");
        let response_key_id = SignerKeyId::from_bytes([0x43; 16]).expect("fixed response key ID");
        let mut response_seed = [0x44; 32];
        let response_signer = EngineResponseSigner::from_seed(response_key_id, &mut response_seed)
            .expect("fixed response signer");
        let response_public_key = response_signer.public_key();
        AuthenticationFixture {
            control_signing,
            control_key_id,
            trusted_control,
            response_signer,
            response_key_id,
            response_public_key,
        }
    })
}

fn unsigned_envelope(
    magic: [u8; 8],
    kind: u8,
    key_id: SignerKeyId,
    body: &[u8],
) -> Option<Vec<u8>> {
    let length = SERVICE_REQUEST_FIXED_BYTES.checked_add(body.len())?;
    if length > MAX_REQUEST_BYTES.checked_sub(ED25519_SIGNATURE_BYTES)? {
        return None;
    }
    let body_length = u32::try_from(body.len()).ok()?;
    let request_id = RequestId::new(REQUEST_ID_VALUE).expect("fixed request ID");
    let mut bytes = vec![0_u8; length];
    bytes[..8].copy_from_slice(&magic);
    bytes[8..10].copy_from_slice(&SERVICE_VERSION.to_le_bytes());
    bytes[10] = kind;
    bytes[12..20].copy_from_slice(&NONCE.to_le_bytes());
    bytes[20..36].copy_from_slice(&request_id.to_bytes());
    bytes[36..52].copy_from_slice(key_id.as_bytes());
    bytes[52..56].copy_from_slice(&body_length.to_le_bytes());
    bytes[SERVICE_REQUEST_FIXED_BYTES..].copy_from_slice(body);
    Some(bytes)
}

fn exercise_request(kind: u8, body: &[u8], authentication: &AuthenticationFixture) {
    let Some(unsigned) =
        unsigned_envelope(REQUEST_MAGIC, kind, authentication.control_key_id, body)
    else {
        return;
    };
    let transcript = control_plane_signature_message(&unsigned)
        .expect("bounded canonical request transcript must encode");
    let signature = authentication.control_signing.sign(&transcript).to_bytes();
    let signed = attach_service_signature(&unsigned, &signature)
        .expect("bounded canonical request signature must attach");
    let _result = AuthenticatedServiceRequest::decode(&signed, &authentication.trusted_control);
}

fn exercise_response(kind: u8, body: &[u8], authentication: &AuthenticationFixture) {
    let Some(mut signed) =
        unsigned_envelope(RESPONSE_MAGIC, kind, authentication.response_key_id, body)
    else {
        return;
    };
    let signature = authentication
        .response_signer
        .sign_response(&signed)
        .expect("bounded canonical response must sign");
    signed.extend_from_slice(&signature);
    let _result = AuthenticatedServiceResponse::decode(
        &signed,
        RequestId::new(REQUEST_ID_VALUE).expect("fixed request ID"),
        NONCE,
        authentication.response_key_id,
        &authentication.response_public_key,
    );
}

fuzz_target!(|data: &[u8]| {
    if data.len() < 2 || data.len() > MAX_CAMPAIGN_BODY_BYTES + 2 {
        return;
    }
    let authentication = fixture();
    if data[0] & 1 == REQUEST_DIRECTION {
        exercise_request(data[1], &data[2..], authentication);
    } else {
        exercise_response(data[1], &data[2..], authentication);
    }
});
