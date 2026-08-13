#![no_main]

use ed25519_dalek::{Signer, SigningKey};
use libfuzzer_sys::fuzz_target;
use personalai_engine_crypto::{
    ED25519_SIGNATURE_BYTES, TrustedControlPlaneKey, TrustedControlPlaneKeys,
    control_plane_signature_message,
};
use personalai_engine_protocol::{
    GENERATION_REQUEST_FIXED_BYTES, GenerationRequestEnvelope, attach_generation_signature,
};
use personalai_engine_types::{MAX_REQUEST_BYTES, SignerKeyId};
use std::sync::OnceLock;

const MAX_CAMPAIGN_BYTES: usize = 256 * 1024;
const MODEL_ID: &[u8] = b"fuzz-model";

struct AuthenticationFixture {
    signing: SigningKey,
    key_id: SignerKeyId,
    trusted: TrustedControlPlaneKeys,
}

fn fixture() -> &'static AuthenticationFixture {
    static FIXTURE: OnceLock<AuthenticationFixture> = OnceLock::new();
    FIXTURE.get_or_init(|| {
        let signing = SigningKey::from_bytes(&[0x51; 32]);
        let key_id = SignerKeyId::from_bytes([0x52; 16]).expect("fixed fuzz key ID");
        let trusted = TrustedControlPlaneKeys::single(
            TrustedControlPlaneKey::active(key_id, signing.verifying_key().to_bytes())
                .expect("fixed fuzz key"),
        )
        .expect("fixed fuzz key ring");
        AuthenticationFixture { signing, key_id, trusted }
    })
}

fuzz_target!(|data: &[u8]| {
    if data.len() > MAX_CAMPAIGN_BYTES {
        return;
    }
    let authentication = fixture();
    let maximum_unsigned = MAX_REQUEST_BYTES.saturating_sub(ED25519_SIGNATURE_BYTES);
    let requested_length =
        GENERATION_REQUEST_FIXED_BYTES.saturating_add(MODEL_ID.len()).saturating_add(data.len());
    if requested_length > maximum_unsigned {
        return;
    }
    let mut unsigned = vec![0_u8; requested_length];
    unsigned[..8].copy_from_slice(b"PAIGEN01");
    unsigned[8..10].copy_from_slice(&1_u16.to_le_bytes());
    unsigned[12..20].copy_from_slice(&1_u64.to_le_bytes());
    unsigned[20..36].copy_from_slice(&1_u128.to_le_bytes());
    unsigned[36..52].copy_from_slice(authentication.key_id.as_bytes());
    unsigned[52..56].copy_from_slice(&1_u32.to_le_bytes());
    unsigned[56..60].copy_from_slice(&16_u32.to_le_bytes());
    unsigned[60..68].copy_from_slice(&(1_u64 << 20).to_le_bytes());
    unsigned[68..76].copy_from_slice(&(1_u64 << 19).to_le_bytes());
    unsigned[76..80].copy_from_slice(&1_u32.to_le_bytes());
    unsigned[80..82].copy_from_slice(
        &u16::try_from(MODEL_ID.len()).expect("fixed model ID length").to_le_bytes(),
    );
    unsigned[84..88].copy_from_slice(&u32::try_from(data.len()).unwrap_or(u32::MAX).to_le_bytes());
    let model_end = GENERATION_REQUEST_FIXED_BYTES + MODEL_ID.len();
    unsigned[GENERATION_REQUEST_FIXED_BYTES..model_end].copy_from_slice(MODEL_ID);
    unsigned[model_end..].copy_from_slice(data);
    let transcript = control_plane_signature_message(&unsigned)
        .expect("bounded fuzz request transcript must encode");
    let signature = authentication.signing.sign(&transcript).to_bytes();
    let signed = attach_generation_signature(&unsigned, &signature)
        .expect("bounded fuzz signature must attach");
    let _result = GenerationRequestEnvelope::decode(&signed, &authentication.trusted);
});
