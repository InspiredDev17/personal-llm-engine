#![no_main]

use libfuzzer_sys::fuzz_target;
use personalai_engine_model_format::{MODEL_REGISTRY_MAX_BYTES, SignedModelRegistryEntryV2};

fuzz_target!(|data: &[u8]| {
    if data.len() <= MODEL_REGISTRY_MAX_BYTES {
        let _result = SignedModelRegistryEntryV2::parse(data);
    }
});
