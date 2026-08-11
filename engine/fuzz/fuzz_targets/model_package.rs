#![no_main]

use libfuzzer_sys::fuzz_target;
use personalai_engine_model_format::ParsedModelPackage;

const MAX_CAMPAIGN_INPUT_BYTES: usize = 8 * 1024 * 1024;

fuzz_target!(|data: &[u8]| {
    if data.len() <= MAX_CAMPAIGN_INPUT_BYTES {
        let _result = ParsedModelPackage::parse(data);
    }
});
