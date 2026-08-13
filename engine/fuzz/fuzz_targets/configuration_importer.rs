#![no_main]

use libfuzzer_sys::fuzz_target;
use personalai_engine_model_importer::convert_aiptok_metadata;

const MAX_CAMPAIGN_BYTES: usize = 256 * 1024;

fuzz_target!(|data: &[u8]| {
    if data.len() <= MAX_CAMPAIGN_BYTES {
        let _result = convert_aiptok_metadata(data);
    }
});
