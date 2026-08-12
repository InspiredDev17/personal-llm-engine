#![no_main]

use libfuzzer_sys::fuzz_target;
use personalai_engine_loader::validate_registered_package_path;
use std::path::Path;

const MAX_CAMPAIGN_INPUT_BYTES: usize = 1024;

fuzz_target!(|data: &[u8]| {
    if data.len() <= MAX_CAMPAIGN_INPUT_BYTES {
        if let Ok(text) = std::str::from_utf8(data) {
            let _result = validate_registered_package_path(Path::new(text));
        }
    }
});
