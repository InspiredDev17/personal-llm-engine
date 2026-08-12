#![no_main]

use libfuzzer_sys::fuzz_target;
use personalai_engine_model_format::{DIRECTORY_MANIFEST_MAX_BYTES, DirectoryManifest};

fuzz_target!(|data: &[u8]| {
    if data.len() <= DIRECTORY_MANIFEST_MAX_BYTES {
        let _result = DirectoryManifest::parse(data);
    }
});
