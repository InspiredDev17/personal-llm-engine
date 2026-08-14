#![no_main]

use libfuzzer_sys::fuzz_target;
use personalai_engine_release::{MAX_RELEASE_MANIFEST_BYTES, ReleaseManifest};

fuzz_target!(|data: &[u8]| {
    if data.len() <= MAX_RELEASE_MANIFEST_BYTES {
        let _result = ReleaseManifest::parse(data);
    }
});
