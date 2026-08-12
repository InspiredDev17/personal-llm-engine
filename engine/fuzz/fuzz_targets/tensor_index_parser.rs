#![no_main]

use libfuzzer_sys::fuzz_target;
use personalai_engine_model_format::{DIRECTORY_TENSOR_RECORD_BYTES, DirectoryTensorRecord};

fuzz_target!(|data: &[u8]| {
    if data.len() <= DIRECTORY_TENSOR_RECORD_BYTES {
        let _result = DirectoryTensorRecord::parse_exact(data);
    }
});
