#![no_main]

use libfuzzer_sys::fuzz_target;
use personalai_engine_protocol::Frame;
use personalai_engine_types::MAX_FRAME_BYTES;

fuzz_target!(|data: &[u8]| {
    if data.len() <= MAX_FRAME_BYTES {
        let _inbound = Frame::decode_inbound_exact(data);
        let _outbound = Frame::decode_outbound_exact(data);
    }
});
