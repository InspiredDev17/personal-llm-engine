#![no_main]

use libfuzzer_sys::fuzz_target;
use personalai_engine_memory::MemoryPool;
use personalai_engine_tokenizer::{EncodeOptions, FirstPartyBpeTokenizer, Tokenizer};
use std::sync::OnceLock;

const MAX_INPUT: usize = 64 * 1024;

fn tokenizer() -> &'static FirstPartyBpeTokenizer {
    static TOKENIZER: OnceLock<FirstPartyBpeTokenizer> = OnceLock::new();
    TOKENIZER.get_or_init(|| {
        let pool = MemoryPool::new(128 * 1024 * 1024).expect("fixed pool must validate");
        FirstPartyBpeTokenizer::parse(
            include_bytes!("../corpus/tokenizer_data/valid-byte-bpe.bin"),
            &pool,
        )
        .expect("checked-in AIPTok seed must parse")
    })
}

fuzz_target!(|data: &[u8]| {
    if data.len() <= MAX_INPUT {
        let _result = tokenizer().encode_bytes_with_options(data, EncodeOptions::byte_round_trip());
    }
});
