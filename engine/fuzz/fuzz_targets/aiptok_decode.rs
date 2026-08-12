#![no_main]

use libfuzzer_sys::fuzz_target;
use personalai_engine_memory::MemoryPool;
use personalai_engine_tokenizer::{FirstPartyBpeTokenizer, Tokenizer};
use std::sync::OnceLock;

const MAX_INPUT: usize = 256 * 1024;

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
        let tokens = data
            .chunks_exact(4)
            .map(|bytes| u32::from_le_bytes(bytes.try_into().expect("chunk is exact")))
            .collect::<Vec<_>>();
        let _result = tokenizer().decode_bytes(&tokens);
    }
});
