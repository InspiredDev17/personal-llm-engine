#![no_main]

use libfuzzer_sys::fuzz_target;
use personalai_engine_memory::MemoryPool;
use personalai_engine_tokenizer::FirstPartyBpeTokenizer;

const MAX_INPUT: usize = 8 * 1024 * 1024;

fuzz_target!(|data: &[u8]| {
    if data.len() <= MAX_INPUT {
        let pool = MemoryPool::new(128 * 1024 * 1024).expect("fixed pool must validate");
        let _result = FirstPartyBpeTokenizer::parse(data, &pool);
    }
});
