#![no_main]

use libfuzzer_sys::fuzz_target;
use personalai_engine_memory::MemoryPool;
use personalai_engine_tokenizer::FirstPartyBpeTokenizer;

const MAX_CAMPAIGN_INPUT_BYTES: usize = 8 * 1024 * 1024;
const FUZZ_MEMORY_POOL_BYTES: u64 = 128 * 1024 * 1024;

fuzz_target!(|data: &[u8]| {
    if data.len() <= MAX_CAMPAIGN_INPUT_BYTES {
        let pool = MemoryPool::new(FUZZ_MEMORY_POOL_BYTES)
            .expect("the fixed fuzz memory ceiling must be valid");
        let _result = FirstPartyBpeTokenizer::parse(data, &pool);
    }
});
