#![no_main]

use libfuzzer_sys::fuzz_target;
use personalai_engine_memory::MemoryPool;
use personalai_engine_sampling::{Sampler, SamplingConfig, SelectableTokenDomain};
use personalai_engine_tensor::{Tensor, TensorShape};
use personalai_engine_types::{ModelPackageHash, RequestId};

const MAX_CAMPAIGN_BYTES: usize = 4096;
const MAX_FUZZ_VOCABULARY: usize = 128;

fn u32_at(data: &[u8], offset: usize) -> u32 {
    let mut bytes = [0_u8; 4];
    let available = data.len().saturating_sub(offset).min(4);
    if available != 0 {
        bytes[..available].copy_from_slice(&data[offset..offset + available]);
    }
    u32::from_le_bytes(bytes)
}

fuzz_target!(|data: &[u8]| {
    if data.is_empty() || data.len() > MAX_CAMPAIGN_BYTES {
        return;
    }
    let temperature = f32::from_bits(u32_at(data, 0));
    let Ok(config) = SamplingConfig::new(temperature) else { return };
    let top_k =
        if data.get(4).copied().unwrap_or(0) & 1 == 0 { None } else { Some(u32_at(data, 5)) };
    let Ok(config) = config.with_top_k(top_k) else { return };
    let Ok(config) = config.with_top_p(f32::from_bits(u32_at(data, 9))) else { return };
    let Ok(config) = config.with_min_p(f32::from_bits(u32_at(data, 13))) else { return };
    let Ok(config) = config.with_repetition_penalty(f32::from_bits(u32_at(data, 17))) else {
        return;
    };
    let config = config.with_seed(u64::from(u32_at(data, 21)));
    let vocabulary = usize::from(data.get(25).copied().unwrap_or(0)).clamp(1, MAX_FUZZ_VOCABULARY);
    let mut logits = Vec::with_capacity(vocabulary);
    for index in 0..vocabulary {
        logits.push(f32::from_bits(u32_at(data, 26 + index.saturating_mul(4))));
    }
    let Ok(pool) = MemoryPool::new(1 << 20) else { return };
    let Ok(shape) = TensorShape::new([u64::try_from(vocabulary).unwrap_or(u64::MAX)]) else {
        return;
    };
    let Ok(tensor) = Tensor::from_f32_values(shape, &logits, &pool) else { return };
    let Ok(vocabulary_u32) = u32::try_from(vocabulary) else { return };
    let model_hash = ModelPackageHash::from_sha384_bytes([1; 48]);
    let tokenizer_hash = ModelPackageHash::from_sha384_bytes([2; 48]);
    let selectable = 1_usize
        .saturating_add(usize::from(data.get(26).copied().unwrap_or(0)) % vocabulary);
    let domain = if selectable == vocabulary {
        SelectableTokenDomain::entire_model_vocabulary(
            vocabulary_u32,
            vocabulary_u32,
            model_hash,
            tokenizer_hash,
        )
    } else {
        let Ok(selectable_u32) = u32::try_from(selectable) else { return };
        SelectableTokenDomain::contiguous_tokenizer_prefix(
            vocabulary_u32,
            selectable_u32,
            model_hash,
            tokenizer_hash,
        )
    };
    let Ok(domain) = domain else {
        return;
    };
    let Ok(mut sampler) = Sampler::new(
        config,
        domain,
        RequestId::new(1).expect("fixed request ID"),
        &pool,
    ) else {
        return;
    };
    let _result = sampler.sample(&tensor, &[]);
});
