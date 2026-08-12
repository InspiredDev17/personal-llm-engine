//! Dependency-free owner-run microbenchmarks for AIPTok.

use personalai_engine_memory::MemoryPool;
use personalai_engine_tokenizer::{
    AipTokDefinition, AipTokMerge, AipTokVocabularyEntry, EncodeOptions, FirstPartyBpeTokenizer,
    IncrementalDecoder, SpecialTokenRole, Tokenizer,
};
use personalai_engine_types::{MAX_REQUEST_BYTES, MAX_TOKEN_COUNT};
use std::hint::black_box;
use std::time::{Duration, Instant};

const ITERATIONS: u32 = 100;

fn fixture() -> Vec<u8> {
    let mut vocabulary = (0_u32..=255)
        .map(|token_id| AipTokVocabularyEntry {
            token_id,
            bytes: vec![u8::try_from(token_id).expect("byte token must fit")],
            special_role: None,
        })
        .collect::<Vec<_>>();
    vocabulary.extend([
        AipTokVocabularyEntry { token_id: 256, bytes: b"th".to_vec(), special_role: None },
        AipTokVocabularyEntry {
            token_id: 257,
            bytes: b"<|bos|>".to_vec(),
            special_role: Some(SpecialTokenRole::BeginningOfSequence),
        },
        AipTokVocabularyEntry {
            token_id: 258,
            bytes: b"<|eos|>".to_vec(),
            special_role: Some(SpecialTokenRole::EndOfSequence),
        },
        AipTokVocabularyEntry {
            token_id: 259,
            bytes: b"<|unk|>".to_vec(),
            special_role: Some(SpecialTokenRole::Unknown),
        },
    ]);
    AipTokDefinition {
        vocabulary,
        merges: vec![AipTokMerge { rank: 0, left: 116, right: 104, result: 256 }],
        beginning_of_sequence_token: 257,
        end_of_sequence_token: 258,
        unknown_token: 259,
        maximum_input_bytes: u32::try_from(MAX_REQUEST_BYTES).expect("request limit fits"),
        maximum_produced_tokens: MAX_TOKEN_COUNT,
    }
    .encode()
    .expect("benchmark definition must encode")
}

fn measure(mut operation: impl FnMut()) -> Duration {
    let start = Instant::now();
    for _ in 0..ITERATIONS {
        operation();
    }
    start.elapsed()
}

fn report(name: &str, duration: Duration) {
    println!("{name}: {ITERATIONS} iterations in {duration:?}");
}

fn main() {
    let data = fixture();
    let pool = MemoryPool::new(512 * 1024 * 1024).expect("benchmark pool must validate");
    let tokenizer = FirstPartyBpeTokenizer::parse(&data, &pool).expect("fixture must parse");
    let short = "Summarize the following local document.";
    let long = "the quick brown fox crosses the local boundary. ".repeat(8_192);
    let generated = tokenizer
        .encode_with_options(&long, EncodeOptions::byte_round_trip())
        .expect("benchmark document must encode");

    report(
        "encode-short",
        measure(|| {
            black_box(tokenizer.encode(black_box(short)).expect("short prompt must encode"));
        }),
    );
    report(
        "encode-long",
        measure(|| {
            black_box(
                tokenizer
                    .encode_with_options(black_box(&long), EncodeOptions::byte_round_trip())
                    .expect("long document must encode"),
            );
        }),
    );
    report(
        "decode-generated",
        measure(|| {
            black_box(tokenizer.decode(black_box(&generated)).expect("tokens must decode"));
        }),
    );
    report(
        "decode-incremental",
        measure(|| {
            let mut decoder = IncrementalDecoder::new(&tokenizer);
            for token in generated.iter().copied() {
                black_box(decoder.push(token).expect("token must stream"));
            }
            decoder.finish().expect("stream must finish");
        }),
    );
    report(
        "load",
        measure(|| {
            black_box(
                FirstPartyBpeTokenizer::parse(black_box(&data), &pool).expect("data must load"),
            );
        }),
    );
}
