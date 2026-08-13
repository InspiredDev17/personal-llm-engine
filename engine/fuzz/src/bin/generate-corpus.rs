use personalai_engine_model_format::{
    DIRECTORY_MANIFEST_HEADER_BYTES, DIRECTORY_SHARD_RECORD_BYTES, DIRECTORY_TENSOR_RECORD_BYTES,
    DirectoryManifestBuilder, DirectoryShardRecord, DirectoryTensorRecord, ParsedModelPackage,
    Quantization,
};
use personalai_engine_model_importer::convert_aiptok_metadata;
use personalai_engine_protocol::{Frame, FrameKind};
use personalai_engine_testkit::DecoderPackageFixtureBuilder;
use personalai_engine_types::{ContextLimit, DType, ModelId, ModelPackageHash, TensorShape};
use std::error::Error;
use std::fs;
use std::path::Path;

fn write_seed(root: &Path, target: &str, name: &str, bytes: &[u8]) -> Result<(), Box<dyn Error>> {
    let directory = root.join(target);
    fs::create_dir_all(&directory)?;
    fs::write(directory.join(name), bytes)?;
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let corpus = Path::new(env!("CARGO_MANIFEST_DIR")).join("corpus");
    let fixture = DecoderPackageFixtureBuilder::new().build()?;
    let parsed = ParsedModelPackage::parse(fixture.bytes())?;
    let tokenizer = parsed.tokenizer_data(fixture.bytes())?;
    let capability = Frame::new(FrameKind::Capability, b"capabilities".to_vec())?.encode()?;
    let hash = |byte| ModelPackageHash::from_sha384_bytes([byte; ModelPackageHash::BYTE_LEN]);
    let tensor = DirectoryTensorRecord::new(
        "token_embedding.weight",
        DType::F32,
        TensorShape::new([4, 4])?,
        0,
        0,
        64,
        64,
        Quantization::None,
        hash(3),
    )?;
    let manifest = DirectoryManifestBuilder::new(
        ModelId::new("fuzz-seed")?,
        (0, 1, 0),
        ContextLimit::new(128)?,
        256,
        4,
        1,
        1,
        1,
        8,
        10_000.0,
        1.0,
        DType::F32,
        Quantization::None,
        false,
        hash(1),
        vec![DirectoryShardRecord::new(0, 64, hash(2))?],
        vec![tensor],
    )?
    .build_bytes()?;
    let tensor_start = DIRECTORY_MANIFEST_HEADER_BYTES + DIRECTORY_SHARD_RECORD_BYTES;
    let tensor_end = tensor_start + DIRECTORY_TENSOR_RECORD_BYTES;
    let safetensors_header = br#"{"seed":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
    let mut safetensors = Vec::new();
    safetensors.extend_from_slice(&u64::try_from(safetensors_header.len())?.to_le_bytes());
    safetensors.extend_from_slice(safetensors_header);
    safetensors.extend_from_slice(&1.0_f32.to_le_bytes());
    let safetensors_f16_header = br#"{"seed":{"dtype":"F16","shape":[1],"data_offsets":[0,2]}}"#;
    let mut safetensors_f16 = Vec::new();
    safetensors_f16.extend_from_slice(&u64::try_from(safetensors_f16_header.len())?.to_le_bytes());
    safetensors_f16.extend_from_slice(safetensors_f16_header);
    safetensors_f16.extend_from_slice(&0x3c00_u16.to_le_bytes());
    let safetensors_bf16_header = br#"{"seed":{"dtype":"BF16","shape":[1],"data_offsets":[0,2]}}"#;
    let mut safetensors_bf16 = Vec::new();
    safetensors_bf16
        .extend_from_slice(&u64::try_from(safetensors_bf16_header.len())?.to_le_bytes());
    safetensors_bf16.extend_from_slice(safetensors_bf16_header);
    safetensors_bf16.extend_from_slice(&0x3f80_u16.to_le_bytes());
    let token_sequence =
        [65_u32, 258, u32::MAX].iter().flat_map(|token| token.to_le_bytes()).collect::<Vec<_>>();
    let incremental_sequence = "世界"
        .as_bytes()
        .iter()
        .flat_map(|byte| u32::from(*byte).to_le_bytes())
        .collect::<Vec<_>>();
    let mut model_status_request = vec![0, 18];
    model_status_request.extend_from_slice(b"fuzz-seed");
    let mut model_status_response = vec![1, 18];
    let mut model_status_body = [0_u8; 44];
    model_status_body[0] = 1;
    model_status_body[1] = 1;
    model_status_body[4..12].copy_from_slice(&0x31_u64.to_le_bytes());
    model_status_body[20..24].copy_from_slice(&1_u32.to_le_bytes());
    model_status_body[24..28].copy_from_slice(&8_u32.to_le_bytes());
    model_status_body[36..44].copy_from_slice(&(64_u64 << 20).to_le_bytes());
    model_status_response.extend_from_slice(&model_status_body);
    let mut tokenizer_metadata = String::from(
        "AIPTOK-METADATA-V1\nfamily=byte-level-bpe-v1\nnormalizer=preserve-bytes\n\
pre_tokenizer=none\ndecoder=concatenate-bytes-skip-special\nunknown_policy=byte-fallback\n\
chat_template=none\nvocabulary_size=259\nbos_token_id=256\neos_token_id=257\n\
unknown_token_id=258\nmaximum_input_bytes=4096\nmaximum_produced_tokens=4096\n",
    );
    for token_id in 0_u32..256 {
        tokenizer_metadata.push_str(&format!("token {token_id} ordinary {token_id:02x}\n"));
    }
    tokenizer_metadata.push_str(
        "token 256 bos 3c7c626f737c3e\ntoken 257 eos 3c7c656f737c3e\ntoken 258 unknown 3c7c756e6b7c3e\nmerge_count=0\n",
    );
    let _validated_tokenizer_seed = convert_aiptok_metadata(tokenizer_metadata.as_bytes())?;
    let mut sampling_parameters = vec![0_u8; 42];
    sampling_parameters[0..4].copy_from_slice(&1.0_f32.to_bits().to_le_bytes());
    sampling_parameters[9..13].copy_from_slice(&1.0_f32.to_bits().to_le_bytes());
    sampling_parameters[13..17].copy_from_slice(&0.0_f32.to_bits().to_le_bytes());
    sampling_parameters[17..21].copy_from_slice(&1.0_f32.to_bits().to_le_bytes());
    sampling_parameters[25] = 4;
    for (index, value) in [0.0_f32, 1.0, 2.0, 3.0].iter().enumerate() {
        let start = 26 + index * 4;
        sampling_parameters[start..start + 4].copy_from_slice(&value.to_bits().to_le_bytes());
    }

    write_seed(&corpus, "model_package", "valid-decoder.paipkg", fixture.bytes())?;
    write_seed(&corpus, "model_package", "truncated-magic", b"PAIMDL")?;
    write_seed(&corpus, "tokenizer_data", "valid-byte-bpe.bin", tokenizer)?;
    write_seed(&corpus, "tokenizer_data", "truncated-magic", b"AIPTOK")?;
    write_seed(&corpus, "aiptok_parser", "valid.aiptok", tokenizer)?;
    write_seed(&corpus, "aiptok_parser", "truncated-magic", b"AIPTOK")?;
    write_seed(&corpus, "aiptok_encode", "utf8-and-null", b"hello \xf0\x9f\xa6\x80\0")?;
    write_seed(&corpus, "aiptok_decode", "mixed-token-ids", &token_sequence)?;
    write_seed(&corpus, "aiptok_incremental", "unicode-byte-tokens", &incremental_sequence)?;
    write_seed(&corpus, "protocol_frame", "valid-capability.bin", &capability)?;
    write_seed(&corpus, "protocol_frame", "truncated-magic", b"PAIE")?;
    write_seed(&corpus, "service_envelope", "valid-health-request", &[0, 12])?;
    write_seed(&corpus, "service_envelope", "valid-health-response", &[1, 12, 1])?;
    write_seed(&corpus, "service_envelope", "valid-model-status-request", &model_status_request)?;
    write_seed(&corpus, "service_envelope", "valid-model-status-response", &model_status_response)?;
    write_seed(&corpus, "service_envelope", "malformed-tokenize-request", &[0, 8, 5, 0, b'x'])?;
    write_seed(&corpus, "manifest_parser", "valid-directory.aipm", &manifest)?;
    write_seed(&corpus, "manifest_parser", "truncated-magic", b"PAIDIR")?;
    write_seed(
        &corpus,
        "tensor_index_parser",
        "valid-tensor-record.aiptidx",
        &manifest[tensor_start..tensor_end],
    )?;
    write_seed(&corpus, "tensor_index_parser", "empty", b"")?;
    write_seed(&corpus, "safetensors_import", "valid-f32.safetensors", &safetensors)?;
    write_seed(&corpus, "safetensors_import", "valid-f16.safetensors", &safetensors_f16)?;
    write_seed(&corpus, "safetensors_import", "valid-bf16.safetensors", &safetensors_bf16)?;
    write_seed(&corpus, "safetensors_import", "truncated-header", &[1, 0, 0])?;
    write_seed(&corpus, "package_path_resolution", "valid-package-name", b"fuzz-seed.paimodel")?;
    write_seed(&corpus, "package_path_resolution", "path-traversal", b"../x.paimodel")?;
    let registry = include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../tests/model-foundry-registry-v2/positive-0002.registry.json"
    ));
    write_seed(&corpus, "model_registry_v2", "valid-entry.json", registry)?;
    write_seed(&corpus, "model_registry_v2", "truncated-entry", b"{\"entry\":\n")?;
    write_seed(
        &corpus,
        "configuration_importer",
        "valid-aiptok-metadata.txt",
        tokenizer_metadata.as_bytes(),
    )?;
    write_seed(
        &corpus,
        "configuration_importer",
        "unsupported-template.txt",
        b"AIPTOK-METADATA-V1\nfamily=byte-level-bpe-v1\nnormalizer=script\n",
    )?;
    write_seed(&corpus, "generation_request", "valid-prompt", b"local prompt")?;
    write_seed(&corpus, "generation_request", "invalid-utf8-prompt", &[0xff, 0xfe])?;
    write_seed(&corpus, "sampling_parameters", "valid-stochastic.bin", &sampling_parameters)?;
    write_seed(
        &corpus,
        "sampling_parameters",
        "nonfinite-temperature.bin",
        &f32::NAN.to_bits().to_le_bytes(),
    )?;
    Ok(())
}
