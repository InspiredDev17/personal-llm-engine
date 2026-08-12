use personalai_engine_model_format::{
    DIRECTORY_MANIFEST_HEADER_BYTES, DIRECTORY_SHARD_RECORD_BYTES, DIRECTORY_TENSOR_RECORD_BYTES,
    DirectoryManifestBuilder, DirectoryShardRecord, DirectoryTensorRecord, ParsedModelPackage,
    Quantization,
};
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
    let token_sequence = [65_u32, 258, u32::MAX]
        .iter()
        .flat_map(|token| token.to_le_bytes())
        .collect::<Vec<_>>();
    let incremental_sequence = "世界"
        .as_bytes()
        .iter()
        .flat_map(|byte| u32::from(*byte).to_le_bytes())
        .collect::<Vec<_>>();

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
    write_seed(&corpus, "safetensors_import", "truncated-header", &[1, 0, 0])?;
    write_seed(&corpus, "package_path_resolution", "valid-package-name", b"fuzz-seed.paimodel")?;
    write_seed(&corpus, "package_path_resolution", "path-traversal", b"../x.paimodel")?;
    Ok(())
}
