use personalai_engine_model_format::ParsedModelPackage;
use personalai_engine_protocol::{Frame, FrameKind};
use personalai_engine_testkit::DecoderPackageFixtureBuilder;
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

    write_seed(&corpus, "model_package", "valid-decoder.paipkg", fixture.bytes())?;
    write_seed(&corpus, "model_package", "truncated-magic", b"PAIMDL")?;
    write_seed(&corpus, "tokenizer_data", "valid-byte-bpe.bin", tokenizer)?;
    write_seed(&corpus, "tokenizer_data", "truncated-magic", b"PAIBPE")?;
    write_seed(&corpus, "protocol_frame", "valid-capability.bin", &capability)?;
    write_seed(&corpus, "protocol_frame", "truncated-magic", b"PAIE")?;
    Ok(())
}
