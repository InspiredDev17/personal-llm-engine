use sovereign_ai_contracts::document::validate_document;
use sovereign_ai_contracts::{ContractError, Result};
use std::env;
use std::fs::{self, File};
use std::io::{self, Read};
use std::path::PathBuf;

const MAX_DOCUMENT_BYTES: u64 = 1 << 20;

fn main() {
    if let Err(error) = run() {
        eprintln!("contract-validate: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let path = parse_path()?;
    let source = read_input(&path)?;
    let result = validate_document(&source)?;
    println!("{}", result.canonical_json);
    Ok(())
}

fn parse_path() -> Result<PathBuf> {
    let mut arguments = env::args_os().skip(1);
    let path = arguments
        .next()
        .map(PathBuf::from)
        .ok_or_else(|| ContractError::field("arguments", "usage: contract-validate FILE"))?;
    if arguments.next().is_some() {
        return Err(ContractError::field(
            "arguments",
            "usage: contract-validate FILE",
        ));
    }
    Ok(path)
}

fn read_input(path: &PathBuf) -> Result<Vec<u8>> {
    if path.as_os_str() == "-" {
        return read_bounded(io::stdin().lock(), "standard input");
    }
    let metadata = fs::symlink_metadata(path)?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err(ContractError::field(
            "input",
            "must be a regular non-symbolic-link file",
        ));
    }
    if metadata.len() > MAX_DOCUMENT_BYTES {
        return Err(ContractError::field(
            "input",
            format!("exceeds the {MAX_DOCUMENT_BYTES}-byte limit"),
        ));
    }
    let file = File::open(path)?;
    let opened = file.metadata()?;
    if !opened.is_file() || !same_file(&metadata, &opened) {
        return Err(ContractError::field(
            "input",
            "changed while it was being opened",
        ));
    }
    read_bounded(file, "input")
}

#[cfg(unix)]
fn same_file(left: &fs::Metadata, right: &fs::Metadata) -> bool {
    use std::os::unix::fs::MetadataExt;
    left.dev() == right.dev() && left.ino() == right.ino()
}

#[cfg(not(unix))]
fn same_file(left: &fs::Metadata, right: &fs::Metadata) -> bool {
    left.len() == right.len() && left.modified().ok() == right.modified().ok()
}

fn read_bounded(mut reader: impl Read, field: &'static str) -> Result<Vec<u8>> {
    let mut source = Vec::new();
    reader
        .by_ref()
        .take(MAX_DOCUMENT_BYTES + 1)
        .read_to_end(&mut source)?;
    if source.len() as u64 > MAX_DOCUMENT_BYTES {
        return Err(ContractError::field(
            field,
            format!("exceeds the {MAX_DOCUMENT_BYTES}-byte limit"),
        ));
    }
    Ok(source)
}

#[cfg(test)]
mod tests {
    use super::read_bounded;
    use std::io::Cursor;

    #[test]
    fn rejects_oversized_input() {
        let length = usize::try_from(super::MAX_DOCUMENT_BYTES + 1).expect("1 MiB fits usize");
        let source = vec![b'x'; length];
        assert!(read_bounded(Cursor::new(source), "input").is_err());
    }
}
