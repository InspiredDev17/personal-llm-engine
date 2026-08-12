//! Canonical hash catalog for directory-backed `.paimodel` packages.

use personalai_engine_types::{EngineError, ModelPackageHash};
use std::collections::BTreeSet;

/// Hash-catalog magic.
pub const HASH_CATALOG_MAGIC: [u8; 8] = *b"PAIHASH1";
/// Fixed hash-catalog header length.
pub const HASH_CATALOG_HEADER_BYTES: usize = 16;
/// Fixed hash-catalog record length.
pub const HASH_CATALOG_RECORD_BYTES: usize = 128;
/// Maximum entries: required metadata plus every bounded shard.
pub const HASH_CATALOG_MAX_ENTRIES: u32 = 1_027;
/// Maximum canonical trusted filename bytes.
pub const HASH_CATALOG_NAME_BYTES: usize = 63;

/// Closed trusted file category.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum PackageFileKind {
    /// Binary directory manifest.
    Manifest,
    /// First-party tokenizer data.
    Tokenizer,
    /// Deterministic provenance JSON.
    Provenance,
    /// Immutable tensor shard.
    TensorShard,
}

impl PackageFileKind {
    const fn encode(self) -> u16 {
        match self {
            Self::Manifest => 1,
            Self::Tokenizer => 2,
            Self::Provenance => 3,
            Self::TensorShard => 4,
        }
    }

    fn decode(value: u16) -> Result<Self, EngineError> {
        match value {
            1 => Ok(Self::Manifest),
            2 => Ok(Self::Tokenizer),
            3 => Ok(Self::Provenance),
            4 => Ok(Self::TensorShard),
            _ => Err(corrupt("hash catalog file kind is unsupported")),
        }
    }
}

/// One signed exact file identity.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HashCatalogEntry {
    kind: PackageFileKind,
    index: u32,
    name: String,
    byte_length: u64,
    content_hash: ModelPackageHash,
}

impl HashCatalogEntry {
    /// Constructs one canonical trusted-file record.
    pub fn new(
        kind: PackageFileKind,
        index: u32,
        name: impl Into<String>,
        byte_length: u64,
        content_hash: ModelPackageHash,
    ) -> Result<Self, EngineError> {
        let name = name.into();
        validate_file_identity(kind, index, &name, byte_length)?;
        Ok(Self { kind, index, name, byte_length, content_hash })
    }

    /// Returns the trusted file category.
    pub const fn kind(&self) -> PackageFileKind {
        self.kind
    }

    /// Returns the shard index, or zero for singleton metadata files.
    pub const fn index(&self) -> u32 {
        self.index
    }

    /// Returns the direct-child canonical filename.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns the exact file byte length.
    pub const fn byte_length(&self) -> u64 {
        self.byte_length
    }

    /// Returns the exact SHA-384 file digest.
    pub const fn content_hash(&self) -> ModelPackageHash {
        self.content_hash
    }
}

/// Signed canonical list of every trusted package data file.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HashCatalog {
    entries: Vec<HashCatalogEntry>,
}

impl HashCatalog {
    /// Constructs a sorted catalog and enforces its exact closed structure.
    pub fn new(mut entries: Vec<HashCatalogEntry>) -> Result<Self, EngineError> {
        entries.sort_by(|left, right| {
            (left.kind, left.index, left.name.as_str()).cmp(&(
                right.kind,
                right.index,
                right.name.as_str(),
            ))
        });
        validate_entries(&entries)?;
        Ok(Self { entries })
    }

    /// Parses one exact catalog without trailing data.
    pub fn parse(bytes: &[u8]) -> Result<Self, EngineError> {
        if bytes.len() < HASH_CATALOG_HEADER_BYTES || bytes[..8] != HASH_CATALOG_MAGIC {
            return Err(corrupt("hash catalog header is invalid or truncated"));
        }
        if read_u16(bytes, 8)? != 1 || read_u16(bytes, 10)? != 0 {
            return Err(corrupt("hash catalog version is unsupported"));
        }
        let count = read_u32(bytes, 12)?;
        if !(4..=HASH_CATALOG_MAX_ENTRIES).contains(&count) {
            return Err(corrupt("hash catalog entry count is outside hard bounds"));
        }
        let expected = usize::try_from(count)
            .ok()
            .and_then(|value| value.checked_mul(HASH_CATALOG_RECORD_BYTES))
            .and_then(|value| value.checked_add(HASH_CATALOG_HEADER_BYTES))
            .ok_or_else(|| EngineError::resource_exhaustion("hash catalog length overflows"))?;
        if bytes.len() != expected {
            return Err(corrupt("hash catalog length or trailing data is invalid"));
        }
        let mut entries = Vec::new();
        entries
            .try_reserve_exact(usize::try_from(count).map_err(|_| {
                EngineError::resource_exhaustion("hash catalog count is not addressable")
            })?)
            .map_err(|_| EngineError::resource_exhaustion("hash catalog allocation failed"))?;
        let mut cursor = HASH_CATALOG_HEADER_BYTES;
        for _ in 0..count {
            let end = cursor
                .checked_add(HASH_CATALOG_RECORD_BYTES)
                .ok_or_else(|| EngineError::resource_exhaustion("hash record offset overflows"))?;
            let record = bytes
                .get(cursor..end)
                .ok_or_else(|| corrupt("hash catalog record is truncated"))?;
            entries.push(parse_entry(record)?);
            cursor = end;
        }
        let catalog = Self::new(entries)?;
        if catalog.encode()?.as_slice() != bytes {
            return Err(corrupt("hash catalog ordering is not canonical"));
        }
        Ok(catalog)
    }

    /// Encodes the exact canonical catalog.
    pub fn encode(&self) -> Result<Vec<u8>, EngineError> {
        validate_entries(&self.entries)?;
        let length = self
            .entries
            .len()
            .checked_mul(HASH_CATALOG_RECORD_BYTES)
            .and_then(|value| value.checked_add(HASH_CATALOG_HEADER_BYTES))
            .ok_or_else(|| EngineError::resource_exhaustion("hash catalog length overflows"))?;
        let mut bytes = zeroed(length)?;
        bytes[..8].copy_from_slice(&HASH_CATALOG_MAGIC);
        write_u16(&mut bytes, 8, 1)?;
        write_u16(&mut bytes, 10, 0)?;
        write_u32(
            &mut bytes,
            12,
            u32::try_from(self.entries.len())
                .map_err(|_| corrupt("hash catalog count exceeds its field"))?,
        )?;
        let mut cursor = HASH_CATALOG_HEADER_BYTES;
        for entry in &self.entries {
            write_u16(&mut bytes, cursor, entry.kind.encode())?;
            write_u16(
                &mut bytes,
                cursor + 2,
                u16::try_from(entry.name.len())
                    .map_err(|_| corrupt("hash catalog name exceeds its field"))?,
            )?;
            write_u32(&mut bytes, cursor + 4, entry.index)?;
            write_u64(&mut bytes, cursor + 8, entry.byte_length)?;
            write_exact(&mut bytes, cursor + 16, entry.content_hash.as_bytes())?;
            write_exact(&mut bytes, cursor + 64, entry.name.as_bytes())?;
            cursor += HASH_CATALOG_RECORD_BYTES;
        }
        Ok(bytes)
    }

    /// Returns every signed file identity.
    pub fn entries(&self) -> &[HashCatalogEntry] {
        &self.entries
    }

    /// Resolves one exact direct-child name.
    pub fn entry(&self, name: &str) -> Option<&HashCatalogEntry> {
        self.entries.iter().find(|entry| entry.name == name)
    }
}

fn parse_entry(record: &[u8]) -> Result<HashCatalogEntry, EngineError> {
    let kind = PackageFileKind::decode(read_u16(record, 0)?)?;
    let name_length = usize::from(read_u16(record, 2)?);
    let index = read_u32(record, 4)?;
    let byte_length = read_u64(record, 8)?;
    let mut hash = [0_u8; ModelPackageHash::BYTE_LEN];
    hash.copy_from_slice(&record[16..64]);
    let name_field = &record[64..128];
    if name_length == 0
        || name_length > HASH_CATALOG_NAME_BYTES
        || name_field[name_length..].iter().any(|byte| *byte != 0)
    {
        return Err(corrupt("hash catalog filename field is invalid"));
    }
    let name = std::str::from_utf8(&name_field[..name_length])
        .map_err(|_| corrupt("hash catalog filename is invalid UTF-8"))?;
    HashCatalogEntry::new(kind, index, name, byte_length, ModelPackageHash::from_sha384_bytes(hash))
}

fn validate_file_identity(
    kind: PackageFileKind,
    index: u32,
    name: &str,
    byte_length: u64,
) -> Result<(), EngineError> {
    if name.is_empty()
        || name.len() > HASH_CATALOG_NAME_BYTES
        || !name.is_ascii()
        || name.contains("..")
        || name.contains('/')
        || name.contains('\\')
        || name.chars().any(char::is_control)
        || byte_length == 0
    {
        return Err(corrupt("hash catalog file identity is invalid"));
    }
    match kind {
        PackageFileKind::Manifest if index != 0 || name != "manifest.aipm" => {
            Err(corrupt("manifest hash-catalog identity is not canonical"))
        }
        PackageFileKind::Tokenizer if index != 0 || name != "tokenizer.aiptok" => {
            Err(corrupt("tokenizer hash-catalog identity is not canonical"))
        }
        PackageFileKind::Provenance if index != 0 || name != "provenance.json" => {
            Err(corrupt("provenance hash-catalog identity is not canonical"))
        }
        PackageFileKind::TensorShard
            if name != format!("tensors-{index:04}.aipt") || index >= 1_024 =>
        {
            Err(corrupt("tensor-shard hash-catalog identity is not canonical"))
        }
        _ => Ok(()),
    }
}

fn validate_entries(entries: &[HashCatalogEntry]) -> Result<(), EngineError> {
    if entries.len() < 4
        || entries.len() > usize::try_from(HASH_CATALOG_MAX_ENTRIES).unwrap_or(usize::MAX)
    {
        return Err(corrupt("hash catalog entry count is outside hard bounds"));
    }
    let names = entries.iter().map(|entry| entry.name.as_str()).collect::<BTreeSet<_>>();
    if names.len() != entries.len() {
        return Err(corrupt("hash catalog contains a duplicate filename"));
    }
    for required in ["manifest.aipm", "tokenizer.aiptok", "provenance.json"] {
        if !names.contains(required) {
            return Err(corrupt("hash catalog omits required metadata"));
        }
    }
    let shards = entries.iter().filter(|entry| entry.kind == PackageFileKind::TensorShard);
    let mut count = 0_usize;
    for (expected, shard) in shards.enumerate() {
        if shard.index != u32::try_from(expected).unwrap_or(u32::MAX) {
            return Err(corrupt("hash catalog shard indices are not contiguous"));
        }
        count += 1;
    }
    if count == 0 || count + 3 != entries.len() {
        return Err(corrupt("hash catalog contains an unexpected trusted file"));
    }
    Ok(())
}

fn read_u16(bytes: &[u8], offset: usize) -> Result<u16, EngineError> {
    Ok(u16::from_le_bytes(
        read_exact(bytes, offset, 2)?
            .try_into()
            .map_err(|_| corrupt("hash catalog integer field is truncated"))?,
    ))
}

fn read_u32(bytes: &[u8], offset: usize) -> Result<u32, EngineError> {
    Ok(u32::from_le_bytes(
        read_exact(bytes, offset, 4)?
            .try_into()
            .map_err(|_| corrupt("hash catalog integer field is truncated"))?,
    ))
}

fn read_u64(bytes: &[u8], offset: usize) -> Result<u64, EngineError> {
    Ok(u64::from_le_bytes(
        read_exact(bytes, offset, 8)?
            .try_into()
            .map_err(|_| corrupt("hash catalog integer field is truncated"))?,
    ))
}

fn read_exact(bytes: &[u8], offset: usize, length: usize) -> Result<&[u8], EngineError> {
    let end = offset
        .checked_add(length)
        .ok_or_else(|| EngineError::resource_exhaustion("hash catalog offset overflows"))?;
    bytes.get(offset..end).ok_or_else(|| corrupt("hash catalog field is truncated"))
}

fn write_u16(bytes: &mut [u8], offset: usize, value: u16) -> Result<(), EngineError> {
    write_exact(bytes, offset, &value.to_le_bytes())
}

fn write_u32(bytes: &mut [u8], offset: usize, value: u32) -> Result<(), EngineError> {
    write_exact(bytes, offset, &value.to_le_bytes())
}

fn write_u64(bytes: &mut [u8], offset: usize, value: u64) -> Result<(), EngineError> {
    write_exact(bytes, offset, &value.to_le_bytes())
}

fn write_exact(bytes: &mut [u8], offset: usize, value: &[u8]) -> Result<(), EngineError> {
    let end = offset
        .checked_add(value.len())
        .ok_or_else(|| EngineError::resource_exhaustion("hash catalog offset overflows"))?;
    bytes
        .get_mut(offset..end)
        .ok_or_else(|| corrupt("hash catalog write range is unavailable"))?
        .copy_from_slice(value);
    Ok(())
}

fn zeroed(length: usize) -> Result<Vec<u8>, EngineError> {
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(length)
        .map_err(|_| EngineError::resource_exhaustion("hash catalog allocation failed"))?;
    bytes.resize(length, 0);
    Ok(bytes)
}

fn corrupt(detail: impl AsRef<str>) -> EngineError {
    EngineError::corrupt_model_package(detail)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hash(byte: u8) -> ModelPackageHash {
        ModelPackageHash::from_sha384_bytes([byte; ModelPackageHash::BYTE_LEN])
    }

    #[test]
    fn catalog_round_trip_is_exact_and_closed() {
        let catalog = HashCatalog::new(vec![
            HashCatalogEntry::new(PackageFileKind::Manifest, 0, "manifest.aipm", 512, hash(1))
                .expect("manifest entry must validate"),
            HashCatalogEntry::new(PackageFileKind::Tokenizer, 0, "tokenizer.aiptok", 40, hash(2))
                .expect("tokenizer entry must validate"),
            HashCatalogEntry::new(PackageFileKind::Provenance, 0, "provenance.json", 2, hash(3))
                .expect("provenance entry must validate"),
            HashCatalogEntry::new(
                PackageFileKind::TensorShard,
                0,
                "tensors-0000.aipt",
                64,
                hash(4),
            )
            .expect("shard entry must validate"),
        ])
        .expect("catalog must validate");
        let bytes = catalog.encode().expect("catalog must encode");
        assert_eq!(HashCatalog::parse(&bytes).expect("catalog must parse"), catalog);
        let mut trailing = bytes;
        trailing.push(0);
        assert!(HashCatalog::parse(&trailing).is_err());
    }
}
