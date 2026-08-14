use serde::{Deserialize, Serialize};
use serde_json::Value;
use sovereign_ai_contracts::common::ContractVersion;
use sovereign_ai_contracts::document::validate_document;
use sovereign_ai_contracts::{ContractError, Result};
use std::env;
use std::fs;
use std::path::{Component, Path, PathBuf};

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Manifest {
    contract_version: ContractVersion,
    fixtures: Vec<ManifestEntry>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ManifestEntry {
    path: String,
    schema: String,
    expected: Expected,
    #[serde(default)]
    kind: Option<String>,
    #[serde(default)]
    reason_code: Option<String>,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
enum Expected {
    Accept,
    Reject,
}

impl Expected {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Accept => "accept",
            Self::Reject => "reject",
        }
    }
}

struct Options {
    fixture_root: PathBuf,
    schema_root: Option<PathBuf>,
    conformance: bool,
}

#[derive(Serialize)]
struct ConformanceFixture {
    actual: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    contract_version: Option<String>,
    expected: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    kind: Option<String>,
    path: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    reason_code: Option<String>,
    schema: String,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("fixture-normalize: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let options = parse_options()?;
    let fixture_root = options.fixture_root;
    let manifest_path = fixture_root.join("manifest.json");
    let manifest_source = fs::read(&manifest_path)?;
    let manifest: Manifest = serde_json::from_slice(&manifest_source)?;
    let _ = manifest.contract_version;

    if options.conformance {
        let schema_root = options.schema_root.ok_or_else(|| {
            ContractError::field("--schema-root", "is required with --conformance")
        })?;
        if !fs::metadata(fs::canonicalize(schema_root)?)?.is_dir() {
            return Err(ContractError::field("--schema-root", "must be a directory"));
        }
        return render_conformance(&fixture_root, &manifest);
    }
    if options.schema_root.is_some() {
        return Err(ContractError::field(
            "--schema-root",
            "is only valid with --conformance",
        ));
    }

    let mut accepted: Vec<&ManifestEntry> = manifest
        .fixtures
        .iter()
        .filter(|entry| entry.expected == Expected::Accept)
        .collect();
    accepted.sort_by(|left, right| left.path.cmp(&right.path));
    if accepted.is_empty() {
        return Err(ContractError::field(
            "fixtures",
            "manifest contains no accepted fixtures",
        ));
    }

    for entry in accepted {
        validate_relative_path("path", &entry.path)?;
        validate_relative_path("schema", &entry.schema)?;
        if entry.reason_code.is_some() {
            return Err(ContractError::field(
                "reason_code",
                "accepted fixtures cannot declare a rejection reason",
            ));
        }
        let fixture_path = fixture_root.join(&entry.path);
        let source = fs::read(&fixture_path)?;
        let envelope: Value = serde_json::from_slice(&source)?;
        let wire_kind = envelope
            .get("kind")
            .and_then(Value::as_str)
            .ok_or_else(|| ContractError::field("kind", "accepted fixture must declare kind"))?;
        let declared_kind = entry
            .kind
            .as_deref()
            .ok_or_else(|| ContractError::field("kind", "accepted manifest entry requires kind"))?;
        if wire_kind != declared_kind {
            return Err(ContractError::field(
                "kind",
                format!(
                    "manifest kind {declared_kind:?} does not match fixture kind {wire_kind:?}"
                ),
            ));
        }
        let normalized = validate_document(&source)?;
        if normalized.kind != declared_kind {
            return Err(ContractError::field(
                "kind",
                "validated kind does not match manifest kind",
            ));
        }
        let declared_schema = entry
            .schema
            .strip_prefix("schemas/")
            .unwrap_or(&entry.schema);
        if normalized.schema_path != declared_schema {
            return Err(ContractError::field(
                "schema",
                format!(
                    "registry schema {:?} does not match manifest schema {declared_schema:?}",
                    normalized.schema_path
                ),
            ));
        }
        println!("{}\t{}", entry.path, normalized.canonical_json);
    }
    Ok(())
}

fn parse_options() -> Result<Options> {
    let mut arguments = env::args_os().skip(1);
    let mut root = None;
    let mut schema_root = None;
    let mut conformance = false;
    while let Some(argument) = arguments.next() {
        match argument.to_str() {
            Some("--fixture-root") => {
                if root.is_some() {
                    return Err(ContractError::field(
                        "--fixture-root",
                        "may be provided only once",
                    ));
                }
                root = arguments.next().map(PathBuf::from);
            }
            Some("--schema-root") => {
                if schema_root.is_some() {
                    return Err(ContractError::field(
                        "--schema-root",
                        "may be provided only once",
                    ));
                }
                schema_root = arguments.next().map(PathBuf::from);
            }
            Some("--conformance") if !conformance => conformance = true,
            _ => {
                return Err(ContractError::field(
                    "arguments",
                    format!("unknown or duplicate argument {argument:?}"),
                ));
            }
        }
    }
    Ok(Options {
        fixture_root: root.ok_or_else(|| ContractError::field("--fixture-root", "is required"))?,
        schema_root,
        conformance,
    })
}

fn render_conformance(fixture_root: &Path, manifest: &Manifest) -> Result<()> {
    let mut fixtures: Vec<&ManifestEntry> = manifest.fixtures.iter().collect();
    fixtures.sort_by(|left, right| left.path.cmp(&right.path));
    for entry in fixtures {
        validate_relative_path("path", &entry.path)?;
        validate_relative_path("schema", &entry.schema)?;
        let source = fs::read(fixture_root.join(&entry.path))?;
        let validation = validate_document(&source);
        let actual = if validation.is_ok() {
            "accept"
        } else {
            "reject"
        };
        if actual != entry.expected.as_str() {
            return Err(ContractError::field(
                "fixture",
                format!(
                    "{:?} expected {} but validator returned {actual}: {:?}",
                    entry.path,
                    entry.expected.as_str(),
                    validation.err()
                ),
            ));
        }
        let declared_schema = entry
            .schema
            .strip_prefix("schemas/")
            .unwrap_or(&entry.schema)
            .to_owned();
        let contract_version = if let Ok(result) = validation {
            if entry.kind.as_deref() != Some(result.kind.as_str())
                || declared_schema != result.schema_path
            {
                return Err(ContractError::field(
                    "fixture",
                    format!(
                        "{:?} manifest registry does not match validator registry",
                        entry.path
                    ),
                ));
            }
            Some(result.contract_version)
        } else {
            None
        };
        let result = ConformanceFixture {
            actual: actual.to_owned(),
            contract_version,
            expected: entry.expected.as_str().to_owned(),
            kind: entry.kind.clone(),
            path: entry.path.clone(),
            reason_code: entry.reason_code.clone(),
            schema: declared_schema,
        };
        println!("{}", serde_json::to_string(&result)?);
    }
    Ok(())
}

fn validate_relative_path(field: &'static str, value: &str) -> Result<()> {
    let path = Path::new(value);
    if path.as_os_str().is_empty()
        || path.is_absolute()
        || path
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
    {
        Err(ContractError::field(
            field,
            "must be a normalized relative path",
        ))
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{validate_relative_path, Expected, Manifest};
    use sovereign_ai_contracts::document::validate_document;
    use std::fs;
    use std::path::PathBuf;

    #[test]
    fn rejects_traversal_and_absolute_fixture_paths() {
        assert!(validate_relative_path("path", "v1/valid/a.json").is_ok());
        assert!(validate_relative_path("path", "../a.json").is_err());
        assert!(validate_relative_path("path", "/tmp/a.json").is_err());
    }

    #[test]
    fn typed_dispatch_rejects_every_published_invalid_fixture() {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../fixtures");
        let manifest: Manifest = serde_json::from_slice(
            &fs::read(root.join("manifest.json")).expect("read fixture manifest"),
        )
        .expect("decode fixture manifest");
        let expected_rejections = manifest
            .fixtures
            .iter()
            .filter(|fixture| fixture.expected == Expected::Reject)
            .count();
        let mut rejected = 0_usize;
        let mut saw_duplicate_key = false;
        for fixture in manifest
            .fixtures
            .iter()
            .filter(|fixture| fixture.expected == Expected::Reject)
        {
            let source = fs::read(root.join(&fixture.path)).expect("read rejected fixture");
            let kind = fixture
                .kind
                .clone()
                .or_else(|| embedded_kind(&source))
                .unwrap_or_else(|| panic!("{} has no discoverable kind", fixture.path));
            let result = validate_document(&source);
            assert!(
                result.is_err(),
                "{} ({:?}) must fail Rust typed validation",
                fixture.path,
                fixture.reason_code
            );
            if fixture.reason_code.as_deref() == Some("duplicate_key") {
                saw_duplicate_key = true;
                assert_eq!(kind, "inference.cancellation_request");
            }
            rejected += 1;
        }
        assert!(expected_rejections > 0);
        assert_eq!(rejected, expected_rejections);
        assert!(saw_duplicate_key, "duplicate-key fixture was not exercised");
    }

    fn embedded_kind(source: &[u8]) -> Option<String> {
        if let Ok(value) = serde_json::from_slice::<serde_json::Value>(source) {
            return value.get("kind")?.as_str().map(str::to_owned);
        }
        // Malformed fixtures can still declare a complete `kind` before the
        // syntax defect. This bounded fallback only selects a strict typed
        // decoder; it never makes malformed JSON acceptable.
        let source = std::str::from_utf8(source).ok()?;
        let tail = source.split_once("\"kind\"")?.1;
        let value = tail.split_once(':')?.1.trim_start();
        let value = value.strip_prefix('"')?;
        Some(value.split_once('"')?.0.to_owned())
    }
}
