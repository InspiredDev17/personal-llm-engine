//! Strict offline verification for component-scoped contract kits.

use crate::component_registry::ALL_COMPONENTS;
use crate::document::parse_strict_value;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::fs;
use std::path::{Component, Path};

const COMPONENT_KIT_FORMAT: &str = "sovereign-ai-component-kit/v2";
const COMPONENT_KIT_MANIFEST: &str = "COMPONENT-KIT.json";
const COMPONENT_KIT_PROFILE: &str = "component-profile.jsonl";
const MAX_MANIFEST_BYTES: u64 = 1024 * 1024;
const MAX_PROFILE_BYTES: u64 = 128 * 1024;
const MAX_SCHEMA_BYTES: u64 = 16 * 1024 * 1024;

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct KitArtifact {
    path: String,
    sha256: String,
    size: u64,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct KitSchema {
    contract_version: String,
    domain: String,
    path: String,
    sha256: String,
    size: u64,
    roles: Vec<String>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct KitManifest {
    format: String,
    component: String,
    source_commit: String,
    profile: KitArtifact,
    schemas: Vec<KitSchema>,
    dependencies: Vec<KitArtifact>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct ProfileContract {
    contract_version: String,
    domain: String,
    schema: String,
    schema_sha256: String,
    roles: Vec<String>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct Profile {
    component: String,
    contracts: Vec<ProfileContract>,
}

/// Successful component-kit integrity verification.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ComponentKitVerification {
    pub component: String,
    pub source_commit: String,
    pub schema_count: usize,
    pub manifest_sha256: String,
}

/// Verifies a component kit against a separately retained canonical profile.
/// This checks integrity only and grants no connectivity or runtime authority.
///
/// # Errors
///
/// Returns an error for malformed or non-canonical metadata, unsafe paths or
/// file types, missing or extra files, profile disagreement, or byte drift.
pub fn verify_component_kit(
    kit_path: &Path,
    trusted_profile_path: &Path,
) -> Result<ComponentKitVerification, String> {
    let root_meta = fs::symlink_metadata(kit_path)
        .map_err(|error| format!("inspect component kit root: {error}"))?;
    if root_meta.file_type().is_symlink() || !root_meta.is_dir() {
        return Err("component kit root must be a non-symbolic-link directory".to_owned());
    }
    let trusted_data = read_regular(trusted_profile_path, MAX_PROFILE_BYTES, "trusted profile")?;
    let trusted: Profile = parse_canonical_json_line(&trusted_data, "trusted profile")?;
    validate_profile(&trusted)?;

    let manifest_path = kit_path.join(COMPONENT_KIT_MANIFEST);
    let manifest_data = read_regular(&manifest_path, MAX_MANIFEST_BYTES, "component kit manifest")?;
    let manifest: KitManifest =
        parse_canonical_json_line(&manifest_data, "component kit manifest")?;
    let digest = format!("{:x}", Sha256::digest(&manifest_data));
    let checksum_data = read_regular(
        &kit_path.join(format!("{COMPONENT_KIT_MANIFEST}.sha256")),
        1024,
        "component kit checksum",
    )?;
    let checksum = std::str::from_utf8(&checksum_data)
        .map_err(|_| "component kit checksum must be UTF-8".to_owned())?;
    if checksum != format!("{digest}  {COMPONENT_KIT_MANIFEST}\n") {
        return Err("component kit manifest checksum mismatch".to_owned());
    }
    validate_manifest_identity(kit_path, &manifest, &trusted)?;

    let packaged_profile = read_regular(
        &kit_path.join(COMPONENT_KIT_PROFILE),
        MAX_PROFILE_BYTES,
        "packaged profile",
    )?;
    if packaged_profile != trusted_data {
        return Err("component kit profile does not match trusted profile pin".to_owned());
    }
    if manifest.profile.path != COMPONENT_KIT_PROFILE
        || manifest.profile.size != usize_to_u64(packaged_profile.len())?
        || manifest.profile.sha256 != format!("{:x}", Sha256::digest(&packaged_profile))
    {
        return Err("component kit profile metadata does not match profile bytes".to_owned());
    }
    if manifest.schemas.len() != trusted.contracts.len()
        || manifest.schemas.is_empty()
        || manifest.schemas.len() > 256
    {
        return Err("component kit schema coverage does not match trusted profile".to_owned());
    }

    let mut expected = BTreeSet::from([
        COMPONENT_KIT_MANIFEST.to_owned(),
        format!("{COMPONENT_KIT_MANIFEST}.sha256"),
        COMPONENT_KIT_PROFILE.to_owned(),
    ]);
    for (index, (schema, contract)) in manifest
        .schemas
        .iter()
        .zip(trusted.contracts.iter())
        .enumerate()
    {
        if schema.contract_version != contract.contract_version
            || schema.domain != contract.domain
            || schema.path != contract.schema
            || schema.sha256 != contract.schema_sha256
            || schema.roles != contract.roles
            || schema.size == 0
            || schema.size > MAX_SCHEMA_BYTES
        {
            return Err(format!(
                "component kit schema {index} does not match trusted profile"
            ));
        }
        validate_schema_path(&schema.path, &schema.domain)?;
        let data = read_regular(
            &kit_path.join(&schema.path),
            MAX_SCHEMA_BYTES,
            "packaged schema",
        )?;
        if usize_to_u64(data.len())? != schema.size
            || format!("{:x}", Sha256::digest(&data)) != schema.sha256
        {
            return Err(format!(
                "component kit schema {:?} bytes do not match manifest",
                schema.path
            ));
        }
        if !expected.insert(schema.path.clone()) {
            return Err("component kit manifest contains a duplicate path".to_owned());
        }
    }
    verify_dependencies(kit_path, &manifest, &mut expected)?;
    let actual = collect_files(kit_path)?;
    if actual != expected {
        return Err("component kit file set contains missing or unexpected files".to_owned());
    }
    Ok(ComponentKitVerification {
        component: manifest.component,
        source_commit: manifest.source_commit,
        schema_count: manifest.schemas.len() + manifest.dependencies.len(),
        manifest_sha256: digest,
    })
}

fn verify_dependencies(
    kit_path: &Path,
    manifest: &KitManifest,
    expected: &mut BTreeSet<String>,
) -> Result<(), String> {
    if manifest.dependencies.len() > 256 {
        return Err("component kit dependency closure exceeds bound".to_owned());
    }
    let roots = manifest
        .schemas
        .iter()
        .map(|schema| schema.path.clone())
        .collect::<BTreeSet<_>>();
    let required = collect_schema_dependencies(kit_path, &roots)?;
    let declared = manifest
        .dependencies
        .iter()
        .map(|dependency| dependency.path.clone())
        .collect::<BTreeSet<_>>();
    if declared.len() != manifest.dependencies.len() || declared != required {
        return Err("component kit dependency closure is incomplete".to_owned());
    }
    for dependency in &manifest.dependencies {
        validate_dependency_path(&dependency.path)?;
        let data = read_regular(
            &kit_path.join(&dependency.path),
            MAX_SCHEMA_BYTES,
            "packaged schema dependency",
        )?;
        if usize_to_u64(data.len())? != dependency.size
            || format!("{:x}", Sha256::digest(&data)) != dependency.sha256
        {
            return Err(format!(
                "component kit dependency {:?} bytes do not match manifest",
                dependency.path
            ));
        }
        if !expected.insert(dependency.path.clone()) {
            return Err("component kit manifest contains a duplicate path".to_owned());
        }
    }
    Ok(())
}

fn validate_manifest_identity(
    kit_path: &Path,
    manifest: &KitManifest,
    trusted: &Profile,
) -> Result<(), String> {
    if manifest.format != COMPONENT_KIT_FORMAT
        || manifest.component != trusted.component
        || !is_lower_hex(&manifest.source_commit, 40)
    {
        return Err("component kit manifest identity is invalid".to_owned());
    }
    let expected = format!(
        "Sovereign-AI-Contracts-component-{}-{}",
        manifest.component, manifest.source_commit
    );
    if kit_path.file_name().and_then(|name| name.to_str()) != Some(expected.as_str()) {
        return Err("component kit directory name does not match manifest identity".to_owned());
    }
    Ok(())
}

fn validate_profile(profile: &Profile) -> Result<(), String> {
    if !ALL_COMPONENTS.contains(&profile.component.as_str())
        || profile.contracts.is_empty()
        || profile.contracts.len() > 256
    {
        return Err("trusted component profile identity or bounds are invalid".to_owned());
    }
    let mut previous: Option<(&str, &str)> = None;
    for contract in &profile.contracts {
        validate_schema_path(&contract.schema, &contract.domain)?;
        if !is_semver(&contract.contract_version)
            || !is_code(&contract.domain)
            || !is_lower_hex(&contract.schema_sha256, 64)
            || !valid_roles(&contract.roles)
        {
            return Err("trusted component profile contract is invalid".to_owned());
        }
        let current = (contract.contract_version.as_str(), contract.domain.as_str());
        if previous.is_some_and(|value| value >= current) {
            return Err("trusted component profile contracts are not sorted".to_owned());
        }
        previous = Some(current);
    }
    Ok(())
}

fn validate_schema_path(path: &str, domain: &str) -> Result<(), String> {
    let value = Path::new(path);
    if value.is_absolute()
        || value
            .components()
            .any(|part| !matches!(part, Component::Normal(_)))
        || !path.starts_with(&format!("schemas/{domain}/v"))
        || !path.ends_with(".schema.json")
        || path.contains('\\')
    {
        return Err(format!("invalid component kit schema path {path:?}"));
    }
    Ok(())
}

fn validate_dependency_path(path: &str) -> Result<(), String> {
    let value = Path::new(path);
    if value.is_absolute()
        || value
            .components()
            .any(|part| !matches!(part, Component::Normal(_)))
        || !path.starts_with("schemas/")
        || !path.ends_with(".schema.json")
        || path.contains('\\')
    {
        return Err(format!("invalid component kit dependency path {path:?}"));
    }
    Ok(())
}

fn collect_schema_dependencies(
    root: &Path,
    roots: &BTreeSet<String>,
) -> Result<BTreeSet<String>, String> {
    const SCHEMA_ID_PREFIX: &str = "https://contracts.sovereign-ai.local/";
    let mut seen = BTreeSet::new();
    let mut pending = roots.iter().cloned().collect::<Vec<_>>();
    while let Some(path) = pending.pop() {
        if !seen.insert(path.clone()) {
            continue;
        }
        let data = read_regular(&root.join(&path), MAX_SCHEMA_BYTES, "packaged schema")?;
        let value = parse_strict_value(&data)
            .map_err(|error| format!("decode packaged schema {path:?}: {error}"))?;
        let mut references = Vec::new();
        collect_references(&value, &mut references);
        for reference in references {
            if let Some(relative) = reference.strip_prefix(SCHEMA_ID_PREFIX) {
                let dependency = relative.split('#').next().unwrap_or_default().to_owned();
                validate_dependency_path(&dependency)?;
                if !seen.contains(&dependency) {
                    pending.push(dependency);
                }
            }
        }
    }
    Ok(seen.difference(roots).cloned().collect())
}

fn collect_references(value: &Value, references: &mut Vec<String>) {
    match value {
        Value::Object(object) => {
            if let Some(Value::String(reference)) = object.get("$ref") {
                references.push(reference.clone());
            }
            for child in object.values() {
                collect_references(child, references);
            }
        }
        Value::Array(values) => {
            for child in values {
                collect_references(child, references);
            }
        }
        Value::Null | Value::Bool(_) | Value::Number(_) | Value::String(_) => {}
    }
}

fn parse_canonical_json_line<T>(data: &[u8], label: &str) -> Result<T, String>
where
    T: for<'de> Deserialize<'de> + Serialize,
{
    if data.is_empty()
        || data.last() != Some(&b'\n')
        || data[..data.len().saturating_sub(1)].contains(&b'\n')
    {
        return Err(format!(
            "{label} must be exactly one newline-terminated JSON value"
        ));
    }
    let value = parse_strict_value(data).map_err(|error| format!("decode {label}: {error}"))?;
    let decoded: T =
        serde_json::from_value(value).map_err(|error| format!("decode {label}: {error}"))?;
    let mut canonical =
        serde_json::to_vec(&decoded).map_err(|error| format!("encode {label}: {error}"))?;
    canonical.push(b'\n');
    if canonical != data {
        return Err(format!("{label} is not canonical JSON"));
    }
    Ok(decoded)
}

fn read_regular(path: &Path, maximum: u64, label: &str) -> Result<Vec<u8>, String> {
    let metadata =
        fs::symlink_metadata(path).map_err(|error| format!("inspect {label}: {error}"))?;
    if metadata.file_type().is_symlink() || !metadata.is_file() || metadata.len() > maximum {
        return Err(format!(
            "{label} must be a bounded regular non-symbolic-link file"
        ));
    }
    let data = fs::read(path).map_err(|error| format!("read {label}: {error}"))?;
    if usize_to_u64(data.len())? > maximum {
        return Err(format!("{label} exceeds its byte limit"));
    }
    Ok(data)
}

fn collect_files(root: &Path) -> Result<BTreeSet<String>, String> {
    fn walk(root: &Path, current: &Path, files: &mut BTreeSet<String>) -> Result<(), String> {
        for entry in fs::read_dir(current)
            .map_err(|error| format!("read component kit directory: {error}"))?
        {
            let entry = entry.map_err(|error| format!("read component kit entry: {error}"))?;
            let path = entry.path();
            let metadata = fs::symlink_metadata(&path)
                .map_err(|error| format!("inspect component kit entry: {error}"))?;
            if metadata.file_type().is_symlink() {
                return Err("component kit contains a symbolic link".to_owned());
            }
            if metadata.is_dir() {
                walk(root, &path, files)?;
            } else if metadata.is_file() {
                let relative = path
                    .strip_prefix(root)
                    .map_err(|_| "component kit path escaped root".to_owned())?
                    .to_str()
                    .ok_or_else(|| "component kit path is not UTF-8".to_owned())?
                    .replace('\\', "/");
                if !files.insert(relative) {
                    return Err("component kit contains a duplicate file".to_owned());
                }
            } else {
                return Err("component kit contains a special file".to_owned());
            }
        }
        Ok(())
    }
    let mut files = BTreeSet::new();
    walk(root, root, &mut files)?;
    Ok(files)
}

fn is_lower_hex(value: &str, length: usize) -> bool {
    value.len() == length
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn is_semver(value: &str) -> bool {
    let parts: Vec<_> = value.split('.').collect();
    parts.len() == 3
        && parts
            .iter()
            .all(|part| !part.is_empty() && part.bytes().all(|byte| byte.is_ascii_digit()))
}

fn is_code(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'_')
}

fn valid_roles(roles: &[String]) -> bool {
    matches!(roles, [role] if role == "consumer" || role == "producer")
        || matches!(roles, [first, second] if first == "consumer" && second == "producer")
}

fn usize_to_u64(value: usize) -> Result<u64, String> {
    u64::try_from(value).map_err(|_| "file size does not fit u64".to_owned())
}
