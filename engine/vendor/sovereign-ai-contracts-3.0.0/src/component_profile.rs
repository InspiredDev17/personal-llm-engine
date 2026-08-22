//! Exact offline schema support profiles for independently deployed components.

use crate::component_registry::{component_contract_registry, ALL_COMPONENTS};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs::{self, File};
use std::io::Read;
use std::path::Path;

const MAX_SCHEMA_BYTES: u64 = 16 * 1024 * 1024;
const MAX_COMPONENT_PROFILE_PIN_BYTES: u64 = 128 * 1024;

/// One exact schema and this component's descriptive interoperability roles.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct ComponentProfileContract {
    pub contract_version: &'static str,
    pub domain: &'static str,
    pub schema: &'static str,
    pub schema_sha256: String,
    pub roles: Vec<&'static str>,
}

/// All exact contract inputs relevant to one component.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct ComponentSupportProfile {
    pub component: &'static str,
    pub contracts: Vec<ComponentProfileContract>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct PinnedComponentProfileContract {
    contract_version: String,
    domain: String,
    schema: String,
    schema_sha256: String,
    roles: Vec<String>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct PinnedComponentSupportProfile {
    component: String,
    contracts: Vec<PinnedComponentProfileContract>,
}

/// Builds every component profile from the compile-bound routes and exact
/// schema bytes. Profiles are pinning evidence, never authorization.
///
/// # Errors
///
/// Returns an error when the schema root or a routed schema cannot be resolved,
/// is not a bounded regular file below that root, or cannot be read safely.
pub fn component_support_profiles(
    schema_root: &Path,
) -> Result<Vec<ComponentSupportProfile>, String> {
    let canonical_root = schema_root
        .canonicalize()
        .map_err(|error| format!("resolve schema root: {error}"))?;
    if !canonical_root.is_dir() {
        return Err("schema root must be a directory".to_owned());
    }
    let routes = component_contract_registry();
    let mut profiles = Vec::with_capacity(ALL_COMPONENTS.len());
    for component in ALL_COMPONENTS {
        let mut contracts = Vec::new();
        for route in &routes {
            let roles = roles_for(component, route.producers, route.consumers);
            if roles.is_empty() {
                continue;
            }
            let relative = route
                .schema
                .strip_prefix("schemas/")
                .ok_or_else(|| format!("invalid schema path {}", route.schema))?;
            let digest = digest_schema(&canonical_root, relative)?;
            contracts.push(ComponentProfileContract {
                contract_version: route.contract_version,
                domain: route.domain,
                schema: route.schema,
                schema_sha256: digest,
                roles,
            });
        }
        if contracts.is_empty() || contracts.len() > 256 {
            return Err(format!(
                "component {component} contract count is out of bounds"
            ));
        }
        profiles.push(ComponentSupportProfile {
            component,
            contracts,
        });
    }
    Ok(profiles)
}

/// Verifies one canonical copied component profile against the exact local
/// schema bytes and compile-bound routes. Verification grants no authority.
///
/// # Errors
///
/// Returns an error when the profile is not a bounded regular
/// non-symbolic-link file, is not exactly one canonical JSON line, names an
/// unknown component, or differs from the expected paths, roles, or digests.
pub fn verify_component_support_profile(
    schema_root: &Path,
    profile_path: &Path,
) -> Result<ComponentSupportProfile, String> {
    let metadata = fs::symlink_metadata(profile_path)
        .map_err(|error| format!("inspect component profile pin: {error}"))?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err("component profile pin must be a regular non-symbolic-link file".to_owned());
    }
    if metadata.len() > MAX_COMPONENT_PROFILE_PIN_BYTES {
        return Err(format!(
            "component profile pin exceeds {MAX_COMPONENT_PROFILE_PIN_BYTES} bytes"
        ));
    }
    let file =
        File::open(profile_path).map_err(|error| format!("open component profile pin: {error}"))?;
    let opened = file
        .metadata()
        .map_err(|error| format!("inspect opened component profile pin: {error}"))?;
    if !opened.is_file() || opened.len() > MAX_COMPONENT_PROFILE_PIN_BYTES {
        return Err("component profile pin changed or exceeds its byte limit".to_owned());
    }
    let mut data = Vec::with_capacity(usize::try_from(opened.len()).unwrap_or(0));
    file.take(MAX_COMPONENT_PROFILE_PIN_BYTES + 1)
        .read_to_end(&mut data)
        .map_err(|error| format!("read component profile pin: {error}"))?;
    verify_component_support_profile_bytes(schema_root, &data)
}

fn verify_component_support_profile_bytes(
    schema_root: &Path,
    data: &[u8],
) -> Result<ComponentSupportProfile, String> {
    if data.is_empty()
        || u64::try_from(data.len()).map_or(true, |length| length > MAX_COMPONENT_PROFILE_PIN_BYTES)
        || data.last() != Some(&b'\n')
        || data[..data.len().saturating_sub(1)].contains(&b'\n')
    {
        return Err(
            "component profile pin must be bounded and exactly one JSON line ending with a newline"
                .to_owned(),
        );
    }
    let pinned: PinnedComponentSupportProfile = serde_json::from_slice(data)
        .map_err(|error| format!("decode component profile pin: {error}"))?;
    let profiles = component_support_profiles(schema_root)?;
    let expected = profiles
        .into_iter()
        .find(|profile| profile.component == pinned.component)
        .ok_or_else(|| {
            format!(
                "component profile pin names unknown component {:?}",
                pinned.component
            )
        })?;
    let expected_pin = owned_profile(&expected);
    let mut canonical = serde_json::to_vec(&expected_pin)
        .map_err(|error| format!("encode expected component profile: {error}"))?;
    canonical.push(b'\n');
    if pinned != expected_pin || data != canonical {
        return Err(format!(
            "component profile pin for {:?} does not exactly match routed schema versions, roles, paths, and digests",
            expected.component
        ));
    }
    Ok(expected)
}

fn owned_profile(profile: &ComponentSupportProfile) -> PinnedComponentSupportProfile {
    PinnedComponentSupportProfile {
        component: profile.component.to_owned(),
        contracts: profile
            .contracts
            .iter()
            .map(|contract| PinnedComponentProfileContract {
                contract_version: contract.contract_version.to_owned(),
                domain: contract.domain.to_owned(),
                schema: contract.schema.to_owned(),
                schema_sha256: contract.schema_sha256.clone(),
                roles: contract.roles.iter().map(ToString::to_string).collect(),
            })
            .collect(),
    }
}

fn roles_for(
    component: &str,
    producers: &[&'static str],
    consumers: &[&'static str],
) -> Vec<&'static str> {
    let mut roles = Vec::with_capacity(2);
    if consumers.binary_search(&component).is_ok() {
        roles.push("consumer");
    }
    if producers.binary_search(&component).is_ok() {
        roles.push("producer");
    }
    roles
}

fn digest_schema(root: &Path, relative: &str) -> Result<String, String> {
    let requested = root.join(relative);
    let link_metadata = fs::symlink_metadata(&requested)
        .map_err(|error| format!("inspect schema {relative}: {error}"))?;
    if link_metadata.file_type().is_symlink() || !link_metadata.is_file() {
        return Err(format!(
            "schema {relative} must be a regular non-symbolic-link file"
        ));
    }
    if link_metadata.len() > MAX_SCHEMA_BYTES {
        return Err(format!(
            "schema {relative} exceeds {MAX_SCHEMA_BYTES} bytes"
        ));
    }
    let canonical = requested
        .canonicalize()
        .map_err(|error| format!("resolve schema {relative}: {error}"))?;
    if !canonical.starts_with(root) {
        return Err(format!("schema {relative} escapes the schema root"));
    }
    let data = fs::read(&canonical).map_err(|error| format!("read schema {relative}: {error}"))?;
    if u64::try_from(data.len()).map_or(true, |length| length > MAX_SCHEMA_BYTES) {
        return Err(format!(
            "schema {relative} exceeds {MAX_SCHEMA_BYTES} bytes"
        ));
    }
    Ok(format!("{:x}", Sha256::digest(data)))
}

#[cfg(test)]
mod tests {
    use super::{component_support_profiles, verify_component_support_profile_bytes};
    use std::path::Path;

    #[test]
    fn profiles_cover_every_component_with_sorted_contracts() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../schemas");
        let profiles = component_support_profiles(&root).expect("build profiles");
        assert_eq!(profiles.len(), 9);
        assert!(profiles
            .windows(2)
            .all(|pair| pair[0].component < pair[1].component));
        for profile in profiles {
            assert!(!profile.contracts.is_empty());
            assert!(profile.contracts.windows(2).all(|pair| {
                (pair[0].contract_version, pair[0].domain)
                    < (pair[1].contract_version, pair[1].domain)
            }));
            assert!(profile.contracts.iter().all(|contract| {
                contract.schema_sha256.len() == 64
                    && !contract.roles.is_empty()
                    && contract.roles.len() <= 2
            }));
        }
    }

    #[test]
    fn verifies_every_canonical_component_pin() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../schemas");
        let canonical = include_str!("../../compatibility/component-support-profiles.jsonl");
        for (index, line) in canonical.lines().enumerate() {
            let data = format!("{line}\n");
            let profile = verify_component_support_profile_bytes(&root, data.as_bytes())
                .expect("verify canonical profile");
            assert_eq!(profile.component, super::ALL_COMPONENTS[index]);
        }
    }

    #[test]
    fn rejects_tampered_malformed_and_oversized_component_pins() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../schemas");
        let line = include_str!("../../compatibility/component-support-profiles.jsonl")
            .lines()
            .next()
            .expect("first profile");
        let tampered = format!("{}\n", line.replacen("db701957", "0b701957", 1));
        assert!(
            verify_component_support_profile_bytes(&root, tampered.as_bytes())
                .expect_err("tampered digest")
                .contains("does not exactly match")
        );
        let unknown = format!(
            "{}\n",
            line.replacen("{\"component\":", "{\"unknown\":true,\"component\":", 1)
        );
        assert!(
            verify_component_support_profile_bytes(&root, unknown.as_bytes())
                .expect_err("unknown field")
                .contains("unknown field")
        );
        let duplicate = format!(
            "{}\n",
            line.replacen(
                "{\"component\":",
                "{\"component\":\"action_broker\",\"component\":",
                1
            )
        );
        assert!(verify_component_support_profile_bytes(&root, duplicate.as_bytes()).is_err());
        let oversized = vec![
            b'x';
            usize::try_from(super::MAX_COMPONENT_PROFILE_PIN_BYTES)
                .expect("profile limit fits usize")
                + 1
        ];
        assert!(verify_component_support_profile_bytes(&root, &oversized)
            .expect_err("oversized profile")
            .contains("bounded"));
    }
}
