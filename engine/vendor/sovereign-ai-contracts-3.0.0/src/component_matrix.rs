//! Exact directed contract routes between independently deployed components.

use crate::component_profile::component_support_profiles;
use crate::component_registry::{component_contract_registry, ALL_COMPONENTS};
use serde::{Deserialize, Serialize};
use std::fs::{self, File};
use std::io::Read;
use std::path::Path;

const MAX_COMPONENT_ROUTE_PIN_BYTES: u64 = 128 * 1024;

/// One exact contract routed from a producer to a consumer.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct ComponentMatrixContract {
    pub contract_version: &'static str,
    pub domain: &'static str,
    pub schema: &'static str,
    pub schema_sha256: String,
}

/// All exact contracts shared by one directed component pair.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct ComponentMatrixPair {
    pub producer: &'static str,
    pub consumer: &'static str,
    pub contracts: Vec<ComponentMatrixContract>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct PinnedComponentMatrixContract {
    contract_version: String,
    domain: String,
    schema: String,
    schema_sha256: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct PinnedComponentMatrixPair {
    producer: String,
    consumer: String,
    contracts: Vec<PinnedComponentMatrixContract>,
}

/// Builds the complete directed route matrix from compile-bound routes and
/// exact schema support profiles. Matrix entries never grant authority.
///
/// # Errors
///
/// Returns an error when a routed schema cannot be safely read and hashed, a
/// digest is missing, or the resulting matrix violates completeness bounds.
pub fn component_route_matrix(schema_root: &Path) -> Result<Vec<ComponentMatrixPair>, String> {
    let profiles = component_support_profiles(schema_root)?;
    let routes = component_contract_registry();
    let mut pairs = Vec::with_capacity(ALL_COMPONENTS.len() * ALL_COMPONENTS.len());

    for producer in ALL_COMPONENTS {
        for consumer in ALL_COMPONENTS {
            let producer_profile = profiles
                .iter()
                .find(|profile| profile.component == *producer)
                .ok_or_else(|| format!("missing component profile for {producer}"))?;
            let mut contracts = Vec::new();
            for route in &routes {
                if route.producers.binary_search(producer).is_err()
                    || route.consumers.binary_search(consumer).is_err()
                {
                    continue;
                }
                let profile_contract = producer_profile
                    .contracts
                    .iter()
                    .find(|contract| {
                        contract.contract_version == route.contract_version
                            && contract.domain == route.domain
                    })
                    .ok_or_else(|| {
                        format!(
                            "missing profile digest for {} at {}",
                            route.domain, route.contract_version
                        )
                    })?;
                contracts.push(ComponentMatrixContract {
                    contract_version: route.contract_version,
                    domain: route.domain,
                    schema: route.schema,
                    schema_sha256: profile_contract.schema_sha256.clone(),
                });
            }
            if contracts.is_empty() || contracts.len() > 256 {
                return Err(format!(
                    "component route pair {producer} to {consumer} contract count is out of bounds"
                ));
            }
            pairs.push(ComponentMatrixPair {
                producer,
                consumer,
                contracts,
            });
        }
    }
    if pairs.len() != ALL_COMPONENTS.len() * ALL_COMPONENTS.len() {
        return Err("component route matrix is incomplete".to_owned());
    }
    Ok(pairs)
}

/// Verifies one canonical copied directed-route pin against the exact local
/// schema bytes and compile-bound routes. Verification grants no connectivity
/// or authority.
///
/// # Errors
///
/// Returns an error when the route is not a bounded regular
/// non-symbolic-link file, is not exactly one canonical JSON line, names an
/// unknown directed component pair, or differs from the expected route.
pub fn verify_component_route_pin(
    schema_root: &Path,
    route_path: &Path,
) -> Result<ComponentMatrixPair, String> {
    let metadata = fs::symlink_metadata(route_path)
        .map_err(|error| format!("inspect component route pin: {error}"))?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err("component route pin must be a regular non-symbolic-link file".to_owned());
    }
    if metadata.len() > MAX_COMPONENT_ROUTE_PIN_BYTES {
        return Err(format!(
            "component route pin exceeds {MAX_COMPONENT_ROUTE_PIN_BYTES} bytes"
        ));
    }
    let file =
        File::open(route_path).map_err(|error| format!("open component route pin: {error}"))?;
    let opened = file
        .metadata()
        .map_err(|error| format!("inspect opened component route pin: {error}"))?;
    if !opened.is_file() || opened.len() > MAX_COMPONENT_ROUTE_PIN_BYTES {
        return Err("component route pin changed or exceeds its byte limit".to_owned());
    }
    let mut data = Vec::with_capacity(usize::try_from(opened.len()).unwrap_or(0));
    file.take(MAX_COMPONENT_ROUTE_PIN_BYTES + 1)
        .read_to_end(&mut data)
        .map_err(|error| format!("read component route pin: {error}"))?;
    verify_component_route_pin_bytes(schema_root, &data)
}

fn verify_component_route_pin_bytes(
    schema_root: &Path,
    data: &[u8],
) -> Result<ComponentMatrixPair, String> {
    if data.is_empty()
        || u64::try_from(data.len()).map_or(true, |length| length > MAX_COMPONENT_ROUTE_PIN_BYTES)
        || data.last() != Some(&b'\n')
        || data[..data.len().saturating_sub(1)].contains(&b'\n')
    {
        return Err(
            "component route pin must be bounded and exactly one JSON line ending with a newline"
                .to_owned(),
        );
    }
    let pinned: PinnedComponentMatrixPair = serde_json::from_slice(data)
        .map_err(|error| format!("decode component route pin: {error}"))?;
    let matrix = component_route_matrix(schema_root)?;
    let expected = matrix
        .into_iter()
        .find(|pair| pair.producer == pinned.producer && pair.consumer == pinned.consumer)
        .ok_or_else(|| {
            format!(
                "component route pin names unknown directed pair {:?} to {:?}",
                pinned.producer, pinned.consumer
            )
        })?;
    let expected_pin = owned_pair(&expected);
    let mut canonical = serde_json::to_vec(&expected_pin)
        .map_err(|error| format!("encode expected component route pin: {error}"))?;
    canonical.push(b'\n');
    if pinned != expected_pin || data != canonical {
        return Err(format!(
            "component route pin from {:?} to {:?} does not exactly match routed schema versions, paths, and digests",
            expected.producer, expected.consumer
        ));
    }
    Ok(expected)
}

fn owned_pair(pair: &ComponentMatrixPair) -> PinnedComponentMatrixPair {
    PinnedComponentMatrixPair {
        producer: pair.producer.to_owned(),
        consumer: pair.consumer.to_owned(),
        contracts: pair
            .contracts
            .iter()
            .map(|contract| PinnedComponentMatrixContract {
                contract_version: contract.contract_version.to_owned(),
                domain: contract.domain.to_owned(),
                schema: contract.schema.to_owned(),
                schema_sha256: contract.schema_sha256.clone(),
            })
            .collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::{component_route_matrix, verify_component_route_pin_bytes};
    use std::path::Path;

    #[test]
    fn matrix_covers_every_directed_pair_with_sorted_contracts() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../schemas");
        let matrix = component_route_matrix(&root).expect("build component route matrix");
        assert_eq!(matrix.len(), 81);
        assert!(matrix.windows(2).all(|pair| {
            (pair[0].producer, pair[0].consumer) < (pair[1].producer, pair[1].consumer)
        }));
        for pair in matrix {
            assert!(!pair.contracts.is_empty());
            assert!(pair.contracts.windows(2).all(|contracts| {
                (contracts[0].contract_version, contracts[0].domain)
                    < (contracts[1].contract_version, contracts[1].domain)
            }));
            assert!(pair
                .contracts
                .iter()
                .all(|contract| contract.schema_sha256.len() == 64));
        }
    }

    #[test]
    fn platform_to_inference_engine_has_exact_shared_routes() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../schemas");
        let matrix = component_route_matrix(&root).expect("build component route matrix");
        let pair = matrix
            .iter()
            .find(|pair| {
                pair.producer == "personal_ai_platform"
                    && pair.consumer == "native_inference_engine"
            })
            .expect("platform to inference engine pair");
        assert_eq!(pair.contracts.len(), 12);
    }

    #[test]
    fn verifies_every_canonical_component_route_pin() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../schemas");
        let canonical = include_str!("../../compatibility/component-route-matrix.jsonl");
        for (index, line) in canonical.lines().enumerate() {
            let data = format!("{line}\n");
            let pair = verify_component_route_pin_bytes(&root, data.as_bytes())
                .expect("verify canonical route");
            assert_eq!(pair.producer, super::ALL_COMPONENTS[index / 9]);
            assert_eq!(pair.consumer, super::ALL_COMPONENTS[index % 9]);
        }
    }

    #[test]
    fn rejects_tampered_malformed_and_oversized_component_route_pins() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../schemas");
        let line = include_str!("../../compatibility/component-route-matrix.jsonl")
            .lines()
            .next()
            .expect("first route");
        let tampered = format!("{}\n", line.replacen("a8a68398", "08a68398", 1));
        assert!(verify_component_route_pin_bytes(&root, tampered.as_bytes())
            .expect_err("tampered digest")
            .contains("does not exactly match"));
        let unknown = format!(
            "{}\n",
            line.replacen("{\"producer\":", "{\"unknown\":true,\"producer\":", 1)
        );
        assert!(verify_component_route_pin_bytes(&root, unknown.as_bytes())
            .expect_err("unknown field")
            .contains("unknown field"));
        let duplicate = format!(
            "{}\n",
            line.replacen(
                "{\"producer\":",
                "{\"producer\":\"action_broker\",\"producer\":",
                1
            )
        );
        assert!(verify_component_route_pin_bytes(&root, duplicate.as_bytes()).is_err());
        let oversized = vec![
            b'x';
            usize::try_from(super::MAX_COMPONENT_ROUTE_PIN_BYTES)
                .expect("route limit fits usize")
                + 1
        ];
        assert!(verify_component_route_pin_bytes(&root, &oversized)
            .expect_err("oversized route")
            .contains("bounded"));
    }
}
