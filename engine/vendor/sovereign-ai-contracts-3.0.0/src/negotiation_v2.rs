use crate::common::{
    check_len, compare_timestamp_values, validate_reason_code, validate_semver, validate_sha256,
    validate_timestamp, validate_timestamp_order, Component,
};
use crate::error::{ContractError, Result, Validate};
use crate::identity::RequestId;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::cmp::Ordering;
use std::collections::BTreeSet;

pub const PRODUCTION_NEGOTIATION_CONTRACT_VERSION: &str = "2.0.0";

fn valid_domain(value: &str) -> bool {
    if value.is_empty() || value.len() > 128 || !value.is_ascii() {
        return false;
    }
    let segments: Vec<&str> = value.split('.').collect();
    !segments.is_empty()
        && segments.len() <= 8
        && segments.iter().all(|segment| {
            !segment.is_empty()
                && segment.len() <= 32
                && segment.as_bytes()[0].is_ascii_lowercase()
                && segment.bytes().all(|byte| {
                    byte.is_ascii_lowercase()
                        || byte.is_ascii_digit()
                        || matches!(byte, b'_' | b'-')
                })
        })
}

fn schema_matches(domain: &str, version: &str, schema_id: &str) -> bool {
    let Some(path) = schema_id.strip_prefix("https://contracts.sovereign-ai.local/schemas/") else {
        return false;
    };
    let parts: Vec<&str> = path.split('/').collect();
    if parts.len() != 3
        || parts[0] != domain
        || !parts[1].starts_with('v')
        || !parts[2].ends_with(".schema.json")
    {
        return false;
    }
    let major = &parts[1][1..];
    !major.is_empty()
        && major.bytes().all(|b| b.is_ascii_digit())
        && version.starts_with(&format!("{major}."))
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ContractLifecycle {
    HistoricalDecodeOnly,
    PreReleaseUnactivated,
    ProductionEligible,
    Retired,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ProductionContractCandidate {
    pub domain: String,
    pub version: String,
    pub schema_id: String,
    pub schema_digest: String,
    pub lifecycle: ContractLifecycle,
}

impl Validate for ProductionContractCandidate {
    fn validate(&self) -> Result<()> {
        if !valid_domain(&self.domain) {
            return Err(ContractError::field(
                "domain",
                "must be a bounded namespaced identifier",
            ));
        }
        validate_semver("version", &self.version)?;
        check_len("schema_id", &self.schema_id, 1, 256)?;
        if !schema_matches(&self.domain, &self.version, &self.schema_id) {
            return Err(ContractError::field(
                "schema_id",
                "must match domain and major version",
            ));
        }
        validate_sha256("schema_digest", &self.schema_digest)
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum ProductionOfferKind {
    #[serde(rename = "contract.production_version_offer")]
    ProductionVersionOffer,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ProductionVersionOffer {
    pub contract_version: String,
    pub kind: ProductionOfferKind,
    pub negotiation_id: RequestId,
    pub sender: Component,
    pub purpose: String,
    pub domain: String,
    pub production_floor_major: u32,
    pub candidates: Vec<ProductionContractCandidate>,
    pub policy_revision: u64,
    pub policy_digest: String,
    pub offered_at: String,
    pub expires_at: String,
    pub nonce_digest: String,
}

impl Validate for ProductionVersionOffer {
    fn validate(&self) -> Result<()> {
        if self.contract_version != PRODUCTION_NEGOTIATION_CONTRACT_VERSION
            || self.purpose != "production"
        {
            return Err(ContractError::field(
                "contract_version",
                "must be production negotiation 2.0.0",
            ));
        }
        self.negotiation_id.validate()?;
        if !valid_domain(&self.domain) {
            return Err(ContractError::field("domain", "invalid namespaced domain"));
        }
        if self.production_floor_major == 0
            || self.production_floor_major > 65_535
            || self.policy_revision == 0
        {
            return Err(ContractError::field(
                "production_policy",
                "floor and revision must be bounded and nonzero",
            ));
        }
        if !(1..=16).contains(&self.candidates.len()) {
            return Err(ContractError::field(
                "candidates",
                "must contain 1 through 16 exact versions",
            ));
        }
        let mut seen = BTreeSet::new();
        for candidate in &self.candidates {
            candidate.validate()?;
            if candidate.domain != self.domain || !seen.insert(candidate.version.clone()) {
                return Err(ContractError::field(
                    "candidates",
                    "domains must match and versions must be unique",
                ));
            }
            let major: u32 = candidate
                .version
                .split('.')
                .next()
                .and_then(|value| value.parse().ok())
                .unwrap_or(0);
            if major < self.production_floor_major {
                return Err(ContractError::field(
                    "candidates.version",
                    "is below the production floor",
                ));
            }
        }
        validate_sha256("policy_digest", &self.policy_digest)?;
        validate_sha256("nonce_digest", &self.nonce_digest)?;
        validate_timestamp("offered_at", &self.offered_at)?;
        validate_timestamp("expires_at", &self.expires_at)?;
        validate_timestamp_order(
            "offered_at",
            &self.offered_at,
            "expires_at",
            &self.expires_at,
        )
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum ProductionSelectionKind {
    #[serde(rename = "contract.production_version_selection")]
    ProductionVersionSelection,
}
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ProductionSelectionOutcome {
    Selected,
    NoCommonVersion,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ProductionVersionSelection {
    pub contract_version: String,
    pub kind: ProductionSelectionKind,
    pub negotiation_id: RequestId,
    pub responder: Component,
    pub offer_digest: String,
    pub outcome: ProductionSelectionOutcome,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub selected: Option<ProductionContractCandidate>,
    pub reason_code: String,
    pub selected_at: String,
    pub expires_at: String,
}

impl Validate for ProductionVersionSelection {
    fn validate(&self) -> Result<()> {
        if self.contract_version != PRODUCTION_NEGOTIATION_CONTRACT_VERSION {
            return Err(ContractError::field("contract_version", "must equal 2.0.0"));
        }
        self.negotiation_id.validate()?;
        validate_sha256("offer_digest", &self.offer_digest)?;
        match (&self.outcome, &self.selected) {
            (ProductionSelectionOutcome::Selected, Some(selected))
                if selected.lifecycle == ContractLifecycle::ProductionEligible =>
            {
                selected.validate()?;
            }
            (ProductionSelectionOutcome::NoCommonVersion, None)
                if self.reason_code == "contract.no_common_version" => {}
            _ => {
                return Err(ContractError::field(
                    "selection",
                    "status, selected version, and lifecycle are incompatible",
                ))
            }
        }
        validate_reason_code("reason_code", &self.reason_code)?;
        validate_timestamp("selected_at", &self.selected_at)?;
        validate_timestamp("expires_at", &self.expires_at)?;
        validate_timestamp_order(
            "selected_at",
            &self.selected_at,
            "expires_at",
            &self.expires_at,
        )
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum ProductionTranscriptKind {
    #[serde(rename = "contract.production_version_transcript")]
    ProductionVersionTranscript,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ProductionVersionTranscript {
    pub contract_version: String,
    pub kind: ProductionTranscriptKind,
    pub offer: ProductionVersionOffer,
    pub selection: ProductionVersionSelection,
    pub verified_at: String,
}

impl Validate for ProductionVersionTranscript {
    fn validate(&self) -> Result<()> {
        if self.contract_version != PRODUCTION_NEGOTIATION_CONTRACT_VERSION {
            return Err(ContractError::field("contract_version", "must equal 2.0.0"));
        }
        self.offer.validate()?;
        self.selection.validate()?;
        if self.offer.negotiation_id != self.selection.negotiation_id
            || production_offer_digest(&self.offer)? != self.selection.offer_digest
        {
            return Err(ContractError::field(
                "selection",
                "must bind the exact offer",
            ));
        }
        if let Some(selected) = &self.selection.selected {
            let major: u32 = selected
                .version
                .split('.')
                .next()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0);
            if major < self.offer.production_floor_major
                || selected.domain != self.offer.domain
                || !self
                    .offer
                    .candidates
                    .iter()
                    .any(|candidate| candidate == selected)
            {
                return Err(ContractError::field(
                    "selected",
                    "must be exactly offered at or above the production floor",
                ));
            }
        }
        validate_timestamp("verified_at", &self.verified_at)?;
        if compare_timestamp_values(&self.verified_at, &self.selection.selected_at)?
            == Ordering::Less
            || compare_timestamp_values(&self.verified_at, &self.selection.expires_at)?
                != Ordering::Less
        {
            return Err(ContractError::field(
                "verified_at",
                "must be at or after selection and before expiry",
            ));
        }
        Ok(())
    }
}

/// Computes the deterministic SHA-256 digest of the canonical production offer.
///
/// # Errors
///
/// Returns an error when the offer cannot be represented as canonical JSON.
pub fn production_offer_digest(offer: &ProductionVersionOffer) -> Result<String> {
    let value = serde_json::to_value(offer)?;
    let canonical = serde_json::to_vec(&crate::document::sort_value(value))?;
    Ok(format!("{:x}", Sha256::digest(canonical)))
}
