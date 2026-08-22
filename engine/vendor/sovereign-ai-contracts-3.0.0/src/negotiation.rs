use crate::common::{
    check_len, check_max, compare_timestamp_values, validate_reason_code, validate_semver,
    validate_sha256, validate_timestamp, validate_timestamp_order, Component,
};
use crate::error::{ContractError, Result, Validate};
use crate::identity::RequestId;
use crate::NEGOTIATION_CONTRACT_VERSION;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};
use std::cmp::Ordering;
use std::collections::BTreeSet;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct NegotiationContractVersion;

impl Serialize for NegotiationContractVersion {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(NEGOTIATION_CONTRACT_VERSION)
    }
}

impl<'de> Deserialize<'de> for NegotiationContractVersion {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        if value == NEGOTIATION_CONTRACT_VERSION {
            Ok(Self)
        } else {
            Err(serde::de::Error::custom(format!(
                "unsupported contract_version {value:?}; expected {NEGOTIATION_CONTRACT_VERSION}"
            )))
        }
    }
}

impl Validate for NegotiationContractVersion {
    fn validate(&self) -> Result<()> {
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ContractDomain {
    Common,
    Identity,
    Inference,
    Models,
    Memory,
    Tools,
    Tasks,
    Security,
    Audit,
    Nodes,
}

impl ContractDomain {
    fn as_str(&self) -> &'static str {
        match self {
            Self::Common => "common",
            Self::Identity => "identity",
            Self::Inference => "inference",
            Self::Models => "models",
            Self::Memory => "memory",
            Self::Tools => "tools",
            Self::Tasks => "tasks",
            Self::Security => "security",
            Self::Audit => "audit",
            Self::Nodes => "nodes",
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SupportedContract {
    pub domain: ContractDomain,
    pub version: String,
    pub schema_id: String,
    pub schema_digest: String,
}

impl Validate for SupportedContract {
    fn validate(&self) -> Result<()> {
        validate_semver("version", &self.version)?;
        check_len("schema_id", &self.schema_id, 1, 256)?;
        let prefix = "https://contracts.sovereign-ai.local/schemas/";
        let suffix = self.schema_id.strip_prefix(prefix).ok_or_else(|| {
            ContractError::field("schema_id", "must be a Sovereign AI schema identifier")
        })?;
        let mut segments = suffix.split('/');
        let domain = segments.next().unwrap_or_default();
        let major_directory = segments.next().unwrap_or_default();
        let filename = segments.next().unwrap_or_default();
        if segments.next().is_some()
            || domain != self.domain.as_str()
            || !major_directory.starts_with('v')
            || !major_directory[1..]
                .bytes()
                .all(|byte| byte.is_ascii_digit())
            || major_directory == "v0"
            || !filename.ends_with(".schema.json")
            || filename
                .bytes()
                .any(|byte| !(byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-')))
        {
            return Err(ContractError::field(
                "schema_id",
                "does not match the declared domain and versioned schema path",
            ));
        }
        let major = &major_directory[1..];
        if !self.version.starts_with(&format!("{major}.")) {
            return Err(ContractError::field(
                "schema_id",
                "major directory does not match version",
            ));
        }
        validate_sha256("schema_digest", &self.schema_digest)
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ContractVersionOffer {
    pub contract_version: NegotiationContractVersion,
    pub kind: OfferKind,
    pub negotiation_id: RequestId,
    pub sender: Component,
    pub supported_contracts: Vec<SupportedContract>,
    pub offered_at: String,
    pub expires_at: String,
    pub nonce_digest: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum OfferKind {
    #[serde(rename = "contract.version_offer")]
    VersionOffer,
}

impl Validate for ContractVersionOffer {
    fn validate(&self) -> Result<()> {
        self.negotiation_id.validate()?;
        if !(1..=64).contains(&self.supported_contracts.len()) {
            return Err(ContractError::field(
                "supported_contracts",
                "must contain 1 through 64 entries",
            ));
        }
        let mut seen = BTreeSet::new();
        for supported in &self.supported_contracts {
            supported.validate()?;
            let identity = format!("{}@{}", supported.domain.as_str(), supported.version);
            if !seen.insert(identity.clone()) {
                return Err(ContractError::field(
                    "supported_contracts",
                    format!("duplicate domain version {identity:?}"),
                ));
            }
        }
        validate_timestamp("offered_at", &self.offered_at)?;
        validate_timestamp("expires_at", &self.expires_at)?;
        validate_timestamp_order(
            "offered_at",
            &self.offered_at,
            "expires_at",
            &self.expires_at,
        )?;
        validate_sha256("nonce_digest", &self.nonce_digest)
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SelectionOutcome {
    Selected,
    NoCommonVersion,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum SelectionKind {
    #[serde(rename = "contract.version_selection")]
    VersionSelection,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ContractVersionSelection {
    pub contract_version: NegotiationContractVersion,
    pub kind: SelectionKind,
    pub negotiation_id: RequestId,
    pub responder: Component,
    pub offer_digest: String,
    pub outcome: SelectionOutcome,
    pub selected_contracts: Vec<SupportedContract>,
    pub reason_code: String,
    pub selected_at: String,
    pub expires_at: String,
}

impl Validate for ContractVersionSelection {
    fn validate(&self) -> Result<()> {
        self.negotiation_id.validate()?;
        validate_sha256("offer_digest", &self.offer_digest)?;
        check_max("selected_contracts", self.selected_contracts.len(), 10)?;
        let mut domains = BTreeSet::new();
        for selected in &self.selected_contracts {
            selected.validate()?;
            if !domains.insert(selected.domain.clone()) {
                return Err(ContractError::field(
                    "selected_contracts",
                    "a domain may be selected only once",
                ));
            }
        }
        match self.outcome {
            SelectionOutcome::Selected if self.selected_contracts.is_empty() => {
                return Err(ContractError::field(
                    "selected_contracts",
                    "must not be empty for selected outcome",
                ));
            }
            SelectionOutcome::NoCommonVersion if !self.selected_contracts.is_empty() => {
                return Err(ContractError::field(
                    "selected_contracts",
                    "must be empty when no common version exists",
                ));
            }
            SelectionOutcome::NoCommonVersion
                if self.reason_code != "contract.no_common_version" =>
            {
                return Err(ContractError::field(
                    "reason_code",
                    "must equal contract.no_common_version",
                ));
            }
            SelectionOutcome::Selected | SelectionOutcome::NoCommonVersion => {}
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
pub enum TranscriptKind {
    #[serde(rename = "contract.version_transcript")]
    VersionTranscript,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct VersionNegotiationTranscript {
    pub contract_version: NegotiationContractVersion,
    pub kind: TranscriptKind,
    pub offer: ContractVersionOffer,
    pub selection: ContractVersionSelection,
    pub verified_at: String,
}

impl Validate for VersionNegotiationTranscript {
    fn validate(&self) -> Result<()> {
        self.offer.validate()?;
        self.selection.validate()?;
        if self.offer.negotiation_id != self.selection.negotiation_id {
            return Err(ContractError::field(
                "selection.negotiation_id",
                "does not match offer",
            ));
        }
        if self.selection.offer_digest != offer_digest(&self.offer)? {
            return Err(ContractError::field(
                "selection.offer_digest",
                "does not match canonical offer digest",
            ));
        }
        if compare_timestamp_values(&self.selection.selected_at, &self.offer.offered_at)?
            != Ordering::Greater
            || compare_timestamp_values(&self.selection.selected_at, &self.offer.expires_at)?
                != Ordering::Less
        {
            return Err(ContractError::field(
                "selection.selected_at",
                "must fall within the offer validity window",
            ));
        }
        if compare_timestamp_values(&self.selection.expires_at, &self.offer.expires_at)?
            == Ordering::Greater
        {
            return Err(ContractError::field(
                "selection.expires_at",
                "must not outlive the offer",
            ));
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
        let offered: BTreeSet<String> = self
            .offer
            .supported_contracts
            .iter()
            .map(contract_identity)
            .collect();
        for selected in &self.selection.selected_contracts {
            if !offered.contains(&contract_identity(selected)) {
                return Err(ContractError::field(
                    "selection.selected_contracts",
                    format!(
                        "selected contract {}@{} was not offered",
                        selected.domain.as_str(),
                        selected.version
                    ),
                ));
            }
        }
        Ok(())
    }
}

fn contract_identity(contract: &SupportedContract) -> String {
    format!(
        "{}\0{}\0{}\0{}",
        contract.domain.as_str(),
        contract.version,
        contract.schema_id,
        contract.schema_digest
    )
}

/// Returns the lowercase SHA-256 digest of the compact canonical JSON offer.
/// The digest is correlation data, not a signature, credential, or authority.
///
/// # Errors
///
/// Returns an error if the offer cannot be represented as canonical JSON.
pub fn offer_digest(offer: &ContractVersionOffer) -> Result<String> {
    let value = serde_json::to_value(offer)?;
    let canonical = serde_json::to_vec(&sort_value(value))?;
    Ok(format!("{:x}", Sha256::digest(canonical)))
}

fn sort_value(value: Value) -> Value {
    match value {
        Value::Object(object) => {
            let mut entries: Vec<(String, Value)> = object.into_iter().collect();
            entries.sort_by(|left, right| left.0.cmp(&right.0));
            Value::Object(
                entries
                    .into_iter()
                    .map(|(key, value)| (key, sort_value(value)))
                    .collect::<Map<String, Value>>(),
            )
        }
        Value::Array(values) => Value::Array(values.into_iter().map(sort_value).collect()),
        other => other,
    }
}

#[cfg(test)]
mod tests {
    use super::{offer_digest, ContractVersionOffer, ContractVersionSelection};
    use crate::Validate;
    use std::fs;
    use std::path::PathBuf;

    #[test]
    fn offer_digest_matches_golden_selection() {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../fixtures/v1/valid");
        let offer: ContractVersionOffer = serde_json::from_slice(
            &fs::read(root.join("version-offer-1.1.0.json")).expect("read offer"),
        )
        .expect("decode offer");
        let selection: ContractVersionSelection = serde_json::from_slice(
            &fs::read(root.join("version-selection-1.1.0.json")).expect("read selection"),
        )
        .expect("decode selection");
        offer.validate().expect("validate offer");
        selection.validate().expect("validate selection");
        assert_eq!(
            offer_digest(&offer).expect("digest offer"),
            selection.offer_digest
        );
    }
}
