//! Experimental, unactivated execution-package identity bindings.
//!
//! These types describe content-free package identity. They do not parse model
//! packages, authorize loading, qualify models, or activate execution.

use crate::common::{check_len, validate_semver, validate_timestamp};
use crate::error::{ContractError, Result, Validate};
use crate::inference_runtime_v3::{ArtifactDigestAlgorithmV3, RuntimeIdentityV3};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::collections::BTreeMap;

pub const EXECUTION_PACKAGE_DESCRIPTOR_CONTRACT_VERSION: &str = "2.0.0";
const MAXIMUM_EXECUTION_PACKAGE_BYTES: u64 = 1_099_511_627_776;
const MAXIMUM_EXECUTION_PACKAGE_MEMBERS: usize = 2_048;
const MAXIMUM_TOKENIZER_BYTES: u64 = 1_073_741_824;
const MAXIMUM_MODEL_VOCABULARY_SIZE: u32 = 10_000_000;
const MAXIMUM_EXECUTION_CONTEXT: u32 = 10_000_000;
const MAXIMUM_TENSOR_INVENTORY: u32 = 1_000_000;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ExecutionPackageDescriptorContractVersion;

impl Serialize for ExecutionPackageDescriptorContractVersion {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(EXECUTION_PACKAGE_DESCRIPTOR_CONTRACT_VERSION)
    }
}

impl<'de> Deserialize<'de> for ExecutionPackageDescriptorContractVersion {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        if value == EXECUTION_PACKAGE_DESCRIPTOR_CONTRACT_VERSION {
            Ok(Self)
        } else {
            Err(serde::de::Error::custom(format!(
                "unsupported contract_version {value:?}; expected {EXECUTION_PACKAGE_DESCRIPTOR_CONTRACT_VERSION}"
            )))
        }
    }
}

impl Validate for ExecutionPackageDescriptorContractVersion {
    fn validate(&self) -> Result<()> {
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PackageDigest {
    pub algorithm: ArtifactDigestAlgorithmV3,
    pub digest: String,
}

impl Validate for PackageDigest {
    fn validate(&self) -> Result<()> {
        validate_package_digest(self)
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionPackageMemberRole {
    PackageManifest,
    TensorInventory,
    Tokenizer,
    TensorShard,
    IntegrityCatalog,
    IntegritySignature,
    Provenance,
    Auxiliary,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ExecutionPackageMember {
    pub member_id: String,
    pub ordinal: u16,
    pub role: ExecutionPackageMemberRole,
    pub content: PackageDigest,
    pub size_bytes: u64,
}

impl Validate for ExecutionPackageMember {
    fn validate(&self) -> Result<()> {
        validate_package_token("member_id", &self.member_id)?;
        if usize::from(self.ordinal) >= MAXIMUM_EXECUTION_PACKAGE_MEMBERS {
            return Err(ContractError::field(
                "ordinal",
                "exceeds package member bound",
            ));
        }
        self.content.validate()?;
        if self.size_bytes == 0 || self.size_bytes > MAXIMUM_EXECUTION_PACKAGE_BYTES {
            return Err(ContractError::field(
                "size_bytes",
                "is outside the execution-package bound",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TokenizerArtifactBinding {
    pub tokenizer_id: String,
    pub member_id: String,
    pub format_id: String,
    pub format_version: String,
    pub content: PackageDigest,
    pub size_bytes: u64,
    pub model_vocabulary_size: u32,
    pub defined_token_count: u32,
    pub selectable_token_count: u32,
}

impl Validate for TokenizerArtifactBinding {
    fn validate(&self) -> Result<()> {
        for (field, value) in [
            ("tokenizer_id", self.tokenizer_id.as_str()),
            ("member_id", self.member_id.as_str()),
            ("format_id", self.format_id.as_str()),
        ] {
            validate_package_token(field, value)?;
        }
        validate_semver("format_version", &self.format_version)?;
        self.content.validate()?;
        if self.size_bytes == 0 || self.size_bytes > MAXIMUM_TOKENIZER_BYTES {
            return Err(ContractError::field(
                "tokenizer.size_bytes",
                "is outside the tokenizer byte bound",
            ));
        }
        if self.model_vocabulary_size == 0
            || self.model_vocabulary_size > MAXIMUM_MODEL_VOCABULARY_SIZE
        {
            return Err(ContractError::field(
                "tokenizer.model_vocabulary_size",
                "is outside the model vocabulary bound",
            ));
        }
        if self.defined_token_count == 0 || self.defined_token_count > self.model_vocabulary_size {
            return Err(ContractError::field(
                "tokenizer.defined_token_count",
                "must be nonzero and no greater than model_vocabulary_size",
            ));
        }
        if self.selectable_token_count == 0
            || self.selectable_token_count > self.defined_token_count
        {
            return Err(ContractError::field(
                "tokenizer.selectable_token_count",
                "must be nonzero and no greater than defined_token_count",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ArchitectureExecutionBinding {
    pub architecture_id: String,
    pub position_semantics_id: String,
    pub context_limit: u32,
}

impl Validate for ArchitectureExecutionBinding {
    fn validate(&self) -> Result<()> {
        validate_package_token("architecture_id", &self.architecture_id)?;
        validate_package_token("position_semantics_id", &self.position_semantics_id)?;
        if self.context_limit == 0 || self.context_limit > MAXIMUM_EXECUTION_CONTEXT {
            return Err(ContractError::field(
                "context_limit",
                "is outside the execution context bound",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PackageStorageDtype {
    F32,
    F16,
    Bf16,
    I8,
    I4,
    Mixed,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct QuantizationBinding {
    pub profile_id: String,
    pub storage_dtype: PackageStorageDtype,
}

impl Validate for QuantizationBinding {
    fn validate(&self) -> Result<()> {
        validate_package_token("profile_id", &self.profile_id)
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TensorInventoryBinding {
    pub member_id: String,
    pub content: PackageDigest,
    pub tensor_count: u32,
}

impl Validate for TensorInventoryBinding {
    fn validate(&self) -> Result<()> {
        validate_package_token("member_id", &self.member_id)?;
        self.content.validate()?;
        if self.tensor_count == 0 || self.tensor_count > MAXIMUM_TENSOR_INVENTORY {
            return Err(ContractError::field(
                "tensor_count",
                "is outside the tensor inventory bound",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum ExecutionPackageDescriptorKind {
    #[serde(rename = "model.execution_package_descriptor")]
    ExecutionPackageDescriptor,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ExecutionPackageDescriptor {
    pub contract_version: ExecutionPackageDescriptorContractVersion,
    pub kind: ExecutionPackageDescriptorKind,
    pub runtime_identity: RuntimeIdentityV3,
    pub package_identity_member_id: String,
    pub package_format_id: String,
    pub package_format_version: String,
    pub package_size_bytes: u64,
    pub tokenizer: TokenizerArtifactBinding,
    pub architecture: ArchitectureExecutionBinding,
    pub quantization: QuantizationBinding,
    pub tensor_inventory: TensorInventoryBinding,
    pub members: Vec<ExecutionPackageMember>,
    pub created_at: String,
}

impl Validate for ExecutionPackageDescriptor {
    #[allow(clippy::too_many_lines)]
    fn validate(&self) -> Result<()> {
        self.runtime_identity.validate()?;
        if all_zero_hex(&self.runtime_identity.artifact_digest) {
            return Err(ContractError::field(
                "runtime_identity.artifact_digest",
                "must not be all zero",
            ));
        }
        validate_package_token(
            "package_identity_member_id",
            &self.package_identity_member_id,
        )?;
        validate_package_token("package_format_id", &self.package_format_id)?;
        validate_semver("package_format_version", &self.package_format_version)?;
        if self.package_size_bytes == 0 || self.package_size_bytes > MAXIMUM_EXECUTION_PACKAGE_BYTES
        {
            return Err(ContractError::field(
                "package_size_bytes",
                "is outside the execution-package bound",
            ));
        }
        self.tokenizer.validate()?;
        self.architecture.validate()?;
        self.quantization.validate()?;
        self.tensor_inventory.validate()?;
        if self.members.len() < 3 || self.members.len() > MAXIMUM_EXECUTION_PACKAGE_MEMBERS {
            return Err(ContractError::field(
                "members",
                "must contain 3 through 2048 entries",
            ));
        }

        let mut members = BTreeMap::new();
        let mut role_counts = BTreeMap::<ExecutionPackageMemberRole, usize>::new();
        let mut total = 0_u64;
        for (index, member) in self.members.iter().enumerate() {
            member.validate()?;
            if usize::from(member.ordinal) != index {
                return Err(ContractError::field(
                    "members",
                    "ordinals must be canonical, contiguous, and match array order",
                ));
            }
            if members.insert(member.member_id.as_str(), member).is_some() {
                return Err(ContractError::field(
                    "members",
                    "contains duplicate member_id",
                ));
            }
            *role_counts.entry(member.role).or_default() += 1;
            total = total
                .checked_add(member.size_bytes)
                .filter(|value| *value <= MAXIMUM_EXECUTION_PACKAGE_BYTES)
                .ok_or_else(|| {
                    ContractError::field("members", "total size overflows package bound")
                })?;
        }
        if total != self.package_size_bytes {
            return Err(ContractError::field(
                "package_size_bytes",
                "must equal the sum of exact member sizes",
            ));
        }
        if role_count(&role_counts, ExecutionPackageMemberRole::PackageManifest) != 1
            || role_count(&role_counts, ExecutionPackageMemberRole::Tokenizer) != 1
            || role_count(&role_counts, ExecutionPackageMemberRole::TensorShard) == 0
        {
            return Err(ContractError::field(
                "members",
                "must contain exactly one package_manifest, exactly one tokenizer, and at least one tensor_shard",
            ));
        }
        for role in [
            ExecutionPackageMemberRole::TensorInventory,
            ExecutionPackageMemberRole::IntegrityCatalog,
            ExecutionPackageMemberRole::IntegritySignature,
            ExecutionPackageMemberRole::Provenance,
        ] {
            if role_count(&role_counts, role) > 1 {
                return Err(ContractError::field(
                    "members",
                    "singleton member role appears more than once",
                ));
            }
        }

        let identity_member = members
            .get(self.package_identity_member_id.as_str())
            .ok_or_else(|| {
                ContractError::field(
                    "package_identity_member_id",
                    "references an unavailable member",
                )
            })?;
        let identity_content = PackageDigest {
            algorithm: self.runtime_identity.artifact_digest_algorithm.clone(),
            digest: self.runtime_identity.artifact_digest.clone(),
        };
        if !matches!(
            identity_member.role,
            ExecutionPackageMemberRole::PackageManifest
                | ExecutionPackageMemberRole::IntegrityCatalog
        ) || identity_member.content != identity_content
        {
            return Err(ContractError::field(
                "package_identity_member_id",
                "must identify a package_manifest or integrity_catalog matching runtime_identity.artifact_digest",
            ));
        }

        let tokenizer_member = members
            .get(self.tokenizer.member_id.as_str())
            .ok_or_else(|| ContractError::field("tokenizer", "references an unavailable member"))?;
        if tokenizer_member.role != ExecutionPackageMemberRole::Tokenizer
            || tokenizer_member.content != self.tokenizer.content
            || tokenizer_member.size_bytes != self.tokenizer.size_bytes
        {
            return Err(ContractError::field(
                "tokenizer",
                "must exactly match its tokenizer member",
            ));
        }

        let inventory_member = members
            .get(self.tensor_inventory.member_id.as_str())
            .ok_or_else(|| {
                ContractError::field("tensor_inventory", "references an unavailable member")
            })?;
        if !matches!(
            inventory_member.role,
            ExecutionPackageMemberRole::PackageManifest
                | ExecutionPackageMemberRole::TensorInventory
        ) || inventory_member.content != self.tensor_inventory.content
        {
            return Err(ContractError::field(
                "tensor_inventory",
                "must exactly match a package_manifest or tensor_inventory member",
            ));
        }
        validate_timestamp("created_at", &self.created_at)
    }
}

fn role_count(
    counts: &BTreeMap<ExecutionPackageMemberRole, usize>,
    role: ExecutionPackageMemberRole,
) -> usize {
    counts.get(&role).copied().unwrap_or_default()
}

fn validate_package_digest(digest: &PackageDigest) -> Result<()> {
    let expected = match digest.algorithm {
        ArtifactDigestAlgorithmV3::Sha256 => 64,
        ArtifactDigestAlgorithmV3::Sha384 => 96,
    };
    if digest.digest.len() != expected
        || !digest
            .digest
            .bytes()
            .all(|byte| byte.is_ascii_digit() || matches!(byte, b'a'..=b'f'))
        || all_zero_hex(&digest.digest)
    {
        return Err(ContractError::field(
            "digest",
            "does not match its declared algorithm or is all zero",
        ));
    }
    Ok(())
}

fn validate_package_token(field: &'static str, value: &str) -> Result<()> {
    check_len(field, value, 2, 128)?;
    let mut bytes = value.bytes();
    if !bytes.next().is_some_and(|byte| byte.is_ascii_lowercase())
        || !bytes.all(|byte| {
            byte.is_ascii_lowercase()
                || byte.is_ascii_digit()
                || matches!(byte, b'.' | b'_' | b':' | b'-')
        })
        || value.ends_with(['.', '_', ':', '-'])
        || value
            .as_bytes()
            .windows(2)
            .any(|pair| matches!(pair, [b'.' | b'_' | b':' | b'-', b'.' | b'_' | b':' | b'-']))
    {
        return Err(ContractError::field(
            field,
            "invalid path-free package identifier",
        ));
    }
    Ok(())
}

fn all_zero_hex(value: &str) -> bool {
    !value.is_empty() && value.bytes().all(|byte| byte == b'0')
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::identity::{ModelArtifactId, ModelId};

    fn digest(byte: char) -> PackageDigest {
        PackageDigest {
            algorithm: ArtifactDigestAlgorithmV3::Sha384,
            digest: byte.to_string().repeat(96),
        }
    }

    fn descriptor() -> ExecutionPackageDescriptor {
        let manifest = ExecutionPackageMember {
            member_id: "manifest".to_owned(),
            ordinal: 0,
            role: ExecutionPackageMemberRole::PackageManifest,
            content: digest('a'),
            size_bytes: 512,
        };
        let tokenizer = ExecutionPackageMember {
            member_id: "tokenizer".to_owned(),
            ordinal: 1,
            role: ExecutionPackageMemberRole::Tokenizer,
            content: digest('b'),
            size_bytes: 4_096,
        };
        let shard = ExecutionPackageMember {
            member_id: "tensor_shard.0000".to_owned(),
            ordinal: 2,
            role: ExecutionPackageMemberRole::TensorShard,
            content: digest('c'),
            size_bytes: 8_192,
        };
        ExecutionPackageDescriptor {
            contract_version: ExecutionPackageDescriptorContractVersion,
            kind: ExecutionPackageDescriptorKind::ExecutionPackageDescriptor,
            runtime_identity: RuntimeIdentityV3 {
                model_id: ModelId::new("mdl_fixture_package").expect("model ID"),
                provider_model_id: "provider.fixture.package".to_owned(),
                artifact_id: ModelArtifactId::new("art_fixture_package").expect("artifact ID"),
                artifact_digest_algorithm: ArtifactDigestAlgorithmV3::Sha384,
                artifact_digest: "a".repeat(96),
                backend_id: "backend.fixture".to_owned(),
                backend_version: "version.1".to_owned(),
                backend_instance_id: "backend.fixture.one".to_owned(),
                engine_instance_id: "engine.fixture".to_owned(),
            },
            package_identity_member_id: "manifest".to_owned(),
            package_format_id: "personalai.directory_paimodel".to_owned(),
            package_format_version: "1.1.0".to_owned(),
            package_size_bytes: 12_800,
            tokenizer: TokenizerArtifactBinding {
                tokenizer_id: "tokenizer.fixture".to_owned(),
                member_id: "tokenizer".to_owned(),
                format_id: "personalai.aiptok".to_owned(),
                format_version: "2.0.0".to_owned(),
                content: digest('b'),
                size_bytes: 4_096,
                model_vocabulary_size: 151_936,
                defined_token_count: 151_665,
                selectable_token_count: 151_665,
            },
            architecture: ArchitectureExecutionBinding {
                architecture_id: "qwen2.5-rms-rope-swiglu-gqa-bias-tied-v1".to_owned(),
                position_semantics_id: "rope.split_half.unscaled.v1".to_owned(),
                context_limit: 32_768,
            },
            quantization: QuantizationBinding {
                profile_id: "unquantized.bf16.v1".to_owned(),
                storage_dtype: PackageStorageDtype::Bf16,
            },
            tensor_inventory: TensorInventoryBinding {
                member_id: "manifest".to_owned(),
                content: digest('a'),
                tensor_count: 338,
            },
            members: vec![manifest, tokenizer, shard],
            created_at: "2026-08-22T12:00:00Z".to_owned(),
        }
    }

    #[test]
    fn validates_exact_execution_package_binding() {
        descriptor().validate().expect("descriptor must validate");
    }

    #[test]
    fn rejects_tokenizer_member_substitution() {
        let mut value = descriptor();
        value.tokenizer.content = digest('e');
        assert!(value.validate().is_err());
    }

    #[test]
    fn rejects_noncanonical_member_order() {
        let mut value = descriptor();
        value.members.swap(0, 1);
        assert!(value.validate().is_err());
    }

    #[test]
    fn rejects_inconsistent_tokenizer_counts() {
        let mut value = descriptor();
        value.tokenizer.defined_token_count = value.tokenizer.model_vocabulary_size + 1;
        assert!(value.validate().is_err());

        let mut value = descriptor();
        value.tokenizer.selectable_token_count = value.tokenizer.defined_token_count + 1;
        assert!(value.validate().is_err());
    }

    #[test]
    fn rejects_multiple_build_metadata_separators() {
        let mut value = descriptor();
        value.package_format_version = "1.1.0+build+other".to_owned();
        assert!(value.validate().is_err());
    }
}
