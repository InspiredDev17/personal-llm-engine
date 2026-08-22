//! Experimental Knowledge 1.2 exact-byte static-prefix acquisition contracts.

use crate::common::{
    timestamp_unix_millis, validate_extensions, validate_reason_code, validate_sha256,
    validate_timestamp, validate_timestamp_order, Component, Extensions, RedactedReference,
    RequiredExtensions,
};
use crate::error::{ContractError, Result, Validate};
use crate::identity::RequestId;
use crate::knowledge::{KnowledgeProducerIdentity, StaticBundleReference};
use crate::knowledge_lifecycle::CacheAuthorizationScope;
use crate::memory::SensitivityClass;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub const KNOWLEDGE_STATIC_PREFIX_CONTRACT_VERSION: &str = "1.2.0";
const MAXIMUM_PREFIX_BYTES: usize = 8 * 1024 * 1024;
const MAXIMUM_PREFIX_MEMBERS: usize = 256;
const MAXIMUM_WINDOW_MILLIS: u64 = 5 * 60 * 1000;
const PREFIX_MAGIC: &[u8] = b"SOVEREIGN-STATIC-KNOWLEDGE-PREFIX-V1\n";

fn field(name: &'static str, message: impl Into<String>) -> ContractError {
    ContractError::field(name, message)
}

fn sensitivity_rank(value: &SensitivityClass) -> u8 {
    match value {
        SensitivityClass::Public => 0,
        SensitivityClass::Internal => 1,
        SensitivityClass::Personal => 2,
        SensitivityClass::Sensitive => 3,
        SensitivityClass::Restricted => 4,
    }
}

fn validate_window(start: &str, end: &str) -> Result<()> {
    validate_timestamp_order("start", start, "expires_at", end)?;
    let start = timestamp_unix_millis("start", start)?;
    let end = timestamp_unix_millis("expires_at", end)?;
    if end - start > MAXIMUM_WINDOW_MILLIS {
        return Err(field("expires_at", "freshness window exceeds five minutes"));
    }
    Ok(())
}

fn validate_active(label: &'static str, start: &str, end: &str, now: &str) -> Result<()> {
    let start = timestamp_unix_millis("start", start)?;
    let end = timestamp_unix_millis("expires_at", end)?;
    let now = timestamp_unix_millis("now", now)?;
    if now < start {
        return Err(field(label, "is not active yet"));
    }
    if now >= end {
        return Err(field(label, "is expired"));
    }
    Ok(())
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum StaticPrefixMaterialStatus {
    Available,
    NotFound,
    Rejected,
    Expired,
    Failed,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct StaticPrefixMaterialRequest {
    pub contract_version: String,
    pub kind: String,
    pub request_id: RequestId,
    pub authorization_scope: CacheAuthorizationScope,
    pub bundle: StaticBundleReference,
    pub expected_sensitivity: SensitivityClass,
    pub sensitivity_ceiling: SensitivityClass,
    pub prefix_reference: RedactedReference,
    pub idempotency_key: String,
    pub requested_at: String,
    pub expires_at: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extensions: Option<Extensions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub required_extensions: Option<RequiredExtensions>,
}

impl Validate for StaticPrefixMaterialRequest {
    fn validate(&self) -> Result<()> {
        if self.contract_version != KNOWLEDGE_STATIC_PREFIX_CONTRACT_VERSION {
            return Err(field("contract_version", "must equal 1.2.0"));
        }
        if self.kind != "knowledge.static_prefix_material_request" {
            return Err(field(
                "kind",
                "must equal knowledge.static_prefix_material_request",
            ));
        }
        validate_extensions(self.extensions.as_ref(), self.required_extensions.as_ref())?;
        self.request_id.validate()?;
        self.authorization_scope.validate()?;
        self.bundle.validate()?;
        if sensitivity_rank(&self.expected_sensitivity)
            > sensitivity_rank(&self.sensitivity_ceiling)
        {
            return Err(field(
                "sensitivity_ceiling",
                "is below expected source sensitivity",
            ));
        }
        self.prefix_reference.validate()?;
        if self.prefix_reference.digest != self.bundle.ordered_prefix_digest {
            return Err(field(
                "prefix_reference.digest",
                "does not bind ordered_prefix_digest",
            ));
        }
        validate_sha256("idempotency_key", &self.idempotency_key)?;
        validate_window(&self.requested_at, &self.expires_at)
    }
}

impl StaticPrefixMaterialRequest {
    /// Validates structure plus deterministic activity at a caller-supplied time.
    ///
    /// # Errors
    ///
    /// Returns an error when the request is invalid, premature, stale, or expired.
    pub fn validate_at(&self, now: &str) -> Result<()> {
        self.validate()?;
        validate_active("request", &self.requested_at, &self.expires_at, now)
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct StaticPrefixMaterialResult {
    pub contract_version: String,
    pub kind: String,
    pub request_id: RequestId,
    pub producer: KnowledgeProducerIdentity,
    pub bundle: StaticBundleReference,
    pub authorization_scope: CacheAuthorizationScope,
    pub sensitivity: SensitivityClass,
    pub status: StaticPrefixMaterialStatus,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub byte_length: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prefix_material_base64: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason_code: Option<String>,
    pub retrieved_at: String,
    pub verified_at: String,
    pub expires_at: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extensions: Option<Extensions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub required_extensions: Option<RequiredExtensions>,
}

impl Validate for StaticPrefixMaterialResult {
    fn validate(&self) -> Result<()> {
        if self.contract_version != KNOWLEDGE_STATIC_PREFIX_CONTRACT_VERSION {
            return Err(field("contract_version", "must equal 1.2.0"));
        }
        if self.kind != "knowledge.static_prefix_material_result" {
            return Err(field(
                "kind",
                "must equal knowledge.static_prefix_material_result",
            ));
        }
        validate_extensions(self.extensions.as_ref(), self.required_extensions.as_ref())?;
        self.request_id.validate()?;
        self.producer.validate()?;
        if self.producer.component != Component::MemoryVault {
            return Err(field("producer.component", "must equal memory_vault"));
        }
        self.bundle.validate()?;
        self.authorization_scope.validate()?;
        validate_timestamp("retrieved_at", &self.retrieved_at)?;
        validate_timestamp_order(
            "retrieved_at",
            &self.retrieved_at,
            "verified_at",
            &self.verified_at,
        )?;
        validate_window(&self.verified_at, &self.expires_at)?;
        match (
            &self.status,
            self.byte_length,
            self.prefix_material_base64.as_ref(),
            self.reason_code.as_ref(),
        ) {
            (StaticPrefixMaterialStatus::Available, Some(length), Some(material), None) => {
                decode_static_prefix_material(
                    material,
                    length,
                    &self.bundle.ordered_prefix_digest,
                )?;
            }
            (
                StaticPrefixMaterialStatus::NotFound
                | StaticPrefixMaterialStatus::Rejected
                | StaticPrefixMaterialStatus::Expired
                | StaticPrefixMaterialStatus::Failed,
                None,
                None,
                Some(reason),
            ) => validate_reason_code("reason_code", reason)?,
            _ => return Err(field("result", "status payload combination is invalid")),
        }
        Ok(())
    }
}

impl StaticPrefixMaterialResult {
    /// Validates structure plus deterministic freshness at a caller-supplied time.
    ///
    /// # Errors
    ///
    /// Returns an error when the result is invalid, premature, stale, or expired.
    pub fn validate_at(&self, now: &str) -> Result<()> {
        self.validate()?;
        validate_active("result", &self.verified_at, &self.expires_at, now)
    }

    /// Correlates the result with the exact request and rejects substitution.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid input, identity substitution, scope or
    /// sensitivity drift, or a result outside the request freshness boundary.
    pub fn validate_against(&self, request: &StaticPrefixMaterialRequest) -> Result<()> {
        self.validate()?;
        request.validate()?;
        if self.request_id != request.request_id
            || self.bundle != request.bundle
            || self.authorization_scope != request.authorization_scope
            || self.sensitivity != request.expected_sensitivity
        {
            return Err(field(
                "result",
                "request identity, bundle, scope, or sensitivity substitution",
            ));
        }
        if sensitivity_rank(&self.sensitivity) > sensitivity_rank(&request.sensitivity_ceiling) {
            return Err(field("sensitivity", "exceeds request ceiling"));
        }
        validate_timestamp_order(
            "requested_at",
            &request.requested_at,
            "retrieved_at",
            &self.retrieved_at,
        )?;
        let request_expiry = timestamp_unix_millis("expires_at", &request.expires_at)?;
        let result_verified = timestamp_unix_millis("verified_at", &self.verified_at)?;
        let result_expiry = timestamp_unix_millis("expires_at", &self.expires_at)?;
        if result_verified > request_expiry || result_expiry > request_expiry {
            return Err(field(
                "expires_at",
                "result exceeds request freshness boundary",
            ));
        }
        Ok(())
    }
}

/// Strictly decodes exact sensitive bytes; callers must never log the output.
///
/// # Errors
///
/// Returns an error for non-canonical base64, invalid length, malformed
/// framing, invalid UTF-8, NUL, or an exact-byte digest mismatch.
pub fn decode_static_prefix_material(
    encoded: &str,
    declared_length: u64,
    expected_digest: &str,
) -> Result<Vec<u8>> {
    if declared_length == 0 || declared_length > MAXIMUM_PREFIX_BYTES as u64 {
        return Err(field("byte_length", "out of range"));
    }
    let decoded = decode_canonical_base64(encoded)?;
    if decoded.len() as u64 != declared_length {
        return Err(field(
            "byte_length",
            "does not equal decoded material length",
        ));
    }
    validate_prefix_framing(&decoded)?;
    validate_sha256("ordered_prefix_digest", expected_digest)?;
    if format!("{:x}", Sha256::digest(&decoded)) != expected_digest {
        return Err(field(
            "ordered_prefix_digest",
            "does not match exact decoded bytes",
        ));
    }
    Ok(decoded)
}

fn decode_canonical_base64(value: &str) -> Result<Vec<u8>> {
    if value.is_empty()
        || value.len() > MAXIMUM_PREFIX_BYTES.div_ceil(3) * 4
        || value.len() % 4 != 0
    {
        return Err(field(
            "prefix_material_base64",
            "must be bounded canonical padded RFC 4648 base64",
        ));
    }
    let bytes = value.as_bytes();
    let mut out = Vec::with_capacity(value.len() / 4 * 3);
    for (chunk_index, chunk) in bytes.chunks_exact(4).enumerate() {
        let last = chunk_index + 1 == bytes.len() / 4;
        let pad = if chunk[3] == b'=' {
            if chunk[2] == b'=' {
                2
            } else {
                1
            }
        } else {
            0
        };
        if (!last && pad != 0)
            || chunk[0] == b'='
            || chunk[1] == b'='
            || (chunk[2] == b'=' && chunk[3] != b'=')
        {
            return Err(field("prefix_material_base64", "has invalid padding"));
        }
        let a = base64_value(chunk[0])?;
        let b = base64_value(chunk[1])?;
        let c = if chunk[2] == b'=' {
            0
        } else {
            base64_value(chunk[2])?
        };
        let d = if chunk[3] == b'=' {
            0
        } else {
            base64_value(chunk[3])?
        };
        if (pad == 2 && b & 0x0f != 0) || (pad == 1 && c & 0x03 != 0) {
            return Err(field(
                "prefix_material_base64",
                "has non-canonical trailing bits",
            ));
        }
        out.push((a << 2) | (b >> 4));
        if pad < 2 {
            out.push((b << 4) | (c >> 2));
        }
        if pad == 0 {
            out.push((c << 6) | d);
        }
    }
    Ok(out)
}

fn base64_value(value: u8) -> Result<u8> {
    match value {
        b'A'..=b'Z' => Ok(value - b'A'),
        b'a'..=b'z' => Ok(value - b'a' + 26),
        b'0'..=b'9' => Ok(value - b'0' + 52),
        b'+' => Ok(62),
        b'/' => Ok(63),
        _ => Err(field(
            "prefix_material_base64",
            "contains a non-base64 character",
        )),
    }
}

fn validate_prefix_framing(value: &[u8]) -> Result<()> {
    if value.len() <= PREFIX_MAGIC.len()
        || value.len() > MAXIMUM_PREFIX_BYTES
        || std::str::from_utf8(value).is_err()
        || value.contains(&0)
        || !value.starts_with(PREFIX_MAGIC)
    {
        return Err(field(
            "prefix_material_base64",
            "decoded material has invalid framing, UTF-8, or NUL",
        ));
    }
    let mut offset = PREFIX_MAGIC.len();
    let mut members = 0;
    while offset < value.len() {
        if members == MAXIMUM_PREFIX_MEMBERS {
            return Err(field(
                "prefix_material_base64",
                "decoded material has too many members",
            ));
        }
        let start = offset;
        let mut length = 0_u64;
        while offset < value.len() && value[offset] != b':' {
            if !value[offset].is_ascii_digit() || offset - start >= 10 {
                return Err(field(
                    "prefix_material_base64",
                    "decoded material has malformed decimal framing",
                ));
            }
            length = length * 10 + u64::from(value[offset] - b'0');
            offset += 1;
        }
        if offset == start
            || offset >= value.len()
            || length == 0
            || (offset - start > 1 && value[start] == b'0')
            || length > (value.len() - offset - 1) as u64
        {
            return Err(field(
                "prefix_material_base64",
                "decoded material is empty, truncated, or non-canonical",
            ));
        }
        offset += 1;
        offset += usize::try_from(length)
            .map_err(|_| field("prefix_material_base64", "member length overflows usize"))?;
        members += 1;
    }
    if members == 0 || offset != value.len() {
        return Err(field(
            "prefix_material_base64",
            "decoded material contains trailing bytes",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{decode_static_prefix_material, MAXIMUM_PREFIX_BYTES, PREFIX_MAGIC};
    use sha2::{Digest, Sha256};

    fn encode_base64(value: &[u8]) -> String {
        const TABLE: &[u8; 64] =
            b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        let mut out = String::with_capacity(value.len().div_ceil(3) * 4);
        for chunk in value.chunks(3) {
            let a = chunk[0];
            let b = chunk.get(1).copied().unwrap_or(0);
            let c = chunk.get(2).copied().unwrap_or(0);
            out.push(TABLE[(a >> 2) as usize] as char);
            out.push(TABLE[(((a & 0x03) << 4) | (b >> 4)) as usize] as char);
            if chunk.len() > 1 {
                out.push(TABLE[(((b & 0x0f) << 2) | (c >> 6)) as usize] as char);
            } else {
                out.push('=');
            }
            if chunk.len() > 2 {
                out.push(TABLE[(c & 0x3f) as usize] as char);
            } else {
                out.push('=');
            }
        }
        out
    }

    #[test]
    fn exact_maximum_prefix_boundary_is_accepted() {
        let mut digits = MAXIMUM_PREFIX_BYTES.to_string().len();
        let mut content_length = MAXIMUM_PREFIX_BYTES - PREFIX_MAGIC.len() - digits - 1;
        while content_length.to_string().len() != digits {
            digits = content_length.to_string().len();
            content_length = MAXIMUM_PREFIX_BYTES - PREFIX_MAGIC.len() - digits - 1;
        }
        let mut material = Vec::with_capacity(MAXIMUM_PREFIX_BYTES);
        material.extend_from_slice(PREFIX_MAGIC);
        material.extend_from_slice(content_length.to_string().as_bytes());
        material.push(b':');
        material.resize(MAXIMUM_PREFIX_BYTES, b'a');
        let digest = format!("{:x}", Sha256::digest(&material));
        let encoded = encode_base64(&material);
        assert!(decode_static_prefix_material(&encoded, material.len() as u64, &digest).is_ok());
    }

    #[test]
    fn alternate_or_malformed_base64_fails_closed() {
        let digest = "e556742d698555c58cbd9e2c752ff00bfa82978b81c154015863dd2cec7309c4";
        assert!(decode_static_prefix_material(
            "U09WRVJFSUdOLVNUQVRJQy1LTk9XTEVER0UtUFJFRklYLVYxCjU6aGVsbG8=",
            44,
            digest
        )
        .is_ok());
        for malformed in ["not-base64", "YR==", "YQ", "YQ==="] {
            assert!(decode_static_prefix_material(malformed, 44, digest).is_err());
        }
    }
}
