use crate::error::{ContractError, Result, Validate};
use crate::CONTRACT_VERSION;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::{Map, Value};
use std::cmp::Ordering;
use std::collections::BTreeMap;

/// Recursively orders every JSON object by its lexical key representation and
/// normalizes integral floating-point numbers to the canonical integer form.
///
/// This is the common canonical object-ordering rule shared by the full
/// document normalizer and the minimal Runtime V3 encoder. Array order remains
/// semantically significant and is preserved.
pub(crate) fn sort_json_value(value: Value) -> Value {
    match value {
        Value::Object(object) => {
            let mut entries: Vec<(String, Value)> = object.into_iter().collect();
            entries.sort_by(|left, right| left.0.cmp(&right.0));
            Value::Object(
                entries
                    .into_iter()
                    .map(|(key, value)| (key, sort_json_value(value)))
                    .collect::<Map<String, Value>>(),
            )
        }
        Value::Array(values) => Value::Array(values.into_iter().map(sort_json_value).collect()),
        Value::Number(number) if number.is_f64() => {
            let value = number
                .as_f64()
                .expect("a serde_json f64 number must remain representable as f64");
            if value == 0.0 {
                Value::Number(0.into())
            } else if value.fract() == 0.0 && value.abs() <= 9_007_199_254_740_991.0 {
                let integer = format!("{value:.0}");
                Value::Number(
                    integer
                        .parse()
                        .expect("a bounded integral f64 must parse as a JSON number"),
                )
            } else {
                Value::Number(number)
            }
        }
        other => other,
    }
}

#[cfg(test)]
mod canonical_json_tests {
    use super::sort_json_value;
    use serde_json::json;

    #[test]
    fn integral_generation_floats_use_canonical_integer_form() {
        let sorted = sort_json_value(json!({
            "top_p": 1.0,
            "temperature": 0.0,
            "negative_zero": -0.0,
            "fractional": 0.25,
        }));
        assert_eq!(
            serde_json::to_string(&sorted).expect("encode canonical JSON"),
            r#"{"fractional":0.25,"negative_zero":0,"temperature":0,"top_p":1}"#
        );
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ContractVersion;

impl Serialize for ContractVersion {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(CONTRACT_VERSION)
    }
}

impl<'de> Deserialize<'de> for ContractVersion {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        if value == CONTRACT_VERSION {
            Ok(Self)
        } else {
            Err(serde::de::Error::custom(format!(
                "unsupported contract_version {value:?}; expected {CONTRACT_VERSION}"
            )))
        }
    }
}

impl Validate for ContractVersion {
    fn validate(&self) -> Result<()> {
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Component {
    PersonalAiPlatform,
    NativeInferenceEngine,
    ModelFoundry,
    MemoryVault,
    ActionBroker,
    AgentGreen,
    NodeRuntime,
    CreatorOs,
    Tradeforge,
}

/// A bounded, namespaced extension map. Values are deliberately constrained to
/// scalars or one-dimensional scalar arrays to keep nesting bounded.
#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(transparent)]
pub struct Extensions(pub BTreeMap<String, Value>);

impl Validate for Extensions {
    fn validate(&self) -> Result<()> {
        check_max("extensions", self.0.len(), 16)?;
        for (name, value) in &self.0 {
            validate_extension_name(name)?;
            validate_extension_value(value)?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(transparent)]
pub struct RequiredExtensions(pub Vec<String>);

impl Validate for RequiredExtensions {
    fn validate(&self) -> Result<()> {
        // These v1 bindings publish no recognized required extensions.
        if let Some(name) = self.0.first() {
            Err(ContractError::UnsupportedRequiredExtension(name.clone()))
        } else {
            Ok(())
        }
    }
}

/// Validates the bounded extension map and fails closed on required extensions.
///
/// # Errors
///
/// Returns an error for malformed extensions, unsupported required extensions,
/// or a required-extension list without a corresponding extension map.
pub fn validate_extensions(
    extensions: Option<&Extensions>,
    required: Option<&RequiredExtensions>,
) -> Result<()> {
    if let Some(extensions) = extensions {
        extensions.validate()?;
    }
    if let Some(required) = required {
        if extensions.is_none() {
            return Err(ContractError::field(
                "required_extensions",
                "extensions must also be present",
            ));
        }
        required.validate()?;
    }
    Ok(())
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RedactedReference {
    pub reference_id: String,
    pub digest: String,
    pub redaction_state: RedactionState,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RedactionState {
    ReferenceOnly,
}

impl Validate for RedactedReference {
    fn validate(&self) -> Result<()> {
        check_len("reference_id", &self.reference_id, 1, 128)?;
        validate_reference_id(&self.reference_id)?;
        validate_sha256("digest", &self.digest)
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ScreenedText {
    pub text: String,
    pub secret_status: SecretStatus,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SecretStatus {
    ScreenedNoSecrets,
    Redacted,
}

impl Validate for ScreenedText {
    fn validate(&self) -> Result<()> {
        check_len("text", &self.text, 1, 32_768)
    }
}

pub(crate) fn check_len(
    field: &'static str,
    value: &str,
    minimum: usize,
    maximum: usize,
) -> Result<()> {
    let length = value.chars().count();
    if (minimum..=maximum).contains(&length) {
        Ok(())
    } else {
        Err(ContractError::field(
            field,
            format!("length must be {minimum} through {maximum}; got {length}"),
        ))
    }
}

pub(crate) fn check_max(field: &'static str, length: usize, maximum: usize) -> Result<()> {
    if length <= maximum {
        Ok(())
    } else {
        Err(ContractError::field(
            field,
            format!("must contain at most {maximum} items; got {length}"),
        ))
    }
}

pub(crate) fn check_range<T>(field: &'static str, value: T, minimum: T, maximum: T) -> Result<()>
where
    T: Copy + PartialOrd + std::fmt::Display,
{
    if value >= minimum && value <= maximum {
        Ok(())
    } else {
        Err(ContractError::field(
            field,
            format!("must be between {minimum} and {maximum}; got {value}"),
        ))
    }
}

pub(crate) fn validate_timestamp(field: &'static str, value: &str) -> Result<()> {
    check_len(field, value, 20, 35)?;
    parse_timestamp(value)
        .map(|_| ())
        .ok_or_else(|| ContractError::field(field, "must be a valid RFC 3339 date-time"))
}

// The minimal runtime-v3 profile uses bounded windows directly; the full
// profile also consumes this two-document ordering helper.
#[cfg_attr(not(feature = "full"), allow(dead_code))]
pub(crate) fn validate_timestamp_order(
    earlier_field: &'static str,
    earlier: &str,
    later_field: &'static str,
    later: &str,
) -> Result<()> {
    let earlier = parse_timestamp(earlier)
        .ok_or_else(|| ContractError::field(earlier_field, "must be a valid RFC 3339 date-time"))?;
    let later = parse_timestamp(later)
        .ok_or_else(|| ContractError::field(later_field, "must be a valid RFC 3339 date-time"))?;
    if compare_timestamps(&later, &earlier) == Ordering::Greater {
        Ok(())
    } else {
        Err(ContractError::field(
            later_field,
            format!("must be later than {earlier_field}"),
        ))
    }
}

pub(crate) fn validate_timestamp_window(
    earlier_field: &'static str,
    earlier: &str,
    later_field: &'static str,
    later: &str,
    maximum_seconds: i64,
) -> Result<()> {
    let earlier_value = parse_timestamp(earlier)
        .ok_or_else(|| ContractError::field(earlier_field, "must be a valid RFC 3339 date-time"))?;
    let later_value = parse_timestamp(later)
        .ok_or_else(|| ContractError::field(later_field, "must be a valid RFC 3339 date-time"))?;
    if compare_timestamps(&later_value, &earlier_value) != Ordering::Greater {
        return Err(ContractError::field(
            later_field,
            format!("must be later than {earlier_field}"),
        ));
    }
    let elapsed_seconds = later_value
        .epoch_seconds
        .checked_sub(earlier_value.epoch_seconds)
        .ok_or_else(|| ContractError::field(later_field, "timestamp interval overflowed"))?;
    if elapsed_seconds > maximum_seconds {
        return Err(ContractError::field(
            later_field,
            format!("must be no more than {maximum_seconds} seconds after {earlier_field}"),
        ));
    }
    Ok(())
}

#[derive(Debug)]
struct ParsedTimestamp {
    epoch_seconds: i64,
    fraction: Vec<u8>,
}

fn parse_timestamp(value: &str) -> Option<ParsedTimestamp> {
    let bytes = value.as_bytes();
    if !(20..=35).contains(&bytes.len())
        || bytes.get(4) != Some(&b'-')
        || bytes.get(7) != Some(&b'-')
        || !matches!(bytes.get(10), Some(b'T' | b't'))
        || bytes.get(13) != Some(&b':')
        || bytes.get(16) != Some(&b':')
    {
        return None;
    }
    let year = fixed_digits(bytes, 0, 4)?;
    let month = fixed_digits(bytes, 5, 2)?;
    let day = fixed_digits(bytes, 8, 2)?;
    let hour = fixed_digits(bytes, 11, 2)?;
    let minute = fixed_digits(bytes, 14, 2)?;
    let second = fixed_digits(bytes, 17, 2)?;
    if year == 0
        || !(1..=12).contains(&month)
        || !(1..=days_in_month(year, month)).contains(&day)
        || hour > 23
        || minute > 59
        || second > 59
    {
        return None;
    }

    let mut cursor = 19;
    let mut fraction = Vec::new();
    if bytes.get(cursor) == Some(&b'.') {
        cursor += 1;
        let fraction_start = cursor;
        while bytes.get(cursor).is_some_and(u8::is_ascii_digit) {
            fraction.push(bytes[cursor]);
            cursor += 1;
        }
        if cursor == fraction_start {
            return None;
        }
    }

    let offset_seconds = match bytes.get(cursor) {
        Some(b'Z' | b'z') if cursor + 1 == bytes.len() => 0_i64,
        Some(sign @ (b'+' | b'-')) if cursor + 6 == bytes.len() => {
            if bytes.get(cursor + 3) != Some(&b':') {
                return None;
            }
            let offset_hour = fixed_digits(bytes, cursor + 1, 2)?;
            let offset_minute = fixed_digits(bytes, cursor + 4, 2)?;
            if offset_hour > 23 || offset_minute > 59 {
                return None;
            }
            let magnitude = i64::from(offset_hour * 3_600 + offset_minute * 60);
            if *sign == b'+' {
                magnitude
            } else {
                -magnitude
            }
        }
        _ => return None,
    };

    let days =
        days_before_year(year) + i64::from(days_before_month(year, month)) + i64::from(day - 1);
    let local_seconds =
        days * 86_400 + i64::from(hour) * 3_600 + i64::from(minute) * 60 + i64::from(second);
    Some(ParsedTimestamp {
        epoch_seconds: local_seconds - offset_seconds,
        fraction,
    })
}

pub(crate) fn timestamp_unix_millis(field: &'static str, value: &str) -> Result<u64> {
    let parsed = parse_timestamp(value)
        .ok_or_else(|| ContractError::field(field, "must be a valid RFC 3339 date-time"))?;
    let unix_epoch_seconds = days_before_year(1970) * 86_400;
    let seconds = parsed
        .epoch_seconds
        .checked_sub(unix_epoch_seconds)
        .filter(|seconds| *seconds >= 0)
        .ok_or_else(|| ContractError::field(field, "must not precede the Unix epoch"))?;
    let mut milliseconds = 0_u64;
    for (index, digit) in parsed.fraction.iter().take(3).enumerate() {
        let place = match index {
            0 => 100,
            1 => 10,
            _ => 1,
        };
        milliseconds += u64::from(*digit - b'0') * place;
    }
    seconds
        .try_into()
        .ok()
        .and_then(|seconds: u64| seconds.checked_mul(1_000))
        .and_then(|value| value.checked_add(milliseconds))
        .ok_or_else(|| ContractError::field(field, "Unix millisecond conversion overflowed"))
}

pub(crate) fn format_unix_millis(field: &'static str, value: u64) -> Result<String> {
    let seconds = value / 1_000;
    let milliseconds = value % 1_000;
    let mut days = seconds / 86_400;
    let day_seconds = seconds % 86_400;
    let mut year = 1970_u32;
    loop {
        let year_days = if is_leap_year(year) { 366 } else { 365 };
        if days < year_days {
            break;
        }
        days -= year_days;
        year = year
            .checked_add(1)
            .filter(|year| *year <= 9_999)
            .ok_or_else(|| {
                ContractError::field(field, "Unix millisecond value exceeds year 9999")
            })?;
    }
    let mut month = 1_u32;
    loop {
        let month_days = u64::from(days_in_month(year, month));
        if days < month_days {
            break;
        }
        days -= month_days;
        month += 1;
    }
    let day = days + 1;
    let hour = day_seconds / 3_600;
    let minute = (day_seconds % 3_600) / 60;
    let second = day_seconds % 60;
    Ok(format!(
        "{year:04}-{month:02}-{day:02}T{hour:02}:{minute:02}:{second:02}.{milliseconds:03}Z"
    ))
}

fn fixed_digits(bytes: &[u8], start: usize, length: usize) -> Option<u32> {
    let digits = bytes.get(start..start + length)?;
    if !digits.iter().all(u8::is_ascii_digit) {
        return None;
    }
    Some(
        digits
            .iter()
            .fold(0_u32, |value, digit| value * 10 + u32::from(*digit - b'0')),
    )
}

const fn is_leap_year(year: u32) -> bool {
    year % 4 == 0 && (year % 100 != 0 || year % 400 == 0)
}

const fn days_in_month(year: u32, month: u32) -> u32 {
    match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 if is_leap_year(year) => 29,
        2 => 28,
        _ => 0,
    }
}

const fn days_before_year(year: u32) -> i64 {
    let previous = year - 1;
    (previous * 365 + previous / 4 - previous / 100 + previous / 400) as i64
}

const fn days_before_month(year: u32, month: u32) -> u32 {
    let ordinary = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334];
    let before = ordinary[(month - 1) as usize];
    if month > 2 && is_leap_year(year) {
        before + 1
    } else {
        before
    }
}

fn compare_timestamps(left: &ParsedTimestamp, right: &ParsedTimestamp) -> Ordering {
    left.epoch_seconds
        .cmp(&right.epoch_seconds)
        .then_with(|| compare_fractions(&left.fraction, &right.fraction))
}

pub(crate) fn compare_timestamp_values(left: &str, right: &str) -> Result<Ordering> {
    let left = parse_timestamp(left)
        .ok_or_else(|| ContractError::field("timestamp", "must be a valid RFC 3339 date-time"))?;
    let right = parse_timestamp(right)
        .ok_or_else(|| ContractError::field("timestamp", "must be a valid RFC 3339 date-time"))?;
    Ok(compare_timestamps(&left, &right))
}

fn compare_fractions(left: &[u8], right: &[u8]) -> Ordering {
    (0..left.len().max(right.len()))
        .map(|index| {
            left.get(index)
                .copied()
                .unwrap_or(b'0')
                .cmp(&right.get(index).copied().unwrap_or(b'0'))
        })
        .find(|ordering| *ordering != Ordering::Equal)
        .unwrap_or(Ordering::Equal)
}

pub(crate) fn validate_reason_code(field: &'static str, value: &str) -> Result<()> {
    check_len(field, value, 1, 96)?;
    let mut bytes = value.bytes();
    if !bytes.next().is_some_and(|byte| byte.is_ascii_lowercase())
        || !bytes.all(|byte| {
            byte.is_ascii_lowercase() || byte.is_ascii_digit() || matches!(byte, b'.' | b'_' | b'-')
        })
    {
        return Err(ContractError::field(field, "invalid reason-code syntax"));
    }
    if value.ends_with(['.', '_', '-'])
        || value
            .as_bytes()
            .windows(2)
            .any(|pair| matches!(pair, [b'.' | b'_' | b'-', b'.' | b'_' | b'-']))
    {
        return Err(ContractError::field(
            field,
            "invalid reason-code separators",
        ));
    }
    Ok(())
}

pub(crate) fn validate_sha256(field: &'static str, value: &str) -> Result<()> {
    if value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || matches!(byte, b'a'..=b'f'))
    {
        Ok(())
    } else {
        Err(ContractError::field(
            field,
            "must be a lowercase hexadecimal SHA-256 digest",
        ))
    }
}

pub(crate) fn validate_semver(field: &'static str, value: &str) -> Result<()> {
    check_len(field, value, 5, 64)?;
    let (without_build, build) = value
        .split_once('+')
        .map_or((value, None), |(prefix, suffix)| (prefix, Some(suffix)));
    if build.is_some_and(|suffix| suffix.is_empty() || suffix.contains('+'))
        || without_build.contains('+')
    {
        return Err(ContractError::field(field, "invalid semantic version"));
    }
    let (core, prerelease) = without_build
        .split_once('-')
        .map_or((without_build, None), |(prefix, suffix)| {
            (prefix, Some(suffix))
        });
    if prerelease.is_some_and(str::is_empty) {
        return Err(ContractError::field(field, "invalid semantic version"));
    }
    let components: Vec<&str> = core.split('.').collect();
    if components.len() != 3
        || components.iter().any(|part| {
            part.is_empty()
                || !part.bytes().all(|byte| byte.is_ascii_digit())
                || (part.len() > 1 && part.starts_with('0'))
        })
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'-' | b'+'))
    {
        Err(ContractError::field(field, "invalid semantic version"))
    } else {
        Ok(())
    }
}

fn validate_extension_name(value: &str) -> Result<()> {
    check_len("extension_name", value, 3, 96)?;
    let segments: Vec<&str> = value.split('.').collect();
    if segments.len() < 2
        || segments.iter().any(|segment| {
            let mut bytes = segment.bytes();
            !bytes.next().is_some_and(|byte| byte.is_ascii_lowercase())
                || !bytes.all(|byte| {
                    byte.is_ascii_lowercase()
                        || byte.is_ascii_digit()
                        || matches!(byte, b'_' | b'-')
                })
        })
    {
        Err(ContractError::field(
            "extension_name",
            "must be a lowercase namespaced name",
        ))
    } else {
        Ok(())
    }
}

fn validate_extension_value(value: &Value) -> Result<()> {
    let validate_scalar = |value: &Value| match value {
        Value::Null | Value::Bool(_) => Ok(()),
        Value::String(text) => check_len("extension_value", text, 0, 1_024),
        Value::Number(number) => {
            let Some(number) = number.as_f64() else {
                return Err(ContractError::field("extension_value", "invalid number"));
            };
            if number.is_finite() && (-1_000_000_000_000.0..=1_000_000_000_000.0).contains(&number)
            {
                Ok(())
            } else {
                Err(ContractError::field(
                    "extension_value",
                    "number out of bounds",
                ))
            }
        }
        _ => Err(ContractError::field("extension_value", "must be a scalar")),
    };
    if let Value::Array(values) = value {
        check_max("extension_value", values.len(), 16)?;
        values.iter().try_for_each(validate_scalar)
    } else {
        validate_scalar(value)
    }
}

fn validate_reference_id(value: &str) -> Result<()> {
    let Some(separator_index) = value
        .bytes()
        .position(|byte| matches!(byte, b'.' | b'_' | b':' | b'/' | b'-'))
    else {
        return Err(ContractError::field(
            "reference_id",
            "invalid reference syntax",
        ));
    };
    let (initial, remainder) = value.split_at(separator_index);
    let mut initial_bytes = initial.bytes();
    if !initial_bytes
        .next()
        .is_some_and(|byte| byte.is_ascii_lowercase())
        || !initial_bytes.all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit())
        || remainder.len() < 2
        || !remainder.bytes().all(|byte| {
            byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b':' | b'/' | b'-')
        })
    {
        Err(ContractError::field(
            "reference_id",
            "invalid reference syntax",
        ))
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{
        validate_timestamp, validate_timestamp_order, ContractVersion, Extensions,
        RequiredExtensions,
    };
    use crate::Validate;

    #[test]
    fn exact_contract_version_is_enforced_during_decode() {
        assert!(serde_json::from_str::<ContractVersion>(r#""1.0.0""#).is_ok());
        assert!(serde_json::from_str::<ContractVersion>(r#""1.0.1""#).is_err());
    }

    #[test]
    fn nested_extensions_are_rejected() {
        let extensions: Extensions =
            serde_json::from_str(r#"{"vendor.feature":{"nested":true}}"#).expect("decode");
        assert!(extensions.validate().is_err());
    }

    #[test]
    fn unknown_required_extensions_fail_closed() {
        let extensions = RequiredExtensions(vec!["vendor.feature".to_owned()]);
        assert!(extensions.validate().is_err());
    }

    #[test]
    fn rejects_impossible_calendar_dates() {
        assert!(validate_timestamp("time", "2025-02-29T00:00:00Z").is_err());
        assert!(validate_timestamp("time", "2024-02-29T00:00:00Z").is_ok());
        assert!(validate_timestamp("time", "2026-04-31T00:00:00Z").is_err());
        assert!(validate_timestamp("time", "2026-01-01T24:00:00Z").is_err());
    }

    #[test]
    fn timestamp_order_accounts_for_offsets_and_fractions() {
        assert!(validate_timestamp_order(
            "issued_at",
            "2026-08-12T09:00:00.1+01:00",
            "expires_at",
            "2026-08-12T08:00:00.2Z",
        )
        .is_ok());
        assert!(validate_timestamp_order(
            "issued_at",
            "2026-08-12T09:00:00Z",
            "expires_at",
            "2026-08-12T09:00:00.000Z",
        )
        .is_err());
    }

    #[test]
    fn rejects_uppercase_reference_prefix_and_empty_semver_suffixes() {
        let reference: super::RedactedReference = serde_json::from_str(
            r#"{"reference_id":"Auth:one","digest":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","redaction_state":"reference_only"}"#,
        )
        .expect("decode");
        assert!(reference.validate().is_err());
        assert!(super::validate_semver("version", "1.0.0-").is_err());
        assert!(super::validate_semver("version", "1.0.0+").is_err());
        assert!(super::validate_semver("version", "1.0.0+build+other").is_err());
        assert!(super::validate_semver("version", "1.0.0-alpha+build").is_ok());
    }
}
