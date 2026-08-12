//! Closed message-specific payloads for authenticated service envelopes.
//!
//! These codecs are intentionally fixed-width and non-recursive. They preserve
//! the PAISVC01/PAISVR01 wire representation while preventing callers from
//! attaching arbitrary bytes to a recognized service message kind.

use crate::ServiceMessageKind;
use personalai_engine_types::{
    EngineCapability, EngineError, MAX_REQUEST_BYTES, MAX_TOKEN_COUNT, ModelId, TokenId,
};

/// Maximum registered model identifiers carried in one response.
pub const MAX_SERVICE_REGISTERED_MODELS: usize = 1_024;
/// Number of fixed `u64` values in a service metrics response.
pub const SERVICE_METRIC_COUNT: usize = 17;

const KNOWN_CAPABILITY_BITS: u64 = 0b11_1111;

/// Closed request payload catalog.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ServiceRequestBody {
    /// Begin protocol negotiation with an exact major and minor version.
    Hello {
        /// Requested protocol major version.
        major: u16,
        /// Requested protocol minor version.
        minor: u16,
    },
    /// Request the implemented capability set.
    ListCapabilities,
    /// Request statically registered model identifiers.
    ListRegisteredModels,
    /// Load one statically registered model.
    LoadModel(ModelId),
    /// Unload one idle model.
    UnloadModel(ModelId),
    /// Tokenize bounded UTF-8 prompt text with one loaded model.
    Tokenize {
        /// Statically registered, loaded model identifier.
        model_id: ModelId,
        /// Bounded UTF-8 prompt text assembled by the Go control plane.
        text: String,
    },
    /// Request service health.
    Health,
    /// Request content-free metrics.
    Metrics,
    /// Request shutdown through the separately authorized administrative path.
    ShutdownRequest,
}

impl ServiceRequestBody {
    /// Returns the only message kind valid for this body.
    pub const fn kind(&self) -> ServiceMessageKind {
        match self {
            Self::Hello { .. } => ServiceMessageKind::Hello,
            Self::ListCapabilities => ServiceMessageKind::ListCapabilities,
            Self::ListRegisteredModels => ServiceMessageKind::ListRegisteredModels,
            Self::LoadModel(_) => ServiceMessageKind::LoadModel,
            Self::UnloadModel(_) => ServiceMessageKind::UnloadModel,
            Self::Tokenize { .. } => ServiceMessageKind::Tokenize,
            Self::Health => ServiceMessageKind::Health,
            Self::Metrics => ServiceMessageKind::Metrics,
            Self::ShutdownRequest => ServiceMessageKind::ShutdownRequest,
        }
    }

    pub(crate) fn encode(&self) -> Result<Vec<u8>, EngineError> {
        match self {
            Self::Hello { major, minor } => Ok([major.to_le_bytes(), minor.to_le_bytes()].concat()),
            Self::ListCapabilities
            | Self::ListRegisteredModels
            | Self::Health
            | Self::Metrics
            | Self::ShutdownRequest => Ok(Vec::new()),
            Self::LoadModel(model_id) | Self::UnloadModel(model_id) => {
                Ok(model_id.as_str().as_bytes().to_vec())
            }
            Self::Tokenize { model_id, text } => encode_tokenize(model_id, text),
        }
    }

    pub(crate) fn decode(kind: ServiceMessageKind, payload: &[u8]) -> Result<Self, EngineError> {
        match kind {
            ServiceMessageKind::Hello => {
                require_length(payload, 4)?;
                Ok(Self::Hello { major: read_u16(payload, 0)?, minor: read_u16(payload, 2)? })
            }
            ServiceMessageKind::ListCapabilities => {
                require_empty(payload)?;
                Ok(Self::ListCapabilities)
            }
            ServiceMessageKind::ListRegisteredModels => {
                require_empty(payload)?;
                Ok(Self::ListRegisteredModels)
            }
            ServiceMessageKind::LoadModel => Ok(Self::LoadModel(decode_model_id(payload)?)),
            ServiceMessageKind::UnloadModel => Ok(Self::UnloadModel(decode_model_id(payload)?)),
            ServiceMessageKind::Tokenize => decode_tokenize(payload),
            ServiceMessageKind::Health => {
                require_empty(payload)?;
                Ok(Self::Health)
            }
            ServiceMessageKind::Metrics => {
                require_empty(payload)?;
                Ok(Self::Metrics)
            }
            ServiceMessageKind::ShutdownRequest => {
                require_empty(payload)?;
                Ok(Self::ShutdownRequest)
            }
            _ => Err(EngineError::protocol_violation("message type has no service request body")),
        }
    }
}

/// Closed, forward-rejecting engine capability bit set.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ServiceCapabilitySet(u64);

impl ServiceCapabilitySet {
    /// Constructs a bit set from reviewed capability values.
    pub fn from_capabilities(
        capabilities: impl IntoIterator<Item = EngineCapability>,
    ) -> Result<Self, EngineError> {
        let mut bits = 0_u64;
        for capability in capabilities {
            bits |= 1_u64 << capability_bit(capability)?;
        }
        Ok(Self(bits))
    }

    /// Returns the exact wire bits.
    pub const fn bits(self) -> u64 {
        self.0
    }

    /// Tests whether one reviewed capability is present.
    pub fn contains(self, capability: EngineCapability) -> Result<bool, EngineError> {
        Ok(self.0 & (1_u64 << capability_bit(capability)?) != 0)
    }

    fn decode(bits: u64) -> Result<Self, EngineError> {
        if bits & !KNOWN_CAPABILITY_BITS != 0 {
            return Err(EngineError::protocol_violation(
                "capability response contains an unknown required bit",
            ));
        }
        Ok(Self(bits))
    }
}

/// Fixed service health state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum ServiceHealthStatus {
    /// Service accepts work and loaded models are healthy.
    Ready = 1,
    /// Service runs but is degraded, including having no loaded model.
    Degraded = 2,
    /// Administrative draining is active.
    Draining = 3,
}

impl ServiceHealthStatus {
    fn decode(value: u8) -> Result<Self, EngineError> {
        match value {
            1 => Ok(Self::Ready),
            2 => Ok(Self::Degraded),
            3 => Ok(Self::Draining),
            _ => Err(EngineError::protocol_violation("health state is unsupported")),
        }
    }
}

/// Stable, content-free service error classification.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum ServiceErrorCode {
    /// Invalid input.
    InvalidInput = 1,
    /// Unsupported architecture.
    UnsupportedArchitecture = 2,
    /// Unsupported tokenizer.
    UnsupportedTokenizer = 3,
    /// Corrupt model package.
    CorruptModelPackage = 4,
    /// Signature verification failure.
    SignatureFailure = 5,
    /// Resource exhaustion.
    ResourceExhaustion = 6,
    /// Cancellation.
    Cancellation = 7,
    /// Backend failure.
    BackendFailure = 8,
    /// Internal invariant violation.
    InternalInvariantViolation = 9,
    /// Authentication failure.
    AuthenticationFailure = 10,
    /// Protocol violation.
    ProtocolViolation = 11,
    /// Feature not implemented.
    NotImplemented = 12,
    /// A future internal classification not exposed to the client.
    Unclassified = 255,
}

impl ServiceErrorCode {
    fn decode(value: u8) -> Result<Self, EngineError> {
        match value {
            1 => Ok(Self::InvalidInput),
            2 => Ok(Self::UnsupportedArchitecture),
            3 => Ok(Self::UnsupportedTokenizer),
            4 => Ok(Self::CorruptModelPackage),
            5 => Ok(Self::SignatureFailure),
            6 => Ok(Self::ResourceExhaustion),
            7 => Ok(Self::Cancellation),
            8 => Ok(Self::BackendFailure),
            9 => Ok(Self::InternalInvariantViolation),
            10 => Ok(Self::AuthenticationFailure),
            11 => Ok(Self::ProtocolViolation),
            12 => Ok(Self::NotImplemented),
            255 => Ok(Self::Unclassified),
            _ => Err(EngineError::protocol_violation("service error code is unsupported")),
        }
    }
}

/// Fixed reason a data-plane shutdown request was rejected.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum ShutdownRejectionReason {
    /// The authenticated data-plane channel has no shutdown authority.
    AdministrativeAuthorizationRequired = 1,
}

/// Fixed-order metrics values. The order is part of PAISVR01.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ServiceMetrics([u64; SERVICE_METRIC_COUNT]);

impl ServiceMetrics {
    /// Constructs metrics in the documented PAISVR01 field order.
    pub const fn new(values: [u64; SERVICE_METRIC_COUNT]) -> Self {
        Self(values)
    }

    /// Returns values in wire order: loaded, active, queued, rejected,
    /// cancelled, prompt tokens, generated tokens, prefill ns, decode ns,
    /// tokens/sec milli, host bytes, device bytes, KV bytes, package failures,
    /// authentication failures, protocol violations, backend failures.
    pub const fn values(&self) -> &[u64; SERVICE_METRIC_COUNT] {
        &self.0
    }
}

/// Closed response payload catalog.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ServiceResponseBody {
    /// Version negotiation succeeded.
    HelloAccepted {
        /// Accepted protocol major version.
        major: u16,
        /// Accepted protocol minor version.
        minor: u16,
    },
    /// Version negotiation failed and reports the supported version.
    HelloRejected {
        /// Supported protocol major version.
        major: u16,
        /// Supported protocol minor version.
        minor: u16,
    },
    /// Reviewed implemented capabilities.
    Capabilities(ServiceCapabilitySet),
    /// Statically registered model identifiers.
    RegisteredModels(Vec<ModelId>),
    /// One requested model is loaded.
    ModelLoaded,
    /// One requested model is unloaded.
    ModelUnloaded,
    /// Tokenizer output.
    Tokenized(Vec<TokenId>),
    /// Current service health.
    Health(ServiceHealthStatus),
    /// Content-free runtime metrics.
    Metrics(ServiceMetrics),
    /// Stable failure category without sensitive diagnostics.
    Error(ServiceErrorCode),
    /// Shutdown lacks separate administrative authorization.
    ShutdownRejected(ShutdownRejectionReason),
    /// Shutdown was accepted through a future separate administrative path.
    ShutdownAccepted,
}

impl ServiceResponseBody {
    /// Returns the only message kind valid for this body.
    pub const fn kind(&self) -> ServiceMessageKind {
        match self {
            Self::HelloAccepted { .. } => ServiceMessageKind::HelloAccepted,
            Self::HelloRejected { .. } => ServiceMessageKind::HelloRejected,
            Self::Capabilities(_) => ServiceMessageKind::ListCapabilities,
            Self::RegisteredModels(_) => ServiceMessageKind::ListRegisteredModels,
            Self::ModelLoaded => ServiceMessageKind::LoadModel,
            Self::ModelUnloaded => ServiceMessageKind::UnloadModel,
            Self::Tokenized(_) => ServiceMessageKind::Tokenize,
            Self::Health(_) => ServiceMessageKind::Health,
            Self::Metrics(_) => ServiceMessageKind::Metrics,
            Self::Error(_) => ServiceMessageKind::Error,
            Self::ShutdownRejected(_) => ServiceMessageKind::ShutdownRejected,
            Self::ShutdownAccepted => ServiceMessageKind::ShutdownAccepted,
        }
    }

    pub(crate) fn encode(&self) -> Result<Vec<u8>, EngineError> {
        match self {
            Self::HelloAccepted { major, minor } | Self::HelloRejected { major, minor } => {
                Ok([major.to_le_bytes(), minor.to_le_bytes()].concat())
            }
            Self::Capabilities(capabilities) => Ok(capabilities.bits().to_le_bytes().to_vec()),
            Self::RegisteredModels(models) => encode_registered_models(models),
            Self::ModelLoaded | Self::ModelUnloaded | Self::ShutdownAccepted => Ok(Vec::new()),
            Self::Tokenized(tokens) => encode_tokens(tokens),
            Self::Health(state) => Ok(vec![*state as u8]),
            Self::Metrics(metrics) => {
                let mut encoded = Vec::with_capacity(SERVICE_METRIC_COUNT * 8);
                for value in metrics.values() {
                    encoded.extend_from_slice(&value.to_le_bytes());
                }
                Ok(encoded)
            }
            Self::Error(code) => Ok(vec![*code as u8]),
            Self::ShutdownRejected(reason) => Ok(vec![*reason as u8]),
        }
    }

    pub(crate) fn decode(kind: ServiceMessageKind, payload: &[u8]) -> Result<Self, EngineError> {
        match kind {
            ServiceMessageKind::HelloAccepted | ServiceMessageKind::HelloRejected => {
                require_length(payload, 4)?;
                let major = read_u16(payload, 0)?;
                let minor = read_u16(payload, 2)?;
                if kind == ServiceMessageKind::HelloAccepted {
                    Ok(Self::HelloAccepted { major, minor })
                } else {
                    Ok(Self::HelloRejected { major, minor })
                }
            }
            ServiceMessageKind::ListCapabilities => {
                require_length(payload, 8)?;
                Ok(Self::Capabilities(ServiceCapabilitySet::decode(read_u64(payload, 0)?)?))
            }
            ServiceMessageKind::ListRegisteredModels => {
                Ok(Self::RegisteredModels(decode_registered_models(payload)?))
            }
            ServiceMessageKind::LoadModel => {
                require_empty(payload)?;
                Ok(Self::ModelLoaded)
            }
            ServiceMessageKind::UnloadModel => {
                require_empty(payload)?;
                Ok(Self::ModelUnloaded)
            }
            ServiceMessageKind::Tokenize => Ok(Self::Tokenized(decode_tokens(payload)?)),
            ServiceMessageKind::Health => {
                require_length(payload, 1)?;
                Ok(Self::Health(ServiceHealthStatus::decode(payload[0])?))
            }
            ServiceMessageKind::Metrics => {
                require_length(payload, SERVICE_METRIC_COUNT * 8)?;
                let mut values = [0_u64; SERVICE_METRIC_COUNT];
                for (index, value) in values.iter_mut().enumerate() {
                    *value = read_u64(payload, index * 8)?;
                }
                Ok(Self::Metrics(ServiceMetrics::new(values)))
            }
            ServiceMessageKind::Error => {
                require_length(payload, 1)?;
                Ok(Self::Error(ServiceErrorCode::decode(payload[0])?))
            }
            ServiceMessageKind::ShutdownRejected => {
                require_length(payload, 1)?;
                if payload[0] != ShutdownRejectionReason::AdministrativeAuthorizationRequired as u8
                {
                    return Err(EngineError::protocol_violation(
                        "shutdown rejection reason is unsupported",
                    ));
                }
                Ok(Self::ShutdownRejected(
                    ShutdownRejectionReason::AdministrativeAuthorizationRequired,
                ))
            }
            ServiceMessageKind::ShutdownAccepted => {
                require_empty(payload)?;
                Ok(Self::ShutdownAccepted)
            }
            _ => Err(EngineError::protocol_violation("message type has no service response body")),
        }
    }
}

fn capability_bit(capability: EngineCapability) -> Result<u32, EngineError> {
    match capability {
        EngineCapability::CpuReference => Ok(0),
        EngineCapability::Arm64 => Ok(1),
        EngineCapability::Metal => Ok(2),
        EngineCapability::Cuda => Ok(3),
        EngineCapability::StreamingGeneration => Ok(4),
        EngineCapability::UnixDomainSocket => Ok(5),
        _ => Err(EngineError::not_implemented("unknown engine capability")),
    }
}

fn encode_tokenize(model_id: &ModelId, text: &str) -> Result<Vec<u8>, EngineError> {
    if text.as_bytes().contains(&0) {
        return Err(EngineError::invalid_input("tokenize text contains NUL"));
    }
    let model = model_id.as_str().as_bytes();
    let model_length = u16::try_from(model.len())
        .map_err(|_| EngineError::resource_exhaustion("model ID exceeds u16"))?;
    let total = 2_usize
        .checked_add(model.len())
        .and_then(|length| length.checked_add(text.len()))
        .filter(|length| *length <= MAX_REQUEST_BYTES)
        .ok_or_else(|| EngineError::resource_exhaustion("tokenize payload is too large"))?;
    let mut encoded = Vec::new();
    encoded
        .try_reserve_exact(total)
        .map_err(|_| EngineError::resource_exhaustion("tokenize payload allocation failed"))?;
    encoded.extend_from_slice(&model_length.to_le_bytes());
    encoded.extend_from_slice(model);
    encoded.extend_from_slice(text.as_bytes());
    Ok(encoded)
}

fn decode_tokenize(payload: &[u8]) -> Result<ServiceRequestBody, EngineError> {
    if payload.len() < 2 {
        return Err(EngineError::protocol_violation("tokenize payload is truncated"));
    }
    let model_length = usize::from(read_u16(payload, 0)?);
    let model_end = 2_usize
        .checked_add(model_length)
        .filter(|end| *end <= payload.len())
        .ok_or_else(|| EngineError::protocol_violation("tokenize model range is invalid"))?;
    let model_id = decode_model_id(&payload[2..model_end])?;
    let text = std::str::from_utf8(&payload[model_end..])
        .map_err(|_| EngineError::protocol_violation("tokenize text is not UTF-8"))?;
    if text.as_bytes().contains(&0) {
        return Err(EngineError::protocol_violation("tokenize text contains NUL"));
    }
    Ok(ServiceRequestBody::Tokenize { model_id, text: text.to_owned() })
}

fn encode_registered_models(models: &[ModelId]) -> Result<Vec<u8>, EngineError> {
    if models.len() > MAX_SERVICE_REGISTERED_MODELS {
        return Err(EngineError::resource_exhaustion("registered model response is too large"));
    }
    let count = u16::try_from(models.len())
        .map_err(|_| EngineError::resource_exhaustion("registered model count exceeds u16"))?;
    let mut encoded = Vec::new();
    encoded.extend_from_slice(&count.to_le_bytes());
    for (index, model) in models.iter().enumerate() {
        if models[..index].contains(model) {
            return Err(EngineError::internal_invariant(
                "registered model response contains a duplicate ID",
            ));
        }
        let bytes = model.as_str().as_bytes();
        let length = u16::try_from(bytes.len())
            .map_err(|_| EngineError::resource_exhaustion("model ID exceeds u16"))?;
        encoded.extend_from_slice(&length.to_le_bytes());
        encoded.extend_from_slice(bytes);
    }
    Ok(encoded)
}

fn decode_registered_models(payload: &[u8]) -> Result<Vec<ModelId>, EngineError> {
    if payload.len() < 2 {
        return Err(EngineError::protocol_violation("registered model response is truncated"));
    }
    let count = usize::from(read_u16(payload, 0)?);
    if count > MAX_SERVICE_REGISTERED_MODELS {
        return Err(EngineError::protocol_violation(
            "registered model count exceeds its hard bound",
        ));
    }
    let mut models = Vec::new();
    models
        .try_reserve_exact(count)
        .map_err(|_| EngineError::resource_exhaustion("registered model allocation failed"))?;
    let mut cursor = 2_usize;
    for _index in 0..count {
        let length = usize::from(read_u16(payload, cursor)?);
        cursor = cursor
            .checked_add(2)
            .ok_or_else(|| EngineError::protocol_violation("model cursor overflows"))?;
        let end = cursor
            .checked_add(length)
            .filter(|end| *end <= payload.len())
            .ok_or_else(|| EngineError::protocol_violation("model range is invalid"))?;
        let model = decode_model_id(&payload[cursor..end])?;
        if models.contains(&model) {
            return Err(EngineError::protocol_violation(
                "registered model response contains a duplicate ID",
            ));
        }
        models.push(model);
        cursor = end;
    }
    if cursor != payload.len() {
        return Err(EngineError::protocol_violation(
            "registered model response contains trailing data",
        ));
    }
    Ok(models)
}

fn encode_tokens(tokens: &[TokenId]) -> Result<Vec<u8>, EngineError> {
    if tokens.len() > usize::try_from(MAX_TOKEN_COUNT).unwrap_or(usize::MAX) {
        return Err(EngineError::resource_exhaustion("token response exceeds its hard bound"));
    }
    let count = u32::try_from(tokens.len())
        .map_err(|_| EngineError::resource_exhaustion("token count exceeds u32"))?;
    let capacity =
        4_usize
            .checked_add(tokens.len().checked_mul(4).ok_or_else(|| {
                EngineError::resource_exhaustion("token response length overflows")
            })?)
            .filter(|length| *length <= MAX_REQUEST_BYTES)
            .ok_or_else(|| EngineError::resource_exhaustion("token response is too large"))?;
    let mut encoded = Vec::new();
    encoded
        .try_reserve_exact(capacity)
        .map_err(|_| EngineError::resource_exhaustion("token response allocation failed"))?;
    encoded.extend_from_slice(&count.to_le_bytes());
    for token in tokens {
        encoded.extend_from_slice(&token.get().to_le_bytes());
    }
    Ok(encoded)
}

fn decode_tokens(payload: &[u8]) -> Result<Vec<TokenId>, EngineError> {
    if payload.len() < 4 {
        return Err(EngineError::protocol_violation("token response is truncated"));
    }
    let count = usize::try_from(read_u32(payload, 0)?)
        .map_err(|_| EngineError::protocol_violation("token count is not addressable"))?;
    if count > usize::try_from(MAX_TOKEN_COUNT).unwrap_or(usize::MAX) {
        return Err(EngineError::protocol_violation("token count exceeds its hard bound"));
    }
    let expected =
        4_usize
            .checked_add(count.checked_mul(4).ok_or_else(|| {
                EngineError::protocol_violation("token response length overflows")
            })?)
            .ok_or_else(|| EngineError::protocol_violation("token response length overflows"))?;
    require_length(payload, expected)?;
    let mut tokens = Vec::new();
    tokens
        .try_reserve_exact(count)
        .map_err(|_| EngineError::resource_exhaustion("token response allocation failed"))?;
    for index in 0..count {
        let offset = 4_usize
            .checked_add(
                index
                    .checked_mul(4)
                    .ok_or_else(|| EngineError::protocol_violation("token offset overflows"))?,
            )
            .ok_or_else(|| EngineError::protocol_violation("token offset overflows"))?;
        tokens.push(
            TokenId::new(read_u32(payload, offset)?)
                .map_err(|_| EngineError::protocol_violation("token ID is invalid"))?,
        );
    }
    Ok(tokens)
}

fn decode_model_id(payload: &[u8]) -> Result<ModelId, EngineError> {
    let text = std::str::from_utf8(payload)
        .map_err(|_| EngineError::protocol_violation("model ID is not UTF-8"))?;
    ModelId::new(text).map_err(|_| EngineError::protocol_violation("model ID is invalid"))
}

fn require_empty(payload: &[u8]) -> Result<(), EngineError> {
    if payload.is_empty() {
        Ok(())
    } else {
        Err(EngineError::protocol_violation("service payload contains trailing fields"))
    }
}

fn require_length(payload: &[u8], expected: usize) -> Result<(), EngineError> {
    if payload.len() == expected {
        Ok(())
    } else {
        Err(EngineError::protocol_violation("service payload has an invalid fixed length"))
    }
}

fn read_u16(bytes: &[u8], offset: usize) -> Result<u16, EngineError> {
    Ok(u16::from_le_bytes(read_exact(bytes, offset)?))
}

fn read_u32(bytes: &[u8], offset: usize) -> Result<u32, EngineError> {
    Ok(u32::from_le_bytes(read_exact(bytes, offset)?))
}

fn read_u64(bytes: &[u8], offset: usize) -> Result<u64, EngineError> {
    Ok(u64::from_le_bytes(read_exact(bytes, offset)?))
}

fn read_exact<const N: usize>(bytes: &[u8], offset: usize) -> Result<[u8; N], EngineError> {
    let end = offset
        .checked_add(N)
        .ok_or_else(|| EngineError::protocol_violation("service body offset overflows"))?;
    bytes
        .get(offset..end)
        .and_then(|field| field.try_into().ok())
        .ok_or_else(|| EngineError::protocol_violation("service body is truncated"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model(value: &str) -> ModelId {
        ModelId::new(value).expect("valid model ID")
    }

    #[test]
    fn every_request_body_round_trips_with_its_derived_kind() {
        let bodies = vec![
            ServiceRequestBody::Hello { major: 2, minor: 0 },
            ServiceRequestBody::ListCapabilities,
            ServiceRequestBody::ListRegisteredModels,
            ServiceRequestBody::LoadModel(model("tiny-load")),
            ServiceRequestBody::UnloadModel(model("tiny-unload")),
            ServiceRequestBody::Tokenize {
                model_id: model("tiny-tokenizer"),
                text: "hello 🌍".to_owned(),
            },
            ServiceRequestBody::Health,
            ServiceRequestBody::Metrics,
            ServiceRequestBody::ShutdownRequest,
        ];
        for body in bodies {
            let encoded = body.encode().expect("request body must encode");
            let decoded = ServiceRequestBody::decode(body.kind(), &encoded)
                .expect("request body must decode");
            assert_eq!(decoded, body);
        }
    }

    #[test]
    fn every_response_body_round_trips_with_its_derived_kind() {
        let capabilities = ServiceCapabilitySet::from_capabilities([
            EngineCapability::CpuReference,
            EngineCapability::UnixDomainSocket,
        ])
        .expect("capabilities");
        let bodies = vec![
            ServiceResponseBody::HelloAccepted { major: 2, minor: 0 },
            ServiceResponseBody::HelloRejected { major: 2, minor: 0 },
            ServiceResponseBody::Capabilities(capabilities),
            ServiceResponseBody::RegisteredModels(vec![model("tiny-a"), model("tiny-b")]),
            ServiceResponseBody::ModelLoaded,
            ServiceResponseBody::ModelUnloaded,
            ServiceResponseBody::Tokenized(vec![
                TokenId::new(0).expect("token"),
                TokenId::new(42).expect("token"),
            ]),
            ServiceResponseBody::Health(ServiceHealthStatus::Degraded),
            ServiceResponseBody::Metrics(ServiceMetrics::new(core::array::from_fn(|index| {
                u64::try_from(index).expect("metric index")
            }))),
            ServiceResponseBody::Error(ServiceErrorCode::ProtocolViolation),
            ServiceResponseBody::ShutdownRejected(
                ShutdownRejectionReason::AdministrativeAuthorizationRequired,
            ),
            ServiceResponseBody::ShutdownAccepted,
        ];
        for body in bodies {
            let encoded = body.encode().expect("response body must encode");
            let decoded = ServiceResponseBody::decode(body.kind(), &encoded)
                .expect("response body must decode");
            assert_eq!(decoded, body);
        }
    }

    #[test]
    fn fixed_and_empty_payloads_reject_truncation_and_trailing_data() {
        assert!(ServiceRequestBody::decode(ServiceMessageKind::Hello, &[2, 0, 0]).is_err());
        assert!(ServiceRequestBody::decode(ServiceMessageKind::Health, &[0]).is_err());
        assert!(ServiceResponseBody::decode(ServiceMessageKind::Health, &[]).is_err());
        assert!(ServiceResponseBody::decode(ServiceMessageKind::Health, &[1, 0]).is_err());
        assert!(ServiceResponseBody::decode(ServiceMessageKind::ShutdownAccepted, &[0]).is_err());
    }

    #[test]
    fn variable_payloads_reject_ambiguity_and_invalid_values() {
        assert!(ServiceRequestBody::decode(ServiceMessageKind::Tokenize, &[5, 0, b'a']).is_err());
        assert!(
            ServiceRequestBody::decode(ServiceMessageKind::Tokenize, &[1, 0, b'm', 0],).is_err()
        );
        assert!(
            ServiceResponseBody::decode(
                ServiceMessageKind::ListCapabilities,
                &(1_u64 << 63).to_le_bytes(),
            )
            .is_err()
        );
        assert!(ServiceResponseBody::decode(ServiceMessageKind::Error, &[13]).is_err());
        assert!(ServiceResponseBody::decode(ServiceMessageKind::ShutdownRejected, &[2]).is_err());
        assert!(
            ServiceResponseBody::decode(
                ServiceMessageKind::Tokenize,
                &[1, 0, 0, 0, 255, 255, 255, 255],
            )
            .is_err()
        );
    }

    #[test]
    fn registered_models_reject_truncation_duplicates_and_trailing_data() {
        let valid = ServiceResponseBody::RegisteredModels(vec![model("tiny")])
            .encode()
            .expect("model response");
        assert!(
            ServiceResponseBody::decode(
                ServiceMessageKind::ListRegisteredModels,
                &valid[..valid.len() - 1],
            )
            .is_err()
        );
        let mut trailing = valid.clone();
        trailing.push(0);
        assert!(
            ServiceResponseBody::decode(ServiceMessageKind::ListRegisteredModels, &trailing)
                .is_err()
        );
        assert!(
            ServiceResponseBody::RegisteredModels(vec![model("tiny"), model("tiny")])
                .encode()
                .is_err()
        );
        let duplicate = [2, 0, 4, 0, b't', b'i', b'n', b'y', 4, 0, b't', b'i', b'n', b'y'];
        assert!(
            ServiceResponseBody::decode(ServiceMessageKind::ListRegisteredModels, &duplicate)
                .is_err()
        );
    }
}
