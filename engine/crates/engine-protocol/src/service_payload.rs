//! Closed message-specific payloads for authenticated service envelopes.
//!
//! These codecs are intentionally fixed-width and non-recursive. They preserve
//! the PAISVC01/PAISVR01 wire representation while preventing callers from
//! attaching arbitrary bytes to a recognized service message kind.

use crate::ServiceMessageKind;
use personalai_engine_types::{
    ArchitectureId, BackendId, ContextLimit, EngineCapability, EngineError, ForbiddenOperation,
    MAX_REQUEST_BYTES, MAX_TOKEN_COUNT, ModelHashAlgorithm, ModelId, ModelPackageHash,
    ModelPackageRejection, RequestId, ResourceKind, SecurityEvent, TokenId,
};

/// Maximum registered model identifiers carried in one response.
pub const MAX_SERVICE_REGISTERED_MODELS: usize = 1_024;
/// Number of fixed `u64` values in a service metrics response.
pub const SERVICE_METRIC_COUNT: usize = 17;
/// Exact bytes in an atomic model-status response.
pub const SERVICE_MODEL_STATUS_BYTES: usize = 44;
/// Maximum sanitized security events returned by one authenticated poll.
pub const MAX_SERVICE_SECURITY_EVENTS: usize = 128;

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
    /// Query one statically registered model's load and scheduler status.
    ModelStatus(ModelId),
    /// Poll a bounded batch of sanitized security events without removing it.
    SecurityEvents {
        /// Positive batch bound no greater than [`MAX_SERVICE_SECURITY_EVENTS`].
        maximum: u16,
    },
    /// Query path-free identity for one registered and loaded model.
    ModelDescriptor(ModelId),
    /// Remove an exact delivered prefix from the security event queue.
    AcknowledgeSecurityEvents {
        /// Positive prefix length no greater than [`MAX_SERVICE_SECURITY_EVENTS`].
        count: u16,
    },
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
            Self::ModelStatus(_) => ServiceMessageKind::ModelStatus,
            Self::SecurityEvents { .. } => ServiceMessageKind::SecurityEvents,
            Self::ModelDescriptor(_) => ServiceMessageKind::ModelDescriptor,
            Self::AcknowledgeSecurityEvents { .. } => ServiceMessageKind::AcknowledgeSecurityEvents,
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
            Self::LoadModel(model_id)
            | Self::UnloadModel(model_id)
            | Self::ModelStatus(model_id)
            | Self::ModelDescriptor(model_id) => Ok(model_id.as_str().as_bytes().to_vec()),
            Self::Tokenize { model_id, text } => encode_tokenize(model_id, text),
            Self::SecurityEvents { maximum } => {
                validate_security_event_maximum(*maximum)?;
                Ok(maximum.to_le_bytes().to_vec())
            }
            Self::AcknowledgeSecurityEvents { count } => {
                validate_security_event_maximum(*count)?;
                Ok(count.to_le_bytes().to_vec())
            }
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
            ServiceMessageKind::ModelStatus => Ok(Self::ModelStatus(decode_model_id(payload)?)),
            ServiceMessageKind::SecurityEvents => {
                require_length(payload, 2)?;
                let maximum = read_u16(payload, 0)?;
                validate_security_event_maximum(maximum)?;
                Ok(Self::SecurityEvents { maximum })
            }
            ServiceMessageKind::ModelDescriptor => {
                Ok(Self::ModelDescriptor(decode_model_id(payload)?))
            }
            ServiceMessageKind::AcknowledgeSecurityEvents => {
                require_length(payload, 2)?;
                let count = read_u16(payload, 0)?;
                validate_security_event_maximum(count)?;
                Ok(Self::AcknowledgeSecurityEvents { count })
            }
            _ => Err(EngineError::protocol_violation("message type has no service request body")),
        }
    }
}

/// Authenticated, path-free identity of one loaded model execution target.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ServiceModelDescriptor {
    package_hash: ModelPackageHash,
    architecture_id: ArchitectureId,
    backend_id: BackendId,
    context_limit: ContextLimit,
}

impl ServiceModelDescriptor {
    /// Creates a validated descriptor from engine-owned metadata.
    pub const fn new(
        package_hash: ModelPackageHash,
        architecture_id: ArchitectureId,
        backend_id: BackendId,
        context_limit: ContextLimit,
    ) -> Self {
        Self { package_hash, architecture_id, backend_id, context_limit }
    }

    /// Returns the verified configured package hash.
    pub const fn package_hash(&self) -> &ModelPackageHash {
        &self.package_hash
    }

    /// Returns the reviewed architecture identifier.
    pub const fn architecture_id(&self) -> &ArchitectureId {
        &self.architecture_id
    }

    /// Returns the executing backend identifier.
    pub const fn backend_id(&self) -> &BackendId {
        &self.backend_id
    }

    /// Returns the loaded model context limit.
    pub const fn context_limit(&self) -> ContextLimit {
        self.context_limit
    }

    fn encode(&self) -> Result<Vec<u8>, EngineError> {
        if self.package_hash.algorithm() != ModelHashAlgorithm::Sha384 {
            return Err(EngineError::internal_invariant(
                "service model hash algorithm is unsupported",
            ));
        }
        let architecture = self.architecture_id.as_str().as_bytes();
        let backend = self.backend_id.as_str().as_bytes();
        let architecture_length = u16::try_from(architecture.len())
            .map_err(|_| EngineError::internal_invariant("architecture identifier exceeds u16"))?;
        let backend_length = u16::try_from(backend.len())
            .map_err(|_| EngineError::internal_invariant("backend identifier exceeds u16"))?;
        let capacity = 1_usize
            .checked_add(ModelPackageHash::BYTE_LEN)
            .and_then(|value| value.checked_add(4))
            .and_then(|value| value.checked_add(2 + architecture.len()))
            .and_then(|value| value.checked_add(2 + backend.len()))
            .ok_or_else(|| EngineError::internal_invariant("model descriptor length overflow"))?;
        let mut bytes = Vec::with_capacity(capacity);
        bytes.push(1); // SHA-384
        bytes.extend_from_slice(self.package_hash.as_bytes());
        bytes.extend_from_slice(&self.context_limit.get().to_le_bytes());
        bytes.extend_from_slice(&architecture_length.to_le_bytes());
        bytes.extend_from_slice(architecture);
        bytes.extend_from_slice(&backend_length.to_le_bytes());
        bytes.extend_from_slice(backend);
        Ok(bytes)
    }

    fn decode(payload: &[u8]) -> Result<Self, EngineError> {
        const FIXED: usize = 1 + ModelPackageHash::BYTE_LEN + 4;
        if payload.len() < FIXED + 4 || payload[0] != 1 {
            return Err(EngineError::protocol_violation("model descriptor header is invalid"));
        }
        let mut digest = [0_u8; ModelPackageHash::BYTE_LEN];
        digest.copy_from_slice(&payload[1..=ModelPackageHash::BYTE_LEN]);
        let context_limit = ContextLimit::new(read_u32(payload, 1 + ModelPackageHash::BYTE_LEN)?)
            .map_err(|_| {
            EngineError::protocol_violation("model descriptor context is invalid")
        })?;
        let mut cursor = FIXED;
        let architecture_length = usize::from(read_u16(payload, cursor)?);
        cursor = cursor
            .checked_add(2)
            .ok_or_else(|| EngineError::protocol_violation("model descriptor length overflow"))?;
        let architecture_end = cursor
            .checked_add(architecture_length)
            .filter(|end| *end <= payload.len())
            .ok_or_else(|| {
                EngineError::protocol_violation("model descriptor architecture is truncated")
            })?;
        let architecture =
            std::str::from_utf8(&payload[cursor..architecture_end]).map_err(|_| {
                EngineError::protocol_violation("model descriptor architecture is invalid UTF-8")
            })?;
        cursor = architecture_end;
        let backend_length = usize::from(read_u16(payload, cursor)?);
        cursor = cursor
            .checked_add(2)
            .ok_or_else(|| EngineError::protocol_violation("model descriptor length overflow"))?;
        let backend_end =
            cursor.checked_add(backend_length).filter(|end| *end == payload.len()).ok_or_else(
                || EngineError::protocol_violation("model descriptor backend length is invalid"),
            )?;
        let backend = std::str::from_utf8(&payload[cursor..backend_end]).map_err(|_| {
            EngineError::protocol_violation("model descriptor backend is invalid UTF-8")
        })?;
        Ok(Self::new(
            ModelPackageHash::from_sha384_bytes(digest),
            ArchitectureId::new(architecture).map_err(|_| {
                EngineError::protocol_violation("model descriptor architecture is invalid")
            })?,
            BackendId::new(backend).map_err(|_| {
                EngineError::protocol_violation("model descriptor backend is invalid")
            })?,
            context_limit,
        ))
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

/// Atomic, content-free status for one registered model and its scheduler.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ServiceModelStatus {
    model_loaded: bool,
    health: ServiceHealthStatus,
    capabilities: ServiceCapabilitySet,
    active_requests: u32,
    queued_requests: u32,
    maximum_active: u32,
    maximum_queued: u32,
    queued_bytes: u64,
    maximum_queued_bytes: u64,
}

impl ServiceModelStatus {
    /// Constructs one internally consistent atomic status snapshot.
    #[expect(clippy::too_many_arguments, reason = "wire fields are intentionally explicit")]
    pub fn new(
        model_loaded: bool,
        health: ServiceHealthStatus,
        capabilities: ServiceCapabilitySet,
        active_requests: u32,
        queued_requests: u32,
        maximum_active: u32,
        maximum_queued: u32,
        queued_bytes: u64,
        maximum_queued_bytes: u64,
    ) -> Result<Self, EngineError> {
        if maximum_active == 0
            || maximum_queued == 0
            || maximum_queued_bytes == 0
            || active_requests > maximum_active
            || queued_requests > maximum_queued
            || queued_bytes > maximum_queued_bytes
        {
            return Err(EngineError::internal_invariant(
                "model status scheduler accounting is inconsistent",
            ));
        }
        Ok(Self {
            model_loaded,
            health,
            capabilities,
            active_requests,
            queued_requests,
            maximum_active,
            maximum_queued,
            queued_bytes,
            maximum_queued_bytes,
        })
    }

    /// Returns whether the requested registered model is loaded.
    pub const fn model_loaded(self) -> bool {
        self.model_loaded
    }

    /// Returns the service health captured in the same snapshot.
    pub const fn health(self) -> ServiceHealthStatus {
        self.health
    }

    /// Returns reviewed engine capabilities captured in the same snapshot.
    pub const fn capabilities(self) -> ServiceCapabilitySet {
        self.capabilities
    }

    /// Returns active request ownership.
    pub const fn active_requests(self) -> u32 {
        self.active_requests
    }

    /// Returns queued request ownership.
    pub const fn queued_requests(self) -> u32 {
        self.queued_requests
    }

    /// Returns the active request ceiling.
    pub const fn maximum_active(self) -> u32 {
        self.maximum_active
    }

    /// Returns the queued request ceiling.
    pub const fn maximum_queued(self) -> u32 {
        self.maximum_queued
    }

    /// Returns bytes currently retained by the request queue.
    pub const fn queued_bytes(self) -> u64 {
        self.queued_bytes
    }

    /// Returns the aggregate queued-byte ceiling.
    pub const fn maximum_queued_bytes(self) -> u64 {
        self.maximum_queued_bytes
    }

    fn encode(self) -> [u8; SERVICE_MODEL_STATUS_BYTES] {
        let mut encoded = [0_u8; SERVICE_MODEL_STATUS_BYTES];
        encoded[0] = u8::from(self.model_loaded);
        encoded[1] = self.health as u8;
        encoded[4..12].copy_from_slice(&self.capabilities.bits().to_le_bytes());
        encoded[12..16].copy_from_slice(&self.active_requests.to_le_bytes());
        encoded[16..20].copy_from_slice(&self.queued_requests.to_le_bytes());
        encoded[20..24].copy_from_slice(&self.maximum_active.to_le_bytes());
        encoded[24..28].copy_from_slice(&self.maximum_queued.to_le_bytes());
        encoded[28..36].copy_from_slice(&self.queued_bytes.to_le_bytes());
        encoded[36..44].copy_from_slice(&self.maximum_queued_bytes.to_le_bytes());
        encoded
    }

    fn decode(payload: &[u8]) -> Result<Self, EngineError> {
        require_length(payload, SERVICE_MODEL_STATUS_BYTES)?;
        if payload[0] > 1 || payload[2] != 0 || payload[3] != 0 {
            return Err(EngineError::protocol_violation(
                "model status flags or reserved bytes are invalid",
            ));
        }
        let status = Self::new(
            payload[0] == 1,
            ServiceHealthStatus::decode(payload[1])?,
            ServiceCapabilitySet::decode(read_u64(payload, 4)?)?,
            read_u32(payload, 12)?,
            read_u32(payload, 16)?,
            read_u32(payload, 20)?,
            read_u32(payload, 24)?,
            read_u64(payload, 28)?,
            read_u64(payload, 36)?,
        )
        .map_err(|_| EngineError::protocol_violation("model status accounting is inconsistent"))?;
        Ok(status)
    }
}

/// Closed sanitized security-event categories exported to the Go control plane.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum ServiceSecurityEventKind {
    /// Kernel peer or service authentication denied a client.
    UnauthorizedClient = 1,
    /// The bounded authentication-failure threshold was reached.
    RepeatedAuthenticationFailure = 2,
    /// A frame violated the fixed grammar.
    MalformedFrame = 3,
    /// A frame exceeded a hard byte ceiling.
    OversizedFrame = 4,
    /// A peer used an unsupported protocol version.
    UnknownProtocolVersion = 5,
    /// A syntactically invalid model identifier was supplied.
    InvalidModelId = 6,
    /// A registered model package signature failed.
    ModelSignatureFailure = 7,
    /// A registered model package hash failed.
    ModelHashFailure = 8,
    /// Descriptor-safe model resolution detected a path escape.
    ModelPathEscapeAttempt = 9,
    /// Repeated resource exhaustion crossed its threshold.
    ResourceExhaustionPattern = 10,
    /// The owner supervisor reported a crash-loop indication.
    CrashLoopIndication = 11,
    /// Repeated rejected requests crossed their threshold.
    RepeatedRejectedRequests = 12,
    /// A request failed authentication.
    AuthenticationFailure = 13,
    /// A request violated the local protocol.
    ProtocolViolation = 14,
    /// A well-formed but unregistered model was requested.
    UnregisteredModelRequested = 15,
    /// A configured data-only model package was rejected.
    ModelPackageRejected = 16,
    /// A finite engine resource limit was exceeded.
    ResourceLimitExceeded = 17,
    /// An inference backend failed.
    BackendFailure = 18,
    /// A forbidden authority operation was attempted.
    ForbiddenOperationAttempt = 19,
    /// A verified internal invariant failed closed.
    InternalInvariantViolation = 20,
}

impl ServiceSecurityEventKind {
    fn decode(value: u8) -> Result<Self, EngineError> {
        match value {
            1 => Ok(Self::UnauthorizedClient),
            2 => Ok(Self::RepeatedAuthenticationFailure),
            3 => Ok(Self::MalformedFrame),
            4 => Ok(Self::OversizedFrame),
            5 => Ok(Self::UnknownProtocolVersion),
            6 => Ok(Self::InvalidModelId),
            7 => Ok(Self::ModelSignatureFailure),
            8 => Ok(Self::ModelHashFailure),
            9 => Ok(Self::ModelPathEscapeAttempt),
            10 => Ok(Self::ResourceExhaustionPattern),
            11 => Ok(Self::CrashLoopIndication),
            12 => Ok(Self::RepeatedRejectedRequests),
            13 => Ok(Self::AuthenticationFailure),
            14 => Ok(Self::ProtocolViolation),
            15 => Ok(Self::UnregisteredModelRequested),
            16 => Ok(Self::ModelPackageRejected),
            17 => Ok(Self::ResourceLimitExceeded),
            18 => Ok(Self::BackendFailure),
            19 => Ok(Self::ForbiddenOperationAttempt),
            20 => Ok(Self::InternalInvariantViolation),
            _ => Err(EngineError::protocol_violation("security event kind is unsupported")),
        }
    }
}

/// One bounded, content-free security event suitable for authenticated polling.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ServiceSecurityEvent {
    kind: ServiceSecurityEventKind,
    detail: u8,
    request_id: Option<RequestId>,
    identifier: Option<String>,
}

impl ServiceSecurityEvent {
    /// Converts one internal typed event without exposing diagnostics or paths.
    pub fn from_event(event: SecurityEvent) -> Result<Self, EngineError> {
        let (kind, detail, request_id, identifier) = match event {
            SecurityEvent::UnauthorizedClient => {
                (ServiceSecurityEventKind::UnauthorizedClient, 0, None, None)
            }
            SecurityEvent::RepeatedAuthenticationFailure => {
                (ServiceSecurityEventKind::RepeatedAuthenticationFailure, 0, None, None)
            }
            SecurityEvent::MalformedFrame => {
                (ServiceSecurityEventKind::MalformedFrame, 0, None, None)
            }
            SecurityEvent::OversizedFrame => {
                (ServiceSecurityEventKind::OversizedFrame, 0, None, None)
            }
            SecurityEvent::UnknownProtocolVersion => {
                (ServiceSecurityEventKind::UnknownProtocolVersion, 0, None, None)
            }
            SecurityEvent::InvalidModelId { request_id } => {
                (ServiceSecurityEventKind::InvalidModelId, 0, request_id, None)
            }
            SecurityEvent::ModelSignatureFailure { model_id } => (
                ServiceSecurityEventKind::ModelSignatureFailure,
                0,
                None,
                Some(model_id.as_str().to_owned()),
            ),
            SecurityEvent::ModelHashFailure { model_id } => (
                ServiceSecurityEventKind::ModelHashFailure,
                0,
                None,
                Some(model_id.as_str().to_owned()),
            ),
            SecurityEvent::ModelPathEscapeAttempt { model_id } => (
                ServiceSecurityEventKind::ModelPathEscapeAttempt,
                0,
                None,
                model_id.map(|value| value.as_str().to_owned()),
            ),
            SecurityEvent::ResourceExhaustionPattern => {
                (ServiceSecurityEventKind::ResourceExhaustionPattern, 0, None, None)
            }
            SecurityEvent::CrashLoopIndication => {
                (ServiceSecurityEventKind::CrashLoopIndication, 0, None, None)
            }
            SecurityEvent::RepeatedRejectedRequests => {
                (ServiceSecurityEventKind::RepeatedRejectedRequests, 0, None, None)
            }
            SecurityEvent::AuthenticationFailure { request_id } => {
                (ServiceSecurityEventKind::AuthenticationFailure, 0, request_id, None)
            }
            SecurityEvent::ProtocolViolation { request_id } => {
                (ServiceSecurityEventKind::ProtocolViolation, 0, request_id, None)
            }
            SecurityEvent::UnregisteredModelRequested { request_id, model_id } => (
                ServiceSecurityEventKind::UnregisteredModelRequested,
                0,
                Some(request_id),
                Some(model_id.as_str().to_owned()),
            ),
            SecurityEvent::ModelPackageRejected { model_id, reason } => (
                ServiceSecurityEventKind::ModelPackageRejected,
                package_rejection_code(reason)?,
                None,
                Some(model_id.as_str().to_owned()),
            ),
            SecurityEvent::ResourceLimitExceeded { request_id, resource, .. } => (
                ServiceSecurityEventKind::ResourceLimitExceeded,
                resource_kind_code(resource)?,
                request_id,
                None,
            ),
            SecurityEvent::BackendFailure { request_id, backend } => (
                ServiceSecurityEventKind::BackendFailure,
                0,
                request_id,
                backend.map(|value| value.as_str().to_owned()),
            ),
            SecurityEvent::ForbiddenOperationAttempt { request_id, operation } => (
                ServiceSecurityEventKind::ForbiddenOperationAttempt,
                forbidden_operation_code(operation)?,
                request_id,
                None,
            ),
            SecurityEvent::InternalInvariantViolation { request_id } => {
                (ServiceSecurityEventKind::InternalInvariantViolation, 0, request_id, None)
            }
            _ => return Err(EngineError::not_implemented("security event has no wire mapping")),
        };
        Ok(Self { kind, detail, request_id, identifier })
    }

    /// Returns the closed event category.
    pub const fn kind(&self) -> ServiceSecurityEventKind {
        self.kind
    }

    /// Returns a category-specific closed detail code, or zero when absent.
    pub const fn detail(&self) -> u8 {
        self.detail
    }

    /// Returns the safe request correlation identifier when available.
    pub const fn request_id(&self) -> Option<RequestId> {
        self.request_id
    }

    /// Returns a configuration-owned model/backend identifier when available.
    pub fn identifier(&self) -> Option<&str> {
        self.identifier.as_deref()
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
    /// Atomic status for one registered model and scheduler.
    ModelStatus(ServiceModelStatus),
    /// Sanitized engine telemetry polled for Go-owned audit and delivery.
    SecurityEvents(Vec<ServiceSecurityEvent>),
    /// Verified identity of one registered and loaded model.
    ModelDescriptor(ServiceModelDescriptor),
    /// One delivered event prefix was removed.
    SecurityEventsAcknowledged,
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
            Self::ModelStatus(_) => ServiceMessageKind::ModelStatus,
            Self::SecurityEvents(_) => ServiceMessageKind::SecurityEvents,
            Self::ModelDescriptor(_) => ServiceMessageKind::ModelDescriptor,
            Self::SecurityEventsAcknowledged => ServiceMessageKind::AcknowledgeSecurityEvents,
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
            Self::ModelLoaded
            | Self::ModelUnloaded
            | Self::ShutdownAccepted
            | Self::SecurityEventsAcknowledged => Ok(Vec::new()),
            Self::Tokenized(tokens) => encode_tokens(tokens),
            Self::Health(state) => Ok(vec![*state as u8]),
            Self::Metrics(metrics) => {
                let mut encoded = Vec::with_capacity(SERVICE_METRIC_COUNT * 8);
                for value in metrics.values() {
                    encoded.extend_from_slice(&value.to_le_bytes());
                }
                Ok(encoded)
            }
            Self::ModelStatus(status) => Ok(status.encode().to_vec()),
            Self::SecurityEvents(events) => encode_security_events(events),
            Self::ModelDescriptor(descriptor) => descriptor.encode(),
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
            ServiceMessageKind::ModelStatus => {
                Ok(Self::ModelStatus(ServiceModelStatus::decode(payload)?))
            }
            ServiceMessageKind::SecurityEvents => {
                Ok(Self::SecurityEvents(decode_security_events(payload)?))
            }
            ServiceMessageKind::ModelDescriptor => {
                Ok(Self::ModelDescriptor(ServiceModelDescriptor::decode(payload)?))
            }
            ServiceMessageKind::AcknowledgeSecurityEvents => {
                require_empty(payload)?;
                Ok(Self::SecurityEventsAcknowledged)
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

fn validate_security_event_maximum(maximum: u16) -> Result<(), EngineError> {
    if maximum == 0 || usize::from(maximum) > MAX_SERVICE_SECURITY_EVENTS {
        return Err(EngineError::protocol_violation(
            "security event batch limit is outside bounds",
        ));
    }
    Ok(())
}

fn encode_security_events(events: &[ServiceSecurityEvent]) -> Result<Vec<u8>, EngineError> {
    if events.len() > MAX_SERVICE_SECURITY_EVENTS {
        return Err(EngineError::resource_exhaustion(
            "security event response exceeds its count bound",
        ));
    }
    let count = u16::try_from(events.len())
        .map_err(|_| EngineError::resource_exhaustion("security event count exceeds u16"))?;
    let mut encoded = Vec::new();
    encoded.extend_from_slice(&count.to_le_bytes());
    for event in events {
        validate_service_security_event(event)?;
        let mut flags = 0_u8;
        if event.request_id.is_some() {
            flags |= 1;
        }
        if event.identifier.is_some() {
            flags |= 2;
        }
        encoded.extend_from_slice(&[event.kind as u8, event.detail, flags, 0]);
        if let Some(request_id) = event.request_id {
            encoded.extend_from_slice(&request_id.to_bytes());
        }
        if let Some(identifier) = &event.identifier {
            let length = u16::try_from(identifier.len())
                .map_err(|_| EngineError::resource_exhaustion("security identifier exceeds u16"))?;
            encoded.extend_from_slice(&length.to_le_bytes());
            encoded.extend_from_slice(identifier.as_bytes());
        }
        if encoded.len() > MAX_REQUEST_BYTES {
            return Err(EngineError::resource_exhaustion(
                "security event response exceeds the request byte bound",
            ));
        }
    }
    Ok(encoded)
}

fn decode_security_events(payload: &[u8]) -> Result<Vec<ServiceSecurityEvent>, EngineError> {
    if payload.len() < 2 {
        return Err(EngineError::protocol_violation("security event response is truncated"));
    }
    let count = usize::from(read_u16(payload, 0)?);
    if count > MAX_SERVICE_SECURITY_EVENTS {
        return Err(EngineError::protocol_violation(
            "security event response exceeds its count bound",
        ));
    }
    let mut cursor = 2_usize;
    let mut events = Vec::new();
    events
        .try_reserve_exact(count)
        .map_err(|_| EngineError::resource_exhaustion("security event allocation failed"))?;
    for _ in 0..count {
        let header_end = cursor
            .checked_add(4)
            .filter(|end| *end <= payload.len())
            .ok_or_else(|| EngineError::protocol_violation("security event header is truncated"))?;
        let kind = ServiceSecurityEventKind::decode(payload[cursor])?;
        let detail = payload[cursor + 1];
        let flags = payload[cursor + 2];
        if flags & !0b11 != 0 || payload[cursor + 3] != 0 {
            return Err(EngineError::protocol_violation("security event flags are invalid"));
        }
        cursor = header_end;
        let request_id =
            if flags & 1 != 0 {
                let end = cursor.checked_add(16).filter(|end| *end <= payload.len()).ok_or_else(
                    || EngineError::protocol_violation("security request ID is truncated"),
                )?;
                let bytes: [u8; 16] = payload[cursor..end].try_into().map_err(|_| {
                    EngineError::protocol_violation("security request ID is invalid")
                })?;
                cursor = end;
                Some(RequestId::from_bytes(bytes).map_err(|_| {
                    EngineError::protocol_violation("security request ID is invalid")
                })?)
            } else {
                None
            };
        let identifier = if flags & 2 != 0 {
            let length = usize::from(read_u16(payload, cursor)?);
            cursor = cursor.checked_add(2).ok_or_else(|| {
                EngineError::protocol_violation("security identifier offset overflows")
            })?;
            let end = cursor.checked_add(length).filter(|end| *end <= payload.len()).ok_or_else(
                || EngineError::protocol_violation("security identifier is truncated"),
            )?;
            let value = std::str::from_utf8(&payload[cursor..end])
                .map_err(|_| EngineError::protocol_violation("security identifier is not UTF-8"))?;
            let validated = ModelId::new(value)
                .map_err(|_| EngineError::protocol_violation("security identifier is invalid"))?;
            cursor = end;
            Some(validated.as_str().to_owned())
        } else {
            None
        };
        let event = ServiceSecurityEvent { kind, detail, request_id, identifier };
        validate_service_security_event(&event)?;
        events.push(event);
    }
    if cursor != payload.len() {
        return Err(EngineError::protocol_violation(
            "security event response contains trailing data",
        ));
    }
    Ok(events)
}

fn validate_service_security_event(event: &ServiceSecurityEvent) -> Result<(), EngineError> {
    let has_request = event.request_id.is_some();
    let has_identifier = event.identifier.is_some();
    let valid_shape = match event.kind {
        ServiceSecurityEventKind::UnauthorizedClient
        | ServiceSecurityEventKind::RepeatedAuthenticationFailure
        | ServiceSecurityEventKind::MalformedFrame
        | ServiceSecurityEventKind::OversizedFrame
        | ServiceSecurityEventKind::UnknownProtocolVersion
        | ServiceSecurityEventKind::ResourceExhaustionPattern
        | ServiceSecurityEventKind::CrashLoopIndication
        | ServiceSecurityEventKind::RepeatedRejectedRequests => !has_request && !has_identifier,
        ServiceSecurityEventKind::InvalidModelId
        | ServiceSecurityEventKind::AuthenticationFailure
        | ServiceSecurityEventKind::ProtocolViolation
        | ServiceSecurityEventKind::ResourceLimitExceeded
        | ServiceSecurityEventKind::ForbiddenOperationAttempt
        | ServiceSecurityEventKind::InternalInvariantViolation => !has_identifier,
        ServiceSecurityEventKind::ModelSignatureFailure
        | ServiceSecurityEventKind::ModelHashFailure
        | ServiceSecurityEventKind::ModelPackageRejected => !has_request && has_identifier,
        ServiceSecurityEventKind::ModelPathEscapeAttempt => !has_request,
        ServiceSecurityEventKind::UnregisteredModelRequested => has_request && has_identifier,
        ServiceSecurityEventKind::BackendFailure => true,
    };
    let valid_detail = match event.kind {
        ServiceSecurityEventKind::ModelPackageRejected => (1..=6).contains(&event.detail),
        ServiceSecurityEventKind::ResourceLimitExceeded => (1..=14).contains(&event.detail),
        ServiceSecurityEventKind::ForbiddenOperationAttempt => (1..=9).contains(&event.detail),
        _ => event.detail == 0,
    };
    if !valid_shape || !valid_detail {
        return Err(EngineError::protocol_violation("security event fields are inconsistent"));
    }
    if let Some(identifier) = &event.identifier {
        ModelId::new(identifier.as_str())
            .map_err(|_| EngineError::protocol_violation("security identifier is invalid"))?;
    }
    Ok(())
}

fn package_rejection_code(reason: ModelPackageRejection) -> Result<u8, EngineError> {
    match reason {
        ModelPackageRejection::Corrupt => Ok(1),
        ModelPackageRejection::SignatureFailure => Ok(2),
        ModelPackageRejection::UnsupportedArchitecture => Ok(3),
        ModelPackageRejection::UnsupportedTokenizer => Ok(4),
        ModelPackageRejection::UnregisteredModel => Ok(5),
        ModelPackageRejection::SizeLimit => Ok(6),
        _ => Err(EngineError::not_implemented("model rejection has no wire code")),
    }
}

fn resource_kind_code(resource: ResourceKind) -> Result<u8, EngineError> {
    match resource {
        ResourceKind::ModelPackageBytes => Ok(1),
        ResourceKind::TensorCount => Ok(2),
        ResourceKind::TensorRank => Ok(3),
        ResourceKind::TensorDimension => Ok(4),
        ResourceKind::TensorElements => Ok(5),
        ResourceKind::TensorBytes => Ok(6),
        ResourceKind::TokenCount => Ok(7),
        ResourceKind::VocabularySize => Ok(8),
        ResourceKind::ContextTokens => Ok(9),
        ResourceKind::RequestBytes => Ok(10),
        ResourceKind::FrameBytes => Ok(11),
        ResourceKind::MemoryBytes => Ok(12),
        ResourceKind::ConcurrentSequences => Ok(13),
        ResourceKind::SequenceIdentifiers => Ok(14),
        _ => Err(EngineError::not_implemented("resource kind has no wire code")),
    }
}

fn forbidden_operation_code(operation: ForbiddenOperation) -> Result<u8, EngineError> {
    match operation {
        ForbiddenOperation::ToolExecution => Ok(1),
        ForbiddenOperation::ApprovalDecision => Ok(2),
        ForbiddenOperation::MemoryMutation => Ok(3),
        ForbiddenOperation::ArbitraryFileAccess => Ok(4),
        ForbiddenOperation::ModelDownload => Ok(5),
        ForbiddenOperation::SelfUpdate => Ok(6),
        ForbiddenOperation::OutboundNetwork => Ok(7),
        ForbiddenOperation::ModelExecutableCode => Ok(8),
        ForbiddenOperation::ModelDefinedKernel => Ok(9),
        _ => Err(EngineError::not_implemented("forbidden operation has no wire code")),
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
            ServiceRequestBody::ModelStatus(model("tiny-status")),
            ServiceRequestBody::SecurityEvents { maximum: 32 },
            ServiceRequestBody::ModelDescriptor(model("tiny-descriptor")),
            ServiceRequestBody::AcknowledgeSecurityEvents { count: 1 },
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
            ServiceResponseBody::ModelStatus(
                ServiceModelStatus::new(
                    true,
                    ServiceHealthStatus::Ready,
                    capabilities,
                    1,
                    2,
                    4,
                    8,
                    1_024,
                    8_192,
                )
                .expect("model status"),
            ),
            ServiceResponseBody::SecurityEvents(vec![
                ServiceSecurityEvent::from_event(SecurityEvent::UnauthorizedClient)
                    .expect("event mapping"),
                ServiceSecurityEvent::from_event(SecurityEvent::UnregisteredModelRequested {
                    request_id: RequestId::new(7).expect("request ID"),
                    model_id: model("unknown-model"),
                })
                .expect("event mapping"),
                ServiceSecurityEvent::from_event(SecurityEvent::ModelPackageRejected {
                    model_id: model("bad-package"),
                    reason: ModelPackageRejection::SignatureFailure,
                })
                .expect("event mapping"),
            ]),
            ServiceResponseBody::ModelDescriptor(ServiceModelDescriptor::new(
                ModelPackageHash::from_sha384_bytes([0x55; ModelPackageHash::BYTE_LEN]),
                ArchitectureId::new("decoder-rms-rope-swiglu-gqa-v1").expect("architecture"),
                BackendId::new("cpu-reference").expect("backend"),
                ContextLimit::new(8_192).expect("context"),
            )),
            ServiceResponseBody::SecurityEventsAcknowledged,
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
        assert!(ServiceRequestBody::decode(ServiceMessageKind::SecurityEvents, &[0, 0]).is_err());
        assert!(
            ServiceRequestBody::decode(ServiceMessageKind::AcknowledgeSecurityEvents, &[0, 0])
                .is_err()
        );
        assert!(
            ServiceRequestBody::decode(
                ServiceMessageKind::SecurityEvents,
                &u16::try_from(MAX_SERVICE_SECURITY_EVENTS + 1)
                    .expect("test bound fits u16")
                    .to_le_bytes(),
            )
            .is_err()
        );
        assert!(
            ServiceResponseBody::decode(
                ServiceMessageKind::ModelStatus,
                &[0; SERVICE_MODEL_STATUS_BYTES - 1],
            )
            .is_err()
        );
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
        let mut inconsistent = [0_u8; SERVICE_MODEL_STATUS_BYTES];
        inconsistent[0] = 2;
        inconsistent[1] = ServiceHealthStatus::Ready as u8;
        inconsistent[20..24].copy_from_slice(&1_u32.to_le_bytes());
        inconsistent[24..28].copy_from_slice(&1_u32.to_le_bytes());
        inconsistent[36..44].copy_from_slice(&1_u64.to_le_bytes());
        assert!(
            ServiceResponseBody::decode(ServiceMessageKind::ModelStatus, &inconsistent).is_err()
        );
        let mut invalid_security = ServiceResponseBody::SecurityEvents(vec![
            ServiceSecurityEvent::from_event(SecurityEvent::UnauthorizedClient)
                .expect("event mapping"),
        ])
        .encode()
        .expect("security response");
        invalid_security[3] = 1;
        assert!(
            ServiceResponseBody::decode(ServiceMessageKind::SecurityEvents, &invalid_security)
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
