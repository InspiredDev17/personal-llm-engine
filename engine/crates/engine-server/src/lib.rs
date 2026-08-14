//! Local-only composition boundary for the native inference engine.
//!
//! This crate validates Unix-domain-socket configuration, admits requests for
//! locally registered model IDs, and can execute the deterministic CPU
//! reference path. It contains no TCP or outbound network capability, tool
//! path, approval authority, arbitrary model path, or platform/model persistence
//! authority. Its only mutable disk state is the bounded, configuration-derived
//! replay journal required before authenticated command execution.

#[cfg(all(unix, feature = "cpu-reference"))]
mod continuous;
#[cfg(unix)]
mod os_lock;
#[cfg(unix)]
mod peer_credential;
#[cfg(unix)]
mod replay;
mod runtime_reservations;
mod runtime_v3_adapter;
#[cfg(all(unix, feature = "cpu-reference"))]
mod runtime_v3_service;
#[cfg(unix)]
mod transport;

#[cfg(all(unix, feature = "cpu-reference"))]
pub use continuous::{
    ContinuousServeLimits, ContinuousServeSummary, ContinuousShutdown, MAX_CONTINUOUS_CONNECTIONS,
};
#[cfg(unix)]
pub use replay::{DurableReplayJournal, MAX_DURABLE_REPLAY_ENTRIES};
pub use runtime_reservations::{
    ArtifactAuthorityId, INFERENCE_RUNTIME_V3_CONTRACT_COMMIT,
    INFERENCE_RUNTIME_V3_CONTRACT_VERSION, LogicalModelId, MAX_RUNTIME_RESERVATION_RECORDS,
    MAX_RUNTIME_V3_AVAILABILITY_TTL_MILLIS, ReservationDecision, ReservationDenial, ReservationId,
    ReservationStatistics, ReservationStatus, RuntimeIdentityBinding, RuntimeInstanceId,
    RuntimeReservationManager, RuntimeReservationRequest, RuntimeResourceCeiling, TaskAuthorityId,
};
pub use runtime_v3_adapter::{
    AdmittedRuntimeExecution, RuntimeIdentityAuthority, RuntimeV3ContractAdapter,
};
#[cfg(all(unix, feature = "cpu-reference", feature = "development-runtime-v3-simulator"))]
pub use runtime_v3_service::RuntimeV3DevelopmentBehavior;
#[cfg(all(unix, feature = "cpu-reference"))]
pub use runtime_v3_service::RuntimeV3ServiceConfig;
#[cfg(unix)]
pub use transport::{UnixFrameConnection, UnixFrameListener, UnixSocketPolicy};

#[cfg(all(unix, feature = "cpu-reference"))]
use personalai_engine_crypto::EngineResponseSigner;
use personalai_engine_crypto::{TrustedControlPlaneKey, TrustedControlPlaneKeys, TrustedSigner};
#[cfg(all(feature = "cpu-reference", test))]
use personalai_engine_crypto::{hash_package, hash_signed_content, verify_package_signature};
#[cfg(all(unix, feature = "cpu-reference"))]
use personalai_engine_loader::DirectorySecurityEventSink;
use personalai_engine_loader::ModelRegistry;
pub use personalai_engine_loader::RegistrySelectionV2;
#[cfg(feature = "cpu-reference")]
use personalai_engine_memory::{MemoryOwner, MemoryPool};
#[cfg(all(feature = "cpu-reference", test))]
use personalai_engine_model_format::ParsedModelPackage;
use personalai_engine_protocol::{
    AuthenticatedServiceRequest, CancellationEnvelope, Frame, FrameKind, GenerationRequestEnvelope,
    PROTOCOL_MAJOR, PROTOCOL_MINOR,
};
#[cfg(all(unix, feature = "cpu-reference"))]
use personalai_engine_protocol::{
    FRAME_HEADER_BYTES, GENERATION_EVENT_AUTH_BYTES, ServiceCapabilitySet, ServiceErrorCode,
    ServiceHealthStatus, ServiceMessageKind, ServiceMetrics, ServiceModelDescriptor,
    ServiceModelStatus, ServiceRequestBody, ServiceResponseBody, ServiceSecurityEvent,
    ShutdownRejectionReason, encode_signed_service_response,
};
#[cfg(feature = "cpu-reference")]
use personalai_engine_sampling::{Sampler, SamplingConfig};
#[cfg(all(unix, feature = "cpu-reference"))]
use personalai_engine_scheduler::CancellationOutcome;
#[cfg(feature = "cpu-reference")]
use personalai_engine_scheduler::MAX_QUEUED_REQUEST_BYTES;
use personalai_engine_scheduler::{Scheduler, SchedulerLimits};
#[cfg(all(feature = "cpu-reference", test))]
use personalai_engine_tensor::Tensor;
#[cfg(feature = "cpu-reference")]
use personalai_engine_tokenizer::{IncrementalDecoder, Tokenizer, build_tokenizer};
#[cfg(feature = "arm64")]
use personalai_engine_transformer::Arm64Backend;
#[cfg(all(feature = "cpu-reference", feature = "metal", target_os = "macos"))]
pub use personalai_engine_transformer::MetalBackend;
use personalai_engine_transformer::{BackendStatus, backend_status};
#[cfg(feature = "cpu-reference")]
use personalai_engine_transformer::{CpuDecoderModel, CpuDecoderSession, TransformerConfig};
#[cfg(all(feature = "cpu-reference", feature = "metal", target_os = "macos"))]
use personalai_engine_transformer::{MetalDecoderModel, MetalDecoderSession};
use personalai_engine_types::{
    BackendId, EngineError, GenerationRequest, MAX_FRAME_BYTES, ModelId, ModelPackageHash,
    RequestId,
};
#[cfg(feature = "cpu-reference")]
use personalai_engine_types::{
    EngineCapability, GenerationEvent, GenerationUsage, MAX_CONCURRENT_SEQUENCES,
    MAX_GENERATION_CHUNK_BYTES, ModelPackageRejection, ResourceBudget, ResourceKind, SecurityEvent,
    SequenceId, TextChunk, TokenId,
};
#[cfg(all(unix, feature = "cpu-reference"))]
use std::collections::BTreeMap;
#[cfg(feature = "cpu-reference")]
use std::collections::VecDeque;
use std::path::{Component, Path, PathBuf};
#[cfg(feature = "cpu-reference")]
use std::sync::Arc;
#[cfg(all(unix, feature = "cpu-reference"))]
use std::sync::Mutex;
#[cfg(feature = "cpu-reference")]
use std::sync::atomic::{AtomicBool, Ordering};
#[cfg(all(unix, feature = "cpu-reference"))]
use std::sync::mpsc::{Receiver, SyncSender, TrySendError};
#[cfg(all(unix, feature = "cpu-reference"))]
use std::thread;
#[cfg(feature = "cpu-reference")]
use std::time::Instant;
#[cfg(all(unix, feature = "cpu-reference"))]
use std::time::{SystemTime, UNIX_EPOCH};

/// Maximum configured Unix socket path bytes for conservative portability.
pub const MAX_UNIX_SOCKET_PATH_BYTES: usize = 100;
/// Hard ceiling for the bounded multi-connection reference batch.
pub const MAX_CONCURRENT_CONNECTION_BATCH: u32 = 64;
/// Maximum signed generation events retained for one local output stream.
#[cfg(all(unix, feature = "cpu-reference"))]
pub const MAX_PENDING_EVENTS_PER_CONNECTION: usize = 8;
/// Maximum sanitized security events retained for Go control-plane polling.
#[cfg(all(unix, feature = "cpu-reference"))]
pub const MAX_PENDING_SECURITY_EVENTS: usize = 256;
#[cfg(all(unix, feature = "cpu-reference"))]
pub(crate) const SECURITY_PATTERN_THRESHOLD: u64 = 8;
#[cfg(all(unix, feature = "cpu-reference"))]
const CANCELLATION_WATCHER_STACK_BYTES: usize = 256 * 1_024;
#[cfg(all(unix, feature = "cpu-reference"))]
const OUTPUT_WRITER_STACK_BYTES: usize = 256 * 1_024;
#[cfg(all(unix, feature = "cpu-reference"))]
const MAX_OUTBOUND_EVENT_FRAME_BYTES: usize = FRAME_HEADER_BYTES
    + 44
    + MAX_GENERATION_CHUNK_BYTES
    + GENERATION_EVENT_AUTH_BYTES
    + personalai_engine_crypto::ED25519_SIGNATURE_BYTES;
#[cfg(all(unix, feature = "cpu-reference"))]
const OUTBOUND_FRAMES_RESERVED_PER_CONNECTION: usize = MAX_PENDING_EVENTS_PER_CONNECTION + 2;
#[cfg(all(unix, feature = "cpu-reference"))]
const OUTBOUND_PIPELINE_RESERVED_BYTES: usize =
    OUTBOUND_FRAMES_RESERVED_PER_CONNECTION * MAX_OUTBOUND_EVENT_FRAME_BYTES;
#[cfg(all(unix, feature = "cpu-reference"))]
const _: () = assert!(MAX_OUTBOUND_EVENT_FRAME_BYTES <= MAX_FRAME_BYTES);
/// Human-readable status for the current reference runtime.
pub const FOUNDATION_STATUS: &str = "signed-package-trusted-backend-uds";

/// Owner-selected execution backend for the protected local service.
///
/// This value is constructed only from startup configuration. It is never
/// decoded from a client request or model package.
#[cfg(feature = "cpu-reference")]
#[derive(Clone, Debug, Default)]
pub enum RuntimeBackend {
    /// Deterministic scalar correctness backend.
    #[default]
    CpuReference,
    /// Reviewed ARM64 host backend with closed scalar fallback per operation.
    #[cfg(all(feature = "arm64", target_arch = "aarch64"))]
    Arm64(Arm64Backend),
    /// Reviewed Apple Metal backend and its fixed device allocation policy.
    #[cfg(all(feature = "metal", target_os = "macos"))]
    Metal(MetalBackend),
}

#[cfg(feature = "cpu-reference")]
impl RuntimeBackend {
    /// Constructs deterministic owner-selected ARM64 execution on a supported host.
    #[cfg(all(feature = "arm64", target_arch = "aarch64"))]
    pub fn arm64() -> Result<Self, EngineError> {
        Ok(Self::Arm64(Arm64Backend::new()?))
    }

    /// Returns the protocol identity pinned by the Go control plane.
    pub fn backend_id(&self) -> Result<BackendId, EngineError> {
        match self {
            Self::CpuReference => BackendId::new("cpu-reference"),
            #[cfg(all(feature = "arm64", target_arch = "aarch64"))]
            Self::Arm64(_) => BackendId::new("arm64"),
            #[cfg(all(feature = "metal", target_os = "macos"))]
            Self::Metal(_) => BackendId::new("metal"),
        }
    }

    fn capabilities(&self) -> Result<ServiceCapabilitySet, EngineError> {
        let capabilities = vec![
            EngineCapability::CpuReference,
            EngineCapability::StreamingGeneration,
            EngineCapability::UnixDomainSocket,
        ];
        #[cfg(all(feature = "arm64", target_arch = "aarch64"))]
        let capabilities = {
            let mut capabilities = capabilities;
            if matches!(self, Self::Arm64(_)) {
                capabilities.push(EngineCapability::Arm64);
            }
            capabilities
        };
        #[cfg(all(feature = "metal", target_os = "macos"))]
        let capabilities = {
            let mut capabilities = capabilities;
            if matches!(self, Self::Metal(_)) {
                capabilities.push(EngineCapability::Metal);
            }
            capabilities
        };
        ServiceCapabilitySet::from_capabilities(capabilities)
    }

    fn device_memory_bytes(&self) -> Result<u64, EngineError> {
        match self {
            Self::CpuReference => Ok(0),
            #[cfg(all(feature = "arm64", target_arch = "aarch64"))]
            Self::Arm64(_) => Ok(0),
            #[cfg(all(feature = "metal", target_os = "macos"))]
            Self::Metal(backend) => Ok(backend.statistics()?.current_allocated_bytes),
        }
    }
}

/// Validated local service configuration that does not open any resource.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ServiceConfig {
    model_root: PathBuf,
    runtime_root: PathBuf,
    socket_path: PathBuf,
}

impl ServiceConfig {
    /// Validates configuration-owned model, runtime-state, and socket paths.
    pub fn new(
        model_root: impl Into<PathBuf>,
        runtime_root: impl Into<PathBuf>,
        socket_path: impl Into<PathBuf>,
    ) -> Result<Self, EngineError> {
        let model_root = model_root.into();
        let runtime_root = runtime_root.into();
        let socket_path = socket_path.into();
        validate_absolute_normal_path(&model_root, "model root")?;
        validate_absolute_normal_path(&runtime_root, "runtime root")?;
        validate_absolute_normal_path(&socket_path, "socket path")?;
        if model_root.starts_with(&runtime_root) || runtime_root.starts_with(&model_root) {
            return Err(EngineError::invalid_input("model and runtime roots must be disjoint"));
        }
        if socket_path.parent() != Some(runtime_root.as_path()) {
            return Err(EngineError::invalid_input(
                "engine socket must be a direct child of its configured runtime root",
            ));
        }
        if socket_path.extension().and_then(|extension| extension.to_str()) != Some("sock") {
            return Err(EngineError::invalid_input(
                "engine socket path must use the .sock extension",
            ));
        }
        if socket_path.as_os_str().as_encoded_bytes().len() > MAX_UNIX_SOCKET_PATH_BYTES {
            return Err(EngineError::invalid_input(
                "engine socket path exceeds the portable hard limit",
            ));
        }
        Ok(Self { model_root, runtime_root, socket_path })
    }

    /// Returns the local configuration-owned model root.
    pub fn model_root(&self) -> &Path {
        &self.model_root
    }

    /// Returns the protected runtime directory.
    pub fn runtime_root(&self) -> &Path {
        &self.runtime_root
    }

    /// Returns the Unix-domain-socket path without opening it.
    pub fn socket_path(&self) -> &Path {
        &self.socket_path
    }
}

/// Read-only protocol metadata for a future Go adapter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProtocolDescriptor {
    /// Protocol major version.
    pub major: u16,
    /// Protocol minor version.
    pub minor: u16,
    /// Maximum accepted complete encoded frame bytes.
    pub maximum_frame_bytes: usize,
    /// Whether a listener exists in this build stage.
    pub listener_implemented: bool,
}

impl ProtocolDescriptor {
    /// Returns truthful foundation protocol metadata.
    pub const fn foundation() -> Self {
        Self {
            major: PROTOCOL_MAJOR,
            minor: PROTOCOL_MINOR,
            maximum_frame_bytes: MAX_FRAME_BYTES,
            listener_implemented: cfg!(unix),
        }
    }
}

/// Receipt proving only bounded scheduler admission, not inference success.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AdmissionReceipt {
    request_id: RequestId,
    queued_requests: usize,
}

/// Authenticated command accepted from the Go control plane.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum InboundCommand {
    /// One fully validated generation request.
    Generate {
        /// Authenticated request body.
        request: GenerationRequest,
        /// Nonzero request nonce bound into every response signature.
        nonce: u64,
    },
    /// Cancellation of one previously submitted request.
    Cancel(RequestId),
    /// One authenticated non-generation service command.
    Service(AuthenticatedServiceRequest),
}

/// Per-connection authentication and strict monotonic anti-replay state.
#[derive(Clone, Debug)]
pub struct AuthenticatedControlSession {
    trusted_keys: TrustedControlPlaneKeys,
    last_nonce: u64,
    authenticated_key_id: Option<personalai_engine_types::SignerKeyId>,
}

impl AuthenticatedControlSession {
    /// Creates a fresh connection session with no accepted nonce.
    pub fn new(trusted_key: TrustedControlPlaneKey) -> Result<Self, EngineError> {
        Ok(Self::new_with_keys(TrustedControlPlaneKeys::single(trusted_key)?))
    }

    /// Creates a fresh connection session with a bounded rotation key ring.
    pub const fn new_with_keys(trusted_keys: TrustedControlPlaneKeys) -> Self {
        Self { trusted_keys, last_nonce: 0, authenticated_key_id: None }
    }

    /// Authenticates, decodes, and replay-checks one inbound command frame.
    pub fn decode_command(&mut self, frame: &Frame) -> Result<InboundCommand, EngineError> {
        let (nonce, control_key_id, command) = match frame.kind() {
            FrameKind::GenerationRequest => {
                let envelope =
                    GenerationRequestEnvelope::decode(frame.payload(), &self.trusted_keys)?;
                let nonce = envelope.nonce();
                (
                    nonce,
                    envelope.control_key_id(),
                    InboundCommand::Generate { request: envelope.into_request(), nonce },
                )
            }
            FrameKind::Cancellation => {
                let envelope = CancellationEnvelope::decode(frame.payload(), &self.trusted_keys)?;
                (
                    envelope.nonce(),
                    envelope.control_key_id(),
                    InboundCommand::Cancel(envelope.request_id()),
                )
            }
            FrameKind::ServiceRequest => {
                let envelope =
                    AuthenticatedServiceRequest::decode(frame.payload(), &self.trusted_keys)?;
                (envelope.nonce(), envelope.control_key_id(), InboundCommand::Service(envelope))
            }
            _ => {
                return Err(EngineError::protocol_violation(
                    "frame kind is not an authenticated command",
                ));
            }
        };
        if self
            .authenticated_key_id
            .is_some_and(|authenticated_key_id| authenticated_key_id != control_key_id)
        {
            return Err(EngineError::authentication_failure(
                "control-plane key changed within one command stream",
            ));
        }
        if nonce <= self.last_nonce {
            return Err(EngineError::authentication_failure(
                "control-plane request nonce was replayed or reordered",
            ));
        }
        self.authenticated_key_id = Some(control_key_id);
        self.last_nonce = nonce;
        Ok(command)
    }

    /// Returns the last accepted nonce.
    pub const fn last_nonce(&self) -> u64 {
        self.last_nonce
    }

    /// Returns the key identity bound to this command stream, if any.
    pub const fn authenticated_key_id(&self) -> Option<personalai_engine_types::SignerKeyId> {
        self.authenticated_key_id
    }
}

impl AdmissionReceipt {
    /// Returns the admitted request identifier.
    pub const fn request_id(self) -> RequestId {
        self.request_id
    }

    /// Returns the queue length immediately after admission.
    pub const fn queued_requests(self) -> usize {
        self.queued_requests
    }
}

/// Foundation composition root with no transport or generation executor.
#[derive(Debug)]
pub struct EngineService {
    config: ServiceConfig,
    registry: ModelRegistry,
    scheduler: Scheduler,
}

/// Fully loaded CPU model and its first-party tokenizer.
#[cfg(feature = "cpu-reference")]
#[derive(Clone)]
pub struct LoadedCpuModel {
    model_id: ModelId,
    model: CpuDecoderModel,
    tokenizer: Arc<dyn Tokenizer>,
}

/// Fully loaded, signed model whose complete decoder executes through Metal.
#[cfg(all(feature = "cpu-reference", feature = "metal", target_os = "macos"))]
#[derive(Clone)]
pub struct LoadedMetalModel {
    model_id: ModelId,
    model: MetalDecoderModel,
    tokenizer: Arc<dyn Tokenizer>,
}

#[cfg(feature = "cpu-reference")]
#[derive(Clone)]
enum LoadedRuntimeModel {
    Cpu(LoadedCpuModel),
    #[cfg(all(feature = "metal", target_os = "macos"))]
    Metal(LoadedMetalModel),
}

#[cfg(feature = "cpu-reference")]
impl LoadedRuntimeModel {
    fn tokenizer(&self) -> &dyn Tokenizer {
        match self {
            Self::Cpu(loaded) => loaded.tokenizer(),
            #[cfg(all(feature = "metal", target_os = "macos"))]
            Self::Metal(loaded) => loaded.tokenizer(),
        }
    }

    fn start_greedy(
        &self,
        request: &GenerationRequest,
        sequence_id: SequenceId,
    ) -> Result<GreedyGeneration, EngineError> {
        match self {
            Self::Cpu(loaded) => loaded.start_greedy(request, sequence_id),
            #[cfg(all(feature = "metal", target_os = "macos"))]
            Self::Metal(loaded) => loaded.start_greedy(request, sequence_id),
        }
    }

    fn memory_pool(&self) -> &MemoryPool {
        match self {
            Self::Cpu(loaded) => loaded.model().memory_pool(),
            #[cfg(all(feature = "metal", target_os = "macos"))]
            Self::Metal(loaded) => loaded.model().memory_pool(),
        }
    }

    fn kv_cache_bytes(&self) -> Result<u64, EngineError> {
        match self {
            Self::Cpu(loaded) => Ok(loaded.model().kv_cache_metrics()?.reserved_bytes),
            #[cfg(all(feature = "metal", target_os = "macos"))]
            Self::Metal(loaded) => Ok(loaded.model().kv_cache_metrics()?.reserved_bytes),
        }
    }

    fn architecture_id(&self) -> &personalai_engine_types::ArchitectureId {
        match self {
            Self::Cpu(loaded) => loaded.model().config().architecture_id(),
            #[cfg(all(feature = "metal", target_os = "macos"))]
            Self::Metal(loaded) => loaded.model().config().architecture_id(),
        }
    }

    fn context_limit(&self) -> personalai_engine_types::ContextLimit {
        match self {
            Self::Cpu(loaded) => loaded.model().config().context_limit(),
            #[cfg(all(feature = "metal", target_os = "macos"))]
            Self::Metal(loaded) => loaded.model().config().context_limit(),
        }
    }

    fn backend_id(&self) -> Result<BackendId, EngineError> {
        match self {
            Self::Cpu(loaded) => loaded.model().backend_id(),
            #[cfg(all(feature = "metal", target_os = "macos"))]
            Self::Metal(_) => BackendId::new("metal"),
        }
    }

    fn unload_if_idle(&self) -> Result<(), EngineError> {
        match self {
            Self::Cpu(_) => Ok(()),
            #[cfg(all(feature = "metal", target_os = "macos"))]
            Self::Metal(loaded) => loaded.model().unload_if_idle().map(|_| ()),
        }
    }
}

#[cfg(feature = "cpu-reference")]
#[derive(Clone)]
enum RuntimeDecoderModel {
    Cpu(CpuDecoderModel),
    #[cfg(all(feature = "metal", target_os = "macos"))]
    Metal(MetalDecoderModel),
}

#[cfg(feature = "cpu-reference")]
enum RuntimeDecoderSession {
    Cpu(CpuDecoderSession),
    #[cfg(all(feature = "metal", target_os = "macos"))]
    Metal(MetalDecoderSession),
}

#[cfg(feature = "cpu-reference")]
impl RuntimeDecoderModel {
    fn config(&self) -> &TransformerConfig {
        match self {
            Self::Cpu(model) => model.config(),
            #[cfg(all(feature = "metal", target_os = "macos"))]
            Self::Metal(model) => model.config(),
        }
    }

    fn memory_pool(&self) -> &MemoryPool {
        match self {
            Self::Cpu(model) => model.memory_pool(),
            #[cfg(all(feature = "metal", target_os = "macos"))]
            Self::Metal(model) => model.memory_pool(),
        }
    }

    fn start_session(
        &self,
        budget: &ResourceBudget,
        request_id: RequestId,
        sequence_id: SequenceId,
    ) -> Result<RuntimeDecoderSession, EngineError> {
        match self {
            Self::Cpu(model) => {
                model.start_session(budget, request_id, sequence_id).map(RuntimeDecoderSession::Cpu)
            }
            #[cfg(all(feature = "metal", target_os = "macos"))]
            Self::Metal(model) => model
                .start_session(budget, request_id, sequence_id)
                .map(RuntimeDecoderSession::Metal),
        }
    }
}

#[cfg(feature = "cpu-reference")]
impl RuntimeDecoderSession {
    fn forward_token(
        &mut self,
        token: u32,
    ) -> Result<personalai_engine_tensor::Tensor, EngineError> {
        match self {
            Self::Cpu(session) => session.forward_token(token),
            #[cfg(all(feature = "metal", target_os = "macos"))]
            Self::Metal(session) => session.decode(token),
        }
    }

    fn token_count(&self) -> u32 {
        match self {
            Self::Cpu(session) => session.token_count(),
            #[cfg(all(feature = "metal", target_os = "macos"))]
            Self::Metal(session) => session.token_count(),
        }
    }
}

#[cfg(all(feature = "cpu-reference", feature = "metal", target_os = "macos"))]
impl LoadedMetalModel {
    /// Returns the configuration-owned model identifier.
    pub const fn model_id(&self) -> &ModelId {
        &self.model_id
    }

    /// Returns the executable Metal decoder.
    pub const fn model(&self) -> &MetalDecoderModel {
        &self.model
    }

    /// Returns the package tokenizer.
    pub fn tokenizer(&self) -> &dyn Tokenizer {
        self.tokenizer.as_ref()
    }

    /// Creates one validated, resumable greedy Metal sequence for fair service scheduling.
    pub fn start_greedy(
        &self,
        request: &GenerationRequest,
        sequence_id: SequenceId,
    ) -> Result<GreedyGeneration, EngineError> {
        start_generation_cursor(
            &self.model_id,
            RuntimeDecoderModel::Metal(self.model.clone()),
            Arc::clone(&self.tokenizer),
            request,
            sequence_id,
            SamplingConfig::greedy(),
        )
    }

    /// Runs deterministic greedy generation through the Metal decoder.
    pub fn generate_greedy(
        &self,
        request: &GenerationRequest,
        sequence_id: SequenceId,
        cancelled: &AtomicBool,
        emit: impl FnMut(InferenceEvent) -> Result<(), EngineError>,
    ) -> Result<GenerationUsage, EngineError> {
        self.generate(request, sequence_id, SamplingConfig::greedy(), cancelled, emit)
    }

    /// Runs tokenize, Metal prefill/decode, sampling, and typed streaming.
    pub fn generate(
        &self,
        request: &GenerationRequest,
        sequence_id: SequenceId,
        sampling: SamplingConfig,
        cancelled: &AtomicBool,
        mut emit: impl FnMut(InferenceEvent) -> Result<(), EngineError>,
    ) -> Result<GenerationUsage, EngineError> {
        emit(InferenceEvent::RequestAccepted { request_id: request.request_id(), sequence_id })?;
        let result = self.generate_inner(request, sequence_id, sampling, cancelled, &mut emit);
        if let Err(error) = &result {
            let event = match error {
                EngineError::Cancellation(_) => InferenceEvent::RequestCancelled {
                    request_id: request.request_id(),
                    sequence_id,
                },
                EngineError::ResourceExhaustion(_) => InferenceEvent::ResourceLimitReached {
                    request_id: request.request_id(),
                    sequence_id,
                    error: error.clone(),
                },
                EngineError::BackendFailure(_) => InferenceEvent::BackendError {
                    request_id: request.request_id(),
                    sequence_id,
                    error: error.clone(),
                },
                _ => InferenceEvent::ModelError {
                    request_id: request.request_id(),
                    sequence_id,
                    error: error.clone(),
                },
            };
            emit(event)?;
        }
        result
    }

    #[allow(clippy::too_many_lines)]
    fn generate_inner(
        &self,
        request: &GenerationRequest,
        sequence_id: SequenceId,
        sampling: SamplingConfig,
        cancelled: &AtomicBool,
        emit: &mut impl FnMut(InferenceEvent) -> Result<(), EngineError>,
    ) -> Result<GenerationUsage, EngineError> {
        if request.model_id() != &self.model_id {
            return Err(EngineError::invalid_input(
                "generation request model differs from the loaded Metal model",
            ));
        }
        if request.context_limit().get() > self.model.config().context_limit().get() {
            return Err(EngineError::resource_exhaustion(
                "request context exceeds the loaded Metal model context",
            ));
        }
        check_cancelled(cancelled, request.request_id())?;
        let input_tokens = self.tokenizer.encode(request.input_text())?;
        let input_count = u32::try_from(input_tokens.len())
            .map_err(|_| EngineError::resource_exhaustion("input token count exceeds u32"))?;
        request.validate_tokenized_input(input_count)?;
        emit(InferenceEvent::PromptTokenized {
            request_id: request.request_id(),
            sequence_id,
            token_count: input_count,
        })?;
        emit(InferenceEvent::PrefillStarted { request_id: request.request_id(), sequence_id })?;
        let mut session = self.model.start_session(
            &request.resource_budget(),
            request.request_id(),
            sequence_id,
        )?;
        let cancellation = session.cancellation();
        let mut logits = session.prefill(&input_tokens)?;
        emit(InferenceEvent::PrefillCompleted { request_id: request.request_id(), sequence_id })?;
        let retained_tokens = input_tokens
            .len()
            .checked_add(usize::try_from(request.max_output_tokens()).map_err(|_| {
                EngineError::resource_exhaustion("output token count is not addressable")
            })?)
            .ok_or_else(|| EngineError::resource_exhaustion("generation history overflows"))?;
        let retained_bytes = retained_tokens
            .checked_mul(std::mem::size_of::<u32>())
            .and_then(|bytes| u64::try_from(bytes).ok())
            .ok_or_else(|| EngineError::resource_exhaustion("generation history bytes overflow"))?;
        let history_pool =
            self.model.memory_pool().scoped(MemoryOwner::Sampling(request.request_id()));
        let _history_reservation = history_pool.reserve(retained_bytes)?;
        let mut history = input_tokens;
        history
            .try_reserve_exact(usize::try_from(request.max_output_tokens()).map_err(|_| {
                EngineError::resource_exhaustion("output token count is not addressable")
            })?)
            .map_err(|_| {
                EngineError::resource_exhaustion("generation history allocation failed")
            })?;
        let mut sampler = Sampler::new(sampling, request.request_id(), self.model.memory_pool());
        let mut decoder = IncrementalDecoder::new_shared(Arc::clone(&self.tokenizer));
        let mut generated = 0_u32;
        loop {
            if cancelled.load(Ordering::Acquire) {
                cancellation.cancel();
                session.cancel()?;
                return Err(EngineError::cancellation(Some(request.request_id())));
            }
            let token = sampler.sample(&logits, &history)?;
            generated = generated
                .checked_add(1)
                .ok_or_else(|| EngineError::internal_invariant("generated count overflows"))?;
            history.push(token);
            let text = decoder.push(token)?;
            emit(InferenceEvent::TokenGenerated {
                request_id: request.request_id(),
                sequence_id,
                token_id: TokenId::new(token)?,
                text: TextChunk::new(text)?,
            })?;
            let stop = if token == self.tokenizer.spec().end_of_sequence_token() {
                Some(GenerationStopReason::EndOfSequence)
            } else if sampler.config().is_stop_token(token) {
                Some(GenerationStopReason::ConfiguredToken)
            } else if generated >= request.max_output_tokens()
                || generated >= sampler.config().maximum_generated_tokens()
            {
                Some(GenerationStopReason::MaximumGeneratedTokens)
            } else {
                None
            };
            if let Some(reason) = stop {
                if decoder.has_pending_bytes() {
                    return Err(EngineError::backend_failure(
                        "generation ended with incomplete UTF-8",
                    ));
                }
                emit(InferenceEvent::StopConditionReached {
                    request_id: request.request_id(),
                    sequence_id,
                    reason,
                })?;
                let usage = GenerationUsage::for_request(request, input_count, generated)?;
                emit(InferenceEvent::RequestCompleted {
                    request_id: request.request_id(),
                    sequence_id,
                    usage,
                })?;
                return Ok(usage);
            }
            logits = session.decode(token)?;
        }
    }
}

#[cfg(feature = "cpu-reference")]
impl LoadedCpuModel {
    /// Returns the configuration-owned model identifier.
    pub const fn model_id(&self) -> &ModelId {
        &self.model_id
    }

    /// Returns the executable CPU decoder.
    pub const fn model(&self) -> &CpuDecoderModel {
        &self.model
    }

    /// Returns the package tokenizer.
    pub fn tokenizer(&self) -> &dyn Tokenizer {
        self.tokenizer.as_ref()
    }

    /// Creates one validated, resumable greedy-generation sequence.
    ///
    /// Each call to [`GreedyGeneration::advance`] performs at most one decoder
    /// token step. This is the scheduling boundary used to interleave sequences
    /// without sharing their KV caches or cancellation state.
    pub fn start_greedy(
        &self,
        request: &GenerationRequest,
        sequence_id: SequenceId,
    ) -> Result<GreedyGeneration, EngineError> {
        self.start_generation(request, sequence_id, SamplingConfig::greedy())
    }

    /// Creates one validated, resumable generation sequence with first-party sampling.
    pub fn start_generation(
        &self,
        request: &GenerationRequest,
        sequence_id: SequenceId,
        sampling: SamplingConfig,
    ) -> Result<GreedyGeneration, EngineError> {
        start_generation_cursor(
            &self.model_id,
            RuntimeDecoderModel::Cpu(self.model.clone()),
            Arc::clone(&self.tokenizer),
            request,
            sequence_id,
            sampling,
        )
    }

    /// Runs deterministic greedy generation and synchronously emits bounded events.
    pub fn generate_greedy(
        &self,
        request: &GenerationRequest,
        sequence_id: SequenceId,
        cancelled: &AtomicBool,
        emit: impl FnMut(GenerationEvent) -> Result<(), EngineError>,
    ) -> Result<GenerationUsage, EngineError> {
        self.generate_greedy_measured(request, sequence_id, cancelled, emit)
            .map(|measurement| measurement.usage)
    }

    fn generate_greedy_measured(
        &self,
        request: &GenerationRequest,
        sequence_id: SequenceId,
        cancelled: &AtomicBool,
        mut emit: impl FnMut(GenerationEvent) -> Result<(), EngineError>,
    ) -> Result<GenerationMeasurement, EngineError> {
        let mut generation = self.start_greedy(request, sequence_id)?;
        loop {
            match generation.advance(cancelled)? {
                GreedyAdvance::Progressed => {}
                GreedyAdvance::Event(event) => {
                    let completed = match &event {
                        GenerationEvent::Completed { usage, .. } => Some(*usage),
                        _ => None,
                    };
                    emit(event)?;
                    if let Some(usage) = completed {
                        return Ok(GenerationMeasurement { usage, timing: generation.timing() });
                    }
                }
            }
        }
    }

    /// Runs complete first-party prefill, sampling, cached decode, and typed streaming.
    pub fn generate(
        &self,
        request: &GenerationRequest,
        sequence_id: SequenceId,
        sampling: SamplingConfig,
        cancelled: &AtomicBool,
        mut emit: impl FnMut(InferenceEvent) -> Result<(), EngineError>,
    ) -> Result<GenerationUsage, EngineError> {
        if request.model_id() != &self.model_id {
            let error = EngineError::invalid_input(
                "generation request model differs from the loaded model",
            );
            emit(InferenceEvent::ModelError {
                request_id: request.request_id(),
                sequence_id,
                error: error.clone(),
            })?;
            return Err(error);
        }
        if request.context_limit().get() > self.model.config().context_limit().get() {
            let error = EngineError::resource_exhaustion(
                "request context exceeds the loaded model context",
            );
            emit(InferenceEvent::ResourceLimitReached {
                request_id: request.request_id(),
                sequence_id,
                error: error.clone(),
            })?;
            return Err(error);
        }
        emit(InferenceEvent::RequestAccepted { request_id: request.request_id(), sequence_id })?;
        let result = self.generate_inner(request, sequence_id, sampling, cancelled, &mut emit);
        if let Err(error) = &result {
            let event = match error {
                EngineError::Cancellation(_) => InferenceEvent::RequestCancelled {
                    request_id: request.request_id(),
                    sequence_id,
                },
                EngineError::ResourceExhaustion(_) => InferenceEvent::ResourceLimitReached {
                    request_id: request.request_id(),
                    sequence_id,
                    error: error.clone(),
                },
                EngineError::BackendFailure(_) => InferenceEvent::BackendError {
                    request_id: request.request_id(),
                    sequence_id,
                    error: error.clone(),
                },
                _ => InferenceEvent::ModelError {
                    request_id: request.request_id(),
                    sequence_id,
                    error: error.clone(),
                },
            };
            emit(event)?;
        }
        result
    }

    fn generate_inner(
        &self,
        request: &GenerationRequest,
        sequence_id: SequenceId,
        sampling: SamplingConfig,
        cancelled: &AtomicBool,
        emit: &mut impl FnMut(InferenceEvent) -> Result<(), EngineError>,
    ) -> Result<GenerationUsage, EngineError> {
        if request.model_id() != &self.model_id {
            return Err(EngineError::invalid_input(
                "generation request model differs from the loaded model",
            ));
        }
        if request.context_limit().get() > self.model.config().context_limit().get() {
            return Err(EngineError::resource_exhaustion(
                "request context exceeds the loaded model context",
            ));
        }
        check_cancelled(cancelled, request.request_id())?;
        let input_tokens = self.tokenizer.encode(request.input_text())?;
        let input_count = u32::try_from(input_tokens.len())
            .map_err(|_| EngineError::resource_exhaustion("input token count exceeds u32"))?;
        request.validate_tokenized_input(input_count)?;
        emit(InferenceEvent::PromptTokenized {
            request_id: request.request_id(),
            sequence_id,
            token_count: input_count,
        })?;
        emit(InferenceEvent::PrefillStarted { request_id: request.request_id(), sequence_id })?;
        let mut session = self.model.start_session(
            &request.resource_budget(),
            request.request_id(),
            sequence_id,
        )?;
        let mut logits = session.prefill(&input_tokens)?;
        emit(InferenceEvent::PrefillCompleted { request_id: request.request_id(), sequence_id })?;
        let retained_tokens = input_tokens
            .len()
            .checked_add(usize::try_from(request.max_output_tokens()).map_err(|_| {
                EngineError::resource_exhaustion("output token count is not addressable")
            })?)
            .ok_or_else(|| EngineError::resource_exhaustion("generation history overflows"))?;
        let retained_bytes = retained_tokens
            .checked_mul(std::mem::size_of::<u32>())
            .and_then(|bytes| u64::try_from(bytes).ok())
            .ok_or_else(|| EngineError::resource_exhaustion("generation history bytes overflow"))?;
        let history_pool =
            self.model.memory_pool().scoped(MemoryOwner::Sampling(request.request_id()));
        let _history_reservation = history_pool.reserve(retained_bytes)?;
        let mut history = input_tokens;
        history
            .try_reserve_exact(usize::try_from(request.max_output_tokens()).map_err(|_| {
                EngineError::resource_exhaustion("output token count is not addressable")
            })?)
            .map_err(|_| {
                EngineError::resource_exhaustion("generation history allocation failed")
            })?;
        let mut sampler = Sampler::new(sampling, request.request_id(), self.model.memory_pool());
        let mut decoder = IncrementalDecoder::new_shared(Arc::clone(&self.tokenizer));
        let mut generated = 0_u32;
        loop {
            check_cancelled(cancelled, request.request_id())?;
            let token = sampler.sample(&logits, &history)?;
            generated = generated
                .checked_add(1)
                .ok_or_else(|| EngineError::internal_invariant("generated count overflows"))?;
            history.push(token);
            let text = decoder.push(token)?;
            emit(InferenceEvent::TokenGenerated {
                request_id: request.request_id(),
                sequence_id,
                token_id: TokenId::new(token)?,
                text: TextChunk::new(text)?,
            })?;
            let stop = if token == self.tokenizer.spec().end_of_sequence_token() {
                Some(GenerationStopReason::EndOfSequence)
            } else if sampler.config().is_stop_token(token) {
                Some(GenerationStopReason::ConfiguredToken)
            } else if generated >= request.max_output_tokens()
                || generated >= sampler.config().maximum_generated_tokens()
            {
                Some(GenerationStopReason::MaximumGeneratedTokens)
            } else {
                None
            };
            if let Some(reason) = stop {
                if decoder.has_pending_bytes() {
                    return Err(EngineError::backend_failure(
                        "generation ended with incomplete UTF-8",
                    ));
                }
                emit(InferenceEvent::StopConditionReached {
                    request_id: request.request_id(),
                    sequence_id,
                    reason,
                })?;
                let usage = GenerationUsage::for_request(request, input_count, generated)?;
                emit(InferenceEvent::RequestCompleted {
                    request_id: request.request_id(),
                    sequence_id,
                    usage,
                })?;
                return Ok(usage);
            }
            logits = session.decode(token)?;
        }
    }
}

/// First-party generation stop classification.
#[cfg(feature = "cpu-reference")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GenerationStopReason {
    /// The tokenizer end-of-sequence ID was produced.
    EndOfSequence,
    /// A caller-configured stop ID was produced.
    ConfiguredToken,
    /// The request or sampler output ceiling was reached.
    MaximumGeneratedTokens,
}

/// Rich internal generation lifecycle; the stable Go wire protocol remains separate.
#[cfg(feature = "cpu-reference")]
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum InferenceEvent {
    /// The request passed initial routing validation.
    RequestAccepted {
        /// Request correlation identifier.
        request_id: RequestId,
        /// Assigned sequence identifier.
        sequence_id: SequenceId,
    },
    /// Prompt tokenization completed.
    PromptTokenized {
        /// Request correlation identifier.
        request_id: RequestId,
        /// Assigned sequence identifier.
        sequence_id: SequenceId,
        /// Validated prompt-token count.
        token_count: u32,
    },
    /// Prompt prefill began.
    PrefillStarted {
        /// Request correlation identifier.
        request_id: RequestId,
        /// Assigned sequence identifier.
        sequence_id: SequenceId,
    },
    /// Prompt prefill completed.
    PrefillCompleted {
        /// Request correlation identifier.
        request_id: RequestId,
        /// Assigned sequence identifier.
        sequence_id: SequenceId,
    },
    /// One token and safe UTF-8 chunk were produced.
    TokenGenerated {
        /// Request correlation identifier.
        request_id: RequestId,
        /// Assigned sequence identifier.
        sequence_id: SequenceId,
        /// Validated generated token.
        token_id: TokenId,
        /// Bounded UTF-8 output fragment.
        text: TextChunk,
    },
    /// A validated stop condition was reached.
    StopConditionReached {
        /// Request correlation identifier.
        request_id: RequestId,
        /// Assigned sequence identifier.
        sequence_id: SequenceId,
        /// Validated stop classification.
        reason: GenerationStopReason,
    },
    /// Request processing completed.
    RequestCompleted {
        /// Request correlation identifier.
        request_id: RequestId,
        /// Assigned sequence identifier.
        sequence_id: SequenceId,
        /// Checked token usage.
        usage: GenerationUsage,
    },
    /// Cancellation was observed.
    RequestCancelled {
        /// Request correlation identifier.
        request_id: RequestId,
        /// Assigned sequence identifier.
        sequence_id: SequenceId,
    },
    /// A finite resource limit prevented progress.
    ResourceLimitReached {
        /// Request correlation identifier.
        request_id: RequestId,
        /// Assigned sequence identifier.
        sequence_id: SequenceId,
        /// Bounded structured failure.
        error: EngineError,
    },
    /// Model, tokenizer, or request validation failed.
    ModelError {
        /// Request correlation identifier.
        request_id: RequestId,
        /// Assigned sequence identifier.
        sequence_id: SequenceId,
        /// Bounded structured failure.
        error: EngineError,
    },
    /// The selected compute backend failed.
    BackendError {
        /// Request correlation identifier.
        request_id: RequestId,
        /// Assigned sequence identifier.
        sequence_id: SequenceId,
        /// Bounded structured failure.
        error: EngineError,
    },
}

/// Result of one bounded greedy-generation scheduling quantum.
#[cfg(feature = "cpu-reference")]
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum GreedyAdvance {
    /// Bounded internal progress completed without a client-visible event.
    ///
    /// This is either request-local session allocation or one prompt-token
    /// decoder step.
    Progressed,
    /// One bounded generation lifecycle event is ready for delivery.
    Event(GenerationEvent),
}

#[cfg(feature = "cpu-reference")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TerminalState {
    Complete,
    IncompleteUtf8AtEndOfSequence,
    IncompleteUtf8AtOutputLimit,
}

/// Mutable state for one independently scheduled greedy-generation sequence.
///
/// The cursor owns no authority and performs no I/O. It borrows one immutable
/// loaded model, owns a request-local decoder session/KV cache, and yields after
/// at most one model token step so an engine-owned scheduler can rotate fairly
/// among active sequences.
#[cfg(feature = "cpu-reference")]
pub struct GreedyGeneration {
    request: GenerationRequest,
    sequence_id: SequenceId,
    model: RuntimeDecoderModel,
    tokenizer: Arc<dyn Tokenizer>,
    session: Option<RuntimeDecoderSession>,
    input_tokens: Vec<u32>,
    input_count: u32,
    history: Vec<u32>,
    sampler: Sampler,
    next_input: usize,
    next_generated_token: Option<u32>,
    generated: u32,
    stream_decoder: IncrementalDecoder<'static>,
    started: bool,
    terminal: Option<TerminalState>,
    usage: Option<GenerationUsage>,
    prefill_nanoseconds: u64,
    decode_nanoseconds: u64,
}

#[cfg(feature = "cpu-reference")]
fn start_generation_cursor(
    model_id: &ModelId,
    model: RuntimeDecoderModel,
    tokenizer: Arc<dyn Tokenizer>,
    request: &GenerationRequest,
    sequence_id: SequenceId,
    sampling: SamplingConfig,
) -> Result<GreedyGeneration, EngineError> {
    if request.model_id() != model_id {
        return Err(EngineError::invalid_input(
            "generation request model differs from the loaded model",
        ));
    }
    if request.context_limit().get() > model.config().context_limit().get() {
        return Err(EngineError::resource_exhaustion(
            "request context exceeds the loaded model context",
        ));
    }
    let input_tokens = tokenizer.encode(request.input_text())?;
    let input_count = u32::try_from(input_tokens.len())
        .map_err(|_| EngineError::resource_exhaustion("input token count exceeds u32"))?;
    request.validate_tokenized_input(input_count)?;
    let sampler = Sampler::new(sampling, request.request_id(), model.memory_pool());
    let history = input_tokens.clone();
    Ok(GreedyGeneration {
        request: request.clone(),
        sequence_id,
        model,
        tokenizer: Arc::clone(&tokenizer),
        session: None,
        input_tokens,
        input_count,
        history,
        sampler,
        next_input: 0,
        next_generated_token: None,
        generated: 0,
        stream_decoder: IncrementalDecoder::new_shared(tokenizer),
        started: false,
        terminal: None,
        usage: None,
        prefill_nanoseconds: 0,
        decode_nanoseconds: 0,
    })
}

#[cfg(feature = "cpu-reference")]
impl GreedyGeneration {
    /// Advances this sequence by at most one decoder token step.
    pub fn advance(&mut self, cancelled: &AtomicBool) -> Result<GreedyAdvance, EngineError> {
        if self.usage.is_some() {
            return Err(EngineError::protocol_violation(
                "completed generation sequence cannot advance",
            ));
        }
        if let Some(terminal) = self.terminal.take() {
            match terminal {
                TerminalState::IncompleteUtf8AtEndOfSequence => {
                    return Err(EngineError::backend_failure(
                        "generation ended with an incomplete UTF-8 sequence",
                    ));
                }
                TerminalState::IncompleteUtf8AtOutputLimit => {
                    return Err(EngineError::backend_failure(
                        "generation limit ended with an incomplete UTF-8 sequence",
                    ));
                }
                TerminalState::Complete => {}
            }
            let usage =
                GenerationUsage::for_request(&self.request, self.input_count, self.generated)?;
            self.usage = Some(usage);
            return Ok(GreedyAdvance::Event(GenerationEvent::Completed {
                request_id: self.request.request_id(),
                sequence_id: self.sequence_id,
                usage,
            }));
        }
        if !self.started {
            self.started = true;
            return Ok(GreedyAdvance::Event(GenerationEvent::Started {
                request_id: self.request.request_id(),
                sequence_id: self.sequence_id,
            }));
        }
        check_cancelled(cancelled, self.request.request_id())?;
        if self.session.is_none() {
            self.session = Some(self.model.start_session(
                &self.request.resource_budget(),
                self.request.request_id(),
                self.sequence_id,
            )?);
            return Ok(GreedyAdvance::Progressed);
        }
        let session = self.session.as_mut().ok_or_else(|| {
            EngineError::internal_invariant("generation decoder session disappeared")
        })?;
        let logits = if let Some(token) = self.input_tokens.get(self.next_input).copied() {
            let started = Instant::now();
            let logits = session.forward_token(token)?;
            add_timing(&mut self.prefill_nanoseconds, started.elapsed())?;
            self.next_input = self.next_input.checked_add(1).ok_or_else(|| {
                EngineError::internal_invariant("generation input position overflows")
            })?;
            if self.next_input < self.input_tokens.len() {
                return Ok(GreedyAdvance::Progressed);
            }
            logits
        } else if let Some(token) = self.next_generated_token.take() {
            let started = Instant::now();
            let logits = session.forward_token(token)?;
            add_timing(&mut self.decode_nanoseconds, started.elapsed())?;
            logits
        } else {
            return Err(EngineError::internal_invariant("tokenizer produced no prompt tokens"));
        };
        let token = self.sampler.sample(&logits, &self.history)?;
        self.generated = self
            .generated
            .checked_add(1)
            .ok_or_else(|| EngineError::internal_invariant("generated token count overflows"))?;
        let text = self.stream_decoder.push(token)?;
        self.history.push(token);
        let end_of_sequence = token == self.tokenizer.spec().end_of_sequence_token()
            || self.sampler.config().is_stop_token(token);
        let output_limit = self.generated >= self.request.max_output_tokens()
            || self.generated >= self.sampler.config().maximum_generated_tokens();
        if end_of_sequence || output_limit {
            self.terminal = Some(if !self.stream_decoder.has_pending_bytes() {
                TerminalState::Complete
            } else if end_of_sequence {
                TerminalState::IncompleteUtf8AtEndOfSequence
            } else {
                TerminalState::IncompleteUtf8AtOutputLimit
            });
        } else {
            self.next_generated_token = Some(token);
        }
        Ok(GreedyAdvance::Event(GenerationEvent::Token {
            request_id: self.request.request_id(),
            sequence_id: self.sequence_id,
            token_id: TokenId::new(token)?,
            text: TextChunk::new(text)?,
        }))
    }

    /// Returns whether this sequence emitted its completed event.
    pub const fn is_complete(&self) -> bool {
        self.usage.is_some()
    }

    /// Returns final usage after completion.
    pub const fn usage(&self) -> Option<GenerationUsage> {
        self.usage
    }

    /// Returns the request correlation identifier.
    pub const fn request_id(&self) -> RequestId {
        self.request.request_id()
    }

    /// Returns the engine-assigned sequence identifier.
    pub const fn sequence_id(&self) -> SequenceId {
        self.sequence_id
    }

    /// Returns decoder tokens committed to this request-local KV cache.
    pub fn processed_tokens(&self) -> u32 {
        self.session.as_ref().map_or(0, RuntimeDecoderSession::token_count)
    }

    /// Returns monotonic CPU forward time split between prompt and decode work.
    pub const fn timing(&self) -> GenerationTiming {
        GenerationTiming {
            prefill_nanoseconds: self.prefill_nanoseconds,
            decode_nanoseconds: self.decode_nanoseconds,
        }
    }

    const fn concurrent_sequence_limit(&self) -> u32 {
        self.request.resource_budget().max_concurrent_sequences()
    }
}

/// Monotonic scalar decoder timing with no prompt or generated content.
#[cfg(feature = "cpu-reference")]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GenerationTiming {
    prefill_nanoseconds: u64,
    decode_nanoseconds: u64,
}

#[cfg(feature = "cpu-reference")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct GenerationMeasurement {
    usage: GenerationUsage,
    timing: GenerationTiming,
}

#[cfg(feature = "cpu-reference")]
impl GenerationTiming {
    /// Returns accumulated prompt-token forward time.
    pub const fn prefill_nanoseconds(self) -> u64 {
        self.prefill_nanoseconds
    }

    /// Returns accumulated generated-token decode forward time.
    pub const fn decode_nanoseconds(self) -> u64 {
        self.decode_nanoseconds
    }
}

#[cfg(feature = "cpu-reference")]
fn duration_nanoseconds(duration: std::time::Duration) -> Result<u64, EngineError> {
    u64::try_from(duration.as_nanos())
        .map_err(|_| EngineError::resource_exhaustion("generation duration exceeds u64"))
}

#[cfg(feature = "cpu-reference")]
fn add_timing(total: &mut u64, duration: std::time::Duration) -> Result<(), EngineError> {
    *total = total
        .checked_add(duration_nanoseconds(duration)?)
        .ok_or_else(|| EngineError::resource_exhaustion("generation timing overflows"))?;
    Ok(())
}

#[cfg(feature = "cpu-reference")]
fn generation_retained_bytes(request: &GenerationRequest) -> Result<usize, EngineError> {
    const TOKENIZER_SPECIAL_TOKENS: usize = 2;
    const COORDINATOR_OVERHEAD_BYTES: usize = 4 * 1_024;

    let request_copies = request.estimated_size_bytes().checked_mul(3).ok_or_else(|| {
        EngineError::resource_exhaustion("active generation request bytes overflow")
    })?;
    let token_slots = request
        .input_text()
        .len()
        .checked_add(TOKENIZER_SPECIAL_TOKENS)
        .ok_or_else(|| EngineError::resource_exhaustion("active token slots overflow"))?;
    let token_bytes = token_slots
        .checked_mul(std::mem::size_of::<u32>())
        .ok_or_else(|| EngineError::resource_exhaustion("active input token bytes overflow"))?;
    request_copies
        .checked_add(token_bytes)
        .and_then(|bytes| bytes.checked_add(COORDINATOR_OVERHEAD_BYTES))
        .ok_or_else(|| EngineError::resource_exhaustion("active retained bytes overflow"))
}

/// One bounded turn completed by [`GenerationRoundRobin`].
#[cfg(feature = "cpu-reference")]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ScheduledGenerationAdvance {
    request_id: RequestId,
    sequence_id: SequenceId,
    decoder_tokens_advanced: u32,
    timing: GenerationTiming,
    event: Option<GenerationEvent>,
}

#[cfg(feature = "cpu-reference")]
impl ScheduledGenerationAdvance {
    /// Returns the request correlation identifier for this turn.
    pub const fn request_id(&self) -> RequestId {
        self.request_id
    }

    /// Returns the engine-assigned sequence identifier for this turn.
    pub const fn sequence_id(&self) -> SequenceId {
        self.sequence_id
    }

    /// Returns decoder tokens committed during this turn; always zero or one.
    pub const fn decoder_tokens_advanced(&self) -> u32 {
        self.decoder_tokens_advanced
    }

    /// Returns CPU forward time accumulated during only this scheduling turn.
    pub const fn timing(&self) -> GenerationTiming {
        self.timing
    }

    /// Returns the event that must be delivered before requesting another turn.
    pub const fn event(&self) -> Option<&GenerationEvent> {
        self.event.as_ref()
    }

    /// Consumes the turn and returns its optional event.
    pub fn into_event(self) -> Option<GenerationEvent> {
        self.event
    }
}

#[cfg(feature = "cpu-reference")]
struct ScheduledGeneration {
    generation: GreedyGeneration,
    cancelled: Arc<AtomicBool>,
    retained_bytes: usize,
}

/// Bounded deterministic round-robin coordinator for active CPU sequences.
///
/// The coordinator buffers no events: callers receive at most one event from
/// each turn and must deliver or reject it before advancing again. Cursor
/// failures become request-local `Failed` events and do not stop neighboring
/// sequences. This type performs no socket I/O and grants no authority.
#[cfg(feature = "cpu-reference")]
pub struct GenerationRoundRobin {
    maximum_active: u32,
    maximum_retained_bytes: usize,
    retained_bytes: usize,
    ready: VecDeque<ScheduledGeneration>,
}

#[cfg(feature = "cpu-reference")]
impl GenerationRoundRobin {
    /// Creates an empty coordinator with a hard active-sequence ceiling.
    pub fn new(maximum_active: u32) -> Result<Self, EngineError> {
        Self::with_limits(maximum_active, MAX_QUEUED_REQUEST_BYTES)
    }

    /// Creates an empty coordinator with active-count and retained-byte ceilings.
    pub fn with_limits(
        maximum_active: u32,
        maximum_retained_bytes: usize,
    ) -> Result<Self, EngineError> {
        if maximum_active == 0 || maximum_active > MAX_CONCURRENT_SEQUENCES {
            return Err(EngineError::invalid_input(
                "round-robin active limit is outside hard bounds",
            ));
        }
        if maximum_retained_bytes == 0 || maximum_retained_bytes > MAX_QUEUED_REQUEST_BYTES {
            return Err(EngineError::invalid_input(
                "round-robin retained-byte limit is outside hard bounds",
            ));
        }
        Ok(Self {
            maximum_active,
            maximum_retained_bytes,
            retained_bytes: 0,
            ready: VecDeque::new(),
        })
    }

    /// Admits one request and creates its cursor only after capacity checks.
    pub fn submit(
        &mut self,
        model: &LoadedCpuModel,
        request: &GenerationRequest,
        sequence_id: SequenceId,
        cancelled: Arc<AtomicBool>,
    ) -> Result<(), EngineError> {
        self.submit_with_retained_bytes(model, request, sequence_id, cancelled, 0)
    }

    /// Admits one request while reserving caller-owned bounded runtime storage.
    ///
    /// The server uses this to charge a stream's fixed outbound queue before
    /// cursor construction. The coordinator does not own or inspect that
    /// storage; it only includes the declared bytes in its aggregate ceiling.
    pub fn submit_with_retained_bytes(
        &mut self,
        model: &LoadedCpuModel,
        request: &GenerationRequest,
        sequence_id: SequenceId,
        cancelled: Arc<AtomicBool>,
        additional_retained_bytes: usize,
    ) -> Result<(), EngineError> {
        self.submit_generation(
            || model.start_greedy(request, sequence_id),
            request,
            sequence_id,
            cancelled,
            additional_retained_bytes,
        )
    }

    fn submit_runtime(
        &mut self,
        model: &LoadedRuntimeModel,
        request: &GenerationRequest,
        sequence_id: SequenceId,
        cancelled: Arc<AtomicBool>,
        additional_retained_bytes: usize,
    ) -> Result<(), EngineError> {
        self.submit_generation(
            || model.start_greedy(request, sequence_id),
            request,
            sequence_id,
            cancelled,
            additional_retained_bytes,
        )
    }

    fn submit_generation(
        &mut self,
        start_generation: impl FnOnce() -> Result<GreedyGeneration, EngineError>,
        request: &GenerationRequest,
        sequence_id: SequenceId,
        cancelled: Arc<AtomicBool>,
        additional_retained_bytes: usize,
    ) -> Result<(), EngineError> {
        if self.ready.iter().any(|scheduled| {
            scheduled.generation.request_id() == request.request_id()
                || scheduled.generation.sequence_id() == sequence_id
        }) {
            return Err(EngineError::protocol_violation(
                "round-robin request or sequence identifier is already active",
            ));
        }
        let current = u32::try_from(self.ready.len())
            .map_err(|_| EngineError::internal_invariant("round-robin active count exceeds u32"))?;
        let next = current.checked_add(1).ok_or_else(|| {
            EngineError::resource_exhaustion("round-robin active count overflows")
        })?;
        if next > self.maximum_active {
            return Err(EngineError::resource_exhaustion(
                "round-robin active sequence limit is reached",
            ));
        }
        let budget_limit = self
            .ready
            .iter()
            .map(|scheduled| scheduled.generation.concurrent_sequence_limit())
            .chain(std::iter::once(request.resource_budget().max_concurrent_sequences()))
            .min()
            .ok_or_else(|| {
                EngineError::internal_invariant("round-robin admission budget disappeared")
            })?;
        if next > budget_limit {
            return Err(EngineError::resource_exhaustion(
                "request concurrency budget does not admit another active sequence",
            ));
        }
        let retained_bytes =
            generation_retained_bytes(request)?.checked_add(additional_retained_bytes).ok_or_else(
                || EngineError::resource_exhaustion("round-robin retained bytes overflow"),
            )?;
        let next_retained_bytes =
            self.retained_bytes.checked_add(retained_bytes).ok_or_else(|| {
                EngineError::resource_exhaustion("round-robin retained bytes overflow")
            })?;
        if next_retained_bytes > self.maximum_retained_bytes {
            return Err(EngineError::resource_exhaustion(
                "round-robin retained-byte limit is reached",
            ));
        }
        self.ready
            .try_reserve(1)
            .map_err(|_| EngineError::resource_exhaustion("round-robin queue allocation failed"))?;
        let generation = start_generation()?;
        self.ready.push_back(ScheduledGeneration { generation, cancelled, retained_bytes });
        self.retained_bytes = next_retained_bytes;
        Ok(())
    }

    /// Advances the next active sequence once, then rotates it to the back.
    ///
    /// `Ok(None)` means no sequence is active. A completed or failed sequence is
    /// removed before this method returns.
    pub fn advance_next(&mut self) -> Result<Option<ScheduledGenerationAdvance>, EngineError> {
        let Some(mut scheduled) = self.ready.pop_front() else {
            return Ok(None);
        };
        self.retained_bytes =
            self.retained_bytes.checked_sub(scheduled.retained_bytes).ok_or_else(|| {
                EngineError::internal_invariant("round-robin retained bytes underflow")
            })?;
        let request_id = scheduled.generation.request_id();
        let sequence_id = scheduled.generation.sequence_id();
        let before = scheduled.generation.processed_tokens();
        let timing_before = scheduled.generation.timing();
        let advance = scheduled.generation.advance(&scheduled.cancelled);
        let after = scheduled.generation.processed_tokens();
        let timing_after = scheduled.generation.timing();
        let decoder_tokens_advanced = after.checked_sub(before).ok_or_else(|| {
            EngineError::internal_invariant("round-robin decoder token count decreased")
        })?;
        if decoder_tokens_advanced > 1 {
            return Err(EngineError::internal_invariant(
                "round-robin turn advanced more than one decoder token",
            ));
        }
        let (event, retain) = match advance {
            Ok(GreedyAdvance::Progressed) => (None, true),
            Ok(GreedyAdvance::Event(event)) => {
                let retain = !scheduled.generation.is_complete();
                (Some(event), retain)
            }
            Err(error) => (
                Some(GenerationEvent::Failed { request_id, sequence_id: Some(sequence_id), error }),
                false,
            ),
        };
        if retain {
            self.retained_bytes =
                self.retained_bytes.checked_add(scheduled.retained_bytes).ok_or_else(|| {
                    EngineError::internal_invariant("round-robin retained bytes overflow")
                })?;
            self.ready.push_back(scheduled);
        }
        Ok(Some(ScheduledGenerationAdvance {
            request_id,
            sequence_id,
            decoder_tokens_advanced,
            timing: GenerationTiming {
                prefill_nanoseconds: timing_after
                    .prefill_nanoseconds
                    .checked_sub(timing_before.prefill_nanoseconds)
                    .ok_or_else(|| {
                        EngineError::internal_invariant("generation prefill timing decreased")
                    })?,
                decode_nanoseconds: timing_after
                    .decode_nanoseconds
                    .checked_sub(timing_before.decode_nanoseconds)
                    .ok_or_else(|| {
                        EngineError::internal_invariant("generation decode timing decreased")
                    })?,
            },
            event,
        }))
    }

    /// Returns the number of active cursors.
    pub fn active_len(&self) -> usize {
        self.ready.len()
    }

    /// Returns whether no cursor remains active.
    pub fn is_empty(&self) -> bool {
        self.ready.is_empty()
    }

    /// Returns conservative bytes retained for active requests and token buffers.
    pub const fn retained_bytes(&self) -> usize {
        self.retained_bytes
    }
}

/// Result of processing one authenticated local control-plane command.
#[cfg(all(unix, feature = "cpu-reference"))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ServeOutcome {
    /// One generation request completed successfully.
    Generated {
        /// Request that completed.
        request_id: RequestId,
        /// Sequence assigned by the engine.
        sequence_id: SequenceId,
        /// Validated usage emitted to the control plane.
        usage: GenerationUsage,
    },
    /// One queued request was cancelled before execution.
    CancelledQueued(RequestId),
    /// One active request was marked for cancellation.
    CancelledActive(RequestId),
    /// Cancellation named no queued or active request.
    CancellationNotFound(RequestId),
    /// One authenticated service-control request received a signed response.
    ServiceResponded {
        /// Correlation identifier from the service header.
        request_id: RequestId,
        /// Exact response category.
        kind: ServiceMessageKind,
    },
}

/// Content-free cumulative and live runtime metrics.
#[cfg(all(unix, feature = "cpu-reference"))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RuntimeMetrics {
    /// Currently loaded model count.
    pub loaded_models: u64,
    /// Currently active generation count.
    pub active_requests: u64,
    /// Currently queued generation count.
    pub queued_requests: u64,
    /// Requests rejected after admission began.
    pub rejected_requests: u64,
    /// Requests cancelled before normal completion.
    pub cancelled_requests: u64,
    /// Successfully processed prompt tokens.
    pub prompt_tokens: u64,
    /// Successfully generated tokens.
    pub generated_tokens: u64,
    /// Accumulated prefill nanoseconds when separately measurable.
    pub prefill_nanoseconds: u64,
    /// Accumulated decode nanoseconds when separately measurable.
    pub decode_nanoseconds: u64,
    /// Generated tokens per second multiplied by 1,000 over measured backend forward time.
    pub tokens_per_second_milli: u64,
    /// Reserved host bytes across loaded model pools.
    pub host_memory_bytes: u64,
    /// Reserved device bytes; zero for the CPU reference backend.
    pub device_memory_bytes: u64,
    /// Reserved KV-cache bytes observable from active sequences.
    pub kv_cache_bytes: u64,
    /// Package verification or validation failures.
    pub package_validation_failures: u64,
    /// Local peer or message authentication failures observed by this runtime.
    pub authentication_failures: u64,
    /// Bounded protocol violations observed by this runtime.
    pub protocol_violations: u64,
    /// Backend failures observed by this runtime.
    pub backend_failures: u64,
}

/// Fail-closed service health classification.
#[cfg(all(unix, feature = "cpu-reference"))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ServiceHealth {
    /// Service is accepting work and all loaded models are healthy.
    Ready,
    /// Service is running but has no loaded model.
    Degraded,
    /// An administrative drain has begun.
    Draining,
}

/// Per-connection result returned by one bounded concurrent serving batch.
#[cfg(all(unix, feature = "cpu-reference"))]
pub type ConnectionServeResult = Result<ServeOutcome, EngineError>;

#[cfg(all(unix, feature = "cpu-reference"))]
struct AuthenticatedConnection {
    input_index: usize,
    connection: UnixFrameConnection,
    session: AuthenticatedControlSession,
    replay_journal: Arc<Mutex<DurableReplayJournal>>,
    command: InboundCommand,
}

#[cfg(all(unix, feature = "cpu-reference"))]
struct ActiveGenerationConnection {
    input_index: usize,
    request_id: RequestId,
    connection: UnixFrameConnection,
    session: Option<AuthenticatedControlSession>,
    replay_journal: Arc<Mutex<DurableReplayJournal>>,
    request_nonce: u64,
    sequence_id: SequenceId,
    next_event_ordinal: u32,
    cancelled: Arc<AtomicBool>,
    watcher_done: Arc<AtomicBool>,
    outbound_sender: Option<SyncSender<Frame>>,
    outbound_receiver: Option<Receiver<Frame>>,
    writer_connection: Option<UnixFrameConnection>,
    writer_failed: Arc<AtomicBool>,
    stream_error: Option<EngineError>,
}

#[cfg(all(unix, feature = "cpu-reference"))]
impl ActiveGenerationConnection {
    const fn request_id(&self) -> RequestId {
        self.request_id
    }
}

#[cfg(all(unix, feature = "cpu-reference"))]
struct CancellationWatcherInput {
    connection: UnixFrameConnection,
    session: AuthenticatedControlSession,
    request_id: RequestId,
    cancelled: Arc<AtomicBool>,
    watcher_done: Arc<AtomicBool>,
    replay_journal: Arc<Mutex<DurableReplayJournal>>,
}

#[cfg(all(unix, feature = "cpu-reference"))]
struct OutputWriterInput {
    input_index: usize,
    connection: UnixFrameConnection,
    receiver: Receiver<Frame>,
    failed: Arc<AtomicBool>,
}

#[cfg(all(unix, feature = "cpu-reference"))]
struct GenerationConnection {
    input_index: usize,
    connection: UnixFrameConnection,
    session: AuthenticatedControlSession,
    replay_journal: Arc<Mutex<DurableReplayJournal>>,
    request: GenerationRequest,
    request_nonce: u64,
}

#[cfg(all(unix, feature = "cpu-reference"))]
struct ConnectionBatchState<'a> {
    service: &'a mut EngineService,
    metrics: &'a mut RuntimeMetrics,
    models: &'a BTreeMap<String, LoadedRuntimeModel>,
    response_signer: &'a EngineResponseSigner,
    next_sequence: &'a mut u64,
    active: &'a mut Vec<ActiveGenerationConnection>,
    results: &'a mut [Option<ConnectionServeResult>],
}

/// Trusted-backend runtime joining authenticated UDS transport to loaded models.
///
/// Production packages enter only through [`Self::load_registered_model`]
/// after owner configuration has registered the model identifier, digest,
/// signer, and relative directory name. Client messages can select only a
/// model identifier and can never install or name a filesystem path.
#[cfg(all(unix, feature = "cpu-reference"))]
pub struct CpuEngineRuntime {
    service: EngineService,
    trusted_control_keys: TrustedControlPlaneKeys,
    response_signer: EngineResponseSigner,
    replay_journals:
        BTreeMap<personalai_engine_types::SignerKeyId, Arc<Mutex<DurableReplayJournal>>>,
    models: BTreeMap<String, LoadedRuntimeModel>,
    backend: RuntimeBackend,
    model_load_budgets: BTreeMap<String, ResourceBudget>,
    security_events: VecDeque<SecurityEvent>,
    metrics: RuntimeMetrics,
    draining: bool,
    next_sequence: u64,
    runtime_v3: Option<runtime_v3_service::RuntimeV3ServiceState>,
}

#[cfg(all(unix, feature = "cpu-reference"))]
fn runtime_metrics(
    service: &EngineService,
    models: &BTreeMap<String, LoadedRuntimeModel>,
    backend: &RuntimeBackend,
    mut metrics: RuntimeMetrics,
) -> Result<RuntimeMetrics, EngineError> {
    metrics.loaded_models = u64::try_from(models.len())
        .map_err(|_| EngineError::internal_invariant("loaded model count exceeds u64"))?;
    metrics.active_requests = u64::try_from(service.scheduler.active_len())
        .map_err(|_| EngineError::internal_invariant("active request count exceeds u64"))?;
    metrics.queued_requests = u64::try_from(service.scheduler.queued_len())
        .map_err(|_| EngineError::internal_invariant("queued request count exceeds u64"))?;
    metrics.host_memory_bytes = models.values().try_fold(0_u64, |total, loaded| {
        let used = loaded.memory_pool().used_bytes()?;
        total
            .checked_add(used)
            .ok_or_else(|| EngineError::resource_exhaustion("loaded model memory metric overflows"))
    })?;
    metrics.kv_cache_bytes = models.values().try_fold(0_u64, |total, loaded| {
        let reserved = loaded.kv_cache_bytes()?;
        total
            .checked_add(reserved)
            .ok_or_else(|| EngineError::resource_exhaustion("KV-cache memory metric overflows"))
    })?;
    metrics.device_memory_bytes = backend.device_memory_bytes()?;
    Ok(metrics)
}

#[cfg(all(unix, feature = "cpu-reference"))]
fn runtime_health(draining: bool, models: &BTreeMap<String, LoadedRuntimeModel>) -> ServiceHealth {
    if draining {
        ServiceHealth::Draining
    } else if models.is_empty() {
        ServiceHealth::Degraded
    } else {
        ServiceHealth::Ready
    }
}

#[cfg(all(unix, feature = "cpu-reference"))]
fn retain_security_event(events: &mut VecDeque<SecurityEvent>, event: SecurityEvent) {
    if events.len() == MAX_PENDING_SECURITY_EVENTS {
        let _oldest = events.pop_front();
    }
    events.push_back(event);
}

#[cfg(all(unix, feature = "cpu-reference"))]
fn record_runtime_error_parts(
    metrics: &mut RuntimeMetrics,
    events: &mut VecDeque<SecurityEvent>,
    backend: &RuntimeBackend,
    request_id: RequestId,
    error: &EngineError,
) -> Result<(), EngineError> {
    let event = match error {
        EngineError::AuthenticationFailure(_) => {
            metrics.authentication_failures = checked_increment(
                metrics.authentication_failures,
                "authentication failure counter",
            )?;
            SecurityEvent::AuthenticationFailure { request_id: Some(request_id) }
        }
        EngineError::ProtocolViolation(_) => {
            metrics.protocol_violations =
                checked_increment(metrics.protocol_violations, "protocol violation counter")?;
            SecurityEvent::ProtocolViolation { request_id: Some(request_id) }
        }
        EngineError::BackendFailure(_) => {
            metrics.backend_failures =
                checked_increment(metrics.backend_failures, "backend failure counter")?;
            SecurityEvent::BackendFailure {
                request_id: Some(request_id),
                backend: Some(backend.backend_id()?),
            }
        }
        EngineError::ResourceExhaustion(detail) => {
            metrics.rejected_requests =
                checked_increment(metrics.rejected_requests, "rejected request counter")?;
            SecurityEvent::ResourceLimitExceeded {
                request_id: Some(request_id),
                resource: detail.resource().unwrap_or(ResourceKind::RequestBytes),
                requested: detail.requested(),
                limit: detail.limit_value().unwrap_or(0),
            }
        }
        _ => SecurityEvent::ProtocolViolation { request_id: Some(request_id) },
    };
    retain_security_event(events, event);
    if matches!(error, EngineError::AuthenticationFailure(_))
        && metrics.authentication_failures % SECURITY_PATTERN_THRESHOLD == 0
    {
        retain_security_event(events, SecurityEvent::RepeatedAuthenticationFailure);
    }
    if matches!(error, EngineError::ResourceExhaustion(_))
        && metrics.rejected_requests % SECURITY_PATTERN_THRESHOLD == 0
    {
        retain_security_event(events, SecurityEvent::ResourceExhaustionPattern);
        retain_security_event(events, SecurityEvent::RepeatedRejectedRequests);
    }
    Ok(())
}

#[cfg(all(unix, feature = "cpu-reference"))]
fn unload_model_parts(
    service: &EngineService,
    models: &mut BTreeMap<String, LoadedRuntimeModel>,
    model_id: &ModelId,
) -> Result<(), EngineError> {
    if service.scheduler.has_active_model(model_id) || service.scheduler.has_queued_model(model_id)
    {
        return Err(EngineError::resource_exhaustion(
            "model cannot unload while a request owns scheduler state",
        ));
    }
    let loaded = models
        .get(model_id.as_str())
        .ok_or_else(|| EngineError::invalid_input("model is not loaded"))?;
    loaded.unload_if_idle()?;
    if models.remove(model_id.as_str()).is_none() {
        return Err(EngineError::invalid_input("model is not loaded"));
    }
    Ok(())
}

#[cfg(all(unix, feature = "cpu-reference"))]
fn load_registered_model_parts(
    service: &EngineService,
    models: &mut BTreeMap<String, LoadedRuntimeModel>,
    backend: &RuntimeBackend,
    metrics: &mut RuntimeMetrics,
    model_id: &ModelId,
    budget: &ResourceBudget,
    events: &mut impl DirectorySecurityEventSink,
) -> Result<(), EngineError> {
    if models.contains_key(model_id.as_str()) {
        return Err(EngineError::invalid_input("model is already loaded"));
    }
    let loaded = match load_runtime_model(service, model_id, budget, backend, events) {
        Ok(loaded) => loaded,
        Err(error) => {
            metrics.package_validation_failures =
                metrics.package_validation_failures.checked_add(1).ok_or_else(|| {
                    EngineError::resource_exhaustion("package validation failure counter exhausted")
                })?;
            return Err(error);
        }
    };
    models.insert(model_id.as_str().to_owned(), loaded);
    Ok(())
}

#[cfg(all(unix, feature = "cpu-reference"))]
fn load_runtime_model(
    service: &EngineService,
    model_id: &ModelId,
    budget: &ResourceBudget,
    backend: &RuntimeBackend,
    events: &mut impl DirectorySecurityEventSink,
) -> Result<LoadedRuntimeModel, EngineError> {
    match backend {
        RuntimeBackend::CpuReference => {
            service.load_registered_cpu_model(model_id, budget, events).map(LoadedRuntimeModel::Cpu)
        }
        #[cfg(all(feature = "arm64", target_arch = "aarch64"))]
        RuntimeBackend::Arm64(arm64) => service
            .load_registered_arm64_model(model_id, budget, arm64.clone(), events)
            .map(LoadedRuntimeModel::Cpu),
        #[cfg(all(feature = "metal", target_os = "macos"))]
        RuntimeBackend::Metal(metal) => service
            .load_registered_metal_model(model_id, budget, metal.clone(), events)
            .map(LoadedRuntimeModel::Metal),
    }
}

#[cfg(all(unix, feature = "cpu-reference"))]
fn model_status_parts(
    service: &EngineService,
    models: &BTreeMap<String, LoadedRuntimeModel>,
    backend: &RuntimeBackend,
    draining: bool,
    model_id: &ModelId,
) -> Result<ServiceResponseBody, EngineError> {
    service.registry.resolve(model_id)?;
    let limits = service.scheduler.limits();
    let status = ServiceModelStatus::new(
        models.contains_key(model_id.as_str()),
        service_health_status(runtime_health(draining, models)),
        service_capabilities(backend)?,
        u32::try_from(service.scheduler.active_len())
            .map_err(|_| EngineError::internal_invariant("active scheduler count exceeds u32"))?,
        u32::try_from(service.scheduler.queued_len())
            .map_err(|_| EngineError::internal_invariant("queued scheduler count exceeds u32"))?,
        limits.maximum_active(),
        u32::try_from(limits.maximum_queued())
            .map_err(|_| EngineError::internal_invariant("queued scheduler limit exceeds u32"))?,
        u64::try_from(service.scheduler.queued_bytes())
            .map_err(|_| EngineError::internal_invariant("queued scheduler bytes exceed u64"))?,
        u64::try_from(limits.maximum_queued_bytes()).map_err(|_| {
            EngineError::internal_invariant("queued scheduler byte limit exceeds u64")
        })?,
    )?;
    Ok(ServiceResponseBody::ModelStatus(status))
}

#[cfg(all(unix, feature = "cpu-reference"))]
fn model_descriptor_parts(
    service: &EngineService,
    models: &BTreeMap<String, LoadedRuntimeModel>,
    model_id: &ModelId,
) -> Result<ServiceResponseBody, EngineError> {
    let registration = service.registry.resolve(model_id)?;
    let loaded = models
        .get(model_id.as_str())
        .ok_or_else(|| EngineError::invalid_input("model is not loaded"))?;
    Ok(ServiceResponseBody::ModelDescriptor(ServiceModelDescriptor::new(
        *registration.expected_hash(),
        loaded.architecture_id().clone(),
        loaded.backend_id()?,
        loaded.context_limit(),
    )))
}

#[cfg(all(unix, feature = "cpu-reference"))]
#[allow(
    clippy::too_many_arguments,
    clippy::too_many_lines,
    reason = "one internal service dispatcher borrows disjoint bounded runtime state without granting a combined authority object"
)]
fn handle_service_request_parts(
    service: &mut EngineService,
    models: &mut BTreeMap<String, LoadedRuntimeModel>,
    backend: &RuntimeBackend,
    model_load_budgets: &BTreeMap<String, ResourceBudget>,
    security_events: &mut VecDeque<SecurityEvent>,
    metrics: &mut RuntimeMetrics,
    draining: bool,
    runtime_v3: &mut Option<runtime_v3_service::RuntimeV3ServiceState>,
    request: &AuthenticatedServiceRequest,
) -> Result<ServiceResponseBody, EngineError> {
    if let Some(state) = runtime_v3.as_mut() {
        if let Some(response) = state.handle(
            service,
            models,
            backend,
            draining,
            request.body(),
            runtime_v3_now_unix_millis()?,
        )? {
            return Ok(response);
        }
    }
    match request.body() {
        ServiceRequestBody::Hello { major, minor } => {
            if *major == PROTOCOL_MAJOR && *minor == PROTOCOL_MINOR {
                Ok(ServiceResponseBody::HelloAccepted {
                    major: PROTOCOL_MAJOR,
                    minor: PROTOCOL_MINOR,
                })
            } else {
                Ok(ServiceResponseBody::HelloRejected {
                    major: PROTOCOL_MAJOR,
                    minor: PROTOCOL_MINOR,
                })
            }
        }
        ServiceRequestBody::ListCapabilities => {
            Ok(ServiceResponseBody::Capabilities(service_capabilities(backend)?))
        }
        ServiceRequestBody::ListRegisteredModels => {
            let registered = service
                .registry
                .registrations()
                .map(|registration| registration.model_id().clone())
                .collect();
            Ok(ServiceResponseBody::RegisteredModels(registered))
        }
        ServiceRequestBody::LoadModel(model_id) => {
            service.registry.resolve(model_id)?;
            if models.contains_key(model_id.as_str()) {
                return Ok(ServiceResponseBody::ModelLoaded);
            }
            let budget = model_load_budgets.get(model_id.as_str()).copied().ok_or_else(|| {
                EngineError::authentication_failure(
                    "registered model has no owner-configured load budget",
                )
            })?;
            let mut emitted = Vec::new();
            let result = load_registered_model_parts(
                service,
                models,
                backend,
                metrics,
                model_id,
                &budget,
                &mut |event| emitted.push(event),
            );
            for event in emitted {
                retain_security_event(security_events, event);
            }
            result?;
            Ok(ServiceResponseBody::ModelLoaded)
        }
        ServiceRequestBody::UnloadModel(model_id) => {
            unload_model_parts(service, models, model_id)?;
            Ok(ServiceResponseBody::ModelUnloaded)
        }
        ServiceRequestBody::Tokenize { model_id, text } => {
            let model = models
                .get(model_id.as_str())
                .ok_or_else(|| EngineError::invalid_input("model is not loaded"))?;
            let tokens = model.tokenizer().encode(text)?;
            let tokens = tokens.into_iter().map(TokenId::new).collect::<Result<Vec<_>, _>>()?;
            Ok(ServiceResponseBody::Tokenized(tokens))
        }
        ServiceRequestBody::Health => {
            Ok(ServiceResponseBody::Health(service_health_status(runtime_health(draining, models))))
        }
        ServiceRequestBody::Metrics => Ok(ServiceResponseBody::Metrics(service_metrics(
            runtime_metrics(service, models, backend, *metrics)?,
        ))),
        ServiceRequestBody::ShutdownRequest => Ok(ServiceResponseBody::ShutdownRejected(
            ShutdownRejectionReason::AdministrativeAuthorizationRequired,
        )),
        ServiceRequestBody::ModelStatus(model_id) => {
            model_status_parts(service, models, backend, draining, model_id)
        }
        ServiceRequestBody::SecurityEvents { maximum } => {
            let count = usize::from(*maximum).min(security_events.len());
            let events = security_events
                .iter()
                .take(count)
                .cloned()
                .map(ServiceSecurityEvent::from_event)
                .collect::<Result<Vec<_>, _>>()?;
            Ok(ServiceResponseBody::SecurityEvents(events))
        }
        ServiceRequestBody::ModelDescriptor(model_id) => {
            model_descriptor_parts(service, models, model_id)
        }
        ServiceRequestBody::AcknowledgeSecurityEvents { count } => {
            let count = usize::from(*count);
            if count > security_events.len() {
                return Err(EngineError::protocol_violation(
                    "security event acknowledgement exceeds the queued prefix",
                ));
            }
            security_events.drain(..count);
            Ok(ServiceResponseBody::SecurityEventsAcknowledged)
        }
        ServiceRequestBody::RuntimeV3Availability
        | ServiceRequestBody::RuntimeV3Reserve(_)
        | ServiceRequestBody::RuntimeV3Cancel(_)
        | ServiceRequestBody::RuntimeV3Release(_)
        | ServiceRequestBody::RuntimeV3Execute(_) => Err(EngineError::not_implemented(
            "runtime v3 authenticated service is not enabled by owner configuration",
        )),
    }
}

#[cfg(all(unix, feature = "cpu-reference"))]
fn runtime_v3_now_unix_millis() -> Result<u64, EngineError> {
    let elapsed = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|_| EngineError::internal_invariant("system clock is before the Unix epoch"))?;
    u64::try_from(elapsed.as_millis())
        .map_err(|_| EngineError::resource_exhaustion("system clock exceeds runtime v3 range"))
}

#[cfg(all(unix, feature = "cpu-reference"))]
impl CpuEngineRuntime {
    /// Creates an unloaded runtime with distinct control and response identities.
    pub fn new(
        service: EngineService,
        trusted_control_key: TrustedControlPlaneKey,
        response_signer: EngineResponseSigner,
    ) -> Result<Self, EngineError> {
        Self::new_with_control_keys(
            service,
            TrustedControlPlaneKeys::single(trusted_control_key)?,
            response_signer,
        )
    }

    /// Creates an unloaded runtime with a bounded control-key overlap window.
    pub fn new_with_control_keys(
        service: EngineService,
        trusted_control_keys: TrustedControlPlaneKeys,
        response_signer: EngineResponseSigner,
    ) -> Result<Self, EngineError> {
        Self::new_with_control_keys_and_backend(
            service,
            trusted_control_keys,
            response_signer,
            RuntimeBackend::CpuReference,
        )
    }

    /// Creates an unloaded runtime with one owner-selected execution backend.
    pub fn new_with_control_keys_and_backend(
        service: EngineService,
        trusted_control_keys: TrustedControlPlaneKeys,
        response_signer: EngineResponseSigner,
        backend: RuntimeBackend,
    ) -> Result<Self, EngineError> {
        for trusted_control_key in trusted_control_keys.configured_keys() {
            if trusted_control_key.key_id() == response_signer.key_id()
                || trusted_control_key.public_key() == response_signer.public_key()
            {
                return Err(EngineError::invalid_input(
                    "control and response signing identities must be distinct",
                ));
            }
        }
        let mut replay_journals = BTreeMap::new();
        for trusted_control_key in trusted_control_keys.active_keys() {
            let journal = DurableReplayJournal::open(
                service.config().runtime_root(),
                trusted_control_key.key_id(),
            )?;
            replay_journals.insert(trusted_control_key.key_id(), Arc::new(Mutex::new(journal)));
        }
        Ok(Self {
            service,
            trusted_control_keys,
            response_signer,
            replay_journals,
            models: BTreeMap::new(),
            backend,
            model_load_budgets: BTreeMap::new(),
            security_events: VecDeque::new(),
            metrics: RuntimeMetrics::default(),
            draining: false,
            next_sequence: 1,
            runtime_v3: None,
        })
    }

    /// Enables canonical runtime-v3 status and reservation handling.
    ///
    /// This owner-only startup method is not reachable from the client
    /// protocol and does not qualify or register any model.
    pub fn enable_runtime_v3(&mut self, config: RuntimeV3ServiceConfig) -> Result<(), EngineError> {
        if self.runtime_v3.is_some() {
            return Err(EngineError::invalid_input("runtime v3 service is already enabled"));
        }
        self.runtime_v3 = Some(runtime_v3_service::RuntimeV3ServiceState::new(config)?);
        Ok(())
    }

    /// Binds one owner-approved control-plane identity to an already
    /// registered engine model. This does not manufacture qualification.
    pub fn configure_runtime_v3_model(
        &mut self,
        identity: RuntimeIdentityBinding,
    ) -> Result<(), EngineError> {
        self.runtime_v3
            .as_mut()
            .ok_or_else(|| EngineError::invalid_input("runtime v3 service is not enabled"))?
            .configure_model(&self.service, &self.backend, identity)
    }

    /// Enables the compile-time-gated deterministic runtime-v3 simulator.
    /// Default builds do not expose this method, and it cannot be selected by
    /// any client message.
    #[cfg(feature = "development-runtime-v3-simulator")]
    pub fn enable_runtime_v3_development_simulator(
        &mut self,
        behavior: RuntimeV3DevelopmentBehavior,
    ) -> Result<(), EngineError> {
        self.runtime_v3
            .as_mut()
            .ok_or_else(|| EngineError::invalid_input("runtime v3 service is not enabled"))?
            .enable_development_simulator(behavior)
    }

    /// Derives engine-owned package/backend identity from one registered model
    /// and binds only the control-plane logical/artifact correlations supplied
    /// by owner startup configuration.
    pub fn configure_runtime_v3_registered_model(
        &mut self,
        logical_model_id: LogicalModelId,
        provider_model_id: ModelId,
        artifact_id: ArtifactAuthorityId,
    ) -> Result<(), EngineError> {
        self.runtime_v3
            .as_mut()
            .ok_or_else(|| EngineError::invalid_input("runtime v3 service is not enabled"))?
            .configure_registered_model(
                &self.service,
                &self.backend,
                logical_model_id,
                provider_model_id,
                artifact_id,
            )
    }

    /// Installs a trusted load budget for one statically registered model.
    ///
    /// This owner-only API is not reachable from the client protocol. A client
    /// `LoadModel` request supplies only the registered model ID.
    pub fn configure_model_load_budget(
        &mut self,
        model_id: &ModelId,
        budget: ResourceBudget,
    ) -> Result<(), EngineError> {
        self.service.registry.resolve(model_id)?;
        self.model_load_budgets.insert(model_id.as_str().to_owned(), budget);
        Ok(())
    }

    /// Returns content-free live and cumulative metrics.
    pub fn metrics(&self) -> Result<RuntimeMetrics, EngineError> {
        runtime_metrics(&self.service, &self.models, &self.backend, self.metrics)
    }

    /// Returns the current local health classification.
    pub fn health(&self) -> ServiceHealth {
        runtime_health(self.draining, &self.models)
    }

    /// Drains sanitized typed telemetry for Go-owned delivery and audit.
    pub fn drain_security_events(&mut self) -> Vec<SecurityEvent> {
        self.security_events.drain(..).collect()
    }

    fn record_runtime_error(
        &mut self,
        request_id: RequestId,
        error: &EngineError,
    ) -> Result<(), EngineError> {
        record_runtime_error_parts(
            &mut self.metrics,
            &mut self.security_events,
            &self.backend,
            request_id,
            error,
        )
    }

    /// Unloads one idle model. Active or queued ownership fails closed.
    pub fn unload_model(&mut self, model_id: &ModelId) -> Result<(), EngineError> {
        unload_model_parts(&self.service, &mut self.models, model_id)
    }

    fn handle_service_request(
        &mut self,
        request: &AuthenticatedServiceRequest,
    ) -> Result<ServiceResponseBody, EngineError> {
        handle_service_request_parts(
            &mut self.service,
            &mut self.models,
            &self.backend,
            &self.model_load_budgets,
            &mut self.security_events,
            &mut self.metrics,
            self.draining,
            &mut self.runtime_v3,
            request,
        )
    }

    /// Verifies and installs one legacy flat fixture package in memory.
    ///
    /// This compatibility helper exists only in this crate's tests. Production
    /// runtime code accepts the signed directory package profile exclusively.
    #[cfg(test)]
    pub fn load_registered_model_bytes(
        &mut self,
        model_id: &ModelId,
        package_bytes: &[u8],
        budget: &personalai_engine_types::ResourceBudget,
    ) -> Result<(), EngineError> {
        if self.models.contains_key(model_id.as_str()) {
            return Err(EngineError::invalid_input("model is already loaded"));
        }
        let loaded =
            self.service.load_registered_cpu_model_bytes(model_id, package_bytes, budget)?;
        self.models.insert(model_id.as_str().to_owned(), LoadedRuntimeModel::Cpu(loaded));
        Ok(())
    }

    /// Verifies and installs one legacy ARM64 fixture for service-integration tests.
    #[cfg(all(test, feature = "arm64", target_arch = "aarch64"))]
    pub fn load_registered_arm64_model_bytes(
        &mut self,
        model_id: &ModelId,
        package_bytes: &[u8],
        budget: &ResourceBudget,
    ) -> Result<(), EngineError> {
        if self.models.contains_key(model_id.as_str()) {
            return Err(EngineError::invalid_input("model is already loaded"));
        }
        let RuntimeBackend::Arm64(backend) = &self.backend else {
            return Err(EngineError::invalid_input(
                "ARM64 fixture requires an ARM64-selected runtime",
            ));
        };
        let loaded = self.service.load_registered_arm64_model_bytes(
            model_id,
            package_bytes,
            budget,
            backend.clone(),
        )?;
        self.models.insert(model_id.as_str().to_owned(), LoadedRuntimeModel::Cpu(loaded));
        Ok(())
    }

    /// Verifies and installs one legacy Metal fixture for service-integration tests.
    #[cfg(all(test, feature = "metal", target_os = "macos"))]
    pub fn load_registered_metal_model_bytes(
        &mut self,
        model_id: &ModelId,
        package_bytes: &[u8],
        budget: &ResourceBudget,
    ) -> Result<(), EngineError> {
        if self.models.contains_key(model_id.as_str()) {
            return Err(EngineError::invalid_input("model is already loaded"));
        }
        let RuntimeBackend::Metal(backend) = &self.backend else {
            return Err(EngineError::invalid_input(
                "Metal fixture requires a Metal-selected runtime",
            ));
        };
        let loaded = self.service.load_registered_metal_model_bytes(
            model_id,
            package_bytes,
            budget,
            backend.clone(),
        )?;
        self.models.insert(model_id.as_str().to_owned(), LoadedRuntimeModel::Metal(loaded));
        Ok(())
    }

    /// Opens, verifies, and loads one registered package from the configured
    /// owner-controlled model root.
    pub fn load_registered_model(
        &mut self,
        model_id: &ModelId,
        budget: &personalai_engine_types::ResourceBudget,
        events: &mut impl DirectorySecurityEventSink,
    ) -> Result<(), EngineError> {
        load_registered_model_parts(
            &self.service,
            &mut self.models,
            &self.backend,
            &mut self.metrics,
            model_id,
            budget,
            events,
        )
    }

    /// Accepts and fairly executes an exact bounded batch of local connections.
    ///
    /// Every connection is authenticated and durably replay-checked before its
    /// command is considered. Generation cursors are interleaved one decoder
    /// token at a time, events remain signed and ordered per stream, and a
    /// cancellation or event-stream failure after admission is isolated to its
    /// request. This bounded reference path does not implement a continuous
    /// accept loop.
    pub fn serve_batch(
        &mut self,
        listener: &UnixFrameListener,
        connection_count: u32,
    ) -> Result<Vec<ConnectionServeResult>, EngineError> {
        let maximum_active = self.service.scheduler.limits().maximum_active();
        if connection_count == 0
            || connection_count > maximum_active
            || connection_count > MAX_CONCURRENT_CONNECTION_BATCH
        {
            return Err(EngineError::invalid_input(
                "concurrent connection batch is outside scheduler bounds",
            ));
        }
        if self.service.scheduler.queued_len() != 0 || self.service.scheduler.active_len() != 0 {
            return Err(EngineError::protocol_violation(
                "concurrent connection batch requires an idle scheduler",
            ));
        }
        let connection_count = usize::try_from(connection_count).map_err(|_| {
            EngineError::resource_exhaustion("concurrent connection count is not addressable")
        })?;
        let mut connections = Vec::new();
        connections.try_reserve_exact(connection_count).map_err(|_| {
            EngineError::resource_exhaustion("concurrent connection allocation failed")
        })?;
        for _index in 0..connection_count {
            connections.push(listener.accept()?);
        }
        self.serve_connections(connections, maximum_active)
    }

    /// Accepts and processes exactly one authenticated local connection.
    ///
    /// A caller-owned outer loop decides service lifetime and restart policy.
    /// The method never opens a network connection or performs filesystem model
    /// discovery.
    #[expect(clippy::too_many_lines, reason = "authenticated connection cleanup remains linear")]
    pub fn serve_one(&mut self, listener: &UnixFrameListener) -> Result<ServeOutcome, EngineError> {
        let mut connection = listener.accept()?;
        let frame = connection.read_inbound_frame()?;
        let mut session =
            AuthenticatedControlSession::new_with_keys(self.trusted_control_keys.clone());
        let command = session.decode_command(&frame)?;
        let authenticated_key_id = session.authenticated_key_id().ok_or_else(|| {
            EngineError::internal_invariant("authenticated command omitted its control key ID")
        })?;
        let replay_journal = self.replay_journals.get(&authenticated_key_id).ok_or_else(|| {
            EngineError::internal_invariant("active control key omitted its replay journal")
        })?;
        admit_durable_nonce(replay_journal, session.last_nonce())?;
        match command {
            InboundCommand::Service(request) => {
                let request_id = request.request_id();
                let nonce = request.nonce();
                let body = match self.handle_service_request(&request) {
                    Ok(response) => response,
                    Err(error) => {
                        self.record_runtime_error(request_id, &error)?;
                        ServiceResponseBody::Error(service_error_code(&error))
                    }
                };
                let kind = body.kind();
                let frame =
                    service_response_frame(request_id, nonce, &body, &self.response_signer)?;
                connection.write_outbound_frame(&frame)?;
                Ok(ServeOutcome::ServiceResponded { request_id, kind })
            }
            InboundCommand::Cancel(request_id) => {
                let outcome = self.service.scheduler.cancel(&request_id)?;
                Ok(match outcome {
                    CancellationOutcome::Queued => {
                        self.metrics.cancelled_requests = checked_increment(
                            self.metrics.cancelled_requests,
                            "cancelled request counter",
                        )?;
                        ServeOutcome::CancelledQueued(request_id)
                    }
                    CancellationOutcome::Active => {
                        self.metrics.cancelled_requests = checked_increment(
                            self.metrics.cancelled_requests,
                            "cancelled request counter",
                        )?;
                        ServeOutcome::CancelledActive(request_id)
                    }
                    CancellationOutcome::NotFound => ServeOutcome::CancellationNotFound(request_id),
                })
            }
            InboundCommand::Generate { request, nonce } => {
                let request_id = request.request_id();
                let model_key = request.model_id().as_str().to_owned();
                if !self.models.contains_key(&model_key) {
                    let error = EngineError::invalid_input("model is not loaded");
                    write_generation_event(
                        &mut connection,
                        &GenerationEvent::Failed {
                            request_id,
                            sequence_id: None,
                            error: error.clone(),
                        },
                        nonce,
                        1,
                        &self.response_signer,
                    )?;
                    return Err(error);
                }
                let sequence_id = SequenceId::new(self.next_sequence)?;
                let next_sequence = self.next_sequence.checked_add(1).ok_or_else(|| {
                    EngineError::resource_exhaustion("runtime sequence identifier exhausted")
                })?;
                self.service.admit(request)?;
                self.next_sequence = next_sequence;
                let model = self.models.get(&model_key).ok_or_else(|| {
                    EngineError::internal_invariant("loaded model disappeared during admission")
                })?;
                let cancellation_connection = connection.try_clone()?;
                let cancelled = AtomicBool::new(false);
                let watcher_done = AtomicBool::new(false);
                let replay_journal = Arc::clone(replay_journal);
                let mut event_ordinal = 1_u32;
                let response_signer = &self.response_signer;
                let generation_result = thread::scope(|scope| {
                    let watcher = scope.spawn(|| {
                        watch_generation_cancellation(
                            cancellation_connection,
                            session,
                            request_id,
                            &cancelled,
                            &watcher_done,
                            &replay_journal,
                        );
                    });
                    let result = self.service.run_next_greedy_runtime(
                        model,
                        sequence_id,
                        &cancelled,
                        |event| {
                            let ordinal = event_ordinal;
                            event_ordinal = event_ordinal.checked_add(1).ok_or_else(|| {
                                EngineError::resource_exhaustion(
                                    "generation event ordinal exhausted",
                                )
                            })?;
                            write_generation_event(
                                &mut connection,
                                &event,
                                nonce,
                                ordinal,
                                response_signer,
                            )
                        },
                    );
                    watcher_done.store(true, Ordering::Release);
                    let shutdown_result = connection.shutdown_inbound();
                    watcher.join().map_err(|_| {
                        EngineError::internal_invariant("cancellation watcher panicked")
                    })?;
                    shutdown_result?;
                    result
                });
                let measurement = generation_result?.ok_or_else(|| {
                    EngineError::internal_invariant("admitted request was not scheduled")
                })?;
                record_generation_measurement(&mut self.metrics, measurement)?;
                let usage = measurement.usage;
                Ok(ServeOutcome::Generated { request_id, sequence_id, usage })
            }
        }
    }

    /// Returns the service configuration and registry composition.
    pub const fn service(&self) -> &EngineService {
        &self.service
    }

    fn serve_connections(
        &mut self,
        connections: Vec<UnixFrameConnection>,
        maximum_active: u32,
    ) -> Result<Vec<ConnectionServeResult>, EngineError> {
        let mut results = Vec::new();
        results
            .try_reserve_exact(connections.len())
            .map_err(|_| EngineError::resource_exhaustion("connection result allocation failed"))?;
        results.resize_with(connections.len(), || None);
        let mut authenticated = Vec::new();
        authenticated.try_reserve_exact(connections.len()).map_err(|_| {
            EngineError::resource_exhaustion("authenticated connection allocation failed")
        })?;
        for (input_index, connection) in connections.into_iter().enumerate() {
            match self.authenticate_connection(input_index, connection) {
                Ok(connection) => authenticated.push(connection),
                Err(error) => store_connection_result(&mut results, input_index, Err(error))?,
            }
        }

        let mut active = Vec::new();
        active
            .try_reserve_exact(authenticated.len())
            .map_err(|_| EngineError::resource_exhaustion("active connection allocation failed"))?;
        let mut coordinator = GenerationRoundRobin::with_limits(
            maximum_active,
            self.service.scheduler.limits().maximum_queued_bytes(),
        )?;
        let mut batch = ConnectionBatchState {
            service: &mut self.service,
            metrics: &mut self.metrics,
            models: &self.models,
            response_signer: &self.response_signer,
            next_sequence: &mut self.next_sequence,
            active: &mut active,
            results: &mut results,
        };
        for authenticated_connection in authenticated {
            if let Err(error) = Self::prepare_authenticated_connection(
                &mut batch,
                authenticated_connection,
                &mut coordinator,
            ) {
                cleanup_active_scheduler(batch.service, batch.active)?;
                return Err(error);
            }
        }
        Self::run_connection_coordinator(
            batch.service,
            batch.metrics,
            batch.response_signer,
            &mut coordinator,
            batch.active,
            batch.results,
        )?;
        drop(coordinator);
        collect_connection_results(results)
    }

    fn authenticate_connection(
        &self,
        input_index: usize,
        mut connection: UnixFrameConnection,
    ) -> Result<AuthenticatedConnection, EngineError> {
        let frame = connection.read_inbound_frame()?;
        let mut session =
            AuthenticatedControlSession::new_with_keys(self.trusted_control_keys.clone());
        let command = session.decode_command(&frame)?;
        let authenticated_key_id = session.authenticated_key_id().ok_or_else(|| {
            EngineError::internal_invariant("authenticated command omitted its control key ID")
        })?;
        let replay_journal =
            Arc::clone(self.replay_journals.get(&authenticated_key_id).ok_or_else(|| {
                EngineError::internal_invariant("active control key omitted its replay journal")
            })?);
        admit_durable_nonce(&replay_journal, session.last_nonce())?;
        Ok(AuthenticatedConnection { input_index, connection, session, replay_journal, command })
    }

    fn prepare_authenticated_connection(
        batch: &mut ConnectionBatchState<'_>,
        authenticated: AuthenticatedConnection,
        coordinator: &mut GenerationRoundRobin,
    ) -> Result<(), EngineError> {
        let AuthenticatedConnection { input_index, connection, session, replay_journal, command } =
            authenticated;
        let InboundCommand::Generate { request, nonce } = command else {
            let request_id = match command {
                InboundCommand::Cancel(request_id) => request_id,
                InboundCommand::Service(_) => {
                    return Err(EngineError::protocol_violation(
                        "service request is not valid in a generation batch",
                    ));
                }
                InboundCommand::Generate { .. } => {
                    return Err(EngineError::internal_invariant(
                        "authenticated command variant disappeared",
                    ));
                }
            };
            let outcome = if let Some(connection) =
                batch.active.iter().find(|connection| connection.request_id == request_id)
            {
                connection.cancelled.store(true, Ordering::Release);
                ServeOutcome::CancelledActive(request_id)
            } else {
                match batch.service.scheduler.cancel(&request_id)? {
                    CancellationOutcome::Queued => ServeOutcome::CancelledQueued(request_id),
                    CancellationOutcome::Active => ServeOutcome::CancelledActive(request_id),
                    CancellationOutcome::NotFound => ServeOutcome::CancellationNotFound(request_id),
                }
            };
            if matches!(
                outcome,
                ServeOutcome::CancelledQueued(_) | ServeOutcome::CancelledActive(_)
            ) {
                batch.metrics.cancelled_requests = checked_increment(
                    batch.metrics.cancelled_requests,
                    "cancelled request counter",
                )?;
            }
            return store_connection_result(batch.results, input_index, Ok(outcome));
        };
        prepare_generation_connection(
            batch,
            coordinator,
            GenerationConnection {
                input_index,
                connection,
                session,
                replay_journal,
                request,
                request_nonce: nonce,
            },
        )
    }

    fn run_connection_coordinator(
        service: &mut EngineService,
        metrics: &mut RuntimeMetrics,
        response_signer: &EngineResponseSigner,
        coordinator: &mut GenerationRoundRobin,
        active: &mut Vec<ActiveGenerationConnection>,
        results: &mut [Option<ConnectionServeResult>],
    ) -> Result<(), EngineError> {
        let (watcher_inputs, writer_inputs) = match prepare_connection_worker_inputs(active) {
            Ok(inputs) => inputs,
            Err(error) => {
                stop_active_watchers(active);
                cleanup_active_scheduler(service, active)?;
                return Err(error);
            }
        };
        thread::scope(|scope| {
            let mut watchers = Vec::new();
            if watchers.try_reserve_exact(watcher_inputs.len()).is_err() {
                cleanup_active_scheduler(service, active)?;
                return Err(EngineError::resource_exhaustion(
                    "cancellation watcher allocation failed",
                ));
            }
            let mut writers = Vec::new();
            if writers.try_reserve_exact(writer_inputs.len()).is_err() {
                cleanup_active_scheduler(service, active)?;
                return Err(EngineError::resource_exhaustion("output writer allocation failed"));
            }
            let mut spawn_error = None;
            for watcher_input in watcher_inputs {
                let spawn = thread::Builder::new()
                    .name("personalai-cancel-watch".to_owned())
                    .stack_size(CANCELLATION_WATCHER_STACK_BYTES)
                    .spawn_scoped(scope, move || {
                        watch_generation_cancellation(
                            watcher_input.connection,
                            watcher_input.session,
                            watcher_input.request_id,
                            &watcher_input.cancelled,
                            &watcher_input.watcher_done,
                            &watcher_input.replay_journal,
                        );
                    });
                if let Ok(watcher) = spawn {
                    watchers.push(watcher);
                } else {
                    spawn_error = Some(EngineError::backend_failure(
                        "cancellation watcher thread creation failed",
                    ));
                    break;
                }
            }
            if spawn_error.is_none() {
                for writer_input in writer_inputs {
                    let input_index = writer_input.input_index;
                    let failed = Arc::clone(&writer_input.failed);
                    let spawn = thread::Builder::new()
                        .name("personalai-event-write".to_owned())
                        .stack_size(OUTPUT_WRITER_STACK_BYTES)
                        .spawn_scoped(scope, move || {
                            let result = write_queued_events(
                                writer_input.connection,
                                &writer_input.receiver,
                            );
                            if result.is_err() {
                                failed.store(true, Ordering::Release);
                            }
                            result
                        });
                    if let Ok(writer) = spawn {
                        writers.push((input_index, writer));
                    } else {
                        spawn_error = Some(EngineError::backend_failure(
                            "output writer thread creation failed",
                        ));
                        break;
                    }
                }
            }
            if let Some(error) = spawn_error {
                stop_active_watchers(active);
                close_output_queues(active);
                let cleanup_result = cleanup_active_scheduler(service, active);
                let watcher_join_result = join_cancellation_watchers(watchers);
                let writer_join_result = join_output_writers(writers, results);
                cleanup_result?;
                watcher_join_result?;
                writer_join_result?;
                return Err(error);
            }

            let run_result = drive_generation_connections(
                service,
                metrics,
                response_signer,
                coordinator,
                active,
                results,
            );
            finish_connection_workers(service, active, results, watchers, writers, run_result)
        })
    }
}

#[cfg(all(unix, feature = "cpu-reference"))]
fn prepare_generation_connection(
    batch: &mut ConnectionBatchState<'_>,
    coordinator: &mut GenerationRoundRobin,
    generation: GenerationConnection,
) -> Result<(), EngineError> {
    let GenerationConnection {
        input_index,
        mut connection,
        session,
        replay_journal,
        request,
        request_nonce,
    } = generation;
    let request_id = request.request_id();
    let model_key = request.model_id().as_str().to_owned();
    let Some(model) = batch.models.get(&model_key) else {
        return reject_generation_connection(
            batch,
            input_index,
            &mut connection,
            request_id,
            None,
            request_nonce,
            EngineError::invalid_input("model is not loaded"),
        );
    };
    let writer_connection = connection.try_clone()?;
    let (outbound_sender, outbound_receiver) =
        std::sync::mpsc::sync_channel(MAX_PENDING_EVENTS_PER_CONNECTION);
    if let Err(error) = batch.service.admit(request.clone()) {
        return reject_generation_connection(
            batch,
            input_index,
            &mut connection,
            request_id,
            None,
            request_nonce,
            error,
        );
    }
    let (sequence_id, scheduled, following_sequence) =
        match activate_queued_request(batch.service, *batch.next_sequence, request_id) {
            Ok(activated) => activated,
            Err(error) => {
                return reject_generation_connection(
                    batch,
                    input_index,
                    &mut connection,
                    request_id,
                    None,
                    request_nonce,
                    error,
                );
            }
        };
    *batch.next_sequence = following_sequence;
    let cancelled = Arc::new(AtomicBool::new(false));
    if let Err(error) = coordinator.submit_runtime(
        model,
        &scheduled,
        sequence_id,
        Arc::clone(&cancelled),
        OUTBOUND_PIPELINE_RESERVED_BYTES,
    ) {
        batch.service.scheduler.finish(&request_id)?;
        return reject_generation_connection(
            batch,
            input_index,
            &mut connection,
            request_id,
            Some(sequence_id),
            request_nonce,
            error,
        );
    }
    if batch.active.iter().any(|connection| connection.request_id() == request_id) {
        return Err(EngineError::internal_invariant("active connection identifier is duplicated"));
    }
    batch.active.push(ActiveGenerationConnection {
        input_index,
        request_id,
        connection,
        session: Some(session),
        replay_journal,
        request_nonce,
        sequence_id,
        next_event_ordinal: 1,
        cancelled,
        watcher_done: Arc::new(AtomicBool::new(false)),
        outbound_sender: Some(outbound_sender),
        outbound_receiver: Some(outbound_receiver),
        writer_connection: Some(writer_connection),
        writer_failed: Arc::new(AtomicBool::new(false)),
        stream_error: None,
    });
    Ok(())
}

#[cfg(all(unix, feature = "cpu-reference"))]
fn activate_queued_request(
    service: &mut EngineService,
    next_sequence: u64,
    request_id: RequestId,
) -> Result<(SequenceId, GenerationRequest, u64), EngineError> {
    let activation = (|| {
        let sequence_id = SequenceId::new(next_sequence)?;
        let following_sequence = next_sequence.checked_add(1).ok_or_else(|| {
            EngineError::resource_exhaustion("runtime sequence identifier exhausted")
        })?;
        let scheduled = service.scheduler.start_next(sequence_id)?.cloned().ok_or_else(|| {
            EngineError::internal_invariant("admitted concurrent request was not scheduled")
        })?;
        Ok((sequence_id, scheduled.request().clone(), following_sequence))
    })();
    if activation.is_err() {
        cancel_queued_after_admission(service, request_id)?;
    }
    activation
}

#[cfg(all(unix, feature = "cpu-reference"))]
fn reject_generation_connection(
    batch: &mut ConnectionBatchState<'_>,
    input_index: usize,
    connection: &mut UnixFrameConnection,
    request_id: RequestId,
    sequence_id: Option<SequenceId>,
    request_nonce: u64,
    error: EngineError,
) -> Result<(), EngineError> {
    let error = write_failed_or_transport_error(
        connection,
        request_id,
        sequence_id,
        request_nonce,
        batch.response_signer,
        error,
    );
    store_connection_result(batch.results, input_index, Err(error))
}

#[cfg(all(unix, feature = "cpu-reference"))]
fn prepare_cancellation_watchers(
    active: &mut [ActiveGenerationConnection],
) -> Result<Vec<CancellationWatcherInput>, EngineError> {
    if active.iter().any(|connection| connection.session.is_none()) {
        return Err(EngineError::internal_invariant(
            "active connection authentication session is missing",
        ));
    }
    let mut cloned_connections = Vec::new();
    cloned_connections.try_reserve_exact(active.len()).map_err(|_| {
        EngineError::resource_exhaustion("cancellation connection allocation failed")
    })?;
    for connection in active.iter() {
        cloned_connections.push(connection.connection.try_clone()?);
    }
    let mut watchers = Vec::new();
    watchers.try_reserve_exact(active.len()).map_err(|_| {
        EngineError::resource_exhaustion("cancellation watcher input allocation failed")
    })?;
    for (connection, cloned_connection) in active.iter_mut().zip(cloned_connections) {
        let session = connection.session.take().ok_or_else(|| {
            EngineError::internal_invariant("active authentication session disappeared")
        })?;
        watchers.push(CancellationWatcherInput {
            connection: cloned_connection,
            session,
            request_id: connection.request_id,
            cancelled: Arc::clone(&connection.cancelled),
            watcher_done: Arc::clone(&connection.watcher_done),
            replay_journal: Arc::clone(&connection.replay_journal),
        });
    }
    Ok(watchers)
}

#[cfg(all(unix, feature = "cpu-reference"))]
fn prepare_connection_worker_inputs(
    active: &mut [ActiveGenerationConnection],
) -> Result<(Vec<CancellationWatcherInput>, Vec<OutputWriterInput>), EngineError> {
    let watcher_inputs = prepare_cancellation_watchers(active)?;
    let writer_inputs = prepare_output_writers(active)?;
    Ok((watcher_inputs, writer_inputs))
}

#[cfg(all(unix, feature = "cpu-reference"))]
fn prepare_output_writers(
    active: &mut [ActiveGenerationConnection],
) -> Result<Vec<OutputWriterInput>, EngineError> {
    if active.iter().any(|connection| {
        connection.outbound_receiver.is_none() || connection.writer_connection.is_none()
    }) {
        return Err(EngineError::internal_invariant("active connection output state is missing"));
    }
    let mut writers = Vec::new();
    writers
        .try_reserve_exact(active.len())
        .map_err(|_| EngineError::resource_exhaustion("output writer input allocation failed"))?;
    for connection in active {
        writers.push(OutputWriterInput {
            input_index: connection.input_index,
            connection: connection.writer_connection.take().ok_or_else(|| {
                EngineError::internal_invariant("active writer connection disappeared")
            })?,
            receiver: connection.outbound_receiver.take().ok_or_else(|| {
                EngineError::internal_invariant("active output receiver disappeared")
            })?,
            failed: Arc::clone(&connection.writer_failed),
        });
    }
    Ok(writers)
}

#[cfg(all(unix, feature = "cpu-reference"))]
fn drive_generation_connections(
    service: &mut EngineService,
    metrics: &mut RuntimeMetrics,
    response_signer: &EngineResponseSigner,
    coordinator: &mut GenerationRoundRobin,
    active: &mut Vec<ActiveGenerationConnection>,
    results: &mut [Option<ConnectionServeResult>],
) -> Result<(), EngineError> {
    while let Some(turn) = coordinator.advance_next()? {
        record_generation_timing(metrics, turn.timing())?;
        let request_id = turn.request_id();
        let position = active
            .iter()
            .position(|connection| connection.request_id == request_id)
            .ok_or_else(|| {
                EngineError::internal_invariant("round-robin turn omitted its active connection")
            })?;
        let Some(event) = turn.into_event() else {
            continue;
        };
        if event.request_id() != request_id {
            return Err(EngineError::internal_invariant(
                "round-robin event request identifier changed",
            ));
        }
        let terminal = terminal_connection_result(&event, active[position].sequence_id)?;
        if let GenerationEvent::Completed { usage, .. } = &event {
            record_generation_usage(metrics, *usage)?;
        }
        let write_result =
            write_active_generation_event(&mut active[position], &event, response_signer);
        if let Err(error) = write_result {
            if active[position].stream_error.is_none() {
                active[position].stream_error = Some(error);
            }
            if terminal.is_none() {
                active[position].cancelled.store(true, Ordering::Release);
                continue;
            }
        }
        if let Some(terminal_result) = terminal {
            let result = match active[position].stream_error.take() {
                Some(error) => Err(error),
                None => terminal_result,
            };
            finish_active_connection(service, active, position, results, result)?;
        }
    }
    Ok(())
}

#[cfg(all(unix, feature = "cpu-reference"))]
fn write_active_generation_event(
    connection: &mut ActiveGenerationConnection,
    event: &GenerationEvent,
    response_signer: &EngineResponseSigner,
) -> Result<(), EngineError> {
    if let Some(error) = &connection.stream_error {
        return Err(error.clone());
    }
    if connection.writer_failed.load(Ordering::Acquire) {
        return Err(EngineError::backend_failure("generation event writer failed"));
    }
    let ordinal = connection.next_event_ordinal;
    connection.next_event_ordinal = ordinal
        .checked_add(1)
        .ok_or_else(|| EngineError::resource_exhaustion("generation event ordinal exhausted"))?;
    let frame = generation_event_frame(event, connection.request_nonce, ordinal, response_signer)?;
    let sender = connection
        .outbound_sender
        .as_ref()
        .ok_or_else(|| EngineError::internal_invariant("active output sender is unavailable"))?;
    enqueue_outbound_frame(sender, frame)
}

#[cfg(all(unix, feature = "cpu-reference"))]
fn terminal_connection_result(
    event: &GenerationEvent,
    expected_sequence_id: SequenceId,
) -> Result<Option<ConnectionServeResult>, EngineError> {
    match event {
        GenerationEvent::Completed { request_id, sequence_id, usage } => {
            if *sequence_id != expected_sequence_id {
                return Err(EngineError::internal_invariant(
                    "completed event sequence identifier changed",
                ));
            }
            Ok(Some(Ok(ServeOutcome::Generated {
                request_id: *request_id,
                sequence_id: *sequence_id,
                usage: *usage,
            })))
        }
        GenerationEvent::Failed { sequence_id, error, .. } => {
            if sequence_id.is_some_and(|sequence_id| sequence_id != expected_sequence_id) {
                return Err(EngineError::internal_invariant(
                    "failed event sequence identifier changed",
                ));
            }
            Ok(Some(Err(error.clone())))
        }
        _ => Ok(None),
    }
}

#[cfg(all(unix, feature = "cpu-reference"))]
fn finish_active_connection(
    service: &mut EngineService,
    active: &mut Vec<ActiveGenerationConnection>,
    position: usize,
    results: &mut [Option<ConnectionServeResult>],
    result: ConnectionServeResult,
) -> Result<(), EngineError> {
    if position >= active.len() {
        return Err(EngineError::internal_invariant(
            "active connection removal position is invalid",
        ));
    }
    let connection = active.remove(position);
    connection.watcher_done.store(true, Ordering::Release);
    let shutdown_result = connection.connection.shutdown_inbound();
    service.scheduler.finish(&connection.request_id)?;
    let result = match shutdown_result {
        Ok(()) => result,
        Err(error) => Err(error),
    };
    store_connection_result(results, connection.input_index, result)
}

#[cfg(all(unix, feature = "cpu-reference"))]
fn stop_active_watchers(active: &[ActiveGenerationConnection]) {
    for connection in active {
        connection.watcher_done.store(true, Ordering::Release);
        let _shutdown_result = connection.connection.shutdown_inbound();
    }
}

#[cfg(all(unix, feature = "cpu-reference"))]
fn close_output_queues(active: &mut [ActiveGenerationConnection]) {
    for connection in active {
        connection.outbound_sender = None;
    }
}

#[cfg(all(unix, feature = "cpu-reference"))]
fn join_cancellation_watchers(
    watchers: Vec<thread::ScopedJoinHandle<'_, ()>>,
) -> Result<(), EngineError> {
    for watcher in watchers {
        watcher
            .join()
            .map_err(|_| EngineError::internal_invariant("cancellation watcher panicked"))?;
    }
    Ok(())
}

#[cfg(all(unix, feature = "cpu-reference"))]
fn join_output_writers(
    writers: Vec<(usize, thread::ScopedJoinHandle<'_, Result<(), EngineError>>)>,
    results: &mut [Option<ConnectionServeResult>],
) -> Result<(), EngineError> {
    for (input_index, writer) in writers {
        let writer_result =
            writer.join().map_err(|_| EngineError::internal_invariant("output writer panicked"))?;
        if let Err(error) = writer_result {
            replace_connection_result_with_error(results, input_index, error)?;
        }
    }
    Ok(())
}

#[cfg(all(unix, feature = "cpu-reference"))]
fn finish_connection_workers(
    service: &mut EngineService,
    active: &mut Vec<ActiveGenerationConnection>,
    results: &mut [Option<ConnectionServeResult>],
    watchers: Vec<thread::ScopedJoinHandle<'_, ()>>,
    writers: Vec<(usize, thread::ScopedJoinHandle<'_, Result<(), EngineError>>)>,
    run_result: Result<(), EngineError>,
) -> Result<(), EngineError> {
    stop_active_watchers(active);
    if run_result.is_err() {
        cleanup_active_scheduler(service, active)?;
    }
    close_output_queues(active);
    let watcher_join_result = join_cancellation_watchers(watchers);
    let writer_join_result = join_output_writers(writers, results);
    run_result?;
    watcher_join_result?;
    writer_join_result?;
    if !active.is_empty() {
        cleanup_active_scheduler(service, active)?;
        return Err(EngineError::internal_invariant(
            "round-robin coordinator stopped with active connections",
        ));
    }
    Ok(())
}

#[cfg(all(unix, feature = "cpu-reference"))]
fn write_queued_events(
    mut connection: UnixFrameConnection,
    receiver: &Receiver<Frame>,
) -> Result<(), EngineError> {
    while let Ok(frame) = receiver.recv() {
        connection.write_outbound_frame(&frame)?;
    }
    Ok(())
}

#[cfg(all(unix, feature = "cpu-reference"))]
fn enqueue_outbound_frame(sender: &SyncSender<Frame>, frame: Frame) -> Result<(), EngineError> {
    match sender.try_send(frame) {
        Ok(()) => Ok(()),
        Err(TrySendError::Full(_frame)) => {
            Err(EngineError::resource_exhaustion("generation event queue is full"))
        }
        Err(TrySendError::Disconnected(_frame)) => {
            Err(EngineError::backend_failure("generation event writer is unavailable"))
        }
    }
}

#[cfg(all(unix, feature = "cpu-reference"))]
fn cleanup_active_scheduler(
    service: &mut EngineService,
    active: &mut Vec<ActiveGenerationConnection>,
) -> Result<(), EngineError> {
    for connection in active.drain(..) {
        connection.watcher_done.store(true, Ordering::Release);
        let _shutdown_result = connection.connection.shutdown_inbound();
        service.scheduler.finish(&connection.request_id)?;
    }
    Ok(())
}

#[cfg(all(unix, feature = "cpu-reference"))]
fn cancel_queued_after_admission(
    service: &mut EngineService,
    request_id: RequestId,
) -> Result<(), EngineError> {
    match service.scheduler.cancel(&request_id)? {
        CancellationOutcome::Queued => Ok(()),
        CancellationOutcome::Active | CancellationOutcome::NotFound => {
            Err(EngineError::internal_invariant("failed concurrent admission was not queued"))
        }
    }
}

#[cfg(all(unix, feature = "cpu-reference"))]
fn store_connection_result(
    results: &mut [Option<ConnectionServeResult>],
    input_index: usize,
    result: ConnectionServeResult,
) -> Result<(), EngineError> {
    let slot = results.get_mut(input_index).ok_or_else(|| {
        EngineError::internal_invariant("connection result index is outside its allocation")
    })?;
    if slot.is_some() {
        return Err(EngineError::internal_invariant("connection result was recorded twice"));
    }
    *slot = Some(result);
    Ok(())
}

#[cfg(all(unix, feature = "cpu-reference"))]
fn replace_connection_result_with_error(
    results: &mut [Option<ConnectionServeResult>],
    input_index: usize,
    error: EngineError,
) -> Result<(), EngineError> {
    let slot = results.get_mut(input_index).ok_or_else(|| {
        EngineError::internal_invariant("connection result index is outside its allocation")
    })?;
    *slot = Some(Err(error));
    Ok(())
}

#[cfg(all(unix, feature = "cpu-reference"))]
fn collect_connection_results(
    results: Vec<Option<ConnectionServeResult>>,
) -> Result<Vec<ConnectionServeResult>, EngineError> {
    let mut collected = Vec::new();
    collected.try_reserve_exact(results.len()).map_err(|_| {
        EngineError::resource_exhaustion("collected connection result allocation failed")
    })?;
    for result in results {
        collected.push(result.ok_or_else(|| {
            EngineError::internal_invariant("connection completed without a result")
        })?);
    }
    Ok(collected)
}

#[cfg(all(unix, feature = "cpu-reference"))]
fn write_failed_or_transport_error(
    connection: &mut UnixFrameConnection,
    request_id: RequestId,
    sequence_id: Option<SequenceId>,
    request_nonce: u64,
    response_signer: &EngineResponseSigner,
    error: EngineError,
) -> EngineError {
    let failed = GenerationEvent::Failed { request_id, sequence_id, error: error.clone() };
    match write_generation_event(connection, &failed, request_nonce, 1, response_signer) {
        Ok(()) => error,
        Err(transport_error) => transport_error,
    }
}

#[cfg(all(unix, feature = "cpu-reference"))]
fn watch_generation_cancellation(
    mut connection: UnixFrameConnection,
    mut session: AuthenticatedControlSession,
    active_request_id: RequestId,
    cancelled: &AtomicBool,
    watcher_done: &AtomicBool,
    replay_journal: &Mutex<DurableReplayJournal>,
) {
    let command = connection
        .read_inbound_frame()
        .and_then(|frame| session.decode_command(&frame))
        .and_then(|command| {
            admit_durable_nonce(replay_journal, session.last_nonce())?;
            Ok(command)
        });
    if watcher_done.load(Ordering::Acquire) {
        return;
    }
    if matches!(command, Ok(InboundCommand::Cancel(request_id)) if request_id == active_request_id)
    {
        cancelled.store(true, Ordering::Release);
        return;
    }
    // A disconnect, malformed follow-up, mismatched cancellation, or second
    // generation command makes the active stream untrustworthy. Fail closed.
    cancelled.store(true, Ordering::Release);
}

#[cfg(all(unix, feature = "cpu-reference"))]
fn admit_durable_nonce(
    replay_journal: &Mutex<DurableReplayJournal>,
    nonce: u64,
) -> Result<(), EngineError> {
    replay_journal
        .lock()
        .map_err(|_| EngineError::internal_invariant("durable replay lock is poisoned"))?
        .admit(nonce)
}

#[cfg(all(unix, feature = "cpu-reference"))]
fn write_generation_event(
    connection: &mut UnixFrameConnection,
    event: &GenerationEvent,
    request_nonce: u64,
    event_ordinal: u32,
    signer: &EngineResponseSigner,
) -> Result<(), EngineError> {
    let frame = generation_event_frame(event, request_nonce, event_ordinal, signer)?;
    connection.write_outbound_frame(&frame)
}

#[cfg(all(unix, feature = "cpu-reference"))]
fn generation_event_frame(
    event: &GenerationEvent,
    request_nonce: u64,
    event_ordinal: u32,
    signer: &EngineResponseSigner,
) -> Result<Frame, EngineError> {
    let payload = personalai_engine_protocol::encode_signed_generation_event(
        event,
        request_nonce,
        event_ordinal,
        signer,
    )?;
    Frame::new(FrameKind::GenerationEvent, payload)
}

#[cfg(all(unix, feature = "cpu-reference"))]
fn service_response_frame(
    request_id: RequestId,
    request_nonce: u64,
    body: &ServiceResponseBody,
    signer: &EngineResponseSigner,
) -> Result<Frame, EngineError> {
    let payload = encode_signed_service_response(request_id, request_nonce, body, signer)?;
    Frame::new(FrameKind::ServiceResponse, payload)
}

#[cfg(all(unix, feature = "cpu-reference"))]
fn service_capabilities(backend: &RuntimeBackend) -> Result<ServiceCapabilitySet, EngineError> {
    backend.capabilities()
}

#[cfg(all(unix, feature = "cpu-reference"))]
const fn service_health_status(health: ServiceHealth) -> ServiceHealthStatus {
    match health {
        ServiceHealth::Ready => ServiceHealthStatus::Ready,
        ServiceHealth::Degraded => ServiceHealthStatus::Degraded,
        ServiceHealth::Draining => ServiceHealthStatus::Draining,
    }
}

#[cfg(all(unix, feature = "cpu-reference"))]
fn service_metrics(metrics: RuntimeMetrics) -> ServiceMetrics {
    ServiceMetrics::new([
        metrics.loaded_models,
        metrics.active_requests,
        metrics.queued_requests,
        metrics.rejected_requests,
        metrics.cancelled_requests,
        metrics.prompt_tokens,
        metrics.generated_tokens,
        metrics.prefill_nanoseconds,
        metrics.decode_nanoseconds,
        metrics.tokens_per_second_milli,
        metrics.host_memory_bytes,
        metrics.device_memory_bytes,
        metrics.kv_cache_bytes,
        metrics.package_validation_failures,
        metrics.authentication_failures,
        metrics.protocol_violations,
        metrics.backend_failures,
    ])
}

#[cfg(all(unix, feature = "cpu-reference"))]
const fn service_error_code(error: &EngineError) -> ServiceErrorCode {
    match error {
        EngineError::InvalidInput(_) => ServiceErrorCode::InvalidInput,
        EngineError::UnsupportedArchitecture(_) => ServiceErrorCode::UnsupportedArchitecture,
        EngineError::UnsupportedTokenizer(_) => ServiceErrorCode::UnsupportedTokenizer,
        EngineError::CorruptModelPackage(_) => ServiceErrorCode::CorruptModelPackage,
        EngineError::SignatureFailure(_) => ServiceErrorCode::SignatureFailure,
        EngineError::ResourceExhaustion(_) => ServiceErrorCode::ResourceExhaustion,
        EngineError::Cancellation(_) => ServiceErrorCode::Cancellation,
        EngineError::BackendFailure(_) => ServiceErrorCode::BackendFailure,
        EngineError::InternalInvariantViolation(_) => ServiceErrorCode::InternalInvariantViolation,
        EngineError::AuthenticationFailure(_) => ServiceErrorCode::AuthenticationFailure,
        EngineError::ProtocolViolation(_) => ServiceErrorCode::ProtocolViolation,
        EngineError::NotImplemented(_) => ServiceErrorCode::NotImplemented,
        _ => ServiceErrorCode::Unclassified,
    }
}

#[cfg(all(unix, feature = "cpu-reference"))]
fn checked_increment(value: u64, name: &'static str) -> Result<u64, EngineError> {
    value.checked_add(1).ok_or_else(|| EngineError::resource_exhaustion(name))
}

#[cfg(all(unix, feature = "cpu-reference"))]
fn record_generation_measurement(
    metrics: &mut RuntimeMetrics,
    measurement: GenerationMeasurement,
) -> Result<(), EngineError> {
    record_generation_timing(metrics, measurement.timing)?;
    record_generation_usage(metrics, measurement.usage)
}

#[cfg(all(unix, feature = "cpu-reference"))]
fn record_generation_timing(
    metrics: &mut RuntimeMetrics,
    timing: GenerationTiming,
) -> Result<(), EngineError> {
    metrics.prefill_nanoseconds = metrics
        .prefill_nanoseconds
        .checked_add(timing.prefill_nanoseconds())
        .ok_or_else(|| EngineError::resource_exhaustion("prefill timing counter exhausted"))?;
    metrics.decode_nanoseconds = metrics
        .decode_nanoseconds
        .checked_add(timing.decode_nanoseconds())
        .ok_or_else(|| EngineError::resource_exhaustion("decode timing counter exhausted"))?;
    update_tokens_per_second(metrics)
}

#[cfg(all(unix, feature = "cpu-reference"))]
fn record_generation_usage(
    metrics: &mut RuntimeMetrics,
    usage: GenerationUsage,
) -> Result<(), EngineError> {
    metrics.prompt_tokens = metrics
        .prompt_tokens
        .checked_add(u64::from(usage.input_tokens()))
        .ok_or_else(|| EngineError::resource_exhaustion("prompt token counter exhausted"))?;
    metrics.generated_tokens = metrics
        .generated_tokens
        .checked_add(u64::from(usage.output_tokens()))
        .ok_or_else(|| EngineError::resource_exhaustion("generated token counter exhausted"))?;
    update_tokens_per_second(metrics)
}

#[cfg(all(unix, feature = "cpu-reference"))]
fn update_tokens_per_second(metrics: &mut RuntimeMetrics) -> Result<(), EngineError> {
    let timed_nanoseconds = metrics
        .prefill_nanoseconds
        .checked_add(metrics.decode_nanoseconds)
        .ok_or_else(|| EngineError::resource_exhaustion("generation timing total exhausted"))?;
    metrics.tokens_per_second_milli = if timed_nanoseconds == 0 {
        0
    } else {
        let scaled = u128::from(metrics.generated_tokens)
            .checked_mul(1_000_000_000_000)
            .ok_or_else(|| EngineError::resource_exhaustion("throughput numerator overflows"))?
            / u128::from(timed_nanoseconds);
        u64::try_from(scaled).unwrap_or(u64::MAX)
    };
    Ok(())
}

#[cfg(feature = "cpu-reference")]
fn check_cancelled(cancelled: &AtomicBool, request_id: RequestId) -> Result<(), EngineError> {
    if cancelled.load(Ordering::Acquire) {
        Err(EngineError::cancellation(Some(request_id)))
    } else {
        Ok(())
    }
}

impl EngineService {
    /// Creates a non-listening service composition.
    pub fn new(config: ServiceConfig, limits: SchedulerLimits) -> Result<Self, EngineError> {
        let registry = ModelRegistry::new(config.model_root().to_path_buf())?;
        Ok(Self { config, registry, scheduler: Scheduler::new(limits) })
    }

    /// Registers a package from trusted local configuration.
    ///
    /// This method is not part of the future client protocol. Clients submit a
    /// model ID only and can never provide an arbitrary path.
    pub fn register_configured_model(
        &mut self,
        model_id: ModelId,
        relative_package_path: impl Into<PathBuf>,
        expected_hash: ModelPackageHash,
        trusted_signer: TrustedSigner,
    ) -> Result<(), EngineError> {
        self.registry.register(model_id, relative_package_path, expected_hash, trusted_signer)
    }

    /// Registers one exact owner-selected Model Foundry V2 decision.
    ///
    /// This is a startup-only configuration path. It verifies the exact signed
    /// registry entry before deriving the package location and never falls back
    /// to the owner-pinned registration mode.
    #[cfg(unix)]
    pub fn register_model_foundry_registry_v2(
        &mut self,
        selection: &RegistrySelectionV2,
    ) -> Result<(), EngineError> {
        self.registry.register_registry_v2(selection)
    }

    /// Admits a request only when its model ID exists in local configuration.
    pub fn admit(&mut self, request: GenerationRequest) -> Result<AdmissionReceipt, EngineError> {
        let request_id = request.request_id();
        self.registry.resolve(request.model_id())?;
        self.scheduler.submit(request)?;
        Ok(AdmissionReceipt { request_id, queued_requests: self.scheduler.queued_len() })
    }

    /// Verifies and materializes legacy flat bytes for unit-test compatibility.
    ///
    /// This method does not open a path. A future descriptor-safe local opener
    /// may call it only after resolving the configuration-owned registration.
    #[cfg(all(feature = "cpu-reference", test))]
    pub fn load_registered_cpu_model_bytes(
        &self,
        model_id: &ModelId,
        package_bytes: &[u8],
        budget: &personalai_engine_types::ResourceBudget,
    ) -> Result<LoadedCpuModel, EngineError> {
        let registration = self.registry.resolve(model_id)?;
        let package_length = u64::try_from(package_bytes.len()).map_err(|_| {
            EngineError::resource_exhaustion("legacy fixture package length is not representable")
        })?;
        if package_length == 0 || package_length > budget.max_memory_bytes() {
            return Err(EngineError::resource_exhaustion(
                "legacy fixture package exceeds its test memory budget",
            ));
        }
        if hash_package(package_bytes) != *registration.expected_hash() {
            return Err(EngineError::signature_failure(
                "legacy fixture package digest differs from test registration",
            ));
        }
        let parsed = ParsedModelPackage::parse(package_bytes)?;
        let signed_range = parsed.header().signed_content_range()?;
        let content_digest = hash_signed_content(&package_bytes[signed_range]);
        verify_package_signature(
            registration.trusted_signer(),
            parsed.signature().key_id(),
            &content_digest,
            parsed.signature().signature(),
        )?;
        let pool = MemoryPool::from_budget(budget)?;
        let config = TransformerConfig::from_manifest(parsed.manifest())?;
        let tokenizer_data = parsed.tokenizer_data(package_bytes)?;
        let tokenizer = Arc::<dyn Tokenizer>::from(build_tokenizer(tokenizer_data, &pool)?);
        if tokenizer.spec().vocabulary_size() != config.vocabulary_size() {
            return Err(EngineError::corrupt_model_package(
                "manifest and tokenizer vocabulary sizes differ",
            ));
        }
        let mut weights = BTreeMap::new();
        for record in parsed.tensors() {
            let bytes = parsed.tensor_data(package_bytes, record)?;
            let tensor =
                Tensor::copy_from_bytes(record.dtype(), record.shape().clone(), bytes, &pool)?;
            if weights.insert(record.name().as_str().to_owned(), tensor).is_some() {
                return Err(EngineError::internal_invariant(
                    "legacy test fixture contains duplicate tensor names",
                ));
            }
        }
        let model = CpuDecoderModel::new_registered(model_id.clone(), config, weights, &pool)?;
        Ok(LoadedCpuModel { model_id: model_id.clone(), model, tokenizer })
    }

    /// Verifies a signed legacy test fixture before constructing the ARM64 decoder.
    #[cfg(all(feature = "cpu-reference", feature = "arm64", target_arch = "aarch64", test))]
    pub fn load_registered_arm64_model_bytes(
        &self,
        model_id: &ModelId,
        package_bytes: &[u8],
        budget: &personalai_engine_types::ResourceBudget,
        backend: Arm64Backend,
    ) -> Result<LoadedCpuModel, EngineError> {
        let registration = self.registry.resolve(model_id)?;
        let package_length = u64::try_from(package_bytes.len()).map_err(|_| {
            EngineError::resource_exhaustion("legacy ARM64 fixture length is not representable")
        })?;
        if package_length == 0 || package_length > budget.max_memory_bytes() {
            return Err(EngineError::resource_exhaustion(
                "legacy ARM64 fixture exceeds its test memory budget",
            ));
        }
        if hash_package(package_bytes) != *registration.expected_hash() {
            return Err(EngineError::signature_failure(
                "legacy ARM64 fixture digest differs from test registration",
            ));
        }
        let parsed = ParsedModelPackage::parse(package_bytes)?;
        let signed_range = parsed.header().signed_content_range()?;
        let content_digest = hash_signed_content(&package_bytes[signed_range]);
        verify_package_signature(
            registration.trusted_signer(),
            parsed.signature().key_id(),
            &content_digest,
            parsed.signature().signature(),
        )?;
        let pool = MemoryPool::from_budget(budget)?;
        let config = TransformerConfig::from_manifest(parsed.manifest())?;
        let tokenizer_data = parsed.tokenizer_data(package_bytes)?;
        let tokenizer = Arc::<dyn Tokenizer>::from(build_tokenizer(tokenizer_data, &pool)?);
        if tokenizer.spec().vocabulary_size() != config.vocabulary_size() {
            return Err(EngineError::corrupt_model_package(
                "manifest and tokenizer vocabulary sizes differ",
            ));
        }
        let mut weights = BTreeMap::new();
        for record in parsed.tensors() {
            let bytes = parsed.tensor_data(package_bytes, record)?;
            let tensor =
                Tensor::copy_from_bytes(record.dtype(), record.shape().clone(), bytes, &pool)?;
            if weights.insert(record.name().as_str().to_owned(), tensor).is_some() {
                return Err(EngineError::internal_invariant(
                    "legacy ARM64 fixture contains duplicate tensor names",
                ));
            }
        }
        let model = CpuDecoderModel::new_registered_arm64(
            model_id.clone(),
            config,
            weights,
            &pool,
            backend,
        )?;
        Ok(LoadedCpuModel { model_id: model_id.clone(), model, tokenizer })
    }

    /// Opens, verifies, and materializes a signed directory-backed package.
    ///
    /// The mandatory sink receives one typed rejection event for every failed
    /// production package admission. No package path is accepted here.
    #[cfg(all(unix, feature = "cpu-reference"))]
    pub fn load_registered_cpu_model(
        &self,
        model_id: &ModelId,
        budget: &personalai_engine_types::ResourceBudget,
        events: &mut impl DirectorySecurityEventSink,
    ) -> Result<LoadedCpuModel, EngineError> {
        let verified = self.registry.load_registered_directory(model_id, budget, events)?;
        let result: Result<LoadedCpuModel, EngineError> = (|| {
            let pool = MemoryPool::from_budget(budget)?;
            let config = TransformerConfig::from_directory_manifest(verified.manifest())?;
            let tokenizer =
                Arc::<dyn Tokenizer>::from(build_tokenizer(verified.tokenizer_bytes(), &pool)?);
            if tokenizer.spec().vocabulary_size() != config.vocabulary_size() {
                return Err(EngineError::corrupt_model_package(
                    "manifest and tokenizer vocabulary sizes differ",
                ));
            }
            let weights = verified.materialize_tensors(&pool)?.into_tensors();
            let model = CpuDecoderModel::new_registered(model_id.clone(), config, weights, &pool)?;
            Ok(LoadedCpuModel { model_id: model_id.clone(), model, tokenizer })
        })();
        if let Err(error) = &result {
            let reason = match error {
                EngineError::UnsupportedArchitecture(_) | EngineError::NotImplemented(_) => {
                    ModelPackageRejection::UnsupportedArchitecture
                }
                EngineError::UnsupportedTokenizer(_) => ModelPackageRejection::UnsupportedTokenizer,
                EngineError::ResourceExhaustion(_) => ModelPackageRejection::SizeLimit,
                _ => ModelPackageRejection::Corrupt,
            };
            events.emit(SecurityEvent::ModelPackageRejected { model_id: model_id.clone(), reason });
        }
        result
    }

    /// Opens one registered signed directory package for complete ARM64 host execution.
    ///
    /// The backend is created only from owner-selected startup configuration.
    /// Package metadata and clients cannot select intrinsics, worker counts, or
    /// operation implementations; unsupported validated operations retain the
    /// scalar correctness fallback inside the reviewed backend.
    #[cfg(all(unix, feature = "cpu-reference", feature = "arm64", target_arch = "aarch64"))]
    pub fn load_registered_arm64_model(
        &self,
        model_id: &ModelId,
        budget: &personalai_engine_types::ResourceBudget,
        backend: Arm64Backend,
        events: &mut impl DirectorySecurityEventSink,
    ) -> Result<LoadedCpuModel, EngineError> {
        let verified = self.registry.load_registered_directory(model_id, budget, events)?;
        let result: Result<LoadedCpuModel, EngineError> = (|| {
            let pool = MemoryPool::from_budget(budget)?;
            let config = TransformerConfig::from_directory_manifest(verified.manifest())?;
            let tokenizer =
                Arc::<dyn Tokenizer>::from(build_tokenizer(verified.tokenizer_bytes(), &pool)?);
            if tokenizer.spec().vocabulary_size() != config.vocabulary_size() {
                return Err(EngineError::corrupt_model_package(
                    "manifest and tokenizer vocabulary sizes differ",
                ));
            }
            let weights = verified.materialize_tensors(&pool)?.into_tensors();
            let model = CpuDecoderModel::new_registered_arm64(
                model_id.clone(),
                config,
                weights,
                &pool,
                backend,
            )?;
            Ok(LoadedCpuModel { model_id: model_id.clone(), model, tokenizer })
        })();
        if let Err(error) = &result {
            let reason = match error {
                EngineError::UnsupportedArchitecture(_) | EngineError::NotImplemented(_) => {
                    ModelPackageRejection::UnsupportedArchitecture
                }
                EngineError::UnsupportedTokenizer(_) => ModelPackageRejection::UnsupportedTokenizer,
                EngineError::ResourceExhaustion(_) => ModelPackageRejection::SizeLimit,
                _ => ModelPackageRejection::Corrupt,
            };
            events.emit(SecurityEvent::ModelPackageRejected { model_id: model_id.clone(), reason });
        }
        result
    }

    /// Opens one registered signed directory package for complete Metal execution.
    ///
    /// The caller selects an already-created local Metal device backend. Clients
    /// still provide only `ModelId`; package paths, shaders, and kernel names are
    /// never accepted by this boundary.
    #[cfg(all(unix, feature = "cpu-reference", feature = "metal", target_os = "macos"))]
    pub fn load_registered_metal_model(
        &self,
        model_id: &ModelId,
        budget: &personalai_engine_types::ResourceBudget,
        backend: MetalBackend,
        events: &mut impl DirectorySecurityEventSink,
    ) -> Result<LoadedMetalModel, EngineError> {
        let verified = self.registry.load_registered_directory(model_id, budget, events)?;
        let result = (|| {
            let pool = MemoryPool::from_budget(budget)?;
            let config = TransformerConfig::from_directory_manifest(verified.manifest())?;
            let tokenizer =
                Arc::<dyn Tokenizer>::from(build_tokenizer(verified.tokenizer_bytes(), &pool)?);
            if tokenizer.spec().vocabulary_size() != config.vocabulary_size() {
                return Err(EngineError::corrupt_model_package(
                    "manifest and tokenizer vocabulary sizes differ",
                ));
            }
            let weights = verified.materialize_tensors(&pool)?.into_tensors();
            let model = MetalDecoderModel::new_registered(
                model_id.clone(),
                config,
                weights,
                &pool,
                backend,
            )?;
            Ok(LoadedMetalModel { model_id: model_id.clone(), model, tokenizer })
        })();
        if let Err(error) = &result {
            let reason = match error {
                EngineError::UnsupportedArchitecture(_) | EngineError::NotImplemented(_) => {
                    ModelPackageRejection::UnsupportedArchitecture
                }
                EngineError::UnsupportedTokenizer(_) => ModelPackageRejection::UnsupportedTokenizer,
                EngineError::ResourceExhaustion(_) => ModelPackageRejection::SizeLimit,
                _ => ModelPackageRejection::Corrupt,
            };
            events.emit(SecurityEvent::ModelPackageRejected { model_id: model_id.clone(), reason });
        }
        result
    }

    /// Verifies a signed legacy test fixture before constructing the Metal decoder.
    #[cfg(all(feature = "cpu-reference", feature = "metal", target_os = "macos", test))]
    pub fn load_registered_metal_model_bytes(
        &self,
        model_id: &ModelId,
        package_bytes: &[u8],
        budget: &personalai_engine_types::ResourceBudget,
        backend: MetalBackend,
    ) -> Result<LoadedMetalModel, EngineError> {
        let registration = self.registry.resolve(model_id)?;
        let package_length = u64::try_from(package_bytes.len()).map_err(|_| {
            EngineError::resource_exhaustion("legacy Metal fixture length is not representable")
        })?;
        if package_length == 0 || package_length > budget.max_memory_bytes() {
            return Err(EngineError::resource_exhaustion(
                "legacy Metal fixture exceeds its test memory budget",
            ));
        }
        if hash_package(package_bytes) != *registration.expected_hash() {
            return Err(EngineError::signature_failure(
                "legacy Metal fixture digest differs from registration",
            ));
        }
        let parsed = ParsedModelPackage::parse(package_bytes)?;
        let signed_range = parsed.header().signed_content_range()?;
        let digest = hash_signed_content(&package_bytes[signed_range]);
        verify_package_signature(
            registration.trusted_signer(),
            parsed.signature().key_id(),
            &digest,
            parsed.signature().signature(),
        )?;
        let pool = MemoryPool::from_budget(budget)?;
        let config = TransformerConfig::from_manifest(parsed.manifest())?;
        let tokenizer = Arc::<dyn Tokenizer>::from(build_tokenizer(
            parsed.tokenizer_data(package_bytes)?,
            &pool,
        )?);
        if tokenizer.spec().vocabulary_size() != config.vocabulary_size() {
            return Err(EngineError::corrupt_model_package(
                "manifest and tokenizer vocabulary sizes differ",
            ));
        }
        let mut weights = BTreeMap::new();
        for record in parsed.tensors() {
            let tensor = Tensor::copy_from_bytes(
                record.dtype(),
                record.shape().clone(),
                parsed.tensor_data(package_bytes, record)?,
                &pool,
            )?;
            if weights.insert(record.name().as_str().to_owned(), tensor).is_some() {
                return Err(EngineError::internal_invariant(
                    "legacy Metal fixture contains duplicate tensor names",
                ));
            }
        }
        let model =
            MetalDecoderModel::new_registered(model_id.clone(), config, weights, &pool, backend)?;
        Ok(LoadedMetalModel { model_id: model_id.clone(), model, tokenizer })
    }

    /// Activates and runs the next queued request on an already loaded model.
    #[cfg(feature = "cpu-reference")]
    pub fn run_next_greedy(
        &mut self,
        loaded: &LoadedCpuModel,
        sequence_id: SequenceId,
        cancelled: &AtomicBool,
        emit: impl FnMut(GenerationEvent) -> Result<(), EngineError>,
    ) -> Result<Option<GenerationUsage>, EngineError> {
        self.run_next_greedy_measured(loaded, sequence_id, cancelled, emit)
            .map(|measurement| measurement.map(|measurement| measurement.usage))
    }

    #[cfg(feature = "cpu-reference")]
    fn run_next_greedy_measured(
        &mut self,
        loaded: &LoadedCpuModel,
        sequence_id: SequenceId,
        cancelled: &AtomicBool,
        emit: impl FnMut(GenerationEvent) -> Result<(), EngineError>,
    ) -> Result<Option<GenerationMeasurement>, EngineError> {
        self.run_next_greedy_cursor(sequence_id, cancelled, emit, |request| {
            loaded.start_greedy(request, sequence_id)
        })
    }

    #[cfg(feature = "cpu-reference")]
    fn run_next_greedy_runtime(
        &mut self,
        loaded: &LoadedRuntimeModel,
        sequence_id: SequenceId,
        cancelled: &AtomicBool,
        emit: impl FnMut(GenerationEvent) -> Result<(), EngineError>,
    ) -> Result<Option<GenerationMeasurement>, EngineError> {
        self.run_next_greedy_cursor(sequence_id, cancelled, emit, |request| {
            loaded.start_greedy(request, sequence_id)
        })
    }

    #[cfg(feature = "cpu-reference")]
    fn run_next_greedy_cursor(
        &mut self,
        sequence_id: SequenceId,
        cancelled: &AtomicBool,
        mut emit: impl FnMut(GenerationEvent) -> Result<(), EngineError>,
        start: impl FnOnce(&GenerationRequest) -> Result<GreedyGeneration, EngineError>,
    ) -> Result<Option<GenerationMeasurement>, EngineError> {
        let Some(active) = self.scheduler.start_next(sequence_id)?.cloned() else {
            return Ok(None);
        };
        let request_id = active.request().request_id();
        let result: Result<GenerationMeasurement, EngineError> = (|| {
            let mut generation = start(active.request())?;
            loop {
                match generation.advance(cancelled)? {
                    GreedyAdvance::Progressed => {}
                    GreedyAdvance::Event(event) => {
                        let completed = match &event {
                            GenerationEvent::Completed { usage, .. } => Some(*usage),
                            _ => None,
                        };
                        emit(event)?;
                        if let Some(usage) = completed {
                            return Ok(GenerationMeasurement {
                                usage,
                                timing: generation.timing(),
                            });
                        }
                    }
                }
            }
        })();
        self.scheduler.finish(&request_id)?;
        match result {
            Ok(measurement) => Ok(Some(measurement)),
            Err(error) => {
                emit(GenerationEvent::Failed {
                    request_id,
                    sequence_id: Some(sequence_id),
                    error: error.clone(),
                })?;
                Err(error)
            }
        }
    }

    /// Returns the validated, unopened service configuration.
    pub const fn config(&self) -> &ServiceConfig {
        &self.config
    }

    /// Returns truthful protocol metadata.
    pub const fn protocol_descriptor(&self) -> ProtocolDescriptor {
        ProtocolDescriptor::foundation()
    }

    /// Returns compile-time and implementation status for one backend.
    pub fn backend_status(&self, backend_id: &BackendId) -> Result<BackendStatus, EngineError> {
        backend_status(backend_id)
    }

    /// Confirms whether reference generation execution is compiled.
    pub fn require_generation_execution(&self) -> Result<(), EngineError> {
        if cfg!(feature = "cpu-reference") {
            Ok(())
        } else {
            Err(EngineError::not_implemented("native model generation execution is not compiled"))
        }
    }
}

fn validate_absolute_normal_path(path: &Path, label: &str) -> Result<(), EngineError> {
    if !path.is_absolute() {
        return Err(EngineError::invalid_input(format!("{label} must be absolute")));
    }
    if path.components().any(|component| {
        matches!(component, Component::Prefix(_) | Component::CurDir | Component::ParentDir)
    }) {
        return Err(EngineError::invalid_input(format!(
            "{label} must be syntactically normalized"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ed25519_dalek::{Signature, Signer, SigningKey, VerifyingKey};
    use personalai_engine_crypto::{
        control_plane_signature_message, engine_response_signature_message,
    };
    use personalai_engine_protocol::{
        AuthenticatedServiceRequest, AuthenticatedServiceResponse, CancellationEnvelope,
        GenerationRequestEnvelope, RuntimeV3Document, ServiceMessageKind,
        attach_cancellation_signature, attach_generation_signature, attach_service_signature,
    };
    #[cfg(feature = "cpu-reference")]
    use personalai_engine_testkit::DecoderPackageFixtureBuilder;
    use personalai_engine_types::{ContextLimit, ModelHashAlgorithm, ResourceBudget, SignerKeyId};
    #[cfg(all(unix, feature = "cpu-reference", feature = "development-runtime-v3-simulator"))]
    use sovereign_ai_contracts::inference::{
        GenerationParameters as ContractGenerationParameters, InferenceMessage, InferenceRole,
    };
    #[cfg(all(unix, feature = "cpu-reference", feature = "development-runtime-v3-simulator"))]
    use sovereign_ai_contracts::inference_runtime_v3::{
        ExecutionRequestKindV3, ExecutionStatusV3, InferenceExecutionRequestV3,
        decode_inference_execution_result_v3, encode_inference_execution_request_v3,
    };
    #[cfg(all(unix, feature = "cpu-reference"))]
    use sovereign_ai_contracts::inference_runtime_v3::{
        InferenceReservationReleaseV3, InferenceReservationRequestV3,
        InferenceRuntimeContractVersionV3, ReservationDecisionV3, ReservationReleaseKindV3,
        ReservationRequestKindV3, RuntimeResourceBudgetV3,
        decode_inference_reservation_decision_v3, decode_runtime_availability_v3,
        encode_inference_reservation_release_v3, encode_inference_reservation_request_v3,
        runtime_v3_timestamp_from_unix_millis,
    };

    #[test]
    fn config_requires_distinct_normal_local_paths() {
        let config = ServiceConfig::new(
            "/var/lib/personalai/models",
            "/run/user/501/personalai-engine",
            "/run/user/501/personalai-engine/engine.sock",
        )
        .expect("normal paths must validate");
        assert_eq!(config.socket_path(), Path::new("/run/user/501/personalai-engine/engine.sock"));
        assert!(ServiceConfig::new("models", "/run/engine", "/run/engine/engine.sock").is_err());
        assert!(ServiceConfig::new("/models", "/run/engine", "/tmp/engine.sock").is_err());
        assert!(ServiceConfig::new("/models", "/models", "/models/engine.sock").is_err());
        assert!(
            ServiceConfig::new("/models", "/models/runtime", "/models/runtime/engine.sock")
                .is_err()
        );
    }

    #[test]
    fn admission_uses_registered_id_and_reports_compiled_execution_truthfully() {
        let mut service = service();
        let model_id = ModelId::new("model-a").expect("model id must validate");
        service
            .register_configured_model(
                model_id.clone(),
                "model-a.paimodel",
                test_hash(),
                test_signer(),
            )
            .expect("configured model must register");
        let receipt = service.admit(request(1, model_id)).expect("registered id must admit");
        assert_eq!(receipt.request_id().get(), 1);
        assert_eq!(receipt.queued_requests(), 1);
        assert_eq!(service.require_generation_execution().is_ok(), cfg!(feature = "cpu-reference"));
    }

    #[test]
    fn unknown_model_id_fails_closed() {
        let mut service = service();
        let unknown = ModelId::new("unknown").expect("model id must validate");
        assert!(matches!(service.admit(request(1, unknown)), Err(EngineError::InvalidInput(_))));
    }

    #[test]
    fn authenticated_control_session_rejects_replay_after_strict_decode() {
        let signing_key = SigningKey::from_bytes(&[0x23; 32]);
        let key_id = SignerKeyId::from_bytes([0x24; 16]).expect("key ID must validate");
        let trusted =
            TrustedControlPlaneKey::active(key_id, signing_key.verifying_key().to_bytes())
                .expect("control key must validate");
        let generation = request(55, ModelId::new("model-a").expect("model ID must validate"));
        let unsigned = GenerationRequestEnvelope::unsigned_bytes(&generation, 1, key_id)
            .expect("request must encode");
        let transcript =
            control_plane_signature_message(&unsigned).expect("transcript must construct");
        let signature = signing_key.sign(&transcript).to_bytes();
        let payload =
            attach_generation_signature(&unsigned, &signature).expect("signature must attach");
        let frame = Frame::new(FrameKind::GenerationRequest, payload).expect("frame must validate");
        let mut session =
            AuthenticatedControlSession::new(trusted).expect("session key must validate");
        assert!(matches!(
            session.decode_command(&frame),
            Ok(InboundCommand::Generate { nonce: 1, .. })
        ));
        assert_eq!(session.last_nonce(), 1);
        assert!(matches!(
            session.decode_command(&frame),
            Err(EngineError::AuthenticationFailure(_))
        ));
    }

    #[test]
    fn authenticated_control_session_accepts_rotation_but_binds_each_stream_to_one_key() {
        let first_signing_key = SigningKey::from_bytes(&[0x25; 32]);
        let second_signing_key = SigningKey::from_bytes(&[0x26; 32]);
        let first_id = SignerKeyId::from_bytes([0x27; 16]).expect("key ID must validate");
        let second_id = SignerKeyId::from_bytes([0x28; 16]).expect("key ID must validate");
        let trusted_keys = TrustedControlPlaneKeys::new([
            TrustedControlPlaneKey::active(first_id, first_signing_key.verifying_key().to_bytes())
                .expect("first control key must validate"),
            TrustedControlPlaneKey::active(
                second_id,
                second_signing_key.verifying_key().to_bytes(),
            )
            .expect("second control key must validate"),
        ])
        .expect("rotation key ring must validate");
        let generation = request(56, ModelId::new("model-a").expect("model ID must validate"));
        let encoded = signed_generation_frame(&generation, 1, first_id, &first_signing_key);
        let frame = Frame::decode_inbound_exact(&encoded).expect("request frame must decode");
        let mut session = AuthenticatedControlSession::new_with_keys(trusted_keys);
        session.decode_command(&frame).expect("first key must authenticate");
        assert_eq!(session.authenticated_key_id(), Some(first_id));

        let unsigned = CancellationEnvelope::unsigned_bytes(generation.request_id(), 2, second_id)
            .expect("cancellation must encode");
        let transcript =
            control_plane_signature_message(&unsigned).expect("transcript must construct");
        let signature = second_signing_key.sign(&transcript).to_bytes();
        let payload =
            attach_cancellation_signature(&unsigned, &signature).expect("signature must attach");
        let cancellation =
            Frame::new(FrameKind::Cancellation, payload).expect("frame must validate");
        assert!(matches!(
            session.decode_command(&cancellation),
            Err(EngineError::AuthenticationFailure(_))
        ));
        assert_eq!(session.authenticated_key_id(), Some(first_id));
        assert_eq!(session.last_nonce(), 1);
    }

    #[cfg(all(unix, feature = "cpu-reference"))]
    #[test]
    fn runtime_rejects_shared_control_and_response_identity() {
        let mut seed = [0x31; 32];
        let key_id = SignerKeyId::from_bytes([0x32; 16]).expect("key ID must validate");
        let signing_key = SigningKey::from_bytes(&seed);
        let trusted =
            TrustedControlPlaneKey::active(key_id, signing_key.verifying_key().to_bytes())
                .expect("control identity must validate");
        let response = EngineResponseSigner::from_seed(key_id, &mut seed)
            .expect("response identity must validate");
        assert!(matches!(
            CpuEngineRuntime::new(service(), trusted, response),
            Err(EngineError::InvalidInput(_))
        ));
    }

    #[cfg(all(unix, feature = "cpu-reference"))]
    #[test]
    fn runtime_opens_one_exclusive_replay_journal_per_active_rotation_key() {
        use std::fs;
        use std::os::unix::fs::PermissionsExt;
        use std::sync::atomic::AtomicU64;

        static NEXT_DIRECTORY: AtomicU64 = AtomicU64::new(1);
        let sequence = NEXT_DIRECTORY.fetch_add(1, Ordering::Relaxed);
        let runtime_root = PathBuf::from("/tmp")
            .join(format!("personalai-engine-key-rotation-{}-{sequence}", std::process::id()));
        fs::create_dir(&runtime_root).expect("runtime root must create");
        fs::set_permissions(&runtime_root, fs::Permissions::from_mode(0o700))
            .expect("runtime root permissions must set");
        let config = ServiceConfig::new("/models", &runtime_root, runtime_root.join("engine.sock"))
            .expect("service config must validate");
        let service = EngineService::new(
            config.clone(),
            SchedulerLimits::new(8, 1, 1 << 20).expect("limits must validate"),
        )
        .expect("service must construct");
        let first_signing_key = SigningKey::from_bytes(&[0x71; 32]);
        let second_signing_key = SigningKey::from_bytes(&[0x72; 32]);
        let first_id = SignerKeyId::from_bytes([0x73; 16]).expect("key ID must validate");
        let second_id = SignerKeyId::from_bytes([0x74; 16]).expect("key ID must validate");
        let trusted_keys = TrustedControlPlaneKeys::new([
            TrustedControlPlaneKey::active(first_id, first_signing_key.verifying_key().to_bytes())
                .expect("first key must validate"),
            TrustedControlPlaneKey::active(
                second_id,
                second_signing_key.verifying_key().to_bytes(),
            )
            .expect("second key must validate"),
        ])
        .expect("key ring must validate");
        let runtime =
            CpuEngineRuntime::new_with_control_keys(service, trusted_keys, test_response_signer())
                .expect("rotating runtime must construct");
        assert_eq!(runtime.replay_journals.len(), 2);
        runtime
            .replay_journals
            .get(&first_id)
            .expect("first journal must exist")
            .lock()
            .expect("journal lock must be available")
            .admit(99)
            .expect("nonce must persist before rotation");
        let paths: Vec<_> = runtime
            .replay_journals
            .values()
            .map(|journal| {
                journal.lock().expect("journal lock must be available").path().to_path_buf()
            })
            .collect();
        assert!(paths.iter().all(|path| path.is_file()));
        drop(runtime);

        let second_only = TrustedControlPlaneKey::active(
            second_id,
            second_signing_key.verifying_key().to_bytes(),
        )
        .expect("remaining key must validate");
        let service = EngineService::new(
            config.clone(),
            SchedulerLimits::new(8, 1, 1 << 20).expect("limits must validate"),
        )
        .expect("service must reconstruct");
        let retired_runtime = CpuEngineRuntime::new(service, second_only, test_response_signer())
            .expect("runtime must allow first-key retirement");
        assert_eq!(retired_runtime.replay_journals.len(), 1);
        assert!(paths.iter().all(|path| path.is_file()));
        drop(retired_runtime);

        let restored_keys = TrustedControlPlaneKeys::new([
            TrustedControlPlaneKey::active(first_id, first_signing_key.verifying_key().to_bytes())
                .expect("first key must validate"),
            TrustedControlPlaneKey::active(
                second_id,
                second_signing_key.verifying_key().to_bytes(),
            )
            .expect("second key must validate"),
        ])
        .expect("restored overlap must validate");
        let service = EngineService::new(
            config,
            SchedulerLimits::new(8, 1, 1 << 20).expect("limits must validate"),
        )
        .expect("service must reconstruct");
        let restored_runtime =
            CpuEngineRuntime::new_with_control_keys(service, restored_keys, test_response_signer())
                .expect("retired key must reopen its retained journal");
        assert!(matches!(
            restored_runtime
                .replay_journals
                .get(&first_id)
                .expect("first journal must reopen")
                .lock()
                .expect("journal lock must be available")
                .admit(99),
            Err(EngineError::AuthenticationFailure(_))
        ));
        drop(restored_runtime);
        for path in paths {
            fs::remove_file(path).expect("journal must clean");
        }
        fs::remove_dir(runtime_root).expect("runtime root must clean");
    }

    #[cfg(all(unix, feature = "cpu-reference"))]
    #[test]
    fn authenticated_same_stream_cancellation_reaches_active_request() {
        use std::fs;
        use std::io::Write;
        use std::os::unix::fs::PermissionsExt;
        use std::os::unix::net::UnixStream;
        use std::sync::atomic::AtomicU64;
        use std::sync::mpsc;
        use std::thread;

        static NEXT_DIRECTORY: AtomicU64 = AtomicU64::new(1);
        let sequence = NEXT_DIRECTORY.fetch_add(1, Ordering::Relaxed);
        let runtime_root = PathBuf::from("/tmp")
            .join(format!("personalai-engine-cancel-{}-{sequence}", std::process::id()));
        fs::create_dir(&runtime_root).expect("runtime root must create");
        fs::set_permissions(&runtime_root, fs::Permissions::from_mode(0o700))
            .expect("runtime root permissions must set");
        let socket_path = runtime_root.join("engine.sock");
        let config = ServiceConfig::new("/models", &runtime_root, &socket_path)
            .expect("runtime config must validate");
        let listener = UnixFrameListener::bind(&config).expect("runtime socket must bind");

        let signing_key = SigningKey::from_bytes(&[0x51; 32]);
        let key_id = SignerKeyId::from_bytes([0x52; 16]).expect("key ID must validate");
        let trusted_key =
            TrustedControlPlaneKey::active(key_id, signing_key.verifying_key().to_bytes())
                .expect("control key must validate");
        let mut replay =
            DurableReplayJournal::open(&runtime_root, key_id).expect("replay journal must open");
        replay.admit(1).expect("generation nonce must persist");
        let replay_path = replay.path().to_path_buf();
        let replay = Mutex::new(replay);
        let request = request(505, ModelId::new("decoder-cancel").expect("model ID must validate"));
        let generation = signed_generation_frame(&request, 1, key_id, &signing_key);
        let unsigned_cancel = CancellationEnvelope::unsigned_bytes(request.request_id(), 2, key_id)
            .expect("cancellation must encode");
        let cancel_transcript = control_plane_signature_message(&unsigned_cancel)
            .expect("cancellation transcript must construct");
        let cancel_signature = signing_key.sign(&cancel_transcript).to_bytes();
        let cancel_payload = attach_cancellation_signature(&unsigned_cancel, &cancel_signature)
            .expect("cancellation signature must attach");
        let cancellation = Frame::new(FrameKind::Cancellation, cancel_payload)
            .expect("cancellation frame must construct")
            .encode()
            .expect("cancellation frame must encode");

        let client_path = socket_path.clone();
        let (release_client, hold_client) = mpsc::sync_channel(0);
        let client = thread::spawn(move || {
            let mut stream = UnixStream::connect(client_path).expect("client must connect");
            stream.write_all(&generation).expect("generation must write");
            stream.write_all(&cancellation).expect("cancellation must write");
            hold_client.recv().expect("server must release client");
        });
        let mut connection = listener.accept().expect("connection must accept");
        let first = connection.read_inbound_frame().expect("generation frame must read");
        let mut session =
            AuthenticatedControlSession::new(trusted_key).expect("session key must validate");
        assert!(matches!(
            session.decode_command(&first),
            Ok(InboundCommand::Generate { nonce: 1, .. })
        ));
        let cancelled = AtomicBool::new(false);
        let done = AtomicBool::new(false);
        watch_generation_cancellation(
            connection,
            session,
            request.request_id(),
            &cancelled,
            &done,
            &replay,
        );
        assert!(cancelled.load(Ordering::Acquire));
        release_client.send(()).expect("client must be released");
        client.join().expect("client must finish");
        drop(replay);
        drop(listener);
        fs::remove_file(replay_path).expect("replay journal must clean");
        fs::remove_dir(&runtime_root).expect("runtime root must clean");
    }

    #[test]
    fn protocol_truthfully_reports_platform_listener_support() {
        let descriptor = service().protocol_descriptor();
        assert_eq!(descriptor.major, 2);
        assert_eq!(descriptor.listener_implemented, cfg!(unix));
    }

    #[cfg(all(unix, feature = "cpu-reference"))]
    #[test]
    fn bounded_output_queue_fails_closed_when_full_or_disconnected() {
        let (sender, receiver) = std::sync::mpsc::sync_channel(1);
        let frame = Frame::new(FrameKind::GenerationEvent, vec![1])
            .expect("bounded test frame must construct");
        enqueue_outbound_frame(&sender, frame.clone()).expect("first frame must queue");
        assert!(matches!(
            enqueue_outbound_frame(&sender, frame.clone()),
            Err(EngineError::ResourceExhaustion(_))
        ));
        drop(receiver);
        assert!(matches!(
            enqueue_outbound_frame(&sender, frame),
            Err(EngineError::BackendFailure(_))
        ));
    }

    #[cfg(all(unix, feature = "cpu-reference"))]
    #[test]
    fn runtime_generation_metrics_accumulate_exact_timing_usage_and_throughput() {
        let mut metrics = RuntimeMetrics::default();
        record_generation_measurement(
            &mut metrics,
            GenerationMeasurement {
                usage: GenerationUsage::new(2, 3).expect("usage must validate"),
                timing: GenerationTiming {
                    prefill_nanoseconds: 2_000_000_000,
                    decode_nanoseconds: 1_000_000_000,
                },
            },
        )
        .expect("first measurement must record");
        assert_eq!(metrics.prompt_tokens, 2);
        assert_eq!(metrics.generated_tokens, 3);
        assert_eq!(metrics.prefill_nanoseconds, 2_000_000_000);
        assert_eq!(metrics.decode_nanoseconds, 1_000_000_000);
        assert_eq!(metrics.tokens_per_second_milli, 1_000);

        record_generation_measurement(
            &mut metrics,
            GenerationMeasurement {
                usage: GenerationUsage::new(1, 1).expect("usage must validate"),
                timing: GenerationTiming {
                    prefill_nanoseconds: 500_000_000,
                    decode_nanoseconds: 500_000_000,
                },
            },
        )
        .expect("second measurement must record");
        assert_eq!(metrics.prompt_tokens, 3);
        assert_eq!(metrics.generated_tokens, 4);
        assert_eq!(metrics.prefill_nanoseconds, 2_500_000_000);
        assert_eq!(metrics.decode_nanoseconds, 1_500_000_000);
        assert_eq!(metrics.tokens_per_second_milli, 1_000);

        let wire = service_metrics(metrics);
        assert_eq!(wire.values()[5], 3);
        assert_eq!(wire.values()[6], 4);
        assert_eq!(wire.values()[7], 2_500_000_000);
        assert_eq!(wire.values()[8], 1_500_000_000);
        assert_eq!(wire.values()[9], 1_000);
    }

    #[cfg(feature = "cpu-reference")]
    #[test]
    fn signed_decoder_package_loads_and_executes_prompt_tokens_end_to_end() {
        let fixture = DecoderPackageFixtureBuilder::new().build().expect("fixture must build");
        let mut service = service();
        let model_id = ModelId::new("decoder-test").expect("model id must validate");
        service
            .register_configured_model(
                model_id.clone(),
                "decoder-test.paimodel",
                fixture.package_hash(),
                fixture.trusted_signer().clone(),
            )
            .expect("fixture must register");
        let budget = ResourceBudget::new(1 << 24, 1 << 20, 1).expect("budget must validate");
        let loaded = service
            .load_registered_cpu_model_bytes(&model_id, fixture.bytes(), &budget)
            .expect("signed exact decoder must load");
        let tokens = loaded.tokenizer().encode("A").expect("prompt must tokenize");
        let mut session = loaded
            .model()
            .start_session(&budget, RequestId::new(1).unwrap(), SequenceId::new(1).unwrap())
            .expect("session must start");
        let mut logits = None;
        for token in tokens {
            logits = Some(session.forward_token(token).expect("token must execute"));
        }
        let logits = logits.expect("prompt must contain at least one token");
        assert_eq!(logits.shape().dimensions(), &[259]);
        assert_eq!(session.token_count(), 2);
    }

    #[cfg(all(feature = "cpu-reference", feature = "metal", target_os = "macos"))]
    #[test]
    #[allow(clippy::too_many_lines)]
    fn signed_decoder_executes_and_streams_through_real_metal() {
        let Ok(devices) = MetalBackend::devices() else { return };
        if devices.is_empty() {
            return;
        }
        let fixture = DecoderPackageFixtureBuilder::new().build().expect("fixture must build");
        let mut service = service();
        let model_id = ModelId::new("decoder-metal").expect("model ID must validate");
        service
            .register_configured_model(
                model_id.clone(),
                "decoder-metal.paimodel",
                fixture.package_hash(),
                fixture.trusted_signer().clone(),
            )
            .expect("fixture must register");
        let budget = ResourceBudget::new(1 << 25, 1 << 20, 1).expect("budget must validate");
        let cpu = service
            .load_registered_cpu_model_bytes(&model_id, fixture.bytes(), &budget)
            .expect("CPU oracle must load");
        let backend = MetalBackend::new(0, 1 << 24, 16).expect("Metal backend must construct");
        let metal = service
            .load_registered_metal_model_bytes(&model_id, fixture.bytes(), &budget, backend.clone())
            .expect("signed model must load for Metal");
        let tokens = metal.tokenizer().encode("A").expect("prompt must tokenize");
        let mut cpu_session = cpu
            .model()
            .start_session(&budget, RequestId::new(701).unwrap(), SequenceId::new(701).unwrap())
            .expect("CPU session must start");
        let mut metal_session = metal
            .model()
            .start_session(&budget, RequestId::new(702).unwrap(), SequenceId::new(702).unwrap())
            .expect("Metal session must start");
        let cpu_started = Instant::now();
        let cpu_logits = cpu_session.prefill(&tokens).expect("CPU prefill must execute");
        let cpu_prefill = cpu_started.elapsed();
        let metal_started = Instant::now();
        let metal_logits = metal_session.prefill(&tokens).expect("Metal prefill must execute");
        let metal_prefill = metal_started.elapsed();
        for index in 0..259 {
            let expected = cpu_logits.f32_value(index).expect("CPU logit must read");
            let actual = metal_logits.f32_value(index).expect("Metal logit must read");
            let tolerance = 1.0e-4_f32.max(expected.abs() * 1.0e-4);
            assert!((actual - expected).abs() <= tolerance, "logit {index} differs");
        }
        assert!(matches!(metal.model().unload_if_idle(), Err(EngineError::ResourceExhaustion(_))));
        drop(cpu_session);
        drop(metal_session);

        let request = GenerationRequest::new(
            RequestId::new(703).expect("request ID must validate"),
            model_id,
            "A",
            2,
            ContextLimit::new(8).expect("context must validate"),
            budget,
        )
        .expect("request must validate");
        let mut events = Vec::new();
        let generation_started = Instant::now();
        let usage = metal
            .generate(
                &request,
                SequenceId::new(703).expect("sequence ID must validate"),
                SamplingConfig::greedy(),
                &AtomicBool::new(false),
                |event| {
                    events.push(event);
                    Ok(())
                },
            )
            .expect("Metal generation must stream");
        let generation_elapsed = generation_started.elapsed();
        assert_eq!(usage.output_tokens(), 2);
        assert!(events.iter().any(|event| matches!(event, InferenceEvent::TokenGenerated { .. })));
        assert!(matches!(events.last(), Some(InferenceEvent::RequestCompleted { .. })));
        let statistics = backend.statistics().expect("statistics must read");
        assert!(statistics.dispatch_count > 0);
        println!(
            "metal-decoder-tiny prompt_tokens={} output_tokens={} cpu_prefill_ns={} metal_prefill_ns={} metal_generation_ns={} dispatches={} h2d_bytes={} d2h_bytes={} peak_device_bytes={}",
            tokens.len(),
            usage.output_tokens(),
            cpu_prefill.as_nanos(),
            metal_prefill.as_nanos(),
            generation_elapsed.as_nanos(),
            statistics.dispatch_count,
            statistics.host_to_device_bytes,
            statistics.device_to_host_bytes,
            statistics.peak_allocated_bytes,
        );

        let mut cancelled = metal
            .model()
            .start_session(&budget, RequestId::new(704).unwrap(), SequenceId::new(704).unwrap())
            .expect("cancellation session must start");
        cancelled.cancellation().cancel();
        assert!(matches!(cancelled.decode(tokens[0]), Err(EngineError::Cancellation(_))));
        assert_eq!(metal.model().kv_cache_metrics().unwrap().active_sequence_count, 0);
        assert!(metal.model().unload_if_idle().is_ok());
        assert!(matches!(
            metal.model().start_session(
                &budget,
                RequestId::new(705).unwrap(),
                SequenceId::new(705).unwrap()
            ),
            Err(EngineError::BackendFailure(_))
        ));
    }

    #[cfg(feature = "cpu-reference")]
    #[test]
    fn greedy_generation_streams_typed_events_and_honors_cancellation() {
        let fixture = DecoderPackageFixtureBuilder::new().build().expect("fixture must build");
        let mut service = service();
        let model_id = ModelId::new("decoder-generate").expect("model id must validate");
        service
            .register_configured_model(
                model_id.clone(),
                "decoder-generate.paimodel",
                fixture.package_hash(),
                fixture.trusted_signer().clone(),
            )
            .expect("fixture must register");
        let budget = ResourceBudget::new(1 << 24, 1 << 20, 1).expect("budget must validate");
        let loaded = service
            .load_registered_cpu_model_bytes(&model_id, fixture.bytes(), &budget)
            .expect("decoder must load");
        let request = GenerationRequest::new(
            RequestId::new(99).expect("request id must validate"),
            model_id.clone(),
            "A",
            4,
            ContextLimit::new(8).expect("context must validate"),
            budget,
        )
        .expect("request must validate");
        let sequence_id = SequenceId::new(7).expect("sequence id must validate");
        let cancelled = AtomicBool::new(false);
        let mut events = Vec::new();
        let usage = loaded
            .generate_greedy(&request, sequence_id, &cancelled, |event| {
                events.push(event);
                Ok(())
            })
            .expect("generation must complete");
        assert_eq!(events.first().and_then(GenerationEvent::sequence_id), Some(sequence_id));
        assert!(matches!(events.last(), Some(GenerationEvent::Completed { .. })));
        assert!(usage.output_tokens() >= 1);

        cancelled.store(true, Ordering::Release);
        assert!(matches!(
            loaded.generate_greedy(&request, sequence_id, &cancelled, |_| Ok(())),
            Err(EngineError::Cancellation(_))
        ));

        cancelled.store(false, Ordering::Release);
        let scheduled = GenerationRequest::new(
            RequestId::new(100).expect("request id must validate"),
            model_id,
            "B",
            4,
            ContextLimit::new(8).expect("context must validate"),
            budget,
        )
        .expect("request must validate");
        service.admit(scheduled).expect("request must queue");
        let mut scheduled_events = Vec::new();
        let scheduled_usage = service
            .run_next_greedy(&loaded, sequence_id, &cancelled, |event| {
                scheduled_events.push(event);
                Ok(())
            })
            .expect("scheduled generation must run")
            .expect("queued generation must produce usage");
        assert!(scheduled_usage.output_tokens() >= 1);
        assert!(matches!(scheduled_events.last(), Some(GenerationEvent::Completed { .. })));
    }

    #[cfg(feature = "cpu-reference")]
    #[test]
    fn signed_multilayer_fixture_runs_full_sampling_lifecycle_deterministically() {
        let fixture = DecoderPackageFixtureBuilder::new().build().expect("fixture must build");
        let mut service = service();
        let model_id = ModelId::new("decoder-sampling-golden").expect("model ID must validate");
        service
            .register_configured_model(
                model_id.clone(),
                "decoder-sampling-golden.paimodel",
                fixture.package_hash(),
                fixture.trusted_signer().clone(),
            )
            .expect("fixture must register");
        let budget = ResourceBudget::new(1 << 24, 1 << 20, 2).expect("budget must validate");
        let loaded = service
            .load_registered_cpu_model_bytes(&model_id, fixture.bytes(), &budget)
            .expect("signed fixture must load");
        let request = GenerationRequest::new(
            RequestId::new(701).unwrap(),
            model_id,
            "AB",
            4,
            ContextLimit::new(8).unwrap(),
            budget,
        )
        .expect("request must validate");
        let sampling = SamplingConfig::new(0.8)
            .unwrap()
            .with_allowed_token_ids(&[65, 66, 67, 257])
            .unwrap()
            .with_top_k(Some(16))
            .unwrap()
            .with_top_p(0.9)
            .unwrap()
            .with_min_p(0.01)
            .unwrap()
            .with_seed(0x5eed)
            .with_maximum_generated_tokens(4)
            .unwrap();
        let cancelled = AtomicBool::new(false);
        let mut first_events = Vec::new();
        let first = loaded
            .generate(
                &request,
                SequenceId::new(31).unwrap(),
                sampling.clone(),
                &cancelled,
                |event| {
                    first_events.push(event);
                    Ok(())
                },
            )
            .expect("first generation must complete");
        let mut second_events = Vec::new();
        let second = loaded
            .generate(&request, SequenceId::new(32).unwrap(), sampling, &cancelled, |event| {
                second_events.push(event);
                Ok(())
            })
            .expect("second generation must complete");
        let token_ids = |events: &[InferenceEvent]| {
            events
                .iter()
                .filter_map(|event| match event {
                    InferenceEvent::TokenGenerated { token_id, .. } => Some(token_id.get()),
                    _ => None,
                })
                .collect::<Vec<_>>()
        };
        assert_eq!(token_ids(&first_events), token_ids(&second_events));
        assert_eq!(first, second);
        assert!(matches!(first_events.first(), Some(InferenceEvent::RequestAccepted { .. })));
        assert!(
            first_events
                .iter()
                .any(|event| matches!(event, InferenceEvent::PrefillCompleted { .. }))
        );
        assert!(matches!(first_events.last(), Some(InferenceEvent::RequestCompleted { .. })));
        // Golden generated solely by this checked scalar implementation from
        // DecoderPackageFixtureBuilder v1, seed 0x5eed, and the settings above.
        assert_eq!(token_ids(&first_events), vec![67, 66, 65, 257]);

        cancelled.store(true, Ordering::Release);
        let mut cancelled_events = Vec::new();
        assert!(matches!(
            loaded.generate(
                &request,
                SequenceId::new(33).unwrap(),
                SamplingConfig::greedy(),
                &cancelled,
                |event| {
                    cancelled_events.push(event);
                    Ok(())
                },
            ),
            Err(EngineError::Cancellation(_))
        ));
        assert!(matches!(
            cancelled_events.as_slice(),
            [InferenceEvent::RequestAccepted { .. }, InferenceEvent::RequestCancelled { .. }]
        ));
    }

    #[cfg(feature = "cpu-reference")]
    #[test]
    fn resumable_generation_interleaves_two_sequence_local_cursors_fairly() {
        let fixture = DecoderPackageFixtureBuilder::new().build().expect("fixture must build");
        let mut service = service();
        let model_id = ModelId::new("decoder-interleaved").expect("model id must validate");
        service
            .register_configured_model(
                model_id.clone(),
                "decoder-interleaved.paimodel",
                fixture.package_hash(),
                fixture.trusted_signer().clone(),
            )
            .expect("fixture must register");
        let budget = ResourceBudget::new(1 << 24, 1 << 20, 2).expect("budget must validate");
        let loaded = service
            .load_registered_cpu_model_bytes(&model_id, fixture.bytes(), &budget)
            .expect("decoder must load");
        let first_request = GenerationRequest::new(
            RequestId::new(201).expect("request id must validate"),
            model_id.clone(),
            "AAA",
            2,
            ContextLimit::new(8).expect("context must validate"),
            budget,
        )
        .expect("first request must validate");
        let second_request = GenerationRequest::new(
            RequestId::new(202).expect("request id must validate"),
            model_id,
            "B",
            2,
            ContextLimit::new(8).expect("context must validate"),
            budget,
        )
        .expect("second request must validate");
        let first_sequence = SequenceId::new(21).expect("sequence must validate");
        let second_sequence = SequenceId::new(22).expect("sequence must validate");
        let first_cancelled = Arc::new(AtomicBool::new(false));
        let second_cancelled = Arc::new(AtomicBool::new(false));
        let mut first_events = Vec::new();
        let mut second_events = Vec::new();
        let mut completion_order = Vec::new();
        let mut coordinator = GenerationRoundRobin::new(2).expect("coordinator must validate");
        coordinator
            .submit(&loaded, &first_request, first_sequence, Arc::clone(&first_cancelled))
            .expect("first cursor must admit");
        coordinator
            .submit(&loaded, &second_request, second_sequence, Arc::clone(&second_cancelled))
            .expect("second cursor must admit");
        assert!(coordinator.retained_bytes() > 0);

        for _turn in 0..64 {
            let turn = coordinator
                .advance_next()
                .expect("round-robin turn must advance")
                .expect("an active cursor must remain");
            assert!(turn.decoder_tokens_advanced() <= 1);
            if let Some(event) = turn.into_event() {
                if matches!(event, GenerationEvent::Completed { .. }) {
                    completion_order.push(event.request_id());
                }
                if event.request_id() == first_request.request_id() {
                    first_events.push(event);
                } else {
                    assert_eq!(event.request_id(), second_request.request_id());
                    second_events.push(event);
                }
            }
            if coordinator.is_empty() {
                break;
            }
        }

        assert!(coordinator.is_empty());
        assert_eq!(coordinator.active_len(), 0);
        assert_eq!(coordinator.retained_bytes(), 0);
        assert_eq!(
            completion_order,
            [second_request.request_id(), first_request.request_id()],
            "the shorter prompt must complete without waiting for the longer sequence"
        );
        assert!(matches!(first_events.first(), Some(GenerationEvent::Started { .. })));
        assert!(matches!(first_events.last(), Some(GenerationEvent::Completed { .. })));
        assert!(matches!(second_events.first(), Some(GenerationEvent::Started { .. })));
        assert!(matches!(second_events.last(), Some(GenerationEvent::Completed { .. })));
        assert!(matches!(coordinator.advance_next(), Ok(None)));
    }

    #[cfg(feature = "cpu-reference")]
    #[test]
    fn round_robin_admission_enforces_active_and_request_limits() {
        let fixture = DecoderPackageFixtureBuilder::new().build().expect("fixture must build");
        let mut service = service();
        let model_id = ModelId::new("decoder-admission").expect("model ID must validate");
        service
            .register_configured_model(
                model_id.clone(),
                "decoder-admission.paimodel",
                fixture.package_hash(),
                fixture.trusted_signer().clone(),
            )
            .expect("fixture must register");
        let budget_two = ResourceBudget::new(1 << 24, 1 << 20, 2).expect("budget must validate");
        let loaded = service
            .load_registered_cpu_model_bytes(&model_id, fixture.bytes(), &budget_two)
            .expect("decoder must load");
        let first_request = GenerationRequest::new(
            RequestId::new(301).expect("request ID must validate"),
            model_id.clone(),
            "A",
            1,
            ContextLimit::new(8).expect("context must validate"),
            budget_two,
        )
        .expect("request must validate");
        let second_request = GenerationRequest::new(
            RequestId::new(302).expect("request ID must validate"),
            model_id.clone(),
            "B",
            1,
            ContextLimit::new(8).expect("context must validate"),
            budget_two,
        )
        .expect("request must validate");
        let first_cancelled = Arc::new(AtomicBool::new(false));
        let second_cancelled = Arc::new(AtomicBool::new(false));

        assert!(matches!(GenerationRoundRobin::new(0), Err(EngineError::InvalidInput(_))));
        assert!(matches!(
            GenerationRoundRobin::new(MAX_CONCURRENT_SEQUENCES + 1),
            Err(EngineError::InvalidInput(_))
        ));
        let mut byte_limited =
            GenerationRoundRobin::with_limits(2, 1).expect("byte limit must validate");
        assert!(matches!(
            byte_limited.submit(
                &loaded,
                &first_request,
                SequenceId::new(30).expect("sequence must validate"),
                Arc::clone(&first_cancelled),
            ),
            Err(EngineError::ResourceExhaustion(_))
        ));
        let mut capacity_one = GenerationRoundRobin::new(1).expect("limit must validate");
        capacity_one
            .submit(
                &loaded,
                &first_request,
                SequenceId::new(31).expect("sequence must validate"),
                Arc::clone(&first_cancelled),
            )
            .expect("first cursor must admit");
        assert!(matches!(
            capacity_one.submit(
                &loaded,
                &second_request,
                SequenceId::new(32).expect("sequence must validate"),
                Arc::clone(&second_cancelled),
            ),
            Err(EngineError::ResourceExhaustion(_))
        ));

        let budget_one = ResourceBudget::new(1 << 24, 1 << 20, 1).expect("budget must validate");
        let budgeted_request = GenerationRequest::new(
            RequestId::new(303).expect("request ID must validate"),
            model_id,
            "C",
            1,
            ContextLimit::new(8).expect("context must validate"),
            budget_one,
        )
        .expect("request must validate");
        let budget_cancelled = Arc::new(AtomicBool::new(false));
        let mut budget_limited = GenerationRoundRobin::new(2).expect("limit must validate");
        budget_limited
            .submit(
                &loaded,
                &budgeted_request,
                SequenceId::new(33).expect("sequence must validate"),
                Arc::clone(&budget_cancelled),
            )
            .expect("budgeted cursor must admit");
        assert!(matches!(
            budget_limited.submit(
                &loaded,
                &second_request,
                SequenceId::new(34).expect("sequence must validate"),
                Arc::clone(&second_cancelled),
            ),
            Err(EngineError::ResourceExhaustion(_))
        ));
    }

    #[cfg(all(unix, feature = "cpu-reference"))]
    #[test]
    fn round_robin_charges_server_output_reservation_before_cursor_creation() {
        let fixture = DecoderPackageFixtureBuilder::new().build().expect("fixture must build");
        let mut service = service();
        let model_id = ModelId::new("decoder-output-budget").expect("model ID must validate");
        service
            .register_configured_model(
                model_id.clone(),
                "decoder-output-budget.paimodel",
                fixture.package_hash(),
                fixture.trusted_signer().clone(),
            )
            .expect("fixture must register");
        let budget = ResourceBudget::new(1 << 24, 1 << 20, 1).expect("budget must validate");
        let loaded = service
            .load_registered_cpu_model_bytes(&model_id, fixture.bytes(), &budget)
            .expect("decoder must load");
        let request = GenerationRequest::new(
            RequestId::new(320).expect("request ID must validate"),
            model_id,
            "A",
            1,
            ContextLimit::new(8).expect("context must validate"),
            budget,
        )
        .expect("request must validate");
        let required = generation_retained_bytes(&request)
            .expect("base retention must calculate")
            .checked_add(OUTBOUND_PIPELINE_RESERVED_BYTES)
            .expect("output reservation must fit");
        let mut insufficient = GenerationRoundRobin::with_limits(1, required - 1)
            .expect("bounded limit must validate");
        assert!(matches!(
            insufficient.submit_with_retained_bytes(
                &loaded,
                &request,
                SequenceId::new(41).expect("sequence must validate"),
                Arc::new(AtomicBool::new(false)),
                OUTBOUND_PIPELINE_RESERVED_BYTES,
            ),
            Err(EngineError::ResourceExhaustion(_))
        ));
        let mut exact =
            GenerationRoundRobin::with_limits(1, required).expect("exact limit must validate");
        exact
            .submit_with_retained_bytes(
                &loaded,
                &request,
                SequenceId::new(42).expect("sequence must validate"),
                Arc::new(AtomicBool::new(false)),
                OUTBOUND_PIPELINE_RESERVED_BYTES,
            )
            .expect("exact reservation must admit");
        assert_eq!(exact.retained_bytes(), required);
    }

    #[cfg(feature = "cpu-reference")]
    #[test]
    fn round_robin_rejects_duplicate_active_identities() {
        let fixture = DecoderPackageFixtureBuilder::new().build().expect("fixture must build");
        let mut service = service();
        let model_id = ModelId::new("decoder-identities").expect("model ID must validate");
        service
            .register_configured_model(
                model_id.clone(),
                "decoder-identities.paimodel",
                fixture.package_hash(),
                fixture.trusted_signer().clone(),
            )
            .expect("fixture must register");
        let budget = ResourceBudget::new(1 << 24, 1 << 20, 2).expect("budget must validate");
        let loaded = service
            .load_registered_cpu_model_bytes(&model_id, fixture.bytes(), &budget)
            .expect("decoder must load");
        let first_request = GenerationRequest::new(
            RequestId::new(351).expect("request ID must validate"),
            model_id.clone(),
            "A",
            1,
            ContextLimit::new(8).expect("context must validate"),
            budget,
        )
        .expect("request must validate");
        let second_request = GenerationRequest::new(
            RequestId::new(352).expect("request ID must validate"),
            model_id.clone(),
            "B",
            1,
            ContextLimit::new(8).expect("context must validate"),
            budget,
        )
        .expect("request must validate");
        let duplicate_request = GenerationRequest::new(
            first_request.request_id(),
            model_id,
            "C",
            1,
            ContextLimit::new(8).expect("context must validate"),
            budget,
        )
        .expect("duplicate-ID request must otherwise validate");
        let first_cancelled = Arc::new(AtomicBool::new(false));
        let second_cancelled = Arc::new(AtomicBool::new(false));
        let duplicate_cancelled = Arc::new(AtomicBool::new(false));
        let duplicate_sequence = SequenceId::new(35).expect("sequence must validate");
        let mut coordinator = GenerationRoundRobin::new(2).expect("limit must validate");
        coordinator
            .submit(&loaded, &first_request, duplicate_sequence, Arc::clone(&first_cancelled))
            .expect("first identity must admit");
        assert!(matches!(
            coordinator.submit(
                &loaded,
                &duplicate_request,
                SequenceId::new(36).expect("sequence must validate"),
                Arc::clone(&duplicate_cancelled),
            ),
            Err(EngineError::ProtocolViolation(_))
        ));
        assert!(matches!(
            coordinator.submit(
                &loaded,
                &second_request,
                duplicate_sequence,
                Arc::clone(&second_cancelled),
            ),
            Err(EngineError::ProtocolViolation(_))
        ));
    }

    #[cfg(feature = "cpu-reference")]
    #[test]
    fn round_robin_cancellation_is_request_local() {
        let fixture = DecoderPackageFixtureBuilder::new().build().expect("fixture must build");
        let mut service = service();
        let model_id = ModelId::new("decoder-cursor-cancel").expect("model ID must validate");
        service
            .register_configured_model(
                model_id.clone(),
                "decoder-cursor-cancel.paimodel",
                fixture.package_hash(),
                fixture.trusted_signer().clone(),
            )
            .expect("fixture must register");
        let budget = ResourceBudget::new(1 << 24, 1 << 20, 2).expect("budget must validate");
        let loaded = service
            .load_registered_cpu_model_bytes(&model_id, fixture.bytes(), &budget)
            .expect("decoder must load");
        let cancelled_request = GenerationRequest::new(
            RequestId::new(401).expect("request ID must validate"),
            model_id.clone(),
            "AAA",
            2,
            ContextLimit::new(8).expect("context must validate"),
            budget,
        )
        .expect("request must validate");
        let neighbor_request = GenerationRequest::new(
            RequestId::new(402).expect("request ID must validate"),
            model_id,
            "B",
            2,
            ContextLimit::new(8).expect("context must validate"),
            budget,
        )
        .expect("request must validate");
        let cancelled = Arc::new(AtomicBool::new(false));
        let neighbor_cancelled = Arc::new(AtomicBool::new(false));
        let mut coordinator = GenerationRoundRobin::new(2).expect("limit must validate");
        coordinator
            .submit(
                &loaded,
                &cancelled_request,
                SequenceId::new(41).expect("sequence must validate"),
                Arc::clone(&cancelled),
            )
            .expect("cancelled cursor must admit");
        coordinator
            .submit(
                &loaded,
                &neighbor_request,
                SequenceId::new(42).expect("sequence must validate"),
                Arc::clone(&neighbor_cancelled),
            )
            .expect("neighbor cursor must admit");

        for expected_request in [cancelled_request.request_id(), neighbor_request.request_id()] {
            let turn = coordinator
                .advance_next()
                .expect("start turn must advance")
                .expect("cursor must remain active");
            assert_eq!(turn.request_id(), expected_request);
            assert!(matches!(turn.event(), Some(GenerationEvent::Started { .. })));
        }
        cancelled.store(true, Ordering::Release);
        let failed = coordinator
            .advance_next()
            .expect("cancelled turn must advance")
            .expect("cancelled cursor must produce a turn");
        assert_eq!(failed.request_id(), cancelled_request.request_id());
        assert!(matches!(
            failed.event(),
            Some(GenerationEvent::Failed { error: EngineError::Cancellation(_), .. })
        ));
        assert_eq!(coordinator.active_len(), 1);

        let mut neighbor_completed = false;
        for _turn in 0..32 {
            let turn = coordinator
                .advance_next()
                .expect("neighbor turn must advance")
                .expect("neighbor cursor must remain until completion");
            assert_eq!(turn.request_id(), neighbor_request.request_id());
            assert!(turn.decoder_tokens_advanced() <= 1);
            if matches!(turn.event(), Some(GenerationEvent::Completed { .. })) {
                neighbor_completed = true;
                break;
            }
        }
        assert!(neighbor_completed);
        assert!(coordinator.is_empty());
    }

    #[cfg(all(unix, feature = "cpu-reference"))]
    #[test]
    fn authenticated_unix_runtime_executes_signed_request_end_to_end() {
        use std::fs;
        use std::os::unix::fs::PermissionsExt;
        use std::sync::atomic::AtomicU64;
        use std::sync::mpsc;
        use std::thread;

        static NEXT_DIRECTORY: AtomicU64 = AtomicU64::new(1);
        let fixture = DecoderPackageFixtureBuilder::new().build().expect("fixture must build");
        let sequence = NEXT_DIRECTORY.fetch_add(1, Ordering::Relaxed);
        let runtime_root = PathBuf::from("/tmp")
            .join(format!("personalai-engine-runtime-{}-{sequence}", std::process::id()));
        fs::create_dir(&runtime_root).expect("runtime root must create");
        fs::set_permissions(&runtime_root, fs::Permissions::from_mode(0o700))
            .expect("runtime root permissions must set");
        let socket_path = runtime_root.join("engine.sock");
        let config = ServiceConfig::new("/models", &runtime_root, &socket_path)
            .expect("runtime config must validate");
        let mut service = EngineService::new(
            config,
            SchedulerLimits::new(8, 1, 1 << 20).expect("limits must validate"),
        )
        .expect("service must construct");
        let model_id = ModelId::new("decoder-runtime").expect("model ID must validate");
        service
            .register_configured_model(
                model_id.clone(),
                "decoder-runtime.paimodel",
                fixture.package_hash(),
                fixture.trusted_signer().clone(),
            )
            .expect("model must register");
        let signing_key = SigningKey::from_bytes(&[0x41; 32]);
        let key_id = SignerKeyId::from_bytes([0x42; 16]).expect("key ID must validate");
        let trusted_key =
            TrustedControlPlaneKey::active(key_id, signing_key.verifying_key().to_bytes())
                .expect("control key must validate");
        let budget = ResourceBudget::new(1 << 24, 1 << 20, 1).expect("budget must validate");
        let response_signer = test_response_signer();
        let response_public_key = response_signer.public_key();
        let mut runtime = CpuEngineRuntime::new(service, trusted_key, response_signer)
            .expect("distinct runtime identities must validate");
        runtime
            .load_registered_model_bytes(&model_id, fixture.bytes(), &budget)
            .expect("model must load");
        let listener =
            UnixFrameListener::bind(runtime.service().config()).expect("runtime socket must bind");

        let request = GenerationRequest::new(
            RequestId::new(404).expect("request ID must validate"),
            model_id,
            "A",
            2,
            ContextLimit::new(8).expect("context must validate"),
            budget,
        )
        .expect("request must validate");
        let unsigned = GenerationRequestEnvelope::unsigned_bytes(&request, 1, key_id)
            .expect("request must encode");
        let transcript =
            control_plane_signature_message(&unsigned).expect("transcript must construct");
        let signature = signing_key.sign(&transcript).to_bytes();
        let payload =
            attach_generation_signature(&unsigned, &signature).expect("signature must attach");
        let encoded = Frame::new(FrameKind::GenerationRequest, payload)
            .expect("request frame must construct")
            .encode()
            .expect("request frame must encode");
        let replay_encoded = encoded.clone();
        let client_path = socket_path.clone();
        let client = thread::spawn(move || {
            runtime_test_client(client_path, &encoded, response_public_key, 1, 3)
        });
        let outcome = runtime.serve_one(&listener).expect("runtime must serve generation");
        assert!(
            matches!(outcome, ServeOutcome::Generated { request_id, .. } if request_id.get() == 404)
        );
        let event_kinds = client.join().expect("client must finish");
        assert_eq!(event_kinds.first(), Some(&1));
        assert_eq!(event_kinds.last(), Some(&3));
        assert!(event_kinds.contains(&2));

        let replay_client_path = socket_path.clone();
        let (release_replay_client, hold_replay_client) = mpsc::sync_channel(0);
        let replay_client = thread::spawn(move || {
            use std::io::Write;
            use std::os::unix::net::UnixStream;

            let mut stream =
                UnixStream::connect(replay_client_path).expect("replay client must connect");
            stream.write_all(&replay_encoded).expect("replayed request must write");
            hold_replay_client.recv().expect("server must release replay client");
        });
        let replay_outcome = runtime.serve_one(&listener);
        assert!(
            matches!(replay_outcome, Err(EngineError::AuthenticationFailure(_))),
            "replayed command must fail authentication: {replay_outcome:?}"
        );
        release_replay_client.send(()).expect("replay client must be released");
        replay_client.join().expect("replay client must finish");

        let replay_path = runtime_root.join(format!("control-replay-{key_id}.bin"));
        drop(listener);
        drop(runtime);
        fs::remove_file(replay_path).expect("replay journal must clean");
        fs::remove_dir(&runtime_root).expect("runtime root must clean");
    }

    #[cfg(all(unix, feature = "cpu-reference"))]
    #[test]
    fn authenticated_batch_serves_two_signed_streams_and_cleans_scheduler_state() {
        let mut fixture = BatchRuntimeFixture::new("success", 2);
        assert!(matches!(
            fixture.runtime.serve_batch(&fixture.listener, 0),
            Err(EngineError::InvalidInput(_))
        ));
        assert!(matches!(
            fixture.runtime.serve_batch(&fixture.listener, 3),
            Err(EngineError::InvalidInput(_))
        ));
        let first_request = fixture.request(601, "AAA", 2);
        let second_request = fixture.request(602, "B", 2);
        let first = signed_generation_frame(
            &first_request,
            1,
            fixture.control_key_id,
            &fixture.control_signing_key,
        );
        let second = signed_generation_frame(
            &second_request,
            2,
            fixture.control_key_id,
            &fixture.control_signing_key,
        );
        let first_socket = fixture.socket_path();
        let second_socket = fixture.socket_path();
        let response_key = fixture.response_public_key;
        let first_client =
            spawn_ordered_runtime_test_client(first_socket, first, response_key, 1, 3);
        let second_client =
            spawn_ordered_runtime_test_client(second_socket, second, response_key, 2, 3);

        let results =
            fixture.runtime.serve_batch(&fixture.listener, 2).expect("bounded batch must run");
        assert_eq!(results.len(), 2);
        let mut completed_ids: Vec<_> = results
            .into_iter()
            .map(|result| match result.expect("both generations must complete") {
                ServeOutcome::Generated { request_id, .. } => request_id.get(),
                other => panic!("unexpected batch outcome: {other:?}"),
            })
            .collect();
        completed_ids.sort_unstable();
        assert_eq!(completed_ids, [601, 602]);
        assert_eq!(first_client.join().expect("first client must finish").last(), Some(&3));
        assert_eq!(second_client.join().expect("second client must finish").last(), Some(&3));
        assert_eq!(fixture.runtime.service.scheduler.queued_len(), 0);
        assert_eq!(fixture.runtime.service.scheduler.active_len(), 0);
        fixture.cleanup();
    }

    #[cfg(all(unix, feature = "metal", target_os = "macos"))]
    #[test]
    fn authenticated_service_streams_through_owner_selected_metal_backend() {
        let backend = MetalBackend::new(0, 1 << 24, 16).expect("Metal backend must construct");
        let mut fixture = BatchRuntimeFixture::new_with_backend(
            "metal-service",
            1,
            RuntimeBackend::Metal(backend),
        );
        let capabilities = service_capabilities(&fixture.runtime.backend)
            .expect("selected capabilities must encode");
        assert!(
            capabilities
                .contains(EngineCapability::CpuReference)
                .expect("CPU capability must decode")
        );
        assert!(
            capabilities.contains(EngineCapability::Metal).expect("Metal capability must decode")
        );
        let descriptor = model_descriptor_parts(
            &fixture.runtime.service,
            &fixture.runtime.models,
            &fixture.model_id,
        )
        .expect("descriptor must construct");
        let ServiceResponseBody::ModelDescriptor(descriptor) = descriptor else {
            panic!("descriptor response kind changed");
        };
        assert_eq!(descriptor.backend_id().as_str(), "metal");
        assert!(
            fixture.runtime.metrics().expect("Metal metrics must read").device_memory_bytes > 0
        );

        let request = fixture.request(609, "A", 2);
        let signed = signed_generation_frame(
            &request,
            1,
            fixture.control_key_id,
            &fixture.control_signing_key,
        );
        let client = spawn_ordered_runtime_test_client(
            fixture.socket_path(),
            signed,
            fixture.response_public_key,
            1,
            3,
        );
        let result = fixture
            .runtime
            .serve_batch(&fixture.listener, 1)
            .expect("Metal service batch must run");
        assert!(matches!(
            result.as_slice(),
            [Ok(ServeOutcome::Generated { request_id, .. })] if request_id.get() == 609
        ));
        assert_eq!(client.join().expect("Metal client must finish").last(), Some(&3));
        fixture.cleanup();
    }

    #[cfg(all(unix, feature = "arm64", target_arch = "aarch64"))]
    #[test]
    fn authenticated_service_streams_through_owner_selected_arm64_backend() {
        let mut fixture = BatchRuntimeFixture::new_with_backend(
            "arm64-service",
            1,
            RuntimeBackend::arm64().expect("ARM64 backend must construct"),
        );
        let capabilities = service_capabilities(&fixture.runtime.backend)
            .expect("selected capabilities must encode");
        assert!(
            capabilities
                .contains(EngineCapability::CpuReference)
                .expect("CPU fallback capability must decode")
        );
        assert!(
            capabilities.contains(EngineCapability::Arm64).expect("ARM64 capability must decode")
        );
        let descriptor = model_descriptor_parts(
            &fixture.runtime.service,
            &fixture.runtime.models,
            &fixture.model_id,
        )
        .expect("descriptor must construct");
        let ServiceResponseBody::ModelDescriptor(descriptor) = descriptor else {
            panic!("descriptor response kind changed");
        };
        assert_eq!(descriptor.backend_id().as_str(), "arm64");
        assert_eq!(
            fixture.runtime.metrics().expect("ARM64 metrics must read").device_memory_bytes,
            0
        );

        let request = fixture.request(608, "A", 2);
        let signed = signed_generation_frame(
            &request,
            1,
            fixture.control_key_id,
            &fixture.control_signing_key,
        );
        let client = spawn_ordered_runtime_test_client(
            fixture.socket_path(),
            signed,
            fixture.response_public_key,
            1,
            3,
        );
        let result = fixture
            .runtime
            .serve_batch(&fixture.listener, 1)
            .expect("ARM64 service batch must run");
        assert!(matches!(
            result.as_slice(),
            [Ok(ServeOutcome::Generated { request_id, .. })] if request_id.get() == 608
        ));
        assert_eq!(client.join().expect("ARM64 client must finish").last(), Some(&3));
        fixture.cleanup();
    }

    #[cfg(all(unix, feature = "cpu-reference"))]
    #[test]
    fn authenticated_batch_isolates_cancellation_and_completes_neighbor() {
        let mut fixture = BatchRuntimeFixture::new("cancel", 3);
        let cancelled_request = fixture.request(611, "AAA", 4);
        let neighbor_request = fixture.request(612, "B", 2);
        let cancelled_stream = signed_generation_frame(
            &cancelled_request,
            10,
            fixture.control_key_id,
            &fixture.control_signing_key,
        );
        let cancellation = signed_cancellation_frame(
            cancelled_request.request_id(),
            11,
            fixture.control_key_id,
            &fixture.control_signing_key,
        );
        let neighbor_stream = signed_generation_frame(
            &neighbor_request,
            12,
            fixture.control_key_id,
            &fixture.control_signing_key,
        );
        let cancelled_socket = fixture.socket_path();
        let cancellation_socket = fixture.socket_path();
        let neighbor_socket = fixture.socket_path();
        let response_key = fixture.response_public_key;
        let cancelled_client = spawn_ordered_runtime_test_client(
            cancelled_socket,
            cancelled_stream,
            response_key,
            10,
            4,
        );
        let cancellation_client =
            spawn_ordered_write_only_client(cancellation_socket, cancellation);
        let neighbor_client = spawn_ordered_runtime_test_client(
            neighbor_socket,
            neighbor_stream,
            response_key,
            12,
            3,
        );

        let results =
            fixture.runtime.serve_batch(&fixture.listener, 3).expect("cancellation batch must run");
        assert!(
            results.iter().any(|result| matches!(result, Err(EngineError::Cancellation(_)))),
            "cancelled stream must fail locally: {results:?}"
        );
        assert!(results.iter().any(|result| {
            matches!(result, Ok(ServeOutcome::CancelledActive(request_id)) if request_id.get() == 611)
        }));
        assert!(results.iter().any(|result| {
            matches!(result, Ok(ServeOutcome::Generated { request_id, .. }) if request_id.get() == 612)
        }));
        assert_eq!(cancelled_client.join().expect("cancelled client must finish").last(), Some(&4));
        assert!(cancellation_client.join().expect("cancellation client must finish").is_empty());
        assert_eq!(neighbor_client.join().expect("neighbor client must finish").last(), Some(&3));
        assert_eq!(fixture.runtime.service.scheduler.active_len(), 0);
        fixture.cleanup();
    }

    #[cfg(all(unix, feature = "cpu-reference"))]
    #[test]
    fn malformed_batch_peer_does_not_block_authenticated_neighbor() {
        use std::io::{Read, Write};
        use std::os::unix::net::UnixStream;
        use std::sync::mpsc;
        use std::thread;

        let mut fixture = BatchRuntimeFixture::new("malformed", 2);
        let malformed_request = fixture.request(621, "A", 2);
        let valid_request = fixture.request(622, "B", 2);
        let mut malformed = signed_generation_frame(
            &malformed_request,
            20,
            fixture.control_key_id,
            &fixture.control_signing_key,
        );
        let final_byte = malformed.last_mut().expect("signed frame must not be empty");
        *final_byte ^= 0x01;
        let valid = signed_generation_frame(
            &valid_request,
            21,
            fixture.control_key_id,
            &fixture.control_signing_key,
        );
        let malformed_socket = fixture.socket_path();
        let valid_socket = fixture.socket_path();
        let response_key = fixture.response_public_key;
        let (malformed_connected, wait_for_malformed) = mpsc::sync_channel(0);
        let malformed_client = thread::spawn(move || {
            let mut stream = UnixStream::connect(malformed_socket).expect("client must connect");
            malformed_connected.send(()).expect("connection order must synchronize");
            stream.write_all(&malformed).expect("malformed frame must write");
            let mut response = Vec::new();
            stream.read_to_end(&mut response).expect("server close must be observable");
            response
        });
        wait_for_malformed.recv().expect("malformed client must connect first");
        let valid_client =
            spawn_ordered_runtime_test_client(valid_socket, valid, response_key, 21, 3);

        let results = fixture
            .runtime
            .serve_batch(&fixture.listener, 2)
            .expect("malformed peer must remain request-local");
        assert!(
            results
                .iter()
                .any(|result| { matches!(result, Err(EngineError::AuthenticationFailure(_))) })
        );
        assert!(results.iter().any(|result| {
            matches!(result, Ok(ServeOutcome::Generated { request_id, .. }) if request_id.get() == 622)
        }));
        assert!(malformed_client.join().expect("malformed client must finish").is_empty());
        assert_eq!(valid_client.join().expect("valid client must finish").last(), Some(&3));
        assert_eq!(fixture.runtime.service.scheduler.active_len(), 0);
        fixture.cleanup();
    }

    #[cfg(all(unix, feature = "cpu-reference"))]
    #[test]
    fn disconnected_batch_peer_does_not_abort_authenticated_neighbor() {
        let mut fixture = BatchRuntimeFixture::new("disconnect", 2);
        let disconnected_request = fixture.request(631, "AAA", 4);
        let neighbor_request = fixture.request(632, "B", 2);
        let disconnected = signed_generation_frame(
            &disconnected_request,
            30,
            fixture.control_key_id,
            &fixture.control_signing_key,
        );
        let neighbor = signed_generation_frame(
            &neighbor_request,
            31,
            fixture.control_key_id,
            &fixture.control_signing_key,
        );
        let disconnected_client =
            spawn_ordered_disconnecting_client(fixture.socket_path(), disconnected);
        let neighbor_client = spawn_ordered_runtime_test_client(
            fixture.socket_path(),
            neighbor,
            fixture.response_public_key,
            31,
            3,
        );

        let results = fixture
            .runtime
            .serve_batch(&fixture.listener, 2)
            .expect("disconnect must remain request-local");
        assert!(results.iter().any(|result| {
            matches!(result, Err(EngineError::Cancellation(_) | EngineError::BackendFailure(_)))
        }));
        assert!(results.iter().any(|result| {
            matches!(result, Ok(ServeOutcome::Generated { request_id, .. }) if request_id.get() == 632)
        }));
        disconnected_client.join().expect("disconnected client must finish");
        assert_eq!(neighbor_client.join().expect("neighbor client must finish").last(), Some(&3));
        assert_eq!(fixture.runtime.service.scheduler.active_len(), 0);
        fixture.cleanup();
    }

    #[cfg(all(unix, feature = "cpu-reference"))]
    #[test]
    fn continuous_reference_serves_ready_peers_and_drains_on_caller_shutdown() {
        use std::thread;

        let mut fixture = BatchRuntimeFixture::new("continuous", 1);
        let first_request = fixture.request(641, "A", 2);
        let second_request = fixture.request(642, "B", 2);
        let first = signed_generation_frame(
            &first_request,
            40,
            fixture.control_key_id,
            &fixture.control_signing_key,
        );
        let second = signed_generation_frame(
            &second_request,
            41,
            fixture.control_key_id,
            &fixture.control_signing_key,
        );
        let first_socket = fixture.socket_path();
        let second_socket = fixture.socket_path();
        let response_key = fixture.response_public_key;
        let shutdown = ContinuousShutdown::new();
        let request_shutdown = shutdown.clone();
        let first_client =
            spawn_ordered_runtime_test_client(first_socket, first, response_key, 40, 3);
        let second_client =
            spawn_ordered_runtime_test_client(second_socket, second, response_key, 41, 3);
        let shutdown_thread = thread::spawn(move || {
            thread::sleep(std::time::Duration::from_millis(200));
            request_shutdown.request();
        });

        let summary = fixture
            .runtime
            .serve_continuous(
                &fixture.listener,
                ContinuousServeLimits::new(2).expect("continuous limits must validate"),
                &shutdown,
            )
            .expect("continuous reference must drain");
        shutdown_thread.join().expect("shutdown request must finish");
        assert_eq!(summary.completed_generations(), 2, "continuous summary: {summary:?}");
        let first_events = first_client.join().expect("first client must finish");
        let second_events = second_client.join().expect("second client must finish");
        assert_eq!(first_events.last(), Some(&3));
        assert_eq!(second_events.last(), Some(&3));
        assert_eq!(summary.accepted_connections(), 2);
        assert_eq!(summary.authenticated_commands(), 2);
        assert_eq!(summary.completed_generations(), 2);
        assert_eq!(summary.failed_connections(), 0);
        assert_eq!(summary.peak_owned_connections(), 2);
        fixture
            .listener
            .set_nonblocking(true)
            .expect("listener mode must be restored before reuse");
        assert!(fixture.listener.try_accept().expect("ready check must succeed").is_none());
        fixture.listener.set_nonblocking(false).expect("listener mode must return to blocking");
        assert_eq!(fixture.runtime.service.scheduler.queued_len(), 0);
        assert_eq!(fixture.runtime.service.scheduler.active_len(), 0);
        fixture.cleanup();
    }

    #[cfg(all(unix, feature = "metal", target_os = "macos"))]
    #[test]
    fn continuous_service_streams_owner_selected_metal_and_drains() {
        use std::thread;

        let backend = MetalBackend::new(0, 1 << 24, 16).expect("Metal backend must construct");
        let mut fixture = BatchRuntimeFixture::new_with_backend(
            "continuous-metal",
            1,
            RuntimeBackend::Metal(backend),
        );
        let request = fixture.request(648, "A", 2);
        let signed = signed_generation_frame(
            &request,
            60,
            fixture.control_key_id,
            &fixture.control_signing_key,
        );
        let client = spawn_ordered_runtime_test_client(
            fixture.socket_path(),
            signed,
            fixture.response_public_key,
            60,
            3,
        );
        let shutdown = ContinuousShutdown::new();
        let request_shutdown = shutdown.clone();
        let shutdown_thread = thread::spawn(move || {
            thread::sleep(std::time::Duration::from_millis(200));
            request_shutdown.request();
        });
        let summary = fixture
            .runtime
            .serve_continuous(
                &fixture.listener,
                ContinuousServeLimits::new(1).expect("continuous limit must validate"),
                &shutdown,
            )
            .expect("continuous Metal service must drain");
        shutdown_thread.join().expect("Metal shutdown request must finish");
        assert_eq!(summary.completed_generations(), 1);
        assert_eq!(summary.failed_connections(), 0);
        assert_eq!(client.join().expect("Metal client must finish").last(), Some(&3));
        fixture.cleanup();
    }

    #[cfg(all(unix, feature = "cpu-reference"))]
    #[test]
    fn continuous_reference_serves_authenticated_control_and_generation_together() {
        use std::io::{Read, Write};
        use std::os::unix::net::UnixStream;
        use std::thread;

        let mut fixture = BatchRuntimeFixture::new("continuous-control", 1);
        let generation_request = fixture.request(643, "A", 2);
        let generation = signed_generation_frame(
            &generation_request,
            50,
            fixture.control_key_id,
            &fixture.control_signing_key,
        );
        let service_request_id = RequestId::new(644).expect("service request ID");
        let unsigned = AuthenticatedServiceRequest::unsigned_bytes(
            service_request_id,
            51,
            fixture.control_key_id,
            &ServiceRequestBody::Health,
        )
        .expect("health request must encode");
        let transcript = control_plane_signature_message(&unsigned).expect("transcript");
        let signature = fixture.control_signing_key.sign(&transcript).to_bytes();
        let service_payload =
            attach_service_signature(&unsigned, &signature).expect("signature must attach");
        let service_frame = Frame::new(FrameKind::ServiceRequest, service_payload)
            .expect("service frame")
            .encode()
            .expect("service frame must encode");

        let generation_client = spawn_ordered_runtime_test_client(
            fixture.socket_path(),
            generation,
            fixture.response_public_key,
            50,
            3,
        );
        let socket_path = fixture.socket_path();
        let response_public_key = fixture.response_public_key;
        let service_client = thread::spawn(move || {
            let mut stream = UnixStream::connect(socket_path).expect("client must connect");
            configure_runtime_test_client(&stream);
            stream.write_all(&service_frame).expect("health request must write");
            let mut header = [0_u8; FRAME_HEADER_BYTES];
            stream.read_exact(&mut header).expect("health response header");
            let payload_length = usize::try_from(u32::from_le_bytes(
                header[12..16].try_into().expect("payload length field"),
            ))
            .expect("payload length must address");
            let mut encoded = header.to_vec();
            encoded.resize(FRAME_HEADER_BYTES + payload_length, 0);
            stream.read_exact(&mut encoded[FRAME_HEADER_BYTES..]).expect("health response payload");
            let response = Frame::decode_outbound_exact(&encoded).expect("response frame");
            let engine_key_id = SignerKeyId::from_bytes([0x62; 16]).expect("engine key ID");
            AuthenticatedServiceResponse::decode(
                response.payload(),
                service_request_id,
                51,
                engine_key_id,
                &response_public_key,
            )
            .expect("health response must authenticate")
        });
        let shutdown = ContinuousShutdown::new();
        let request_shutdown = shutdown.clone();
        let shutdown_thread = thread::spawn(move || {
            thread::sleep(std::time::Duration::from_millis(200));
            request_shutdown.request();
        });

        let summary = fixture
            .runtime
            .serve_continuous(
                &fixture.listener,
                ContinuousServeLimits::new(2).expect("continuous limits must validate"),
                &shutdown,
            )
            .expect("continuous service must handle control and generation");
        shutdown_thread.join().expect("shutdown request must finish");
        assert_eq!(generation_client.join().expect("generation client").last(), Some(&3));
        let response = service_client.join().expect("service client must finish");
        assert_eq!(response.kind(), ServiceMessageKind::Health);
        assert_eq!(response.body(), &ServiceResponseBody::Health(ServiceHealthStatus::Ready));
        assert_eq!(summary.accepted_connections(), 2);
        assert_eq!(summary.authenticated_commands(), 2);
        assert_eq!(summary.completed_generations(), 1);
        assert_eq!(summary.failed_connections(), 0);
        fixture.cleanup();
    }

    #[cfg(all(unix, feature = "cpu-reference"))]
    #[test]
    fn continuous_reference_applies_authenticated_idle_model_unload() {
        use std::io::{Read, Write};
        use std::os::unix::net::UnixStream;
        use std::thread;

        let mut fixture = BatchRuntimeFixture::new("continuous-unload", 1);
        let request_id = RequestId::new(645).expect("service request ID");
        let unsigned = AuthenticatedServiceRequest::unsigned_bytes(
            request_id,
            1,
            fixture.control_key_id,
            &ServiceRequestBody::UnloadModel(fixture.model_id.clone()),
        )
        .expect("unload request must encode");
        let transcript = control_plane_signature_message(&unsigned).expect("transcript");
        let signature = fixture.control_signing_key.sign(&transcript).to_bytes();
        let payload = attach_service_signature(&unsigned, &signature).expect("signature");
        let encoded = Frame::new(FrameKind::ServiceRequest, payload)
            .expect("service frame")
            .encode()
            .expect("service frame must encode");
        let socket_path = fixture.socket_path();
        let response_public_key = fixture.response_public_key;
        let shutdown = ContinuousShutdown::new();
        let request_shutdown = shutdown.clone();
        let client = thread::spawn(move || {
            let mut stream = UnixStream::connect(socket_path).expect("client must connect");
            configure_runtime_test_client(&stream);
            stream.write_all(&encoded).expect("unload request must write");
            let mut header = [0_u8; FRAME_HEADER_BYTES];
            stream.read_exact(&mut header).expect("unload response header");
            let payload_length = usize::try_from(u32::from_le_bytes(
                header[12..16].try_into().expect("payload length field"),
            ))
            .expect("payload length must address");
            let mut response_bytes = header.to_vec();
            response_bytes.resize(FRAME_HEADER_BYTES + payload_length, 0);
            stream
                .read_exact(&mut response_bytes[FRAME_HEADER_BYTES..])
                .expect("unload response payload");
            let response =
                Frame::decode_outbound_exact(&response_bytes).expect("response frame must decode");
            let engine_key_id = SignerKeyId::from_bytes([0x62; 16]).expect("engine key ID");
            let verified = AuthenticatedServiceResponse::decode(
                response.payload(),
                request_id,
                1,
                engine_key_id,
                &response_public_key,
            )
            .expect("unload response must authenticate");
            request_shutdown.request();
            verified
        });

        let summary = fixture
            .runtime
            .serve_continuous(
                &fixture.listener,
                ContinuousServeLimits::new(1).expect("continuous limits must validate"),
                &shutdown,
            )
            .expect("continuous service must handle model unload");
        let response = client.join().expect("service client must finish");
        assert_eq!(response.kind(), ServiceMessageKind::UnloadModel);
        assert_eq!(response.body(), &ServiceResponseBody::ModelUnloaded);
        assert_eq!(fixture.runtime.health(), ServiceHealth::Degraded);
        assert_eq!(fixture.runtime.metrics().expect("metrics").loaded_models, 0);
        assert_eq!(summary.accepted_connections(), 1);
        assert_eq!(summary.authenticated_commands(), 1);
        assert_eq!(summary.failed_connections(), 0);
        fixture.cleanup();
    }

    #[cfg(all(unix, feature = "cpu-reference"))]
    #[test]
    fn model_unload_rejects_queued_scheduler_ownership_then_releases_cleanly() {
        let mut fixture = BatchRuntimeFixture::new("unload-ownership", 1);
        let request = fixture.request(646, "A", 2);
        let request_id = request.request_id();
        fixture.runtime.service.admit(request).expect("request must queue");
        assert!(matches!(
            fixture.runtime.unload_model(&fixture.model_id),
            Err(EngineError::ResourceExhaustion(_))
        ));
        assert_eq!(
            fixture
                .runtime
                .service
                .scheduler
                .cancel(&request_id)
                .expect("queued request must cancel"),
            CancellationOutcome::Queued
        );
        fixture
            .runtime
            .unload_model(&fixture.model_id)
            .expect("idle model must unload after scheduler release");
        assert_eq!(fixture.runtime.health(), ServiceHealth::Degraded);
        fixture.cleanup();
    }

    #[cfg(all(unix, feature = "cpu-reference"))]
    #[test]
    fn multiple_registered_models_have_independent_residency_and_unload_state() {
        let mut fixture = BatchRuntimeFixture::new("multi-model-residency", 2);
        let second_fixture = DecoderPackageFixtureBuilder::new().build().expect("fixture");
        let second_id = ModelId::new("decoder-cold").expect("second model ID");
        fixture
            .runtime
            .service
            .register_configured_model(
                second_id.clone(),
                "decoder-cold.paimodel",
                second_fixture.package_hash(),
                second_fixture.trusted_signer().clone(),
            )
            .expect("second model must register");
        fixture
            .runtime
            .load_registered_model_bytes(&second_id, second_fixture.bytes(), &fixture.budget)
            .expect("second model must load independently");
        assert_eq!(fixture.runtime.metrics().expect("metrics").loaded_models, 2);
        assert!(fixture.runtime.models.contains_key(fixture.model_id.as_str()));
        assert!(fixture.runtime.models.contains_key(second_id.as_str()));

        fixture.runtime.unload_model(&second_id).expect("second model must unload");
        assert_eq!(fixture.runtime.metrics().expect("metrics").loaded_models, 1);
        assert!(fixture.runtime.models.contains_key(fixture.model_id.as_str()));
        assert!(!fixture.runtime.models.contains_key(second_id.as_str()));
        assert!(matches!(
            fixture.runtime.unload_model(&second_id),
            Err(EngineError::InvalidInput(_))
        ));
        fixture.cleanup();
    }

    #[cfg(all(unix, feature = "cpu-reference"))]
    #[test]
    fn continuous_shutdown_interrupts_an_incomplete_authenticated_ingress_read() {
        use std::io::Read;
        use std::os::unix::net::UnixStream;
        use std::sync::mpsc;
        use std::thread;
        use std::time::Duration;

        let mut fixture = BatchRuntimeFixture::new("continuous-shutdown", 1);
        let socket = fixture.socket_path();
        let (connected, wait_for_connection) = mpsc::sync_channel(0);
        let client = thread::spawn(move || {
            let mut stream = UnixStream::connect(socket).expect("silent client must connect");
            connected.send(()).expect("connection must synchronize");
            let mut response = Vec::new();
            stream.read_to_end(&mut response).expect("shutdown must close owned ingress");
            response
        });
        wait_for_connection.recv().expect("silent client must be connected");
        let shutdown = ContinuousShutdown::new();
        let request_shutdown = shutdown.clone();
        let shutdown_thread = thread::spawn(move || {
            thread::sleep(Duration::from_millis(20));
            request_shutdown.request();
        });

        let summary = fixture
            .runtime
            .serve_continuous(
                &fixture.listener,
                ContinuousServeLimits::new(1).expect("continuous limits must validate"),
                &shutdown,
            )
            .expect("incomplete ingress must drain");
        shutdown_thread.join().expect("shutdown request must finish");
        assert!(client.join().expect("silent client must finish").is_empty());
        assert_eq!(summary.accepted_connections(), 1);
        assert_eq!(summary.authenticated_commands(), 0);
        assert_eq!(summary.failed_connections(), 1);
        assert_eq!(fixture.runtime.service.scheduler.queued_len(), 0);
        assert_eq!(fixture.runtime.service.scheduler.active_len(), 0);
        fixture.cleanup();
    }

    #[cfg(all(unix, feature = "cpu-reference"))]
    #[test]
    fn continuous_peer_credential_rejection_is_counted_and_request_local() {
        use std::os::unix::net::UnixStream;
        use std::sync::mpsc;
        use std::thread;
        use std::time::Duration;

        let mut fixture = BatchRuntimeFixture::new("continuous-peer-identity", 1);
        fixture.listener.reject_current_peer_for_test();
        let socket = fixture.socket_path();
        let (connected, wait_for_connection) = mpsc::sync_channel(0);
        let client = thread::spawn(move || {
            drop(UnixStream::connect(socket).expect("rejected peer must connect"));
            connected.send(()).expect("rejected peer must synchronize");
        });
        wait_for_connection.recv().expect("rejected peer must be ready");
        let shutdown = ContinuousShutdown::new();
        let request_shutdown = shutdown.clone();
        let shutdown_thread = thread::spawn(move || {
            thread::sleep(Duration::from_millis(20));
            request_shutdown.request();
        });

        let summary = fixture
            .runtime
            .serve_continuous(
                &fixture.listener,
                ContinuousServeLimits::new(1).expect("continuous limits must validate"),
                &shutdown,
            )
            .expect("credential rejection must not stop continuous service");
        client.join().expect("rejected peer must finish");
        shutdown_thread.join().expect("shutdown request must finish");
        assert_eq!(summary.accepted_connections(), 1);
        assert_eq!(summary.authenticated_commands(), 0);
        assert_eq!(summary.failed_connections(), 1);
        assert_eq!(fixture.runtime.metrics().expect("metrics").authentication_failures, 1);
        assert_eq!(fixture.runtime.drain_security_events(), [SecurityEvent::UnauthorizedClient]);
        assert_eq!(fixture.runtime.service.scheduler.queued_len(), 0);
        assert_eq!(fixture.runtime.service.scheduler.active_len(), 0);
        fixture.cleanup();
    }

    #[cfg(all(unix, feature = "cpu-reference"))]
    #[test]
    fn continuous_invalid_ingress_emits_exact_typed_security_telemetry() {
        use std::io::{Read, Write};
        use std::os::unix::net::UnixStream;
        use std::thread;

        fn assert_event(label: &str, encoded: Vec<u8>, expected: SecurityEvent) {
            let mut fixture = BatchRuntimeFixture::new(label, 1);
            let socket = fixture.socket_path();
            let shutdown = ContinuousShutdown::new();
            let request_shutdown = shutdown.clone();
            let client = thread::spawn(move || {
                let mut stream = UnixStream::connect(socket).expect("client must connect");
                configure_runtime_test_client(&stream);
                stream.write_all(&encoded).expect("invalid frame must write");
                let mut response = Vec::new();
                stream.read_to_end(&mut response).expect("rejection must close the stream");
                request_shutdown.request();
                response
            });
            let summary = fixture
                .runtime
                .serve_continuous(
                    &fixture.listener,
                    ContinuousServeLimits::new(1).expect("continuous limits must validate"),
                    &shutdown,
                )
                .expect("invalid ingress must remain connection-local");
            assert!(client.join().expect("client must finish").is_empty());
            assert_eq!(summary.accepted_connections(), 1);
            assert_eq!(summary.authenticated_commands(), 0);
            assert_eq!(summary.failed_connections(), 1);
            assert_eq!(fixture.runtime.metrics().expect("metrics").protocol_violations, 1);
            assert_eq!(fixture.runtime.drain_security_events(), [expected]);
            fixture.cleanup();
        }

        let mut malformed = vec![0_u8; FRAME_HEADER_BYTES + 1];
        malformed[12..16].copy_from_slice(&1_u32.to_le_bytes());
        assert_event("continuous-malformed", malformed, SecurityEvent::MalformedFrame);

        let mut wrong_version = Frame::new(FrameKind::Capability, vec![1])
            .expect("frame")
            .encode()
            .expect("encoded frame");
        wrong_version[4..6].copy_from_slice(&99_u16.to_le_bytes());
        assert_event("continuous-version", wrong_version, SecurityEvent::UnknownProtocolVersion);

        let mut oversized = vec![0_u8; FRAME_HEADER_BYTES];
        oversized[..4].copy_from_slice(&personalai_engine_protocol::FRAME_MAGIC);
        oversized[4..6].copy_from_slice(&PROTOCOL_MAJOR.to_le_bytes());
        oversized[6..8].copy_from_slice(&PROTOCOL_MINOR.to_le_bytes());
        oversized[8] = FrameKind::Capability as u8;
        let declared = u32::try_from(personalai_engine_protocol::MAX_FRAME_PAYLOAD_BYTES + 1)
            .expect("frame cap must fit u32");
        oversized[12..16].copy_from_slice(&declared.to_le_bytes());
        assert_event("continuous-oversized", oversized, SecurityEvent::OversizedFrame);
    }

    #[cfg(all(unix, feature = "cpu-reference"))]
    #[test]
    fn bounded_continuous_soak_isolates_delayed_readers_across_restarts() {
        use std::sync::mpsc;
        use std::thread;
        use std::time::Duration;

        const SOAK_CYCLES: u128 = 8;
        const FAST_CLIENTS_PER_CYCLE: u128 = 5;
        const CONNECTIONS_PER_CYCLE: u32 = 6;

        for cycle in 0..SOAK_CYCLES {
            let mut fixture = BatchRuntimeFixture::new("continuous-soak", 3);
            let nonce_base = 1_000_u64
                .checked_add(u64::try_from(cycle).expect("cycle must fit") * 100)
                .expect("soak nonce must fit");
            let request_base =
                10_000_u128.checked_add(cycle * 100).expect("soak request ID must fit");
            let delayed_request = fixture.request(request_base, "S", 2);
            let delayed_frame = signed_generation_frame(
                &delayed_request,
                nonce_base,
                fixture.control_key_id,
                &fixture.control_signing_key,
            );
            let (release_delayed, delayed_client) = spawn_ordered_delayed_runtime_test_client(
                fixture.socket_path(),
                delayed_frame,
                fixture.response_public_key,
                nonce_base,
                3,
            );

            let mut fast_clients = Vec::new();
            fast_clients
                .try_reserve_exact(
                    usize::try_from(FAST_CLIENTS_PER_CYCLE)
                        .expect("fast client count must address"),
                )
                .expect("fast client handles must reserve");
            for fast_index in 1..=FAST_CLIENTS_PER_CYCLE {
                let request_id =
                    request_base.checked_add(fast_index).expect("fast request ID must fit");
                let nonce = nonce_base
                    .checked_add(u64::try_from(fast_index).expect("fast index must fit"))
                    .expect("fast nonce must fit");
                let request = fixture.request(request_id, "fast", 2);
                let frame = signed_generation_frame(
                    &request,
                    nonce,
                    fixture.control_key_id,
                    &fixture.control_signing_key,
                );
                fast_clients.push(spawn_ordered_runtime_test_client(
                    fixture.socket_path(),
                    frame,
                    fixture.response_public_key,
                    nonce,
                    3,
                ));
            }

            let shutdown = ContinuousShutdown::new();
            let request_shutdown = shutdown.clone();
            let (cycle_result_sender, cycle_result_receiver) = mpsc::sync_channel(1);
            let coordinator = thread::spawn(move || {
                let mut fast_completed = true;
                for client in fast_clients {
                    fast_completed &= client.join().is_ok_and(|events| events.last() == Some(&3));
                }
                let release_succeeded = release_delayed.send(()).is_ok();
                let delayed_completed =
                    delayed_client.join().is_ok_and(|(released_before_timeout, events)| {
                        released_before_timeout && events.last() == Some(&3)
                    });
                request_shutdown.request();
                cycle_result_sender
                    .send((fast_completed, release_succeeded, delayed_completed))
                    .expect("soak cycle result must send");
            });

            let summary = fixture
                .runtime
                .serve_continuous(
                    &fixture.listener,
                    ContinuousServeLimits::new(CONNECTIONS_PER_CYCLE)
                        .expect("soak limits must validate"),
                    &shutdown,
                )
                .expect("bounded soak cycle must drain");
            let cycle_result = cycle_result_receiver
                .recv_timeout(Duration::from_secs(10))
                .expect("soak cycle must complete within its bound");
            coordinator.join().expect("soak coordinator must join");
            assert_eq!(cycle_result, (true, true, true));
            assert_eq!(summary.accepted_connections(), u64::from(CONNECTIONS_PER_CYCLE));
            assert_eq!(summary.authenticated_commands(), u64::from(CONNECTIONS_PER_CYCLE));
            assert_eq!(summary.completed_generations(), u64::from(CONNECTIONS_PER_CYCLE));
            assert_eq!(summary.failed_connections(), 0);
            assert!(summary.peak_owned_connections() <= CONNECTIONS_PER_CYCLE);
            assert_eq!(fixture.runtime.service.scheduler.queued_len(), 0);
            assert_eq!(fixture.runtime.service.scheduler.active_len(), 0);
            fixture.cleanup();
        }
    }

    #[cfg(all(unix, feature = "cpu-reference"))]
    #[test]
    fn continuous_connection_limits_are_exact_and_bounded() {
        assert!(matches!(ContinuousServeLimits::new(0), Err(EngineError::InvalidInput(_))));
        assert!(matches!(
            ContinuousServeLimits::new(MAX_CONTINUOUS_CONNECTIONS + 1),
            Err(EngineError::InvalidInput(_))
        ));
        assert_eq!(
            ContinuousServeLimits::new(MAX_CONTINUOUS_CONNECTIONS)
                .expect("hard limit must validate")
                .maximum_connections(),
            MAX_CONTINUOUS_CONNECTIONS
        );
    }

    #[cfg(all(unix, feature = "cpu-reference"))]
    struct BatchRuntimeFixture {
        runtime: CpuEngineRuntime,
        listener: UnixFrameListener,
        runtime_root: PathBuf,
        model_id: ModelId,
        budget: ResourceBudget,
        control_signing_key: SigningKey,
        control_key_id: SignerKeyId,
        response_public_key: [u8; 32],
    }

    #[cfg(all(unix, feature = "cpu-reference"))]
    impl BatchRuntimeFixture {
        fn new(label: &str, maximum_active: u32) -> Self {
            Self::new_with_backend(label, maximum_active, RuntimeBackend::CpuReference)
        }

        fn new_with_backend(label: &str, maximum_active: u32, backend: RuntimeBackend) -> Self {
            use std::fs;
            use std::os::unix::fs::PermissionsExt;
            use std::sync::atomic::AtomicU64;

            static NEXT_DIRECTORY: AtomicU64 = AtomicU64::new(1);
            let sequence = NEXT_DIRECTORY.fetch_add(1, Ordering::Relaxed);
            let runtime_root = PathBuf::from("/tmp")
                .join(format!("personalai-engine-batch-{label}-{}-{sequence}", std::process::id()));
            fs::create_dir(&runtime_root).expect("runtime root must create");
            fs::set_permissions(&runtime_root, fs::Permissions::from_mode(0o700))
                .expect("runtime root permissions must set");
            let socket_path = runtime_root.join("engine.sock");
            let config = ServiceConfig::new("/models", &runtime_root, &socket_path)
                .expect("runtime config must validate");
            let mut service = EngineService::new(
                config,
                SchedulerLimits::new(8, maximum_active, 4 << 20).expect("limits must validate"),
            )
            .expect("service must construct");
            let fixture = DecoderPackageFixtureBuilder::new().build().expect("fixture must build");
            let model_id = ModelId::new("decoder-batch").expect("model ID must validate");
            service
                .register_configured_model(
                    model_id.clone(),
                    "decoder-batch.paimodel",
                    fixture.package_hash(),
                    fixture.trusted_signer().clone(),
                )
                .expect("model must register");
            let control_signing_key = SigningKey::from_bytes(&[0x81; 32]);
            let control_key_id = SignerKeyId::from_bytes([0x82; 16]).expect("key ID must validate");
            let trusted_key = TrustedControlPlaneKey::active(
                control_key_id,
                control_signing_key.verifying_key().to_bytes(),
            )
            .expect("control key must validate");
            let budget = ResourceBudget::new(1 << 24, 1 << 20, maximum_active)
                .expect("budget must validate");
            let response_signer = test_response_signer();
            let response_public_key = response_signer.public_key();
            let trusted_keys = TrustedControlPlaneKeys::single(trusted_key)
                .expect("control key ring must validate");
            let backend_id =
                backend.backend_id().expect("backend identity must validate").as_str().to_owned();
            let mut runtime = CpuEngineRuntime::new_with_control_keys_and_backend(
                service,
                trusted_keys,
                response_signer,
                backend,
            )
            .expect("runtime must construct");
            if backend_id == "metal" {
                #[cfg(all(feature = "metal", target_os = "macos"))]
                runtime
                    .load_registered_metal_model_bytes(&model_id, fixture.bytes(), &budget)
                    .expect("Metal model must load");
            } else if backend_id == "arm64" {
                #[cfg(all(feature = "arm64", target_arch = "aarch64"))]
                runtime
                    .load_registered_arm64_model_bytes(&model_id, fixture.bytes(), &budget)
                    .expect("ARM64 model must load");
            } else {
                runtime
                    .load_registered_model_bytes(&model_id, fixture.bytes(), &budget)
                    .expect("CPU model must load");
            }
            let listener =
                UnixFrameListener::bind(runtime.service().config()).expect("socket must bind");
            Self {
                runtime,
                listener,
                runtime_root,
                model_id,
                budget,
                control_signing_key,
                control_key_id,
                response_public_key,
            }
        }

        fn request(&self, id: u128, prompt: &str, maximum_output_tokens: u32) -> GenerationRequest {
            GenerationRequest::new(
                RequestId::new(id).expect("request ID must validate"),
                self.model_id.clone(),
                prompt,
                maximum_output_tokens,
                ContextLimit::new(8).expect("context must validate"),
                self.budget,
            )
            .expect("request must validate")
        }

        fn socket_path(&self) -> PathBuf {
            self.listener.socket_path().to_path_buf()
        }

        fn runtime_v3_identity(&self) -> RuntimeIdentityBinding {
            let registration = self
                .runtime
                .service
                .registry
                .resolve(&self.model_id)
                .expect("fixture registration");
            RuntimeIdentityBinding::new(
                LogicalModelId::new("mdl_runtime-fixture").expect("logical model ID"),
                self.model_id.clone(),
                ArtifactAuthorityId::new("art_runtime-fixture").expect("artifact ID"),
                *registration.expected_hash(),
                self.runtime.backend.backend_id().expect("backend ID"),
                RuntimeInstanceId::new("version.1").expect("backend version"),
                RuntimeInstanceId::new("backend.instance-1").expect("backend instance"),
                RuntimeInstanceId::new("engine.instance-1").expect("engine instance"),
            )
        }

        fn enable_runtime_v3(&mut self) -> RuntimeIdentityBinding {
            let identity = self.runtime_v3_identity();
            let config = RuntimeV3ServiceConfig::new(
                RuntimeInstanceId::new("version.1").expect("backend version"),
                RuntimeInstanceId::new("backend.instance-1").expect("backend instance"),
                RuntimeInstanceId::new("engine.instance-1").expect("engine instance"),
                2,
                2 << 20,
                30_000,
            )
            .expect("runtime v3 config");
            self.runtime.enable_runtime_v3(config).expect("enable runtime v3");
            self.runtime
                .configure_runtime_v3_model(identity.clone())
                .expect("configure runtime v3 model");
            identity
        }

        fn authenticated_service_request(
            &self,
            nonce: u64,
            body: &ServiceRequestBody,
        ) -> AuthenticatedServiceRequest {
            let request_id = RequestId::new(u128::from(nonce)).expect("request ID");
            let unsigned = AuthenticatedServiceRequest::unsigned_bytes(
                request_id,
                nonce,
                self.control_key_id,
                body,
            )
            .expect("request encode");
            let transcript = control_plane_signature_message(&unsigned).expect("transcript");
            let signature = self.control_signing_key.sign(&transcript).to_bytes();
            let payload = attach_service_signature(&unsigned, &signature).expect("signature");
            let trusted = TrustedControlPlaneKeys::single(
                TrustedControlPlaneKey::active(
                    self.control_key_id,
                    self.control_signing_key.verifying_key().to_bytes(),
                )
                .expect("trusted key"),
            )
            .expect("trusted keys");
            AuthenticatedServiceRequest::decode(&payload, &trusted).expect("request decode")
        }

        fn cleanup(self) {
            use std::fs;

            let replay_path =
                self.runtime_root.join(format!("control-replay-{}.bin", self.control_key_id));
            drop(self.listener);
            drop(self.runtime);
            fs::remove_file(replay_path).expect("replay journal must clean");
            fs::remove_dir(self.runtime_root).expect("runtime root must clean");
        }
    }

    #[cfg(all(unix, feature = "cpu-reference"))]
    #[test]
    #[allow(
        clippy::too_many_lines,
        reason = "one authenticated socket transcript proves the complete reservation and execution lifecycle"
    )]
    fn authenticated_runtime_v3_availability_reservation_and_release_are_exact() {
        let mut fixture = BatchRuntimeFixture::new("runtime-v3-contract", 1);
        let identity = fixture.enable_runtime_v3();

        let availability_request =
            fixture.authenticated_service_request(41, &ServiceRequestBody::RuntimeV3Availability);
        let ServiceResponseBody::RuntimeV3Availability(document) = fixture
            .runtime
            .handle_service_request(&availability_request)
            .expect("availability response")
        else {
            panic!("runtime v3 availability response kind changed");
        };
        let availability =
            decode_runtime_availability_v3(document.as_bytes()).expect("availability contract");
        assert_eq!(availability.engine_instance_id, identity.engine_instance_id().as_str());
        assert_eq!(availability.models.len(), 1);
        assert_eq!(availability.models[0].identity.provider_model_id, fixture.model_id.as_str());

        let now = runtime_v3_now_unix_millis().expect("service time");
        let reservation = InferenceReservationRequestV3 {
            contract_version: InferenceRuntimeContractVersionV3,
            kind: ReservationRequestKindV3::ReservationRequest,
            reservation_id: "rsv_service-fixture".to_owned(),
            request_id: sovereign_ai_contracts::identity::RequestId::new(
                "req_11111111111111111111111111111111",
            )
            .expect("contract request ID"),
            task_id: sovereign_ai_contracts::identity::TaskId::new("tsk_service-fixture")
                .expect("contract task ID"),
            identity: availability.models[0].identity.clone(),
            budget: RuntimeResourceBudgetV3 {
                maximum_input_tokens: 1_024,
                maximum_output_tokens: 64,
                maximum_inference_memory_bytes: 1 << 20,
                maximum_execution_millis: 5_000,
            },
            idempotency_key: "11".repeat(32),
            created_at: runtime_v3_timestamp_from_unix_millis(now).expect("created time"),
            expires_at: runtime_v3_timestamp_from_unix_millis(now + 10_000).expect("expiry time"),
            extensions: None,
            required_extensions: None,
        };
        let reserve_body = ServiceRequestBody::RuntimeV3Reserve(
            RuntimeV3Document::new(
                encode_inference_reservation_request_v3(&reservation).expect("reservation"),
            )
            .expect("bounded reservation"),
        );
        let reserve = fixture.authenticated_service_request(42, &reserve_body);
        let ServiceResponseBody::RuntimeV3Reservation(document) =
            fixture.runtime.handle_service_request(&reserve).expect("reservation response")
        else {
            panic!("runtime v3 reservation response kind changed");
        };
        let decision = decode_inference_reservation_decision_v3(document.as_bytes())
            .expect("reservation decision");
        assert_eq!(decision.decision, ReservationDecisionV3::Accepted);

        #[cfg(feature = "development-runtime-v3-simulator")]
        {
            let execution_created = runtime_v3_now_unix_millis().expect("execution service time");
            fixture
                .runtime
                .enable_runtime_v3_development_simulator(
                    RuntimeV3DevelopmentBehavior::fixed_success(
                        "development runtime v3 fixture output",
                    )
                    .expect("development output"),
                )
                .expect("enable explicit development simulator");
            let execution = InferenceExecutionRequestV3 {
                contract_version: InferenceRuntimeContractVersionV3,
                kind: ExecutionRequestKindV3::ExecutionRequest,
                reservation_id: reservation.reservation_id.clone(),
                request_id: reservation.request_id.clone(),
                task_id: reservation.task_id.clone(),
                identity: reservation.identity.clone(),
                messages: vec![InferenceMessage {
                    role: InferenceRole::User,
                    content: "deterministic fixture input".to_owned(),
                    name: None,
                }],
                parameters: ContractGenerationParameters {
                    max_output_tokens: 8,
                    temperature: 0.0,
                    top_p: 1.0,
                    seed: Some(7),
                    stop_sequences: None,
                },
                idempotency_key: "22".repeat(32),
                created_at: runtime_v3_timestamp_from_unix_millis(execution_created)
                    .expect("execution creation"),
                deadline_at: runtime_v3_timestamp_from_unix_millis(now + 9_000)
                    .expect("execution deadline"),
                extensions: None,
                required_extensions: None,
            };
            let execution_body = ServiceRequestBody::RuntimeV3Execute(
                RuntimeV3Document::new(
                    encode_inference_execution_request_v3(&execution).expect("execution request"),
                )
                .expect("bounded execution request"),
            );
            let execute = fixture.authenticated_service_request(43, &execution_body);
            let ServiceResponseBody::RuntimeV3Execution(document) =
                fixture.runtime.handle_service_request(&execute).expect("execution response")
            else {
                panic!("runtime v3 execution response kind changed");
            };
            let result = decode_inference_execution_result_v3(document.as_bytes())
                .expect("execution result");
            result.validate_against(&execution).expect("result binding");
            assert_eq!(result.status, ExecutionStatusV3::Succeeded);
            assert_eq!(result.output.as_deref(), Some("development runtime v3 fixture output"));
        }

        let release = InferenceReservationReleaseV3 {
            contract_version: InferenceRuntimeContractVersionV3,
            kind: ReservationReleaseKindV3::ReservationRelease,
            reservation_id: reservation.reservation_id,
            request_id: reservation.request_id,
            task_id: reservation.task_id,
            reason_code: "owner_release".to_owned(),
            requested_at: runtime_v3_timestamp_from_unix_millis(now + 1).expect("release time"),
            extensions: None,
            required_extensions: None,
        };
        let release_body = ServiceRequestBody::RuntimeV3Release(
            RuntimeV3Document::new(
                encode_inference_reservation_release_v3(&release).expect("release"),
            )
            .expect("bounded release"),
        );
        let release = fixture.authenticated_service_request(44, &release_body);
        assert_eq!(
            fixture.runtime.handle_service_request(&release),
            Ok(ServiceResponseBody::RuntimeV3Released)
        );
        fixture.cleanup();
    }

    #[cfg(all(unix, feature = "cpu-reference"))]
    #[test]
    fn authenticated_runtime_v3_availability_round_trips_over_protected_socket() {
        use std::io::{Read, Write};
        use std::os::unix::net::UnixStream;

        let mut fixture = BatchRuntimeFixture::new("runtime-v3-uds", 1);
        let identity = fixture.enable_runtime_v3();
        let request_id = RequestId::new(0x5353).expect("request ID");
        let nonce = 51;
        let body = ServiceRequestBody::RuntimeV3Availability;
        let unsigned = AuthenticatedServiceRequest::unsigned_bytes(
            request_id,
            nonce,
            fixture.control_key_id,
            &body,
        )
        .expect("service request");
        let transcript = control_plane_signature_message(&unsigned).expect("transcript");
        let signature = fixture.control_signing_key.sign(&transcript).to_bytes();
        let payload = attach_service_signature(&unsigned, &signature).expect("signature");
        let encoded = Frame::new(FrameKind::ServiceRequest, payload)
            .expect("service frame")
            .encode()
            .expect("encoded service frame");
        let socket_path = fixture.socket_path();
        let response_public_key = fixture.response_public_key;
        let expected_engine_instance = identity.engine_instance_id().as_str().to_owned();
        let client = std::thread::spawn(move || {
            let mut stream = UnixStream::connect(socket_path).expect("client must connect");
            stream.write_all(&encoded).expect("request must write");
            let mut header = [0_u8; FRAME_HEADER_BYTES];
            stream.read_exact(&mut header).expect("response header");
            let payload_length = usize::try_from(u32::from_le_bytes(
                header[12..16].try_into().expect("payload length field"),
            ))
            .expect("payload length");
            let mut frame = header.to_vec();
            frame.resize(FRAME_HEADER_BYTES + payload_length, 0);
            stream.read_exact(&mut frame[FRAME_HEADER_BYTES..]).expect("response payload");
            let response = Frame::decode_outbound_exact(&frame).expect("response frame");
            let engine_key_id = SignerKeyId::from_bytes([0x62; 16]).expect("engine key ID");
            let verified = AuthenticatedServiceResponse::decode(
                response.payload(),
                request_id,
                nonce,
                engine_key_id,
                &response_public_key,
            )
            .expect("service response must verify");
            let ServiceResponseBody::RuntimeV3Availability(document) = verified.body() else {
                panic!("runtime v3 availability response kind changed");
            };
            let availability = decode_runtime_availability_v3(document.as_bytes())
                .expect("canonical availability");
            assert_eq!(availability.engine_instance_id, expected_engine_instance);
            assert_eq!(availability.models.len(), 1);
        });
        assert!(matches!(
            fixture.runtime.serve_one(&fixture.listener),
            Ok(ServeOutcome::ServiceResponded {
                request_id: returned,
                kind: ServiceMessageKind::RuntimeV3Availability,
            }) if returned == request_id
        ));
        client.join().expect("client must finish");
        fixture.cleanup();
    }

    #[cfg(all(unix, feature = "cpu-reference"))]
    #[test]
    fn authenticated_local_health_request_receives_signed_service_response() {
        use std::io::{Read, Write};
        use std::os::unix::net::UnixStream;

        let mut fixture = BatchRuntimeFixture::new("service-health", 1);
        let request_id = RequestId::new(0x5151).expect("request ID");
        let unsigned = AuthenticatedServiceRequest::unsigned_bytes(
            request_id,
            1,
            fixture.control_key_id,
            &ServiceRequestBody::Health,
        )
        .expect("service request must encode");
        let transcript = control_plane_signature_message(&unsigned).expect("transcript");
        let signature = fixture.control_signing_key.sign(&transcript).to_bytes();
        let payload = attach_service_signature(&unsigned, &signature).expect("signature");
        let encoded = Frame::new(FrameKind::ServiceRequest, payload)
            .expect("service frame")
            .encode()
            .expect("encoded service frame");
        let socket_path = fixture.socket_path();
        let response_public_key = fixture.response_public_key;
        let client = std::thread::spawn(move || {
            let mut stream = UnixStream::connect(socket_path).expect("client must connect");
            stream.write_all(&encoded).expect("request must write");
            let mut header = [0_u8; FRAME_HEADER_BYTES];
            stream.read_exact(&mut header).expect("response header");
            let payload_length = usize::try_from(u32::from_le_bytes(
                header[12..16].try_into().expect("payload length field"),
            ))
            .expect("payload length");
            let mut frame = header.to_vec();
            frame.resize(FRAME_HEADER_BYTES + payload_length, 0);
            stream.read_exact(&mut frame[FRAME_HEADER_BYTES..]).expect("response payload");
            let response = Frame::decode_outbound_exact(&frame).expect("response frame");
            assert_eq!(response.kind(), FrameKind::ServiceResponse);
            let engine_key_id = SignerKeyId::from_bytes([0x62; 16]).expect("engine key ID");
            let verified = AuthenticatedServiceResponse::decode(
                response.payload(),
                request_id,
                1,
                engine_key_id,
                &response_public_key,
            )
            .expect("service response must verify");
            assert_eq!(verified.kind(), ServiceMessageKind::Health);
            assert_eq!(verified.body(), &ServiceResponseBody::Health(ServiceHealthStatus::Ready));
        });
        assert!(matches!(
            fixture.runtime.serve_one(&fixture.listener),
            Ok(ServeOutcome::ServiceResponded {
                request_id: returned,
                kind: ServiceMessageKind::Health,
            }) if returned == request_id
        ));
        client.join().expect("client must finish");
        fixture.cleanup();
    }

    #[cfg(all(unix, feature = "cpu-reference"))]
    #[test]
    fn authenticated_model_status_is_one_signed_scheduler_snapshot() {
        use std::io::{Read, Write};
        use std::os::unix::net::UnixStream;

        let mut fixture = BatchRuntimeFixture::new("service-model-status", 1);
        let request_id = RequestId::new(0x5252).expect("request ID");
        let request_body = ServiceRequestBody::ModelStatus(fixture.model_id.clone());
        let unsigned = AuthenticatedServiceRequest::unsigned_bytes(
            request_id,
            2,
            fixture.control_key_id,
            &request_body,
        )
        .expect("service request must encode");
        let transcript = control_plane_signature_message(&unsigned).expect("transcript");
        let signature = fixture.control_signing_key.sign(&transcript).to_bytes();
        let payload = attach_service_signature(&unsigned, &signature).expect("signature");
        let encoded = Frame::new(FrameKind::ServiceRequest, payload)
            .expect("service frame")
            .encode()
            .expect("encoded service frame");
        let socket_path = fixture.socket_path();
        let response_public_key = fixture.response_public_key;
        let client = std::thread::spawn(move || {
            let mut stream = UnixStream::connect(socket_path).expect("client must connect");
            stream.write_all(&encoded).expect("request must write");
            let mut header = [0_u8; FRAME_HEADER_BYTES];
            stream.read_exact(&mut header).expect("response header");
            let payload_length = usize::try_from(u32::from_le_bytes(
                header[12..16].try_into().expect("payload length field"),
            ))
            .expect("payload length");
            let mut frame = header.to_vec();
            frame.resize(FRAME_HEADER_BYTES + payload_length, 0);
            stream.read_exact(&mut frame[FRAME_HEADER_BYTES..]).expect("response payload");
            let response = Frame::decode_outbound_exact(&frame).expect("response frame");
            let engine_key_id = SignerKeyId::from_bytes([0x62; 16]).expect("engine key ID");
            let verified = AuthenticatedServiceResponse::decode(
                response.payload(),
                request_id,
                2,
                engine_key_id,
                &response_public_key,
            )
            .expect("service response must verify");
            let ServiceResponseBody::ModelStatus(status) = verified.body() else {
                panic!("model status response kind changed");
            };
            assert!(status.model_loaded());
            assert_eq!(status.health(), ServiceHealthStatus::Ready);
            assert!(
                status
                    .capabilities()
                    .contains(EngineCapability::CpuReference)
                    .expect("known capability")
            );
            assert_eq!(status.active_requests(), 0);
            assert_eq!(status.queued_requests(), 0);
            assert_eq!(status.maximum_active(), 1);
            assert_eq!(status.maximum_queued(), 8);
            assert_eq!(status.queued_bytes(), 0);
            assert_eq!(status.maximum_queued_bytes(), 4 << 20);
        });
        assert!(matches!(
            fixture.runtime.serve_one(&fixture.listener),
            Ok(ServeOutcome::ServiceResponded {
                request_id: returned,
                kind: ServiceMessageKind::ModelStatus,
            }) if returned == request_id
        ));
        client.join().expect("client must finish");
        fixture.cleanup();
    }

    #[cfg(all(unix, feature = "cpu-reference"))]
    #[test]
    fn security_event_poll_is_non_destructive_until_exact_acknowledgement() {
        let mut fixture = BatchRuntimeFixture::new("service-security-events", 1);
        fixture.runtime.security_events.push_back(SecurityEvent::UnauthorizedClient);
        fixture.runtime.security_events.push_back(SecurityEvent::MalformedFrame);

        let decode = |fixture: &BatchRuntimeFixture, nonce: u64, body: ServiceRequestBody| {
            let request_id = RequestId::new(u128::from(nonce)).expect("request ID");
            let unsigned = AuthenticatedServiceRequest::unsigned_bytes(
                request_id,
                nonce,
                fixture.control_key_id,
                &body,
            )
            .expect("request encode");
            let transcript = control_plane_signature_message(&unsigned).expect("transcript");
            let signature = fixture.control_signing_key.sign(&transcript).to_bytes();
            let payload = attach_service_signature(&unsigned, &signature).expect("signature");
            let trusted = TrustedControlPlaneKeys::single(
                TrustedControlPlaneKey::active(
                    fixture.control_key_id,
                    fixture.control_signing_key.verifying_key().to_bytes(),
                )
                .expect("trusted key"),
            )
            .expect("trusted keys");
            AuthenticatedServiceRequest::decode(&payload, &trusted).expect("request decode")
        };

        let poll = decode(&fixture, 1, ServiceRequestBody::SecurityEvents { maximum: 1 });
        assert!(matches!(
            fixture.runtime.handle_service_request(&poll),
            Ok(ServiceResponseBody::SecurityEvents(events)) if events.len() == 1
        ));
        assert_eq!(fixture.runtime.security_events.len(), 2);

        let acknowledge =
            decode(&fixture, 2, ServiceRequestBody::AcknowledgeSecurityEvents { count: 1 });
        assert_eq!(
            fixture.runtime.handle_service_request(&acknowledge),
            Ok(ServiceResponseBody::SecurityEventsAcknowledged)
        );
        assert_eq!(fixture.runtime.security_events.len(), 1);

        let excessive =
            decode(&fixture, 3, ServiceRequestBody::AcknowledgeSecurityEvents { count: 2 });
        assert!(fixture.runtime.handle_service_request(&excessive).is_err());
        assert_eq!(fixture.runtime.security_events.len(), 1);

        let idempotent_load =
            decode(&fixture, 4, ServiceRequestBody::LoadModel(fixture.model_id.clone()));
        assert_eq!(
            fixture.runtime.handle_service_request(&idempotent_load),
            Ok(ServiceResponseBody::ModelLoaded)
        );
        fixture.cleanup();
    }

    #[cfg(all(unix, feature = "cpu-reference"))]
    fn spawn_ordered_runtime_test_client(
        socket_path: PathBuf,
        encoded: Vec<u8>,
        response_public_key: [u8; 32],
        expected_nonce: u64,
        terminal_kind: u8,
    ) -> std::thread::JoinHandle<Vec<u8>> {
        use std::os::unix::net::UnixStream;
        use std::sync::mpsc;
        use std::thread;

        let (connected, wait_for_connection) = mpsc::sync_channel(0);
        let client = thread::spawn(move || {
            let stream = UnixStream::connect(socket_path).expect("client must connect");
            connected.send(()).expect("connection order must synchronize");
            runtime_test_stream(
                stream,
                &encoded,
                response_public_key,
                expected_nonce,
                terminal_kind,
            )
        });
        wait_for_connection.recv().expect("client must connect before the next peer");
        client
    }

    #[cfg(all(unix, feature = "cpu-reference"))]
    fn spawn_ordered_delayed_runtime_test_client(
        socket_path: PathBuf,
        encoded: Vec<u8>,
        response_public_key: [u8; 32],
        expected_nonce: u64,
        terminal_kind: u8,
    ) -> (std::sync::mpsc::SyncSender<()>, std::thread::JoinHandle<(bool, Vec<u8>)>) {
        use std::io::Write;
        use std::os::unix::net::UnixStream;
        use std::sync::mpsc;
        use std::thread;
        use std::time::Duration;

        let (connected, wait_for_connection) = mpsc::sync_channel(0);
        let (release, wait_for_release) = mpsc::sync_channel(0);
        let client = thread::spawn(move || {
            let mut stream = UnixStream::connect(socket_path).expect("delayed client must connect");
            configure_runtime_test_client(&stream);
            stream.write_all(&encoded).expect("delayed request must write");
            connected.send(()).expect("delayed connection must synchronize");
            let released_before_timeout =
                wait_for_release.recv_timeout(Duration::from_secs(5)).is_ok();
            let events =
                runtime_test_events(stream, response_public_key, expected_nonce, terminal_kind);
            (released_before_timeout, events)
        });
        wait_for_connection.recv().expect("delayed client must connect before fast peers");
        (release, client)
    }

    #[cfg(all(unix, feature = "cpu-reference"))]
    fn spawn_ordered_write_only_client(
        socket_path: PathBuf,
        encoded: Vec<u8>,
    ) -> std::thread::JoinHandle<Vec<u8>> {
        use std::io::{Read, Write};
        use std::os::unix::net::UnixStream;
        use std::sync::mpsc;
        use std::thread;

        let (connected, wait_for_connection) = mpsc::sync_channel(0);
        let client = thread::spawn(move || {
            let mut stream = UnixStream::connect(socket_path).expect("client must connect");
            connected.send(()).expect("connection order must synchronize");
            stream.write_all(&encoded).expect("command must write");
            let mut response = Vec::new();
            stream.read_to_end(&mut response).expect("server close must be observable");
            response
        });
        wait_for_connection.recv().expect("client must connect before the next peer");
        client
    }

    #[cfg(all(unix, feature = "cpu-reference"))]
    fn spawn_ordered_disconnecting_client(
        socket_path: PathBuf,
        encoded: Vec<u8>,
    ) -> std::thread::JoinHandle<()> {
        use personalai_engine_protocol::FRAME_HEADER_BYTES;
        use std::io::{Read, Write};
        use std::os::unix::net::UnixStream;
        use std::sync::mpsc;
        use std::thread;

        let (connected, wait_for_connection) = mpsc::sync_channel(0);
        let client = thread::spawn(move || {
            let mut stream = UnixStream::connect(socket_path).expect("client must connect");
            connected.send(()).expect("connection order must synchronize");
            stream.write_all(&encoded).expect("request must write before disconnect");
            let mut header = [0_u8; FRAME_HEADER_BYTES];
            stream.read_exact(&mut header).expect("first event header must read");
            let length = usize::try_from(u32::from_le_bytes(
                header[12..16].try_into().expect("length field must exist"),
            ))
            .expect("event length must address");
            let mut frame = Vec::from(header);
            frame.resize(FRAME_HEADER_BYTES + length, 0);
            stream.read_exact(&mut frame[FRAME_HEADER_BYTES..]).expect("first event must read");
            let event = Frame::decode_outbound_exact(&frame).expect("first event must decode");
            assert_eq!(event.payload()[10], 1);
            drop(stream);
        });
        wait_for_connection.recv().expect("client must connect before the next peer");
        client
    }

    #[cfg(all(unix, feature = "cpu-reference"))]
    fn runtime_test_client(
        socket_path: PathBuf,
        encoded: &[u8],
        response_public_key: [u8; 32],
        expected_nonce: u64,
        terminal_kind: u8,
    ) -> Vec<u8> {
        use std::os::unix::net::UnixStream;

        let stream = UnixStream::connect(socket_path).expect("client must connect");
        runtime_test_stream(stream, encoded, response_public_key, expected_nonce, terminal_kind)
    }

    #[cfg(all(unix, feature = "cpu-reference"))]
    fn runtime_test_stream(
        mut stream: std::os::unix::net::UnixStream,
        encoded: &[u8],
        response_public_key: [u8; 32],
        expected_nonce: u64,
        terminal_kind: u8,
    ) -> Vec<u8> {
        use std::io::Write;

        configure_runtime_test_client(&stream);
        stream.write_all(encoded).expect("request must write");
        runtime_test_events(stream, response_public_key, expected_nonce, terminal_kind)
    }

    #[cfg(all(unix, feature = "cpu-reference"))]
    fn configure_runtime_test_client(stream: &std::os::unix::net::UnixStream) {
        use std::time::Duration;

        stream
            .set_read_timeout(Some(Duration::from_secs(5)))
            .expect("test client read timeout must set");
        stream
            .set_write_timeout(Some(Duration::from_secs(5)))
            .expect("test client write timeout must set");
    }

    #[cfg(all(unix, feature = "cpu-reference"))]
    fn runtime_test_events(
        mut stream: std::os::unix::net::UnixStream,
        response_public_key: [u8; 32],
        expected_nonce: u64,
        terminal_kind: u8,
    ) -> Vec<u8> {
        use personalai_engine_protocol::{FRAME_HEADER_BYTES, GENERATION_EVENT_AUTH_BYTES};
        use std::io::Read;

        let mut event_kinds = Vec::new();
        let response_key =
            VerifyingKey::from_bytes(&response_public_key).expect("response key must parse");
        let expected_key_id = [0x62; 16];
        let mut expected_ordinal = 1_u32;
        loop {
            let mut header = [0_u8; FRAME_HEADER_BYTES];
            stream.read_exact(&mut header).expect("event header must read");
            let length = usize::try_from(u32::from_le_bytes(
                header[12..16].try_into().expect("length field must exist"),
            ))
            .expect("event length must address");
            let mut frame = Vec::from(header);
            frame.resize(FRAME_HEADER_BYTES + length, 0);
            stream.read_exact(&mut frame[FRAME_HEADER_BYTES..]).expect("event payload must read");
            let event = Frame::decode_outbound_exact(&frame).expect("event must decode");
            let payload = event.payload();
            let unsigned_end = payload.len() - personalai_engine_crypto::ED25519_SIGNATURE_BYTES;
            let auth_start = unsigned_end - GENERATION_EVENT_AUTH_BYTES;
            assert_eq!(&payload[auth_start..auth_start + 8], &expected_nonce.to_le_bytes());
            assert_eq!(&payload[auth_start + 8..auth_start + 12], &expected_ordinal.to_le_bytes());
            assert_eq!(&payload[auth_start + 16..unsigned_end], &expected_key_id);
            let signature = Signature::from_bytes(
                &payload[unsigned_end..].try_into().expect("event signature must exist"),
            );
            let transcript = engine_response_signature_message(&payload[..unsigned_end])
                .expect("event transcript must construct");
            response_key
                .verify_strict(&transcript, &signature)
                .expect("runtime event signature must verify");
            expected_ordinal = expected_ordinal.checked_add(1).expect("ordinal must fit");
            event_kinds.push(event.payload()[10]);
            if event.payload()[10] == terminal_kind {
                break;
            }
        }
        event_kinds
    }

    fn signed_generation_frame(
        request: &GenerationRequest,
        nonce: u64,
        key_id: SignerKeyId,
        signing_key: &SigningKey,
    ) -> Vec<u8> {
        let unsigned = GenerationRequestEnvelope::unsigned_bytes(request, nonce, key_id)
            .expect("request must encode");
        let transcript =
            control_plane_signature_message(&unsigned).expect("transcript must construct");
        let signature = signing_key.sign(&transcript).to_bytes();
        let payload =
            attach_generation_signature(&unsigned, &signature).expect("signature must attach");
        Frame::new(FrameKind::GenerationRequest, payload)
            .expect("request frame must construct")
            .encode()
            .expect("request frame must encode")
    }

    fn signed_cancellation_frame(
        request_id: RequestId,
        nonce: u64,
        key_id: SignerKeyId,
        signing_key: &SigningKey,
    ) -> Vec<u8> {
        let unsigned = CancellationEnvelope::unsigned_bytes(request_id, nonce, key_id)
            .expect("cancellation must encode");
        let transcript =
            control_plane_signature_message(&unsigned).expect("transcript must construct");
        let signature = signing_key.sign(&transcript).to_bytes();
        let payload =
            attach_cancellation_signature(&unsigned, &signature).expect("signature must attach");
        Frame::new(FrameKind::Cancellation, payload)
            .expect("cancellation frame must construct")
            .encode()
            .expect("cancellation frame must encode")
    }

    fn test_response_signer() -> EngineResponseSigner {
        let mut seed = [0x61; 32];
        let key_id = SignerKeyId::from_bytes([0x62; 16]).expect("key ID must validate");
        EngineResponseSigner::from_seed(key_id, &mut seed).expect("response signer must validate")
    }

    fn service() -> EngineService {
        let config = ServiceConfig::new(
            "/var/lib/personalai/models",
            "/run/user/501/personalai-engine",
            "/run/user/501/personalai-engine/engine.sock",
        )
        .expect("config must validate");
        EngineService::new(
            config,
            SchedulerLimits::new(8, 2, 1 << 20).expect("limits must validate"),
        )
        .expect("service must construct")
    }

    fn request(id: u128, model_id: ModelId) -> GenerationRequest {
        GenerationRequest::new(
            RequestId::new(id).expect("request id must validate"),
            model_id,
            "bounded input",
            16,
            ContextLimit::new(128).expect("context must validate"),
            ResourceBudget::new(1 << 20, 1 << 19, 1).expect("budget must validate"),
        )
        .expect("request must validate")
    }

    fn test_hash() -> ModelPackageHash {
        ModelPackageHash::from_bytes(ModelHashAlgorithm::Sha384, [3_u8; 48])
    }

    fn test_signer() -> TrustedSigner {
        let key_id = SignerKeyId::from_bytes([1_u8; 16]).expect("key id must validate");
        let public_key = [
            0xd7, 0x5a, 0x98, 0x01, 0x82, 0xb1, 0x0a, 0xb7, 0xd5, 0x4b, 0xfe, 0xd3, 0xc9, 0x64,
            0x07, 0x3a, 0x0e, 0xe1, 0x72, 0xf3, 0xda, 0xa6, 0x23, 0x25, 0xaf, 0x02, 0x1a, 0x68,
            0xf7, 0x07, 0x51, 0x1a,
        ];
        TrustedSigner::active(key_id, public_key).expect("RFC test key must validate")
    }
}
