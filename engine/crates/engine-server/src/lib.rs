//! Local-only composition boundary for the native inference engine.
//!
//! This crate validates Unix-domain-socket configuration, admits requests for
//! locally registered model IDs, and can execute the deterministic CPU
//! reference path. It contains no TCP or outbound network capability, tool
//! path, approval authority, arbitrary model path, or persistence authority.

#[cfg(unix)]
mod transport;

#[cfg(unix)]
pub use transport::{UnixFrameConnection, UnixFrameListener};

use personalai_engine_crypto::{TrustedControlPlaneKey, TrustedSigner};
use personalai_engine_loader::ModelRegistry;
#[cfg(feature = "cpu-reference")]
use personalai_engine_memory::MemoryPool;
use personalai_engine_protocol::{
    CancellationEnvelope, Frame, FrameKind, GenerationRequestEnvelope, PROTOCOL_MAJOR,
    PROTOCOL_MINOR,
};
#[cfg(feature = "cpu-reference")]
use personalai_engine_sampling::{SamplingConfig, select_next_token_tensor};
#[cfg(all(unix, feature = "cpu-reference"))]
use personalai_engine_scheduler::CancellationOutcome;
use personalai_engine_scheduler::{Scheduler, SchedulerLimits};
#[cfg(feature = "cpu-reference")]
use personalai_engine_tokenizer::{MAX_TOKEN_PIECE_BYTES, Tokenizer, build_tokenizer};
use personalai_engine_transformer::{BackendStatus, backend_status};
#[cfg(feature = "cpu-reference")]
use personalai_engine_transformer::{CpuDecoderModel, TransformerConfig};
use personalai_engine_types::{
    BackendId, EngineError, GenerationRequest, MAX_FRAME_BYTES, ModelId, ModelPackageHash,
    RequestId,
};
#[cfg(feature = "cpu-reference")]
use personalai_engine_types::{GenerationEvent, GenerationUsage, SequenceId, TextChunk, TokenId};
#[cfg(all(unix, feature = "cpu-reference"))]
use std::collections::BTreeMap;
use std::path::{Component, Path, PathBuf};
#[cfg(feature = "cpu-reference")]
use std::sync::atomic::{AtomicBool, Ordering};
#[cfg(all(unix, feature = "cpu-reference"))]
use std::thread;

/// Maximum configured Unix socket path bytes for conservative portability.
pub const MAX_UNIX_SOCKET_PATH_BYTES: usize = 100;
/// Human-readable status for the current reference runtime.
pub const FOUNDATION_STATUS: &str = "signed-package-cpu-reference-uds";

/// Validated local service configuration that does not open any resource.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ServiceConfig {
    model_root: PathBuf,
    runtime_root: PathBuf,
    socket_path: PathBuf,
}

impl ServiceConfig {
    /// Validates configuration-owned model and future socket paths.
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

    /// Returns the future protected runtime directory.
    pub fn runtime_root(&self) -> &Path {
        &self.runtime_root
    }

    /// Returns the future Unix-domain-socket path without opening it.
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
    Generate(GenerationRequest),
    /// Cancellation of one previously submitted request.
    Cancel(RequestId),
}

/// Per-connection authentication and strict monotonic anti-replay state.
#[derive(Clone, Debug)]
pub struct AuthenticatedControlSession {
    trusted_key: TrustedControlPlaneKey,
    last_nonce: u64,
}

impl AuthenticatedControlSession {
    /// Creates a fresh connection session with no accepted nonce.
    pub const fn new(trusted_key: TrustedControlPlaneKey) -> Self {
        Self { trusted_key, last_nonce: 0 }
    }

    /// Authenticates, decodes, and replay-checks one inbound command frame.
    pub fn decode_command(&mut self, frame: &Frame) -> Result<InboundCommand, EngineError> {
        let (nonce, command) = match frame.kind() {
            FrameKind::GenerationRequest => {
                let envelope =
                    GenerationRequestEnvelope::decode(frame.payload(), &self.trusted_key)?;
                (envelope.nonce(), InboundCommand::Generate(envelope.into_request()))
            }
            FrameKind::Cancellation => {
                let envelope = CancellationEnvelope::decode(frame.payload(), &self.trusted_key)?;
                (envelope.nonce(), InboundCommand::Cancel(envelope.request_id()))
            }
            _ => {
                return Err(EngineError::protocol_violation(
                    "frame kind is not an authenticated command",
                ));
            }
        };
        if nonce <= self.last_nonce {
            return Err(EngineError::authentication_failure(
                "control-plane request nonce was replayed or reordered",
            ));
        }
        self.last_nonce = nonce;
        Ok(command)
    }

    /// Returns the last accepted nonce.
    pub const fn last_nonce(&self) -> u64 {
        self.last_nonce
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
pub struct LoadedCpuModel {
    model_id: ModelId,
    model: CpuDecoderModel,
    tokenizer: Box<dyn Tokenizer>,
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

    /// Runs deterministic greedy generation and synchronously emits bounded events.
    pub fn generate_greedy(
        &self,
        request: &GenerationRequest,
        sequence_id: SequenceId,
        cancelled: &AtomicBool,
        mut emit: impl FnMut(GenerationEvent) -> Result<(), EngineError>,
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
        let input_tokens = self.tokenizer.encode(request.input_text())?;
        let input_count = u32::try_from(input_tokens.len())
            .map_err(|_| EngineError::resource_exhaustion("input token count exceeds u32"))?;
        request.validate_tokenized_input(input_count)?;
        emit(GenerationEvent::Started { request_id: request.request_id(), sequence_id })?;
        let mut session = self.model.start_session(&request.resource_budget())?;
        let mut logits = None;
        for token in input_tokens {
            check_cancelled(cancelled, request.request_id())?;
            logits = Some(session.forward_token(token)?);
        }
        let mut logits = logits.ok_or_else(|| {
            EngineError::internal_invariant("tokenizer produced no prompt tokens")
        })?;
        let mut generated = 0_u32;
        let mut pending_utf8 = [0_u8; 1_024];
        let mut pending_length = 0_usize;
        while generated < request.max_output_tokens() {
            check_cancelled(cancelled, request.request_id())?;
            let token = select_next_token_tensor(&logits, SamplingConfig::greedy())?;
            generated = generated.checked_add(1).ok_or_else(|| {
                EngineError::internal_invariant("generated token count overflows")
            })?;
            let text = append_streaming_piece(
                self.tokenizer.as_ref(),
                token,
                &mut pending_utf8,
                &mut pending_length,
            )?;
            emit(GenerationEvent::Token {
                request_id: request.request_id(),
                sequence_id,
                token_id: TokenId::new(token)?,
                text: TextChunk::new(text)?,
            })?;
            if token == self.tokenizer.spec().end_of_sequence_token() {
                if pending_length != 0 {
                    return Err(EngineError::backend_failure(
                        "generation ended with an incomplete UTF-8 sequence",
                    ));
                }
                break;
            }
            if generated < request.max_output_tokens() {
                logits = session.forward_token(token)?;
            }
        }
        if pending_length != 0 {
            return Err(EngineError::backend_failure(
                "generation limit ended with an incomplete UTF-8 sequence",
            ));
        }
        let usage = GenerationUsage::for_request(request, input_count, generated)?;
        emit(GenerationEvent::Completed { request_id: request.request_id(), sequence_id, usage })?;
        Ok(usage)
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
}

/// CPU-reference runtime joining authenticated UDS transport to loaded models.
///
/// Package bytes enter only through [`Self::load_registered_model_bytes`]
/// after the owner configuration has registered the model identifier, digest,
/// signer, and relative package name. Client messages can select only a model
/// identifier and can never install or name a filesystem path.
#[cfg(all(unix, feature = "cpu-reference"))]
pub struct CpuEngineRuntime {
    service: EngineService,
    trusted_control_key: TrustedControlPlaneKey,
    models: BTreeMap<String, LoadedCpuModel>,
    next_sequence: u64,
}

#[cfg(all(unix, feature = "cpu-reference"))]
impl CpuEngineRuntime {
    /// Creates an unloaded runtime around the Go-authorized control key.
    pub const fn new(service: EngineService, trusted_control_key: TrustedControlPlaneKey) -> Self {
        Self { service, trusted_control_key, models: BTreeMap::new(), next_sequence: 1 }
    }

    /// Verifies and installs one already bounded registered package in memory.
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
        self.models.insert(model_id.as_str().to_owned(), loaded);
        Ok(())
    }

    /// Opens, verifies, and loads one registered package from the configured
    /// owner-controlled model root.
    pub fn load_registered_model(
        &mut self,
        model_id: &ModelId,
        budget: &personalai_engine_types::ResourceBudget,
    ) -> Result<(), EngineError> {
        let bytes = self.service.registry.read_registered_package(model_id, budget)?;
        self.load_registered_model_bytes(model_id, &bytes, budget)
    }

    /// Accepts and processes exactly one authenticated local connection.
    ///
    /// A caller-owned outer loop decides service lifetime and restart policy.
    /// The method never opens a network connection or performs filesystem model
    /// discovery.
    pub fn serve_one(&mut self, listener: &UnixFrameListener) -> Result<ServeOutcome, EngineError> {
        let mut connection = listener.accept()?;
        let frame = connection.read_inbound_frame()?;
        let mut session = AuthenticatedControlSession::new(self.trusted_control_key.clone());
        match session.decode_command(&frame)? {
            InboundCommand::Cancel(request_id) => {
                let outcome = self.service.scheduler.cancel(&request_id)?;
                Ok(match outcome {
                    CancellationOutcome::Queued => ServeOutcome::CancelledQueued(request_id),
                    CancellationOutcome::Active => ServeOutcome::CancelledActive(request_id),
                    CancellationOutcome::NotFound => ServeOutcome::CancellationNotFound(request_id),
                })
            }
            InboundCommand::Generate(request) => {
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
                let generation_result = thread::scope(|scope| {
                    let watcher = scope.spawn(|| {
                        watch_generation_cancellation(
                            cancellation_connection,
                            session,
                            request_id,
                            &cancelled,
                            &watcher_done,
                        );
                    });
                    let result =
                        self.service.run_next_greedy(model, sequence_id, &cancelled, |event| {
                            write_generation_event(&mut connection, &event)
                        });
                    watcher_done.store(true, Ordering::Release);
                    let shutdown_result = connection.shutdown_inbound();
                    watcher.join().map_err(|_| {
                        EngineError::internal_invariant("cancellation watcher panicked")
                    })?;
                    shutdown_result?;
                    result
                });
                let usage = generation_result?.ok_or_else(|| {
                    EngineError::internal_invariant("admitted request was not scheduled")
                })?;
                Ok(ServeOutcome::Generated { request_id, sequence_id, usage })
            }
        }
    }

    /// Returns the service configuration and registry composition.
    pub const fn service(&self) -> &EngineService {
        &self.service
    }
}

#[cfg(all(unix, feature = "cpu-reference"))]
fn watch_generation_cancellation(
    mut connection: UnixFrameConnection,
    mut session: AuthenticatedControlSession,
    active_request_id: RequestId,
    cancelled: &AtomicBool,
    watcher_done: &AtomicBool,
) {
    let command = connection.read_inbound_frame().and_then(|frame| session.decode_command(&frame));
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
fn write_generation_event(
    connection: &mut UnixFrameConnection,
    event: &GenerationEvent,
) -> Result<(), EngineError> {
    let payload = personalai_engine_protocol::encode_generation_event(event)?;
    let frame = Frame::new(FrameKind::GenerationEvent, payload)?;
    connection.write_outbound_frame(&frame)
}

#[cfg(feature = "cpu-reference")]
fn check_cancelled(cancelled: &AtomicBool, request_id: RequestId) -> Result<(), EngineError> {
    if cancelled.load(Ordering::Acquire) {
        Err(EngineError::cancellation(Some(request_id)))
    } else {
        Ok(())
    }
}

#[cfg(feature = "cpu-reference")]
fn append_streaming_piece(
    tokenizer: &dyn Tokenizer,
    token: u32,
    pending: &mut [u8; 1_024],
    pending_length: &mut usize,
) -> Result<String, EngineError> {
    if tokenizer.is_special_token(token)? {
        return Ok(String::new());
    }
    let piece = tokenizer.token_piece(token)?;
    if piece.len() > usize::from(MAX_TOKEN_PIECE_BYTES) {
        return Err(EngineError::internal_invariant(
            "validated tokenizer piece exceeds its hard bound",
        ));
    }
    let end = pending_length
        .checked_add(piece.len())
        .filter(|end| *end <= pending.len())
        .ok_or_else(|| EngineError::backend_failure("generated UTF-8 prefix remains invalid"))?;
    pending[*pending_length..end].copy_from_slice(piece);
    *pending_length = end;
    match std::str::from_utf8(&pending[..end]) {
        Ok(text) => {
            let text = text.to_owned();
            *pending_length = 0;
            Ok(text)
        }
        Err(error) if error.error_len().is_none() => Ok(String::new()),
        Err(_) => Err(EngineError::backend_failure("generated token bytes are not valid UTF-8")),
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

    /// Admits a request only when its model ID exists in local configuration.
    pub fn admit(&mut self, request: GenerationRequest) -> Result<AdmissionReceipt, EngineError> {
        let request_id = request.request_id();
        self.registry.resolve(request.model_id())?;
        self.scheduler.submit(request)?;
        Ok(AdmissionReceipt { request_id, queued_requests: self.scheduler.queued_len() })
    }

    /// Verifies and materializes bytes obtained for one registered model.
    ///
    /// This method does not open a path. A future descriptor-safe local opener
    /// may call it only after resolving the configuration-owned registration.
    #[cfg(feature = "cpu-reference")]
    pub fn load_registered_cpu_model_bytes(
        &self,
        model_id: &ModelId,
        package_bytes: &[u8],
        budget: &personalai_engine_types::ResourceBudget,
    ) -> Result<LoadedCpuModel, EngineError> {
        let pool = MemoryPool::from_budget(budget)?;
        let verified = self.registry.verify_package_bytes(model_id, package_bytes, budget)?;
        let config = TransformerConfig::from_manifest(verified.parsed().manifest())?;
        let tokenizer_data = verified.parsed().tokenizer_data(package_bytes)?;
        let tokenizer = build_tokenizer(tokenizer_data, &pool)?;
        if tokenizer.spec().vocabulary_size() != config.vocabulary_size() {
            return Err(EngineError::corrupt_model_package(
                "manifest and tokenizer vocabulary sizes differ",
            ));
        }
        let weights = verified.materialize_tensors(&pool)?.into_tensors();
        let model = CpuDecoderModel::new(config, weights, &pool)?;
        Ok(LoadedCpuModel { model_id: model_id.clone(), model, tokenizer })
    }

    /// Activates and runs the next queued request on an already loaded model.
    #[cfg(feature = "cpu-reference")]
    pub fn run_next_greedy(
        &mut self,
        loaded: &LoadedCpuModel,
        sequence_id: SequenceId,
        cancelled: &AtomicBool,
        mut emit: impl FnMut(GenerationEvent) -> Result<(), EngineError>,
    ) -> Result<Option<GenerationUsage>, EngineError> {
        let Some(active) = self.scheduler.start_next(sequence_id)?.cloned() else {
            return Ok(None);
        };
        let request_id = active.request().request_id();
        let result = loaded.generate_greedy(active.request(), sequence_id, cancelled, &mut emit);
        self.scheduler.finish(&request_id)?;
        match result {
            Ok(usage) => Ok(Some(usage)),
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
    use ed25519_dalek::{Signer, SigningKey};
    use personalai_engine_crypto::control_plane_signature_message;
    use personalai_engine_protocol::{
        CancellationEnvelope, GenerationRequestEnvelope, attach_cancellation_signature,
        attach_generation_signature,
    };
    #[cfg(feature = "cpu-reference")]
    use personalai_engine_testkit::DecoderPackageFixtureBuilder;
    use personalai_engine_types::{ContextLimit, ModelHashAlgorithm, ResourceBudget, SignerKeyId};

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
        let mut session = AuthenticatedControlSession::new(trusted);
        assert!(matches!(session.decode_command(&frame), Ok(InboundCommand::Generate(_))));
        assert_eq!(session.last_nonce(), 1);
        assert!(matches!(
            session.decode_command(&frame),
            Err(EngineError::AuthenticationFailure(_))
        ));
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
        let mut session = AuthenticatedControlSession::new(trusted_key);
        assert!(matches!(session.decode_command(&first), Ok(InboundCommand::Generate(_))));
        let cancelled = AtomicBool::new(false);
        let done = AtomicBool::new(false);
        watch_generation_cancellation(connection, session, request.request_id(), &cancelled, &done);
        assert!(cancelled.load(Ordering::Acquire));
        release_client.send(()).expect("client must be released");
        client.join().expect("client must finish");
        drop(listener);
        fs::remove_dir(&runtime_root).expect("runtime root must clean");
    }

    #[test]
    fn protocol_truthfully_reports_platform_listener_support() {
        let descriptor = service().protocol_descriptor();
        assert_eq!(descriptor.major, 1);
        assert_eq!(descriptor.listener_implemented, cfg!(unix));
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
        let mut session = loaded.model().start_session(&budget).expect("session must start");
        let mut logits = None;
        for token in tokens {
            logits = Some(session.forward_token(token).expect("token must execute"));
        }
        let logits = logits.expect("prompt must contain at least one token");
        assert_eq!(logits.shape().dimensions(), &[258]);
        assert_eq!(session.token_count(), 2);
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

    #[cfg(all(unix, feature = "cpu-reference"))]
    #[test]
    fn authenticated_unix_runtime_executes_signed_request_end_to_end() {
        use std::fs;
        use std::os::unix::fs::PermissionsExt;
        use std::sync::atomic::AtomicU64;
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
        let mut runtime = CpuEngineRuntime::new(service, trusted_key);
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
        let client_path = socket_path.clone();
        let client = thread::spawn(move || runtime_test_client(client_path, &encoded));
        let outcome = runtime.serve_one(&listener).expect("runtime must serve generation");
        assert!(
            matches!(outcome, ServeOutcome::Generated { request_id, .. } if request_id.get() == 404)
        );
        let event_kinds = client.join().expect("client must finish");
        assert_eq!(event_kinds.first(), Some(&1));
        assert_eq!(event_kinds.last(), Some(&3));
        assert!(event_kinds.iter().any(|kind| *kind == 2));
        drop(listener);
        fs::remove_dir(&runtime_root).expect("runtime root must clean");
    }

    #[cfg(all(unix, feature = "cpu-reference"))]
    fn runtime_test_client(socket_path: PathBuf, encoded: &[u8]) -> Vec<u8> {
        use personalai_engine_protocol::FRAME_HEADER_BYTES;
        use std::io::{Read, Write};
        use std::os::unix::net::UnixStream;

        let mut stream = UnixStream::connect(socket_path).expect("client must connect");
        stream.write_all(encoded).expect("request must write");
        let mut event_kinds = Vec::new();
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
            event_kinds.push(event.payload()[10]);
            if event.payload()[10] == 3 {
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
