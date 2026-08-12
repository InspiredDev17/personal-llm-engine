//! Local-only composition boundary for the native inference engine.
//!
//! This crate validates Unix-domain-socket configuration, admits requests for
//! locally registered model IDs, and can execute the deterministic CPU
//! reference path. It contains no TCP or outbound network capability, tool
//! path, approval authority, arbitrary model path, or platform/model persistence
//! authority. Its only mutable disk state is the bounded, configuration-derived
//! replay journal required before authenticated command execution.

#[cfg(unix)]
mod os_lock;
#[cfg(unix)]
mod replay;
#[cfg(unix)]
mod transport;

#[cfg(unix)]
pub use replay::{DurableReplayJournal, MAX_DURABLE_REPLAY_ENTRIES};
#[cfg(unix)]
pub use transport::{UnixFrameConnection, UnixFrameListener};

use personalai_engine_crypto::{
    EngineResponseSigner, TrustedControlPlaneKey, TrustedControlPlaneKeys, TrustedSigner,
};
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
use personalai_engine_transformer::{CpuDecoderModel, CpuDecoderSession, TransformerConfig};
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
use std::sync::{Arc, Mutex};
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

    /// Creates one validated, resumable greedy-generation sequence.
    ///
    /// Each call to [`GreedyGeneration::advance`] performs at most one decoder
    /// token step. This is the scheduling boundary used to interleave sequences
    /// without sharing their KV caches or cancellation state.
    pub fn start_greedy<'a>(
        &'a self,
        request: &'a GenerationRequest,
        sequence_id: SequenceId,
    ) -> Result<GreedyGeneration<'a>, EngineError> {
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
        Ok(GreedyGeneration {
            request,
            sequence_id,
            model: &self.model,
            tokenizer: self.tokenizer.as_ref(),
            session: None,
            input_tokens,
            input_count,
            next_input: 0,
            next_generated_token: None,
            generated: 0,
            pending_utf8: [0_u8; 1_024],
            pending_length: 0,
            started: false,
            terminal: None,
            usage: None,
        })
    }

    /// Runs deterministic greedy generation and synchronously emits bounded events.
    pub fn generate_greedy(
        &self,
        request: &GenerationRequest,
        sequence_id: SequenceId,
        cancelled: &AtomicBool,
        mut emit: impl FnMut(GenerationEvent) -> Result<(), EngineError>,
    ) -> Result<GenerationUsage, EngineError> {
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
                        return Ok(usage);
                    }
                }
            }
        }
    }
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
pub struct GreedyGeneration<'a> {
    request: &'a GenerationRequest,
    sequence_id: SequenceId,
    model: &'a CpuDecoderModel,
    tokenizer: &'a dyn Tokenizer,
    session: Option<CpuDecoderSession<'a>>,
    input_tokens: Vec<u32>,
    input_count: u32,
    next_input: usize,
    next_generated_token: Option<u32>,
    generated: u32,
    pending_utf8: [u8; 1_024],
    pending_length: usize,
    started: bool,
    terminal: Option<TerminalState>,
    usage: Option<GenerationUsage>,
}

#[cfg(feature = "cpu-reference")]
impl GreedyGeneration<'_> {
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
                GenerationUsage::for_request(self.request, self.input_count, self.generated)?;
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
            self.session = Some(self.model.start_session(&self.request.resource_budget())?);
            return Ok(GreedyAdvance::Progressed);
        }
        let session = self.session.as_mut().ok_or_else(|| {
            EngineError::internal_invariant("generation decoder session disappeared")
        })?;
        let logits = if let Some(token) = self.input_tokens.get(self.next_input).copied() {
            let logits = session.forward_token(token)?;
            self.next_input = self.next_input.checked_add(1).ok_or_else(|| {
                EngineError::internal_invariant("generation input position overflows")
            })?;
            if self.next_input < self.input_tokens.len() {
                return Ok(GreedyAdvance::Progressed);
            }
            logits
        } else if let Some(token) = self.next_generated_token.take() {
            session.forward_token(token)?
        } else {
            return Err(EngineError::internal_invariant("tokenizer produced no prompt tokens"));
        };
        let token = select_next_token_tensor(&logits, SamplingConfig::greedy())?;
        self.generated = self
            .generated
            .checked_add(1)
            .ok_or_else(|| EngineError::internal_invariant("generated token count overflows"))?;
        let text = append_streaming_piece(
            self.tokenizer,
            token,
            &mut self.pending_utf8,
            &mut self.pending_length,
        )?;
        let end_of_sequence = token == self.tokenizer.spec().end_of_sequence_token();
        let output_limit = self.generated >= self.request.max_output_tokens();
        if end_of_sequence || output_limit {
            self.terminal = Some(if self.pending_length == 0 {
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
        self.session.as_ref().map_or(0, CpuDecoderSession::token_count)
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
    trusted_control_keys: TrustedControlPlaneKeys,
    response_signer: EngineResponseSigner,
    replay_journals:
        BTreeMap<personalai_engine_types::SignerKeyId, Arc<Mutex<DurableReplayJournal>>>,
    models: BTreeMap<String, LoadedCpuModel>,
    next_sequence: u64,
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
            next_sequence: 1,
        })
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
            InboundCommand::Cancel(request_id) => {
                let outcome = self.service.scheduler.cancel(&request_id)?;
                Ok(match outcome {
                    CancellationOutcome::Queued => ServeOutcome::CancelledQueued(request_id),
                    CancellationOutcome::Active => ServeOutcome::CancelledActive(request_id),
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
                    let result =
                        self.service.run_next_greedy(model, sequence_id, &cancelled, |event| {
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
    let payload = personalai_engine_protocol::encode_signed_generation_event(
        event,
        request_nonce,
        event_ordinal,
        signer,
    )?;
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
    use ed25519_dalek::{Signature, Signer, SigningKey, VerifyingKey};
    use personalai_engine_crypto::{
        control_plane_signature_message, engine_response_signature_message,
    };
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
        let mut first =
            loaded.start_greedy(&first_request, first_sequence).expect("first cursor must start");
        let mut second = loaded
            .start_greedy(&second_request, second_sequence)
            .expect("second cursor must start");
        let first_cancelled = AtomicBool::new(false);
        let second_cancelled = AtomicBool::new(false);
        let mut first_events = Vec::new();
        let mut second_events = Vec::new();
        let mut completion_order = Vec::new();

        for _round in 0..64 {
            if !first.is_complete() {
                let before = first.processed_tokens();
                if let GreedyAdvance::Event(event) =
                    first.advance(&first_cancelled).expect("first cursor must advance")
                {
                    if matches!(event, GenerationEvent::Completed { .. }) {
                        completion_order.push(event.request_id());
                    }
                    first_events.push(event);
                }
                let advanced = first
                    .processed_tokens()
                    .checked_sub(before)
                    .expect("first cursor token count must not decrease");
                assert!(advanced <= 1);
            }
            if !second.is_complete() {
                let before = second.processed_tokens();
                if let GreedyAdvance::Event(event) =
                    second.advance(&second_cancelled).expect("second cursor must advance")
                {
                    if matches!(event, GenerationEvent::Completed { .. }) {
                        completion_order.push(event.request_id());
                    }
                    second_events.push(event);
                }
                let advanced = second
                    .processed_tokens()
                    .checked_sub(before)
                    .expect("second cursor token count must not decrease");
                assert!(advanced <= 1);
            }
            if first.is_complete() && second.is_complete() {
                break;
            }
        }

        assert!(first.is_complete());
        assert!(second.is_complete());
        assert_eq!(
            completion_order,
            [second_request.request_id(), first_request.request_id()],
            "the shorter prompt must complete without waiting for the longer sequence"
        );
        assert!(matches!(first_events.first(), Some(GenerationEvent::Started { .. })));
        assert!(matches!(first_events.last(), Some(GenerationEvent::Completed { .. })));
        assert!(matches!(second_events.first(), Some(GenerationEvent::Started { .. })));
        assert!(matches!(second_events.last(), Some(GenerationEvent::Completed { .. })));
        assert_eq!(first.request_id(), first_request.request_id());
        assert_eq!(second.sequence_id(), second_sequence);
        assert!(first.usage().is_some());
        assert!(second.usage().is_some());
        assert!(matches!(first.advance(&first_cancelled), Err(EngineError::ProtocolViolation(_))));
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
        let client =
            thread::spawn(move || runtime_test_client(client_path, &encoded, response_public_key));
        let outcome = runtime.serve_one(&listener).expect("runtime must serve generation");
        assert!(
            matches!(outcome, ServeOutcome::Generated { request_id, .. } if request_id.get() == 404)
        );
        let event_kinds = client.join().expect("client must finish");
        assert_eq!(event_kinds.first(), Some(&1));
        assert_eq!(event_kinds.last(), Some(&3));
        assert!(event_kinds.iter().any(|kind| *kind == 2));

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
    fn runtime_test_client(
        socket_path: PathBuf,
        encoded: &[u8],
        response_public_key: [u8; 32],
    ) -> Vec<u8> {
        use personalai_engine_protocol::{FRAME_HEADER_BYTES, GENERATION_EVENT_AUTH_BYTES};
        use std::io::{Read, Write};
        use std::os::unix::net::UnixStream;

        let mut stream = UnixStream::connect(socket_path).expect("client must connect");
        stream.write_all(encoded).expect("request must write");
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
            assert_eq!(&payload[auth_start..auth_start + 8], &1_u64.to_le_bytes());
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
