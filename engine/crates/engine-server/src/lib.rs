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
#[cfg(unix)]
mod transport;

#[cfg(all(unix, feature = "cpu-reference"))]
pub use continuous::{
    ContinuousServeLimits, ContinuousServeSummary, ContinuousShutdown, MAX_CONTINUOUS_CONNECTIONS,
};
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
    CancellationEnvelope, FRAME_HEADER_BYTES, Frame, FrameKind, GENERATION_EVENT_AUTH_BYTES,
    GenerationRequestEnvelope, PROTOCOL_MAJOR, PROTOCOL_MINOR,
};
#[cfg(feature = "cpu-reference")]
use personalai_engine_sampling::{SamplingConfig, select_next_token_tensor};
#[cfg(all(unix, feature = "cpu-reference"))]
use personalai_engine_scheduler::CancellationOutcome;
use personalai_engine_scheduler::{MAX_QUEUED_REQUEST_BYTES, Scheduler, SchedulerLimits};
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
use personalai_engine_types::{
    GenerationEvent, GenerationUsage, MAX_CONCURRENT_SEQUENCES, MAX_GENERATION_CHUNK_BYTES,
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

/// Maximum configured Unix socket path bytes for conservative portability.
pub const MAX_UNIX_SOCKET_PATH_BYTES: usize = 100;
/// Hard ceiling for the bounded multi-connection reference batch.
pub const MAX_CONCURRENT_CONNECTION_BATCH: u32 = 64;
/// Maximum signed generation events retained for one local output stream.
#[cfg(all(unix, feature = "cpu-reference"))]
pub const MAX_PENDING_EVENTS_PER_CONNECTION: usize = 8;
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
        request: &GenerationRequest,
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
            request: request.clone(),
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
    request: GenerationRequest,
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

    const fn concurrent_sequence_limit(&self) -> u32 {
        self.request.resource_budget().max_concurrent_sequences()
    }
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
struct ScheduledGeneration<'a> {
    generation: GreedyGeneration<'a>,
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
pub struct GenerationRoundRobin<'a> {
    maximum_active: u32,
    maximum_retained_bytes: usize,
    retained_bytes: usize,
    ready: VecDeque<ScheduledGeneration<'a>>,
}

#[cfg(feature = "cpu-reference")]
impl<'a> GenerationRoundRobin<'a> {
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
        model: &'a LoadedCpuModel,
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
        model: &'a LoadedCpuModel,
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
        let generation = model.start_greedy(request, sequence_id)?;
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
        let advance = scheduled.generation.advance(&scheduled.cancelled);
        let after = scheduled.generation.processed_tokens();
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
    models: &'a BTreeMap<String, LoadedCpuModel>,
    response_signer: &'a EngineResponseSigner,
    next_sequence: &'a mut u64,
    active: &'a mut Vec<ActiveGenerationConnection>,
    results: &'a mut [Option<ConnectionServeResult>],
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

        let mut coordinator = GenerationRoundRobin::with_limits(
            maximum_active,
            self.service.scheduler.limits().maximum_queued_bytes(),
        )?;
        let mut active = Vec::new();
        active
            .try_reserve_exact(authenticated.len())
            .map_err(|_| EngineError::resource_exhaustion("active connection allocation failed"))?;
        let mut batch = ConnectionBatchState {
            service: &mut self.service,
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
            batch.response_signer,
            &mut coordinator,
            batch.active,
            batch.results,
        )?;
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

    fn prepare_authenticated_connection<'a>(
        batch: &mut ConnectionBatchState<'a>,
        authenticated: AuthenticatedConnection,
        coordinator: &mut GenerationRoundRobin<'a>,
    ) -> Result<(), EngineError> {
        let AuthenticatedConnection { input_index, connection, session, replay_journal, command } =
            authenticated;
        let InboundCommand::Generate { request, nonce } = command else {
            let InboundCommand::Cancel(request_id) = command else {
                return Err(EngineError::internal_invariant(
                    "authenticated command variant disappeared",
                ));
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
        response_signer: &EngineResponseSigner,
        coordinator: &mut GenerationRoundRobin<'_>,
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
fn prepare_generation_connection<'a>(
    batch: &mut ConnectionBatchState<'a>,
    coordinator: &mut GenerationRoundRobin<'a>,
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
    if let Err(error) = coordinator.submit_with_retained_bytes(
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
    response_signer: &EngineResponseSigner,
    coordinator: &mut GenerationRoundRobin<'_>,
    active: &mut Vec<ActiveGenerationConnection>,
    results: &mut [Option<ConnectionServeResult>],
) -> Result<(), EngineError> {
    while let Some(turn) = coordinator.advance_next()? {
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
        assert_eq!(fixture.runtime.service.scheduler.queued_len(), 0);
        assert_eq!(fixture.runtime.service.scheduler.active_len(), 0);
        fixture.cleanup();
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
            let mut runtime = CpuEngineRuntime::new(service, trusted_key, response_signer)
                .expect("runtime must construct");
            runtime
                .load_registered_model_bytes(&model_id, fixture.bytes(), &budget)
                .expect("model must load");
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
