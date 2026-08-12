//! Bounded continuous admission and graceful-drain reference service.

use super::{
    AuthenticatedControlSession, BTreeMap, CANCELLATION_WATCHER_STACK_BYTES, CancellationOutcome,
    ConnectionServeResult, CpuEngineRuntime, DurableReplayJournal, EngineError,
    EngineResponseSigner, EngineService, Frame, GenerationEvent, GenerationRequest,
    GenerationRoundRobin, InboundCommand, LoadedCpuModel, MAX_CONCURRENT_CONNECTION_BATCH,
    MAX_PENDING_EVENTS_PER_CONNECTION, OUTBOUND_PIPELINE_RESERVED_BYTES, OUTPUT_WRITER_STACK_BYTES,
    RequestId, SequenceId, ServeOutcome, SyncSender, TrustedControlPlaneKeys, UnixFrameConnection,
    UnixFrameListener, activate_queued_request, admit_durable_nonce, enqueue_outbound_frame,
    generation_event_frame, terminal_connection_result, watch_generation_cancellation,
    write_queued_events,
};
use personalai_engine_types::SignerKeyId;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::Duration;

/// Hard ceiling spanning authenticating, queued, active, and draining streams.
pub const MAX_CONTINUOUS_CONNECTIONS: u32 = MAX_CONCURRENT_CONNECTION_BATCH;

const CONTINUOUS_IDLE_POLL: Duration = Duration::from_millis(1);
const INGRESS_WORKER_STACK_BYTES: usize = 256 * 1_024;

/// Validated connection ceiling for one continuous reference service run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ContinuousServeLimits {
    maximum_connections: u32,
}

impl ContinuousServeLimits {
    /// Creates a positive ceiling no larger than the engine hard limit.
    pub fn new(maximum_connections: u32) -> Result<Self, EngineError> {
        if maximum_connections == 0 || maximum_connections > MAX_CONTINUOUS_CONNECTIONS {
            return Err(EngineError::invalid_input(
                "continuous connection limit is outside hard bounds",
            ));
        }
        Ok(Self { maximum_connections })
    }

    /// Returns the total authenticating, queued, active, and draining ceiling.
    pub const fn maximum_connections(self) -> u32 {
        self.maximum_connections
    }
}

/// Cloneable caller-owned graceful-shutdown signal.
#[derive(Clone, Debug, Default)]
pub struct ContinuousShutdown {
    requested: Arc<AtomicBool>,
}

impl ContinuousShutdown {
    /// Creates a signal in the running state.
    pub fn new() -> Self {
        Self::default()
    }

    /// Requests an idempotent stop-admit, cancel, and drain transition.
    pub fn request(&self) {
        self.requested.store(true, Ordering::Release);
    }

    /// Returns whether graceful shutdown was requested.
    pub fn is_requested(&self) -> bool {
        self.requested.load(Ordering::Acquire)
    }
}

/// Bounded counters returned after a continuous reference run drains.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ContinuousServeSummary {
    accepted_connections: u64,
    authenticated_commands: u64,
    completed_generations: u64,
    cancelled_commands: u64,
    failed_connections: u64,
    peak_owned_connections: u32,
}

impl ContinuousServeSummary {
    /// Returns accepted local streams.
    pub const fn accepted_connections(self) -> u64 {
        self.accepted_connections
    }

    /// Returns initial commands that passed signature and durable replay admission.
    pub const fn authenticated_commands(self) -> u64 {
        self.authenticated_commands
    }

    /// Returns generations whose signed output writers completed successfully.
    pub const fn completed_generations(self) -> u64 {
        self.completed_generations
    }

    /// Returns separate authenticated cancellation streams processed by the runtime.
    pub const fn cancelled_commands(self) -> u64 {
        self.cancelled_commands
    }

    /// Returns streams rejected or failed before successful completion.
    pub const fn failed_connections(self) -> u64 {
        self.failed_connections
    }

    /// Returns the largest simultaneous authenticating, queued, active, and draining count.
    pub const fn peak_owned_connections(self) -> u32 {
        self.peak_owned_connections
    }
}

struct ContinuousAuthenticatedConnection {
    connection: UnixFrameConnection,
    session: AuthenticatedControlSession,
    replay_journal: Arc<Mutex<DurableReplayJournal>>,
    command: InboundCommand,
}

struct IngressWorker {
    control: UnixFrameConnection,
    handle: JoinHandle<Result<ContinuousAuthenticatedConnection, EngineError>>,
}

struct StreamWorkers {
    control: UnixFrameConnection,
    outbound_sender: Option<SyncSender<Frame>>,
    writer_failed: Arc<AtomicBool>,
    cancelled: Arc<AtomicBool>,
    watcher_done: Arc<AtomicBool>,
    watcher: JoinHandle<()>,
    writer: JoinHandle<Result<(), EngineError>>,
}

struct PendingGenerationConnection {
    request: GenerationRequest,
    request_nonce: u64,
    workers: StreamWorkers,
}

struct ContinuousActiveConnection {
    request_id: RequestId,
    request_nonce: u64,
    sequence_id: SequenceId,
    next_event_ordinal: u32,
    stream_error: Option<EngineError>,
    workers: StreamWorkers,
}

struct RetiringConnection {
    result: ConnectionServeResult,
    watcher: JoinHandle<()>,
    writer: JoinHandle<Result<(), EngineError>>,
}

struct ContinuousState<'a> {
    service: &'a mut EngineService,
    trusted_control_keys: Arc<TrustedControlPlaneKeys>,
    response_signer: &'a EngineResponseSigner,
    replay_journals: Arc<BTreeMap<SignerKeyId, Arc<Mutex<DurableReplayJournal>>>>,
    models: &'a BTreeMap<String, LoadedCpuModel>,
    next_sequence: &'a mut u64,
    coordinator: GenerationRoundRobin<'a>,
    ingress: Vec<IngressWorker>,
    pending: Vec<PendingGenerationConnection>,
    active: Vec<ContinuousActiveConnection>,
    retiring: Vec<RetiringConnection>,
    summary: ContinuousServeSummary,
    maximum_connections: usize,
}

impl CpuEngineRuntime {
    /// Serves bounded continuously admitted local streams until shutdown.
    ///
    /// Pending authentication, queued requests, active generation, and draining
    /// output writers share one hard connection ceiling. The caller owns the
    /// shutdown signal. On request, admission stops, incomplete handshakes are
    /// interrupted, queued work is rejected, active work is cancelled, signed
    /// terminal events drain, and all workers are joined before return.
    pub fn serve_continuous(
        &mut self,
        listener: &UnixFrameListener,
        limits: ContinuousServeLimits,
        shutdown: &ContinuousShutdown,
    ) -> Result<ContinuousServeSummary, EngineError> {
        if self.service.scheduler.queued_len() != 0 || self.service.scheduler.active_len() != 0 {
            return Err(EngineError::protocol_violation(
                "continuous serving requires an idle scheduler",
            ));
        }
        listener.set_nonblocking(true)?;
        let result = self.run_continuous(listener, limits, shutdown);
        let restore = listener.set_nonblocking(false);
        match (result, restore) {
            (Ok(summary), Ok(())) => Ok(summary),
            (Err(error), _) => Err(error),
            (Ok(_summary), Err(error)) => Err(error),
        }
    }

    fn run_continuous(
        &mut self,
        listener: &UnixFrameListener,
        limits: ContinuousServeLimits,
        shutdown: &ContinuousShutdown,
    ) -> Result<ContinuousServeSummary, EngineError> {
        let maximum_active = self.service.scheduler.limits().maximum_active();
        let maximum_retained = self.service.scheduler.limits().maximum_queued_bytes();
        let coordinator = GenerationRoundRobin::with_limits(maximum_active, maximum_retained)?;
        let maximum_connections = usize::try_from(limits.maximum_connections()).map_err(|_| {
            EngineError::resource_exhaustion("continuous connection limit is not addressable")
        })?;
        let mut state = ContinuousState {
            service: &mut self.service,
            trusted_control_keys: Arc::new(self.trusted_control_keys.clone()),
            response_signer: &self.response_signer,
            replay_journals: Arc::new(self.replay_journals.clone()),
            models: &self.models,
            next_sequence: &mut self.next_sequence,
            coordinator,
            ingress: bounded_vec(maximum_connections, "continuous ingress")?,
            pending: bounded_vec(maximum_connections, "continuous pending")?,
            active: bounded_vec(maximum_connections, "continuous active")?,
            retiring: bounded_vec(maximum_connections, "continuous retiring")?,
            summary: ContinuousServeSummary::default(),
            maximum_connections,
        };
        state.run(listener, shutdown)
    }
}

impl ContinuousState<'_> {
    fn run(
        &mut self,
        listener: &UnixFrameListener,
        shutdown: &ContinuousShutdown,
    ) -> Result<ContinuousServeSummary, EngineError> {
        let mut draining = false;
        loop {
            let mut progressed = false;
            if shutdown.is_requested() && !draining {
                self.begin_graceful_drain()?;
                draining = true;
                progressed = true;
            }
            if !draining {
                progressed |= self.accept_ready(listener)?;
            }
            progressed |= self.collect_finished_ingress(draining)?;
            progressed |= self.cancel_disconnected_pending()?;
            if !draining {
                progressed |= self.activate_pending()?;
            }
            progressed |= self.drive_one_generation()?;
            progressed |= self.reap_retiring()?;
            self.update_peak()?;

            if draining && self.is_drained() {
                break;
            }
            if !progressed {
                thread::sleep(CONTINUOUS_IDLE_POLL);
            }
        }
        if !self.coordinator.is_empty()
            || self.service.scheduler.queued_len() != 0
            || self.service.scheduler.active_len() != 0
        {
            return Err(EngineError::internal_invariant(
                "continuous drain left scheduler state behind",
            ));
        }
        Ok(self.summary)
    }

    fn accept_ready(&mut self, listener: &UnixFrameListener) -> Result<bool, EngineError> {
        let mut accepted = false;
        while self.owned_connections() < self.maximum_connections {
            let connection = match listener.try_accept() {
                Ok(Some(connection)) => connection,
                Ok(None) => break,
                Err(EngineError::AuthenticationFailure(_)) => {
                    accepted = true;
                    increment(&mut self.summary.accepted_connections, "accepted connection count")?;
                    increment(&mut self.summary.failed_connections, "failed connection count")?;
                    continue;
                }
                Err(error) => return Err(error),
            };
            accepted = true;
            increment(&mut self.summary.accepted_connections, "accepted connection count")?;
            let control = match connection.try_clone() {
                Ok(control) => control,
                Err(_error) => {
                    increment(&mut self.summary.failed_connections, "failed connection count")?;
                    continue;
                }
            };
            let trusted_keys = self.trusted_control_keys.clone();
            let replay_journals = self.replay_journals.clone();
            let Ok(handle) = thread::Builder::new()
                .name("personalai-ingress".to_owned())
                .stack_size(INGRESS_WORKER_STACK_BYTES)
                .spawn(move || {
                    authenticate_continuous(
                        connection,
                        trusted_keys.as_ref(),
                        replay_journals.as_ref(),
                    )
                })
            else {
                increment(&mut self.summary.failed_connections, "failed connection count")?;
                continue;
            };
            self.ingress.push(IngressWorker { control, handle });
        }
        Ok(accepted)
    }

    fn collect_finished_ingress(&mut self, draining: bool) -> Result<bool, EngineError> {
        let mut progressed = false;
        let mut index = self.ingress.len();
        while index > 0 {
            index -= 1;
            if !self.ingress[index].handle.is_finished() {
                continue;
            }
            progressed = true;
            let worker = self.ingress.swap_remove(index);
            let result = worker
                .handle
                .join()
                .map_err(|_| EngineError::internal_invariant("ingress worker panicked"))?;
            match result {
                Ok(authenticated) if !draining => self.accept_authenticated(authenticated)?,
                Ok(_) | Err(_) => {
                    increment(&mut self.summary.failed_connections, "failed connection count")?;
                }
            }
        }
        Ok(progressed)
    }

    fn accept_authenticated(
        &mut self,
        authenticated: ContinuousAuthenticatedConnection,
    ) -> Result<(), EngineError> {
        increment(&mut self.summary.authenticated_commands, "authenticated command count")?;
        let ContinuousAuthenticatedConnection { connection, session, replay_journal, command } =
            authenticated;
        match command {
            InboundCommand::Cancel(request_id) => {
                self.apply_cancellation(request_id)?;
                increment(&mut self.summary.cancelled_commands, "cancellation command count")
            }
            InboundCommand::Generate { request, nonce } => {
                let request_id = request.request_id();
                let workers =
                    match start_stream_workers(connection, session, replay_journal, request_id) {
                        Ok(workers) => workers,
                        Err(_error) => {
                            increment(
                                &mut self.summary.failed_connections,
                                "failed connection count",
                            )?;
                            return Ok(());
                        }
                    };
                if !self.models.contains_key(request.model_id().as_str()) {
                    self.retire_rejected(
                        workers,
                        &request,
                        nonce,
                        None,
                        EngineError::invalid_input("model is not loaded"),
                    );
                    return Ok(());
                }
                if let Err(error) = self.service.admit(request.clone()) {
                    self.retire_rejected(workers, &request, nonce, None, error);
                    return Ok(());
                }
                self.pending.push(PendingGenerationConnection {
                    request,
                    request_nonce: nonce,
                    workers,
                });
                Ok(())
            }
        }
    }

    fn activate_pending(&mut self) -> Result<bool, EngineError> {
        let mut progressed = false;
        while !self.pending.is_empty()
            && self.coordinator.active_len()
                < usize::try_from(self.service.scheduler.limits().maximum_active()).map_err(
                    |_| {
                        EngineError::internal_invariant("scheduler active limit is not addressable")
                    },
                )?
        {
            let request_id = self.pending[0].request.request_id();
            let model =
                self.models.get(self.pending[0].request.model_id().as_str()).ok_or_else(|| {
                    EngineError::internal_invariant("queued continuous model disappeared")
                })?;
            let (sequence_id, scheduled, following_sequence) =
                activate_queued_request(self.service, *self.next_sequence, request_id)?;
            *self.next_sequence = following_sequence;
            let submission = self.coordinator.submit_with_retained_bytes(
                model,
                &scheduled,
                sequence_id,
                Arc::clone(&self.pending[0].workers.cancelled),
                OUTBOUND_PIPELINE_RESERVED_BYTES,
            );
            if let Err(error) = submission {
                self.service.scheduler.finish(&request_id)?;
                let pending = self.pending.remove(0);
                self.retire_rejected(
                    pending.workers,
                    &pending.request,
                    pending.request_nonce,
                    Some(sequence_id),
                    error,
                );
                progressed = true;
                continue;
            }
            let pending = self.pending.remove(0);
            self.active.push(ContinuousActiveConnection {
                request_id,
                request_nonce: pending.request_nonce,
                sequence_id,
                next_event_ordinal: 1,
                stream_error: None,
                workers: pending.workers,
            });
            progressed = true;
        }
        Ok(progressed)
    }

    fn drive_one_generation(&mut self) -> Result<bool, EngineError> {
        let Some(turn) = self.coordinator.advance_next()? else {
            return Ok(false);
        };
        let request_id = turn.request_id();
        let position = self
            .active
            .iter()
            .position(|connection| connection.request_id == request_id)
            .ok_or_else(|| {
                EngineError::internal_invariant("continuous turn omitted its active connection")
            })?;
        let Some(event) = turn.into_event() else {
            return Ok(true);
        };
        let terminal = terminal_connection_result(&event, self.active[position].sequence_id)?;
        let write_result =
            enqueue_active_event(&mut self.active[position], &event, self.response_signer);
        if let Err(error) = write_result {
            if self.active[position].stream_error.is_none() {
                self.active[position].stream_error = Some(error);
            }
            if terminal.is_none() {
                self.active[position].workers.cancelled.store(true, Ordering::Release);
                return Ok(true);
            }
        }
        if let Some(terminal_result) = terminal {
            let result = self.active[position].stream_error.take().map_or(terminal_result, Err);
            self.finish_active(position, result)?;
        }
        Ok(true)
    }

    fn cancel_disconnected_pending(&mut self) -> Result<bool, EngineError> {
        let mut progressed = false;
        let mut index = self.pending.len();
        while index > 0 {
            index -= 1;
            if !self.pending[index].workers.cancelled.load(Ordering::Acquire) {
                continue;
            }
            progressed = true;
            let request_id = self.pending[index].request.request_id();
            match self.service.scheduler.cancel(&request_id)? {
                CancellationOutcome::Queued => {}
                CancellationOutcome::Active | CancellationOutcome::NotFound => {
                    return Err(EngineError::internal_invariant(
                        "pending continuous cancellation lost scheduler state",
                    ));
                }
            }
            let pending = self.pending.remove(index);
            self.retire_rejected(
                pending.workers,
                &pending.request,
                pending.request_nonce,
                None,
                EngineError::cancellation(Some(pending.request.request_id())),
            );
        }
        Ok(progressed)
    }

    fn apply_cancellation(&mut self, request_id: RequestId) -> Result<(), EngineError> {
        if let Some(active) =
            self.active.iter().find(|connection| connection.request_id == request_id)
        {
            active.workers.cancelled.store(true, Ordering::Release);
            return Ok(());
        }
        match self.service.scheduler.cancel(&request_id)? {
            CancellationOutcome::Queued => {
                let position = self
                    .pending
                    .iter()
                    .position(|connection| connection.request.request_id() == request_id)
                    .ok_or_else(|| {
                        EngineError::internal_invariant(
                            "cancelled queued request omitted its continuous connection",
                        )
                    })?;
                let pending = self.pending.remove(position);
                self.retire_rejected(
                    pending.workers,
                    &pending.request,
                    pending.request_nonce,
                    None,
                    EngineError::cancellation(Some(request_id)),
                );
                Ok(())
            }
            CancellationOutcome::Active | CancellationOutcome::NotFound => Ok(()),
        }
    }

    fn retire_rejected(
        &mut self,
        workers: StreamWorkers,
        request: &GenerationRequest,
        request_nonce: u64,
        sequence_id: Option<SequenceId>,
        error: EngineError,
    ) {
        let failed = GenerationEvent::Failed {
            request_id: request.request_id(),
            sequence_id,
            error: error.clone(),
        };
        let result = match generation_event_frame(&failed, request_nonce, 1, self.response_signer)
            .and_then(|frame| {
                let sender = workers.outbound_sender.as_ref().ok_or_else(|| {
                    EngineError::internal_invariant("continuous output sender disappeared")
                })?;
                enqueue_outbound_frame(sender, frame)
            }) {
            Ok(()) => Err(error),
            Err(output_error) => Err(output_error),
        };
        self.retire_workers(workers, result);
    }

    fn finish_active(
        &mut self,
        position: usize,
        result: ConnectionServeResult,
    ) -> Result<(), EngineError> {
        let request_id = self
            .active
            .get(position)
            .ok_or_else(|| {
                EngineError::internal_invariant("continuous active removal position is invalid")
            })?
            .request_id;
        self.service.scheduler.finish(&request_id)?;
        let active = self.active.remove(position);
        self.retire_workers(active.workers, result);
        Ok(())
    }

    fn retire_workers(&mut self, mut workers: StreamWorkers, result: ConnectionServeResult) {
        workers.watcher_done.store(true, Ordering::Release);
        let shutdown_result = workers.control.shutdown_inbound();
        workers.outbound_sender = None;
        let result = match shutdown_result {
            Ok(()) => result,
            Err(error) => Err(error),
        };
        self.retiring.push(RetiringConnection {
            result,
            watcher: workers.watcher,
            writer: workers.writer,
        });
    }

    fn reap_retiring(&mut self) -> Result<bool, EngineError> {
        let mut progressed = false;
        let mut index = self.retiring.len();
        while index > 0 {
            index -= 1;
            if !self.retiring[index].watcher.is_finished()
                || !self.retiring[index].writer.is_finished()
            {
                continue;
            }
            progressed = true;
            let retiring = self.retiring.swap_remove(index);
            let watcher_result = retiring
                .watcher
                .join()
                .map_err(|_| EngineError::internal_invariant("cancellation watcher panicked"));
            let writer_result = retiring
                .writer
                .join()
                .map_err(|_| EngineError::internal_invariant("output writer panicked"));
            watcher_result?;
            let writer_result = writer_result?;
            let result = match writer_result {
                Ok(()) => retiring.result,
                Err(error) => Err(error),
            };
            self.record_result(&result)?;
        }
        Ok(progressed)
    }

    fn record_result(&mut self, result: &ConnectionServeResult) -> Result<(), EngineError> {
        match result {
            Ok(ServeOutcome::Generated { .. }) => {
                increment(&mut self.summary.completed_generations, "completed generation count")
            }
            Ok(
                ServeOutcome::CancelledQueued(_)
                | ServeOutcome::CancelledActive(_)
                | ServeOutcome::CancellationNotFound(_),
            ) => increment(&mut self.summary.cancelled_commands, "cancellation outcome count"),
            Err(_) => increment(&mut self.summary.failed_connections, "failed connection count"),
        }
    }

    fn begin_graceful_drain(&mut self) -> Result<(), EngineError> {
        for worker in &self.ingress {
            let _shutdown_result = worker.control.shutdown_inbound();
        }
        while let Some(pending) = self.pending.pop() {
            let cancellation = self.service.scheduler.cancel(&pending.request.request_id());
            match cancellation {
                Ok(CancellationOutcome::Queued) => {}
                Ok(CancellationOutcome::Active | CancellationOutcome::NotFound) => {
                    self.retire_rejected(
                        pending.workers,
                        &pending.request,
                        pending.request_nonce,
                        None,
                        EngineError::internal_invariant(
                            "graceful drain lost queued scheduler state",
                        ),
                    );
                    return Err(EngineError::internal_invariant(
                        "graceful drain lost queued scheduler state",
                    ));
                }
                Err(error) => {
                    self.retire_rejected(
                        pending.workers,
                        &pending.request,
                        pending.request_nonce,
                        None,
                        error.clone(),
                    );
                    return Err(error);
                }
            }
            self.retire_rejected(
                pending.workers,
                &pending.request,
                pending.request_nonce,
                None,
                EngineError::cancellation(Some(pending.request.request_id())),
            );
        }
        for active in &self.active {
            active.workers.cancelled.store(true, Ordering::Release);
        }
        Ok(())
    }

    fn owned_connections(&self) -> usize {
        self.ingress.len() + self.pending.len() + self.active.len() + self.retiring.len()
    }

    fn update_peak(&mut self) -> Result<(), EngineError> {
        let current = u32::try_from(self.owned_connections())
            .map_err(|_| EngineError::internal_invariant("owned connection count exceeds u32"))?;
        self.summary.peak_owned_connections = self.summary.peak_owned_connections.max(current);
        if current > u32::try_from(self.maximum_connections).unwrap_or(u32::MAX) {
            return Err(EngineError::internal_invariant(
                "continuous connection ownership exceeded its ceiling",
            ));
        }
        Ok(())
    }

    fn is_drained(&self) -> bool {
        self.ingress.is_empty()
            && self.pending.is_empty()
            && self.active.is_empty()
            && self.retiring.is_empty()
    }
}

impl Drop for ContinuousState<'_> {
    fn drop(&mut self) {
        for worker in &self.ingress {
            let _shutdown_result = worker.control.shutdown_inbound();
        }
        for worker in self.ingress.drain(..) {
            let _join_result = worker.handle.join();
        }
        for pending in self.pending.drain(..) {
            let _cancel_result = self.service.scheduler.cancel(&pending.request.request_id());
            stop_stream_workers(pending.workers);
        }
        for active in self.active.drain(..) {
            let _finish_result = self.service.scheduler.finish(&active.request_id);
            stop_stream_workers(active.workers);
        }
        for retiring in self.retiring.drain(..) {
            let _watcher_join = retiring.watcher.join();
            let _writer_join = retiring.writer.join();
        }
    }
}

fn authenticate_continuous(
    mut connection: UnixFrameConnection,
    trusted_control_keys: &TrustedControlPlaneKeys,
    replay_journals: &BTreeMap<SignerKeyId, Arc<Mutex<DurableReplayJournal>>>,
) -> Result<ContinuousAuthenticatedConnection, EngineError> {
    let frame = connection.read_inbound_frame()?;
    let mut session = AuthenticatedControlSession::new_with_keys(trusted_control_keys.clone());
    let command = session.decode_command(&frame)?;
    let authenticated_key_id = session.authenticated_key_id().ok_or_else(|| {
        EngineError::internal_invariant("authenticated command omitted its control key ID")
    })?;
    let replay_journal =
        Arc::clone(replay_journals.get(&authenticated_key_id).ok_or_else(|| {
            EngineError::internal_invariant("active control key omitted its replay journal")
        })?);
    admit_durable_nonce(&replay_journal, session.last_nonce())?;
    Ok(ContinuousAuthenticatedConnection { connection, session, replay_journal, command })
}

fn start_stream_workers(
    connection: UnixFrameConnection,
    session: AuthenticatedControlSession,
    replay_journal: Arc<Mutex<DurableReplayJournal>>,
    request_id: RequestId,
) -> Result<StreamWorkers, EngineError> {
    let cancellation_connection = connection.try_clone()?;
    let writer_connection = connection.try_clone()?;
    let (outbound_sender, outbound_receiver) =
        std::sync::mpsc::sync_channel(MAX_PENDING_EVENTS_PER_CONNECTION);
    let writer_failed = Arc::new(AtomicBool::new(false));
    let writer_failed_worker = Arc::clone(&writer_failed);
    let writer = thread::Builder::new()
        .name("personalai-event-write".to_owned())
        .stack_size(OUTPUT_WRITER_STACK_BYTES)
        .spawn(move || {
            let result = write_queued_events(writer_connection, &outbound_receiver);
            if result.is_err() {
                writer_failed_worker.store(true, Ordering::Release);
            }
            result
        })
        .map_err(|_| EngineError::backend_failure("output writer thread creation failed"))?;
    let cancelled = Arc::new(AtomicBool::new(false));
    let watcher_done = Arc::new(AtomicBool::new(false));
    let watcher_cancelled = Arc::clone(&cancelled);
    let watcher_finished = Arc::clone(&watcher_done);
    let watcher = thread::Builder::new()
        .name("personalai-cancel-watch".to_owned())
        .stack_size(CANCELLATION_WATCHER_STACK_BYTES)
        .spawn(move || {
            watch_generation_cancellation(
                cancellation_connection,
                session,
                request_id,
                &watcher_cancelled,
                &watcher_finished,
                &replay_journal,
            );
        });
    let Ok(watcher) = watcher else {
        drop(outbound_sender);
        writer.join().map_err(|_| EngineError::internal_invariant("output writer panicked"))??;
        return Err(EngineError::backend_failure("cancellation watcher thread creation failed"));
    };
    Ok(StreamWorkers {
        control: connection,
        outbound_sender: Some(outbound_sender),
        writer_failed,
        cancelled,
        watcher_done,
        watcher,
        writer,
    })
}

fn stop_stream_workers(mut workers: StreamWorkers) {
    workers.watcher_done.store(true, Ordering::Release);
    let _shutdown_result = workers.control.shutdown_inbound();
    workers.outbound_sender = None;
    let _watcher_join = workers.watcher.join();
    let _writer_join = workers.writer.join();
}

fn enqueue_active_event(
    connection: &mut ContinuousActiveConnection,
    event: &GenerationEvent,
    response_signer: &EngineResponseSigner,
) -> Result<(), EngineError> {
    if let Some(error) = &connection.stream_error {
        return Err(error.clone());
    }
    if connection.workers.writer_failed.load(Ordering::Acquire) {
        return Err(EngineError::backend_failure("generation event writer failed"));
    }
    let ordinal = connection.next_event_ordinal;
    connection.next_event_ordinal = ordinal
        .checked_add(1)
        .ok_or_else(|| EngineError::resource_exhaustion("generation event ordinal exhausted"))?;
    let frame = generation_event_frame(event, connection.request_nonce, ordinal, response_signer)?;
    let sender =
        connection.workers.outbound_sender.as_ref().ok_or_else(|| {
            EngineError::internal_invariant("continuous output sender disappeared")
        })?;
    enqueue_outbound_frame(sender, frame)
}

fn bounded_vec<T>(capacity: usize, label: &'static str) -> Result<Vec<T>, EngineError> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(capacity)
        .map_err(|_| EngineError::resource_exhaustion(format!("{label} allocation failed")))?;
    Ok(values)
}

fn increment(value: &mut u64, label: &'static str) -> Result<(), EngineError> {
    *value = value
        .checked_add(1)
        .ok_or_else(|| EngineError::resource_exhaustion(format!("{label} overflow")))?;
    Ok(())
}
