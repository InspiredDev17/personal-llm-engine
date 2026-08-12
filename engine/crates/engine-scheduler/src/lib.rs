//! Bounded, deterministic inference-request scheduler.
//!
//! This scheduler owns only inference queue and sequence lifecycle. It is not a
//! model router, task scheduler, agent runtime, policy engine, approval system,
//! or authorization authority. Queue admission is bounded by both count and
//! aggregate encoded request bytes.

use personalai_engine_types::{
    EngineError, GenerationRequest, MAX_CONCURRENT_SEQUENCES, ModelId, RequestId, SequenceId,
};
use std::collections::VecDeque;
use std::time::Instant;

/// Hard maximum queued generation requests.
pub const MAX_QUEUED_REQUESTS: usize = 4_096;
/// Hard maximum aggregate encoded bytes retained in the request queue.
pub const MAX_QUEUED_REQUEST_BYTES: usize = 64 * 1_024 * 1_024;
/// Maximum UTF-8 bytes in configuration-owned agent quota metadata.
pub const MAX_AGENT_ID_BYTES: usize = 64;

/// Closed scheduling priority classes.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum PriorityClass {
    /// Latency-sensitive interactive work.
    Interactive,
    /// Normal platform work.
    #[default]
    Standard,
    /// Background work that must still make bounded progress.
    Background,
}

impl PriorityClass {
    const fn index(self) -> usize {
        match self {
            Self::Interactive => 0,
            Self::Standard => 1,
            Self::Background => 2,
        }
    }
}

/// Validated non-authoritative agent identity used only for quota accounting.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct AgentId(String);

impl AgentId {
    /// Creates bounded ASCII quota metadata; this grants no authorization.
    pub fn new(value: impl Into<String>) -> Result<Self, EngineError> {
        let value = value.into();
        if value.is_empty()
            || value.len() > MAX_AGENT_ID_BYTES
            || !value
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.'))
        {
            return Err(EngineError::invalid_input("scheduler agent ID is outside safe bounds"));
        }
        Ok(Self(value))
    }

    /// Returns the bounded identifier.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// Per-agent queue and active ceilings supplied by trusted configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AgentQuota {
    maximum_queued: usize,
    maximum_active: u32,
}

impl AgentQuota {
    /// Creates a quota no larger than workspace hard limits.
    pub fn new(maximum_queued: usize, maximum_active: u32) -> Result<Self, EngineError> {
        if maximum_queued == 0 || maximum_queued > MAX_QUEUED_REQUESTS {
            return Err(EngineError::invalid_input("agent queue quota is outside hard bounds"));
        }
        if maximum_active == 0 || maximum_active > MAX_CONCURRENT_SEQUENCES {
            return Err(EngineError::invalid_input("agent active quota is outside hard bounds"));
        }
        Ok(Self { maximum_queued, maximum_active })
    }

    /// Returns the queued-request ceiling.
    pub const fn maximum_queued(self) -> usize {
        self.maximum_queued
    }

    /// Returns the active-request ceiling.
    pub const fn maximum_active(self) -> u32 {
        self.maximum_active
    }
}

/// Trusted scheduling metadata kept outside the generation request wire body.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SchedulingMetadata {
    priority: PriorityClass,
    agent_id: Option<AgentId>,
    agent_quota: Option<AgentQuota>,
    deadline: Option<Instant>,
    maximum_model_concurrency: u32,
}

impl SchedulingMetadata {
    /// Creates validated metadata. Agent ID and quota must be present together.
    pub fn new(
        priority: PriorityClass,
        agent_id: Option<AgentId>,
        agent_quota: Option<AgentQuota>,
        deadline: Option<Instant>,
        maximum_model_concurrency: u32,
    ) -> Result<Self, EngineError> {
        if agent_id.is_some() != agent_quota.is_some() {
            return Err(EngineError::invalid_input(
                "agent ID and quota must be configured together",
            ));
        }
        if maximum_model_concurrency == 0 || maximum_model_concurrency > MAX_CONCURRENT_SEQUENCES {
            return Err(EngineError::invalid_input(
                "per-model concurrency limit is outside hard bounds",
            ));
        }
        Ok(Self { priority, agent_id, agent_quota, deadline, maximum_model_concurrency })
    }

    /// Returns conservative defaults used by the stable generation adapter.
    pub const fn standard() -> Self {
        Self {
            priority: PriorityClass::Standard,
            agent_id: None,
            agent_quota: None,
            deadline: None,
            maximum_model_concurrency: MAX_CONCURRENT_SEQUENCES,
        }
    }

    /// Returns the priority class.
    pub const fn priority(&self) -> PriorityClass {
        self.priority
    }

    /// Returns quota metadata, when configured.
    pub fn agent_id(&self) -> Option<&AgentId> {
        self.agent_id.as_ref()
    }

    /// Returns the absolute monotonic deadline, when configured.
    pub const fn deadline(&self) -> Option<Instant> {
        self.deadline
    }
}

/// Validated scheduler capacity limits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SchedulerLimits {
    maximum_queued: usize,
    maximum_active: u32,
    maximum_queued_bytes: usize,
}

impl SchedulerLimits {
    /// Creates bounded scheduler limits.
    pub fn new(
        maximum_queued: usize,
        maximum_active: u32,
        maximum_queued_bytes: usize,
    ) -> Result<Self, EngineError> {
        if maximum_queued == 0 || maximum_queued > MAX_QUEUED_REQUESTS {
            return Err(EngineError::invalid_input("scheduler queue limit is outside hard bounds"));
        }
        if maximum_active == 0 || maximum_active > MAX_CONCURRENT_SEQUENCES {
            return Err(EngineError::invalid_input(
                "scheduler active limit is outside hard bounds",
            ));
        }
        if maximum_queued_bytes == 0 || maximum_queued_bytes > MAX_QUEUED_REQUEST_BYTES {
            return Err(EngineError::invalid_input(
                "scheduler queued-byte limit is outside hard bounds",
            ));
        }
        Ok(Self { maximum_queued, maximum_active, maximum_queued_bytes })
    }

    /// Returns the maximum queued requests.
    pub const fn maximum_queued(self) -> usize {
        self.maximum_queued
    }

    /// Returns the maximum active sequences.
    pub const fn maximum_active(self) -> u32 {
        self.maximum_active
    }

    /// Returns the maximum aggregate queued request bytes.
    pub const fn maximum_queued_bytes(self) -> usize {
        self.maximum_queued_bytes
    }
}

/// One active request bound to an engine sequence ID.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ActiveRequest {
    request: GenerationRequest,
    sequence_id: SequenceId,
    metadata: SchedulingMetadata,
}

impl ActiveRequest {
    /// Returns the validated generation request.
    pub const fn request(&self) -> &GenerationRequest {
        &self.request
    }

    /// Returns the assigned engine sequence ID.
    pub const fn sequence_id(&self) -> SequenceId {
        self.sequence_id
    }

    /// Returns trusted scheduling metadata associated with this request.
    pub const fn metadata(&self) -> &SchedulingMetadata {
        &self.metadata
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct QueuedRequest {
    request: GenerationRequest,
    metadata: SchedulingMetadata,
}

/// Result of a request cancellation attempt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CancellationOutcome {
    /// A queued request was removed before activation.
    Queued,
    /// An active request was removed and must stop backend work.
    Active,
    /// No queued or active request matched the identifier.
    NotFound,
}

/// Deterministic FIFO scheduler with hard count and aggregate-byte bounds.
#[derive(Debug)]
pub struct Scheduler {
    limits: SchedulerLimits,
    queued: [VecDeque<QueuedRequest>; 3],
    queued_bytes: usize,
    active: Vec<ActiveRequest>,
    dispatch_slot: usize,
    rejected_deadlines: u64,
}

const DISPATCH_ORDER: [PriorityClass; 7] = [
    PriorityClass::Interactive,
    PriorityClass::Standard,
    PriorityClass::Interactive,
    PriorityClass::Background,
    PriorityClass::Interactive,
    PriorityClass::Standard,
    PriorityClass::Interactive,
];

impl Scheduler {
    /// Creates an empty scheduler.
    pub const fn new(limits: SchedulerLimits) -> Self {
        Self {
            limits,
            queued: [VecDeque::new(), VecDeque::new(), VecDeque::new()],
            queued_bytes: 0,
            active: Vec::new(),
            dispatch_slot: 0,
            rejected_deadlines: 0,
        }
    }

    /// Returns the configured hard scheduler limits.
    pub const fn limits(&self) -> SchedulerLimits {
        self.limits
    }

    /// Admits a unique validated request to the FIFO queue.
    pub fn submit(&mut self, request: GenerationRequest) -> Result<(), EngineError> {
        self.submit_with_metadata(request, SchedulingMetadata::standard())
    }

    /// Admits a request with trusted priority, quota, deadline, and model limits.
    pub fn submit_with_metadata(
        &mut self,
        request: GenerationRequest,
        metadata: SchedulingMetadata,
    ) -> Result<(), EngineError> {
        if self.queued_len() >= self.limits.maximum_queued() {
            return Err(EngineError::resource_exhaustion("generation request queue is full"));
        }
        if metadata.deadline().is_some_and(|deadline| deadline <= Instant::now()) {
            return Err(EngineError::cancellation(Some(request.request_id())));
        }
        let request_id = request.request_id();
        if self.active.iter().any(|active| active.request.request_id() == request_id)
            || self
                .queued
                .iter()
                .flat_map(VecDeque::iter)
                .any(|queued| queued.request.request_id() == request_id)
        {
            return Err(EngineError::protocol_violation(
                "generation request identifier is already scheduled",
            ));
        }
        let next_bytes = self
            .queued_bytes
            .checked_add(request.estimated_size_bytes())
            .ok_or_else(|| EngineError::resource_exhaustion("queued request bytes overflow"))?;
        if next_bytes > self.limits.maximum_queued_bytes() {
            return Err(EngineError::resource_exhaustion(
                "aggregate generation request queue bytes are exhausted",
            ));
        }
        if let (Some(agent_id), Some(quota)) = (&metadata.agent_id, metadata.agent_quota) {
            let queued_for_agent = self
                .queued
                .iter()
                .flat_map(VecDeque::iter)
                .filter(|queued| queued.metadata.agent_id.as_ref() == Some(agent_id))
                .count();
            if queued_for_agent >= quota.maximum_queued() {
                return Err(EngineError::resource_exhaustion("agent queue quota is exhausted"));
            }
        }
        let queue = &mut self.queued[metadata.priority().index()];
        queue.try_reserve(1).map_err(|_| {
            EngineError::resource_exhaustion("generation request queue allocation failed")
        })?;
        queue.push_back(QueuedRequest { request, metadata });
        self.queued_bytes = next_bytes;
        Ok(())
    }

    /// Activates the next queued request under an externally assigned sequence.
    ///
    /// Sequence IDs are supplied by trusted engine orchestration, never by the
    /// model package. `Ok(None)` means the queue is empty.
    pub fn start_next(
        &mut self,
        sequence_id: SequenceId,
    ) -> Result<Option<&ActiveRequest>, EngineError> {
        self.expire_deadlines(Instant::now())?;
        let active_count = u32::try_from(self.active.len())
            .map_err(|_| EngineError::internal_invariant("active scheduler count exceeds u32"))?;
        if active_count >= self.limits.maximum_active() {
            return Err(EngineError::resource_exhaustion(
                "active generation sequence limit is reached",
            ));
        }
        if self.active.iter().any(|active| active.sequence_id == sequence_id) {
            return Err(EngineError::protocol_violation(
                "generation sequence identifier is already active",
            ));
        }
        let Some((queue_index, position)) = self.next_admissible_position(active_count)? else {
            return Ok(None);
        };
        let front = self.queued[queue_index]
            .get(position)
            .ok_or_else(|| EngineError::internal_invariant("selected request disappeared"))?;
        let next_queued_bytes = self
            .queued_bytes
            .checked_sub(front.request.estimated_size_bytes())
            .ok_or_else(|| EngineError::internal_invariant("queued byte accounting underflows"))?;
        self.active
            .try_reserve(1)
            .map_err(|_| EngineError::resource_exhaustion("active request allocation failed"))?;
        let queued = self.queued[queue_index]
            .remove(position)
            .ok_or_else(|| EngineError::internal_invariant("queued request disappeared"))?;
        self.queued_bytes = next_queued_bytes;
        self.active.push(ActiveRequest {
            request: queued.request,
            sequence_id,
            metadata: queued.metadata,
        });
        Ok(self.active.last())
    }

    /// Completes and removes one active request.
    pub fn finish(&mut self, request_id: &RequestId) -> Result<ActiveRequest, EngineError> {
        let position = self
            .active
            .iter()
            .position(|active| active.request.request_id() == *request_id)
            .ok_or_else(|| EngineError::protocol_violation("generation request is not active"))?;
        Ok(self.active.remove(position))
    }

    /// Cancels a queued or active request without treating a miss as success.
    pub fn cancel(&mut self, request_id: &RequestId) -> Result<CancellationOutcome, EngineError> {
        for queue in &mut self.queued {
            if let Some(position) =
                queue.iter().position(|queued| queued.request.request_id() == *request_id)
            {
                let removed = queue
                    .remove(position)
                    .ok_or_else(|| EngineError::internal_invariant("queued request disappeared"))?;
                self.queued_bytes = self
                    .queued_bytes
                    .checked_sub(removed.request.estimated_size_bytes())
                    .ok_or_else(|| {
                        EngineError::internal_invariant("queued byte accounting underflows")
                    })?;
                return Ok(CancellationOutcome::Queued);
            }
        }
        if let Some(position) =
            self.active.iter().position(|active| active.request.request_id() == *request_id)
        {
            let _removed = self.active.remove(position);
            return Ok(CancellationOutcome::Active);
        }
        Ok(CancellationOutcome::NotFound)
    }

    /// Returns the queued request count.
    pub fn queued_len(&self) -> usize {
        self.queued.iter().map(VecDeque::len).sum()
    }

    /// Returns aggregate checked bytes retained by queued requests.
    pub const fn queued_bytes(&self) -> usize {
        self.queued_bytes
    }

    /// Returns the active request count.
    pub fn active_len(&self) -> usize {
        self.active.len()
    }

    /// Returns the number of requests rejected after their queue deadline elapsed.
    pub const fn rejected_deadlines(&self) -> u64 {
        self.rejected_deadlines
    }

    /// Returns whether a model currently owns active sequence state.
    pub fn has_active_model(&self, model_id: &ModelId) -> bool {
        self.active.iter().any(|active| active.request.model_id() == model_id)
    }

    /// Returns whether a model currently owns queued request state.
    pub fn has_queued_model(&self, model_id: &ModelId) -> bool {
        self.queued
            .iter()
            .flat_map(VecDeque::iter)
            .any(|queued| queued.request.model_id() == model_id)
    }

    fn expire_deadlines(&mut self, now: Instant) -> Result<(), EngineError> {
        for queue in &mut self.queued {
            let mut retained = VecDeque::new();
            retained.try_reserve(queue.len()).map_err(|_| {
                EngineError::resource_exhaustion("deadline queue allocation failed")
            })?;
            while let Some(queued) = queue.pop_front() {
                if queued.metadata.deadline.is_some_and(|deadline| deadline <= now) {
                    self.queued_bytes = self
                        .queued_bytes
                        .checked_sub(queued.request.estimated_size_bytes())
                        .ok_or_else(|| {
                            EngineError::internal_invariant("queued byte accounting underflows")
                        })?;
                    self.rejected_deadlines =
                        self.rejected_deadlines.checked_add(1).ok_or_else(|| {
                            EngineError::resource_exhaustion("deadline rejection counter exhausted")
                        })?;
                } else {
                    retained.push_back(queued);
                }
            }
            *queue = retained;
        }
        Ok(())
    }

    fn next_admissible_position(
        &mut self,
        active_count: u32,
    ) -> Result<Option<(usize, usize)>, EngineError> {
        for offset in 0..DISPATCH_ORDER.len() {
            let slot = (self.dispatch_slot + offset) % DISPATCH_ORDER.len();
            let queue_index = DISPATCH_ORDER[slot].index();
            for (position, queued) in self.queued[queue_index].iter().enumerate() {
                if self.request_is_admissible(queued, active_count)? {
                    self.dispatch_slot = (slot + 1) % DISPATCH_ORDER.len();
                    return Ok(Some((queue_index, position)));
                }
            }
        }
        Ok(None)
    }

    fn request_is_admissible(
        &self,
        queued: &QueuedRequest,
        active_count: u32,
    ) -> Result<bool, EngineError> {
        let next_active = active_count
            .checked_add(1)
            .ok_or_else(|| EngineError::resource_exhaustion("active request count overflows"))?;
        if next_active > queued.request.resource_budget().max_concurrent_sequences() {
            return Ok(false);
        }
        let model_active = self
            .active
            .iter()
            .filter(|active| active.request.model_id() == queued.request.model_id())
            .count();
        if u32::try_from(model_active)
            .map_err(|_| EngineError::internal_invariant("per-model active count exceeds u32"))?
            >= queued.metadata.maximum_model_concurrency
        {
            return Ok(false);
        }
        if let (Some(agent_id), Some(quota)) =
            (&queued.metadata.agent_id, queued.metadata.agent_quota)
        {
            let agent_active = self
                .active
                .iter()
                .filter(|active| active.metadata.agent_id.as_ref() == Some(agent_id))
                .count();
            if u32::try_from(agent_active).map_err(|_| {
                EngineError::internal_invariant("per-agent active count exceeds u32")
            })? >= quota.maximum_active()
            {
                return Ok(false);
            }
        }
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use personalai_engine_types::{ContextLimit, ModelId, ResourceBudget};
    use std::time::Duration;

    #[test]
    fn schedules_fifo_and_tracks_active_bounds() {
        let limits = SchedulerLimits::new(2, 1, 1 << 20).expect("limits must validate");
        let mut scheduler = Scheduler::new(limits);
        let first = request(1);
        let second = request(2);
        scheduler.submit(first).expect("first request must queue");
        scheduler.submit(second).expect("second request must queue");
        assert!(scheduler.queued_bytes() > 0);
        let active = scheduler
            .start_next(SequenceId::new(1).expect("sequence must validate"))
            .expect("request must activate")
            .expect("queue must not be empty");
        assert_eq!(active.request().request_id().get(), 1);
        assert!(scheduler.start_next(SequenceId::new(2).expect("sequence must validate")).is_err());
    }

    #[test]
    fn duplicate_count_and_aggregate_bytes_fail_closed() {
        let first = request(1);
        let exact_bytes = first.estimated_size_bytes();
        let limits = SchedulerLimits::new(2, 1, exact_bytes).expect("limits must validate");
        let mut scheduler = Scheduler::new(limits);
        scheduler.submit(first).expect("request must queue");
        assert!(scheduler.submit(request(1)).is_err());
        assert!(scheduler.submit(request(2)).is_err());
        assert_eq!(scheduler.queued_bytes(), exact_bytes);
    }

    #[test]
    fn cancellation_distinguishes_queue_active_and_missing() {
        let limits = SchedulerLimits::new(2, 1, 1 << 20).expect("limits must validate");
        let mut scheduler = Scheduler::new(limits);
        let queued_id = RequestId::new(1).expect("id must validate");
        scheduler.submit(request(1)).expect("request must queue");
        assert_eq!(
            scheduler.cancel(&queued_id).expect("cancellation must be accounted"),
            CancellationOutcome::Queued
        );
        assert_eq!(scheduler.queued_bytes(), 0);
        scheduler.submit(request(2)).expect("request must queue");
        scheduler
            .start_next(SequenceId::new(1).expect("sequence must validate"))
            .expect("request must activate");
        let active_id = RequestId::new(2).expect("id must validate");
        assert_eq!(
            scheduler.cancel(&active_id).expect("cancellation must be accounted"),
            CancellationOutcome::Active
        );
        assert_eq!(
            scheduler.cancel(&active_id).expect("cancellation must be accounted"),
            CancellationOutcome::NotFound
        );
    }

    #[test]
    fn weighted_priority_order_is_deterministic_and_nonstarving() {
        let limits = SchedulerLimits::new(8, 1, 1 << 20).expect("limits");
        let mut scheduler = Scheduler::new(limits);
        scheduler
            .submit_with_metadata(request(1), metadata(PriorityClass::Background, 1))
            .expect("background");
        scheduler
            .submit_with_metadata(request(2), metadata(PriorityClass::Interactive, 1))
            .expect("interactive");
        scheduler
            .submit_with_metadata(request(3), metadata(PriorityClass::Standard, 1))
            .expect("standard");
        let mut observed = Vec::new();
        for sequence in 1..=3 {
            let id = scheduler
                .start_next(SequenceId::new(sequence).expect("sequence"))
                .expect("start")
                .expect("request")
                .request()
                .request_id();
            observed.push(id.get());
            scheduler.finish(&id).expect("finish");
        }
        assert_eq!(observed, vec![2, 3, 1]);
    }

    #[test]
    fn quotas_model_limits_and_deadlines_fail_closed() {
        let limits = SchedulerLimits::new(8, 4, 1 << 20).expect("limits");
        let mut scheduler = Scheduler::new(limits);
        let agent = AgentId::new("creator-agent").expect("agent");
        let quota = AgentQuota::new(1, 1).expect("quota");
        let first_metadata = SchedulingMetadata::new(
            PriorityClass::Interactive,
            Some(agent.clone()),
            Some(quota),
            None,
            1,
        )
        .expect("metadata");
        scheduler.submit_with_metadata(request(1), first_metadata.clone()).expect("first");
        assert!(scheduler.submit_with_metadata(request(2), first_metadata).is_err());
        let first_id = scheduler
            .start_next(SequenceId::new(1).expect("sequence"))
            .expect("start")
            .expect("request")
            .request()
            .request_id();
        scheduler
            .submit_with_metadata(
                request(2),
                SchedulingMetadata::new(PriorityClass::Standard, Some(agent), Some(quota), None, 1)
                    .expect("metadata"),
            )
            .expect("queue after activation");
        assert!(
            scheduler
                .start_next(SequenceId::new(2).expect("sequence"))
                .expect("blocked model/agent")
                .is_none()
        );
        scheduler.finish(&first_id).expect("finish");

        let expired = SchedulingMetadata::new(
            PriorityClass::Standard,
            None,
            None,
            Some(
                Instant::now()
                    .checked_sub(Duration::from_millis(1))
                    .expect("one millisecond before now is representable"),
            ),
            1,
        )
        .expect("expired metadata");
        assert!(scheduler.submit_with_metadata(request(3), expired).is_err());
    }

    fn metadata(priority: PriorityClass, maximum_model_concurrency: u32) -> SchedulingMetadata {
        SchedulingMetadata::new(priority, None, None, None, maximum_model_concurrency)
            .expect("metadata")
    }

    fn request(id: u128) -> GenerationRequest {
        GenerationRequest::new(
            RequestId::new(id).expect("request id must validate"),
            ModelId::new("model-a").expect("model id must validate"),
            "bounded input",
            16,
            ContextLimit::new(128).expect("context must validate"),
            ResourceBudget::new(1 << 20, 1 << 19, 1).expect("budget must validate"),
        )
        .expect("request must validate")
    }
}
