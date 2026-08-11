//! Bounded, deterministic inference-request scheduler.
//!
//! This scheduler owns only inference queue and sequence lifecycle. It is not a
//! model router, task scheduler, agent runtime, policy engine, approval system,
//! or authorization authority. Queue admission is bounded by both count and
//! aggregate encoded request bytes.

use personalai_engine_types::{
    EngineError, GenerationRequest, MAX_CONCURRENT_SEQUENCES, RequestId, SequenceId,
};
use std::collections::VecDeque;

/// Hard maximum queued generation requests.
pub const MAX_QUEUED_REQUESTS: usize = 4_096;
/// Hard maximum aggregate encoded bytes retained in the request queue.
pub const MAX_QUEUED_REQUEST_BYTES: usize = 64 * 1_024 * 1_024;

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
    queued: VecDeque<GenerationRequest>,
    queued_bytes: usize,
    active: Vec<ActiveRequest>,
}

impl Scheduler {
    /// Creates an empty scheduler.
    pub const fn new(limits: SchedulerLimits) -> Self {
        Self { limits, queued: VecDeque::new(), queued_bytes: 0, active: Vec::new() }
    }

    /// Admits a unique validated request to the FIFO queue.
    pub fn submit(&mut self, request: GenerationRequest) -> Result<(), EngineError> {
        if self.queued.len() >= self.limits.maximum_queued() {
            return Err(EngineError::resource_exhaustion("generation request queue is full"));
        }
        let request_id = request.request_id();
        if self.active.iter().any(|active| active.request.request_id() == request_id)
            || self.queued.iter().any(|queued| queued.request_id() == request_id)
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
        self.queued.try_reserve(1).map_err(|_| {
            EngineError::resource_exhaustion("generation request queue allocation failed")
        })?;
        self.queued.push_back(request);
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
        let Some(front) = self.queued.front() else {
            return Ok(None);
        };
        let next_active_count = active_count
            .checked_add(1)
            .ok_or_else(|| EngineError::resource_exhaustion("active request count overflows"))?;
        if next_active_count > front.resource_budget().max_concurrent_sequences() {
            return Err(EngineError::resource_exhaustion(
                "request concurrency budget does not admit another active sequence",
            ));
        }
        let next_queued_bytes = self
            .queued_bytes
            .checked_sub(front.estimated_size_bytes())
            .ok_or_else(|| EngineError::internal_invariant("queued byte accounting underflows"))?;
        self.active
            .try_reserve(1)
            .map_err(|_| EngineError::resource_exhaustion("active request allocation failed"))?;
        let request = self
            .queued
            .pop_front()
            .ok_or_else(|| EngineError::internal_invariant("queued request disappeared"))?;
        self.queued_bytes = next_queued_bytes;
        self.active.push(ActiveRequest { request, sequence_id });
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
        if let Some(position) =
            self.queued.iter().position(|queued| queued.request_id() == *request_id)
        {
            let removed = self
                .queued
                .remove(position)
                .ok_or_else(|| EngineError::internal_invariant("queued request disappeared"))?;
            self.queued_bytes =
                self.queued_bytes.checked_sub(removed.estimated_size_bytes()).ok_or_else(|| {
                    EngineError::internal_invariant("queued byte accounting underflows")
                })?;
            return Ok(CancellationOutcome::Queued);
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
        self.queued.len()
    }

    /// Returns aggregate checked bytes retained by queued requests.
    pub const fn queued_bytes(&self) -> usize {
        self.queued_bytes
    }

    /// Returns the active request count.
    pub fn active_len(&self) -> usize {
        self.active.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use personalai_engine_types::{ContextLimit, ModelId, ResourceBudget};

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
