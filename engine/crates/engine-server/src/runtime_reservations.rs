//! Bounded inference-runtime-v3 reservation lifecycle.
//!
//! This module is the engine-owned semantic state machine for the canonical
//! Sovereign AI Contracts inference runtime v3 contract. It is deliberately
//! transport-neutral and contains no cognitive role, qualification decision,
//! cloud policy, tool authority, or user policy. A separately authenticated
//! protocol adapter must validate and map canonical documents before calling it.

use personalai_engine_types::{BackendId, EngineError, ModelId, ModelPackageHash, RequestId};
use std::collections::BTreeMap;
use std::sync::Mutex;

/// Exact Sovereign AI Contracts commit defining inference runtime v3.
pub const INFERENCE_RUNTIME_V3_CONTRACT_COMMIT: &str = "77f987c3c1c61153fc17698336d25b95edaf2304";
/// Exact canonical contract version implemented by this lifecycle.
pub const INFERENCE_RUNTIME_V3_CONTRACT_VERSION: &str = "3.0.0";
/// Maximum active and historical reservation records retained in memory.
pub const MAX_RUNTIME_RESERVATION_RECORDS: usize = 16_384;
const MAX_AUTHORITY_ID_BYTES: usize = 128;
const MAX_RUNTIME_TOKENS: u64 = 10_000_000;
const MAX_RUNTIME_MEMORY_BYTES: u64 = 1 << 50;
const MAX_RUNTIME_MILLIS: u64 = 86_400_000;

macro_rules! authority_id {
    ($(#[$meta:meta])* $name:ident, $field:literal, $prefix:literal) => {
        $(#[$meta])*
        #[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
        pub struct $name(String);

        impl $name {
            /// Constructs one path-free authority identifier.
            pub fn new(value: impl Into<String>) -> Result<Self, EngineError> {
                let value = value.into();
                validate_authority_id($field, &value, $prefix)?;
                Ok(Self(value))
            }

            /// Returns the exact identifier text.
            pub fn as_str(&self) -> &str {
                &self.0
            }
        }
    };
}

authority_id!(
    /// Control-plane logical model identifier. This is not the engine registry key.
    LogicalModelId,
    "model_id",
    "mdl_"
);
authority_id!(
    /// Model Foundry artifact identity bound by the control plane.
    ArtifactAuthorityId,
    "artifact_id",
    "art_"
);
authority_id!(
    /// Control-plane task identity carried only for correlation and isolation.
    TaskAuthorityId,
    "task_id",
    "tsk_"
);
authority_id!(
    /// One non-transferable runtime reservation identity.
    ReservationId,
    "reservation_id",
    "rsv_"
);

/// Path-free configured runtime instance or version identifier.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RuntimeInstanceId(String);

impl RuntimeInstanceId {
    /// Constructs a path-free runtime identifier using the v3 grammar.
    pub fn new(value: impl Into<String>) -> Result<Self, EngineError> {
        let value = value.into();
        validate_runtime_id("runtime_instance_id", &value)?;
        Ok(Self(value))
    }

    /// Returns exact identifier text.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// Exact runtime target that admission and execution may not substitute.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RuntimeIdentityBinding {
    logical_model_id: LogicalModelId,
    provider_model_id: ModelId,
    artifact_id: ArtifactAuthorityId,
    package_hash: ModelPackageHash,
    backend_id: BackendId,
    backend_version: RuntimeInstanceId,
    backend_instance_id: RuntimeInstanceId,
    engine_instance_id: RuntimeInstanceId,
}

impl RuntimeIdentityBinding {
    /// Constructs one exact path-free execution target.
    #[allow(clippy::too_many_arguments)]
    pub const fn new(
        logical_model_id: LogicalModelId,
        provider_model_id: ModelId,
        artifact_id: ArtifactAuthorityId,
        package_hash: ModelPackageHash,
        backend_id: BackendId,
        backend_version: RuntimeInstanceId,
        backend_instance_id: RuntimeInstanceId,
        engine_instance_id: RuntimeInstanceId,
    ) -> Self {
        Self {
            logical_model_id,
            provider_model_id,
            artifact_id,
            package_hash,
            backend_id,
            backend_version,
            backend_instance_id,
            engine_instance_id,
        }
    }

    /// Returns the logical control-plane model identity.
    pub const fn logical_model_id(&self) -> &LogicalModelId {
        &self.logical_model_id
    }

    /// Returns the engine registry model identity.
    pub const fn provider_model_id(&self) -> &ModelId {
        &self.provider_model_id
    }

    /// Returns the Model Foundry artifact identity.
    pub const fn artifact_id(&self) -> &ArtifactAuthorityId {
        &self.artifact_id
    }

    /// Returns the exact verified model-package digest.
    pub const fn package_hash(&self) -> &ModelPackageHash {
        &self.package_hash
    }

    /// Returns the configured backend identity.
    pub const fn backend_id(&self) -> &BackendId {
        &self.backend_id
    }

    /// Returns the configured backend version.
    pub const fn backend_version(&self) -> &RuntimeInstanceId {
        &self.backend_version
    }

    /// Returns the configured backend process/device instance.
    pub const fn backend_instance_id(&self) -> &RuntimeInstanceId {
        &self.backend_instance_id
    }

    /// Returns the configured engine instance.
    pub const fn engine_instance_id(&self) -> &RuntimeInstanceId {
        &self.engine_instance_id
    }

    /// Revalidates every engine-owned identity immediately before reservation
    /// or execution. The logical model and artifact IDs remain control-plane
    /// correlations; the engine independently proves the registry model,
    /// package digest, backend, and configured instances.
    pub fn revalidate_engine_owned(
        &self,
        provider_model_id: &ModelId,
        package_hash: &ModelPackageHash,
        backend_id: &BackendId,
        backend_version: &RuntimeInstanceId,
        backend_instance_id: &RuntimeInstanceId,
        engine_instance_id: &RuntimeInstanceId,
    ) -> Result<(), EngineError> {
        if self.provider_model_id != *provider_model_id
            || self.package_hash != *package_hash
            || self.backend_id != *backend_id
            || self.backend_version != *backend_version
            || self.backend_instance_id != *backend_instance_id
            || self.engine_instance_id != *engine_instance_id
        {
            return Err(EngineError::authentication_failure(
                "runtime identity differs from current engine-owned state",
            ));
        }
        Ok(())
    }
}

/// Provider-neutral request ceilings admitted by the engine.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RuntimeResourceCeiling {
    maximum_input_tokens: u64,
    maximum_output_tokens: u64,
    maximum_inference_memory_bytes: u64,
    maximum_execution_millis: u64,
}

impl RuntimeResourceCeiling {
    /// Constructs bounded nonzero resource ceilings.
    pub fn new(
        maximum_input_tokens: u64,
        maximum_output_tokens: u64,
        maximum_inference_memory_bytes: u64,
        maximum_execution_millis: u64,
    ) -> Result<Self, EngineError> {
        if !(1..=MAX_RUNTIME_TOKENS).contains(&maximum_input_tokens)
            || !(1..=MAX_RUNTIME_TOKENS).contains(&maximum_output_tokens)
            || !(1..=MAX_RUNTIME_MEMORY_BYTES).contains(&maximum_inference_memory_bytes)
            || !(1..=MAX_RUNTIME_MILLIS).contains(&maximum_execution_millis)
        {
            return Err(EngineError::invalid_input(
                "runtime resource ceiling contains zero or exceeds the v3 contract bound",
            ));
        }
        Ok(Self {
            maximum_input_tokens,
            maximum_output_tokens,
            maximum_inference_memory_bytes,
            maximum_execution_millis,
        })
    }

    /// Returns maximum input tokens.
    pub const fn maximum_input_tokens(self) -> u64 {
        self.maximum_input_tokens
    }

    /// Returns maximum output tokens.
    pub const fn maximum_output_tokens(self) -> u64 {
        self.maximum_output_tokens
    }

    /// Returns maximum inference memory bytes.
    pub const fn maximum_inference_memory_bytes(self) -> u64 {
        self.maximum_inference_memory_bytes
    }

    /// Returns maximum execution wall time in milliseconds.
    pub const fn maximum_execution_millis(self) -> u64 {
        self.maximum_execution_millis
    }
}

/// Fully bound reservation request after canonical contract validation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RuntimeReservationRequest {
    reservation_id: ReservationId,
    request_id: RequestId,
    task_id: TaskAuthorityId,
    identity: RuntimeIdentityBinding,
    budget: RuntimeResourceCeiling,
    idempotency_key: [u8; 32],
    expires_unix_millis: u64,
}

impl RuntimeReservationRequest {
    /// Constructs a reservation request. Absolute freshness is checked by the manager.
    pub fn new(
        reservation_id: ReservationId,
        request_id: RequestId,
        task_id: TaskAuthorityId,
        identity: RuntimeIdentityBinding,
        budget: RuntimeResourceCeiling,
        idempotency_key: [u8; 32],
        expires_unix_millis: u64,
    ) -> Result<Self, EngineError> {
        if idempotency_key == [0_u8; 32] {
            return Err(EngineError::invalid_field("idempotency_key", "must not be all zero"));
        }
        if expires_unix_millis == 0 {
            return Err(EngineError::invalid_field("expires_at", "must not be zero"));
        }
        Ok(Self {
            reservation_id,
            request_id,
            task_id,
            identity,
            budget,
            idempotency_key,
            expires_unix_millis,
        })
    }

    /// Returns the reservation identity.
    pub const fn reservation_id(&self) -> &ReservationId {
        &self.reservation_id
    }

    /// Returns the request correlation identity.
    pub const fn request_id(&self) -> RequestId {
        self.request_id
    }

    /// Returns the task authority identity.
    pub const fn task_id(&self) -> &TaskAuthorityId {
        &self.task_id
    }

    /// Returns the exact runtime target.
    pub const fn identity(&self) -> &RuntimeIdentityBinding {
        &self.identity
    }

    /// Returns requested provider-neutral ceilings.
    pub const fn budget(&self) -> RuntimeResourceCeiling {
        self.budget
    }

    /// Returns reservation expiry as Unix milliseconds.
    pub const fn expires_unix_millis(&self) -> u64 {
        self.expires_unix_millis
    }
}

/// Stable typed resource-admission denial.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReservationDenial {
    /// Manager record capacity is exhausted.
    RecordCapacity,
    /// Concurrent accepted reservation ceiling is exhausted.
    ConcurrentCapacity,
    /// Inference-memory reservation capacity is exhausted.
    MemoryCapacity,
    /// Request was already expired when evaluated.
    Expired,
}

/// Exact deterministic decision for one reservation request.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReservationDecision {
    /// Reservation was accepted with ceilings no broader than requested.
    Accepted(RuntimeResourceCeiling),
    /// Reservation was denied without consuming authority or resources.
    Denied(ReservationDenial),
}

/// Stored reservation lifecycle state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReservationStatus {
    /// Capacity is held but execution has not consumed it.
    Reserved,
    /// Exactly one execution has consumed the reservation.
    Consumed,
    /// Exact cancellation released capacity.
    Cancelled,
    /// Exact release returned capacity.
    Released,
    /// Expiry returned capacity.
    Expired,
    /// Reservation request was denied and never held capacity.
    Denied,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct ReservationRecord {
    request: RuntimeReservationRequest,
    decision: ReservationDecision,
    status: ReservationStatus,
}

#[derive(Debug)]
struct ReservationState {
    records: BTreeMap<ReservationId, ReservationRecord>,
    idempotency: BTreeMap<[u8; 32], ReservationId>,
    active: usize,
    reserved_memory_bytes: u64,
}

/// Bounded concurrent reservation statistics.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ReservationStatistics {
    /// Retained active and historical records.
    pub records: usize,
    /// Reservations currently holding capacity.
    pub active: usize,
    /// Inference-memory bytes currently held.
    pub reserved_memory_bytes: u64,
}

/// Thread-safe bounded reservation manager for one engine instance.
#[derive(Debug)]
pub struct RuntimeReservationManager {
    maximum_active: usize,
    maximum_memory_bytes: u64,
    state: Mutex<ReservationState>,
}

impl RuntimeReservationManager {
    /// Constructs an empty manager with explicit engine-wide limits.
    pub fn new(maximum_active: usize, maximum_memory_bytes: u64) -> Result<Self, EngineError> {
        if maximum_active == 0 || maximum_active > MAX_RUNTIME_RESERVATION_RECORDS {
            return Err(EngineError::invalid_input(
                "maximum active runtime reservations is outside hard bounds",
            ));
        }
        if !(1..=MAX_RUNTIME_MEMORY_BYTES).contains(&maximum_memory_bytes) {
            return Err(EngineError::invalid_input(
                "maximum runtime reservation memory is outside hard bounds",
            ));
        }
        Ok(Self {
            maximum_active,
            maximum_memory_bytes,
            state: Mutex::new(ReservationState {
                records: BTreeMap::new(),
                idempotency: BTreeMap::new(),
                active: 0,
                reserved_memory_bytes: 0,
            }),
        })
    }

    /// Evaluates and records one request. Exact idempotent replays return the
    /// original decision; altered replays fail closed.
    pub fn reserve(
        &self,
        request: RuntimeReservationRequest,
        now_unix_millis: u64,
    ) -> Result<ReservationDecision, EngineError> {
        let mut state = self.lock()?;
        expire_locked(&mut state, now_unix_millis)?;
        if let Some(existing_id) = state.idempotency.get(&request.idempotency_key) {
            let existing = state.records.get(existing_id).ok_or_else(|| {
                EngineError::internal_invariant("reservation idempotency index is inconsistent")
            })?;
            if existing.request == request {
                return Ok(existing.decision);
            }
            return Err(EngineError::authentication_failure(
                "runtime reservation idempotency key was reused with different authority",
            ));
        }
        if state.records.contains_key(&request.reservation_id) {
            return Err(EngineError::authentication_failure(
                "runtime reservation identifier was reused",
            ));
        }

        let decision = if request.expires_unix_millis <= now_unix_millis {
            ReservationDecision::Denied(ReservationDenial::Expired)
        } else if state.records.len() >= MAX_RUNTIME_RESERVATION_RECORDS {
            return Ok(ReservationDecision::Denied(ReservationDenial::RecordCapacity));
        } else if state.active >= self.maximum_active {
            ReservationDecision::Denied(ReservationDenial::ConcurrentCapacity)
        } else {
            let requested_memory = request.budget.maximum_inference_memory_bytes;
            let next_memory = state
                .reserved_memory_bytes
                .checked_add(requested_memory)
                .ok_or_else(|| EngineError::resource_exhaustion("reservation memory overflows"))?;
            if next_memory > self.maximum_memory_bytes {
                ReservationDecision::Denied(ReservationDenial::MemoryCapacity)
            } else {
                state.reserved_memory_bytes = next_memory;
                state.active = state.active.checked_add(1).ok_or_else(|| {
                    EngineError::resource_exhaustion("active reservation count overflows")
                })?;
                ReservationDecision::Accepted(request.budget)
            }
        };
        let status = if matches!(decision, ReservationDecision::Accepted(_)) {
            ReservationStatus::Reserved
        } else {
            ReservationStatus::Denied
        };
        state.idempotency.insert(request.idempotency_key, request.reservation_id.clone());
        state.records.insert(
            request.reservation_id.clone(),
            ReservationRecord { request, decision, status },
        );
        Ok(decision)
    }

    /// Atomically consumes one exact accepted reservation at most once.
    pub fn consume(
        &self,
        reservation_id: &ReservationId,
        request_id: RequestId,
        task_id: &TaskAuthorityId,
        identity: &RuntimeIdentityBinding,
        requested_output_tokens: u64,
        now_unix_millis: u64,
    ) -> Result<(), EngineError> {
        let mut state = self.lock()?;
        expire_locked(&mut state, now_unix_millis)?;
        let record = state
            .records
            .get_mut(reservation_id)
            .ok_or_else(|| EngineError::invalid_input("runtime reservation does not exist"))?;
        validate_record_authority(record, request_id, task_id, identity)?;
        if record.status != ReservationStatus::Reserved {
            return Err(EngineError::authentication_failure(
                "runtime reservation is not available for first consumption",
            ));
        }
        let ReservationDecision::Accepted(granted) = record.decision else {
            return Err(EngineError::authentication_failure(
                "denied runtime reservation cannot be consumed",
            ));
        };
        if requested_output_tokens == 0 || requested_output_tokens > granted.maximum_output_tokens {
            return Err(EngineError::invalid_field(
                "max_output_tokens",
                "exceeds the accepted runtime reservation",
            ));
        }
        record.status = ReservationStatus::Consumed;
        Ok(())
    }

    /// Cancels an exact reserved or consumed request and releases capacity.
    /// Repeating the exact cancellation is idempotent and returns `false`.
    pub fn cancel(
        &self,
        reservation_id: &ReservationId,
        request_id: RequestId,
        task_id: &TaskAuthorityId,
        identity: &RuntimeIdentityBinding,
        now_unix_millis: u64,
    ) -> Result<bool, EngineError> {
        self.transition_terminal(
            reservation_id,
            request_id,
            task_id,
            identity,
            now_unix_millis,
            ReservationStatus::Cancelled,
        )
    }

    /// Releases an exact reserved or consumed request. Repeating the exact
    /// release is idempotent and returns `false`.
    pub fn release(
        &self,
        reservation_id: &ReservationId,
        request_id: RequestId,
        task_id: &TaskAuthorityId,
        identity: &RuntimeIdentityBinding,
        now_unix_millis: u64,
    ) -> Result<bool, EngineError> {
        self.transition_terminal(
            reservation_id,
            request_id,
            task_id,
            identity,
            now_unix_millis,
            ReservationStatus::Released,
        )
    }

    /// Expires stale reservations and returns the number whose state changed.
    pub fn expire(&self, now_unix_millis: u64) -> Result<usize, EngineError> {
        let mut state = self.lock()?;
        expire_locked(&mut state, now_unix_millis)
    }

    /// Returns one stored status without exposing mutable authority.
    pub fn status(
        &self,
        reservation_id: &ReservationId,
    ) -> Result<Option<ReservationStatus>, EngineError> {
        let state = self.lock()?;
        Ok(state.records.get(reservation_id).map(|record| record.status))
    }

    /// Returns content-free bounded counters.
    pub fn statistics(&self) -> Result<ReservationStatistics, EngineError> {
        let state = self.lock()?;
        Ok(ReservationStatistics {
            records: state.records.len(),
            active: state.active,
            reserved_memory_bytes: state.reserved_memory_bytes,
        })
    }

    fn transition_terminal(
        &self,
        reservation_id: &ReservationId,
        request_id: RequestId,
        task_id: &TaskAuthorityId,
        identity: &RuntimeIdentityBinding,
        now_unix_millis: u64,
        target: ReservationStatus,
    ) -> Result<bool, EngineError> {
        let mut state = self.lock()?;
        expire_locked(&mut state, now_unix_millis)?;
        let record = state
            .records
            .get_mut(reservation_id)
            .ok_or_else(|| EngineError::invalid_input("runtime reservation does not exist"))?;
        validate_record_authority(record, request_id, task_id, identity)?;
        if record.status == target {
            return Ok(false);
        }
        if !matches!(record.status, ReservationStatus::Reserved | ReservationStatus::Consumed) {
            return Err(EngineError::authentication_failure(
                "terminal runtime reservation cannot change state",
            ));
        }
        let memory = record.request.budget.maximum_inference_memory_bytes;
        record.status = target;
        release_accounting(&mut state, memory)?;
        Ok(true)
    }

    fn lock(&self) -> Result<std::sync::MutexGuard<'_, ReservationState>, EngineError> {
        self.state
            .lock()
            .map_err(|_| EngineError::internal_invariant("runtime reservation lock was poisoned"))
    }
}

fn validate_record_authority(
    record: &ReservationRecord,
    request_id: RequestId,
    task_id: &TaskAuthorityId,
    identity: &RuntimeIdentityBinding,
) -> Result<(), EngineError> {
    if record.request.request_id != request_id
        || record.request.task_id != *task_id
        || record.request.identity != *identity
    {
        return Err(EngineError::authentication_failure(
            "runtime reservation authority does not match",
        ));
    }
    Ok(())
}

fn expire_locked(state: &mut ReservationState, now_unix_millis: u64) -> Result<usize, EngineError> {
    let mut expired = 0_usize;
    let mut released_memory = 0_u64;
    for record in state.records.values_mut() {
        if matches!(record.status, ReservationStatus::Reserved | ReservationStatus::Consumed)
            && record.request.expires_unix_millis <= now_unix_millis
        {
            record.status = ReservationStatus::Expired;
            expired = expired.checked_add(1).ok_or_else(|| {
                EngineError::resource_exhaustion("expired reservation count overflows")
            })?;
            released_memory = released_memory
                .checked_add(record.request.budget.maximum_inference_memory_bytes)
                .ok_or_else(|| {
                    EngineError::resource_exhaustion("expired reservation memory overflows")
                })?;
        }
    }
    state.active = state.active.checked_sub(expired).ok_or_else(|| {
        EngineError::internal_invariant("runtime active reservation count underflows")
    })?;
    state.reserved_memory_bytes =
        state.reserved_memory_bytes.checked_sub(released_memory).ok_or_else(|| {
            EngineError::internal_invariant("runtime reserved memory count underflows")
        })?;
    Ok(expired)
}

fn release_accounting(state: &mut ReservationState, memory: u64) -> Result<(), EngineError> {
    state.active = state.active.checked_sub(1).ok_or_else(|| {
        EngineError::internal_invariant("runtime active reservation count underflows")
    })?;
    state.reserved_memory_bytes =
        state.reserved_memory_bytes.checked_sub(memory).ok_or_else(|| {
            EngineError::internal_invariant("runtime reserved memory count underflows")
        })?;
    Ok(())
}

fn validate_authority_id(
    field: &'static str,
    value: &str,
    prefix: &str,
) -> Result<(), EngineError> {
    if !value.starts_with(prefix) {
        return Err(EngineError::invalid_field(field, "has the wrong authority prefix"));
    }
    validate_runtime_id(field, value)
}

fn validate_runtime_id(field: &'static str, value: &str) -> Result<(), EngineError> {
    if !(2..=MAX_AUTHORITY_ID_BYTES).contains(&value.len())
        || !value.is_ascii()
        || !value.as_bytes()[0].is_ascii_lowercase()
        || value.ends_with(['.', '_', ':', '-'])
        || !value.bytes().all(|byte| {
            byte.is_ascii_lowercase()
                || byte.is_ascii_digit()
                || matches!(byte, b'.' | b'_' | b':' | b'-')
        })
        || value
            .as_bytes()
            .windows(2)
            .any(|pair| matches!(pair, [b'.' | b'_' | b':' | b'-', b'.' | b'_' | b':' | b'-']))
    {
        return Err(EngineError::invalid_field(
            field,
            "must be a bounded path-free lowercase runtime identifier",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        ArtifactAuthorityId, LogicalModelId, ReservationDecision, ReservationDenial, ReservationId,
        ReservationStatus, RuntimeIdentityBinding, RuntimeInstanceId, RuntimeReservationManager,
        RuntimeReservationRequest, RuntimeResourceCeiling, TaskAuthorityId,
    };
    use personalai_engine_types::{
        BackendId, ModelHashAlgorithm, ModelId, ModelPackageHash, RequestId,
    };
    use std::sync::{Arc, Barrier};
    use std::thread;

    fn identity(provider_model: &str) -> RuntimeIdentityBinding {
        RuntimeIdentityBinding::new(
            LogicalModelId::new("mdl_fixture").unwrap(),
            ModelId::new(provider_model).unwrap(),
            ArtifactAuthorityId::new("art_fixture").unwrap(),
            ModelPackageHash::from_bytes(ModelHashAlgorithm::Sha384, [7_u8; 48]),
            BackendId::new("cpu-reference").unwrap(),
            RuntimeInstanceId::new("version.1").unwrap(),
            RuntimeInstanceId::new("backend.instance-1").unwrap(),
            RuntimeInstanceId::new("engine.instance-1").unwrap(),
        )
    }

    fn request(index: u128, memory: u64) -> RuntimeReservationRequest {
        RuntimeReservationRequest::new(
            ReservationId::new(format!("rsv_fixture-{index}")).unwrap(),
            RequestId::new(index).unwrap(),
            TaskAuthorityId::new(format!("tsk_fixture-{index}")).unwrap(),
            identity("fixture-model"),
            RuntimeResourceCeiling::new(1024, 64, memory, 5_000).unwrap(),
            [u8::try_from(index).unwrap_or(255); 32],
            10_000,
        )
        .unwrap()
    }

    #[test]
    fn exact_replay_is_idempotent_but_substitution_fails_closed() {
        let manager = RuntimeReservationManager::new(2, 200).unwrap();
        let original = request(1, 100);
        let decision = manager.reserve(original.clone(), 1).unwrap();
        assert_eq!(decision, ReservationDecision::Accepted(original.budget()));
        assert_eq!(manager.reserve(original.clone(), 2).unwrap(), decision);
        assert_eq!(manager.statistics().unwrap().active, 1);

        let mut substituted = original;
        substituted.identity = identity("other-model");
        assert!(manager.reserve(substituted, 2).is_err());
    }

    #[test]
    fn consume_cancel_release_and_expiry_are_exact_and_bounded() {
        let manager = RuntimeReservationManager::new(2, 100).unwrap();
        let first = request(1, 100);
        manager.reserve(first.clone(), 1).unwrap();
        assert!(
            manager
                .consume(
                    first.reservation_id(),
                    first.request_id(),
                    first.task_id(),
                    first.identity(),
                    64,
                    2,
                )
                .is_ok()
        );
        assert!(
            manager
                .consume(
                    first.reservation_id(),
                    first.request_id(),
                    first.task_id(),
                    first.identity(),
                    64,
                    2,
                )
                .is_err()
        );
        assert!(
            manager
                .cancel(
                    first.reservation_id(),
                    first.request_id(),
                    first.task_id(),
                    first.identity(),
                    3,
                )
                .unwrap()
        );
        assert!(
            !manager
                .cancel(
                    first.reservation_id(),
                    first.request_id(),
                    first.task_id(),
                    first.identity(),
                    3,
                )
                .unwrap()
        );
        assert_eq!(manager.statistics().unwrap().reserved_memory_bytes, 0);

        let second = request(2, 100);
        manager.reserve(second.clone(), 4).unwrap();
        assert_eq!(manager.expire(10_000).unwrap(), 1);
        assert_eq!(
            manager.status(second.reservation_id()).unwrap(),
            Some(ReservationStatus::Expired)
        );
        assert_eq!(manager.statistics().unwrap().active, 0);
    }

    #[test]
    fn concurrent_reservations_cannot_overcommit() {
        let manager = Arc::new(RuntimeReservationManager::new(1, 100).unwrap());
        let barrier = Arc::new(Barrier::new(3));
        let mut handles = Vec::new();
        for index in [1_u128, 2] {
            let manager = Arc::clone(&manager);
            let barrier = Arc::clone(&barrier);
            handles.push(thread::spawn(move || {
                barrier.wait();
                manager.reserve(request(index, 100), 1).unwrap()
            }));
        }
        barrier.wait();
        let decisions: Vec<_> = handles.into_iter().map(|handle| handle.join().unwrap()).collect();
        assert_eq!(
            decisions
                .iter()
                .filter(|decision| matches!(decision, ReservationDecision::Accepted(_)))
                .count(),
            1
        );
        assert_eq!(
            decisions
                .iter()
                .filter(|decision| {
                    matches!(
                        decision,
                        ReservationDecision::Denied(ReservationDenial::ConcurrentCapacity)
                    )
                })
                .count(),
            1
        );
        assert_eq!(manager.statistics().unwrap().reserved_memory_bytes, 100);
    }

    #[test]
    fn identity_and_budget_grammars_fail_closed() {
        assert!(LogicalModelId::new("mdl_../escape").is_err());
        assert!(ArtifactAuthorityId::new("mdl_wrong").is_err());
        assert!(RuntimeInstanceId::new("backend//instance").is_err());
        assert!(RuntimeResourceCeiling::new(0, 1, 1, 1).is_err());
    }

    #[test]
    fn engine_owned_identity_is_revalidated_exactly() {
        let target = identity("fixture-model");
        target
            .revalidate_engine_owned(
                target.provider_model_id(),
                target.package_hash(),
                target.backend_id(),
                target.backend_version(),
                target.backend_instance_id(),
                target.engine_instance_id(),
            )
            .unwrap();
        assert!(
            target
                .revalidate_engine_owned(
                    &ModelId::new("substituted-model").unwrap(),
                    target.package_hash(),
                    target.backend_id(),
                    target.backend_version(),
                    target.backend_instance_id(),
                    target.engine_instance_id(),
                )
                .is_err()
        );
        assert!(
            target
                .revalidate_engine_owned(
                    target.provider_model_id(),
                    &ModelPackageHash::from_bytes(ModelHashAlgorithm::Sha384, [8_u8; 48]),
                    target.backend_id(),
                    target.backend_version(),
                    target.backend_instance_id(),
                    target.engine_instance_id(),
                )
                .is_err()
        );
    }

    #[test]
    fn reservation_theft_and_output_escalation_fail_closed() {
        let manager = RuntimeReservationManager::new(1, 100).unwrap();
        let reserved = request(1, 100);
        manager.reserve(reserved.clone(), 1).unwrap();
        assert!(
            manager
                .consume(
                    reserved.reservation_id(),
                    RequestId::new(2).unwrap(),
                    reserved.task_id(),
                    reserved.identity(),
                    64,
                    2,
                )
                .is_err()
        );
        assert!(
            manager
                .consume(
                    reserved.reservation_id(),
                    reserved.request_id(),
                    reserved.task_id(),
                    reserved.identity(),
                    65,
                    2,
                )
                .is_err()
        );
        assert_eq!(
            manager.status(reserved.reservation_id()).unwrap(),
            Some(ReservationStatus::Reserved)
        );
    }
}
