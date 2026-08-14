//! Strict canonical inference-runtime-v3 document adapter.
//!
//! The enclosing authenticated Unix-domain-socket service is responsible for
//! peer authentication, confidentiality, and replay protection. This adapter
//! owns only bounded contract decoding, exact identity revalidation, resource
//! admission, single-consumption, cancellation, and release. It deliberately
//! has no cognitive role, qualification, user policy, cloud policy, tool, or
//! memory authority.

use crate::{
    ArtifactAuthorityId, LogicalModelId, MAX_RUNTIME_RESERVATION_RECORDS, ReservationDecision,
    ReservationDenial, ReservationId, ReservationStatus, RuntimeIdentityBinding, RuntimeInstanceId,
    RuntimeReservationManager, RuntimeReservationRequest, RuntimeResourceCeiling, TaskAuthorityId,
};
use personalai_engine_types::{
    BackendId, EngineError, MAX_FRAME_BYTES, ModelHashAlgorithm, ModelId, ModelPackageHash,
    RequestId,
};
use sovereign_ai_contracts::inference_runtime_v3::{
    ArtifactDigestAlgorithmV3, InferenceExecutionRequestV3, InferenceExecutionResultV3,
    InferenceReservationDecisionV3, InferenceReservationRequestV3,
    InferenceRuntimeContractVersionV3, ReservationDecisionKindV3, ReservationDecisionV3,
    RuntimeIdentityV3, RuntimeResourceBudgetV3, RuntimeResourceUnitV3, RuntimeResourceV3,
    decode_inference_execution_request_v3, decode_inference_reservation_release_v3,
    decode_inference_reservation_request_v3, decode_runtime_cancellation_request_v3,
    encode_inference_execution_result_v3, encode_inference_reservation_decision_v3,
    runtime_v3_timestamp_from_unix_millis,
};
use std::collections::BTreeMap;
use std::sync::Mutex;

/// Revalidates engine-owned runtime identity immediately before admission and
/// again immediately before execution.
pub trait RuntimeIdentityAuthority: Send + Sync {
    /// Proves that the exact provider model, package digest, backend, and
    /// configured runtime instances remain current.
    fn revalidate(&self, identity: &RuntimeIdentityBinding) -> Result<(), EngineError>;
}

#[derive(Clone, Debug, PartialEq)]
struct AdapterRecord {
    request: InferenceReservationRequestV3,
    decision: InferenceReservationDecisionV3,
    encoded_decision: Vec<u8>,
    identity: RuntimeIdentityBinding,
    request_id: RequestId,
    task_id: TaskAuthorityId,
    reservation_id: ReservationId,
    result: Option<InferenceExecutionResultV3>,
    encoded_result: Option<Vec<u8>>,
}

#[derive(Debug)]
struct AdapterState {
    records: BTreeMap<ReservationId, AdapterRecord>,
    next_status_revision: u64,
}

/// Immutable execution admission returned only after exact reservation
/// consumption and a second engine-owned identity proof.
#[derive(Clone, Debug, PartialEq)]
pub struct AdmittedRuntimeExecution {
    request: InferenceExecutionRequestV3,
    identity: RuntimeIdentityBinding,
    request_id: RequestId,
    task_id: TaskAuthorityId,
    reservation_id: ReservationId,
}

impl AdmittedRuntimeExecution {
    /// Returns the validated canonical execution request.
    pub const fn request(&self) -> &InferenceExecutionRequestV3 {
        &self.request
    }

    /// Returns the exact engine-owned target revalidated before consumption.
    pub const fn identity(&self) -> &RuntimeIdentityBinding {
        &self.identity
    }

    /// Returns the engine request correlation identifier.
    pub const fn engine_request_id(&self) -> RequestId {
        self.request_id
    }

    /// Returns the exact task authority correlation.
    pub const fn task_id(&self) -> &TaskAuthorityId {
        &self.task_id
    }

    /// Returns the exact consumed reservation identifier.
    pub const fn reservation_id(&self) -> &ReservationId {
        &self.reservation_id
    }
}

/// Bounded bridge from canonical Sovereign Contracts documents to the
/// engine-owned runtime reservation state machine.
#[derive(Debug)]
pub struct RuntimeV3ContractAdapter<A> {
    reservations: RuntimeReservationManager,
    identity_authority: A,
    state: Mutex<AdapterState>,
}

impl<A: RuntimeIdentityAuthority> RuntimeV3ContractAdapter<A> {
    /// Constructs an empty adapter with explicit engine-wide reservation
    /// limits and one engine-owned identity authority.
    pub fn new(
        maximum_active: usize,
        maximum_memory_bytes: u64,
        identity_authority: A,
    ) -> Result<Self, EngineError> {
        Ok(Self {
            reservations: RuntimeReservationManager::new(maximum_active, maximum_memory_bytes)?,
            identity_authority,
            state: Mutex::new(AdapterState { records: BTreeMap::new(), next_status_revision: 1 }),
        })
    }

    /// Strictly decodes, identity-checks, and evaluates one canonical
    /// reservation request. Exact replay returns the original encoded decision.
    pub fn reserve_document(
        &self,
        source: &[u8],
        now_unix_millis: u64,
    ) -> Result<Vec<u8>, EngineError> {
        let request = decode_reservation(source)?;
        let created = contract_result(request.created_unix_millis())?;
        let expires = contract_result(request.expires_unix_millis())?;
        if now_unix_millis < created {
            return Err(EngineError::invalid_field(
                "created_at",
                "is in the future relative to the authenticated service clock",
            ));
        }

        let mapped = map_reservation(&request, expires)?;
        self.identity_authority.revalidate(mapped.identity())?;

        let mut state = self.lock_state()?;
        if let Some(existing) = state.records.get(mapped.reservation_id()) {
            if existing.request == request {
                return Ok(existing.encoded_decision.clone());
            }
            return Err(EngineError::authentication_failure(
                "runtime reservation identifier was replayed with different authority",
            ));
        }
        if state.records.len() >= MAX_RUNTIME_RESERVATION_RECORDS {
            return Err(EngineError::resource_exhaustion(
                "runtime v3 adapter record capacity is exhausted",
            ));
        }

        let decision = self.reservations.reserve(mapped.clone(), now_unix_millis)?;
        let status_revision = state.next_status_revision;
        state.next_status_revision = state
            .next_status_revision
            .checked_add(1)
            .ok_or_else(|| EngineError::internal_invariant("runtime status revision overflowed"))?;
        let canonical = canonical_decision(&request, decision, status_revision, now_unix_millis)?;
        contract_result(canonical.validate_against(&request))?;
        let encoded = contract_result(encode_inference_reservation_decision_v3(&canonical))?;
        state.records.insert(
            mapped.reservation_id().clone(),
            AdapterRecord {
                request,
                decision: canonical,
                encoded_decision: encoded.clone(),
                identity: mapped.identity().clone(),
                request_id: mapped.request_id(),
                task_id: mapped.task_id().clone(),
                reservation_id: mapped.reservation_id().clone(),
                result: None,
                encoded_result: None,
            },
        );
        Ok(encoded)
    }

    /// Strictly decodes one execution request, proves it consumes the exact
    /// accepted reservation, revalidates live identity, and consumes once.
    pub fn admit_execution_document(
        &self,
        source: &[u8],
        now_unix_millis: u64,
    ) -> Result<AdmittedRuntimeExecution, EngineError> {
        check_document_bound(source)?;
        let request = contract_result(decode_inference_execution_request_v3(source))?;
        let created = contract_result(request.created_unix_millis())?;
        let deadline = contract_result(request.deadline_unix_millis())?;
        if now_unix_millis < created || now_unix_millis >= deadline {
            return Err(EngineError::invalid_field(
                "deadline_at",
                "execution is outside its authenticated time window",
            ));
        }

        let reservation_id = ReservationId::new(request.reservation_id.clone())?;
        let state = self.lock_state()?;
        let record = state.records.get(&reservation_id).ok_or_else(|| {
            EngineError::authentication_failure("runtime reservation decision does not exist")
        })?;
        contract_result(request.validate_against(&record.decision))?;
        let mapped_identity = map_identity(&request.identity)?;
        let mapped_request = map_request_id(request.request_id.as_str())?;
        let mapped_task = TaskAuthorityId::new(request.task_id.as_str())?;
        if mapped_identity != record.identity
            || mapped_request != record.request_id
            || mapped_task != record.task_id
        {
            return Err(EngineError::authentication_failure(
                "runtime execution changed reserved identity or correlation",
            ));
        }
        self.identity_authority.revalidate(&mapped_identity)?;
        self.reservations.consume(
            &reservation_id,
            mapped_request,
            &mapped_task,
            &mapped_identity,
            u64::from(request.parameters.max_output_tokens),
            now_unix_millis,
        )?;
        Ok(AdmittedRuntimeExecution {
            request,
            identity: mapped_identity,
            request_id: mapped_request,
            task_id: mapped_task,
            reservation_id,
        })
    }

    /// Strictly validates and applies one exact cancellation. Repeating the
    /// exact cancellation is idempotent and returns `false`.
    pub fn cancel_document(
        &self,
        source: &[u8],
        now_unix_millis: u64,
    ) -> Result<bool, EngineError> {
        check_document_bound(source)?;
        let cancellation = contract_result(decode_runtime_cancellation_request_v3(source))?;
        let requested = contract_result(cancellation.requested_unix_millis())?;
        if now_unix_millis < requested {
            return Err(EngineError::invalid_field(
                "requested_at",
                "is in the future relative to the authenticated service clock",
            ));
        }
        let reservation_id = ReservationId::new(cancellation.reservation_id)?;
        let state = self.lock_state()?;
        let record = state.records.get(&reservation_id).ok_or_else(|| {
            EngineError::authentication_failure("runtime reservation decision does not exist")
        })?;
        validate_terminal_correlation(
            cancellation.request_id.as_str(),
            cancellation.task_id.as_str(),
            record,
        )?;
        self.reservations.cancel(
            &record.reservation_id,
            record.request_id,
            &record.task_id,
            &record.identity,
            now_unix_millis,
        )
    }

    /// Strictly validates and applies one exact release. Repeating the exact
    /// release is idempotent and returns `false`.
    pub fn release_document(
        &self,
        source: &[u8],
        now_unix_millis: u64,
    ) -> Result<bool, EngineError> {
        check_document_bound(source)?;
        let release = contract_result(decode_inference_reservation_release_v3(source))?;
        let requested = contract_result(release.requested_unix_millis())?;
        if now_unix_millis < requested {
            return Err(EngineError::invalid_field(
                "requested_at",
                "is in the future relative to the authenticated service clock",
            ));
        }
        let reservation_id = ReservationId::new(release.reservation_id)?;
        let state = self.lock_state()?;
        let record = state.records.get(&reservation_id).ok_or_else(|| {
            EngineError::authentication_failure("runtime reservation decision does not exist")
        })?;
        validate_terminal_correlation(
            release.request_id.as_str(),
            release.task_id.as_str(),
            record,
        )?;
        self.reservations.release(
            &record.reservation_id,
            record.request_id,
            &record.task_id,
            &record.identity,
            now_unix_millis,
        )
    }

    /// Validates and serializes one identity-bound terminal result, enforcing
    /// the accepted token and resource ceilings before releasing capacity.
    /// A result is finalized at most once.
    pub fn finalize_execution_result(
        &self,
        admitted: &AdmittedRuntimeExecution,
        result: &InferenceExecutionResultV3,
        now_unix_millis: u64,
    ) -> Result<Vec<u8>, EngineError> {
        let mut state = self.lock_state()?;
        let record = state.records.get_mut(&admitted.reservation_id).ok_or_else(|| {
            EngineError::authentication_failure("runtime reservation decision does not exist")
        })?;
        if let Some(existing) = record.result.as_ref() {
            if existing == result {
                return record.encoded_result.clone().ok_or_else(|| {
                    EngineError::internal_invariant(
                        "final runtime result is missing its encoded document",
                    )
                });
            }
            return Err(EngineError::authentication_failure(
                "runtime execution result was replayed with different content",
            ));
        }
        if admitted.request_id != record.request_id
            || admitted.task_id != record.task_id
            || admitted.identity != record.identity
        {
            return Err(EngineError::authentication_failure(
                "admitted runtime execution no longer matches its reservation",
            ));
        }
        contract_result(admitted.request.validate_against(&record.decision))?;
        contract_result(result.validate_against(&admitted.request))?;
        enforce_result_budget(result, &record.decision)?;

        if matches!(
            result.status,
            sovereign_ai_contracts::inference_runtime_v3::ExecutionStatusV3::Succeeded
        ) {
            if let Err(error) = self.identity_authority.revalidate(&record.identity) {
                let _ = self.reservations.release(
                    &record.reservation_id,
                    record.request_id,
                    &record.task_id,
                    &record.identity,
                    now_unix_millis,
                );
                return Err(error);
            }
        }

        let encoded = contract_result(encode_inference_execution_result_v3(result))?;
        match self.reservations.status(&record.reservation_id)? {
            Some(ReservationStatus::Consumed) => {
                if !self.reservations.release(
                    &record.reservation_id,
                    record.request_id,
                    &record.task_id,
                    &record.identity,
                    now_unix_millis,
                )? {
                    return Err(EngineError::internal_invariant(
                        "consumed runtime reservation was not released",
                    ));
                }
            }
            Some(ReservationStatus::Cancelled)
                if matches!(
                    result.status,
                    sovereign_ai_contracts::inference_runtime_v3::ExecutionStatusV3::Cancelled
                ) => {}
            _ => {
                return Err(EngineError::authentication_failure(
                    "runtime reservation is not eligible for terminal result finalization",
                ));
            }
        }
        record.result = Some(result.clone());
        record.encoded_result = Some(encoded.clone());
        Ok(encoded)
    }

    fn lock_state(&self) -> Result<std::sync::MutexGuard<'_, AdapterState>, EngineError> {
        self.state
            .lock()
            .map_err(|_| EngineError::internal_invariant("runtime v3 adapter lock was poisoned"))
    }
}

fn decode_reservation(source: &[u8]) -> Result<InferenceReservationRequestV3, EngineError> {
    check_document_bound(source)?;
    contract_result(decode_inference_reservation_request_v3(source))
}

fn check_document_bound(source: &[u8]) -> Result<(), EngineError> {
    if source.is_empty() || source.len() > MAX_FRAME_BYTES {
        return Err(EngineError::invalid_field(
            "runtime_document",
            "is empty or exceeds the authenticated frame bound",
        ));
    }
    Ok(())
}

fn map_reservation(
    request: &InferenceReservationRequestV3,
    expires_unix_millis: u64,
) -> Result<RuntimeReservationRequest, EngineError> {
    RuntimeReservationRequest::new(
        ReservationId::new(request.reservation_id.clone())?,
        map_request_id(request.request_id.as_str())?,
        TaskAuthorityId::new(request.task_id.as_str())?,
        map_identity(&request.identity)?,
        map_budget(&request.budget)?,
        parse_sha256(&request.idempotency_key)?,
        expires_unix_millis,
    )
}

fn map_identity(identity: &RuntimeIdentityV3) -> Result<RuntimeIdentityBinding, EngineError> {
    if !matches!(identity.artifact_digest_algorithm, ArtifactDigestAlgorithmV3::Sha384) {
        return Err(EngineError::invalid_field(
            "artifact_digest_algorithm",
            "first-party AIPack identity requires sha384",
        ));
    }
    Ok(RuntimeIdentityBinding::new(
        LogicalModelId::new(identity.model_id.as_str())?,
        ModelId::new(identity.provider_model_id.clone())?,
        ArtifactAuthorityId::new(identity.artifact_id.as_str())?,
        ModelPackageHash::from_hex(ModelHashAlgorithm::Sha384, &identity.artifact_digest)?,
        BackendId::new(identity.backend_id.clone())?,
        RuntimeInstanceId::new(identity.backend_version.clone())?,
        RuntimeInstanceId::new(identity.backend_instance_id.clone())?,
        RuntimeInstanceId::new(identity.engine_instance_id.clone())?,
    ))
}

fn map_budget(budget: &RuntimeResourceBudgetV3) -> Result<RuntimeResourceCeiling, EngineError> {
    RuntimeResourceCeiling::new(
        budget.maximum_input_tokens,
        budget.maximum_output_tokens,
        budget.maximum_inference_memory_bytes,
        budget.maximum_execution_millis,
    )
}

fn map_request_id(value: &str) -> Result<RequestId, EngineError> {
    let encoded = value.strip_prefix("req_").ok_or_else(|| {
        EngineError::invalid_field("request_id", "must use the req_ authority prefix")
    })?;
    if encoded.len() != 32
        || !encoded.bytes().all(|byte| byte.is_ascii_digit() || matches!(byte, b'a'..=b'f'))
    {
        return Err(EngineError::invalid_field(
            "request_id",
            "native runtime profile requires exactly 128 bits of lowercase hexadecimal entropy",
        ));
    }
    let value = u128::from_str_radix(encoded, 16)
        .map_err(|_| EngineError::invalid_field("request_id", "contains invalid hexadecimal"))?;
    RequestId::new(value)
}

fn parse_sha256(value: &str) -> Result<[u8; 32], EngineError> {
    if value.len() != 64 {
        return Err(EngineError::invalid_field(
            "idempotency_key",
            "must contain exactly 64 lowercase hexadecimal characters",
        ));
    }
    let mut digest = [0_u8; 32];
    for (index, pair) in value.as_bytes().chunks_exact(2).enumerate() {
        let high = hex_nibble(pair[0]).ok_or_else(|| {
            EngineError::invalid_field("idempotency_key", "contains invalid lowercase hexadecimal")
        })?;
        let low = hex_nibble(pair[1]).ok_or_else(|| {
            EngineError::invalid_field("idempotency_key", "contains invalid lowercase hexadecimal")
        })?;
        digest[index] = (high << 4) | low;
    }
    Ok(digest)
}

const fn hex_nibble(value: u8) -> Option<u8> {
    match value {
        b'0'..=b'9' => Some(value - b'0'),
        b'a'..=b'f' => Some(value - b'a' + 10),
        _ => None,
    }
}

fn canonical_decision(
    request: &InferenceReservationRequestV3,
    decision: ReservationDecision,
    status_revision: u64,
    now_unix_millis: u64,
) -> Result<InferenceReservationDecisionV3, EngineError> {
    let observed_at = contract_result(runtime_v3_timestamp_from_unix_millis(now_unix_millis))?;
    let (decision, granted_budget, denial_code, expires_at) = match decision {
        ReservationDecision::Accepted(budget) => (
            ReservationDecisionV3::Accepted,
            Some(RuntimeResourceBudgetV3 {
                maximum_input_tokens: budget.maximum_input_tokens(),
                maximum_output_tokens: budget.maximum_output_tokens(),
                maximum_inference_memory_bytes: budget.maximum_inference_memory_bytes(),
                maximum_execution_millis: budget.maximum_execution_millis(),
            }),
            None,
            Some(request.expires_at.clone()),
        ),
        ReservationDecision::Denied(reason) => (
            ReservationDecisionV3::Denied,
            None,
            Some(
                match reason {
                    ReservationDenial::RecordCapacity => "record_capacity",
                    ReservationDenial::ConcurrentCapacity => "concurrent_capacity",
                    ReservationDenial::MemoryCapacity => "memory_capacity",
                    ReservationDenial::Expired => "expired",
                }
                .to_owned(),
            ),
            None,
        ),
    };
    Ok(InferenceReservationDecisionV3 {
        contract_version: InferenceRuntimeContractVersionV3,
        kind: ReservationDecisionKindV3::ReservationDecisionV3,
        reservation_id: request.reservation_id.clone(),
        request_id: request.request_id.clone(),
        task_id: request.task_id.clone(),
        identity: request.identity.clone(),
        decision,
        granted_budget,
        denial_code,
        status_revision,
        observed_at,
        expires_at,
        extensions: None,
        required_extensions: None,
    })
}

fn enforce_result_budget(
    result: &InferenceExecutionResultV3,
    decision: &InferenceReservationDecisionV3,
) -> Result<(), EngineError> {
    let granted = decision.granted_budget.as_ref().ok_or_else(|| {
        EngineError::authentication_failure("runtime result has no accepted resource authority")
    })?;
    if result.usage.input_tokens > granted.maximum_input_tokens
        || result.usage.output_tokens > granted.maximum_output_tokens
    {
        return Err(EngineError::resource_exhaustion(
            "runtime result token usage exceeds the accepted reservation",
        ));
    }

    let mut has_memory = false;
    let mut has_wall_time = false;
    for usage in &result.resource_usage {
        match (&usage.resource, &usage.unit) {
            (RuntimeResourceV3::InferenceMemory, RuntimeResourceUnitV3::Bytes) => {
                has_memory = true;
                if usage.units > granted.maximum_inference_memory_bytes {
                    return Err(EngineError::resource_exhaustion(
                        "runtime result memory usage exceeds the accepted reservation",
                    ));
                }
            }
            (RuntimeResourceV3::WallTime, RuntimeResourceUnitV3::Milliseconds) => {
                has_wall_time = true;
                if usage.units > granted.maximum_execution_millis {
                    return Err(EngineError::resource_exhaustion(
                        "runtime result wall time exceeds the accepted reservation",
                    ));
                }
            }
            (
                RuntimeResourceV3::CpuTime | RuntimeResourceV3::AcceleratorTime,
                RuntimeResourceUnitV3::Milliseconds,
            ) => {
                if usage.units > granted.maximum_execution_millis {
                    return Err(EngineError::resource_exhaustion(
                        "runtime result compute time exceeds the accepted reservation",
                    ));
                }
            }
            _ => {
                return Err(EngineError::invalid_input(
                    "runtime result contains a resource/unit mismatch",
                ));
            }
        }
    }
    if !has_memory || !has_wall_time {
        return Err(EngineError::invalid_input(
            "runtime result must report inference memory and wall time",
        ));
    }
    Ok(())
}

fn validate_terminal_correlation(
    request_id: &str,
    task_id: &str,
    record: &AdapterRecord,
) -> Result<(), EngineError> {
    if map_request_id(request_id)? != record.request_id
        || TaskAuthorityId::new(task_id)? != record.task_id
    {
        return Err(EngineError::authentication_failure(
            "runtime terminal request changed reservation correlation",
        ));
    }
    Ok(())
}

fn contract_result<T>(result: sovereign_ai_contracts::Result<T>) -> Result<T, EngineError> {
    result.map_err(|error| EngineError::invalid_input(format!("runtime v3 contract: {error}")))
}

#[cfg(test)]
mod tests {
    use super::{RuntimeIdentityAuthority, RuntimeV3ContractAdapter, map_identity, map_request_id};
    use crate::RuntimeIdentityBinding;
    use personalai_engine_types::EngineError;
    use sha2::{Digest, Sha256};
    use sovereign_ai_contracts::identity::{ModelArtifactId, ModelId, RequestId, TaskId};
    use sovereign_ai_contracts::inference::{
        GenerationParameters, GenerationUsage, InferenceMessage, InferenceRole,
    };
    use sovereign_ai_contracts::inference_runtime_v3::{
        ArtifactDigestAlgorithmV3, ExecutionRequestKindV3, ExecutionResultKindV3,
        ExecutionStatusV3, InferenceExecutionRequestV3, InferenceExecutionResultV3,
        InferenceReservationReleaseV3, InferenceReservationRequestV3,
        InferenceRuntimeContractVersionV3, ReservationDecisionV3, ReservationReleaseKindV3,
        ReservationRequestKindV3, RuntimeCancellationRequestKindV3, RuntimeCancellationRequestV3,
        RuntimeFailureClassV3, RuntimeFailureV3, RuntimeIdentityV3, RuntimeResourceBudgetV3,
        RuntimeResourceUnitV3, RuntimeResourceUsageV3, RuntimeResourceV3,
        decode_inference_execution_result_v3, decode_inference_reservation_decision_v3,
        encode_inference_execution_request_v3, encode_inference_reservation_release_v3,
        encode_inference_reservation_request_v3, encode_runtime_cancellation_request_v3,
        runtime_v3_timestamp_from_unix_millis,
    };
    use std::fmt::Write as _;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};

    #[derive(Debug)]
    struct ExactIdentityAuthority {
        expected: RuntimeIdentityBinding,
        available: AtomicBool,
    }

    impl RuntimeIdentityAuthority for Arc<ExactIdentityAuthority> {
        fn revalidate(&self, identity: &RuntimeIdentityBinding) -> Result<(), EngineError> {
            if !self.available.load(Ordering::SeqCst) || *identity != self.expected {
                return Err(EngineError::authentication_failure(
                    "fixture runtime identity is unavailable or changed",
                ));
            }
            Ok(())
        }
    }

    fn identity() -> RuntimeIdentityV3 {
        RuntimeIdentityV3 {
            model_id: ModelId::new("mdl_fixture").expect("logical model"),
            provider_model_id: "fixture-model".to_owned(),
            artifact_id: ModelArtifactId::new("art_fixture").expect("artifact"),
            artifact_digest_algorithm: ArtifactDigestAlgorithmV3::Sha384,
            artifact_digest: "07".repeat(48),
            backend_id: "cpu-reference".to_owned(),
            backend_version: "version.1".to_owned(),
            backend_instance_id: "backend.instance-1".to_owned(),
            engine_instance_id: "engine.instance-1".to_owned(),
        }
    }

    fn reservation() -> InferenceReservationRequestV3 {
        InferenceReservationRequestV3 {
            contract_version: InferenceRuntimeContractVersionV3,
            kind: ReservationRequestKindV3::ReservationRequest,
            reservation_id: "rsv_fixture-1".to_owned(),
            request_id: RequestId::new("req_11111111111111111111111111111111").expect("request"),
            task_id: TaskId::new("tsk_fixture").expect("task"),
            identity: identity(),
            budget: RuntimeResourceBudgetV3 {
                maximum_input_tokens: 1_024,
                maximum_output_tokens: 64,
                maximum_inference_memory_bytes: 1_024,
                maximum_execution_millis: 5_000,
            },
            idempotency_key: "11".repeat(32),
            created_at: "2026-08-14T12:00:00.000Z".to_owned(),
            expires_at: "2026-08-14T12:00:10.000Z".to_owned(),
            extensions: None,
            required_extensions: None,
        }
    }

    fn execution(reservation: &InferenceReservationRequestV3) -> InferenceExecutionRequestV3 {
        let created = reservation.created_unix_millis().expect("creation") + 1;
        InferenceExecutionRequestV3 {
            contract_version: InferenceRuntimeContractVersionV3,
            kind: ExecutionRequestKindV3::ExecutionRequest,
            reservation_id: reservation.reservation_id.clone(),
            request_id: reservation.request_id.clone(),
            task_id: reservation.task_id.clone(),
            identity: reservation.identity.clone(),
            messages: vec![InferenceMessage {
                role: InferenceRole::User,
                content: "deterministic fixture prompt".to_owned(),
                name: None,
            }],
            parameters: GenerationParameters {
                max_output_tokens: 32,
                temperature: 0.0,
                top_p: 1.0,
                seed: Some(7),
                stop_sequences: None,
            },
            idempotency_key: "22".repeat(32),
            created_at: runtime_v3_timestamp_from_unix_millis(created).expect("created timestamp"),
            deadline_at: runtime_v3_timestamp_from_unix_millis(created + 5_000)
                .expect("deadline timestamp"),
            extensions: None,
            required_extensions: None,
        }
    }

    fn adapter()
    -> (RuntimeV3ContractAdapter<Arc<ExactIdentityAuthority>>, Arc<ExactIdentityAuthority>) {
        let expected = map_identity(&identity()).expect("map fixture identity");
        let authority =
            Arc::new(ExactIdentityAuthority { expected, available: AtomicBool::new(true) });
        (
            RuntimeV3ContractAdapter::new(2, 2_048, Arc::clone(&authority))
                .expect("construct adapter"),
            authority,
        )
    }

    fn successful_result(execution: &InferenceExecutionRequestV3) -> InferenceExecutionResultV3 {
        let started = execution.created_unix_millis().expect("created") + 1;
        let output = "deterministic fixture output".to_owned();
        let mut output_digest = String::with_capacity(64);
        for byte in Sha256::digest(output.as_bytes()) {
            write!(&mut output_digest, "{byte:02x}").expect("write digest");
        }
        InferenceExecutionResultV3 {
            contract_version: InferenceRuntimeContractVersionV3,
            kind: ExecutionResultKindV3::ExecutionResult,
            reservation_id: execution.reservation_id.clone(),
            request_id: execution.request_id.clone(),
            task_id: execution.task_id.clone(),
            identity: execution.identity.clone(),
            status: ExecutionStatusV3::Succeeded,
            started_at: runtime_v3_timestamp_from_unix_millis(started).expect("start timestamp"),
            completed_at: runtime_v3_timestamp_from_unix_millis(started + 10)
                .expect("completion timestamp"),
            usage: GenerationUsage { input_tokens: 12, output_tokens: 4, total_tokens: 16 },
            resource_usage: vec![
                RuntimeResourceUsageV3 {
                    resource: RuntimeResourceV3::InferenceMemory,
                    units: 512,
                    unit: RuntimeResourceUnitV3::Bytes,
                },
                RuntimeResourceUsageV3 {
                    resource: RuntimeResourceV3::WallTime,
                    units: 10,
                    unit: RuntimeResourceUnitV3::Milliseconds,
                },
            ],
            output_digest: Some(output_digest),
            output: Some(output),
            failure: None,
            extensions: None,
            required_extensions: None,
        }
    }

    fn cancelled_result(execution: &InferenceExecutionRequestV3) -> InferenceExecutionResultV3 {
        let started = execution.created_unix_millis().expect("created") + 1;
        InferenceExecutionResultV3 {
            contract_version: InferenceRuntimeContractVersionV3,
            kind: ExecutionResultKindV3::ExecutionResult,
            reservation_id: execution.reservation_id.clone(),
            request_id: execution.request_id.clone(),
            task_id: execution.task_id.clone(),
            identity: execution.identity.clone(),
            status: ExecutionStatusV3::Cancelled,
            started_at: runtime_v3_timestamp_from_unix_millis(started).expect("start timestamp"),
            completed_at: runtime_v3_timestamp_from_unix_millis(started + 10)
                .expect("completion timestamp"),
            usage: GenerationUsage { input_tokens: 0, output_tokens: 0, total_tokens: 0 },
            resource_usage: vec![
                RuntimeResourceUsageV3 {
                    resource: RuntimeResourceV3::InferenceMemory,
                    units: 0,
                    unit: RuntimeResourceUnitV3::Bytes,
                },
                RuntimeResourceUsageV3 {
                    resource: RuntimeResourceV3::WallTime,
                    units: 10,
                    unit: RuntimeResourceUnitV3::Milliseconds,
                },
            ],
            output_digest: None,
            output: None,
            failure: Some(RuntimeFailureV3 {
                class: RuntimeFailureClassV3::Cancelled,
                code: "owner_cancelled".to_owned(),
                retryable: false,
            }),
            extensions: None,
            required_extensions: None,
        }
    }

    #[test]
    fn exact_reservation_replay_and_single_execution_are_enforced() {
        let (adapter, _) = adapter();
        let reservation = reservation();
        let source = encode_inference_reservation_request_v3(&reservation).expect("encode reserve");
        let now = reservation.created_unix_millis().expect("created") + 1;
        let first = adapter.reserve_document(&source, now).expect("reserve exact identity");
        assert_eq!(adapter.reserve_document(&source, now + 1).expect("exact replay"), first);
        let decision = decode_inference_reservation_decision_v3(&first).expect("decode decision");
        assert_eq!(decision.decision, ReservationDecisionV3::Accepted);

        let execution = execution(&reservation);
        let source = encode_inference_execution_request_v3(&execution).expect("encode execution");
        let admitted = adapter
            .admit_execution_document(&source, execution.created_unix_millis().expect("created"))
            .expect("admit once");
        assert_eq!(admitted.request(), &execution);
        assert!(
            adapter
                .admit_execution_document(
                    &source,
                    execution.created_unix_millis().expect("created")
                )
                .is_err()
        );
    }

    #[test]
    fn identity_is_revalidated_immediately_before_consumption() {
        let (adapter, authority) = adapter();
        let reservation = reservation();
        let source = encode_inference_reservation_request_v3(&reservation).expect("encode reserve");
        let now = reservation.created_unix_millis().expect("created") + 1;
        adapter.reserve_document(&source, now).expect("reserve");
        let execution = execution(&reservation);
        let source = encode_inference_execution_request_v3(&execution).expect("encode execution");

        authority.available.store(false, Ordering::SeqCst);
        assert!(adapter.admit_execution_document(&source, now + 1).is_err());
        authority.available.store(true, Ordering::SeqCst);
        adapter
            .admit_execution_document(&source, now + 1)
            .expect("reservation was not consumed by failed identity proof");
    }

    #[test]
    fn terminal_result_is_identity_bound_budgeted_and_idempotent() {
        let (adapter, _) = adapter();
        let reservation = reservation();
        let source = encode_inference_reservation_request_v3(&reservation).expect("encode reserve");
        let now = reservation.created_unix_millis().expect("created") + 1;
        adapter.reserve_document(&source, now).expect("reserve");
        let execution = execution(&reservation);
        let source = encode_inference_execution_request_v3(&execution).expect("encode execution");
        let admitted = adapter
            .admit_execution_document(&source, execution.created_unix_millis().expect("created"))
            .expect("admit");
        let result = successful_result(&execution);
        let completed = result.completed_unix_millis().expect("completed");
        let encoded =
            adapter.finalize_execution_result(&admitted, &result, completed).expect("finalize");
        assert_eq!(decode_inference_execution_result_v3(&encoded).expect("decode result"), result);
        assert_eq!(
            adapter
                .finalize_execution_result(&admitted, &result, completed)
                .expect("exact result replay"),
            encoded
        );

        let mut substituted = result;
        substituted.identity.backend_instance_id = "backend.instance-2".to_owned();
        assert!(adapter.finalize_execution_result(&admitted, &substituted, completed).is_err());
    }

    #[test]
    fn terminal_result_cannot_exceed_reserved_resource_accounting() {
        let (adapter, _) = adapter();
        let reservation = reservation();
        let source = encode_inference_reservation_request_v3(&reservation).expect("encode reserve");
        let now = reservation.created_unix_millis().expect("created") + 1;
        adapter.reserve_document(&source, now).expect("reserve");
        let execution = execution(&reservation);
        let source = encode_inference_execution_request_v3(&execution).expect("encode execution");
        let admitted = adapter
            .admit_execution_document(&source, execution.created_unix_millis().expect("created"))
            .expect("admit");
        let mut result = successful_result(&execution);
        result.resource_usage[0].units = 1_025;
        assert!(
            adapter
                .finalize_execution_result(
                    &admitted,
                    &result,
                    result.completed_unix_millis().expect("completed")
                )
                .is_err()
        );
    }

    #[test]
    fn artifact_substitution_and_noncanonical_request_ids_fail_closed() {
        let (adapter, _) = adapter();
        let reservation = reservation();
        let source = encode_inference_reservation_request_v3(&reservation).expect("encode reserve");
        let now = reservation.created_unix_millis().expect("created") + 1;
        adapter.reserve_document(&source, now).expect("reserve");
        let mut execution = execution(&reservation);
        execution.identity.artifact_digest = "08".repeat(48);
        let source =
            encode_inference_execution_request_v3(&execution).expect("encode substitution");
        assert!(adapter.admit_execution_document(&source, now + 1).is_err());
        assert!(map_request_id("req_short").is_err());
        assert!(map_request_id("req_00000000000000000000000000000000").is_err());
    }

    #[test]
    fn cancellation_and_release_are_exact_and_idempotent() {
        let (adapter, _) = adapter();
        let reservation = reservation();
        let source = encode_inference_reservation_request_v3(&reservation).expect("encode reserve");
        let now = reservation.created_unix_millis().expect("created") + 1;
        adapter.reserve_document(&source, now).expect("reserve");

        let cancellation = RuntimeCancellationRequestV3 {
            contract_version: InferenceRuntimeContractVersionV3,
            kind: RuntimeCancellationRequestKindV3::RuntimeCancellationRequestV3,
            reservation_id: reservation.reservation_id.clone(),
            request_id: reservation.request_id.clone(),
            task_id: reservation.task_id.clone(),
            cancellation_request_id: RequestId::new("req_22222222222222222222222222222222")
                .expect("cancel request"),
            reason_code: "owner_cancelled".to_owned(),
            requested_at: runtime_v3_timestamp_from_unix_millis(now).expect("cancel time"),
            extensions: None,
            required_extensions: None,
        };
        let source = encode_runtime_cancellation_request_v3(&cancellation).expect("encode cancel");
        assert!(adapter.cancel_document(&source, now).expect("cancel once"));
        assert!(!adapter.cancel_document(&source, now).expect("cancel replay"));

        let other = InferenceReservationReleaseV3 {
            contract_version: InferenceRuntimeContractVersionV3,
            kind: ReservationReleaseKindV3::ReservationRelease,
            reservation_id: reservation.reservation_id,
            request_id: reservation.request_id,
            task_id: TaskId::new("tsk_other").expect("other task"),
            reason_code: "complete".to_owned(),
            requested_at: runtime_v3_timestamp_from_unix_millis(now).expect("release time"),
            extensions: None,
            required_extensions: None,
        };
        let source = encode_inference_reservation_release_v3(&other).expect("encode release");
        assert!(adapter.release_document(&source, now).is_err());
    }

    #[test]
    fn cancelled_execution_accepts_only_a_bound_cancelled_terminal_result() {
        let (adapter, _) = adapter();
        let reservation = reservation();
        let source = encode_inference_reservation_request_v3(&reservation).expect("encode reserve");
        let now = reservation.created_unix_millis().expect("created") + 1;
        adapter.reserve_document(&source, now).expect("reserve");
        let execution = execution(&reservation);
        let source = encode_inference_execution_request_v3(&execution).expect("encode execution");
        let admitted = adapter
            .admit_execution_document(&source, execution.created_unix_millis().expect("created"))
            .expect("admit");

        let cancellation = RuntimeCancellationRequestV3 {
            contract_version: InferenceRuntimeContractVersionV3,
            kind: RuntimeCancellationRequestKindV3::RuntimeCancellationRequestV3,
            reservation_id: reservation.reservation_id,
            request_id: reservation.request_id,
            task_id: reservation.task_id,
            cancellation_request_id: RequestId::new("req_33333333333333333333333333333333")
                .expect("cancel request"),
            reason_code: "owner_cancelled".to_owned(),
            requested_at: runtime_v3_timestamp_from_unix_millis(now + 1).expect("cancel time"),
            extensions: None,
            required_extensions: None,
        };
        let source = encode_runtime_cancellation_request_v3(&cancellation).expect("encode cancel");
        assert!(adapter.cancel_document(&source, now + 1).expect("cancel running execution"));

        let result = cancelled_result(&execution);
        let completed = result.completed_unix_millis().expect("completed");
        let encoded = adapter
            .finalize_execution_result(&admitted, &result, completed)
            .expect("finalize cancellation");
        assert_eq!(decode_inference_execution_result_v3(&encoded).expect("decode result"), result);
        assert_eq!(
            adapter
                .finalize_execution_result(&admitted, &result, completed)
                .expect("exact result replay"),
            encoded
        );
    }
}
