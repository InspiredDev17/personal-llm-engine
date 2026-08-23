//! Authenticated service composition for canonical inference-runtime-v3 documents.
//!
//! This module binds the contract-only reservation adapter to live engine
//! registry, package, backend, and loaded-model state. Configuration remains
//! owner-only; no client message can install an identity or manufacture
//! qualification authority.

use crate::{
    AdmittedRuntimeExecution, ArtifactAuthorityId, EngineService, GenerationEvent, GreedyAdvance,
    LoadedRuntimeModel, LogicalModelId, MAX_RUNTIME_V3_AVAILABILITY_TTL_MILLIS,
    QWEN2_DECODER_ARCHITECTURE_ID, ReservationId, RuntimeBackend, RuntimeIdentityAuthority,
    RuntimeIdentityBinding, RuntimeInstanceId, RuntimeV3ContractAdapter, RuntimeV3PromptEncoding,
    package_hash_hex,
};
use personalai_engine_knowledge_contract::{ArtifactDigest, CacheRuntimeIdentity};
use personalai_engine_protocol::{RuntimeV3Document, ServiceRequestBody, ServiceResponseBody};
use personalai_engine_sampling::SamplingConfig;
use personalai_engine_types::{
    ContextLimit, EngineError, GenerationRequest, MAX_REQUEST_BYTES, MAX_TOKEN_COUNT, ModelId,
    ResourceBudget, SequenceId,
};
use sha2::{Digest, Sha256};
use sovereign_ai_contracts::inference::GenerationUsage;
use sovereign_ai_contracts::inference::{InferenceMessage, InferenceRole};
use sovereign_ai_contracts::inference_runtime_v3::{
    ArtifactDigestAlgorithmV3, InferenceRuntimeContractVersionV3, RuntimeAvailabilityKindV3,
    RuntimeAvailabilityV3, RuntimeIdentityV3, RuntimeModelStateV3, RuntimeModelStatusV3,
    RuntimeStateV3, decode_runtime_cancellation_request_v3, encode_runtime_availability_v3,
    runtime_v3_timestamp_from_unix_millis,
};
use sovereign_ai_contracts::inference_runtime_v3::{
    ExecutionResultKindV3, ExecutionStatusV3, InferenceExecutionResultV3, RuntimeFailureClassV3,
    RuntimeFailureV3, RuntimeResourceUnitV3, RuntimeResourceUsageV3, RuntimeResourceV3,
};
use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

const MAX_RUNTIME_V3_OUTPUT_BYTES: usize = 1_048_576;
const QWEN_CHAT_START: &str = "<|im_start|>";
const QWEN_CHAT_END: &str = "<|im_end|>";

fn execution_deadline_reached(
    observed_unix_millis: u64,
    deadline_unix_millis: u64,
    elapsed: std::time::Duration,
    maximum_execution_millis: u64,
) -> bool {
    observed_unix_millis >= deadline_unix_millis
        || elapsed.as_millis() >= u128::from(maximum_execution_millis)
}

fn accepted_context_limit(
    maximum_input_tokens: u64,
    maximum_output_tokens: u32,
    loaded_context_limit: u32,
) -> Result<ContextLimit, EngineError> {
    let requested_context = maximum_input_tokens
        .checked_add(u64::from(maximum_output_tokens))
        .ok_or_else(|| EngineError::resource_exhaustion("runtime context ceiling overflows"))?;
    if requested_context > u64::from(loaded_context_limit)
        || requested_context > u64::from(MAX_TOKEN_COUNT)
    {
        return Err(EngineError::resource_exhaustion(
            "accepted runtime context ceiling exceeds the loaded model",
        ));
    }
    ContextLimit::new(
        u32::try_from(requested_context)
            .map_err(|_| EngineError::resource_exhaustion("runtime context ceiling exceeds u32"))?,
    )
}

/// Owner-controlled engine instance and reservation ceilings for runtime v3.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RuntimeV3ServiceConfig {
    backend_version: RuntimeInstanceId,
    backend_instance_id: RuntimeInstanceId,
    engine_instance_id: RuntimeInstanceId,
    maximum_active_reservations: usize,
    maximum_inference_memory_bytes: u64,
    availability_ttl_millis: u64,
}

/// Explicit non-production behavior for deterministic contract integration.
/// This type is absent from default builds and never represents model quality.
#[cfg(feature = "development-runtime-v3-simulator")]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RuntimeV3DevelopmentBehavior {
    /// Return one fixed bounded output and synthetic resource accounting.
    FixedSuccess(String),
    /// Return a retryable structured engine failure.
    TransientFailure,
    /// Return a non-retryable structured engine failure.
    TerminalFailure,
    /// Remain active until the request-local cancellation signal is observed.
    WaitUntilCancelled,
}

#[cfg(feature = "development-runtime-v3-simulator")]
impl RuntimeV3DevelopmentBehavior {
    /// Constructs bounded deterministic success output.
    pub fn fixed_success(output: impl Into<String>) -> Result<Self, EngineError> {
        let output = output.into();
        if output.is_empty() || output.len() > 4_096 || output.contains('\0') {
            return Err(EngineError::invalid_input(
                "development runtime v3 output is outside fixture bounds",
            ));
        }
        Ok(Self::FixedSuccess(output))
    }
}

impl RuntimeV3ServiceConfig {
    /// Validates one explicit runtime-v3 service configuration.
    pub fn new(
        backend_version: RuntimeInstanceId,
        backend_instance_id: RuntimeInstanceId,
        engine_instance_id: RuntimeInstanceId,
        maximum_active_reservations: usize,
        maximum_inference_memory_bytes: u64,
        availability_ttl_millis: u64,
    ) -> Result<Self, EngineError> {
        RuntimeV3ContractAdapter::new(maximum_active_reservations, maximum_inference_memory_bytes)?;
        if availability_ttl_millis == 0
            || availability_ttl_millis > MAX_RUNTIME_V3_AVAILABILITY_TTL_MILLIS
        {
            return Err(EngineError::invalid_input(
                "runtime v3 availability TTL is outside canonical bounds",
            ));
        }
        Ok(Self {
            backend_version,
            backend_instance_id,
            engine_instance_id,
            maximum_active_reservations,
            maximum_inference_memory_bytes,
            availability_ttl_millis,
        })
    }

    /// Returns the configured engine instance identity.
    pub const fn engine_instance_id(&self) -> &RuntimeInstanceId {
        &self.engine_instance_id
    }

    /// Returns the owner-configured backend implementation version.
    pub const fn backend_version(&self) -> &RuntimeInstanceId {
        &self.backend_version
    }

    /// Returns the owner-configured backend process or device identity.
    pub const fn backend_instance_id(&self) -> &RuntimeInstanceId {
        &self.backend_instance_id
    }
}

pub(crate) enum PreparedRuntimeV3Execution {
    Replay(Vec<u8>),
    Admitted {
        execution: Box<AdmittedRuntimeExecution>,
        model: Box<LoadedRuntimeModel>,
        pending_cache: Option<Box<crate::PendingCacheAttachment>>,
        cancelled: Arc<AtomicBool>,
    },
}

impl PreparedRuntimeV3Execution {
    pub(crate) fn cancel(&self) {
        if let Self::Admitted { cancelled, .. } = self {
            cancelled.store(true, Ordering::Release);
        }
    }
}

#[derive(Debug)]
struct SealedRuntimeIdentityAuthority {
    identity: RuntimeIdentityBinding,
}

impl RuntimeIdentityAuthority for SealedRuntimeIdentityAuthority {
    fn revalidate(&self, identity: &RuntimeIdentityBinding) -> Result<(), EngineError> {
        if identity != &self.identity {
            return Err(EngineError::authentication_failure(
                "runtime execution identity differs from its admitted authority",
            ));
        }
        Ok(())
    }
}

#[derive(Debug)]
pub(crate) struct RuntimeV3ServiceState {
    config: RuntimeV3ServiceConfig,
    adapter: RuntimeV3ContractAdapter,
    bindings: BTreeMap<String, RuntimeIdentityBinding>,
    next_availability_revision: AtomicU64,
    next_execution_sequence: AtomicU64,
    active_execution_cancellations: Mutex<BTreeMap<ReservationId, Arc<AtomicBool>>>,
    #[cfg(feature = "development-runtime-v3-simulator")]
    development_behavior: Option<RuntimeV3DevelopmentBehavior>,
}

impl RuntimeV3ServiceState {
    pub(crate) fn new(config: RuntimeV3ServiceConfig) -> Result<Self, EngineError> {
        let adapter = RuntimeV3ContractAdapter::new(
            config.maximum_active_reservations,
            config.maximum_inference_memory_bytes,
        )?;
        Ok(Self {
            config,
            adapter,
            bindings: BTreeMap::new(),
            next_availability_revision: AtomicU64::new(0),
            next_execution_sequence: AtomicU64::new(1),
            active_execution_cancellations: Mutex::new(BTreeMap::new()),
            #[cfg(feature = "development-runtime-v3-simulator")]
            development_behavior: None,
        })
    }

    pub(crate) const fn maximum_execution_workers(&self) -> usize {
        self.config.maximum_active_reservations
    }

    #[cfg(test)]
    pub(crate) fn insert_active_execution_for_test(
        &self,
        reservation_id: ReservationId,
    ) -> Result<(), EngineError> {
        let previous = self
            .active_execution_cancellations
            .lock()
            .map_err(|_| {
                EngineError::internal_invariant("runtime execution cancellation lock is poisoned")
            })?
            .insert(reservation_id, Arc::new(AtomicBool::new(false)));
        if previous.is_some() {
            return Err(EngineError::internal_invariant("test execution slot already exists"));
        }
        Ok(())
    }

    #[cfg(test)]
    pub(crate) fn clear_active_executions_for_test(&self) -> Result<(), EngineError> {
        self.active_execution_cancellations
            .lock()
            .map_err(|_| {
                EngineError::internal_invariant("runtime execution cancellation lock is poisoned")
            })?
            .clear();
        Ok(())
    }

    #[cfg(feature = "development-runtime-v3-simulator")]
    pub(crate) fn enable_development_simulator(
        &mut self,
        behavior: RuntimeV3DevelopmentBehavior,
    ) -> Result<(), EngineError> {
        if self.development_behavior.is_some() {
            return Err(EngineError::invalid_input(
                "development runtime v3 simulator is already configured",
            ));
        }
        self.development_behavior = Some(behavior);
        Ok(())
    }

    pub(crate) fn configure_model(
        &mut self,
        service: &EngineService,
        backend: &RuntimeBackend,
        identity: RuntimeIdentityBinding,
    ) -> Result<(), EngineError> {
        let registration = service.registry.resolve(identity.provider_model_id())?;
        if let Some(registry_artifact_id) = registration.registry_artifact_id() {
            if registry_artifact_id != identity.artifact_id().as_str() {
                return Err(EngineError::authentication_failure(
                    "runtime v3 artifact identity differs from signed registry selection",
                ));
            }
        }
        if let Some(canonical) = registration.execution_runtime_identity() {
            if identity.logical_model_id().as_str() != canonical.model_id.as_str()
                || identity.provider_model_id().as_str() != canonical.provider_model_id
                || identity.artifact_id().as_str() != canonical.artifact_id.as_str()
                || package_hash_hex(*identity.package_hash()) != canonical.artifact_digest
                || identity.backend_id().as_str() != canonical.backend_id
                || identity.backend_version().as_str() != canonical.backend_version
                || identity.backend_instance_id().as_str() != canonical.backend_instance_id
                || identity.engine_instance_id().as_str() != canonical.engine_instance_id
            {
                return Err(EngineError::authentication_failure(
                    "runtime v3 owner identity differs from signed Models 2.0 identity",
                ));
            }
        }
        identity.revalidate_engine_owned(
            registration.model_id(),
            registration.expected_hash(),
            &backend.backend_id()?,
            &self.config.backend_version,
            &self.config.backend_instance_id,
            &self.config.engine_instance_id,
        )?;
        let key = identity.provider_model_id().as_str().to_owned();
        if let Some(existing) = self.bindings.get(&key) {
            if existing == &identity {
                return Ok(());
            }
            return Err(EngineError::authentication_failure(
                "runtime v3 model identity cannot be replaced in place",
            ));
        }
        self.bindings.insert(key, identity);
        Ok(())
    }

    pub(crate) fn configure_registered_model(
        &mut self,
        service: &EngineService,
        backend: &RuntimeBackend,
        logical_model_id: LogicalModelId,
        provider_model_id: ModelId,
        artifact_id: ArtifactAuthorityId,
    ) -> Result<(), EngineError> {
        let registration = service.registry.resolve(&provider_model_id)?;
        let identity = RuntimeIdentityBinding::new(
            logical_model_id,
            provider_model_id,
            artifact_id,
            *registration.expected_hash(),
            backend.backend_id()?,
            self.config.backend_version.clone(),
            self.config.backend_instance_id.clone(),
            self.config.engine_instance_id.clone(),
        );
        self.configure_model(service, backend, identity)
    }

    pub(crate) fn has_active_model(
        &self,
        model_id: &ModelId,
        now_unix_millis: u64,
    ) -> Result<bool, EngineError> {
        self.adapter.has_active_model(model_id, now_unix_millis)
    }

    pub(crate) fn validate_cache_model(&self, model_id: &ModelId) -> Result<(), EngineError> {
        self.bindings.get(model_id.as_str()).map(|_| ()).ok_or_else(|| {
            EngineError::authentication_failure(
                "cache model is not bound by owner runtime-v3 configuration",
            )
        })
    }

    pub(crate) fn validate_cache_runtime_identity(
        &self,
        runtime: &CacheRuntimeIdentity,
    ) -> Result<(), EngineError> {
        let identity =
            self.bindings.get(runtime.provider_model_id().as_str()).ok_or_else(|| {
                EngineError::authentication_failure(
                    "cache model is not bound by owner runtime-v3 configuration",
                )
            })?;
        if runtime.logical_model_id().as_str() != identity.logical_model_id().as_str()
            || runtime.artifact_id().as_str() != identity.artifact_id().as_str()
            || runtime.artifact_digest() != ArtifactDigest::Sha384(*identity.package_hash())
            || runtime.backend_id() != identity.backend_id()
            || runtime.backend_version().as_str() != identity.backend_version().as_str()
            || runtime.backend_instance_id().as_str() != identity.backend_instance_id().as_str()
            || runtime.engine_instance_id().as_str() != identity.engine_instance_id().as_str()
        {
            return Err(EngineError::authentication_failure(
                "cache runtime identity differs from owner runtime-v3 configuration",
            ));
        }
        Ok(())
    }

    pub(crate) fn validate_cache_logical_model(
        &self,
        model_id: &ModelId,
        logical_model_id: &personalai_engine_knowledge_contract::KnowledgeIdentity,
    ) -> Result<(), EngineError> {
        let identity = self.bindings.get(model_id.as_str()).ok_or_else(|| {
            EngineError::authentication_failure(
                "cache model is not bound by owner runtime-v3 configuration",
            )
        })?;
        if logical_model_id.as_str() != identity.logical_model_id().as_str() {
            return Err(EngineError::authentication_failure(
                "cache logical model differs from owner runtime-v3 configuration",
            ));
        }
        Ok(())
    }

    #[allow(
        clippy::too_many_arguments,
        reason = "each live authority input remains explicit rather than being hidden in a broad aggregate"
    )]
    pub(crate) fn handle(
        &self,
        service: &EngineService,
        models: &BTreeMap<String, LoadedRuntimeModel>,
        pending_cache_attachments: &mut BTreeMap<
            personalai_engine_types::RequestId,
            crate::PendingCacheAttachment,
        >,
        backend: &RuntimeBackend,
        draining: bool,
        request: &ServiceRequestBody,
        now_unix_millis: u64,
    ) -> Result<Option<ServiceResponseBody>, EngineError> {
        let authority = LiveRuntimeIdentityAuthority {
            service,
            models,
            backend,
            config: &self.config,
            bindings: &self.bindings,
        };
        let response = match request {
            ServiceRequestBody::RuntimeV3Availability => {
                self.adapter.expire(now_unix_millis)?;
                ServiceResponseBody::RuntimeV3Availability(RuntimeV3Document::new(
                    self.availability_document(
                        service,
                        models,
                        backend,
                        draining,
                        now_unix_millis,
                    )?,
                )?)
            }
            ServiceRequestBody::RuntimeV3Reserve(document) => {
                ServiceResponseBody::RuntimeV3Reservation(RuntimeV3Document::new(
                    self.adapter.reserve_document(
                        document.as_bytes(),
                        now_unix_millis,
                        &authority,
                    )?,
                )?)
            }
            ServiceRequestBody::RuntimeV3Cancel(document) => {
                let changed = self.adapter.cancel_document(document.as_bytes(), now_unix_millis)?;
                if changed {
                    let cancellation = contract_result(decode_runtime_cancellation_request_v3(
                        document.as_bytes(),
                    ))?;
                    let reservation_id = ReservationId::new(cancellation.reservation_id)?;
                    if let Some(cancelled) = self
                        .active_execution_cancellations
                        .lock()
                        .map_err(|_| {
                            EngineError::internal_invariant(
                                "runtime execution cancellation lock is poisoned",
                            )
                        })?
                        .get(&reservation_id)
                    {
                        cancelled.store(true, Ordering::Release);
                    }
                }
                ServiceResponseBody::RuntimeV3Cancelled
            }
            ServiceRequestBody::RuntimeV3Release(document) => {
                let _changed =
                    self.adapter.release_document(document.as_bytes(), now_unix_millis)?;
                ServiceResponseBody::RuntimeV3Released
            }
            ServiceRequestBody::RuntimeV3Execute(document) => {
                ServiceResponseBody::RuntimeV3Execution(RuntimeV3Document::new(
                    self.execute_document(
                        document.as_bytes(),
                        now_unix_millis,
                        models,
                        pending_cache_attachments,
                        &authority,
                    )?,
                )?)
            }
            _ => return Ok(None),
        };
        Ok(Some(response))
    }

    fn execute_document(
        &self,
        document: &[u8],
        now_unix_millis: u64,
        models: &BTreeMap<String, LoadedRuntimeModel>,
        pending_cache_attachments: &mut BTreeMap<
            personalai_engine_types::RequestId,
            crate::PendingCacheAttachment,
        >,
        authority: &impl RuntimeIdentityAuthority,
    ) -> Result<Vec<u8>, EngineError> {
        match self.prepare_execution_document(
            document,
            now_unix_millis,
            models,
            pending_cache_attachments,
            authority,
            Arc::new(AtomicBool::new(false)),
        )? {
            PreparedRuntimeV3Execution::Replay(encoded) => Ok(encoded),
            prepared @ PreparedRuntimeV3Execution::Admitted { .. } => {
                self.execute_prepared(prepared, now_unix_millis)
            }
        }
    }

    #[allow(
        clippy::too_many_arguments,
        reason = "continuous preparation receives explicit authority stores plus one request-local cancellation signal"
    )]
    pub(crate) fn prepare_continuous_execution(
        &self,
        document: &[u8],
        now_unix_millis: u64,
        service: &EngineService,
        models: &BTreeMap<String, LoadedRuntimeModel>,
        backend: &RuntimeBackend,
        pending_cache_attachments: &mut BTreeMap<
            personalai_engine_types::RequestId,
            crate::PendingCacheAttachment,
        >,
        cancelled: Arc<AtomicBool>,
    ) -> Result<PreparedRuntimeV3Execution, EngineError> {
        let authority = LiveRuntimeIdentityAuthority {
            service,
            models,
            backend,
            config: &self.config,
            bindings: &self.bindings,
        };
        self.prepare_execution_document(
            document,
            now_unix_millis,
            models,
            pending_cache_attachments,
            &authority,
            cancelled,
        )
    }

    fn prepare_execution_document(
        &self,
        document: &[u8],
        now_unix_millis: u64,
        models: &BTreeMap<String, LoadedRuntimeModel>,
        pending_cache_attachments: &mut BTreeMap<
            personalai_engine_types::RequestId,
            crate::PendingCacheAttachment,
        >,
        authority: &impl RuntimeIdentityAuthority,
        cancelled: Arc<AtomicBool>,
    ) -> Result<PreparedRuntimeV3Execution, EngineError> {
        if let Some(encoded) = self.adapter.replay_execution_document(document)? {
            return Ok(PreparedRuntimeV3Execution::Replay(encoded));
        }
        // Close every service-local rejectable gate before advancing the
        // reservation state machine to `Consumed`. Canonical validation and
        // live authority are repeated by admission immediately afterward.
        let preflight = self.adapter.preflight_execution_document(document, now_unix_millis)?;
        let model = models
            .get(preflight.identity().provider_model_id().as_str())
            .cloned()
            .ok_or_else(|| EngineError::resource_exhaustion("runtime v3 model is not loaded"))?;
        let mut active = self.active_execution_cancellations.lock().map_err(|_| {
            EngineError::internal_invariant("runtime execution cancellation lock is poisoned")
        })?;
        if active.len() >= self.config.maximum_active_reservations {
            return Err(EngineError::resource_exhaustion(
                "runtime execution worker limit is reached",
            ));
        }
        if active.contains_key(preflight.reservation_id()) {
            return Err(EngineError::resource_exhaustion(
                "runtime execution is already preparing or admitted",
            ));
        }
        active.insert(preflight.reservation_id().clone(), Arc::clone(&cancelled));
        drop(active);
        let execution =
            match self.adapter.admit_execution_document(document, now_unix_millis, authority) {
                Ok(execution) => execution,
                Err(error) => {
                    let removed = self
                        .active_execution_cancellations
                        .lock()
                        .map_err(|_| {
                            EngineError::internal_invariant(
                                "runtime execution cancellation lock is poisoned",
                            )
                        })?
                        .remove(preflight.reservation_id());
                    if removed.is_none() {
                        return Err(EngineError::internal_invariant(
                            "failed runtime admission lost its preflight cancellation slot",
                        ));
                    }
                    return Err(error);
                }
            };
        // Consume the exact request-bound attachment only after every admission
        // step that can reject without creating an active execution. This keeps
        // a capacity/duplicate rejection from silently discarding cache authority.
        let pending_cache = pending_cache_attachments.remove(&execution.engine_request_id());
        Ok(PreparedRuntimeV3Execution::Admitted {
            execution: Box::new(execution),
            model: Box::new(model),
            pending_cache: pending_cache.map(Box::new),
            cancelled,
        })
    }

    pub(crate) fn cancel_prepared_execution(
        &self,
        prepared: PreparedRuntimeV3Execution,
        now_unix_millis: u64,
    ) -> Result<Vec<u8>, EngineError> {
        prepared.cancel();
        self.execute_prepared(prepared, now_unix_millis)
    }

    pub(crate) fn execute_prepared(
        &self,
        prepared: PreparedRuntimeV3Execution,
        now_unix_millis: u64,
    ) -> Result<Vec<u8>, EngineError> {
        let PreparedRuntimeV3Execution::Admitted { execution, model, pending_cache, cancelled } =
            prepared
        else {
            return Err(EngineError::internal_invariant(
                "replayed runtime execution entered the execution worker",
            ));
        };
        let authority = SealedRuntimeIdentityAuthority { identity: execution.identity().clone() };
        #[cfg(feature = "development-runtime-v3-simulator")]
        let result = if let Some(behavior) = self.development_behavior.as_ref() {
            development_result(&execution, behavior, now_unix_millis, cancelled.as_ref())
        } else {
            self.execute_loaded_model(
                &execution,
                model.as_ref(),
                pending_cache.map(|attachment| *attachment),
                now_unix_millis,
                cancelled.as_ref(),
            )
        };
        #[cfg(not(feature = "development-runtime-v3-simulator"))]
        let result = self.execute_loaded_model(
            &execution,
            model.as_ref(),
            pending_cache.map(|attachment| *attachment),
            now_unix_millis,
            cancelled.as_ref(),
        );
        let result = match result {
            Ok(result) => result,
            Err(error) => build_runtime_result(
                &execution,
                terminal_from_engine_error(&error, now_unix_millis, execution.request())?,
                String::new(),
                now_unix_millis,
                now_unix_millis,
                std::time::Duration::ZERO,
            )?,
        };
        let completed = contract_result(result.completed_unix_millis())?;
        let finalized =
            self.adapter.finalize_execution_result(&execution, &result, completed, &authority);
        let removed = self
            .active_execution_cancellations
            .lock()
            .map_err(|_| {
                EngineError::internal_invariant("runtime execution cancellation lock is poisoned")
            })?
            .remove(execution.reservation_id());
        if removed.is_none() {
            return Err(EngineError::internal_invariant(
                "runtime execution cancellation signal disappeared",
            ));
        }
        finalized
    }

    #[allow(
        clippy::too_many_lines,
        reason = "one bounded loop keeps pre/post-step cancellation, deadline, output, and terminal accounting ordering visible"
    )]
    fn execute_loaded_model(
        &self,
        admitted: &AdmittedRuntimeExecution,
        loaded: &LoadedRuntimeModel,
        pending_cache: Option<crate::PendingCacheAttachment>,
        now_unix_millis: u64,
        request_cancelled: &AtomicBool,
    ) -> Result<InferenceExecutionResultV3, EngineError> {
        let request = admitted.request();
        if request_cancelled.load(Ordering::Acquire)
            || self.adapter.execution_is_cancelled(admitted)?
        {
            request_cancelled.store(true, Ordering::Release);
            return build_runtime_result(
                admitted,
                RuntimeExecutionTerminal::Cancelled,
                String::new(),
                now_unix_millis,
                crate::runtime_v3_now_unix_millis()?,
                std::time::Duration::ZERO,
            );
        }
        let (prompt, prompt_encoding) =
            compose_runtime_prompt(&request.messages, loaded.architecture_id().as_str())?;
        let sampling = runtime_sampling(&request.parameters)?;
        let ceiling = admitted.resource_ceiling();
        let context_limit = accepted_context_limit(
            ceiling.maximum_input_tokens(),
            request.parameters.max_output_tokens,
            loaded.context_limit().get(),
        )?;
        let budget = ResourceBudget::new(
            ceiling.maximum_inference_memory_bytes(),
            ceiling.maximum_inference_memory_bytes(),
            1,
        )?;
        let generation_request = GenerationRequest::new(
            admitted.engine_request_id(),
            admitted.identity().provider_model_id().clone(),
            prompt,
            request.parameters.max_output_tokens,
            context_limit,
            budget,
        )?;
        let sequence = self
            .next_execution_sequence
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |current| current.checked_add(1))
            .map_err(|_| EngineError::resource_exhaustion("runtime sequence ID counter wraps"))?;
        let sequence = SequenceId::new(sequence)?;
        let mut generation = loaded.start_runtime_v3_generation(
            &generation_request,
            sequence,
            sampling,
            prompt_encoding,
            pending_cache,
        )?;
        let started = Instant::now();
        let mut output = String::new();
        let terminal = loop {
            let observed = crate::runtime_v3_now_unix_millis()?;
            if request_cancelled.load(Ordering::Acquire)
                || self.adapter.execution_is_cancelled(admitted)?
            {
                request_cancelled.store(true, Ordering::Release);
                break RuntimeExecutionTerminal::Cancelled;
            }
            if execution_deadline_reached(
                observed,
                contract_result(request.deadline_unix_millis())?,
                started.elapsed(),
                ceiling.maximum_execution_millis(),
            ) {
                request_cancelled.store(true, Ordering::Release);
                break RuntimeExecutionTerminal::DeadlineExceeded;
            }
            let advance = generation.advance(request_cancelled);
            let observed_after_step = crate::runtime_v3_now_unix_millis()?;
            if request_cancelled.load(Ordering::Acquire)
                || self.adapter.execution_is_cancelled(admitted)?
            {
                request_cancelled.store(true, Ordering::Release);
                break RuntimeExecutionTerminal::Cancelled;
            }
            if execution_deadline_reached(
                observed_after_step,
                contract_result(request.deadline_unix_millis())?,
                started.elapsed(),
                ceiling.maximum_execution_millis(),
            ) {
                request_cancelled.store(true, Ordering::Release);
                break RuntimeExecutionTerminal::DeadlineExceeded;
            }
            match advance {
                Ok(
                    GreedyAdvance::Progressed
                    | GreedyAdvance::Event(GenerationEvent::Started { .. }),
                ) => {}
                Ok(GreedyAdvance::Event(GenerationEvent::Token { text, .. })) => {
                    let next = output.len().checked_add(text.as_str().len()).ok_or_else(|| {
                        EngineError::resource_exhaustion("runtime v3 output length overflows")
                    })?;
                    if next > MAX_RUNTIME_V3_OUTPUT_BYTES {
                        break RuntimeExecutionTerminal::Failed(
                            RuntimeFailureClassV3::ResourceUnavailable,
                            "output_size_exceeded",
                            false,
                        );
                    }
                    output.push_str(text.as_str());
                }
                Ok(GreedyAdvance::Event(GenerationEvent::Completed { usage, .. })) => {
                    break RuntimeExecutionTerminal::Succeeded(usage);
                }
                Ok(GreedyAdvance::Event(GenerationEvent::Failed { error, .. })) | Err(error) => {
                    break terminal_from_engine_error(&error, observed, request)?;
                }
                Ok(GreedyAdvance::Event(_)) => {
                    return Err(EngineError::internal_invariant(
                        "runtime v3 generation emitted an unknown event",
                    ));
                }
            }
        };
        let observed_completion = crate::runtime_v3_now_unix_millis()?;
        build_runtime_result(
            admitted,
            terminal,
            output,
            now_unix_millis,
            observed_completion,
            started.elapsed(),
        )
    }

    fn availability_document(
        &self,
        service: &EngineService,
        models: &BTreeMap<String, LoadedRuntimeModel>,
        backend: &RuntimeBackend,
        draining: bool,
        now_unix_millis: u64,
    ) -> Result<Vec<u8>, EngineError> {
        let statistics = self.adapter.reservation_statistics()?;
        let available_memory_bytes = self
            .adapter
            .maximum_memory_bytes()
            .checked_sub(statistics.reserved_memory_bytes)
            .ok_or_else(|| {
                EngineError::internal_invariant("runtime reservation memory accounting underflowed")
            })?;
        let active_requests = u16::try_from(statistics.active)
            .map_err(|_| EngineError::internal_invariant("runtime active count exceeds u16"))?;
        let maximum_concurrent_requests = u16::try_from(self.adapter.maximum_active())
            .map_err(|_| EngineError::internal_invariant("runtime active ceiling exceeds u16"))?;
        let mut statuses = Vec::new();
        statuses.try_reserve_exact(self.bindings.len()).map_err(|_| {
            EngineError::resource_exhaustion("runtime availability model allocation failed")
        })?;
        for identity in self.bindings.values() {
            let registration = service.registry.resolve(identity.provider_model_id())?;
            identity.revalidate_engine_owned(
                registration.model_id(),
                registration.expected_hash(),
                &backend.backend_id()?,
                &self.config.backend_version,
                &self.config.backend_instance_id,
                &self.config.engine_instance_id,
            )?;
            let loaded = models.contains_key(identity.provider_model_id().as_str());
            let state = if draining {
                RuntimeModelStateV3::Draining
            } else if loaded {
                RuntimeModelStateV3::Loaded
            } else {
                RuntimeModelStateV3::Loadable
            };
            statuses.push(RuntimeModelStatusV3 {
                identity: contract_identity(identity)?,
                state,
                available_memory_bytes,
                active_requests,
                maximum_concurrent_requests,
            });
        }
        let state = if draining {
            RuntimeStateV3::Draining
        } else if self.bindings.is_empty() {
            RuntimeStateV3::Unavailable
        } else if models.keys().any(|model| self.bindings.contains_key(model))
            && statistics.active < self.adapter.maximum_active()
            && available_memory_bytes > 0
        {
            RuntimeStateV3::Available
        } else {
            RuntimeStateV3::Constrained
        };
        // Wall-clock-derived revisions survive ordinary process restart without
        // requiring a second mutable journal. If the host clock rolls back,
        // consumers retaining a newer revision reject this status until the
        // clock catches up, so availability fails closed.
        let revision = loop {
            let current = self.next_availability_revision.load(Ordering::Acquire);
            let revision = current.max(now_unix_millis.max(1));
            let next = revision.checked_add(1).ok_or_else(|| {
                EngineError::internal_invariant("availability revision overflowed")
            })?;
            if self
                .next_availability_revision
                .compare_exchange(current, next, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                break revision;
            }
        };
        let expires = now_unix_millis
            .checked_add(self.config.availability_ttl_millis)
            .ok_or_else(|| EngineError::resource_exhaustion("availability expiry overflows"))?;
        let availability = RuntimeAvailabilityV3 {
            contract_version: InferenceRuntimeContractVersionV3,
            kind: RuntimeAvailabilityKindV3::RuntimeAvailabilityV3,
            engine_instance_id: self.config.engine_instance_id.as_str().to_owned(),
            status_revision: revision,
            state,
            models: statuses,
            observed_at: contract_result(runtime_v3_timestamp_from_unix_millis(now_unix_millis))?,
            expires_at: contract_result(runtime_v3_timestamp_from_unix_millis(expires))?,
            extensions: None,
            required_extensions: None,
        };
        contract_result(encode_runtime_availability_v3(&availability))
    }
}

#[derive(Debug)]
enum RuntimeExecutionTerminal {
    Succeeded(personalai_engine_types::GenerationUsage),
    Failed(RuntimeFailureClassV3, &'static str, bool),
    Cancelled,
    DeadlineExceeded,
}

#[allow(
    clippy::cast_possible_truncation,
    reason = "the following round-trip comparison rejects an f32 conversion that would expand caller sampling authority"
)]
fn runtime_sampling(
    parameters: &sovereign_ai_contracts::inference::GenerationParameters,
) -> Result<SamplingConfig, EngineError> {
    if parameters.stop_sequences.as_ref().is_some_and(|values| !values.is_empty()) {
        return Err(EngineError::not_implemented(
            "runtime v3 text stop sequences are not implemented by the native tokenizer path",
        ));
    }
    let temperature = parameters.temperature as f32;
    let top_p = parameters.top_p as f32;
    if f64::from(temperature) > parameters.temperature && parameters.temperature > 0.0 {
        return Err(EngineError::invalid_field(
            "temperature",
            "cannot be represented without expanding sampling authority",
        ));
    }
    if f64::from(top_p) > parameters.top_p {
        return Err(EngineError::invalid_field(
            "top_p",
            "cannot be represented without expanding sampling authority",
        ));
    }
    let mut sampling = SamplingConfig::new(temperature)?.with_top_p(top_p)?;
    if let Some(seed) = parameters.seed {
        sampling = sampling.with_seed(u64::from(seed));
    }
    sampling.with_maximum_generated_tokens(parameters.max_output_tokens)
}

fn compose_runtime_prompt(
    messages: &[InferenceMessage],
    architecture_id: &str,
) -> Result<(String, RuntimeV3PromptEncoding), EngineError> {
    if architecture_id != QWEN2_DECODER_ARCHITECTURE_ID {
        let [message] = messages else {
            return Err(EngineError::not_implemented(
                "runtime v3 multi-message formatting is unavailable for this architecture",
            ));
        };
        if message.role != InferenceRole::User || message.name.is_some() {
            return Err(EngineError::not_implemented(
                "runtime v3 role formatting is unavailable for this architecture",
            ));
        }
        validate_runtime_message_content(&message.content)?;
        return Ok((message.content.clone(), RuntimeV3PromptEncoding::PlainUserMessage));
    }

    let mut prompt = String::new();
    prompt
        .try_reserve(messages.iter().try_fold(32_usize, |total, message| {
            total
                .checked_add(message.content.len())
                .and_then(|value| value.checked_add(32))
                .ok_or_else(|| EngineError::resource_exhaustion("runtime prompt size overflows"))
        })?)
        .map_err(|_| EngineError::resource_exhaustion("runtime prompt allocation failed"))?;
    for message in messages {
        if message.name.is_some() {
            return Err(EngineError::not_implemented(
                "named runtime v3 messages have no reviewed Qwen2.5 chat mapping",
            ));
        }
        validate_runtime_message_content(&message.content)?;
        if message.content.contains(QWEN_CHAT_START) || message.content.contains(QWEN_CHAT_END) {
            return Err(EngineError::invalid_field(
                "messages.content",
                "contains an engine-owned Qwen chat delimiter",
            ));
        }
        let role = match message.role {
            InferenceRole::System => "system",
            InferenceRole::User => "user",
            InferenceRole::Assistant => "assistant",
            InferenceRole::Tool => "tool",
        };
        prompt.push_str(QWEN_CHAT_START);
        prompt.push_str(role);
        prompt.push('\n');
        prompt.push_str(&message.content);
        prompt.push_str(QWEN_CHAT_END);
        prompt.push('\n');
        if prompt.len() > MAX_REQUEST_BYTES {
            return Err(EngineError::resource_exhaustion(
                "runtime prompt exceeds the native request bound",
            ));
        }
    }
    prompt.push_str(QWEN_CHAT_START);
    prompt.push_str("assistant\n");
    if prompt.len() > MAX_REQUEST_BYTES {
        return Err(EngineError::resource_exhaustion(
            "runtime prompt exceeds the native request bound",
        ));
    }
    Ok((prompt, RuntimeV3PromptEncoding::Qwen25Chat))
}

fn validate_runtime_message_content(content: &str) -> Result<(), EngineError> {
    if content.is_empty() || content.contains('\0') {
        return Err(EngineError::invalid_field(
            "messages.content",
            "must be nonempty UTF-8 without NUL",
        ));
    }
    Ok(())
}

fn terminal_from_engine_error(
    error: &EngineError,
    observed_unix_millis: u64,
    request: &sovereign_ai_contracts::inference_runtime_v3::InferenceExecutionRequestV3,
) -> Result<RuntimeExecutionTerminal, EngineError> {
    if matches!(error, EngineError::Cancellation(_)) {
        return if observed_unix_millis >= contract_result(request.deadline_unix_millis())? {
            Ok(RuntimeExecutionTerminal::DeadlineExceeded)
        } else {
            Ok(RuntimeExecutionTerminal::Cancelled)
        };
    }
    Ok(match error {
        EngineError::ResourceExhaustion(_) => RuntimeExecutionTerminal::Failed(
            RuntimeFailureClassV3::ResourceUnavailable,
            "resource_unavailable",
            true,
        ),
        EngineError::BackendFailure(_) => RuntimeExecutionTerminal::Failed(
            RuntimeFailureClassV3::BackendUnavailable,
            "backend_failure",
            true,
        ),
        EngineError::AuthenticationFailure(_)
        | EngineError::SignatureFailure(_)
        | EngineError::CorruptModelPackage(_) => RuntimeExecutionTerminal::Failed(
            RuntimeFailureClassV3::InvalidIdentity,
            "runtime_identity_invalid",
            false,
        ),
        EngineError::UnsupportedArchitecture(_) | EngineError::UnsupportedTokenizer(_) => {
            RuntimeExecutionTerminal::Failed(
                RuntimeFailureClassV3::ModelUnavailable,
                "model_execution_unsupported",
                false,
            )
        }
        EngineError::InvalidInput(_) | EngineError::ProtocolViolation(_) => {
            RuntimeExecutionTerminal::Failed(
                RuntimeFailureClassV3::InvalidRequest,
                "runtime_request_invalid",
                false,
            )
        }
        EngineError::NotImplemented(_) => RuntimeExecutionTerminal::Failed(
            RuntimeFailureClassV3::BackendUnavailable,
            "runtime_path_unavailable",
            false,
        ),
        EngineError::InternalInvariantViolation(_) => RuntimeExecutionTerminal::Failed(
            RuntimeFailureClassV3::InternalFailure,
            "engine_internal_failure",
            false,
        ),
        EngineError::Cancellation(_) => unreachable!("cancellation returned above"),
        _ => RuntimeExecutionTerminal::Failed(
            RuntimeFailureClassV3::InternalFailure,
            "engine_unclassified_failure",
            false,
        ),
    })
}

fn build_runtime_result(
    admitted: &AdmittedRuntimeExecution,
    terminal: RuntimeExecutionTerminal,
    output: String,
    started_unix_millis: u64,
    observed_completion_unix_millis: u64,
    elapsed: std::time::Duration,
) -> Result<InferenceExecutionResultV3, EngineError> {
    let request = admitted.request();
    let deadline = contract_result(request.deadline_unix_millis())?;
    let minimum_completion = started_unix_millis
        .checked_add(1)
        .ok_or_else(|| EngineError::resource_exhaustion("runtime completion time overflows"))?;
    let completed_unix_millis =
        observed_completion_unix_millis.max(minimum_completion).min(deadline);
    if completed_unix_millis <= started_unix_millis {
        return Err(EngineError::invalid_input(
            "runtime deadline does not permit a canonical terminal interval",
        ));
    }
    let wall_millis = u64::try_from(elapsed.as_millis())
        .unwrap_or(u64::MAX)
        .min(admitted.resource_ceiling().maximum_execution_millis());
    let resource_usage = vec![RuntimeResourceUsageV3 {
        resource: RuntimeResourceV3::WallTime,
        units: wall_millis,
        unit: RuntimeResourceUnitV3::Milliseconds,
    }];
    let (status, usage, output, output_digest, failure) = match terminal {
        RuntimeExecutionTerminal::Succeeded(usage) => {
            let digest = sha256_hex(output.as_bytes())?;
            (
                ExecutionStatusV3::Succeeded,
                GenerationUsage {
                    input_tokens: u64::from(usage.input_tokens()),
                    output_tokens: u64::from(usage.output_tokens()),
                    total_tokens: u64::from(usage.total_tokens()),
                },
                Some(output),
                Some(digest),
                None,
            )
        }
        RuntimeExecutionTerminal::Failed(class, code, retryable) => (
            ExecutionStatusV3::Failed,
            GenerationUsage { input_tokens: 0, output_tokens: 0, total_tokens: 0 },
            None,
            None,
            Some(RuntimeFailureV3 { class, code: code.to_owned(), retryable }),
        ),
        RuntimeExecutionTerminal::Cancelled => (
            ExecutionStatusV3::Cancelled,
            GenerationUsage { input_tokens: 0, output_tokens: 0, total_tokens: 0 },
            None,
            None,
            Some(RuntimeFailureV3 {
                class: RuntimeFailureClassV3::Cancelled,
                code: "owner_cancelled".to_owned(),
                retryable: false,
            }),
        ),
        RuntimeExecutionTerminal::DeadlineExceeded => (
            ExecutionStatusV3::DeadlineExceeded,
            GenerationUsage { input_tokens: 0, output_tokens: 0, total_tokens: 0 },
            None,
            None,
            Some(RuntimeFailureV3 {
                class: RuntimeFailureClassV3::DeadlineExceeded,
                code: "execution_deadline_exceeded".to_owned(),
                retryable: false,
            }),
        ),
    };
    Ok(InferenceExecutionResultV3 {
        contract_version: InferenceRuntimeContractVersionV3,
        kind: ExecutionResultKindV3::ExecutionResult,
        reservation_id: request.reservation_id.clone(),
        request_id: request.request_id.clone(),
        task_id: request.task_id.clone(),
        identity: request.identity.clone(),
        status,
        started_at: contract_result(runtime_v3_timestamp_from_unix_millis(started_unix_millis))?,
        completed_at: contract_result(runtime_v3_timestamp_from_unix_millis(
            completed_unix_millis,
        ))?,
        usage,
        resource_usage,
        output,
        output_digest,
        failure,
        extensions: None,
        required_extensions: None,
    })
}

fn sha256_hex(bytes: &[u8]) -> Result<String, EngineError> {
    let mut digest = String::with_capacity(64);
    for byte in Sha256::digest(bytes) {
        write!(&mut digest, "{byte:02x}")
            .map_err(|_| EngineError::internal_invariant("output digest formatting failed"))?;
    }
    Ok(digest)
}

#[cfg(feature = "development-runtime-v3-simulator")]
#[allow(
    clippy::too_many_lines,
    reason = "the explicit closed simulator behaviors share one canonical result constructor for reviewability"
)]
fn development_result(
    admitted: &AdmittedRuntimeExecution,
    behavior: &RuntimeV3DevelopmentBehavior,
    now_unix_millis: u64,
    cancelled: &AtomicBool,
) -> Result<InferenceExecutionResultV3, EngineError> {
    let request = admitted.request();
    if matches!(behavior, RuntimeV3DevelopmentBehavior::WaitUntilCancelled) {
        let started = Instant::now();
        let deadline = contract_result(request.deadline_unix_millis())?;
        loop {
            if cancelled.load(Ordering::Acquire) {
                return build_runtime_result(
                    admitted,
                    RuntimeExecutionTerminal::Cancelled,
                    String::new(),
                    now_unix_millis,
                    crate::runtime_v3_now_unix_millis()?,
                    started.elapsed(),
                );
            }
            let observed = crate::runtime_v3_now_unix_millis()?;
            if execution_deadline_reached(
                observed,
                deadline,
                started.elapsed(),
                admitted.resource_ceiling().maximum_execution_millis(),
            ) {
                return build_runtime_result(
                    admitted,
                    RuntimeExecutionTerminal::DeadlineExceeded,
                    String::new(),
                    now_unix_millis,
                    observed,
                    started.elapsed(),
                );
            }
            std::thread::yield_now();
        }
    }
    let nominal_completion = now_unix_millis.checked_add(1).ok_or_else(|| {
        EngineError::resource_exhaustion("development completion time overflowed")
    })?;
    let deadline_unix_millis = contract_result(request.deadline_unix_millis())?;
    let deadline_exceeded = nominal_completion > deadline_unix_millis;
    let completed_unix_millis = nominal_completion.min(deadline_unix_millis);
    let started_at = contract_result(runtime_v3_timestamp_from_unix_millis(now_unix_millis))?;
    let completed_at =
        contract_result(runtime_v3_timestamp_from_unix_millis(completed_unix_millis))?;
    let resources = vec![RuntimeResourceUsageV3 {
        resource: RuntimeResourceV3::WallTime,
        units: 1,
        unit: RuntimeResourceUnitV3::Milliseconds,
    }];
    let (status, usage, output, output_digest, failure) = if deadline_exceeded {
        (
            ExecutionStatusV3::DeadlineExceeded,
            GenerationUsage { input_tokens: 0, output_tokens: 0, total_tokens: 0 },
            None,
            None,
            Some(RuntimeFailureV3 {
                class: RuntimeFailureClassV3::DeadlineExceeded,
                code: "development_deadline_exceeded".to_owned(),
                retryable: false,
            }),
        )
    } else {
        match behavior {
            RuntimeV3DevelopmentBehavior::FixedSuccess(output) => {
                let mut digest = String::with_capacity(64);
                for byte in Sha256::digest(output.as_bytes()) {
                    write!(&mut digest, "{byte:02x}").map_err(|_| {
                        EngineError::internal_invariant(
                            "development output digest formatting failed",
                        )
                    })?;
                }
                (
                    ExecutionStatusV3::Succeeded,
                    GenerationUsage { input_tokens: 1, output_tokens: 1, total_tokens: 2 },
                    Some(output.clone()),
                    Some(digest),
                    None,
                )
            }
            RuntimeV3DevelopmentBehavior::TransientFailure => (
                ExecutionStatusV3::Failed,
                GenerationUsage { input_tokens: 0, output_tokens: 0, total_tokens: 0 },
                None,
                None,
                Some(RuntimeFailureV3 {
                    class: RuntimeFailureClassV3::InternalFailure,
                    code: "development_transient_failure".to_owned(),
                    retryable: true,
                }),
            ),
            RuntimeV3DevelopmentBehavior::TerminalFailure => (
                ExecutionStatusV3::Failed,
                GenerationUsage { input_tokens: 0, output_tokens: 0, total_tokens: 0 },
                None,
                None,
                Some(RuntimeFailureV3 {
                    class: RuntimeFailureClassV3::InternalFailure,
                    code: "development_terminal_failure".to_owned(),
                    retryable: false,
                }),
            ),
            RuntimeV3DevelopmentBehavior::WaitUntilCancelled => unreachable!(
                "wait-until-cancelled behavior returns before deterministic result assembly"
            ),
        }
    };
    Ok(InferenceExecutionResultV3 {
        contract_version: InferenceRuntimeContractVersionV3,
        kind: ExecutionResultKindV3::ExecutionResult,
        reservation_id: request.reservation_id.clone(),
        request_id: request.request_id.clone(),
        task_id: request.task_id.clone(),
        identity: request.identity.clone(),
        status,
        started_at,
        completed_at,
        usage,
        resource_usage: resources,
        output,
        output_digest,
        failure,
        extensions: None,
        required_extensions: None,
    })
}

struct LiveRuntimeIdentityAuthority<'a> {
    service: &'a EngineService,
    models: &'a BTreeMap<String, LoadedRuntimeModel>,
    backend: &'a RuntimeBackend,
    config: &'a RuntimeV3ServiceConfig,
    bindings: &'a BTreeMap<String, RuntimeIdentityBinding>,
}

impl RuntimeIdentityAuthority for LiveRuntimeIdentityAuthority<'_> {
    fn revalidate(&self, identity: &RuntimeIdentityBinding) -> Result<(), EngineError> {
        let configured =
            self.bindings.get(identity.provider_model_id().as_str()).ok_or_else(|| {
                EngineError::authentication_failure("runtime v3 model identity is not configured")
            })?;
        if configured != identity {
            return Err(EngineError::authentication_failure(
                "runtime v3 model identity changed after configuration",
            ));
        }
        let registration = self.service.registry.resolve(identity.provider_model_id())?;
        let loaded = self
            .models
            .get(identity.provider_model_id().as_str())
            .ok_or_else(|| EngineError::resource_exhaustion("runtime v3 model is not loaded"))?;
        identity.revalidate_engine_owned(
            registration.model_id(),
            registration.expected_hash(),
            &loaded.backend_id()?,
            &self.config.backend_version,
            &self.config.backend_instance_id,
            &self.config.engine_instance_id,
        )?;
        if loaded.backend_id()? != self.backend.backend_id()? {
            return Err(EngineError::authentication_failure(
                "loaded model backend differs from selected runtime backend",
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
#[allow(
    clippy::items_after_test_module,
    reason = "focused private helper tests remain next to the live authority implementation they exercise"
)]
mod tests {
    use super::{accepted_context_limit, execution_deadline_reached};
    use personalai_engine_types::EngineError;
    use std::time::Duration;

    #[test]
    fn accepted_context_limit_never_silently_narrows_the_admitted_budget() {
        assert_eq!(accepted_context_limit(96, 32, 128).expect("exact accepted context").get(), 128);
        assert!(matches!(
            accepted_context_limit(97, 32, 128),
            Err(EngineError::ResourceExhaustion(_))
        ));
        assert!(matches!(
            accepted_context_limit(u64::MAX, 1, u32::MAX),
            Err(EngineError::ResourceExhaustion(_))
        ));
    }

    #[test]
    fn deadline_is_resampled_after_a_decoder_step_before_output_is_retained() {
        assert!(!execution_deadline_reached(9_999, 10_000, Duration::from_millis(99), 100,));
        assert!(execution_deadline_reached(10_000, 10_000, Duration::from_millis(99), 100,));
        assert!(execution_deadline_reached(9_999, 10_000, Duration::from_millis(100), 100,));
    }
}

fn contract_identity(identity: &RuntimeIdentityBinding) -> Result<RuntimeIdentityV3, EngineError> {
    let mut artifact_digest = String::with_capacity(96);
    for byte in identity.package_hash().as_bytes() {
        write!(&mut artifact_digest, "{byte:02x}")
            .map_err(|_| EngineError::internal_invariant("artifact digest formatting failed"))?;
    }
    Ok(RuntimeIdentityV3 {
        model_id: sovereign_ai_contracts::identity::ModelId::new(
            identity.logical_model_id().as_str(),
        )
        .map_err(|error| EngineError::internal_invariant(format!("logical model ID: {error}")))?,
        provider_model_id: identity.provider_model_id().as_str().to_owned(),
        artifact_id: sovereign_ai_contracts::identity::ModelArtifactId::new(
            identity.artifact_id().as_str(),
        )
        .map_err(|error| EngineError::internal_invariant(format!("artifact ID: {error}")))?,
        artifact_digest_algorithm: ArtifactDigestAlgorithmV3::Sha384,
        artifact_digest,
        backend_id: identity.backend_id().as_str().to_owned(),
        backend_version: identity.backend_version().as_str().to_owned(),
        backend_instance_id: identity.backend_instance_id().as_str().to_owned(),
        engine_instance_id: identity.engine_instance_id().as_str().to_owned(),
    })
}

fn contract_result<T>(result: sovereign_ai_contracts::Result<T>) -> Result<T, EngineError> {
    result.map_err(|error| EngineError::invalid_input(format!("runtime v3 contract: {error}")))
}
