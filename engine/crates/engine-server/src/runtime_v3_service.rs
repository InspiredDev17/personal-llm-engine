//! Authenticated service composition for canonical inference-runtime-v3 documents.
//!
//! This module binds the contract-only reservation adapter to live engine
//! registry, package, backend, and loaded-model state. Configuration remains
//! owner-only; no client message can install an identity or manufacture
//! qualification authority.

#[cfg(feature = "development-runtime-v3-simulator")]
use crate::AdmittedRuntimeExecution;
use crate::{
    ArtifactAuthorityId, EngineService, LoadedRuntimeModel, LogicalModelId,
    MAX_RUNTIME_V3_AVAILABILITY_TTL_MILLIS, RuntimeBackend, RuntimeIdentityAuthority,
    RuntimeIdentityBinding, RuntimeInstanceId, RuntimeV3ContractAdapter,
};
use personalai_engine_knowledge_contract::{ArtifactDigest, CacheRuntimeIdentity};
use personalai_engine_protocol::{RuntimeV3Document, ServiceRequestBody, ServiceResponseBody};
use personalai_engine_types::EngineError;
use personalai_engine_types::ModelId;
#[cfg(feature = "development-runtime-v3-simulator")]
use sha2::{Digest, Sha256};
#[cfg(feature = "development-runtime-v3-simulator")]
use sovereign_ai_contracts::inference::GenerationUsage;
use sovereign_ai_contracts::inference_runtime_v3::{
    ArtifactDigestAlgorithmV3, InferenceRuntimeContractVersionV3, RuntimeAvailabilityKindV3,
    RuntimeAvailabilityV3, RuntimeIdentityV3, RuntimeModelStateV3, RuntimeModelStatusV3,
    RuntimeStateV3, encode_runtime_availability_v3, runtime_v3_timestamp_from_unix_millis,
};
#[cfg(feature = "development-runtime-v3-simulator")]
use sovereign_ai_contracts::inference_runtime_v3::{
    ExecutionResultKindV3, ExecutionStatusV3, InferenceExecutionResultV3, RuntimeFailureClassV3,
    RuntimeFailureV3, RuntimeResourceUnitV3, RuntimeResourceUsageV3, RuntimeResourceV3,
};
use std::collections::BTreeMap;
use std::fmt::Write as _;

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
}

#[derive(Debug)]
pub(crate) struct RuntimeV3ServiceState {
    config: RuntimeV3ServiceConfig,
    adapter: RuntimeV3ContractAdapter,
    bindings: BTreeMap<String, RuntimeIdentityBinding>,
    next_availability_revision: u64,
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
            next_availability_revision: 0,
            #[cfg(feature = "development-runtime-v3-simulator")]
            development_behavior: None,
        })
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

    pub(crate) fn handle(
        &mut self,
        service: &EngineService,
        models: &BTreeMap<String, LoadedRuntimeModel>,
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
                let _changed =
                    self.adapter.cancel_document(document.as_bytes(), now_unix_millis)?;
                ServiceResponseBody::RuntimeV3Cancelled
            }
            ServiceRequestBody::RuntimeV3Release(document) => {
                let _changed =
                    self.adapter.release_document(document.as_bytes(), now_unix_millis)?;
                ServiceResponseBody::RuntimeV3Released
            }
            ServiceRequestBody::RuntimeV3Execute(document) => {
                ServiceResponseBody::RuntimeV3Execution(RuntimeV3Document::new(
                    self.execute_document(document.as_bytes(), now_unix_millis, &authority)?,
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
        authority: &impl RuntimeIdentityAuthority,
    ) -> Result<Vec<u8>, EngineError> {
        #[cfg(not(feature = "development-runtime-v3-simulator"))]
        {
            let _ = (document, now_unix_millis, authority);
            Err(EngineError::not_implemented("runtime v3 execution backend is not configured"))
        }
        #[cfg(feature = "development-runtime-v3-simulator")]
        {
            let behavior = self.development_behavior.as_ref().ok_or_else(|| {
                EngineError::not_implemented("runtime v3 execution backend is not configured")
            })?;
            if let Some(encoded) = self.adapter.replay_execution_document(document)? {
                return Ok(encoded);
            }
            let admitted =
                self.adapter.admit_execution_document(document, now_unix_millis, authority)?;
            let result = development_result(&admitted, behavior, now_unix_millis)?;
            self.adapter.finalize_execution_result(
                &admitted,
                &result,
                now_unix_millis.saturating_add(1),
                authority,
            )
        }
    }

    fn availability_document(
        &mut self,
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
        let revision = self.next_availability_revision.max(now_unix_millis.max(1));
        self.next_availability_revision = revision
            .checked_add(1)
            .ok_or_else(|| EngineError::internal_invariant("availability revision overflowed"))?;
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

#[cfg(feature = "development-runtime-v3-simulator")]
fn development_result(
    admitted: &AdmittedRuntimeExecution,
    behavior: &RuntimeV3DevelopmentBehavior,
    now_unix_millis: u64,
) -> Result<InferenceExecutionResultV3, EngineError> {
    let request = admitted.request();
    let nominal_completion = now_unix_millis.checked_add(1).ok_or_else(|| {
        EngineError::resource_exhaustion("development completion time overflowed")
    })?;
    let deadline_unix_millis = contract_result(request.deadline_unix_millis())?;
    let deadline_exceeded = nominal_completion > deadline_unix_millis;
    let completed_unix_millis = nominal_completion.min(deadline_unix_millis);
    let started_at = contract_result(runtime_v3_timestamp_from_unix_millis(now_unix_millis))?;
    let completed_at =
        contract_result(runtime_v3_timestamp_from_unix_millis(completed_unix_millis))?;
    let resources = vec![
        RuntimeResourceUsageV3 {
            resource: RuntimeResourceV3::InferenceMemory,
            units: 1,
            unit: RuntimeResourceUnitV3::Bytes,
        },
        RuntimeResourceUsageV3 {
            resource: RuntimeResourceV3::WallTime,
            units: 1,
            unit: RuntimeResourceUnitV3::Milliseconds,
        },
    ];
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
