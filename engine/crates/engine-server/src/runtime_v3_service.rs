//! Authenticated service composition for canonical inference-runtime-v3 documents.
//!
//! This module binds the contract-only reservation adapter to live engine
//! registry, package, backend, and loaded-model state. Configuration remains
//! owner-only; no client message can install an identity or manufacture
//! qualification authority.

use crate::{
    ArtifactAuthorityId, EngineService, LoadedRuntimeModel, LogicalModelId,
    MAX_RUNTIME_V3_AVAILABILITY_TTL_MILLIS, RuntimeBackend, RuntimeIdentityAuthority,
    RuntimeIdentityBinding, RuntimeInstanceId, RuntimeV3ContractAdapter,
};
use personalai_engine_protocol::{RuntimeV3Document, ServiceRequestBody, ServiceResponseBody};
use personalai_engine_types::EngineError;
use personalai_engine_types::ModelId;
use sovereign_ai_contracts::inference_runtime_v3::{
    ArtifactDigestAlgorithmV3, InferenceRuntimeContractVersionV3, RuntimeAvailabilityKindV3,
    RuntimeAvailabilityV3, RuntimeIdentityV3, RuntimeModelStateV3, RuntimeModelStatusV3,
    RuntimeStateV3, encode_runtime_availability_v3, runtime_v3_timestamp_from_unix_millis,
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
}

impl RuntimeV3ServiceState {
    pub(crate) fn new(config: RuntimeV3ServiceConfig) -> Result<Self, EngineError> {
        let adapter = RuntimeV3ContractAdapter::new(
            config.maximum_active_reservations,
            config.maximum_inference_memory_bytes,
        )?;
        Ok(Self { config, adapter, bindings: BTreeMap::new(), next_availability_revision: 0 })
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
            _ => return Ok(None),
        };
        Ok(Some(response))
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
