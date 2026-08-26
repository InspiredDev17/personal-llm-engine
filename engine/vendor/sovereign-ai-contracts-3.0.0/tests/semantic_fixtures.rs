use serde::de::DeserializeOwned;
use serde::Deserialize;
use sovereign_ai_contracts::cognitive::{
    validate_execution_against_request, CognitiveExecutionRequest, CognitiveExecutionResult,
};
use sovereign_ai_contracts::common::ContractVersion;
use sovereign_ai_contracts::identity::{
    EvaluationRevisionId, ModelArtifactId, ModelId, QualificationEvidenceId, RequestId,
};
use sovereign_ai_contracts::inference_runtime_v3::{
    ArtifactDigestAlgorithmV3, InferenceExecutionRequestV3, InferenceExecutionResultV3,
    InferenceReservationDecisionV3, InferenceReservationRequestV3, RuntimeAvailabilityV3,
};
use sovereign_ai_contracts::knowledge::{
    canonical_memory_revision_invalidation_digest, canonical_static_knowledge_bundle_digest,
    validate_memory_revision_invalidation_order, BundleInvalidationState, CacheCompatibility,
    CacheCompatibilityOutcome, CacheResidency, CacheStatus, CagCacheBuildRequest,
    CagCacheDescriptor, CagCacheQueryResult, CagEligibility, KnowledgePlanRequest,
    KnowledgePlanResult, KnowledgePlanStatus, MemoryRevisionInvalidated, StaticBundleReference,
    StaticKnowledgeBundle,
};
use sovereign_ai_contracts::knowledge_prefix::{
    StaticPrefixMaterialRequest, StaticPrefixMaterialResult,
};
use sovereign_ai_contracts::memory::SensitivityClass;
use sovereign_ai_contracts::model_admission_v1::{
    canonical_model_execution_admission_digest, validate_model_execution_admission_revision,
    ModelAdmissionState, ModelAdmissionTrustState, ModelExecutionAdmission,
};
use sovereign_ai_contracts::qualification_v2::{
    canonical_model_capability_digest, validate_manifest_revision, ModelCapabilityManifest,
};
use sovereign_ai_contracts::qualification_v2_1::{
    canonical_model_capability_digest_v21, validate_manifest_revision_v21,
    ModelCapabilityManifestV21, ModelQualificationStateV21, QualificationTrustStateV21,
};
use sovereign_ai_contracts::{Result, Validate};

#[derive(Deserialize)]
struct Manifest {
    contract_version: String,
    cases: Vec<Case>,
}
#[derive(Deserialize)]
struct Case {
    name: String,
    suite: String,
    mutation: String,
}

fn fixture<T: DeserializeOwned + Validate>(name: &str) -> T {
    let path = format!("../fixtures/v1/valid/{name}");
    let data = std::fs::read(path).expect("read fixture");
    let value: T = serde_json::from_slice(&data).expect("decode fixture");
    value.validate().expect("valid fixture");
    value
}

#[test]
fn shared_semantic_fixtures_fail_closed() {
    let data = std::fs::read("../fixtures/semantic/manifest.json").expect("semantic manifest");
    let manifest: Manifest = serde_json::from_slice(&data).expect("decode manifest");
    assert_eq!(manifest.contract_version, "1.0.0");
    assert!(!manifest.cases.is_empty());
    for case in manifest.cases {
        let outcome = match case.suite.as_str() {
            "runtime_v3" => runtime_mutation(&case.mutation),
            "cognitive" => cognitive_mutation(&case.mutation),
            "qualification_v2" => qualification_mutation(&case.mutation),
            "qualification_v2_1" => qualification_v21_mutation(&case.mutation),
            "model_admission_v1" => model_admission_v1_mutation(&case.mutation),
            "knowledge" => knowledge_mutation(&case.mutation),
            "knowledge_prefix" => knowledge_prefix_mutation(&case.mutation),
            other => panic!("unknown suite {other}"),
        };
        assert!(
            outcome.is_err(),
            "{} ({}) did not fail closed",
            case.name,
            case.mutation
        );
    }
}

fn finalize_admission(value: &mut ModelExecutionAdmission) -> Result<()> {
    value.canonical_digest = canonical_model_execution_admission_digest(&value.payload)?;
    Ok(())
}

fn next_admission(previous: &ModelExecutionAdmission) -> Result<ModelExecutionAdmission> {
    let mut current = previous.clone();
    current.payload.revision += 1;
    current.payload.registry_sequence += 1;
    current.payload.previous_revision_digest = Some(previous.canonical_digest.clone());
    current.payload.issued_at = "2026-08-25T12:02:00Z".into();
    finalize_admission(&mut current)?;
    Ok(current)
}

fn model_admission_v1_mutation(mutation: &str) -> Result<()> {
    let mut previous: ModelExecutionAdmission = fixture("model-execution-admission-v2.1.json");
    match mutation {
        "development_production_activation" => {
            previous.payload.admission_state = ModelAdmissionState::ProductionActivated;
            previous.payload.development_execution_authorized = false;
            previous.payload.production_activation = true;
            previous.validate()
        }
        "same_revision_replay" => validate_model_execution_admission_revision(&previous, &previous),
        "maximum_revision_terminal" => {
            previous.payload.revision = u64::MAX;
            previous.payload.previous_revision_digest = Some("a".repeat(64));
            finalize_admission(&mut previous)?;
            validate_model_execution_admission_revision(&previous, &previous)
        }
        "revoked_terminal" => {
            previous.payload.admission_state = ModelAdmissionState::Revoked;
            previous.payload.development_execution_authorized = false;
            finalize_admission(&mut previous)?;
            let current = next_admission(&previous)?;
            validate_model_execution_admission_revision(&previous, &current)
        }
        "trust_downgrade" => {
            previous.payload.trust_state = ModelAdmissionTrustState::Verified;
            finalize_admission(&mut previous)?;
            let mut current = next_admission(&previous)?;
            current.payload.trust_state = ModelAdmissionTrustState::DevelopmentFixture;
            finalize_admission(&mut current)?;
            validate_model_execution_admission_revision(&previous, &current)
        }
        _ => {
            let mut current = next_admission(&previous)?;
            match mutation {
                "runtime_identity_substitution" => {
                    current.payload.runtime_identity.provider_model_id =
                        "provider.substituted".into();
                }
                "descriptor_substitution" => {
                    current.payload.execution_package_descriptor_digest.digest = format!(
                        "a{}",
                        &current.payload.execution_package_descriptor_digest.digest[1..]
                    );
                }
                "signer_substitution" => {
                    current.payload.package_signer_identity.authority_id =
                        "package.authority.other".into();
                }
                "registry_floor_rollback" => {
                    current.payload.minimum_registry_sequence -= 1;
                }
                "expiry_extension" => {
                    current.payload.expires_at = "2026-10-25T12:01:00Z".into();
                }
                other => panic!("unknown model admission v1 mutation {other}"),
            }
            finalize_admission(&mut current)?;
            validate_model_execution_admission_revision(&previous, &current)
        }
    }
}

fn finalize_v21(manifest: &mut ModelCapabilityManifestV21) -> Result<()> {
    manifest.canonical_digest = canonical_model_capability_digest_v21(&manifest.payload)?;
    Ok(())
}

fn next_v21(previous: &ModelCapabilityManifestV21) -> Result<ModelCapabilityManifestV21> {
    let mut current = previous.clone();
    current.payload.revision += 1;
    current.payload.anti_rollback_metadata.sequence += 1;
    current.payload.previous_revision_digest = Some(previous.canonical_digest.clone());
    current.payload.issued_at = "2026-08-15T12:10:00Z".into();
    finalize_v21(&mut current)?;
    Ok(current)
}

fn apply_fresh_evaluation_v21(manifest: &mut ModelCapabilityManifestV21) -> Result<()> {
    manifest.payload.issued_at = "2026-08-15T12:20:00Z".into();
    manifest.payload.evaluation_revision = EvaluationRevisionId::new("evr_fixture_2")?;
    manifest.payload.qualification_timestamp = "2026-08-15T12:14:00Z".into();
    manifest.payload.state_transition.occurred_at = "2026-08-15T12:15:00Z".into();
    manifest.payload.state_transition.reason_code = "evaluation.reaffirmed".into();
    let evidence_id = QualificationEvidenceId::new("qev_fixture_2")?;
    manifest.payload.evidence[0].evidence_id = evidence_id.clone();
    manifest.payload.evidence[0].evaluation_revision = manifest.payload.evaluation_revision.clone();
    manifest.payload.evidence[0].evidence_reference.reference_id = "evidence:rust-v2".into();
    manifest.payload.evidence[0].evidence_reference.digest = "d".repeat(64);
    manifest.payload.evidence[0].observed_at = "2026-08-15T12:13:00Z".into();
    manifest.payload.capabilities[0].evaluation_revision =
        manifest.payload.evaluation_revision.clone();
    manifest.payload.capabilities[0].evidence_ids = vec![evidence_id];
    Ok(())
}

#[allow(clippy::too_many_lines)]
fn qualification_v21_mutation(mutation: &str) -> Result<()> {
    let mut previous: ModelCapabilityManifestV21 = fixture("model-capability-manifest-v2.1.json");
    match mutation {
        "maximum_revision_terminal" => {
            previous.payload.revision = u64::MAX;
            previous.payload.anti_rollback_metadata.sequence = u64::MAX;
            previous.payload.previous_revision_digest = Some("a".repeat(64));
            finalize_v21(&mut previous)?;
            validate_manifest_revision_v21(&previous, &previous)
        }
        "revoked_terminal" => {
            previous.payload.qualification_state = ModelQualificationStateV21::Revoked;
            previous.payload.state_transition.reason_code = "security.revoked".into();
            finalize_v21(&mut previous)?;
            let current = next_v21(&previous)?;
            validate_manifest_revision_v21(&previous, &current)
        }
        "trust_downgrade" => {
            previous.payload.trust_state = QualificationTrustStateV21::Verified;
            finalize_v21(&mut previous)?;
            let mut current = next_v21(&previous)?;
            current.payload.trust_state = QualificationTrustStateV21::DevelopmentFixture;
            finalize_v21(&mut current)?;
            validate_manifest_revision_v21(&previous, &current)
        }
        "requalification_without_evidence" => {
            previous.payload.trust_state = QualificationTrustStateV21::Verified;
            previous.payload.qualification_state = ModelQualificationStateV21::Suspended;
            previous.payload.state_transition.reason_code = "security.suspended".into();
            finalize_v21(&mut previous)?;
            let mut current = next_v21(&previous)?;
            current.payload.qualification_state = ModelQualificationStateV21::Qualified;
            current.payload.state_transition.occurred_at = "2026-08-15T12:05:00Z".into();
            current.payload.state_transition.reason_code = "evaluation.requalified".into();
            finalize_v21(&mut current)?;
            validate_manifest_revision_v21(&previous, &current)
        }
        "evidence_id_reuse_with_new_evaluation" => {
            let mut current = next_v21(&previous)?;
            apply_fresh_evaluation_v21(&mut current)?;
            current.payload.evidence[0].evidence_id =
                previous.payload.evidence[0].evidence_id.clone();
            current.payload.capabilities[0].evidence_ids =
                vec![previous.payload.evidence[0].evidence_id.clone()];
            finalize_v21(&mut current)?;
            validate_manifest_revision_v21(&previous, &current)
        }
        "evidence_reference_reuse_with_new_evaluation" => {
            let mut current = next_v21(&previous)?;
            apply_fresh_evaluation_v21(&mut current)?;
            current.payload.evidence[0].evidence_reference =
                previous.payload.evidence[0].evidence_reference.clone();
            finalize_v21(&mut current)?;
            validate_manifest_revision_v21(&previous, &current)
        }
        "same_state_event_without_evaluation" => {
            let mut current = next_v21(&previous)?;
            current.payload.state_transition.occurred_at = "2026-08-15T12:05:00Z".into();
            current.payload.state_transition.reason_code = "evaluation.reaffirmed".into();
            finalize_v21(&mut current)?;
            validate_manifest_revision_v21(&previous, &current)
        }
        "qualified_rejection_without_evaluation" => {
            previous.payload.trust_state = QualificationTrustStateV21::Verified;
            previous.payload.qualification_state = ModelQualificationStateV21::Qualified;
            previous.payload.state_transition.reason_code = "fixture.qualified".into();
            finalize_v21(&mut previous)?;
            let mut current = next_v21(&previous)?;
            current.payload.qualification_state = ModelQualificationStateV21::Rejected;
            current.payload.state_transition.occurred_at = "2026-08-15T12:05:00Z".into();
            current.payload.state_transition.reason_code = "evaluation.rejected".into();
            finalize_v21(&mut current)?;
            validate_manifest_revision_v21(&previous, &current)
        }
        _ => {
            let mut current = next_v21(&previous)?;
            match mutation {
                "runtime_identity_substitution" => {
                    current.payload.runtime_identity.engine_instance_id =
                        "engine.instance-2".into();
                }
                "lineage_substitution" => {
                    current.payload.anti_rollback_metadata.lineage_id = "lineage:other".into();
                }
                "state_timestamp_rollback" => {
                    current.payload.state_transition.occurred_at = "2026-08-15T11:57:15Z".into();
                }
                "issued_timestamp_rollback" => {
                    current.payload.issued_at = "2026-08-15T11:57:45Z".into();
                }
                "expiry_extension_without_evaluation" => {
                    current.payload.expires_at = "2026-10-14T11:58:00Z".into();
                }
                "capability_substitution_without_evaluation" => {
                    current.payload.capabilities[0].score_basis_points += 1;
                }
                "evidence_substitution_without_evaluation" => {
                    current.payload.evidence[0].observed_at = "2026-08-15T11:54:00Z".into();
                }
                "resource_substitution_without_evaluation" => {
                    current.payload.resource_requirements.expected_memory_bytes += 1;
                }
                "modality_substitution_without_evaluation" => {
                    current.payload.modalities = vec!["audio".into(), "text".into()];
                }
                "taxonomy_substitution_without_evaluation" => {
                    current.payload.taxonomy_revision = "cognitive.v3".into();
                }
                other => panic!("unknown qualification v2.1 mutation {other}"),
            }
            finalize_v21(&mut current)?;
            validate_manifest_revision_v21(&previous, &current)
        }
    }
}

fn knowledge_prefix_mutation(mutation: &str) -> Result<()> {
    let request: StaticPrefixMaterialRequest =
        fixture("knowledge-prefix-request-project-1.2.0.json");
    let mut result: StaticPrefixMaterialResult =
        fixture("knowledge-prefix-result-available-1.2.0.json");
    match mutation {
        "bundle_id" => {
            result.bundle.bundle_id =
                sovereign_ai_contracts::identity::StaticKnowledgeBundleId::new("kbn_other_bundle")?;
        }
        "bundle_revision" => result.bundle.bundle_revision += 1,
        "bundle_digest" => {
            result.bundle.bundle_digest =
                "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".into();
        }
        "scope" => {
            result.authorization_scope.project_id = Some(
                sovereign_ai_contracts::identity::ProjectId::new("prj_other")?,
            );
        }
        "sensitivity" => result.sensitivity = SensitivityClass::Public,
        "expired_request" => return request.validate_at("2026-08-21T12:05:00Z"),
        "expired_result" => return result.validate_at("2026-08-21T12:04:00Z"),
        other => panic!("unknown knowledge prefix mutation {other}"),
    }
    result.validate_against(&request)
}

#[test]
fn knowledge_prefix_rejects_malformed_utf8() {
    let mut data =
        br#"{"contract_version":"1.2.0","kind":"knowledge.static_prefix_material_request"}"#
            .to_vec();
    let index = data.len() - 2;
    data[index] = 0xff;
    assert!(serde_json::from_slice::<StaticPrefixMaterialRequest>(&data).is_err());
}

#[allow(clippy::too_many_lines)]
fn knowledge_mutation(mutation: &str) -> Result<()> {
    let mut descriptor: CagCacheDescriptor = fixture("knowledge-cache-descriptor.json");
    let mut compatibility = CacheCompatibility {
        contract_version: ContractVersion,
        kind: "knowledge.cache_compatibility".into(),
        request_id: RequestId::new("req_cache_compat_fixture")?,
        expected: descriptor.binding.clone(),
        cache_id: Some(descriptor.cache_id.clone()),
        outcome: CacheCompatibilityOutcome::Compatible,
        mismatch_reasons: vec![],
        evaluated_at: "2026-08-21T12:04:00Z".into(),
    };
    match mutation {
        "model_mismatch" => {
            compatibility.expected.runtime_identity.model_id = ModelId::new("mdl_other")?;
        }
        "provider_model_mismatch" => {
            compatibility.expected.runtime_identity.provider_model_id =
                "provider.model.other".into();
        }
        "artifact_mismatch" => {
            compatibility.expected.runtime_identity.artifact_id =
                ModelArtifactId::new("art_other")?;
        }
        "tokenizer_mismatch" => {
            compatibility.expected.tokenizer_digest =
                "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd".into();
        }
        "architecture_mismatch" => {
            compatibility.expected.architecture_id = "transformer.other".into();
        }
        "cache_format_mismatch" => compatibility.expected.cache_format_version = "2.0.0".into(),
        "prefix_mismatch" => {
            compatibility.expected.bundle.ordered_prefix_digest =
                "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd".into();
        }
        "token_count_mismatch" => compatibility.expected.token_count += 1,
        "scope_mismatch" => {
            compatibility.expected.scope.project_id =
                sovereign_ai_contracts::identity::ProjectId::new("prj_other")?;
        }
        "sensitivity_downgrade" => {
            compatibility.expected.sensitivity = SensitivityClass::Public;
        }
        "engine_mismatch" => {
            compatibility.expected.runtime_identity.engine_instance_id = "engine.other".into();
        }
        "cache_byte_overflow" => {
            let mut request: CagCacheBuildRequest = fixture("knowledge-cache-build-request.json");
            request.maximum_cache_bytes = u64::MAX;
            return request.validate();
        }
        "required_zero_sources" => {
            let mut request: KnowledgePlanRequest = fixture("knowledge-plan-request.json");
            request.requirements[0].minimum_sources = 0;
            return request.validate();
        }
        "expired_ready_inventory" => {
            let mut result: CagCacheQueryResult = fixture("knowledge-cache-query-result.json");
            result.observed_at = "2026-08-23T00:00:00Z".into();
            result.expires_at = "2026-08-23T00:01:00Z".into();
            return result.validate();
        }
        "failed_plan_payload" => {
            let mut result: KnowledgePlanResult = fixture("knowledge-plan-direct.json");
            result.status = KnowledgePlanStatus::Failed;
            result.mode = None;
            result.degraded = true;
            return result.validate();
        }
        "stale_cache" => {
            return compatibility.validate_against(&descriptor, "2026-08-23T00:00:00Z");
        }
        "invalidated_cache" => {
            descriptor.status = CacheStatus::Invalidated;
            descriptor.residency = CacheResidency::Absent;
            descriptor.reason_code = Some("knowledge.cache.invalidated".into());
        }
        "bundle_member_order" => {
            let mut bundle: StaticKnowledgeBundle = fixture("knowledge-static-bundle.json");
            bundle.ordered_members.swap(0, 1);
            return bundle.validate();
        }
        "invalidated_bundle" => {
            let mut bundle: StaticKnowledgeBundle = fixture("knowledge-static-bundle.json");
            bundle.invalidation_state = BundleInvalidationState::Invalidated;
            bundle.cag_eligibility = CagEligibility::IneligibleInvalidated;
            bundle.invalidated_at = Some("2026-08-21T13:00:00Z".into());
            bundle.invalidation_reason = Some("knowledge.bundle.invalidated".into());
            bundle.bundle_digest = canonical_static_knowledge_bundle_digest(&bundle)?;
            descriptor.binding.bundle = StaticBundleReference {
                bundle_id: bundle.bundle_id.clone(),
                bundle_revision: bundle.bundle_revision,
                bundle_digest: bundle.bundle_digest.clone(),
                ordered_prefix_digest: bundle.ordered_prefix_digest.clone(),
            };
            return descriptor.validate_usable_against_bundle(&bundle, "2026-08-21T13:01:00Z");
        }
        "deleted_source" => {
            let mut event: MemoryRevisionInvalidated =
                fixture("knowledge-revision-invalidated.json");
            event.payload.deleted = true;
            event.payload.replacement_revision_id = None;
            event.canonical_digest = canonical_memory_revision_invalidation_digest(&event.payload)?;
            return descriptor.validate_unaffected_by_invalidation(&event);
        }
        "conflicting_replay" | "rollback_sequence" => {
            let previous: MemoryRevisionInvalidated =
                fixture("knowledge-revision-invalidated.json");
            let mut current = previous.clone();
            if mutation == "conflicting_replay" {
                current.payload.event_id =
                    sovereign_ai_contracts::identity::KnowledgeInvalidationEventId::new(
                        "kie_conflicting_event",
                    )?;
            } else {
                current.payload.sequence -= 1;
                current.payload.event_id =
                    sovereign_ai_contracts::identity::KnowledgeInvalidationEventId::new(
                        "kie_rollback_event",
                    )?;
            }
            current.canonical_digest =
                canonical_memory_revision_invalidation_digest(&current.payload)?;
            return validate_memory_revision_invalidation_order(&previous, &current).map(|_| ());
        }
        other => panic!("unknown knowledge mutation {other}"),
    }
    compatibility.validate_against(&descriptor, "2026-08-21T12:04:00Z")
}

fn runtime_mutation(mutation: &str) -> Result<()> {
    let availability: RuntimeAvailabilityV3 = fixture("inference-v3-runtime-availability.json");
    let reservation: InferenceReservationRequestV3 =
        fixture("inference-v3-reservation-request.json");
    let mut decision: InferenceReservationDecisionV3 =
        fixture("inference-v3-reservation-decision.json");
    let mut request: InferenceExecutionRequestV3 = fixture("inference-v3-execution-request.json");
    let mut result: InferenceExecutionResultV3 = fixture("inference-v3-execution-result.json");
    match mutation {
        "availability_expired" => availability.validate_fresh_at("2026-08-14T12:01:00Z"),
        "reservation_expired" => decision.validate_usable_at("2026-08-14T12:02:00Z"),
        "reservation_identity_mismatch" => {
            decision.identity.provider_model_id = "runtime-model-2".into();
            decision.validate_against(&reservation)
        }
        "reservation_budget_escalation" => {
            decision
                .granted_budget
                .as_mut()
                .expect("accepted")
                .maximum_output_tokens = 513;
            decision.validate_against(&reservation)
        }
        "execution_wrong_engine" => {
            request.identity.engine_instance_id = "engine.instance-2".into();
            request.validate_against(&decision)
        }
        "execution_artifact_mismatch" => {
            request.identity.artifact_id = ModelArtifactId::new("art_other")?;
            request.validate_against(&decision)
        }
        "execution_outside_window" => {
            request.created_at = "2026-08-14T12:02:01Z".into();
            request.deadline_at = "2026-08-14T12:02:02Z".into();
            request.validate_against(&decision)
        }
        "result_wrong_request" => {
            result.request_id = RequestId::new("req_other")?;
            result.validate_against(&request)
        }
        "result_model_mismatch" => {
            result.identity.model_id = ModelId::new("mdl_other")?;
            result.validate_against(&request)
        }
        "result_after_deadline" => {
            result.completed_at = "2026-08-14T12:00:32Z".into();
            result.validate_against(&request)
        }
        "result_token_escalation" => {
            result.usage.output_tokens = 513;
            result.usage.total_tokens = result.usage.input_tokens + result.usage.output_tokens;
            result.validate_against(&request)
        }
        other => panic!("unknown runtime mutation {other}"),
    }
}

fn cognitive_mutation(mutation: &str) -> Result<()> {
    let request: CognitiveExecutionRequest = fixture("cognitive-execution-request.json");
    let mut result: CognitiveExecutionResult = fixture("cognitive-execution-result.json");
    match mutation {
        "wrong_request" => result.request_id = RequestId::new("req_other")?,
        "wrong_engine" => result.runtime_identity.engine_instance_id = "engine.instance-2".into(),
        "artifact_mismatch" => {
            result.runtime_identity.artifact_id = ModelArtifactId::new("art_other")?;
        }
        "idempotency_mismatch" => result.idempotency_key = "idem.request-2".into(),
        "digest_mismatch" => {
            result.immutable_request_digest =
                "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".into();
        }
        "resource_escalation" => {
            result.resource_usage[0].units = request.resource_ceiling.maximum_wall_clock_millis + 1;
        }
        "plan_revision_mismatch" => result.plan_revision = request.plan_revision + 1,
        other => panic!("unknown cognitive mutation {other}"),
    }
    validate_execution_against_request(&request, &result)
}

fn qualification_mutation(mutation: &str) -> Result<()> {
    let mut previous: ModelCapabilityManifest = fixture("model-capability-manifest-v2.json");
    if mutation == "maximum_revision_terminal" {
        previous.payload.revision = u64::MAX;
        previous.payload.anti_rollback_metadata.sequence = u64::MAX;
        previous.payload.previous_revision_digest = Some("a".repeat(64));
        previous.canonical_digest = canonical_model_capability_digest(&previous.payload)?;
        return validate_manifest_revision(&previous, &previous);
    }
    let mut current = previous.clone();
    current.payload.revision = 2;
    current.payload.anti_rollback_metadata.sequence = 2;
    current.payload.previous_revision_digest = Some(previous.canonical_digest.clone());
    match mutation {
        "stale_revision" => {
            current.payload.revision = 1;
            current.payload.anti_rollback_metadata.sequence = 1;
        }
        "logical_model_substitution" => {
            current.payload.logical_model_id = ModelId::new("mdl_other")?;
            current.payload.runtime_identity.model_id = ModelId::new("mdl_other")?;
        }
        "provider_model_substitution" => {
            current.payload.runtime_identity.provider_model_id = "runtime-model-2".into();
        }
        "artifact_substitution" => {
            current.payload.runtime_identity.artifact_id = ModelArtifactId::new("art_other")?;
            for evidence in &mut current.payload.evidence {
                evidence.artifact_id = ModelArtifactId::new("art_other")?;
            }
        }
        "artifact_digest_substitution" => {
            current.payload.runtime_identity.artifact_digest = "d".repeat(64);
        }
        "artifact_digest_algorithm_substitution" => {
            current.payload.runtime_identity.artifact_digest_algorithm =
                ArtifactDigestAlgorithmV3::Sha384;
            current.payload.runtime_identity.artifact_digest = "d".repeat(96);
        }
        "backend_substitution" => {
            current.payload.runtime_identity.backend_id = "backend.other".into();
        }
        "backend_version_substitution" => {
            current.payload.runtime_identity.backend_version = "version.2".into();
        }
        "backend_instance_substitution" => {
            current.payload.runtime_identity.backend_instance_id = "backend.instance-2".into();
        }
        "engine_instance_substitution" => {
            current.payload.runtime_identity.engine_instance_id = "engine.instance-2".into();
        }
        other => panic!("unknown qualification mutation {other}"),
    }
    current.canonical_digest = canonical_model_capability_digest(&current.payload)?;
    validate_manifest_revision(&previous, &current)
}

#[test]
fn qualification_v2_accepts_identity_stable_revision() {
    let previous: ModelCapabilityManifest = fixture("model-capability-manifest-v2.json");
    let mut current = previous.clone();
    current.payload.revision = 2;
    current.payload.anti_rollback_metadata.sequence = 2;
    current.payload.previous_revision_digest = Some(previous.canonical_digest.clone());
    current.canonical_digest =
        canonical_model_capability_digest(&current.payload).expect("canonical revision digest");
    validate_manifest_revision(&previous, &current).expect("identity-stable revision");
}
