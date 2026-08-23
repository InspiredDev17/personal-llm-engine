use sovereign_ai_contracts::identity::{EvaluationRevisionId, QualificationEvidenceId};
use sovereign_ai_contracts::qualification_v2::{
    canonical_model_capability_digest, validate_manifest_revision, ModelCapabilityManifest,
};
use sovereign_ai_contracts::qualification_v2_1::{
    canonical_model_capability_digest_v21, model_capability_signing_transcript_v21,
    validate_manifest_revision_v21, ModelCapabilityManifestV21, ModelQualificationStateV21,
    QualificationTrustStateV21,
};
use sovereign_ai_contracts::ContractError;

fn fixture() -> ModelCapabilityManifestV21 {
    serde_json::from_slice(include_bytes!(
        "../../fixtures/v1/valid/model-capability-manifest-v2.1.json"
    ))
    .expect("valid v2.1 fixture syntax")
}

fn fixture_v2() -> ModelCapabilityManifest {
    serde_json::from_slice(include_bytes!(
        "../../fixtures/v1/valid/model-capability-manifest-v2.json"
    ))
    .expect("valid v2 fixture syntax")
}

fn finalize(manifest: &mut ModelCapabilityManifestV21) {
    manifest.canonical_digest =
        canonical_model_capability_digest_v21(&manifest.payload).expect("valid canonical payload");
}

fn next(previous: &ModelCapabilityManifestV21) -> ModelCapabilityManifestV21 {
    let mut current = previous.clone();
    current.payload.revision += 1;
    current.payload.anti_rollback_metadata.sequence += 1;
    current.payload.previous_revision_digest = Some(previous.canonical_digest.clone());
    current.payload.issued_at = "2026-08-15T12:10:00Z".into();
    finalize(&mut current);
    current
}

fn apply_fresh_evaluation(manifest: &mut ModelCapabilityManifestV21) {
    manifest.payload.issued_at = "2026-08-15T12:20:00Z".into();
    manifest.payload.evaluation_revision =
        EvaluationRevisionId::new("evr_fixture_2").expect("evaluation id");
    manifest.payload.qualification_timestamp = "2026-08-15T12:14:00Z".into();
    manifest.payload.state_transition.occurred_at = "2026-08-15T12:15:00Z".into();
    manifest.payload.state_transition.reason_code = "evaluation.reaffirmed".into();
    let evidence_id = QualificationEvidenceId::new("qev_fixture_2").expect("evidence id");
    manifest.payload.evidence[0].evidence_id = evidence_id.clone();
    manifest.payload.evidence[0].evaluation_revision = manifest.payload.evaluation_revision.clone();
    manifest.payload.evidence[0].evidence_reference.reference_id = "evidence:rust-v2".into();
    manifest.payload.evidence[0].evidence_reference.digest = "d".repeat(64);
    manifest.payload.evidence[0].observed_at = "2026-08-15T12:13:00Z".into();
    manifest.payload.capabilities[0].evaluation_revision =
        manifest.payload.evaluation_revision.clone();
    manifest.payload.capabilities[0].evidence_ids = vec![evidence_id];
}

#[test]
fn fixture_digest_matches() {
    let manifest = fixture();
    assert_eq!(
        canonical_model_capability_digest_v21(&manifest.payload).unwrap(),
        manifest.canonical_digest
    );
}

#[test]
fn strict_document_decode_and_transcript_are_version_bound() {
    let data = include_bytes!("../../fixtures/v1/valid/model-capability-manifest-v2.1.json");
    sovereign_ai_contracts::document::validate_document(data).expect("strict document");

    let mut unknown = br#"{"unexpected":true,"#.to_vec();
    unknown.extend_from_slice(&data[1..]);
    assert!(sovereign_ai_contracts::document::validate_document(&unknown).is_err());

    let mut duplicate = br#"{"contract_version":"2.1.0","#.to_vec();
    duplicate.extend_from_slice(&data[1..]);
    assert!(sovereign_ai_contracts::document::validate_document(&duplicate).is_err());

    let manifest = fixture();
    let transcript = model_capability_signing_transcript_v21(&manifest.canonical_digest).unwrap();
    let prefix = b"SOVEREIGN-MODEL-CAPABILITY-SIGNATURE-V2.1\0";
    assert!(transcript.starts_with(prefix));
    let digest_bytes: Vec<u8> = manifest
        .canonical_digest
        .as_bytes()
        .chunks_exact(2)
        .map(|pair| u8::from_str_radix(std::str::from_utf8(pair).unwrap(), 16).unwrap())
        .collect();
    assert_eq!(&transcript[prefix.len()..], digest_bytes);
}

#[test]
fn identity_stable_revision_is_accepted() {
    let previous = fixture();
    let current = next(&previous);
    validate_manifest_revision_v21(&previous, &current).expect("identity-stable revision");
}

#[test]
fn runtime_producer_and_lineage_substitution_are_rejected() {
    let previous = fixture();
    let mut current = next(&previous);
    current.payload.runtime_identity.backend_instance_id = "backend.instance-2".into();
    finalize(&mut current);
    assert!(validate_manifest_revision_v21(&previous, &current).is_err());

    let mut current = next(&previous);
    current.payload.producer_identity.producer_id = "model_foundry.other".into();
    finalize(&mut current);
    assert!(validate_manifest_revision_v21(&previous, &current).is_err());

    let mut current = next(&previous);
    current.payload.anti_rollback_metadata.lineage_id = "lineage:other".into();
    finalize(&mut current);
    assert!(validate_manifest_revision_v21(&previous, &current).is_err());
}

#[test]
fn revocation_is_terminal() {
    let previous = fixture();
    let mut revoked = next(&previous);
    revoked.payload.qualification_state = ModelQualificationStateV21::Revoked;
    revoked.payload.state_transition.occurred_at = "2026-08-15T12:05:00Z".into();
    revoked.payload.state_transition.reason_code = "security.revoked".into();
    finalize(&mut revoked);
    validate_manifest_revision_v21(&previous, &revoked).expect("valid revocation");

    let mut later = next(&revoked);
    later.payload.issued_at = "2026-08-15T12:20:00Z".into();
    finalize(&mut later);
    assert!(validate_manifest_revision_v21(&revoked, &later).is_err());
}

#[test]
fn suspension_recovery_requires_fresh_evaluation() {
    let mut qualified = fixture();
    qualified.payload.trust_state = QualificationTrustStateV21::Verified;
    qualified.payload.qualification_state = ModelQualificationStateV21::Qualified;
    qualified.payload.state_transition.reason_code = "fixture.qualified".into();
    finalize(&mut qualified);

    let mut suspended = next(&qualified);
    suspended.payload.qualification_state = ModelQualificationStateV21::Suspended;
    suspended.payload.state_transition.occurred_at = "2026-08-15T12:05:00Z".into();
    suspended.payload.state_transition.reason_code = "security.suspended".into();
    finalize(&mut suspended);
    validate_manifest_revision_v21(&qualified, &suspended).expect("valid suspension");

    let mut recovered = next(&suspended);
    recovered.payload.issued_at = "2026-08-15T12:20:00Z".into();
    recovered.payload.qualification_state = ModelQualificationStateV21::Qualified;
    recovered.payload.state_transition.occurred_at = "2026-08-15T12:15:00Z".into();
    recovered.payload.state_transition.reason_code = "evaluation.requalified".into();
    finalize(&mut recovered);
    assert!(validate_manifest_revision_v21(&suspended, &recovered).is_err());

    apply_fresh_evaluation(&mut recovered);
    recovered.payload.qualification_state = ModelQualificationStateV21::Qualified;
    recovered.payload.state_transition.reason_code = "evaluation.requalified".into();
    finalize(&mut recovered);
    validate_manifest_revision_v21(&suspended, &recovered).expect("fresh requalification");
}

#[test]
fn same_state_reevaluation_is_auditable() {
    let mut previous = fixture();
    previous.payload.trust_state = QualificationTrustStateV21::Verified;
    previous.payload.qualification_state = ModelQualificationStateV21::Qualified;
    previous.payload.state_transition.reason_code = "fixture.qualified".into();
    finalize(&mut previous);

    let mut current = next(&previous);
    apply_fresh_evaluation(&mut current);
    finalize(&mut current);
    validate_manifest_revision_v21(&previous, &current).expect("same-state re-evaluation");

    let mut without_evaluation = next(&previous);
    without_evaluation.payload.state_transition.occurred_at = "2026-08-15T12:05:00Z".into();
    without_evaluation.payload.state_transition.reason_code = "evaluation.reaffirmed".into();
    finalize(&mut without_evaluation);
    assert!(validate_manifest_revision_v21(&previous, &without_evaluation).is_err());
}

#[test]
fn evidence_identity_cannot_be_relabeled() {
    let previous = fixture();
    let mut current = next(&previous);
    apply_fresh_evaluation(&mut current);
    current.payload.evidence[0].evidence_id = previous.payload.evidence[0].evidence_id.clone();
    current.payload.capabilities[0].evidence_ids =
        vec![previous.payload.evidence[0].evidence_id.clone()];
    finalize(&mut current);
    assert!(validate_manifest_revision_v21(&previous, &current).is_err());
}

#[test]
fn evidence_reference_cannot_be_relabeled() {
    let previous = fixture();
    let mut current = next(&previous);
    apply_fresh_evaluation(&mut current);
    current.payload.evidence[0].evidence_reference =
        previous.payload.evidence[0].evidence_reference.clone();
    finalize(&mut current);
    assert!(validate_manifest_revision_v21(&previous, &current).is_err());
}

#[test]
fn qualified_or_suspended_rejection_requires_fresh_evidence() {
    for state in [
        ModelQualificationStateV21::Qualified,
        ModelQualificationStateV21::Suspended,
    ] {
        let mut previous = fixture();
        previous.payload.trust_state = QualificationTrustStateV21::Verified;
        previous.payload.qualification_state = state;
        previous.payload.state_transition.reason_code = "fixture.previous_state".into();
        finalize(&mut previous);

        let mut without_evaluation = next(&previous);
        without_evaluation.payload.qualification_state = ModelQualificationStateV21::Rejected;
        without_evaluation.payload.state_transition.occurred_at = "2026-08-15T12:05:00Z".into();
        without_evaluation.payload.state_transition.reason_code = "evaluation.rejected".into();
        finalize(&mut without_evaluation);
        assert!(validate_manifest_revision_v21(&previous, &without_evaluation).is_err());

        let mut with_evaluation = next(&previous);
        apply_fresh_evaluation(&mut with_evaluation);
        with_evaluation.payload.qualification_state = ModelQualificationStateV21::Rejected;
        with_evaluation.payload.state_transition.reason_code = "evaluation.rejected".into();
        finalize(&mut with_evaluation);
        validate_manifest_revision_v21(&previous, &with_evaluation)
            .expect("evidence-based rejection");
    }
}

#[test]
fn maximum_revision_is_a_typed_terminal_error_in_v2_and_v21() {
    let mut previous_v2 = fixture_v2();
    previous_v2.payload.revision = u64::MAX;
    previous_v2.payload.anti_rollback_metadata.sequence = u64::MAX;
    previous_v2.payload.previous_revision_digest = Some("a".repeat(64));
    previous_v2.canonical_digest =
        canonical_model_capability_digest(&previous_v2.payload).expect("v2 digest");
    assert!(matches!(
        validate_manifest_revision(&previous_v2, &previous_v2),
        Err(ContractError::RevisionTerminal { revision: u64::MAX })
    ));

    let mut previous_v21 = fixture();
    previous_v21.payload.revision = u64::MAX;
    previous_v21.payload.anti_rollback_metadata.sequence = u64::MAX;
    previous_v21.payload.previous_revision_digest = Some("a".repeat(64));
    finalize(&mut previous_v21);
    assert!(matches!(
        validate_manifest_revision_v21(&previous_v21, &previous_v21),
        Err(ContractError::RevisionTerminal { revision: u64::MAX })
    ));
}

#[test]
fn lifecycle_and_trust_rollback_are_rejected() {
    let previous = fixture();
    let mut current = next(&previous);
    current.payload.state_transition.occurred_at = "2026-08-15T11:57:15Z".into();
    finalize(&mut current);
    assert!(validate_manifest_revision_v21(&previous, &current).is_err());

    let mut verified = previous;
    verified.payload.trust_state = QualificationTrustStateV21::Verified;
    finalize(&mut verified);
    let mut current = next(&verified);
    current.payload.trust_state = QualificationTrustStateV21::DevelopmentFixture;
    finalize(&mut current);
    assert!(validate_manifest_revision_v21(&verified, &current).is_err());
}
