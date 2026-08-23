use serde::de::DeserializeOwned;
use sovereign_ai_contracts::knowledge_invalidation_feed::{
    canonical_knowledge_invalidation_feed_result_digest,
    canonical_knowledge_invalidation_tombstone_digest,
    knowledge_invalidation_feed_result_signing_transcript,
    knowledge_invalidation_tombstone_signing_transcript,
    resolve_knowledge_invalidation_source_sensitivity,
    validate_knowledge_invalidation_feed_page_order,
    validate_knowledge_invalidation_feed_request_replay,
    validate_knowledge_invalidation_feed_request_replay_at,
    validate_knowledge_invalidation_tombstone_order, KnowledgeInvalidationDisposition,
    KnowledgeInvalidationFeedRequest, KnowledgeInvalidationFeedRequestDisposition,
    KnowledgeInvalidationFeedResult, KnowledgeInvalidationStreamHistory,
    KnowledgeInvalidationTombstone,
};
use sovereign_ai_contracts::memory::SensitivityClass;
use sovereign_ai_contracts::Validate;

const TOMBSTONE_DIGEST: &str = "2b59e154d1ad1bf53c29ed8c3b6471f5d3fdf9c89b8ecf2d893a29a7ccec4c9f";
const FEED_DIGEST: &str = "3ba14a4e4453a1c08f66b3135e52b19391ada3efbfcb28752573157b1007a7c1";

fn fixture<T: DeserializeOwned + Validate>(name: &str) -> T {
    let path = format!("../fixtures/v1/valid/{name}");
    let data = std::fs::read(path).expect("read fixture");
    let value: T = serde_json::from_slice(&data).expect("decode fixture");
    value.validate().expect("validate fixture");
    value
}

#[test]
fn golden_documents_and_transcripts_match_go() {
    let request: KnowledgeInvalidationFeedRequest =
        fixture("knowledge-invalidation-feed-request-1.3.0.json");
    let tombstone: KnowledgeInvalidationTombstone =
        fixture("knowledge-invalidation-tombstone-1.3.0.json");
    let result: KnowledgeInvalidationFeedResult =
        fixture("knowledge-invalidation-feed-result-1.3.0.json");

    assert_eq!(tombstone.canonical_digest, TOMBSTONE_DIGEST);
    assert_eq!(result.canonical_digest, FEED_DIGEST);
    result
        .validate_against(&request, "platform.fixture")
        .expect("correlate result");
    request
        .validate_at("2026-08-22T13:02:00Z")
        .expect("active request");
    result
        .validate_at("2026-08-22T13:02:00Z")
        .expect("active result");

    let tombstone_transcript =
        knowledge_invalidation_tombstone_signing_transcript(TOMBSTONE_DIGEST)
            .expect("tombstone transcript");
    assert!(tombstone_transcript
        .starts_with(b"SOVEREIGN-KNOWLEDGE-INVALIDATION-TOMBSTONE-SIGNATURE-V1\0"));
    assert_eq!(
        tombstone_transcript.len(),
        b"SOVEREIGN-KNOWLEDGE-INVALIDATION-TOMBSTONE-SIGNATURE-V1\0".len() + 32
    );
    let feed_transcript = knowledge_invalidation_feed_result_signing_transcript(FEED_DIGEST)
        .expect("feed transcript");
    assert!(
        feed_transcript.starts_with(b"SOVEREIGN-KNOWLEDGE-INVALIDATION-FEED-RESULT-SIGNATURE-V1\0")
    );
    assert_eq!(
        feed_transcript.len(),
        b"SOVEREIGN-KNOWLEDGE-INVALIDATION-FEED-RESULT-SIGNATURE-V1\0".len() + 32
    );
}

#[test]
fn authenticated_consumer_and_authorization_revision_are_bound() {
    let mut request: KnowledgeInvalidationFeedRequest =
        fixture("knowledge-invalidation-feed-request-1.3.0.json");
    let result: KnowledgeInvalidationFeedResult =
        fixture("knowledge-invalidation-feed-result-1.3.0.json");

    assert!(result.validate_against(&request, "platform.other").is_err());
    request.expected_authorization_revision = "authorization.revision.8".into();
    assert!(result
        .validate_against(&request, "platform.fixture")
        .is_err());
}

#[test]
fn visible_sequence_gaps_and_exact_replay_are_deterministic() {
    let previous: KnowledgeInvalidationTombstone =
        fixture("knowledge-invalidation-tombstone-1.3.0.json");
    let mut current = previous.clone();
    current.payload.sequence = 11;
    current.payload.event_id = sovereign_ai_contracts::identity::KnowledgeInvalidationEventId::new(
        "kie_fixture_tombstone_next",
    )
    .expect("event id");
    current.payload.effective_at = "2026-08-22T13:00:03Z".into();
    current.payload.issued_at = "2026-08-22T13:00:04Z".into();
    current.canonical_digest =
        canonical_knowledge_invalidation_tombstone_digest(&current.payload).expect("digest");

    assert_eq!(
        validate_knowledge_invalidation_tombstone_order(&previous, &current)
            .expect("gapped next event"),
        KnowledgeInvalidationDisposition::Next
    );
    assert_eq!(
        validate_knowledge_invalidation_tombstone_order(&current, &current)
            .expect("duplicate event"),
        KnowledgeInvalidationDisposition::Duplicate
    );

    let mut conflict = current.clone();
    conflict.payload.event_id =
        sovereign_ai_contracts::identity::KnowledgeInvalidationEventId::new(
            "kie_fixture_conflicting_tombstone",
        )
        .expect("event id");
    conflict.canonical_digest =
        canonical_knowledge_invalidation_tombstone_digest(&conflict.payload).expect("digest");
    assert!(validate_knowledge_invalidation_tombstone_order(&current, &conflict).is_err());
}

#[test]
fn request_replay_and_bootstrap_pins_fail_closed() {
    let previous: KnowledgeInvalidationFeedRequest =
        fixture("knowledge-invalidation-feed-request-1.3.0.json");
    assert_eq!(
        validate_knowledge_invalidation_feed_request_replay(&previous, &previous)
            .expect("exact duplicate"),
        KnowledgeInvalidationFeedRequestDisposition::Duplicate
    );
    assert!(validate_knowledge_invalidation_feed_request_replay_at(
        &previous,
        &previous,
        "2026-08-22T13:05:01Z"
    )
    .is_err());
    assert_eq!(
        validate_knowledge_invalidation_feed_request_replay_at(
            &previous,
            &previous,
            "2026-08-22T13:02:00Z"
        )
        .expect("active exact replay"),
        KnowledgeInvalidationFeedRequestDisposition::Duplicate
    );

    let mut conflict = previous.clone();
    conflict.limit -= 1;
    assert!(validate_knowledge_invalidation_feed_request_replay(&previous, &conflict).is_err());

    let mut bootstrap = previous;
    bootstrap.after_sequence = 0;
    bootstrap.expected_authorization_view_id = None;
    bootstrap.expected_stream_epoch = None;
    bootstrap.validate().expect("sequence-zero bootstrap");
    bootstrap.after_sequence = 1;
    assert!(bootstrap.validate().is_err());
}

#[test]
fn cutover_rejects_fabricated_legacy_events_and_accepts_restricted_metadata() {
    let mut result: KnowledgeInvalidationFeedResult =
        fixture("knowledge-invalidation-feed-result-1.3.0.json");
    result.payload.stream_history =
        KnowledgeInvalidationStreamHistory::CutoverRequiresReconciliation;
    result.payload.epoch_started_at = "2026-08-22T13:00:02Z".into();
    assert!(canonical_knowledge_invalidation_feed_result_digest(&result.payload).is_err());

    let mut restricted: KnowledgeInvalidationFeedResult =
        fixture("knowledge-invalidation-feed-result-1.3.0.json");
    restricted.payload.tombstones[0].payload.source_sensitivity = SensitivityClass::Restricted;
    restricted.payload.tombstones[0].canonical_digest =
        canonical_knowledge_invalidation_tombstone_digest(
            &restricted.payload.tombstones[0].payload,
        )
        .expect("tombstone digest");
    restricted.canonical_digest =
        canonical_knowledge_invalidation_feed_result_digest(&restricted.payload)
            .expect("feed digest");
    restricted
        .validate()
        .expect("restricted revocation metadata");
}

#[test]
fn source_sensitivity_uses_the_stricter_side() {
    assert_eq!(
        resolve_knowledge_invalidation_source_sensitivity(
            &SensitivityClass::Public,
            Some(&SensitivityClass::Restricted),
        ),
        SensitivityClass::Restricted
    );
    assert_eq!(
        resolve_knowledge_invalidation_source_sensitivity(
            &SensitivityClass::Restricted,
            Some(&SensitivityClass::Public),
        ),
        SensitivityClass::Restricted
    );
    assert_eq!(
        resolve_knowledge_invalidation_source_sensitivity(&SensitivityClass::Personal, None),
        SensitivityClass::Personal
    );
}

#[test]
fn non_adjacent_event_id_reuse_is_rejected() {
    let mut result: KnowledgeInvalidationFeedResult =
        fixture("knowledge-invalidation-feed-result-1.3.0.json");
    let first = result.payload.tombstones[0].clone();
    let mut second = first.clone();
    second.payload.sequence = 8;
    second.payload.event_id = sovereign_ai_contracts::identity::KnowledgeInvalidationEventId::new(
        "kie_fixture_tombstone_second",
    )
    .expect("event ID");
    second.payload.effective_at = "2026-08-22T13:00:02Z".into();
    second.payload.issued_at = "2026-08-22T13:00:02Z".into();
    second.canonical_digest =
        canonical_knowledge_invalidation_tombstone_digest(&second.payload).expect("digest");
    let mut third = first.clone();
    third.payload.sequence = 9;
    third.payload.effective_at = "2026-08-22T13:00:03Z".into();
    third.payload.issued_at = "2026-08-22T13:00:03Z".into();
    third.canonical_digest =
        canonical_knowledge_invalidation_tombstone_digest(&third.payload).expect("digest");
    result.payload.tombstones = vec![first, second, third];
    assert!(result.payload.validate().is_err());
}

#[test]
fn page_continuity_rejects_cursor_rollback() {
    let previous: KnowledgeInvalidationFeedResult =
        fixture("knowledge-invalidation-feed-result-1.3.0.json");
    let mut current = previous.clone();
    current.payload.request_id =
        sovereign_ai_contracts::identity::RequestId::new("req_invalidation_feed_next_page")
            .expect("request id");
    current.payload.after_sequence = previous.payload.observed_through_sequence;
    current.payload.tombstones.clear();
    current.payload.observed_through_sequence = 12;
    current.payload.high_watermark_sequence = 12;
    current.payload.observed_at = "2026-08-22T13:01:00Z".into();
    current.canonical_digest =
        canonical_knowledge_invalidation_feed_result_digest(&current.payload).expect("digest");
    assert_eq!(
        validate_knowledge_invalidation_feed_page_order(&previous, &current).expect("next page"),
        KnowledgeInvalidationDisposition::Next
    );

    let mut rollback = current;
    rollback.payload.after_sequence -= 1;
    rollback.canonical_digest =
        canonical_knowledge_invalidation_feed_result_digest(&rollback.payload).expect("digest");
    assert!(validate_knowledge_invalidation_feed_page_order(&previous, &rollback).is_err());
}

#[test]
fn cross_page_event_id_reuse_is_rejected() {
    let previous: KnowledgeInvalidationFeedResult =
        fixture("knowledge-invalidation-feed-result-1.3.0.json");
    let mut current = previous.clone();
    current.payload.request_id =
        sovereign_ai_contracts::identity::RequestId::new("req_invalidation_feed_reused_event")
            .expect("request ID");
    current.payload.after_sequence = previous.payload.observed_through_sequence;
    let mut reused = previous.payload.tombstones[0].clone();
    reused.payload.sequence = 11;
    reused.payload.effective_at = "2026-08-22T13:01:01Z".into();
    reused.payload.issued_at = "2026-08-22T13:01:02Z".into();
    reused.canonical_digest =
        canonical_knowledge_invalidation_tombstone_digest(&reused.payload).expect("digest");
    current.payload.tombstones = vec![reused];
    current.payload.observed_through_sequence = 11;
    current.payload.high_watermark_sequence = 11;
    current.payload.observed_at = "2026-08-22T13:01:03Z".into();
    current.canonical_digest =
        canonical_knowledge_invalidation_feed_result_digest(&current.payload).expect("digest");
    assert!(validate_knowledge_invalidation_feed_page_order(&previous, &current).is_err());
}

#[test]
fn strict_document_decode_rejects_explicit_nulls() {
    let data = std::fs::read_to_string(
        "../fixtures/v1/valid/knowledge-invalidation-feed-request-1.3.0.json",
    )
    .expect("read fixture");
    let data = data.replacen(
        r#""expires_at": "2026-08-22T13:05:00Z""#,
        r#""expires_at": "2026-08-22T13:05:00Z", "extensions": null"#,
        1,
    );
    assert!(sovereign_ai_contracts::document::validate_document(data.as_bytes()).is_err());
}
