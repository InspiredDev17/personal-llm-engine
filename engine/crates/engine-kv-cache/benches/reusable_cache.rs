//! Dependency-free synthetic reusable-prefix lookup benchmark.

use personalai_engine_knowledge_contract::{
    ArtifactDigest, CacheAuthorizationScope, CacheBinding, CacheNamespace, CacheRuntimeIdentity,
    CacheSensitivity, KnowledgeIdentity, StaticBundleBinding,
};
use personalai_engine_kv_cache::{
    AttentionPositionSemantics, CacheRetentionPolicy, ContiguousKvCache, KvCacheCompatibility,
    KvCacheLookup, KvCachePlan, ReusableKvCacheLimits, ReusableKvCacheStore,
};
use personalai_engine_memory::MemoryPool;
use personalai_engine_types::{
    ArchitectureId, BackendId, DType, ModelId, ModelPackageHash, RequestId, ResourceBudget,
    SequenceId, Sha256Digest,
};
use std::hint::black_box;
use std::time::Instant;

const ITERATIONS: u32 = 10_000;

#[allow(
    clippy::too_many_lines,
    reason = "the standalone benchmark keeps its complete synthetic identity and accounting fixture visible"
)]
fn main() {
    let budget = ResourceBudget::new(16 << 20, 8 << 20, 4).expect("benchmark budget");
    let pool = MemoryPool::from_budget(&budget).expect("benchmark pool");
    let model_id = ModelId::new("synthetic-cag-benchmark").expect("model ID");
    let plan = KvCachePlan::new(2, 2, 8, 512, DType::F32, &budget).expect("cache plan");
    let cache = ContiguousKvCache::new(model_id.clone(), plan, 4, pool.clone()).expect("cache");
    let source = cache
        .create_sequence(RequestId::new(1).expect("request"), SequenceId::new(1).expect("sequence"))
        .expect("source sequence");
    for token in 0..256_u32 {
        let mut writer = cache.begin_token(source).expect("token writer");
        for layer in 0..2_u32 {
            let value = f32::from(u16::try_from(token + layer).expect("small fixture")) / 256.0;
            writer.write_layer(layer, &[value; 16], &[value + 0.5; 16]).expect("layer write");
        }
        writer.commit().expect("commit");
    }
    let limits =
        ReusableKvCacheLimits::new(16, 4 << 20, cache.sequence_allocation_bytes(), 16 << 20)
            .expect("store limits");
    let store = ReusableKvCacheStore::new(model_id.clone(), pool, limits).expect("store");
    let compatibility = KvCacheCompatibility::new(
        model_id,
        ModelPackageHash::from_sha384_bytes([1; 48]),
        ModelPackageHash::from_sha384_bytes([2; 48]),
        Sha256Digest::from_bytes([2; 32]),
        ArchitectureId::new("decoder-rms-rope-swiglu-gqa-v1").expect("architecture"),
        BackendId::new("cpu-reference").expect("backend"),
        AttentionPositionSemantics::CausalZeroBasedV1,
        10_000.0,
        1.0,
        DType::F32,
    )
    .expect("compatibility");
    let scope = CacheAuthorizationScope::new(
        CacheNamespace::Project,
        KnowledgeIdentity::new("owner.synthetic").expect("owner"),
        Some(KnowledgeIdentity::new("project.synthetic").expect("project")),
        None,
        None,
    )
    .expect("scope");
    let tokens = (0..256_u32).collect::<Vec<_>>();
    let binding = CacheBinding::new(
        CacheRuntimeIdentity::new(
            KnowledgeIdentity::new("logical.synthetic").expect("logical model"),
            compatibility.model_id().clone(),
            KnowledgeIdentity::new("artifact.synthetic").expect("artifact"),
            ArtifactDigest::Sha384(compatibility.model_artifact_hash()),
            compatibility.backend_id().clone(),
            KnowledgeIdentity::new("version.1").expect("backend version"),
            KnowledgeIdentity::new("backend.synthetic").expect("backend instance"),
            KnowledgeIdentity::new("engine.synthetic").expect("engine instance"),
        )
        .expect("runtime identity"),
        KnowledgeIdentity::new("tokenizer.synthetic").expect("tokenizer"),
        Sha256Digest::from_bytes([2; 32]),
        compatibility.architecture_id().clone(),
        KnowledgeIdentity::new("rope.causal-zero-based-v1").expect("RoPE semantics"),
        compatibility.kv_dtype(),
        StaticBundleBinding::new(
            KnowledgeIdentity::new("bundle.synthetic").expect("bundle"),
            1,
            Sha256Digest::from_bytes([3; 32]),
            Sha256Digest::from_bytes([4; 32]),
        )
        .expect("bundle binding"),
        scope.clone(),
        CacheSensitivity::Internal,
        256,
    )
    .expect("cache binding");
    store
        .create_from_sequence(
            &cache,
            source,
            compatibility.clone(),
            binding,
            &tokens,
            &[0.0; 256],
            1,
            CacheRetentionPolicy::explicit(),
        )
        .expect("seal cache");

    let started = Instant::now();
    for iteration in 0..ITERATIONS {
        let result = store
            .lookup_longest_prefix(
                black_box(&compatibility),
                black_box(&scope),
                black_box(&tokens),
                u64::from(iteration) + 2,
            )
            .expect("lookup");
        assert!(matches!(result, KvCacheLookup::Hit(_)));
    }
    let elapsed = started.elapsed();
    println!(
        "synthetic exact-prefix lookup: iterations={ITERATIONS} prefix_tokens={} elapsed_ns={} (not production performance evidence)",
        tokens.len(),
        elapsed.as_nanos()
    );
}
