//! Synthetic parity between fully resident and bounded streamed expert storage.

#![cfg(all(feature = "cpu-reference", unix))]

use personalai_engine_crypto::{TrustedSigner, hash_package};
use personalai_engine_loader::ModelRegistry;
use personalai_engine_model_format::{ParsedModelPackage, Quantization};
use personalai_engine_model_importer::{ImportSpec, SourceMapping, build_package};
use personalai_engine_placement::{
    BackendPlacementCapabilities, ExecutionPlacementPlan, ExpertByteSource, PlacementCancellation,
    PlacementCancellationProbe, PlacementError, PlacementPolicy, PlacementResourceEnvelope,
    SPARSE_EXPERT_CACHE_SLOT_BYTES, SparseExpertCache, SparseExpertDescriptor,
    SparseExpertExecutionPlan, SparseExpertId, SparseExpertLayerPlan, SparseExpertPlanIdentity,
    SparseExpertResidencyMode, SparseExpertResourceEnvelope, SparseExpertSelection,
};
use personalai_engine_quantization::QuantizationProfile;
use personalai_engine_testkit::DecoderPackageFixtureBuilder;
use personalai_engine_transformer::ScalarSparseMoeLayer;
use personalai_engine_types::{
    ContextLimit, DType, EngineErrorKind, ModelId, ResourceBudget, SignerKeyId,
};
use std::collections::BTreeMap;
use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_TEMP: AtomicU64 = AtomicU64::new(1);

struct FixtureRoot(PathBuf);

impl FixtureRoot {
    fn new() -> Self {
        let sequence = NEXT_TEMP.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir()
            .join(format!("personalai-sparse-transformer-{}-{sequence}", std::process::id()));
        fs::create_dir(&path).expect("fixture root must create");
        Self(path)
    }
}

impl Drop for FixtureRoot {
    fn drop(&mut self) {
        unseal_tree(&self.0);
        let _result = fs::remove_dir_all(&self.0);
    }
}

fn unseal_tree(path: &Path) {
    if let Ok(metadata) = fs::symlink_metadata(path) {
        if metadata.is_dir() {
            let _result = fs::set_permissions(path, fs::Permissions::from_mode(0o700));
            if let Ok(entries) = fs::read_dir(path) {
                for entry in entries.flatten() {
                    unseal_tree(&entry.path());
                }
            }
        } else {
            let _result = fs::set_permissions(path, fs::Permissions::from_mode(0o600));
        }
    }
}

fn external_name(canonical: &str) -> String {
    match canonical {
        "token_embedding.weight" => "model.embed_tokens.weight".to_owned(),
        "final_norm.weight" => "model.norm.weight".to_owned(),
        "output.weight" => "lm_head.weight".to_owned(),
        name => {
            let rest = name.strip_prefix("layers.").expect("canonical layer prefix");
            let (layer, suffix) = rest.split_once('.').expect("canonical layer suffix");
            let layer = layer.parse::<u32>().expect("canonical layer number");
            format!("model.layers.{layer}.{suffix}")
                .replace("attn_norm.weight", "input_layernorm.weight")
                .replace("ffn_norm.weight", "post_attention_layernorm.weight")
                .replace("q_proj.weight", "self_attn.q_proj.weight")
                .replace("k_proj.weight", "self_attn.k_proj.weight")
                .replace("v_proj.weight", "self_attn.v_proj.weight")
                .replace("o_proj.weight", "self_attn.o_proj.weight")
                .replace("gate_proj.weight", "mlp.gate_proj.weight")
                .replace("up_proj.weight", "mlp.up_proj.weight")
                .replace("down_proj.weight", "mlp.down_proj.weight")
        }
    }
}

fn source_and_tokenizer() -> (Vec<u8>, Vec<u8>) {
    let fixture = DecoderPackageFixtureBuilder::new().build().expect("fixture must build");
    let parsed = ParsedModelPackage::parse(fixture.bytes()).expect("fixture must parse");
    let tokenizer = parsed.tokenizer_data(fixture.bytes()).expect("tokenizer").to_vec();
    let mut records = Vec::new();
    let mut data = Vec::new();
    for tensor in parsed.tensors() {
        let bytes = parsed.tensor_data(fixture.bytes(), tensor).expect("tensor bytes");
        let start = data.len();
        data.extend_from_slice(bytes);
        let dimensions =
            tensor.shape().dimensions().iter().map(u64::to_string).collect::<Vec<_>>().join(",");
        records.push(format!(
            "\"{}\":{{\"dtype\":\"F32\",\"shape\":[{dimensions}],\"data_offsets\":[{start},{}]}}",
            external_name(tensor.name().as_str()),
            data.len()
        ));
    }
    let header = format!("{{{}}}", records.join(","));
    let mut source = Vec::new();
    source.extend_from_slice(&u64::try_from(header.len()).unwrap().to_le_bytes());
    source.extend_from_slice(header.as_bytes());
    source.extend_from_slice(&data);
    (source, tokenizer)
}

fn verified_base_plans(
    quantization_profile: QuantizationProfile,
) -> (FixtureRoot, ExecutionPlacementPlan, ExecutionPlacementPlan) {
    let root = FixtureRoot::new();
    let model_id = ModelId::new("synthetic-sparse-moe").unwrap();
    let spec = ImportSpec::new(
        model_id.clone(),
        SourceMapping::LlamaSafetensorsV1,
        ContextLimit::new(8).unwrap(),
        259,
        4,
        2,
        2,
        1,
        8,
        10_000.0,
        1.0,
        false,
        "sparse-transformer-test",
        "test-only",
        quantization_profile,
    )
    .unwrap();
    let (source, tokenizer) = source_and_tokenizer();
    let key_id = SignerKeyId::from_bytes([0x71; 16]).unwrap();
    let mut seed = [0x72; 32];
    let imported =
        build_package(&spec, "model.safetensors", &source, &tokenizer, key_id, &mut seed).unwrap();
    let package_path = root.0.join("sparse-transformer.paimodel");
    fs::create_dir(&package_path).unwrap();
    for (name, bytes) in imported.files() {
        let path = package_path.join(name);
        fs::write(&path, bytes).unwrap();
        fs::set_permissions(path, fs::Permissions::from_mode(0o400)).unwrap();
    }
    fs::set_permissions(&package_path, fs::Permissions::from_mode(0o500)).unwrap();
    fs::set_permissions(&root.0, fs::Permissions::from_mode(0o500)).unwrap();
    let signer =
        TrustedSigner::active(imported.signer_key_id(), imported.signer_public_key()).unwrap();
    let mut registry = ModelRegistry::new(&root.0).unwrap();
    registry
        .register(model_id.clone(), "sparse-transformer.paimodel", imported.package_hash(), signer)
        .unwrap();
    let budget = ResourceBudget::new(1 << 24, 1 << 20, 1).unwrap();
    let package = registry.load_registered_directory(&model_id, &budget, &mut |_event| {}).unwrap();
    let capabilities =
        BackendPlacementCapabilities::cpu_reference(budget.max_memory_bytes()).unwrap();
    let envelope = PlacementResourceEnvelope::new(
        budget.max_memory_bytes(),
        budget.max_memory_bytes(),
        budget.max_memory_bytes(),
        budget.max_kv_cache_bytes(),
    )
    .unwrap();
    let first = ExecutionPlacementPlan::derive(
        &package,
        &capabilities,
        envelope,
        PlacementPolicy::fully_resident(),
        0,
        1,
    )
    .unwrap();
    let second = ExecutionPlacementPlan::derive(
        &package,
        &capabilities,
        envelope,
        PlacementPolicy::fully_resident(),
        0,
        2,
    )
    .unwrap();
    (root, first, second)
}

#[derive(Clone)]
struct FixtureSource {
    identity: SparseExpertPlanIdentity,
    payloads: BTreeMap<SparseExpertId, Vec<u8>>,
}

impl ExpertByteSource for FixtureSource {
    fn identity(&self) -> &SparseExpertPlanIdentity {
        &self.identity
    }

    fn read_exact_expert(
        &self,
        descriptor: &SparseExpertDescriptor,
        destination: &mut [u8],
        cancellation: &dyn PlacementCancellationProbe,
    ) -> Result<usize, PlacementError> {
        if cancellation.is_cancelled() {
            return Err(PlacementError::Cancelled);
        }
        let payload =
            self.payloads.get(&descriptor.id()).ok_or(PlacementError::ArtifactIdentityMismatch)?;
        if payload.len() != destination.len() {
            return Ok(payload.len());
        }
        destination.copy_from_slice(payload);
        Ok(payload.len())
    }
}

fn encode(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|value| value.to_le_bytes()).collect()
}

fn descriptors(payloads: &[Vec<u8>]) -> Vec<SparseExpertDescriptor> {
    descriptors_with_storage(payloads, DType::F32, Quantization::None)
}

fn descriptors_with_storage(
    payloads: &[Vec<u8>],
    storage_dtype: DType,
    quantization: Quantization,
) -> Vec<SparseExpertDescriptor> {
    payloads
        .iter()
        .enumerate()
        .map(|(expert, payload)| {
            SparseExpertDescriptor::new(
                SparseExpertId::new(0, u32::try_from(expert).unwrap()).unwrap(),
                u64::try_from(payload.len()).unwrap(),
                u64::try_from(payload.len()).unwrap(),
                hash_package(payload),
                storage_dtype,
                quantization,
            )
            .unwrap()
        })
        .collect()
}

fn plan(
    base_plan: &ExecutionPlacementPlan,
    descriptors: Vec<SparseExpertDescriptor>,
    mode: SparseExpertResidencyMode,
    cache_bytes: u64,
) -> Arc<SparseExpertExecutionPlan> {
    Arc::new(
        SparseExpertExecutionPlan::derive(
            base_plan,
            vec![SparseExpertLayerPlan::new(0, 2, descriptors).unwrap()],
            mode,
            SparseExpertResourceEnvelope::new(2_048, cache_bytes, 128, 64).unwrap(),
            0,
        )
        .unwrap(),
    )
}

fn selection(
    plan: &SparseExpertExecutionPlan,
    layer: &ScalarSparseMoeLayer,
    input: &[f32],
) -> SparseExpertSelection {
    let selected = layer
        .route(input)
        .unwrap()
        .into_iter()
        .map(|routed| SparseExpertId::new(0, routed.expert()).unwrap())
        .collect();
    SparseExpertSelection::new(plan, 0, selected).unwrap()
}

fn execute_from_leases(
    layer: &ScalarSparseMoeLayer,
    input: &[f32],
    leases: &[personalai_engine_placement::SparseExpertLease],
) -> Vec<f32> {
    layer.execute_cached_f32(input, leases, &PlacementCancellation::new()).unwrap()
}

#[test]
fn fully_resident_and_streamed_expert_caches_are_bit_exact() {
    let layer =
        ScalarSparseMoeLayer::new(2, 2, 3, 2, vec![1.0, 0.0, 0.0, 2.0, -1.0, -1.0]).unwrap();
    let payloads = vec![
        encode(&[1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, -1.0, 1.0, 0.0, 0.0, 1.0]),
        encode(&[2.0, 0.0, 0.0, 2.0, 1.0, -1.0, -1.0, 1.0, 0.5, 0.0, 0.0, 0.5]),
        encode(&[-1.0, 0.0, 0.0, -1.0, 3.0, 3.0, 3.0, 3.0, 9.0, 0.0, 0.0, 9.0]),
    ];
    assert!(payloads.iter().all(|payload| payload.len() == layer.expert_payload_bytes()));
    let input = [1.0, 0.5];
    let (_root, resident_base, streamed_base) =
        verified_base_plans(QuantizationProfile::ReferenceF32);
    let resident_plan = plan(
        &resident_base,
        descriptors(&payloads),
        SparseExpertResidencyMode::FullyResident,
        3 * SPARSE_EXPERT_CACHE_SLOT_BYTES + 3 * 48,
    );
    let streamed_plan = plan(
        &streamed_base,
        descriptors(&payloads),
        SparseExpertResidencyMode::ExpertStreamed,
        3 * SPARSE_EXPERT_CACHE_SLOT_BYTES + 2 * 48,
    );
    let resident_layer = layer.clone().bind_to_plan(&resident_plan, 0).unwrap();
    let streamed_layer = layer.clone().bind_to_plan(&streamed_plan, 0).unwrap();
    let resident_source = FixtureSource {
        identity: resident_plan.identity().clone(),
        payloads: payloads
            .iter()
            .enumerate()
            .map(|(expert, payload)| {
                (SparseExpertId::new(0, u32::try_from(expert).unwrap()).unwrap(), payload.clone())
            })
            .collect(),
    };
    let streamed_source = FixtureSource {
        identity: streamed_plan.identity().clone(),
        payloads: resident_source.payloads.clone(),
    };
    let cancellation = PlacementCancellation::new();

    let resident_pool = personalai_engine_memory::MemoryPool::new(2_048).unwrap();
    let resident_cache =
        SparseExpertCache::new(Arc::clone(&resident_plan), &resident_pool).unwrap();
    resident_cache.populate_fully_resident(&resident_source, &cancellation).unwrap();
    let resident_selection = selection(&resident_plan, &layer, &input);
    let resident_leases = resident_cache
        .acquire_selection(&resident_selection, &resident_source, &cancellation)
        .unwrap();
    let resident_output = execute_from_leases(&resident_layer, &input, &resident_leases);

    let streamed_pool = personalai_engine_memory::MemoryPool::new(2_048).unwrap();
    let streamed_cache =
        SparseExpertCache::new(Arc::clone(&streamed_plan), &streamed_pool).unwrap();
    let streamed_selection = selection(&streamed_plan, &layer, &input);
    let streamed_leases = streamed_cache
        .acquire_selection(&streamed_selection, &streamed_source, &cancellation)
        .unwrap();
    let streamed_output = execute_from_leases(&streamed_layer, &input, &streamed_leases);

    assert_eq!(resident_output, streamed_output);
    assert!(resident_layer.execute_cached_f32(&input, &streamed_leases, &cancellation).is_err());
    let differently_routed_resident =
        ScalarSparseMoeLayer::new(2, 2, 3, 2, vec![9.0, 9.0, -4.0, 1.0, 2.0, -8.0])
            .unwrap()
            .bind_to_plan(&resident_plan, 0)
            .unwrap();
    assert_eq!(
        differently_routed_resident
            .execute_cached_f32(&input, &streamed_leases, &cancellation)
            .unwrap_err()
            .kind(),
        EngineErrorKind::AuthenticationFailure
    );
    assert!(layer.clone().bind_to_plan(&streamed_plan, 1).is_err());
    assert_eq!(resident_cache.metrics().unwrap().loads, 3);
    assert_eq!(resident_cache.metrics().unwrap().hits, 2);
    assert_eq!(streamed_cache.metrics().unwrap().loads, 2);
    assert_eq!(
        streamed_cache.metrics().unwrap().resident_bytes,
        3 * SPARSE_EXPERT_CACHE_SLOT_BYTES + 2 * 48
    );

    let second_streamed_pool = personalai_engine_memory::MemoryPool::new(2_048).unwrap();
    let second_streamed_cache =
        SparseExpertCache::new(Arc::clone(&streamed_plan), &second_streamed_pool).unwrap();
    let first_id = streamed_selection.experts()[0];
    let second_id = streamed_selection.experts()[1];
    let mixed_leases = vec![
        streamed_cache.acquire(first_id, &streamed_source, &cancellation).unwrap(),
        second_streamed_cache.acquire(second_id, &streamed_source, &cancellation).unwrap(),
    ];
    let mixed_error =
        streamed_layer.execute_cached_f32(&input, &mixed_leases, &cancellation).unwrap_err();
    assert_eq!(mixed_error.kind(), EngineErrorKind::AuthenticationFailure);
}

#[test]
fn scalar_sparse_oracle_rejects_divergent_inventory_from_same_base_plan() {
    let layer = ScalarSparseMoeLayer::new(1, 1, 2, 2, vec![1.0, -1.0]).unwrap();
    let first_payloads = vec![encode(&[1.0, 2.0, 3.0]), encode(&[-1.0, 2.0, 3.0])];
    let second_payloads = vec![encode(&[9.0, 2.0, 3.0]), encode(&[-1.0, 2.0, 3.0])];
    let input = [1.0];
    let (_root, base, _) = verified_base_plans(QuantizationProfile::ReferenceF32);
    let cache_bytes = 2 * SPARSE_EXPERT_CACHE_SLOT_BYTES + 24;
    let first_plan = plan(
        &base,
        descriptors(&first_payloads),
        SparseExpertResidencyMode::ExpertStreamed,
        cache_bytes,
    );
    let second_plan = plan(
        &base,
        descriptors(&second_payloads),
        SparseExpertResidencyMode::ExpertStreamed,
        cache_bytes,
    );
    assert_eq!(first_plan.identity(), second_plan.identity());
    assert_ne!(first_plan.plan_digest(), second_plan.plan_digest());

    let source = FixtureSource {
        identity: second_plan.identity().clone(),
        payloads: second_payloads
            .iter()
            .enumerate()
            .map(|(expert, payload)| {
                (SparseExpertId::new(0, u32::try_from(expert).unwrap()).unwrap(), payload.clone())
            })
            .collect(),
    };
    let pool = personalai_engine_memory::MemoryPool::new(2_048).unwrap();
    let cache = SparseExpertCache::new(Arc::clone(&second_plan), &pool).unwrap();
    let leases = cache
        .acquire_selection(
            &selection(&second_plan, &layer, &input),
            &source,
            &PlacementCancellation::new(),
        )
        .unwrap();
    let bound = layer.bind_to_plan(&first_plan, 0).unwrap();
    assert_eq!(
        bound
            .execute_cached_f32(&input, &leases, &PlacementCancellation::new())
            .unwrap_err()
            .kind(),
        EngineErrorKind::AuthenticationFailure
    );
}

#[test]
fn scalar_sparse_oracle_rejects_unsupported_storage_encodings_explicitly() {
    let layer = ScalarSparseMoeLayer::new(1, 1, 2, 2, vec![1.0, -1.0]).unwrap();
    let payloads = vec![encode(&[1.0, 2.0, 3.0]), encode(&[-1.0, 2.0, 3.0])];
    let input = [1.0];

    for dtype in [DType::F16, DType::BFloat16] {
        let (_root, base, _) = verified_base_plans(QuantizationProfile::ReferenceF32);
        let plan = plan(
            &base,
            descriptors_with_storage(&payloads, dtype, Quantization::None),
            SparseExpertResidencyMode::ExpertStreamed,
            2 * SPARSE_EXPERT_CACHE_SLOT_BYTES + 24,
        );
        let pool = personalai_engine_memory::MemoryPool::new(2_048).unwrap();
        let cache = SparseExpertCache::new(Arc::clone(&plan), &pool).unwrap();
        let source = FixtureSource {
            identity: plan.identity().clone(),
            payloads: payloads
                .iter()
                .enumerate()
                .map(|(expert, payload)| {
                    (
                        SparseExpertId::new(0, u32::try_from(expert).unwrap()).unwrap(),
                        payload.clone(),
                    )
                })
                .collect(),
        };
        let selection = selection(&plan, &layer, &input);
        let leases =
            cache.acquire_selection(&selection, &source, &PlacementCancellation::new()).unwrap();
        let bound = layer.clone().bind_to_plan(&plan, 0).unwrap();
        let error =
            bound.execute_cached_f32(&input, &leases, &PlacementCancellation::new()).unwrap_err();
        assert_eq!(error.kind(), EngineErrorKind::NotImplemented);
    }

    for profile in [QuantizationProfile::BalancedQ8, QuantizationProfile::CompactQ4] {
        let (_root, base, _) = verified_base_plans(profile);
        let plan = plan(
            &base,
            descriptors_with_storage(&payloads, DType::F32, Quantization::None),
            SparseExpertResidencyMode::ExpertStreamed,
            2 * SPARSE_EXPERT_CACHE_SLOT_BYTES + 24,
        );
        let pool = personalai_engine_memory::MemoryPool::new(2_048).unwrap();
        let cache = SparseExpertCache::new(Arc::clone(&plan), &pool).unwrap();
        let source = FixtureSource {
            identity: plan.identity().clone(),
            payloads: payloads
                .iter()
                .enumerate()
                .map(|(expert, payload)| {
                    (
                        SparseExpertId::new(0, u32::try_from(expert).unwrap()).unwrap(),
                        payload.clone(),
                    )
                })
                .collect(),
        };
        let selection = selection(&plan, &layer, &input);
        let leases =
            cache.acquire_selection(&selection, &source, &PlacementCancellation::new()).unwrap();
        let bound = layer.clone().bind_to_plan(&plan, 0).unwrap();
        let error =
            bound.execute_cached_f32(&input, &leases, &PlacementCancellation::new()).unwrap_err();
        assert_eq!(error.kind(), EngineErrorKind::NotImplemented);
    }
}
