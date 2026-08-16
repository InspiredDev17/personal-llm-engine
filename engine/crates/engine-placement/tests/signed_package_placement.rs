//! Signed-package integration coverage for trusted execution placement.

#![cfg(unix)]

use personalai_engine_crypto::TrustedSigner;
use personalai_engine_loader::ModelRegistry;
use personalai_engine_memory::MemoryPool;
use personalai_engine_model_format::ParsedModelPackage;
use personalai_engine_model_importer::{ImportSpec, SourceMapping, build_package};
use personalai_engine_placement::{
    BackendPlacementCapabilities, ExecutionPlacementPlan, MemoryTopology, NoPlacementFaults,
    PlacementCancellation, PlacementError, PlacementPolicy, PlacementResourceEnvelope,
    PlacementStageKind, WeightResidencyMode, run_layer_streamed,
};
use personalai_engine_quantization::QuantizationProfile;
use personalai_engine_testkit::DecoderPackageFixtureBuilder;
use personalai_engine_types::{ContextLimit, ModelId, ResourceBudget, SignerKeyId};
use std::fs;
use std::io::{Read, Seek, SeekFrom, Write};
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
            .join(format!("personalai-placement-{}-{sequence}", std::process::id()));
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

fn verified_package()
-> (FixtureRoot, Arc<personalai_engine_loader::VerifiedDirectoryPackage>, ResourceBudget) {
    let root = FixtureRoot::new();
    let model_id = ModelId::new("placement-fixture").unwrap();
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
        "placement-test",
        "test-only",
        QuantizationProfile::ReferenceF32,
    )
    .unwrap();
    let (source, tokenizer) = source_and_tokenizer();
    let key_id = SignerKeyId::from_bytes([0x61; 16]).unwrap();
    let mut seed = [0x62; 32];
    let imported =
        build_package(&spec, "model.safetensors", &source, &tokenizer, key_id, &mut seed).unwrap();
    let package_name = "placement-fixture.paimodel";
    let package_path = root.0.join(package_name);
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
    registry.register(model_id.clone(), package_name, imported.package_hash(), signer).unwrap();
    let budget = ResourceBudget::new(1 << 24, 1 << 20, 1).unwrap();
    let package = registry.load_registered_directory(&model_id, &budget, &mut |_event| {}).unwrap();
    (root, Arc::new(package), budget)
}

fn envelope(budget: ResourceBudget) -> PlacementResourceEnvelope {
    PlacementResourceEnvelope::new(
        budget.max_memory_bytes(),
        budget.max_memory_bytes(),
        budget.max_memory_bytes(),
        budget.max_kv_cache_bytes(),
    )
    .unwrap()
}

#[test]
fn signed_package_derives_exact_resident_and_streamed_plans() {
    let (_root, package, budget) = verified_package();
    let capabilities =
        BackendPlacementCapabilities::cpu_reference(budget.max_memory_bytes()).unwrap();
    let resident = ExecutionPlacementPlan::derive(
        &package,
        &capabilities,
        envelope(budget),
        PlacementPolicy::fully_resident(),
        0,
        1,
    )
    .unwrap();
    assert_eq!(resident.mode(), WeightResidencyMode::FullyResident);
    assert_eq!(resident.backend_id().as_str(), "cpu-reference");
    assert_eq!(resident.model_id(), package.registration().model_id());
    assert_eq!(resident.package_digest(), *package.registration().expected_hash());
    assert_eq!(resident.stages().len(), 4);
    assert_eq!(resident.stages()[1].kind(), PlacementStageKind::DecoderLayer(0));
    assert_eq!(resident.stages()[2].kind(), PlacementStageKind::DecoderLayer(1));
    assert!(resident.complete_execution_bytes() > 0);
    assert_eq!(resident.guaranteed_staging_bytes(), 0);

    let streamed = ExecutionPlacementPlan::derive(
        &package,
        &capabilities,
        envelope(budget),
        PlacementPolicy::layer_streamed(),
        0,
        2,
    )
    .unwrap();
    assert_eq!(streamed.mode(), WeightResidencyMode::LayerStreamed);
    assert_eq!(streamed.prefetch_depth(), 1);
    assert!(streamed.guaranteed_staging_bytes() < streamed.complete_execution_bytes());
}

#[test]
fn signed_package_streams_in_order_with_bounded_accounted_staging() {
    let (_root, package, budget) = verified_package();
    let capabilities =
        BackendPlacementCapabilities::cpu_reference(budget.max_memory_bytes()).unwrap();
    let plan = ExecutionPlacementPlan::derive(
        &package,
        &capabilities,
        envelope(budget),
        PlacementPolicy::layer_streamed(),
        0,
        1,
    )
    .unwrap();
    let pool = MemoryPool::new(plan.guaranteed_staging_bytes()).unwrap();
    let mut order = Vec::new();
    let metrics = run_layer_streamed(
        &plan,
        &package,
        &pool,
        &PlacementCancellation::new(),
        &NoPlacementFaults,
        &mut |_event| {},
        |stage, tensors| {
            assert!(!tensors.tensors().is_empty());
            order.push(stage.ordinal());
            Ok(())
        },
    )
    .unwrap();
    assert_eq!(order, vec![0, 1, 2, 3]);
    assert_eq!(metrics.stages_executed, 4);
    assert_eq!(metrics.prefetch_started, 3);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[test]
fn unsupported_hybrid_and_cancelled_streaming_fail_explicitly() {
    let (_root, package, budget) = verified_package();
    let capabilities = BackendPlacementCapabilities::new(
        personalai_engine_types::BackendId::new("metal").unwrap(),
        MemoryTopology::Unified,
        budget.max_memory_bytes(),
        budget.max_memory_bytes(),
        false,
    )
    .unwrap();
    let hybrid = ExecutionPlacementPlan::derive(
        &package,
        &capabilities,
        envelope(budget),
        PlacementPolicy::new(WeightResidencyMode::HybridResident, 0).unwrap(),
        0,
        1,
    );
    assert!(matches!(hybrid, Err(PlacementError::UnsupportedPlacement { .. })));

    let cpu = BackendPlacementCapabilities::cpu_reference(budget.max_memory_bytes()).unwrap();
    let plan = ExecutionPlacementPlan::derive(
        &package,
        &cpu,
        envelope(budget),
        PlacementPolicy::layer_streamed(),
        0,
        1,
    )
    .unwrap();
    let cancellation = PlacementCancellation::new();
    cancellation.cancel();
    let result = run_layer_streamed(
        &plan,
        &package,
        &MemoryPool::new(plan.guaranteed_staging_bytes()).unwrap(),
        &cancellation,
        &NoPlacementFaults,
        &mut |_event| {},
        |_stage, _tensors| Ok(()),
    );
    assert_eq!(result, Err(PlacementError::Cancelled));
}

#[test]
fn mapped_package_mutation_after_verification_fails_before_stage_exposure() {
    let (root, package, budget) = verified_package();
    let capabilities =
        BackendPlacementCapabilities::cpu_reference(budget.max_memory_bytes()).unwrap();
    let plan = ExecutionPlacementPlan::derive(
        &package,
        &capabilities,
        envelope(budget),
        PlacementPolicy::layer_streamed(),
        0,
        1,
    )
    .unwrap();
    let shard = root.0.join("placement-fixture.paimodel/tensors-0000.aipt");
    fs::set_permissions(&shard, fs::Permissions::from_mode(0o600)).unwrap();
    let mut file = fs::OpenOptions::new().read(true).write(true).open(&shard).unwrap();
    let mut byte = [0_u8; 1];
    file.read_exact(&mut byte).unwrap();
    file.seek(SeekFrom::Start(0)).unwrap();
    byte[0] ^= 1;
    file.write_all(&byte).unwrap();
    file.sync_all().unwrap();
    fs::set_permissions(&shard, fs::Permissions::from_mode(0o400)).unwrap();

    let mut events = Vec::new();
    let result = run_layer_streamed(
        &plan,
        &package,
        &MemoryPool::new(plan.guaranteed_staging_bytes()).unwrap(),
        &PlacementCancellation::new(),
        &NoPlacementFaults,
        &mut |event| events.push(event),
        |_stage, _tensors| Ok(()),
    );
    assert_eq!(result, Err(PlacementError::ArtifactIdentityMismatch));
    assert_eq!(events.len(), 1);
}
