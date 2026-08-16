//! Signed-package parity and cancellation tests for streamed scalar execution.

use personalai_engine_crypto::TrustedSigner;
use personalai_engine_loader::{ModelRegistry, VerifiedDirectoryPackage};
use personalai_engine_memory::MemoryPool;
use personalai_engine_model_format::ParsedModelPackage;
use personalai_engine_model_importer::{ImportSpec, SourceMapping, build_package};
use personalai_engine_placement::{
    BackendPlacementCapabilities, ExecutionPlacementPlan, NoPlacementFaults, PlacementCancellation,
    PlacementFaultInjector, PlacementPolicy, PlacementResourceEnvelope,
};
use personalai_engine_quantization::QuantizationProfile;
use personalai_engine_testkit::DecoderPackageFixtureBuilder;
use personalai_engine_transformer::{CpuDecoderModel, TransformerConfig};
use personalai_engine_types::{
    ContextLimit, EngineError, ModelId, RequestId, ResourceBudget, SequenceId, SignerKeyId,
};
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
            .join(format!("personalai-streamed-decoder-{}-{sequence}", std::process::id()));
        fs::create_dir(&path).expect("fixture root must create");
        Self(path)
    }
}

impl Drop for FixtureRoot {
    fn drop(&mut self) {
        let _ = make_tree_owner_writable(&self.0);
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn make_tree_owner_writable(path: &Path) -> std::io::Result<()> {
    if !path.exists() {
        return Ok(());
    }
    let metadata = fs::symlink_metadata(path)?;
    if metadata.is_dir() {
        fs::set_permissions(path, fs::Permissions::from_mode(0o700))?;
        for entry in fs::read_dir(path)? {
            make_tree_owner_writable(&entry?.path())?;
        }
    } else {
        fs::set_permissions(path, fs::Permissions::from_mode(0o600))?;
    }
    Ok(())
}

fn external_name(canonical: &str) -> String {
    if canonical == "token_embedding.weight" {
        return "model.embed_tokens.weight".to_owned();
    }
    if canonical == "final_norm.weight" {
        return "model.norm.weight".to_owned();
    }
    if canonical == "output.weight" {
        return "lm_head.weight".to_owned();
    }
    let suffixes = [
        ("attn_norm.weight", "input_layernorm.weight"),
        ("q_proj.weight", "self_attn.q_proj.weight"),
        ("k_proj.weight", "self_attn.k_proj.weight"),
        ("v_proj.weight", "self_attn.v_proj.weight"),
        ("o_proj.weight", "self_attn.o_proj.weight"),
        ("ffn_norm.weight", "post_attention_layernorm.weight"),
        ("gate_proj.weight", "mlp.gate_proj.weight"),
        ("up_proj.weight", "mlp.up_proj.weight"),
        ("down_proj.weight", "mlp.down_proj.weight"),
    ];
    for (canonical_suffix, external_suffix) in suffixes {
        if let Some(prefix) = canonical.strip_suffix(canonical_suffix) {
            let layer = prefix.trim_start_matches("layers.").trim_end_matches('.');
            let index = layer.parse::<u32>().expect("canonical fixture layer");
            return format!("model.layers.{index}.{external_suffix}");
        }
    }
    panic!("unexpected fixture tensor name")
}

fn source_and_tokenizer() -> (Vec<u8>, Vec<u8>) {
    let fixture = DecoderPackageFixtureBuilder::new().build().expect("fixture");
    let parsed = ParsedModelPackage::parse(fixture.bytes()).expect("parsed fixture");
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
    source.extend_from_slice(&(header.len() as u64).to_le_bytes());
    source.extend_from_slice(header.as_bytes());
    source.extend_from_slice(&data);
    (source, tokenizer)
}

fn verified_package() -> (FixtureRoot, Arc<VerifiedDirectoryPackage>, ResourceBudget) {
    let root = FixtureRoot::new();
    let model_id = ModelId::new("streamed-decoder-fixture").unwrap();
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
        "streamed-decoder-test",
        "test-only",
        QuantizationProfile::ReferenceF32,
    )
    .unwrap();
    let (source, tokenizer) = source_and_tokenizer();
    let key_id = SignerKeyId::from_bytes([0x71; 16]).unwrap();
    let mut seed = [0x72; 32];
    let imported =
        build_package(&spec, "model.safetensors", &source, &tokenizer, key_id, &mut seed).unwrap();
    let package_name = "streamed-decoder-fixture.paimodel";
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
    let budget = ResourceBudget::new(1 << 24, 1 << 20, 2).unwrap();
    let package = registry.load_registered_directory(&model_id, &budget, &mut |_event| {}).unwrap();
    (root, Arc::new(package), budget)
}

fn streamed_plan(
    package: &VerifiedDirectoryPackage,
    budget: ResourceBudget,
) -> Arc<ExecutionPlacementPlan> {
    let capabilities =
        BackendPlacementCapabilities::cpu_reference(budget.max_memory_bytes()).unwrap();
    let envelope = PlacementResourceEnvelope::new(
        budget.max_memory_bytes(),
        budget.max_memory_bytes(),
        budget.max_memory_bytes(),
        budget.max_kv_cache_bytes(),
    )
    .unwrap();
    Arc::new(
        ExecutionPlacementPlan::derive(
            package,
            &capabilities,
            envelope,
            PlacementPolicy::layer_streamed(),
            0,
            1,
        )
        .unwrap(),
    )
}

#[test]
fn streamed_prefill_and_decode_are_bit_exact_with_fully_resident_scalar() {
    let (_root, package, budget) = verified_package();
    let config = TransformerConfig::from_directory_manifest(package.manifest()).unwrap();

    let resident_pool = MemoryPool::from_budget(&budget).unwrap();
    let resident_weights = package.materialize_tensors(&resident_pool).unwrap().into_tensors();
    let resident = CpuDecoderModel::new_registered(
        package.registration().model_id().clone(),
        config.clone(),
        resident_weights,
        &resident_pool,
    )
    .unwrap();

    let plan = streamed_plan(&package, budget);
    let undersized_pool = MemoryPool::new(plan.required_envelope_bytes() - 1).unwrap();
    let undersized = CpuDecoderModel::new_registered_streamed(
        package.registration().model_id().clone(),
        config.clone(),
        Arc::clone(&package),
        Arc::clone(&plan),
        &undersized_pool,
    );
    assert!(matches!(undersized, Err(EngineError::ResourceExhaustion(_))));

    let streamed_pool = MemoryPool::from_budget(&budget).unwrap();
    let streamed = CpuDecoderModel::new_registered_streamed(
        package.registration().model_id().clone(),
        config,
        Arc::clone(&package),
        plan,
        &streamed_pool,
    )
    .unwrap();
    let mut resident_session = resident
        .start_session(&budget, RequestId::new(11).unwrap(), SequenceId::new(11).unwrap())
        .unwrap();
    let mut streamed_session = streamed
        .start_session(&budget, RequestId::new(12).unwrap(), SequenceId::new(12).unwrap())
        .unwrap();
    let cancellation = PlacementCancellation::new();
    for token in [256, 65, 66] {
        let expected = resident_session.forward_token(token).unwrap();
        let actual = streamed_session
            .forward_token_streamed(token, &cancellation, &NoPlacementFaults, &mut |_event| {})
            .unwrap();
        assert_eq!(actual.as_bytes(), expected.as_bytes());
    }
    assert_eq!(streamed_session.token_count(), 3);
}

struct CancelAtFirstLayer(PlacementCancellation);

impl PlacementFaultInjector for CancelAtFirstLayer {
    fn before_materialize(
        &self,
        stage: &personalai_engine_placement::PlacementStage,
    ) -> Result<(), EngineError> {
        if matches!(stage.kind(), personalai_engine_placement::PlacementStageKind::DecoderLayer(0))
        {
            self.0.cancel();
        }
        Ok(())
    }
}

#[test]
fn cancellation_during_layer_materialization_invalidates_partial_kv_state() {
    let (_root, package, budget) = verified_package();
    let config = TransformerConfig::from_directory_manifest(package.manifest()).unwrap();
    let pool = MemoryPool::from_budget(&budget).unwrap();
    let model = CpuDecoderModel::new_registered_streamed(
        package.registration().model_id().clone(),
        config,
        Arc::clone(&package),
        streamed_plan(&package, budget),
        &pool,
    )
    .unwrap();
    let mut session = model
        .start_session(&budget, RequestId::new(21).unwrap(), SequenceId::new(21).unwrap())
        .unwrap();
    let cancellation = PlacementCancellation::new();
    let result = session.forward_token_streamed(
        256,
        &cancellation,
        &CancelAtFirstLayer(cancellation.clone()),
        &mut |_event| {},
    );
    assert!(matches!(result, Err(EngineError::Cancellation(_))));
    assert_eq!(session.token_count(), 0);
    let retry = session.forward_token_streamed(
        256,
        &PlacementCancellation::new(),
        &NoPlacementFaults,
        &mut |_event| {},
    );
    assert!(retry.is_err(), "a partially written sequence must not become usable again");
}
