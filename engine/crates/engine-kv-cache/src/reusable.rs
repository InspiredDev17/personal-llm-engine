//! Immutable reusable prefix K/V state and bounded cache lifecycle.
//!
//! This module deliberately implements exact token-prefix reuse only. It has no
//! retrieval, semantic matching, document ordering, or durable-memory policy.

use crate::{ContiguousKvCache, SequenceHandle};
use personalai_engine_crypto::hash_package;
use personalai_engine_knowledge_contract::{
    ArtifactDigest, CacheAuthorizationScope, CacheBinding, KnowledgeIdentity,
};
use personalai_engine_memory::{MemoryOwner, MemoryPool, MemoryReservation};
use personalai_engine_types::{
    ArchitectureId, BackendId, DType, EngineError, MAX_CONTEXT_TOKENS,
    MAX_REUSABLE_CACHE_TTL_SECONDS, ModelHashAlgorithm, ModelId, ModelPackageHash, Sha256Digest,
};
use std::collections::BTreeMap;
use std::fmt;
use std::sync::{Arc, Mutex, MutexGuard};

/// First reusable prefix cache representation understood by this engine.
pub const REUSABLE_KV_CACHE_FORMAT_VERSION: u16 = 1;
/// Maximum live reusable cache objects in one model store.
pub const MAX_REUSABLE_KV_CACHES: u32 = 1_024;
/// Conservative accounted metadata bytes per reusable cache entry.
const CACHE_METADATA_BYTES: u64 = 768;

/// Deterministic content-derived reusable cache identifier.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct KvCacheId(ModelPackageHash);

impl KvCacheId {
    /// Constructs an identifier from an exact SHA-384 value.
    pub const fn from_hash(hash: ModelPackageHash) -> Self {
        Self(hash)
    }

    /// Returns the algorithm-tagged identifier hash.
    pub const fn as_hash(self) -> ModelPackageHash {
        self.0
    }

    /// Parses exactly 96 lowercase or uppercase hexadecimal characters.
    pub fn from_hex(value: &str) -> Result<Self, EngineError> {
        ModelPackageHash::from_hex(ModelHashAlgorithm::Sha384, value).map(Self)
    }
}

impl fmt::Debug for KvCacheId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.debug_tuple("KvCacheId").field(&self.0.to_string()).finish()
    }
}

impl fmt::Display for KvCacheId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}", self.0)
    }
}

/// Closed positional semantics for cached attention state.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum AttentionPositionSemantics {
    /// Zero-based causal positions with the next token at `cached_token_count`.
    CausalZeroBasedV1 = 1,
}

/// Exact model/runtime facts that must match before cache reuse.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct KvCacheCompatibility {
    model_id: ModelId,
    model_artifact_hash: ModelPackageHash,
    tokenizer_hash: ModelPackageHash,
    tokenizer_sha256: Sha256Digest,
    architecture_id: ArchitectureId,
    backend_id: BackendId,
    attention_positions: AttentionPositionSemantics,
    rope_theta_bits: u32,
    rope_scaling_bits: u32,
    kv_dtype: DType,
}

impl KvCacheCompatibility {
    /// Constructs an exact compatibility descriptor.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        model_id: ModelId,
        model_artifact_hash: ModelPackageHash,
        tokenizer_hash: ModelPackageHash,
        tokenizer_sha256: Sha256Digest,
        architecture_id: ArchitectureId,
        backend_id: BackendId,
        attention_positions: AttentionPositionSemantics,
        rope_theta: f32,
        rope_scaling: f32,
        kv_dtype: DType,
    ) -> Result<Self, EngineError> {
        if !rope_theta.is_finite() || rope_theta <= 1.0 {
            return Err(EngineError::invalid_input("cache RoPE theta is invalid"));
        }
        if !rope_scaling.is_finite() || rope_scaling <= 0.0 {
            return Err(EngineError::invalid_input("cache RoPE scaling is invalid"));
        }
        if !matches!(kv_dtype, DType::F32 | DType::F16 | DType::BFloat16) {
            return Err(EngineError::not_implemented("cache K/V dtype is unsupported"));
        }
        Ok(Self {
            model_id,
            model_artifact_hash,
            tokenizer_hash,
            tokenizer_sha256,
            architecture_id,
            backend_id,
            attention_positions,
            rope_theta_bits: rope_theta.to_bits(),
            rope_scaling_bits: rope_scaling.to_bits(),
            kv_dtype,
        })
    }

    /// Returns the exact model identity.
    pub const fn model_id(&self) -> &ModelId {
        &self.model_id
    }

    /// Returns the exact verified model artifact hash.
    pub const fn model_artifact_hash(&self) -> ModelPackageHash {
        self.model_artifact_hash
    }

    /// Returns the exact tokenizer data hash.
    pub const fn tokenizer_hash(&self) -> ModelPackageHash {
        self.tokenizer_hash
    }

    /// Returns the canonical SHA-256 tokenizer digest.
    pub const fn tokenizer_sha256(&self) -> Sha256Digest {
        self.tokenizer_sha256
    }

    /// Returns the reviewed architecture identity.
    pub const fn architecture_id(&self) -> &ArchitectureId {
        &self.architecture_id
    }

    /// Returns the exact compatible execution backend.
    pub const fn backend_id(&self) -> &BackendId {
        &self.backend_id
    }

    /// Returns attention-position semantics.
    pub const fn attention_positions(&self) -> AttentionPositionSemantics {
        self.attention_positions
    }

    /// Returns the exact RoPE theta.
    pub const fn rope_theta(&self) -> f32 {
        f32::from_bits(self.rope_theta_bits)
    }

    /// Returns the exact RoPE scaling value.
    pub const fn rope_scaling(&self) -> f32 {
        f32::from_bits(self.rope_scaling_bits)
    }

    /// Returns the K/V storage type.
    pub const fn kv_dtype(&self) -> DType {
        self.kv_dtype
    }
}

/// Deterministic exact-prefix lookup key.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct KvCacheKey {
    compatibility: KvCacheCompatibility,
    binding: CacheBinding,
    ordered_token_hash: ModelPackageHash,
    token_count: u32,
}

impl KvCacheKey {
    /// Derives a key from exact ordered token IDs.
    pub fn new(
        compatibility: KvCacheCompatibility,
        binding: CacheBinding,
        tokens: &[u32],
    ) -> Result<Self, EngineError> {
        let token_count = u32::try_from(tokens.len())
            .map_err(|_| EngineError::resource_exhaustion("cache token count exceeds u32"))?;
        if token_count == 0 || token_count > MAX_CONTEXT_TOKENS {
            return Err(EngineError::invalid_input("cache token count is outside hard bounds"));
        }
        if binding.token_count() != token_count
            || binding.runtime().provider_model_id() != compatibility.model_id()
            || binding.architecture_id() != compatibility.architecture_id()
            || binding.runtime().backend_id() != compatibility.backend_id()
            || binding.tokenizer_digest() != compatibility.tokenizer_sha256()
            || binding.kv_dtype() != compatibility.kv_dtype()
            || binding.rope_semantics_id().as_str() != "rope.causal-zero-based-v1"
            || !matches!(
                binding.runtime().artifact_digest(),
                ArtifactDigest::Sha384(hash) if hash == compatibility.model_artifact_hash()
            )
        {
            return Err(EngineError::authentication_failure(
                "canonical cache binding differs from loaded execution identity",
            ));
        }
        Ok(Self { compatibility, binding, ordered_token_hash: hash_tokens(tokens)?, token_count })
    }

    /// Returns the exact compatibility descriptor.
    pub const fn compatibility(&self) -> &KvCacheCompatibility {
        &self.compatibility
    }

    /// Returns the exact authorization scope.
    pub const fn binding(&self) -> &CacheBinding {
        &self.binding
    }

    /// Returns the exact authorization scope.
    pub const fn scope(&self) -> &CacheAuthorizationScope {
        self.binding.scope()
    }

    /// Returns the ordered token-sequence hash.
    pub const fn ordered_token_hash(&self) -> ModelPackageHash {
        self.ordered_token_hash
    }

    /// Returns the cached prefix length.
    pub const fn token_count(&self) -> u32 {
        self.token_count
    }

    /// Derives the stable content-based cache identifier.
    pub fn cache_id(&self) -> Result<KvCacheId, EngineError> {
        Ok(KvCacheId::from_hash(hash_package(&encode_key(self)?)))
    }
}

/// Lifecycle of a reusable cache object.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KvCacheState {
    /// Immutable and eligible for compatible attachment.
    Sealed,
}

/// Physical residency tier. Only CPU RAM is implemented in V1.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum KvCacheLocation {
    /// Accounted process-local CPU memory.
    CpuRam = 1,
    /// Trusted GPU memory, reserved for a later backend implementation.
    GpuMemory = 2,
    /// Protected local persistent storage, reserved for a later format.
    LocalSsd = 3,
}

/// Bounded retention controls supplied by authenticated control-plane policy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CacheRetentionPolicy {
    ttl_seconds: Option<u64>,
    eviction_priority: u8,
    pinned: bool,
}

impl CacheRetentionPolicy {
    /// Constructs a retention policy. TTL must be between one second and 365 days.
    pub fn new(
        ttl_seconds: Option<u64>,
        eviction_priority: u8,
        pinned: bool,
    ) -> Result<Self, EngineError> {
        if ttl_seconds.is_some_and(|ttl| ttl == 0 || ttl > MAX_REUSABLE_CACHE_TTL_SECONDS) {
            return Err(EngineError::invalid_input("cache TTL is outside its bounded range"));
        }
        Ok(Self { ttl_seconds, eviction_priority, pinned })
    }

    /// Returns a conservative evictable explicit-retention policy.
    pub const fn explicit() -> Self {
        Self { ttl_seconds: None, eviction_priority: 128, pinned: false }
    }

    /// Returns optional TTL seconds.
    pub const fn ttl_seconds(self) -> Option<u64> {
        self.ttl_seconds
    }

    /// Returns deterministic eviction priority; lower values evict first.
    pub const fn eviction_priority(self) -> u8 {
        self.eviction_priority
    }

    /// Returns whether pressure eviction is prohibited.
    pub const fn pinned(self) -> bool {
        self.pinned
    }
}

/// Immutable public metadata for one reusable cache.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KvCacheDescriptor {
    cache_id: KvCacheId,
    key: KvCacheKey,
    state: KvCacheState,
    location: KvCacheLocation,
    byte_size: u64,
    created_at_unix_seconds: u64,
    last_access_unix_seconds: u64,
    hit_count: u64,
    retention: CacheRetentionPolicy,
    resident: bool,
    transfer_cost_nanoseconds: Option<u64>,
}

/// Deterministic result of sealing a reusable cache key.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KvCacheCreateOutcome {
    descriptor: KvCacheDescriptor,
    reused: bool,
}

impl KvCacheCreateOutcome {
    /// Returns the exact immutable cache descriptor.
    pub const fn descriptor(&self) -> &KvCacheDescriptor {
        &self.descriptor
    }

    /// Returns whether the exact pre-existing cache object was reused.
    pub const fn reused(&self) -> bool {
        self.reused
    }
}

impl std::ops::Deref for KvCacheCreateOutcome {
    type Target = KvCacheDescriptor;

    fn deref(&self) -> &Self::Target {
        &self.descriptor
    }
}

impl KvCacheDescriptor {
    /// Returns the deterministic cache ID.
    pub const fn cache_id(&self) -> KvCacheId {
        self.cache_id
    }

    /// Returns the exact compatibility and prefix key.
    pub const fn key(&self) -> &KvCacheKey {
        &self.key
    }

    /// Returns the sealed state.
    pub const fn state(&self) -> KvCacheState {
        self.state
    }

    /// Returns the actual residency tier.
    pub const fn location(&self) -> KvCacheLocation {
        self.location
    }

    /// Returns accounted reusable-cache bytes.
    pub const fn byte_size(&self) -> u64 {
        self.byte_size
    }

    /// Returns trusted creation time.
    pub const fn created_at_unix_seconds(&self) -> u64 {
        self.created_at_unix_seconds
    }

    /// Returns the last trusted access time.
    pub const fn last_access_unix_seconds(&self) -> u64 {
        self.last_access_unix_seconds
    }

    /// Returns successful attachment/lookup count.
    pub const fn hit_count(&self) -> u64 {
        self.hit_count
    }

    /// Returns the retention policy.
    pub const fn retention(&self) -> CacheRetentionPolicy {
        self.retention
    }

    /// Returns whether bytes are currently resident.
    pub const fn resident(&self) -> bool {
        self.resident
    }

    /// Returns a measured transfer cost when one exists; CPU V1 is zero.
    pub const fn transfer_cost_nanoseconds(&self) -> Option<u64> {
        self.transfer_cost_nanoseconds
    }
}

/// Exact or shorter-prefix cache hit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KvCacheHitKind {
    /// Requested tokens exactly equal the sealed cached prefix.
    ExactPrefix,
    /// A shorter sealed prefix exactly matches the beginning of the request.
    PartialPrefix,
}

/// Successful deterministic lookup result.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KvCacheHit {
    descriptor: KvCacheDescriptor,
    kind: KvCacheHitKind,
    reused_tokens: u32,
}

impl KvCacheHit {
    /// Returns cache metadata.
    pub const fn descriptor(&self) -> &KvCacheDescriptor {
        &self.descriptor
    }

    /// Returns exact or partial-prefix classification.
    pub const fn kind(&self) -> KvCacheHitKind {
        self.kind
    }

    /// Returns the number of prefix tokens reused.
    pub const fn reused_tokens(&self) -> u32 {
        self.reused_tokens
    }
}

/// Deterministic cache miss classification without sensitive content.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KvCacheMiss {
    /// No sealed compatible prefix exists.
    NoCompatiblePrefix,
    /// The logical cache ID is unknown or expired.
    UnknownCache,
    /// Authenticated scope does not own the cache.
    ScopeMismatch,
    /// Exact model/runtime compatibility differs.
    CompatibilityMismatch,
    /// Expected token count differs from the sealed prefix.
    TokenCountMismatch,
}

/// Lookup result preserving explicit hit and miss semantics.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum KvCacheLookup {
    /// Longest compatible exact prefix.
    Hit(Box<KvCacheHit>),
    /// Deterministic miss.
    Miss(KvCacheMiss),
}

/// Cache reuse behavior accepted by trusted internal APIs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CacheReusePolicy {
    /// Do not perform reusable cache lookup.
    Disabled,
    /// Require an exact full-prefix object.
    ExactOnly,
    /// Select the longest sealed exact token prefix.
    LongestExactPrefix,
    /// Reserved extension; always unsupported in V1.
    SelectiveCacheRecompositionDisabled,
}

/// Bounded model-local reusable-cache limits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ReusableKvCacheLimits {
    maximum_entries: u32,
    maximum_bytes: u64,
    active_sequence_headroom_bytes: u64,
}

impl ReusableKvCacheLimits {
    /// Constructs limits that preserve active-sequence memory headroom.
    pub fn new(
        maximum_entries: u32,
        maximum_bytes: u64,
        active_sequence_headroom_bytes: u64,
        pool_limit_bytes: u64,
    ) -> Result<Self, EngineError> {
        if maximum_entries == 0 || maximum_entries > MAX_REUSABLE_KV_CACHES {
            return Err(EngineError::invalid_input("reusable cache entry limit is invalid"));
        }
        if maximum_bytes == 0
            || active_sequence_headroom_bytes == 0
            || maximum_bytes
                .checked_add(active_sequence_headroom_bytes)
                .is_none_or(|total| total > pool_limit_bytes)
        {
            return Err(EngineError::invalid_input(
                "reusable cache bytes do not preserve active-sequence headroom",
            ));
        }
        Ok(Self { maximum_entries, maximum_bytes, active_sequence_headroom_bytes })
    }

    /// Returns maximum live cache objects.
    pub const fn maximum_entries(self) -> u32 {
        self.maximum_entries
    }

    /// Returns maximum reusable cache bytes.
    pub const fn maximum_bytes(self) -> u64 {
        self.maximum_bytes
    }

    /// Returns reserved active-sequence headroom.
    pub const fn active_sequence_headroom_bytes(self) -> u64 {
        self.active_sequence_headroom_bytes
    }
}

/// Content-free reusable-cache metrics.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ReusableKvCacheMetrics {
    /// Cache lookup attempts.
    pub lookup_count: u64,
    /// Successful prefix matches.
    pub hit_count: u64,
    /// Deterministic misses.
    pub miss_count: u64,
    /// Exact-prefix matches.
    pub exact_prefix_hits: u64,
    /// Shorter longest-prefix matches.
    pub partial_prefix_hits: u64,
    /// Tokens restored from sealed state.
    pub tokens_reused: u64,
    /// Requested tokens not covered by a hit.
    pub tokens_recomputed: u64,
    /// K/V bytes restored from sealed state.
    pub bytes_reused: u64,
    /// Current CPU-resident reusable bytes.
    pub cpu_cache_bytes: u64,
    /// Current GPU-resident reusable bytes; zero in V1.
    pub gpu_cache_bytes: u64,
    /// Pressure/TTL evictions.
    pub evictions: u64,
    /// Explicit invalidations.
    pub invalidations: u64,
    /// Measured transfer nanoseconds; CPU V1 has no transfer.
    pub transfer_nanoseconds: u64,
    /// Measured time saved only; never estimated in V1.
    pub measured_time_saved_nanoseconds: u64,
    /// Measured request time-to-first-token samples.
    pub time_to_first_token_nanoseconds: u64,
    /// Measured request prefill latency samples.
    pub prefill_nanoseconds: u64,
}

#[derive(Debug)]
struct CacheData {
    tokens: Box<[u32]>,
    kv_bytes: Box<[u8]>,
    final_logits: Box<[f32]>,
    content_hash: ModelPackageHash,
    reservation: MemoryReservation,
}

#[derive(Debug)]
struct CacheEntry {
    descriptor: KvCacheDescriptor,
    data: Arc<CacheData>,
    expires_at: Option<u64>,
}

#[derive(Debug, Default)]
struct StoreState {
    entries: BTreeMap<KvCacheId, CacheEntry>,
    bytes: u64,
    metrics: ReusableKvCacheMetrics,
}

/// Model-local immutable reusable prefix store.
#[derive(Debug)]
pub struct ReusableKvCacheStore {
    model_id: ModelId,
    pool: MemoryPool,
    limits: ReusableKvCacheLimits,
    state: Mutex<StoreState>,
}

impl ReusableKvCacheStore {
    /// Creates an empty model-scoped store without reserving memory.
    pub fn new(
        model_id: ModelId,
        pool: MemoryPool,
        limits: ReusableKvCacheLimits,
    ) -> Result<Self, EngineError> {
        if limits.maximum_bytes() > pool.limit_bytes() {
            return Err(EngineError::invalid_input("cache limit exceeds its memory pool"));
        }
        Ok(Self { model_id, pool, limits, state: Mutex::new(StoreState::default()) })
    }

    /// Returns the model identity owning every entry.
    pub const fn model_id(&self) -> &ModelId {
        &self.model_id
    }

    /// Seals one committed request-local prefix into immutable CPU RAM state.
    #[allow(clippy::too_many_arguments, clippy::too_many_lines)]
    pub fn create_from_sequence(
        &self,
        cache: &ContiguousKvCache,
        handle: SequenceHandle,
        compatibility: KvCacheCompatibility,
        binding: CacheBinding,
        tokens: &[u32],
        final_logits: &[f32],
        created_at_unix_seconds: u64,
        retention: CacheRetentionPolicy,
    ) -> Result<KvCacheCreateOutcome, EngineError> {
        if created_at_unix_seconds == 0 {
            return Err(EngineError::invalid_input("cache creation time must be nonzero"));
        }
        if cache.model_id() != &self.model_id || compatibility.model_id() != &self.model_id {
            return Err(EngineError::authentication_failure(
                "reusable cache model identity does not match its store",
            ));
        }
        if compatibility.kv_dtype() != cache.plan().dtype() {
            return Err(EngineError::invalid_input("reusable cache K/V dtype differs"));
        }
        let token_count = u32::try_from(tokens.len())
            .map_err(|_| EngineError::resource_exhaustion("cache token count exceeds u32"))?;
        if cache.token_count(handle)? != token_count {
            return Err(EngineError::invalid_input(
                "sealed token count differs from committed K/V state",
            ));
        }
        if final_logits.is_empty() || final_logits.iter().any(|value| !value.is_finite()) {
            return Err(EngineError::invalid_input("cache final logits are empty or non-finite"));
        }
        let key = KvCacheKey::new(compatibility, binding, tokens)?;
        let cache_id = key.cache_id()?;
        let mut state = self.lock_state()?;
        evict_expired_locked(&mut state, created_at_unix_seconds)?;
        if let Some(existing) = state.entries.get(&cache_id) {
            if existing.descriptor.key != key {
                return Err(EngineError::internal_invariant("cache identifier collision"));
            }
            return Ok(KvCacheCreateOutcome {
                descriptor: existing.descriptor.clone(),
                reused: true,
            });
        }

        let kv_bytes = cache.copy_committed_prefix(handle, token_count)?;
        let kv_length = kv_bytes.len();
        let token_bytes = allocation_bytes::<u32>(tokens.len(), "cache token bytes")?;
        let logit_bytes = allocation_bytes::<f32>(final_logits.len(), "cache logit bytes")?;
        let logical_bytes = [
            u64::try_from(kv_length)
                .map_err(|_| EngineError::resource_exhaustion("cache K/V bytes exceed u64"))?,
            token_bytes,
            logit_bytes,
            CACHE_METADATA_BYTES,
        ]
        .into_iter()
        .try_fold(0_u64, |total, value| {
            total
                .checked_add(value)
                .ok_or_else(|| EngineError::resource_exhaustion("cache byte count overflows"))
        })?;
        make_room_locked(
            &self.pool,
            self.limits,
            &mut state,
            logical_bytes,
            created_at_unix_seconds,
        )?;
        let mut reservation = self
            .pool
            .reserve_for(logical_bytes, MemoryOwner::ReusableKvCache(self.model_id.clone()))?;
        let token_copy = fallible_copy(tokens, "cache token allocation failed")?;
        let logit_copy = fallible_copy(final_logits, "cache logit allocation failed")?;
        let allocated_bytes = CACHE_METADATA_BYTES
            .checked_add(capacity_bytes::<u8>(kv_bytes.capacity())?)
            .and_then(|value| value.checked_add(capacity_bytes::<u32>(token_copy.capacity()).ok()?))
            .and_then(|value| value.checked_add(capacity_bytes::<f32>(logit_copy.capacity()).ok()?))
            .ok_or_else(|| EngineError::resource_exhaustion("cache allocation bytes overflow"))?;
        if allocated_bytes > logical_bytes {
            self.pool.grow_reservation(&mut reservation, allocated_bytes - logical_bytes)?;
        }
        let content_hash = hash_cache_data(&token_copy, &kv_bytes, &logit_copy)?;
        let data = Arc::new(CacheData {
            tokens: token_copy.into_boxed_slice(),
            kv_bytes: kv_bytes.into_boxed_slice(),
            final_logits: logit_copy.into_boxed_slice(),
            content_hash,
            reservation,
        });
        let byte_size = data.reservation.bytes();
        let expires_at = retention
            .ttl_seconds()
            .map(|ttl| {
                created_at_unix_seconds
                    .checked_add(ttl)
                    .ok_or_else(|| EngineError::resource_exhaustion("cache expiry time overflows"))
            })
            .transpose()?;
        let descriptor = KvCacheDescriptor {
            cache_id,
            key,
            state: KvCacheState::Sealed,
            location: KvCacheLocation::CpuRam,
            byte_size,
            created_at_unix_seconds,
            last_access_unix_seconds: created_at_unix_seconds,
            hit_count: 0,
            retention,
            resident: true,
            transfer_cost_nanoseconds: Some(0),
        };
        state.bytes = state
            .bytes
            .checked_add(byte_size)
            .ok_or_else(|| EngineError::resource_exhaustion("cache store bytes overflow"))?;
        state.metrics.cpu_cache_bytes = state.bytes;
        state
            .entries
            .insert(cache_id, CacheEntry { descriptor: descriptor.clone(), data, expires_at });
        Ok(KvCacheCreateOutcome { descriptor, reused: false })
    }

    /// Finds the longest sealed cache whose exact token sequence prefixes the request.
    pub fn lookup_longest_prefix(
        &self,
        compatibility: &KvCacheCompatibility,
        scope: &CacheAuthorizationScope,
        requested_tokens: &[u32],
        now_unix_seconds: u64,
    ) -> Result<KvCacheLookup, EngineError> {
        let requested_count = u64::try_from(requested_tokens.len())
            .map_err(|_| EngineError::resource_exhaustion("requested token count exceeds u64"))?;
        let mut state = self.lock_state()?;
        evict_expired_locked(&mut state, now_unix_seconds)?;
        state.metrics.lookup_count = increment(state.metrics.lookup_count, "cache lookup counter")?;
        let selected = state
            .entries
            .iter()
            .filter(|(_id, entry)| {
                entry.descriptor.key.compatibility() == compatibility
                    && entry.descriptor.key.scope() == scope
                    && requested_tokens.starts_with(&entry.data.tokens)
            })
            .max_by_key(|(id, entry)| (entry.descriptor.key.token_count(), **id))
            .map(|(id, _entry)| *id);
        let Some(cache_id) = selected else {
            state.metrics.miss_count = increment(state.metrics.miss_count, "cache miss counter")?;
            state.metrics.tokens_recomputed = state
                .metrics
                .tokens_recomputed
                .checked_add(requested_count)
                .ok_or_else(|| EngineError::resource_exhaustion("recomputed token counter"))?;
            return Ok(KvCacheLookup::Miss(KvCacheMiss::NoCompatiblePrefix));
        };
        let entry = state.entries.get_mut(&cache_id).ok_or_else(|| {
            EngineError::internal_invariant("selected reusable cache disappeared")
        })?;
        let kind = if usize::try_from(entry.descriptor.key.token_count()).ok()
            == Some(requested_tokens.len())
        {
            KvCacheHitKind::ExactPrefix
        } else {
            KvCacheHitKind::PartialPrefix
        };
        entry.descriptor.last_access_unix_seconds = now_unix_seconds;
        entry.descriptor.hit_count = increment(entry.descriptor.hit_count, "cache hit counter")?;
        let descriptor = entry.descriptor.clone();
        let reused = u64::from(descriptor.key.token_count());
        let bytes =
            prefix_bytes_for_tokens(&descriptor, &entry.data, descriptor.key.token_count())?;
        record_hit_metrics(&mut state.metrics, kind, reused, requested_count, bytes)?;
        Ok(KvCacheLookup::Hit(Box::new(KvCacheHit {
            descriptor,
            kind,
            reused_tokens: u32::try_from(reused)
                .map_err(|_| EngineError::internal_invariant("cache token count exceeds u32"))?,
        })))
    }

    /// Copies one sealed cache into a distinct request-local sequence.
    #[allow(clippy::too_many_arguments)]
    pub fn attach_to_sequence(
        &self,
        cache_id: KvCacheId,
        expected_binding: &CacheBinding,
        compatibility: &KvCacheCompatibility,
        cache: &ContiguousKvCache,
        handle: SequenceHandle,
        now_unix_seconds: u64,
    ) -> Result<KvCacheAttachment, EngineError> {
        let mut state = self.lock_state()?;
        evict_expired_locked(&mut state, now_unix_seconds)?;
        state.metrics.lookup_count = increment(state.metrics.lookup_count, "cache lookup counter")?;
        let Some(entry) = state.entries.get_mut(&cache_id) else {
            state.metrics.miss_count = increment(state.metrics.miss_count, "cache miss counter")?;
            return Err(EngineError::invalid_input("reusable cache ID is unavailable"));
        };
        if entry.descriptor.key.scope() != expected_binding.scope() {
            state.metrics.miss_count = increment(state.metrics.miss_count, "cache miss counter")?;
            return Err(EngineError::authentication_failure("cache scope is not authorized"));
        }
        if entry.descriptor.key.compatibility() != compatibility
            || entry.descriptor.key.binding() != expected_binding
            || compatibility.model_id() != cache.model_id()
        {
            state.metrics.miss_count = increment(state.metrics.miss_count, "cache miss counter")?;
            return Err(EngineError::authentication_failure("cache compatibility differs"));
        }
        let expected_token_count = expected_binding.token_count();
        if hash_cache_data(&entry.data.tokens, &entry.data.kv_bytes, &entry.data.final_logits)?
            != entry.data.content_hash
        {
            return Err(EngineError::internal_invariant("sealed cache content changed"));
        }
        cache.restore_committed_prefix(handle, expected_token_count, &entry.data.kv_bytes)?;
        entry.descriptor.last_access_unix_seconds = now_unix_seconds;
        entry.descriptor.hit_count = increment(entry.descriptor.hit_count, "cache hit counter")?;
        let bytes = prefix_bytes_for_tokens(&entry.descriptor, &entry.data, expected_token_count)?;
        let descriptor = entry.descriptor.clone();
        let data = Arc::clone(&entry.data);
        record_hit_metrics(
            &mut state.metrics,
            KvCacheHitKind::ExactPrefix,
            u64::from(expected_token_count),
            u64::from(expected_token_count),
            bytes,
        )?;
        Ok(KvCacheAttachment { cache_id, descriptor, data })
    }

    /// Returns one cache descriptor only to its exact authorized scope.
    pub fn describe(
        &self,
        cache_id: KvCacheId,
        authorized_scope: &CacheAuthorizationScope,
        now_unix_seconds: u64,
    ) -> Result<KvCacheDescriptor, EngineError> {
        let mut state = self.lock_state()?;
        evict_expired_locked(&mut state, now_unix_seconds)?;
        let entry = state
            .entries
            .get(&cache_id)
            .ok_or_else(|| EngineError::invalid_input("reusable cache ID is unavailable"))?;
        if entry.descriptor.key.scope() != authorized_scope {
            return Err(EngineError::authentication_failure("cache scope is not authorized"));
        }
        Ok(entry.descriptor.clone())
    }

    /// Lists a bounded deterministic cache descriptor prefix for one exact scope.
    pub fn list(
        &self,
        authorized_scope: &CacheAuthorizationScope,
        maximum: u16,
        now_unix_seconds: u64,
    ) -> Result<Vec<KvCacheDescriptor>, EngineError> {
        if maximum == 0 || u32::from(maximum) > self.limits.maximum_entries() {
            return Err(EngineError::invalid_input("cache list maximum is outside bounds"));
        }
        let mut state = self.lock_state()?;
        evict_expired_locked(&mut state, now_unix_seconds)?;
        let mut descriptors = Vec::new();
        descriptors
            .try_reserve_exact(usize::from(maximum))
            .map_err(|_| EngineError::resource_exhaustion("cache descriptor allocation failed"))?;
        for entry in state.entries.values() {
            if entry.descriptor.key.scope() == authorized_scope {
                descriptors.push(entry.descriptor.clone());
                if descriptors.len() == usize::from(maximum) {
                    break;
                }
            }
        }
        Ok(descriptors)
    }

    /// Explicitly invalidates one cache; active forks remain protected.
    pub fn invalidate_cache(
        &self,
        cache_id: KvCacheId,
        authorized_scope: &CacheAuthorizationScope,
    ) -> Result<bool, EngineError> {
        let mut state = self.lock_state()?;
        let Some(entry) = state.entries.get(&cache_id) else {
            return Ok(false);
        };
        if entry.descriptor.key.scope() != authorized_scope {
            return Err(EngineError::authentication_failure("cache scope is not authorized"));
        }
        if Arc::strong_count(&entry.data) != 1 {
            return Err(EngineError::resource_exhaustion("cache has an active request lease"));
        }
        remove_entry(&mut state, cache_id, false)?;
        state.metrics.invalidations =
            increment(state.metrics.invalidations, "invalidation counter")?;
        Ok(true)
    }

    /// Invalidates every idle cache in an exact scope.
    pub fn invalidate_namespace(
        &self,
        scope: &CacheAuthorizationScope,
    ) -> Result<u32, EngineError> {
        let mut state = self.lock_state()?;
        let ids = state
            .entries
            .iter()
            .filter(|(_id, entry)| {
                entry.descriptor.key.scope() == scope && Arc::strong_count(&entry.data) == 1
            })
            .map(|(id, _entry)| *id)
            .collect::<Vec<_>>();
        for id in &ids {
            remove_entry(&mut state, *id, false)?;
        }
        let count = u32::try_from(ids.len())
            .map_err(|_| EngineError::internal_invariant("invalidation count exceeds u32"))?;
        state.metrics.invalidations =
            state.metrics.invalidations.checked_add(u64::from(count)).ok_or_else(|| {
                EngineError::resource_exhaustion("invalidation counter overflows")
            })?;
        Ok(count)
    }

    /// Invalidates idle caches for one exact logical model inside one scope.
    pub fn invalidate_logical_model_in_scope(
        &self,
        scope: &CacheAuthorizationScope,
        logical_model_id: &KnowledgeIdentity,
    ) -> Result<u32, EngineError> {
        self.invalidate_matching(|entry| {
            entry.descriptor.key.scope() == scope
                && entry.descriptor.key.binding().runtime().logical_model_id() == logical_model_id
        })
    }

    /// Invalidates idle caches for one exact bundle revision inside one scope.
    pub fn invalidate_bundle_revision_in_scope(
        &self,
        scope: &CacheAuthorizationScope,
        bundle_id: &KnowledgeIdentity,
        revision: u64,
    ) -> Result<u32, EngineError> {
        if revision == 0 {
            return Err(EngineError::invalid_input("bundle revision must be nonzero"));
        }
        self.invalidate_matching(|entry| {
            let binding = entry.descriptor.key.binding();
            entry.descriptor.key.scope() == scope
                && binding.bundle().bundle_id() == bundle_id
                && binding.bundle().bundle_revision() == revision
        })
    }

    /// Invalidates idle caches for one exact canonical source revision in one scope.
    pub fn invalidate_source_revision_in_scope(
        &self,
        scope: &CacheAuthorizationScope,
        source_id: &KnowledgeIdentity,
        revision_id: &KnowledgeIdentity,
    ) -> Result<u32, EngineError> {
        self.invalidate_matching(|entry| {
            entry.descriptor.key.scope() == scope
                && entry.descriptor.key.binding().source_revision().is_some_and(|source| {
                    source.source_id() == source_id && source.revision_id() == revision_id
                })
        })
    }

    /// Invalidates all idle model caches before model unload.
    pub fn invalidate_model(&self) -> Result<u32, EngineError> {
        self.invalidate_matching(|_entry| true)
    }

    /// Evicts deterministic unpinned LRU entries until requested memory is available.
    pub fn evict_for_memory_pressure(
        &self,
        required_available_bytes: u64,
        now_unix_seconds: u64,
    ) -> Result<u32, EngineError> {
        let mut state = self.lock_state()?;
        evict_expired_locked(&mut state, now_unix_seconds)?;
        let mut evicted = 0_u32;
        while self.pool.available_bytes()? < required_available_bytes {
            let Some(id) = eviction_candidate(&state) else {
                return Err(EngineError::resource_exhaustion(
                    "reusable caches cannot preserve requested inference headroom",
                ));
            };
            remove_entry(&mut state, id, true)?;
            evicted = evicted
                .checked_add(1)
                .ok_or_else(|| EngineError::resource_exhaustion("eviction count overflows"))?;
        }
        Ok(evicted)
    }

    /// Returns an atomic content-free metrics snapshot.
    pub fn metrics(&self) -> Result<ReusableKvCacheMetrics, EngineError> {
        Ok(self.lock_state()?.metrics)
    }

    /// Records measured request execution values without estimating saved time.
    pub fn record_execution_measurement(
        &self,
        tokens_recomputed: u32,
        prefill_nanoseconds: u64,
        time_to_first_token_nanoseconds: u64,
    ) -> Result<(), EngineError> {
        let mut state = self.lock_state()?;
        state.metrics.tokens_recomputed =
            state.metrics.tokens_recomputed.checked_add(u64::from(tokens_recomputed)).ok_or_else(
                || EngineError::resource_exhaustion("cache recompute metric overflows"),
            )?;
        state.metrics.prefill_nanoseconds =
            state.metrics.prefill_nanoseconds.checked_add(prefill_nanoseconds).ok_or_else(
                || EngineError::resource_exhaustion("cache prefill metric overflows"),
            )?;
        state.metrics.time_to_first_token_nanoseconds = state
            .metrics
            .time_to_first_token_nanoseconds
            .checked_add(time_to_first_token_nanoseconds)
            .ok_or_else(|| EngineError::resource_exhaustion("cache TTFT metric overflows"))?;
        Ok(())
    }

    /// Rejects unimplemented tier migration without mutating cache state.
    pub fn migrate(&self, cache_id: KvCacheId, target: KvCacheLocation) -> Result<(), EngineError> {
        if !self.lock_state()?.entries.contains_key(&cache_id) {
            return Err(EngineError::invalid_input("reusable cache ID is unavailable"));
        }
        if target == KvCacheLocation::CpuRam {
            return Ok(());
        }
        Err(EngineError::not_implemented(
            "reusable cache GPU and SSD migration are not implemented",
        ))
    }

    fn invalidate_matching(
        &self,
        predicate: impl Fn(&CacheEntry) -> bool,
    ) -> Result<u32, EngineError> {
        let mut state = self.lock_state()?;
        if state
            .entries
            .values()
            .any(|entry| predicate(entry) && Arc::strong_count(&entry.data) != 1)
        {
            return Err(EngineError::resource_exhaustion(
                "matching reusable cache has an active request lease",
            ));
        }
        let ids = state
            .entries
            .iter()
            .filter(|(_id, entry)| predicate(entry))
            .map(|(id, _entry)| *id)
            .collect::<Vec<_>>();
        for id in &ids {
            remove_entry(&mut state, *id, false)?;
        }
        let count = u32::try_from(ids.len())
            .map_err(|_| EngineError::internal_invariant("invalidation count exceeds u32"))?;
        state.metrics.invalidations =
            state.metrics.invalidations.checked_add(u64::from(count)).ok_or_else(|| {
                EngineError::resource_exhaustion("invalidation counter overflows")
            })?;
        Ok(count)
    }

    fn lock_state(&self) -> Result<MutexGuard<'_, StoreState>, EngineError> {
        self.state
            .lock()
            .map_err(|_| EngineError::internal_invariant("reusable cache lock is poisoned"))
    }
}

/// Immutable lease retained while request-local state uses one cache identity.
#[derive(Clone, Debug)]
pub struct KvCacheAttachment {
    cache_id: KvCacheId,
    descriptor: KvCacheDescriptor,
    data: Arc<CacheData>,
}

impl KvCacheAttachment {
    /// Returns the exact cache ID.
    pub const fn cache_id(&self) -> KvCacheId {
        self.cache_id
    }

    /// Returns the sealed prefix token count.
    pub const fn token_count(&self) -> u32 {
        self.descriptor.key.token_count()
    }

    /// Returns immutable cached token IDs for request history accounting.
    pub fn tokens(&self) -> &[u32] {
        &self.data.tokens
    }

    /// Returns immutable logits after the final cached prefix token.
    pub fn final_logits(&self) -> &[f32] {
        &self.data.final_logits
    }

    /// Returns content-free cache metadata.
    pub const fn descriptor(&self) -> &KvCacheDescriptor {
        &self.descriptor
    }
}

fn hash_tokens(tokens: &[u32]) -> Result<ModelPackageHash, EngineError> {
    let capacity = tokens
        .len()
        .checked_mul(4)
        .ok_or_else(|| EngineError::resource_exhaustion("cache token hash bytes overflow"))?;
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(capacity)
        .map_err(|_| EngineError::resource_exhaustion("cache token hash allocation failed"))?;
    for token in tokens {
        bytes.extend_from_slice(&token.to_le_bytes());
    }
    Ok(hash_package(&bytes))
}

fn encode_key(key: &KvCacheKey) -> Result<Vec<u8>, EngineError> {
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(512)
        .map_err(|_| EngineError::resource_exhaustion("cache key allocation failed"))?;
    bytes.extend_from_slice(b"PAICAGK1");
    bytes.extend_from_slice(&REUSABLE_KV_CACHE_FORMAT_VERSION.to_le_bytes());
    append_text(&mut bytes, key.compatibility.model_id().as_str())?;
    bytes.extend_from_slice(key.compatibility.model_artifact_hash().as_bytes());
    bytes.extend_from_slice(key.compatibility.tokenizer_hash().as_bytes());
    bytes.extend_from_slice(key.compatibility.tokenizer_sha256().as_bytes());
    append_text(&mut bytes, key.compatibility.architecture_id().as_str())?;
    append_text(&mut bytes, key.compatibility.backend_id().as_str())?;
    bytes.push(key.compatibility.attention_positions() as u8);
    bytes.extend_from_slice(&key.compatibility.rope_theta_bits.to_le_bytes());
    bytes.extend_from_slice(&key.compatibility.rope_scaling_bits.to_le_bytes());
    bytes.push(dtype_code(key.compatibility.kv_dtype())?);
    append_binding(&mut bytes, key.binding())?;
    bytes.extend_from_slice(key.ordered_token_hash.as_bytes());
    bytes.extend_from_slice(&key.token_count.to_le_bytes());
    Ok(bytes)
}

fn append_binding(bytes: &mut Vec<u8>, binding: &CacheBinding) -> Result<(), EngineError> {
    let runtime = binding.runtime();
    append_text(bytes, runtime.logical_model_id().as_str())?;
    append_text(bytes, runtime.provider_model_id().as_str())?;
    append_text(bytes, runtime.artifact_id().as_str())?;
    match runtime.artifact_digest() {
        ArtifactDigest::Sha256(digest) => {
            bytes.push(1);
            bytes.extend_from_slice(digest.as_bytes());
        }
        ArtifactDigest::Sha384(digest) => {
            bytes.push(2);
            bytes.extend_from_slice(digest.as_bytes());
        }
    }
    append_text(bytes, runtime.backend_id().as_str())?;
    append_text(bytes, runtime.backend_version().as_str())?;
    append_text(bytes, runtime.backend_instance_id().as_str())?;
    append_text(bytes, runtime.engine_instance_id().as_str())?;
    append_text(bytes, binding.tokenizer_id().as_str())?;
    bytes.extend_from_slice(binding.tokenizer_digest().as_bytes());
    append_text(bytes, binding.architecture_id().as_str())?;
    append_text(bytes, binding.rope_semantics_id().as_str())?;
    bytes.push(dtype_code(binding.kv_dtype())?);
    let bundle = binding.bundle();
    append_text(bytes, bundle.bundle_id().as_str())?;
    bytes.extend_from_slice(&bundle.bundle_revision().to_le_bytes());
    bytes.extend_from_slice(bundle.bundle_digest().as_bytes());
    bytes.extend_from_slice(bundle.ordered_prefix_digest().as_bytes());
    match binding.source_revision() {
        Some(source) => {
            bytes.push(1);
            append_text(bytes, source.source_id().as_str())?;
            append_text(bytes, source.revision_id().as_str())?;
        }
        None => bytes.push(0),
    }
    let scope = binding.scope();
    bytes.push(scope.namespace() as u8);
    append_text(bytes, scope.owner_id().as_str())?;
    append_optional_identity(bytes, scope.project_id())?;
    append_optional_identity(bytes, scope.task_id())?;
    append_optional_identity(bytes, scope.agent_id())?;
    bytes.push(binding.sensitivity() as u8);
    bytes.extend_from_slice(&binding.token_count().to_le_bytes());
    Ok(())
}

fn append_optional_identity(
    bytes: &mut Vec<u8>,
    identity: Option<&KnowledgeIdentity>,
) -> Result<(), EngineError> {
    match identity {
        Some(identity) => {
            bytes.push(1);
            append_text(bytes, identity.as_str())?;
        }
        None => bytes.push(0),
    }
    Ok(())
}

fn append_text(bytes: &mut Vec<u8>, value: &str) -> Result<(), EngineError> {
    let length = u16::try_from(value.len())
        .map_err(|_| EngineError::resource_exhaustion("cache identity text exceeds u16"))?;
    bytes.extend_from_slice(&length.to_le_bytes());
    bytes.extend_from_slice(value.as_bytes());
    Ok(())
}

fn dtype_code(dtype: DType) -> Result<u8, EngineError> {
    match dtype {
        DType::F32 => Ok(1),
        DType::F16 => Ok(2),
        DType::BFloat16 => Ok(3),
        _ => Err(EngineError::not_implemented("cache K/V dtype has no identity code")),
    }
}

fn hash_cache_data(
    tokens: &[u32],
    kv_bytes: &[u8],
    final_logits: &[f32],
) -> Result<ModelPackageHash, EngineError> {
    let capacity = tokens
        .len()
        .checked_mul(4)
        .and_then(|value| value.checked_add(kv_bytes.len()))
        .and_then(|value| value.checked_add(final_logits.len().checked_mul(4)?))
        .ok_or_else(|| EngineError::resource_exhaustion("cache content hash bytes overflow"))?;
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(capacity)
        .map_err(|_| EngineError::resource_exhaustion("cache content hash allocation failed"))?;
    for token in tokens {
        bytes.extend_from_slice(&token.to_le_bytes());
    }
    bytes.extend_from_slice(kv_bytes);
    for logit in final_logits {
        bytes.extend_from_slice(&logit.to_bits().to_le_bytes());
    }
    Ok(hash_package(&bytes))
}

fn fallible_copy<T: Copy>(source: &[T], reason: &'static str) -> Result<Vec<T>, EngineError> {
    let mut values = Vec::new();
    values.try_reserve_exact(source.len()).map_err(|_| EngineError::resource_exhaustion(reason))?;
    values.extend_from_slice(source);
    Ok(values)
}

fn allocation_bytes<T>(count: usize, label: &'static str) -> Result<u64, EngineError> {
    count
        .checked_mul(std::mem::size_of::<T>())
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or_else(|| EngineError::resource_exhaustion(label))
}

fn capacity_bytes<T>(capacity: usize) -> Result<u64, EngineError> {
    allocation_bytes::<T>(capacity, "cache allocation capacity overflows")
}

fn make_room_locked(
    pool: &MemoryPool,
    limits: ReusableKvCacheLimits,
    state: &mut StoreState,
    required: u64,
    now: u64,
) -> Result<(), EngineError> {
    if required > limits.maximum_bytes() {
        return Err(EngineError::resource_exhaustion("one reusable cache exceeds its byte limit"));
    }
    evict_expired_locked(state, now)?;
    loop {
        let entry_count = u32::try_from(state.entries.len())
            .map_err(|_| EngineError::internal_invariant("cache entry count exceeds u32"))?;
        let store_fits =
            state.bytes.checked_add(required).is_some_and(|total| total <= limits.maximum_bytes());
        let required_with_headroom = required
            .checked_add(limits.active_sequence_headroom_bytes())
            .ok_or_else(|| EngineError::resource_exhaustion("cache headroom bytes overflow"))?;
        let pool_fits = pool.available_bytes()? >= required_with_headroom;
        if entry_count < limits.maximum_entries() && store_fits && pool_fits {
            return Ok(());
        }
        let Some(id) = eviction_candidate(state) else {
            return Err(EngineError::resource_exhaustion(
                "reusable cache store cannot preserve inference headroom",
            ));
        };
        remove_entry(state, id, true)?;
    }
}

fn eviction_candidate(state: &StoreState) -> Option<KvCacheId> {
    state
        .entries
        .iter()
        .filter(|(_id, entry)| {
            !entry.descriptor.retention.pinned() && Arc::strong_count(&entry.data) == 1
        })
        .min_by_key(|(id, entry)| {
            (
                entry.descriptor.retention.eviction_priority(),
                entry.descriptor.last_access_unix_seconds,
                **id,
            )
        })
        .map(|(id, _entry)| *id)
}

fn evict_expired_locked(state: &mut StoreState, now: u64) -> Result<(), EngineError> {
    let ids = state
        .entries
        .iter()
        .filter(|(_id, entry)| {
            entry.expires_at.is_some_and(|expiry| expiry <= now)
                && !entry.descriptor.retention.pinned()
                && Arc::strong_count(&entry.data) == 1
        })
        .map(|(id, _entry)| *id)
        .collect::<Vec<_>>();
    for id in ids {
        remove_entry(state, id, true)?;
    }
    Ok(())
}

fn remove_entry(
    state: &mut StoreState,
    cache_id: KvCacheId,
    eviction: bool,
) -> Result<(), EngineError> {
    let entry = state
        .entries
        .remove(&cache_id)
        .ok_or_else(|| EngineError::internal_invariant("cache entry disappeared"))?;
    state.bytes = state
        .bytes
        .checked_sub(entry.descriptor.byte_size)
        .ok_or_else(|| EngineError::internal_invariant("cache byte accounting underflows"))?;
    state.metrics.cpu_cache_bytes = state.bytes;
    if eviction {
        state.metrics.evictions = increment(state.metrics.evictions, "cache eviction counter")?;
    }
    Ok(())
}

fn prefix_bytes_for_tokens(
    descriptor: &KvCacheDescriptor,
    data: &CacheData,
    token_count: u32,
) -> Result<u64, EngineError> {
    if token_count != descriptor.key.token_count() {
        return Err(EngineError::internal_invariant("cache prefix byte request differs"));
    }
    u64::try_from(data.kv_bytes.len())
        .map_err(|_| EngineError::resource_exhaustion("cache prefix bytes exceed u64"))
}

fn record_hit_metrics(
    metrics: &mut ReusableKvCacheMetrics,
    kind: KvCacheHitKind,
    reused: u64,
    requested: u64,
    bytes: u64,
) -> Result<(), EngineError> {
    metrics.hit_count = increment(metrics.hit_count, "cache hit counter")?;
    match kind {
        KvCacheHitKind::ExactPrefix => {
            metrics.exact_prefix_hits =
                increment(metrics.exact_prefix_hits, "exact cache hit counter")?;
        }
        KvCacheHitKind::PartialPrefix => {
            metrics.partial_prefix_hits =
                increment(metrics.partial_prefix_hits, "partial cache hit counter")?;
        }
    }
    metrics.tokens_reused = metrics
        .tokens_reused
        .checked_add(reused)
        .ok_or_else(|| EngineError::resource_exhaustion("reused token counter overflows"))?;
    metrics.tokens_recomputed = metrics
        .tokens_recomputed
        .checked_add(requested.checked_sub(reused).ok_or_else(|| {
            EngineError::internal_invariant("cache reused more tokens than requested")
        })?)
        .ok_or_else(|| EngineError::resource_exhaustion("recomputed token counter overflows"))?;
    metrics.bytes_reused = metrics
        .bytes_reused
        .checked_add(bytes)
        .ok_or_else(|| EngineError::resource_exhaustion("reused byte counter overflows"))?;
    Ok(())
}

fn increment(value: u64, reason: &'static str) -> Result<u64, EngineError> {
    value.checked_add(1).ok_or_else(|| EngineError::resource_exhaustion(reason))
}

#[cfg(test)]
mod tests {
    use super::*;
    use personalai_engine_knowledge_contract::{
        CacheNamespace, CacheRuntimeIdentity, CacheSensitivity, SourceRevisionBinding,
        StaticBundleBinding,
    };
    use personalai_engine_types::{RequestId, ResourceBudget, SequenceId, Sha256Digest};

    fn hash(byte: u8) -> ModelPackageHash {
        ModelPackageHash::from_sha384_bytes([byte; ModelPackageHash::BYTE_LEN])
    }

    fn compatibility(model_id: &ModelId) -> KvCacheCompatibility {
        KvCacheCompatibility::new(
            model_id.clone(),
            hash(1),
            hash(2),
            Sha256Digest::from_bytes([2; 32]),
            ArchitectureId::new("decoder-rms-rope-swiglu-gqa-v1").unwrap(),
            BackendId::new("cpu-reference").unwrap(),
            AttentionPositionSemantics::CausalZeroBasedV1,
            10_000.0,
            1.0,
            DType::F32,
        )
        .unwrap()
    }

    fn cache_scope(owner: &str) -> CacheAuthorizationScope {
        CacheAuthorizationScope::new(
            CacheNamespace::Project,
            KnowledgeIdentity::new("owner.test").unwrap(),
            Some(KnowledgeIdentity::new(owner).unwrap()),
            None,
            None,
        )
        .unwrap()
    }

    fn binding(
        model_id: &ModelId,
        scope: CacheAuthorizationScope,
        token_count: u32,
        revision: u64,
    ) -> CacheBinding {
        CacheBinding::new(
            CacheRuntimeIdentity::new(
                KnowledgeIdentity::new("logical.test").unwrap(),
                model_id.clone(),
                KnowledgeIdentity::new("artifact.test").unwrap(),
                ArtifactDigest::Sha384(hash(1)),
                BackendId::new("cpu-reference").unwrap(),
                KnowledgeIdentity::new("version.1").unwrap(),
                KnowledgeIdentity::new("backend.test").unwrap(),
                KnowledgeIdentity::new("engine.test").unwrap(),
            )
            .unwrap(),
            KnowledgeIdentity::new("tokenizer.test").unwrap(),
            Sha256Digest::from_bytes([2; 32]),
            ArchitectureId::new("decoder-rms-rope-swiglu-gqa-v1").unwrap(),
            KnowledgeIdentity::new("rope.causal-zero-based-v1").unwrap(),
            DType::F32,
            StaticBundleBinding::new(
                KnowledgeIdentity::new("bundle.test").unwrap(),
                revision,
                Sha256Digest::from_bytes([3; 32]),
                Sha256Digest::from_bytes([4; 32]),
            )
            .unwrap(),
            scope,
            CacheSensitivity::Internal,
            token_count,
        )
        .unwrap()
        .with_source_revision(SourceRevisionBinding::new(
            KnowledgeIdentity::new("source.test").unwrap(),
            KnowledgeIdentity::new(format!("revision.{revision}")).unwrap(),
        ))
    }

    fn cache_and_store() -> (ContiguousKvCache, ReusableKvCacheStore, CacheAuthorizationScope) {
        let model_id = ModelId::new("cag-test-model").unwrap();
        let budget = ResourceBudget::new(1 << 20, 1 << 18, 4).unwrap();
        let pool = MemoryPool::new(budget.max_memory_bytes()).unwrap();
        let plan = crate::KvCachePlan::new(2, 1, 2, 8, DType::F32, &budget).unwrap();
        let headroom = plan.bytes();
        let cache = ContiguousKvCache::new(model_id.clone(), plan, 4, pool.clone()).unwrap();
        let limits = ReusableKvCacheLimits::new(8, 1 << 17, headroom, pool.limit_bytes()).unwrap();
        let store = ReusableKvCacheStore::new(model_id, pool, limits).unwrap();
        let scope = cache_scope("project.alpha");
        (cache, store, scope)
    }

    fn filled_sequence(
        cache: &ContiguousKvCache,
        request: u128,
        sequence: u64,
        tokens: u32,
    ) -> SequenceHandle {
        let handle = cache
            .create_sequence(RequestId::new(request).unwrap(), SequenceId::new(sequence).unwrap())
            .unwrap();
        let width =
            usize::try_from(cache.plan().kv_head_count() * cache.plan().head_dimension()).unwrap();
        for token in 0..tokens {
            let mut writer = cache.begin_token(handle).unwrap();
            for layer in 0..cache.plan().layer_count() {
                let base = f32::from(u16::try_from(token * 10 + layer).unwrap());
                writer
                    .write_layer(layer, &vec![base + 1.0; width], &vec![base + 2.0; width])
                    .unwrap();
            }
            writer.commit().unwrap();
        }
        handle
    }

    #[test]
    fn exact_partial_and_reordered_prefixes_are_deterministic() {
        let (cache, store, scope) = cache_and_store();
        let handle = filled_sequence(&cache, 1, 1, 3);
        let descriptor = store
            .create_from_sequence(
                &cache,
                handle,
                compatibility(cache.model_id()),
                binding(cache.model_id(), scope.clone(), 3, 1),
                &[1, 2, 3],
                &[0.1, 0.2],
                100,
                CacheRetentionPolicy::explicit(),
            )
            .unwrap();
        assert_eq!(descriptor.cache_id(), descriptor.key().cache_id().unwrap());
        let exact = store
            .lookup_longest_prefix(&compatibility(cache.model_id()), &scope, &[1, 2, 3], 101)
            .unwrap();
        assert!(
            matches!(exact, KvCacheLookup::Hit(hit) if hit.kind() == KvCacheHitKind::ExactPrefix)
        );
        let partial = store
            .lookup_longest_prefix(&compatibility(cache.model_id()), &scope, &[1, 2, 3, 4], 102)
            .unwrap();
        assert!(
            matches!(partial, KvCacheLookup::Hit(hit) if hit.kind() == KvCacheHitKind::PartialPrefix)
        );
        assert_eq!(
            store
                .lookup_longest_prefix(&compatibility(cache.model_id()), &scope, &[2, 1, 3], 103)
                .unwrap(),
            KvCacheLookup::Miss(KvCacheMiss::NoCompatiblePrefix)
        );
    }

    #[test]
    fn attach_forks_state_and_scope_or_compatibility_mismatch_fails() {
        let (cache, store, scope) = cache_and_store();
        let source = filled_sequence(&cache, 1, 1, 2);
        let descriptor = store
            .create_from_sequence(
                &cache,
                source,
                compatibility(cache.model_id()),
                binding(cache.model_id(), scope.clone(), 2, 1),
                &[4, 5],
                &[0.3, 0.4],
                100,
                CacheRetentionPolicy::explicit(),
            )
            .unwrap();
        let destination =
            cache.create_sequence(RequestId::new(2).unwrap(), SequenceId::new(2).unwrap()).unwrap();
        let attachment = store
            .attach_to_sequence(
                descriptor.cache_id(),
                descriptor.key().binding(),
                &compatibility(cache.model_id()),
                &cache,
                destination,
                101,
            )
            .unwrap();
        assert_eq!(cache.token_count(destination).unwrap(), 2);
        assert_eq!(attachment.tokens(), &[4, 5]);
        cache.cancel_sequence(destination).unwrap();
        assert_eq!(cache.token_count(source).unwrap(), 2);
        let wrong_scope = cache_scope("project.beta");
        assert!(store.describe(descriptor.cache_id(), &wrong_scope, 102).is_err());
        let mut wrong = compatibility(cache.model_id());
        wrong.rope_theta_bits = 20_000.0_f32.to_bits();
        let empty =
            cache.create_sequence(RequestId::new(3).unwrap(), SequenceId::new(3).unwrap()).unwrap();
        assert!(
            store
                .attach_to_sequence(
                    descriptor.cache_id(),
                    descriptor.key().binding(),
                    &wrong,
                    &cache,
                    empty,
                    103,
                )
                .is_err()
        );
    }

    #[test]
    fn ttl_pressure_pins_and_active_leases_are_bounded() {
        let (cache, store, scope) = cache_and_store();
        let first = filled_sequence(&cache, 1, 1, 1);
        let ttl = store
            .create_from_sequence(
                &cache,
                first,
                compatibility(cache.model_id()),
                binding(cache.model_id(), scope.clone(), 1, 1),
                &[1],
                &[0.1],
                100,
                CacheRetentionPolicy::new(Some(1), 1, false).unwrap(),
            )
            .unwrap();
        assert!(store.describe(ttl.cache_id(), &scope, 101).is_err());
        let second = filled_sequence(&cache, 2, 2, 1);
        let pinned = store
            .create_from_sequence(
                &cache,
                second,
                compatibility(cache.model_id()),
                binding(cache.model_id(), scope.clone(), 1, 1),
                &[2],
                &[0.2],
                102,
                CacheRetentionPolicy::new(None, 0, true).unwrap(),
            )
            .unwrap();
        assert!(store.evict_for_memory_pressure(store.pool.limit_bytes(), 103).is_err());
        assert!(store.describe(pinned.cache_id(), &scope, 103).is_ok());
    }

    #[test]
    fn retention_ttl_is_strictly_bounded() {
        assert!(CacheRetentionPolicy::new(Some(0), 1, false).is_err());
        assert!(CacheRetentionPolicy::new(Some(MAX_REUSABLE_CACHE_TTL_SECONDS), 1, false).is_ok());
        assert!(
            CacheRetentionPolicy::new(Some(MAX_REUSABLE_CACHE_TTL_SECONDS + 1), 1, false).is_err()
        );
    }

    #[test]
    fn model_tokenizer_rope_dtype_and_namespace_changes_miss() {
        let (cache, store, scope) = cache_and_store();
        let handle = filled_sequence(&cache, 1, 1, 1);
        store
            .create_from_sequence(
                &cache,
                handle,
                compatibility(cache.model_id()),
                binding(cache.model_id(), scope.clone(), 1, 1),
                &[7],
                &[0.7],
                100,
                CacheRetentionPolicy::explicit(),
            )
            .unwrap();
        let mut cases = Vec::new();
        let mut model = compatibility(cache.model_id());
        model.model_artifact_hash = hash(9);
        cases.push(model);
        let mut tokenizer = compatibility(cache.model_id());
        tokenizer.tokenizer_hash = hash(9);
        cases.push(tokenizer);
        let mut rope = compatibility(cache.model_id());
        rope.rope_theta_bits = 20_000.0_f32.to_bits();
        cases.push(rope);
        let mut dtype = compatibility(cache.model_id());
        dtype.kv_dtype = DType::F16;
        cases.push(dtype);
        for changed in cases {
            assert_eq!(
                store.lookup_longest_prefix(&changed, &scope, &[7], 101).unwrap(),
                KvCacheLookup::Miss(KvCacheMiss::NoCompatiblePrefix)
            );
        }
        let other_scope = CacheAuthorizationScope::new(
            CacheNamespace::Task,
            KnowledgeIdentity::new("owner.test").unwrap(),
            Some(KnowledgeIdentity::new("project.alpha").unwrap()),
            Some(KnowledgeIdentity::new("task.other").unwrap()),
            None,
        )
        .unwrap();
        assert_eq!(
            store
                .lookup_longest_prefix(&compatibility(cache.model_id()), &other_scope, &[7], 102,)
                .unwrap(),
            KvCacheLookup::Miss(KvCacheMiss::NoCompatiblePrefix)
        );
    }

    #[test]
    fn simultaneous_readers_fork_independent_sequences() {
        let (cache, store, scope) = cache_and_store();
        let source = filled_sequence(&cache, 1, 1, 2);
        let compatibility = compatibility(cache.model_id());
        let descriptor = store
            .create_from_sequence(
                &cache,
                source,
                compatibility.clone(),
                binding(cache.model_id(), scope.clone(), 2, 1),
                &[1, 2],
                &[0.1, 0.2],
                100,
                CacheRetentionPolicy::explicit(),
            )
            .unwrap();
        let first =
            cache.create_sequence(RequestId::new(2).unwrap(), SequenceId::new(2).unwrap()).unwrap();
        let second =
            cache.create_sequence(RequestId::new(3).unwrap(), SequenceId::new(3).unwrap()).unwrap();
        std::thread::scope(|threads| {
            let first_thread = threads.spawn(|| {
                store
                    .attach_to_sequence(
                        descriptor.cache_id(),
                        descriptor.key().binding(),
                        &compatibility,
                        &cache,
                        first,
                        101,
                    )
                    .unwrap()
            });
            let second_thread = threads.spawn(|| {
                store
                    .attach_to_sequence(
                        descriptor.cache_id(),
                        descriptor.key().binding(),
                        &compatibility,
                        &cache,
                        second,
                        102,
                    )
                    .unwrap()
            });
            let first_lease = first_thread.join().unwrap();
            let second_lease = second_thread.join().unwrap();
            assert_eq!(first_lease.tokens(), second_lease.tokens());
            assert_eq!(cache.token_count(first).unwrap(), 2);
            assert_eq!(cache.token_count(second).unwrap(), 2);
        });
        cache.reset_sequence(first).unwrap();
        assert_eq!(cache.token_count(second).unwrap(), 2);
    }

    #[test]
    fn source_revision_and_tier_operations_fail_or_invalidate_exactly() {
        let (cache, store, scope) = cache_and_store();
        let first = filled_sequence(&cache, 1, 1, 1);
        let first_descriptor = store
            .create_from_sequence(
                &cache,
                first,
                compatibility(cache.model_id()),
                binding(cache.model_id(), scope.clone(), 1, 1),
                &[1],
                &[0.1],
                100,
                CacheRetentionPolicy::explicit(),
            )
            .unwrap();
        let second = filled_sequence(&cache, 2, 2, 1);
        let second_descriptor = store
            .create_from_sequence(
                &cache,
                second,
                compatibility(cache.model_id()),
                binding(cache.model_id(), scope.clone(), 1, 2),
                &[2],
                &[0.2],
                101,
                CacheRetentionPolicy::explicit(),
            )
            .unwrap();
        assert!(matches!(
            store.migrate(first_descriptor.cache_id(), KvCacheLocation::GpuMemory),
            Err(EngineError::NotImplemented(_))
        ));
        assert_eq!(
            store
                .invalidate_source_revision_in_scope(
                    &scope,
                    &KnowledgeIdentity::new("source.test").unwrap(),
                    &KnowledgeIdentity::new("revision.1").unwrap(),
                )
                .unwrap(),
            1
        );
        assert!(store.describe(first_descriptor.cache_id(), &scope, 102).is_err());
        assert!(store.describe(second_descriptor.cache_id(), &scope, 102).is_ok());
        assert_eq!(store.invalidate_model().unwrap(), 1);
        assert!(store.describe(second_descriptor.cache_id(), &scope, 103).is_err());
    }

    #[test]
    fn exact_invalidation_is_idempotent_single_match_and_restart_is_empty() {
        let (cache, store, scope) = cache_and_store();
        let first = filled_sequence(&cache, 1, 1, 1);
        let first_descriptor = store
            .create_from_sequence(
                &cache,
                first,
                compatibility(cache.model_id()),
                binding(cache.model_id(), scope.clone(), 1, 1),
                &[1],
                &[0.1],
                100,
                CacheRetentionPolicy::explicit(),
            )
            .unwrap();
        let second = filled_sequence(&cache, 2, 2, 1);
        let second_descriptor = store
            .create_from_sequence(
                &cache,
                second,
                compatibility(cache.model_id()),
                binding(cache.model_id(), scope.clone(), 1, 2),
                &[2],
                &[0.2],
                101,
                CacheRetentionPolicy::explicit(),
            )
            .unwrap();
        assert!(store.invalidate_cache(first_descriptor.cache_id(), &scope).unwrap());
        assert!(!store.invalidate_cache(first_descriptor.cache_id(), &scope).unwrap());
        assert!(store.describe(second_descriptor.cache_id(), &scope, 102).is_ok());

        let budget = ResourceBudget::new(1 << 20, 1 << 18, 4).unwrap();
        let restart_pool = MemoryPool::new(budget.max_memory_bytes()).unwrap();
        let restart = ReusableKvCacheStore::new(
            cache.model_id().clone(),
            restart_pool,
            ReusableKvCacheLimits::new(8, 1 << 17, 1 << 16, 1 << 20).unwrap(),
        )
        .unwrap();
        assert!(restart.describe(second_descriptor.cache_id(), &scope, 103).is_err());
    }
}
