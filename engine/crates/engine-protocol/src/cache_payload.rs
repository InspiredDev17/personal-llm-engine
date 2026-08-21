//! Bounded, path-free reusable K/V cache service payloads.

use crate::ServiceMessageKind;
pub use personalai_engine_knowledge_contract::{
    CacheAuthorizationScope as ServiceCacheScope, CacheNamespace as ServiceCacheNamespace,
};
use personalai_engine_knowledge_contract::{CacheBinding, KnowledgeIdentity};
use personalai_engine_types::{
    EngineError, MAX_REQUEST_BYTES, MAX_REUSABLE_CACHE_TTL_SECONDS, ModelId, ModelPackageHash,
    RequestId,
};

/// Maximum cache descriptors returned by one list request.
pub const MAX_SERVICE_CACHE_RESULTS: u16 = 256;
/// Exact counter count in canonical Knowledge 1.1 `CacheMetrics`.
pub const SERVICE_CACHE_METRIC_COUNT: usize = 16;
/// Largest integer represented exactly by canonical cross-language JSON consumers.
pub const MAX_SERVICE_CACHE_COUNTER: u64 = 9_007_199_254_740_991;

/// SHA-384-derived logical reusable-cache identifier.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ServiceCacheId(ModelPackageHash);

impl ServiceCacheId {
    /// Constructs from an exact SHA-384 value.
    pub const fn new(hash: ModelPackageHash) -> Self {
        Self(hash)
    }

    /// Returns the identifier hash.
    pub const fn hash(self) -> ModelPackageHash {
        self.0
    }
}

/// Optional source identity used for exact invalidation, never semantic matching.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ServiceCacheSource {
    source_id: KnowledgeIdentity,
    revision_id: KnowledgeIdentity,
}

impl ServiceCacheSource {
    /// Creates a nonzero exact source revision.
    pub const fn new(source_id: KnowledgeIdentity, revision_id: KnowledgeIdentity) -> Self {
        Self { source_id, revision_id }
    }

    /// Returns the source hash.
    pub const fn source_id(&self) -> &KnowledgeIdentity {
        &self.source_id
    }

    /// Returns the source revision.
    pub const fn revision_id(&self) -> &KnowledgeIdentity {
        &self.revision_id
    }
}

/// Bounded retention settings selected by the control plane.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ServiceCacheRetention {
    ttl_seconds: Option<u64>,
    priority: u8,
    pinned: bool,
}

impl ServiceCacheRetention {
    /// Constructs retention settings with the shared 365-day maximum.
    pub fn new(ttl_seconds: Option<u64>, priority: u8, pinned: bool) -> Result<Self, EngineError> {
        if ttl_seconds.is_some_and(|ttl| ttl == 0 || ttl > MAX_REUSABLE_CACHE_TTL_SECONDS) {
            return Err(EngineError::invalid_input("cache TTL is outside its bounded range"));
        }
        Ok(Self { ttl_seconds, priority, pinned })
    }

    /// Returns the optional TTL.
    pub const fn ttl_seconds(self) -> Option<u64> {
        self.ttl_seconds
    }

    /// Returns deterministic eviction priority.
    pub const fn priority(self) -> u8 {
        self.priority
    }

    /// Returns whether pressure eviction is prohibited.
    pub const fn pinned(self) -> bool {
        self.pinned
    }
}

/// Closed invalidation selector; no filesystem or semantic query is accepted.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ServiceCacheInvalidation {
    /// One exact logical cache.
    Exact(ServiceCacheId),
    /// Every cache in the authorized namespace/owner scope.
    Scope,
    /// Every cache for one exact logical model.
    LogicalModel(KnowledgeIdentity),
    /// Every cache with one exact source identity.
    Source(ServiceCacheSource),
}

/// Closed reusable-cache request catalog.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ServiceCacheRequest {
    /// Prefill bounded text and seal its exact token/KV prefix.
    Create {
        /// Loaded registered model.
        model_id: ModelId,
        /// Authorization scope.
        binding: Box<CacheBinding>,
        /// Exact Memory Vault prefix transport bytes; no normalization is permitted.
        prefix: Vec<u8>,
        /// Request-local hard maximum for the sealed cache object.
        maximum_cache_bytes: u64,
        /// Retention policy.
        retention: ServiceCacheRetention,
    },
    /// Describe one cache.
    Describe {
        /// Loaded registered model.
        model_id: ModelId,
        /// Exact cache ID.
        cache_id: ServiceCacheId,
        /// Authorized scope.
        scope: ServiceCacheScope,
    },
    /// List a bounded scope.
    List {
        /// Loaded registered model.
        model_id: ModelId,
        /// Authorized scope.
        scope: ServiceCacheScope,
        /// Maximum returned descriptors.
        maximum: u16,
    },
    /// Stage a cache for the generation request sharing this envelope request ID.
    Attach {
        /// Loaded registered model.
        model_id: ModelId,
        /// Exact cache ID.
        cache_id: ServiceCacheId,
        /// Generation request that will consume the attachment.
        target_request_id: RequestId,
        /// Authorized scope.
        binding: Box<CacheBinding>,
    },
    /// Release a staged attachment owned by another request ID.
    Release {
        /// Generation request whose staged attachment is released.
        target_request_id: RequestId,
        /// Exact cache identity.
        cache_id: ServiceCacheId,
        /// Producing engine instance identity.
        engine_instance_id: KnowledgeIdentity,
        /// Exact authorization scope.
        scope: ServiceCacheScope,
    },
    /// Invalidate through a closed exact selector.
    Invalidate {
        /// Loaded registered model.
        model_id: ModelId,
        /// Authorized scope.
        scope: ServiceCacheScope,
        /// Closed invalidation selector.
        selector: ServiceCacheInvalidation,
    },
    /// Query one loaded model's content-free cache metrics.
    Metrics {
        /// Loaded registered model.
        model_id: ModelId,
        /// Canonical logical model identity.
        logical_model_id: KnowledgeIdentity,
        /// Exact authorization scope.
        scope: ServiceCacheScope,
    },
}

impl ServiceCacheRequest {
    /// Returns the corresponding service message kind.
    pub const fn kind(&self) -> ServiceMessageKind {
        match self {
            Self::Create { .. } => ServiceMessageKind::CreateReusableCache,
            Self::Describe { .. } => ServiceMessageKind::DescribeCache,
            Self::List { .. } => ServiceMessageKind::ListCaches,
            Self::Attach { .. } => ServiceMessageKind::AttachCache,
            Self::Release { .. } => ServiceMessageKind::ReleaseCache,
            Self::Invalidate { .. } => ServiceMessageKind::InvalidateCache,
            Self::Metrics { .. } => ServiceMessageKind::CacheMetrics,
        }
    }

    pub(crate) fn encode(&self) -> Result<Vec<u8>, EngineError> {
        let mut out = Vec::new();
        match self {
            Self::Create { model_id, binding, prefix, maximum_cache_bytes, retention } => {
                if *maximum_cache_bytes == 0 || *maximum_cache_bytes > (1_u64 << 40) {
                    return Err(EngineError::invalid_input(
                        "maximum cache bytes is outside contract bounds",
                    ));
                }
                put_model(&mut out, model_id)?;
                put_binding(&mut out, binding)?;
                out.extend_from_slice(&maximum_cache_bytes.to_le_bytes());
                let ttl = retention.ttl_seconds().ok_or_else(|| {
                    EngineError::invalid_input("canonical cache creation requires a TTL")
                })?;
                out.extend_from_slice(&ttl.to_le_bytes());
                out.push(retention.priority());
                out.push(u8::from(retention.pinned()));
                put_bytes_u32(&mut out, prefix, "static knowledge prefix")?;
            }
            Self::Describe { model_id, cache_id, scope } => {
                put_model(&mut out, model_id)?;
                put_hash(&mut out, cache_id.hash());
                put_scope(&mut out, scope)?;
            }
            Self::Attach { model_id, cache_id, target_request_id, binding } => {
                put_model(&mut out, model_id)?;
                put_hash(&mut out, cache_id.hash());
                out.extend_from_slice(&target_request_id.to_bytes());
                put_binding(&mut out, binding)?;
            }
            Self::List { model_id, scope, maximum } => {
                validate_maximum(*maximum)?;
                put_model(&mut out, model_id)?;
                put_scope(&mut out, scope)?;
                out.extend_from_slice(&maximum.to_le_bytes());
            }
            Self::Release { target_request_id, cache_id, engine_instance_id, scope } => {
                out.extend_from_slice(&target_request_id.to_bytes());
                put_hash(&mut out, cache_id.hash());
                put_identity(&mut out, engine_instance_id)?;
                put_scope(&mut out, scope)?;
            }
            Self::Invalidate { model_id, scope, selector } => {
                put_model(&mut out, model_id)?;
                put_scope(&mut out, scope)?;
                match selector {
                    ServiceCacheInvalidation::Exact(id) => {
                        out.push(1);
                        put_hash(&mut out, id.hash());
                    }
                    ServiceCacheInvalidation::Scope => out.push(2),
                    ServiceCacheInvalidation::LogicalModel(model_id) => {
                        out.push(3);
                        put_identity(&mut out, model_id)?;
                    }
                    ServiceCacheInvalidation::Source(source) => {
                        out.push(4);
                        put_identity(&mut out, source.source_id())?;
                        put_identity(&mut out, source.revision_id())?;
                    }
                }
            }
            Self::Metrics { model_id, logical_model_id, scope } => {
                put_model(&mut out, model_id)?;
                put_identity(&mut out, logical_model_id)?;
                put_scope(&mut out, scope)?;
            }
        }
        if out.len() > MAX_REQUEST_BYTES {
            return Err(EngineError::resource_exhaustion("cache service payload is too large"));
        }
        Ok(out)
    }

    // Keeping the closed message catalog in one match makes unknown kinds and
    // trailing-data rejection auditable alongside every accepted shape.
    #[allow(clippy::too_many_lines)]
    pub(crate) fn decode(kind: ServiceMessageKind, payload: &[u8]) -> Result<Self, EngineError> {
        let mut cursor = Cursor::new(payload);
        let request = match kind {
            ServiceMessageKind::CreateReusableCache => {
                let model_id = cursor.model()?;
                let binding = Box::new(cursor.binding()?);
                let maximum_cache_bytes = cursor.u64()?;
                if maximum_cache_bytes == 0 || maximum_cache_bytes > (1_u64 << 40) {
                    return Err(EngineError::protocol_violation(
                        "maximum cache bytes is outside contract bounds",
                    ));
                }
                let ttl = cursor.u64()?;
                if ttl == 0 {
                    return Err(EngineError::protocol_violation(
                        "canonical cache creation requires a TTL",
                    ));
                }
                let priority = cursor.u8()?;
                let pinned = decode_bool(cursor.u8()?)?;
                let prefix = cursor.bytes_u32("static knowledge prefix")?.to_vec();
                if prefix.is_empty() || prefix.contains(&0) {
                    return Err(EngineError::protocol_violation(
                        "static knowledge prefix is invalid",
                    ));
                }
                Self::Create {
                    model_id,
                    binding,
                    prefix,
                    maximum_cache_bytes,
                    retention: ServiceCacheRetention::new(Some(ttl), priority, pinned).map_err(
                        |_| EngineError::protocol_violation("cache retention is invalid"),
                    )?,
                }
            }
            ServiceMessageKind::DescribeCache => Self::Describe {
                model_id: cursor.model()?,
                cache_id: ServiceCacheId::new(cursor.hash()?),
                scope: cursor.scope()?,
            },
            ServiceMessageKind::ListCaches => {
                let model_id = cursor.model()?;
                let scope = cursor.scope()?;
                let maximum = cursor.u16()?;
                validate_maximum(maximum)?;
                Self::List { model_id, scope, maximum }
            }
            ServiceMessageKind::AttachCache => {
                let model_id = cursor.model()?;
                let cache_id = ServiceCacheId::new(cursor.hash()?);
                let target_request_id =
                    RequestId::from_bytes(cursor.array_16()?).map_err(|_| {
                        EngineError::protocol_violation("cache target request ID is invalid")
                    })?;
                let binding = Box::new(cursor.binding()?);
                Self::Attach { model_id, cache_id, target_request_id, binding }
            }
            ServiceMessageKind::ReleaseCache => {
                let bytes = cursor.array_16()?;
                Self::Release {
                    target_request_id: RequestId::from_bytes(bytes).map_err(|_| {
                        EngineError::protocol_violation("cache release request ID is invalid")
                    })?,
                    cache_id: ServiceCacheId::new(cursor.hash()?),
                    engine_instance_id: cursor.identity()?,
                    scope: cursor.scope()?,
                }
            }
            ServiceMessageKind::InvalidateCache => {
                let model_id = cursor.model()?;
                let scope = cursor.scope()?;
                let selector = match cursor.u8()? {
                    1 => ServiceCacheInvalidation::Exact(ServiceCacheId::new(cursor.hash()?)),
                    2 => ServiceCacheInvalidation::Scope,
                    3 => ServiceCacheInvalidation::LogicalModel(cursor.identity()?),
                    4 => ServiceCacheInvalidation::Source(ServiceCacheSource::new(
                        cursor.identity()?,
                        cursor.identity()?,
                    )),
                    _ => {
                        return Err(EngineError::protocol_violation(
                            "cache invalidation selector is unsupported",
                        ));
                    }
                };
                Self::Invalidate { model_id, scope, selector }
            }
            ServiceMessageKind::CacheMetrics => Self::Metrics {
                model_id: cursor.model()?,
                logical_model_id: cursor.identity()?,
                scope: cursor.scope()?,
            },
            _ => return Err(EngineError::protocol_violation("message has no cache request body")),
        };
        cursor.finish()?;
        Ok(request)
    }
}

/// Path-free cache descriptor returned to an authorized caller.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ServiceCacheDescriptor {
    cache_id: ServiceCacheId,
    binding: Box<CacheBinding>,
    byte_size: u64,
    created_at_unix_seconds: u64,
    verified_at_unix_seconds: u64,
    expires_at_unix_seconds: Option<u64>,
    last_access_unix_seconds: u64,
    hit_count: u64,
    pinned: bool,
    resident: bool,
}

impl ServiceCacheDescriptor {
    /// Constructs one validated content-free descriptor.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        cache_id: ServiceCacheId,
        binding: CacheBinding,
        byte_size: u64,
        created_at_unix_seconds: u64,
        verified_at_unix_seconds: u64,
        expires_at_unix_seconds: Option<u64>,
        last_access_unix_seconds: u64,
        hit_count: u64,
        pinned: bool,
        resident: bool,
    ) -> Result<Self, EngineError> {
        if byte_size == 0
            || created_at_unix_seconds == 0
            || verified_at_unix_seconds < created_at_unix_seconds
            || expires_at_unix_seconds.is_some_and(|expiry| expiry <= verified_at_unix_seconds)
        {
            return Err(EngineError::invalid_input("cache descriptor bounds are invalid"));
        }
        Ok(Self {
            cache_id,
            binding: Box::new(binding),
            byte_size,
            created_at_unix_seconds,
            verified_at_unix_seconds,
            expires_at_unix_seconds,
            last_access_unix_seconds,
            hit_count,
            pinned,
            resident,
        })
    }

    /// Returns cache ID.
    pub const fn cache_id(&self) -> ServiceCacheId {
        self.cache_id
    }
    /// Returns model ID.
    pub const fn binding(&self) -> &CacheBinding {
        &self.binding
    }
    /// Returns authorization scope.
    pub const fn scope(&self) -> &ServiceCacheScope {
        self.binding.scope()
    }
    /// Returns prefix tokens.
    pub const fn token_count(&self) -> u32 {
        self.binding.token_count()
    }
    /// Returns accounted bytes.
    pub const fn byte_size(&self) -> u64 {
        self.byte_size
    }
    /// Returns creation time.
    pub const fn created_at_unix_seconds(&self) -> u64 {
        self.created_at_unix_seconds
    }
    /// Returns trusted verification time.
    pub const fn verified_at_unix_seconds(&self) -> u64 {
        self.verified_at_unix_seconds
    }
    /// Returns trusted expiry time when TTL-bound.
    pub const fn expires_at_unix_seconds(&self) -> Option<u64> {
        self.expires_at_unix_seconds
    }
    /// Returns last access time.
    pub const fn last_access_unix_seconds(&self) -> u64 {
        self.last_access_unix_seconds
    }
    /// Returns hit count.
    pub const fn hit_count(&self) -> u64 {
        self.hit_count
    }
    /// Returns pin state.
    pub const fn pinned(&self) -> bool {
        self.pinned
    }
    /// Returns whether the cache is currently resident.
    pub const fn resident(&self) -> bool {
        self.resident
    }
}

/// Fixed content-free reusable-cache metrics wire representation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ServiceCacheMetrics(pub [u64; SERVICE_CACHE_METRIC_COUNT]);

impl ServiceCacheMetrics {
    fn validate(self) -> Result<Self, EngineError> {
        if self.0.iter().any(|value| *value > MAX_SERVICE_CACHE_COUNTER)
            || self.0[1] > self.0[0]
            || self.0[2] > self.0[0]
            || self.0[3] > self.0[1]
            || self.0[4] > self.0[1]
        {
            return Err(EngineError::protocol_violation(
                "cache metric counters violate canonical bounds",
            ));
        }
        Ok(self)
    }
}

/// Closed cache response catalog.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ServiceCacheResponse {
    /// Cache creation completed.
    Created(ServiceCacheDescriptor),
    /// An exact pre-existing cache object satisfied an idempotent create.
    Reused(ServiceCacheDescriptor),
    /// One descriptor was returned.
    Described(ServiceCacheDescriptor),
    /// The authorized cache identity is absent or unusable.
    Missing,
    /// A bounded descriptor list was returned.
    Listed {
        /// Authorized descriptor prefix.
        caches: Vec<ServiceCacheDescriptor>,
        /// Whether additional authorized descriptors exist.
        truncated: bool,
    },
    /// Attachment was staged for correlated generation.
    Attached {
        /// Exact cache ID.
        cache_id: ServiceCacheId,
        /// Exact cached prefix length.
        token_count: u32,
    },
    /// A staged attachment was released.
    Released,
    /// The exact attachment had already been released or consumed.
    AlreadyReleased,
    /// Number of invalidated cache objects.
    Invalidated(u32),
    /// Content-free cache metrics.
    Metrics(ServiceCacheMetrics),
}

impl ServiceCacheResponse {
    /// Returns the corresponding service message kind.
    pub const fn kind(&self) -> ServiceMessageKind {
        match self {
            Self::Created(_) | Self::Reused(_) => ServiceMessageKind::CreateReusableCache,
            Self::Described(_) | Self::Missing => ServiceMessageKind::DescribeCache,
            Self::Listed { .. } => ServiceMessageKind::ListCaches,
            Self::Attached { .. } => ServiceMessageKind::AttachCache,
            Self::Released | Self::AlreadyReleased => ServiceMessageKind::ReleaseCache,
            Self::Invalidated(_) => ServiceMessageKind::InvalidateCache,
            Self::Metrics(_) => ServiceMessageKind::CacheMetrics,
        }
    }

    pub(crate) fn encode(&self) -> Result<Vec<u8>, EngineError> {
        let mut out = Vec::new();
        match self {
            Self::Created(value) => {
                out.push(1);
                put_descriptor(&mut out, value)?;
            }
            Self::Reused(value) => {
                out.push(2);
                put_descriptor(&mut out, value)?;
            }
            Self::Described(value) => {
                out.push(1);
                put_descriptor(&mut out, value)?;
            }
            Self::Missing | Self::AlreadyReleased => out.push(2),
            Self::Listed { caches, truncated } => {
                let count = u16::try_from(caches.len()).map_err(|_| {
                    EngineError::resource_exhaustion("cache descriptor count exceeds u16")
                })?;
                validate_maximum(count.max(1))?;
                out.extend_from_slice(&count.to_le_bytes());
                out.push(u8::from(*truncated));
                for value in caches {
                    put_descriptor(&mut out, value)?;
                }
            }
            Self::Attached { cache_id, token_count } => {
                if *token_count == 0 {
                    return Err(EngineError::invalid_input("cache token count is zero"));
                }
                put_hash(&mut out, cache_id.hash());
                out.extend_from_slice(&token_count.to_le_bytes());
            }
            Self::Released => out.push(1),
            Self::Invalidated(count) => out.extend_from_slice(&count.to_le_bytes()),
            Self::Metrics(metrics) => {
                for value in metrics.validate()?.0 {
                    out.extend_from_slice(&value.to_le_bytes());
                }
            }
        }
        Ok(out)
    }

    pub(crate) fn decode(kind: ServiceMessageKind, payload: &[u8]) -> Result<Self, EngineError> {
        let mut cursor = Cursor::new(payload);
        let response = match kind {
            ServiceMessageKind::CreateReusableCache => match cursor.u8()? {
                1 => Self::Created(cursor.descriptor()?),
                2 => Self::Reused(cursor.descriptor()?),
                _ => {
                    return Err(EngineError::protocol_violation("cache create status is invalid"));
                }
            },
            ServiceMessageKind::DescribeCache => match cursor.u8()? {
                1 => Self::Described(cursor.descriptor()?),
                2 => Self::Missing,
                _ => {
                    return Err(EngineError::protocol_violation(
                        "cache describe status is invalid",
                    ));
                }
            },
            ServiceMessageKind::ListCaches => {
                let count = cursor.u16()?;
                if count > MAX_SERVICE_CACHE_RESULTS {
                    return Err(EngineError::protocol_violation("cache list exceeds bound"));
                }
                let truncated = decode_bool(cursor.u8()?)?;
                let mut caches = Vec::new();
                caches.try_reserve_exact(usize::from(count)).map_err(|_| {
                    EngineError::resource_exhaustion("cache list allocation failed")
                })?;
                for _ in 0..count {
                    caches.push(cursor.descriptor()?);
                }
                Self::Listed { caches, truncated }
            }
            ServiceMessageKind::AttachCache => {
                let cache_id = ServiceCacheId::new(cursor.hash()?);
                let token_count = cursor.u32()?;
                if token_count == 0 {
                    return Err(EngineError::protocol_violation("cache token count is zero"));
                }
                Self::Attached { cache_id, token_count }
            }
            ServiceMessageKind::ReleaseCache => match cursor.u8()? {
                1 => Self::Released,
                2 => Self::AlreadyReleased,
                _ => {
                    return Err(EngineError::protocol_violation("cache release status is invalid"));
                }
            },
            ServiceMessageKind::InvalidateCache => Self::Invalidated(cursor.u32()?),
            ServiceMessageKind::CacheMetrics => {
                let mut values = [0_u64; SERVICE_CACHE_METRIC_COUNT];
                for value in &mut values {
                    *value = cursor.u64()?;
                }
                Self::Metrics(ServiceCacheMetrics(values).validate()?)
            }
            _ => return Err(EngineError::protocol_violation("message has no cache response body")),
        };
        cursor.finish()?;
        Ok(response)
    }
}

fn put_descriptor(out: &mut Vec<u8>, value: &ServiceCacheDescriptor) -> Result<(), EngineError> {
    put_hash(out, value.cache_id.hash());
    put_binding(out, &value.binding)?;
    out.extend_from_slice(&value.byte_size.to_le_bytes());
    out.extend_from_slice(&value.created_at_unix_seconds.to_le_bytes());
    out.extend_from_slice(&value.verified_at_unix_seconds.to_le_bytes());
    out.extend_from_slice(&value.expires_at_unix_seconds.unwrap_or(0).to_le_bytes());
    out.extend_from_slice(&value.last_access_unix_seconds.to_le_bytes());
    out.extend_from_slice(&value.hit_count.to_le_bytes());
    out.push(u8::from(value.pinned));
    out.push(u8::from(value.resident));
    Ok(())
}

fn put_model(out: &mut Vec<u8>, model: &ModelId) -> Result<(), EngineError> {
    put_bytes_u16(out, model.as_str().as_bytes(), "model ID")
}

fn put_scope(out: &mut Vec<u8>, scope: &ServiceCacheScope) -> Result<(), EngineError> {
    out.push(scope.namespace() as u8);
    put_identity(out, scope.owner_id())?;
    put_optional_identity(out, scope.project_id())?;
    put_optional_identity(out, scope.task_id())?;
    put_optional_identity(out, scope.agent_id())
}

fn put_identity(out: &mut Vec<u8>, identity: &KnowledgeIdentity) -> Result<(), EngineError> {
    put_bytes_u16(out, identity.as_str().as_bytes(), "knowledge identity")
}

fn put_optional_identity(
    out: &mut Vec<u8>,
    identity: Option<&KnowledgeIdentity>,
) -> Result<(), EngineError> {
    match identity {
        Some(identity) => {
            out.push(1);
            put_identity(out, identity)?;
        }
        None => out.push(0),
    }
    Ok(())
}

fn put_binding(out: &mut Vec<u8>, binding: &CacheBinding) -> Result<(), EngineError> {
    put_bytes_u32(out, &binding.encode_binary()?, "cache binding")
}

fn put_hash(out: &mut Vec<u8>, hash: ModelPackageHash) {
    out.extend_from_slice(hash.as_bytes());
}

fn put_bytes_u16(out: &mut Vec<u8>, value: &[u8], label: &'static str) -> Result<(), EngineError> {
    let length = u16::try_from(value.len()).map_err(|_| EngineError::resource_exhaustion(label))?;
    out.extend_from_slice(&length.to_le_bytes());
    out.extend_from_slice(value);
    Ok(())
}

fn put_bytes_u32(out: &mut Vec<u8>, value: &[u8], label: &'static str) -> Result<(), EngineError> {
    let length = u32::try_from(value.len()).map_err(|_| EngineError::resource_exhaustion(label))?;
    out.extend_from_slice(&length.to_le_bytes());
    out.extend_from_slice(value);
    Ok(())
}

fn validate_maximum(value: u16) -> Result<(), EngineError> {
    if value == 0 || value > MAX_SERVICE_CACHE_RESULTS {
        return Err(EngineError::protocol_violation("cache result maximum is outside bounds"));
    }
    Ok(())
}

fn decode_bool(value: u8) -> Result<bool, EngineError> {
    match value {
        0 => Ok(false),
        1 => Ok(true),
        _ => Err(EngineError::protocol_violation("cache boolean is invalid")),
    }
}

struct Cursor<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> Cursor<'a> {
    const fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }
    fn take(&mut self, length: usize) -> Result<&'a [u8], EngineError> {
        let end = self
            .offset
            .checked_add(length)
            .filter(|end| *end <= self.bytes.len())
            .ok_or_else(|| EngineError::protocol_violation("cache payload is truncated"))?;
        let value = &self.bytes[self.offset..end];
        self.offset = end;
        Ok(value)
    }
    fn u8(&mut self) -> Result<u8, EngineError> {
        Ok(self.take(1)?[0])
    }
    fn u16(&mut self) -> Result<u16, EngineError> {
        Ok(u16::from_le_bytes(
            self.take(2)?
                .try_into()
                .map_err(|_| EngineError::protocol_violation("u16 is truncated"))?,
        ))
    }
    fn u32(&mut self) -> Result<u32, EngineError> {
        Ok(u32::from_le_bytes(
            self.take(4)?
                .try_into()
                .map_err(|_| EngineError::protocol_violation("u32 is truncated"))?,
        ))
    }
    fn u64(&mut self) -> Result<u64, EngineError> {
        Ok(u64::from_le_bytes(
            self.take(8)?
                .try_into()
                .map_err(|_| EngineError::protocol_violation("u64 is truncated"))?,
        ))
    }
    fn array_16(&mut self) -> Result<[u8; 16], EngineError> {
        self.take(16)?
            .try_into()
            .map_err(|_| EngineError::protocol_violation("request ID is truncated"))
    }
    fn hash(&mut self) -> Result<ModelPackageHash, EngineError> {
        let bytes: [u8; ModelPackageHash::BYTE_LEN] = self
            .take(ModelPackageHash::BYTE_LEN)?
            .try_into()
            .map_err(|_| EngineError::protocol_violation("cache hash is truncated"))?;
        Ok(ModelPackageHash::from_sha384_bytes(bytes))
    }
    fn model(&mut self) -> Result<ModelId, EngineError> {
        let value = self.string_u16("model ID")?;
        ModelId::new(value).map_err(|_| EngineError::protocol_violation("model ID is invalid"))
    }
    fn scope(&mut self) -> Result<ServiceCacheScope, EngineError> {
        let namespace = ServiceCacheNamespace::from_u8(self.u8()?)?;
        let owner = self.identity()?;
        ServiceCacheScope::new(
            namespace,
            owner,
            self.optional_identity()?,
            self.optional_identity()?,
            self.optional_identity()?,
        )
        .map_err(|_| EngineError::protocol_violation("cache scope is invalid"))
    }
    fn identity(&mut self) -> Result<KnowledgeIdentity, EngineError> {
        KnowledgeIdentity::new(self.string_u16("knowledge identity")?)
            .map_err(|_| EngineError::protocol_violation("knowledge identity is invalid"))
    }
    fn optional_identity(&mut self) -> Result<Option<KnowledgeIdentity>, EngineError> {
        match self.u8()? {
            0 => Ok(None),
            1 => self.identity().map(Some),
            _ => Err(EngineError::protocol_violation("optional identity flag is invalid")),
        }
    }
    fn binding(&mut self) -> Result<CacheBinding, EngineError> {
        CacheBinding::decode_binary(self.bytes_u32("cache binding")?)
    }
    fn string_u16(&mut self, label: &'static str) -> Result<String, EngineError> {
        let length = usize::from(self.u16()?);
        self.string(length, label)
    }
    fn bytes_u32(&mut self, label: &'static str) -> Result<&'a [u8], EngineError> {
        let length =
            usize::try_from(self.u32()?).map_err(|_| EngineError::protocol_violation(label))?;
        self.take(length)
    }
    fn string(&mut self, length: usize, label: &'static str) -> Result<String, EngineError> {
        let value = std::str::from_utf8(self.take(length)?)
            .map_err(|_| EngineError::protocol_violation(label))?;
        Ok(value.to_owned())
    }
    fn descriptor(&mut self) -> Result<ServiceCacheDescriptor, EngineError> {
        let cache_id = ServiceCacheId::new(self.hash()?);
        let binding = self.binding()?;
        let byte_size = self.u64()?;
        let created = self.u64()?;
        let verified = self.u64()?;
        let expiry = self.u64()?;
        let accessed = self.u64()?;
        let hits = self.u64()?;
        let pinned = decode_bool(self.u8()?)?;
        let resident = decode_bool(self.u8()?)?;
        ServiceCacheDescriptor::new(
            cache_id,
            binding,
            byte_size,
            created,
            verified,
            (expiry != 0).then_some(expiry),
            accessed,
            hits,
            pinned,
            resident,
        )
        .map_err(|_| EngineError::protocol_violation("cache descriptor is invalid"))
    }
    fn finish(self) -> Result<(), EngineError> {
        if self.offset == self.bytes.len() {
            Ok(())
        } else {
            Err(EngineError::protocol_violation("cache payload has trailing bytes"))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use personalai_engine_crypto::hash_sha256;
    use personalai_engine_knowledge_contract::{
        ArtifactDigest, CacheRuntimeIdentity, CacheSensitivity, SourceRevisionBinding,
        StaticBundleBinding,
    };
    use personalai_engine_types::{ArchitectureId, BackendId, DType, Sha256Digest};

    fn hash(byte: u8) -> ModelPackageHash {
        ModelPackageHash::from_sha384_bytes([byte; 48])
    }
    fn scope() -> ServiceCacheScope {
        ServiceCacheScope::new(
            ServiceCacheNamespace::Project,
            KnowledgeIdentity::new("usr_owner").unwrap(),
            Some(KnowledgeIdentity::new("prj_a").unwrap()),
            None,
            None,
        )
        .unwrap()
    }
    fn binding(token_count: u32, prefix_digest: Sha256Digest) -> CacheBinding {
        CacheBinding::new(
            CacheRuntimeIdentity::new(
                KnowledgeIdentity::new("mdl_logical").unwrap(),
                ModelId::new("tiny").unwrap(),
                KnowledgeIdentity::new("artifact.tiny").unwrap(),
                ArtifactDigest::Sha384(hash(1)),
                BackendId::new("cpu-reference").unwrap(),
                KnowledgeIdentity::new("version.1").unwrap(),
                KnowledgeIdentity::new("backend.one").unwrap(),
                KnowledgeIdentity::new("engine.one").unwrap(),
            )
            .unwrap(),
            KnowledgeIdentity::new("tokenizer.tiny").unwrap(),
            Sha256Digest::from_bytes([2; 32]),
            ArchitectureId::new("decoder-rms-rope-swiglu-gqa-v1").unwrap(),
            KnowledgeIdentity::new("rope.causal-zero-based-v1").unwrap(),
            DType::F32,
            StaticBundleBinding::new(
                KnowledgeIdentity::new("bundle.tiny").unwrap(),
                1,
                Sha256Digest::from_bytes([3; 32]),
                prefix_digest,
            )
            .unwrap(),
            scope(),
            CacheSensitivity::Internal,
            token_count,
        )
        .unwrap()
        .with_source_revision(SourceRevisionBinding::new(
            KnowledgeIdentity::new("ksr_source").unwrap(),
            KnowledgeIdentity::new("krv_revision").unwrap(),
        ))
    }

    #[test]
    fn every_cache_request_round_trips_and_trailing_data_fails() {
        assert!(KnowledgeIdentity::new("../project/a").is_err());
        assert!(ServiceCacheRetention::new(Some(0), 1, false).is_err());
        assert!(ServiceCacheRetention::new(Some(MAX_REUSABLE_CACHE_TTL_SECONDS), 1, false).is_ok());
        assert!(
            ServiceCacheRetention::new(Some(MAX_REUSABLE_CACHE_TTL_SECONDS + 1), 1, false).is_err()
        );
        let model_id = ModelId::new("tiny").unwrap();
        let id = ServiceCacheId::new(hash(1));
        let prefix = b"SOVEREIGN-STATIC-KNOWLEDGE-PREFIX-V1\n6:prefix".to_vec();
        let binding = binding(2, hash_sha256(&prefix));
        let requests = vec![
            ServiceCacheRequest::Create {
                model_id: model_id.clone(),
                binding: Box::new(binding.clone()),
                prefix,
                maximum_cache_bytes: 1 << 20,
                retention: ServiceCacheRetention::new(Some(60), 2, true).unwrap(),
            },
            ServiceCacheRequest::Describe {
                model_id: model_id.clone(),
                cache_id: id,
                scope: scope(),
            },
            ServiceCacheRequest::List { model_id: model_id.clone(), scope: scope(), maximum: 8 },
            ServiceCacheRequest::Attach {
                model_id: model_id.clone(),
                cache_id: id,
                target_request_id: RequestId::new(8).unwrap(),
                binding: Box::new(binding.clone()),
            },
            ServiceCacheRequest::Release {
                target_request_id: RequestId::new(9).unwrap(),
                cache_id: id,
                engine_instance_id: binding.runtime().engine_instance_id().clone(),
                scope: scope(),
            },
            ServiceCacheRequest::Invalidate {
                model_id: model_id.clone(),
                scope: scope(),
                selector: ServiceCacheInvalidation::Source(ServiceCacheSource::new(
                    KnowledgeIdentity::new("ksr_source").unwrap(),
                    KnowledgeIdentity::new("krv_revision").unwrap(),
                )),
            },
            ServiceCacheRequest::Metrics {
                model_id,
                logical_model_id: KnowledgeIdentity::new("mdl_logical").unwrap(),
                scope: scope(),
            },
        ];
        for request in requests {
            let encoded = request.encode().unwrap();
            assert_eq!(ServiceCacheRequest::decode(request.kind(), &encoded).unwrap(), request);
            let mut trailing = encoded;
            trailing.push(0);
            assert!(ServiceCacheRequest::decode(request.kind(), &trailing).is_err());
        }
    }

    #[test]
    fn descriptors_and_metrics_round_trip() {
        let descriptor = ServiceCacheDescriptor::new(
            ServiceCacheId::new(hash(4)),
            binding(3, Sha256Digest::from_bytes([4; 32])),
            99,
            1,
            1,
            Some(60),
            2,
            3,
            false,
            true,
        )
        .unwrap();
        let responses = vec![
            ServiceCacheResponse::Created(descriptor.clone()),
            ServiceCacheResponse::Reused(descriptor.clone()),
            ServiceCacheResponse::Described(descriptor.clone()),
            ServiceCacheResponse::Missing,
            ServiceCacheResponse::Listed { caches: vec![descriptor], truncated: true },
            ServiceCacheResponse::Attached {
                cache_id: ServiceCacheId::new(hash(5)),
                token_count: 3,
            },
            ServiceCacheResponse::Released,
            ServiceCacheResponse::AlreadyReleased,
            ServiceCacheResponse::Invalidated(2),
            ServiceCacheResponse::Metrics(ServiceCacheMetrics([7; SERVICE_CACHE_METRIC_COUNT])),
        ];
        for response in responses {
            let encoded = response.encode().unwrap();
            assert_eq!(ServiceCacheResponse::decode(response.kind(), &encoded).unwrap(), response);
        }
        assert!(
            ServiceCacheResponse::Metrics(ServiceCacheMetrics(
                [u64::MAX; SERVICE_CACHE_METRIC_COUNT]
            ))
            .encode()
            .is_err()
        );
    }

    #[test]
    fn json_and_duplicate_keys_are_never_an_engine_deserialization_surface() {
        let duplicate = br#"{"contract_version":"1.1.0","contract_version":"1.1.0","kind":"knowledge.cag_cache_create_request"}"#;
        assert!(
            ServiceCacheRequest::decode(ServiceMessageKind::CreateReusableCache, duplicate,)
                .is_err()
        );
    }
}
