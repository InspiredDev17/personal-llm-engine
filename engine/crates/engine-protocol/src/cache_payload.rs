//! Bounded, path-free reusable K/V cache service payloads.

use crate::ServiceMessageKind;
use personalai_engine_types::{
    EngineError, MAX_REQUEST_BYTES, MAX_REUSABLE_CACHE_TTL_SECONDS, ModelId, ModelPackageHash,
    RequestId,
};

/// Maximum cache descriptors returned by one list request.
pub const MAX_SERVICE_CACHE_RESULTS: u16 = 256;
/// Maximum cache scope owner bytes.
pub const MAX_SERVICE_CACHE_OWNER_BYTES: usize = 96;

/// Closed authorization namespace carried from the Go control plane.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum ServiceCacheNamespace {
    /// Explicitly reviewed globally shareable static prefix.
    GlobalSafeStatic = 1,
    /// One user scope.
    User = 2,
    /// One project scope.
    Project = 3,
    /// One task scope.
    Task = 4,
    /// One agent scope.
    Agent = 5,
}

impl ServiceCacheNamespace {
    fn decode(value: u8) -> Result<Self, EngineError> {
        match value {
            1 => Ok(Self::GlobalSafeStatic),
            2 => Ok(Self::User),
            3 => Ok(Self::Project),
            4 => Ok(Self::Task),
            5 => Ok(Self::Agent),
            _ => Err(EngineError::protocol_violation("cache namespace is unsupported")),
        }
    }
}

/// Exact authorization scope; owner is a bounded opaque identifier, not content.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ServiceCacheScope {
    namespace: ServiceCacheNamespace,
    owner: String,
}

impl ServiceCacheScope {
    /// Constructs a validated scope.
    pub fn new(
        namespace: ServiceCacheNamespace,
        owner: impl Into<String>,
    ) -> Result<Self, EngineError> {
        let owner = owner.into();
        if owner.is_empty()
            || owner.len() > MAX_SERVICE_CACHE_OWNER_BYTES
            || !owner.bytes().all(|byte| {
                byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.' | b':')
            })
            || owner.contains("..")
        {
            return Err(EngineError::invalid_input(
                "cache scope owner is outside the canonical path-free grammar",
            ));
        }
        Ok(Self { namespace, owner })
    }

    /// Returns the closed namespace.
    pub const fn namespace(&self) -> ServiceCacheNamespace {
        self.namespace
    }

    /// Returns the bounded opaque owner identifier.
    pub fn owner(&self) -> &str {
        &self.owner
    }
}

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
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ServiceCacheSource {
    hash: ModelPackageHash,
    revision: u64,
}

impl ServiceCacheSource {
    /// Creates a nonzero exact source revision.
    pub fn new(hash: ModelPackageHash, revision: u64) -> Result<Self, EngineError> {
        if revision == 0 {
            return Err(EngineError::invalid_input("cache source revision must be nonzero"));
        }
        Ok(Self { hash, revision })
    }

    /// Returns the source hash.
    pub const fn hash(self) -> ModelPackageHash {
        self.hash
    }

    /// Returns the source revision.
    pub const fn revision(self) -> u64 {
        self.revision
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
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ServiceCacheInvalidation {
    /// One exact logical cache.
    Exact(ServiceCacheId),
    /// Every cache in the authorized namespace/owner scope.
    Scope,
    /// Every cache for the loaded model.
    Model,
    /// Every cache with one exact source identity.
    Source(ServiceCacheSource),
    /// Every cache with one exact source revision.
    Revision(u64),
}

/// Closed reusable-cache request catalog.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ServiceCacheRequest {
    /// Prefill bounded text and seal its exact token/KV prefix.
    Create {
        /// Loaded registered model.
        model_id: ModelId,
        /// Authorization scope.
        scope: ServiceCacheScope,
        /// Prompt text already assembled by Go.
        text: String,
        /// Optional exact source identity.
        source: Option<ServiceCacheSource>,
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
        /// Authorized scope.
        scope: ServiceCacheScope,
        /// Exact expected prefix length.
        expected_token_count: u32,
    },
    /// Release a staged attachment owned by another request ID.
    Release {
        /// Generation request whose staged attachment is released.
        target_request_id: RequestId,
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
            Self::Create { model_id, scope, text, source, retention } => {
                put_model(&mut out, model_id)?;
                put_scope(&mut out, scope)?;
                out.push(u8::from(source.is_some()));
                if let Some(source) = source {
                    put_hash(&mut out, source.hash());
                    out.extend_from_slice(&source.revision().to_le_bytes());
                }
                out.extend_from_slice(&retention.ttl_seconds().unwrap_or(0).to_le_bytes());
                out.push(retention.priority());
                out.push(u8::from(retention.pinned()));
                put_bytes_u32(&mut out, text.as_bytes(), "cache text")?;
            }
            Self::Describe { model_id, cache_id, scope }
            | Self::Attach { model_id, cache_id, scope, expected_token_count: _ } => {
                put_model(&mut out, model_id)?;
                put_hash(&mut out, cache_id.hash());
                put_scope(&mut out, scope)?;
                if let Self::Attach { expected_token_count, .. } = self {
                    if *expected_token_count == 0 {
                        return Err(EngineError::invalid_input(
                            "expected cache token count must be nonzero",
                        ));
                    }
                    out.extend_from_slice(&expected_token_count.to_le_bytes());
                }
            }
            Self::List { model_id, scope, maximum } => {
                validate_maximum(*maximum)?;
                put_model(&mut out, model_id)?;
                put_scope(&mut out, scope)?;
                out.extend_from_slice(&maximum.to_le_bytes());
            }
            Self::Release { target_request_id } => {
                out.extend_from_slice(&target_request_id.to_bytes());
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
                    ServiceCacheInvalidation::Model => out.push(3),
                    ServiceCacheInvalidation::Source(source) => {
                        out.push(4);
                        put_hash(&mut out, source.hash());
                        out.extend_from_slice(&source.revision().to_le_bytes());
                    }
                    ServiceCacheInvalidation::Revision(revision) => {
                        if *revision == 0 {
                            return Err(EngineError::invalid_input(
                                "cache revision must be nonzero",
                            ));
                        }
                        out.push(5);
                        out.extend_from_slice(&revision.to_le_bytes());
                    }
                }
            }
            Self::Metrics { model_id } => put_model(&mut out, model_id)?,
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
                let scope = cursor.scope()?;
                let source =
                    match cursor.u8()? {
                        0 => None,
                        1 => Some(ServiceCacheSource::new(cursor.hash()?, cursor.u64()?).map_err(
                            |_| EngineError::protocol_violation("cache source is invalid"),
                        )?),
                        _ => {
                            return Err(EngineError::protocol_violation(
                                "cache source flag is invalid",
                            ));
                        }
                    };
                let ttl = cursor.u64()?;
                let priority = cursor.u8()?;
                let pinned = decode_bool(cursor.u8()?)?;
                let text = cursor.string_u32("cache text")?;
                if text.is_empty() || text.as_bytes().contains(&0) {
                    return Err(EngineError::protocol_violation("cache text is invalid"));
                }
                Self::Create {
                    model_id,
                    scope,
                    text,
                    source,
                    retention: ServiceCacheRetention::new(
                        (ttl != 0).then_some(ttl),
                        priority,
                        pinned,
                    )
                    .map_err(|_| EngineError::protocol_violation("cache retention is invalid"))?,
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
                let scope = cursor.scope()?;
                let expected_token_count = cursor.u32()?;
                if expected_token_count == 0 {
                    return Err(EngineError::protocol_violation(
                        "expected cache token count must be nonzero",
                    ));
                }
                Self::Attach { model_id, cache_id, scope, expected_token_count }
            }
            ServiceMessageKind::ReleaseCache => {
                let bytes = cursor.array_16()?;
                Self::Release {
                    target_request_id: RequestId::from_bytes(bytes).map_err(|_| {
                        EngineError::protocol_violation("cache release request ID is invalid")
                    })?,
                }
            }
            ServiceMessageKind::InvalidateCache => {
                let model_id = cursor.model()?;
                let scope = cursor.scope()?;
                let selector = match cursor.u8()? {
                    1 => ServiceCacheInvalidation::Exact(ServiceCacheId::new(cursor.hash()?)),
                    2 => ServiceCacheInvalidation::Scope,
                    3 => ServiceCacheInvalidation::Model,
                    4 => ServiceCacheInvalidation::Source(
                        ServiceCacheSource::new(cursor.hash()?, cursor.u64()?).map_err(|_| {
                            EngineError::protocol_violation("cache source is invalid")
                        })?,
                    ),
                    5 => {
                        let revision = cursor.u64()?;
                        if revision == 0 {
                            return Err(EngineError::protocol_violation(
                                "cache revision must be nonzero",
                            ));
                        }
                        ServiceCacheInvalidation::Revision(revision)
                    }
                    _ => {
                        return Err(EngineError::protocol_violation(
                            "cache invalidation selector is unsupported",
                        ));
                    }
                };
                Self::Invalidate { model_id, scope, selector }
            }
            ServiceMessageKind::CacheMetrics => Self::Metrics { model_id: cursor.model()? },
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
    model_id: ModelId,
    scope: ServiceCacheScope,
    token_count: u32,
    byte_size: u64,
    created_at_unix_seconds: u64,
    last_access_unix_seconds: u64,
    hit_count: u64,
    pinned: bool,
}

impl ServiceCacheDescriptor {
    /// Constructs one validated content-free descriptor.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        cache_id: ServiceCacheId,
        model_id: ModelId,
        scope: ServiceCacheScope,
        token_count: u32,
        byte_size: u64,
        created_at_unix_seconds: u64,
        last_access_unix_seconds: u64,
        hit_count: u64,
        pinned: bool,
    ) -> Result<Self, EngineError> {
        if token_count == 0 || byte_size == 0 || created_at_unix_seconds == 0 {
            return Err(EngineError::invalid_input("cache descriptor bounds are invalid"));
        }
        Ok(Self {
            cache_id,
            model_id,
            scope,
            token_count,
            byte_size,
            created_at_unix_seconds,
            last_access_unix_seconds,
            hit_count,
            pinned,
        })
    }

    /// Returns cache ID.
    pub const fn cache_id(&self) -> ServiceCacheId {
        self.cache_id
    }
    /// Returns model ID.
    pub const fn model_id(&self) -> &ModelId {
        &self.model_id
    }
    /// Returns authorization scope.
    pub const fn scope(&self) -> &ServiceCacheScope {
        &self.scope
    }
    /// Returns prefix tokens.
    pub const fn token_count(&self) -> u32 {
        self.token_count
    }
    /// Returns accounted bytes.
    pub const fn byte_size(&self) -> u64 {
        self.byte_size
    }
    /// Returns creation time.
    pub const fn created_at_unix_seconds(&self) -> u64 {
        self.created_at_unix_seconds
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
}

/// Fixed content-free reusable-cache metrics wire representation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ServiceCacheMetrics(pub [u64; 17]);

/// Closed cache response catalog.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ServiceCacheResponse {
    /// Cache creation completed.
    Created(ServiceCacheDescriptor),
    /// One descriptor was returned.
    Described(ServiceCacheDescriptor),
    /// A bounded descriptor list was returned.
    Listed(Vec<ServiceCacheDescriptor>),
    /// Attachment was staged for correlated generation.
    Attached {
        /// Exact cache ID.
        cache_id: ServiceCacheId,
        /// Exact cached prefix length.
        token_count: u32,
    },
    /// A staged attachment was released.
    Released,
    /// Number of invalidated cache objects.
    Invalidated(u32),
    /// Content-free cache metrics.
    Metrics(ServiceCacheMetrics),
}

impl ServiceCacheResponse {
    /// Returns the corresponding service message kind.
    pub const fn kind(&self) -> ServiceMessageKind {
        match self {
            Self::Created(_) => ServiceMessageKind::CreateReusableCache,
            Self::Described(_) => ServiceMessageKind::DescribeCache,
            Self::Listed(_) => ServiceMessageKind::ListCaches,
            Self::Attached { .. } => ServiceMessageKind::AttachCache,
            Self::Released => ServiceMessageKind::ReleaseCache,
            Self::Invalidated(_) => ServiceMessageKind::InvalidateCache,
            Self::Metrics(_) => ServiceMessageKind::CacheMetrics,
        }
    }

    pub(crate) fn encode(&self) -> Result<Vec<u8>, EngineError> {
        let mut out = Vec::new();
        match self {
            Self::Created(value) | Self::Described(value) => put_descriptor(&mut out, value)?,
            Self::Listed(values) => {
                let count = u16::try_from(values.len()).map_err(|_| {
                    EngineError::resource_exhaustion("cache descriptor count exceeds u16")
                })?;
                validate_maximum(count.max(1))?;
                out.extend_from_slice(&count.to_le_bytes());
                for value in values {
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
            Self::Released => {}
            Self::Invalidated(count) => out.extend_from_slice(&count.to_le_bytes()),
            Self::Metrics(metrics) => {
                for value in metrics.0 {
                    out.extend_from_slice(&value.to_le_bytes());
                }
            }
        }
        Ok(out)
    }

    pub(crate) fn decode(kind: ServiceMessageKind, payload: &[u8]) -> Result<Self, EngineError> {
        let mut cursor = Cursor::new(payload);
        let response = match kind {
            ServiceMessageKind::CreateReusableCache => Self::Created(cursor.descriptor()?),
            ServiceMessageKind::DescribeCache => Self::Described(cursor.descriptor()?),
            ServiceMessageKind::ListCaches => {
                let count = cursor.u16()?;
                if count > MAX_SERVICE_CACHE_RESULTS {
                    return Err(EngineError::protocol_violation("cache list exceeds bound"));
                }
                let mut values = Vec::new();
                values.try_reserve_exact(usize::from(count)).map_err(|_| {
                    EngineError::resource_exhaustion("cache list allocation failed")
                })?;
                for _ in 0..count {
                    values.push(cursor.descriptor()?);
                }
                Self::Listed(values)
            }
            ServiceMessageKind::AttachCache => {
                let cache_id = ServiceCacheId::new(cursor.hash()?);
                let token_count = cursor.u32()?;
                if token_count == 0 {
                    return Err(EngineError::protocol_violation("cache token count is zero"));
                }
                Self::Attached { cache_id, token_count }
            }
            ServiceMessageKind::ReleaseCache => Self::Released,
            ServiceMessageKind::InvalidateCache => Self::Invalidated(cursor.u32()?),
            ServiceMessageKind::CacheMetrics => {
                let mut values = [0_u64; 17];
                for value in &mut values {
                    *value = cursor.u64()?;
                }
                Self::Metrics(ServiceCacheMetrics(values))
            }
            _ => return Err(EngineError::protocol_violation("message has no cache response body")),
        };
        cursor.finish()?;
        Ok(response)
    }
}

fn put_descriptor(out: &mut Vec<u8>, value: &ServiceCacheDescriptor) -> Result<(), EngineError> {
    put_hash(out, value.cache_id.hash());
    put_model(out, &value.model_id)?;
    put_scope(out, &value.scope)?;
    out.extend_from_slice(&value.token_count.to_le_bytes());
    out.extend_from_slice(&value.byte_size.to_le_bytes());
    out.extend_from_slice(&value.created_at_unix_seconds.to_le_bytes());
    out.extend_from_slice(&value.last_access_unix_seconds.to_le_bytes());
    out.extend_from_slice(&value.hit_count.to_le_bytes());
    out.push(u8::from(value.pinned));
    Ok(())
}

fn put_model(out: &mut Vec<u8>, model: &ModelId) -> Result<(), EngineError> {
    put_bytes_u16(out, model.as_str().as_bytes(), "model ID")
}

fn put_scope(out: &mut Vec<u8>, scope: &ServiceCacheScope) -> Result<(), EngineError> {
    out.push(scope.namespace as u8);
    put_bytes_u16(out, scope.owner.as_bytes(), "cache scope owner")
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
        let namespace = ServiceCacheNamespace::decode(self.u8()?)?;
        let owner = self.string_u16("cache owner")?;
        ServiceCacheScope::new(namespace, owner)
            .map_err(|_| EngineError::protocol_violation("cache scope is invalid"))
    }
    fn string_u16(&mut self, label: &'static str) -> Result<String, EngineError> {
        let length = usize::from(self.u16()?);
        self.string(length, label)
    }
    fn string_u32(&mut self, label: &'static str) -> Result<String, EngineError> {
        let length = usize::try_from(self.u32()?)
            .map_err(|_| EngineError::protocol_violation("cache string length exceeds usize"))?;
        self.string(length, label)
    }
    fn string(&mut self, length: usize, label: &'static str) -> Result<String, EngineError> {
        let value = std::str::from_utf8(self.take(length)?)
            .map_err(|_| EngineError::protocol_violation(label))?;
        Ok(value.to_owned())
    }
    fn descriptor(&mut self) -> Result<ServiceCacheDescriptor, EngineError> {
        let cache_id = ServiceCacheId::new(self.hash()?);
        let model_id = self.model()?;
        let scope = self.scope()?;
        let token_count = self.u32()?;
        let byte_size = self.u64()?;
        let created = self.u64()?;
        let accessed = self.u64()?;
        let hits = self.u64()?;
        let pinned = decode_bool(self.u8()?)?;
        ServiceCacheDescriptor::new(
            cache_id,
            model_id,
            scope,
            token_count,
            byte_size,
            created,
            accessed,
            hits,
            pinned,
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

    fn hash(byte: u8) -> ModelPackageHash {
        ModelPackageHash::from_sha384_bytes([byte; 48])
    }
    fn scope() -> ServiceCacheScope {
        ServiceCacheScope::new(ServiceCacheNamespace::Project, "project:a").unwrap()
    }

    #[test]
    fn every_cache_request_round_trips_and_trailing_data_fails() {
        assert!(ServiceCacheScope::new(ServiceCacheNamespace::Project, "../project/a").is_err());
        assert!(ServiceCacheRetention::new(Some(0), 1, false).is_err());
        assert!(ServiceCacheRetention::new(Some(MAX_REUSABLE_CACHE_TTL_SECONDS), 1, false).is_ok());
        assert!(
            ServiceCacheRetention::new(Some(MAX_REUSABLE_CACHE_TTL_SECONDS + 1), 1, false).is_err()
        );
        let model_id = ModelId::new("tiny").unwrap();
        let id = ServiceCacheId::new(hash(1));
        let requests = vec![
            ServiceCacheRequest::Create {
                model_id: model_id.clone(),
                scope: scope(),
                text: "prefix".into(),
                source: Some(ServiceCacheSource::new(hash(2), 3).unwrap()),
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
                scope: scope(),
                expected_token_count: 2,
            },
            ServiceCacheRequest::Release { target_request_id: RequestId::new(9).unwrap() },
            ServiceCacheRequest::Invalidate {
                model_id: model_id.clone(),
                scope: scope(),
                selector: ServiceCacheInvalidation::Source(
                    ServiceCacheSource::new(hash(3), 4).unwrap(),
                ),
            },
            ServiceCacheRequest::Metrics { model_id },
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
            ModelId::new("tiny").unwrap(),
            scope(),
            3,
            99,
            1,
            2,
            3,
            false,
        )
        .unwrap();
        let responses = vec![
            ServiceCacheResponse::Created(descriptor.clone()),
            ServiceCacheResponse::Described(descriptor.clone()),
            ServiceCacheResponse::Listed(vec![descriptor]),
            ServiceCacheResponse::Attached {
                cache_id: ServiceCacheId::new(hash(5)),
                token_count: 3,
            },
            ServiceCacheResponse::Released,
            ServiceCacheResponse::Invalidated(2),
            ServiceCacheResponse::Metrics(ServiceCacheMetrics([7; 17])),
        ];
        for response in responses {
            let encoded = response.encode().unwrap();
            assert_eq!(ServiceCacheResponse::decode(response.kind(), &encoded).unwrap(), response);
        }
    }
}
