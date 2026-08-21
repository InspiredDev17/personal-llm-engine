//! Narrow, dependency-policy-compatible projection of Sovereign Knowledge 1.1.
//!
//! The canonical JSON schemas and language bindings remain owned by
//! Sovereign-AI-Contracts. This crate carries only the closed execution facts
//! needed behind the engine's authenticated binary transport. It deliberately
//! excludes retrieval, ranking, qualification, memory, tools, and policy.

use personalai_engine_crypto::hash_sha256;
use personalai_engine_types::{
    ArchitectureId, BackendId, DType, EngineError, ModelId, ModelPackageHash, Sha256Digest,
};

/// Exact Contracts source revision audited by this adapter.
pub const SOVEREIGN_CONTRACT_COMMIT: &str = "2eaa6c23e6334df9b1309dcb28a0a8d15ebb5356";
/// Canonical Knowledge cache lifecycle contract version.
pub const KNOWLEDGE_CACHE_LIFECYCLE_VERSION: &str = "1.1.0";
/// Canonical Knowledge 1.1 schema digest from the component support profile.
pub const KNOWLEDGE_CACHE_SCHEMA_SHA256: &str =
    "4976b42a04c35efbbf9f00be71d2e26343399d1c8528811eecf6fa4aa07451f9";
/// Canonical Knowledge exact-prefix acquisition contract version.
pub const KNOWLEDGE_STATIC_PREFIX_CONTRACT_VERSION: &str = "1.2.0";
/// Canonical Knowledge 1.2 exact-prefix schema digest.
pub const KNOWLEDGE_STATIC_PREFIX_SCHEMA_SHA256: &str =
    "0f711d87f54c22992887e21a3ef4b2a4c7814f5ff3c47bb95f8d29fbd2ed68b9";
/// Pinned canonical accepted lifecycle fixture digests in operation/request-result order:
/// attach, create, describe, invalidate, list, metrics, and release.
pub const KNOWLEDGE_ACCEPTED_FIXTURE_SHA256: [&str; 14] = [
    "347f7d9d1853b6a395a3544f5652e9833e407c3e301f55b8bda39eeb5ef74f4e",
    "6560ef159196ed2be07c6ab81d32a652d1ef137de65c2640c1156d2681daa1c3",
    "a881c2da87ec5936806aa78568a39807bb6f045ea63acecb81e40c13d472b9e4",
    "f7ab03a38b048b2d28f22845d3738744ce3823226fcae5b95900f5ebaf710d33",
    "34ba69774c56116abd3ae6bd833773b874685fd1a228a1c37de792976eb1ca21",
    "4be00a14f248a93f05c5be209af0b835ce0b12f810ffd15d2b7ee97e9d1c4f79",
    "a5c394a4425141d252d5185e1ae0553b7ab9efbdca1c06ecec573348dbc92b24",
    "5ac4d1492e3aed1d16ea6667b89aefd821724df34dc852926db6dc70cdc2b38f",
    "b658a018bf760b9e8d5a907e6e03d1a2f34c1a57dfa12e9149411965ec93309b",
    "d927c25b0805561e66540747eb7bf101304662416fa0c26d6cbdee43d990049b",
    "4c4722e0030771fe6249a4cad32e39849c5d174742057e6e5f548f060b9518c9",
    "418f5fe5211e0c9b3d5c7b8afb40d799e2282c85dfd2ad1d41214e103d2d5f48",
    "5242b4394e5515ea70e2ab5e789eeaeee2ab8f4545965c77a9b6f7549a4f82c3",
    "a3c7793b48a49587eabdbf12288e21d793537c84c0851020e426ed2fd959ce1c",
];
/// Pinned canonical rejected fixture digests: invalid scope shape, invalid status
/// payload, counter overflow, truncation, unknown namespace, and unknown version.
pub const KNOWLEDGE_REJECTED_FIXTURE_SHA256: [&str; 6] = [
    "84770a0057b35c378f8fe3f8a326d2678001713696714076b24c8dbbbc4bf404",
    "5ff5eb8440514fdcf1b648d4453a6208302274a349651f8bb7585eb752d4bbc1",
    "8c19570cec8c5b0450b98267e74ab4fbb07f4684dd96ff3504d794dd8085c3ea",
    "7bcea3aeaf36a8c95deccaf5fbbc8e2f69ba016fd52762c3c664e44dda848ea1",
    "97e074116930e11ab448e120fe4602c576aef71f897582beb034113fd4d123a0",
    "3dda876c5aa389f989f68926a7e461faa125bfa72f956148527e7f648d944de0",
];
/// Pinned canonical accepted Knowledge 1.2 request/result fixture digests.
pub const KNOWLEDGE_STATIC_PREFIX_ACCEPTED_FIXTURE_SHA256: [&str; 5] = [
    "efa60e9dd9073d0797d4a4e8814eec7249bb914bd1908666ef6319f507323432",
    "8f109b58b658c046161fa947aa8d360bbc0158f942ce9061958a76b08de7c82a",
    "daabb9dab31e0d302a34e5460d39b01547a2d714f216163575996517a2a40e3a",
    "6f3b67b11f348afd8be7fb0de57b993020efb7096efac308fc13afdd5faab3c4",
    "16b0c404ad6d64e39edb465feaf3f7d44df083fb7aa7fa80ae186b800980841e",
];
/// Pinned canonical rejected Knowledge 1.2 exact-prefix fixture digests.
pub const KNOWLEDGE_STATIC_PREFIX_REJECTED_FIXTURE_SHA256: [&str; 18] = [
    "15dcaa018ecb951ad0fdd0e94cab30e93418ebc42410a2a970c8687f77244b28",
    "37f1fffb3adb84fae5847850c917f93d0b7edb05a7afea4c29a83eb65282f3fa",
    "a41be899eaa84137d5b831291407c552e0867d0f7f298060e19c8563649762b8",
    "2309e42095fd119a08f60880c34897fc5c202484f5ab4be9b5be14602139b128",
    "2767ccc74968e0c45edc31f6dd3e6ee51e0af5c3f5c08210ab8f9fff04848ea6",
    "161997c312eaae5ba8d4d3992f8220ca3b408b7ecb5b494d175e4790ca996548",
    "c85371481fbbcad13a2dd965f14df2d829c218961ce17238c89fb3ecad4cb23f",
    "fa1fec3ec75b2aef6f09128c19c84a88ebed1bf78b14e3407d2f36de3c7884dd",
    "41e320fbe6141218fdf98d513bd07cbde09d4a46ec05f630de0ceac251ed6b3c",
    "6e45277b9167a00c808266497b6c70756cc1a71edc6549364508d223c488dc03",
    "ac29add4c904709fa48bbe03b5eaa4371182669c2e584293e0ba8ce830a8eae7",
    "e7872e705896e1915d40adbfe9e2d17aee437ad0d088bee49476b7bb2ab05379",
    "45122d519463d46d52f906d3cddb2fb3a97f6adbe65fae0724d5055a0d812e6b",
    "fb1c880f6399b6b5519f5a14467ec1f29f2786439090df74cb35c1221d4b4c2c",
    "e81feb670fde986abf3de6bc09c8ff95c8fd964f6176acac557264c877e8ed24",
    "874d4994a003c1ef59485c8eac67c63fb468fd95e288bb8cbe78bb53c34f7eee",
    "54584cc2eb619fd8417daabb73ce6d8c86565abf1e32fdd1c474c335fe5ab7bd",
    "5d83a89000e79b0eba2854e982334f6699ea158487e54ab7e9e7872fcffbfad2",
];
/// First engine cache representation projected into the canonical contract.
pub const CACHE_FORMAT_VERSION: &str = "1.0.0";
/// Exact prefix-material domain emitted by Memory Vault.
pub const STATIC_PREFIX_DOMAIN: &[u8] = b"SOVEREIGN-STATIC-KNOWLEDGE-PREFIX-V1\n";
/// Canonical Knowledge 1.2 decoded prefix-material ceiling.
pub const MAX_STATIC_PREFIX_BYTES: usize = 8 * 1024 * 1024;
/// Maximum exact ordered source members accepted in one prefix.
pub const MAX_STATIC_PREFIX_MEMBERS: usize = 256;
/// Maximum canonical decimal digits in one member byte length.
pub const MAX_STATIC_PREFIX_LENGTH_DIGITS: usize = 10;
/// Maximum bytes in one canonical identity.
pub const MAX_KNOWLEDGE_IDENTITY_BYTES: usize = 128;

/// Bounded path-free canonical identity.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct KnowledgeIdentity(String);

impl KnowledgeIdentity {
    /// Validates one opaque canonical identity without interpreting authority.
    pub fn new(value: impl Into<String>) -> Result<Self, EngineError> {
        let value = value.into();
        if value.is_empty()
            || value.len() > MAX_KNOWLEDGE_IDENTITY_BYTES
            || !value.is_ascii()
            || !value.as_bytes()[0].is_ascii_lowercase()
            || !value.bytes().all(|byte| {
                byte.is_ascii_lowercase()
                    || byte.is_ascii_digit()
                    || matches!(byte, b'.' | b'_' | b'-' | b':')
            })
            || value.contains("..")
            || value.contains("//")
            || value.ends_with(['.', '_', '-', ':'])
        {
            return Err(EngineError::invalid_input(
                "knowledge identity is outside its bounded path-free grammar",
            ));
        }
        Ok(Self(value))
    }

    /// Returns the opaque identity text.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// Closed authorization namespace from Knowledge 1.1.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum CacheNamespace {
    /// One owner.
    Owner = 1,
    /// One owner/project pair.
    Project = 2,
    /// One owner/project/task tuple.
    Task = 3,
    /// One owner/project/agent tuple.
    Agent = 4,
}

impl CacheNamespace {
    /// Decodes one closed binary adapter value.
    pub fn from_u8(value: u8) -> Result<Self, EngineError> {
        match value {
            1 => Ok(Self::Owner),
            2 => Ok(Self::Project),
            3 => Ok(Self::Task),
            4 => Ok(Self::Agent),
            _ => Err(EngineError::protocol_violation("cache namespace is unsupported")),
        }
    }
}

/// Exact authorization scope asserted by the authenticated Go control plane.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CacheAuthorizationScope {
    namespace: CacheNamespace,
    owner_id: KnowledgeIdentity,
    project_id: Option<KnowledgeIdentity>,
    task_id: Option<KnowledgeIdentity>,
    agent_id: Option<KnowledgeIdentity>,
}

impl CacheAuthorizationScope {
    /// Constructs a scope and enforces the canonical namespace/identity shape.
    pub fn new(
        namespace: CacheNamespace,
        owner_id: KnowledgeIdentity,
        project_id: Option<KnowledgeIdentity>,
        task_id: Option<KnowledgeIdentity>,
        agent_id: Option<KnowledgeIdentity>,
    ) -> Result<Self, EngineError> {
        let valid = match namespace {
            CacheNamespace::Owner => {
                project_id.is_none() && task_id.is_none() && agent_id.is_none()
            }
            CacheNamespace::Project => {
                project_id.is_some() && task_id.is_none() && agent_id.is_none()
            }
            CacheNamespace::Task => project_id.is_some() && task_id.is_some() && agent_id.is_none(),
            CacheNamespace::Agent => {
                project_id.is_some() && task_id.is_none() && agent_id.is_some()
            }
        };
        if !valid {
            return Err(EngineError::invalid_input(
                "cache authorization namespace and identities disagree",
            ));
        }
        Ok(Self { namespace, owner_id, project_id, task_id, agent_id })
    }

    /// Returns the closed namespace.
    pub const fn namespace(&self) -> CacheNamespace {
        self.namespace
    }

    /// Returns the owner identity.
    pub const fn owner_id(&self) -> &KnowledgeIdentity {
        &self.owner_id
    }

    /// Returns the project identity when required by the namespace.
    pub const fn project_id(&self) -> Option<&KnowledgeIdentity> {
        self.project_id.as_ref()
    }

    /// Returns the task identity when required by the namespace.
    pub const fn task_id(&self) -> Option<&KnowledgeIdentity> {
        self.task_id.as_ref()
    }

    /// Returns the agent identity when required by the namespace.
    pub const fn agent_id(&self) -> Option<&KnowledgeIdentity> {
        self.agent_id.as_ref()
    }
}

/// Closed Knowledge sensitivity classification.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum CacheSensitivity {
    /// Public data.
    Public = 1,
    /// Internal data.
    Internal = 2,
    /// Personal data.
    Personal = 3,
    /// Sensitive data.
    Sensitive = 4,
    /// Restricted data.
    Restricted = 5,
}

impl CacheSensitivity {
    /// Decodes one closed binary adapter value.
    pub fn from_u8(value: u8) -> Result<Self, EngineError> {
        match value {
            1 => Ok(Self::Public),
            2 => Ok(Self::Internal),
            3 => Ok(Self::Personal),
            4 => Ok(Self::Sensitive),
            5 => Ok(Self::Restricted),
            _ => Err(EngineError::protocol_violation("cache sensitivity is unsupported")),
        }
    }
}

/// Artifact digest admitted by the canonical runtime identity.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ArtifactDigest {
    /// Canonical SHA-256 artifact identity.
    Sha256(Sha256Digest),
    /// Native AIPack SHA-384 artifact identity.
    Sha384(ModelPackageHash),
}

/// Exact execution identity bound to reusable state.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct CacheRuntimeIdentity {
    logical_model_id: KnowledgeIdentity,
    provider_model_id: ModelId,
    artifact_id: KnowledgeIdentity,
    artifact_digest: ArtifactDigest,
    backend_id: BackendId,
    backend_version: KnowledgeIdentity,
    backend_instance_id: KnowledgeIdentity,
    engine_instance_id: KnowledgeIdentity,
}

impl CacheRuntimeIdentity {
    /// Constructs an exact runtime identity and rejects logical/provider substitution.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        logical_model_id: KnowledgeIdentity,
        provider_model_id: ModelId,
        artifact_id: KnowledgeIdentity,
        artifact_digest: ArtifactDigest,
        backend_id: BackendId,
        backend_version: KnowledgeIdentity,
        backend_instance_id: KnowledgeIdentity,
        engine_instance_id: KnowledgeIdentity,
    ) -> Result<Self, EngineError> {
        if logical_model_id.as_str() == provider_model_id.as_str() {
            return Err(EngineError::invalid_input(
                "logical and provider model identities must remain distinct",
            ));
        }
        Ok(Self {
            logical_model_id,
            provider_model_id,
            artifact_id,
            artifact_digest,
            backend_id,
            backend_version,
            backend_instance_id,
            engine_instance_id,
        })
    }

    /// Returns the logical control-plane model identity.
    pub const fn logical_model_id(&self) -> &KnowledgeIdentity {
        &self.logical_model_id
    }
    /// Returns the exact engine/provider model identity.
    pub const fn provider_model_id(&self) -> &ModelId {
        &self.provider_model_id
    }
    /// Returns the exact artifact identity.
    pub const fn artifact_id(&self) -> &KnowledgeIdentity {
        &self.artifact_id
    }
    /// Returns the exact artifact digest.
    pub const fn artifact_digest(&self) -> ArtifactDigest {
        self.artifact_digest
    }
    /// Returns the trusted backend identity.
    pub const fn backend_id(&self) -> &BackendId {
        &self.backend_id
    }
    /// Returns the backend implementation version.
    pub const fn backend_version(&self) -> &KnowledgeIdentity {
        &self.backend_version
    }
    /// Returns the backend instance identity.
    pub const fn backend_instance_id(&self) -> &KnowledgeIdentity {
        &self.backend_instance_id
    }
    /// Returns the producing engine instance identity.
    pub const fn engine_instance_id(&self) -> &KnowledgeIdentity {
        &self.engine_instance_id
    }
}

/// Exact immutable Memory Vault bundle identity.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct StaticBundleBinding {
    bundle_id: KnowledgeIdentity,
    bundle_revision: u64,
    bundle_digest: Sha256Digest,
    ordered_prefix_digest: Sha256Digest,
}

/// Optional exact source/revision identity used by canonical invalidation.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct SourceRevisionBinding {
    source_id: KnowledgeIdentity,
    revision_id: KnowledgeIdentity,
}

impl SourceRevisionBinding {
    /// Constructs one exact source revision identity pair.
    pub const fn new(source_id: KnowledgeIdentity, revision_id: KnowledgeIdentity) -> Self {
        Self { source_id, revision_id }
    }
    /// Returns the source identity.
    pub const fn source_id(&self) -> &KnowledgeIdentity {
        &self.source_id
    }
    /// Returns the revision identity.
    pub const fn revision_id(&self) -> &KnowledgeIdentity {
        &self.revision_id
    }
}

impl StaticBundleBinding {
    /// Constructs one exact nonzero bundle revision.
    pub fn new(
        bundle_id: KnowledgeIdentity,
        bundle_revision: u64,
        bundle_digest: Sha256Digest,
        ordered_prefix_digest: Sha256Digest,
    ) -> Result<Self, EngineError> {
        if bundle_revision == 0 {
            return Err(EngineError::invalid_input("bundle revision must be nonzero"));
        }
        Ok(Self { bundle_id, bundle_revision, bundle_digest, ordered_prefix_digest })
    }
    /// Returns the bundle identity.
    pub const fn bundle_id(&self) -> &KnowledgeIdentity {
        &self.bundle_id
    }
    /// Returns the exact bundle revision.
    pub const fn bundle_revision(&self) -> u64 {
        self.bundle_revision
    }
    /// Returns the canonical bundle digest.
    pub const fn bundle_digest(&self) -> Sha256Digest {
        self.bundle_digest
    }
    /// Returns the digest of exact prefix transport bytes.
    pub const fn ordered_prefix_digest(&self) -> Sha256Digest {
        self.ordered_prefix_digest
    }
}

/// Complete canonical compatibility and authorization binding.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct CacheBinding {
    runtime: CacheRuntimeIdentity,
    tokenizer_id: KnowledgeIdentity,
    tokenizer_digest: Sha256Digest,
    architecture_id: ArchitectureId,
    rope_semantics_id: KnowledgeIdentity,
    kv_dtype: DType,
    bundle: StaticBundleBinding,
    source_revision: Option<SourceRevisionBinding>,
    scope: CacheAuthorizationScope,
    sensitivity: CacheSensitivity,
    token_count: u32,
}

impl CacheBinding {
    /// Constructs a local-native cache binding with the only supported format.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        runtime: CacheRuntimeIdentity,
        tokenizer_id: KnowledgeIdentity,
        tokenizer_digest: Sha256Digest,
        architecture_id: ArchitectureId,
        rope_semantics_id: KnowledgeIdentity,
        kv_dtype: DType,
        bundle: StaticBundleBinding,
        scope: CacheAuthorizationScope,
        sensitivity: CacheSensitivity,
        token_count: u32,
    ) -> Result<Self, EngineError> {
        if token_count == 0 || token_count > 2_000_000 {
            return Err(EngineError::invalid_input("cache token count is outside contract bounds"));
        }
        if !matches!(kv_dtype, DType::F32 | DType::F16 | DType::BFloat16) {
            return Err(EngineError::not_implemented("canonical cache K/V dtype is unsupported"));
        }
        if scope.owner_id().as_str().is_empty() {
            return Err(EngineError::internal_invariant("cache scope owner disappeared"));
        }
        Ok(Self {
            runtime,
            tokenizer_id,
            tokenizer_digest,
            architecture_id,
            rope_semantics_id,
            kv_dtype,
            bundle,
            source_revision: None,
            scope,
            sensitivity,
            token_count,
        })
    }
    /// Returns the exact runtime identity.
    pub const fn runtime(&self) -> &CacheRuntimeIdentity {
        &self.runtime
    }
    /// Returns the tokenizer identity.
    pub const fn tokenizer_id(&self) -> &KnowledgeIdentity {
        &self.tokenizer_id
    }
    /// Returns the tokenizer content digest.
    pub const fn tokenizer_digest(&self) -> Sha256Digest {
        self.tokenizer_digest
    }
    /// Returns the reviewed architecture identity.
    pub const fn architecture_id(&self) -> &ArchitectureId {
        &self.architecture_id
    }
    /// Returns exact RoPE/position semantics identity.
    pub const fn rope_semantics_id(&self) -> &KnowledgeIdentity {
        &self.rope_semantics_id
    }
    /// Returns the cache storage dtype.
    pub const fn kv_dtype(&self) -> DType {
        self.kv_dtype
    }
    /// Returns exact bundle identity.
    pub const fn bundle(&self) -> &StaticBundleBinding {
        &self.bundle
    }
    /// Adds one exact source/revision invalidation identity.
    #[must_use]
    pub fn with_source_revision(mut self, source_revision: SourceRevisionBinding) -> Self {
        self.source_revision = Some(source_revision);
        self
    }
    /// Returns optional exact source/revision invalidation identity.
    pub const fn source_revision(&self) -> Option<&SourceRevisionBinding> {
        self.source_revision.as_ref()
    }
    /// Returns the authorization scope.
    pub const fn scope(&self) -> &CacheAuthorizationScope {
        &self.scope
    }
    /// Returns the sensitivity classification.
    pub const fn sensitivity(&self) -> CacheSensitivity {
        self.sensitivity
    }
    /// Returns the exact cached prefix token count.
    pub const fn token_count(&self) -> u32 {
        self.token_count
    }

    /// Encodes the narrow authenticated-binary adapter representation.
    pub fn encode_binary(&self) -> Result<Vec<u8>, EngineError> {
        let mut out = Vec::new();
        out.try_reserve_exact(768)
            .map_err(|_| EngineError::resource_exhaustion("cache binding allocation failed"))?;
        out.extend_from_slice(b"SKCAGB11");
        put_text(&mut out, CACHE_FORMAT_VERSION)?;
        let runtime = self.runtime();
        put_text(&mut out, runtime.logical_model_id().as_str())?;
        put_text(&mut out, runtime.provider_model_id().as_str())?;
        put_text(&mut out, runtime.artifact_id().as_str())?;
        match runtime.artifact_digest() {
            ArtifactDigest::Sha256(digest) => {
                out.push(1);
                out.extend_from_slice(digest.as_bytes());
            }
            ArtifactDigest::Sha384(digest) => {
                out.push(2);
                out.extend_from_slice(digest.as_bytes());
            }
        }
        put_text(&mut out, runtime.backend_id().as_str())?;
        put_text(&mut out, runtime.backend_version().as_str())?;
        put_text(&mut out, runtime.backend_instance_id().as_str())?;
        put_text(&mut out, runtime.engine_instance_id().as_str())?;
        put_text(&mut out, self.tokenizer_id().as_str())?;
        out.extend_from_slice(self.tokenizer_digest().as_bytes());
        put_text(&mut out, self.architecture_id().as_str())?;
        put_text(&mut out, self.rope_semantics_id().as_str())?;
        out.push(dtype_code(self.kv_dtype())?);
        put_text(&mut out, self.bundle().bundle_id().as_str())?;
        out.extend_from_slice(&self.bundle().bundle_revision().to_le_bytes());
        out.extend_from_slice(self.bundle().bundle_digest().as_bytes());
        out.extend_from_slice(self.bundle().ordered_prefix_digest().as_bytes());
        match self.source_revision() {
            Some(source) => {
                out.push(1);
                put_text(&mut out, source.source_id().as_str())?;
                put_text(&mut out, source.revision_id().as_str())?;
            }
            None => out.push(0),
        }
        put_scope(&mut out, self.scope())?;
        out.push(self.sensitivity() as u8);
        out.extend_from_slice(&self.token_count().to_le_bytes());
        Ok(out)
    }

    /// Decodes the closed adapter representation and rejects unknown/trailing data.
    pub fn decode_binary(bytes: &[u8]) -> Result<Self, EngineError> {
        let mut cursor = BindingCursor::new(bytes);
        if cursor.take(8)? != b"SKCAGB11" {
            return Err(EngineError::protocol_violation("cache binding magic is invalid"));
        }
        if cursor.text()? != CACHE_FORMAT_VERSION {
            return Err(EngineError::protocol_violation("cache format version is unsupported"));
        }
        let logical_model_id = cursor.identity()?;
        let provider_model_id = ModelId::new(cursor.text()?)
            .map_err(|_| EngineError::protocol_violation("provider model ID is invalid"))?;
        let artifact_id = cursor.identity()?;
        let artifact_digest = match cursor.u8()? {
            1 => ArtifactDigest::Sha256(Sha256Digest::from_bytes(cursor.array_32()?)),
            2 => ArtifactDigest::Sha384(ModelPackageHash::from_sha384_bytes(cursor.array_48()?)),
            _ => {
                return Err(EngineError::protocol_violation("artifact digest type is unsupported"));
            }
        };
        let backend_id = BackendId::new(cursor.text()?)
            .map_err(|_| EngineError::protocol_violation("backend ID is invalid"))?;
        let runtime = CacheRuntimeIdentity::new(
            logical_model_id,
            provider_model_id,
            artifact_id,
            artifact_digest,
            backend_id,
            cursor.identity()?,
            cursor.identity()?,
            cursor.identity()?,
        )
        .map_err(|_| EngineError::protocol_violation("cache runtime identity is invalid"))?;
        let tokenizer_id = cursor.identity()?;
        let tokenizer_digest = Sha256Digest::from_bytes(cursor.array_32()?);
        let architecture_id = ArchitectureId::new(cursor.text()?)
            .map_err(|_| EngineError::protocol_violation("architecture ID is invalid"))?;
        let rope_semantics_id = cursor.identity()?;
        let kv_dtype = decode_dtype(cursor.u8()?)?;
        let bundle = StaticBundleBinding::new(
            cursor.identity()?,
            cursor.u64()?,
            Sha256Digest::from_bytes(cursor.array_32()?),
            Sha256Digest::from_bytes(cursor.array_32()?),
        )
        .map_err(|_| EngineError::protocol_violation("static bundle binding is invalid"))?;
        let source_revision = match cursor.u8()? {
            0 => None,
            1 => Some(SourceRevisionBinding::new(cursor.identity()?, cursor.identity()?)),
            _ => {
                return Err(EngineError::protocol_violation(
                    "source revision binding flag is invalid",
                ));
            }
        };
        let scope = cursor.scope()?;
        let sensitivity = CacheSensitivity::from_u8(cursor.u8()?)?;
        let token_count = cursor.u32()?;
        cursor.finish()?;
        let binding = Self::new(
            runtime,
            tokenizer_id,
            tokenizer_digest,
            architecture_id,
            rope_semantics_id,
            kv_dtype,
            bundle,
            scope,
            sensitivity,
            token_count,
        )
        .map_err(|_| EngineError::protocol_violation("cache binding is invalid"))?;
        Ok(match source_revision {
            Some(source_revision) => binding.with_source_revision(source_revision),
            None => binding,
        })
    }
}

fn put_text(out: &mut Vec<u8>, value: &str) -> Result<(), EngineError> {
    let length = u16::try_from(value.len())
        .map_err(|_| EngineError::resource_exhaustion("cache binding text exceeds u16"))?;
    out.extend_from_slice(&length.to_le_bytes());
    out.extend_from_slice(value.as_bytes());
    Ok(())
}

fn put_optional_identity(
    out: &mut Vec<u8>,
    value: Option<&KnowledgeIdentity>,
) -> Result<(), EngineError> {
    match value {
        Some(value) => {
            out.push(1);
            put_text(out, value.as_str())?;
        }
        None => out.push(0),
    }
    Ok(())
}

fn put_scope(out: &mut Vec<u8>, scope: &CacheAuthorizationScope) -> Result<(), EngineError> {
    out.push(scope.namespace() as u8);
    put_text(out, scope.owner_id().as_str())?;
    put_optional_identity(out, scope.project_id())?;
    put_optional_identity(out, scope.task_id())?;
    put_optional_identity(out, scope.agent_id())?;
    Ok(())
}

fn dtype_code(dtype: DType) -> Result<u8, EngineError> {
    match dtype {
        DType::F32 => Ok(1),
        DType::F16 => Ok(2),
        DType::BFloat16 => Ok(3),
        _ => Err(EngineError::not_implemented("cache binding dtype is unsupported")),
    }
}

fn decode_dtype(value: u8) -> Result<DType, EngineError> {
    match value {
        1 => Ok(DType::F32),
        2 => Ok(DType::F16),
        3 => Ok(DType::BFloat16),
        _ => Err(EngineError::protocol_violation("cache binding dtype is unsupported")),
    }
}

struct BindingCursor<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> BindingCursor<'a> {
    const fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    fn take(&mut self, length: usize) -> Result<&'a [u8], EngineError> {
        let end = self
            .offset
            .checked_add(length)
            .filter(|end| *end <= self.bytes.len())
            .ok_or_else(|| EngineError::protocol_violation("cache binding is truncated"))?;
        let value = &self.bytes[self.offset..end];
        self.offset = end;
        Ok(value)
    }

    fn u8(&mut self) -> Result<u8, EngineError> {
        Ok(self.take(1)?[0])
    }

    fn u16(&mut self) -> Result<u16, EngineError> {
        let bytes: [u8; 2] = self
            .take(2)?
            .try_into()
            .map_err(|_| EngineError::internal_invariant("binding u16 slice length changed"))?;
        Ok(u16::from_le_bytes(bytes))
    }

    fn u32(&mut self) -> Result<u32, EngineError> {
        let bytes: [u8; 4] = self
            .take(4)?
            .try_into()
            .map_err(|_| EngineError::internal_invariant("binding u32 slice length changed"))?;
        Ok(u32::from_le_bytes(bytes))
    }

    fn u64(&mut self) -> Result<u64, EngineError> {
        let bytes: [u8; 8] = self
            .take(8)?
            .try_into()
            .map_err(|_| EngineError::internal_invariant("binding u64 slice length changed"))?;
        Ok(u64::from_le_bytes(bytes))
    }

    fn array_32(&mut self) -> Result<[u8; 32], EngineError> {
        self.take(32)?
            .try_into()
            .map_err(|_| EngineError::internal_invariant("binding digest slice length changed"))
    }

    fn array_48(&mut self) -> Result<[u8; 48], EngineError> {
        self.take(48)?
            .try_into()
            .map_err(|_| EngineError::internal_invariant("binding digest slice length changed"))
    }

    fn text(&mut self) -> Result<&'a str, EngineError> {
        let length = usize::from(self.u16()?);
        std::str::from_utf8(self.take(length)?)
            .map_err(|_| EngineError::protocol_violation("cache binding text is invalid UTF-8"))
    }

    fn identity(&mut self) -> Result<KnowledgeIdentity, EngineError> {
        KnowledgeIdentity::new(self.text()?)
            .map_err(|_| EngineError::protocol_violation("cache binding identity is invalid"))
    }

    fn optional_identity(&mut self) -> Result<Option<KnowledgeIdentity>, EngineError> {
        match self.u8()? {
            0 => Ok(None),
            1 => self.identity().map(Some),
            _ => Err(EngineError::protocol_violation("optional identity flag is invalid")),
        }
    }

    fn scope(&mut self) -> Result<CacheAuthorizationScope, EngineError> {
        CacheAuthorizationScope::new(
            CacheNamespace::from_u8(self.u8()?)?,
            self.identity()?,
            self.optional_identity()?,
            self.optional_identity()?,
            self.optional_identity()?,
        )
        .map_err(|_| EngineError::protocol_violation("cache authorization scope is invalid"))
    }

    fn finish(self) -> Result<(), EngineError> {
        if self.offset == self.bytes.len() {
            Ok(())
        } else {
            Err(EngineError::protocol_violation("cache binding has trailing bytes"))
        }
    }
}

/// A validated, exact Memory Vault prefix transport.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StaticKnowledgePrefix {
    bytes: Box<[u8]>,
    member_count: u16,
}

impl StaticKnowledgePrefix {
    /// Validates domain, deterministic decimal framing, digest, UTF-8, and NUL policy.
    pub fn parse(
        bytes: &[u8],
        expected_digest: Sha256Digest,
        maximum_bytes: usize,
    ) -> Result<Self, EngineError> {
        if maximum_bytes < STATIC_PREFIX_DOMAIN.len()
            || maximum_bytes > MAX_STATIC_PREFIX_BYTES
            || bytes.len() > maximum_bytes
            || !bytes.starts_with(STATIC_PREFIX_DOMAIN)
            || bytes.contains(&0)
            || std::str::from_utf8(bytes).is_err()
            || hash_sha256(bytes) != expected_digest
        {
            return Err(EngineError::invalid_input("static knowledge prefix is invalid"));
        }
        let mut offset = STATIC_PREFIX_DOMAIN.len();
        let mut members = 0_usize;
        while offset < bytes.len() {
            let length_start = offset;
            while offset < bytes.len() && bytes[offset].is_ascii_digit() {
                offset = offset.checked_add(1).ok_or_else(|| {
                    EngineError::resource_exhaustion("prefix frame offset overflows")
                })?;
            }
            if offset == length_start
                || offset >= bytes.len()
                || bytes[offset] != b':'
                || offset - length_start > MAX_STATIC_PREFIX_LENGTH_DIGITS
                || (bytes[length_start] == b'0' && offset - length_start != 1)
            {
                return Err(EngineError::invalid_input("static prefix length frame is malformed"));
            }
            let length_text = std::str::from_utf8(&bytes[length_start..offset])
                .map_err(|_| EngineError::invalid_input("static prefix length is not ASCII"))?;
            let length = length_text
                .parse::<usize>()
                .map_err(|_| EngineError::invalid_input("static prefix length overflows"))?;
            if length == 0 {
                return Err(EngineError::invalid_input("static prefix member is empty"));
            }
            offset = offset
                .checked_add(1)
                .and_then(|value| value.checked_add(length))
                .filter(|value| *value <= bytes.len())
                .ok_or_else(|| EngineError::invalid_input("static prefix member is truncated"))?;
            members = members.checked_add(1).ok_or_else(|| {
                EngineError::resource_exhaustion("static prefix member count overflows")
            })?;
            if members > MAX_STATIC_PREFIX_MEMBERS {
                return Err(EngineError::resource_exhaustion(
                    "static prefix member count exceeds its bound",
                ));
            }
        }
        if members == 0 {
            return Err(EngineError::invalid_input("static prefix contains no members"));
        }
        let mut owned = Vec::new();
        owned
            .try_reserve_exact(bytes.len())
            .map_err(|_| EngineError::resource_exhaustion("static prefix allocation failed"))?;
        owned.extend_from_slice(bytes);
        Ok(Self {
            bytes: owned.into_boxed_slice(),
            member_count: u16::try_from(members).map_err(|_| {
                EngineError::internal_invariant("static prefix member count exceeds u16")
            })?,
        })
    }

    /// Returns the exact unmodified UTF-8 transport bytes.
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Returns exact unmodified UTF-8 text for tokenizer input.
    pub fn as_str(&self) -> &str {
        // Construction proves UTF-8 and the bytes are immutable.
        std::str::from_utf8(&self.bytes).expect("validated prefix bytes remain UTF-8")
    }

    /// Returns the exact framed member count.
    pub const fn member_count(&self) -> u16 {
        self.member_count
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn binding() -> CacheBinding {
        CacheBinding::new(
            CacheRuntimeIdentity::new(
                KnowledgeIdentity::new("mdl_logical_fixture").unwrap(),
                ModelId::new("provider-fixture").unwrap(),
                KnowledgeIdentity::new("artifact.fixture").unwrap(),
                ArtifactDigest::Sha384(ModelPackageHash::from_sha384_bytes([1; 48])),
                BackendId::new("cpu-reference").unwrap(),
                KnowledgeIdentity::new("version.0-1-0").unwrap(),
                KnowledgeIdentity::new("backend.fixture").unwrap(),
                KnowledgeIdentity::new("engine.fixture").unwrap(),
            )
            .unwrap(),
            KnowledgeIdentity::new("tokenizer.fixture").unwrap(),
            Sha256Digest::from_bytes([2; 32]),
            ArchitectureId::new("decoder-rms-rope-swiglu-gqa-v1").unwrap(),
            KnowledgeIdentity::new("rope.causal-zero-based-v1").unwrap(),
            DType::F32,
            StaticBundleBinding::new(
                KnowledgeIdentity::new("kbundle_fixture").unwrap(),
                7,
                Sha256Digest::from_bytes([3; 32]),
                Sha256Digest::from_bytes([4; 32]),
            )
            .unwrap(),
            CacheAuthorizationScope::new(
                CacheNamespace::Project,
                KnowledgeIdentity::new("usr_fixture_owner").unwrap(),
                Some(KnowledgeIdentity::new("prj_fixture_project").unwrap()),
                None,
                None,
            )
            .unwrap(),
            CacheSensitivity::Internal,
            17,
        )
        .unwrap()
        .with_source_revision(SourceRevisionBinding::new(
            KnowledgeIdentity::new("ksr_fixture_source").unwrap(),
            KnowledgeIdentity::new("krv_fixture_revision").unwrap(),
        ))
    }

    #[test]
    fn prefix_transport_is_exact_ordered_and_digest_bound() {
        let bytes = b"SOVEREIGN-STATIC-KNOWLEDGE-PREFIX-V1\n5:alpha4:beta";
        let parsed = StaticKnowledgePrefix::parse(bytes, hash_sha256(bytes), 1024).unwrap();
        assert_eq!(parsed.as_bytes(), bytes);
        assert_eq!(parsed.member_count(), 2);
        let reordered = b"SOVEREIGN-STATIC-KNOWLEDGE-PREFIX-V1\n4:beta5:alpha";
        assert!(StaticKnowledgePrefix::parse(reordered, hash_sha256(bytes), 1024).is_err());
    }

    #[test]
    fn canonical_knowledge_1_2_available_fixture_bytes_are_accepted_exactly() {
        let bytes = b"SOVEREIGN-STATIC-KNOWLEDGE-PREFIX-V1\n5:hello5:world";
        let digest = Sha256Digest::from_hex(
            "b6d8d9788107e9b6d0a32665b2045af4f4d60558af65f2e679fc092c97864484",
        )
        .unwrap();
        let parsed = StaticKnowledgePrefix::parse(bytes, digest, bytes.len()).unwrap();
        assert_eq!(parsed.as_bytes(), bytes);
        assert_eq!(parsed.member_count(), 2);
    }

    #[test]
    fn prefix_transport_rejects_malformed_truncated_nul_and_bounds() {
        for bytes in [
            &b"5:alpha"[..],
            &b"SOVEREIGN-STATIC-KNOWLEDGE-PREFIX-V1\n05:alpha"[..],
            &b"SOVEREIGN-STATIC-KNOWLEDGE-PREFIX-V1\n6:alpha"[..],
            &b"SOVEREIGN-STATIC-KNOWLEDGE-PREFIX-V1\n1:\0"[..],
            &b"SOVEREIGN-STATIC-KNOWLEDGE-PREFIX-V1\n0:"[..],
            &b"SOVEREIGN-STATIC-KNOWLEDGE-PREFIX-V1\n12345678901:x"[..],
            &b"SOVEREIGN-STATIC-KNOWLEDGE-PREFIX-V1\n1:aX"[..],
            &b"SOVEREIGN-STATIC-KNOWLEDGE-PREFIX-V1\n1:\xff"[..],
        ] {
            assert!(StaticKnowledgePrefix::parse(bytes, hash_sha256(bytes), 1024).is_err());
        }
        let bytes = b"SOVEREIGN-STATIC-KNOWLEDGE-PREFIX-V1\n5:alpha";
        assert!(StaticKnowledgePrefix::parse(bytes, hash_sha256(bytes), bytes.len() - 1).is_err());
        assert!(
            StaticKnowledgePrefix::parse(bytes, hash_sha256(bytes), MAX_STATIC_PREFIX_BYTES + 1)
                .is_err()
        );

        let mut too_many = STATIC_PREFIX_DOMAIN.to_vec();
        for _ in 0..=MAX_STATIC_PREFIX_MEMBERS {
            too_many.extend_from_slice(b"1:a");
        }
        assert!(
            StaticKnowledgePrefix::parse(&too_many, hash_sha256(&too_many), too_many.len())
                .is_err()
        );
    }

    #[test]
    fn scope_and_runtime_substitution_fail_closed() {
        let owner = KnowledgeIdentity::new("usr_owner").unwrap();
        assert!(
            CacheAuthorizationScope::new(CacheNamespace::Project, owner.clone(), None, None, None)
                .is_err()
        );
        let same = KnowledgeIdentity::new("mdl_same").unwrap();
        assert!(
            CacheRuntimeIdentity::new(
                same,
                ModelId::new("mdl_same").unwrap(),
                KnowledgeIdentity::new("art_fixture").unwrap(),
                ArtifactDigest::Sha256(Sha256Digest::from_bytes([1; 32])),
                BackendId::new("cpu-reference").unwrap(),
                KnowledgeIdentity::new("version.1").unwrap(),
                KnowledgeIdentity::new("backend.one").unwrap(),
                KnowledgeIdentity::new("engine.one").unwrap(),
            )
            .is_err()
        );
    }

    #[test]
    fn canonical_binding_binary_is_deterministic_closed_and_truncation_safe() {
        let binding = binding();
        let encoded = binding.encode_binary().unwrap();
        assert_eq!(binding.encode_binary().unwrap(), encoded);
        assert_eq!(CacheBinding::decode_binary(&encoded).unwrap(), binding);
        for length in 0..encoded.len() {
            assert!(CacheBinding::decode_binary(&encoded[..length]).is_err());
        }
        let mut trailing = encoded.clone();
        trailing.push(0);
        assert!(CacheBinding::decode_binary(&trailing).is_err());
        let mut unknown_version = encoded;
        unknown_version[10] = b'9';
        assert!(CacheBinding::decode_binary(&unknown_version).is_err());
    }

    #[test]
    fn contracts_revision_schema_and_fixture_evidence_are_exact() {
        assert_eq!(SOVEREIGN_CONTRACT_COMMIT.len(), 40);
        assert_eq!(KNOWLEDGE_CACHE_LIFECYCLE_VERSION, "1.1.0");
        assert_eq!(KNOWLEDGE_CACHE_SCHEMA_SHA256.len(), 64);
        assert_eq!(KNOWLEDGE_STATIC_PREFIX_CONTRACT_VERSION, "1.2.0");
        assert_eq!(KNOWLEDGE_STATIC_PREFIX_SCHEMA_SHA256.len(), 64);
        assert_eq!(KNOWLEDGE_ACCEPTED_FIXTURE_SHA256.len(), 14);
        assert!(KNOWLEDGE_ACCEPTED_FIXTURE_SHA256.iter().all(|digest| digest.len() == 64));
        assert_eq!(KNOWLEDGE_REJECTED_FIXTURE_SHA256.len(), 6);
        assert!(KNOWLEDGE_REJECTED_FIXTURE_SHA256.iter().all(|digest| digest.len() == 64));
        assert_eq!(KNOWLEDGE_STATIC_PREFIX_ACCEPTED_FIXTURE_SHA256.len(), 5);
        assert!(
            KNOWLEDGE_STATIC_PREFIX_ACCEPTED_FIXTURE_SHA256.iter().all(|digest| digest.len() == 64)
        );
        assert_eq!(KNOWLEDGE_STATIC_PREFIX_REJECTED_FIXTURE_SHA256.len(), 18);
        assert!(
            KNOWLEDGE_STATIC_PREFIX_REJECTED_FIXTURE_SHA256.iter().all(|digest| digest.len() == 64)
        );
    }
}
