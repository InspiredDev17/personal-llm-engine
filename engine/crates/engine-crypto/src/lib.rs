//! Cryptographic identity and signature verification for model packages.
//!
//! This crate performs SHA-384 hashing and strict Ed25519 operations through
//! reviewed general-purpose cryptographic dependencies. Model-package signing
//! keys remain outside the engine. The one private key accepted here is the
//! owner-provisioned engine response identity; its seed is zeroized after
//! construction and the underlying signing key zeroizes on drop.

use ed25519_dalek::{Signature, Signer, SigningKey, VerifyingKey};
use personalai_engine_types::{EngineError, ModelPackageHash, Sha256Digest, SignerKeyId};
use sha2::{Digest, Sha256, Sha384};
use std::collections::BTreeMap;
use std::io::Read;
use zeroize::Zeroize;

/// Domain separator for version-one model-package content signatures.
pub const MODEL_PACKAGE_SIGNATURE_DOMAIN_V1: &[u8] = b"PersonalAI.ModelPackage.Signature.v1\0";
/// Domain separator for directory-backed package hash catalogs.
pub const DIRECTORY_MODEL_PACKAGE_SIGNATURE_DOMAIN_V1: &[u8] =
    b"PersonalAI.DirectoryModelPackage.Signature.v1\0";
/// Domain separator for Model Foundry registry V2 decisions.
pub const MODEL_FOUNDRY_REGISTRY_SIGNATURE_DOMAIN_V2: &[u8] =
    b"PersonalAI.ModelRegistry.Entry.Signature.v2\0";
/// Domain separator for Go control-plane request signatures.
pub const CONTROL_PLANE_REQUEST_SIGNATURE_DOMAIN_V1: &[u8] =
    b"PersonalAI.ControlPlane.Request.Signature.v1\0";
/// Domain separator for version-two engine response signatures.
pub const ENGINE_RESPONSE_SIGNATURE_DOMAIN_V2: &[u8] = b"PersonalAI.Engine.Response.Signature.v2\0";
/// Domain separator for version-one installed-engine release manifests.
pub const ENGINE_RELEASE_SIGNATURE_DOMAIN_V1: &[u8] = b"PersonalAI.Engine.Release.Signature.v1\0";
/// Ed25519 public-key length.
pub const ED25519_PUBLIC_KEY_BYTES: usize = 32;
/// Ed25519 signature length.
pub const ED25519_SIGNATURE_BYTES: usize = 64;
/// Maximum simultaneously accepted Go control-plane verification identities.
pub const MAX_TRUSTED_CONTROL_PLANE_KEYS: usize = 2;

/// Computes a canonical SHA-256 content digest.
///
/// SHA-256 is used only where the pinned Sovereign contract requires it. Model
/// packages continue to use their separately tagged SHA-384 identity.
pub fn hash_sha256(bytes: &[u8]) -> Sha256Digest {
    let digest = Sha256::digest(bytes);
    let mut exact = [0_u8; Sha256Digest::BYTE_LEN];
    exact.copy_from_slice(&digest);
    Sha256Digest::from_bytes(exact)
}

/// Computes SHA-256 over a stream while enforcing an exact byte ceiling.
///
/// This is the streaming counterpart to [`hash_sha256`]. It lets offline
/// inspection bind a large local artifact to one already-open file descriptor
/// without allocating the complete artifact in memory.
pub fn hash_sha256_bounded_reader(
    reader: &mut impl Read,
    maximum_bytes: u64,
) -> Result<(Sha256Digest, u64), EngineError> {
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 16 * 1024];
    let mut total = 0_u64;
    loop {
        let read = reader
            .read(&mut buffer)
            .map_err(|_| EngineError::invalid_input("bounded SHA-256 input could not be read"))?;
        if read == 0 {
            break;
        }
        total = total
            .checked_add(u64::try_from(read).map_err(|_| {
                EngineError::internal_invariant("bounded SHA-256 read length overflows")
            })?)
            .ok_or_else(|| EngineError::resource_exhaustion("bounded SHA-256 length overflows"))?;
        if total > maximum_bytes {
            return Err(EngineError::resource_exhaustion("bounded SHA-256 input exceeds limit"));
        }
        hasher.update(&buffer[..read]);
    }
    let digest = hasher.finalize();
    let mut bytes = [0_u8; Sha256Digest::BYTE_LEN];
    bytes.copy_from_slice(&digest);
    Ok((Sha256Digest::from_bytes(bytes), total))
}

/// Operator-configured Ed25519 verification key.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TrustedSigner {
    key_id: SignerKeyId,
    public_key: [u8; ED25519_PUBLIC_KEY_BYTES],
    active: bool,
}

/// Operator-configured release-signing verification identity.
///
/// This deliberately distinct type prevents model-package, request, and
/// response trust identities from being passed to release verification APIs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TrustedReleaseSigner {
    key_id: SignerKeyId,
    public_key: [u8; ED25519_PUBLIC_KEY_BYTES],
    active: bool,
}

/// Offline Ed25519 identity used only to sign engine release manifests.
///
/// The value is neither cloneable nor printable and zeroizes the caller's
/// seed during construction. Production inference services never construct it.
pub struct EngineReleaseSigner {
    key_id: SignerKeyId,
    signing_key: SigningKey,
}

/// Operator-configured Go control-plane request-verification key.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TrustedControlPlaneKey {
    key_id: SignerKeyId,
    public_key: [u8; ED25519_PUBLIC_KEY_BYTES],
    active: bool,
}

/// Bounded, owner-configured control-plane verification key ring.
///
/// The two-entry ceiling supports a staged current/previous overlap without
/// allowing unbounded trust expansion or runtime key discovery.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TrustedControlPlaneKeys {
    keys: BTreeMap<SignerKeyId, TrustedControlPlaneKey>,
}

/// Owner-provisioned Ed25519 identity used only to sign engine responses.
///
/// The value is intentionally neither cloneable nor printable. It confers no
/// model, tool, filesystem, network, approval, or control-plane authority.
pub struct EngineResponseSigner {
    key_id: SignerKeyId,
    signing_key: SigningKey,
}

impl EngineResponseSigner {
    /// Constructs a response signer and clears the caller's seed in place.
    pub fn from_seed(
        key_id: SignerKeyId,
        seed: &mut [u8; ED25519_PUBLIC_KEY_BYTES],
    ) -> Result<Self, EngineError> {
        let signing_key = SigningKey::from_bytes(seed);
        seed.zeroize();
        validate_public_key(&signing_key.verifying_key().to_bytes())?;
        Ok(Self { key_id, signing_key })
    }

    /// Returns the configured response-key identifier.
    pub const fn key_id(&self) -> SignerKeyId {
        self.key_id
    }

    /// Returns the public verification key for configuration and tests.
    pub fn public_key(&self) -> [u8; ED25519_PUBLIC_KEY_BYTES] {
        self.signing_key.verifying_key().to_bytes()
    }

    /// Signs one exact unsigned response using the v2 domain transcript.
    pub fn sign_response(
        &self,
        unsigned_response: &[u8],
    ) -> Result<[u8; ED25519_SIGNATURE_BYTES], EngineError> {
        let message = engine_response_signature_message(unsigned_response)?;
        Ok(self.signing_key.sign(&message).to_bytes())
    }
}

impl EngineReleaseSigner {
    /// Constructs a release signer and clears the caller's seed in place.
    pub fn from_seed(
        key_id: SignerKeyId,
        seed: &mut [u8; ED25519_PUBLIC_KEY_BYTES],
    ) -> Result<Self, EngineError> {
        let signing_key = SigningKey::from_bytes(seed);
        seed.zeroize();
        validate_public_key(&signing_key.verifying_key().to_bytes())?;
        Ok(Self { key_id, signing_key })
    }

    /// Returns the configured release-key identifier.
    pub const fn key_id(&self) -> SignerKeyId {
        self.key_id
    }

    /// Returns the public verification key for owner configuration.
    pub fn public_key(&self) -> [u8; ED25519_PUBLIC_KEY_BYTES] {
        self.signing_key.verifying_key().to_bytes()
    }

    /// Signs one exact release-manifest digest under the release-only domain.
    pub fn sign_manifest_digest(
        &self,
        manifest_digest: &[u8; ModelPackageHash::BYTE_LEN],
    ) -> Result<[u8; ED25519_SIGNATURE_BYTES], EngineError> {
        let message = engine_release_signature_message(manifest_digest)?;
        Ok(self.signing_key.sign(&message).to_bytes())
    }
}

impl TrustedReleaseSigner {
    /// Creates one active release signer from exact public-key bytes.
    pub fn active(
        key_id: SignerKeyId,
        public_key: [u8; ED25519_PUBLIC_KEY_BYTES],
    ) -> Result<Self, EngineError> {
        validate_public_key(&public_key)?;
        Ok(Self { key_id, public_key, active: true })
    }

    /// Creates one explicitly revoked release signer.
    pub fn revoked(
        key_id: SignerKeyId,
        public_key: [u8; ED25519_PUBLIC_KEY_BYTES],
    ) -> Result<Self, EngineError> {
        validate_public_key(&public_key)?;
        Ok(Self { key_id, public_key, active: false })
    }

    /// Returns the configured release-key identifier.
    pub const fn key_id(&self) -> SignerKeyId {
        self.key_id
    }

    /// Returns whether this release key accepts signatures.
    pub const fn is_active(&self) -> bool {
        self.active
    }
}

impl TrustedControlPlaneKey {
    /// Creates one active trusted control-plane key.
    pub fn active(
        key_id: SignerKeyId,
        public_key: [u8; ED25519_PUBLIC_KEY_BYTES],
    ) -> Result<Self, EngineError> {
        validate_public_key(&public_key)?;
        Ok(Self { key_id, public_key, active: true })
    }

    /// Creates one explicitly revoked control-plane key.
    pub fn revoked(
        key_id: SignerKeyId,
        public_key: [u8; ED25519_PUBLIC_KEY_BYTES],
    ) -> Result<Self, EngineError> {
        validate_public_key(&public_key)?;
        Ok(Self { key_id, public_key, active: false })
    }

    /// Returns the configured key identifier.
    pub const fn key_id(&self) -> SignerKeyId {
        self.key_id
    }

    /// Returns the configured public verification key.
    pub const fn public_key(&self) -> [u8; ED25519_PUBLIC_KEY_BYTES] {
        self.public_key
    }

    /// Returns whether this control-plane key currently accepts signatures.
    pub const fn is_active(&self) -> bool {
        self.active
    }
}

impl TrustedControlPlaneKeys {
    /// Constructs a bounded key ring from explicit local configuration.
    pub fn new(
        keys: impl IntoIterator<Item = TrustedControlPlaneKey>,
    ) -> Result<Self, EngineError> {
        let mut configured = BTreeMap::new();
        for key in keys {
            if configured.len() == MAX_TRUSTED_CONTROL_PLANE_KEYS {
                return Err(EngineError::invalid_input(
                    "control-plane key ring exceeds its hard bound",
                ));
            }
            if configured
                .values()
                .any(|existing: &TrustedControlPlaneKey| existing.public_key() == key.public_key())
            {
                return Err(EngineError::invalid_input(
                    "control-plane key ring contains a duplicate public key",
                ));
            }
            if configured.insert(key.key_id(), key).is_some() {
                return Err(EngineError::invalid_input(
                    "control-plane key ring contains a duplicate key ID",
                ));
            }
        }
        if configured.is_empty() {
            return Err(EngineError::invalid_input("control-plane key ring is empty"));
        }
        if !configured.values().any(TrustedControlPlaneKey::is_active) {
            return Err(EngineError::invalid_input("control-plane key ring has no active key"));
        }
        Ok(Self { keys: configured })
    }

    /// Constructs a one-key ring for non-rotating deployments.
    pub fn single(key: TrustedControlPlaneKey) -> Result<Self, EngineError> {
        Self::new([key])
    }

    /// Resolves an explicitly configured active key without fallback.
    pub fn resolve(&self, key_id: SignerKeyId) -> Result<&TrustedControlPlaneKey, EngineError> {
        let key = self.keys.get(&key_id).ok_or_else(|| {
            EngineError::authentication_failure(
                "control-plane key does not match local configuration",
            )
        })?;
        if !key.is_active() {
            return Err(EngineError::authentication_failure(
                "control-plane authentication key is revoked",
            ));
        }
        Ok(key)
    }

    /// Iterates over active configured keys in deterministic key-ID order.
    pub fn active_keys(&self) -> impl Iterator<Item = &TrustedControlPlaneKey> {
        self.keys.values().filter(|key| key.is_active())
    }

    /// Iterates over every configured key in deterministic key-ID order.
    pub fn configured_keys(&self) -> impl Iterator<Item = &TrustedControlPlaneKey> {
        self.keys.values()
    }
}

impl TrustedSigner {
    /// Creates one active trusted signer from exact public-key bytes.
    pub fn active(
        key_id: SignerKeyId,
        public_key: [u8; ED25519_PUBLIC_KEY_BYTES],
    ) -> Result<Self, EngineError> {
        validate_public_key(&public_key)?;
        Ok(Self { key_id, public_key, active: true })
    }

    /// Creates one explicitly revoked signer for denial-path tests and state.
    pub fn revoked(
        key_id: SignerKeyId,
        public_key: [u8; ED25519_PUBLIC_KEY_BYTES],
    ) -> Result<Self, EngineError> {
        validate_public_key(&public_key)?;
        Ok(Self { key_id, public_key, active: false })
    }

    /// Returns the configured key identifier.
    pub const fn key_id(&self) -> SignerKeyId {
        self.key_id
    }

    /// Returns the exact public verification key bytes.
    pub const fn public_key(&self) -> [u8; ED25519_PUBLIC_KEY_BYTES] {
        self.public_key
    }

    /// Returns whether this signer is currently active.
    pub const fn is_active(&self) -> bool {
        self.active
    }
}

/// Computes the canonical SHA-384 identity of complete package bytes.
pub fn hash_package(package_bytes: &[u8]) -> ModelPackageHash {
    let mut accumulator = PackageHashAccumulator::new();
    accumulator.update(package_bytes);
    accumulator.finalize()
}

/// Incremental SHA-384 package-identity accumulator.
///
/// This narrow wrapper lets bounded engine readers verify large immutable
/// regions in cancellation-sized chunks without importing cryptographic
/// implementation details into loader or placement crates. Callers remain
/// responsible for canonical framing when hashing multiple logical fields.
pub struct PackageHashAccumulator(Sha384);

impl PackageHashAccumulator {
    /// Starts one empty SHA-384 transcript.
    pub fn new() -> Self {
        Self(Sha384::new())
    }

    /// Adds exact bytes to the transcript.
    pub fn update(&mut self, bytes: &[u8]) {
        self.0.update(bytes);
    }

    /// Finalizes the transcript as the engine's SHA-384-tagged digest type.
    pub fn finalize(self) -> ModelPackageHash {
        let digest = self.0.finalize();
        let mut bytes = [0_u8; ModelPackageHash::BYTE_LEN];
        bytes.copy_from_slice(&digest);
        ModelPackageHash::from_sha384_bytes(bytes)
    }
}

impl Default for PackageHashAccumulator {
    fn default() -> Self {
        Self::new()
    }
}

/// Computes the SHA-384 content digest used by the signature transcript.
pub fn hash_signed_content(content_bytes: &[u8]) -> [u8; ModelPackageHash::BYTE_LEN] {
    let digest = Sha384::digest(content_bytes);
    let mut bytes = [0_u8; ModelPackageHash::BYTE_LEN];
    bytes.copy_from_slice(&digest);
    bytes
}

/// Computes SHA-384 over a stream while enforcing an exact byte ceiling.
///
/// The returned length lets callers bind filesystem metadata without loading
/// the complete artifact into memory.
pub fn hash_bounded_reader(
    reader: &mut impl Read,
    maximum_bytes: u64,
) -> Result<([u8; ModelPackageHash::BYTE_LEN], u64), EngineError> {
    let mut hasher = Sha384::new();
    let mut buffer = [0_u8; 16 * 1024];
    let mut total = 0_u64;
    loop {
        let read = reader
            .read(&mut buffer)
            .map_err(|_| EngineError::invalid_input("bounded hash input could not be read"))?;
        if read == 0 {
            break;
        }
        total = total
            .checked_add(u64::try_from(read).map_err(|_| {
                EngineError::internal_invariant("bounded hash read length overflows")
            })?)
            .ok_or_else(|| EngineError::resource_exhaustion("bounded hash length overflows"))?;
        if total > maximum_bytes {
            return Err(EngineError::resource_exhaustion("bounded hash input exceeds limit"));
        }
        hasher.update(&buffer[..read]);
    }
    let digest = hasher.finalize();
    let mut bytes = [0_u8; ModelPackageHash::BYTE_LEN];
    bytes.copy_from_slice(&digest);
    Ok((bytes, total))
}

/// Builds the bounded, domain-separated message signed by package tooling.
pub fn signature_message(
    content_digest: &[u8; ModelPackageHash::BYTE_LEN],
) -> Result<Vec<u8>, EngineError> {
    domain_separated_message(MODEL_PACKAGE_SIGNATURE_DOMAIN_V1, content_digest)
}

/// Builds the bounded message signed by offline directory-package tooling.
pub fn directory_package_signature_message(
    catalog_digest: &[u8; ModelPackageHash::BYTE_LEN],
) -> Result<Vec<u8>, EngineError> {
    domain_separated_message(DIRECTORY_MODEL_PACKAGE_SIGNATURE_DOMAIN_V1, catalog_digest)
}

/// Builds the bounded message signed by Model Foundry registry V2 tooling.
pub fn model_foundry_registry_signature_message(
    entry_digest: &[u8; ModelPackageHash::BYTE_LEN],
) -> Result<Vec<u8>, EngineError> {
    domain_separated_message(MODEL_FOUNDRY_REGISTRY_SIGNATURE_DOMAIN_V2, entry_digest)
}

/// Builds the bounded message signed by the Go control plane.
pub fn control_plane_signature_message(request_bytes: &[u8]) -> Result<Vec<u8>, EngineError> {
    let digest = hash_signed_content(request_bytes);
    domain_separated_message(CONTROL_PLANE_REQUEST_SIGNATURE_DOMAIN_V1, &digest)
}

/// Builds the bounded message signed by the native engine for Go.
pub fn engine_response_signature_message(response_bytes: &[u8]) -> Result<Vec<u8>, EngineError> {
    let digest = hash_signed_content(response_bytes);
    domain_separated_message(ENGINE_RESPONSE_SIGNATURE_DOMAIN_V2, &digest)
}

/// Builds the bounded transcript signed by offline release tooling.
pub fn engine_release_signature_message(
    manifest_digest: &[u8; ModelPackageHash::BYTE_LEN],
) -> Result<Vec<u8>, EngineError> {
    domain_separated_message(ENGINE_RELEASE_SIGNATURE_DOMAIN_V1, manifest_digest)
}

/// Strictly verifies one release manifest under a release-only trust identity.
pub fn verify_engine_release_signature(
    signer: &TrustedReleaseSigner,
    envelope_key_id: SignerKeyId,
    manifest_digest: &[u8; ModelPackageHash::BYTE_LEN],
    signature_bytes: &[u8; ED25519_SIGNATURE_BYTES],
) -> Result<(), EngineError> {
    if !signer.is_active() {
        return Err(EngineError::signature_failure("engine release signer is revoked"));
    }
    if signer.key_id() != envelope_key_id {
        return Err(EngineError::signature_failure(
            "engine release signer does not match owner configuration",
        ));
    }
    let verifying_key = VerifyingKey::from_bytes(&signer.public_key)
        .map_err(|_| EngineError::signature_failure("configured release key is invalid"))?;
    if verifying_key.is_weak() {
        return Err(EngineError::signature_failure("configured release key is weak"));
    }
    let signature = Signature::from_bytes(signature_bytes);
    let message = engine_release_signature_message(manifest_digest)?;
    verifying_key
        .verify_strict(&message, &signature)
        .map_err(|_| EngineError::signature_failure("engine release signature is invalid"))
}

/// Verifies one strict Ed25519 package signature against operator trust state.
pub fn verify_package_signature(
    signer: &TrustedSigner,
    envelope_key_id: SignerKeyId,
    content_digest: &[u8; ModelPackageHash::BYTE_LEN],
    signature_bytes: &[u8; ED25519_SIGNATURE_BYTES],
) -> Result<(), EngineError> {
    if !signer.is_active() {
        return Err(EngineError::signature_failure("model package signer is revoked"));
    }
    if signer.key_id() != envelope_key_id {
        return Err(EngineError::signature_failure(
            "model package signer does not match local configuration",
        ));
    }
    let verifying_key = VerifyingKey::from_bytes(&signer.public_key)
        .map_err(|_| EngineError::signature_failure("configured Ed25519 key is invalid"))?;
    if verifying_key.is_weak() {
        return Err(EngineError::signature_failure("configured Ed25519 key is weak"));
    }
    let signature = Signature::from_bytes(signature_bytes);
    let message = signature_message(content_digest)?;
    verifying_key
        .verify_strict(&message, &signature)
        .map_err(|_| EngineError::signature_failure("model package signature is invalid"))
}

/// Verifies one strict Ed25519 directory-package hash-catalog signature.
pub fn verify_directory_package_signature(
    signer: &TrustedSigner,
    envelope_key_id: SignerKeyId,
    catalog_digest: &[u8; ModelPackageHash::BYTE_LEN],
    signature_bytes: &[u8; ED25519_SIGNATURE_BYTES],
) -> Result<(), EngineError> {
    if !signer.is_active() {
        return Err(EngineError::signature_failure("model package signer is revoked"));
    }
    if signer.key_id() != envelope_key_id {
        return Err(EngineError::signature_failure(
            "model package signer does not match local configuration",
        ));
    }
    let verifying_key = VerifyingKey::from_bytes(&signer.public_key)
        .map_err(|_| EngineError::signature_failure("configured Ed25519 key is invalid"))?;
    if verifying_key.is_weak() {
        return Err(EngineError::signature_failure("configured Ed25519 key is weak"));
    }
    let signature = Signature::from_bytes(signature_bytes);
    let message = directory_package_signature_message(catalog_digest)?;
    verifying_key
        .verify_strict(&message, &signature)
        .map_err(|_| EngineError::signature_failure("model package signature is invalid"))
}

/// Verifies one exact Model Foundry registry V2 entry under active owner trust.
pub fn verify_model_foundry_registry_signature(
    signer: &TrustedSigner,
    envelope_key_id: SignerKeyId,
    entry_digest: &[u8; ModelPackageHash::BYTE_LEN],
    signature_bytes: &[u8; ED25519_SIGNATURE_BYTES],
) -> Result<(), EngineError> {
    if !signer.is_active() {
        return Err(EngineError::signature_failure("model registry signer is revoked"));
    }
    if signer.key_id() != envelope_key_id {
        return Err(EngineError::signature_failure(
            "model registry signer does not match owner configuration",
        ));
    }
    let verifying_key = VerifyingKey::from_bytes(&signer.public_key)
        .map_err(|_| EngineError::signature_failure("configured registry key is invalid"))?;
    if verifying_key.is_weak() {
        return Err(EngineError::signature_failure("configured registry key is weak"));
    }
    let signature = Signature::from_bytes(signature_bytes);
    let message = model_foundry_registry_signature_message(entry_digest)?;
    verifying_key
        .verify_strict(&message, &signature)
        .map_err(|_| EngineError::signature_failure("model registry signature is invalid"))
}

/// Strictly verifies one authenticated Go control-plane request body.
pub fn verify_control_plane_signature(
    trusted_key: &TrustedControlPlaneKey,
    envelope_key_id: SignerKeyId,
    request_bytes: &[u8],
    signature_bytes: &[u8; ED25519_SIGNATURE_BYTES],
) -> Result<(), EngineError> {
    if !trusted_key.active {
        return Err(EngineError::authentication_failure(
            "control-plane authentication key is revoked",
        ));
    }
    if trusted_key.key_id() != envelope_key_id {
        return Err(EngineError::authentication_failure(
            "control-plane key does not match local configuration",
        ));
    }
    let verifying_key = VerifyingKey::from_bytes(&trusted_key.public_key)
        .map_err(|_| EngineError::authentication_failure("control-plane key is invalid"))?;
    if verifying_key.is_weak() {
        return Err(EngineError::authentication_failure("control-plane key is weak"));
    }
    let signature = Signature::from_bytes(signature_bytes);
    let message = control_plane_signature_message(request_bytes)?;
    verifying_key
        .verify_strict(&message, &signature)
        .map_err(|_| EngineError::authentication_failure("control-plane signature is invalid"))
}

/// Resolves and strictly verifies one request against a bounded key ring.
pub fn verify_control_plane_signature_with_keys(
    trusted_keys: &TrustedControlPlaneKeys,
    envelope_key_id: SignerKeyId,
    request_bytes: &[u8],
    signature_bytes: &[u8; ED25519_SIGNATURE_BYTES],
) -> Result<(), EngineError> {
    let trusted_key = trusted_keys.resolve(envelope_key_id)?;
    verify_control_plane_signature(trusted_key, envelope_key_id, request_bytes, signature_bytes)
}

/// Strictly verifies one engine response against an exact pinned identity.
pub fn verify_engine_response_signature(
    expected_key_id: SignerKeyId,
    expected_public_key: &[u8; ED25519_PUBLIC_KEY_BYTES],
    envelope_key_id: SignerKeyId,
    response_bytes: &[u8],
    signature_bytes: &[u8; ED25519_SIGNATURE_BYTES],
) -> Result<(), EngineError> {
    if expected_key_id != envelope_key_id {
        return Err(EngineError::authentication_failure(
            "engine response key does not match local configuration",
        ));
    }
    let verifying_key = VerifyingKey::from_bytes(expected_public_key)
        .map_err(|_| EngineError::authentication_failure("engine response key is invalid"))?;
    if verifying_key.is_weak() {
        return Err(EngineError::authentication_failure("engine response key is weak"));
    }
    let signature = Signature::from_bytes(signature_bytes);
    let transcript = engine_response_signature_message(response_bytes)?;
    verifying_key
        .verify_strict(&transcript, &signature)
        .map_err(|_| EngineError::authentication_failure("engine response signature is invalid"))
}

fn domain_separated_message(
    domain: &[u8],
    digest: &[u8; ModelPackageHash::BYTE_LEN],
) -> Result<Vec<u8>, EngineError> {
    let message_length = domain
        .len()
        .checked_add(digest.len())
        .ok_or_else(|| EngineError::internal_invariant("signature transcript length overflows"))?;
    let mut message = Vec::new();
    message
        .try_reserve_exact(message_length)
        .map_err(|_| EngineError::resource_exhaustion("signature transcript allocation failed"))?;
    message.extend_from_slice(domain);
    message.extend_from_slice(digest);
    Ok(message)
}

fn validate_public_key(public_key: &[u8; ED25519_PUBLIC_KEY_BYTES]) -> Result<(), EngineError> {
    let key = VerifyingKey::from_bytes(public_key)
        .map_err(|_| EngineError::invalid_input("trusted Ed25519 public key is invalid"))?;
    if key.is_weak() {
        return Err(EngineError::invalid_input("trusted Ed25519 public key is weak"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ed25519_dalek::{Signer, SigningKey};

    const TEST_SECRET: [u8; 32] = [7_u8; 32];

    #[test]
    fn sha384_known_answer_matches_nist_vector() {
        let hash = hash_package(b"abc");
        assert_eq!(
            hash.to_string(),
            concat!(
                "sha384:cb00753f45a35e8bb5a03d699ac65007272c32ab0eded163",
                "1a8b605a43ff5bed8086072ba1e7cc2358baeca134c825a7"
            )
        );
    }

    #[test]
    fn incremental_sha384_matches_single_slice_identity() {
        let mut accumulator = PackageHashAccumulator::new();
        accumulator.update(b"a");
        accumulator.update(b"b");
        accumulator.update(b"c");
        assert_eq!(accumulator.finalize(), hash_package(b"abc"));
    }

    #[test]
    fn strict_signature_accepts_exact_transcript_and_rejects_tampering() {
        let signing_key = SigningKey::from_bytes(&TEST_SECRET);
        let key_id = SignerKeyId::from_bytes([1_u8; 16]).expect("key id must validate");
        let signer = TrustedSigner::active(key_id, signing_key.verifying_key().to_bytes())
            .expect("public key must validate");
        let digest = hash_signed_content(b"signed package content");
        let message = signature_message(&digest).expect("message must construct");
        let signature = signing_key.sign(&message).to_bytes();
        verify_package_signature(&signer, key_id, &digest, &signature)
            .expect("exact signature must verify");

        let mut tampered = digest;
        tampered[0] ^= 1;
        assert!(verify_package_signature(&signer, key_id, &tampered, &signature).is_err());
        let revoked = TrustedSigner::revoked(key_id, signing_key.verifying_key().to_bytes())
            .expect("public key must validate");
        assert!(verify_package_signature(&revoked, key_id, &digest, &signature).is_err());
    }

    #[test]
    fn response_signer_zeroizes_seed_and_produces_strict_v2_signature() {
        let mut seed = [0x5a; 32];
        let key_id = SignerKeyId::from_bytes([2_u8; 16]).expect("key id must validate");
        let signer = EngineResponseSigner::from_seed(key_id, &mut seed)
            .expect("response signer must validate");
        assert_eq!(seed, [0_u8; 32]);
        let unsigned = b"bounded response";
        let signature =
            Signature::from_bytes(&signer.sign_response(unsigned).expect("response must sign"));
        let transcript = engine_response_signature_message(unsigned)
            .expect("response transcript must construct");
        signer
            .signing_key
            .verifying_key()
            .verify_strict(&transcript, &signature)
            .expect("exact response signature must verify");
        let tampered = engine_response_signature_message(b"tampered")
            .expect("tampered transcript must construct");
        assert!(signer.signing_key.verifying_key().verify_strict(&tampered, &signature).is_err());
    }

    #[test]
    fn release_signer_is_role_separated_zeroized_and_strict() {
        let key_id = SignerKeyId::from_bytes([0x71; 16]).expect("key ID must validate");
        let mut seed = [0x72; 32];
        let signer =
            EngineReleaseSigner::from_seed(key_id, &mut seed).expect("release signer validates");
        assert_eq!(seed, [0_u8; 32]);
        let trusted = TrustedReleaseSigner::active(key_id, signer.public_key())
            .expect("release trust validates");
        let digest = hash_signed_content(b"canonical engine release manifest");
        let signature = signer.sign_manifest_digest(&digest).expect("release manifest must sign");
        verify_engine_release_signature(&trusted, key_id, &digest, &signature)
            .expect("exact release signature must verify");
        let mut tampered = digest;
        tampered[0] ^= 1;
        assert!(verify_engine_release_signature(&trusted, key_id, &tampered, &signature).is_err());
        let revoked = TrustedReleaseSigner::revoked(key_id, signer.public_key())
            .expect("revoked release key validates");
        assert!(verify_engine_release_signature(&revoked, key_id, &digest, &signature).is_err());
        let package_transcript = signature_message(&digest).expect("package transcript builds");
        let release_signature = Signature::from_bytes(&signature);
        assert!(
            signer
                .signing_key
                .verifying_key()
                .verify_strict(&package_transcript, &release_signature)
                .is_err(),
            "release signatures must not verify under the model-package domain"
        );
    }

    #[test]
    fn bounded_stream_hash_matches_slice_hash_and_rejects_excess() {
        let bytes = vec![0x5a; 40_000];
        let (digest, length) =
            hash_bounded_reader(&mut bytes.as_slice(), 40_000).expect("bounded hash succeeds");
        assert_eq!(digest, hash_signed_content(&bytes));
        assert_eq!(length, 40_000);
        assert!(hash_bounded_reader(&mut bytes.as_slice(), 39_999).is_err());
    }

    #[test]
    fn bounded_sha256_stream_matches_slice_hash_and_rejects_excess() {
        let bytes = vec![0xa5; 140_000];
        let (digest, length) = hash_sha256_bounded_reader(&mut bytes.as_slice(), 140_000)
            .expect("bounded SHA-256 succeeds");
        assert_eq!(digest, hash_sha256(&bytes));
        assert_eq!(length, 140_000);
        assert!(hash_sha256_bounded_reader(&mut bytes.as_slice(), 139_999).is_err());
    }

    #[test]
    fn control_key_ring_is_bounded_unique_and_fail_closed() {
        let first_signing_key = SigningKey::from_bytes(&[0x31; 32]);
        let second_signing_key = SigningKey::from_bytes(&[0x32; 32]);
        let third_signing_key = SigningKey::from_bytes(&[0x33; 32]);
        let first_id = SignerKeyId::from_bytes([0x41; 16]).expect("key ID must validate");
        let second_id = SignerKeyId::from_bytes([0x42; 16]).expect("key ID must validate");
        let third_id = SignerKeyId::from_bytes([0x43; 16]).expect("key ID must validate");
        let first =
            TrustedControlPlaneKey::active(first_id, first_signing_key.verifying_key().to_bytes())
                .expect("first key must validate");
        let second = TrustedControlPlaneKey::active(
            second_id,
            second_signing_key.verifying_key().to_bytes(),
        )
        .expect("second key must validate");
        let ring = TrustedControlPlaneKeys::new([first.clone(), second.clone()])
            .expect("overlap ring must validate");
        assert_eq!(ring.resolve(first_id), Ok(&first));
        assert_eq!(ring.resolve(second_id), Ok(&second));
        assert!(ring.resolve(third_id).is_err());

        assert!(TrustedControlPlaneKeys::new(Vec::new()).is_err());
        assert!(TrustedControlPlaneKeys::new([first.clone(), first]).is_err());
        let duplicate_public_key =
            TrustedControlPlaneKey::active(third_id, first_signing_key.verifying_key().to_bytes())
                .expect("duplicate material is individually valid");
        assert!(TrustedControlPlaneKeys::new([second.clone(), duplicate_public_key]).is_ok());
        let duplicate_public_key =
            TrustedControlPlaneKey::active(third_id, second_signing_key.verifying_key().to_bytes())
                .expect("duplicate material is individually valid");
        assert!(TrustedControlPlaneKeys::new([second, duplicate_public_key]).is_err());

        let third =
            TrustedControlPlaneKey::active(third_id, third_signing_key.verifying_key().to_bytes())
                .expect("third key must validate");
        assert!(
            TrustedControlPlaneKeys::new([
                TrustedControlPlaneKey::active(
                    first_id,
                    first_signing_key.verifying_key().to_bytes(),
                )
                .expect("first key must validate"),
                TrustedControlPlaneKey::active(
                    second_id,
                    second_signing_key.verifying_key().to_bytes(),
                )
                .expect("second key must validate"),
                third,
            ])
            .is_err()
        );
    }

    #[test]
    fn revoked_control_key_is_retained_but_never_resolved() {
        let signing_key = SigningKey::from_bytes(&[0x61; 32]);
        let key_id = SignerKeyId::from_bytes([0x62; 16]).expect("key ID must validate");
        let revoked =
            TrustedControlPlaneKey::revoked(key_id, signing_key.verifying_key().to_bytes())
                .expect("revoked key must validate");
        assert!(TrustedControlPlaneKeys::single(revoked.clone()).is_err());
        let active_signing_key = SigningKey::from_bytes(&[0x63; 32]);
        let active_id = SignerKeyId::from_bytes([0x64; 16]).expect("key ID must validate");
        let active = TrustedControlPlaneKey::active(
            active_id,
            active_signing_key.verifying_key().to_bytes(),
        )
        .expect("active key must validate");
        let ring = TrustedControlPlaneKeys::new([revoked, active])
            .expect("ring with an active key must validate");
        assert!(ring.resolve(key_id).is_err());
        assert!(ring.resolve(active_id).is_ok());
    }
}
