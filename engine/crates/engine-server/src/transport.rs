//! Isolated local Unix-domain-socket frame transport.

use crate::{ServiceConfig, peer_credential};
use personalai_engine_protocol::{FRAME_HEADER_BYTES, Frame, MAX_FRAME_PAYLOAD_BYTES};
use personalai_engine_types::EngineError;
use std::fs;
use std::io::{Read, Write};
use std::net::Shutdown;
use std::os::unix::fs::{FileTypeExt, MetadataExt, PermissionsExt};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::PathBuf;
use std::time::Duration;

const DEFAULT_IO_TIMEOUT: Duration = Duration::from_secs(30);
const MIN_IO_TIMEOUT: Duration = Duration::from_millis(10);
const MAX_IO_TIMEOUT: Duration = Duration::from_secs(300);

/// Validated finite local connection and frame timeouts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct UnixSocketPolicy {
    idle_timeout: Duration,
    frame_timeout: Duration,
    recover_stale_socket: bool,
}

impl UnixSocketPolicy {
    /// Creates a same-user policy with finite bounded timeouts.
    pub fn new(
        idle_timeout: Duration,
        frame_timeout: Duration,
        recover_stale_socket: bool,
    ) -> Result<Self, EngineError> {
        if !(MIN_IO_TIMEOUT..=MAX_IO_TIMEOUT).contains(&idle_timeout)
            || !(MIN_IO_TIMEOUT..=MAX_IO_TIMEOUT).contains(&frame_timeout)
        {
            return Err(EngineError::invalid_input("Unix socket timeout is outside hard bounds"));
        }
        Ok(Self { idle_timeout, frame_timeout, recover_stale_socket })
    }

    /// Returns the maximum time waiting for a new command on a connection.
    pub const fn idle_timeout(self) -> Duration {
        self.idle_timeout
    }

    /// Returns the maximum time waiting for an individual frame read or write.
    pub const fn frame_timeout(self) -> Duration {
        self.frame_timeout
    }

    /// Returns whether inode-checked stale socket recovery is enabled.
    pub const fn recover_stale_socket(self) -> bool {
        self.recover_stale_socket
    }
}

impl Default for UnixSocketPolicy {
    fn default() -> Self {
        Self {
            idle_timeout: DEFAULT_IO_TIMEOUT,
            frame_timeout: DEFAULT_IO_TIMEOUT,
            recover_stale_socket: true,
        }
    }
}

/// Bound local-only socket with inode-safe cleanup ownership.
#[derive(Debug)]
pub struct UnixFrameListener {
    listener: UnixListener,
    socket_path: PathBuf,
    device: u64,
    inode: u64,
    allowed_peer_user_id: u32,
    policy: UnixSocketPolicy,
}

impl UnixFrameListener {
    /// Binds the configured socket only inside an existing owner-private runtime directory.
    pub fn bind(config: &ServiceConfig) -> Result<Self, EngineError> {
        Self::bind_with_policy(config, UnixSocketPolicy::default())
    }

    /// Binds with explicit finite I/O timeouts and stale-socket policy.
    pub fn bind_with_policy(
        config: &ServiceConfig,
        policy: UnixSocketPolicy,
    ) -> Result<Self, EngineError> {
        let runtime_metadata = fs::symlink_metadata(config.runtime_root())
            .map_err(|_| EngineError::invalid_input("runtime root is unavailable"))?;
        if !runtime_metadata.is_dir()
            || runtime_metadata.file_type().is_symlink()
            || runtime_metadata.permissions().mode() & 0o077 != 0
        {
            return Err(EngineError::authentication_failure(
                "runtime root must be a non-symlink owner-private directory",
            ));
        }
        prepare_socket_path(config, policy)?;
        let listener = UnixListener::bind(config.socket_path())
            .map_err(|_| EngineError::backend_failure("Unix socket bind failed"))?;
        fs::set_permissions(config.socket_path(), fs::Permissions::from_mode(0o600))
            .map_err(|_| EngineError::backend_failure("Unix socket permission setup failed"))?;
        let socket_metadata = fs::symlink_metadata(config.socket_path())
            .map_err(|_| EngineError::backend_failure("Unix socket metadata is unavailable"))?;
        if !socket_metadata.file_type().is_socket() {
            return Err(EngineError::internal_invariant("bound Unix socket path is not a socket"));
        }
        Ok(Self {
            listener,
            socket_path: config.socket_path().to_path_buf(),
            device: socket_metadata.dev(),
            inode: socket_metadata.ino(),
            allowed_peer_user_id: peer_credential::effective_user_id(),
            policy,
        })
    }

    /// Accepts one local connection and installs finite I/O timeouts.
    pub fn accept(&self) -> Result<UnixFrameConnection, EngineError> {
        loop {
            match self.accept_once() {
                Err(EngineError::AuthenticationFailure(_)) => {}
                result => return result,
            }
        }
    }

    fn accept_once(&self) -> Result<UnixFrameConnection, EngineError> {
        let (stream, _address) = self
            .listener
            .accept()
            .map_err(|_| EngineError::backend_failure("Unix socket accept failed"))?;
        self.verify_peer(&stream)?;
        configure_stream(&stream, self.policy)?;
        Ok(UnixFrameConnection { stream })
    }

    /// Switches the listener between blocking and readiness-polled admission.
    pub fn set_nonblocking(&self, nonblocking: bool) -> Result<(), EngineError> {
        self.listener
            .set_nonblocking(nonblocking)
            .map_err(|_| EngineError::backend_failure("Unix listener mode setup failed"))
    }

    /// Accepts one ready local connection without waiting when none is pending.
    ///
    /// The listener must have been placed in nonblocking mode by its owner.
    pub fn try_accept(&self) -> Result<Option<UnixFrameConnection>, EngineError> {
        match self.listener.accept() {
            Ok((stream, _address)) => {
                self.verify_peer(&stream)?;
                configure_stream(&stream, self.policy)?;
                Ok(Some(UnixFrameConnection { stream }))
            }
            Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => Ok(None),
            Err(_) => Err(EngineError::backend_failure("Unix socket accept failed")),
        }
    }

    /// Returns the exact bound socket path.
    pub fn socket_path(&self) -> &std::path::Path {
        &self.socket_path
    }

    /// Returns the only effective user ID admitted by this listener.
    pub const fn allowed_peer_user_id(&self) -> u32 {
        self.allowed_peer_user_id
    }

    #[cfg(test)]
    pub(crate) fn reject_current_peer_for_test(&mut self) {
        self.allowed_peer_user_id = self.allowed_peer_user_id.checked_add(1).unwrap_or(0);
    }

    fn verify_peer(&self, stream: &UnixStream) -> Result<(), EngineError> {
        let peer_user_id = peer_credential::peer_effective_user_id(stream)?;
        if peer_user_id != self.allowed_peer_user_id {
            return Err(EngineError::authentication_failure(
                "Unix peer effective user ID is not authorized",
            ));
        }
        Ok(())
    }
}

fn configure_stream(stream: &UnixStream, policy: UnixSocketPolicy) -> Result<(), EngineError> {
    stream
        .set_nonblocking(false)
        .map_err(|_| EngineError::backend_failure("Unix stream blocking mode setup failed"))?;
    stream
        .set_read_timeout(Some(policy.frame_timeout().min(policy.idle_timeout())))
        .map_err(|_| EngineError::backend_failure("Unix socket read timeout setup failed"))?;
    stream
        .set_write_timeout(Some(policy.frame_timeout()))
        .map_err(|_| EngineError::backend_failure("Unix socket write timeout setup failed"))
}

fn prepare_socket_path(
    config: &ServiceConfig,
    policy: UnixSocketPolicy,
) -> Result<(), EngineError> {
    let metadata = match fs::symlink_metadata(config.socket_path()) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(_) => return Err(EngineError::authentication_failure("socket path metadata failed")),
    };
    if !policy.recover_stale_socket()
        || !metadata.file_type().is_socket()
        || metadata.uid() != peer_credential::effective_user_id()
        || metadata.permissions().mode() & 0o077 != 0
    {
        return Err(EngineError::authentication_failure(
            "existing socket path is not eligible for safe recovery",
        ));
    }
    match UnixStream::connect(config.socket_path()) {
        Ok(_active) => {
            return Err(EngineError::authentication_failure(
                "configured engine socket is already active",
            ));
        }
        Err(error)
            if matches!(
                error.kind(),
                std::io::ErrorKind::ConnectionRefused | std::io::ErrorKind::NotFound
            ) => {}
        Err(_) => {
            return Err(EngineError::authentication_failure(
                "existing socket liveness could not be established",
            ));
        }
    }
    let current = fs::symlink_metadata(config.socket_path())
        .map_err(|_| EngineError::authentication_failure("stale socket changed during recovery"))?;
    if !current.file_type().is_socket()
        || current.dev() != metadata.dev()
        || current.ino() != metadata.ino()
        || current.uid() != metadata.uid()
    {
        return Err(EngineError::authentication_failure(
            "stale socket identity changed during recovery",
        ));
    }
    fs::remove_file(config.socket_path())
        .map_err(|_| EngineError::authentication_failure("stale socket removal failed"))
}

impl Drop for UnixFrameListener {
    fn drop(&mut self) {
        let Ok(metadata) = fs::symlink_metadata(&self.socket_path) else {
            return;
        };
        if metadata.file_type().is_socket()
            && metadata.dev() == self.device
            && metadata.ino() == self.inode
        {
            let _cleanup_result = fs::remove_file(&self.socket_path);
        }
    }
}

/// One accepted local stream supporting exact bounded protocol frames.
#[derive(Debug)]
pub struct UnixFrameConnection {
    stream: UnixStream,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum InboundWatchEvent {
    Timeout,
    PeerClosed,
    UnexpectedData,
}

impl UnixFrameConnection {
    /// Clones the underlying local stream for a bounded full-duplex control watcher.
    pub fn try_clone(&self) -> Result<Self, EngineError> {
        let stream = self
            .stream
            .try_clone()
            .map_err(|_| EngineError::backend_failure("Unix stream clone failed"))?;
        Ok(Self { stream })
    }

    /// Stops further inbound reads without closing the outbound event stream.
    pub fn shutdown_inbound(&self) -> Result<(), EngineError> {
        match self.stream.shutdown(Shutdown::Read) {
            Ok(()) => Ok(()),
            // A client may close immediately after receiving its signed
            // terminal event. The cancellation watcher is already unable to
            // receive another command in that state, so ENOTCONN is the same
            // idempotent terminal condition rather than a failed generation.
            Err(error) if error.kind() == std::io::ErrorKind::NotConnected => Ok(()),
            Err(_) => Err(EngineError::backend_failure("Unix stream read shutdown failed")),
        }
    }

    /// Waits for peer liveness or unexpected follow-up data without mapping a
    /// finite socket timeout to a protocol violation. Runtime execution uses
    /// this only as a disconnect watcher; canonical cancellation arrives on a
    /// separately authenticated connection.
    pub(crate) fn watch_inbound_event(&mut self) -> Result<InboundWatchEvent, EngineError> {
        let mut byte = [0_u8; 1];
        loop {
            match self.stream.read(&mut byte) {
                Ok(0) => return Ok(InboundWatchEvent::PeerClosed),
                Ok(_) => return Ok(InboundWatchEvent::UnexpectedData),
                Err(error)
                    if matches!(
                        error.kind(),
                        std::io::ErrorKind::TimedOut | std::io::ErrorKind::WouldBlock
                    ) =>
                {
                    return Ok(InboundWatchEvent::Timeout);
                }
                Err(error) if error.kind() == std::io::ErrorKind::Interrupted => {}
                Err(_) => {
                    return Err(EngineError::backend_failure("Unix stream liveness watch failed"));
                }
            }
        }
    }

    /// Reads exactly one inbound frame without trusting its declared allocation size.
    pub fn read_inbound_frame(&mut self) -> Result<Frame, EngineError> {
        let mut header = [0_u8; FRAME_HEADER_BYTES];
        self.stream
            .read_exact(&mut header)
            .map_err(|_| EngineError::protocol_violation("Unix frame header read failed"))?;
        let payload_length = usize::try_from(u32::from_le_bytes(
            header[12..16]
                .try_into()
                .map_err(|_| EngineError::protocol_violation("frame length field is invalid"))?,
        ))
        .map_err(|_| EngineError::protocol_violation("frame length is not addressable"))?;
        if payload_length == 0 || payload_length > MAX_FRAME_PAYLOAD_BYTES {
            return Err(EngineError::protocol_violation(
                "declared Unix frame payload is outside hard bounds",
            ));
        }
        let total_length = FRAME_HEADER_BYTES
            .checked_add(payload_length)
            .ok_or_else(|| EngineError::protocol_violation("Unix frame length overflows"))?;
        let mut encoded = Vec::new();
        encoded
            .try_reserve_exact(total_length)
            .map_err(|_| EngineError::resource_exhaustion("Unix frame allocation failed"))?;
        encoded.extend_from_slice(&header);
        encoded.resize(total_length, 0);
        self.stream
            .read_exact(&mut encoded[FRAME_HEADER_BYTES..])
            .map_err(|_| EngineError::protocol_violation("Unix frame payload read failed"))?;
        Frame::decode_inbound_exact(&encoded)
    }

    /// Writes exactly one outbound frame and flushes it before returning.
    pub fn write_outbound_frame(&mut self, frame: &Frame) -> Result<(), EngineError> {
        let encoded = frame.encode()?;
        Frame::decode_outbound_exact(&encoded)?;
        self.stream
            .write_all(&encoded)
            .and_then(|()| self.stream.flush())
            .map_err(|_| EngineError::backend_failure("Unix frame write failed"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use personalai_engine_protocol::FrameKind;
    use std::net::Shutdown;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::mpsc;
    use std::thread;

    static NEXT_DIRECTORY: AtomicU64 = AtomicU64::new(1);

    #[test]
    fn owner_private_socket_round_trips_bounded_frames_and_cleans_its_inode() {
        let directory = temporary_directory();
        fs::create_dir(&directory).expect("temporary runtime directory must create");
        fs::set_permissions(&directory, fs::Permissions::from_mode(0o700))
            .expect("runtime permissions must set");
        let socket = directory.join("engine.sock");
        let config = ServiceConfig::new("/models", &directory, &socket)
            .expect("temporary config must validate");
        let listener = UnixFrameListener::bind(&config).expect("socket must bind");
        assert_eq!(listener.allowed_peer_user_id(), peer_credential::effective_user_id());
        let client_socket = socket.clone();
        let client = thread::spawn(move || {
            let mut stream = UnixStream::connect(client_socket).expect("client must connect");
            let inbound = Frame::new(FrameKind::Capability, vec![1, 0, 2])
                .expect("inbound frame must construct")
                .encode()
                .expect("inbound frame must encode");
            stream.write_all(&inbound).expect("client frame must write");
            let mut response = [0_u8; FRAME_HEADER_BYTES + 3];
            stream.read_exact(&mut response).expect("response must read");
            Frame::decode_outbound_exact(&response).expect("response must decode")
        });
        let mut connection = listener.accept().expect("server must accept");
        let inbound = connection.read_inbound_frame().expect("server must read");
        assert_eq!(inbound.payload(), &[1, 0, 2]);
        connection
            .write_outbound_frame(
                &Frame::new(FrameKind::Capability, vec![3, 2, 1]).expect("response must construct"),
            )
            .expect("server must write");
        let response = client.join().expect("client must finish");
        assert_eq!(response.payload(), &[3, 2, 1]);
        drop(listener);
        assert!(!socket.exists());
        fs::remove_dir(&directory).expect("temporary directory must remove");
    }

    #[test]
    fn mismatched_kernel_peer_user_is_rejected_before_frame_read() {
        let directory = temporary_directory();
        fs::create_dir(&directory).expect("temporary runtime directory must create");
        fs::set_permissions(&directory, fs::Permissions::from_mode(0o700))
            .expect("runtime permissions must set");
        let socket = directory.join("engine.sock");
        let config = ServiceConfig::new("/models", &directory, &socket)
            .expect("temporary config must validate");
        let mut listener = UnixFrameListener::bind(&config).expect("socket must bind");
        listener.reject_current_peer_for_test();
        assert_ne!(listener.allowed_peer_user_id(), peer_credential::effective_user_id());
        let client_socket = socket.clone();
        let client = thread::spawn(move || {
            drop(UnixStream::connect(client_socket).expect("client must connect"));
        });
        assert!(matches!(listener.accept_once(), Err(EngineError::AuthenticationFailure(_))));
        client.join().expect("client must finish");
        drop(listener);
        assert!(!socket.exists());
        fs::remove_dir(&directory).expect("temporary directory must remove");
    }

    #[test]
    fn permissive_runtime_directory_and_existing_socket_path_are_rejected() {
        let directory = temporary_directory();
        fs::create_dir(&directory).expect("temporary runtime directory must create");
        fs::set_permissions(&directory, fs::Permissions::from_mode(0o755))
            .expect("runtime permissions must set");
        let socket = directory.join("engine.sock");
        let config = ServiceConfig::new("/models", &directory, &socket)
            .expect("temporary config must validate");
        assert!(matches!(
            UnixFrameListener::bind(&config),
            Err(EngineError::AuthenticationFailure(_))
        ));
        fs::remove_dir(&directory).expect("temporary directory must remove");
    }

    #[test]
    fn stale_socket_recovery_is_inode_checked_and_configurable() {
        let directory = temporary_directory();
        fs::create_dir(&directory).expect("temporary runtime directory must create");
        fs::set_permissions(&directory, fs::Permissions::from_mode(0o700))
            .expect("runtime permissions must set");
        let socket = directory.join("engine.sock");
        let config = ServiceConfig::new("/models", &directory, &socket)
            .expect("temporary config must validate");
        let stale = UnixListener::bind(&socket).expect("stale socket must bind");
        fs::set_permissions(&socket, fs::Permissions::from_mode(0o600))
            .expect("stale socket permissions must set");
        drop(stale);

        let strict = UnixSocketPolicy::new(Duration::from_secs(1), Duration::from_secs(1), false)
            .expect("policy must validate");
        assert!(matches!(
            UnixFrameListener::bind_with_policy(&config, strict),
            Err(EngineError::AuthenticationFailure(_))
        ));
        let listener = UnixFrameListener::bind(&config).expect("stale socket must recover");
        drop(listener);
        fs::remove_dir(&directory).expect("temporary directory must remove");
    }

    #[test]
    fn socket_timeout_policy_is_bounded() {
        assert!(UnixSocketPolicy::new(Duration::ZERO, Duration::from_secs(1), true).is_err());
        assert!(
            UnixSocketPolicy::new(Duration::from_secs(1), Duration::from_secs(301), true,).is_err()
        );
        assert_eq!(
            UnixSocketPolicy::new(Duration::from_secs(2), Duration::from_secs(3), true,)
                .expect("policy")
                .frame_timeout(),
            Duration::from_secs(3)
        );
    }

    #[test]
    fn inbound_shutdown_is_idempotent_after_peer_disconnect() {
        let (server, peer) = UnixStream::pair().expect("local pair must create");
        let connection = UnixFrameConnection { stream: server };
        drop(peer);
        connection.shutdown_inbound().expect("disconnected inbound shutdown must be terminal");
        connection.shutdown_inbound().expect("repeated inbound shutdown must be idempotent");
    }

    #[test]
    fn inbound_liveness_watch_distinguishes_timeout_data_and_peer_close() {
        let (server, mut peer) = UnixStream::pair().expect("local pair must create");
        server
            .set_read_timeout(Some(Duration::from_millis(10)))
            .expect("read timeout must configure");
        let mut connection = UnixFrameConnection { stream: server };
        assert_eq!(
            connection.watch_inbound_event().expect("finite liveness poll"),
            InboundWatchEvent::Timeout
        );
        peer.write_all(&[0xff]).expect("unexpected follow-up byte must write");
        assert_eq!(
            connection.watch_inbound_event().expect("follow-up activity"),
            InboundWatchEvent::UnexpectedData
        );
        drop(peer);
        assert_eq!(
            connection.watch_inbound_event().expect("peer closure"),
            InboundWatchEvent::PeerClosed
        );
    }

    #[test]
    fn truncated_payload_fails_without_waiting_for_the_io_timeout() {
        let directory = temporary_directory();
        fs::create_dir(&directory).expect("temporary runtime directory must create");
        fs::set_permissions(&directory, fs::Permissions::from_mode(0o700))
            .expect("runtime permissions must set");
        let socket = directory.join("engine.sock");
        let config = ServiceConfig::new("/models", &directory, &socket)
            .expect("temporary config must validate");
        let listener = UnixFrameListener::bind(&config).expect("socket must bind");
        let client_socket = socket.clone();
        let (release_client, hold_client) = mpsc::sync_channel(0);
        let client = thread::spawn(move || {
            let mut stream = UnixStream::connect(client_socket).expect("client must connect");
            let encoded = Frame::new(FrameKind::Capability, vec![1, 2, 3])
                .expect("frame must construct")
                .encode()
                .expect("frame must encode");
            stream.write_all(&encoded[..=FRAME_HEADER_BYTES]).expect("partial frame must write");
            stream.shutdown(Shutdown::Write).expect("client write side must close");
            hold_client.recv().expect("server must release client");
        });
        let mut connection = listener.accept().expect("server must accept");
        assert!(matches!(connection.read_inbound_frame(), Err(EngineError::ProtocolViolation(_))));
        release_client.send(()).expect("client must be released");
        client.join().expect("client must finish");
        drop(listener);
        fs::remove_dir(&directory).expect("temporary directory must remove");
    }

    fn temporary_directory() -> PathBuf {
        let sequence = NEXT_DIRECTORY.fetch_add(1, Ordering::Relaxed);
        PathBuf::from("/tmp")
            .join(format!("personalai-engine-transport-{}-{sequence}", std::process::id()))
    }
}
