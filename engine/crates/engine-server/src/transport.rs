//! Isolated local Unix-domain-socket frame transport.

use crate::ServiceConfig;
use personalai_engine_protocol::{FRAME_HEADER_BYTES, Frame, MAX_FRAME_PAYLOAD_BYTES};
use personalai_engine_types::EngineError;
use std::fs;
use std::io::{Read, Write};
use std::net::Shutdown;
use std::os::unix::fs::{FileTypeExt, MetadataExt, PermissionsExt};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::PathBuf;
use std::time::Duration;

const IO_TIMEOUT: Duration = Duration::from_secs(30);

/// Bound local-only socket with inode-safe cleanup ownership.
#[derive(Debug)]
pub struct UnixFrameListener {
    listener: UnixListener,
    socket_path: PathBuf,
    device: u64,
    inode: u64,
}

impl UnixFrameListener {
    /// Binds the configured socket only inside an existing owner-private runtime directory.
    pub fn bind(config: &ServiceConfig) -> Result<Self, EngineError> {
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
        if fs::symlink_metadata(config.socket_path()).is_ok() {
            return Err(EngineError::invalid_input("configured engine socket path already exists"));
        }
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
        })
    }

    /// Accepts one local connection and installs finite I/O timeouts.
    pub fn accept(&self) -> Result<UnixFrameConnection, EngineError> {
        let (stream, _address) = self
            .listener
            .accept()
            .map_err(|_| EngineError::backend_failure("Unix socket accept failed"))?;
        configure_stream(&stream)?;
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
                configure_stream(&stream)?;
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
}

fn configure_stream(stream: &UnixStream) -> Result<(), EngineError> {
    stream
        .set_nonblocking(false)
        .map_err(|_| EngineError::backend_failure("Unix stream blocking mode setup failed"))?;
    stream
        .set_read_timeout(Some(IO_TIMEOUT))
        .map_err(|_| EngineError::backend_failure("Unix socket read timeout setup failed"))?;
    stream
        .set_write_timeout(Some(IO_TIMEOUT))
        .map_err(|_| EngineError::backend_failure("Unix socket write timeout setup failed"))
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
        self.stream
            .shutdown(Shutdown::Read)
            .map_err(|_| EngineError::backend_failure("Unix stream read shutdown failed"))
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
