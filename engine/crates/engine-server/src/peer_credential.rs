//! Isolated Unix peer-credential inspection for local transport admission.

#![allow(unsafe_code)]

use personalai_engine_types::EngineError;
use std::os::fd::AsRawFd;
use std::os::unix::net::UnixStream;

/// Returns the effective user ID that owns this engine process.
pub(crate) fn effective_user_id() -> u32 {
    // SAFETY: `geteuid` takes no arguments, retains no state, and always
    // returns the calling process's effective user ID.
    unsafe { libc::geteuid() }
}

/// Returns the kernel-authenticated effective user ID of one connected peer.
#[cfg(any(target_os = "linux", target_os = "android"))]
pub(crate) fn peer_effective_user_id(stream: &UnixStream) -> Result<u32, EngineError> {
    let mut credentials = std::mem::MaybeUninit::<libc::ucred>::uninit();
    let mut length = libc::socklen_t::try_from(std::mem::size_of::<libc::ucred>())
        .map_err(|_| EngineError::internal_invariant("peer credential size is not addressable"))?;
    // SAFETY: the descriptor is borrowed from a live connected `UnixStream`.
    // The output points to correctly aligned `ucred` storage and `length`
    // starts at its exact size. `getsockopt` retains neither pointer.
    let result = unsafe {
        libc::getsockopt(
            stream.as_raw_fd(),
            libc::SOL_SOCKET,
            libc::SO_PEERCRED,
            credentials.as_mut_ptr().cast(),
            &mut length,
        )
    };
    if result != 0 || usize::try_from(length).ok() != Some(std::mem::size_of::<libc::ucred>()) {
        return Err(EngineError::authentication_failure("Unix peer credentials are unavailable"));
    }
    // SAFETY: the successful kernel call reported the complete `ucred` size.
    let credentials = unsafe { credentials.assume_init() };
    Ok(credentials.uid)
}

/// Returns the kernel-authenticated effective user ID of one connected peer.
#[cfg(any(
    target_os = "macos",
    target_os = "ios",
    target_os = "freebsd",
    target_os = "dragonfly",
    target_os = "netbsd",
    target_os = "openbsd"
))]
pub(crate) fn peer_effective_user_id(stream: &UnixStream) -> Result<u32, EngineError> {
    let mut user_id: libc::uid_t = 0;
    let mut group_id: libc::gid_t = 0;
    // SAFETY: the descriptor is borrowed from a live connected `UnixStream`.
    // Both outputs point to initialized values of the exact ABI types and the
    // call retains neither pointer.
    let result =
        unsafe { libc::getpeereid(stream.as_raw_fd(), &raw mut user_id, &raw mut group_id) };
    if result != 0 {
        return Err(EngineError::authentication_failure("Unix peer credentials are unavailable"));
    }
    Ok(user_id)
}

/// Fails closed on Unix targets without a reviewed credential primitive.
#[cfg(not(any(
    target_os = "linux",
    target_os = "android",
    target_os = "macos",
    target_os = "ios",
    target_os = "freebsd",
    target_os = "dragonfly",
    target_os = "netbsd",
    target_os = "openbsd"
)))]
pub(crate) fn peer_effective_user_id(_stream: &UnixStream) -> Result<u32, EngineError> {
    Err(EngineError::not_implemented(
        "Unix peer credential admission is unsupported on this target",
    ))
}
