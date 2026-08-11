//! Isolated Unix FFI for an exclusive nonblocking file lock.
//!
//! The safe wrapper owns the only file descriptor used for the lock and never
//! exposes or clones it. Closing that descriptor releases the advisory lock.

#![allow(unsafe_code)]

use std::fs::{File, Metadata};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::os::fd::AsRawFd;

/// An owned file protected by a lifetime-held exclusive advisory lock.
pub(super) struct ExclusiveFile {
    file: File,
}

impl ExclusiveFile {
    /// Acquires an exclusive lock without waiting for another runtime.
    pub(super) fn try_acquire(file: File) -> io::Result<Self> {
        // SAFETY: `as_raw_fd` is valid for the duration of this call because
        // `file` is owned here. `flock` receives no pointer, retains no Rust
        // reference, and the return value is checked before the safe wrapper is
        // constructed. The wrapper never exposes or clones the descriptor.
        let result = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
        if result == 0 { Ok(Self { file }) } else { Err(io::Error::last_os_error()) }
    }

    pub(super) fn metadata(&self) -> io::Result<Metadata> {
        self.file.metadata()
    }

    pub(super) fn seek(&mut self, position: SeekFrom) -> io::Result<u64> {
        self.file.seek(position)
    }

    pub(super) fn read_exact(&mut self, bytes: &mut [u8]) -> io::Result<()> {
        self.file.read_exact(bytes)
    }

    pub(super) fn write_all(&mut self, bytes: &[u8]) -> io::Result<()> {
        self.file.write_all(bytes)
    }

    pub(super) fn sync_data(&self) -> io::Result<()> {
        self.file.sync_data()
    }

    pub(super) fn sync_all(&self) -> io::Result<()> {
        self.file.sync_all()
    }
}
