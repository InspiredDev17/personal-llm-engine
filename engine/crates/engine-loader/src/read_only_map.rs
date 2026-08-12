//! Isolated read-only file mapping for verified tensor shards.

#![allow(unsafe_code)]

use personalai_engine_types::EngineError;
use std::fs::File;
use std::os::fd::AsRawFd;
use std::ptr::NonNull;

/// Owned private read-only mapping backed by one securely opened file.
#[derive(Debug)]
pub(crate) struct ReadOnlyFileMap {
    pointer: NonNull<u8>,
    length: usize,
    _file: File,
}

impl ReadOnlyFileMap {
    /// Maps the complete nonempty file without executable permission.
    pub(crate) fn new(file: File, length: usize) -> Result<Self, EngineError> {
        if length == 0 {
            return Err(EngineError::corrupt_model_package("tensor shard must not be empty"));
        }
        // SAFETY: the descriptor belongs to a live read-only regular `File`;
        // length is its exact checked nonzero size; offset zero is page aligned.
        // `PROT_READ` excludes writes and execution, and `MAP_PRIVATE` prevents
        // the mapping from mutating the package file.
        let pointer = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                length,
                libc::PROT_READ,
                libc::MAP_PRIVATE,
                file.as_raw_fd(),
                0,
            )
        };
        if pointer == libc::MAP_FAILED {
            return Err(EngineError::backend_failure("read-only tensor shard mapping failed"));
        }
        let pointer = NonNull::new(pointer.cast::<u8>())
            .ok_or_else(|| EngineError::internal_invariant("mmap returned a null mapping"))?;
        Ok(Self { pointer, length, _file: file })
    }

    /// Returns the immutable mapped file bytes.
    pub(crate) fn bytes(&self) -> &[u8] {
        // SAFETY: the mapping remains live for `self`, is exactly `length`
        // bytes, and was created read-only. No mutable slice is exposed.
        unsafe { std::slice::from_raw_parts(self.pointer.as_ptr(), self.length) }
    }
}

impl Drop for ReadOnlyFileMap {
    fn drop(&mut self) {
        // SAFETY: this is the exact pointer/length returned by the successful
        // `mmap` call and this non-cloneable owner drops it exactly once.
        let _result = unsafe { libc::munmap(self.pointer.as_ptr().cast(), self.length) };
    }
}
