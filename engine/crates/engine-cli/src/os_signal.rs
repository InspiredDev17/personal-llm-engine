//! Isolated Unix process-signal bridge for graceful owner shutdown.

#![allow(unsafe_code)]

use personalai_engine_types::EngineError;
use std::mem::MaybeUninit;
use std::sync::atomic::{AtomicBool, Ordering};

static INSTALLED: AtomicBool = AtomicBool::new(false);
static SHUTDOWN_REQUESTED: AtomicBool = AtomicBool::new(false);

/// Owned installation of the process-wide interrupt and termination handlers.
pub(crate) struct ProcessSignalGuard {
    previous_interrupt: libc::sigaction,
    previous_terminate: libc::sigaction,
    active: bool,
}

impl ProcessSignalGuard {
    /// Installs handlers that perform only one lock-free atomic store.
    pub(crate) fn install() -> Result<Self, EngineError> {
        if INSTALLED.compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire).is_err() {
            return Err(EngineError::backend_failure(
                "process shutdown handlers are already installed",
            ));
        }
        SHUTDOWN_REQUESTED.store(false, Ordering::Release);

        // SAFETY: `sigaction` is initialized before the kernel reads it, its
        // mask is initialized by `sigemptyset`, and both previous-action output
        // pointers refer to live `MaybeUninit` storage. The installed handler
        // has C ABI, retains no pointer, and performs only an atomic store.
        let installed = unsafe {
            let mut action: libc::sigaction = std::mem::zeroed();
            action.sa_sigaction = record_shutdown as libc::sighandler_t;
            action.sa_flags = libc::SA_RESTART;
            if libc::sigemptyset(&mut action.sa_mask) != 0 {
                None
            } else {
                let mut previous_interrupt = MaybeUninit::<libc::sigaction>::uninit();
                let mut previous_terminate = MaybeUninit::<libc::sigaction>::uninit();
                if libc::sigaction(libc::SIGINT, &action, previous_interrupt.as_mut_ptr()) != 0 {
                    None
                } else if libc::sigaction(libc::SIGTERM, &action, previous_terminate.as_mut_ptr())
                    != 0
                {
                    let _restore_result = libc::sigaction(
                        libc::SIGINT,
                        previous_interrupt.as_ptr(),
                        std::ptr::null_mut(),
                    );
                    None
                } else {
                    Some((previous_interrupt.assume_init(), previous_terminate.assume_init()))
                }
            }
        };

        let Some((previous_interrupt, previous_terminate)) = installed else {
            SHUTDOWN_REQUESTED.store(false, Ordering::Release);
            INSTALLED.store(false, Ordering::Release);
            return Err(EngineError::backend_failure(
                "process shutdown handler installation failed",
            ));
        };
        Ok(Self { previous_interrupt, previous_terminate, active: true })
    }

    /// Returns whether the process manager requested graceful shutdown.
    pub(crate) fn shutdown_requested() -> bool {
        SHUTDOWN_REQUESTED.load(Ordering::Acquire)
    }

    /// Restores the exact handlers that preceded this installation.
    pub(crate) fn restore(mut self) -> Result<(), EngineError> {
        self.restore_inner()
    }

    fn restore_inner(&mut self) -> Result<(), EngineError> {
        if !self.active {
            return Ok(());
        }
        // SAFETY: both actions were initialized by successful `sigaction`
        // calls and remain owned by this non-cloneable guard. Null output
        // pointers request restoration without exposing kernel-written data.
        let restored = unsafe {
            let interrupt =
                libc::sigaction(libc::SIGINT, &self.previous_interrupt, std::ptr::null_mut());
            let terminate =
                libc::sigaction(libc::SIGTERM, &self.previous_terminate, std::ptr::null_mut());
            interrupt == 0 && terminate == 0
        };
        self.active = false;
        SHUTDOWN_REQUESTED.store(false, Ordering::Release);
        if restored {
            INSTALLED.store(false, Ordering::Release);
            Ok(())
        } else {
            Err(EngineError::backend_failure("process shutdown handler restoration failed"))
        }
    }
}

impl Drop for ProcessSignalGuard {
    fn drop(&mut self) {
        let _restore_result = self.restore_inner();
    }
}

extern "C" fn record_shutdown(_signal: libc::c_int) {
    SHUTDOWN_REQUESTED.store(true, Ordering::Release);
}

#[cfg(test)]
pub(crate) fn request_shutdown_for_test() {
    record_shutdown(0);
}
