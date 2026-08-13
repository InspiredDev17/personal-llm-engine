//! Bounded first-party worker pool for ARM64 host kernels.

use personalai_engine_types::EngineError;
use std::fmt;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc::{Receiver, SyncSender, sync_channel};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};

/// Hard ceiling on persistent ARM64 workers in one backend instance.
pub const MAX_ARM64_WORKERS: usize = 16;

type JobResult = Result<(), EngineError>;
type JobFunction = Box<dyn FnOnce() -> JobResult + Send + 'static>;

enum Message {
    Run(JobFunction, SyncSender<JobResult>),
    Shutdown,
}

/// Cooperative request cancellation shared with bounded kernels.
#[derive(Clone, Debug, Default)]
pub struct CancellationToken(Arc<AtomicBool>);

impl CancellationToken {
    /// Creates a live token.
    pub fn new() -> Self {
        Self::default()
    }

    /// Requests cancellation. This operation is idempotent.
    pub fn cancel(&self) {
        self.0.store(true, Ordering::Release);
    }

    /// Returns whether cancellation has been requested.
    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::Acquire)
    }
}

/// Snapshot of bounded worker-pool resource and failure counters.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WorkerPoolMetrics {
    /// Fixed worker count.
    pub worker_count: usize,
    /// Jobs currently queued or executing.
    pub in_flight_jobs: usize,
    /// Peak simultaneous jobs.
    pub peak_in_flight_jobs: usize,
    /// Jobs for which a worker returned a completion, including isolated failures.
    pub completed_jobs: u64,
    /// Jobs that observed cooperative cancellation.
    pub cancelled_jobs: u64,
    /// Panics isolated at the worker boundary.
    pub isolated_panics: u64,
}

#[derive(Default)]
struct PoolCounters {
    in_flight: AtomicUsize,
    peak_in_flight: AtomicUsize,
    completed: AtomicU64,
    cancelled: AtomicU64,
    panics: AtomicU64,
}

/// Fixed-size pool with bounded ingress and deterministic shutdown.
pub(crate) struct WorkerPool {
    sender: SyncSender<Message>,
    handles: Mutex<Vec<JoinHandle<()>>>,
    counters: Arc<PoolCounters>,
    worker_count: usize,
}

impl fmt::Debug for WorkerPool {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.debug_struct("WorkerPool").field("metrics", &self.metrics()).finish()
    }
}

impl WorkerPool {
    pub(crate) fn new(worker_count: usize) -> Result<Self, EngineError> {
        if worker_count == 0 || worker_count > MAX_ARM64_WORKERS {
            return Err(EngineError::invalid_input("ARM64 worker count is outside its hard bound"));
        }
        let (sender, receiver) = sync_channel(worker_count);
        let receiver = Arc::new(Mutex::new(receiver));
        let counters = Arc::new(PoolCounters::default());
        let mut handles = Vec::new();
        handles.try_reserve_exact(worker_count).map_err(|_| {
            EngineError::resource_exhaustion("ARM64 worker handle allocation failed")
        })?;
        for index in 0..worker_count {
            let receiver = Arc::clone(&receiver);
            let counters = Arc::clone(&counters);
            let handle = thread::Builder::new()
                .name(format!("personalai-arm64-{index}"))
                .spawn(move || worker_loop(&receiver, &counters))
                .map_err(|_| EngineError::backend_failure("ARM64 worker creation failed"))?;
            handles.push(handle);
        }
        Ok(Self { sender, handles: Mutex::new(handles), counters, worker_count })
    }

    pub(crate) fn run(
        &self,
        jobs: Vec<JobFunction>,
        cancellation: &CancellationToken,
    ) -> Result<(), EngineError> {
        if jobs.len() > self.worker_count {
            return Err(EngineError::internal_invariant(
                "ARM64 work partition exceeds the fixed worker count",
            ));
        }
        let mut completions = Vec::new();
        completions
            .try_reserve_exact(jobs.len())
            .map_err(|_| EngineError::resource_exhaustion("ARM64 completion allocation failed"))?;
        for job in jobs {
            if cancellation.is_cancelled() {
                self.counters.cancelled.fetch_add(1, Ordering::Relaxed);
                return Err(EngineError::cancellation(None));
            }
            let (sender, receiver) = sync_channel(1);
            increment_in_flight(&self.counters);
            if self.sender.send(Message::Run(job, sender)).is_err() {
                self.counters.in_flight.fetch_sub(1, Ordering::AcqRel);
                return Err(EngineError::backend_failure("ARM64 worker pool is unavailable"));
            }
            completions.push(receiver);
        }
        let mut first_error = None;
        for completion in completions {
            match completion.recv() {
                Ok(Ok(())) => {}
                Ok(Err(error)) => {
                    if matches!(error, EngineError::Cancellation(_)) {
                        self.counters.cancelled.fetch_add(1, Ordering::Relaxed);
                    }
                    if first_error.is_none() {
                        first_error = Some(error);
                    }
                }
                Err(_) => {
                    if first_error.is_none() {
                        first_error = Some(EngineError::backend_failure(
                            "ARM64 worker completion channel closed",
                        ));
                    }
                }
            }
        }
        first_error.map_or(Ok(()), Err)
    }

    pub(crate) fn metrics(&self) -> WorkerPoolMetrics {
        WorkerPoolMetrics {
            worker_count: self.worker_count,
            in_flight_jobs: self.counters.in_flight.load(Ordering::Acquire),
            peak_in_flight_jobs: self.counters.peak_in_flight.load(Ordering::Acquire),
            completed_jobs: self.counters.completed.load(Ordering::Relaxed),
            cancelled_jobs: self.counters.cancelled.load(Ordering::Relaxed),
            isolated_panics: self.counters.panics.load(Ordering::Relaxed),
        }
    }
}

impl Drop for WorkerPool {
    fn drop(&mut self) {
        for _ in 0..self.worker_count {
            let _ignored = self.sender.send(Message::Shutdown);
        }
        let handles = match self.handles.get_mut() {
            Ok(handles) => handles,
            Err(poisoned) => poisoned.into_inner(),
        };
        for handle in handles.drain(..) {
            let _ignored = handle.join();
        }
    }
}

fn worker_loop(receiver: &Mutex<Receiver<Message>>, counters: &PoolCounters) {
    loop {
        let message = match receiver.lock() {
            Ok(receiver) => receiver.recv(),
            Err(poisoned) => poisoned.into_inner().recv(),
        };
        match message {
            Ok(Message::Run(job, completion)) => {
                let result = catch_unwind(AssertUnwindSafe(job));
                let result = if let Ok(result) = result {
                    result
                } else {
                    counters.panics.fetch_add(1, Ordering::Relaxed);
                    Err(EngineError::backend_failure("ARM64 worker panic was isolated"))
                };
                counters.completed.fetch_add(1, Ordering::Relaxed);
                counters.in_flight.fetch_sub(1, Ordering::AcqRel);
                let _ignored = completion.send(result);
            }
            Ok(Message::Shutdown) | Err(_) => break,
        }
    }
}

fn increment_in_flight(counters: &PoolCounters) {
    let current = counters.in_flight.fetch_add(1, Ordering::AcqRel) + 1;
    let mut peak = counters.peak_in_flight.load(Ordering::Acquire);
    while current > peak {
        match counters.peak_in_flight.compare_exchange_weak(
            peak,
            current,
            Ordering::AcqRel,
            Ordering::Acquire,
        ) {
            Ok(_) => break,
            Err(observed) => peak = observed,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pool_is_bounded_completes_and_shuts_down() {
        assert!(WorkerPool::new(0).is_err());
        assert!(WorkerPool::new(MAX_ARM64_WORKERS + 1).is_err());
        let pool = WorkerPool::new(2).unwrap();
        let count = Arc::new(AtomicUsize::new(0));
        let jobs = (0..2)
            .map(|_| {
                let count = Arc::clone(&count);
                Box::new(move || {
                    count.fetch_add(1, Ordering::Relaxed);
                    Ok(())
                }) as JobFunction
            })
            .collect();
        pool.run(jobs, &CancellationToken::new()).unwrap();
        assert_eq!(count.load(Ordering::Relaxed), 2);
        assert_eq!(pool.metrics().completed_jobs, 2);
        assert_eq!(pool.metrics().in_flight_jobs, 0);
    }

    #[test]
    fn cancellation_and_panics_are_isolated() {
        let pool = WorkerPool::new(1).unwrap();
        let token = CancellationToken::new();
        token.cancel();
        assert!(matches!(
            pool.run(vec![Box::new(|| Ok(()))], &token),
            Err(EngineError::Cancellation(_))
        ));
        let result = pool.run(
            vec![Box::new(|| -> JobResult { panic!("injected worker panic") })],
            &CancellationToken::new(),
        );
        assert!(matches!(result, Err(EngineError::BackendFailure(_))));
        assert_eq!(pool.metrics().isolated_panics, 1);
    }
}
