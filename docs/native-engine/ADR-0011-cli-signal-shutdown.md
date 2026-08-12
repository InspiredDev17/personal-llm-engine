# ADR-0011: Explicit continuous CLI mode and process-signal shutdown

- Status: Accepted
- Date: 2026-08-11

## Context

ADR-0010 proves bounded continuous admission and graceful drain behind a
caller-owned atomic shutdown signal. The owner CLI still selects the older
single-connection loop because it has no reviewed way to translate an external
process-manager stop into that signal. Replacing `serve` implicitly would also
change established behavior without an explicit owner resource configuration.

The shutdown bridge must not add a network path, wire command, model authority,
unbounded worker, third-party dependency, or signal-handler operation that is
unsafe in asynchronous signal context.

## Decision

Keep `serve` unchanged and add an explicit `serve-continuous` command. It
requires both `--max-active-sequences` and `--max-connections`; neither may be
zero, both retain central hard limits, and active sequences cannot exceed total
owned connections. The configured active limit also becomes the scheduler and
load-budget concurrency ceiling.

On Unix, the CLI installs process-wide `SIGINT` and `SIGTERM` actions before it
enters continuous serving. The signal handler performs exactly one operation: a
release store to a static `AtomicBool`. A single bounded-stack watcher polls
that flag and invokes the existing safe `ContinuousShutdown::request`. The
continuous server then stops admission, interrupts incomplete authentication,
cancels queued and active work, drains signed terminal events, joins all of its
workers, restores listener mode, and returns its bounded summary. The CLI stops
and joins the watcher and restores the exact prior process actions before it
returns.

The OS calls are isolated in `engine-cli/src/os_signal.rs`. Workspace unsafe is
locally allowed only in that module. One documented unsafe block initializes and
installs both actions; a second restores them. The non-cloneable guard prevents
overlapping installations in one process. Partial installation rolls back the
first action and fails closed. No raw pointer or signal number is exposed by the
safe wrapper.

The existing direct `libc` version is reused with default features disabled.
There is no new resolved package, build script, dynamic loading, socket type,
network-client capability, or wire-protocol change.

## Invariants

- Unix process permissions decide which local principals may signal the engine;
  no model or protocol request can select, suppress, or synthesize shutdown.
- Signal context performs no allocation, lock, I/O, logging, pointer access, or
  engine-state mutation beyond the atomic flag.
- Only one handler guard can be live in a process.
- A failed install prevents continuous serving; a failed restore is reported
  and keeps another in-process installation fail-closed.
- The watcher has one fixed 64 KiB stack, is stopped and joined on every normal
  or server-error return, and owns no socket or secret.
- Connection and active-sequence limits are explicit and checked before model
  or listener startup.
- `serve` remains the backward-compatible single-worker mode.
- Go retains all routing, policy, approval, tool, memory, and Agent Green
  authority.

## Evidence

Unit tests reject missing, duplicate, malformed, and inconsistent continuous
limits. A direct bridge test proves the handler flag requests the existing
graceful-shutdown type. A bounded child-process test installs the real actions,
receives an OS `SIGTERM`, observes it through the bridge, restores the prior
actions, and exits normally. Architecture tests require exactly two unsafe
blocks and one unsafe-lint allowance in the signal module while rejecting those
markers everywhere else except the separately approved replay lock module.

## Consequences

The owner can now select continuous local serving and receive ordinary service
manager termination without abandoning authenticated streams or scheduler
state. This is reference-runtime activation, not production qualification.
Operated slow-reader/multi-client soak, production key injection and recovery,
package conversion, performance evidence, canary, and rollback work remain.

ADR-0012 subsequently adds scheduled bounded synthetic delayed-reader and
restart soak. ADR-0013 adds same-service-user kernel peer admission.
Production-size operated qualification remains distinct.

## Rejected alternatives

- **Model or client shutdown command:** expands the wire protocol and grants a
  remote peer process-lifetime influence that this slice does not require.
- **Second shutdown socket or arbitrary path:** adds an unauthenticated local
  surface and filesystem configuration outside the existing service boundary.
- **Unbounded detached stdin watcher:** can outlive failed startup and couples
  secret provisioning EOF to process lifetime.
- **Third-party signal crate:** adds resolved supply-chain surface for two small
  audited OS calls already available through the approved `libc` dependency.
- **Implicitly replace `serve`:** removes the owner's reversible selection and
  changes concurrency semantics for existing launch configuration.
