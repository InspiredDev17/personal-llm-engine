# ADR-0006: Always-offline Rust data plane

- Status: Accepted
- Date: 2026-08-11

## Context

The native engine handles prompts, model packages, generated tokens, and
security events. None of those data require Internet access. A configurable
"offline" flag inside Rust would be weaker than the charter because an invalid
configuration or future caller could disable it. Dependency-only review also
cannot independently contain a future source-level regression.

The Go control plane may eventually operate explicitly approved integrations,
but it alone owns provider registration, routing, fallback, and telemetry
delivery. Denying all Go networking would conflate inference confidentiality
with separately authorized platform integrations.

## Decision

The Rust data plane is always offline and exposes no switch to enable Internet
access. Its only communications transport is the authenticated protected
Unix-domain stream owned by `engine-server`.

The production Rust workspace has three reinforcing controls:

1. the exact dependency allowlist contains no network client or telemetry
   exporter;
2. architecture tests reject Internet socket APIs, raw socket FFI,
   asynchronous network modules, and subprocess-spawning egress paths in
   first-party production Rust sections; and
3. each production service definition must independently deny outbound
   networking at the operating-system boundary.

The approved Unix transport may reference `std::net::Shutdown` solely to close
one direction of a Unix stream. Explicit Internet socket markers remain denied
even in that module. Local audit and sanitized security telemetry are retained:
events travel to Go over the authenticated UDS. Rust has no cloud exporter and
no direct Agent Green client.

Go implements the separately reviewed platform-wide local-only mode. That mode
must reject remote primary, fallback, verifier, and manual-override model
destinations before invocation. It must not rely on a Rust command-line flag.

## Consequences

- A future network client, raw socket, or subprocess requirement is an
  architecture change requiring owner approval and a replacement ADR; it is
  not an ordinary dependency addition.
- Local UDS operation remains available under OS network confinement.
- The exact Linux, macOS, and accelerator-compatible service profiles are
  target-specific release artifacts and require operated tests on supported
  hardware.
- This ADR does not claim that the current host is OS-confined. Mandatory
  production routing remains blocked until the target service profile and
  no-egress evidence exist.

## Rejected alternatives

- **Rust `--offline` flag:** mutable configuration is not a security boundary.
- **Dependency review alone:** first-party standard-library or FFI code could
  still add egress.
- **Disable local audit/security events:** removes defensive visibility without
  improving Internet isolation.
- **Deny every Go network operation:** would break separately authorized tools
  and integrations instead of enforcing the narrower inference boundary.
