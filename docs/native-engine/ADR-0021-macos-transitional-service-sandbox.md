# ADR-0021: Transitional macOS service sandbox

- Status: Accepted for development and operated qualification
- Date: 2026-08-12

## Decision

The macOS CPU service may be qualified under the checked-in Seatbelt profile.
It denies by default, permits model reads, confines writes to the configured
runtime root, and permits bind/connect only for the exact canonical
`/private/var/run/personalai-engine/engine.sock` Unix socket. It contains no IP,
DNS, discovery, generic inbound, or wildcard network grant.

The local proof derives an otherwise identical temporary profile containing one
exact temporary socket path. It proves that UDS communication succeeds and
that an already-listening loopback TCP endpoint remains unreachable. It never
contacts an external host. Static architecture tests prevent broad network
rules from entering the profile.

## Limitation

Apple marks `sandbox-exec` deprecated. This profile is therefore transitional
defense-in-depth, not the final macOS distribution boundary. Production release
still requires a signed hardened launcher or App Sandbox design, reviewed file
and Unix-socket access, dedicated identity and secret injection, and repeated
no-egress evidence on each supported macOS release. No successful local proof
changes that release gate.
