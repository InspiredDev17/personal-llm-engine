# macOS local-service sandbox

`personalai-engine.sb` is a transitional Seatbelt profile for the owner-run
native engine. It denies by default, grants read access required by the signed
binary and read-only model store, confines writes to the configured runtime
directory, and permits networking only for the configured Unix socket. There
is no IP, DNS, service-discovery, or arbitrary Unix-socket rule.

Run `verify-sandbox.sh` from the repository root on every supported macOS
release. The proof creates only temporary local listeners: a configured Unix
socket must work and a live loopback TCP listener must remain unreachable from
the sandbox. This is deterministic isolation testing and contacts no external
host.

Apple labels `sandbox-exec` deprecated. Accordingly, this profile is
defense-in-depth for development and operated qualification, not the final
distribution mechanism. Production macOS release still requires a signed,
hardened launcher/App Sandbox entitlement review, a dedicated service identity,
keychain or launchd credential injection, and repeated no-egress proof. Do not
add client/server network entitlements: the UDS must remain the only transport.
