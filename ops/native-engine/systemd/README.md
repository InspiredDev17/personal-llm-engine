# Linux systemd sandbox profile

This reviewed unit is the first operated-service template for the native
inference data plane. It denies IP networking twice: `RestrictAddressFamilies`
permits only Unix-domain sockets and `IPAddressDeny=any` denies every IP
address. The service has no ambient or bounding capabilities, cannot gain
privileges, cannot write the model store, and can write only its protected
runtime directory.

The operator must create the `personalai` service identity, mode-0755 root-owned
model directory, root-owned mode-0600 configuration/signing files, install the
OS-authenticated verifier at `/usr/libexec/personalai-engine-release`, and
install the closed signed release beneath `/usr/lib/personalai-engine/release`.
The root verifier unit runs before every engine start, advances state under
`/var/lib/personalai-engine/release-state`, and retains content-free failures
under `/var/lib/personalai-engine/release-events`. Both state directories are
root-owned; only event records are group-readable. systemd creates the mode-0700 runtime directory.
The verifier's active marker is `PartOf` the engine service, so stopping or
restarting the engine clears prior success and forces a new verification.
`engine.env` contains public identifiers,
public verification keys, fixed package registration, and numeric budgets. It
must never contain the response signing seed. systemd presents the exact
32-byte seed from its credential directory on standard input; the engine clears
its copy after signer construction.

`release.env` contains only the release signer key ID and public key. The
verifier receives no private key and has no network address family. The engine
unit has no write access to the release, ledger, or event directories.

The template also sets a nonexistent home, removes common cloud/proxy secret
variables, disables core dumps and swap, restricts namespaces, and caps open
files at 1,024, processes/tasks at 128, and resident memory at 16 GiB. Review
and lower the memory ceiling for the selected package and host. Raising any
limit requires recorded capacity and denial-of-service review.

Before enabling the unit, substitute a legally approved registered package and
run `systemd-analyze security personalai-engine.service`. Operated proof must
also show that generation succeeds over `/run/personalai-engine/engine.sock`
while an attempted IPv4 and IPv6 connection from the service identity is
denied. Static repository tests verify the mandatory directives but are not a
substitute for that host-level test. macOS requires a separately reviewed
launchd plus application-sandbox design; this Linux unit makes no macOS claim.
