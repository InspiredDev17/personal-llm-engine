# Linux systemd sandbox profile

This reviewed unit is the first operated-service template for the native
inference data plane. It denies IP networking twice: `RestrictAddressFamilies`
permits only Unix-domain sockets and `IPAddressDeny=any` denies every IP
address. The service has no ambient or bounding capabilities, cannot gain
privileges, cannot write the model store, and can write only its protected
runtime directory.

The operator must create the `personalai` service identity, mode-0755 root-owned
model directory, root-owned mode-0600
configuration/signing files, and install the audited release binary at
`/usr/libexec/personalai-engine`. systemd creates the mode-0700 runtime directory.
`engine.env` contains public identifiers,
public verification keys, fixed package registration, and numeric budgets. It
must never contain the response signing seed. systemd presents the exact
32-byte seed from its credential directory on standard input; the engine clears
its copy after signer construction.

Before enabling the unit, substitute a legally approved registered package and
run `systemd-analyze security personalai-engine.service`. Operated proof must
also show that generation succeeds over `/run/personalai-engine/engine.sock`
while an attempted IPv4 and IPv6 connection from the service identity is
denied. Static repository tests verify the mandatory directives but are not a
substitute for that host-level test. macOS requires a separately reviewed
launchd plus application-sandbox design; this Linux unit makes no macOS claim.
