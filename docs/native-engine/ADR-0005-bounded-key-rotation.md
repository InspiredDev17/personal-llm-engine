# ADR-0005: Bounded Native-Engine Key Rotation

- Status: Accepted
- Date: 2026-08-11

## Context

The local protocol has separate Ed25519 identities: Go signs control commands
and Rust signs response events. Replacing either identity in one instantaneous
configuration change would require coordinated downtime and would make safe
rollback difficult. Accepting an unbounded key set, discovering keys at runtime,
or sharing private keys across the boundary would expand trust unnecessarily.

Control-key replacement must also preserve durable replay history. Reusing a
fresh journal for a previously accepted identity would make old signed nonces
eligible again.

## Decision

Permit an explicit two-entry current/previous overlap window in each verifier:

- Rust accepts at most two locally configured Go control public keys. It
  rejects duplicate key IDs and public keys, requires an active identity, binds
  one connection stream to the first authenticated key, and opens one
  exclusively locked durable replay journal per active key.
- Go signs commands with exactly one current control private key. A control-key
  cutover starts Rust with the desired current public key plus the previous
  public key, switches Go to the desired private key, drains old streams, and
  then removes the previous public key.
- Rust signs responses with exactly one current response private key supplied
  through stdin. Go accepts an explicit current/previous response public-key
  pair. A response-key cutover configures Go with the desired key as current and
  the deployed key as previous, restarts Rust with the desired private key,
  drains old streams, and then removes the previous public key.
- Removal from active configuration never deletes a replay journal. Re-adding a
  retired control key ID therefore reopens its existing journal and retains its
  nonce history. Journal archival or destruction is a separate operator action
  that is prohibited while any corresponding signed command could remain
  valid.

The wire protocol does not change. Key IDs already exist in signed request,
cancellation, and event bindings. There is no key negotiation, fallback to an
unknown key, network lookup, filesystem discovery, or model-selected identity.

## Invariants

1. Control and response identities remain distinct by both key ID and public
   key material, including overlap identities.
2. A command stream cannot switch control keys between generation and
   cancellation.
3. Each accepted control identity has a separate, bounded, mode-0600,
   exclusively owned replay journal.
4. Rust holds no Go private key. Go holds no Rust response private key.
5. At most two verification identities are accepted in either direction.
6. Unknown, duplicate, partially configured, revoked, or invalid identities
   fail before command execution or event parsing.
7. Rotation configuration comes only from the owner-controlled process
   configuration; clients and model packages cannot add keys.

## Consequences

The engine can rotate both mutual-authentication identities without a wire
change or trust discovery. Rollback during the bounded overlap is possible, and
control-key rollback does not reset replay state. The mechanism does not prove
production secret injection, backup/restore, crash-dump policy, journal archival,
or operated rollback. Those require deployment evidence before `SECRET` routing
or mandatory production use.
