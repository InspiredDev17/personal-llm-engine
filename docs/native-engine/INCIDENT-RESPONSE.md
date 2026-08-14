# Native Engine Incident Response

## Immediate containment

1. Stop new Go routing to the affected engine. Do not convert failure into
   cloud fallback unless a new explicit cloud authorization is approved.
2. Ask the supervisor for graceful shutdown; if cleanup cannot complete within
   its fixed timeout, terminate the unprivileged service without executing any
   model/package-provided command.
3. Preserve content-free structured events, exact release/model hashes,
   revision/toolchain/OS versions, signer key IDs, replay-journal metadata, and
   supervisor status. Do not preserve prompts, generated content, raw memory,
   private keys, or model tensors in ordinary incident logs.
4. Make the model store and candidate release read-only, revoke affected
   signer/control/response keys where exposure is plausible, and quarantine
   affected package IDs in owner-controlled configuration.

## Classification and response

| Signal | Required response |
| --- | --- |
| signature, hash, signer, path, or unexpected-file failure | reject/unload package; quarantine ID; preserve catalog/signature hashes; investigate source/store mutation |
| unauthorized/repeated authentication | close peer; rotate control key if compromise is plausible; inspect same-user/service-account isolation |
| malformed/oversized/version/replay pattern | keep request rejected; rate-limit at Go/supervisor boundary; preserve category/count only |
| resource exhaustion or GPU/backend failure | cancel affected request; verify cleanup/accounting; unload only after ownership drains; restart under bounded supervisor policy if necessary |
| engine/metallib mismatch or downgrade | the supervisor verifier keeps the engine stopped and retains a content-free ring event; preserve the release root and ledger, then follow owner-authorized recovery without weakening rollback protection |
| crash loop | disable restart after the configured threshold; retain supervisor evidence; do not expose prompts in crash reports or enable core dumps |
| suspected cross-request/KV disclosure | stop service, revoke response/control credentials, preserve only minimal metadata, and treat all requests in the affected process lifetime as potentially exposed |
| prompt/tool injection | treat output as untrusted; do not execute; preserve Go approval/Action Broker evidence and verify no control-plane bypass occurred |
| Agent Green delivery failure | retain bounded engine events in Go's durable path; model output cannot acknowledge or suppress them; restore the separate authenticated liaison |

## Recovery gate

Recovery requires an exact root cause, a deterministic regression test when
applicable, fresh package/release verification, key rotation where applicable,
successful full Rust/Go/security gates, memory/KV cleanup proof, no-egress proof,
and explicit owner approval of the exact candidate. Re-enabling one model does
not authorize every registered model. Do not delete evidence until the owner
retention policy permits it.

## Communication boundary

Security events flow Rust to Go as bounded types. Go owns audit and any
allowlisted authenticated Agent Green delivery. The engine and model cannot
contact Agent Green, change policy, suppress an event, approve a tool, modify
memory, or select an incident action.
