# Model Foundry Phase 0 fixture ledger

Review bases:

- Native Engine: `f458c46cd7cd69e73fddd575669604f99373acce`
- Model Foundry: `d6ab7b99fa083a76688ac5ef4b32b2e31480118c`

The Native Engine test
`deterministic_signed_import_builds_exact_files` regenerates the synthetic
directory package and enforces this exact ledger:

| File | Bytes | SHA-384 |
| --- | ---: | --- |
| `hashes.aiph` | 528 | `5b97582279ce8e9dc1ed114b0b46b243fd1f6655f9ed694def28c1e52cbfeffdf1eb5e27f19fab5c7529b91d6471b329` |
| `manifest.aipm` | 2,880 | `6e417bdfd2682c1e0c06a324670797b462089b2e0bff06f20615b543497f664d8509e0a8cb4e7f5ed7049cd8b3c42115` |
| `provenance.json` | 471 | `b10362b19f93ea9ecfbc721af0635835ecc319ca71134608eca7dcff047c480d37e2349b53681b317654f3130cf4bdd5` |
| `signature.ed25519` | 96 | `1fd392c088ac1c2a22d18307f0f909743b76a27595cbdb9e12e8f99a58dbe31f76e883d726ec8a6347f44704621fb3b0` |
| `tensors-0000.aipt` | 9,120 | `3f283f50f5c5937ad8625da6c692c31bdf5833733589520842ea946921c17eb2842aa31c2b09c0310baec907d3ff1d72` |
| `tokenizer.aiptok` | 1,328 | `1c6de2b8090bc7b45683292075d18ea846361269e57f98d2d158eb38731eb8c2a342e58eeb15d6632520fd01ac40d194` |

Package identity is the `hashes.aiph` SHA-384 above. The non-production package
signer key ID is `51515151515151515151515151515151`; its public key is
`2012cb90ca60e8e5d8daf66e2272d2233e0486d557e8c66141ed8920177d7eb7`.

The canonical registry V2 vectors and negative decisions are frozen in Model
Foundry. This engine intentionally does not copy a parser or claim acceptance in
Phase 0. ADR-0020 defines the future boundary.
