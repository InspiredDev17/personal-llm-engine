# Model Foundry Phase 0 fixture ledger

Review bases:

- Native Engine: integrated `agent/native-inference-engine` and Model Foundry
  admission histories
- Model Foundry: `d6ab7b99fa083a76688ac5ef4b32b2e31480118c`

The Native Engine test
`deterministic_signed_import_builds_exact_files` regenerates the synthetic
directory package and enforces this exact ledger:

| File | Bytes | SHA-384 |
| --- | ---: | --- |
| `hashes.aiph` | 528 | `24c0d89c9a04166200cda5fdd29647fed34d81a4b662ca195a0148182b840072b0859828f89d78003ddb5d44907d10a4` |
| `manifest.aipm` | 4,608 | `6a2a5349c18916a515b3afac4253ae5240e45d0e3fb808bf5a8093d7c45cc469b45fbdf6ad95f88e94f7d93a1d62941d` |
| `provenance.json` | 720 | `d55e94e28620461d21bd1103e695d2e4c6c62470984a9f2912d1fc566bafecdb11c5725a4bd5009441e224044094affc` |
| `signature.ed25519` | 96 | `6292cef765d385e5c796b9533a2bc98fca7300d668ac75558c5bb1e650044f6f695d888a3afeddf6e80dc6e714a394af` |
| `tensors-0000.aipt` | 9,904 | `b0ca20c39137666448ca362acc76abfdbc1f79fdde96729f852bda4a85d0cf8314b9791e49ce48d83c90fb510447a4cd` |
| `tokenizer.aiptok` | 2,501 | `c366ab2a065461ce28b6456c9d8391a9e83e69fb9938f82ec5dcf6f9246c30f84f0928d55d375e82e4c25be3163abce4` |

Current package identity is the `hashes.aiph` SHA-384 above. The non-production package
signer key ID is `51515151515151515151515151515151`; its public key is
`2012cb90ca60e8e5d8daf66e2272d2233e0486d557e8c66141ed8920177d7eb7`.

The canonical registry V2 vectors remain frozen in Model Foundry and still bind
the prior `5b9758...b329` package ledger. The engine deliberately rejects that
signed decision when presented with the current deterministic package. A new
Model Foundry registry decision must be produced through its offline approval
authority before registry-v2 activation can succeed; the engine does not carry
or reconstruct the registry private key. ADR-0020 defines this fail-closed
boundary.
