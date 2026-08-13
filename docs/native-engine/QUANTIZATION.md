# First-Party Quantization

## Boundary

Quantization is an offline storage transformation and a reviewed scalar CPU
execution path. It does not add an architecture, operation, kernel selector,
plugin, model code, network capability, or third-party runtime. The offline
operator must select exactly one profile: `reference-f32`, `balanced-q8`, or
`compact-q4`. The importer never chooses a profile implicitly. Version one
quantizes only rank-two architecture-owned weight matrices; normalization
vectors remain finite F32.

The package manifest declares one package-wide profile. Each tensor record is
either unquantized or uses exactly that profile. Mixing AIPQ8 and AIPQ4 in one
package, declaring a quantized profile with no quantized tensors, or declaring
quantized storage with a logical dtype other than F32 is rejected.

## Common tensor payload

Every quantized tensor is one complete, self-describing payload aligned to 64
bytes in its signed shard. The 32-byte header is:

| Offset | Width | Meaning |
| --- | ---: | --- |
| 0 | 8 | `AIPQ8V1\0` or `AIPQ4V1\0` |
| 8 | 2 | little-endian version `1` |
| 10 | 2 | little-endian block value count `32` |
| 12 | 1 | scale type `1`, little-endian IEEE binary32 |
| 13 | 1 | `0` for Q8; `1` for Q4 low-nibble-first |
| 14 | 2 | reserved zero |
| 16 | 8 | exact little-endian logical element count |
| 24 | 8 | exact little-endian block count |

There is no compression, variable integer, native-endian field, recursion, or
optional behavior. The parser checks the exact payload length before reading a
block and rejects trailing data. Tail values are encoded as zero and verified
as zero. NaN and either infinity are rejected before conversion. A block whose
source values are all zero has positive-zero scale and only zero codes.

## AIPQ8 v1

Each 36-byte block is a four-byte little-endian F32 scale followed by 32 signed
bytes. Conversion computes `scale = max(abs(value)) / 127`, rounds each quotient
to nearest with halfway cases away from zero using Rust's defined F32 `round`,
then clamps to `[-127, 127]`. Encoding `-128` is reserved and rejected. A
nonzero canonical block must contain at least one code with absolute value 127;
this binds the scale to the deterministic encoder. If division would underflow
the scale to zero for a subnormal-only block, the scale is the maximum absolute
source value and the canonical maximum code is one; this avoids converting
finite nonzero input into an invalid zero-scale block.
If finite F32 scale multiplication rounds just beyond the representable F32
maximum during dequantization, the result saturates to signed `F32::MAX`; NaN
and every other non-finite result remain rejected.

## AIPQ4 v1

AIPQ4 was admitted only after the AIPQ8 parser, importer, loader, kernels, and
targeted lint/test gate passed. Each 20-byte block is a four-byte little-endian
F32 scale followed by 16 packed bytes. Even logical indices occupy the low
nibble and odd indices the high nibble. Codes `0..=7` mean `0..=7`; codes
`8..=14` mean `-7..=-1`; code `15` is reserved and rejected. Conversion uses
`scale = max(abs(value)) / 7`, the same rounding rule, and clamp `[-7, 7]`.
The same subnormal scale-underflow rule applies. Zero is uniquely code zero.
Unused tail nibbles must be zero.

## Integrity, loading, and execution

The source tensor SHA-384, quantized tensor SHA-384, format, maximum and mean
absolute error, root-mean-square error, maximum defined relative error, and
relative-value count are recorded per tensor in signed provenance. Signed
provenance also records per-layer maxima instead of reporting only an aggregate
average. The manifest tensor hash, shard hash, closed hash catalog, and Ed25519
package signature cover the exact quantized bytes.

The production loader validates the manifest profile, exact payload header,
block count, packed length, scale, canonical codes, and tail before exposing a
package. It maps signed shards read-only and non-executable. The current CPU
service materializes quantized weights into aggregate-accounted F32 tensors;
the quantization crate separately provides scalar matrix-vector and
matrix-matrix kernels that operate directly on validated AIPQ bytes. Those
kernels are the correctness reference, not a performance claim.

## Numerical evidence and limitations

The deterministic two-layer synthetic fixture compares the same prompt and
weights under F32, AIPQ8, and AIPQ4. AIPQ8 observed maximum logit deviation
`2.775073051e-3`, mean absolute deviation `5.673158340e-4`, RMS deviation
`8.298536572e-4`, and zero divergent greedy tokens out of four. AIPQ4 observed
maximum logit deviation `3.539389372e-2`, mean absolute deviation
`1.445611193e-2`, RMS deviation `1.818871859e-2`, and four divergent greedy
tokens out of four.

These synthetic results prove deterministic mechanics only. In particular,
the AIPQ4 token divergence is poor on this fixture and must not be described as
quality parity. No real-model quality, perplexity, throughput, memory-residency,
or hardware-backend claim exists. Real-model activation still requires
independent per-layer/logit/generation evaluation and owner acceptance.

One bounded optimized local run on 2026-08-13 used a `128 x 128` matrix and 25
iterations per measurement. AIPQ8 averaged 161,523 ns to quantize, 983,823 ns
to dequantize into an accounted F32 tensor, and 130,431 ns for direct scalar
matrix-vector multiplication. AIPQ4 averaged 214,636 ns, 962,750 ns, and
138,326 ns respectively. These wall-clock numbers are a methodology record for
this run, not a portable throughput claim; the benchmark has no warmup,
statistical confidence interval, production-size matrix, or hardware pinning.
