# Qwen2.5 Tokenizer Conformance Vectors

## Status and provenance

These are independently derived vectors for the pinned 7,031,645-byte
`tokenizer.json` whose SHA-256 is
`c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539`.
They remain non-activating evidence; the current public runtime cannot tokenize
Qwen input. This revision executes the scanner-boundary vectors against
trusted synthetic Unicode classifications and exercises BPE against compact
first-party tables. A private adapter replays the normalization, category, and
contraction-boundary vectors against the admitted Unicode-16 data. A
feature-gated content-free offline diagnostic now also replays all listed
production token IDs, full decode, and incremental decode against the exact
canonical conversion. The production table is not committed or returned by
that diagnostic.

| Input notation | Fixed scanner chunks | Expected token IDs |
| --- | --- | --- |
| `Hello world` | `Hello`, ` world` | `9707, 1879` |
| `I'm WE'RE he'd` | `I`, `'m`, ` WE`, `'RE`, ` he`, `'d` | `40, 2776, 19677, 94153, 566, 4172` |
| `abc123Ⅷ½` | `abc`, `1`, `2`, `3`, `Ⅷ`, `½` | `13683, 16, 17, 18, 70467, 100, 26062` |
| `Hello, world!!!\n` | `Hello`, `,`, ` world`, `!!!\n` | `9707, 11, 1879, 79295` |
| `A\r\nB\n\n` | `A`, `\r\n`, `B`, `\n\n` | `32, 319, 33, 271` |
| `A  B ` | `A`, ` `, ` B`, ` ` | `32, 220, 425, 220` |
| `\t\r\n` | `\t\r\n` | `4474` |
| `Cafe\u{0301}` after NFC | `Café` | `34, 2577, 963` |
| `\u{1100}\u{1161}` after NFC | `가` | `19969` |
| `\u{0301}b` | `\u{0301}b` | `53839, 65` |
| `\0A` | `\0A` | `188, 32` |
| `A中Ж` | `A中Ж` | `32, 15946, 132042` |
| ` !?` | ` !?` | `753, 30` |
| ` 😀\n` | ` 😀\n` | `90316, 198` |
| `NBSP + !` | `NBSP`, `!` | `4102, 0` |
| `ZWSP + !` | `ZWSP + !` | `15692, 0` |

The exact special-token behavior targets are:

- `Hello` produces `9707` with no implicit BOS.
- `<|im_start|>` is token `151644` only under an explicit allow policy; reject
  returns a typed error; ordinary-byte treatment splits it as `<|`, `im`,
  `_start`, `|>` and produces `27, 91, 318, 4906, 91, 29`.
- `<tool_call>` is the ordinary added token `151657`; diagnostic byte treatment
  produces `27, 14172, 13429, 29`.
- model-output IDs `151665..=151935` have no tokenizer entry and must fail
  decode or be excluded by the reviewed sampling domain.

## Unicode-version boundaries

The profile requires Unicode 16.0 as one coherent normalization/category
authority. U+088F and U+11DE0 are unassigned under that profile. The reviewed
normalization-order vector
`0061 1ACF 0315 0300 05AE 0062` becomes
`0061 1ACF 05AE 0300 0315 0062` under Unicode 16 behavior. Unicode 17 moves
U+1ACF because its assigned combining class changed, so a Unicode-17 NFC table
combined with Unicode-16 categories is explicitly incompatible.

The corresponding pinned-table ID target is:

```text
64, 157, 104, 237, 146, 106, 96549, 136, 243, 65
```

The private production adapter now replays this boundary before any activation.
It remains nonconstructible outside its module and does not by itself establish
the pinned production table or tokenizer parity.
