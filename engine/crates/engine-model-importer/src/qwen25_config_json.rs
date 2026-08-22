//! Exact configuration validator for the pinned Qwen2.5 1.5B source profiles.

use crate::bounded_json::JsonCursor;
use personalai_engine_crypto::{hash_package, hash_sha256};
use personalai_engine_types::{EngineError, ModelPackageHash, Sha256Digest};

/// Source configuration format identifier.
pub const QWEN25_CONFIG_JSON_SOURCE_FORMAT: &str = "qwen2.5-config-json-v1";
/// Exact byte length shared by the three reviewed model tensor sources.
pub const QWEN25_MODEL_SAFETENSORS_BYTES: u64 = 3_087_467_144;

const GENERAL_SHA256: &str = "98d2ff8cc47488d08a2b0b3acf4eb99ef210779b42bd48605f6b8e36acdbf670";
const MATH_SHA256: &str = "a48e6ab280ad87a25f06da604d486aa819b9c1bc467dae7c25fe5d05c57f5511";
const CODER_SHA256: &str = "88f9a17863c05fb313515d2ff74b1098e0c35579f99068e32beda00618508ae0";
const GENERAL_MODEL_SHA256: &str =
    "dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee";
const MATH_MODEL_SHA256: &str = "08faf1e950173bb44c11d26df28c55da7237dc55c77dbc1cbae4a5d7584540d9";
const CODER_MODEL_SHA256: &str = "c1b9b30e907950516ba3c646bdf570d8084c25a6410a0cdca80cf04b11bc13a8";

/// Exact pinned source configuration variant.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Qwen25ConfigProfile {
    /// Qwen2.5 1.5B Instruct.
    Instruct,
    /// Qwen2.5 Math 1.5B Instruct.
    MathInstruct,
    /// Qwen2.5 Coder 1.5B Instruct.
    CoderInstruct,
}

/// Validated execution facts derived from a pinned data-only configuration.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Qwen25Config {
    profile: Qwen25ConfigProfile,
    source_sha256: Sha256Digest,
    source_sha384: ModelPackageHash,
    model_source_sha256: Sha256Digest,
    context_limit: u32,
    rope_theta: f32,
}

impl Qwen25Config {
    /// Parses only one of the three exact pinned startup configurations.
    pub fn parse(bytes: &[u8]) -> Result<Self, EngineError> {
        if !matches!(bytes.len(), 656 | 660) {
            return Err(corrupt("Qwen2.5 configuration byte length is not pinned"));
        }
        let digest = hash_sha256(bytes);
        let profile = match (bytes.len(), digest.to_string().as_str()) {
            (660, GENERAL_SHA256) => Qwen25ConfigProfile::Instruct,
            (656, MATH_SHA256) => Qwen25ConfigProfile::MathInstruct,
            (660, CODER_SHA256) => Qwen25ConfigProfile::CoderInstruct,
            _ => return Err(corrupt("Qwen2.5 configuration identity is not pinned")),
        };
        let parsed = parse_graph(bytes)?;
        let (context_limit, rope_theta): (u32, f32) = match profile {
            Qwen25ConfigProfile::MathInstruct => (4_096, 10_000.0),
            Qwen25ConfigProfile::Instruct | Qwen25ConfigProfile::CoderInstruct => {
                (32_768, 1_000_000.0)
            }
        };
        let model_hash = match profile {
            Qwen25ConfigProfile::Instruct => GENERAL_MODEL_SHA256,
            Qwen25ConfigProfile::MathInstruct => MATH_MODEL_SHA256,
            Qwen25ConfigProfile::CoderInstruct => CODER_MODEL_SHA256,
        };
        if parsed.max_position_embeddings != context_limit
            || parsed.sliding_window != context_limit
            || parsed.rope_theta.to_bits() != f64::from(rope_theta).to_bits()
            || parsed.max_window_layers
                != if profile == Qwen25ConfigProfile::CoderInstruct { 28 } else { 21 }
        {
            return Err(corrupt("Qwen2.5 configuration profile fields do not match its identity"));
        }
        Ok(Self {
            profile,
            source_sha256: digest,
            source_sha384: hash_package(bytes),
            model_source_sha256: Sha256Digest::from_hex(model_hash).map_err(|_| {
                EngineError::internal_invariant("pinned Qwen2.5 model digest is invalid")
            })?,
            context_limit,
            rope_theta,
        })
    }

    /// Returns the exact source variant.
    pub const fn profile(self) -> Qwen25ConfigProfile {
        self.profile
    }

    /// Returns the source SHA-256 identity.
    pub const fn source_sha256(self) -> Sha256Digest {
        self.source_sha256
    }

    /// Returns the source SHA-384 provenance identity.
    pub const fn source_sha384(self) -> ModelPackageHash {
        self.source_sha384
    }

    /// Returns the exact reviewed `model.safetensors` SHA-256 identity.
    pub const fn model_source_sha256(self) -> Sha256Digest {
        self.model_source_sha256
    }

    /// Returns the exact reviewed `model.safetensors` byte length.
    pub const fn model_source_bytes(self) -> u64 {
        QWEN25_MODEL_SAFETENSORS_BYTES
    }

    /// Returns the accepted context limit.
    pub const fn context_limit(self) -> u32 {
        self.context_limit
    }

    /// Returns the accepted RoPE theta.
    pub const fn rope_theta(self) -> f32 {
        self.rope_theta
    }

    /// Returns the fixed hidden dimension.
    pub const fn hidden_dimension(self) -> u32 {
        1_536
    }

    /// Returns the fixed layer count.
    pub const fn layer_count(self) -> u32 {
        28
    }

    /// Returns the fixed attention-head count.
    pub const fn attention_head_count(self) -> u32 {
        12
    }

    /// Returns the fixed grouped-query key/value-head count.
    pub const fn kv_head_count(self) -> u32 {
        2
    }

    /// Returns the fixed SwiGLU intermediate dimension.
    pub const fn feed_forward_dimension(self) -> u32 {
        8_960
    }

    /// Returns the fixed RMS-normalization epsilon.
    pub const fn rms_norm_epsilon(self) -> f32 {
        1.0e-6
    }

    /// Returns the model embedding-row count.
    pub const fn vocabulary_size(self) -> u32 {
        151_936
    }
}

#[derive(Default)]
struct ParsedConfig {
    max_position_embeddings: u32,
    max_window_layers: u32,
    rope_theta: f64,
    sliding_window: u32,
}

#[allow(clippy::too_many_lines)]
fn parse_graph(bytes: &[u8]) -> Result<ParsedConfig, EngineError> {
    let mut cursor = JsonCursor::new(bytes)?;
    cursor.expect(b'{')?;
    let mut seen = 0_u32;
    let mut parsed = ParsedConfig::default();
    loop {
        let key = cursor.key(64)?;
        let field = match key.as_str() {
            "architectures" => {
                cursor.expect(b'[')?;
                exact_string(&mut cursor, "Qwen2ForCausalLM")?;
                cursor.expect(b']')?;
                0
            }
            "attention_dropout" => {
                exact_f64(&mut cursor, 0.0)?;
                1
            }
            "bos_token_id" => {
                exact_u32(&mut cursor, 151_643)?;
                2
            }
            "eos_token_id" => {
                exact_u32(&mut cursor, 151_645)?;
                3
            }
            "hidden_act" => {
                exact_string(&mut cursor, "silu")?;
                4
            }
            "hidden_size" => {
                exact_u32(&mut cursor, 1_536)?;
                5
            }
            "initializer_range" => {
                exact_f64(&mut cursor, 0.02)?;
                6
            }
            "intermediate_size" => {
                exact_u32(&mut cursor, 8_960)?;
                7
            }
            "max_position_embeddings" => {
                parsed.max_position_embeddings = cursor.u32()?;
                8
            }
            "max_window_layers" => {
                parsed.max_window_layers = cursor.u32()?;
                9
            }
            "model_type" => {
                exact_string(&mut cursor, "qwen2")?;
                10
            }
            "num_attention_heads" => {
                exact_u32(&mut cursor, 12)?;
                11
            }
            "num_hidden_layers" => {
                exact_u32(&mut cursor, 28)?;
                12
            }
            "num_key_value_heads" => {
                exact_u32(&mut cursor, 2)?;
                13
            }
            "rms_norm_eps" => {
                exact_f64(&mut cursor, 0.000_001)?;
                14
            }
            "rope_theta" => {
                parsed.rope_theta = cursor.f64()?;
                15
            }
            "sliding_window" => {
                parsed.sliding_window = cursor.u32()?;
                16
            }
            "tie_word_embeddings" => {
                exact_bool(&mut cursor, true)?;
                17
            }
            "torch_dtype" => {
                exact_string(&mut cursor, "bfloat16")?;
                18
            }
            "transformers_version" => {
                let version = cursor.string(16)?;
                if !matches!(version.as_str(), "4.43.1" | "4.44.0") {
                    return Err(corrupt("Qwen2.5 source tool version is unsupported"));
                }
                19
            }
            "use_cache" => {
                exact_bool(&mut cursor, true)?;
                20
            }
            "use_sliding_window" => {
                exact_bool(&mut cursor, false)?;
                21
            }
            "vocab_size" => {
                exact_u32(&mut cursor, 151_936)?;
                22
            }
            _ => return Err(corrupt("Qwen2.5 configuration field is unsupported")),
        };
        let mask = 1_u32 << field;
        if seen & mask != 0 {
            return Err(corrupt("Qwen2.5 configuration field is duplicated"));
        }
        seen |= mask;
        if cursor.consume(b'}') {
            break;
        }
        cursor.expect(b',')?;
    }
    cursor.finish()?;
    if seen != (1_u32 << 23) - 1
        || !matches!(parsed.max_position_embeddings, 4_096 | 32_768)
        || !matches!(parsed.max_window_layers, 21 | 28)
        || !matches!(parsed.rope_theta.to_bits(), value if value == 10_000_f64.to_bits() || value == 1_000_000_f64.to_bits())
        || parsed.sliding_window != parsed.max_position_embeddings
    {
        return Err(corrupt("Qwen2.5 configuration is incomplete or inconsistent"));
    }
    Ok(parsed)
}

fn exact_string(cursor: &mut JsonCursor<'_>, expected: &str) -> Result<(), EngineError> {
    if cursor.string(64)? == expected {
        Ok(())
    } else {
        Err(corrupt("Qwen2.5 configuration string is unsupported"))
    }
}

fn exact_u32(cursor: &mut JsonCursor<'_>, expected: u32) -> Result<(), EngineError> {
    if cursor.u32()? == expected {
        Ok(())
    } else {
        Err(corrupt("Qwen2.5 configuration integer is unsupported"))
    }
}

fn exact_f64(cursor: &mut JsonCursor<'_>, expected: f64) -> Result<(), EngineError> {
    if cursor.f64()?.to_bits() == expected.to_bits() {
        Ok(())
    } else {
        Err(corrupt("Qwen2.5 configuration number is unsupported"))
    }
}

fn exact_bool(cursor: &mut JsonCursor<'_>, expected: bool) -> Result<(), EngineError> {
    if cursor.boolean()? == expected {
        Ok(())
    } else {
        Err(corrupt("Qwen2.5 configuration boolean is unsupported"))
    }
}

fn corrupt(detail: impl AsRef<str>) -> EngineError {
    EngineError::corrupt_model_package(detail)
}

#[cfg(test)]
mod tests {
    use super::*;

    const INSTRUCT_CONFIG: &[u8] =
        include_bytes!("../tests/fixtures/qwen25-1.5b-instruct-config.json");
    const MATH_CONFIG: &[u8] =
        include_bytes!("../tests/fixtures/qwen25-math-1.5b-instruct-config.json");
    const CODER_CONFIG: &[u8] =
        include_bytes!("../tests/fixtures/qwen25-coder-1.5b-instruct-config.json");

    #[test]
    fn all_three_exact_compact_configuration_fixtures_validate() {
        let cases = [
            (
                INSTRUCT_CONFIG,
                Qwen25ConfigProfile::Instruct,
                32_768,
                1_000_000.0_f32,
                GENERAL_MODEL_SHA256,
            ),
            (
                MATH_CONFIG,
                Qwen25ConfigProfile::MathInstruct,
                4_096,
                10_000.0_f32,
                MATH_MODEL_SHA256,
            ),
            (
                CODER_CONFIG,
                Qwen25ConfigProfile::CoderInstruct,
                32_768,
                1_000_000.0_f32,
                CODER_MODEL_SHA256,
            ),
        ];
        for (fixture, profile, context, theta, model_digest) in cases {
            let parsed = Qwen25Config::parse(fixture).expect("pinned config must validate");
            assert_eq!(parsed.profile(), profile);
            assert_eq!(parsed.context_limit(), context);
            assert_eq!(parsed.rope_theta().to_bits(), theta.to_bits());
            assert_eq!(parsed.model_source_bytes(), QWEN25_MODEL_SAFETENSORS_BYTES);
            assert_eq!(parsed.model_source_sha256().to_string(), model_digest);
        }
    }

    #[test]
    fn unpinned_configuration_and_unknown_fields_fail_closed() {
        assert!(Qwen25Config::parse(b"{}").is_err());
        assert!(parse_graph(br#"{"unknown":1}"#).is_err());
    }
}
