use crate::error::{ContractError, Result, Validate};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::fmt::{Display, Formatter};

fn validate_id(kind: &'static str, prefix: &str, value: &str) -> Result<()> {
    let Some(body) = value.strip_prefix(prefix) else {
        return Err(ContractError::InvalidIdentifier {
            kind,
            reason: format!("must start with {prefix}"),
        });
    };
    let maximum = match prefix {
        "usr_" | "agt_" | "tsk_" | "req_" | "ses_" | "mdl_" | "art_" | "mem_" | "tol_" | "nod_"
        | "sec_" | "aud_" => 63,
        _ => 64,
    };
    if !(2..=maximum).contains(&body.len()) {
        return Err(ContractError::InvalidIdentifier {
            kind,
            reason: format!("suffix must contain 2 through {maximum} ASCII characters"),
        });
    }
    let mut bytes = body.bytes();
    if !bytes
        .next()
        .is_some_and(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit())
    {
        return Err(ContractError::InvalidIdentifier {
            kind,
            reason: "suffix must begin with a lowercase ASCII letter or digit".to_owned(),
        });
    }
    if !bytes.all(|byte| {
        byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'_' || byte == b'-'
    }) {
        return Err(ContractError::InvalidIdentifier {
            kind,
            reason: "suffix contains a forbidden character".to_owned(),
        });
    }
    Ok(())
}

macro_rules! id_type {
    ($name:ident, $prefix:literal) => {
        #[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
        pub struct $name(String);

        impl $name {
            /// Constructs and validates an opaque typed identifier.
            ///
            /// # Errors
            ///
            /// Returns an error for a wrong prefix, invalid character, or
            /// suffix outside the published length bounds.
            pub fn new(value: impl Into<String>) -> Result<Self> {
                let value = value.into();
                validate_id(stringify!($name), $prefix, &value)?;
                Ok(Self(value))
            }

            #[must_use]
            pub fn as_str(&self) -> &str {
                &self.0
            }
        }

        impl Validate for $name {
            fn validate(&self) -> Result<()> {
                validate_id(stringify!($name), $prefix, &self.0)
            }
        }

        impl Display for $name {
            fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
                formatter.write_str(&self.0)
            }
        }

        impl TryFrom<String> for $name {
            type Error = ContractError;

            fn try_from(value: String) -> Result<Self> {
                Self::new(value)
            }
        }

        impl Serialize for $name {
            fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
            where
                S: Serializer,
            {
                serializer.serialize_str(&self.0)
            }
        }

        impl<'de> Deserialize<'de> for $name {
            fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
            where
                D: Deserializer<'de>,
            {
                let value = String::deserialize(deserializer)?;
                Self::new(value).map_err(serde::de::Error::custom)
            }
        }
    };
}

id_type!(UserId, "usr_");
id_type!(AgentId, "agt_");
id_type!(TaskId, "tsk_");
id_type!(RequestId, "req_");
id_type!(SessionId, "ses_");
id_type!(ModelId, "mdl_");
id_type!(ModelArtifactId, "art_");
id_type!(MemoryId, "mem_");
id_type!(ToolId, "tol_");
id_type!(NodeId, "nod_");
id_type!(SecurityEventId, "sec_");
id_type!(AuditEventId, "aud_");
id_type!(CognitivePlanId, "cpl_");
id_type!(CognitivePlanNodeId, "cpn_");
id_type!(CognitiveExecutionResultId, "cer_");
id_type!(QualificationEvidenceId, "qev_");
id_type!(EvaluationRevisionId, "evr_");
id_type!(ProjectId, "prj_");
id_type!(KnowledgePlanId, "kpl_");
id_type!(KnowledgeQueryId, "kqy_");
id_type!(KnowledgeSourceId, "ksr_");
id_type!(KnowledgeRevisionId, "krv_");
id_type!(StaticKnowledgeBundleId, "kbn_");
id_type!(CagCacheId, "cag_");
id_type!(KnowledgeInvalidationEventId, "kie_");

#[cfg(test)]
mod tests {
    use super::{RequestId, StaticKnowledgeBundleId, UserId};

    #[test]
    fn accepts_valid_typed_id() {
        let id = UserId::new("usr_ab").expect("valid ID");
        assert_eq!(id.as_str(), "usr_ab");
    }

    #[test]
    fn suffix_boundary_matches_published_schema() {
        assert!(UserId::new(format!("usr_{}", "a".repeat(63))).is_ok());
        assert!(UserId::new(format!("usr_{}", "a".repeat(64))).is_err());
        assert!(StaticKnowledgeBundleId::new(format!("kbn_{}", "a".repeat(64))).is_ok());
    }

    #[test]
    fn rejects_database_like_and_wrong_prefix_ids() {
        assert!(UserId::new("42").is_err());
        assert!(UserId::new("req_ab").is_err());
        assert!(RequestId::new("req_AA").is_err());
    }

    #[test]
    fn serde_rejects_invalid_id() {
        assert!(serde_json::from_str::<UserId>(r#""usr_a""#).is_err());
    }
}
