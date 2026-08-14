use std::fmt::{Display, Formatter};

/// Validation or framing failure. Messages are stable enough for diagnostics,
/// but are not themselves a wire contract.
#[derive(Debug)]
pub enum ContractError {
    InvalidField { field: &'static str, reason: String },
    InvalidIdentifier { kind: &'static str, reason: String },
    UnsupportedRequiredExtension(String),
    Json(serde_json::Error),
    Io(std::io::Error),
    Frame(&'static str),
}

impl ContractError {
    #[must_use]
    pub fn field(field: &'static str, reason: impl Into<String>) -> Self {
        Self::InvalidField {
            field,
            reason: reason.into(),
        }
    }
}

impl Display for ContractError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidField { field, reason } => {
                write!(formatter, "invalid field {field}: {reason}")
            }
            Self::InvalidIdentifier { kind, reason } => {
                write!(formatter, "invalid {kind}: {reason}")
            }
            Self::UnsupportedRequiredExtension(name) => {
                write!(formatter, "unsupported required extension: {name}")
            }
            Self::Json(error) => Display::fmt(error, formatter),
            Self::Io(error) => Display::fmt(error, formatter),
            Self::Frame(reason) => write!(formatter, "invalid SAE1 frame: {reason}"),
        }
    }
}

impl std::error::Error for ContractError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Json(error) => Some(error),
            Self::Io(error) => Some(error),
            _ => None,
        }
    }
}

impl From<serde_json::Error> for ContractError {
    fn from(value: serde_json::Error) -> Self {
        Self::Json(value)
    }
}

impl From<std::io::Error> for ContractError {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value)
    }
}

pub type Result<T> = std::result::Result<T, ContractError>;

/// Performs semantic and cross-field validation after strict deserialization.
pub trait Validate {
    /// Validates all bounds and cross-field invariants for this contract value.
    ///
    /// # Errors
    ///
    /// Returns a [`ContractError`] when any contract invariant is violated.
    fn validate(&self) -> Result<()>;
}
