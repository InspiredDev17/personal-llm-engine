//! Narrow canonical JSON support for signed package metadata.

pub(crate) fn sort_value(value: serde_json::Value) -> serde_json::Value {
    match value {
        serde_json::Value::Array(values) => {
            serde_json::Value::Array(values.into_iter().map(sort_value).collect())
        }
        serde_json::Value::Object(values) => {
            let mut entries: Vec<(String, serde_json::Value)> = values.into_iter().collect();
            entries.sort_by(|left, right| left.0.cmp(&right.0));
            serde_json::Value::Object(
                entries.into_iter().map(|(key, value)| (key, sort_value(value))).collect(),
            )
        }
        scalar => scalar,
    }
}

pub(crate) fn to_vec(value: serde_json::Value) -> Result<Vec<u8>, serde_json::Error> {
    serde_json::to_vec(&sort_value(value))
}
