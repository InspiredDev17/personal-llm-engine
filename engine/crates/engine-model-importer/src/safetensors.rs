//! Narrow, explicit parser for data-only safetensors used by the offline importer.

use personalai_engine_types::{
    DType, EngineError, MAX_DIMENSION_SIZE, MAX_MODEL_PACKAGE_BYTES, MAX_TENSOR_COUNT,
    MAX_TENSOR_RANK, TensorShape,
};
use std::collections::BTreeSet;
use std::ops::Range;

/// Largest accepted safetensors JSON header: 64 MiB.
pub const MAX_SAFETENSORS_HEADER_BYTES: usize = 64 * 1024 * 1024;
/// Largest external tensor name inspected by the importer.
pub const MAX_SOURCE_TENSOR_NAME_BYTES: usize = 256;

/// One validated tensor view into an immutable source file.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SafeTensorRecord {
    name: String,
    dtype: DType,
    shape: TensorShape,
    data_range: Range<usize>,
}

impl SafeTensorRecord {
    /// Returns the source tensor name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns the source tensor dtype.
    pub const fn dtype(&self) -> DType {
        self.dtype
    }

    /// Returns the checked source tensor shape.
    pub const fn shape(&self) -> &TensorShape {
        &self.shape
    }

    /// Returns the exact absolute source byte range.
    pub fn data_range(&self) -> Range<usize> {
        self.data_range.clone()
    }
}

/// Fully validated, zero-copy safetensors metadata view.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParsedSafetensors<'a> {
    bytes: &'a [u8],
    tensors: Vec<SafeTensorRecord>,
}

impl<'a> ParsedSafetensors<'a> {
    /// Parses only the bounded safetensors structures needed by the importer.
    pub fn parse(bytes: &'a [u8]) -> Result<Self, EngineError> {
        if bytes.len() < 10 {
            return Err(corrupt("safetensors file is truncated"));
        }
        if u64::try_from(bytes.len()).unwrap_or(u64::MAX) > MAX_MODEL_PACKAGE_BYTES {
            return Err(EngineError::resource_exhaustion("safetensors file exceeds hard limit"));
        }
        let header_length = usize::try_from(u64::from_le_bytes(
            bytes[..8].try_into().map_err(|_| corrupt("safetensors header length is truncated"))?,
        ))
        .map_err(|_| EngineError::resource_exhaustion("safetensors header is not addressable"))?;
        if !(2..=MAX_SAFETENSORS_HEADER_BYTES).contains(&header_length) {
            return Err(corrupt("safetensors header length is outside hard bounds"));
        }
        let data_start = 8_usize
            .checked_add(header_length)
            .ok_or_else(|| EngineError::resource_exhaustion("safetensors data offset overflows"))?;
        let header = bytes
            .get(8..data_start)
            .ok_or_else(|| corrupt("safetensors JSON header is truncated"))?;
        let header_text = std::str::from_utf8(header)
            .map_err(|_| corrupt("safetensors header is not valid UTF-8"))?;
        let relative = HeaderParser::new(header_text).parse()?;
        if relative.is_empty()
            || relative.len() > usize::try_from(MAX_TENSOR_COUNT).unwrap_or(usize::MAX)
        {
            return Err(corrupt("safetensors tensor count is outside hard bounds"));
        }
        let mut tensors = Vec::new();
        tensors.try_reserve_exact(relative.len()).map_err(|_| {
            EngineError::resource_exhaustion("safetensors metadata allocation failed")
        })?;
        let mut previous_end = 0_usize;
        for tensor in relative {
            if tensor.start != previous_end || tensor.end <= tensor.start {
                return Err(corrupt("safetensors tensor ranges are overlapping or noncanonical"));
            }
            let absolute_start = data_start
                .checked_add(tensor.start)
                .ok_or_else(|| EngineError::resource_exhaustion("tensor offset overflows"))?;
            let absolute_end = data_start
                .checked_add(tensor.end)
                .ok_or_else(|| EngineError::resource_exhaustion("tensor end overflows"))?;
            let expected = usize::try_from(tensor.shape.byte_len(tensor.dtype)?).map_err(|_| {
                EngineError::resource_exhaustion("tensor length is not addressable")
            })?;
            if tensor.end - tensor.start != expected || absolute_end > bytes.len() {
                return Err(corrupt("safetensors tensor range is out of bounds or wrong length"));
            }
            previous_end = tensor.end;
            tensors.push(SafeTensorRecord {
                name: tensor.name,
                dtype: tensor.dtype,
                shape: tensor.shape,
                data_range: absolute_start..absolute_end,
            });
        }
        if data_start.checked_add(previous_end) != Some(bytes.len()) {
            return Err(corrupt("safetensors data has trailing or unindexed bytes"));
        }
        Ok(Self { bytes, tensors })
    }

    /// Returns every source tensor in canonical source-offset order.
    pub fn tensors(&self) -> &[SafeTensorRecord] {
        &self.tensors
    }

    /// Returns the exact immutable bytes for one validated tensor.
    pub fn tensor_bytes(&self, tensor: &SafeTensorRecord) -> Result<&'a [u8], EngineError> {
        self.bytes
            .get(tensor.data_range())
            .ok_or_else(|| corrupt("validated safetensors range became unavailable"))
    }
}

#[derive(Debug)]
struct RelativeTensor {
    name: String,
    dtype: DType,
    shape: TensorShape,
    start: usize,
    end: usize,
}

struct HeaderParser<'a> {
    bytes: &'a [u8],
    cursor: usize,
}

impl<'a> HeaderParser<'a> {
    const fn new(text: &'a str) -> Self {
        Self { bytes: text.as_bytes(), cursor: 0 }
    }

    fn parse(mut self) -> Result<Vec<RelativeTensor>, EngineError> {
        self.expect(b'{')?;
        let mut tensors = Vec::new();
        let mut names = BTreeSet::new();
        self.whitespace();
        if self.consume(b'}') {
            return Ok(tensors);
        }
        loop {
            let name = self.string(MAX_SOURCE_TENSOR_NAME_BYTES)?;
            self.expect(b':')?;
            if name == "__metadata__" {
                self.metadata()?;
            } else {
                validate_source_name(&name)?;
                if !names.insert(name.clone()) {
                    return Err(corrupt("safetensors contains a duplicate tensor name"));
                }
                tensors.push(self.tensor(name)?);
            }
            self.whitespace();
            if self.consume(b'}') {
                break;
            }
            self.expect(b',')?;
        }
        self.whitespace();
        if self.cursor != self.bytes.len() {
            return Err(corrupt("safetensors header contains trailing JSON"));
        }
        tensors.sort_by_key(|tensor| tensor.start);
        Ok(tensors)
    }

    fn tensor(&mut self, name: String) -> Result<RelativeTensor, EngineError> {
        self.expect(b'{')?;
        let mut dtype = None;
        let mut shape = None;
        let mut offsets = None;
        loop {
            let key = self.string(32)?;
            self.expect(b':')?;
            match key.as_str() {
                "dtype" if dtype.is_none() => dtype = Some(parse_dtype(&self.string(16)?)?),
                "shape" if shape.is_none() => shape = Some(self.shape()?),
                "data_offsets" if offsets.is_none() => offsets = Some(self.offsets()?),
                "dtype" | "shape" | "data_offsets" => {
                    return Err(corrupt("safetensors tensor field is duplicated"));
                }
                _ => return Err(corrupt("safetensors tensor field is unsupported")),
            }
            self.whitespace();
            if self.consume(b'}') {
                break;
            }
            self.expect(b',')?;
        }
        let (start, end) = offsets.ok_or_else(|| corrupt("safetensors offsets are missing"))?;
        Ok(RelativeTensor {
            name,
            dtype: dtype.ok_or_else(|| corrupt("safetensors dtype is missing"))?,
            shape: shape.ok_or_else(|| corrupt("safetensors shape is missing"))?,
            start,
            end,
        })
    }

    fn metadata(&mut self) -> Result<(), EngineError> {
        self.expect(b'{')?;
        self.whitespace();
        if self.consume(b'}') {
            return Ok(());
        }
        loop {
            self.string(256)?;
            self.expect(b':')?;
            self.string(4_096)?;
            self.whitespace();
            if self.consume(b'}') {
                return Ok(());
            }
            self.expect(b',')?;
        }
    }

    fn shape(&mut self) -> Result<TensorShape, EngineError> {
        self.expect(b'[')?;
        let mut dimensions = Vec::new();
        self.whitespace();
        if self.consume(b']') {
            return TensorShape::new([]).map_err(|_| corrupt("safetensors scalar is invalid"));
        }
        loop {
            if dimensions.len() >= MAX_TENSOR_RANK {
                return Err(corrupt("safetensors tensor rank exceeds hard limit"));
            }
            let dimension = self.number()?;
            if dimension == 0 || dimension > MAX_DIMENSION_SIZE {
                return Err(corrupt("safetensors tensor dimension is outside hard bounds"));
            }
            dimensions.push(dimension);
            self.whitespace();
            if self.consume(b']') {
                break;
            }
            self.expect(b',')?;
        }
        TensorShape::new(dimensions).map_err(|_| corrupt("safetensors tensor shape is invalid"))
    }

    fn offsets(&mut self) -> Result<(usize, usize), EngineError> {
        self.expect(b'[')?;
        let start = usize::try_from(self.number()?).map_err(|_| {
            EngineError::resource_exhaustion("safetensors offset is not addressable")
        })?;
        self.expect(b',')?;
        let end = usize::try_from(self.number()?).map_err(|_| {
            EngineError::resource_exhaustion("safetensors offset is not addressable")
        })?;
        self.expect(b']')?;
        Ok((start, end))
    }

    fn string(&mut self, maximum: usize) -> Result<String, EngineError> {
        self.whitespace();
        if !self.consume_raw(b'"') {
            return Err(corrupt("safetensors JSON string is missing"));
        }
        let start = self.cursor;
        while let Some(byte) = self.bytes.get(self.cursor).copied() {
            if byte == b'"' {
                let value = self
                    .bytes
                    .get(start..self.cursor)
                    .ok_or_else(|| corrupt("safetensors JSON string range is invalid"))?;
                self.cursor += 1;
                if value.is_empty()
                    || value.len() > maximum
                    || value.contains(&b'\\')
                    || value.iter().any(|byte| *byte < 0x20)
                {
                    return Err(corrupt("safetensors JSON string is noncanonical"));
                }
                return std::str::from_utf8(value)
                    .map(str::to_owned)
                    .map_err(|_| corrupt("safetensors JSON string is invalid UTF-8"));
            }
            self.cursor += 1;
        }
        Err(corrupt("safetensors JSON string is unterminated"))
    }

    fn number(&mut self) -> Result<u64, EngineError> {
        self.whitespace();
        let start = self.cursor;
        while self.bytes.get(self.cursor).is_some_and(u8::is_ascii_digit) {
            self.cursor += 1;
        }
        if start == self.cursor {
            return Err(corrupt("safetensors JSON integer is missing"));
        }
        if self.cursor - start > 1 && self.bytes[start] == b'0' {
            return Err(corrupt("safetensors JSON integer has ambiguous leading zeroes"));
        }
        let text = std::str::from_utf8(&self.bytes[start..self.cursor])
            .map_err(|_| corrupt("safetensors JSON integer is invalid"))?;
        text.parse().map_err(|_| corrupt("safetensors JSON integer overflows"))
    }

    fn expect(&mut self, expected: u8) -> Result<(), EngineError> {
        self.whitespace();
        if self.consume_raw(expected) {
            Ok(())
        } else {
            Err(corrupt("safetensors JSON punctuation is invalid"))
        }
    }

    fn consume(&mut self, expected: u8) -> bool {
        self.whitespace();
        self.consume_raw(expected)
    }

    fn consume_raw(&mut self, expected: u8) -> bool {
        if self.bytes.get(self.cursor) == Some(&expected) {
            self.cursor += 1;
            true
        } else {
            false
        }
    }

    fn whitespace(&mut self) {
        while self
            .bytes
            .get(self.cursor)
            .is_some_and(|byte| matches!(byte, b' ' | b'\n' | b'\r' | b'\t'))
        {
            self.cursor += 1;
        }
    }
}

fn parse_dtype(value: &str) -> Result<DType, EngineError> {
    match value {
        "F32" => Ok(DType::F32),
        "F16" => Ok(DType::F16),
        "BF16" => Ok(DType::BFloat16),
        _ => Err(EngineError::not_implemented("safetensors source dtype is unsupported")),
    }
}

fn validate_source_name(name: &str) -> Result<(), EngineError> {
    if name.len() > MAX_SOURCE_TENSOR_NAME_BYTES
        || !name.is_ascii()
        || name.contains("..")
        || name.chars().any(char::is_control)
        || !name
            .as_bytes()
            .iter()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_'))
    {
        return Err(corrupt("safetensors tensor name is invalid"));
    }
    Ok(())
}

fn corrupt(detail: impl AsRef<str>) -> EngineError {
    EngineError::corrupt_model_package(detail)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture(header: &str, data: &[u8]) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(
            &u64::try_from(header.len()).expect("length must fit").to_le_bytes(),
        );
        bytes.extend_from_slice(header.as_bytes());
        bytes.extend_from_slice(data);
        bytes
    }

    #[test]
    fn parses_exact_data_only_supported_source_dtypes() {
        for (dtype, data) in [("F32", &[0_u8; 8][..]), ("F16", &[0_u8; 4]), ("BF16", &[0_u8; 4])] {
            let header = format!(
                "{{\"weight\":{{\"dtype\":\"{dtype}\",\"shape\":[2],\"data_offsets\":[0,{}]}}}}",
                data.len()
            );
            let bytes = fixture(&header, data);
            let parsed = ParsedSafetensors::parse(&bytes).expect("source must parse");
            assert_eq!(parsed.tensors().len(), 1);
            assert_eq!(parsed.tensor_bytes(&parsed.tensors()[0]).expect("bytes must exist"), data);
        }
    }

    #[test]
    fn rejects_trailing_overlap_dtype_rank_and_invalid_utf8() {
        for header in [
            r#"{"a":{"dtype":"F64","shape":[1],"data_offsets":[0,8]}}"#,
            r#"{"a":{"dtype":"F32","shape":[1,1,1,1,1,1,1,1,1],"data_offsets":[0,4]}}"#,
            r#"{"a":{"dtype":"F32","shape":[1],"data_offsets":[1,5]}}"#,
        ] {
            assert!(ParsedSafetensors::parse(&fixture(header, &[0; 8])).is_err());
        }
        let mut invalid = fixture("{}", &[]);
        invalid[8] = 0xff;
        assert!(ParsedSafetensors::parse(&invalid).is_err());
    }
}
