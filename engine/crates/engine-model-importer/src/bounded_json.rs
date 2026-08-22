//! Small non-recursive JSON cursor for closed offline source profiles.

use personalai_engine_types::{EngineError, MAX_REQUEST_BYTES};

/// Largest accepted source JSON document for the offline importer.
pub(crate) const MAX_SOURCE_JSON_BYTES: usize = 8 * 1024 * 1024;

pub(crate) struct JsonCursor<'a> {
    bytes: &'a [u8],
    cursor: usize,
}

impl<'a> JsonCursor<'a> {
    pub(crate) fn new(bytes: &'a [u8]) -> Result<Self, EngineError> {
        if bytes.is_empty()
            || bytes.len() > MAX_SOURCE_JSON_BYTES
            || bytes.len() > MAX_REQUEST_BYTES.saturating_mul(2)
            || std::str::from_utf8(bytes).is_err()
        {
            return Err(corrupt("source JSON bytes are outside bounds or invalid UTF-8"));
        }
        Ok(Self { bytes, cursor: 0 })
    }

    pub(crate) fn expect(&mut self, expected: u8) -> Result<(), EngineError> {
        self.whitespace();
        if self.consume_raw(expected) {
            Ok(())
        } else {
            Err(corrupt("source JSON punctuation is invalid"))
        }
    }

    pub(crate) fn consume(&mut self, value: u8) -> bool {
        self.whitespace();
        self.consume_raw(value)
    }

    pub(crate) fn null(&mut self) -> Result<(), EngineError> {
        self.literal(b"null", "source JSON null is invalid")
    }

    pub(crate) fn boolean(&mut self) -> Result<bool, EngineError> {
        self.whitespace();
        if self.consume_bytes(b"true") {
            Ok(true)
        } else if self.consume_bytes(b"false") {
            Ok(false)
        } else {
            Err(corrupt("source JSON boolean is invalid"))
        }
    }

    pub(crate) fn u32(&mut self) -> Result<u32, EngineError> {
        let text = self.number_text(10)?;
        if text.starts_with('-') || text.contains(['.', 'e', 'E']) {
            return Err(corrupt("source JSON integer is invalid"));
        }
        text.parse().map_err(|_| corrupt("source JSON integer overflows"))
    }

    pub(crate) fn f64(&mut self) -> Result<f64, EngineError> {
        let text = self.number_text(32)?;
        let value: f64 = text.parse().map_err(|_| corrupt("source JSON number is invalid"))?;
        if !value.is_finite() {
            return Err(corrupt("source JSON number is non-finite"));
        }
        Ok(value)
    }

    pub(crate) fn string(&mut self, maximum_bytes: usize) -> Result<String, EngineError> {
        self.whitespace();
        if maximum_bytes == 0 || !self.consume_raw(b'"') {
            return Err(corrupt("source JSON string is missing"));
        }
        let mut output = Vec::new();
        output.try_reserve_exact(maximum_bytes.min(256)).map_err(|_| {
            EngineError::resource_exhaustion("source JSON string allocation failed")
        })?;
        loop {
            let byte = *self
                .bytes
                .get(self.cursor)
                .ok_or_else(|| corrupt("source JSON string is unterminated"))?;
            self.cursor += 1;
            match byte {
                b'"' => break,
                b'\\' => self.escape(&mut output, maximum_bytes)?,
                0x00..=0x1f => return Err(corrupt("source JSON string contains a control byte")),
                _ => push_bounded(&mut output, byte, maximum_bytes)?,
            }
        }
        String::from_utf8(output).map_err(|_| corrupt("source JSON string is invalid UTF-8"))
    }

    pub(crate) fn key(&mut self, maximum_bytes: usize) -> Result<String, EngineError> {
        let key = self.string(maximum_bytes)?;
        self.expect(b':')?;
        Ok(key)
    }

    pub(crate) fn finish(mut self) -> Result<(), EngineError> {
        self.whitespace();
        if self.cursor == self.bytes.len() {
            Ok(())
        } else {
            Err(corrupt("source JSON contains trailing data"))
        }
    }

    fn escape(&mut self, output: &mut Vec<u8>, maximum: usize) -> Result<(), EngineError> {
        let escaped = *self
            .bytes
            .get(self.cursor)
            .ok_or_else(|| corrupt("source JSON escape is truncated"))?;
        self.cursor += 1;
        match escaped {
            b'"' | b'\\' | b'/' => push_bounded(output, escaped, maximum),
            b'b' => push_bounded(output, 0x08, maximum),
            b'f' => push_bounded(output, 0x0c, maximum),
            b'n' => push_bounded(output, b'\n', maximum),
            b'r' => push_bounded(output, b'\r', maximum),
            b't' => push_bounded(output, b'\t', maximum),
            b'u' => {
                let first = self.hex_quad()?;
                let scalar = if (0xd800..=0xdbff).contains(&first) {
                    if !self.consume_raw(b'\\') || !self.consume_raw(b'u') {
                        return Err(corrupt("source JSON high surrogate is unpaired"));
                    }
                    let second = self.hex_quad()?;
                    if !(0xdc00..=0xdfff).contains(&second) {
                        return Err(corrupt("source JSON surrogate pair is invalid"));
                    }
                    0x1_0000 + ((u32::from(first) - 0xd800) << 10) + (u32::from(second) - 0xdc00)
                } else if (0xdc00..=0xdfff).contains(&first) {
                    return Err(corrupt("source JSON low surrogate is unpaired"));
                } else {
                    u32::from(first)
                };
                let character = char::from_u32(scalar)
                    .ok_or_else(|| corrupt("source JSON Unicode escape is invalid"))?;
                let mut encoded = [0_u8; 4];
                for byte in character.encode_utf8(&mut encoded).as_bytes() {
                    push_bounded(output, *byte, maximum)?;
                }
                Ok(())
            }
            _ => Err(corrupt("source JSON escape is unsupported")),
        }
    }

    fn hex_quad(&mut self) -> Result<u16, EngineError> {
        let bytes = self
            .bytes
            .get(self.cursor..self.cursor.saturating_add(4))
            .ok_or_else(|| corrupt("source JSON Unicode escape is truncated"))?;
        self.cursor += 4;
        bytes.iter().try_fold(0_u16, |value, byte| {
            let nibble = match byte {
                b'0'..=b'9' => u16::from(*byte - b'0'),
                b'a'..=b'f' => u16::from(*byte - b'a' + 10),
                b'A'..=b'F' => u16::from(*byte - b'A' + 10),
                _ => return Err(corrupt("source JSON Unicode escape is invalid")),
            };
            Ok((value << 4) | nibble)
        })
    }

    fn number_text(&mut self, maximum_bytes: usize) -> Result<&'a str, EngineError> {
        self.whitespace();
        let start = self.cursor;
        while self.bytes.get(self.cursor).is_some_and(|byte| {
            byte.is_ascii_digit() || matches!(byte, b'-' | b'+' | b'.' | b'e' | b'E')
        }) {
            self.cursor += 1;
        }
        let bytes = self
            .bytes
            .get(start..self.cursor)
            .filter(|value| !value.is_empty() && value.len() <= maximum_bytes)
            .ok_or_else(|| corrupt("source JSON number is missing or oversized"))?;
        let text = std::str::from_utf8(bytes)
            .map_err(|_| corrupt("source JSON number is invalid UTF-8"))?;
        validate_number_grammar(text)?;
        Ok(text)
    }

    fn literal(&mut self, expected: &[u8], error: &'static str) -> Result<(), EngineError> {
        self.whitespace();
        if self.consume_bytes(expected) { Ok(()) } else { Err(corrupt(error)) }
    }

    fn consume_bytes(&mut self, expected: &[u8]) -> bool {
        if self.bytes.get(self.cursor..self.cursor.saturating_add(expected.len())) == Some(expected)
        {
            self.cursor += expected.len();
            true
        } else {
            false
        }
    }

    fn consume_raw(&mut self, value: u8) -> bool {
        if self.bytes.get(self.cursor) == Some(&value) {
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

fn validate_number_grammar(text: &str) -> Result<(), EngineError> {
    let bytes = text.as_bytes();
    let mut cursor = usize::from(bytes.first() == Some(&b'-'));
    if cursor == bytes.len() {
        return Err(corrupt("source JSON number is noncanonical"));
    }
    if bytes[cursor] == b'0' {
        cursor += 1;
        if bytes.get(cursor).is_some_and(u8::is_ascii_digit) {
            return Err(corrupt("source JSON number has a leading zero"));
        }
    } else if matches!(bytes[cursor], b'1'..=b'9') {
        cursor += 1;
        while bytes.get(cursor).is_some_and(u8::is_ascii_digit) {
            cursor += 1;
        }
    } else {
        return Err(corrupt("source JSON number integer part is invalid"));
    }
    if bytes.get(cursor) == Some(&b'.') {
        cursor += 1;
        let fraction_start = cursor;
        while bytes.get(cursor).is_some_and(u8::is_ascii_digit) {
            cursor += 1;
        }
        if cursor == fraction_start {
            return Err(corrupt("source JSON number fraction is empty"));
        }
    }
    if bytes.get(cursor).is_some_and(|byte| matches!(byte, b'e' | b'E')) {
        cursor += 1;
        if bytes.get(cursor).is_some_and(|byte| matches!(byte, b'+' | b'-')) {
            cursor += 1;
        }
        let exponent_start = cursor;
        while bytes.get(cursor).is_some_and(u8::is_ascii_digit) {
            cursor += 1;
        }
        if cursor == exponent_start
            || (cursor - exponent_start > 1 && bytes[exponent_start] == b'0')
        {
            return Err(corrupt("source JSON number exponent is noncanonical"));
        }
    }
    if cursor != bytes.len() {
        return Err(corrupt("source JSON number contains invalid syntax"));
    }
    if text.starts_with('-')
        && text.parse::<f64>().is_ok_and(|value| value.to_bits() == (-0.0_f64).to_bits())
    {
        return Err(corrupt("source JSON negative zero is noncanonical"));
    }
    Ok(())
}

fn push_bounded(output: &mut Vec<u8>, byte: u8, maximum: usize) -> Result<(), EngineError> {
    if output.len() >= maximum {
        return Err(corrupt("source JSON string exceeds its hard bound"));
    }
    output.push(byte);
    Ok(())
}

fn corrupt(detail: impl AsRef<str>) -> EngineError {
    EngineError::corrupt_model_package(detail)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cursor_decodes_bounded_strings_and_rejects_trailing_or_surrogates() {
        let mut cursor = JsonCursor::new(br#" {"value":"a\n\u00e9\ud83d\ude00"} "#).unwrap();
        cursor.expect(b'{').unwrap();
        assert_eq!(cursor.key(16).unwrap(), "value");
        assert_eq!(cursor.string(16).unwrap(), "a\né😀");
        cursor.expect(b'}').unwrap();
        cursor.finish().unwrap();

        let mut invalid = JsonCursor::new(br#""\ud800""#).unwrap();
        assert!(invalid.string(16).is_err());
        let mut trailing = JsonCursor::new(b"null x").unwrap();
        trailing.null().unwrap();
        assert!(trailing.finish().is_err());
    }

    #[test]
    fn cursor_rejects_noncanonical_json_number_forms() {
        for invalid in
            ["01", "-01", "00", "-0", "-0.0", "+1", ".1", "1.", "1e", "1e+", "1e01", "1+2", "--1"]
        {
            let mut cursor = JsonCursor::new(invalid.as_bytes()).unwrap();
            assert!(cursor.f64().is_err(), "accepted {invalid}");
        }
        for valid in ["0", "1", "-1", "0.1", "-0.1", "1e0", "1E+2", "1e-2"] {
            let mut cursor = JsonCursor::new(valid.as_bytes()).unwrap();
            assert!(cursor.f64().is_ok(), "rejected {valid}");
            assert!(cursor.finish().is_ok(), "left trailing data for {valid}");
        }
    }
}
