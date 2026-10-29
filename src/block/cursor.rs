// SPDX-License-Identifier: MIT OR Apache-2.0
//! Tolerant little-endian byte cursor used by every `from_bytes`.
//!
//! Firmware revisions have appended fields to several blocks, so readers must
//! accept short payloads. Every read on an exhausted cursor returns the type's
//! default (zero) and sets [`Cursor::truncated`], instead of failing. Callers
//! that need a field to be present check `truncated()` or use the `try_*`
//! variants.

/// Little-endian reader over a byte slice that never panics.
#[derive(Debug, Clone)]
pub struct Cursor<'a> {
    data: &'a [u8],
    pos: usize,
    truncated: bool,
}

impl<'a> Cursor<'a> {
    /// Wrap a slice.
    #[inline]
    pub fn new(data: &'a [u8]) -> Self {
        Cursor {
            data,
            pos: 0,
            truncated: false,
        }
    }

    /// Current position in bytes.
    #[inline]
    pub fn position(&self) -> usize {
        self.pos
    }

    /// Bytes not yet consumed.
    #[inline]
    pub fn remaining(&self) -> usize {
        self.data.len().saturating_sub(self.pos)
    }

    /// True once any read ran past the end of the data.
    #[inline]
    pub fn truncated(&self) -> bool {
        self.truncated
    }

    /// The unread tail of the slice.
    #[inline]
    pub fn rest(&self) -> &'a [u8] {
        self.data.get(self.pos..).unwrap_or(&[])
    }

    /// Take `n` bytes, or `None` (and mark truncated) if fewer remain.
    #[inline]
    pub fn try_bytes(&mut self, n: usize) -> Option<&'a [u8]> {
        let end = self.pos.checked_add(n)?;
        match self.data.get(self.pos..end) {
            Some(s) => {
                self.pos = end;
                Some(s)
            }
            None => {
                self.truncated = true;
                self.pos = self.data.len();
                None
            }
        }
    }

    /// Take exactly `N` bytes as an array, zero-filled and marked truncated if short.
    #[inline]
    pub fn array<const N: usize>(&mut self) -> [u8; N] {
        let mut out = [0u8; N];
        if let Some(s) = self.try_bytes(N) {
            out.copy_from_slice(s);
        }
        out
    }

    /// Skip `n` bytes.
    #[inline]
    pub fn skip(&mut self, n: usize) {
        let _ = self.try_bytes(n);
    }

    /// Read a `u8`, 0 if exhausted.
    #[inline]
    pub fn u8(&mut self) -> u8 {
        self.array::<1>()[0]
    }

    /// Read a little-endian `u16`, 0 if exhausted.
    #[inline]
    pub fn u16(&mut self) -> u16 {
        u16::from_le_bytes(self.array())
    }

    /// Read a little-endian `i16`, 0 if exhausted.
    #[inline]
    pub fn i16(&mut self) -> i16 {
        i16::from_le_bytes(self.array())
    }

    /// Read a little-endian `u32`, 0 if exhausted.
    #[inline]
    pub fn u32(&mut self) -> u32 {
        u32::from_le_bytes(self.array())
    }

    /// Read a little-endian `i32`, 0 if exhausted.
    #[inline]
    pub fn i32(&mut self) -> i32 {
        i32::from_le_bytes(self.array())
    }

    /// Read a little-endian `u64`, 0 if exhausted.
    #[inline]
    pub fn u64(&mut self) -> u64 {
        u64::from_le_bytes(self.array())
    }

    /// Read a `u16` only if fully present.
    #[inline]
    pub fn try_u16(&mut self) -> Option<u16> {
        self.try_bytes(2).map(|s| u16::from_le_bytes([s[0], s[1]]))
    }

    /// Read a `u32` only if fully present.
    #[inline]
    pub fn try_u32(&mut self) -> Option<u32> {
        self.try_bytes(4)
            .map(|s| u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
    }

    /// Read a `u64` only if fully present.
    #[inline]
    pub fn try_u64(&mut self) -> Option<u64> {
        self.try_bytes(8).map(|s| {
            let mut a = [0u8; 8];
            a.copy_from_slice(s);
            u64::from_le_bytes(a)
        })
    }
}

/// Interpret a fixed-size, NUL-padded C string field as `&str`.
///
/// Stops at the first NUL, trims trailing whitespace, and replaces invalid
/// UTF-8 by truncating at the first bad byte.
pub fn c_str(bytes: &[u8]) -> &str {
    let end = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
    let bytes = &bytes[..end];
    let s = match core::str::from_utf8(bytes) {
        Ok(s) => s,
        Err(e) => core::str::from_utf8(&bytes[..e.valid_up_to()]).unwrap_or(""),
    };
    s.trim_end()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_little_endian() {
        let d = [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a];
        let mut c = Cursor::new(&d);
        assert_eq!(c.u16(), 0x0201);
        assert_eq!(c.u32(), 0x0605_0403);
        assert_eq!(c.remaining(), 4);
        assert!(!c.truncated());
        assert_eq!(c.u64(), 0); // short read -> zero, truncated
        assert!(c.truncated());
        assert_eq!(c.remaining(), 0);
    }

    #[test]
    fn c_str_trims() {
        assert_eq!(
            c_str(b"Canon EOS 5D Mark III\0\0\0\0"),
            "Canon EOS 5D Mark III"
        );
        assert_eq!(c_str(b"abc   "), "abc");
        assert_eq!(c_str(b""), "");
        assert_eq!(c_str(b"ok\xff\xfe"), "ok");
    }
}
