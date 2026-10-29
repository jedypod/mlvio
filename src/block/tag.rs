// SPDX-License-Identifier: MIT OR Apache-2.0
//! Four-character block tags and the common 16-byte block header.

use core::fmt;

use super::cursor::Cursor;
use crate::{Error, Result};

/// Four ASCII bytes identifying a block type (`blockType` in `mlv_hdr_t`).
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Tag(pub [u8; 4]);

macro_rules! tags {
    ($($(#[$m:meta])* $name:ident = $lit:literal;)*) => {
        impl Tag {
            $( $(#[$m])* pub const $name: Tag = Tag(*$lit); )*
        }
    };
}

tags! {
    /// File header; first block of every chunk.
    MLVI = b"MLVI";
    /// Video frame.
    VIDF = b"VIDF";
    /// Audio frame.
    AUDF = b"AUDF";
    /// Raw format info (`raw_info`).
    RAWI = b"RAWI";
    /// Raw capture info (sensor, binning).
    RAWC = b"RAWC";
    /// WAV audio format.
    WAVI = b"WAVI";
    /// Exposure (ISO, shutter).
    EXPO = b"EXPO";
    /// Lens.
    LENS = b"LENS";
    /// Extended lens (variable-length name).
    ELNS = b"ELNS";
    /// Real-time clock.
    RTCI = b"RTCI";
    /// Camera identity.
    IDNT = b"IDNT";
    /// Frame offset index written in post.
    XREF = b"XREF";
    /// Free-form info string.
    INFO = b"INFO";
    /// Dual-ISO settings.
    DISO = b"DISO";
    /// User marker.
    MARK = b"MARK";
    /// Picture style.
    STYL = b"STYL";
    /// Electronic level.
    ELVL = b"ELVL";
    /// White balance.
    WBAL = b"WBAL";
    /// Debug log text.
    DEBG = b"DEBG";
    /// Module version string.
    VERS = b"VERS";
    /// Decode curve LUT.
    CURV = b"CURV";
    /// Averaged dark frame.
    DARK = b"DARK";
    /// Padding.
    NULL = b"NULL";
    /// Backup / padding (MLV-App).
    BKUP = b"BKUP";
}

impl Tag {
    /// Tag as a `&str` if all four bytes are printable ASCII.
    pub fn as_str(&self) -> Option<&str> {
        if self.0.iter().all(u8::is_ascii_graphic) {
            core::str::from_utf8(&self.0).ok()
        } else {
            None
        }
    }

    /// True for the two frame-carrying block types.
    #[inline]
    pub fn is_frame(&self) -> bool {
        *self == Tag::VIDF || *self == Tag::AUDF
    }
}

impl fmt::Display for Tag {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.as_str() {
            Some(s) => f.write_str(s),
            None => write!(
                f,
                "{:02x}{:02x}{:02x}{:02x}",
                self.0[0], self.0[1], self.0[2], self.0[3]
            ),
        }
    }
}

impl fmt::Debug for Tag {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Tag({self})")
    }
}

impl From<[u8; 4]> for Tag {
    fn from(b: [u8; 4]) -> Self {
        Tag(b)
    }
}

/// The 16-byte header every block starts with (`mlv_hdr_t`).
///
/// For `MLVI` the `timestamp` slot holds `versionString[8]`; see
/// [`BlockHeader::version_bytes`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct BlockHeader {
    /// Block type.
    pub tag: Tag,
    /// Total block size in bytes including this header (`blockSize`).
    pub size: u32,
    /// Microseconds since recording start (`timestamp`), or the raw version
    /// string bytes for `MLVI`.
    pub timestamp: u64,
}

impl BlockHeader {
    /// Header length in bytes.
    pub const SIZE: usize = 16;

    /// Parse from at least 16 bytes. Fails on short input or `size < 16`.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < Self::SIZE {
            return Err(Error::BadBlock {
                offset: 0,
                reason: "block header shorter than 16 bytes",
            });
        }
        let mut c = Cursor::new(bytes);
        let tag = Tag(c.array());
        let size = c.u32();
        let timestamp = c.u64();
        if size < Self::SIZE as u32 {
            return Err(Error::BadBlock {
                offset: 0,
                reason: "blockSize smaller than header",
            });
        }
        Ok(BlockHeader {
            tag,
            size,
            timestamp,
        })
    }

    /// Payload length (`size - 16`).
    #[inline]
    pub fn payload_len(&self) -> usize {
        (self.size as usize).saturating_sub(Self::SIZE)
    }

    /// For `MLVI`, the eight version-string bytes stored in the timestamp slot.
    #[inline]
    pub fn version_bytes(&self) -> [u8; 8] {
        self.timestamp.to_le_bytes()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::string::ToString;

    #[test]
    fn header_roundtrip() {
        let mut b = [0u8; 16];
        b[..4].copy_from_slice(b"VIDF");
        b[4..8].copy_from_slice(&1234u32.to_le_bytes());
        b[8..16].copy_from_slice(&99u64.to_le_bytes());
        let h = BlockHeader::from_bytes(&b).unwrap();
        assert_eq!(h.tag, Tag::VIDF);
        assert_eq!(h.size, 1234);
        assert_eq!(h.timestamp, 99);
        assert_eq!(h.payload_len(), 1218);
        assert_eq!(Tag::VIDF.to_string(), "VIDF");
    }

    #[test]
    fn rejects_small_size() {
        let mut b = [0u8; 16];
        b[..4].copy_from_slice(b"NULL");
        b[4..8].copy_from_slice(&8u32.to_le_bytes());
        assert!(BlockHeader::from_bytes(&b).is_err());
        assert!(BlockHeader::from_bytes(&b[..10]).is_err());
    }
}
