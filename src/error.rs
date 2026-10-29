// SPDX-License-Identifier: MIT OR Apache-2.0
//! Error types. Library code never panics on input data; every failure is one
//! of these variants.

use alloc::string::String;
use core::fmt;

use crate::block::Tag;

/// Convenience alias used throughout the crate.
pub type Result<T, E = Error> = core::result::Result<T, E>;

/// Video class flags / formats the crate recognises but does not decode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Unsupported {
    /// `MLV_VIDEO_CLASS_FLAG_LZMA` (0x80).
    Lzma,
    /// `MLV_VIDEO_CLASS_FLAG_DELTA` (0x40).
    Delta,
    /// `MLV_VIDEO_CLASS_FLAG_JPEG2K` (0x10, MLV-App extension).
    Jpeg2000,
    /// McRAW container pretending to be MLV (0x100, MLV-App extension).
    McRaw,
    /// Any other `videoClass` whose base class is not RAW (1).
    VideoClass(u16),
    /// `bits_per_pixel` other than 10, 12 or 14 for uncompressed frames.
    BitDepth(u32),
    /// File has an audio class but no video (`videoClass == 0`).
    AudioOnly,
}

impl fmt::Display for Unsupported {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Unsupported::Lzma => write!(f, "LZMA-compressed frames"),
            Unsupported::Delta => write!(f, "delta-encoded frames"),
            Unsupported::Jpeg2000 => write!(f, "JPEG 2000 frames"),
            Unsupported::McRaw => write!(f, "McRAW container"),
            Unsupported::VideoClass(c) => write!(f, "video class {c:#x}"),
            Unsupported::BitDepth(b) => write!(f, "{b}-bit uncompressed frames"),
            Unsupported::AudioOnly => write!(f, "clip has no video stream"),
        }
    }
}

/// Lossless JPEG decoder failures.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Lj92Error {
    /// Stream does not begin with SOI (`FF D8`).
    NoSoi,
    /// A marker segment extends past the end of the data.
    Truncated,
    /// SOF3 missing or inconsistent (zero size, components > 4, bad precision).
    BadFrameHeader,
    /// DHT segment malformed or no table defined for a scanned component.
    BadHuffmanTable,
    /// SOS references a component or table that does not exist, or uses a
    /// predictor outside 0..=7.
    BadScanHeader,
    /// A `DRI`/`RSTn` marker was found. Magic Lantern never emits restart
    /// intervals and the decoder does not support them.
    RestartMarkers,
    /// Entropy-coded data ended before the image was complete.
    ScanTruncated,
    /// Huffman code not present in the table (corrupt data).
    BadCode,
    /// Caller supplied an output slice whose length is not `w * h * comps`.
    OutputSize {
        /// Required length in samples.
        expected: usize,
        /// Supplied length.
        got: usize,
    },
    /// Image dimensions multiply past `u32::MAX` or are too wide for the LUT.
    TooLarge,
    /// Only SOF3 (lossless, Huffman) is supported.
    NotLossless(u8),
}

impl fmt::Display for Lj92Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Lj92Error::NoSoi => write!(f, "no SOI marker"),
            Lj92Error::Truncated => write!(f, "marker segment truncated"),
            Lj92Error::BadFrameHeader => write!(f, "bad SOF3 frame header"),
            Lj92Error::BadHuffmanTable => write!(f, "bad Huffman table"),
            Lj92Error::BadScanHeader => write!(f, "bad SOS scan header"),
            Lj92Error::RestartMarkers => write!(f, "restart markers not supported"),
            Lj92Error::ScanTruncated => write!(f, "entropy-coded data truncated"),
            Lj92Error::BadCode => write!(f, "invalid Huffman code"),
            Lj92Error::OutputSize { expected, got } => {
                write!(f, "output buffer has {got} samples, need {expected}")
            }
            Lj92Error::TooLarge => write!(f, "image dimensions too large"),
            Lj92Error::NotLossless(m) => write!(f, "SOF marker {m:#x} is not lossless JPEG"),
        }
    }
}

impl core::error::Error for Lj92Error {}

/// The crate error type.
#[derive(Debug)]
#[non_exhaustive]
pub enum Error {
    /// Underlying I/O failure.
    #[cfg(feature = "std")]
    Io(std::io::Error),
    /// The file does not start with an `MLVI` block.
    NotMlv {
        /// Path or description of the source.
        path: String,
    },
    /// A block header or payload is inconsistent.
    BadBlock {
        /// Byte offset of the block within its chunk.
        offset: u64,
        /// Human-readable reason.
        reason: &'static str,
    },
    /// The chunk ends in the middle of a block (camera lost power, card
    /// full). The clip layer treats this as a warning and stops scanning.
    Truncated {
        /// Byte offset of the incomplete block.
        offset: u64,
    },
    /// A block required for decoding (`RAWI`, `MLVI`) was not found.
    MissingBlock(Tag),
    /// A sibling chunk carries a different `fileGuid`.
    ChunkGuidMismatch {
        /// Path of the chunk that disagrees.
        path: String,
        /// GUID of the main `.MLV` file.
        expected: u64,
        /// GUID found in the offending chunk.
        got: u64,
    },
    /// Frame index out of `0..frame_count()`.
    FrameOutOfRange(usize),
    /// Uncompressed payload is shorter than `width * height * bpp / 8`.
    FrameTooShort {
        /// Bytes required.
        expected: usize,
        /// Bytes present.
        got: usize,
    },
    /// Decoded sample count does not match `xRes * yRes`, or the caller's
    /// destination slice has the wrong length.
    GeometryMismatch {
        /// Expected sample count.
        expected: usize,
        /// Actual sample count.
        got: usize,
    },
    /// Recognised but unsupported encoding.
    Unsupported(Unsupported),
    /// Lossless JPEG decode failure.
    Lj92(Lj92Error),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            #[cfg(feature = "std")]
            Error::Io(e) => write!(f, "I/O error: {e}"),
            Error::NotMlv { path } => write!(f, "{path}: not an MLV file (no MLVI header)"),
            Error::BadBlock { offset, reason } => {
                write!(f, "bad block at offset {offset}: {reason}")
            }
            Error::Truncated { offset } => {
                write!(f, "file truncated inside block at offset {offset}")
            }
            Error::MissingBlock(tag) => write!(f, "required block {tag} not found"),
            Error::ChunkGuidMismatch {
                path,
                expected,
                got,
            } => write!(
                f,
                "chunk {path} has fileGuid {got:#018x}, expected {expected:#018x}"
            ),
            Error::FrameOutOfRange(i) => write!(f, "frame index {i} out of range"),
            Error::FrameTooShort { expected, got } => {
                write!(f, "frame payload is {got} bytes, need {expected}")
            }
            Error::GeometryMismatch { expected, got } => {
                write!(f, "decoded {got} samples, expected {expected}")
            }
            Error::Unsupported(u) => write!(f, "unsupported: {u}"),
            Error::Lj92(e) => write!(f, "LJ92 decode failed: {e}"),
        }
    }
}

impl core::error::Error for Error {
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        match self {
            #[cfg(feature = "std")]
            Error::Io(e) => Some(e),
            Error::Lj92(e) => Some(e),
            _ => None,
        }
    }
}

#[cfg(feature = "std")]
impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::Io(e)
    }
}

impl From<Lj92Error> for Error {
    fn from(e: Lj92Error) -> Self {
        Error::Lj92(e)
    }
}

impl From<Unsupported> for Error {
    fn from(u: Unsupported) -> Self {
        Error::Unsupported(u)
    }
}
