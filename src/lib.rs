// SPDX-License-Identifier: MIT OR Apache-2.0
//! # mlvio
//!
//! Reader for Magic Lantern Video (MLV) raw clips.
//!
//! The crate is layered, bottom up:
//!
//! - [`block`]: block tags, the 16-byte block header and one typed struct per
//!   metadata block, all parsed from byte slices. No I/O.
//! - [`scan`]: the [`ReadAt`] positional-read trait and a block
//!   iterator that walks a chunk reading only headers.
//! - [`clip`]: chunk discovery (`NAME.MLV` + `NAME.M00..M99`), the frame
//!   index, metadata timelines and the [`Clip`] API.
//! - [`decode`]: 10/12/14-bit unpackers and the lossless JPEG (LJ92) decoder.
//!
//! The output contract is a `u16` Bayer mosaic plane at the stored bit depth
//! plus enough metadata ([`ClipInfo`]) to linearise and colour it.
//!
//! ```no_run
//! use mlvio::Clip;
//!
//! let clip = Clip::open("CLIP.MLV")?;
//! let info = clip.info();
//! println!("{}x{} {}-bit, {} frames", info.width, info.height,
//!          info.bits_per_pixel, clip.frame_count());
//! let frame = clip.frame(0)?;
//! assert_eq!(frame.data.len(), frame.width * frame.height);
//! # Ok::<(), mlvio::Error>(())
//! ```
//!
//! Scope of 0.1: raw Bayer video, uncompressed or LJ92, split chunks,
//! metadata. Audio, LZMA/delta, JPEG2000, McRAW and dual-ISO processing are
//! detected and reported but not decoded.

#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]
#![warn(missing_docs)]

extern crate alloc;

pub mod block;
pub mod decode;
mod error;

#[cfg(feature = "std")]
pub mod clip;
#[cfg(feature = "std")]
pub mod scan;

pub use error::{Error, Lj92Error, Result, Unsupported};

pub use block::{Block, BlockHeader, Cfa, Tag};

#[cfg(feature = "std")]
pub use clip::{
    CameraId, Clip, ClipInfo, Compression, Frame, FrameMeta, FrameRef, FrameScratch, MetaRef, Rect,
    Warning, WarningKind,
};
#[cfg(feature = "std")]
pub use scan::ReadAt;
