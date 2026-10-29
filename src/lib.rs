// SPDX-License-Identifier: MIT OR Apache-2.0
//! # mlvio
//!
//! Reader for Magic Lantern Video (MLV) raw clips.
//!
//! The crate is layered, bottom up:
//!
//! - [`block`]: block tags, the 16-byte block header and one typed struct per
//!   metadata block, all parsed from byte slices. No I/O.
//!
//! The output contract is a `u16` Bayer mosaic plane at the stored bit depth
//! plus enough metadata ([`ClipInfo`]) to linearise and colour it.
//!
//! Scope of 0.1: raw Bayer video, uncompressed or LJ92, split chunks,
//! metadata. Audio, LZMA/delta, JPEG2000, McRAW and dual-ISO processing are
//! detected and reported but not decoded.

#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]
#![warn(missing_docs)]

extern crate alloc;

pub mod block;
mod error;

pub use error::{Error, Lj92Error, Result, Unsupported};

pub use block::{Block, BlockHeader, Cfa, Tag};

