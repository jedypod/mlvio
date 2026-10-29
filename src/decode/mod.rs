// SPDX-License-Identifier: MIT OR Apache-2.0
//! Pixel decoders. No I/O; everything works on byte slices.
//!
//! - [`unpack`](mod@unpack): Magic Lantern's 10/12/14-bit packed Bayer layouts to `u16`.
//! - [`lj92`]: lossless JPEG (ITU T.81 process 14, SOF3) decoder.

pub mod lj92;
pub mod unpack;

pub use lj92::{Lj92Header, decode_into as lj92_decode_into, parse as lj92_parse};
pub use unpack::{unpack, unpack10, unpack12, unpack14};
