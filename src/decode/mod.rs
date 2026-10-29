// SPDX-License-Identifier: MIT OR Apache-2.0
//! Pixel decoders. No I/O; everything works on byte slices.
//!
//! - [`unpack`](mod@unpack): Magic Lantern's 10/12/14-bit packed Bayer layouts to `u16`.

pub mod unpack;

pub use unpack::{unpack, unpack10, unpack12, unpack14};
