// SPDX-License-Identifier: MIT OR Apache-2.0
//! Block tags, the common header and one typed struct per block, parsed from
//! byte slices. This module performs no I/O and is `no_std + alloc` clean.
//!
//! Parsing is tolerant: every `from_bytes` accepts a payload shorter than the
//! struct Magic Lantern writes today, fills missing trailing fields with
//! zero/default, and records that in a `truncated` flag where it matters.
//! This mirrors the `min(blockSize, sizeof(struct))` copy that ML's own
//! readers perform, because firmware revisions have appended fields.

pub mod cursor;
mod tag;

pub use cursor::{Cursor, c_str};
pub use tag::{BlockHeader, Tag};
