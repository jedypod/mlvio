// SPDX-License-Identifier: MIT OR Apache-2.0
//! Smoke test for the synthetic MLV writer used by the integration tests.

mod common;

use common::{MlvWriter, lcg_pixels};
use mlvio::Tag;
use mlvio::scan::BlockIter;

#[test]
fn two_frame_14bit_clip() {
    let (w, h, bpp) = (64u16, 32u16, 14u32);
    let n = usize::from(w) * usize::from(h);
    let frame_bytes = n * bpp as usize / 8;

    let mut mw = MlvWriter::new(0x1234_5678_9abc_def0, 0, 1);
    mw.rawi(w, h, bpp, 2048, 15000);
    mw.vidf(1000, 0, 0, &lcg_pixels(n, bpp, 1), bpp);
    mw.vidf_padded(41000, 1, 8, &lcg_pixels(n, bpp, 2), bpp, 4);
    let bytes = mw.finish();

    let vidf0 = 32 + frame_bytes;
    let vidf1 = 32 + 8 + frame_bytes + 4;
    assert_eq!(bytes.len(), 52 + 180 + vidf0 + vidf1);
    // videoFrameCount was bumped by each vidf call.
    assert_eq!(u32::from_le_bytes(bytes[36..40].try_into().unwrap()), 2);

    let src: &[u8] = &bytes;
    let blocks: Vec<_> = BlockIter::new(src).collect::<Result<_, _>>().unwrap();
    let tags: Vec<Tag> = blocks.iter().map(|(_, h)| h.tag).collect();
    assert_eq!(tags, [Tag::MLVI, Tag::RAWI, Tag::VIDF, Tag::VIDF]);
    assert_eq!(blocks[0].1.version_bytes(), *b"v2.0\0\0\0\0");
    assert_eq!(blocks[1].1.size, 180);
    assert_eq!(blocks[2].0, 52 + 180);
    assert_eq!(blocks[2].1.size as usize, vidf0);
    assert_eq!(blocks[3].1.size as usize, vidf1);
    assert_eq!(blocks[3].1.timestamp, 41000);
}
