// SPDX-License-Identifier: MIT OR Apache-2.0
//! Unpackers for Magic Lantern's uncompressed 10/12/14-bit Bayer layouts.
//!
//! The packed data is a sequence of little-endian `u16` words. Pixel bits
//! form a big-endian bitstream over those words: the first pixel occupies the
//! most significant bits of the first word, and a pixel that does not fit in
//! the remainder of a word continues in the top bits of the next one. Groups
//! repeat every 8 pixels / 14 bytes (14-bit), 4 pixels / 6 bytes (12-bit) and
//! 8 pixels / 10 bytes (10-bit).
//!
//! Output samples are at the stored bit depth (`0..2^bpp`), never rescaled.

use alloc::vec::Vec;

use crate::{Error, Unsupported};

/// Number of source bytes needed to hold `pixels` samples of `bpp` bits.
///
/// The stream is made of whole little-endian `u16` words, so this is
/// `ceil(pixels * bpp / 16) * 2`. For every real frame (`pixels * bpp` a
/// multiple of 16) it equals `pixels * bpp / 8`; for odd tails it can be one
/// byte more than `ceil(pixels * bpp / 8)` because the last bits sit in the
/// high byte of a word, which is the *second* byte in memory.
#[inline]
pub fn needed_bytes(pixels: usize, bpp: u32) -> usize {
    let bits = pixels.saturating_mul(bpp as usize);
    bits.div_ceil(16).saturating_mul(2)
}

#[inline(always)]
fn word(b: &[u8], i: usize) -> u16 {
    u16::from_le_bytes([b[2 * i], b[2 * i + 1]])
}

/// Slow path: read `count` pixels of `bpp` bits starting at pixel `first`
/// of `src`'s bitstream. Used only for tails shorter than a group.
fn unpack_tail(bpp: u32, src: &[u8], first: usize, dst: &mut [u16]) {
    let bpp = bpp as usize;
    for (k, out) in dst.iter_mut().enumerate() {
        let start = (first + k) * bpp;
        let mut v: u32 = 0;
        for bit in start..start + bpp {
            let w = word(src, bit / 16);
            let b = (w >> (15 - (bit % 16))) & 1;
            v = (v << 1) | u32::from(b);
        }
        *out = v as u16;
    }
}

fn check_len(src: &[u8], dst: &[u16], bpp: u32) -> Result<(), Error> {
    let expected = needed_bytes(dst.len(), bpp);
    if src.len() < expected {
        return Err(Error::FrameTooShort {
            expected,
            got: src.len(),
        });
    }
    Ok(())
}

/// Unpack 14-bit data (8 pixels per 14 bytes) into `dst`.
///
/// Fails with [`Error::FrameTooShort`] if `src` is shorter than
/// [`needed_bytes`]`(dst.len(), 14)`. Extra source bytes are ignored.
pub fn unpack14(src: &[u8], dst: &mut [u16]) -> Result<(), Error> {
    check_len(src, dst, 14)?;
    let (out, tail) = dst.as_chunks_mut::<8>();
    let groups = out.len();
    for (o, s) in out.iter_mut().zip(src.as_chunks::<14>().0) {
        let w0 = word(s, 0);
        let w1 = word(s, 1);
        let w2 = word(s, 2);
        let w3 = word(s, 3);
        let w4 = word(s, 4);
        let w5 = word(s, 5);
        let w6 = word(s, 6);
        o[0] = w0 >> 2;
        o[1] = ((w0 & 0x3) << 12) | (w1 >> 4);
        o[2] = ((w1 & 0xf) << 10) | (w2 >> 6);
        o[3] = ((w2 & 0x3f) << 8) | (w3 >> 8);
        o[4] = ((w3 & 0xff) << 6) | (w4 >> 10);
        o[5] = ((w4 & 0x3ff) << 4) | (w5 >> 12);
        o[6] = ((w5 & 0xfff) << 2) | (w6 >> 14);
        o[7] = w6 & 0x3fff;
    }
    if !tail.is_empty() {
        unpack_tail(14, &src[groups * 14..], 0, tail);
    }
    Ok(())
}

/// Unpack 12-bit data (4 pixels per 6 bytes) into `dst`.
///
/// Fails with [`Error::FrameTooShort`] if `src` is shorter than
/// [`needed_bytes`]`(dst.len(), 12)`. Extra source bytes are ignored.
pub fn unpack12(src: &[u8], dst: &mut [u16]) -> Result<(), Error> {
    check_len(src, dst, 12)?;
    let (out, tail) = dst.as_chunks_mut::<4>();
    let groups = out.len();
    for (o, s) in out.iter_mut().zip(src.as_chunks::<6>().0) {
        let w0 = word(s, 0);
        let w1 = word(s, 1);
        let w2 = word(s, 2);
        o[0] = w0 >> 4;
        o[1] = ((w0 & 0xf) << 8) | (w1 >> 8);
        o[2] = ((w1 & 0xff) << 4) | (w2 >> 12);
        o[3] = w2 & 0xfff;
    }
    if !tail.is_empty() {
        unpack_tail(12, &src[groups * 6..], 0, tail);
    }
    Ok(())
}

/// Unpack 10-bit data (8 pixels per 10 bytes) into `dst`.
///
/// Fails with [`Error::FrameTooShort`] if `src` is shorter than
/// [`needed_bytes`]`(dst.len(), 10)`. Extra source bytes are ignored.
pub fn unpack10(src: &[u8], dst: &mut [u16]) -> Result<(), Error> {
    check_len(src, dst, 10)?;
    let (out, tail) = dst.as_chunks_mut::<8>();
    let groups = out.len();
    for (o, s) in out.iter_mut().zip(src.as_chunks::<10>().0) {
        let w0 = word(s, 0);
        let w1 = word(s, 1);
        let w2 = word(s, 2);
        let w3 = word(s, 3);
        let w4 = word(s, 4);
        o[0] = w0 >> 6;
        o[1] = ((w0 & 0x3f) << 4) | (w1 >> 12);
        o[2] = (w1 >> 2) & 0x3ff;
        o[3] = ((w1 & 0x3) << 8) | (w2 >> 8);
        o[4] = ((w2 & 0xff) << 2) | (w3 >> 14);
        o[5] = (w3 >> 4) & 0x3ff;
        o[6] = ((w3 & 0xf) << 6) | (w4 >> 10);
        o[7] = w4 & 0x3ff;
    }
    if !tail.is_empty() {
        unpack_tail(10, &src[groups * 10..], 0, tail);
    }
    Ok(())
}

/// Unpack `bpp`-bit data into `dst`, dispatching to [`unpack10`],
/// [`unpack12`] or [`unpack14`].
///
/// Any other depth fails with [`Unsupported::BitDepth`].
pub fn unpack(bpp: u32, src: &[u8], dst: &mut [u16]) -> Result<(), Error> {
    match bpp {
        10 => unpack10(src, dst),
        12 => unpack12(src, dst),
        14 => unpack14(src, dst),
        _ => Err(Error::Unsupported(Unsupported::BitDepth(bpp))),
    }
}

/// Pack samples into Magic Lantern's layout; the exact inverse of [`unpack`].
///
/// This is a test and fixture helper (it is slow, bit by bit) and not part of
/// the reading API. It appends [`needed_bytes`]`(src.len(), bpp)` bytes to
/// `dst`, masks each sample to its low `bpp` bits, and zero-fills unused bits
/// in the final word. Any `bpp` in `1..=16` is accepted.
///
/// # Panics
///
/// Panics if `bpp` is 0 or greater than 16.
pub fn pack(bpp: u32, src: &[u16], dst: &mut Vec<u8>) {
    assert!((1..=16).contains(&bpp), "pack: bpp must be 1..=16");
    let n_words = needed_bytes(src.len(), bpp) / 2;
    let mut words: Vec<u16> = alloc::vec![0; n_words];
    let bpp = bpp as usize;
    let mut bit = 0usize;
    for &v in src {
        for i in (0..bpp).rev() {
            if (v >> i) & 1 != 0 {
                words[bit / 16] |= 1 << (15 - (bit % 16));
            }
            bit += 1;
        }
    }
    dst.reserve(n_words * 2);
    for w in words {
        dst.extend_from_slice(&w.to_le_bytes());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    fn lcg(n: usize, bpp: u32, mut seed: u32) -> Vec<u16> {
        let mask = (1u32 << bpp) - 1;
        (0..n)
            .map(|_| {
                seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                ((seed >> 8) & mask) as u16
            })
            .collect()
    }

    #[test]
    fn round_trip_all_depths() {
        for bpp in [10u32, 12, 14] {
            for n in [
                0usize, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 23, 64, 1001, 4096,
            ] {
                let px = lcg(n, bpp, 0xC0FFEE ^ (n as u32) ^ bpp);
                let mut packed = Vec::new();
                pack(bpp, &px, &mut packed);
                assert_eq!(packed.len(), needed_bytes(n, bpp));
                let mut out = vec![0xffffu16; n];
                unpack(bpp, &packed, &mut out).unwrap();
                assert_eq!(out, px, "bpp {bpp} n {n}");
            }
        }
    }

    #[test]
    fn extremes_round_trip() {
        for bpp in [10u32, 12, 14] {
            let max = ((1u32 << bpp) - 1) as u16;
            let px: Vec<u16> = (0..37).map(|i| if i % 2 == 0 { max } else { 0 }).collect();
            let mut packed = Vec::new();
            pack(bpp, &px, &mut packed);
            let mut out = vec![0u16; px.len()];
            unpack(bpp, &packed, &mut out).unwrap();
            assert_eq!(out, px);
        }
    }

    #[test]
    fn group_sizes() {
        assert_eq!(needed_bytes(8, 14), 14);
        assert_eq!(needed_bytes(4, 12), 6);
        assert_eq!(needed_bytes(8, 10), 10);
        assert_eq!(needed_bytes(0, 14), 0);
        assert_eq!(needed_bytes(1, 14), 2);
        // 4 px * 10 bit = 40 bits: the last 8 bits live in the high byte of
        // word 2, which is byte 5, so 6 bytes are needed.
        assert_eq!(needed_bytes(4, 10), 6);
        assert_eq!(needed_bytes(1920 * 1080, 14), 1920 * 1080 * 14 / 8);
    }

    #[test]
    fn hand_vector_14bit() {
        // Pixels 0..=7, 14 bits each, as one 112-bit big-endian stream:
        // word0 = a<<2 | b>>12               = 0<<2 | 0       = 0x0000
        // word1 = (b&0xfff)<<4 | c>>10       = 1<<4 | 0       = 0x0010
        // word2 = (c&0x3ff)<<6 | d>>8        = 2<<6 | 0       = 0x0080
        // word3 = (d&0xff)<<8 | e>>6         = 3<<8 | 0       = 0x0300
        // word4 = (e&0x3f)<<10 | f>>4        = 4<<10 | 0      = 0x1000
        // word5 = (f&0xf)<<12 | g>>2         = 5<<12 | 1      = 0x5001
        // word6 = (g&0x3)<<14 | h            = 2<<14 | 7      = 0x8007
        let expected: [u8; 14] = [
            0x00, 0x00, 0x10, 0x00, 0x80, 0x00, 0x00, 0x03, 0x00, 0x10, 0x01, 0x50, 0x07, 0x80,
        ];
        let px: Vec<u16> = (0..8).collect();
        let mut packed = Vec::new();
        pack(14, &px, &mut packed);
        assert_eq!(packed, expected);
        let mut out = [0u16; 8];
        unpack14(&expected, &mut out).unwrap();
        assert_eq!(out, [0, 1, 2, 3, 4, 5, 6, 7]);
    }

    #[test]
    fn too_short() {
        let src = [0u8; 13];
        let mut dst = [0u16; 8];
        match unpack14(&src, &mut dst) {
            Err(Error::FrameTooShort {
                expected: 14,
                got: 13,
            }) => {}
            other => panic!("unexpected {other:?}"),
        }
        let mut dst = [0u16; 5];
        match unpack12(&src[..7], &mut dst) {
            Err(Error::FrameTooShort {
                expected: 8,
                got: 7,
            }) => {}
            other => panic!("unexpected {other:?}"),
        }
        let mut dst = [0u16; 16];
        match unpack10(&src, &mut dst) {
            Err(Error::FrameTooShort {
                expected: 20,
                got: 13,
            }) => {}
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn unsupported_depth() {
        let mut dst = [0u16; 4];
        match unpack(16, &[0u8; 8], &mut dst) {
            Err(Error::Unsupported(Unsupported::BitDepth(16))) => {}
            other => panic!("unexpected {other:?}"),
        }
    }
}
