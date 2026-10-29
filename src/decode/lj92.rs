// SPDX-License-Identifier: MIT
//! Lossless JPEG (ITU-T T.81 process 14, SOF3) decoder.
//!
//! Magic Lantern's `lossless.c` drives the camera's JPEG core to produce a
//! complete LJ92 stream per frame: SOI, one or more DHT, SOF3, SOS, entropy
//! data, EOI. The frame is encoded as two interleaved components with a
//! geometry of `width x height/2`, so the SOF3 dimensions do not match the
//! `RAWI` resolution; only `width * height * components` does. This decoder
//! therefore trusts the stream's own geometry and writes samples in scan
//! order to a flat buffer. The caller checks the product.
//!
//! Derived from liblj92 by Andrew Baldwin (MIT) via the Rust translation in
//! the `libmlv` crate by Ilia Sibiryakov (MIT). Rewritten to be bounds safe
//! on corrupt input, to accept all four DHT slots, and to support the general
//! predictor set with a per-predictor specialised inner loop.
//!
//! ```text
//! lj92.rs
//! (c) Andrew Baldwin 2014
//! (c) Ilia Sibiryakov 2024 (translated to Rust)
//! (c) 2026 mlvio contributors
//!
//! Permission is hereby granted, free of charge, to any person obtaining a
//! copy of this software and associated documentation files (the "Software"),
//! to deal in the Software without restriction, including without limitation
//! the rights to use, copy, modify, merge, publish, distribute, sublicense,
//! and/or sell copies of the Software, and to permit persons to whom the
//! Software is furnished to do so, subject to the following conditions:
//!
//! The above copyright notice and this permission notice shall be included in
//! all copies or substantial portions of the Software.
//!
//! THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//! IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//! FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
//! THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//! LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
//! FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
//! DEALINGS IN THE SOFTWARE.
//! ```

use alloc::vec::Vec;

use crate::Lj92Error;

type Result<T> = core::result::Result<T, Lj92Error>;

const M_SOI: u8 = 0xD8;
const M_EOI: u8 = 0xD9;
const M_SOS: u8 = 0xDA;
const M_DHT: u8 = 0xC4;
const M_DRI: u8 = 0xDD;
const M_SOF3: u8 = 0xC3;

/// Maximum components per scan this decoder accepts.
pub const MAX_COMPONENTS: usize = 4;

/// Lookup entry for a bit pattern that is not a code: SSSS 0 with a code
/// length of 63, which no real code can have. "Consuming" it wraps the bit
/// reader's count past 64 and the row check reports `BadCode`.
const BAD_CODE_ENTRY: u16 = 63;

/// One Huffman table expanded to a direct lookup.
///
/// Entry `i` (indexed by the next `max_bits` bits of the stream) holds
/// `(ssss << 8) | code_length`, or [`BAD_CODE_ENTRY`] for a bit pattern that
/// is not a code.
#[derive(Debug, Clone, PartialEq, Eq)]
struct HuffTable {
    max_bits: u8,
    lut: Vec<u16>,
}

impl HuffTable {
    /// Build from the 16 `BITS` counts and the `HUFFVAL` list of one DHT table.
    fn build(bits: &[u8; 16], vals: &[u8]) -> Result<Self> {
        let total: usize = bits.iter().map(|&b| b as usize).sum();
        if total == 0 || total > vals.len() || total > 256 {
            return Err(Lj92Error::BadHuffmanTable);
        }
        let max_bits = 16 - bits.iter().rev().position(|&b| b != 0).unwrap_or(16) as u8;
        if max_bits == 0 {
            return Err(Lj92Error::BadHuffmanTable);
        }
        let size = 1usize << max_bits;
        let mut lut = alloc::vec![BAD_CODE_ENTRY; size];

        // Canonical code assignment: codes of length L are consecutive,
        // starting at (previous code + 1) << 1.
        let mut code: u32 = 0;
        let mut vi = 0usize;
        for len in 1..=max_bits {
            let count = bits[len as usize - 1] as u32;
            for _ in 0..count {
                let val = vals[vi];
                vi += 1;
                // Lossless JPEG only defines SSSS 0..=16.
                if val > 16 {
                    return Err(Lj92Error::BadHuffmanTable);
                }
                let shift = max_bits - len;
                let start = (code as usize) << shift;
                let end = start + (1usize << shift);
                if end > size {
                    // Over-subscribed code space.
                    return Err(Lj92Error::BadHuffmanTable);
                }
                let entry = ((val as u16) << 8) | len as u16;
                for e in &mut lut[start..end] {
                    *e = entry;
                }
                code += 1;
            }
            code <<= 1;
        }
        Ok(HuffTable { max_bits, lut })
    }
}

/// Borrowed view of a [`HuffTable`] held by value in the decode loop so the
/// table pointer, length and code width stay in registers.
#[derive(Clone, Copy)]
struct TableRef<'a> {
    lut: &'a [u16],
    max_bits: u32,
}

impl HuffTable {
    fn as_ref(&self) -> TableRef<'_> {
        TableRef {
            lut: &self.lut,
            max_bits: self.max_bits as u32,
        }
    }
}

/// Parsed stream header: geometry, tables and the location of the scan data.
///
/// Obtain with [`parse`], then call [`decode_into`]. The header is `Send`
/// and may be reused only for the same byte slice it was parsed from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Lj92Header {
    /// Samples per line (`X` in SOF3).
    pub width: u16,
    /// Number of lines (`Y` in SOF3).
    pub height: u16,
    /// Sample precision in bits (`P`, 2..=16).
    pub precision: u8,
    /// Number of image components in the scan (`Ns`).
    pub components: u8,
    /// Predictor selection value from SOS (`Ss`, 0..=7).
    pub predictor: u8,
    /// Point transform (`Al` in SOS). Lowers the initial prediction; samples
    /// are returned as coded, not shifted back.
    pub point_transform: u8,
    /// Table selector per scan component.
    table_of: [u8; MAX_COMPONENTS],
    tables: [Option<HuffTable>; MAX_COMPONENTS],
    /// Offset of the first entropy-coded byte.
    scan_start: usize,
}

impl Lj92Header {
    /// Total samples the decoder will write: `width * height * components`.
    #[inline]
    pub fn sample_count(&self) -> usize {
        self.width as usize * self.height as usize * self.components as usize
    }
}

#[inline]
fn be16(data: &[u8], pos: usize) -> Result<u16> {
    match (data.get(pos), data.get(pos + 1)) {
        (Some(&a), Some(&b)) => Ok(u16::from_be_bytes([a, b])),
        _ => Err(Lj92Error::Truncated),
    }
}

/// Return the marker segment body starting at `pos` (which points at the
/// 2-byte length) and the position just after the segment.
fn segment(data: &[u8], pos: usize) -> Result<(&[u8], usize)> {
    let len = be16(data, pos)? as usize;
    if len < 2 {
        return Err(Lj92Error::Truncated);
    }
    let end = pos.checked_add(len).ok_or(Lj92Error::Truncated)?;
    data.get(pos + 2..end)
        .map(|s| (s, end))
        .ok_or(Lj92Error::Truncated)
}

/// Parse the marker segments up to and including SOS.
///
/// Accepts leading garbage before SOI (skips to the first `FF D8`), any
/// number of DHT segments (each may define several tables), exactly one
/// SOF3, and skips APPn/COM/DQT and other segments. DRI or a non-lossless
/// SOF are rejected.
pub fn parse(data: &[u8]) -> Result<Lj92Header> {
    // Find SOI.
    let mut pos = data
        .windows(2)
        .position(|w| w == [0xFF, M_SOI])
        .ok_or(Lj92Error::NoSoi)?
        + 2;

    let mut tables: [Option<HuffTable>; MAX_COMPONENTS] = [None, None, None, None];
    let mut sof: Option<(u16, u16, u8, Vec<u8>)> = None; // width, height, precision, component ids

    loop {
        // Expect a marker: FF (possibly several fill FFs) then a code.
        if *data.get(pos).ok_or(Lj92Error::Truncated)? != 0xFF {
            return Err(Lj92Error::Truncated);
        }
        while data.get(pos) == Some(&0xFF) {
            pos += 1;
        }
        let marker = *data.get(pos).ok_or(Lj92Error::Truncated)?;
        pos += 1;

        match marker {
            M_DHT => {
                let (body, next) = segment(data, pos)?;
                let mut p = 0;
                while p < body.len() {
                    let tc_th = body[p];
                    let class = tc_th >> 4;
                    let id = (tc_th & 0x0F) as usize;
                    if class != 0 || id >= MAX_COMPONENTS {
                        return Err(Lj92Error::BadHuffmanTable);
                    }
                    let bits: [u8; 16] = body
                        .get(p + 1..p + 17)
                        .and_then(|s| s.try_into().ok())
                        .ok_or(Lj92Error::BadHuffmanTable)?;
                    let total: usize = bits.iter().map(|&b| b as usize).sum();
                    let vals = body
                        .get(p + 17..p + 17 + total)
                        .ok_or(Lj92Error::BadHuffmanTable)?;
                    tables[id] = Some(HuffTable::build(&bits, vals)?);
                    p += 17 + total;
                }
                pos = next;
            }
            M_SOF3 => {
                let (body, next) = segment(data, pos)?;
                if body.len() < 6 {
                    return Err(Lj92Error::BadFrameHeader);
                }
                let precision = body[0];
                let height = u16::from_be_bytes([body[1], body[2]]);
                let width = u16::from_be_bytes([body[3], body[4]]);
                let nf = body[5] as usize;
                if !(2..=16).contains(&precision) || width == 0 || height == 0 {
                    return Err(Lj92Error::BadFrameHeader);
                }
                if nf == 0 || nf > MAX_COMPONENTS || body.len() < 6 + 3 * nf {
                    return Err(Lj92Error::BadFrameHeader);
                }
                let ids = (0..nf).map(|i| body[6 + 3 * i]).collect();
                sof = Some((width, height, precision, ids));
                pos = next;
            }
            0xC0..=0xC2 | 0xC5..=0xC7 | 0xC9..=0xCB | 0xCD..=0xCF => {
                return Err(Lj92Error::NotLossless(marker));
            }
            M_DRI => return Err(Lj92Error::RestartMarkers),
            M_SOS => {
                let (body, next) = segment(data, pos)?;
                let (width, height, precision, ids) = sof.ok_or(Lj92Error::BadFrameHeader)?;
                let ns = *body.first().ok_or(Lj92Error::BadScanHeader)? as usize;
                if ns == 0 || ns > MAX_COMPONENTS || ns != ids.len() {
                    return Err(Lj92Error::BadScanHeader);
                }
                if body.len() < 1 + 2 * ns + 3 {
                    return Err(Lj92Error::BadScanHeader);
                }
                let mut table_of = [0u8; MAX_COMPONENTS];
                for i in 0..ns {
                    let cs = body[1 + 2 * i];
                    let td = body[2 + 2 * i] >> 4;
                    if !ids.contains(&cs) || td as usize >= MAX_COMPONENTS {
                        return Err(Lj92Error::BadScanHeader);
                    }
                    if tables[td as usize].is_none() {
                        return Err(Lj92Error::BadHuffmanTable);
                    }
                    table_of[i] = td;
                }
                let predictor = body[1 + 2 * ns];
                let point_transform = body[3 + 2 * ns] & 0x0F;
                if predictor > 7 || point_transform >= precision {
                    return Err(Lj92Error::BadScanHeader);
                }
                let samples = (width as u64) * (height as u64) * (ns as u64);
                if samples > u32::MAX as u64 {
                    return Err(Lj92Error::TooLarge);
                }
                return Ok(Lj92Header {
                    width,
                    height,
                    precision,
                    components: ns as u8,
                    predictor,
                    point_transform,
                    table_of,
                    tables,
                    scan_start: next,
                });
            }
            M_EOI => return Err(Lj92Error::BadScanHeader),
            0xD0..=0xD7 => return Err(Lj92Error::RestartMarkers),
            // Standalone markers without a length field.
            0x01 | M_SOI => {}
            _ => {
                let (_, next) = segment(data, pos)?;
                pos = next;
            }
        }
    }
}

/// Bit reader over entropy-coded data with `FF 00` unstuffing.
///
/// The accumulator is left-aligned: the next bit of the stream is bit 63 of
/// `acc`, and `cnt` bits are valid. [`BitReader::refill`] guarantees at
/// least 32 bits, so after one refill a full symbol (a code of at most 16
/// bits plus up to 16 value bits) can be consumed without further checks.
///
/// Refilling takes a fast path when eight raw bytes are available and none of
/// them is `0xFF`: the bytes are loaded as one big-endian word and inserted
/// whole. Otherwise it falls back to a byte-at-a-time loop that unstuffs
/// `FF 00`, stops at any other marker and at the end of the data, and feeds
/// zero bits from then on while counting them in `pad_bits`;
/// [`BitReader::finish`] turns an overrun into an error.
///
/// Decoding never fails mid-symbol. An unassigned code pattern hits
/// [`BAD_CODE_ENTRY`], which "consumes" 63 bits and so drives `cnt` past 64
/// (it can never legitimately exceed 64); the row loop checks that once per
/// row, which keeps the `Result` out of the per-sample path.
struct BitReader<'a> {
    data: &'a [u8],
    pos: usize,
    /// Left-aligned bit accumulator.
    acc: u64,
    /// Valid bits in `acc`.
    cnt: u32,
    /// Zero bits fed after the real data ended.
    pad_bits: u32,
    /// True once a marker (other than a stuffed `FF 00`) was reached.
    hit_marker: bool,
}

/// True if any byte of `w` is `0xFF`.
#[inline(always)]
fn has_ff_byte(w: u64) -> bool {
    // Classic "has zero byte" test applied to the complement.
    let x = !w;
    const LO: u64 = 0x0101_0101_0101_0101;
    const HI: u64 = 0x8080_8080_8080_8080;
    (x.wrapping_sub(LO) & !x & HI) != 0
}

impl<'a> BitReader<'a> {
    fn new(data: &'a [u8], pos: usize) -> Self {
        BitReader {
            data,
            pos,
            acc: 0,
            cnt: 0,
            pad_bits: 0,
            hit_marker: false,
        }
    }

    /// Ensure at least 32 valid bits are buffered (enough for one symbol:
    /// a code of at most 16 bits plus at most 16 value bits). Refilling only
    /// when the accumulator is half empty lets the fast path insert four or
    /// more bytes at a time.
    #[inline(always)]
    fn refill(&mut self) {
        if self.cnt >= 32 {
            return;
        }
        if let Some(chunk) = self.data.get(self.pos..self.pos + 8) {
            let w = u64::from_be_bytes(chunk.try_into().expect("8-byte slice"));
            if !has_ff_byte(w) {
                // Insert as many whole bytes as fit above the valid bits.
                let nbytes = (64 - self.cnt) >> 3;
                let keep = nbytes << 3;
                self.acc |= (w >> (64 - keep)) << (64 - keep - self.cnt);
                self.cnt += keep;
                self.pos += nbytes as usize;
                return;
            }
        }
        self.refill_slow();
    }

    /// Byte-at-a-time refill: handles `FF 00` stuffing, markers and the end
    /// of the data.
    #[cold]
    #[inline(never)]
    fn refill_slow(&mut self) {
        while self.cnt <= 56 {
            let b = self.next_byte();
            self.acc |= (b as u64) << (56 - self.cnt);
            self.cnt += 8;
        }
    }

    fn next_byte(&mut self) -> u8 {
        if self.hit_marker {
            self.pad_bits += 8;
            return 0;
        }
        match self.data.get(self.pos) {
            Some(&0xFF) => match self.data.get(self.pos + 1) {
                Some(&0x00) => {
                    self.pos += 2;
                    0xFF
                }
                _ => {
                    self.hit_marker = true;
                    self.pad_bits += 8;
                    0
                }
            },
            Some(&b) => {
                self.pos += 1;
                b
            }
            None => {
                self.hit_marker = true;
                self.pad_bits += 8;
                0
            }
        }
    }

    /// Decode one difference value using `t`.
    ///
    /// Branch free apart from the refill check. An unassigned bit pattern
    /// hits [`BAD_CODE_ENTRY`], which poisons `cnt`; the caller checks it
    /// once per row.
    #[inline(always)]
    fn next_diff(&mut self, t: TableRef<'_>) -> i32 {
        self.refill();
        let idx = (self.acc >> (64 - t.max_bits)) as usize;
        // `idx < 1 << max_bits == lut.len()` by construction; the fallback
        // only exists to keep the index in bounds without a panic path.
        let entry = t.lut.get(idx).copied().unwrap_or(BAD_CODE_ENTRY) as u32;
        let len = entry & 0xFF;
        let ssss = entry >> 8;
        // SSSS 16 carries no value bits and means +32768; otherwise SSSS
        // value bits follow the code.
        let extra = ssss & 15;
        // `>> 1 >> (63 - extra)` reads `extra` bits and is well defined for
        // `extra == 0`.
        let v = ((self.acc << len) >> 1 >> (63 - extra)) as i32;
        self.acc <<= len + extra;
        // Wraps only for `BAD_CODE_ENTRY`, which `check_row` detects.
        self.cnt = self.cnt.wrapping_sub(len + extra);
        // Values below 2^(extra-1) are negative: v - 2^extra + 1.
        let half = (1i32 << extra) >> 1;
        let neg = -((v < half) as i32);
        v + (neg & (1 - (1i32 << extra))) + (((ssss >> 4) as i32) << 15)
    }

    /// Per-row health check: a bad code is fatal, and a stream that ended
    /// more than a word ago is reported as truncated instead of producing a
    /// frame of garbage.
    #[inline(always)]
    fn check_row(&self) -> Result<()> {
        if self.cnt > 64 {
            Err(Lj92Error::BadCode)
        } else if self.pad_bits > 64 + self.cnt {
            Err(Lj92Error::ScanTruncated)
        } else {
            Ok(())
        }
    }

    /// Fail if more bits were consumed than the data contained.
    fn finish(self) -> Result<()> {
        if self.cnt > 64 {
            Err(Lj92Error::BadCode)
        } else if self.pad_bits > self.cnt {
            Err(Lj92Error::ScanTruncated)
        } else {
            Ok(())
        }
    }
}

/// Decode the scan described by `hdr` from `data` (the same slice passed to
/// [`parse`]) into `out`, which must hold exactly
/// [`Lj92Header::sample_count`] values.
///
/// Samples are written in scan order: row-major, components interleaved.
pub fn decode_into(data: &[u8], hdr: &Lj92Header, out: &mut [u16]) -> Result<()> {
    let expected = hdr.sample_count();
    if out.len() != expected {
        return Err(Lj92Error::OutputSize {
            expected,
            got: out.len(),
        });
    }
    let ncomp = hdr.components as usize;
    let mut tables: [TableRef<'_>; MAX_COMPONENTS] = [EMPTY_TABLE.as_ref(); MAX_COMPONENTS];
    for (slot, &t) in tables.iter_mut().zip(&hdr.table_of).take(ncomp) {
        *slot = hdr.tables[t as usize]
            .as_ref()
            .ok_or(Lj92Error::BadHuffmanTable)?
            .as_ref();
    }
    let mut br = BitReader::new(data, hdr.scan_start);
    let init = 1i32 << (hdr.precision - hdr.point_transform - 1);
    let p = Params {
        width: hdr.width as usize,
        ncomp,
        init,
    };
    match hdr.predictor {
        0 => decode_rows::<0>(&mut br, &tables, &p, out)?,
        1 => decode_rows::<1>(&mut br, &tables, &p, out)?,
        2 => decode_rows::<2>(&mut br, &tables, &p, out)?,
        3 => decode_rows::<3>(&mut br, &tables, &p, out)?,
        4 => decode_rows::<4>(&mut br, &tables, &p, out)?,
        5 => decode_rows::<5>(&mut br, &tables, &p, out)?,
        6 => decode_rows::<6>(&mut br, &tables, &p, out)?,
        7 => decode_rows::<7>(&mut br, &tables, &p, out)?,
        _ => return Err(Lj92Error::BadScanHeader),
    }
    br.finish()
}

static EMPTY_TABLE: HuffTable = HuffTable {
    max_bits: 0,
    lut: Vec::new(),
};

struct Params {
    width: usize,
    ncomp: usize,
    init: i32,
}

/// Predict from left `a`, above `b`, above-left `c`.
#[inline(always)]
fn predict<const P: u8>(a: i32, b: i32, c: i32) -> i32 {
    match P {
        0 => 0,
        1 => a,
        2 => b,
        3 => c,
        4 => a + b - c,
        5 => a + ((b - c) >> 1),
        6 => b + ((a - c) >> 1),
        _ => (a + b) >> 1,
    }
}

fn decode_rows<const P: u8>(
    br: &mut BitReader<'_>,
    tables: &[TableRef<'_>; MAX_COMPONENTS],
    p: &Params,
    out: &mut [u16],
) -> Result<()> {
    let ncomp = p.ncomp;
    let row_len = p.width * ncomp;
    let mut rows = out.chunks_exact_mut(row_len);

    // Row 0: first sample of each component from the initial value, the
    // rest from the left neighbour of the same component.
    let first = rows.next().ok_or(Lj92Error::BadFrameHeader)?;
    for (c, s) in first.iter_mut().take(ncomp).enumerate() {
        *s = (p.init + br.next_diff(tables[c])) as u16;
    }
    for x in ncomp..row_len {
        let d = br.next_diff(tables[x % ncomp]);
        first[x] = (first[x - ncomp] as i32 + d) as u16;
    }
    br.check_row()?;

    // Remaining rows: first sample of each component from above, the rest
    // by predictor P over left `a`, above `b`, above-left `c`.
    let mut prev: &[u16] = first;
    for cur in rows {
        for c in 0..ncomp {
            cur[c] = (prev[c] as i32 + br.next_diff(tables[c])) as u16;
        }
        match ncomp {
            1 => {
                let t = tables[0];
                let mut a = cur[0] as i32;
                let above = prev[1..].iter().zip(prev.iter());
                for (o, (&b, &c)) in cur[1..].iter_mut().zip(above) {
                    let d = br.next_diff(t);
                    a = (predict::<P>(a, b as i32, c as i32) + d) as u16 as i32;
                    *o = a as u16;
                }
            }
            2 => {
                let (t0, t1) = (tables[0], tables[1]);
                let mut a0 = cur[0] as i32;
                let mut a1 = cur[1] as i32;
                // Pairs: `b` is the pair above, `c` the pair above-left.
                let above = prev[2..].as_chunks::<2>().0.iter();
                let above_left = prev.as_chunks::<2>().0.iter();
                let out_pairs = cur[2..].as_chunks_mut::<2>().0.iter_mut();
                for (o, (b, c)) in out_pairs.zip(above.zip(above_left)) {
                    let d0 = br.next_diff(t0);
                    a0 = (predict::<P>(a0, b[0] as i32, c[0] as i32) + d0) as u16 as i32;
                    o[0] = a0 as u16;
                    let d1 = br.next_diff(t1);
                    a1 = (predict::<P>(a1, b[1] as i32, c[1] as i32) + d1) as u16 as i32;
                    o[1] = a1 as u16;
                }
            }
            _ => {
                for x in ncomp..row_len {
                    let a = cur[x - ncomp] as i32;
                    let b = prev[x] as i32;
                    let c = prev[x - ncomp] as i32;
                    let d = br.next_diff(tables[x % ncomp]);
                    cur[x] = (predict::<P>(a, b, c) + d) as u16;
                }
            }
        }
        br.check_row()?;
        prev = cur;
    }
    Ok(())
}

/// Parse and decode in one step. `out.len()` must equal the stream's sample count.
pub fn decode(data: &[u8], out: &mut [u16]) -> Result<Lj92Header> {
    let hdr = parse(data)?;
    decode_into(data, &hdr, out)?;
    Ok(hdr)
}

#[cfg(test)]
pub(crate) mod test_encoder {
    //! Minimal LJ92 encoder for round-trip tests: one Huffman table with
    //! 5-bit codes for every SSSS class, arbitrary predictor and component
    //! count. Not optimised and not part of the public API.
    use alloc::vec::Vec;

    pub struct Encoder {
        out: Vec<u8>,
        acc: u64,
        cnt: u32,
    }

    impl Encoder {
        fn put_bits(&mut self, v: u32, n: u32) {
            if n == 0 {
                return;
            }
            self.acc = (self.acc << n) | (v as u64 & ((1u64 << n) - 1));
            self.cnt += n;
            while self.cnt >= 8 {
                let b = (self.acc >> (self.cnt - 8)) as u8;
                self.out.push(b);
                if b == 0xFF {
                    self.out.push(0);
                }
                self.cnt -= 8;
            }
        }

        fn flush(&mut self) {
            if self.cnt > 0 {
                let pad = 8 - self.cnt;
                self.put_bits((1 << pad) - 1, pad);
            }
        }

        fn diff(&mut self, d: i32) {
            let (ssss, bits) = if d == 0 {
                (0u32, 0u32)
            } else if d == 32768 || d == -32768 {
                (16, 0)
            } else {
                let t = 32 - d.unsigned_abs().leading_zeros();
                let v = if d < 0 { (d - 1) as u32 } else { d as u32 };
                (t, v)
            };
            // Code for class ssss: 5 bits, value == ssss (canonical, all length 5).
            self.put_bits(ssss, 5);
            self.put_bits(bits, ssss);
        }
    }

    /// Encode `img` (row-major, interleaved) as lossless JPEG.
    #[allow(clippy::too_many_arguments)]
    pub fn encode(
        img: &[u16],
        width: u16,
        height: u16,
        comps: u8,
        precision: u8,
        predictor: u8,
        pt: u8,
        tables_per_component: bool,
    ) -> Vec<u8> {
        let w = width as usize;
        let nc = comps as usize;
        assert_eq!(img.len(), w * height as usize * nc);
        let mut v = Vec::new();
        v.extend_from_slice(&[0xFF, 0xD8]);
        // APP0-ish comment segment to exercise skipping.
        v.extend_from_slice(&[0xFF, 0xFE, 0x00, 0x05, b'h', b'i', b'!']);
        // DHT: table(s) with 17 symbols of length 5.
        let ntab = if tables_per_component { nc } else { 1 };
        let mut dht = Vec::new();
        for t in 0..ntab {
            dht.push(t as u8);
            let mut bits = [0u8; 16];
            bits[4] = 17;
            dht.extend_from_slice(&bits);
            dht.extend(0u8..=16);
        }
        v.extend_from_slice(&[0xFF, 0xC4]);
        v.extend_from_slice(&((dht.len() + 2) as u16).to_be_bytes());
        v.extend_from_slice(&dht);
        // SOF3
        v.extend_from_slice(&[0xFF, 0xC3]);
        v.extend_from_slice(&((8 + 3 * nc) as u16).to_be_bytes());
        v.push(precision);
        v.extend_from_slice(&height.to_be_bytes());
        v.extend_from_slice(&width.to_be_bytes());
        v.push(comps);
        for c in 0..nc {
            v.extend_from_slice(&[c as u8 + 1, 0x11, 0]);
        }
        // SOS
        v.extend_from_slice(&[0xFF, 0xDA]);
        v.extend_from_slice(&((6 + 2 * nc) as u16).to_be_bytes());
        v.push(comps);
        for c in 0..nc {
            let td = if tables_per_component { c as u8 } else { 0 };
            v.extend_from_slice(&[c as u8 + 1, td << 4]);
        }
        v.extend_from_slice(&[predictor, 0, pt]);

        let mut e = Encoder {
            out: Vec::new(),
            acc: 0,
            cnt: 0,
        };
        let init = 1i32 << (precision - pt - 1);
        let row_len = w * nc;
        for y in 0..height as usize {
            for x in 0..row_len {
                let cur = img[y * row_len + x] as i32;
                let px = if y == 0 && x < nc {
                    init
                } else if y == 0 {
                    img[x - nc] as i32
                } else if x < nc {
                    img[(y - 1) * row_len + x] as i32
                } else {
                    let a = img[y * row_len + x - nc] as i32;
                    let b = img[(y - 1) * row_len + x] as i32;
                    let c = img[(y - 1) * row_len + x - nc] as i32;
                    match predictor {
                        0 => 0,
                        1 => a,
                        2 => b,
                        3 => c,
                        4 => a + b - c,
                        5 => a + ((b - c) >> 1),
                        6 => b + ((a - c) >> 1),
                        _ => (a + b) >> 1,
                    }
                };
                // Difference modulo 2^16 into -32768..=32767, 32768 coded as class 16.
                let mut d = (cur - px) & 0xFFFF;
                if d >= 32768 {
                    d -= 65536;
                }
                e.diff(d);
            }
        }
        e.flush();
        v.extend_from_slice(&e.out);
        v.extend_from_slice(&[0xFF, 0xD9]);
        v
    }
}

#[cfg(test)]
mod tests {
    use super::test_encoder::encode;
    use super::*;

    fn lcg(seed: &mut u32) -> u32 {
        *seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        *seed >> 8
    }

    fn image(w: usize, h: usize, nc: usize, bits: u32, seed: u32) -> Vec<u16> {
        let mut s = seed;
        let mask = (1u32 << bits) - 1;
        // Smooth-ish image with occasional big jumps to hit all SSSS classes.
        let mut v = Vec::with_capacity(w * h * nc);
        let mut cur = 1u32 << (bits - 1);
        for _ in 0..w * h * nc {
            let r = lcg(&mut s);
            cur = if r.is_multiple_of(97) {
                lcg(&mut s) & mask
            } else {
                (cur as i32 + (r % 33) as i32 - 16).clamp(0, mask as i32) as u32
            };
            v.push(cur as u16);
        }
        v
    }

    #[test]
    fn roundtrip_all_predictors_and_components() {
        for &nc in &[1u8, 2, 3, 4] {
            for pred in 0..=7u8 {
                for &per_comp in &[false, true] {
                    let (w, h) = (23u16, 7u16);
                    let img = image(w as usize, h as usize, nc as usize, 14, 7 + pred as u32);
                    let data = encode(&img, w, h, nc, 14, pred, 0, per_comp);
                    let hdr =
                        parse(&data).unwrap_or_else(|e| panic!("parse nc={nc} pred={pred}: {e}"));
                    assert_eq!(
                        (hdr.width, hdr.height, hdr.components, hdr.predictor),
                        (w, h, nc, pred)
                    );
                    let mut out = vec![0u16; hdr.sample_count()];
                    decode_into(&data, &hdr, &mut out)
                        .unwrap_or_else(|e| panic!("decode nc={nc} pred={pred}: {e}"));
                    assert_eq!(out, img, "nc={nc} pred={pred} per_comp={per_comp}");
                }
            }
        }
    }

    #[test]
    fn roundtrip_16bit_full_range_and_point_transform() {
        let (w, h) = (16u16, 16u16);
        let mut s = 1u32;
        let img: Vec<u16> = (0..w as usize * h as usize)
            .map(|_| lcg(&mut s) as u16)
            .collect();
        let data = encode(&img, w, h, 1, 16, 6, 0, false);
        let mut out = vec![0u16; img.len()];
        decode(&data, &mut out).unwrap();
        assert_eq!(out, img);

        // Point transform only changes the initial prediction.
        let img12 = image(w as usize, h as usize, 1, 12, 3);
        let data = encode(&img12, w, h, 1, 14, 1, 2, false);
        let hdr = parse(&data).unwrap();
        assert_eq!(hdr.point_transform, 2);
        decode_into(&data, &hdr, &mut out).unwrap();
        assert_eq!(out, img12);
    }

    #[test]
    fn rejects_bad_output_size() {
        let img = image(8, 8, 1, 14, 1);
        let data = encode(&img, 8, 8, 1, 14, 6, 0, false);
        let hdr = parse(&data).unwrap();
        let mut out = vec![0u16; 63];
        assert_eq!(
            decode_into(&data, &hdr, &mut out),
            Err(Lj92Error::OutputSize {
                expected: 64,
                got: 63
            })
        );
    }

    #[test]
    fn truncated_scan_is_an_error_not_a_panic() {
        let img = image(64, 32, 2, 14, 9);
        let data = encode(&img, 64, 32, 2, 14, 6, 0, false);
        let hdr = parse(&data).unwrap();
        let mut out = vec![0u16; hdr.sample_count()];
        // Cut deep inside the entropy data.
        let cut = hdr.scan_start + (data.len() - hdr.scan_start) / 2;
        let r = decode_into(&data[..cut], &hdr, &mut out);
        assert_eq!(r, Err(Lj92Error::ScanTruncated));
        // Cut off just the EOI: still decodes.
        decode_into(&data[..data.len() - 2], &hdr, &mut out).unwrap();
        assert_eq!(out, img);
    }

    #[test]
    fn corrupt_headers_are_errors() {
        assert_eq!(parse(&[]), Err(Lj92Error::NoSoi));
        assert_eq!(parse(&[0xFF, 0xD8]), Err(Lj92Error::Truncated));
        assert_eq!(
            parse(&[0xFF, 0xD8, 0xFF, 0xC4, 0x00]),
            Err(Lj92Error::Truncated)
        );
        // Baseline SOF0 is not lossless.
        assert_eq!(
            parse(&[0xFF, 0xD8, 0xFF, 0xC0, 0x00, 0x02]),
            Err(Lj92Error::NotLossless(0xC0))
        );
        // DRI rejected.
        assert_eq!(
            parse(&[0xFF, 0xD8, 0xFF, 0xDD, 0x00, 0x04, 0, 1]),
            Err(Lj92Error::RestartMarkers)
        );
        // SOS before SOF.
        let img = image(4, 4, 1, 8, 1);
        let mut data = encode(&img, 4, 4, 1, 8, 1, 0, false);
        let sof = data.windows(2).position(|w| w == [0xFF, 0xC3]).unwrap();
        data[sof + 1] = 0xFE; // turn SOF3 into a comment
        assert!(matches!(parse(&data), Err(Lj92Error::BadFrameHeader)));

        // Fuzz-ish: random byte flips never panic.
        let img = image(16, 8, 2, 12, 5);
        let good = encode(&img, 16, 8, 2, 12, 6, 0, true);
        let mut s = 42u32;
        for _ in 0..2000 {
            let mut d = good.clone();
            let n = 1 + lcg(&mut s) as usize % 4;
            for _ in 0..n {
                let i = lcg(&mut s) as usize % d.len();
                d[i] ^= 1 << (lcg(&mut s) % 8);
            }
            if let Ok(h) = parse(&d) {
                let mut out = vec![0u16; h.sample_count().min(1 << 20)];
                let _ = decode_into(&d, &h, &mut out);
            }
        }
    }

    #[test]
    fn huffman_table_rejects_oversubscription() {
        let mut bits = [0u8; 16];
        bits[0] = 3; // three 1-bit codes: impossible
        assert_eq!(
            HuffTable::build(&bits, &[0, 1, 2]),
            Err(Lj92Error::BadHuffmanTable)
        );
        let bits = [0u8; 16];
        assert_eq!(
            HuffTable::build(&bits, &[]),
            Err(Lj92Error::BadHuffmanTable)
        );
        let mut bits = [0u8; 16];
        bits[1] = 2;
        bits[2] = 2;
        let t = HuffTable::build(&bits, &[0, 1, 2, 3]).unwrap();
        assert_eq!(t.max_bits, 3);
        // Codes: 00, 01, 100, 101; 11x are unassigned.
        assert_eq!(t.lut[0b000] & 0xFF, 2);
        assert_eq!(t.lut[0b100] >> 8, 2);
        assert_eq!(t.lut[0b110], BAD_CODE_ENTRY);
        assert_eq!(t.lut[0b111], BAD_CODE_ENTRY);
    }
}
