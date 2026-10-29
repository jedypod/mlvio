// SPDX-License-Identifier: MIT OR Apache-2.0
//! Synthetic MLV chunk writer for integration tests.
//!
//! Builds one chunk in memory: an `MLVI` header followed by whatever blocks
//! the test appends. Layouts follow the format facts in DESIGN.md section 1.

#![allow(dead_code)]

use mlvio::decode::unpack::pack;

/// Byte offsets of patchable `MLVI` fields within the chunk.
const MLVI_FILE_FLAGS: usize = 28;
const MLVI_VIDEO_CLASS: usize = 32;
const MLVI_VIDEO_FRAME_COUNT: usize = 36;
const MLVI_FPS_NOM: usize = 44;
const MLVI_FPS_DENOM: usize = 48;

/// Size of the `MLVI` block.
pub const MLVI_SIZE: usize = 52;
/// Size of the `RAWI` block.
pub const RAWI_SIZE: usize = 180;
/// Size of the fixed part of a `VIDF` block (header + fields).
pub const VIDF_HEADER_SIZE: usize = 32;

/// In-memory MLV chunk builder.
pub struct MlvWriter {
    buf: Vec<u8>,
}

impl MlvWriter {
    /// Start a chunk with an `MLVI` block: RAW video class, no audio,
    /// 25 fps (25000/1000), zero flags, version string "v2.0".
    pub fn new(guid: u64, file_num: u16, file_count: u16) -> Self {
        let mut p = Vec::with_capacity(36);
        p.extend_from_slice(&guid.to_le_bytes());
        p.extend_from_slice(&file_num.to_le_bytes());
        p.extend_from_slice(&file_count.to_le_bytes());
        p.extend_from_slice(&0u32.to_le_bytes()); // fileFlags
        p.extend_from_slice(&1u16.to_le_bytes()); // videoClass = RAW
        p.extend_from_slice(&0u16.to_le_bytes()); // audioClass
        p.extend_from_slice(&0u32.to_le_bytes()); // videoFrameCount
        p.extend_from_slice(&0u32.to_le_bytes()); // audioFrameCount
        p.extend_from_slice(&25000u32.to_le_bytes()); // sourceFpsNom
        p.extend_from_slice(&1000u32.to_le_bytes()); // sourceFpsDenom
        let mut w = MlvWriter { buf: Vec::new() };
        w.raw_block(*b"MLVI", u64::from_le_bytes(*b"v2.0\0\0\0\0"), &p);
        debug_assert_eq!(w.buf.len(), MLVI_SIZE);
        w
    }

    fn patch(&mut self, at: usize, bytes: &[u8]) {
        self.buf[at..at + bytes.len()].copy_from_slice(bytes);
    }

    /// Set `MLVI.videoClass` (1 = RAW, `| 0x20` = LJ92).
    pub fn video_class(mut self, class: u16) -> Self {
        self.patch(MLVI_VIDEO_CLASS, &class.to_le_bytes());
        self
    }

    /// Set `MLVI.sourceFpsNom` / `sourceFpsDenom`.
    pub fn fps(mut self, num: u32, den: u32) -> Self {
        self.patch(MLVI_FPS_NOM, &num.to_le_bytes());
        self.patch(MLVI_FPS_DENOM, &den.to_le_bytes());
        self
    }

    /// Set `MLVI.fileFlags` (1 = out of order, 2 = dropped frames).
    pub fn file_flags(mut self, flags: u32) -> Self {
        self.patch(MLVI_FILE_FLAGS, &flags.to_le_bytes());
        self
    }

    /// Set `MLVI.videoFrameCount` explicitly (otherwise each `vidf` call
    /// increments it).
    pub fn video_frame_count(mut self, n: u32) -> Self {
        self.patch(MLVI_VIDEO_FRAME_COUNT, &n.to_le_bytes());
        self
    }

    /// Current chunk length in bytes.
    pub fn len(&self) -> usize {
        self.buf.len()
    }

    /// Append a block with an arbitrary tag and payload.
    pub fn raw_block(&mut self, tag: [u8; 4], timestamp: u64, payload: &[u8]) -> &mut Self {
        let size = u32::try_from(16 + payload.len()).expect("block too large");
        self.buf.extend_from_slice(&tag);
        self.buf.extend_from_slice(&size.to_le_bytes());
        self.buf.extend_from_slice(&timestamp.to_le_bytes());
        self.buf.extend_from_slice(payload);
        self
    }

    /// Append a 180-byte `RAWI` block (xRes, yRes, 160-byte `raw_info`).
    pub fn rawi(&mut self, x_res: u16, y_res: u16, bpp: u32, black: i32, white: i32) -> &mut Self {
        let width = i32::from(x_res);
        let height = i32::from(y_res);
        let pitch = width * bpp as i32 / 8;
        let frame_size = pitch * height;
        let mut p = Vec::with_capacity(164);
        p.extend_from_slice(&x_res.to_le_bytes());
        p.extend_from_slice(&y_res.to_le_bytes());
        let ri_start = p.len();
        let i32s = |p: &mut Vec<u8>, vals: &[i32]| {
            for v in vals {
                p.extend_from_slice(&v.to_le_bytes());
            }
        };
        p.extend_from_slice(&1u32.to_le_bytes()); // api_version
        p.extend_from_slice(&[0u8; 4]); // buffer pointer slot
        i32s(
            &mut p,
            &[height, width, pitch, frame_size, bpp as i32, black, white],
        );
        i32s(&mut p, &[0, 0, width, height]); // crop origin x,y / size w,h
        i32s(&mut p, &[0, 0, height, width]); // active_area y1,x1,y2,x2
        i32s(&mut p, &[0, 1]); // exposure_bias
        p.extend_from_slice(&0x0201_0100u32.to_le_bytes()); // cfa_pattern (RGGB)
        i32s(&mut p, &[21]); // calibration_illuminant1 (D65)
        for r in 0..3 {
            for c in 0..3 {
                i32s(&mut p, &[if r == c { 10000 } else { 0 }, 10000]);
            }
        }
        i32s(&mut p, &[0]); // dynamic_range
        assert_eq!(p.len() - ri_start, 0xa0, "raw_info must be 160 bytes");
        self.raw_block(*b"RAWI", 0, &p);
        self
    }

    /// Append an `EXPO` block (isoMode 0, isoValue = isoAnalog = `iso`,
    /// digitalGain 0, shutterValue in microseconds).
    pub fn expo(&mut self, timestamp: u64, iso: u32, shutter_us: u64) -> &mut Self {
        let mut p = Vec::with_capacity(24);
        p.extend_from_slice(&0u32.to_le_bytes());
        p.extend_from_slice(&iso.to_le_bytes());
        p.extend_from_slice(&iso.to_le_bytes());
        p.extend_from_slice(&0u32.to_le_bytes());
        p.extend_from_slice(&shutter_us.to_le_bytes());
        self.raw_block(*b"EXPO", timestamp, &p)
    }

    /// Append a `WBAL` block (wb_mode, kelvin; gains and shifts zero).
    pub fn wbal(&mut self, timestamp: u64, mode: u32, kelvin: u32) -> &mut Self {
        let mut p = Vec::with_capacity(28);
        p.extend_from_slice(&mode.to_le_bytes());
        p.extend_from_slice(&kelvin.to_le_bytes());
        p.extend_from_slice(&[0u8; 20]); // wbgain_r/g/b, wbs_gm, wbs_ba
        self.raw_block(*b"WBAL", timestamp, &p)
    }

    /// Append an `IDNT` block (cameraName[32], cameraModel, cameraSerial[32]).
    pub fn idnt(&mut self, name: &str, model: u32, serial: &str) -> &mut Self {
        let mut p = Vec::with_capacity(68);
        p.extend_from_slice(&fixed_str::<32>(name));
        p.extend_from_slice(&model.to_le_bytes());
        p.extend_from_slice(&fixed_str::<32>(serial));
        self.raw_block(*b"IDNT", 0, &p)
    }

    /// Append an uncompressed `VIDF` block holding `pixels` packed at `bpp`.
    pub fn vidf(
        &mut self,
        timestamp: u64,
        frame_number: u32,
        frame_space: u32,
        pixels: &[u16],
        bpp: u32,
    ) -> &mut Self {
        self.vidf_padded(timestamp, frame_number, frame_space, pixels, bpp, 0)
    }

    /// Like [`MlvWriter::vidf`] but with `extra` zero bytes after the pixel
    /// data (payload longer than the frame).
    pub fn vidf_padded(
        &mut self,
        timestamp: u64,
        frame_number: u32,
        frame_space: u32,
        pixels: &[u16],
        bpp: u32,
        extra: usize,
    ) -> &mut Self {
        let mut data = Vec::new();
        pack(bpp, pixels, &mut data);
        self.vidf_bytes(timestamp, frame_number, frame_space, &data, extra)
    }

    /// Append a `VIDF` block with an already-encoded payload (e.g. LJ92).
    pub fn vidf_bytes(
        &mut self,
        timestamp: u64,
        frame_number: u32,
        frame_space: u32,
        data: &[u8],
        extra: usize,
    ) -> &mut Self {
        let mut p = Vec::with_capacity(16 + frame_space as usize + data.len() + extra);
        p.extend_from_slice(&frame_number.to_le_bytes());
        p.extend_from_slice(&[0u8; 8]); // cropPosX/Y, panPosX/Y
        p.extend_from_slice(&frame_space.to_le_bytes());
        p.resize(p.len() + frame_space as usize, 0);
        p.extend_from_slice(data);
        p.resize(p.len() + extra, 0);
        self.raw_block(*b"VIDF", timestamp, &p);
        let n = u32::from_le_bytes(
            self.buf[MLVI_VIDEO_FRAME_COUNT..MLVI_VIDEO_FRAME_COUNT + 4]
                .try_into()
                .unwrap(),
        );
        self.patch(MLVI_VIDEO_FRAME_COUNT, &(n + 1).to_le_bytes());
        self
    }

    /// Append a `NULL` padding block of total size `size` (>= 16).
    pub fn null(&mut self, size: u32) -> &mut Self {
        assert!(size >= 16, "NULL block must be at least 16 bytes");
        let payload = vec![0u8; size as usize - 16];
        self.raw_block(*b"NULL", 0, &payload)
    }

    /// Cut the chunk to `len` bytes, simulating a truncated recording.
    pub fn truncate_to(&mut self, len: usize) -> &mut Self {
        self.buf.truncate(len);
        self
    }

    /// Return the chunk bytes.
    pub fn finish(self) -> Vec<u8> {
        self.buf
    }
}

fn fixed_str<const N: usize>(s: &str) -> [u8; N] {
    let mut out = [0u8; N];
    let b = s.as_bytes();
    let n = b.len().min(N - 1);
    out[..n].copy_from_slice(&b[..n]);
    out
}

/// Deterministic pseudo-random pixels in `0..2^bpp` (LCG, no dependencies).
pub fn lcg_pixels(n: usize, bpp: u32, seed: u32) -> Vec<u16> {
    let mask = (1u32 << bpp) - 1;
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((s >> 8) & mask) as u16
        })
        .collect()
}
