// SPDX-License-Identifier: MIT OR Apache-2.0
//! One plain struct per MLV block type, plus the [`Block`] enum and the
//! [`parse_block`] dispatcher.
//!
//! Every `from_bytes` takes the block payload, that is the bytes that follow
//! the 16-byte [`BlockHeader`]. Payloads shorter than the layout written by
//! current firmware are accepted: missing trailing fields read as zero and the
//! struct's `truncated` flag is set. Longer payloads are fine too; extra bytes
//! are ignored.
//!
//! Field names are Rust-style; each field's doc comment gives the name used by
//! the reference C headers so readers can map back to them.

use alloc::string::String;
use alloc::vec::Vec;
use core::fmt;

use super::tag::{BlockHeader, Tag};
use super::{Cursor, c_str};

/// Decode a byte run as text: stop at the first NUL, replace invalid UTF-8,
/// trim trailing whitespace.
fn text_from(bytes: &[u8]) -> String {
    let end = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
    let s = String::from_utf8_lossy(&bytes[..end]);
    String::from(s.trim_end())
}

// ---------------------------------------------------------------------------
// MLVI
// ---------------------------------------------------------------------------

/// `MLVI`: file header, the first block of every chunk.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FileHeader {
    /// `versionString`: format revision as a NUL-padded string (e.g. `v2.0`).
    /// Stored where other blocks keep their timestamp.
    pub version: [u8; 8],
    /// `fileGuid`: identifier shared by all chunks of one recording.
    pub file_guid: u64,
    /// `fileNum`: index of this chunk within the group.
    pub file_num: u16,
    /// `fileCount`: number of chunks in the group (often 0 or stale).
    pub file_count: u16,
    /// `fileFlags`: see [`FileHeader::out_of_order`] and
    /// [`FileHeader::dropped_frames`].
    pub file_flags: u32,
    /// `videoClass`: low nibble is the base class (1 = raw), upper bits are
    /// compression flags (`FLAG_*`).
    pub video_class: u16,
    /// `audioClass`: 0 = none, 1 = WAV.
    pub audio_class: u16,
    /// `videoFrameCount`: video frames in this chunk (0 if recording was
    /// interrupted before the header was rewritten).
    pub video_frame_count: u32,
    /// `audioFrameCount`: audio frames in this chunk.
    pub audio_frame_count: u32,
    /// `sourceFpsNom`: frame rate numerator.
    pub fps_num: u32,
    /// `sourceFpsDenom`: frame rate denominator (typically 1000 or 1001).
    pub fps_den: u32,
    /// Payload was shorter than the full structure.
    pub truncated: bool,
}

impl FileHeader {
    /// Base video class value for raw Bayer frames.
    pub const VIDEO_CLASS_RAW: u16 = 0x01;
    /// Video class flag: frames are LJ92 lossless JPEG.
    pub const FLAG_LJ92: u16 = 0x20;
    /// Video class flag: frames are delta-encoded.
    pub const FLAG_DELTA: u16 = 0x40;
    /// Video class flag: frames are LZMA-compressed.
    pub const FLAG_LZMA: u16 = 0x80;
    /// Video class flag: frames are JPEG 2000 (MLV-App extension).
    pub const FLAG_JPEG2K: u16 = 0x10;
    /// Video class flag: McRAW container (MLV-App extension).
    pub const FLAG_MCRAW: u16 = 0x100;

    /// Payload length written by current firmware.
    pub const PAYLOAD_SIZE: usize = 36;

    /// Parse from the block header (which carries the version string) and
    /// the payload.
    pub fn from_bytes(header: &BlockHeader, payload: &[u8]) -> Self {
        let mut c = Cursor::new(payload);
        let mut h = FileHeader {
            version: header.version_bytes(),
            file_guid: c.u64(),
            file_num: c.u16(),
            file_count: c.u16(),
            file_flags: c.u32(),
            video_class: c.u16(),
            audio_class: c.u16(),
            video_frame_count: c.u32(),
            audio_frame_count: c.u32(),
            fps_num: c.u32(),
            fps_den: c.u32(),
            truncated: false,
        };
        h.truncated = c.truncated();
        h
    }

    /// Version string, trimmed at the first NUL.
    pub fn version_str(&self) -> &str {
        c_str(&self.version)
    }

    /// Frame rate as a float, 0.0 if the denominator is zero.
    pub fn fps(&self) -> f64 {
        if self.fps_den == 0 {
            0.0
        } else {
            f64::from(self.fps_num) / f64::from(self.fps_den)
        }
    }

    /// Frames may be stored out of order (`fileFlags` bit 0).
    pub fn out_of_order(&self) -> bool {
        self.file_flags & 1 != 0
    }

    /// The recorder dropped frames (`fileFlags` bit 1).
    pub fn dropped_frames(&self) -> bool {
        self.file_flags & 2 != 0
    }

    /// Base video class is raw Bayer.
    pub fn is_raw(&self) -> bool {
        self.video_class & 0x0F == Self::VIDEO_CLASS_RAW
    }

    /// Frames are LJ92 lossless JPEG.
    pub fn is_lj92(&self) -> bool {
        self.video_class & Self::FLAG_LJ92 != 0
    }
}

// ---------------------------------------------------------------------------
// RAWI
// ---------------------------------------------------------------------------

/// Active sensor area in raw buffer coordinates (`raw_info.active_area`).
///
/// Stored in the file in the order `y1, x1, y2, x2`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ActiveArea {
    /// `y1`: top edge (inclusive).
    pub y1: i32,
    /// `x1`: left edge (inclusive).
    pub x1: i32,
    /// `y2`: bottom edge (exclusive).
    pub y2: i32,
    /// `x2`: right edge (exclusive).
    pub x2: i32,
}

/// Bayer colour filter arrangement of the top-left 2x2 tile.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub enum Cfa {
    /// Red, green / green, blue (pattern `0x02010100`).
    Rggb,
    /// Blue, green / green, red (pattern `0x00010102`).
    Bggr,
    /// Green, red / blue, green (pattern `0x01020001`).
    Grbg,
    /// Green, blue / red, green (pattern `0x01000201`).
    Gbrg,
    /// Any other `cfa_pattern` value.
    Unknown(u32),
}

impl Cfa {
    /// Interpret a `raw_info.cfa_pattern` value.
    ///
    /// The value is four little-endian bytes giving the colour index
    /// (0 = R, 1 = G, 2 = B) of the top-left, top-right, bottom-left and
    /// bottom-right pixels.
    pub fn from_pattern(pattern: u32) -> Self {
        match pattern.to_le_bytes() {
            [0, 1, 1, 2] => Cfa::Rggb,
            [2, 1, 1, 0] => Cfa::Bggr,
            [1, 0, 2, 1] => Cfa::Grbg,
            [1, 2, 0, 1] => Cfa::Gbrg,
            _ => Cfa::Unknown(pattern),
        }
    }

    /// Colour indices (0 = R, 1 = G, 2 = B) of the 2x2 tile in the order
    /// top-left, top-right, bottom-left, bottom-right. `None` for
    /// [`Cfa::Unknown`].
    pub fn pattern(&self) -> Option<[u8; 4]> {
        match self {
            Cfa::Rggb => Some([0, 1, 1, 2]),
            Cfa::Bggr => Some([2, 1, 1, 0]),
            Cfa::Grbg => Some([1, 0, 2, 1]),
            Cfa::Gbrg => Some([1, 2, 0, 1]),
            Cfa::Unknown(_) => None,
        }
    }

    /// The arrangement seen from a window whose origin is `dx` columns and
    /// `dy` rows from this tile's origin. Only the parity of each offset
    /// matters; [`Cfa::Unknown`] is returned unchanged.
    pub fn shifted(self, dx: i64, dy: i64) -> Self {
        let Some(p) = self.pattern() else {
            return self;
        };
        let (sx, sy) = ((dx & 1) as usize, (dy & 1) as usize);
        let at = |x: usize, y: usize| p[((y + sy) & 1) * 2 + ((x + sx) & 1)];
        Self::from_pattern(u32::from_le_bytes([at(0, 0), at(1, 0), at(0, 1), at(1, 1)]))
    }
}

impl fmt::Display for Cfa {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Cfa::Rggb => f.write_str("RGGB"),
            Cfa::Bggr => f.write_str("BGGR"),
            Cfa::Grbg => f.write_str("GRBG"),
            Cfa::Gbrg => f.write_str("GBRG"),
            Cfa::Unknown(p) => write!(f, "unknown({p:#010x})"),
        }
    }
}

/// `RAWI`: raw frame format (`xRes`, `yRes` followed by ML's `raw_info`).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct RawInfo {
    /// `xRes`: width of the stored frames in pixels.
    pub x_res: u16,
    /// `yRes`: height of the stored frames in pixels.
    pub y_res: u16,
    /// `raw_info.api_version`.
    pub api_version: u32,
    /// `raw_info.height`: raw buffer height (may differ from `y_res`).
    pub height: i32,
    /// `raw_info.width`: raw buffer width (may differ from `x_res`).
    pub width: i32,
    /// `raw_info.pitch`: raw buffer line length in bytes.
    pub pitch: i32,
    /// `raw_info.frame_size`: raw buffer size in bytes.
    pub frame_size: i32,
    /// `raw_info.bits_per_pixel`: 10, 12 or 14.
    pub bits_per_pixel: i32,
    /// `raw_info.black_level`.
    pub black_level: i32,
    /// `raw_info.white_level`.
    pub white_level: i32,
    /// `raw_info.crop.origin` (aliases `jpeg.x`, `jpeg.y`).
    pub crop_origin: [i32; 2],
    /// `raw_info.crop.size` (aliases `jpeg.width`, `jpeg.height`).
    pub crop_size: [i32; 2],
    /// `raw_info.active_area`.
    pub active_area: ActiveArea,
    /// `raw_info.exposure_bias`: rational (numerator, denominator).
    pub exposure_bias: [i32; 2],
    /// `raw_info.cfa_pattern`; see [`RawInfo::cfa`].
    pub cfa_pattern: u32,
    /// `raw_info.calibration_illuminant1`: EXIF light source code.
    pub calibration_illuminant1: i32,
    /// `raw_info.color_matrix1`: 3x3 matrix as nine (numerator, denominator)
    /// pairs, row major; see [`RawInfo::color_matrix`].
    pub color_matrix1: [i32; 18],
    /// `raw_info.dynamic_range`: EV x 100.
    pub dynamic_range: i32,
    /// Payload was shorter than the full structure.
    pub truncated: bool,
}

impl RawInfo {
    /// Payload length written by current firmware (4 + 160).
    pub const PAYLOAD_SIZE: usize = 164;

    /// Parse from the block payload.
    pub fn from_bytes(payload: &[u8]) -> Self {
        let mut c = Cursor::new(payload);
        let x_res = c.u16();
        let y_res = c.u16();
        let api_version = c.u32();
        // `raw_info.buffer`: a 32-bit camera pointer, meaningless on disk.
        c.skip(4);
        let height = c.i32();
        let width = c.i32();
        let pitch = c.i32();
        let frame_size = c.i32();
        let bits_per_pixel = c.i32();
        let black_level = c.i32();
        let white_level = c.i32();
        let crop_origin = c.i32s();
        let crop_size = c.i32s();
        let active_area = ActiveArea {
            y1: c.i32(),
            x1: c.i32(),
            y2: c.i32(),
            x2: c.i32(),
        };
        let exposure_bias = c.i32s();
        let cfa_pattern = c.u32();
        let calibration_illuminant1 = c.i32();
        let color_matrix1 = c.i32s();
        let dynamic_range = c.i32();
        RawInfo {
            x_res,
            y_res,
            api_version,
            height,
            width,
            pitch,
            frame_size,
            bits_per_pixel,
            black_level,
            white_level,
            crop_origin,
            crop_size,
            active_area,
            exposure_bias,
            cfa_pattern,
            calibration_illuminant1,
            color_matrix1,
            dynamic_range,
            truncated: c.truncated(),
        }
    }

    /// Bayer arrangement from `cfa_pattern`.
    pub fn cfa(&self) -> Cfa {
        Cfa::from_pattern(self.cfa_pattern)
    }

    /// `color_matrix1` as nine floats, row major. `None` if any denominator
    /// is zero or every value is zero.
    pub fn color_matrix(&self) -> Option<[f64; 9]> {
        if self.color_matrix1.iter().all(|&v| v == 0) {
            return None;
        }
        let mut out = [0.0f64; 9];
        for (o, pair) in out.iter_mut().zip(self.color_matrix1.as_chunks::<2>().0) {
            let [num, den] = *pair;
            if den == 0 {
                return None;
            }
            *o = f64::from(num) / f64::from(den);
        }
        Some(out)
    }

    /// `active_area.x2 - active_area.x1`, clamped to `0..=u32::MAX`.
    pub fn active_width(&self) -> u32 {
        span(self.active_area.x1, self.active_area.x2)
    }

    /// `active_area.y2 - active_area.y1`, clamped to `0..=u32::MAX`.
    pub fn active_height(&self) -> u32 {
        span(self.active_area.y1, self.active_area.y2)
    }
}

fn span(lo: i32, hi: i32) -> u32 {
    (i64::from(hi) - i64::from(lo)).clamp(0, i64::from(u32::MAX)) as u32
}

// ---------------------------------------------------------------------------
// VIDF / AUDF
// ---------------------------------------------------------------------------

/// `VIDF`: fixed header of a video frame block. Pixel data follows after
/// `frame_space` bytes of padding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct VideoFrameHeader {
    /// `frameNumber`: frame index within the recording.
    pub frame_number: u32,
    /// `cropPosX`: horizontal sensor offset of the frame (coarse).
    pub crop_pos_x: u16,
    /// `cropPosY`: vertical sensor offset of the frame (coarse).
    pub crop_pos_y: u16,
    /// `panPosX`: horizontal sensor offset of the frame (pixel precision).
    pub pan_pos_x: u16,
    /// `panPosY`: vertical sensor offset of the frame (pixel precision).
    pub pan_pos_y: u16,
    /// `frameSpace`: padding bytes between this header and the pixel data.
    pub frame_space: u32,
    /// Payload was shorter than the fixed header.
    pub truncated: bool,
}

impl VideoFrameHeader {
    /// Fixed header length after the block header.
    pub const SIZE: usize = 16;

    /// Parse from at least the first [`Self::SIZE`] payload bytes.
    pub fn from_bytes(payload: &[u8]) -> Self {
        let mut c = Cursor::new(payload);
        let mut h = VideoFrameHeader {
            frame_number: c.u32(),
            crop_pos_x: c.u16(),
            crop_pos_y: c.u16(),
            pan_pos_x: c.u16(),
            pan_pos_y: c.u16(),
            frame_space: c.u32(),
            truncated: false,
        };
        h.truncated = c.truncated();
        h
    }

    /// Offset of the pixel data from the start of the block.
    pub fn data_offset(&self) -> u64 {
        (BlockHeader::SIZE + Self::SIZE) as u64 + u64::from(self.frame_space)
    }

    /// Pixel data length for a block of `block_size` bytes; `None` if
    /// `frame_space` does not fit in the block.
    pub fn data_len(&self, block_size: u32) -> Option<u32> {
        block_size
            .checked_sub((BlockHeader::SIZE + Self::SIZE) as u32)?
            .checked_sub(self.frame_space)
    }
}

/// `AUDF`: fixed header of an audio frame block.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct AudioFrameHeader {
    /// `frameNumber`: audio frame index.
    pub frame_number: u32,
    /// `frameSpace`: padding bytes between this header and the samples.
    pub frame_space: u32,
    /// Payload was shorter than the fixed header.
    pub truncated: bool,
}

impl AudioFrameHeader {
    /// Fixed header length after the block header.
    pub const SIZE: usize = 8;

    /// Parse from at least the first [`Self::SIZE`] payload bytes.
    pub fn from_bytes(payload: &[u8]) -> Self {
        let mut c = Cursor::new(payload);
        let mut h = AudioFrameHeader {
            frame_number: c.u32(),
            frame_space: c.u32(),
            truncated: false,
        };
        h.truncated = c.truncated();
        h
    }

    /// Offset of the sample data from the start of the block.
    pub fn data_offset(&self) -> u64 {
        (BlockHeader::SIZE + Self::SIZE) as u64 + u64::from(self.frame_space)
    }

    /// Sample data length for a block of `block_size` bytes; `None` on
    /// underflow.
    pub fn data_len(&self, block_size: u32) -> Option<u32> {
        block_size
            .checked_sub((BlockHeader::SIZE + Self::SIZE) as u32)?
            .checked_sub(self.frame_space)
    }
}

// ---------------------------------------------------------------------------
// EXPO, LENS, ELNS, WBAL, IDNT
// ---------------------------------------------------------------------------

/// `EXPO`: exposure settings. May repeat mid-clip.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Exposure {
    /// `isoMode`: 0 = manual, 1 = auto.
    pub iso_mode: u32,
    /// `isoValue`: ISO reported by the camera.
    pub iso_value: u32,
    /// `isoAnalog`: ISO from analog amplification.
    pub iso_analog: u32,
    /// `digitalGain`: digital gain not baked into the raw data
    /// (1024 = 1 EV).
    pub digital_gain: u32,
    /// `shutterValue`: exposure time in microseconds.
    pub shutter_us: u64,
    /// Payload was shorter than the full structure.
    pub truncated: bool,
}

impl Exposure {
    /// Parse from the block payload.
    pub fn from_bytes(payload: &[u8]) -> Self {
        let mut c = Cursor::new(payload);
        let mut e = Exposure {
            iso_mode: c.u32(),
            iso_value: c.u32(),
            iso_analog: c.u32(),
            digital_gain: c.u32(),
            shutter_us: c.u64(),
            truncated: false,
        };
        e.truncated = c.truncated();
        e
    }

    /// Exposure time in seconds.
    pub fn shutter_seconds(&self) -> f64 {
        self.shutter_us as f64 / 1_000_000.0
    }
}

/// `LENS`: lens state. May repeat mid-clip.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Lens {
    /// `focalLength`: millimetres.
    pub focal_length_mm: u16,
    /// `focalDist`: millimetres, 65535 for infinity.
    pub focus_distance_mm: u16,
    /// `aperture`: f-number x 100.
    pub aperture_x100: u16,
    /// `stabilizerMode`: 0 = off, 1 = on.
    pub stabilizer_mode: u8,
    /// `autofocusMode`.
    pub autofocus_mode: u8,
    /// `flags`.
    pub flags: u32,
    /// `lensID`.
    pub lens_id: u32,
    /// `lensName`: NUL-padded string.
    pub name: [u8; 32],
    /// `lensSerial`: NUL-padded string.
    pub serial: [u8; 32],
    /// Payload was shorter than the full structure.
    pub truncated: bool,
}

impl Lens {
    /// Parse from the block payload.
    pub fn from_bytes(payload: &[u8]) -> Self {
        let mut c = Cursor::new(payload);
        let mut l = Lens {
            focal_length_mm: c.u16(),
            focus_distance_mm: c.u16(),
            aperture_x100: c.u16(),
            stabilizer_mode: c.u8(),
            autofocus_mode: c.u8(),
            flags: c.u32(),
            lens_id: c.u32(),
            name: c.array(),
            serial: c.array(),
            truncated: false,
        };
        l.truncated = c.truncated();
        l
    }

    /// Lens name, trimmed.
    pub fn name_str(&self) -> &str {
        c_str(&self.name)
    }

    /// Lens serial number, trimmed.
    pub fn serial_str(&self) -> &str {
        c_str(&self.serial)
    }

    /// f-number.
    pub fn aperture(&self) -> f64 {
        f64::from(self.aperture_x100) / 100.0
    }
}

/// `ELNS`: extended lens description with a variable-length name.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ExtendedLens {
    /// `focalLengthMin`: millimetres.
    pub focal_length_min: u16,
    /// `focalLengthMax`: millimetres.
    pub focal_length_max: u16,
    /// `apertureMin`: f-number x 100.
    pub aperture_min: u16,
    /// `apertureMax`: f-number x 100.
    pub aperture_max: u16,
    /// `version`: lens protocol version.
    pub version: u32,
    /// `extenderInfo`.
    pub extender_info: u8,
    /// `capabilities`.
    pub capabilities: u8,
    /// `chipped`: non-zero when the lens communicates with the body.
    pub chipped: u8,
    /// `lensName`: text up to the first NUL or the end of the payload.
    pub name: String,
    /// Payload was shorter than the fixed fields.
    pub truncated: bool,
}

impl ExtendedLens {
    /// Parse from the block payload.
    pub fn from_bytes(payload: &[u8]) -> Self {
        let mut c = Cursor::new(payload);
        let focal_length_min = c.u16();
        let focal_length_max = c.u16();
        let aperture_min = c.u16();
        let aperture_max = c.u16();
        let version = c.u32();
        let extender_info = c.u8();
        let capabilities = c.u8();
        let chipped = c.u8();
        let name = text_from(c.rest());
        ExtendedLens {
            focal_length_min,
            focal_length_max,
            aperture_min,
            aperture_max,
            version,
            extender_info,
            capabilities,
            chipped,
            name,
            truncated: c.truncated(),
        }
    }
}

/// `WBAL`: white balance settings. May repeat mid-clip.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct WhiteBalance {
    /// `wb_mode`: Canon white balance preset (0 auto, 1 daylight, 2 cloudy,
    /// 3 tungsten, 4 fluorescent, 5 flash, 6 custom, 8 shade, 9 kelvin).
    pub mode: u32,
    /// `kelvin`: colour temperature, meaningful in kelvin mode.
    pub kelvin: u32,
    /// `wbgain_r`: red gain, 1024 = 1.0, meaningful in custom mode.
    pub gain_r: u32,
    /// `wbgain_g`: green gain, 1024 = 1.0.
    pub gain_g: u32,
    /// `wbgain_b`: blue gain, 1024 = 1.0.
    pub gain_b: u32,
    /// `wbs_gm`: green/magenta shift.
    pub shift_gm: u32,
    /// `wbs_ba`: blue/amber shift.
    pub shift_ba: u32,
    /// Payload was shorter than the full structure.
    pub truncated: bool,
}

impl WhiteBalance {
    /// Parse from the block payload.
    pub fn from_bytes(payload: &[u8]) -> Self {
        let mut c = Cursor::new(payload);
        let mut w = WhiteBalance {
            mode: c.u32(),
            kelvin: c.u32(),
            gain_r: c.u32(),
            gain_g: c.u32(),
            gain_b: c.u32(),
            shift_gm: c.u32(),
            shift_ba: c.u32(),
            truncated: false,
        };
        w.truncated = c.truncated();
        w
    }
}

/// `IDNT`: camera identity.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Identity {
    /// `cameraName`: NUL-padded model name.
    pub camera_name: [u8; 32],
    /// `cameraModel`: Canon model id.
    pub camera_model: u32,
    /// `cameraSerial`: NUL-padded serial number string.
    pub camera_serial: [u8; 32],
    /// Payload was shorter than the full structure.
    pub truncated: bool,
}

impl Identity {
    /// Parse from the block payload.
    pub fn from_bytes(payload: &[u8]) -> Self {
        let mut c = Cursor::new(payload);
        let mut i = Identity {
            camera_name: c.array(),
            camera_model: c.u32(),
            camera_serial: c.array(),
            truncated: false,
        };
        i.truncated = c.truncated();
        i
    }

    /// Camera name, trimmed.
    pub fn camera_name_str(&self) -> &str {
        c_str(&self.camera_name)
    }

    /// Camera serial, trimmed.
    pub fn camera_serial_str(&self) -> &str {
        c_str(&self.camera_serial)
    }
}

// ---------------------------------------------------------------------------
// RTCI
// ---------------------------------------------------------------------------

/// A calendar date and time without time zone.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct DateTime {
    /// Year of the common era.
    pub year: i32,
    /// Month, 1..=12.
    pub month: u8,
    /// Day of month, 1..=31.
    pub day: u8,
    /// Hour, 0..=23.
    pub hour: u8,
    /// Minute, 0..=59.
    pub minute: u8,
    /// Second, 0..=60.
    pub second: u8,
}

impl fmt::Display for DateTime {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}",
            self.year, self.month, self.day, self.hour, self.minute, self.second
        )
    }
}

/// `RTCI`: camera wall-clock time at the block's timestamp, in C `struct tm`
/// conventions.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct WallClock {
    /// `tm_sec`: seconds, 0..=60.
    pub sec: u16,
    /// `tm_min`: minutes, 0..=59.
    pub min: u16,
    /// `tm_hour`: hours, 0..=23.
    pub hour: u16,
    /// `tm_mday`: day of month, 1..=31.
    pub mday: u16,
    /// `tm_mon`: month, 0..=11.
    pub mon: u16,
    /// `tm_year`: years since 1900.
    pub year: u16,
    /// `tm_wday`: day of week, 0 = Sunday.
    pub wday: u16,
    /// `tm_yday`: day of year, 0..=365.
    pub yday: u16,
    /// `tm_isdst`: daylight saving flag.
    pub isdst: u16,
    /// `tm_gmtoff`: offset from UTC as written by the camera.
    pub gmtoff: u16,
    /// `tm_zone`: NUL-padded zone abbreviation.
    pub zone: [u8; 8],
    /// Payload was shorter than the full structure.
    pub truncated: bool,
}

impl WallClock {
    /// Parse from the block payload.
    pub fn from_bytes(payload: &[u8]) -> Self {
        let mut c = Cursor::new(payload);
        let mut w = WallClock {
            sec: c.u16(),
            min: c.u16(),
            hour: c.u16(),
            mday: c.u16(),
            mon: c.u16(),
            year: c.u16(),
            wday: c.u16(),
            yday: c.u16(),
            isdst: c.u16(),
            gmtoff: c.u16(),
            zone: c.array(),
            truncated: false,
        };
        w.truncated = c.truncated();
        w
    }

    /// Year of the common era (`tm_year + 1900`).
    pub fn year_ce(&self) -> u16 {
        self.year.saturating_add(1900)
    }

    /// Month 1..=12 (`tm_mon + 1`).
    pub fn month(&self) -> u8 {
        u8::try_from(self.mon.saturating_add(1)).unwrap_or(u8::MAX)
    }

    /// Time zone abbreviation, trimmed.
    pub fn zone_str(&self) -> &str {
        c_str(&self.zone)
    }

    /// Calendar date and time, or `None` if any field is out of range.
    pub fn to_datetime(&self) -> Option<DateTime> {
        if self.mon > 11
            || !(1..=31).contains(&self.mday)
            || self.hour > 23
            || self.min > 59
            || self.sec > 60
        {
            return None;
        }
        Some(DateTime {
            year: 1900 + i32::from(self.year),
            month: self.mon as u8 + 1,
            day: self.mday as u8,
            hour: self.hour as u8,
            minute: self.min as u8,
            second: self.sec as u8,
        })
    }
}

// ---------------------------------------------------------------------------
// RAWC, DISO, CURV, DARK
// ---------------------------------------------------------------------------

/// `RAWC`: sensor readout geometry (resolution, crop factor, binning,
/// skipping, offsets).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct RawCapture {
    /// `sensor_res_x`: full sensor width in pixels.
    pub sensor_res_x: u16,
    /// `sensor_res_y`: full sensor height in pixels.
    pub sensor_res_y: u16,
    /// `sensor_crop`: sensor crop factor x 100.
    pub sensor_crop_x100: u16,
    /// `reserved`.
    pub reserved: u16,
    /// `binning_x`: horizontal pixels combined per output pixel.
    pub binning_x: u8,
    /// `skipping_x`: horizontal pixels skipped per output pixel.
    pub skipping_x: u8,
    /// `binning_y`: vertical lines combined per output line.
    pub binning_y: u8,
    /// `skipping_y`: vertical lines skipped per output line.
    pub skipping_y: u8,
    /// `offset_x`: horizontal offset of the capture on the full-res sensor.
    pub offset_x: i16,
    /// `offset_y`: vertical offset of the capture on the full-res sensor.
    pub offset_y: i16,
    /// Payload was shorter than the full structure.
    pub truncated: bool,
}

impl RawCapture {
    /// Parse from the block payload.
    pub fn from_bytes(payload: &[u8]) -> Self {
        let mut c = Cursor::new(payload);
        let mut r = RawCapture {
            sensor_res_x: c.u16(),
            sensor_res_y: c.u16(),
            sensor_crop_x100: c.u16(),
            reserved: c.u16(),
            binning_x: c.u8(),
            skipping_x: c.u8(),
            binning_y: c.u8(),
            skipping_y: c.u8(),
            offset_x: c.i16(),
            offset_y: c.i16(),
            truncated: false,
        };
        r.truncated = c.truncated();
        r
    }
}

/// `DISO`: dual-ISO recording mode.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct DualIso {
    /// `dualMode`: 0 = off, non-zero = alternating ISO lines.
    pub dual_mode: u32,
    /// `isoValue`: the second ISO.
    pub iso_value: u32,
    /// Payload was shorter than the full structure.
    pub truncated: bool,
}

impl DualIso {
    /// Parse from the block payload.
    pub fn from_bytes(payload: &[u8]) -> Self {
        let mut c = Cursor::new(payload);
        let mut d = DualIso {
            dual_mode: c.u32(),
            iso_value: c.u32(),
            truncated: false,
        };
        d.truncated = c.truncated();
        d
    }

    /// Frames use interleaved dual-ISO lines.
    pub fn is_active(&self) -> bool {
        self.dual_mode != 0
    }
}

/// `CURV`: lookup table from stored values to linear values.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Curve {
    /// `data`: the table; may hold fewer than 65536 entries.
    pub lut: Vec<u16>,
    /// Payload had an odd length; the trailing byte was ignored.
    pub truncated: bool,
}

impl Curve {
    /// Parse from the block payload.
    pub fn from_bytes(payload: &[u8]) -> Self {
        let lut = payload
            .as_chunks::<2>()
            .0
            .iter()
            .map(|p| u16::from_le_bytes(*p))
            .collect();
        Curve {
            lut,
            truncated: !payload.len().is_multiple_of(2),
        }
    }
}

/// `DARK`: averaged dark frame with the settings it was captured under.
///
/// Pixel data is not copied; it starts at [`DarkFrame::FIXED_SIZE`] bytes
/// into the payload.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct DarkFrame {
    /// `samplesAveraged`: number of frames averaged.
    pub samples_averaged: u32,
    /// `cameraModel`: Canon model id.
    pub camera_model: u32,
    /// `xRes`: dark frame width.
    pub x_res: u16,
    /// `yRes`: dark frame height.
    pub y_res: u16,
    /// `rawWidth`: raw buffer width.
    pub raw_width: u32,
    /// `rawHeight`: raw buffer height.
    pub raw_height: u32,
    /// `bits_per_pixel`.
    pub bits_per_pixel: u32,
    /// `black_level`.
    pub black_level: u32,
    /// `white_level`.
    pub white_level: u32,
    /// `sourceFpsNom`: frame rate numerator.
    pub fps_num: u32,
    /// `sourceFpsDenom`: frame rate denominator.
    pub fps_den: u32,
    /// `isoMode`: 0 = manual, 1 = auto.
    pub iso_mode: u32,
    /// `isoValue`.
    pub iso_value: u32,
    /// `isoAnalog`.
    pub iso_analog: u32,
    /// `digitalGain`.
    pub digital_gain: u32,
    /// `shutterValue`: microseconds.
    pub shutter_us: u64,
    /// `binning_x`.
    pub binning_x: u8,
    /// `skipping_x`.
    pub skipping_x: u8,
    /// `binning_y`.
    pub binning_y: u8,
    /// `skipping_y`.
    pub skipping_y: u8,
    /// Payload was shorter than the fixed fields.
    pub truncated: bool,
}

impl DarkFrame {
    /// Length of the fixed fields preceding the pixel data; `frameData`
    /// starts at this offset in the payload.
    pub const FIXED_SIZE: usize = 68;

    /// Parse the fixed fields from the block payload.
    pub fn from_bytes(payload: &[u8]) -> Self {
        let mut c = Cursor::new(payload);
        let mut d = DarkFrame {
            samples_averaged: c.u32(),
            camera_model: c.u32(),
            x_res: c.u16(),
            y_res: c.u16(),
            raw_width: c.u32(),
            raw_height: c.u32(),
            bits_per_pixel: c.u32(),
            black_level: c.u32(),
            white_level: c.u32(),
            fps_num: c.u32(),
            fps_den: c.u32(),
            iso_mode: c.u32(),
            iso_value: c.u32(),
            iso_analog: c.u32(),
            digital_gain: c.u32(),
            shutter_us: c.u64(),
            binning_x: c.u8(),
            skipping_x: c.u8(),
            binning_y: c.u8(),
            skipping_y: c.u8(),
            truncated: false,
        };
        d.truncated = c.truncated();
        d
    }
}

// ---------------------------------------------------------------------------
// INFO, VERS, DEBG, MARK, STYL, ELVL
// ---------------------------------------------------------------------------

/// `INFO`: free-form text entered by the user.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Info {
    /// `stringData`: text up to the first NUL, invalid UTF-8 replaced.
    pub text: String,
    /// Always false; present for uniformity with the other blocks.
    pub truncated: bool,
}

impl Info {
    /// Parse from the block payload.
    pub fn from_bytes(payload: &[u8]) -> Self {
        Info {
            text: text_from(payload),
            truncated: false,
        }
    }
}

/// Read a `length`-prefixed string; returns the text and whether `length`
/// exceeded the bytes available.
fn length_prefixed(c: &Cursor<'_>, length: u32) -> (String, bool) {
    let rest = c.rest();
    let n = (length as usize).min(rest.len());
    (text_from(&rest[..n]), n < length as usize)
}

/// `VERS`: version string of one recording module, `"<module> <version>"`.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Version {
    /// `length`: string length in bytes (the block may be padded).
    pub length: u32,
    /// `stringData`.
    pub text: String,
    /// Payload shorter than the length field or `length` claims.
    pub truncated: bool,
}

impl Version {
    /// Parse from the block payload.
    pub fn from_bytes(payload: &[u8]) -> Self {
        let mut c = Cursor::new(payload);
        let length = c.u32();
        let (text, short) = length_prefixed(&c, length);
        Version {
            length,
            text,
            truncated: c.truncated() || short,
        }
    }
}

/// `DEBG`: development log text. Carries no image data.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct DebugLog {
    /// `type`: 0 = text log.
    pub kind: u32,
    /// `length`: string length in bytes.
    pub length: u32,
    /// `stringData`.
    pub text: String,
    /// Payload shorter than the fixed fields or `length` claims.
    pub truncated: bool,
}

impl DebugLog {
    /// Parse from the block payload.
    pub fn from_bytes(payload: &[u8]) -> Self {
        let mut c = Cursor::new(payload);
        let kind = c.u32();
        let length = c.u32();
        let (text, short) = length_prefixed(&c, length);
        DebugLog {
            kind,
            length,
            text,
            truncated: c.truncated() || short,
        }
    }
}

/// `MARK`: user marker set during recording.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Marker {
    /// `type`: marker kind (button / user-defined value).
    pub kind: u32,
    /// Payload was shorter than the full structure.
    pub truncated: bool,
}

impl Marker {
    /// Parse from the block payload.
    pub fn from_bytes(payload: &[u8]) -> Self {
        let mut c = Cursor::new(payload);
        let kind = c.u32();
        Marker {
            kind,
            truncated: c.truncated(),
        }
    }
}

/// `STYL`: Canon picture style in effect.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PictureStyle {
    /// `picStyleId`.
    pub pic_style_id: u32,
    /// `contrast`.
    pub contrast: i32,
    /// `sharpness`.
    pub sharpness: i32,
    /// `saturation`.
    pub saturation: i32,
    /// `colortone`.
    pub colortone: i32,
    /// `picStyleName`: NUL-padded string.
    pub name: [u8; 16],
    /// Payload was shorter than the full structure.
    pub truncated: bool,
}

impl PictureStyle {
    /// Parse from the block payload.
    pub fn from_bytes(payload: &[u8]) -> Self {
        let mut c = Cursor::new(payload);
        let mut s = PictureStyle {
            pic_style_id: c.u32(),
            contrast: c.i32(),
            sharpness: c.i32(),
            saturation: c.i32(),
            colortone: c.i32(),
            name: c.array(),
            truncated: false,
        };
        s.truncated = c.truncated();
        s
    }

    /// Style name, trimmed.
    pub fn name_str(&self) -> &str {
        c_str(&self.name)
    }
}

/// `ELVL`: electronic level (camera orientation).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Level {
    /// `roll`: degrees x 100.
    pub roll: u32,
    /// `pitch`: degrees x 100.
    pub pitch: u32,
    /// Payload was shorter than the full structure.
    pub truncated: bool,
}

impl Level {
    /// Parse from the block payload.
    pub fn from_bytes(payload: &[u8]) -> Self {
        let mut c = Cursor::new(payload);
        let mut l = Level {
            roll: c.u32(),
            pitch: c.u32(),
            truncated: false,
        };
        l.truncated = c.truncated();
        l
    }
}

// ---------------------------------------------------------------------------
// XREF, WAVI
// ---------------------------------------------------------------------------

/// One entry of an [`Xref`] index.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct XrefEntry {
    /// `fileNumber`: chunk index holding the frame.
    pub file_number: u16,
    /// `frameType`: 0 = unspecified, 1 = VIDF, 2 = AUDF.
    pub frame_type: u8,
    /// `frameOffset`: byte offset of the block within its chunk.
    pub frame_offset: u64,
}

impl XrefEntry {
    /// Encoded entry length.
    pub const SIZE: usize = 12;
}

/// `XREF`: frame offset index, written by post-processing tools.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Xref {
    /// `frameType`: bitmask, 1 = video, 2 = audio.
    pub frame_type: u32,
    /// `entryCount`: number of entries declared.
    pub entry_count: u32,
    /// The entries actually present (at most `entry_count`).
    pub entries: Vec<XrefEntry>,
    /// Fewer entries present than `entry_count` declares, or the fixed
    /// fields were short.
    pub truncated: bool,
}

impl Xref {
    /// Parse from the block payload.
    pub fn from_bytes(payload: &[u8]) -> Self {
        let mut c = Cursor::new(payload);
        let frame_type = c.u32();
        let entry_count = c.u32();
        let available = c.remaining() / XrefEntry::SIZE;
        let n = (entry_count as usize).min(available);
        let mut entries = Vec::with_capacity(n);
        for _ in 0..n {
            let file_number = c.u16();
            let _empty = c.u8();
            let frame_type = c.u8();
            let frame_offset = c.u64();
            entries.push(XrefEntry {
                file_number,
                frame_type,
                frame_offset,
            });
        }
        Xref {
            frame_type,
            entry_count,
            entries,
            truncated: c.truncated() || n < entry_count as usize,
        }
    }
}

/// `WAVI`: audio stream format (WAVEFORMATEX-like).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct WavInfo {
    /// `format`: 1 = PCM.
    pub format: u16,
    /// `channels`.
    pub channels: u16,
    /// `samplingRate`: Hz.
    pub sampling_rate: u32,
    /// `bytesPerSecond`.
    pub bytes_per_second: u32,
    /// `blockAlign`.
    pub block_align: u16,
    /// `bitsPerSample`.
    pub bits_per_sample: u16,
    /// Payload was shorter than the full structure.
    pub truncated: bool,
}

impl WavInfo {
    /// Parse from the block payload.
    pub fn from_bytes(payload: &[u8]) -> Self {
        let mut c = Cursor::new(payload);
        let mut w = WavInfo {
            format: c.u16(),
            channels: c.u16(),
            sampling_rate: c.u32(),
            bytes_per_second: c.u32(),
            block_align: c.u16(),
            bits_per_sample: c.u16(),
            truncated: false,
        };
        w.truncated = c.truncated();
        w
    }
}

// ---------------------------------------------------------------------------
// Block enum
// ---------------------------------------------------------------------------

/// A parsed block of any type.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub enum Block {
    /// `MLVI`.
    FileHeader(FileHeader),
    /// `RAWI`.
    RawInfo(RawInfo),
    /// `RAWC`.
    RawCapture(RawCapture),
    /// `VIDF` (fixed header only).
    VideoFrame(VideoFrameHeader),
    /// `AUDF` (fixed header only).
    AudioFrame(AudioFrameHeader),
    /// `WAVI`.
    WavInfo(WavInfo),
    /// `EXPO`.
    Exposure(Exposure),
    /// `LENS`.
    Lens(Lens),
    /// `ELNS`.
    ExtendedLens(ExtendedLens),
    /// `WBAL`.
    WhiteBalance(WhiteBalance),
    /// `IDNT`.
    Identity(Identity),
    /// `RTCI`.
    WallClock(WallClock),
    /// `DISO`.
    DualIso(DualIso),
    /// `CURV`.
    Curve(Curve),
    /// `DARK` (fixed fields only).
    DarkFrame(DarkFrame),
    /// `INFO`.
    Info(Info),
    /// `VERS`.
    Version(Version),
    /// `DEBG`.
    Debug(DebugLog),
    /// `MARK`.
    Marker(Marker),
    /// `STYL`.
    PictureStyle(PictureStyle),
    /// `ELVL`.
    Level(Level),
    /// `XREF`.
    Xref(Xref),
    /// `NULL` or `BKUP` padding.
    Null,
    /// Any tag this crate does not know.
    Unknown {
        /// The block's tag.
        tag: Tag,
    },
}

impl Block {
    /// The tag this block was parsed from. [`Block::Null`] reports
    /// [`Tag::NULL`] even when it came from a `BKUP` block.
    pub fn tag(&self) -> Tag {
        match self {
            Block::FileHeader(_) => Tag::MLVI,
            Block::RawInfo(_) => Tag::RAWI,
            Block::RawCapture(_) => Tag::RAWC,
            Block::VideoFrame(_) => Tag::VIDF,
            Block::AudioFrame(_) => Tag::AUDF,
            Block::WavInfo(_) => Tag::WAVI,
            Block::Exposure(_) => Tag::EXPO,
            Block::Lens(_) => Tag::LENS,
            Block::ExtendedLens(_) => Tag::ELNS,
            Block::WhiteBalance(_) => Tag::WBAL,
            Block::Identity(_) => Tag::IDNT,
            Block::WallClock(_) => Tag::RTCI,
            Block::DualIso(_) => Tag::DISO,
            Block::Curve(_) => Tag::CURV,
            Block::DarkFrame(_) => Tag::DARK,
            Block::Info(_) => Tag::INFO,
            Block::Version(_) => Tag::VERS,
            Block::Debug(_) => Tag::DEBG,
            Block::Marker(_) => Tag::MARK,
            Block::PictureStyle(_) => Tag::STYL,
            Block::Level(_) => Tag::ELVL,
            Block::Xref(_) => Tag::XREF,
            Block::Null => Tag::NULL,
            Block::Unknown { tag } => *tag,
        }
    }
}

/// Parse a block payload according to `header.tag`.
///
/// `payload` is the bytes after the 16-byte header. For `VIDF`/`AUDF` it may
/// be just the fixed frame header; pixel and sample data are never read.
/// Never fails: unknown tags become [`Block::Unknown`], short payloads set the
/// struct's `truncated` flag.
pub fn parse_block(header: &BlockHeader, payload: &[u8]) -> Block {
    match header.tag {
        Tag::MLVI => Block::FileHeader(FileHeader::from_bytes(header, payload)),
        Tag::RAWI => Block::RawInfo(RawInfo::from_bytes(payload)),
        Tag::RAWC => Block::RawCapture(RawCapture::from_bytes(payload)),
        Tag::VIDF => Block::VideoFrame(VideoFrameHeader::from_bytes(payload)),
        Tag::AUDF => Block::AudioFrame(AudioFrameHeader::from_bytes(payload)),
        Tag::WAVI => Block::WavInfo(WavInfo::from_bytes(payload)),
        Tag::EXPO => Block::Exposure(Exposure::from_bytes(payload)),
        Tag::LENS => Block::Lens(Lens::from_bytes(payload)),
        Tag::ELNS => Block::ExtendedLens(ExtendedLens::from_bytes(payload)),
        Tag::WBAL => Block::WhiteBalance(WhiteBalance::from_bytes(payload)),
        Tag::IDNT => Block::Identity(Identity::from_bytes(payload)),
        Tag::RTCI => Block::WallClock(WallClock::from_bytes(payload)),
        Tag::DISO => Block::DualIso(DualIso::from_bytes(payload)),
        Tag::CURV => Block::Curve(Curve::from_bytes(payload)),
        Tag::DARK => Block::DarkFrame(DarkFrame::from_bytes(payload)),
        Tag::INFO => Block::Info(Info::from_bytes(payload)),
        Tag::VERS => Block::Version(Version::from_bytes(payload)),
        Tag::DEBG => Block::Debug(DebugLog::from_bytes(payload)),
        Tag::MARK => Block::Marker(Marker::from_bytes(payload)),
        Tag::STYL => Block::PictureStyle(PictureStyle::from_bytes(payload)),
        Tag::ELVL => Block::Level(Level::from_bytes(payload)),
        Tag::XREF => Block::Xref(Xref::from_bytes(payload)),
        Tag::NULL | Tag::BKUP => Block::Null,
        tag => Block::Unknown { tag },
    }
}

#[cfg(test)]
mod tests {
    use super::Cfa;

    #[test]
    fn cfa_shift_follows_offset_parity() {
        assert_eq!(Cfa::Rggb.shifted(0, 0), Cfa::Rggb);
        assert_eq!(Cfa::Rggb.shifted(1, 0), Cfa::Grbg);
        assert_eq!(Cfa::Rggb.shifted(0, 1), Cfa::Gbrg);
        assert_eq!(Cfa::Rggb.shifted(1, 1), Cfa::Bggr);
        assert_eq!(Cfa::Rggb.shifted(-64, 178), Cfa::Rggb);
        assert_eq!(Cfa::Gbrg.shifted(-1, 0), Cfa::Bggr);
        assert_eq!(Cfa::Unknown(7).shifted(1, 1), Cfa::Unknown(7));
    }

    use super::*;
    use alloc::string::ToString;
    use alloc::vec;

    /// Little-endian payload builder.
    #[derive(Default)]
    struct B(Vec<u8>);

    impl B {
        fn u8(mut self, v: u8) -> Self {
            self.0.push(v);
            self
        }
        fn u16(mut self, v: u16) -> Self {
            self.0.extend_from_slice(&v.to_le_bytes());
            self
        }
        fn i16(mut self, v: i16) -> Self {
            self.0.extend_from_slice(&v.to_le_bytes());
            self
        }
        fn u32(mut self, v: u32) -> Self {
            self.0.extend_from_slice(&v.to_le_bytes());
            self
        }
        fn i32(mut self, v: i32) -> Self {
            self.0.extend_from_slice(&v.to_le_bytes());
            self
        }
        fn u64(mut self, v: u64) -> Self {
            self.0.extend_from_slice(&v.to_le_bytes());
            self
        }
        fn bytes(mut self, v: &[u8]) -> Self {
            self.0.extend_from_slice(v);
            self
        }
        fn padded(mut self, v: &[u8], n: usize) -> Self {
            let start = self.0.len();
            self.0.extend_from_slice(v);
            self.0.resize(start + n, 0);
            self
        }
    }

    fn hdr(tag: Tag, payload_len: usize, timestamp: u64) -> BlockHeader {
        BlockHeader {
            tag,
            size: (16 + payload_len) as u32,
            timestamp,
        }
    }

    #[test]
    fn file_header() {
        let p = B::default()
            .u64(0x1122_3344_5566_7788)
            .u16(1)
            .u16(3)
            .u32(0b11)
            .u16(0x21)
            .u16(0)
            .u32(500)
            .u32(0)
            .u32(23976)
            .u32(1000)
            .0;
        assert_eq!(p.len(), FileHeader::PAYLOAD_SIZE);
        let h = hdr(Tag::MLVI, p.len(), u64::from_le_bytes(*b"v2.0\0\0\0\0"));
        let f = FileHeader::from_bytes(&h, &p);
        assert!(!f.truncated);
        assert_eq!(f.version_str(), "v2.0");
        assert_eq!(f.file_guid, 0x1122_3344_5566_7788);
        assert_eq!((f.file_num, f.file_count), (1, 3));
        assert!(f.out_of_order() && f.dropped_frames());
        assert!(f.is_raw() && f.is_lj92());
        assert_eq!(f.video_frame_count, 500);
        assert!((f.fps() - 23.976).abs() < 1e-9);

        let zero = FileHeader {
            fps_den: 0,
            ..f.clone()
        };
        assert_eq!(zero.fps(), 0.0);
        assert_eq!(parse_block(&h, &p), Block::FileHeader(f));
    }

    fn rawi_payload() -> Vec<u8> {
        let mut b = B::default()
            .u16(1920)
            .u16(1080)
            .u32(1) // api_version
            .u32(0xdead_beef) // buffer pointer slot
            .i32(1290) // height
            .i32(1888) // width
            .i32(1888 * 14 / 8) // pitch
            .i32(1888 * 14 / 8 * 1290) // frame_size
            .i32(14)
            .i32(2047)
            .i32(15000)
            .i32(10)
            .i32(20) // crop_origin
            .i32(1920)
            .i32(1080) // crop_size
            .i32(28)
            .i32(146)
            .i32(1290)
            .i32(1888) // active_area y1 x1 y2 x2
            .i32(0)
            .i32(0) // exposure_bias
            .u32(0x0201_0100)
            .i32(21);
        let m = [6722, -635, -963, -4287, 12460, 2028, -908, 2162, 5668];
        for v in m {
            b = b.i32(v).i32(10000);
        }
        b.i32(1150).0
    }

    #[test]
    fn raw_info() {
        let p = rawi_payload();
        assert_eq!(p.len(), 164);
        assert_eq!(p.len() + 16, 180);
        let r = RawInfo::from_bytes(&p);
        assert!(!r.truncated);
        assert_eq!((r.x_res, r.y_res), (1920, 1080));
        assert_eq!(r.api_version, 1);
        assert_eq!((r.width, r.height), (1888, 1290));
        assert_eq!(r.bits_per_pixel, 14);
        assert_eq!((r.black_level, r.white_level), (2047, 15000));
        assert_eq!(r.crop_origin, [10, 20]);
        assert_eq!(r.crop_size, [1920, 1080]);
        assert_eq!(
            r.active_area,
            ActiveArea {
                y1: 28,
                x1: 146,
                y2: 1290,
                x2: 1888
            }
        );
        assert_eq!(r.active_width(), 1888 - 146);
        assert_eq!(r.active_height(), 1290 - 28);
        assert_eq!(r.cfa(), Cfa::Rggb);
        assert_eq!(r.calibration_illuminant1, 21);
        assert_eq!(r.dynamic_range, 1150);
        let m = r.color_matrix().unwrap();
        assert!((m[0] - 0.6722).abs() < 1e-12);
        assert!((m[8] - 0.5668).abs() < 1e-12);

        let mut bad = r.clone();
        bad.color_matrix1[3] = 0;
        assert_eq!(bad.color_matrix(), None);
        bad.color_matrix1 = [0; 18];
        assert_eq!(bad.color_matrix(), None);

        let mut inv = r.clone();
        inv.active_area.x2 = 0;
        assert_eq!(inv.active_width(), 0);
    }

    #[test]
    fn raw_info_truncated_payload() {
        let p = rawi_payload();
        // Cut inside color_matrix1: trailing fields read as zero.
        let r = RawInfo::from_bytes(&p[..100]);
        assert!(r.truncated);
        assert_eq!(r.x_res, 1920);
        assert_eq!(r.cfa_pattern, 0x0201_0100);
        assert_eq!(r.dynamic_range, 0);
        assert_eq!(r.color_matrix1[17], 0);
    }

    #[test]
    fn video_frame_header() {
        let p = B::default()
            .u32(42)
            .u16(1)
            .u16(2)
            .u16(3)
            .u16(4)
            .u32(4064)
            .bytes(&[0xAA; 8])
            .0;
        let v = VideoFrameHeader::from_bytes(&p);
        assert!(!v.truncated);
        assert_eq!(v.frame_number, 42);
        assert_eq!((v.crop_pos_x, v.crop_pos_y), (1, 2));
        assert_eq!((v.pan_pos_x, v.pan_pos_y), (3, 4));
        assert_eq!(v.data_offset(), 32 + 4064);
        assert_eq!(v.data_len(32 + 4064 + 1000), Some(1000));
        assert_eq!(v.data_len(32 + 4063), None);
        assert_eq!(v.data_len(10), None);

        let h = hdr(Tag::VIDF, 100_000, 7);
        assert_eq!(parse_block(&h, &p[..16]), Block::VideoFrame(v));

        let a = AudioFrameHeader::from_bytes(&B::default().u32(5).u32(8).0);
        assert_eq!(a.data_offset(), 24 + 8);
        assert_eq!(a.data_len(100), Some(68));
    }

    #[test]
    fn exposure() {
        let p = B::default().u32(0).u32(800).u32(800).u32(0).u64(20_000).0;
        let e = Exposure::from_bytes(&p);
        assert!(!e.truncated);
        assert_eq!(e.iso_value, 800);
        assert_eq!(e.shutter_us, 20_000);
        assert!((e.shutter_seconds() - 0.02).abs() < 1e-12);
    }

    #[test]
    fn wall_clock() {
        let p = B::default()
            .u16(5)
            .u16(4)
            .u16(13)
            .u16(28)
            .u16(9)
            .u16(126)
            .u16(3)
            .u16(300)
            .u16(0)
            .u16(0)
            .padded(b"UTC", 8)
            .0;
        assert_eq!(p.len(), 28);
        let w = WallClock::from_bytes(&p);
        assert!(!w.truncated);
        assert_eq!(w.year_ce(), 2026);
        assert_eq!(w.month(), 10);
        assert_eq!(w.zone_str(), "UTC");
        let dt = w.to_datetime().unwrap();
        assert_eq!(
            dt,
            DateTime {
                year: 2026,
                month: 10,
                day: 28,
                hour: 13,
                minute: 4,
                second: 5
            }
        );
        assert_eq!(dt.to_string(), "2026-10-28T13:04:05");

        let bad = WallClock {
            mday: 0,
            ..w.clone()
        };
        assert_eq!(bad.to_datetime(), None);
        let bad = WallClock { mon: 12, ..w };
        assert_eq!(bad.to_datetime(), None);
    }

    #[test]
    fn lens_name_trimming() {
        let p = B::default()
            .u16(50)
            .u16(65535)
            .u16(180)
            .u8(1)
            .u8(0)
            .u32(0)
            .u32(0x1234)
            .padded(b"EF50mm f/1.8 STM  ", 32)
            .padded(b"0000012345", 32)
            .0;
        let l = Lens::from_bytes(&p);
        assert!(!l.truncated);
        assert_eq!(l.name_str(), "EF50mm f/1.8 STM");
        assert_eq!(l.serial_str(), "0000012345");
        assert!((l.aperture() - 1.8).abs() < 1e-12);
        assert_eq!(l.lens_id, 0x1234);

        let e = ExtendedLens::from_bytes(
            &B::default()
                .u16(24)
                .u16(70)
                .u16(280)
                .u16(2200)
                .u32(1)
                .u8(0)
                .u8(0)
                .u8(1)
                .bytes(b"EF24-70mm f/2.8L\0junk")
                .0,
        );
        assert!(!e.truncated);
        assert_eq!(e.name, "EF24-70mm f/2.8L");
        assert_eq!(e.chipped, 1);
    }

    #[test]
    fn cfa_mapping() {
        assert_eq!(Cfa::from_pattern(0x0201_0100), Cfa::Rggb);
        assert_eq!(Cfa::from_pattern(0x0001_0102), Cfa::Bggr);
        assert_eq!(Cfa::from_pattern(0x0102_0001), Cfa::Grbg);
        assert_eq!(Cfa::from_pattern(0x0100_0201), Cfa::Gbrg);
        assert_eq!(Cfa::from_pattern(7), Cfa::Unknown(7));
        for c in [Cfa::Rggb, Cfa::Bggr, Cfa::Grbg, Cfa::Gbrg] {
            let p = c.pattern().unwrap();
            assert_eq!(Cfa::from_pattern(u32::from_le_bytes(p)), c);
        }
        assert_eq!(Cfa::Unknown(7).pattern(), None);
        assert_eq!(Cfa::Gbrg.to_string(), "GBRG");
        assert_eq!(Cfa::Unknown(7).to_string(), "unknown(0x00000007)");
    }

    #[test]
    fn short_payloads_set_truncated() {
        let e = Exposure::from_bytes(&B::default().u32(1).u32(400).0);
        assert!(e.truncated);
        assert_eq!(e.iso_value, 400);
        assert_eq!((e.iso_analog, e.digital_gain, e.shutter_us), (0, 0, 0));

        let r = RawCapture::from_bytes(&B::default().u16(5796).u16(3870).u16(100).0);
        assert!(r.truncated);
        assert_eq!(r.sensor_crop_x100, 100);
        assert_eq!(r.offset_y, 0);

        let full = RawCapture::from_bytes(
            &B::default()
                .u16(1)
                .u16(2)
                .u16(3)
                .u16(0)
                .u8(3)
                .u8(0)
                .u8(3)
                .u8(0)
                .i16(-4)
                .i16(8)
                .0,
        );
        assert!(!full.truncated);
        assert_eq!((full.offset_x, full.offset_y), (-4, 8));

        let v = VideoFrameHeader::from_bytes(&[1, 0, 0, 0]);
        assert!(v.truncated);
        assert_eq!(v.frame_space, 0);

        let x = Xref::from_bytes(
            &B::default()
                .u32(1)
                .u32(3)
                .u16(0)
                .u8(0)
                .u8(1)
                .u64(1234)
                .bytes(&[0; 5])
                .0,
        );
        assert!(x.truncated);
        assert_eq!(
            x.entries,
            vec![XrefEntry {
                file_number: 0,
                frame_type: 1,
                frame_offset: 1234
            }]
        );

        let ver = Version::from_bytes(&B::default().u32(20).bytes(b"mlv_lite 1").0);
        assert!(ver.truncated);
        assert_eq!(ver.text, "mlv_lite 1");

        let c = Curve::from_bytes(&[1, 0, 2, 0, 9]);
        assert!(c.truncated);
        assert_eq!(c.lut, vec![1, 2]);
    }

    #[test]
    fn dispatch() {
        let p = B::default().u32(1).u32(1600).0;
        let b = parse_block(&hdr(Tag::DISO, p.len(), 0), &p);
        match &b {
            Block::DualIso(d) => assert!(d.is_active() && d.iso_value == 1600),
            other => panic!("wrong variant {other:?}"),
        }
        assert_eq!(b.tag(), Tag::DISO);

        let info = parse_block(&hdr(Tag::INFO, 9, 0), b"take 3\0\0\0");
        assert_eq!(
            info,
            Block::Info(Info {
                text: "take 3".into(),
                truncated: false
            })
        );

        let vers = B::default().u32(15).bytes(b"mlv_lite abc123\0").0;
        match parse_block(&hdr(Tag::VERS, vers.len(), 0), &vers) {
            Block::Version(v) => {
                assert_eq!(v.text, "mlv_lite abc123");
                assert!(!v.truncated);
            }
            other => panic!("wrong variant {other:?}"),
        }

        assert_eq!(parse_block(&hdr(Tag::NULL, 4, 0), &[0; 4]), Block::Null);
        assert_eq!(parse_block(&hdr(Tag::BKUP, 0, 0), &[]), Block::Null);

        let unk = Tag(*b"ZZZZ");
        let u = parse_block(&hdr(unk, 3, 0), &[1, 2, 3]);
        assert_eq!(u, Block::Unknown { tag: unk });
        assert_eq!(u.tag(), unk);

        let idnt = B::default()
            .padded(b"Canon EOS 5D Mark III", 32)
            .u32(0x8000_0285)
            .padded(b"123456789", 32)
            .0;
        match parse_block(&hdr(Tag::IDNT, idnt.len(), 0), &idnt) {
            Block::Identity(i) => {
                assert_eq!(i.camera_name_str(), "Canon EOS 5D Mark III");
                assert_eq!(i.camera_serial_str(), "123456789");
                assert_eq!(i.camera_model, 0x8000_0285);
            }
            other => panic!("wrong variant {other:?}"),
        }

        let rawi = rawi_payload();
        let b = parse_block(&hdr(Tag::RAWI, rawi.len(), 0), &rawi);
        assert_eq!(b.tag(), Tag::RAWI);
        assert!(matches!(b, Block::RawInfo(ref r) if r.bits_per_pixel == 14));

        let dark = DarkFrame::from_bytes(&[0u8; 70]);
        assert!(!dark.truncated);
        assert_eq!(DarkFrame::FIXED_SIZE, 68);
        assert!(DarkFrame::from_bytes(&[0u8; 60]).truncated);
    }
}
