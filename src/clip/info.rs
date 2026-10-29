// SPDX-License-Identifier: MIT OR Apache-2.0
//! `ClipInfo`: everything a raw consumer needs, flattened from the first
//! occurrence of each one-per-clip block.

use crate::block::{
    Block, Cfa, Curve, DateTime, DualIso, Exposure, FileHeader, Identity, Info, Lens, RawCapture,
    RawInfo, Version, WallClock, WavInfo, WhiteBalance,
};
use crate::{Error, Result, Tag};

/// Axis-aligned rectangle in pixels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Rect {
    /// Left edge.
    pub x: u32,
    /// Top edge.
    pub y: u32,
    /// Width.
    pub width: u32,
    /// Height.
    pub height: u32,
}

/// Frame encoding, from `videoClass`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Compression {
    /// Packed 10/12/14-bit samples.
    #[default]
    None,
    /// Lossless JPEG.
    Lj92,
    /// Something this crate does not decode; the raw `videoClass` value.
    Unsupported(u16),
}

/// Camera identity from `IDNT`.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct CameraId {
    /// Model name, e.g. `Canon EOS 5D Mark III`.
    pub name: String,
    /// Canon model id (`cameraModel`).
    pub model_id: u32,
    /// Body serial number, if the camera provided one.
    pub serial: String,
}

/// Clip-wide metadata.
///
/// The flattened fields at the top are what a raw pipeline needs. The
/// `Option<Block>` fields underneath expose the parsed source blocks for
/// anything else.
#[derive(Debug, Clone, PartialEq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ClipInfo {
    /// Stored frame width (`RAWI.xRes`).
    pub width: u32,
    /// Stored frame height (`RAWI.yRes`).
    pub height: u32,
    /// Bits per sample of the decoded data (`raw_info.bits_per_pixel`).
    pub bits_per_pixel: u32,
    /// Black level in sample units.
    pub black_level: u32,
    /// White level in sample units.
    pub white_level: u32,
    /// Colour filter array layout.
    pub cfa: Cfa,
    /// DNG-style active area, clamped to the stored frame.
    pub active_area: Rect,
    /// DNG default crop (`raw_info.crop`), clamped to the stored frame.
    pub crop: Rect,
    /// `ColorMatrix1` as 9 row-major XYZ-to-camera coefficients.
    pub color_matrix: Option<[f64; 9]>,
    /// `CalibrationIlluminant1` (EXIF light source code, 21 = D65).
    pub calibration_illuminant: u32,
    /// Nominal frame rate.
    pub fps: f64,
    /// Frame rate as a rational `(numerator, denominator)`.
    pub fps_rational: (u32, u32),
    /// Decodable video frames across all chunks.
    pub frame_count: usize,
    /// Audio frames across all chunks (not decoded by this crate).
    pub audio_frame_count: usize,
    /// Encoding of the frame payloads.
    pub compression: Compression,
    /// True if a `DISO` block reports dual-ISO active. The data is returned
    /// as recorded; interleaved-ISO processing is the consumer's job.
    pub dual_iso: bool,
    /// Camera identity.
    pub camera: CameraId,
    /// Wall-clock time of the first `RTCI` block.
    pub recorded_at: Option<DateTime>,
    /// Number of chunk files.
    pub chunk_count: usize,
    /// `fileFlags` bit 0: frames were written out of order.
    pub out_of_order: bool,
    /// `fileFlags` bit 1: frames were dropped while recording.
    pub dropped_frames: bool,

    /// Main file header.
    pub file_header: FileHeader,
    /// Raw format block.
    pub raw_info: RawInfo,
    /// Sensor capture info, if present.
    pub raw_capture: Option<RawCapture>,
    /// Camera identity block, if present.
    pub identity: Option<Identity>,
    /// First real-time clock block, if present.
    pub wall_clock: Option<WallClock>,
    /// Dual-ISO block, if present.
    pub dual_iso_block: Option<DualIso>,
    /// Decode curve LUT, if present. Not applied by this crate.
    pub curve: Option<Curve>,
    /// Free-form info string, if present.
    pub info: Option<Info>,
    /// Module version strings.
    pub versions: Vec<Version>,
    /// Audio format, if the clip has audio.
    pub wav_info: Option<WavInfo>,
    /// Exposure at the start of the clip.
    pub exposure: Option<Exposure>,
    /// White balance at the start of the clip.
    pub white_balance: Option<WhiteBalance>,
    /// Lens at the start of the clip.
    pub lens: Option<Lens>,
    /// Location of the `DARK` block, if present: (chunk, offset).
    pub dark_frame: Option<(u16, u64)>,
}

/// Accumulates first-occurrence blocks during the index pass.
pub(super) struct InfoBuilder {
    file_header: FileHeader,
    raw_info: Option<RawInfo>,
    raw_capture: Option<RawCapture>,
    identity: Option<Identity>,
    wall_clock: Option<(u64, WallClock)>,
    dual_iso: Option<DualIso>,
    curve: Option<Curve>,
    info: Option<Info>,
    versions: Vec<Version>,
    wav_info: Option<WavInfo>,
    dark_frame: Option<(u16, u64)>,
    chunks: u16,
    pub(super) audio_frames: usize,
}

impl InfoBuilder {
    pub(super) fn new(file_header: FileHeader) -> Self {
        InfoBuilder {
            file_header,
            raw_info: None,
            raw_capture: None,
            identity: None,
            wall_clock: None,
            dual_iso: None,
            curve: None,
            info: None,
            versions: Vec::new(),
            wav_info: None,
            dark_frame: None,
            chunks: 0,
            audio_frames: 0,
        }
    }

    pub(super) fn absorb(&mut self, timestamp: u64, block: Block, chunk: u16, offset: u64) {
        self.chunks = self.chunks.max(chunk + 1);
        match block {
            Block::RawInfo(b) => {
                self.raw_info.get_or_insert(b);
            }
            Block::RawCapture(b) => {
                self.raw_capture.get_or_insert(b);
            }
            Block::Identity(b) => {
                self.identity.get_or_insert(b);
            }
            Block::WallClock(b) => {
                // Keep the earliest by timestamp; RTCI can repeat.
                if self.wall_clock.as_ref().is_none_or(|(t, _)| timestamp < *t) {
                    self.wall_clock = Some((timestamp, b));
                }
            }
            Block::DualIso(b) => {
                self.dual_iso.get_or_insert(b);
            }
            Block::Curve(b) => {
                self.curve.get_or_insert(b);
            }
            Block::Info(b) => {
                self.info.get_or_insert(b);
            }
            Block::Version(b) => self.versions.push(b),
            Block::WavInfo(b) => {
                self.wav_info.get_or_insert(b);
            }
            Block::DarkFrame(_) => {
                self.dark_frame.get_or_insert((chunk, offset));
            }
            _ => {}
        }
    }

    pub(super) fn finish(
        self,
        frame_count: usize,
        expo: &[(u64, Exposure)],
        wbal: &[(u64, WhiteBalance)],
        lens: &[(u64, Lens)],
    ) -> Result<ClipInfo> {
        let raw_info = self.raw_info.ok_or(Error::MissingBlock(Tag::RAWI))?;
        let fh = self.file_header;
        let width = raw_info.x_res as u32;
        let height = raw_info.y_res as u32;

        let class = fh.video_class;
        let compression = if class & 0x0F != FileHeader::VIDEO_CLASS_RAW
            || class
                & (FileHeader::FLAG_LZMA
                    | FileHeader::FLAG_DELTA
                    | FileHeader::FLAG_JPEG2K
                    | FileHeader::FLAG_MCRAW)
                != 0
        {
            Compression::Unsupported(class)
        } else if class & FileHeader::FLAG_LJ92 != 0 {
            Compression::Lj92
        } else {
            Compression::None
        };

        let aa = &raw_info.active_area;
        let active_area = clamp_rect(aa.x1, aa.y1, aa.x2 - aa.x1, aa.y2 - aa.y1, width, height);
        let crop = clamp_rect(
            raw_info.crop_origin[0],
            raw_info.crop_origin[1],
            raw_info.crop_size[0],
            raw_info.crop_size[1],
            width,
            height,
        );

        let camera = self
            .identity
            .as_ref()
            .map(|id| CameraId {
                name: id.camera_name_str().to_string(),
                model_id: id.camera_model,
                serial: id.camera_serial_str().to_string(),
            })
            .unwrap_or_default();

        Ok(ClipInfo {
            width,
            height,
            bits_per_pixel: raw_info.bits_per_pixel.max(0) as u32,
            black_level: raw_info.black_level.max(0) as u32,
            white_level: raw_info.white_level.max(0) as u32,
            cfa: raw_info.cfa(),
            active_area,
            crop,
            color_matrix: raw_info.color_matrix(),
            calibration_illuminant: raw_info.calibration_illuminant1.max(0) as u32,
            fps: fh.fps(),
            fps_rational: (fh.fps_num, fh.fps_den),
            frame_count,
            audio_frame_count: self.audio_frames,
            compression,
            dual_iso: self.dual_iso.as_ref().is_some_and(|d| d.is_active()),
            camera,
            recorded_at: self.wall_clock.as_ref().and_then(|(_, w)| w.to_datetime()),
            chunk_count: self.chunks.max(1) as usize,
            out_of_order: fh.out_of_order(),
            dropped_frames: fh.dropped_frames(),
            file_header: fh,
            raw_info,
            raw_capture: self.raw_capture,
            identity: self.identity,
            wall_clock: self.wall_clock.map(|(_, w)| w),
            dual_iso_block: self.dual_iso,
            curve: self.curve,
            info: self.info,
            versions: self.versions,
            wav_info: self.wav_info,
            exposure: expo.first().map(|e| e.1.clone()),
            white_balance: wbal.first().map(|e| e.1.clone()),
            lens: lens.first().map(|e| e.1.clone()),
            dark_frame: self.dark_frame,
        })
    }
}

/// Clamp a possibly-garbage rectangle to the frame; falls back to the whole
/// frame when the input is empty or inverted.
fn clamp_rect(x: i32, y: i32, w: i32, h: i32, fw: u32, fh: u32) -> Rect {
    if w <= 0 || h <= 0 || x < 0 || y < 0 || x as u32 >= fw || y as u32 >= fh {
        return Rect {
            x: 0,
            y: 0,
            width: fw,
            height: fh,
        };
    }
    let x = x as u32;
    let y = y as u32;
    Rect {
        x,
        y,
        width: (w as u32).min(fw - x),
        height: (h as u32).min(fh - y),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clamp_rect_handles_garbage() {
        assert_eq!(
            clamp_rect(0, 0, 1600, 900, 1600, 900),
            Rect {
                x: 0,
                y: 0,
                width: 1600,
                height: 900
            }
        );
        assert_eq!(
            clamp_rect(10, 20, 5000, 5000, 1600, 900),
            Rect {
                x: 10,
                y: 20,
                width: 1590,
                height: 880
            }
        );
        assert_eq!(
            clamp_rect(-5, 0, 10, 10, 100, 100),
            Rect {
                x: 0,
                y: 0,
                width: 100,
                height: 100
            }
        );
        assert_eq!(
            clamp_rect(0, 0, 0, 0, 100, 100),
            Rect {
                x: 0,
                y: 0,
                width: 100,
                height: 100
            }
        );
    }
}
