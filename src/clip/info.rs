// SPDX-License-Identifier: MIT OR Apache-2.0
//! `ClipInfo`: everything a raw consumer needs, flattened from the first
//! occurrence of each one-per-clip block.

use crate::block::{
    Block, Cfa, Curve, DateTime, DualIso, Exposure, FileHeader, Identity, Info, Lens, RawCapture,
    RawInfo, Version, WallClock, WavInfo, WhiteBalance,
};
use crate::{Error, Result, Tag, Unsupported};

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

/// Frame encoding, derived from `videoClass`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Compression {
    /// Packed 10/12/14-bit samples.
    None,
    /// Lossless JPEG.
    Lj92,
    /// Something this crate does not decode. The raw `videoClass` value is
    /// in [`ClipInfo::file_header`].
    Unsupported(Unsupported),
}

impl Compression {
    /// Classify a `videoClass` value.
    pub fn from_video_class(class: u16) -> Compression {
        let unsupported = if class == 0 {
            Some(Unsupported::AudioOnly)
        } else if class & FileHeader::FLAG_MCRAW != 0 {
            Some(Unsupported::McRaw)
        } else if class & FileHeader::FLAG_LZMA != 0 {
            Some(Unsupported::Lzma)
        } else if class & FileHeader::FLAG_DELTA != 0 {
            Some(Unsupported::Delta)
        } else if class & FileHeader::FLAG_JPEG2K != 0 {
            Some(Unsupported::Jpeg2000)
        } else if class & 0x0F != FileHeader::VIDEO_CLASS_RAW {
            Some(Unsupported::VideoClass(class))
        } else {
            None
        };
        match unsupported {
            Some(u) => Compression::Unsupported(u),
            None if class & FileHeader::FLAG_LJ92 != 0 => Compression::Lj92,
            None => Compression::None,
        }
    }
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
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
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
    pub(super) audio_frames: usize,
}

impl InfoBuilder {
    /// Tags whose payload the builder or the timelines consume. Other
    /// metadata blocks are indexed by location only and never parsed.
    pub(super) fn wants(tag: Tag) -> bool {
        matches!(
            tag,
            Tag::RAWI
                | Tag::RAWC
                | Tag::IDNT
                | Tag::RTCI
                | Tag::DISO
                | Tag::CURV
                | Tag::INFO
                | Tag::VERS
                | Tag::WAVI
                | Tag::DARK
                | Tag::EXPO
                | Tag::WBAL
                | Tag::LENS
        )
    }

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
            audio_frames: 0,
        }
    }

    pub(super) fn absorb(&mut self, timestamp: u64, block: Block, chunk: u16, offset: u64) {
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
        chunk_count: usize,
        frame_count: usize,
        expo: &[(u64, Exposure)],
        wbal: &[(u64, WhiteBalance)],
        lens: &[(u64, Lens)],
    ) -> Result<ClipInfo> {
        let raw_info = self.raw_info.ok_or(Error::MissingBlock(Tag::RAWI))?;
        let fh = self.file_header;
        let width = u32::from(raw_info.x_res);
        let height = u32::from(raw_info.y_res);

        let aa = &raw_info.active_area;
        let active_area = clamp_rect(
            aa.x1,
            aa.y1,
            raw_info.active_width(),
            raw_info.active_height(),
            width,
            height,
        );
        let crop = clamp_rect(
            raw_info.crop_origin[0],
            raw_info.crop_origin[1],
            raw_info.crop_size[0].max(0) as u32,
            raw_info.crop_size[1].max(0) as u32,
            width,
            height,
        );

        let camera = self
            .identity
            .as_ref()
            .map_or_else(CameraId::default, |id| CameraId {
                name: id.camera_name_str().to_string(),
                model_id: id.camera_model,
                serial: id.camera_serial_str().to_string(),
            });

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
            compression: Compression::from_video_class(fh.video_class),
            dual_iso: self.dual_iso.as_ref().is_some_and(|d| d.is_active()),
            camera,
            recorded_at: self.wall_clock.as_ref().and_then(|(_, w)| w.to_datetime()),
            chunk_count,
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
/// frame when the input is empty or starts outside it.
fn clamp_rect(x: i32, y: i32, w: u32, h: u32, fw: u32, fh: u32) -> Rect {
    let whole = Rect {
        x: 0,
        y: 0,
        width: fw,
        height: fh,
    };
    let (Ok(x), Ok(y)) = (u32::try_from(x), u32::try_from(y)) else {
        return whole;
    };
    if w == 0 || h == 0 || x >= fw || y >= fh {
        return whole;
    }
    Rect {
        x,
        y,
        width: w.min(fw - x),
        height: h.min(fh - y),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clamp_rect_handles_garbage() {
        let whole = Rect {
            x: 0,
            y: 0,
            width: 100,
            height: 100,
        };
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
        assert_eq!(clamp_rect(-5, 0, 10, 10, 100, 100), whole);
        assert_eq!(clamp_rect(0, 0, 0, 0, 100, 100), whole);
        assert_eq!(clamp_rect(i32::MIN, i32::MAX, u32::MAX, 1, 100, 100), whole);
    }

    #[test]
    fn video_class_classification() {
        assert_eq!(Compression::from_video_class(0x01), Compression::None);
        assert_eq!(Compression::from_video_class(0x21), Compression::Lj92);
        assert_eq!(
            Compression::from_video_class(0),
            Compression::Unsupported(Unsupported::AudioOnly)
        );
        assert_eq!(
            Compression::from_video_class(0x81),
            Compression::Unsupported(Unsupported::Lzma)
        );
        assert_eq!(
            Compression::from_video_class(0xA1),
            Compression::Unsupported(Unsupported::Lzma)
        );
        assert_eq!(
            Compression::from_video_class(0x41),
            Compression::Unsupported(Unsupported::Delta)
        );
        assert_eq!(
            Compression::from_video_class(0x11),
            Compression::Unsupported(Unsupported::Jpeg2000)
        );
        assert_eq!(
            Compression::from_video_class(0x101),
            Compression::Unsupported(Unsupported::McRaw)
        );
        assert_eq!(
            Compression::from_video_class(0x02),
            Compression::Unsupported(Unsupported::VideoClass(2))
        );
    }
}
