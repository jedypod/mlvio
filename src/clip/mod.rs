// SPDX-License-Identifier: MIT OR Apache-2.0
//! Clip-level API: chunk discovery, frame index, metadata timelines and
//! frame decoding.
//!
//! A clip is `NAME.MLV` plus optional `NAME.M00` .. `NAME.M99` chunks that
//! share a `fileGuid`. [`Clip::open`] resolves the siblings, scans every
//! block header once, and builds a timestamp-sorted frame index. Frames are
//! decoded on demand with [`Clip::read_frame`] or [`Clip::frame`]; the
//! `Clip` takes `&self` throughout and is `Sync`, so callers can decode in
//! parallel from several threads over one open clip.

mod info;

use std::fs::File;
use std::path::{Path, PathBuf};

use crate::block::{
    Block, BlockHeader, Exposure, FileHeader, Lens, Tag, VideoFrameHeader, WhiteBalance,
    parse_block,
};
use crate::decode::{lj92, unpack};
use crate::scan::{BlockIter, MAX_META_PAYLOAD, ReadAt, read_payload};
use crate::{Error, Result, Unsupported};

pub use info::{CameraId, ClipInfo, Compression, Rect};

/// Location of one video frame's pixel payload.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FrameRef {
    /// Index into the clip's chunk list.
    pub chunk: u16,
    /// Offset of the `VIDF` block within the chunk.
    pub block_offset: u64,
    /// Offset of the first pixel byte within the chunk.
    pub data_offset: u64,
    /// Pixel payload length in bytes (after `frameSpace` padding).
    pub data_len: u32,
    /// Block timestamp, microseconds since recording start.
    pub timestamp: u64,
    /// `frameNumber` as written by the camera.
    pub frame_number: u32,
    /// `cropPosX`, `cropPosY`.
    pub crop: (u16, u16),
    /// `panPosX`, `panPosY`.
    pub pan: (u16, u16),
}

/// Location of a non-frame block, for callers that want to re-read it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct MetaRef {
    /// Index into the clip's chunk list.
    pub chunk: u16,
    /// Offset of the block within the chunk.
    pub offset: u64,
    /// The block header.
    pub header: BlockHeader,
}

/// A non-fatal problem found while indexing.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Warning {
    /// Chunk index the problem was found in.
    pub chunk: u16,
    /// Byte offset within the chunk.
    pub offset: u64,
    /// Description.
    pub message: String,
}

/// Per-frame metadata: the frame's own header fields plus the exposure,
/// white balance and lens settings in effect at its timestamp.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FrameMeta {
    /// Index in the sorted frame list.
    pub index: usize,
    /// Camera frame number.
    pub frame_number: u32,
    /// Microseconds since recording start.
    pub timestamp: u64,
    /// Sensor crop position in 8x2 blocks (`cropPosX/Y`).
    pub crop_pos: (u16, u16),
    /// Pan position in pixels (`panPosX/Y`).
    pub pan_pos: (u16, u16),
    /// Chunk the frame lives in.
    pub chunk: u16,
    /// Latest `EXPO` at or before the frame, else the first in the clip.
    pub exposure: Option<Exposure>,
    /// Latest `WBAL` at or before the frame, else the first in the clip.
    pub white_balance: Option<WhiteBalance>,
    /// Latest `LENS` at or before the frame, else the first in the clip.
    pub lens: Option<Lens>,
}

/// An owned, decoded frame.
#[derive(Debug, Clone, PartialEq)]
pub struct Frame {
    /// Width in pixels.
    pub width: usize,
    /// Height in pixels.
    pub height: usize,
    /// Row-major Bayer mosaic at the stored bit depth.
    pub data: Vec<u16>,
    /// Metadata for this frame.
    pub meta: FrameMeta,
}

/// Reusable byte buffer for frame payloads, to avoid an allocation per frame.
#[derive(Debug, Default)]
pub struct FrameScratch {
    bytes: Vec<u8>,
}

impl FrameScratch {
    /// Empty scratch; grows on first use.
    pub fn new() -> Self {
        Self::default()
    }
}

struct Chunk<R> {
    source: R,
    path: Option<PathBuf>,
    header: FileHeader,
}

/// An opened MLV clip. See the [module docs](self).
pub struct Clip<R = File> {
    chunks: Vec<Chunk<R>>,
    frames: Vec<FrameRef>,
    meta: Vec<MetaRef>,
    info: ClipInfo,
    expo: Vec<(u64, Exposure)>,
    wbal: Vec<(u64, WhiteBalance)>,
    lens: Vec<(u64, Lens)>,
    warnings: Vec<Warning>,
}

impl Clip<File> {
    /// Open `path` and its sibling chunks.
    ///
    /// Chunks are `NAME.M00` .. `NAME.M99` next to the file, matched case
    /// insensitively, stopping at the first missing number. Each must carry
    /// the same `fileGuid` as the main file, otherwise
    /// [`Error::ChunkGuidMismatch`] is returned; use [`Clip::open_single`]
    /// to ignore siblings.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let mut paths = vec![path.to_path_buf()];
        paths.extend(find_chunks(path)?);
        Self::open_paths(paths)
    }

    /// Open exactly one file, ignoring any sibling chunks.
    pub fn open_single(path: impl AsRef<Path>) -> Result<Self> {
        Self::open_paths(vec![path.as_ref().to_path_buf()])
    }

    /// Open an explicit list of chunk files. The first is treated as the main
    /// file for GUID checking; order is taken from each chunk's `fileNum`.
    pub fn open_paths(paths: Vec<PathBuf>) -> Result<Self> {
        let mut sources = Vec::with_capacity(paths.len());
        for p in &paths {
            sources.push(
                File::open(p)
                    .map_err(|e| std::io::Error::new(e.kind(), format!("{}: {e}", p.display())))?,
            );
        }
        Self::build(sources, paths.into_iter().map(Some).collect())
    }
}

impl<R: ReadAt> Clip<R> {
    /// Build a clip from in-memory or custom sources, one per chunk.
    pub fn from_sources(sources: Vec<R>) -> Result<Self> {
        let n = sources.len();
        Self::build(sources, vec![None; n])
    }

    fn build(sources: Vec<R>, paths: Vec<Option<PathBuf>>) -> Result<Self> {
        if sources.is_empty() {
            return Err(Error::NotMlv {
                path: String::from("<no sources>"),
            });
        }
        // Read every MLVI first so GUID/order problems surface before the scan.
        let mut chunks = Vec::with_capacity(sources.len());
        for (source, path) in sources.into_iter().zip(paths) {
            let name = path
                .as_ref()
                .map(|p| p.display().to_string())
                .unwrap_or_else(|| String::from("<memory>"));
            let header =
                read_file_header(&source).ok_or_else(|| Error::NotMlv { path: name.clone() })?;
            chunks.push(Chunk {
                source,
                path,
                header,
            });
        }
        let guid = chunks[0].header.file_guid;
        if let Some(bad) = chunks.iter().find(|c| c.header.file_guid != guid) {
            return Err(Error::ChunkGuidMismatch {
                path: bad
                    .path
                    .as_ref()
                    .map(|p| p.display().to_string())
                    .unwrap_or_default(),
                expected: guid,
                got: bad.header.file_guid,
            });
        }
        chunks.sort_by_key(|c| c.header.file_num);

        let mut clip = Clip {
            chunks,
            frames: Vec::new(),
            meta: Vec::new(),
            info: ClipInfo::default(),
            expo: Vec::new(),
            wbal: Vec::new(),
            lens: Vec::new(),
            warnings: Vec::new(),
        };
        clip.index()?;
        Ok(clip)
    }

    fn index(&mut self) -> Result<()> {
        let mut builder = info::InfoBuilder::new(self.chunks[0].header.clone());
        for ci in 0..self.chunks.len() {
            let chunk_idx = ci as u16;
            let source = &self.chunks[ci].source;
            for item in BlockIter::new(source) {
                let (offset, header) = match item {
                    Ok(x) => x,
                    Err(Error::Truncated { offset }) => {
                        self.warnings.push(Warning {
                            chunk: chunk_idx,
                            offset,
                            message: String::from(
                                "file ends inside a block; remaining data ignored",
                            ),
                        });
                        break;
                    }
                    Err(Error::BadBlock { offset, reason }) => {
                        self.warnings.push(Warning {
                            chunk: chunk_idx,
                            offset,
                            message: format!("bad block ({reason}); scan of this chunk stopped"),
                        });
                        break;
                    }
                    Err(e) => return Err(e),
                };
                match header.tag {
                    Tag::VIDF => {
                        let payload =
                            read_payload(source, offset, &header, VideoFrameHeader::SIZE)?;
                        let vh = VideoFrameHeader::from_bytes(&payload);
                        let Some(data_len) = vh.data_len(header.size) else {
                            self.warnings.push(Warning {
                                chunk: chunk_idx,
                                offset,
                                message: String::from(
                                    "VIDF frameSpace larger than block; frame skipped",
                                ),
                            });
                            continue;
                        };
                        self.frames.push(FrameRef {
                            chunk: chunk_idx,
                            block_offset: offset,
                            data_offset: offset + vh.data_offset(),
                            data_len,
                            timestamp: header.timestamp,
                            frame_number: vh.frame_number,
                            crop: (vh.crop_pos_x, vh.crop_pos_y),
                            pan: (vh.pan_pos_x, vh.pan_pos_y),
                        });
                    }
                    Tag::AUDF => {
                        self.meta.push(MetaRef {
                            chunk: chunk_idx,
                            offset,
                            header,
                        });
                        builder.audio_frames += 1;
                    }
                    Tag::MLVI | Tag::NULL | Tag::BKUP => {
                        self.meta.push(MetaRef {
                            chunk: chunk_idx,
                            offset,
                            header,
                        });
                    }
                    _ => {
                        self.meta.push(MetaRef {
                            chunk: chunk_idx,
                            offset,
                            header,
                        });
                        // DARK frames can be large; read only the fixed header for those.
                        let cap = if header.tag == Tag::DARK {
                            128
                        } else {
                            MAX_META_PAYLOAD
                        };
                        let payload = read_payload(source, offset, &header, cap)?;
                        match parse_block(&header, &payload) {
                            Block::Exposure(e) => self.expo.push((header.timestamp, e)),
                            Block::WhiteBalance(w) => self.wbal.push((header.timestamp, w)),
                            Block::Lens(l) => self.lens.push((header.timestamp, l)),
                            other => builder.absorb(header.timestamp, other, chunk_idx, offset),
                        }
                    }
                }
            }
        }

        // Timestamp order, tie-break on frame number, drop duplicate numbers.
        self.frames.sort_by_key(|f| (f.timestamp, f.frame_number));
        let before = self.frames.len();
        let mut seen = std::collections::HashSet::with_capacity(before);
        self.frames.retain(|f| seen.insert(f.frame_number));
        if self.frames.len() != before {
            self.warnings.push(Warning {
                chunk: 0,
                offset: 0,
                message: format!(
                    "{} duplicate frame number(s) dropped",
                    before - self.frames.len()
                ),
            });
        }
        self.expo.sort_by_key(|e| e.0);
        self.wbal.sort_by_key(|e| e.0);
        self.lens.sort_by_key(|e| e.0);

        self.info = builder.finish(self.frames.len(), &self.expo, &self.wbal, &self.lens)?;
        Ok(())
    }

    /// Clip-wide metadata.
    pub fn info(&self) -> &ClipInfo {
        &self.info
    }

    /// Number of decodable video frames.
    pub fn frame_count(&self) -> usize {
        self.frames.len()
    }

    /// Frame locations in playback (timestamp) order.
    pub fn frames(&self) -> &[FrameRef] {
        &self.frames
    }

    /// Every non-`VIDF` block seen while indexing, in file order.
    pub fn meta_blocks(&self) -> &[MetaRef] {
        &self.meta
    }

    /// Non-fatal problems found while indexing.
    pub fn warnings(&self) -> &[Warning] {
        &self.warnings
    }

    /// Number of chunk files.
    pub fn chunk_count(&self) -> usize {
        self.chunks.len()
    }

    /// Path of chunk `i`, if it was opened from a file.
    pub fn chunk_path(&self, i: usize) -> Option<&Path> {
        self.chunks.get(i).and_then(|c| c.path.as_deref())
    }

    /// `MLVI` header of chunk `i`.
    pub fn chunk_header(&self, i: usize) -> Option<&FileHeader> {
        self.chunks.get(i).map(|c| &c.header)
    }

    /// Underlying source of chunk `i`.
    pub fn chunk_source(&self, i: usize) -> Option<&R> {
        self.chunks.get(i).map(|c| &c.source)
    }

    /// `EXPO` timeline: (timestamp, exposure) sorted by time.
    pub fn exposures(&self) -> &[(u64, Exposure)] {
        &self.expo
    }

    /// `WBAL` timeline.
    pub fn white_balances(&self) -> &[(u64, WhiteBalance)] {
        &self.wbal
    }

    /// `LENS` timeline.
    pub fn lenses(&self) -> &[(u64, Lens)] {
        &self.lens
    }

    fn frame_ref(&self, i: usize) -> Result<&FrameRef> {
        self.frames.get(i).ok_or(Error::FrameOutOfRange(i))
    }

    /// Metadata for frame `i`.
    pub fn frame_meta(&self, i: usize) -> Result<FrameMeta> {
        let f = self.frame_ref(i)?;
        Ok(FrameMeta {
            index: i,
            frame_number: f.frame_number,
            timestamp: f.timestamp,
            crop_pos: f.crop,
            pan_pos: f.pan,
            chunk: f.chunk,
            exposure: latest_at(&self.expo, f.timestamp).cloned(),
            white_balance: latest_at(&self.wbal, f.timestamp).cloned(),
            lens: latest_at(&self.lens, f.timestamp).cloned(),
        })
    }

    /// Read the undecoded pixel payload of frame `i` into `buf` (cleared first).
    pub fn raw_payload(&self, i: usize, buf: &mut Vec<u8>) -> Result<()> {
        let f = self.frame_ref(i)?;
        let source = &self.chunks[f.chunk as usize].source;
        buf.clear();
        buf.resize(f.data_len as usize, 0);
        source.read_exact_at(f.data_offset, buf)?;
        Ok(())
    }

    /// Decode frame `i` into `dst`, which must hold `width * height` samples.
    pub fn read_frame(&self, i: usize, dst: &mut [u16]) -> Result<()> {
        let mut scratch = FrameScratch::new();
        self.read_frame_with(i, dst, &mut scratch)
    }

    /// Like [`read_frame`](Self::read_frame) but reusing `scratch` for the
    /// compressed/packed bytes.
    pub fn read_frame_with(
        &self,
        i: usize,
        dst: &mut [u16],
        scratch: &mut FrameScratch,
    ) -> Result<()> {
        let info = &self.info;
        let pixels = info.width as usize * info.height as usize;
        if dst.len() != pixels {
            return Err(Error::GeometryMismatch {
                expected: pixels,
                got: dst.len(),
            });
        }
        match info.compression {
            Compression::Unsupported(flags) => return Err(unsupported_for(flags)),
            Compression::None | Compression::Lj92 => {}
        }
        self.raw_payload(i, &mut scratch.bytes)?;
        let bytes = &scratch.bytes;
        match info.compression {
            Compression::None => {
                let need = unpack::needed_bytes(pixels, info.bits_per_pixel);
                if bytes.len() < need {
                    return Err(Error::FrameTooShort {
                        expected: need,
                        got: bytes.len(),
                    });
                }
                unpack::unpack(info.bits_per_pixel, bytes, dst)
            }
            Compression::Lj92 => {
                let hdr = lj92::parse(bytes)?;
                let got = hdr.sample_count();
                if got != pixels {
                    return Err(Error::GeometryMismatch {
                        expected: pixels,
                        got,
                    });
                }
                lj92::decode_into(bytes, &hdr, dst)?;
                Ok(())
            }
            Compression::Unsupported(flags) => Err(unsupported_for(flags)),
        }
    }

    /// Decode frame `i` into a new [`Frame`].
    pub fn frame(&self, i: usize) -> Result<Frame> {
        let meta = self.frame_meta(i)?;
        let (width, height) = (self.info.width as usize, self.info.height as usize);
        let mut data = vec![0u16; width * height];
        self.read_frame(i, &mut data)?;
        Ok(Frame {
            width,
            height,
            data,
            meta,
        })
    }
}

/// Map `videoClass` flags to the specific unsupported variant.
fn unsupported_for(flags: u16) -> Error {
    let u = if flags & FileHeader::FLAG_MCRAW != 0 {
        Unsupported::McRaw
    } else if flags & FileHeader::FLAG_LZMA != 0 {
        Unsupported::Lzma
    } else if flags & FileHeader::FLAG_DELTA != 0 {
        Unsupported::Delta
    } else if flags & FileHeader::FLAG_JPEG2K != 0 {
        Unsupported::Jpeg2000
    } else if flags == 0 {
        Unsupported::AudioOnly
    } else {
        Unsupported::VideoClass(flags)
    };
    Error::Unsupported(u)
}

/// Latest entry with timestamp <= `ts`, else the first entry.
fn latest_at<T>(timeline: &[(u64, T)], ts: u64) -> Option<&T> {
    let n = timeline.partition_point(|e| e.0 <= ts);
    if n > 0 {
        Some(&timeline[n - 1].1)
    } else {
        timeline.first().map(|e| &e.1)
    }
}

fn read_file_header<R: ReadAt + ?Sized>(source: &R) -> Option<FileHeader> {
    let mut hdr = [0u8; BlockHeader::SIZE];
    source.read_exact_at(0, &mut hdr).ok()?;
    let header = BlockHeader::from_bytes(&hdr).ok()?;
    if header.tag != Tag::MLVI {
        return None;
    }
    let payload = read_payload(source, 0, &header, 4096).ok()?;
    Some(FileHeader::from_bytes(&header, &payload))
}

/// Find `NAME.M00`.. siblings of `path`, case-insensitively, stopping at the
/// first gap. Returns them in numeric order.
fn find_chunks(path: &Path) -> Result<Vec<PathBuf>> {
    let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else {
        return Ok(Vec::new());
    };
    let dir = match path.parent() {
        Some(p) if !p.as_os_str().is_empty() => p.to_path_buf(),
        _ => PathBuf::from("."),
    };
    // One directory listing, lower-cased names -> real paths.
    let mut names = std::collections::HashMap::new();
    for entry in std::fs::read_dir(&dir)? {
        let entry = entry?;
        if let Some(n) = entry.file_name().to_str() {
            names.insert(n.to_ascii_lowercase(), entry.path());
        }
    }
    let stem_lc = stem.to_ascii_lowercase();
    let mut out = Vec::new();
    for n in 0..100 {
        match names.get(&format!("{stem_lc}.m{n:02}")) {
            Some(p) => out.push(p.clone()),
            None => break,
        }
    }
    Ok(out)
}

// `Clip` holds only its sources plus plain data, so it is `Send + Sync`
// whenever the source is. Spelled out so a regression is a compile error.
#[allow(dead_code)]
fn assert_sync<R: ReadAt + Send + Sync>(c: &Clip<R>) -> &(dyn Sync + '_) {
    c
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn latest_at_picks_floor_or_first() {
        let tl = vec![(10u64, 'a'), (20, 'b'), (30, 'c')];
        assert_eq!(latest_at(&tl, 5), Some(&'a'));
        assert_eq!(latest_at(&tl, 10), Some(&'a'));
        assert_eq!(latest_at(&tl, 25), Some(&'b'));
        assert_eq!(latest_at(&tl, 99), Some(&'c'));
        assert_eq!(latest_at::<char>(&[], 1), None);
    }

    #[test]
    fn find_chunks_is_case_insensitive_and_stops_at_gap() {
        let dir = std::env::temp_dir().join(format!("mlvio-chunks-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        for n in ["Clip.MLV", "Clip.m00", "Clip.M01", "Clip.M03"] {
            std::fs::write(dir.join(n), b"x").unwrap();
        }
        let found = find_chunks(&dir.join("Clip.MLV")).unwrap();
        let names: Vec<_> = found
            .iter()
            .map(|p| p.file_name().unwrap().to_str().unwrap().to_string())
            .collect();
        assert_eq!(names, ["Clip.m00", "Clip.M01"]);
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
