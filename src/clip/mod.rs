// SPDX-License-Identifier: MIT OR Apache-2.0
//! Clip-level API: chunk discovery, frame index, metadata timelines and
//! frame decoding.
//!
//! A clip is `NAME.MLV` plus optional `NAME.M00` .. `NAME.M99` chunks that
//! share a `fileGuid`. [`Clip::open`] resolves the siblings, scans every
//! block header once, and builds a timestamp-sorted frame index. Frames are
//! decoded on demand with [`Clip::read_frame`] or [`Clip::frame`]; the
//! `Clip` takes `&self` throughout and is `Sync` whenever its source is, so
//! callers can decode in parallel from several threads over one open clip.
//!
//! For a pipeline that separates I/O from decoding, [`Clip::raw_payload`]
//! fetches a frame's bytes and [`Clip::decode_payload`] turns bytes into
//! samples; [`Clip::read_frame_with`] combines the two and skips the copy
//! when the source is already in memory (a slice, `Vec<u8>` or memory map).

mod info;

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::fs::File;
use std::io;
use std::path::{Path, PathBuf};

use crate::block::{
    Block, BlockHeader, DarkFrame, Exposure, FileHeader, Lens, Tag, VideoFrameHeader, WhiteBalance,
    parse_block,
};
use crate::decode::{lj92, unpack};
use crate::scan::{BlockIter, MAX_META_PAYLOAD, ReadAt, read_payload, read_payload_into};
use crate::{Error, Result};

pub use info::{CameraId, ClipInfo, Compression, Rect};

/// Location of one video frame's pixel payload.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
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
    /// Sensor crop position in 8x2 blocks (`cropPosX`, `cropPosY`).
    pub crop_pos: (u16, u16),
    /// Pan position in pixels (`panPosX`, `panPosY`).
    pub pan_pos: (u16, u16),
}

/// Location of a non-frame block, for callers that want to re-read it with
/// [`Clip::read_meta`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct MetaRef {
    /// Index into the clip's chunk list.
    pub chunk: u16,
    /// Offset of the block within the chunk.
    pub offset: u64,
    /// The block header.
    pub header: BlockHeader,
}

/// What a [`Warning`] is about.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum WarningKind {
    /// The chunk ends inside a block; the rest of that chunk was ignored.
    TruncatedTail,
    /// A block header was inconsistent; the scan of that chunk stopped.
    BadBlock(String),
    /// A `VIDF` block's `frameSpace` exceeds the block; the frame was skipped.
    FrameSpaceOverflow,
    /// This many frames repeated an earlier frame number and were dropped.
    DuplicateFrames(usize),
}

/// A non-fatal problem found while indexing.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Warning {
    /// What happened.
    pub kind: WarningKind,
    /// Chunk index, when the problem is tied to one place in a file.
    pub chunk: Option<u16>,
    /// Byte offset within the chunk, when the problem is tied to one place.
    pub offset: Option<u64>,
}

impl fmt::Display for Warning {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let (Some(c), Some(o)) = (self.chunk, self.offset) {
            write!(f, "chunk {c} offset {o}: ")?;
        }
        match &self.kind {
            WarningKind::TruncatedTail => {
                write!(f, "file ends inside a block; remaining data ignored")
            }
            WarningKind::BadBlock(reason) => {
                write!(f, "bad block ({reason}); scan of this chunk stopped")
            }
            WarningKind::FrameSpaceOverflow => {
                write!(f, "VIDF frameSpace larger than block; frame skipped")
            }
            WarningKind::DuplicateFrames(n) => write!(f, "{n} duplicate frame number(s) dropped"),
        }
    }
}

/// Per-frame metadata: the frame's location and header fields plus the
/// exposure, white balance and lens settings in effect at its timestamp.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FrameMeta {
    /// Index in the sorted frame list.
    pub index: usize,
    /// Location and `VIDF` header fields.
    pub frame: FrameRef,
    /// Latest `EXPO` at or before the frame, else the first in the clip.
    pub exposure: Option<Exposure>,
    /// Latest `WBAL` at or before the frame, else the first in the clip.
    pub white_balance: Option<WhiteBalance>,
    /// Latest `LENS` at or before the frame, else the first in the clip.
    pub lens: Option<Lens>,
}

/// An owned, decoded frame.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
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

/// What one pass over the chunks collects, before `ClipInfo` is built.
#[derive(Default)]
struct Scan {
    frames: Vec<FrameRef>,
    meta: Vec<MetaRef>,
    expo: Vec<(u64, Exposure)>,
    wbal: Vec<(u64, WhiteBalance)>,
    lens: Vec<(u64, Lens)>,
    warnings: Vec<Warning>,
}

impl Scan {
    fn warn(&mut self, chunk: u16, offset: u64, kind: WarningKind) {
        self.warnings.push(Warning {
            kind,
            chunk: Some(chunk),
            offset: Some(offset),
        });
    }
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
    /// insensitively, stopping at the first missing number (see
    /// [`sibling_chunks`]). Each must carry the same `fileGuid` as the main
    /// file, otherwise [`Error::ChunkGuidMismatch`] is returned; use
    /// [`Clip::open_single`] to ignore siblings.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        Self::open_with(path, |p| File::open(p))
    }

    /// Open exactly one file, ignoring any sibling chunks.
    pub fn open_single(path: impl AsRef<Path>) -> Result<Self> {
        Self::open_paths(vec![path.as_ref().to_path_buf()])
    }

    /// Open an explicit list of chunk files. The first is treated as the main
    /// file for GUID checking; order is taken from each chunk's `fileNum`.
    pub fn open_paths(paths: Vec<PathBuf>) -> Result<Self> {
        Self::open_paths_with(paths, |p| File::open(p))
    }
}

impl<R: ReadAt> Clip<R> {
    /// Like [`Clip::open`], but every chunk file is turned into a source by
    /// `open`. This is how to use memory maps or a custom reader while
    /// keeping chunk discovery and per-chunk paths in errors:
    ///
    /// ```no_run
    /// # #[cfg(feature = "mmap")] {
    /// let clip = mlvio::Clip::open_with("CLIP.MLV", |p| {
    ///     let f = std::fs::File::open(p)?;
    ///     // SAFETY: the caller promises the file is not modified while mapped.
    ///     unsafe { memmap2::Mmap::map(&f) }
    /// })?;
    /// # }
    /// # Ok::<(), mlvio::Error>(())
    /// ```
    pub fn open_with<F>(path: impl AsRef<Path>, open: F) -> Result<Self>
    where
        F: FnMut(&Path) -> io::Result<R>,
    {
        let path = path.as_ref();
        let mut paths = vec![path.to_path_buf()];
        paths.extend(sibling_chunks(path)?);
        Self::open_paths_with(paths, open)
    }

    /// Like [`Clip::open_paths`] with a custom source constructor.
    pub fn open_paths_with<F>(paths: Vec<PathBuf>, mut open: F) -> Result<Self>
    where
        F: FnMut(&Path) -> io::Result<R>,
    {
        let mut sources = Vec::with_capacity(paths.len());
        for p in &paths {
            let source = open(p).map_err(|source| Error::Open {
                path: p.clone(),
                source,
            })?;
            sources.push(source);
        }
        Self::build(sources, paths.into_iter().map(Some).collect())
    }

    /// Build a clip from in-memory or custom sources, one per chunk, in any
    /// order. Errors carry no paths.
    pub fn from_sources(sources: Vec<R>) -> Result<Self> {
        let n = sources.len();
        Self::build(sources, vec![None; n])
    }

    fn build(sources: Vec<R>, paths: Vec<Option<PathBuf>>) -> Result<Self> {
        if sources.is_empty() {
            return Err(Error::NotMlv { path: None });
        }
        if u16::try_from(sources.len()).is_err() {
            return Err(Error::Io(io::Error::new(
                io::ErrorKind::InvalidInput,
                "more than 65535 chunk sources",
            )));
        }
        // Read every MLVI first so GUID/order problems surface before the scan.
        let mut chunks = Vec::with_capacity(sources.len());
        for (source, path) in sources.into_iter().zip(paths) {
            let header =
                read_file_header(&source)?.ok_or_else(|| Error::NotMlv { path: path.clone() })?;
            chunks.push(Chunk {
                source,
                path,
                header,
            });
        }
        let guid = chunks[0].header.file_guid;
        if let Some(bad) = chunks.iter().find(|c| c.header.file_guid != guid) {
            return Err(Error::ChunkGuidMismatch {
                path: bad.path.clone(),
                expected: guid,
                got: bad.header.file_guid,
            });
        }
        chunks.sort_by_key(|c| c.header.file_num);

        let (scan, info) = index(&chunks)?;
        Ok(Clip {
            chunks,
            frames: scan.frames,
            meta: scan.meta,
            info,
            expo: scan.expo,
            wbal: scan.wbal,
            lens: scan.lens,
            warnings: scan.warnings,
        })
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
        let f = *self.frame_ref(i)?;
        Ok(FrameMeta {
            index: i,
            exposure: latest_at(&self.expo, f.timestamp).cloned(),
            white_balance: latest_at(&self.wbal, f.timestamp).cloned(),
            lens: latest_at(&self.lens, f.timestamp).cloned(),
            frame: f,
        })
    }

    /// Re-read and parse the block a [`MetaRef`] points to (payload capped at
    /// [`MAX_META_PAYLOAD`]).
    pub fn read_meta(&self, m: &MetaRef) -> Result<Block> {
        let source = &self
            .chunks
            .get(usize::from(m.chunk))
            .ok_or(Error::FrameOutOfRange(usize::from(m.chunk)))?
            .source;
        let payload = read_payload(source, m.offset, &m.header, MAX_META_PAYLOAD)?;
        Ok(parse_block(&m.header, &payload))
    }

    /// Read the undecoded pixel payload of frame `i` into `buf`, replacing
    /// its contents. The buffer is resized, not zero-filled.
    pub fn raw_payload(&self, i: usize, buf: &mut Vec<u8>) -> Result<()> {
        let f = self.frame_ref(i)?;
        let source = &self.chunks[usize::from(f.chunk)].source;
        buf.resize(f.data_len as usize, 0);
        source.read_exact_at(f.data_offset, buf)?;
        Ok(())
    }

    /// Decode a frame payload (as returned by [`Clip::raw_payload`]) into
    /// `dst`, which must hold `width * height` samples.
    ///
    /// Uncompressed payloads are unpacked at the clip's bit depth; LJ92
    /// payloads are parsed, checked against the `RAWI` geometry and decoded.
    /// Unsupported encodings fail with [`Error::Unsupported`].
    pub fn decode_payload(&self, bytes: &[u8], dst: &mut [u16]) -> Result<()> {
        let info = &self.info;
        let pixels = info.width as usize * info.height as usize;
        if dst.len() != pixels {
            return Err(Error::DestinationSize {
                expected: pixels,
                got: dst.len(),
            });
        }
        match info.compression {
            Compression::None => unpack::unpack(info.bits_per_pixel, bytes, dst),
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
            Compression::Unsupported(u) => Err(Error::Unsupported(u)),
        }
    }

    /// Decode frame `i` into `dst`, which must hold `width * height` samples.
    pub fn read_frame(&self, i: usize, dst: &mut [u16]) -> Result<()> {
        let mut scratch = FrameScratch::new();
        self.read_frame_with(i, dst, &mut scratch)
    }

    /// Like [`read_frame`](Self::read_frame) but reusing `scratch` for the
    /// compressed/packed bytes. When the chunk source is already in memory
    /// ([`ReadAt::as_slice`]) the payload is decoded in place and `scratch`
    /// is not touched.
    pub fn read_frame_with(
        &self,
        i: usize,
        dst: &mut [u16],
        scratch: &mut FrameScratch,
    ) -> Result<()> {
        if let Compression::Unsupported(u) = self.info.compression {
            return Err(Error::Unsupported(u));
        }
        let f = self.frame_ref(i)?;
        let source = &self.chunks[usize::from(f.chunk)].source;
        if let Some(all) = source.as_slice() {
            let start = usize::try_from(f.data_offset).ok();
            let bytes = start
                .and_then(|s| all.get(s..s.checked_add(f.data_len as usize)?))
                .ok_or(Error::Truncated {
                    offset: f.block_offset,
                })?;
            return self.decode_payload(bytes, dst);
        }
        self.raw_payload(i, &mut scratch.bytes)?;
        self.decode_payload(&scratch.bytes, dst)
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

/// Scan every chunk once and build the frame index, metadata timelines,
/// block list and `ClipInfo`.
fn index<R: ReadAt>(chunks: &[Chunk<R>]) -> Result<(Scan, ClipInfo)> {
    let mut builder = info::InfoBuilder::new(chunks[0].header.clone());
    let mut scan = Scan::default();
    for (ci, chunk) in chunks.iter().enumerate() {
        // `build` has already checked that the count fits.
        let chunk_idx = ci as u16;
        index_chunk(chunk_idx, &chunk.source, &mut scan, &mut builder)?;
    }

    // Timestamp order, tie-break on frame number, drop duplicate numbers.
    scan.frames.sort_by_key(|f| (f.timestamp, f.frame_number));
    let before = scan.frames.len();
    let mut seen = HashSet::with_capacity(before);
    scan.frames.retain(|f| seen.insert(f.frame_number));
    if scan.frames.len() != before {
        scan.warnings.push(Warning {
            kind: WarningKind::DuplicateFrames(before - scan.frames.len()),
            chunk: None,
            offset: None,
        });
    }
    scan.expo.sort_by_key(|e| e.0);
    scan.wbal.sort_by_key(|e| e.0);
    scan.lens.sort_by_key(|e| e.0);

    let info = builder.finish(
        chunks.len(),
        scan.frames.len(),
        &scan.expo,
        &scan.wbal,
        &scan.lens,
    )?;
    Ok((scan, info))
}

/// Walk the block headers of one chunk. Frames become `FrameRef`s, every
/// other block becomes a `MetaRef`, and the few metadata tags the clip layer
/// consumes are parsed on the spot. A truncated or corrupt tail ends the
/// chunk with a warning.
fn index_chunk<R: ReadAt>(
    chunk_idx: u16,
    source: &R,
    scan: &mut Scan,
    builder: &mut info::InfoBuilder,
) -> Result<()> {
    for item in BlockIter::new(source) {
        let (offset, header) = match item {
            Ok(x) => x,
            Err(Error::Truncated { offset }) => {
                scan.warn(chunk_idx, offset, WarningKind::TruncatedTail);
                break;
            }
            Err(Error::BadBlock { offset, reason }) => {
                scan.warn(chunk_idx, offset, WarningKind::BadBlock(reason.to_string()));
                break;
            }
            Err(e) => return Err(e),
        };
        if header.tag == Tag::VIDF {
            // Only the fixed 16-byte frame header is needed; no allocation.
            let mut buf = [0u8; VideoFrameHeader::SIZE];
            read_payload_into(source, offset, &header, &mut buf)?;
            let vh = VideoFrameHeader::from_bytes(&buf);
            let Some(data_len) = vh.data_len(header.size) else {
                scan.warn(chunk_idx, offset, WarningKind::FrameSpaceOverflow);
                continue;
            };
            scan.frames.push(FrameRef {
                chunk: chunk_idx,
                block_offset: offset,
                data_offset: offset + vh.data_offset(),
                data_len,
                timestamp: header.timestamp,
                frame_number: vh.frame_number,
                crop_pos: (vh.crop_pos_x, vh.crop_pos_y),
                pan_pos: (vh.pan_pos_x, vh.pan_pos_y),
            });
            continue;
        }
        scan.meta.push(MetaRef {
            chunk: chunk_idx,
            offset,
            header,
        });
        if header.tag == Tag::AUDF {
            builder.audio_frames += 1;
        } else if info::InfoBuilder::wants(header.tag) {
            // DARK carries a whole frame; only its fixed header is parsed.
            let cap = if header.tag == Tag::DARK {
                DarkFrame::FIXED_SIZE
            } else {
                MAX_META_PAYLOAD
            };
            let payload = read_payload(source, offset, &header, cap)?;
            match parse_block(&header, &payload) {
                Block::Exposure(e) => scan.expo.push((header.timestamp, e)),
                Block::WhiteBalance(w) => scan.wbal.push((header.timestamp, w)),
                Block::Lens(l) => scan.lens.push((header.timestamp, l)),
                other => builder.absorb(header.timestamp, other, chunk_idx, offset),
            }
        }
    }
    Ok(())
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

/// Read and parse the `MLVI` block at the start of `source`. `Ok(None)` means
/// the source is readable but does not start with `MLVI`.
fn read_file_header<R: ReadAt + ?Sized>(source: &R) -> Result<Option<FileHeader>> {
    let mut hdr = [0u8; BlockHeader::SIZE];
    match source.read_exact_at(0, &mut hdr) {
        Ok(()) => {}
        Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => return Ok(None),
        Err(e) => return Err(Error::Io(e)),
    }
    let Ok(header) = BlockHeader::from_bytes(&hdr) else {
        return Ok(None);
    };
    if header.tag != Tag::MLVI {
        return Ok(None);
    }
    let payload = read_payload(source, 0, &header, 4096)?;
    Ok(Some(FileHeader::from_bytes(&header, &payload)))
}

/// Find the `NAME.M00` .. `NAME.M99` siblings of `path`, matched case
/// insensitively with one directory listing, stopping at the first missing
/// number. Returns them in numeric order; the main file is not included.
pub fn sibling_chunks(path: &Path) -> io::Result<Vec<PathBuf>> {
    let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else {
        return Ok(Vec::new());
    };
    let dir = match path.parent() {
        Some(p) if !p.as_os_str().is_empty() => p.to_path_buf(),
        _ => PathBuf::from("."),
    };
    let mut names = HashMap::new();
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
const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<Clip<File>>();
    assert_send_sync::<Clip<Vec<u8>>>();
    assert_send_sync::<Clip<&[u8]>>();
};

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
    fn sibling_chunks_is_case_insensitive_and_stops_at_gap() {
        let dir = std::env::temp_dir().join(format!("mlvio-chunks-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        for n in ["Clip.MLV", "Clip.m00", "Clip.M01", "Clip.M03"] {
            std::fs::write(dir.join(n), b"x").unwrap();
        }
        let found = sibling_chunks(&dir.join("Clip.MLV")).unwrap();
        let names: Vec<_> = found
            .iter()
            .map(|p| p.file_name().unwrap().to_str().unwrap().to_string())
            .collect();
        assert_eq!(names, ["Clip.m00", "Clip.M01"]);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn open_reports_io_errors_with_path() {
        let missing = std::env::temp_dir().join("mlvio-definitely-missing.MLV");
        match Clip::open_single(&missing) {
            Err(Error::Open { path, source }) => {
                assert_eq!(path, missing);
                assert_eq!(source.kind(), io::ErrorKind::NotFound);
            }
            other => panic!("unexpected {:?}", other.map(|_| ())),
        }
    }
}
