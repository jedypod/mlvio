# MLV reader crate: design and architecture plan

Status: implemented 2026-10-28 (phases 1-4 done; see README.md). This document
records the design as proposed; where the code differs, the code is current.

Scope for v0.1 (agreed): read Magic Lantern Video (MLV) clips, including
split chunks, and decode raw Bayer frames that are either uncompressed
(10/12/14-bit packed) or LJ92 lossless-JPEG compressed. Expose clip and
per-frame metadata. Out of scope for now: audio, LZMA/delta, JPEG2000,
McRAW, dual-ISO and the rest of MLV-App's `llrawproc`, MAPP index caches,
writing MLV files.

The crate is a standalone library. vkraw is the first consumer but the
crate must not depend on vkraw types. The output contract is "u16 mosaic
plane plus enough metadata to linearise it".

## 1. Format facts the design depends on

Verified against `mlv.h`, `raw.h`, `mlv_lite.c`, `lossless.c` (Magic
Lantern) and `video_mlv.c` (MLV-App).

- Container: flat sequence of blocks. Every block starts with a 16-byte
  header: 4 ASCII bytes tag, u32 LE `blockSize` (whole block including the
  header), u64 LE `timestamp` (microseconds since recording start). The
  first block of every chunk is `MLVI` whose "timestamp" slot instead holds
  `versionString[8]`. All multi-byte fields are little-endian, packed, no
  padding.
- Chunks: a clip is `NAME.MLV` plus optional `NAME.M00` .. `NAME.M99`.
  Every chunk starts with its own `MLVI`. Chunks of one clip share
  `fileGuid`; `fileNum`/`fileCount` give order and total. Extension case
  varies in the wild (`.mlv`, `.m00`).
- Frames: `VIDF` block = 16-byte header + `frameNumber` u32 + `cropPosX/Y`
  u16 + `panPosX/Y` u16 + `frameSpace` u32 (32 bytes total), then
  `frameSpace` bytes of alignment padding, then pixel data. Pixel data
  offset = block offset + 32 + frameSpace; length = blockSize - 32 -
  frameSpace. Frames can be written out of order (`fileFlags & 1`) and
  frames can be dropped (`fileFlags & 2`), so frame numbers may have gaps.
- `RAWI`: `xRes`, `yRes` (u16) then the `raw_info` struct: api_version,
  a 32-bit dummy (pointer slot), height, width, pitch, frame_size,
  bits_per_pixel, black_level, white_level, crop origin/size (4 x i32),
  active_area y1,x1,y2,x2, exposure_bias[2], cfa_pattern, 
  calibration_illuminant1, color_matrix1[18] (9 rationals as
  numerator/denominator i32 pairs), dynamic_range. `xRes`/`yRes` are the
  frame dimensions actually stored; `raw_info.width/height` can differ.
- Uncompressed pixel layout: groups of 8 pixels in 14 bytes (14-bit),
  4 pixels in 6 bytes (12-bit), 8 pixels in 10 bytes (10-bit). Each group
  is read as little-endian u16 words and the pixel bits are taken
  big-endian within the 16-bit words (the `raw_pixblock_*` bitfields in
  `raw.h`). Uncompressed frame byte length = xRes * yRes * bpp / 8, but
  the stored payload may be longer (padding), never assume equality.
- LJ92 (videoClass & 0x20): payload is a complete lossless JPEG stream
  (SOI, DHT, SOF3, SOS, EOI). The camera's JPEG core encodes the frame as
  2 interleaved components with a slice geometry of width x height/2
  (`lossless.c`), so the SOF3 dimensions do not equal xRes x yRes. The
  decoder must trust the JPEG's own geometry, write samples in scan order
  to a flat buffer, and check `w * h * components == xRes * yRes`.
  Predictors 1..7 occur; predictor 6 is the common one from ML.
  Only one Huffman table per component in practice, but SOF3 permits up
  to 4 DHTs; support the general case.
- Lossless with reduced bit depth (`OUTPUT_12BIT_LOSSLESS` etc.): the
  container is always 14-bit; ML applies digital gain in camera and
  rescales black/white levels in RAWI. The reader needs no special
  handling, just trust `bits_per_pixel`, `black_level`, `white_level`.
- Other video class flags: LZMA 0x80, DELTA 0x40, JPEG2K 0x10, MCRAW
  0x100. Reject with a typed error, do not misdecode.
- Metadata blocks that matter to a raw consumer: `IDNT` (camera name,
  model id, serial), `EXPO` (iso, analog/digital gain, shutter in
  microseconds), `LENS`/`ELNS`, `WBAL`, `RTCI` (wall-clock), `RAWC`
  (sensor res, crop factor, binning/skipping, offsets), `DISO` (dual-ISO
  flag, important to detect even though we do not process it), `CURV`
  (decode LUT, rare), `DARK` (averaged dark frame, rare), `INFO`, `VERS`,
  `MARK`, `STYL`, `ELVL`, `XREF`, `NULL`/`BKUP` padding. `EXPO`, `WBAL`,
  `LENS` can repeat mid-clip when settings change; they apply from their
  timestamp onward.
- Block size tolerance: readers must copy `min(blockSize, sizeof(struct))`
  bytes because firmware revisions have appended fields. Parse what is
  present; default the rest.

## 2. Existing Rust code (surveyed 2026-10-28)

| Crate | Author | License | What it has | Verdict |
|---|---|---|---|---|
| `libmlv` 0.1.0 (lib name `mlv`) | ilia3101 (MLV-App author) | GPL-3.0-only OR GPL-2.0-only OR MIT OR Apache-2.0 | Block scanner, timestamp-sorted VIDF/AUDF index, 10/12/14 unpackers, a 344-line Rust translation of liblj92. No chunk support, no metadata beyond MLVI/RAWI/WAVI, `Option` errors, `println!` in library code, `unsafe set_len`, panics on truncated LJ92 data. README calls it a WIP being rewritten. | Not usable as a dependency. Its `lj92.rs` and `decode.rs` are MIT-licensable and are the best permissive starting point for our decoder. |
| `mlv` 0.1.1 | grego | MIT | Typed structs for every block via `packbytes` derive, header read, frame iteration, MLV writing. No chunks, no index, no pixel decoding, no LJ92. | Useful as a cross-check for field order. Owns the crate name `mlv` on crates.io. |
| `ljpeg` 0.1.2 | grego | LGPL-2.1 | Lossless JPEG decoder + encoder derived from dnglab/rawler. Full predictor set, multi-component, Sony/csfix quirks, `no_std`. Author measured ~2.5x faster than liblj92. | Best quality and speed, but copyleft. Acceptable only as an optional feature or a dev-dependency for benchmarks. |
| `rawler` 0.7/0.8 | dnglab | LGPL-2.1 | `decompressors::ljpeg::LjpegDecompressor`. | Same decoder as `ljpeg` with a very heavy dependency tree. vkraw already depends on it, so vkraw could reuse it, but the standalone crate should not. |
| `liblj92` (C) | Andrew Baldwin | MIT | 1219 lines, decoder with a predictor-6 fast path, encoder, no deps. | The reference. Port or FFI are both license-clean. |

No crate offers chunk handling or a usable clip-level API. Writing our own
crate is justified.

## 3. Licensing

Goal: the crate is `MIT OR Apache-2.0`, the same as vkraw, so it can be
used in any permissive or proprietary project without obligations beyond
attribution.

What may and may not flow into it:

- Magic Lantern headers (`mlv.h`, `raw.h`): GPL-2.0-or-later. Struct
  layouts, field names, constants and bit layouts are facts about the file
  format and are not copyrightable expression. Reimplementing the format
  from these is fine. Do not copy the comment prose verbatim; write our
  own doc comments. The MLV spec is also published on the ML wiki.
- MLV-App (`video_mlv.c`, `llrawproc/*`): GPL-3.0. Use as a behavioural
  reference only (read to understand edge cases, then write independently).
  No line-by-line porting. This matters most for `llrawproc` features such
  as dual-ISO, which descend from ML's GPL `cr2hdr`: if those are ever
  wanted they must be clean-room or live in a separately licensed crate.
- `liblj92` (C): MIT. Port freely; keep the copyright notice.
- `libmlv` `lj92.rs` / `decode.rs`: offered under MIT at the user's
  choice. Vendoring under MIT with attribution to Andrew Baldwin (original)
  and Ilia Sibiryakov (translation) is clean.
- `ljpeg` / `rawler`: LGPL-2.1. Depending on an LGPL crate does not change
  our crate's license, but every binary that statically links it (all Rust
  binaries) inherits the LGPL relink/source obligation for that component.
  vkraw already carries this via `rawler`, so nothing new for vkraw, but a
  general-purpose permissive crate should not require it. If used at all:
  optional feature, default off, or dev-dependency only.

Repo hygiene: `LICENSE-MIT`, `LICENSE-APACHE`, `license = "MIT OR
Apache-2.0"` in Cargo.toml, a `NOTICE` file listing third-party origins
(liblj92, libmlv translation), SPDX headers on vendored files.

Crate name: `mlv` and `libmlv` are taken on crates.io. The folder name
can stay `mlv`; the package name must differ if it is ever published.
Suggestion: `mlvio` (short, unclaimed as of today). Decide before the
first commit so the lib name does not churn.

## 4. Architecture

Layered, bottom up. Lower layers never depend on higher ones.

```
mlvio
├── block/      tags, header, typed block structs, parse-from-bytes      (no I/O)
├── scan/       block iterator over a ReadAt source                     (I/O, headers only)
├── clip/       chunk discovery, index, metadata timeline, Clip API     (I/O)
├── decode/
│   ├── unpack  10/12/14-bit packed -> u16                              (no I/O)
│   └── lj92    lossless JPEG decoder                                   (no I/O)
└── bin/mlvinfo (feature "cli")  dump blocks, index, extract frames
```

### 4.1 `block`: typed blocks, zero I/O

- `Tag([u8; 4])` with constants (`Tag::MLVI`, ...) and `Display`.
- `BlockHeader { tag, size: u32, timestamp: u64 }`.
- One plain struct per block (`FileHeader`, `RawInfo`, `RawCapture`,
  `VideoFrameHeader`, `Exposure`, `Lens`, `ExtendedLens`, `WhiteBalance`,
  `Identity`, `WallClock`, `DualIso`, `Curve`, `DarkFrame`, `Info`,
  `Version`, `Marker`, `PictureStyle`, `Level`, `Xref`, `WavInfo`,
  `AudioFrameHeader`). Fields use Rust names and native types; the C
  names go in doc comments so readers can map back to `mlv.h`.
- Parsing: hand-written `from_bytes(&[u8]) -> Result<Self, Error>` using a
  tiny cursor helper (`u8/u16/u32/u64/i32/bytes` little-endian reads).
  No `repr(C, packed)` + transmute, no `bytemuck`: the format has
  version-dependent lengths and we want tolerant parsing. Short blocks
  fill missing trailing fields with defaults and record `truncated: bool`
  where it matters. `RawInfo` skips the 4-byte pointer slot explicitly.
- `enum Block` wrapping the typed structs plus `Unknown { tag, payload
  range }` so the scanner never fails on tags we do not know.
- Derived conveniences: `RawInfo::cfa()` -> `Cfa::Rggb/Gbrg/...` from
  `cfa_pattern` (0x02010100 = RGGB), `RawInfo::color_matrix()` ->
  `[f64; 9]`, `FileHeader::fps()` -> `(num, den)` and `f64`,
  `Identity::camera_name()` as trimmed `&str`.
- `no_std` + `alloc` compatible by construction; `std` feature gates the
  I/O layers. Not a goal to polish for v0.1, just do not preclude it.

### 4.2 `scan`: block iteration

- `trait ReadAt { fn read_exact_at(&self, pos: u64, buf: &mut [u8]) ->
  io::Result<()>; fn len(&self) -> u64; }` implemented for `std::fs::File`
  (via `FileExt::read_exact_at` on Unix, `seek_read` on Windows), for
  `&[u8]`, and for `memmap2::Mmap` behind an optional feature. Positional
  reads need no `&mut` and no seek state, so one `File` can be shared by
  many decoding threads without a mutex (MLV-App uses a mutex per chunk;
  we avoid that).
- `BlockIter<'a, R: ReadAt>`: yields `Result<(u64 offset, BlockHeader)>`
  and advances by `blockSize`. Reads 16 bytes per block, never the
  payload. Validates `size >= 16`, `offset + size <= len`. Last block
  truncated (camera lost power) -> `Error::Truncated { offset }` which the
  clip layer downgrades to a warning and stops.
- `read_block(&R, offset, header) -> Block` reads and parses one payload
  on demand, capped at a sane maximum for non-frame blocks.

### 4.3 `clip`: chunks, index, metadata, public API

- `Clip::open(path)`: resolves sibling chunks (`.M00`..`.M99`, case
  insensitive, stop at the first missing number), opens each, reads each
  `MLVI`, checks `fileGuid` matches (mismatch -> `Error::ChunkGuidMismatch`
  listing the offender; caller may retry with `Clip::open_single`), orders
  by `fileNum`. `Clip::from_sources(Vec<impl ReadAt>)` for non-file use.
- Index build: single pass of `BlockIter` over every chunk. Records
  `FrameRef { chunk: u16, block_offset: u64, data_offset: u64, data_len:
  u32, timestamp: u64, frame_number: u32, crop: (u16,u16), pan:
  (u16,u16) }` for each `VIDF`, and stores every non-frame block as a
  `MetaRef { chunk, offset, header }` plus a parsed copy of the small
  metadata blocks. Sort frames by `timestamp`, tie-break by
  `frame_number`. Duplicate frame numbers (seen in some ML builds after
  a dropped frame) keep the first. `XREF` blocks are ignored in v0.1; a
  full header scan of a 20 GB clip is tens of thousands of 16-byte reads
  plus seeks, well under a second on local storage.
- Metadata model: `ClipInfo` holds the first occurrence of each
  one-per-clip block (`MLVI`, `RAWI`, `RAWC`, `IDNT`, `RTCI`, `DISO`,
  `VERS`, `INFO`, `WAVI`), plus `Vec<(timestamp, Exposure)>`,
  `Vec<(timestamp, WhiteBalance)>`, `Vec<(timestamp, Lens)>` timelines.
  `Clip::frame_meta(i)` returns the frame's own header fields plus the
  latest `EXPO`/`WBAL`/`LENS` at or before its timestamp (binary search).
- `ClipInfo` exposes the raw-consumer essentials directly: `width`,
  `height`, `bits_per_pixel`, `black_level`, `white_level`, `cfa`,
  `active_area: Rect`, `crop: Rect`, `color_matrix: Option<[f64; 9]>`,
  `calibration_illuminant`, `fps`, `frame_count`, `compression:
  Compression::{None, Lj92, Unsupported(flags)}`, `dual_iso: bool`,
  `camera: { name, model_id, serial }`, `recorded_at: Option<DateTime>`
  (plain struct, no chrono dependency).
- Frame access:
  - `Clip::frame_count()`.
  - `Clip::read_frame(i, &mut [u16]) -> Result<()>`: reads the payload
    into a scratch buffer and decodes into the caller's `width*height`
    slice. Reentrant: takes `&self`, allocates only the scratch (caller
    may pass a reusable `FrameScratch`).
  - `Clip::frame(i) -> Result<Frame>` convenience returning an owned
    `Frame { width, height, data: Vec<u16>, meta: FrameMeta }`.
  - `Clip::raw_payload(i, &mut Vec<u8>)` for callers that want the
    undecoded bytes (re-muxing, debugging).
  - Parallel decode is the caller's job (`(0..n).into_par_iter().map(|i|
    clip.frame(i))` works because `Clip: Sync`).
- Thread safety: `Clip` is `Send + Sync` given `R: ReadAt + Sync`.

### 4.4 `decode::unpack`

- `unpack14(src, dst)`, `unpack12`, `unpack10`: group-wise as in
  `raw.h`, written with explicit shifts (the `libmlv` versions are a
  correct MIT reference). Length checks up front: `src.len() >=
  dst.len() * bpp / 8` else `Error::FrameTooShort`. SIMD is not needed
  for v0.1; `chunks_exact` loops auto-vectorise reasonably.
- Sensor data is shifted to the stored bit depth, not to 16 bits. Scaling
  to a common range is the consumer's job using `black_level` and
  `white_level`.

### 4.5 `decode::lj92`

- Own module, MIT, attributed. Start from the `libmlv` translation of
  liblj92 and harden it:
  - bounds-checked byte cursor and bit reader, `Result` everywhere, no
    indexing panics on truncated or corrupt input;
  - parse all DHT tables (up to 4), per-component table selection from
    SOS, predictors 0..7, point transform (Pt) and `Al` ignored as in
    liblj92 but validated;
  - geometry taken from SOF3; `decode_into(&mut [u16])` requires
    `dst.len() == w * h * comps`;
  - restart markers: ML never emits them; detect and error rather than
    misdecode;
  - optional predictor-6 fast path ported from liblj92 once a benchmark
    shows it matters.
- API: `Lj92::parse(&[u8]) -> Result<Lj92Header>` then
  `decode_into(&[u8], &Lj92Header, &mut [u16])`, so the clip layer can
  validate geometry before allocating.
- Fuzz target from day one (`cargo fuzz`, decoder only, no I/O).
- Benchmark against the `ljpeg` crate as a dev-dependency only. If the
  gap is large, close it in our code; do not ship the LGPL dependency.

### 4.6 Errors

One `Error` enum (hand-written or `thiserror`, either is fine):
`Io(io::Error)`, `NotMlv { path }`, `BadBlock { offset, reason }`,
`Truncated { offset }`, `MissingBlock(Tag)`, `ChunkGuidMismatch { .. }`,
`FrameOutOfRange(usize)`, `FrameTooShort { expected, got }`,
`GeometryMismatch { expected, got }`, `Unsupported(Unsupported)` with
`Unsupported::{Lzma, Delta, Jpeg2000, McRaw, VideoClass(u16), BitDepth(u32),
AudioOnly}`, `Lj92(Lj92Error)`. Library code never panics on input data.

### 4.7 Dependencies

Required: none beyond `std`. Optional features: `mmap` (memmap2),
`serde` (derive on metadata types), `cli` (clap, for the `mlvinfo` binary).
Edition 2024. MSRV 1.89 to match vkraw.

### 4.8 `mlvinfo` binary

`mlvinfo FILE.MLV` prints chunks, `ClipInfo`, block histogram, metadata
timelines. `mlvinfo FILE.MLV --frame N --out frame.pgm` writes a 16-bit
PGM for eyeballing and diffing against `mlv_dump` output. Lives behind
the `cli` feature so library users pay nothing.

## 5. Testing strategy

- No MLV sample files were found on this machine during a partial search
  (the search was stopped before it finished). Needed early: at least one
  uncompressed 14-bit clip, one 12-bit, one 10-bit, one LJ92 clip, one
  multi-chunk clip, one clip with mid-clip `EXPO`/`WBAL` changes, one
  dual-ISO clip. Sources: ML forum sample threads, MLV-App's test
  material, or record on camera. Keep fixtures out of git (size); store
  under `tests/fixtures/` git-ignored with a download script, or truncate
  real clips to the first few frames with `mlv_dump -f 0-3`.
- Synthetic fixtures: a small test-only MLV writer (header + RAWI + N
  VIDF) covering uncompressed 10/12/14-bit, chunk splitting, out-of-order
  timestamps, truncated tail, padded frames. For LJ92 test vectors use
  liblj92's C encoder (compile in a `build.rs` of a test helper, MIT) or
  the `ljpeg` crate's encoder as a dev-dependency.
- Golden tests: decode frame N, compare against pixels from `mlv_dump
  --dng` (ML repo, GPL, used as a tool not as code) via a tiny DNG strip
  reader in tests, or against a stored hash.
- Fuzz: `block::*::from_bytes`, `BlockIter` over `&[u8]`, `lj92`.
- Property test: unpack(pack(x)) round trip for 10/12/14.

## 6. Phases

1. Skeleton: crate name decision, licenses, `NOTICE`, CI (fmt, clippy,
   test), fixture download script. Half a day.
2. `block` + `scan`: all structs, tolerant parsing, iterator, `mlvinfo`
   dumping blocks. 1 to 2 days.
3. `clip`: chunk discovery, index, `ClipInfo`, metadata timelines,
   uncompressed decode, PGM export, golden tests vs `mlv_dump`. 1 to 2 days.
4. `lj92`: vendor, harden, fuzz, bench, golden tests on real LJ92 clips.
   1 to 2 days.
5. Polish: docs with the `mlv.h` field mapping, examples, `serde`
   feature, publish 0.1. Half a day.

Then vkraw integration (separate plan): add `.mlv` to discovery, group
chunks so `.M00` files are not imported as clips, map `ClipInfo` to
`RawFrame`/`CameraMetadata`, model one clip as a frame sequence.

## 7. Open decisions

- Package name (`mlvio` suggested).
- Whether `Clip::open` should fail or warn on a `fileGuid` mismatch
  between chunks (proposal: fail, with an explicit `open_single` escape).
- Whether `CURV` LUTs should be applied by the crate or surfaced to the
  consumer (proposal: surface only; vkraw linearises).
- `no_std` for the `block`/`decode` layers: free if kept in mind from the
  start, so the proposal is to keep those modules `core`/`alloc` only.
