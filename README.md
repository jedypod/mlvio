# mlvio

Reader for Magic Lantern Video (MLV) raw clips, written in Rust.

- Opens a clip and its split chunks (`NAME.MLV` + `NAME.M00` .. `NAME.M99`),
  checks they belong together, and builds a timestamp-sorted frame index.
- Decodes uncompressed 10/12/14-bit packed frames and LJ92 (lossless JPEG)
  frames to a `u16` Bayer mosaic at the stored bit depth.
- Exposes clip metadata (resolution, levels, CFA, colour matrix, camera,
  lens, exposure, white balance, wall clock) and per-frame metadata with
  mid-clip `EXPO`/`WBAL`/`LENS` changes applied by timestamp.
- Positional reads only: one open `Clip` is `Sync` and can be decoded from
  many threads without locking.
- No required dependencies. `block` and `decode` are `no_std + alloc`.
- Never panics on malformed input; every failure is a typed `Error`.

Not in scope for 0.1: audio, LZMA/delta, JPEG 2000, McRAW, dual-ISO
processing, `CURV` application, `XREF` index caches, writing MLV. Unsupported
encodings are detected and reported as `Error::Unsupported`.

## Usage

```rust
use mlvio::Clip;

let clip = Clip::open("CLIP.MLV")?;
let info = clip.info();
println!("{}x{} {}-bit {:?} {} frames, black {} white {}",
    info.width, info.height, info.bits_per_pixel, info.compression,
    clip.frame_count(), info.black_level, info.white_level);

let frame = clip.frame(0)?;             // owned Vec<u16>, width*height
let mut buf = vec![0u16; (info.width * info.height) as usize];
clip.read_frame(1, &mut buf)?;          // decode into your own buffer
let meta = clip.frame_meta(1)?;         // ISO, shutter, lens, WB at that frame
# Ok::<(), mlvio::Error>(())
```

Parallel decode is the caller's job and needs nothing special:

```rust,ignore
use rayon::prelude::*;
let frames: Vec<_> = (0..clip.frame_count()).into_par_iter().map(|i| clip.frame(i)).collect();
```

For a pipeline that separates I/O from decoding, `Clip::raw_payload` fetches
a frame's bytes and `Clip::decode_payload` turns bytes into samples;
`read_frame_with` combines them with a reusable scratch buffer. When the
source is already in memory (a slice, `Vec<u8>`, or a memory map opened with
`Clip::open_with` and the `mmap` feature) frames are decoded in place
without copying the payload.

Single-thread decode speed on an EPYC 7302P core: about 150 Msamples/s for
LJ92 (a 1600x800 14-bit frame in 8.5 ms) and about 1 Gpx/s for packed
14-bit; `mlvinfo --check --threads N` reports the parallel figure.

## Features

| Feature | Default | Effect |
|---|---|---|
| `std`   | yes | `scan` and `clip` layers (file I/O). Without it only `block` and `decode` build. |
| `mmap`  | no  | `ReadAt` for `memmap2::Mmap`. |
| `serde` | no  | `Serialize`/`Deserialize` on metadata types. |
| `cli`   | no  | Builds the `mlvinfo` binary (pulls in `clap`). |

## mlvinfo

```text
cargo install --path . --features cli
mlvinfo CLIP.MLV                      # chunks, clip info, block histogram
mlvinfo CLIP.MLV --blocks --meta      # every block header and parsed metadata
mlvinfo CLIP.MLV --frame 12 --out f12.pgm    # decode one frame to 16-bit PGM
mlvinfo CLIP.MLV --check              # decode every frame, report speed and errors
mlvinfo CLIP.MLV --check --threads 8  # the same on eight threads
```

## Layout

```text
src/block    tags, header, one struct per block, tolerant from_bytes   (no I/O)
src/scan     ReadAt trait, BlockIter over block headers               (I/O)
src/clip     chunk discovery, frame index, ClipInfo, frame decoding   (I/O)
src/decode   unpack10/12/14 and the LJ92 decoder                      (no I/O)
src/bin      mlvinfo
tests        synthetic clips (tests/common writer) and real fixtures
fuzz         cargo-fuzz targets for the parsers and the LJ92 decoder
```

See `DESIGN.md` for the format facts and the reasoning behind the layout.

## Testing

`cargo test` runs unit tests, synthetic-clip integration tests and, when
present, tests against real clips in `tests/fixtures/` (git-ignored). The
fixture tests expect:

| File | Content |
|---|---|
| `tests/fixtures/lj92_14bit.MLV` | 1600x800, 14-bit LJ92, 814 frames, Canon 60D |
| `tests/fixtures/uncompressed_14bit_dualiso.MLV` | 1600x900, 14-bit uncompressed, 237 frames |

`scripts/fetch-fixtures.sh` links or copies them from a local directory
(`MLV_FIXTURE_DIR`) or downloads from `MLV_FIXTURE_URL`. Any other clips you
drop in that directory can be checked with `mlvinfo FILE --check`.

Fuzzing (nightly): `cargo install cargo-fuzz && cargo fuzz run lj92`.

## License

MIT OR Apache-2.0. The LJ92 decoder derives from liblj92 (Andrew Baldwin,
MIT) via Ilia Sibiryakov's Rust translation; see `NOTICE`. Magic Lantern and
MLV-App (GPL) were used as format references only; no code from them is
included.
