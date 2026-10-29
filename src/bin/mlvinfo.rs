// SPDX-License-Identifier: MIT OR Apache-2.0
//! `mlvinfo`: inspect an MLV clip and optionally dump a frame as 16-bit PGM.

use std::collections::BTreeMap;
use std::io::Write;
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use clap::Parser;
use mlvio::block::{Block, parse_block};
use mlvio::scan::{BlockIter, read_payload};
use mlvio::{Clip, FrameScratch, Tag};

type CliResult = Result<(), Box<dyn std::error::Error>>;

#[derive(Parser)]
#[command(version, about = "Inspect Magic Lantern Video (MLV) clips")]
struct Args {
    /// The .MLV file (sibling .M00.. chunks are picked up automatically).
    file: PathBuf,
    /// Ignore sibling chunk files.
    #[arg(long)]
    single: bool,
    /// List every block header in file order.
    #[arg(long)]
    blocks: bool,
    /// Print the parsed contents of metadata blocks.
    #[arg(long)]
    meta: bool,
    /// Decode frame N (playback order) and report statistics.
    #[arg(long, value_name = "N")]
    frame: Option<usize>,
    /// Write the decoded frame as a 16-bit binary PGM (requires --frame).
    #[arg(long, value_name = "FILE")]
    out: Option<PathBuf>,
    /// Write the undecoded frame payload instead of a PGM (requires --frame).
    #[arg(long, value_name = "FILE")]
    raw_out: Option<PathBuf>,
    /// Decode every frame and report timing and errors.
    #[arg(long)]
    check: bool,
    /// Number of decoding threads used by --check.
    #[arg(long, value_name = "N", default_value = "1")]
    threads: NonZeroUsize,
}

fn main() -> ExitCode {
    let args = Args::parse();
    match run(&args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("mlvinfo: {e}");
            ExitCode::FAILURE
        }
    }
}

fn run(args: &Args) -> CliResult {
    let clip = if args.single {
        Clip::open_single(&args.file)?
    } else {
        Clip::open(&args.file)?
    };

    print_summary(&clip);
    if args.blocks || args.meta {
        print_blocks(&clip, args)?;
    }
    if let Some(n) = args.frame {
        dump_frame(&clip, n, args)?;
    } else if args.out.is_some() || args.raw_out.is_some() {
        return Err("--out/--raw-out require --frame".into());
    }
    if args.check {
        check_all(&clip, args.threads.get())?;
    }
    Ok(())
}

/// Chunk list, clip-level metadata, block histogram and scan warnings.
fn print_summary(clip: &Clip) {
    let info = clip.info();

    println!("Chunks: {}", clip.chunk_count());
    for i in 0..clip.chunk_count() {
        let h = clip.chunk_header(i).unwrap();
        println!(
            "  [{i}] {}  fileNum {} of {}  {}  videoClass {:#x} audioClass {}  frames {} flags {:#x}",
            clip.chunk_path(i)
                .map(|p| p.display().to_string())
                .unwrap_or_default(),
            h.file_num,
            h.file_count,
            h.version_str(),
            h.video_class,
            h.audio_class,
            h.video_frame_count,
            h.file_flags,
        );
    }

    println!("Clip:");
    println!(
        "  camera        {} (model {:#x}, serial {})",
        info.camera.name, info.camera.model_id, info.camera.serial
    );
    println!(
        "  resolution    {}x{}  {}-bit  {}",
        info.width, info.height, info.bits_per_pixel, info.cfa
    );
    println!(
        "  levels        black {}  white {}",
        info.black_level, info.white_level
    );
    println!("  compression   {:?}", info.compression);
    println!(
        "  fps           {:.3} ({}/{})",
        info.fps, info.fps_rational.0, info.fps_rational.1
    );
    println!(
        "  frames        {} video, {} audio",
        info.frame_count, info.audio_frame_count
    );
    println!("  active area   {:?}", info.active_area);
    println!("  crop          {:?}", info.crop);
    println!("  illuminant    {}", info.calibration_illuminant);
    if let Some(m) = info.color_matrix {
        println!("  color matrix  {:.4} {:.4} {:.4}", m[0], m[1], m[2]);
        println!("                {:.4} {:.4} {:.4}", m[3], m[4], m[5]);
        println!("                {:.4} {:.4} {:.4}", m[6], m[7], m[8]);
    }
    println!("  dual iso      {}", info.dual_iso);
    if let Some(t) = &info.recorded_at {
        println!("  recorded at   {t}");
    }
    if let Some(e) = &info.exposure {
        println!(
            "  exposure      ISO {} (analog {}, digital gain {})  shutter {:.2} ms",
            e.iso_value,
            e.iso_analog,
            e.digital_gain,
            e.shutter_us as f64 / 1000.0
        );
    }
    if let Some(l) = &info.lens {
        println!(
            "  lens          {}  {} mm  f/{:.1}  focus {} mm",
            l.name_str(),
            l.focal_length_mm,
            l.aperture(),
            l.focus_distance_mm
        );
    }
    if let Some(w) = &info.white_balance {
        println!(
            "  white balance mode {}  {} K  gains {}/{}/{}",
            w.mode, w.kelvin, w.gain_r, w.gain_g, w.gain_b
        );
    }
    if let Some(rc) = &info.raw_capture {
        println!(
            "  sensor        {}x{}  crop x{:.2}  binning {}x{}  skipping {}x{}",
            rc.sensor_res_x,
            rc.sensor_res_y,
            rc.sensor_crop_x100 as f64 / 100.0,
            rc.binning_x,
            rc.binning_y,
            rc.skipping_x,
            rc.skipping_y
        );
    }
    for v in &info.versions {
        println!("  version       {}", v.text);
    }
    if let Some(i) = info.info.as_ref().filter(|i| !i.text.is_empty()) {
        println!("  info          {}", i.text);
    }
    println!(
        "  out of order  {}   dropped frames {}",
        info.out_of_order, info.dropped_frames
    );
    if info.exposure.is_some() || clip.exposures().len() > 1 {
        println!(
            "  timelines     EXPO {}  WBAL {}  LENS {}",
            clip.exposures().len(),
            clip.white_balances().len(),
            clip.lenses().len()
        );
    }

    // Block histogram.
    let mut hist: BTreeMap<String, (usize, u64)> = BTreeMap::new();
    for m in clip.meta_blocks() {
        let e = hist.entry(m.header.tag.to_string()).or_default();
        e.0 += 1;
        e.1 += m.header.size as u64;
    }
    let vbytes: u64 = clip.frames().iter().map(|f| f.data_len as u64).sum();
    println!("Blocks:");
    for (tag, (n, bytes)) in &hist {
        println!("  {tag:<6} {n:>8}  {bytes:>14} bytes");
    }
    println!(
        "  {:<6} {:>8}  {:>14} bytes (payload)",
        "VIDF",
        clip.frame_count(),
        vbytes
    );

    for w in clip.warnings() {
        println!("Warning: {w}");
    }
}

/// `--blocks` / `--meta`: every block header per chunk, optionally with the
/// parsed contents of metadata blocks.
fn print_blocks(clip: &Clip, args: &Args) -> CliResult {
    for ci in 0..clip.chunk_count() {
        let src = clip.chunk_source(ci).unwrap();
        println!("Chunk {ci} blocks:");
        for item in BlockIter::new(src) {
            let (offset, header) = match item {
                Ok(x) => x,
                Err(e) => {
                    println!("  {offset:>12}  ERROR {e}", offset = "");
                    break;
                }
            };
            println!(
                "  {offset:>12}  {}  size {:>10}  ts {}",
                header.tag, header.size, header.timestamp
            );
            if args.meta && !header.tag.is_frame() && header.tag != Tag::MLVI {
                let payload = read_payload(src, offset, &header, 4096)?;
                match parse_block(&header, &payload) {
                    Block::Unknown { .. } | Block::Null => {}
                    b => println!("      {b:?}"),
                }
            }
        }
    }
    Ok(())
}

/// `--frame N` with optional `--raw-out` / `--out`.
fn dump_frame(clip: &Clip, n: usize, args: &Args) -> CliResult {
    let meta = clip.frame_meta(n)?;
    println!(
        "Frame {n}: number {}  ts {} us  chunk {}  crop {:?}  pan {:?}",
        meta.frame.frame_number,
        meta.frame.timestamp,
        meta.frame.chunk,
        meta.frame.crop_pos,
        meta.frame.pan_pos
    );
    if let Some(e) = &meta.exposure {
        println!(
            "  ISO {}  shutter {:.2} ms",
            e.iso_value,
            e.shutter_us as f64 / 1000.0
        );
    }
    if let Some(p) = &args.raw_out {
        let mut raw = Vec::new();
        clip.raw_payload(n, &mut raw)?;
        std::fs::write(p, &raw)?;
        println!("  wrote {} payload bytes to {}", raw.len(), p.display());
    }
    let t = Instant::now();
    let frame = clip.frame(n)?;
    let dt = t.elapsed();
    let (mut lo, mut hi, mut sum) = (u16::MAX, 0u16, 0u64);
    for &v in &frame.data {
        lo = lo.min(v);
        hi = hi.max(v);
        sum += v as u64;
    }
    println!(
        "  decoded in {:.2} ms  min {lo}  max {hi}  mean {:.1}",
        dt.as_secs_f64() * 1e3,
        sum as f64 / frame.data.len() as f64
    );
    if let Some(p) = &args.out {
        write_pgm(
            p,
            frame.width,
            frame.height,
            &frame.data,
            clip.info().bits_per_pixel,
        )?;
        println!("  wrote {}", p.display());
    }
    Ok(())
}

/// `--check`: decode every frame over `threads` worker threads and report
/// throughput and failures.
fn check_all(clip: &Clip, threads: usize) -> CliResult {
    let info = clip.info();
    let pixels = (info.width * info.height) as usize;
    let count = clip.frame_count();
    let threads = threads.min(count).max(1);

    let t = Instant::now();
    let failures = if threads == 1 {
        // Report failures as they happen, in frame order.
        let mut buf = vec![0u16; pixels];
        let mut scratch = FrameScratch::new();
        let mut failures = 0usize;
        for i in 0..count {
            if let Err(e) = clip.read_frame_with(i, &mut buf, &mut scratch) {
                failures += 1;
                println!("  frame {i}: {e}");
            }
        }
        failures
    } else {
        let next = AtomicUsize::new(0);
        let mut errors: Vec<(usize, String)> = std::thread::scope(|s| {
            let workers: Vec<_> = (0..threads)
                .map(|_| {
                    s.spawn(|| {
                        let mut buf = vec![0u16; pixels];
                        let mut scratch = FrameScratch::new();
                        let mut errors = Vec::new();
                        loop {
                            let i = next.fetch_add(1, Ordering::Relaxed);
                            if i >= count {
                                break errors;
                            }
                            if let Err(e) = clip.read_frame_with(i, &mut buf, &mut scratch) {
                                errors.push((i, e.to_string()));
                            }
                        }
                    })
                })
                .collect();
            workers
                .into_iter()
                .flat_map(|w| w.join().expect("decode thread panicked"))
                .collect()
        });
        errors.sort_unstable_by_key(|&(i, _)| i);
        for (i, e) in &errors {
            println!("  frame {i}: {e}");
        }
        errors.len()
    };
    let dt = t.elapsed().as_secs_f64();

    print!(
        "Checked {} frames in {:.2} s ({:.1} fps), {} failed",
        count,
        dt,
        count as f64 / dt,
        failures
    );
    if threads > 1 {
        print!(" ({threads} threads)");
    }
    println!();
    if failures > 0 {
        return Err(format!("{failures} frame(s) failed to decode").into());
    }
    Ok(())
}

/// Write a binary PGM. PGM caps maxval at 65535, so `bpp` is clamped to
/// 1..=16 (a corrupt header could otherwise overflow the shift).
fn write_pgm(path: &Path, w: usize, h: usize, data: &[u16], bpp: u32) -> std::io::Result<()> {
    let bpp = bpp.clamp(1, 16);
    let maxval = (1u32 << bpp) - 1;
    let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
    write!(f, "P5\n{w} {h}\n{maxval}\n")?;
    let mut bytes = Vec::with_capacity(data.len() * 2);
    for &v in data {
        bytes.extend_from_slice(&v.to_be_bytes());
    }
    f.write_all(&bytes)?;
    f.flush()
}
