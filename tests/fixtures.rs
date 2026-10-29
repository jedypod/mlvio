//! Tests against real camera clips under `tests/fixtures/` (git-ignored).
//! Each test is skipped when its file is absent so CI without fixtures passes.

use std::path::PathBuf;

use mlvio::{Cfa, Clip, Compression};

fn fixture(name: &str) -> Option<PathBuf> {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name);
    if p.exists() {
        Some(p)
    } else {
        eprintln!("skipping: fixture {name} not present");
        None
    }
}

/// Mean and fraction of samples inside [black, white], for a sanity check
/// that decoded data looks like a photograph rather than noise.
fn stats(data: &[u16], black: u32, white: u32) -> (f64, f64) {
    let n = data.len() as f64;
    let mean = data.iter().map(|&v| v as f64).sum::<f64>() / n;
    let inside = data
        .iter()
        .filter(|&&v| (v as u32) >= black.saturating_sub(64) && (v as u32) <= white)
        .count();
    (mean, inside as f64 / n)
}

/// Neighbouring same-colour samples in a real image are highly correlated;
/// a mis-decoded frame is not. Returns mean |x[i] - x[i+2]| relative to the
/// signal range.
fn roughness(data: &[u16], width: usize, white: u32) -> f64 {
    let mut acc = 0u64;
    let mut n = 0u64;
    for row in data.chunks_exact(width) {
        for w in row.windows(3) {
            acc += (w[0] as i64 - w[2] as i64).unsigned_abs();
            n += 1;
        }
    }
    acc as f64 / n as f64 / white as f64
}

#[test]
fn lj92_14bit_clip_decodes() {
    let Some(p) = fixture("lj92_14bit.MLV") else {
        return;
    };
    let clip = Clip::open(&p).unwrap();
    let info = clip.info();
    assert_eq!(info.compression, Compression::Lj92);
    assert_eq!((info.width, info.height), (1600, 800));
    assert_eq!(info.bits_per_pixel, 14);
    assert_eq!((info.black_level, info.white_level), (2048, 15000));
    assert_eq!(info.cfa, Cfa::Rggb);
    assert_eq!(clip.frame_count(), 814);
    assert_eq!(info.fps_rational, (11988, 1000));
    assert!(!info.camera.name.is_empty(), "IDNT missing");
    assert!(info.recorded_at.is_some());
    assert!(info.color_matrix.is_some());
    assert!(clip.warnings().is_empty(), "{:?}", clip.warnings());

    let mut scratch = mlvio::FrameScratch::new();
    let mut buf = vec![0u16; 1600 * 800];
    for i in [0usize, 1, 100, 813] {
        clip.read_frame_with(i, &mut buf, &mut scratch)
            .unwrap_or_else(|e| panic!("frame {i}: {e}"));
        let (mean, inside) = stats(&buf, info.black_level, info.white_level);
        assert!(
            mean > info.black_level as f64 && mean < info.white_level as f64,
            "frame {i} mean {mean}"
        );
        assert!(inside > 0.99, "frame {i}: {inside} inside range");
        assert!(
            roughness(&buf, 1600, info.white_level) < 0.05,
            "frame {i} looks like noise"
        );
    }
    let f0 = clip.frame(0).unwrap();
    assert_eq!(f0.meta.frame.frame_number, 0);
    assert!(f0.meta.exposure.is_some());
    assert!(f0.meta.lens.is_some());
}

#[test]
fn uncompressed_14bit_dual_iso_clip_decodes() {
    let Some(p) = fixture("uncompressed_14bit_dualiso.MLV") else {
        return;
    };
    let clip = Clip::open(&p).unwrap();
    let info = clip.info();
    assert_eq!(info.compression, Compression::None);
    assert_eq!((info.width, info.height), (1600, 900));
    assert_eq!(info.bits_per_pixel, 14);
    assert_eq!(clip.frame_count(), 237);
    assert_eq!(info.fps_rational, (23976, 1000));
    assert!(
        clip.exposures().len() >= 2,
        "EXPO repeats mid-clip in this fixture"
    );

    let f = clip.frame(0).unwrap();
    let (mean, inside) = stats(&f.data, info.black_level, info.white_level);
    assert!(
        mean > info.black_level as f64 && mean < info.white_level as f64,
        "mean {mean}"
    );
    assert!(inside > 0.99);
    // Dual-ISO alternates exposure every two rows, so compare rows 4 apart.
    let w = 1600;
    let mut acc = 0u64;
    for y in 0..(900 - 4) {
        for x in 0..w {
            acc += (f.data[y * w + x] as i64 - f.data[(y + 4) * w + x] as i64).unsigned_abs();
        }
    }
    let r = acc as f64 / ((900 - 4) * w) as f64 / info.white_level as f64;
    assert!(r < 0.05, "frame looks like noise: {r}");

    // Raw payload length matches the packed size exactly for this camera.
    let mut raw = Vec::new();
    clip.raw_payload(0, &mut raw).unwrap();
    assert_eq!(raw.len(), 1600 * 900 * 14 / 8);
}
