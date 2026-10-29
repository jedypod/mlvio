//! Clip-level behaviour on synthetic MLV data built with the test writer.

mod common;

use common::{MlvWriter, lcg_pixels};
use mlvio::{Clip, Compression, Error, Unsupported};

const W: u16 = 64;
const H: u16 = 24;

fn frame(seed: u32, bpp: u32) -> Vec<u16> {
    lcg_pixels(W as usize * H as usize, bpp, seed)
}

fn basic(bpp: u32) -> MlvWriter {
    let mut w = MlvWriter::new(0xDEAD_BEEF, 0, 1);
    w.rawi(W, H, bpp, 2048, 15000);
    w.idnt("Canon EOS 5D Mark III", 0x285, "123456789012");
    w.expo(0, 800, 20_000);
    w.wbal(0, 9, 5600);
    w
}

#[test]
fn uncompressed_10_12_14_roundtrip() {
    for bpp in [10u32, 12, 14] {
        let mut w = basic(bpp);
        let f0 = frame(1, bpp);
        let f1 = frame(2, bpp);
        w.vidf(1000, 0, 0, &f0, bpp);
        w.vidf_padded(2000, 1, 32, &f1, bpp, 100); // frameSpace and trailing padding
        let clip = Clip::from_sources(vec![w.finish()]).unwrap();
        let info = clip.info();
        assert_eq!(
            (info.width, info.height, info.bits_per_pixel),
            (W as u32, H as u32, bpp)
        );
        assert_eq!(info.compression, Compression::None);
        assert_eq!(info.black_level, 2048);
        assert_eq!(info.camera.name, "Canon EOS 5D Mark III");
        assert_eq!(info.camera.model_id, 0x285);
        assert_eq!(clip.frame_count(), 2);
        assert_eq!(clip.frame(0).unwrap().data, f0, "bpp {bpp}");
        assert_eq!(clip.frame(1).unwrap().data, f1, "bpp {bpp}");
        assert!(clip.warnings().is_empty());
    }
}

#[test]
fn out_of_order_frames_are_sorted_by_timestamp_and_duplicates_dropped() {
    let mut w = basic(14).file_flags(1);
    let frames: Vec<_> = (0..4).map(|i| frame(10 + i, 14)).collect();
    w.vidf(3000, 2, 0, &frames[2], 14);
    w.vidf(1000, 0, 0, &frames[0], 14);
    w.vidf(4000, 3, 0, &frames[3], 14);
    w.vidf(2000, 1, 0, &frames[1], 14);
    w.vidf(2000, 1, 0, &frames[3], 14); // duplicate frame number, must be dropped
    let clip = Clip::from_sources(vec![w.finish()]).unwrap();
    assert_eq!(clip.frame_count(), 4);
    assert!(clip.info().out_of_order);
    for (i, expected) in frames.iter().enumerate() {
        let f = clip.frame(i).unwrap();
        assert_eq!(f.meta.frame.frame_number, i as u32);
        assert_eq!(&f.data, expected);
    }
    assert_eq!(clip.warnings().len(), 1);
}

#[test]
fn chunks_are_merged_in_file_num_order_and_guid_checked() {
    let f: Vec<_> = (0..3).map(|i| frame(20 + i, 14)).collect();
    let mut c0 = basic(14);
    c0.vidf(1000, 0, 0, &f[0], 14);
    let mut c1 = MlvWriter::new(0xDEAD_BEEF, 1, 3);
    c1.vidf(2000, 1, 0, &f[1], 14);
    let mut c2 = MlvWriter::new(0xDEAD_BEEF, 2, 3);
    c2.vidf(3000, 2, 0, &f[2], 14);
    // Pass chunks in scrambled order; fileNum must win.
    let clip = Clip::from_sources(vec![c0.finish(), c2.finish(), c1.finish()]).unwrap();
    assert_eq!(clip.chunk_count(), 3);
    assert_eq!(clip.info().chunk_count, 3);
    assert_eq!(clip.frame_count(), 3);
    for (i, expected) in f.iter().enumerate() {
        let fr = clip.frame(i).unwrap();
        assert_eq!(&fr.data, expected);
        assert_eq!(fr.meta.frame.chunk as usize, i);
    }

    let mut bad = MlvWriter::new(0xBAD, 1, 2);
    bad.vidf(2000, 1, 0, &f[1], 14);
    let mut c0 = basic(14);
    c0.vidf(1000, 0, 0, &f[0], 14);
    match Clip::from_sources(vec![c0.finish(), bad.finish()]) {
        Err(Error::ChunkGuidMismatch { expected, got, .. }) => {
            assert_eq!((expected, got), (0xDEAD_BEEF, 0xBAD));
        }
        other => panic!("expected guid mismatch, got {:?}", other.map(|_| ())),
    }
}

#[test]
fn metadata_timeline_applies_from_timestamp_onward() {
    let mut w = basic(14);
    w.expo(2500, 1600, 10_000);
    w.wbal(3500, 0, 0);
    for i in 0..4u64 {
        w.vidf(1000 * (i + 1), i as u32, 0, &frame(i as u32, 14), 14);
    }
    let clip = Clip::from_sources(vec![w.finish()]).unwrap();
    assert_eq!(clip.exposures().len(), 2);
    let iso = |i| clip.frame_meta(i).unwrap().exposure.unwrap().iso_value;
    assert_eq!([iso(0), iso(1), iso(2), iso(3)], [800, 800, 1600, 1600]);
    let wb = |i| clip.frame_meta(i).unwrap().white_balance.unwrap().kelvin;
    assert_eq!([wb(0), wb(2), wb(3)], [5600, 5600, 0]);
    assert_eq!(clip.info().exposure.as_ref().unwrap().iso_value, 800);
}

#[test]
fn truncated_tail_is_a_warning_not_an_error() {
    let mut w = basic(14);
    w.vidf(1000, 0, 0, &frame(1, 14), 14);
    let full = w.len();
    w.vidf(2000, 1, 0, &frame(2, 14), 14);
    w.truncate_to(full + 40); // header + part of the second frame
    let clip = Clip::from_sources(vec![w.finish()]).unwrap();
    assert_eq!(clip.frame_count(), 1);
    assert_eq!(clip.warnings().len(), 1);
    assert_eq!(clip.warnings()[0].kind, mlvio::WarningKind::TruncatedTail);
    assert!(clip.warnings()[0].to_string().contains("ends inside"));
    assert_eq!(clip.frame(0).unwrap().data, frame(1, 14));
}

#[test]
fn errors_are_typed() {
    // Not MLV.
    assert!(matches!(
        Clip::from_sources(vec![b"RIFF....".to_vec()]),
        Err(Error::NotMlv { .. })
    ));
    // Missing RAWI.
    let mut w = MlvWriter::new(1, 0, 1);
    w.vidf(1, 0, 0, &frame(1, 14), 14);
    assert!(
        matches!(Clip::from_sources(vec![w.finish()]), Err(Error::MissingBlock(t)) if t == mlvio::Tag::RAWI)
    );
    // Frame too short.
    let mut w = basic(14);
    let f = frame(1, 14);
    let mut packed = Vec::new();
    mlvio::decode::unpack::pack(14, &f, &mut packed);
    w.vidf_bytes(1, 0, 0, &packed[..packed.len() - 10], 0);
    let clip = Clip::from_sources(vec![w.finish()]).unwrap();
    assert!(matches!(clip.frame(0), Err(Error::FrameTooShort { .. })));
    assert!(matches!(clip.frame(1), Err(Error::FrameOutOfRange(1))));
    let mut small = vec![0u16; 3];
    assert!(matches!(
        clip.read_frame(0, &mut small),
        Err(Error::DestinationSize { .. })
    ));
    // Unsupported classes open fine but refuse to decode.
    for (class, want) in [
        (0x81u16, Unsupported::Lzma),
        (0x41, Unsupported::Delta),
        (0x11, Unsupported::Jpeg2000),
        (0x101, Unsupported::McRaw),
        (0x02, Unsupported::VideoClass(2)),
    ] {
        let mut w = basic(14).video_class(class);
        w.vidf(1, 0, 0, &f, 14);
        let clip = Clip::from_sources(vec![w.finish()]).unwrap();
        assert_eq!(clip.info().compression, Compression::Unsupported(want));
        match clip.frame(0) {
            Err(Error::Unsupported(u)) => assert_eq!(u, want),
            other => panic!("class {class:#x}: {:?}", other.map(|_| ())),
        }
    }
    // Unsupported bit depth.
    let mut w = basic(16);
    w.vidf_bytes(1, 0, 0, &vec![0u8; W as usize * H as usize * 2], 0);
    let clip = Clip::from_sources(vec![w.finish()]).unwrap();
    assert!(matches!(
        clip.frame(0),
        Err(Error::Unsupported(Unsupported::BitDepth(16)))
    ));
}

#[test]
fn clip_is_sync_and_frames_decode_in_parallel() {
    let mut w = basic(14);
    let frames: Vec<_> = (0..8).map(|i| frame(100 + i, 14)).collect();
    for (i, f) in frames.iter().enumerate() {
        w.vidf(1000 * (i as u64 + 1), i as u32, 0, f, 14);
    }
    let clip = std::sync::Arc::new(Clip::from_sources(vec![w.finish()]).unwrap());
    let handles: Vec<_> = (0..8)
        .map(|i| {
            let c = clip.clone();
            std::thread::spawn(move || c.frame(i).unwrap().data)
        })
        .collect();
    for (i, h) in handles.into_iter().enumerate() {
        assert_eq!(h.join().unwrap(), frames[i]);
    }
}

#[test]
fn open_discovers_chunk_files_on_disk() {
    let dir = std::env::temp_dir().join(format!("mlvio-open-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let f: Vec<_> = (0..2).map(|i| frame(30 + i, 12)).collect();
    let mut c0 = basic(12);
    c0.vidf(1000, 0, 0, &f[0], 12);
    let mut c1 = MlvWriter::new(0xDEAD_BEEF, 1, 2);
    c1.vidf(2000, 1, 0, &f[1], 12);
    std::fs::write(dir.join("Take.MLV"), c0.finish()).unwrap();
    std::fs::write(dir.join("Take.m00"), c1.finish()).unwrap();

    let clip = Clip::open(dir.join("Take.MLV")).unwrap();
    assert_eq!(clip.chunk_count(), 2);
    assert_eq!(clip.frame_count(), 2);
    assert_eq!(clip.frame(1).unwrap().data, f[1]);
    assert!(clip.chunk_path(1).unwrap().ends_with("Take.m00"));

    let single = Clip::open_single(dir.join("Take.MLV")).unwrap();
    assert_eq!(single.frame_count(), 1);
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn hostile_rawi_rectangles_fall_back_to_the_whole_frame() {
    let mut w = MlvWriter::new(7, 0, 1);
    w.rawi_with_areas(
        W,
        H,
        14,
        2048,
        15000,
        [i32::MIN, 5, i32::MAX, -1],
        [i32::MAX, i32::MIN, i32::MIN, i32::MAX],
    );
    w.vidf(1, 0, 0, &frame(1, 14), 14);
    let clip = Clip::from_sources(vec![w.finish()]).unwrap();
    let whole = mlvio::Rect {
        x: 0,
        y: 0,
        width: W as u32,
        height: H as u32,
    };
    assert_eq!(clip.info().active_area, whole);
    assert_eq!(clip.info().crop, whole);
    // A sane, partial active area survives.
    let mut w = MlvWriter::new(7, 0, 1);
    w.rawi_with_areas(W, H, 14, 2048, 15000, [2, 4, 10, 8], [1, 3, 20, 60]);
    w.vidf(1, 0, 0, &frame(1, 14), 14);
    let info = Clip::from_sources(vec![w.finish()]).unwrap().info().clone();
    assert_eq!(
        (info.crop.x, info.crop.y, info.crop.width, info.crop.height),
        (2, 4, 10, 8)
    );
    let a = info.active_area;
    assert_eq!((a.x, a.y, a.width, a.height), (3, 1, 57, 19));
}

#[test]
fn raw_payload_plus_decode_payload_matches_read_frame_and_read_meta_works() {
    let mut w = basic(12);
    let f0 = frame(5, 12);
    w.vidf_padded(1000, 0, 16, &f0, 12, 8);
    let clip = Clip::from_sources(vec![w.finish()]).unwrap();

    let mut bytes = vec![0xAAu8; 3]; // stale content must be replaced
    clip.raw_payload(0, &mut bytes).unwrap();
    let mut via_payload = vec![0u16; f0.len()];
    clip.decode_payload(&bytes, &mut via_payload).unwrap();
    let mut via_read = vec![0u16; f0.len()];
    let mut scratch = mlvio::FrameScratch::new();
    clip.read_frame_with(0, &mut via_read, &mut scratch)
        .unwrap();
    assert_eq!(via_payload, f0);
    assert_eq!(via_read, f0);

    let idnt = clip
        .meta_blocks()
        .iter()
        .find(|m| m.header.tag == mlvio::Tag::IDNT)
        .unwrap();
    match clip.read_meta(idnt).unwrap() {
        mlvio::Block::Identity(id) => assert_eq!(id.camera_name_str(), "Canon EOS 5D Mark III"),
        other => panic!("unexpected {other:?}"),
    }
    // Blocks the clip layer does not consume are still listed.
    assert!(
        clip.meta_blocks()
            .iter()
            .any(|m| m.header.tag == mlvio::Tag::MLVI)
    );
}
