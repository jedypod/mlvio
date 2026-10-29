#![no_main]
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // Whole pipeline: open from memory, decode every frame (bounded).
    if let Ok(clip) = mlvio::Clip::from_sources(vec![data.to_vec()]) {
        let info = clip.info();
        let n = info.width as usize * info.height as usize;
        if n == 0 || n > 1 << 22 {
            return;
        }
        let mut buf = vec![0u16; n];
        let mut scratch = mlvio::FrameScratch::new();
        for i in 0..clip.frame_count().min(8) {
            let _ = clip.read_frame_with(i, &mut buf, &mut scratch);
            let _ = clip.frame_meta(i);
        }
    }
});
