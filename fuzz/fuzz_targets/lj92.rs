#![no_main]
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if let Ok(hdr) = mlvio::decode::lj92::parse(data) {
        // Cap the allocation so the fuzzer explores the decoder, not OOM.
        let n = hdr.sample_count();
        if n <= 1 << 22 {
            let mut out = vec![0u16; n];
            let _ = mlvio::decode::lj92::decode_into(data, &hdr, &mut out);
        }
    }
});
