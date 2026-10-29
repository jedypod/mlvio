#![no_main]
use libfuzzer_sys::fuzz_target;
use mlvio::block::{BlockHeader, Tag, parse_block};

const TAGS: [Tag; 24] = [
    Tag::MLVI, Tag::VIDF, Tag::AUDF, Tag::RAWI, Tag::RAWC, Tag::WAVI, Tag::EXPO, Tag::LENS,
    Tag::ELNS, Tag::RTCI, Tag::IDNT, Tag::XREF, Tag::INFO, Tag::DISO, Tag::MARK, Tag::STYL,
    Tag::ELVL, Tag::WBAL, Tag::DEBG, Tag::VERS, Tag::CURV, Tag::DARK, Tag::NULL, Tag::BKUP,
];

fuzz_target!(|data: &[u8]| {
    // Every block parser over the same payload, plus the header parser and
    // the block iterator over the raw bytes.
    for tag in TAGS {
        let header = BlockHeader { tag, size: (data.len() + 16) as u32, timestamp: 0 };
        let _ = parse_block(&header, data);
    }
    let _ = BlockHeader::from_bytes(data);
    for item in mlvio::scan::BlockIter::new(data) {
        if item.is_err() {
            break;
        }
    }
    let _ = mlvio::Clip::from_sources(vec![data.to_vec()]);
});
