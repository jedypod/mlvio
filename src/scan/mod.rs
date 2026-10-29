// SPDX-License-Identifier: MIT OR Apache-2.0
//! Positional reads and block iteration over one MLV chunk.
//!
//! [`ReadAt`] abstracts "read these bytes at this offset" without `&mut` or
//! seek state, so one open file can be shared by many decoding threads.
//! [`BlockIter`] walks a chunk reading only the 16-byte block headers.

use std::io;

use crate::Error;
use crate::block::BlockHeader;

/// Cap on payload bytes read for non-frame blocks (16 MiB). `CURV` and `DARK`
/// blocks can legitimately be several megabytes.
pub const MAX_META_PAYLOAD: usize = 16 << 20;

/// Positional, shared-reference reads from a byte source.
///
/// `Clip<R>` is `Send`/`Sync` exactly when `R` is, so implement this for a
/// `Sync` type (such as `File`, a slice or a memory map) to decode frames
/// from several threads over one open clip.
pub trait ReadAt {
    /// Fill `buf` with the bytes starting at `pos`.
    ///
    /// Fails with [`io::ErrorKind::UnexpectedEof`] if the source ends before
    /// `buf` is full.
    fn read_exact_at(&self, pos: u64, buf: &mut [u8]) -> io::Result<()>;

    /// Total size of the source in bytes.
    fn size(&self) -> u64;

    /// The whole source as a slice, if it is already in memory.
    ///
    /// Sources that can provide this (slices, `Vec<u8>`, memory maps) let
    /// the clip layer decode frames without copying the payload first.
    fn as_slice(&self) -> Option<&[u8]> {
        None
    }
}

fn eof() -> io::Error {
    io::Error::new(io::ErrorKind::UnexpectedEof, "read past end of source")
}

fn slice_read_at(data: &[u8], pos: u64, buf: &mut [u8]) -> io::Result<()> {
    let start = usize::try_from(pos).map_err(|_| eof())?;
    let end = start.checked_add(buf.len()).ok_or_else(eof)?;
    let src = data.get(start..end).ok_or_else(eof)?;
    buf.copy_from_slice(src);
    Ok(())
}

impl ReadAt for [u8] {
    fn read_exact_at(&self, pos: u64, buf: &mut [u8]) -> io::Result<()> {
        slice_read_at(self, pos, buf)
    }

    fn size(&self) -> u64 {
        self.len() as u64
    }

    fn as_slice(&self) -> Option<&[u8]> {
        Some(self)
    }
}

impl ReadAt for alloc::vec::Vec<u8> {
    fn read_exact_at(&self, pos: u64, buf: &mut [u8]) -> io::Result<()> {
        slice_read_at(self, pos, buf)
    }

    fn size(&self) -> u64 {
        self.len() as u64
    }

    fn as_slice(&self) -> Option<&[u8]> {
        Some(self)
    }
}

macro_rules! forward_read_at {
    ($($ty:ty),*) => {$(
        impl<T: ReadAt + ?Sized> ReadAt for $ty {
            fn read_exact_at(&self, pos: u64, buf: &mut [u8]) -> io::Result<()> {
                (**self).read_exact_at(pos, buf)
            }

            fn size(&self) -> u64 {
                (**self).size()
            }

            fn as_slice(&self) -> Option<&[u8]> {
                (**self).as_slice()
            }
        }
    )*};
}

forward_read_at!(&T, alloc::boxed::Box<T>, std::sync::Arc<T>);

/// Reads use `pread`-style positional I/O. [`ReadAt::size`] queries the file
/// metadata on every call; a failed query reports a size of 0.
impl ReadAt for std::fs::File {
    #[cfg(unix)]
    fn read_exact_at(&self, pos: u64, buf: &mut [u8]) -> io::Result<()> {
        std::os::unix::fs::FileExt::read_exact_at(self, buf, pos)
    }

    #[cfg(windows)]
    fn read_exact_at(&self, mut pos: u64, mut buf: &mut [u8]) -> io::Result<()> {
        use std::os::windows::fs::FileExt;
        while !buf.is_empty() {
            match self.seek_read(buf, pos) {
                Ok(0) => return Err(eof()),
                Ok(n) => {
                    buf = &mut buf[n..];
                    pos += n as u64;
                }
                Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
                Err(e) => return Err(e),
            }
        }
        Ok(())
    }

    fn size(&self) -> u64 {
        self.metadata().map(|m| m.len()).unwrap_or(0)
    }
}

#[cfg(feature = "mmap")]
impl ReadAt for memmap2::Mmap {
    fn read_exact_at(&self, pos: u64, buf: &mut [u8]) -> io::Result<()> {
        slice_read_at(self, pos, buf)
    }

    fn size(&self) -> u64 {
        self.len() as u64
    }

    fn as_slice(&self) -> Option<&[u8]> {
        Some(self)
    }
}

/// Map an I/O failure at block `offset`: a short read means the chunk is
/// truncated, anything else is passed through.
fn io_at(offset: u64) -> impl FnOnce(io::Error) -> Error {
    move |e| {
        if e.kind() == io::ErrorKind::UnexpectedEof {
            Error::Truncated { offset }
        } else {
            Error::Io(e)
        }
    }
}

fn header_at(source_len: u64, offset: u64, bytes: &[u8; 16]) -> Result<BlockHeader, Error> {
    let header = match BlockHeader::from_bytes(bytes) {
        Ok(h) => h,
        Err(Error::BadBlock { reason, .. }) => return Err(Error::BadBlock { offset, reason }),
        Err(e) => return Err(e),
    };
    match offset.checked_add(u64::from(header.size)) {
        Some(end) if end <= source_len => Ok(header),
        _ => Err(Error::Truncated { offset }),
    }
}

/// Read and validate the block header at `offset`.
///
/// Fails with [`Error::Truncated`] if fewer than 16 bytes remain or the block
/// extends past the end of the source, and with [`Error::BadBlock`] if
/// `blockSize < 16`.
pub fn read_header<R: ReadAt + ?Sized>(source: &R, offset: u64) -> Result<BlockHeader, Error> {
    read_header_len(source, source.size(), offset)
}

fn read_header_len<R: ReadAt + ?Sized>(
    source: &R,
    len: u64,
    offset: u64,
) -> Result<BlockHeader, Error> {
    match offset.checked_add(BlockHeader::SIZE as u64) {
        Some(end) if end <= len => {}
        _ => return Err(Error::Truncated { offset }),
    }
    let mut buf = [0u8; 16];
    source
        .read_exact_at(offset, &mut buf)
        .map_err(io_at(offset))?;
    header_at(len, offset, &buf)
}

/// Read the start of the payload of the block at `offset` (the bytes after
/// its 16-byte header) into `buf`, up to `buf.len()` bytes.
///
/// Returns the number of bytes read, which is the smaller of the payload
/// length and `buf.len()`. Reads nothing beyond the block, and does not
/// allocate or query the source size, so it suits per-block work during
/// indexing.
pub fn read_payload_into<R: ReadAt + ?Sized>(
    source: &R,
    offset: u64,
    header: &BlockHeader,
    buf: &mut [u8],
) -> Result<usize, Error> {
    let n = header.payload_len().min(buf.len());
    let start = offset
        .checked_add(BlockHeader::SIZE as u64)
        .ok_or(Error::Truncated { offset })?;
    source
        .read_exact_at(start, &mut buf[..n])
        .map_err(io_at(offset))?;
    Ok(n)
}

/// Read the payload of the block at `offset` (the bytes after its 16-byte
/// header) into a new vector, at most `max_len` bytes.
///
/// Use [`MAX_META_PAYLOAD`] as `max_len` for metadata blocks.
pub fn read_payload<R: ReadAt + ?Sized>(
    source: &R,
    offset: u64,
    header: &BlockHeader,
    max_len: usize,
) -> Result<Vec<u8>, Error> {
    let mut buf = alloc::vec![0u8; header.payload_len().min(max_len)];
    read_payload_into(source, offset, header, &mut buf)?;
    Ok(buf)
}

/// Iterator over the block headers of one chunk.
///
/// Yields `(offset, header)` for each block and advances by `blockSize`.
/// Ends with `None` exactly at the end of the source. After yielding an error
/// ([`Error::Truncated`], [`Error::BadBlock`] or [`Error::Io`]) it is fused
/// and returns `None` forever.
#[derive(Debug)]
pub struct BlockIter<'a, R: ReadAt + ?Sized> {
    source: &'a R,
    offset: u64,
    len: u64,
    done: bool,
}

impl<'a, R: ReadAt + ?Sized> BlockIter<'a, R> {
    /// Iterate from the start of `source`.
    pub fn new(source: &'a R) -> Self {
        Self::from_offset(source, 0)
    }

    /// Iterate starting at byte `offset`, which must be a block boundary.
    ///
    /// The source size is sampled once here.
    pub fn from_offset(source: &'a R, offset: u64) -> Self {
        BlockIter {
            source,
            offset,
            len: source.size(),
            done: false,
        }
    }

    /// Offset of the next block to be read.
    pub fn offset(&self) -> u64 {
        self.offset
    }

    /// Source size as sampled when the iterator was created.
    pub fn source_size(&self) -> u64 {
        self.len
    }
}

impl<R: ReadAt + ?Sized> Iterator for BlockIter<'_, R> {
    type Item = Result<(u64, BlockHeader), Error>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done || self.offset >= self.len {
            self.done = true;
            return None;
        }
        let offset = self.offset;
        match read_header_len(self.source, self.len, offset) {
            Ok(h) => {
                self.offset = offset + u64::from(h.size);
                Some(Ok((offset, h)))
            }
            Err(e) => {
                self.done = true;
                Some(Err(e))
            }
        }
    }
}

impl<R: ReadAt + ?Sized> core::iter::FusedIterator for BlockIter<'_, R> {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::Tag;

    fn block(out: &mut Vec<u8>, tag: &[u8; 4], ts: u64, payload: &[u8]) {
        out.extend_from_slice(tag);
        out.extend_from_slice(&((16 + payload.len()) as u32).to_le_bytes());
        out.extend_from_slice(&ts.to_le_bytes());
        out.extend_from_slice(payload);
    }

    fn three() -> Vec<u8> {
        let mut v = Vec::new();
        block(&mut v, b"MLVI", 0, &[1u8; 36]);
        block(&mut v, b"NULL", 5, &[]);
        block(&mut v, b"VIDF", 7, &[9u8; 40]);
        v
    }

    #[test]
    fn iterates_three_blocks() {
        let data = three();
        let src: &[u8] = &data;
        let items: Vec<_> = BlockIter::new(src).map(|r| r.unwrap()).collect();
        assert_eq!(items.len(), 3);
        assert_eq!(items[0].0, 0);
        assert_eq!(items[0].1.tag, Tag::MLVI);
        assert_eq!(items[0].1.size, 52);
        assert_eq!(items[1].0, 52);
        assert_eq!(items[1].1.tag, Tag::NULL);
        assert_eq!(items[1].1.timestamp, 5);
        assert_eq!(items[2].0, 68);
        assert_eq!(items[2].1.tag, Tag::VIDF);
        assert_eq!(items[2].1.size, 56);

        let p = read_payload(src, items[2].0, &items[2].1, MAX_META_PAYLOAD).unwrap();
        assert_eq!(p, vec![9u8; 40]);
        let p = read_payload(src, items[2].0, &items[2].1, 10).unwrap();
        assert_eq!(p.len(), 10);
        let mut small = [0u8; 64];
        assert_eq!(
            read_payload_into(src, items[2].0, &items[2].1, &mut small).unwrap(),
            40
        );
        assert_eq!(&small[..40], &[9u8; 40]);
        assert_eq!(read_header(src, 52).unwrap().tag, Tag::NULL);

        // Vec and Box / Arc wrappers work too, and expose the slice.
        assert_eq!(BlockIter::new(&data).count(), 3);
        let b: Box<[u8]> = data.clone().into_boxed_slice();
        assert_eq!(BlockIter::new(&b).count(), 3);
        let a = std::sync::Arc::new(data.clone());
        assert_eq!(BlockIter::new(&a).count(), 3);
        assert_eq!(a.as_slice().unwrap().len(), data.len());
    }

    #[test]
    fn truncated_block_then_fused() {
        let mut data = three();
        data.truncate(data.len() - 1);
        let src: &[u8] = &data;
        let mut it = BlockIter::new(src);
        assert!(it.next().unwrap().is_ok());
        assert!(it.next().unwrap().is_ok());
        match it.next() {
            Some(Err(Error::Truncated { offset: 68 })) => {}
            other => panic!("unexpected {other:?}"),
        }
        assert!(it.next().is_none());
        assert!(it.next().is_none());
    }

    #[test]
    fn partial_header_is_truncated() {
        let mut data = three();
        data.extend_from_slice(b"VID");
        let src: &[u8] = &data;
        let mut it = BlockIter::new(src);
        for _ in 0..3 {
            assert!(it.next().unwrap().is_ok());
        }
        match it.next() {
            Some(Err(Error::Truncated { offset: 124 })) => {}
            other => panic!("unexpected {other:?}"),
        }
        assert!(it.next().is_none());
        assert!(matches!(
            read_header(src, 124),
            Err(Error::Truncated { offset: 124 })
        ));
    }

    #[test]
    fn small_size_is_bad_block() {
        let mut data = three();
        let mut bad = Vec::new();
        bad.extend_from_slice(b"NULL");
        bad.extend_from_slice(&8u32.to_le_bytes());
        bad.extend_from_slice(&0u64.to_le_bytes());
        data.extend_from_slice(&bad);
        let src: &[u8] = &data;
        let mut it = BlockIter::new(src);
        for _ in 0..3 {
            assert!(it.next().unwrap().is_ok());
        }
        match it.next() {
            Some(Err(Error::BadBlock { offset: 124, .. })) => {}
            other => panic!("unexpected {other:?}"),
        }
        assert!(it.next().is_none());
    }

    #[test]
    fn empty_source() {
        let src: &[u8] = &[];
        assert!(BlockIter::new(src).next().is_none());
    }

    #[test]
    fn slice_short_read_is_eof() {
        let src: &[u8] = &[1, 2, 3];
        let mut buf = [0u8; 4];
        let e = ReadAt::read_exact_at(src, 0, &mut buf).unwrap_err();
        assert_eq!(e.kind(), io::ErrorKind::UnexpectedEof);
        let mut buf = [0u8; 2];
        ReadAt::read_exact_at(src, 1, &mut buf).unwrap();
        assert_eq!(buf, [2, 3]);
    }

    #[test]
    fn payload_past_end_is_truncated() {
        let mut data = three();
        data.truncate(data.len() - 5);
        let src: &[u8] = &data;
        let h = BlockHeader {
            tag: Tag::VIDF,
            size: 56,
            timestamp: 7,
        };
        assert!(matches!(
            read_payload(src, 68, &h, MAX_META_PAYLOAD),
            Err(Error::Truncated { offset: 68 })
        ));
    }
}
