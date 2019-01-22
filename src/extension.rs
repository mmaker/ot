// XXX: check architecture
use std::arch::x86_64::_mm256_movemask_epi8;

// XXX: we're assuming it's LittleEndian
use byteorder::{NativeEndian, ByteOrder};
use packed_simd::u8x32;
use digest::Digest;
use rand::{CryptoRng, Rng};

use crate::{Sender, Receiver};


#[inline]
pub fn movemask8x32(x: u8x32) -> i32 {
    unsafe { _mm256_movemask_epi8(std::mem::transmute(x)) }
}

#[inline]
fn interleave_left(l: u8x32, r: u8x32) -> u8x32 {
    shuffle!(l, r, [0, 32, 1, 33, 2, 34, 3, 35, 4, 36, 5, 37, 6, 38, 7, 39, 8, 40, 9, 41, 10, 42, 11, 43, 12, 44, 13, 45, 14, 46, 15, 47])
}


#[inline]
fn interleave_right(l: u8x32, r: u8x32) -> u8x32 {
    shuffle!(l, r, [16, 48, 17, 49, 18, 50, 19, 51, 20, 52, 21, 53, 22, 54, 23, 55, 24, 56, 25, 57, 26, 58, 27, 59, 28, 60, 29, 61, 30, 62, 31, 63])
}


#[inline]
fn transpose_round(src: &[u8], dst: &mut [u8], l: usize) {
    assert_eq!(src.len() % (32*2), 0);
    let half: usize = src.len() / 2;

    let mut j = 0;
    for i in (0 .. half).step_by(l) {
        let r0 = u8x32::from_slice_unaligned(&src[i .. i+32]);
        let r1 = u8x32::from_slice_unaligned(&src[half+i .. half+i+32]);

        let t0 = interleave_left(r0, r1);
        let t1 = interleave_right(r0, r1);

        t0.write_to_slice_unaligned(&mut dst[j .. j+32]);
        t1.write_to_slice_unaligned(&mut dst[j+32 .. j+64]);
        j += 64;
    }
}

#[inline]
fn transpose_round_inplace(dst: &mut [u8]) {
    assert_eq!(dst.len() % 128, 0);
    let half: usize = dst.len() / 2 ;

    // Copy the first half of the matrix:
    // when interleaving we write two consecutive rows, but we read only one.
    // We gain about 1 kc in moving this outside the round.
    // Also, note that at the i-th step I need to copy i lines.
    // This means that 1/4 of the size should suffice
    let mut src  =  vec![0u8; half];
    src.clone_from_slice(&dst[ .. half]);

    for i in (0 .. half).step_by(32) {
        let r0 = u8x32::from_slice_unaligned(&src[i .. i+32]);
        let r1 = u8x32::from_slice_unaligned(&dst[half+i .. half+i+32]);

        let t0 = interleave_left(r0, r1);
        let t1 = interleave_right(r0, r1);

        t0.write_to_slice_unaligned(&mut dst[2*i .. 2*i+32]);
        t1.write_to_slice_unaligned(&mut dst[2*i+32 .. 2*i+64]);
    }
}

#[inline]
fn transpose_u8(m: &mut [u8]) {
    // 128.log2() rounds will bring us to the permutation we want
    for _i in 0 .. 7 {
       transpose_round_inplace(m);
    }
}


pub fn transpose128(m: &mut [u8]) {
    // transpose byte-wise. We are left with a 256x128 byte matrix
    transpose_u8(m);

    // let mut time = unsafe { -std::arch::x86_64::_rdtsc() };
    // … code …
    // time += unsafe { std::arch::x86_64::_rdtsc() };
    // println!("{}", time);

    // We are left with transposing 1x8 bit matrices.
    // We do so in chunks of one row
    for i in (0 ..  m.len()).step_by(128) {
        let mut t0 = u8x32::from_slice_unaligned(&m[i .. i+32]);
        let mut t1 = u8x32::from_slice_unaligned(&m[i+32 .. i+64]);
        let mut t2 = u8x32::from_slice_unaligned(&m[i+64 .. i+96]);
        let mut t3 = u8x32::from_slice_unaligned(&m[i+96 .. i+128]);

        let mut chunks = [0i32; 32];
        for j in 0 .. 8 {
            chunks[j*4 + 0] = movemask8x32(t0);
            chunks[j*4 + 1] = movemask8x32(t1);
            chunks[j*4 + 2] = movemask8x32(t2);
            chunks[j*4 + 3] = movemask8x32(t3);

            t0 <<= 1;
            t1 <<= 1;
            t2 <<= 1;
            t3 <<= 1;
        }
        NativeEndian::write_i32_into(&chunks[..], &mut m[i .. i+128]);
    }
}



fn xor3(t0: &mut [u8], t1: &[u8], t2: &[u8]) {
    for i in (0 .. t0.len()).step_by(32 * 4) {
        let r0 =
            u8x32::from_slice_unaligned(&t0[i .. i+32]) ^
            u8x32::from_slice_unaligned(&t1[i .. i+32]) ^
            u8x32::from_slice_unaligned(&t2[i .. i+32]);
        let r1 =
            u8x32::from_slice_unaligned(&t0[i+32 .. i+64]) ^
            u8x32::from_slice_unaligned(&t1[i+32 .. i+64]) ^
            u8x32::from_slice_unaligned(&t2[i+32 .. i+64]);
        let r2 =
            u8x32::from_slice_unaligned(&t0[i+64 .. i+96]) ^
            u8x32::from_slice_unaligned(&t1[i+64 .. i+96]) ^
            u8x32::from_slice_unaligned(&t2[i+64 .. i+96]);
        let r3 =
            u8x32::from_slice_unaligned(&t0[i+96 .. i+128]) ^
            u8x32::from_slice_unaligned(&t1[i+96 .. i+128]) ^
            u8x32::from_slice_unaligned(&t2[i+96 .. i+128]);

        r0.write_to_slice_unaligned(&mut t0[i .. i+32]);
        r1.write_to_slice_unaligned(&mut t0[i+32 .. i+64]);
        r2.write_to_slice_unaligned(&mut t0[i+64 .. i+96]);
        r3.write_to_slice_unaligned(&mut t0[i+96 .. i+128]);
    }
}


pub struct ReceiverExt(Sender);
pub struct SenderExt(Receiver);

impl ReceiverExt {
    pub fn new<R>(csprng: &mut R) -> (Self, [u8; 32])
        where R: CryptoRng + Rng
    {
        let (sender, s) = Sender::new(csprng);
        (ReceiverExt(sender), s)
    }

    fn key_decompress(dst: &mut [u8], key: &[u8]) {
        let one = u8x32::splat(0xff);

        for (i, &k) in (0 .. 128).zip(key) {
            (k * one).write_to_slice_unaligned(&mut dst[i*32 .. i*32+32]);
        }
    }

    // pub fn keys<D>(&self, r_bytes: &[u8], c8: &[u8]) -> Option<Vec<[u8; 16]>>
    //     where D: Digest + Default
    // {
    //     if r_bytes.len() != 128 * 32 {
    //         return None;
    //     }

    //     let mut t0 = [0u8; 32*128];
    //     let mut t1 = [0u8; 32*128];
    //     let mut choices = [0u8; 32*128];

    //     Self::key_decompress(&mut choices, c8);
    //     for i in 0 .. 128 {
    //         let keys = self.0.keys::<D>(&r_bytes[i*32 .. i*32+32], c8[i] as usize)?;

    //         t0[i*32 .. i*32+32].copy_from_slice(keys[0].as_slice());
    //         t1[i*32 .. i*32+32].copy_from_slice(keys[1].as_slice());
    //     }

    //     xor3(&mut t0, &t1, &choices);
    //     transpose128x256(&mut t0);

    //     let mut chunks : Vec<[u8; 16]>= vec![[0u8; 16]; 32];
    //     for (i, chunk) in t0.chunks(16).enumerate() {
    //         chunks[i].copy_from_slice(chunk)
    //     }
    //     Some(chunks)
    // }
}

// fn addmul(mut key: u128, u: &[u8], t: &mut [u8]) {
//     for i in (0 .. t.len()).step_by(32 * 4) {
//         let b = (key & 1) as u8;
//         let mut r0 =
//             u8x32::from_slice_unaligned(&t[i .. i+32]) ^
//             ( * u8x32::from_slice_unaligned(&u[i .. i+32]));
//         let mut r1 =
//             u8x32::from_slice_unaligned(&t[i+32 .. i+64]) ^
//             (b * u8x32::from_slice_unaligned(&u[i+32 .. i+64]));
//         let mut r2 =
//             u8x32::from_slice_unaligned(&t[i+64 .. i+96]) ^
//             (b * u8x32::from_slice_unaligned(&u[i+64 .. i+96]));
//         let mut r3 =
//             u8x32::from_slice_unaligned(&t[i+96 .. i+128]) ^
//             (b * u8x32::from_slice_unaligned(&u[i+96 .. i+128]));

//         key >>= 1;
//         r0.write_to_slice_unaligned(&mut t[i .. i+32]);
//         r1.write_to_slice_unaligned(&mut t[i+32 .. i+64]);
//         r2.write_to_slice_unaligned(&mut t[i+64 .. i+96]);
//         r3.write_to_slice_unaligned(&mut t[i+96 .. i+128]);
//     }
// }


#[cfg(test)]
mod tests {
    use super::*;
    use rand::{RngCore, thread_rng};

    fn get_bit(src: &[u8], i: usize) -> u8 {
        let byte = src[i / 8];
        let bit_pos = 7 - (i % 8);
        (byte & (1 << bit_pos) != 0) as u8
    }

    fn set_bit(dst: &mut [u8], i: usize, b: u8) {
        let bit_pos = i % 8;
        if b == 1 {
            dst[i / 8] |= 1 << bit_pos;
        } else {
            dst[i / 8] &= !(1 << bit_pos);
        }
    }

    #[test]
    fn test_interleave() {
        use std::io::Read;

        let test =
            b"\xc1i3\xf6\xbe1\xcc\xfd0f\xa9>%\xb7\x8a\xf8,\x08t\x9e\xe7\xc9$\xb8\xec\xfbZTsN\
              \x1c\xc4\x01\xe7\xe0\x19\xfa\x07@N\xb5\x7fIl\x8e\xeb\xb4N\xd6\xf4Q\x94^GlG\xe3\
              {\x16\xe9\xa3\xa9\xde\x08";
        let expected =
            b"\xc1\x01i\xe73\xe0\xf6\x19\xbe\xfa1\x07\xcc@\xfdN0\xb5f\x7f\xa9I>l%\x8e\xb7\xeb\
              \x8a\xb4\xf8N,\xd6\x08\xf4tQ\x9e\x94\xe7^\xc9G$l\xb8G\xec\xe3\xfb{Z\x16T\xe9s\xa3\
              N\xa9\x1c\xde\xc4\x08";
        let mut got = [0u8; 64];
        let a = u8x32::from_slice_unaligned(&test[ .. 32]);
        let b = u8x32::from_slice_unaligned(&test[32 .. ]);
        interleave_left(a, b).write_to_slice_unaligned(&mut got[ .. 32]);
        interleave_right(a, b).write_to_slice_unaligned(&mut got[32 ..]);
        assert_eq!(&got[..], &expected[..]);
    }

    #[test]
    fn test_transpose_round() {
        let mut test : Vec<u8> =
            b"Q)FSg\xedb\x02\'q\xdd\x9c\x8e\x12\x1d\xf9\x08\xe66\xf9\x1d\xf1\x90:b\x9fi_\xe3yf\
              \xff\x10\xdf\xd3r\xea\"\xd5f\xe0\x98{\xd9\x9ffb*\xe8\xda5kwe\xf8H2\x0e\xdf/\xc9i\
              \xd5\xa0\xdbHD\x15\ns~\xaa1\xfe\x86H\xa9.\x89\xa2\x03\x86\xef\xc6\xd3\x19D\xed\xab\
              \x16\xfbh\xee\xf9\x0e\xbeI\x17\x04)\xca\xac\x07\xd0V-\xe6v|G\\\xff\xd5B\x08\xddM\xf2\
              \nd\x85\xfe@\xe9\xfc\xab\xf4N".into_iter().cloned().cycle().take(32 * 128).collect();
        let expected =
            b"QQ))FFSSgg\xed\xedbb\x02\x02\'\'qq\xdd\xdd\x9c\x9c\x8e\x8e\x12\x12\x1d\x1d\xf9\xf9\x08\
            \x08\xe6\xe666\xf9\xf9\x1d\x1d\xf1\xf1\x90\x90::bb\x9f\x9fii__\xe3\xe3yyff\xff\xff\x10\
            \x10\xdf\xdf\xd3\xd3rr\xea\xea\"\"\xd5\xd5ff\xe0\xe0\x98\x98{{\xd9\xd9\x9f\x9fffbb**\xe8\
            \xe8\xda\xda55kkwwee\xf8\xf8HH22\x0e\x0e\xdf\xdf//\xc9\xc9ii\xd5\xd5\xa0\xa0\xdb\xdbHHDD\
            \x15\x15\n\nss~~\xaa\xaa11\xfe\xfe\x86\x86HH\xa9\xa9..\x89\x89\xa2\xa2\x03\x03\x86\x86\xef\
            \xef\xc6\xc6\xd3\xd3\x19\x19DD\xed\xed\xab\xab\x16\x16\xfb\xfbhh\xee\xee\xf9\xf9\x0e\x0e\xbe\
            \xbeII\x17\x17\x04\x04))\xca\xca\xac\xac\x07\x07\xd0\xd0VV--\xe6\xe6vv||GG\\\\\xff\xff\xd5\xd5\
            BB\x08\x08\xdd\xddMM\xf2\xf2\n\ndd\x85\x85\xfe\xfe@@\xe9\xe9\xfc\xfc\xab\xab\xf4\xf4NN";
        transpose_round_inplace(&mut test);
        assert_eq!(&test[.. 256], &expected[.. 256]);
    }

    fn transpose256_u8_naif(dst: &mut [u8], src: &[u8]) {
        assert!(src.len() % 32 == 0);
        let r = src.len() / 32;

        for (i, &x) in src.iter().enumerate() {
            let (row, col) = (i / 32, i % 32);
            dst[col * r + row] = x;
        }
    }

    #[test]
    fn test_transpose256_u8() {
        let mut a = [0u8; 128*32];
        let mut prng = thread_rng();

        // then test transpose bytes acts correctly
        prng.fill_bytes(&mut a);
        let mut b = [0u8; 128*32];
        transpose256_u8_naif(&mut b, &a);
        transpose_u8(&mut a);
        assert_eq!(&a[..], &b[..]);

    }

    // Below, src is a matrix n x m = l.
    fn transpose_naif(dst: &mut [u8], src: &[u8], m: usize) {
        assert!(src.len() % m == 0);
        let l = src.len() * 8;
        let n = l / m;

        for i in 0 .. l {
            let bit = get_bit(src, i);
            let (row, col) = (i / m, i % m);
            set_bit(dst, col * n + row, bit);
        }
    }

    #[test]
    fn test_transpose256() {
        let mut a = [0u8; 128*32];
        let mut prng = thread_rng();

        // then test transpose bytes acts correctly
        prng.fill_bytes(&mut a[..]);
        let mut b = [0u8; 128*32];
        transpose_naif(&mut b, &a, 256);
        transpose128(&mut a);
        assert_eq!(&a[..], &b[..]);
    }

    #[test]
    fn test_transpose128() {
        for p in 5 .. 14 {
            let brows = 1 << p;
            let cols = 128;

            let mut a = vec![0u8; brows*cols];
            let mut b = vec![0u8; brows*cols];
            let mut prng = thread_rng();
            // then test transpose bytes acts correctly
            prng.fill_bytes(&mut a[..]);
            transpose_naif(&mut b, &a, brows*8);
            transpose128(&mut a);
            assert_eq!(&a[..], &b[..]);
        }
    }
}
