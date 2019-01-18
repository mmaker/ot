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
fn transpose256_round(src: &[u8], dst: &mut [u8]) {
    assert_eq!(src.len() % (32*2), 0);
    let half: usize = src.len() / 2;

    for i in (0 .. half).step_by(32) {
        let r0 = u8x32::from_slice_unaligned(&src[i .. i+32]);
        let r1 = u8x32::from_slice_unaligned(&src[half+i .. half+i+32]);

        let t0 = interleave_left(r0, r1);
        let t1 = interleave_right(r0, r1);

        t0.write_to_slice_unaligned(&mut dst[2*i .. 2*i+32]);
        t1.write_to_slice_unaligned(&mut dst[2*i+32 .. 2*i+64]);
    }
}

#[inline]
fn transpose128x256_bytes(m: &mut [u8]) {
    let n = &mut [0u8; 128*32];
    // the following line is costing us +300 cycles.
    // XXX. make transpose256_round inplace.
    n.clone_from_slice(m);

    // 128.log2() rounds will bring us to the permutation we want
    transpose256_round(n, m);
    transpose256_round(m, n);
    transpose256_round(n, m);
    transpose256_round(m, n);
    transpose256_round(n, m);
    transpose256_round(m, n);
    transpose256_round(n, m);
}


pub fn transpose128x256(m: &mut [u8]) {
    transpose128x256_bytes(m);

    let mut chunk = [0i32; 8];
    for i in 0 .. 32 {
        let mut t0 = u8x32::from_slice_unaligned(&m[i*32 .. i*32+32]);

        chunk[7] = movemask8x32(t0);
        for j in (0 .. 7).rev() {
            t0 <<= 1;
            chunk[j] = movemask8x32(t0);
        }
        NativeEndian::write_i32_into(&chunk, &mut m[i*32 .. i*32+32])
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
        (src[i / 8] & (1 << (i % 8)) != 0) as u8
    }

    fn set_bit(dst: &mut [u8], i: usize, b: u8) {
        if b == 1 {
            dst[i / 8] |= 1 << (i % 8);
        } else {
            dst[i / 8] &= !(1 << (i % 8));
        }
    }

    #[test]
    fn test_set_get_bit() {
        let mut a = [0xffu8, 0x00];
        assert_eq!(get_bit(&a, 7), 1);
        assert_eq!(get_bit(&a, 9), 0);
        set_bit(&mut a, 7, 0);
        assert_eq!(get_bit(&a, 7), 0);
    }

    #[test]
    fn test_interleave() {
        let a : Vec<_> = (0u8 .. 32u8).collect();
        let a = u8x32::from_slice_unaligned(&a);

        let b : Vec<_> = (32u8 .. 64u8).collect();
        let b = u8x32::from_slice_unaligned(&b);


        let got = interleave_left(a, b);
        assert_eq!(got.extract(0), 0);
        assert_eq!(got.extract(1), 32);
        assert_eq!(got.extract(2), 1);
        assert_eq!(got.extract(31), 32+16-1);
    }

    #[test]
    fn test_transpose128x256() {
        let mut a = [0u8; 128*32];
        let mut prng = thread_rng();

        // first test interleave acts correctly on a single round.
        a[0] = 1;
        a[a.len() / 2] = 1;
        a[1] = 1;
        let b = a.clone();
        transpose256_round(&b, &mut a);
        assert_eq!(a[0], 1);
        assert_eq!(a[1], 1);
        assert_eq!(a[2], 1);

        // then test transpose bytes acts correctly
        prng.fill_bytes(&mut a);
        let b = a.clone();
        transpose128x256_bytes(&mut a);
        assert_eq!(a[0], b[0]);
        assert_eq!(a[1], b[32]);
        assert_eq!(a[2], b[64]);

        // finally, test bit transposition.
        prng.fill_bytes(&mut a);
        let b = a.clone();
        transpose128x256(&mut a);
        assert_eq!(get_bit(&a, 0), get_bit(&b, 0));
        assert_eq!(get_bit(&a, 1), get_bit(&b, 256));
        assert_eq!(get_bit(&a, 2), get_bit(&b, 512));
    }
}
