// XXX: check architecture
use packed_simd::*;
use byteorder::{NativeEndian, BigEndian, LittleEndian, ByteOrder};
use std::ops::{BitXorAssign, BitAndAssign};


use std::arch::x86_64::_mm256_movemask_epi8;

#[inline]
pub fn movemask8x32(x: u8x32) -> i32 {
    unsafe { _mm256_movemask_epi8(std::mem::transmute(x)) }
}


#[derive(Debug)]
pub struct BitIterator<E> {
    t: E,
    n: usize,
}

impl<E: AsRef<[u8]>> BitIterator<E> {
    pub fn new(t: E) -> Self {
        let n = t.as_ref().len() * 8;

        BitIterator { t, n }
    }
}

impl<E: AsRef<[u8]>> Iterator for BitIterator<E> {
    type Item = bool;

    fn next(&mut self) -> Option<bool> {
        if self.n == 0 {
            None
        } else {
            self.n -= 1;
            let part = self.n / 8;
            let bit = self.n - (8 * part);

            Some(self.t.as_ref()[part] & (1 << bit) > 0)
        }
    }
}


pub struct Bits<E: AsRef<[u8]>> (E);

impl<E: AsRef<[u8]>> Bits<E> {
    fn new(v: E) -> Self {
        Bits(v)
    }

    fn iter(self) -> BitIterator<E> {
        BitIterator::new(self.0)
    }
}

impl<L, R> BitXorAssign<Bits<R>> for Bits<L> where
    R: AsRef<[u8]>,
    L: AsRef<[u8]> + AsMut<[u8]>
{
    #[inline]
    fn bitxor_assign(&mut self, rhs: Bits<R>)
    {
        let va = self.0.as_mut();
        let vb = rhs.0.as_ref();

        assert_eq!(va.len(), vb.len());
        assert!(va.len() % 256 == 0);
        for i in (0 .. va.len()).step_by(32 * 4) {
            let t0 =
                u8x32::from_slice_unaligned(&va[i .. i+32]) ^
                u8x32::from_slice_unaligned(&vb[i .. i+32]);
            let t1 =
                u8x32::from_slice_unaligned(&va[i+32 .. i+64]) ^
                u8x32::from_slice_unaligned(&vb[i+32 .. i+64]);
            let t2 =
                u8x32::from_slice_unaligned(&va[i+64 .. i+96]) ^
                u8x32::from_slice_unaligned(&vb[i+64 .. i+96]);
            let t3 =
                u8x32::from_slice_unaligned(&va[i+96 .. i+128]) ^
                u8x32::from_slice_unaligned(&vb[i+96 .. i+128]);

            t0.write_to_slice_unaligned(&mut va[i .. i+32]);
            t1.write_to_slice_unaligned(&mut va[i+32 .. i+64]);
            t2.write_to_slice_unaligned(&mut va[i+64 .. i+96]);
            t3.write_to_slice_unaligned(&mut va[i+96 .. i+128]);
        }
    }
}

impl<L, R> BitAndAssign<Bits<R>> for Bits<L> where
    R: AsRef<[u8]>,
    L: AsRef<[u8]> + AsMut<[u8]>
{
    #[inline]
    fn bitand_assign(&mut self, rhs: Bits<R>)
    {
        let va : &mut [u8] = self.0.as_mut();
        let vb = rhs.0.as_ref();

        assert_eq!(va.len(), vb.len());
        assert!(va.len() % 256 == 0);
        for i in (0 .. va.len()).step_by(32 * 4) {
            let t0 =
                u8x32::from_slice_unaligned(&va[i .. i+32]) &
                u8x32::from_slice_unaligned(&vb[i .. i+32]);
            let t1 =
                u8x32::from_slice_unaligned(&va[i+32 .. i+64]) &
                u8x32::from_slice_unaligned(&vb[i+32 .. i+64]);
            let t2 =
                u8x32::from_slice_unaligned(&va[i+64 .. i+96]) &
                u8x32::from_slice_unaligned(&vb[i+64 .. i+96]);
            let t3 =
                u8x32::from_slice_unaligned(&va[i+96 .. i+128]) &
                u8x32::from_slice_unaligned(&vb[i+96 .. i+128]);

            t0.write_to_slice_unaligned(&mut va[i .. i+32]);
            t1.write_to_slice_unaligned(&mut va[i+32 .. i+64]);
            t2.write_to_slice_unaligned(&mut va[i+64 .. i+96]);
            t3.write_to_slice_unaligned(&mut va[i+96 .. i+128]);
        }
    }
}



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

pub fn transpose256x256(dst: &mut [u8], src: &[u8]) {
    for i in 0 .. 256 {
        for j in 0 .. 256 {
            let b = get_bit(src, i * 256 + j);
            set_bit(dst, j * 256 + i, b)
        }
    }
}

pub fn simd_transpose256x256(dst: &mut [u8], src: &[u8]) {
    assert_eq!(src.len(), 32 * 256);

    for l in 0 .. 8 {
        for i in 0 .. 256/8 {
            let mut t0 = u8x32::new(
                src[i + 32*0 + 1024*l],
                src[i + 32*1 + 1024*l],
                src[i + 32*2 + 1024*l],
                src[i + 32*3 + 1024*l],
                src[i + 32*4 + 1024*l],
                src[i + 32*5 + 1024*l],
                src[i + 32*6 + 1024*l],
                src[i + 32*7 + 1024*l],
                src[i + 32*8 + 1024*l],
                src[i + 32*9 + 1024*l],
                src[i + 32*10 + 1024*l],
                src[i + 32*11 + 1024*l],
                src[i + 32*12 + 1024*l],
                src[i + 32*13 + 1024*l],
                src[i + 32*14 + 1024*l],
                src[i + 32*15 + 1024*l],
                src[i + 32*16 + 1024*l],
                src[i + 32*17 + 1024*l],
                src[i + 32*18 + 1024*l],
                src[i + 32*19 + 1024*l],
                src[i + 32*20 + 1024*l],
                src[i + 32*21 + 1024*l],
                src[i + 32*22 + 1024*l],
                src[i + 32*23 + 1024*l],
                src[i + 32*24 + 1024*l],
                src[i + 32*25 + 1024*l],
                src[i + 32*26 + 1024*l],
                src[i + 32*27 + 1024*l],
                src[i + 32*28 + 1024*l],
                src[i + 32*29 + 1024*l],
                src[i + 32*30 + 1024*l],
                src[i + 32*31 + 1024*l]
            );

            for j in (0 .. 8).rev() {
                let got = [movemask8x32(t0)];
                t0 <<= 1;
                NativeEndian::write_i32_into(&got, &mut dst[j*32+i*8*32+l*4.. i*8*32+j*32+l*4+4]);
            }
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use rand::{RngCore, thread_rng};

    #[test]
    fn test_bit_xor() {
        let mut _a = [0u8; 32 * 256];
        let _b = [0xffu8; 32 * 256];

        let mut a = Bits(&mut _a[..]);
        let b = Bits(&_b[..]);

        a ^= b;
        assert_eq!(a.0[0], 0xff);
        assert_eq!(a.0[100], 0xff);
    }

    #[test]
    fn test_and() {
        let mut _a = [0xffu8; 32 * 256];
        let _b = [0u8; 32 * 256];

        let mut a = Bits(&mut _a[..]);
        let b = Bits(&_b[..]);

        a &= b;
        assert_eq!(a.0[0], 0);
        assert_eq!(a.0[100], 0);
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
    fn test_transpose() {
        // transpose the identity matrix
        let mut a = [0u8; 256 * 256 / 8];
        let mut b = [0u8; 256 * 256 / 8];
        for i in 0 .. 256 {
            set_bit(&mut a,  i * 256 + i, 1);
        }

        transpose256x256(&mut b, &a);
        for i in 0 .. 256 {
            assert_eq!(get_bit(&b, i*256+i), 1)
        }
    }

    #[test]
    fn test_simd_transpose() {
        // test with the identity matrix.
        let mut a = [0u8; 256 * 32];
        let mut got = [0u8; 256 * 32];

        for i in 0 .. 256 {
            set_bit(&mut a,  i * 256 + i, 1);
        }
        set_bit(&mut a, 256, 1);

        simd_transpose256x256(&mut got, &a);
        assert_eq!(get_bit(&got, 0), 1);
        assert_eq!(get_bit(&got, 256 + 1), 1);
        assert_eq!(get_bit(&got, 1), 1);
        assert_eq!(get_bit(&got, 256*32 + 32), 1);
        assert!((0 .. 256).all(|i| get_bit(&got, i*256 + i) == 1));

        // test with a randomly generated matrix and compare with the naive algorithm.
        let mut expected = [0u8; 256 * 32];
        let mut prng = thread_rng();
        prng.fill_bytes(&mut a);
        simd_transpose256x256(&mut got, &a);
        transpose256x256(&mut expected, &a);
        assert!((0 .. 256*256).all(|i| get_bit(&got, i) == get_bit(&expected, i)));
    }
}
