/// This implementation is half stolen from the linux kernel.
use byteorder::{NativeEndian, ByteOrder};

const PRIMITIVE_POLYNOMIALS : [u32; 11] = [
    0x25, 0x43, 0x83, 0x11d, 0x211, 0x409, 0x805, 0x1053, 0x201b, 0x402b, 0x8003,
];
const MIN_M: usize = 5;
const MAX_M: usize = 15;

macro_rules! gf_size {
    ($x:expr) => {(1<<$x) - 1}
}

macro_rules! rem2 {
    ($x: expr, $n: expr) => { if $x >= $n { $x - $n } else { $x } }
}

macro_rules! ceil {
    ($x: expr, $n: expr) => {($x + ($n - 1)) / $n}
}

// The m-th extension of GF(2) with n elements generated by f.
struct GF2m {
    pow: Vec<u32>,
    log: Vec<usize>,
    f: u32,
    m: usize,
    n: usize,
}

impl GF2m {
    fn new(m: usize) -> GF2m {
        if m < MIN_M || m > MAX_M {
            panic!("Unsupported extension field.")
        }

        // the primitive polynomial
        let f = PRIMITIVE_POLYNOMIALS[m - MIN_M];
        let k = 1 << m;
        let n = (1<<m) - 1;
        // the power tables
        let mut pow = vec![1u32; k as usize];
        let mut log = vec![0usize; k as usize];

        let mut x = 1u32;
        for i in 0 .. n {
            pow[i] = x;
            log[x as usize] = i;
            assert!(i == 0 || x != 1);
            x <<= 1;
            if x & k != 0 { x ^= f }
        }
        GF2m {pow, log, f, m, n}
    }

    fn mul(&self, a: u32, b: u32) -> u32 {
        if a == 0 || b == 0 {
            0
        } else {
            let i = self.log[a as usize] + self.log[b as usize];
            self.pow[rem2!(i, self.n)]
        }
    }
}

struct Polynomial {
    deg: usize,
    c: Vec<u32>,
}

pub struct BCH {
    ff: GF2m,
    g: Polynomial,
    reminders: Vec<u32>,
    t: usize,
}

impl BCH {
    pub fn new(m: usize, t: usize) -> Self {
        let ff = GF2m::new(m);
        let g = Self::generator_polynomial(&ff, t);
        let reminders = Self::rem8_tables(&ff, &g);
        BCH {ff, g, reminders, t}
    }

    fn generator_polynomial(ff: &GF2m, t: usize) -> Polynomial {
        // XXXX why on the linux kernel this is n+1?
        let mut roots = vec![false; ff.n];

        // the complete defining set of the code
        for i in 0 .. t {
            let mut r = 2*i + 1;
            for _ in 0 .. ff.m {
                roots[r] = true;
                r = rem2!(r*2, ff.n);
            }
        }

        // g = \prod_r (x-r) has at most t roots with all conjugates.
        let mut g = vec![0; ff.m*t+1];
        let mut deg = 0;
        g[0] = 1;
        for i in (0 .. ff.n).filter(|&i| roots[i]) {
            let r = ff.pow[i];
            g[deg+1] = 1;
            for j in (1 .. deg+1).rev() {
                g[j] = ff.mul(g[j], r)^g[j-1];
            }
            g[0] = ff.mul(g[0], r);
            deg += 1;
        }
        let mut c = vec![0u32; ceil!(deg, 32)];
        // now the polynomial has coefficients in GF2.
        // Compress it.
        for (i, chunk) in g.chunks(32).enumerate() {
            let mut word = 0;
            for (j, &x) in chunk.iter().rev().enumerate() {
                if x != 0 { word |= 1u32 << j }
            }
            c[i] = word;
        }
        Polynomial {c, deg}
    }

    fn deg(x: u32) -> usize {
        match x {
            0 | 1 => 0,
            _ => Self::deg(x >> 1) + 1
        }
    }

    fn rem8_tables(ff: &GF2m, g: &Polynomial) -> Vec<u32> {
        let ecclen = ceil!(g.deg, 32);
        let plen = ceil!(g.deg+1, 32);
        let l = ceil!(ff.n, 32);

        let mut rem8 = vec![0u32; 4*256*l];
        // for every polynomial p of max degree 7
        for p in 0 .. 256 {
            // in blocks of 32 bits
            for b in 0 .. 4 {
                let mut rem8_pb = &mut rem8[(b*256 + 1) * l as usize ..];
                // q = p x^(8b)
                let mut q = p << (b*8);
                while q != 0 {
                    let d = Self::deg(q);
                    q ^= g.c[0] >> (31-d);
                    for j in 0 .. ecclen {
			let hi = if d < 31 {g.c[j] << (d+1)} else {0};
			let lo = if j+1 < plen { g.c[j+1] >> (31-d)} else {0};
			rem8_pb[j] ^= hi|lo;
                    }
                }
            }
        }
        rem8
    }

    pub fn encode(&self, w: &[u8], len: usize, dst: &mut [u8]) {
        let rem0 = &self.reminders[ .. ];
        let rem1 = &rem0[256*(self.ff.n+1) ..];
        let rem2 = &rem1[256*(self.ff.n+1) ..];
        let rem3 = &rem2[256*(self.ff.n+1) ..];
        let l = ceil!(self.ff.n-1, 32);

        for i in (0 .. len).step_by(4) {
            let p0 = &rem0[l*w[i] as usize ..];
            let p1 = &rem1[l*w[i+1] as usize ..];
            let p2 = &rem2[l*w[i+2] as usize ..];
            let p3 = &rem3[l*w[i+3] as usize ..];

            for j in 0 .. l-1 {
                let b =
                    NativeEndian::read_u32(&dst[(j+1)*4 .. (j+2)*4]) ^
                    p0[j] ^ p1[j] ^ p2[j] ^ p3[j];
                NativeEndian::write_u32_into(&[b], &mut dst[j .. j+4])
            }
            let b = p0[l-1] ^ p1[l-1] ^ p2[l-1] ^ p3[l-1];
            NativeEndian::write_u32_into(&[b], &mut dst[l-1 .. l-1+4])
        }
    }
}

#[test]
fn test_ff() {
    let ff = GF2m::new(5);
    assert_eq!(ff.m, 5);

    assert_eq!(ff.pow[0], 1);
    assert_eq!(ff.pow[ff.n], 1);

    assert_eq!(ff.mul(ff.pow[4], ff.pow[1]), ff.pow[5]);
    assert_eq!(ff.mul(ff.pow[1], ff.pow[3]), ff.pow[4]);
    assert_eq!(ff.mul(ff.pow[1], ff.pow[ff.n-1]), 1)
}


#[test]
fn test_bch() {
    // there is a BCH code [m=6, n=63, k=24, t=7]
    let code = BCH::new(6, 7);
    assert_eq!(code.ff.n - code.g.deg, 24);

}
