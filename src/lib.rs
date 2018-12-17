extern crate curve25519_dalek;
extern crate rand;
extern crate digest;
extern crate generic_array;

#[cfg(test)]
extern crate sha2;

use rand::{CryptoRng, Rng};

use curve25519_dalek::scalar::Scalar;
use curve25519_dalek::constants;
use curve25519_dalek::ristretto::RistrettoPoint;
use curve25519_dalek::ristretto::CompressedRistretto;

use digest::Digest;

use generic_array::GenericArray;


pub struct Sender {
    s_bytes: [u8; 32],
    t: RistrettoPoint,
    y: Scalar,
}

impl Sender {
    pub fn new<R>(csprng: &mut R) -> (Self, [u8; 32])
        where R: CryptoRng + Rng
    {
        let y = Scalar::random(csprng);
        let s = &y * &constants::RISTRETTO_BASEPOINT_TABLE;
        let t = &y * &s;
        let s_bytes = *s.compress().as_bytes();
        let sender = Sender {s_bytes, t, y};
        (sender, s_bytes)
    }

    pub fn keys<D>(&self, r_bytes: &[u8], n: usize) -> Option<Vec<GenericArray<u8, D::OutputSize>>>
        where D: Digest + Default
    {
        let r = CompressedRistretto::from_slice(r_bytes).decompress()?;

        let mut secrets = Vec::new();
        let mut secret = self.y * r;
        for _ in 1 .. n {
            secrets.push(secret.compress());
            secret -= self.t;
        }
        secrets.push(secret.compress());

        let mut hash = D::default();
        let mut keys = Vec::new();
        for secret in secrets.iter() {
            hash.input(self.s_bytes);
            hash.input(r_bytes);
            hash.input(secret.as_bytes());
            keys.push(hash.result_reset());
        }
        Some(keys)
    }
}


pub struct Receiver {
    x: Scalar,
    s: RistrettoPoint,
    s_bytes: [u8; 32],
    r_bytes: [u8; 32],
}

impl Receiver {
    pub fn new<R>(csprng: &mut R, choice: usize, s_bytes: &[u8; 32]) -> Option<(Self, [u8; 32])>
        where R: Rng + CryptoRng
    {
        let s = CompressedRistretto::from_slice(s_bytes).decompress()?;
        let choice = Scalar::from(choice as u64);

        let x = Scalar::random(csprng);
        let r = &choice * &s + &x * &constants::RISTRETTO_BASEPOINT_TABLE;;

        let rc = r.compress();
        let r_bytes = rc.as_bytes();
        let receiver = Receiver {x: x,
                                 s: s,
                                 s_bytes: *s_bytes,
                                 r_bytes: *r_bytes};
        Some((receiver, *r_bytes))
    }

    pub fn key<D>(self) -> GenericArray<u8, D::OutputSize>
        where D: Digest + Default
    {
        let mut hash = D::default();
        hash.input(self.s_bytes);
        hash.input(self.r_bytes);
        hash.input((&self.x * &self.s).compress().as_bytes());
        hash.result()
    }

}


#[cfg(test)]
mod tests {
    use super::{Sender, Receiver};
    use rand::rngs::OsRng;
    use sha2::Sha256;

    #[test]
    fn test_ot() {
        let n = 10;
        let mut csprng: OsRng = OsRng::new().unwrap();
        let (sender, s) = Sender::new(&mut csprng);

        for choice in 0 .. n {
            let (receiver, r) = Receiver::new(&mut csprng, choice, &s).unwrap();

            let secret_s = sender.keys::<Sha256>(&r, n).unwrap();
            let secret_r = receiver.key::<Sha256>();

            assert_eq!(secret_r, secret_s[choice as usize]);
        }
    }
}
