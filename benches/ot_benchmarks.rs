#[macro_use]
extern crate criterion;
extern crate rand;
extern crate sha2;

extern crate ot_dalek;

use criterion::Criterion;
use rand::rngs::OsRng;
use ot_dalek::{Sender, Receiver};
use sha2::Sha256;

fn bench_sender_new(c: &mut Criterion) {
    c.bench_function("sender setup", move |b| {
        let mut csprng = OsRng::new().unwrap();
        b.iter(|| {
            Sender::new(&mut csprng);
        })
    });
}

fn bench_sender_keys(c: &mut Criterion) {
    c.bench_function("sender keys", move |b| {
        let mut csprng = OsRng::new().unwrap();
        let (sender, s) = Sender::new(&mut csprng);
        let (_receiver, r) = Receiver::new(&mut csprng, 1, &s).unwrap();

        b.iter(|| {
            let _keys = sender.keys::<Sha256>(&r, 2);
        })
    });
}

fn bench_receiver(c: &mut Criterion) {
    c.bench_function("receiver", move |b| {
        let mut csprng = OsRng::new().unwrap();
        let (_sender, s) = Sender::new(&mut csprng);

        b.iter(|| {
            let (receiver, _r) = Receiver::new(&mut csprng, 1, &s).unwrap();
            receiver.key::<Sha256>();
        })
    });
}

criterion_group!{
    name = benches;
    config = Criterion::default();
    targets = bench_sender_new, bench_sender_keys, bench_receiver
}
criterion_main!(benches);
