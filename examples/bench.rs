//! Time fANOVA importance computation without pulling in criterion.
//! Run with `cargo run --release --example bench`.

use fanova::Fanova;
use std::time::Instant;

fn make_fanova() -> Fanova {
    let mut feature1 = Vec::new();
    let mut feature2 = Vec::new();
    let mut feature3 = Vec::new();
    let mut target = Vec::new();
    for _ in 0..100 {
        let f1: f64 = rand::random();
        let f2: f64 = rand::random();
        let f3: f64 = rand::random();
        let t = f1 / 100.0 + (f2 - 0.5) * (f3 - 0.5);
        feature1.push(f1);
        feature2.push(f2);
        feature3.push(f3);
        target.push(t);
    }
    Fanova::fit(vec![&feature1, &feature2, &feature3], &target).unwrap()
}

fn measure<F: FnMut()>(label: &str, samples: usize, mut f: F) {
    for _ in 0..3 {
        f();
    }
    let mut times = Vec::with_capacity(samples);
    for _ in 0..samples {
        let t = Instant::now();
        f();
        times.push(t.elapsed().as_nanos());
    }
    times.sort_unstable();
    let min = times[0];
    let median = times[times.len() / 2];
    let max = *times.last().unwrap();
    println!("{label:<30}  min = {min:>10} ns   median = {median:>10} ns   max = {max:>10} ns");
}

fn main() {
    let mut fanova = make_fanova();
    measure("k=1, features=3, n=100", 50, || {
        for i in 0..3 {
            fanova.clear();
            fanova.quantify_importance(&[i]);
        }
    });

    let mut fanova = make_fanova();
    measure("k=1+2, features=3, n=100", 30, || {
        fanova.clear();
        for i in [&[0][..], &[1], &[2], &[0, 1], &[0, 2], &[2, 3]] {
            fanova.quantify_importance(i);
        }
    });
}
