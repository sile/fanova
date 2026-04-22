//! Time fANOVA importance computation without pulling in criterion.
//! Run with `cargo run --release --example bench`.

use fanova::Fanova;
use std::time::Instant;

fn make_fanova(n_features: usize, n_rows: usize) -> Fanova {
    let mut features: Vec<Vec<f64>> = (0..n_features).map(|_| Vec::with_capacity(n_rows)).collect();
    let mut target = Vec::with_capacity(n_rows);
    for _ in 0..n_rows {
        let mut t = 0.0;
        for (i, f) in features.iter_mut().enumerate() {
            let v: f64 = rand::random();
            f.push(v);
            t += v * (i as f64 + 1.0) / (n_features as f64);
        }
        target.push(t);
    }
    let cols: Vec<&[f64]> = features.iter().map(|f| f.as_slice()).collect();
    Fanova::fit(cols, &target).unwrap()
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
    println!("{label:<32}  min = {min:>10} ns   median = {median:>10} ns   max = {max:>10} ns");
}

fn main() {
    let mut fanova = make_fanova(3, 100);
    measure("k=1, features=3, n=100", 50, || {
        for i in 0..3 {
            fanova.clear();
            fanova.quantify_importance(&[i]);
        }
    });

    let mut fanova = make_fanova(3, 100);
    measure("k=1+2, features=3, n=100", 30, || {
        fanova.clear();
        for i in [&[0][..], &[1], &[2], &[0, 1], &[0, 2], &[2, 3]] {
            fanova.quantify_importance(i);
        }
    });

    let mut fanova = make_fanova(10, 1000);
    measure("k=1, features=10, n=1000", 20, || {
        for i in 0..10 {
            fanova.clear();
            fanova.quantify_importance(&[i]);
        }
    });

    let mut fanova = make_fanova(10, 1000);
    measure("k=1+2, features=10, n=1000", 10, || {
        fanova.clear();
        for i in 0..10 {
            fanova.quantify_importance(&[i]);
        }
        for i in 0..10 {
            for j in (i + 1)..10 {
                fanova.quantify_importance(&[i, j]);
            }
        }
    });
}
