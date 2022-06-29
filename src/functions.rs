pub fn mean(xs: impl Iterator<Item = f64>) -> f64 {
    mean_and_stddev(xs).0
}

pub fn mean_and_stddev(xs: impl Iterator<Item = f64>) -> (f64, f64) {
    let (mut mean, mut s, mut n) = (0.0, 0.0, 0.0);
    for x in xs {
        n += 1.0;
        let delta = x - mean;
        mean += delta / n;
        s += delta * (x - mean);
    }
    assert!(n >= 0.0, "Need at least one value");
    if n == 1.0 {
        (mean, 0.0)
    } else {
        (mean, (s / (n - 1.0)).sqrt())
    }
}

pub fn mse(xs: impl Iterator<Item = f64> + Clone) -> f64 {
    let n = xs.clone().count() as f64;
    let m = mean(xs.clone());
    xs.map(|x| (x - m).powi(2)).sum::<f64>() / n
}
