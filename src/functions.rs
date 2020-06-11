pub fn mean(xs: impl Iterator<Item = f64>) -> f64 {
    let mut count = 0;
    let mut total = 0.0;
    for x in xs {
        count += 1;
        total += x;
    }
    assert_ne!(count, 0);
    total / count as f64
}

pub fn mean_and_stddev(xs: impl Iterator<Item = f64> + Clone) -> (f64, f64) {
    let m = mean(xs.clone());
    let s = xs.map(|x| (x - m).powi(2)).sum::<f64>().sqrt();
    (m, s)
}

pub fn mse(xs: impl Iterator<Item = f64> + Clone) -> f64 {
    let n = xs.clone().count() as f64;
    let m = mean(xs.clone());
    xs.map(|x| (x - m).powi(2)).sum::<f64>() / n
}
