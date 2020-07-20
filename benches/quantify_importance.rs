use criterion::{criterion_group, criterion_main, Criterion};
use fanova::Fanova;
use rand;

fn k1(c: &mut Criterion) {
    let mut feature1 = Vec::new();
    let mut feature2 = Vec::new();
    let mut feature3 = Vec::new();
    let mut target = Vec::new();

    for _ in 0..100 {
        let f1 = rand::random();
        let f2 = rand::random();
        let f3 = rand::random();
        let t = f1 / 100.0 + (f2 - 0.5) * (f3 - 0.5);

        feature1.push(f1);
        feature2.push(f2);
        feature3.push(f3);
        target.push(t);
    }

    let mut fanova = Fanova::fit(vec![&feature1, &feature2, &feature3], &target).unwrap();

    c.bench_function("k=1, features=3, n=100", |b| {
        b.iter(|| {
            for i in 0..3 {
                fanova.clear();
                fanova.quantify_importance(&[i]);
            }
        })
    });
}

fn k2(c: &mut Criterion) {
    let mut feature1 = Vec::new();
    let mut feature2 = Vec::new();
    let mut feature3 = Vec::new();
    let mut target = Vec::new();

    for _ in 0..100 {
        let f1 = rand::random();
        let f2 = rand::random();
        let f3 = rand::random();
        let t = f1 / 100.0 + (f2 - 0.5) * (f3 - 0.5);

        feature1.push(f1);
        feature2.push(f2);
        feature3.push(f3);
        target.push(t);
    }

    let mut fanova = Fanova::fit(vec![&feature1, &feature2, &feature3], &target).unwrap();

    c.bench_function("k=1, features=3, n=100", |b| {
        b.iter(|| {
            fanova.clear();
            for i in &[
                &[0][..],
                &[1][..],
                &[2][..],
                &[0, 1][..],
                &[0, 2][..],
                &[2, 3][..],
            ] {
                fanova.quantify_importance(i);
            }
        })
    });
}

criterion_group!(benches, k1, k2);
criterion_main!(benches);
