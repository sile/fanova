fanova
=======

[![fanova](https://img.shields.io/crates/v/fanova.svg)](https://crates.io/crates/fanova)
[![Documentation](https://docs.rs/fanova/badge.svg)](https://docs.rs/fanova)
[![Actions Status](https://github.com/sile/fanova/workflows/CI/badge.svg)](https://github.com/sile/fanova/actions)
[![Coverage Status](https://coveralls.io/repos/github/sile/fanova/badge.svg?branch=master)](https://coveralls.io/github/sile/fanova?branch=master)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A Rust [fANOVA] (functional analysis of variance) implementation.

fANOVA provides a way to calculate feature importance.

Examples
--------

```
use fanova::{FanovaOptions, RandomForestOptions};
use rand::{Rng, SeedableRng};

let mut feature1 = Vec::new();
let mut feature2 = Vec::new();
let mut feature3 = Vec::new();
let mut target = Vec::new();

let mut rng = rand::rngs::StdRng::from_seed([0u8; 32]);
for _ in 0..100 {
    let f1 = rng.gen();
    let f2 = rng.gen();
    let f3 = rng.gen();
    let t = f1 + f2 * 2.0 + f3 * 3.0;

    feature1.push(f1);
    feature2.push(f2);
    feature3.push(f3);
    target.push(t);
}

let mut fanova = FanovaOptions::new()
    .random_forest(RandomForestOptions::new().seed(0))
    .fit(vec![&feature1, &feature2, &feature3], &target).unwrap();
let importances = (0..3)
    .map(|i| fanova.quantify_importance(&[i]).mean)
    .collect::<Vec<_>>();

assert_eq!(
    importances,
    vec![0.02744461966313835, 0.22991883769286145, 0.6288784011550144]
);
```

References
----------

- [An Efficient Approach for Assessing Hyperparameter Importance][fANOVA]

[fANOVA]: http://proceedings.mlr.press/v32/hutter14.html
