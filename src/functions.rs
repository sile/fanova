use ordered_float::OrderedFloat;
use std::collections::HashMap;

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

pub fn most_frequent(xs: impl Iterator<Item = f64>) -> f64 {
    let mut counter = HashMap::<_, usize>::new();
    for x in xs {
        *counter.entry(OrderedFloat(x)).or_default() += 1;
    }
    counter
        .iter()
        .max_by_key(|t| t.1)
        .expect("never fails")
        .0
         .0
}
