#[derive(Debug)]
pub struct Table<'a> {
    row_indices: Vec<usize>,
    features: Vec<&'a [f64]>,
    target: &'a [f64],
}
