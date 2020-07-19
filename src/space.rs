use crate::table::Table;
use ordered_float::OrderedFloat;
use std::ops::Range;

#[derive(Debug, Clone)]
pub struct FeatureSpace(Vec<Range<f64>>);

impl FeatureSpace {
    pub(crate) fn from_table(table: &Table) -> Self {
        let ranges = (0..table.features_len())
            .map(|i| {
                let start = table
                    .column(i)
                    .min_by_key(|&v| OrderedFloat(v))
                    .expect("never fails");
                let end = table
                    .column(i)
                    .max_by_key(|&v| OrderedFloat(v))
                    .expect("never fails");
                Range { start, end }
            })
            .collect();
        Self(ranges)
    }

    pub fn ranges(&self) -> &[Range<f64>] {
        &self.0
    }

    pub(crate) fn split(&self, feature_index: usize, split_point: f64) -> (Self, Self) {
        debug_assert!(feature_index < self.0.len());
        debug_assert!(self.0[feature_index].start <= split_point);
        debug_assert!(split_point <= self.0[feature_index].end);

        let mut lower = self.clone();
        let mut upper = self.clone();
        lower.0[feature_index].end = split_point;
        upper.0[feature_index].start = split_point;
        (lower, upper)
    }

    pub(crate) fn covers(&self, space: &SparseFeatureSpace) -> bool {
        space
            .iter()
            .all(|(i, r)| self.0[i].start <= r.start && r.end <= self.0[i].end)
    }

    pub(crate) fn to_sparse(&self, features: &[usize]) -> SparseFeatureSpace {
        SparseFeatureSpace(
            features
                .iter()
                .copied()
                .map(|i| (i, self.0[i].clone()))
                .collect(),
        )
    }
}

#[derive(Debug, Clone)]
pub struct SparseFeatureSpace(Vec<(usize, Range<f64>)>);

impl SparseFeatureSpace {
    pub fn new(space: Vec<(usize, Range<f64>)>) -> Self {
        Self(space)
    }

    pub fn size(&self) -> f64 {
        self.0.iter().map(|(_, r)| r.end - r.start).product()
    }

    pub fn iter<'a>(&'a self) -> impl 'a + Iterator<Item = (usize, Range<f64>)> {
        self.0.iter().map(|x| (x.0, x.1.clone()))
    }
}
