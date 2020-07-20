use crate::table::Table;
use ordered_float::OrderedFloat;
use std::ops::Range;

#[derive(Debug, Clone)]
pub struct FeatureSpace(Vec<Range<f64>>);

impl FeatureSpace {
    pub fn from_table(table: &Table) -> Self {
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

    pub fn split(&self, feature_index: usize, split_point: f64) -> (Self, Self) {
        debug_assert!(feature_index < self.0.len());
        debug_assert!(self.0[feature_index].start <= split_point);
        debug_assert!(split_point <= self.0[feature_index].end);

        let mut lower = self.clone();
        let mut upper = self.clone();
        lower.0[feature_index].end = split_point;
        upper.0[feature_index].start = split_point;
        (lower, upper)
    }

    pub fn marginal_size(&self, fixed: &[usize]) -> f64 {
        self.0
            .iter()
            .enumerate()
            .filter(|(i, _)| !fixed.contains(i))
            .map(|(_, r)| r.end - r.start)
            .product()
    }

    pub fn partial_size(&self, features: &[usize]) -> f64 {
        features
            .iter()
            .map(|&i| self.0[i].end - self.0[i].start)
            .product()
    }

    pub fn size(&self) -> f64 {
        self.0.iter().map(|r| r.end - r.start).product()
    }
}
