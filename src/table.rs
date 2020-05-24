use ordered_float::OrderedFloat;
use std::ops::Range;
use thiserror::Error;

#[derive(Debug)]
pub struct Table<'a> {
    row_indices: Vec<usize>,
    row_range: Range<usize>,
    features: Vec<&'a [f64]>,
    target: &'a [f64],
}

impl<'a> Table<'a> {
    pub fn new(features: Vec<&'a [f64]>, target: &'a [f64]) -> Result<Self, TableError> {
        if features.is_empty() {
            return Err(TableError::EmptyFeature);
        }
        if features
            .iter()
            .skip(1)
            .any(|f| f.len() != features[0].len())
        {
            return Err(TableError::FeatureSizeMismatch);
        }
        if features[0].len() != target.len() {
            return Err(TableError::TargetSizeMismatch);
        }
        if target.iter().any(|t| t.is_nan()) {
            return Err(TableError::NanTarget);
        }

        Ok(Self {
            row_indices: (0..target.len()).collect(),
            row_range: Range {
                start: 0,
                end: target.len(),
            },
            features,
            target,
        })
    }

    pub fn is_single_target(&self) -> bool {
        self.target.iter().skip(1).all(|&t| t == self.target[0])
    }

    pub fn target<'b>(&'b self) -> impl 'b + Iterator<Item = f64> + Clone {
        self.row_indices[self.row_range.start..self.row_range.end]
            .iter()
            .map(move |&i| self.target[i])
    }

    pub fn features(&self) -> &[&'a [f64]] {
        &self.features
    }

    pub fn sort_rows_by_feature(&mut self, column: usize) {
        let features = &self.features;
        (&mut self.row_indices[self.row_range.start..self.row_range.end])
            .sort_by_key(|&x| OrderedFloat(features[column][x]))
    }

    pub fn thresholds<'b>(&'b self, column: usize) -> impl 'b + Iterator<Item = (usize, f64)> {
        let feature = self.features[column];
        let indices = &self.row_indices;
        (self.row_range.start..self.row_range.end)
            .map(move |i| feature[indices[i]])
            .enumerate()
            .scan(None, |prev, (i, x)| {
                if *prev != Some(x) {
                    let y = prev.expect("never fails");
                    *prev = Some(x);
                    Some((i, (x + y) / 2.0))
                } else {
                    None
                }
            })
    }

    pub fn with_split<F, T>(&mut self, row: usize, mut f: F) -> (T, T)
    where
        F: FnMut(&mut Self) -> T,
    {
        let original = self.row_range.clone();

        self.row_range.end = row;
        let left = f(self);
        self.row_range.end = original.end;

        self.row_range.start = row;
        let right = f(self);
        self.row_range.start = original.start;

        (left, right)
    }
}

#[derive(Debug, Error, Clone)]
pub enum TableError {
    #[error("a table must have at least one feature")]
    EmptyFeature,

    #[error("some of features have a different row count from others")]
    FeatureSizeMismatch,

    #[error("target row count is different from feature row count")]
    TargetSizeMismatch,

    #[error("target must not contain NaN values")]
    NanTarget,
}
