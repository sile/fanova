use crate::partition::TreePartitions;
use crate::random_forest::{RandomForestOptions, RandomForestRegressor};
use crate::table::Table;
use ordered_float::OrderedFloat;
use rand;
use std::collections::BTreeMap;
use std::num::NonZeroUsize;
use std::ops::Range;
use thiserror::Error;

#[derive(Debug, Clone, Default)]
pub struct FanovaOptions {
    random_forest: RandomForestOptions,
    feature_space: Option<Vec<Range<f64>>>,
    // target_cutoff
    normalize_importance: bool,
    // max_subset_size: NonZeroUsize,
}

impl FanovaOptions {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn trees(mut self, trees: NonZeroUsize) -> Self {
        self.random_forest.trees = trees;
        self
    }

    pub fn max_features(mut self, max_features: usize) -> Self {
        self.random_forest.max_features = Some(max_features);
        self
    }
}

#[derive(Debug)]
pub struct Fanova<'a> {
    space: Vec<Range<f64>>,
    table: Table<'a>,
    options: FanovaOptions,
}

impl<'a> Fanova<'a> {
    pub fn new(features: Vec<&'a [f64]>, target: &'a [f64]) -> Result<Self, FanovaError> {
        Self::with_options(features, target, Default::default())
    }

    pub fn with_options(
        features: Vec<&'a [f64]>,
        target: &'a [f64],
        options: FanovaOptions,
    ) -> Result<Self, FanovaError> {
        if let Some(space) = &options.feature_space {
            if space.len() != features.len() {
                return Err(FanovaError::FeatureSpaceSizeMismatch);
            }

            for (i, (range, feature)) in space.iter().zip(features.iter()).enumerate() {
                if feature.iter().any(|&f| f < range.start || range.end <= f) {
                    return Err(FanovaError::TooNarrowFeatureSpace { feature: i });
                }
            }
        }

        todo!()
        // Ok(Self {
        //     space,
        //     table,
        //     options,
        // })
    }
}

#[non_exhaustive]
#[derive(Debug, Error, Clone)]
pub enum FanovaError {
    #[error("TODO")]
    FeatureSpaceSizeMismatch,

    #[error("TODO")]
    TooNarrowFeatureSpace { feature: usize },

    #[error("table must have at least one column and one row")]
    EmptyTable,

    #[error("some of columns have a different row count from others")]
    RowSizeMismatch,

    #[error("target column contains non finite numbers")]
    NonFiniteTarget,
}

pub fn quantify_importance(config_space: Vec<Range<f64>>, table: Table) -> Vec<f64> {
    let mut importances = vec![0.0; table.features_len()];
    let regressor = RandomForestRegressor::fit(&mut rand::thread_rng(), table, Default::default());
    for tree in regressor.forest().iter() {
        let partitioning = TreePartitions::new(tree, config_space.clone());
        let (mean, total_variance) = partitioning.mean_and_variance();
        for (i, u) in config_space.iter().enumerate() {
            let subspaces = subspaces(partitioning.partitions().map(|p| p.space[i].clone()));
            let variance = subspaces
                .map(|s| {
                    let v = partitioning.marginal_predict(&[i], &s);
                    (v - mean).powi(2) * (s.end - s.start)
                })
                .sum::<f64>();
            let v = variance / (u.end - u.start) / total_variance; // TODO: Also save standard deviation.
            importances[i] += v;
        }
    }

    importances
        .iter_mut()
        .for_each(|v| *v /= regressor.forest().len() as f64);
    let sum = importances.iter().map(|&v| v).sum::<f64>();
    importances.iter().map(|&v| v / sum).collect()
}

fn subspaces(partitions: impl Iterator<Item = Range<f64>>) -> impl Iterator<Item = Range<f64>> {
    let mut subspaces = BTreeMap::new();
    for p in partitions {
        insert_subspace(&mut subspaces, p);
    }
    subspaces.into_iter().map(|(_, v)| v)
}

fn insert_subspace(subspaces: &mut BTreeMap<OrderedFloat<f64>, Range<f64>>, mut p: Range<f64>) {
    if p.start == p.end {
        return;
    }

    if let Some(mut q) = subspaces
        .range(..=OrderedFloat(p.start))
        .rev()
        .nth(0)
        .map(|(_, q)| q.clone())
    {
        if q.start == p.start {
            if q.end > p.end {
                subspaces.remove(&OrderedFloat(q.start));

                q.start = p.end;
                subspaces.insert(OrderedFloat(p.start), p);
                subspaces.insert(OrderedFloat(q.start), q);
            } else {
                assert!(q.end <= p.end);
                p.start = q.end;
                insert_subspace(subspaces, p);
            }
        } else {
            assert!(q.start < p.start);
            if q.end > p.end {
                subspaces.remove(&OrderedFloat(q.start));

                let r = Range {
                    start: p.end,
                    end: q.end,
                };
                q.end = p.start;
                subspaces.insert(OrderedFloat(q.start), q);
                subspaces.insert(OrderedFloat(p.start), p);
                subspaces.insert(OrderedFloat(r.start), r);
            } else {
                assert!(q.end <= p.end);
                subspaces.remove(&OrderedFloat(q.start));

                let r = Range {
                    start: q.end,
                    end: p.end,
                };
                q.end = p.start;
                p.end = r.start;
                subspaces.insert(OrderedFloat(q.start), q);
                subspaces.insert(OrderedFloat(p.start), p);
                insert_subspace(subspaces, r);
            }
        }
    } else {
        subspaces.insert(OrderedFloat(p.start), p);
    }
}
