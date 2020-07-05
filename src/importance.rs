use crate::decision_tree::DecisionTreeRegressor;
use crate::functions;
use crate::partition::TreePartitions;
use crate::random_forest::{RandomForestOptions, RandomForestRegressor};
use crate::table::{Table, TableError};
use itertools::Itertools as _;
use ordered_float::OrderedFloat;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::num::NonZeroUsize;
use std::ops::Range;
use thiserror::Error;

#[derive(Debug, Clone)]
pub struct FanovaOptions {
    random_forest: RandomForestOptions,
    feature_space: Option<Vec<Range<f64>>>,
    max_importance_dimension: NonZeroUsize, // target_cutoff
                                            // normalize_importance: bool
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

    pub fn feature_space(mut self, space: Vec<Range<f64>>) -> Self {
        self.feature_space = Some(space);
        self
    }

    pub fn max_importance_dimension(mut self, max: NonZeroUsize) -> Self {
        self.max_importance_dimension = max;
        self
    }

    pub fn seed(mut self, seed: u64) -> Self {
        self.random_forest.seed = Some(seed);
        self
    }
}

impl Default for FanovaOptions {
    fn default() -> Self {
        Self {
            random_forest: RandomForestOptions::default(),
            feature_space: None,
            max_importance_dimension: NonZeroUsize::new(1).expect("never fails"),
        }
    }
}

#[derive(Debug)]
pub struct Fanova<'a> {
    space: Vec<Range<f64>>,
    table: Option<Table<'a>>,
    options: FanovaOptions,
}

impl<'a> Fanova<'a> {
    pub fn new(features: Vec<&'a [f64]>, target: &'a [f64]) -> Result<Self, FanovaError> {
        Self::with_options(features, target, Default::default())
    }

    pub fn with_options(
        features: Vec<&'a [f64]>,
        target: &'a [f64],
        mut options: FanovaOptions,
    ) -> Result<Self, FanovaError> {
        let mut columns = features;
        columns.push(target);
        let table = Table::new(columns)?;

        let space = if let Some(space) = options.feature_space.take() {
            if space.len() != table.features_len() {
                return Err(FanovaError::FeatureSpaceSizeMismatch);
            }

            for (i, range) in space.iter().enumerate() {
                let mut feature = table.column(i);
                if feature.any(|f| f < range.start || range.end <= f) {
                    return Err(FanovaError::TooNarrowFeatureSpace { feature: i });
                }
            }

            space
        } else {
            (0..table.features_len())
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
                .collect()
        };

        Ok(Self {
            space,
            table: Some(table),
            options,
        })
    }

    // TODO: quantify_importance_parallel

    fn calc_effect(
        &self,
        subspace: &[(usize, Range<f64>)],
        partitioning: &TreePartitions,
        mean: f64,
    ) -> f64 {
        let mut v = partitioning.marginal_predict(subspace);
        for k in 1..subspace.len() {
            for subspace in subspace.iter().cloned().combinations(k) {
                v -= self.calc_effect(&subspace, partitioning, mean);
            }
        }
        v - mean
    }

    fn quantify_importance_tree(
        &self,
        tree: &DecisionTreeRegressor,
    ) -> HashMap<BTreeSet<usize>, f64> {
        let space = self.space.clone();
        let mut importances = HashMap::<BTreeSet<usize>, f64>::new();
        let partitioning = TreePartitions::new(tree, space.clone());
        let (mean, total_variance) = partitioning.mean_and_variance();

        for k in 1..self.options.max_importance_dimension.get() {
            for columns in (0..space.len()).combinations(k) {
                let variance = columns
                    .iter()
                    .copied()
                    .map(|i| {
                        subspaces(partitioning.partitions().map(|p| p.space[i].clone()))
                            .map(|space| (i, space))
                            .collect::<Vec<_>>()
                    })
                    .multi_cartesian_product()
                    .map(|subspace| {
                        let effect = self.calc_effect(&subspace, &partitioning, mean);
                        let size = subspace.iter().map(|(_, s)| s.end - s.start).sum::<f64>();
                        effect.powi(2) * size
                    })
                    .sum::<f64>();

                let size = columns
                    .iter()
                    .map(|&i| &space[i])
                    .map(|s| s.end - s.start)
                    .sum::<f64>();
                let v = variance / size / total_variance;

                let old = importances.insert(columns.into_iter().collect(), v);
                assert!(old.is_none());
            }
        }

        importances
    }

    pub fn quantify_importance(mut self) -> Vec<Importance> {
        let table = self.table.take().expect("never fails");

        let mut importances = HashMap::<BTreeSet<usize>, Vec<f64>>::new();
        let regressor = RandomForestRegressor::fit(table, Default::default());
        for tree in regressor.forest().iter() {
            for (k, v) in self.quantify_importance_tree(tree) {
                importances.entry(k).or_default().push(v);
            }
        }

        importances
            .into_iter()
            .map(|(features, vs)| {
                let (mean, stddev) = functions::mean_and_stddev(vs.into_iter());
                Importance {
                    features,
                    mean,
                    stddev,
                }
            })
            .collect()
    }
}

#[derive(Debug, Clone)]
pub struct Importance {
    pub features: BTreeSet<usize>,
    pub mean: f64,
    pub stddev: f64,
}

#[non_exhaustive]
#[derive(Debug, Error, Clone)]
pub enum FanovaError {
    #[error("TODO")]
    FeatureSpaceSizeMismatch,

    #[error("TODO")]
    TooNarrowFeatureSpace { feature: usize },

    #[error("features and target must have one or more rows")]
    EmptyRows,

    #[error("some of features or target have a different row count from others")]
    RowSizeMismatch,

    #[error("target contains non finite numbers")]
    NonFiniteTarget,
}

impl From<TableError> for FanovaError {
    fn from(f: TableError) -> Self {
        match f {
            TableError::EmptyTable => FanovaError::EmptyRows,
            TableError::NonFiniteTarget => FanovaError::NonFiniteTarget,
            TableError::RowSizeMismatch => FanovaError::RowSizeMismatch,
        }
    }
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
