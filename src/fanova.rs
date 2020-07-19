use crate::decision_tree::DecisionTreeRegressor;
use crate::functions;
use crate::partition::TreePartitions;
use crate::random_forest::{RandomForestOptions, RandomForestRegressor};
use crate::space::SparseFeatureSpace;
use crate::table::{Table, TableError};
use crate::FeatureSpace;
use itertools::Itertools as _;
use ordered_float::OrderedFloat;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use std::borrow::Cow;
use std::collections::BTreeMap;
use std::ops::Range;
use thiserror::Error;

#[derive(Debug, Clone)]
pub struct FanovaOptions {
    random_forest: RandomForestOptions,
    feature_space: Option<FeatureSpace>,
    parallel: bool,
    target_cutoff: Range<f64>,
}

impl FanovaOptions {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn random_forest(mut self, options: RandomForestOptions) -> Self {
        self.random_forest = options;
        self
    }

    pub fn feature_space(mut self, space: FeatureSpace) -> Self {
        self.feature_space = Some(space);
        self
    }

    pub fn parallel(mut self) -> Self {
        self.parallel = true;
        self
    }

    pub fn target_cutoff_low(mut self, v: f64) -> Self {
        self.target_cutoff.start = v;
        self
    }

    pub fn target_cutoff_high(mut self, v: f64) -> Self {
        self.target_cutoff.end = v;
        self
    }

    pub fn fit(mut self, features: Vec<&[f64]>, target: &[f64]) -> Result<Fanova, FitError> {
        let mut columns = features;
        let mut target = Cow::Borrowed(target);

        if self.target_cutoff.start != std::f64::NEG_INFINITY
            || self.target_cutoff.end != std::f64::INFINITY
        {
            target = Cow::Owned(
                target
                    .iter()
                    .map(|&v| v.max(self.target_cutoff.start).min(self.target_cutoff.end))
                    .collect::<Vec<_>>(),
            );
        }
        columns.push(&target);
        let table = Table::new(columns)?;

        let feature_space = if let Some(space) = self.feature_space.take() {
            if space.ranges().len() != table.features_len() {
                return Err(FitError::FeatureSpaceSizeMismatch);
            }

            for (i, range) in space.ranges().iter().enumerate() {
                let mut feature = table.column(i);
                if feature.any(|f| f < range.start || range.end <= f) {
                    return Err(FitError::TooNarrowFeatureSpace { feature: i });
                }
            }

            space
        } else {
            FeatureSpace::from_table(&table)
        };

        let random_forest = if self.parallel {
            RandomForestRegressor::fit_parallel(table, self.random_forest)
        } else {
            RandomForestRegressor::fit(table, self.random_forest)
        };
        Ok(Fanova {
            random_forest,
            feature_space,
            parallel: self.parallel,
        })
    }
}

impl Default for FanovaOptions {
    fn default() -> Self {
        Self {
            random_forest: RandomForestOptions::default(),
            feature_space: None,
            parallel: false,
            target_cutoff: Range {
                start: std::f64::NEG_INFINITY,
                end: std::f64::INFINITY,
            },
        }
    }
}

#[derive(Debug)]
pub struct Fanova {
    random_forest: RandomForestRegressor,
    feature_space: FeatureSpace,
    parallel: bool,
}

impl Fanova {
    pub fn fit(features: Vec<&[f64]>, target: &[f64]) -> Result<Self, FitError> {
        FanovaOptions::default().fit(features, target)
    }

    pub fn quantify_importance(&self, features: &[usize]) -> Importance {
        let importances = if self.parallel {
            self.random_forest
                .forest()
                .par_iter()
                .map(|tree| self.quantify_importance_tree(tree, features))
                .collect::<Vec<_>>()
        } else {
            self.random_forest
                .forest()
                .iter()
                .map(|tree| self.quantify_importance_tree(tree, features))
                .collect::<Vec<_>>()
        };

        let (mean, stddev) = functions::mean_and_stddev(importances.into_iter());
        Importance { mean, stddev }
    }

    pub fn feature_combinations(&self, k: usize) -> impl Iterator<Item = Vec<usize>> {
        let features = self.feature_space.ranges().len();
        (1..=k).flat_map(move |k| (0..features).combinations(k))
    }

    fn calc_effect(
        &self,
        subspace: &SparseFeatureSpace,
        partitioning: &TreePartitions,
        mean: f64,
    ) -> f64 {
        let mut v = partitioning.marginal_predict(subspace);
        for k in 1..subspace.features() {
            for subspace in subspace.iter().combinations(k).map(SparseFeatureSpace::new) {
                v -= self.calc_effect(&subspace, partitioning, mean);
            }
        }
        v - mean
    }

    fn quantify_importance_tree(&self, tree: &DecisionTreeRegressor, features: &[usize]) -> f64 {
        // TODO: out-of-range check
        let partitioning = TreePartitions::new(tree, self.feature_space.clone());
        let (mean, total_variance) = partitioning.mean_and_variance();

        let variance = features
            .iter()
            .copied()
            .map(|i| {
                subspaces(
                    partitioning
                        .partitions()
                        .map(|p| p.space.ranges()[i].clone()),
                )
                .map(|space| (i, space))
                .collect::<Vec<_>>()
            })
            .multi_cartesian_product()
            .map(SparseFeatureSpace::new)
            .map(|subspace| {
                let effect = self.calc_effect(&subspace, &partitioning, mean);
                effect.powi(2) * subspace.size() as f64
            })
            .sum::<f64>();

        let size = self.feature_space.to_sparse(features).size();
        return variance / size / total_variance;
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Importance {
    pub mean: f64,
    pub stddev: f64,
}

#[non_exhaustive]
#[derive(Debug, Error, Clone)]
pub enum FitError {
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

impl From<TableError> for FitError {
    fn from(f: TableError) -> Self {
        match f {
            TableError::EmptyTable => Self::EmptyRows,
            TableError::NonFiniteTarget => Self::NonFiniteTarget,
            TableError::RowSizeMismatch => Self::RowSizeMismatch,
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

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    #[test]
    fn quantify_importance_k1_works() -> anyhow::Result<()> {
        let mut feature1 = Vec::new();
        let mut feature2 = Vec::new();
        let mut feature3 = Vec::new();
        let mut target = Vec::new();

        let mut rng = StdRng::from_seed([0u8; 32]);
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

        let fanova = FanovaOptions::default()
            .random_forest(RandomForestOptions::default().seed(0))
            .fit(vec![&feature1, &feature2, &feature3], &target)?;
        let importances = (0..3)
            .map(|i| fanova.quantify_importance(&[i]).mean)
            .collect::<Vec<_>>();
        assert_eq!(
            importances,
            vec![0.02744461966313835, 0.2299188376928614, 0.6288784011550144]
        );

        Ok(())
    }

    #[test]
    fn quantify_importance_k2_works() -> anyhow::Result<()> {
        let mut feature1 = Vec::new();
        let mut feature2 = Vec::new();
        let mut feature3 = Vec::new();
        let mut target = Vec::new();

        let mut rng = StdRng::from_seed([0u8; 32]);
        for _ in 0..100 {
            let f1 = rng.gen();
            let f2 = rng.gen();
            let f3 = rng.gen();
            let t = f1 / 100.0 + (f2 - 0.5) * (f3 - 0.5);

            feature1.push(f1);
            feature2.push(f2);
            feature3.push(f3);
            target.push(t);
        }

        let fanova = FanovaOptions::default()
            .random_forest(RandomForestOptions::default().seed(0))
            .parallel()
            .fit(vec![&feature1, &feature2, &feature3], &target)?;
        let importances = fanova
            .feature_combinations(2)
            .map(|i| fanova.quantify_importance(&i).mean)
            .collect::<Vec<_>>();
        assert_eq!(
            importances,
            vec![
                0.08594444775918132,
                0.13762622244474054,
                0.1436884807101818,
                0.0970577365624792,
                0.0766824385057356,
                0.4098183710868284
            ]
        );

        Ok(())
    }
}
