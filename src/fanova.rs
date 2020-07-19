use crate::decision_tree::DecisionTreeRegressor;
use crate::functions;
use crate::partition::TreePartitions;
use crate::random_forest::{RandomForestOptions, RandomForestRegressor};
use crate::space::{FeatureSpace, SparseFeatureSpace};
use crate::table::{Table, TableError};
use itertools::Itertools as _;
use ordered_float::OrderedFloat;
use rayon::iter::{IntoParallelIterator, IntoParallelRefMutIterator, ParallelIterator};
use std::collections::BTreeMap;
use std::ops::Range;
use thiserror::Error;

#[derive(Debug, Clone)]
pub struct FanovaOptions {
    random_forest: RandomForestOptions,
    parallel: bool,
}

impl FanovaOptions {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn random_forest(mut self, options: RandomForestOptions) -> Self {
        self.random_forest = options;
        self
    }

    pub fn parallel(mut self) -> Self {
        self.parallel = true;
        self
    }

    pub fn fit(self, features: Vec<&[f64]>, target: &[f64]) -> Result<Fanova, FitError> {
        let mut columns = features;
        columns.push(target);
        let table = Table::new(columns)?;

        let feature_space = FeatureSpace::from_table(&table);

        let trees = if self.parallel {
            RandomForestRegressor::fit_parallel(table, self.random_forest)
                .into_trees()
                .into_par_iter()
                .map(|tree| Tree::new(tree, feature_space.clone()))
                .collect()
        } else {
            RandomForestRegressor::fit(table, self.random_forest)
                .into_trees()
                .into_iter()
                .map(|tree| Tree::new(tree, feature_space.clone()))
                .collect()
        };

        Ok(Fanova {
            feature_space,
            parallel: self.parallel,
            trees,
        })
    }
}

impl Default for FanovaOptions {
    fn default() -> Self {
        Self {
            random_forest: RandomForestOptions::default(),
            parallel: false,
        }
    }
}

#[derive(Debug)]
struct Tree {
    partitions: TreePartitions,
    mean: f64,
    variance: f64,
    importances: BTreeMap<Vec<usize>, f64>,
}

impl Tree {
    fn new(regressor: DecisionTreeRegressor, feature_space: FeatureSpace) -> Self {
        let partitions = TreePartitions::new(&regressor, feature_space);
        let (mean, variance) = partitions.mean_and_variance();
        Self {
            partitions,
            mean,
            variance,
            importances: BTreeMap::new(),
        }
    }
}

#[derive(Debug)]
pub struct Fanova {
    trees: Vec<Tree>,
    feature_space: FeatureSpace,
    parallel: bool,
}

impl Fanova {
    pub fn fit(features: Vec<&[f64]>, target: &[f64]) -> Result<Self, FitError> {
        FanovaOptions::default().fit(features, target)
    }

    pub fn quantify_importance(&mut self, features: &[usize]) -> Importance {
        if features
            .iter()
            .any(|&f| f >= self.feature_space.ranges().len())
        {
            return Importance {
                mean: 0.0,
                stddev: 0.0,
            };
        }

        let mut trees = std::mem::replace(&mut self.trees, Vec::new());
        let importances = if self.parallel {
            trees
                .par_iter_mut()
                .map(|tree| self.quantify_importance_tree(tree, features))
                .collect::<Vec<_>>()
        } else {
            trees
                .iter_mut()
                .map(|tree| self.quantify_importance_tree(tree, features))
                .collect::<Vec<_>>()
        };
        self.trees = trees;

        let (mean, stddev) = functions::mean_and_stddev(importances.into_iter());
        Importance { mean, stddev }
    }

    pub fn feature_combinations(&self, k: usize) -> impl Iterator<Item = Vec<usize>> {
        let features = self.feature_space.ranges().len();
        (1..=k).flat_map(move |k| (0..features).combinations(k))
    }

    fn quantify_importance_tree(&self, tree: &mut Tree, features: &[usize]) -> f64 {
        if let Some(&importance) = tree.importances.get(features) {
            return importance;
        }

        let variance = features
            .iter()
            .copied()
            .map(|i| {
                subspaces(
                    tree.partitions
                        .partitions()
                        .map(|p| p.space.ranges()[i].clone()),
                )
                .map(|space| (i, space))
                .collect::<Vec<_>>()
            })
            .multi_cartesian_product()
            .map(SparseFeatureSpace::new)
            .map(|subspace| {
                let variance = tree.partitions.marginal_predict(&subspace) - tree.mean;
                let weight = subspace.size() as f64;
                variance.powi(2) * weight
            })
            .sum::<f64>();

        let size = self.feature_space.to_sparse(features).size();
        let mut importance = variance / size / tree.variance;
        for k in 1..features.len() {
            for sub_features in features.iter().copied().combinations(k) {
                importance -= self.quantify_importance_tree(tree, &sub_features);
            }
        }

        tree.importances.insert(features.to_owned(), importance);
        importance
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

        let mut fanova = FanovaOptions::default()
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

        let mut fanova = FanovaOptions::default()
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
                0.09705773656247917,
                0.07668243850573553,
                0.40981837108682856
            ]
        );

        Ok(())
    }
}
