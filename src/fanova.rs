use crate::decision_tree::DecisionTreeRegressor;
use crate::functions;
use crate::partition::TreePartitions;
use crate::random_forest::{RandomForestOptions, RandomForestRegressor};
use crate::space::FeatureSpace;
use crate::table::{Table, TableError};
use itertools::Itertools as _;
use rayon::iter::{IntoParallelIterator, IntoParallelRefMutIterator, ParallelIterator};
use std::cmp::Ordering;
use std::collections::BTreeMap;
use std::ops::Range;

#[derive(Debug, Clone, Copy)]
struct TotalF64(f64);

impl PartialEq for TotalF64 {
    fn eq(&self, other: &Self) -> bool {
        self.0.total_cmp(&other.0).is_eq()
    }
}

impl Eq for TotalF64 {}

impl PartialOrd for TotalF64 {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for TotalF64 {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.total_cmp(&other.0)
    }
}

/// fANOVA options.
#[derive(Debug, Clone, Default)]
pub struct FanovaOptions {
    random_forest: RandomForestOptions,
    parallel: bool,
}

impl FanovaOptions {
    /// Make `FanovaOptions` with the default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the `RandomForestOptions`.
    ///
    /// The default value is `RandomForestOptions::default()`.
    pub fn random_forest(mut self, options: RandomForestOptions) -> Self {
        self.random_forest = options;
        self
    }

    /// Enables parallel executions of `Fanova::{fit, quantify_importance}`.
    ///
    /// This library use `rayon` for parallel execution.
    /// Please see [the rayon document](https://docs.rs/rayon) if you want to configure the behavior
    /// (e.g., the number of worker threads).
    pub fn parallel(mut self) -> Self {
        self.parallel = true;
        self
    }

    /// Builds an fANOVA model for the given features and target.
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

    fn clear(&mut self) {
        self.importances.clear();
    }
}

/// fANOVA object.
#[derive(Debug)]
pub struct Fanova {
    trees: Vec<Tree>,
    feature_space: FeatureSpace,
    parallel: bool,
}

impl Fanova {
    /// Builds an fANOVA model for the given features and target.
    ///
    /// This is equivalent to `FanovaOptions::new().fit(features, target)`.
    pub fn fit(features: Vec<&[f64]>, target: &[f64]) -> Result<Self, FitError> {
        FanovaOptions::default().fit(features, target)
    }

    /// Calculates the importance of the given features.
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

        let mut trees = std::mem::take(&mut self.trees);
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

    /// Clears the internal cache.
    pub fn clear(&mut self) {
        for t in &mut self.trees {
            t.clear();
        }
    }

    #[cfg(test)]
    fn feature_combinations(&self, k: usize) -> impl Iterator<Item = Vec<usize>> + use<> {
        let features = self.feature_space.ranges().len();
        (1..=k).flat_map(move |k| (0..features).combinations(k))
    }

    fn traverse_covered_subspaces<F>(
        &self,
        marginal_value_index: usize,
        partition: &crate::partition::Partition,
        feature_subspaces: &[(usize, Vec<Range<f64>>)],
        f: &mut F,
    ) where
        F: FnMut(usize),
    {
        if feature_subspaces.is_empty() {
            f(marginal_value_index);
            return;
        }

        let (feature, subspaces) = &feature_subspaces[0];
        let range = partition.space.ranges()[*feature].clone();
        let start = subspaces
            .binary_search_by(|x| x.start.total_cmp(&range.start))
            .unwrap_or_else(|index| index);

        for i in (start..subspaces.len()).take_while(|&i| subspaces[i].end <= range.end) {
            self.traverse_covered_subspaces(
                marginal_value_index * subspaces.len() + i,
                partition,
                &feature_subspaces[1..],
                f,
            );
        }
    }

    fn quantify_importance_tree(&self, tree: &mut Tree, features: &[usize]) -> f64 {
        if let Some(&importance) = tree.importances.get(features) {
            return importance;
        }

        let feature_subspaces = features
            .iter()
            .copied()
            .map(|i| {
                let subspaces =
                    subspaces(tree.partitions.iter().map(|p| p.space.ranges()[i].clone()));
                (i, subspaces)
            })
            .collect::<Vec<_>>();

        let overall_marginal_size = self.feature_space.marginal_size(features);
        let mut marginal_values =
            vec![0.0; feature_subspaces.iter().map(|(_, s)| s.len()).product()];

        for p in tree.partitions.iter() {
            let partition_marginal_size = p.space.marginal_size(features);
            let weighted_value = p.value * (partition_marginal_size / overall_marginal_size);

            self.traverse_covered_subspaces(0, p, &feature_subspaces, &mut |index| {
                marginal_values[index] += weighted_value
            });
        }

        let mut variance = 0.0;
        for (mut i, v) in marginal_values.into_iter().enumerate() {
            let mut weight = 1.0;
            for (_, subspaces) in feature_subspaces.iter().rev() {
                let j = i % subspaces.len();
                i /= subspaces.len();
                weight *= subspaces[j].end - subspaces[j].start;
            }

            variance += (v - tree.mean).powi(2) * weight;
        }

        let size = self.feature_space.partial_size(features);
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

/// Importance of a feature set.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Importance {
    /// Average of importances across random forest trees.
    pub mean: f64,

    /// Standard deviation of importances across random forest trees.
    pub stddev: f64,
}

/// Possible errors which could be returned by `Fanove::fit` method.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum FitError {
    /// Features and target must have one or more rows.
    EmptyRows,

    /// Some of features or target have a different row count from others.
    RowSizeMismatch,

    /// Target contains non finite numbers.
    NonFiniteTarget,
}

impl std::fmt::Display for FitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyRows => write!(f, "features and target must have one or more rows"),
            Self::RowSizeMismatch => write!(
                f,
                "some of features or target have a different row count from others"
            ),
            Self::NonFiniteTarget => write!(f, "target contains non finite numbers"),
        }
    }
}

impl std::error::Error for FitError {}

impl From<TableError> for FitError {
    fn from(f: TableError) -> Self {
        match f {
            TableError::EmptyTable => Self::EmptyRows,
            TableError::NonFiniteTarget => Self::NonFiniteTarget,
            TableError::RowSizeMismatch => Self::RowSizeMismatch,
        }
    }
}

fn subspaces(partitions: impl Iterator<Item = Range<f64>>) -> Vec<Range<f64>> {
    let mut subspaces = BTreeMap::new();
    for p in partitions {
        insert_subspace(&mut subspaces, p);
    }
    subspaces.into_values().collect()
}

fn insert_subspace(subspaces: &mut BTreeMap<TotalF64, Range<f64>>, mut p: Range<f64>) {
    if (p.start - p.end).abs() < f64::EPSILON {
        return;
    }

    if let Some(mut q) = subspaces
        .range(..=TotalF64(p.start))
        .next_back()
        .map(|(_, q)| q.clone())
    {
        if (q.start - p.start).abs() < f64::EPSILON {
            if q.end > p.end {
                subspaces.remove(&TotalF64(q.start));

                q.start = p.end;
                subspaces.insert(TotalF64(p.start), p);
                subspaces.insert(TotalF64(q.start), q);
            } else {
                assert!(q.end <= p.end);
                p.start = q.end;
                insert_subspace(subspaces, p);
            }
        } else {
            assert!(q.start < p.start);
            if q.end > p.end {
                subspaces.remove(&TotalF64(q.start));

                let r = Range {
                    start: p.end,
                    end: q.end,
                };
                q.end = p.start;
                subspaces.insert(TotalF64(q.start), q);
                subspaces.insert(TotalF64(p.start), p);
                subspaces.insert(TotalF64(r.start), r);
            } else {
                assert!(q.end <= p.end);
                subspaces.remove(&TotalF64(q.start));

                let r = Range {
                    start: q.end,
                    end: p.end,
                };
                q.end = p.start;
                p.end = r.start;
                subspaces.insert(TotalF64(q.start), q);
                subspaces.insert(TotalF64(p.start), p);
                insert_subspace(subspaces, r);
            }
        }
    } else {
        subspaces.insert(TotalF64(p.start), p);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::{RngExt, SeedableRng};
    type TestResult = Result<(), Box<dyn std::error::Error>>;

    #[test]
    fn quantify_importance_k1_works() -> TestResult {
        let mut feature1 = Vec::new();
        let mut feature2 = Vec::new();
        let mut feature3 = Vec::new();
        let mut target = Vec::new();

        let mut rng = StdRng::seed_from_u64(0);
        for _ in 0..100 {
            let f1 = rng.random();
            let f2 = rng.random();
            let f3 = rng.random();
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
            vec![0.03949614161205558, 0.24001507447005044, 0.5934922097988682]
        );

        Ok(())
    }

    #[test]
    fn quantify_importance_k2_works() -> TestResult {
        let mut feature1 = Vec::new();
        let mut feature2 = Vec::new();
        let mut feature3 = Vec::new();
        let mut target = Vec::new();

        let mut rng = StdRng::seed_from_u64(0);
        for _ in 0..100 {
            let f1 = rng.random();
            let f2 = rng.random();
            let f3 = rng.random();
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
                0.06743725813997076,
                0.22594018609277702,
                0.19671183110117801,
                0.07562742199761226,
                0.05954537473187141,
                0.33055886109798627
            ]
        );

        Ok(())
    }
}
