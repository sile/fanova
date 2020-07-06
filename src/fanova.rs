use crate::decision_tree::DecisionTreeRegressor;
use crate::functions;
use crate::partition::TreePartitions;
use crate::random_forest::{RandomForestOptions, RandomForestRegressor};
use crate::space::SparseFeatureSpace;
use crate::table::{Table, TableError};
use crate::FeatureSpace;
use itertools::Itertools as _;
use ordered_float::OrderedFloat;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::num::NonZeroUsize;
use std::ops::Range;
use thiserror::Error;

#[derive(Debug, Clone)]
pub struct FanovaOptions {
    random_forest: RandomForestOptions,
    feature_space: Option<FeatureSpace>,
    parallel: bool,
    // TODO: target_cutoff
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

    pub fn parallel(mut self, enabled: bool) -> Self {
        self.parallel = enabled;
        self
    }

    pub fn fit(self, features: Vec<&'a [f64]>, target: &'a [f64]) -> Result<Self, FitError> {
        let mut columns = features;
        columns.push(target);
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
        })
    }
}

impl Default for FanovaOptions {
    fn default() -> Self {
        Self {
            random_forest: RandomForestOptions::default(),
            feature_space: None,
            parallel: false,
        }
    }
}

#[derive(Debug)]
pub struct Fanova {
    random_forest: RandomForestRegressor,
    feature_space: FeatureSpace,
}

impl Fanova {
    pub fn fit(features: Vec<&'a [f64]>, target: &'a [f64]) -> Result<Self, FitError> {
        FanovaOptions::default().fit(features, target)
    }

    pub fn quantify_importance(&self, features: Vec<usize>) -> Importance {
        todo!()
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Importance {
    pub mean: f64,
    pub variance: f64,
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

// #[derive(Debug)]
// pub struct Fanova<'a> {
//     space: FeatureSpace,
//     table: Option<Table<'a>>,
//     options: FanovaOptions,
// }

// impl<'a> Fanova<'a> {
//     pub fn new(features: Vec<&'a [f64]>, target: &'a [f64]) -> Result<Self, FanovaError> {
//         Self::with_options(features, target, Default::default())
//     }

//     pub fn with_options(
//         features: Vec<&'a [f64]>,
//         target: &'a [f64],
//         mut options: FanovaOptions,
//     ) -> Result<Self, FanovaError> {
//         let mut columns = features;
//         columns.push(target);
//         let table = Table::new(columns)?;

//         let space = if let Some(space) = options.feature_space.take() {
//             if space.ranges().len() != table.features_len() {
//                 return Err(FanovaError::FeatureSpaceSizeMismatch);
//             }

//             for (i, range) in space.ranges().iter().enumerate() {
//                 let mut feature = table.column(i);
//                 if feature.any(|f| f < range.start || range.end <= f) {
//                     return Err(FanovaError::TooNarrowFeatureSpace { feature: i });
//                 }
//             }

//             space
//         } else {
//             FeatureSpace::from_table(&table)
//         };

//         Ok(Self {
//             space,
//             table: Some(table),
//             options,
//         })
//     }

//     // TODO: quantify_importance_parallel

//     fn calc_effect(
//         &self,
//         subspace: &SparseFeatureSpace,
//         partitioning: &TreePartitions,
//         mean: f64,
//     ) -> f64 {
//         let mut v = partitioning.marginal_predict(subspace);
//         for k in 1..subspace.features() {
//             for subspace in subspace.iter().combinations(k).map(SparseFeatureSpace::new) {
//                 v -= self.calc_effect(&subspace, partitioning, mean);
//             }
//         }
//         v - mean
//     }

//     fn quantify_importance_tree(
//         &self,
//         tree: &DecisionTreeRegressor,
//     ) -> HashMap<BTreeSet<usize>, f64> {
//         let space = self.space.clone();
//         let mut importances = HashMap::<BTreeSet<usize>, f64>::new();
//         let partitioning = TreePartitions::new(tree, space.clone());
//         let (mean, total_variance) = partitioning.mean_and_variance();

//         for k in 1..self.options.max_importance_dimension.get() {
//             for columns in (0..space.ranges().len()).combinations(k) {
//                 let variance = columns
//                     .iter()
//                     .copied()
//                     .map(|i| {
//                         subspaces(
//                             partitioning
//                                 .partitions()
//                                 .map(|p| p.space.ranges()[i].clone()),
//                         )
//                         .map(|space| (i, space))
//                         .collect::<Vec<_>>()
//                     })
//                     .multi_cartesian_product()
//                     .map(SparseFeatureSpace::new)
//                     .map(|subspace| {
//                         let effect = self.calc_effect(&subspace, &partitioning, mean);
//                         let size = subspace.iter().map(|(_, s)| s.end - s.start).sum::<f64>();
//                         effect.powi(2) * size
//                     })
//                     .sum::<f64>();

//                 let size = columns
//                     .iter()
//                     .map(|&i| &space.ranges()[i])
//                     .map(|s| s.end - s.start)
//                     .sum::<f64>();
//                 let v = variance / size / total_variance;

//                 let old = importances.insert(columns.into_iter().collect(), v);
//                 assert!(old.is_none());
//             }
//         }

//         importances
//     }

//     pub fn quantify_importance(mut self) -> Vec<Importance> {
//         let table = self.table.take().expect("never fails");

//         let mut importances = HashMap::<BTreeSet<usize>, Vec<f64>>::new();
//         let regressor = RandomForestRegressor::fit(table, Default::default());
//         for tree in regressor.forest().iter() {
//             for (k, v) in self.quantify_importance_tree(tree) {
//                 importances.entry(k).or_default().push(v);
//             }
//         }

//         importances
//             .into_iter()
//             .map(|(features, vs)| {
//                 let (mean, stddev) = functions::mean_and_stddev(vs.into_iter());
//                 Importance {
//                     features,
//                     mean,
//                     stddev,
//                 }
//             })
//             .collect()
//     }
// }

// #[derive(Debug, Clone)]
// pub struct Importance {
//     pub features: BTreeSet<usize>,
//     pub mean: f64,
//     pub stddev: f64,
// }

// fn subspaces(partitions: impl Iterator<Item = Range<f64>>) -> impl Iterator<Item = Range<f64>> {
//     let mut subspaces = BTreeMap::new();
//     for p in partitions {
//         insert_subspace(&mut subspaces, p);
//     }
//     subspaces.into_iter().map(|(_, v)| v)
// }

// fn insert_subspace(subspaces: &mut BTreeMap<OrderedFloat<f64>, Range<f64>>, mut p: Range<f64>) {
//     if p.start == p.end {
//         return;
//     }

//     if let Some(mut q) = subspaces
//         .range(..=OrderedFloat(p.start))
//         .rev()
//         .nth(0)
//         .map(|(_, q)| q.clone())
//     {
//         if q.start == p.start {
//             if q.end > p.end {
//                 subspaces.remove(&OrderedFloat(q.start));

//                 q.start = p.end;
//                 subspaces.insert(OrderedFloat(p.start), p);
//                 subspaces.insert(OrderedFloat(q.start), q);
//             } else {
//                 assert!(q.end <= p.end);
//                 p.start = q.end;
//                 insert_subspace(subspaces, p);
//             }
//         } else {
//             assert!(q.start < p.start);
//             if q.end > p.end {
//                 subspaces.remove(&OrderedFloat(q.start));

//                 let r = Range {
//                     start: p.end,
//                     end: q.end,
//                 };
//                 q.end = p.start;
//                 subspaces.insert(OrderedFloat(q.start), q);
//                 subspaces.insert(OrderedFloat(p.start), p);
//                 subspaces.insert(OrderedFloat(r.start), r);
//             } else {
//                 assert!(q.end <= p.end);
//                 subspaces.remove(&OrderedFloat(q.start));

//                 let r = Range {
//                     start: q.end,
//                     end: p.end,
//                 };
//                 q.end = p.start;
//                 p.end = r.start;
//                 subspaces.insert(OrderedFloat(q.start), q);
//                 subspaces.insert(OrderedFloat(p.start), p);
//                 insert_subspace(subspaces, r);
//             }
//         }
//     } else {
//         subspaces.insert(OrderedFloat(p.start), p);
//     }
// }
