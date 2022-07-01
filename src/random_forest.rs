use crate::decision_tree::{DecisionTreeOptions, DecisionTreeRegressor};
#[cfg(test)]
use crate::functions;
use crate::table::Table;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use std::num::NonZeroUsize;

/// Random forest options.
#[derive(Debug, Clone)]
pub struct RandomForestOptions {
    trees: NonZeroUsize,
    max_features: Option<NonZeroUsize>,
    seed: Option<u64>,
}

impl RandomForestOptions {
    /// Makes a `RandomForestOptions` instance with the default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the random generator seed.
    ///
    /// The default value is random.
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Sets the number of decision trees.
    ///
    /// The default value is `100`.
    pub fn trees(mut self, trees: NonZeroUsize) -> Self {
        self.trees = trees;
        self
    }

    /// Sets the number of maximum candidate features used to determine each decision tree node.
    ///
    /// The default value is `sqrt(the number of features)`.
    pub fn max_features(mut self, max: NonZeroUsize) -> Self {
        self.max_features = Some(max);
        self
    }
}

impl Default for RandomForestOptions {
    fn default() -> Self {
        Self {
            trees: NonZeroUsize::new(100).expect("never fails"),
            max_features: None,
            seed: None,
        }
    }
}

impl RandomForestOptions {
    fn tree_rngs(&self) -> impl Iterator<Item = StdRng> {
        let seed = self.seed.unwrap_or_else(|| rand::thread_rng().gen());
        let mut rng = StdRng::seed_from_u64(seed);
        (0..self.trees.get()).map(move |_| StdRng::seed_from_u64(rng.gen()))
    }
}

#[derive(Debug)]
pub struct RandomForestRegressor {
    forest: Vec<DecisionTreeRegressor>,
}

impl RandomForestRegressor {
    pub fn fit(table: Table, options: RandomForestOptions) -> Self {
        let max_features = Self::decide_max_features(&table, &options);
        let forest = options
            .tree_rngs()
            .map(|mut rng| Self::tree_fit(&mut rng, &table, max_features))
            .collect::<Vec<_>>();
        Self { forest }
    }

    pub fn fit_parallel(table: Table, options: RandomForestOptions) -> Self {
        let max_features = Self::decide_max_features(&table, &options);
        let forest = options
            .tree_rngs()
            .collect::<Vec<_>>()
            .into_par_iter()
            .map(|mut rng| Self::tree_fit(&mut rng, &table, max_features))
            .collect::<Vec<_>>();
        Self { forest }
    }

    fn decide_max_features(table: &Table, options: &RandomForestOptions) -> usize {
        if let Some(n) = options.max_features {
            n.get()
        } else {
            (table.features_len() as f64).sqrt().ceil() as usize
        }
    }

    fn tree_fit<R: Rng + ?Sized>(
        rng: &mut R,
        table: &Table,
        max_features: usize,
    ) -> DecisionTreeRegressor {
        let table = table.bootstrap_sample(rng);
        let tree_options = DecisionTreeOptions {
            max_features: Some(max_features),
        };
        DecisionTreeRegressor::fit(rng, table, tree_options)
    }

    pub fn into_trees(self) -> Vec<DecisionTreeRegressor> {
        self.forest
    }

    #[cfg(test)]
    fn predict(&self, xs: &[f64]) -> f64 {
        functions::mean(self.forest.iter().map(|tree| tree.predict(xs)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn regression_works() -> Result<(), anyhow::Error> {
        let columns = vec![
            // Features.
            &[
                0.0, 0.0, 1.0, 2.0, 2.0, 2.0, 1.0, 0.0, 0.0, 2.0, 0.0, 1.0, 1.0, 2.0,
            ],
            &[
                2.0, 2.0, 2.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 2.0, 1.0,
            ],
            &[
                1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0,
            ],
            &[
                0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0,
            ],
            // Target.
            &[
                25.0, 30.0, 46.0, 45.0, 52.0, 23.0, 43.0, 35.0, 38.0, 46.0, 48.0, 52.0, 44.0, 30.0,
            ],
        ];
        let train_len = columns[0].len() - 2;

        let table = Table::new(columns.iter().map(|f| &f[..train_len]).collect())?;

        let mut options = RandomForestOptions::default();
        options.seed = Some(0);
        let regressor = RandomForestRegressor::fit(table, options);
        assert_eq!(
            regressor.predict(&columns.iter().map(|f| f[train_len]).collect::<Vec<_>>()),
            41.063500000000005
        );
        assert_eq!(
            regressor.predict(&columns.iter().map(|f| f[train_len + 1]).collect::<Vec<_>>()),
            44.425000000000004
        );

        Ok(())
    }
}
