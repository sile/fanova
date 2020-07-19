use crate::decision_tree::{DecisionTreeOptions, DecisionTreeRegressor};
#[cfg(test)]
use crate::functions;
use crate::table::Table;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use std::num::NonZeroUsize;

#[derive(Debug, Clone)]
pub struct RandomForestOptions {
    // TODO: Make these fields private.
    pub trees: NonZeroUsize,
    pub max_features: Option<usize>,
    pub seed: Option<u64>,
}

impl RandomForestOptions {
    pub fn trees(mut self, trees: NonZeroUsize) -> Self {
        self.trees = trees;
        self
    }

    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
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
        let seed_u64 = self.seed.unwrap_or_else(|| rand::thread_rng().gen());
        let mut seed = [0u8; 32];
        (&mut seed[0..8]).copy_from_slice(&seed_u64.to_be_bytes()[..]);
        let mut rng = StdRng::from_seed(seed);
        (0..self.trees.get()).map(move |_| {
            let mut seed = [0u8; 32];
            rng.fill(&mut seed);
            StdRng::from_seed(seed)
        })
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
        options
            .max_features
            .unwrap_or_else(|| (table.features_len() as f64).sqrt().ceil() as usize)
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
            41.9785
        );
        assert_eq!(
            regressor.predict(&columns.iter().map(|f| f[train_len + 1]).collect::<Vec<_>>()),
            43.50333333333333
        );

        Ok(())
    }
}
