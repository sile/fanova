use crate::decision_tree::{DecisionTreeOptions, DecisionTreeRegressor};
use crate::functions;
use crate::table::Table;
use rand::Rng;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use std::num::NonZeroUsize;

#[derive(Debug, Clone)]
pub struct RandomForestOptions {
    pub trees: NonZeroUsize,
    pub max_features: Option<usize>,
}

impl Default for RandomForestOptions {
    fn default() -> Self {
        Self {
            trees: NonZeroUsize::new(100).expect("never fails"),
            max_features: None,
        }
    }
}

#[derive(Debug)]
pub struct RandomForestRegressor {
    forest: Vec<DecisionTreeRegressor>,
}

impl RandomForestRegressor {
    pub fn fit<R: Rng + ?Sized>(rng: &mut R, table: Table, options: RandomForestOptions) -> Self {
        let max_features = Self::decide_max_features(&table, &options);
        let forest = (0..options.trees.get())
            .map(|_| Self::tree_fit(rng, &table, max_features))
            .collect::<Vec<_>>();
        Self { forest }
    }

    pub fn fit_parallel(table: Table, options: RandomForestOptions) -> Self {
        let max_features = Self::decide_max_features(&table, &options);
        let forest = (0..options.trees.get())
            .into_par_iter()
            .map(|_| Self::tree_fit(&mut rand::thread_rng(), &table, max_features))
            .collect::<Vec<_>>();
        Self { forest }
    }

    fn decide_max_features(table: &Table, options: &RandomForestOptions) -> usize {
        options
            .max_features
            .unwrap_or_else(|| (table.features_len() as f64 / 3.0).ceil() as usize)
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

    pub fn forest(&self) -> &[DecisionTreeRegressor] {
        &self.forest
    }

    pub fn predict(&self, xs: &[f64]) -> f64 {
        functions::mean(self.forest.iter().map(|tree| tree.predict(xs)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{self, SeedableRng};

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

        let mut rng = rand::rngs::StdRng::from_seed(Default::default());
        let regressor = RandomForestRegressor::fit(&mut rng, table, Default::default());
        assert_eq!(
            regressor.predict(&columns.iter().map(|f| f[train_len]).collect::<Vec<_>>()),
            41.92138095238095
        );
        assert_eq!(
            regressor.predict(&columns.iter().map(|f| f[train_len + 1]).collect::<Vec<_>>()),
            45.140666666666675
        );

        Ok(())
    }
}
