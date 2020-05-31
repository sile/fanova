use crate::decision_tree::{DecisionTreeOptions, DecisionTreeRegressor};
use crate::functions;
use crate::table::Table;
use rand::Rng;
use std::num::NonZeroUsize;

#[derive(Debug, Clone)]
pub struct RandomForestOptions {
    pub trees: NonZeroUsize,
}

impl Default for RandomForestOptions {
    fn default() -> Self {
        Self {
            trees: NonZeroUsize::new(100).unwrap(),
            //trees: NonZeroUsize::new(1).unwrap(),
        }
    }
}

#[derive(Debug)]
pub struct RandomForestRegressor {
    forest: Vec<DecisionTreeRegressor>,
}

impl RandomForestRegressor {
    pub fn fit<R: Rng + ?Sized>(rng: &mut R, table: Table, options: RandomForestOptions) -> Self {
        let forest = (0..options.trees.get())
            .map(|_| {
                let mut table = table.clone();
                table.subsample(rng, table.rows_len());
                let options = DecisionTreeOptions {
                    max_features: Some(std::cmp::max(1, table.features_len() / 3)),
                    //max_features: Some(std::cmp::max(10, table.features_len() / 3)),
                };
                let tree = DecisionTreeRegressor::fit(rng, table, options);
                tree
            })
            .collect::<Vec<_>>();
        Self { forest }
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
        let features = vec![
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
        ];
        let target = &[
            25.0, 30.0, 46.0, 45.0, 52.0, 23.0, 43.0, 35.0, 38.0, 46.0, 48.0, 52.0, 44.0, 30.0,
        ];
        let train_len = target.len() - 2;

        let table = Table::new(
            features.iter().map(|f| &f[..train_len]).collect(),
            &target[..train_len],
        )?;

        let mut rng = rand::rngs::StdRng::from_seed(Default::default());
        let regressor = RandomForestRegressor::fit(&mut rng, table, Default::default());
        assert_eq!(
            regressor.predict(&features.iter().map(|f| f[train_len]).collect::<Vec<_>>()),
            40.50667857142857
        );
        assert_eq!(
            regressor.predict(
                &features
                    .iter()
                    .map(|f| f[train_len + 1])
                    .collect::<Vec<_>>()
            ),
            43.81471428571429
        );

        Ok(())
    }
}
