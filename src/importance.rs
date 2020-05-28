use crate::partitioning::ForestPartitioning;
use crate::random_forest::RandomForestRegressor;
use crate::table::Table;
use rand;
use std::ops::Range;

pub fn quantify_importance(config_space: Vec<Range<f64>>, table: Table) {
    let regressor = RandomForestRegressor::fit(&mut rand::thread_rng(), table, Default::default());
    let partitioning = ForestPartitioning::new(&regressor, config_space);
}
