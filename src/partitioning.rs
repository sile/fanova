use crate::decision_tree::DecisionTreeRegressor;
use crate::random_forest::RandomForestRegressor;
use std::ops::Range;

#[derive(Debug)]
pub struct Partition {
    pub label: f64,
    pub config_space: Vec<Range<f64>>,
}

impl Partition {
    fn contains(&self, xs: &[(usize, f64)]) -> bool {
        xs.iter().all(|&(i, v)| self.config_space[i].contains(&v))
    }
}

fn compute_partitions(
    regressor: &DecisionTreeRegressor,
    config_space: Vec<Range<f64>>,
) -> Vec<Partition> {
    regressor.fold(
        config_space,
        Vec::new(),
        |config_space, split| {
            let mut cs_l = config_space.clone();
            let mut cs_r = config_space;
            cs_l[split.column].end = split.threshold;
            cs_r[split.column].start = split.threshold;
            (cs_l, cs_r)
        },
        |mut acc, config_space, label| {
            acc.push(Partition {
                label,
                config_space,
            });
            acc
        },
    )
}

#[derive(Debug)]
pub struct TreePartitioning {
    partitions: Vec<Partition>,
    config_space: Vec<Range<f64>>,
}

impl TreePartitioning {
    pub fn new(regressor: &DecisionTreeRegressor, config_space: Vec<Range<f64>>) -> Self {
        Self {
            partitions: compute_partitions(regressor, config_space.clone()),
            config_space,
        }
    }

    pub fn marginal_predict(&self, partial_config: &[(usize, f64)]) -> f64 {
        let total_size = self
            .config_space
            .iter()
            .enumerate()
            .filter(|(i, _)| partial_config.iter().find(|(j, _)| i == j).is_none())
            .map(|(_, s)| s.end - s.start)
            .sum::<f64>();
        self.partitions
            .iter()
            .filter(|p| p.contains(partial_config))
            .map(|p| {
                let size = p
                    .config_space
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| partial_config.iter().find(|(j, _)| i == j).is_none())
                    .map(|(_, s)| s.end - s.start)
                    .sum::<f64>();
                (size / total_size) * p.label
            })
            .sum()
    }
}

#[derive(Debug)]
pub struct ForestPartitioning {
    forest: Vec<TreePartitioning>,
}

impl ForestPartitioning {
    pub fn new(regressor: &RandomForestRegressor, config_space: Vec<Range<f64>>) -> Self {
        Self {
            forest: regressor
                .forest()
                .iter()
                .map(|tree| TreePartitioning::new(tree, config_space.clone()))
                .collect(),
        }
    }

    pub fn marginal_predict(&self, partial_config: &[(usize, f64)]) -> f64 {
        let sum = self
            .forest
            .iter()
            .map(|t| t.marginal_predict(partial_config))
            .sum::<f64>();
        sum / self.forest.len() as f64
    }
}
