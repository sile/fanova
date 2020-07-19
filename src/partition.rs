use crate::decision_tree::DecisionTreeRegressor;
use crate::space::{FeatureSpace, SparseFeatureSpace};

#[derive(Debug)]
pub struct Partition {
    pub value: f64,
    pub space: FeatureSpace,
}

#[derive(Debug)]
pub struct TreePartitions {
    partitions: Vec<Partition>,
    space: FeatureSpace,
}

impl TreePartitions {
    pub fn new(regressor: &DecisionTreeRegressor, space: FeatureSpace) -> Self {
        let partitions = regressor.fold(
            space.clone(),
            |space, split| space.split(split.column, split.threshold),
            Vec::new(),
            |mut acc, space, value| {
                acc.push(Partition { value, space });
                acc
            },
        );
        Self { partitions, space }
    }

    pub fn marginal_predict(&self, fixed_space: &SparseFeatureSpace) -> f64 {
        let overall_size = self.space.marginal_size(fixed_space);
        self.partitions
            .iter()
            .filter(|p| p.space.covers(&fixed_space))
            .map(|p| {
                let size = p.space.marginal_size(fixed_space);
                (size / overall_size) * p.value
            })
            .sum()
    }

    pub fn mean_and_variance(&self) -> (f64, f64) {
        let overall_size = self.space.size();
        let weights = self
            .iter()
            .map(|p| p.space.size() / overall_size)
            .collect::<Vec<_>>();
        let mean = self
            .iter()
            .zip(weights.iter())
            .map(|(p, w)| w * p.value)
            .sum::<f64>();
        let variance = self
            .iter()
            .zip(weights.iter())
            .map(|(p, w)| w * (p.value - mean).powi(2))
            .sum::<f64>();
        (mean, variance)
    }

    pub fn iter(&self) -> impl Iterator<Item = &Partition> {
        self.partitions.iter()
    }
}
