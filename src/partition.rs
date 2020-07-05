use crate::decision_tree::DecisionTreeRegressor;
use crate::space::SparseFeatureSpace;
use crate::FeatureSpace;

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
        let total_size = self
            .space
            .ranges()
            .iter()
            .enumerate()
            .filter(|(i, _)| fixed_space.iter().find(|(j, _)| i == j).is_none())
            .map(|(_, s)| s.end - s.start)
            .product::<f64>();
        self.partitions
            .iter()
            .filter(|p| p.space.covers(&fixed_space))
            .map(|p| {
                let size = p
                    .space
                    .ranges()
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| fixed_space.iter().find(|(j, _)| i == j).is_none())
                    .map(|(_, s)| s.end - s.start)
                    .product::<f64>();
                (size / total_size) * p.value
            })
            .sum()
    }

    pub fn mean_and_variance(&self) -> (f64, f64) {
        let weights = self
            .partitions()
            .map(|p| {
                p.space
                    .ranges()
                    .iter()
                    .zip(self.space.ranges().iter())
                    .map(|(cs0, cs1)| (cs0.end - cs0.start) / (cs1.end - cs1.start))
                    .product::<f64>()
            })
            .collect::<Vec<_>>();
        let mean = self
            .partitions()
            .zip(weights.iter())
            .map(|(p, w)| w * p.value)
            .sum::<f64>();
        let variance = self
            .partitions()
            .zip(weights.iter())
            .map(|(p, w)| w * (p.value - mean).powi(2))
            .sum::<f64>();
        (mean, variance)
    }

    pub fn partitions(&self) -> impl Iterator<Item = &Partition> {
        self.partitions.iter()
    }
}
