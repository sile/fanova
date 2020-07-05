use crate::decision_tree::DecisionTreeRegressor;
use std::ops::Range;

#[derive(Debug)]
pub struct Partition {
    pub value: f64,
    pub space: Vec<Range<f64>>,
}

#[derive(Debug)]
pub struct TreePartitions {
    partitions: Vec<Partition>,
    config_space: Vec<Range<f64>>,
}

impl TreePartitions {
    pub fn new(regressor: &DecisionTreeRegressor, config_space: Vec<Range<f64>>) -> Self {
        let partitions = regressor.fold(
            config_space.clone(),
            |config_space, split| {
                let mut cs_l = config_space.clone();
                let mut cs_r = config_space;
                cs_l[split.column].end = split.threshold;
                cs_r[split.column].start = split.threshold;
                (cs_l, cs_r)
            },
            Vec::new(),
            |mut acc, space, value| {
                acc.push(Partition { value, space });
                acc
            },
        );
        Self {
            partitions,
            config_space,
        }
    }

    pub fn marginal_predict(&self, fixed_space: &[(usize, Range<f64>)]) -> f64 {
        fn contains(a: &Range<f64>, b: &Range<f64>) -> bool {
            a.start <= b.start && b.end <= a.end
        }

        let total_size = self
            .config_space
            .iter()
            .enumerate()
            .filter(|(i, _)| fixed_space.iter().find(|(j, _)| i == j).is_none())
            .map(|(_, s)| s.end - s.start)
            .product::<f64>();
        self.partitions
            .iter()
            .filter(|p| {
                fixed_space
                    .iter()
                    .all(|(c, space)| contains(&p.space[*c], space))
            })
            .map(|p| {
                let size = p
                    .space
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
                    .iter()
                    .zip(self.config_space.iter())
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
