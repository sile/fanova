use crate::decision_tree::DecisionTreeRegressor;
//use crate::random_forest::RandomForestRegressor;
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

    fn contains2(&self, i: usize, space: &Range<f64>) -> bool {
        self.config_space[i].start <= space.start && space.end <= self.config_space[i].end
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

    // TODO: delete
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

    pub fn marginal_predict2(&self, column: usize, space: &Range<f64>) -> f64 {
        let total_size = self
            .config_space
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != column)
            .map(|(_, s)| s.end - s.start)
            .product::<f64>();
        self.partitions
            .iter()
            .filter(|p| p.contains2(column, space))
            .map(|p| {
                let size = p
                    .config_space
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| *i != column)
                    .map(|(_, s)| s.end - s.start)
                    .product::<f64>();
                (size / total_size) * p.label
            })
            .sum()
    }

    pub fn mean(&self) -> f64 {
        self.partitions()
            .map(|p| {
                let v = p
                    .config_space
                    .iter()
                    .zip(self.config_space.iter())
                    .map(|(cs0, cs1)| (cs0.end - cs0.start) / (cs1.end - cs1.start))
                    .product::<f64>();
                v * p.label
            })
            .sum()
    }

    pub fn variance(&self) -> f64 {
        let m = self.mean();
        self.partitions()
            .map(|p| {
                let v = p
                    .config_space
                    .iter()
                    .zip(self.config_space.iter())
                    .map(|(cs0, cs1)| (cs0.end - cs0.start) / (cs1.end - cs1.start))
                    .product::<f64>();
                v * (p.label - m).powi(2)
            })
            .sum()
    }

    pub fn partitions(&self) -> impl Iterator<Item = &Partition> {
        self.partitions.iter()
    }
}

// #[derive(Debug)]
// pub struct ForestPartitioning {
//     forest: Vec<TreePartitioning>,
//     config_space: Vec<Range<f64>>,
// }

// impl ForestPartitioning {
//     pub fn new(regressor: &RandomForestRegressor, config_space: Vec<Range<f64>>) -> Self {
//         Self {
//             forest: regressor
//                 .forest()
//                 .iter()
//                 .map(|tree| TreePartitioning::new(tree, config_space.clone()))
//                 .collect(),
//             config_space,
//         }
//     }

//     pub fn mean(&self) -> f64 {
//         self.partitions()
//             .map(|p| {
//                 let v = p
//                     .config_space
//                     .iter()
//                     .zip(self.config_space.iter())
//                     .map(|(cs0, cs1)| (cs0.end - cs0.start) / (cs1.end - cs1.start))
//                     .product::<f64>();
//                 v * p.label
//             })
//             .sum()
//     }

//     pub fn variance(&self) -> f64 {
//         let m = self.mean();
//         self.partitions()
//             .map(|p| {
//                 let v = p
//                     .config_space
//                     .iter()
//                     .zip(self.config_space.iter())
//                     .map(|(cs0, cs1)| (cs0.end - cs0.start) / (cs1.end - cs1.start))
//                     .product::<f64>();
//                 v * (p.label - m).powi(2)
//             })
//             .sum()
//     }

//     pub fn partitions(&self) -> impl Iterator<Item = &Partition> {
//         self.forest.iter().flat_map(|t| t.partitions.iter())
//     }

//     pub fn marginal_predict(&self, partial_config: &[(usize, f64)]) -> f64 {
//         let sum = self
//             .forest
//             .iter()
//             .map(|t| t.marginal_predict(partial_config))
//             .sum::<f64>();
//         sum / self.forest.len() as f64 // TODO(?): remove
//     }

//     pub fn marginal_predict2(&self, column: usize, space: &Range<f64>) -> f64 {
//         self.forest
//             .iter()
//             .map(|t| t.marginal_predict2(column, space))
//             .sum()
//     }
// }
