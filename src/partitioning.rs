use crate::decision_tree::DecisionTreeRegressor;
//use crate::random_forest::RandomForestRegressor;
use std::ops::Range;

#[derive(Debug)]
pub struct Partition {
    pub label: f64,
    pub config_space: Vec<Range<f64>>,
    valid_columns: Vec<usize>,
}

impl Partition {
    fn contains2(&self, i: usize, space: &Range<f64>) -> bool {
        if !self.valid_columns.contains(&i) {
            return false;
        }
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
        |mut acc, config_space, label, valid_columns| {
            acc.push(Partition {
                label,
                config_space,
                valid_columns,
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

    pub fn marginal_predict2(&self, column: usize, space: &Range<f64>) -> f64 {
        // let total_size = self
        //     .config_space
        //     .iter()
        //     .enumerate()
        //     .filter(|(i, _)| *i != column) // TODO(?): filter other invalid columns
        //     .map(|(_, s)| s.end - s.start)
        //     .product::<f64>();
        self.partitions
            .iter()
            .filter(|p| p.contains2(column, space))
            .map(|p| {
                // let size = p
                //     .config_space
                //     .iter()
                //     .enumerate() // TODO(?): filter other invalid columns
                //     .filter(|(i, _)| *i != column)
                //     .filter(|(i, _)| p.valid_columns.contains(i))
                //     .map(|(_, s)| s.end - s.start)
                //     .product::<f64>();
                let v = p
                    .valid_columns
                    .iter()
                    .filter(|&i| *i != column)
                    .map(|&i| {
                        let cs0 = &p.config_space[i];
                        let cs1 = &self.config_space[i];
                        (cs0.end - cs0.start) / (cs1.end - cs1.start)
                    })
                    .product::<f64>();

                v * p.label
            })
            .sum()
    }

    pub fn mean(&self) -> f64 {
        self.partitions()
            .map(|p| {
                let v = p
                    .valid_columns
                    .iter()
                    .map(|&i| {
                        let cs0 = &p.config_space[i];
                        let cs1 = &self.config_space[i];
                        (cs0.end - cs0.start) / (cs1.end - cs1.start)
                    })
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
                    .valid_columns
                    .iter()
                    .map(|&i| {
                        let cs0 = &p.config_space[i];
                        let cs1 = &self.config_space[i];
                        (cs0.end - cs0.start) / (cs1.end - cs1.start)
                    })
                    .product::<f64>();
                v * (p.label - m).powi(2)
            })
            .sum()
    }

    pub fn partitions(&self) -> impl Iterator<Item = &Partition> {
        self.partitions.iter()
    }
}
