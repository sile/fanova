use crate::functions;
use crate::table::Table;

#[derive(Debug)]
pub struct DecisionTreeRegressor {
    tree: Tree,
}

impl DecisionTreeRegressor {
    pub fn fit<'a>(table: Table<'a>) -> Self {
        let tree = Tree::fit(table, Mse, false);
        Self { tree }
    }

    pub fn predict(&self, xs: &[f64]) -> f64 {
        self.tree.predict(xs)
    }
}

pub trait Criterion {
    fn calculate<T>(&self, target: T) -> f64
    where
        T: Iterator<Item = f64> + Clone;
}

#[derive(Debug)]
pub struct Mse;

impl Criterion for Mse {
    fn calculate<T>(&self, target: T) -> f64
    where
        T: Iterator<Item = f64> + Clone,
    {
        let n = target.clone().count() as f64;
        let m = functions::mean(target.clone());
        target.map(|y| (y - m).powi(2)).sum::<f64>() / n
    }
}

#[derive(Debug)]
pub struct Tree {
    root: Node,
}

impl Tree {
    pub fn fit<'a>(mut table: Table<'a>, criterion: impl Criterion, classification: bool) -> Self {
        let mut builder = NodeBuilder {
            criterion,
            classification,
        };
        let root = builder.build(&mut table);
        Self { root }
    }

    pub fn predict(&self, xs: &[f64]) -> f64 {
        self.root.predict(xs)
    }
}

#[derive(Debug)]
pub struct Node {
    label: f64,
    children: Option<Children>,
}

impl Node {
    fn new(label: f64) -> Self {
        Self {
            label,
            children: None,
        }
    }

    fn predict(&self, xs: &[f64]) -> f64 {
        if let Some(children) = &self.children {
            if xs[children.split.column] <= children.split.threshold {
                children.left.predict(xs)
            } else {
                children.right.predict(xs)
            }
        } else {
            self.label
        }
    }
}

#[derive(Debug)]
pub struct Children {
    split: SplitPoint,
    left: Box<Node>,
    right: Box<Node>,
}

#[derive(Debug)]
struct SplitPoint {
    information_gain: f64,
    column: usize,
    threshold: f64,
}

#[derive(Debug)]
struct NodeBuilder<C> {
    criterion: C,
    classification: bool,
}

impl<C> NodeBuilder<C>
where
    C: Criterion,
{
    fn build(&mut self, table: &mut Table) -> Node {
        if table.is_single_target() {
            let label = table.target().nth(0).expect("never fails");
            return Node::new(label);
        }

        let label = if self.classification {
            functions::most_frequent(table.target())
        } else {
            functions::mean(table.target())
        };

        let mut node = Node::new(label);
        let mut best: Option<SplitPoint> = None;
        let impurity = self.criterion.calculate(table.target());
        let rows = table.target().count();

        for column in 0..table.features_len() {
            if table.feature(column).any(|f| f.is_nan()) {
                continue;
            }

            table.sort_rows_by_feature(column);
            for (row, threshold) in table.thresholds(column) {
                let impurity_l = self.criterion.calculate(table.target().take(row));
                let impurity_r = self.criterion.calculate(table.target().skip(row));
                let n_l = row as f64 / rows as f64;
                let n_r = 1.0 - n_l;

                let information_gain = impurity - (n_l * impurity_l + n_r * impurity_r);
                if best
                    .as_ref()
                    .map_or(true, |t| t.information_gain < information_gain)
                {
                    best = Some(SplitPoint {
                        information_gain,
                        column,
                        threshold,
                    });
                }
            }
        }

        let best = best.expect("never fails");
        node.children = Some(self.build_children(table, best));
        node
    }

    fn build_children(&mut self, table: &mut Table, split: SplitPoint) -> Children {
        table.sort_rows_by_feature(split.column);
        let row = table
            .feature(split.column)
            .take_while(|&f| f <= split.threshold)
            .count();
        let (left, right) = table.with_split(row, |table| Box::new(self.build(table)));
        Children { split, left, right }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

        let regressor = DecisionTreeRegressor::fit(table);
        assert_eq!(
            regressor.predict(&features.iter().map(|f| f[train_len]).collect::<Vec<_>>()),
            46.0
        );
        assert_eq!(
            regressor.predict(
                &features
                    .iter()
                    .map(|f| f[train_len + 1])
                    .collect::<Vec<_>>()
            ),
            52.0
        );

        Ok(())
    }
}
