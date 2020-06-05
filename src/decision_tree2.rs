use crate::functions;
use crate::table2::Table;
use rand::seq::SliceRandom as _;
use rand::Rng;

#[derive(Debug, Clone, Default)]
pub struct DecisionTreeOptions {
    pub max_features: Option<usize>,
}

#[derive(Debug)]
pub struct DecisionTreeRegressor {
    tree: Tree,
}

impl DecisionTreeRegressor {
    pub fn fit<'a, R: Rng + ?Sized>(
        rng: &mut R,
        table: Table<'a>,
        options: DecisionTreeOptions,
    ) -> Self {
        let tree = Tree::fit(rng, table, options);
        Self { tree }
    }

    pub fn predict(&self, xs: &[f64]) -> f64 {
        self.tree.predict(xs)
    }

    // // TODO: rename
    // pub fn fold<F, LeafF, T, LeafT>(
    //     &self,
    //     init: T,
    //     mut leaf_init: LeafT,
    //     mut f: F,
    //     mut leaf_f: LeafF,
    // ) -> LeafT
    // where
    //     F: FnMut(T, &SplitPoint) -> (T, T),
    //     LeafF: FnMut(LeafT, T, f64) -> LeafT,
    // {
    //     let mut stack = vec![(&self.tree.root, init)];
    //     while let Some((node, acc)) = stack.pop() {
    //         if let Some(c) = &node.children {
    //             let (acc_l, acc_r) = f(acc, &c.split);
    //             stack.push((&c.left, acc_l));
    //             stack.push((&c.right, acc_r));
    //         } else {
    //             leaf_init = leaf_f(leaf_init, acc, node.label);
    //         }
    //     }
    //     leaf_init
    // }
}

#[derive(Debug)]
pub struct Tree {
    root: Node,
}

impl Tree {
    pub fn fit<'a, R: Rng + ?Sized>(
        rng: &mut R,
        mut table: Table<'a>,
        options: DecisionTreeOptions,
    ) -> Self {
        let mut builder = NodeBuilder { rng, options };
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

// impl Node {
//     fn new(label: f64) -> Self {
//         Self {
//             label,
//             children: None,
//         }
//     }

//     fn predict(&self, xs: &[f64]) -> f64 {
//         if let Some(children) = &self.children {
//             if xs[children.split.column] <= children.split.threshold {
//                 children.left.predict(xs)
//             } else {
//                 children.right.predict(xs)
//             }
//         } else {
//             self.label
//         }
//     }
// }

#[derive(Debug)]
pub struct Children {
    split: SplitPoint,
    left: Box<Node>,
    right: Box<Node>,
}

#[derive(Debug)]
pub struct SplitPoint {
    pub information_gain: f64,
    pub column: usize,
    pub threshold: f64,
}

#[derive(Debug)]
struct NodeBuilder<R> {
    rng: R,
    options: DecisionTreeOptions,
}

impl<R: Rng> NodeBuilder<R> {
    fn build(&mut self, table: &mut Table) -> Node {
        if table.is_single_target() {
            let label = table.target().nth(0).expect("never fails");
            return Node::new(label);
        }

        let label = functions::mean(table.target());

        let mut node = Node::new(label);
        let impurity = functions::mse(table.target());
        let rows = table.target().count();

        let max_features = self
            .options
            .max_features
            .unwrap_or_else(|| table.features_len());
        let columns = (0..table.features_len())
            .filter(|&i| !table.feature(i).any(|f| f.is_nan()))
            .collect::<Vec<_>>();

        let mut best: Option<SplitPoint> = None;
        for &column in
            columns.choose_multiple(&mut self.rng, std::cmp::min(columns.len(), max_features))
        {
            table.sort_rows_by_feature(column);
            for (row, threshold) in table.thresholds(column) {
                let impurity_l = functions::mse(table.target().take(row));
                let impurity_r = functions::mse(table.target().skip(row));
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

        if let Some(best) = best {
            node.children = Some(self.build_children(table, best));
        }
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

// #[cfg(test)]
// mod tests {
//     use super::*;
//     use rand;

//     #[test]
//     fn regression_works() -> Result<(), anyhow::Error> {
//         let features = vec![
//             &[
//                 0.0, 0.0, 1.0, 2.0, 2.0, 2.0, 1.0, 0.0, 0.0, 2.0, 0.0, 1.0, 1.0, 2.0,
//             ],
//             &[
//                 2.0, 2.0, 2.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 2.0, 1.0,
//             ],
//             &[
//                 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0,
//             ],
//             &[
//                 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0,
//             ],
//         ];
//         let target = &[
//             25.0, 30.0, 46.0, 45.0, 52.0, 23.0, 43.0, 35.0, 38.0, 46.0, 48.0, 52.0, 44.0, 30.0,
//         ];
//         let train_len = target.len() - 2;

//         let table = Table::new(
//             features.iter().map(|f| &f[..train_len]).collect(),
//             &target[..train_len],
//         )?;

//         let regressor =
//             DecisionTreeRegressor::fit(&mut rand::thread_rng(), table, Default::default());
//         assert_eq!(
//             regressor.predict(&features.iter().map(|f| f[train_len]).collect::<Vec<_>>()),
//             46.0
//         );
//         assert_eq!(
//             regressor.predict(
//                 &features
//                     .iter()
//                     .map(|f| f[train_len + 1])
//                     .collect::<Vec<_>>()
//             ),
//             52.0
//         );

//         Ok(())
//     }
// }
