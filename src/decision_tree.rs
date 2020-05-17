use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use ordered_float::OrderedFloat;
use std::iter::FromIterator as _;
use thiserror::Error;

#[derive(Debug)]
pub struct TreeOptions<C> {
    criterion: C,
}

impl<C> TreeOptions<C>
where
    C: Criterion,
{
    pub fn new(criterion: C) -> Self {
        Self { criterion }
    }

    pub fn fit(
        &self,
        features: &ArrayView2<f64>,
        target: &ArrayView1<f64>,
    ) -> Result<Tree, FitError> {
        let node = self.build_node(features, target)?;
        Ok(Tree { root: node })
    }

    fn build_node(
        &self,
        features: &ArrayView2<f64>,
        target: &ArrayView1<f64>,
    ) -> Result<Node, FitError> {
        if target.is_empty() {
            return Err(FitError::EmptyTarget);
        }

        if features.shape()[0] != target.len() {
            return Err(FitError::SampleSizeMismatched);
        }

        if is_single(target) {
            return Ok(Node::new(target[0]));
        }

        let label = if self.criterion.kind() == TreeKind::Classification {
            most_frequent_class(target)
        } else {
            target.mean().expect("never fails")
        };

        let mut node = Node::new(label);
        let impurity = self.criterion.calculate(target);

        let mut best = None;
        struct Best {
            information_gain: f64,
            feature: usize,
            threshold: f64,
        }

        for col in 0..features.shape()[1] {
            let features = features.index_axis(Axis(1), col);
            for threshold in self.thresholds(&features) {
                let (target_l, target_r) = features.iter().zip(target.iter()).fold(
                    (Vec::new(), Vec::new()),
                    |(mut l, mut r), (&x, &y)| {
                        if x <= threshold {
                            l.push(y);
                        } else {
                            r.push(y);
                        }
                        (l, r)
                    },
                );
                let impurity_l = self.criterion.calculate(&ArrayView1::from(&target_l));
                let impurity_r = self.criterion.calculate(&ArrayView1::from(&target_r));
                let n_l = target_l.len() as f64 / target.len() as f64;
                let n_r = target_r.len() as f64 / target.len() as f64;

                let information_gain = impurity - (n_l * impurity_l + n_r * impurity_r);
                if best
                    .as_ref()
                    .map_or(true, |t: &Best| t.information_gain < information_gain)
                {
                    best = Some(Best {
                        information_gain,
                        feature: col,
                        threshold,
                    });
                }
            }
        }

        let best = best.expect("never fails");
        node.children =
            Some(self.build_children(features, target, best.feature, best.threshold)?);
        Ok(node)
    }

    fn build_children(
        &self,
        features: &ArrayView2<f64>,
        target: &ArrayView1<f64>,
        feature_index: usize,
        threshold: f64,
    ) -> Result<Children, FitError> {
        let features_l = self.filter_features(features, feature_index, |x| x <= threshold)?;
        let target_l = self.filter_target(target, features, feature_index, |x| x <= threshold)?;
        let left = self.build_node(&features_l.view(), &target_l.view())?;

        let features_r = self.filter_features(features, feature_index, |x| x > threshold)?;
        let target_r = self.filter_target(target, features, feature_index, |x| x > threshold)?;
        let right = self.build_node(&features_r.view(), &target_r.view())?;

        Ok(Children {
            feature_index,
            threshold,
            left: Box::new(left),
            right: Box::new(right),
        })
    }

    fn filter_features<F>(
        &self,
        xss: &ArrayView2<f64>,
        index: usize,
        f: F,
    ) -> Result<Array2<f64>, ndarray::ShapeError>
    where
        F: Fn(f64) -> bool,
    {
        let mut rows = 0;
        let mut storage = Vec::<f64>::new();
        for xs in xss.genrows() {
            if f(xs[index]) {
                storage.extend(xs);
                rows += 1;
            }
        }
        Array2::from_shape_vec((rows, xss.shape()[1]), storage)
    }

    fn filter_target<F>(
        &self,
        ys: &ArrayView1<f64>,
        xss: &ArrayView2<f64>,
        index: usize,
        f: F,
    ) -> Result<Array1<f64>, ndarray::ShapeError>
    where
        F: Fn(f64) -> bool,
    {
        let mut rows = 0;
        let mut storage = Vec::<f64>::new();
        for (&y, xs) in ys.iter().zip(xss.genrows()) {
            if f(xs[index]) {
                storage.push(y);
                rows += 1;
            }
        }
        Array1::from_shape_vec((rows,), storage)
    }

    fn thresholds(&self, xs: &ArrayView1<f64>) -> Vec<f64> {
        let values = std::collections::BTreeSet::from_iter(xs.iter().copied().map(OrderedFloat));
        values
            .iter()
            .zip(values.iter().skip(1))
            .map(|(a, b)| (a.0 + b.0) / 2.0)
            .collect()
    }
}

#[derive(Debug)]
pub struct Children {
    feature_index: usize,
    threshold: f64,
    left: Box<Node>,
    right: Box<Node>,
}

#[derive(Debug)]
pub struct Node {
    label: f64,
    children: Option<Children>,
}

impl Node {
    pub fn new(label: f64) -> Self {
        Self {
            label,
            children: None,
        }
    }

    fn predict(&self, xs: &ArrayView1<f64>) -> f64 {
        if let Some(children) = &self.children {
            if xs[children.feature_index] <= children.threshold {
                children.left.predict(xs)
            } else {
                children.right.predict(xs)
            }
        } else {
            self.label
        }
    }
}

fn is_single(target: &ArrayView1<f64>) -> bool {
    let x = target[0];
    target.iter().skip(1).all(|&y| x == y)
}

fn most_frequent_class(ys: &ArrayView1<f64>) -> f64 {
    let mut counter = std::collections::HashMap::<_, usize>::new();
    for y in ys {
        *counter.entry(OrderedFloat(*y)).or_default() += 1;
    }

    counter
        .iter()
        .max_by_key(|t| t.1)
        .expect("never fails")
        .0
         .0
}

#[derive(Debug)]
pub struct Tree {
    root: Node,
}

impl Tree {
    pub fn predict(&self, xs: &ArrayView1<f64>) -> f64 {
        self.root.predict(xs)
    }
}

pub trait Criterion {
    fn calculate(&self, target: &ArrayView1<f64>) -> f64;
    fn kind(&self) -> TreeKind;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TreeKind {
    Classification,
    Regression,
}

#[derive(Debug, Error)]
pub enum FitError {
    #[error("target data is empty")]
    EmptyTarget,

    #[error("the sample counts of features and target are mismatched")]
    SampleSizeMismatched,

    #[error("invalid array shapse")]
    InvalidShape(#[from] ndarray::ShapeError),
}
