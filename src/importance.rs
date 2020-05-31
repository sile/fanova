use crate::partitioning::TreePartitioning;
use crate::random_forest::RandomForestRegressor;
use crate::table::Table;
use ordered_float::OrderedFloat;
use rand;
use std::collections::BTreeMap;
use std::ops::Range;

pub fn quantify_importance(config_space: Vec<Range<f64>>, table: Table) -> Vec<f64> {
    let mut importances = vec![0.0; table.features_len()];
    let regressor = RandomForestRegressor::fit(&mut rand::thread_rng(), table, Default::default());
    for tree in regressor.forest().iter() {
        let partitioning = TreePartitioning::new(tree, config_space.clone());
        let mean = partitioning.mean(); // TODO: optimize
        let total_variance = partitioning.variance();
        for (i, u) in config_space.iter().enumerate() {
            let subspaces = subspaces(partitioning.partitions().map(|p| p.config_space[i].clone()));
            let variance = subspaces
                .map(|s| {
                    let v = partitioning.marginal_predict2(i, &s);
                    (v - mean).powi(2) * (s.end - s.start)
                })
                .sum::<f64>();
            let v = variance / (u.end - u.start) / total_variance; // TODO: Also save standard deviation.
            importances[i] += v;
        }
    }

    importances
        .iter_mut()
        .for_each(|v| *v /= regressor.forest().len() as f64);
    let sum = importances.iter().map(|&v| v).sum::<f64>();
    importances.iter().map(|&v| v / sum).collect()
}

fn subspaces(partitions: impl Iterator<Item = Range<f64>>) -> impl Iterator<Item = Range<f64>> {
    let mut subspaces = BTreeMap::new();
    for p in partitions {
        insert_subspace(&mut subspaces, p);
    }
    subspaces.into_iter().map(|(_, v)| v)
}

fn insert_subspace(subspaces: &mut BTreeMap<OrderedFloat<f64>, Range<f64>>, mut p: Range<f64>) {
    if p.start == p.end {
        return;
    }

    if let Some(mut q) = subspaces
        .range(..=OrderedFloat(p.start))
        .rev()
        .nth(0)
        .map(|(_, q)| q.clone())
    {
        if q.start == p.start {
            if q.end > p.end {
                subspaces.remove(&OrderedFloat(q.start));

                q.start = p.end;
                subspaces.insert(OrderedFloat(p.start), p);
                subspaces.insert(OrderedFloat(q.start), q);
            } else {
                assert!(q.end <= p.end);
                p.start = q.end;
                insert_subspace(subspaces, p);
            }
        } else {
            assert!(q.start < p.start);
            if q.end > p.end {
                subspaces.remove(&OrderedFloat(q.start));

                let r = Range {
                    start: p.end,
                    end: q.end,
                };
                q.end = p.start;
                subspaces.insert(OrderedFloat(q.start), q);
                subspaces.insert(OrderedFloat(p.start), p);
                subspaces.insert(OrderedFloat(r.start), r);
            } else {
                assert!(q.end <= p.end);
                subspaces.remove(&OrderedFloat(q.start));

                let r = Range {
                    start: q.end,
                    end: p.end,
                };
                q.end = p.start;
                p.end = r.start;
                subspaces.insert(OrderedFloat(q.start), q);
                subspaces.insert(OrderedFloat(p.start), p);
                insert_subspace(subspaces, r);
            }
        }
    } else {
        subspaces.insert(OrderedFloat(p.start), p);
    }
}