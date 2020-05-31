use anyhow::ensure;
use fanova::table::Table;
use serde::Deserialize;
use std::ops::Range;
use structopt::StructOpt;

#[derive(Debug, Deserialize)]
struct Column {
    name: String,
    // TODO: type: categorical|numerical
    // TODO: distribution: uniform | log-uniform
    low: f64,
    high: f64,
    data: Vec<f64>,
}

#[derive(Debug, StructOpt)]
struct Opt {
    // TODO: target_clip{low,high}
// TODO: categorical_encoding
// TODO: target_column
}

fn main() -> anyhow::Result<()> {
    let _opt = Opt::from_args();
    let columns: Vec<Column> = serde_json::from_reader(std::io::stdin().lock())?;
    ensure!(columns.len() > 2, "too few columns");

    let features = columns
        .iter()
        .take(columns.len() - 1)
        .map(|c| c.data.as_slice())
        .collect();
    let target = &columns[columns.len() - 1].data;
    let table = Table::new(features, target)?;

    let config_space = columns
        .iter()
        .take(columns.len() - 1)
        .map(|c| Range {
            start: c.low,
            end: c.high,
        })
        .collect();
    let importances = fanova::importance::quantify_importance(config_space, table);

    let result = columns
        .iter()
        .zip(importances.iter())
        .map(|(c, v)| (&c.name, v))
        .collect::<std::collections::BTreeMap<_, _>>();
    serde_json::to_writer_pretty(std::io::stdout().lock(), &result)?;

    Ok(())
}
