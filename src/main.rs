//use fanova::dataframe::{DataFrame, Series};
//use std::fs::File;
use std::path::PathBuf;
use structopt::StructOpt;

#[derive(Debug, StructOpt)]
struct Opt {
    x_training_set: PathBuf,
    y_training_set: PathBuf,
    y_test_set: PathBuf,
}

fn main() -> anyhow::Result<()> {
    // let opt = Opt::from_args();
    // let x_training_set: DataFrame = serde_json::from_reader(File::open(opt.x_training_set)?)?;
    // let y_training_set: Series = serde_json::from_reader(File::open(opt.y_training_set)?)?;
    // let y_test_set: Series = serde_json::from_reader(File::open(opt.y_test_set)?)?;
    // println!("{:?}", y_test_set);
    Ok(())
}
