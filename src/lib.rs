pub use importance::{Fanova, FanovaOptions};
pub use random_forest::RandomForestOptions;
pub use space::{FeatureSpace, FeatureSpaceError};

pub mod fanova;

mod decision_tree;
mod functions;
mod importance;
mod partition;
mod random_forest;
mod space;
mod table;
