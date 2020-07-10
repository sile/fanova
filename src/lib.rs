pub use fanova::{Fanova, FanovaOptions};
pub use random_forest::RandomForestOptions;
pub use space::{FeatureSpace, FeatureSpaceError};

mod decision_tree;
mod fanova;
mod functions;
mod partition;
mod random_forest;
mod space;
mod table;
