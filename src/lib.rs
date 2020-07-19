pub use self::fanova::{Fanova, FanovaOptions, FitError, Importance};
pub use self::random_forest::RandomForestOptions;
pub use self::space::{FeatureSpace, FeatureSpaceError};

mod decision_tree;
mod fanova;
mod functions;
mod partition;
mod random_forest;
mod space;
mod table;
