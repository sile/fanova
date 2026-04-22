//! A Rust [fANOVA] (functional analysis of variance) implementation.
//!
//! fANOVA provides a way to calculate feature importance.
//!
//! # Examples
//!
//! ```
//! use fanova::{FanovaOptions, RandomForestOptions};
//! use rand::{RngExt, SeedableRng};
//!
//! let mut feature1 = Vec::new();
//! let mut feature2 = Vec::new();
//! let mut feature3 = Vec::new();
//! let mut target = Vec::new();
//!
//! let mut rng = rand::rngs::StdRng::seed_from_u64(0);
//! for _ in 0..100 {
//!     let f1 = rng.random();
//!     let f2 = rng.random();
//!     let f3 = rng.random();
//!     let t = f1 + f2 * 2.0 + f3 * 3.0;
//!
//!     feature1.push(f1);
//!     feature2.push(f2);
//!     feature3.push(f3);
//!     target.push(t);
//! }
//!
//! let mut fanova = FanovaOptions::new()
//!     .random_forest(RandomForestOptions::new().seed(0))
//!     .fit(vec![&feature1, &feature2, &feature3], &target).unwrap();
//! let importances = (0..3)
//!     .map(|i| fanova.quantify_importance(&[i]).mean)
//!     .collect::<Vec<_>>();
//!
//! assert_eq!(
//!     importances,
//!     vec![0.03949614161205558, 0.24001507447005044, 0.5934922097988682]
//! );
//! ```
//!
//! # References
//!
//! - [An Efficient Approach for Assessing Hyperparameter Importance][fANOVA]
//!
//! [fANOVA]: http://proceedings.mlr.press/v32/hutter14.html
//!
#![warn(missing_docs)]
pub use self::fanova::{Fanova, FanovaOptions, FitError, Importance};
pub use self::random_forest::RandomForestOptions;

mod combinations;
mod decision_tree;
mod fanova;
mod functions;
mod partition;
mod random_forest;
mod space;
mod table;
