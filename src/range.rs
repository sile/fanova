use thiserror::Error;

#[derive(Debug, Clone)]
pub struct ParamRange {
    pub(crate) start: f64,
    pub(crate) end: f64,
}

impl ParamRange {
    pub fn new(start: f64, end: f64) -> Result<Self, ParamRangeError> {
        if !start.is_finite() || !end.is_finite() {
            Err(ParamRangeError::NonFiniteValue)
        } else if start > end {
            Err(ParamRangeError::NegativeRange)
        } else {
            Ok(Self { start, end })
        }
    }

    pub const fn start(&self) -> f64 {
        self.start
    }

    pub const fn end(&self) -> f64 {
        self.end
    }

    pub fn contains(&self, other: &Self) -> bool {
        self.start <= other.start && other.end <= self.end
    }
}

#[derive(Debug, Error, Clone)]
pub enum ParamRangeError {
    #[error("the start and end of a range must be finite numbers")]
    NonFiniteValue,

    #[error("the start of a range must be less than or equal to the end")]
    NegativeRange,
}
