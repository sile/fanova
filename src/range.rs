use thiserror::Error;

#[derive(Debug, Clone)]
pub struct Range {
    start: f64,
    end: f64,
}

impl Range {
    pub fn new(start: f64, end: f64) -> Result<Self, RangeError> {
        if !start.is_finite() || !end.is_finite() {
            Err(RangeError::NonFiniteValue)
        } else if start > end {
            Err(RangeError::NegativeRange)
        } else {
            Ok(Range { start, end })
        }
    }

    pub const fn start(&self) -> f64 {
        self.start
    }

    pub const fn end(&self) -> f64 {
        self.end
    }
}

#[derive(Debug, Error, Clone)]
pub enum RangeError {
    #[error("the start and end of a range must be finite numbers")]
    NonFiniteValue,

    #[error("the start of a range must be less than or equal to the end")]
    NegativeRange,
}
