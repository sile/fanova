use ordered_float::OrderedFloat;
use rand::Rng;
use std::ops::Range;
use thiserror::Error;

#[derive(Debug, Clone)]
pub struct Table<'a> {
    row_index: Vec<usize>,
    row_range: Range<usize>,
    columns: Vec<&'a [f64]>,
}

impl<'a> Table<'a> {
    pub fn new(columns: Vec<&'a [f64]>) -> Result<Self, TableError> {
        if columns.is_empty() || columns[0].is_empty() {
            return Err(TableError::EmptyTable);
        }

        let rows_len = columns[0].len();
        if columns.iter().skip(1).any(|c| c.len() != rows_len) {
            return Err(TableError::RowSizeMismatch);
        }

        if columns[columns.len() - 1].iter().any(|t| !t.is_finite()) {
            return Err(TableError::NonFiniteTarget);
        }

        Ok(Self {
            row_index: (0..rows_len).collect(),
            row_range: Range {
                start: 0,
                end: rows_len,
            },
            columns,
        })
    }

    pub fn target(&self) -> impl '_ + Iterator<Item = f64> + Clone {
        self.column(self.columns.len() - 1)
    }

    pub fn column(&self, column_index: usize) -> impl '_ + Iterator<Item = f64> + Clone {
        self.rows().map(move |i| self.columns[column_index][i])
    }

    pub fn features_len(&self) -> usize {
        self.columns.len() - 1
    }

    pub fn rows_len(&self) -> usize {
        self.row_range.end - self.row_range.start
    }

    fn rows(&self) -> impl '_ + Iterator<Item = usize> + Clone {
        self.row_index[self.row_range.start..self.row_range.end]
            .iter()
            .copied()
    }

    pub fn sort_rows_by_column(&mut self, column: usize) {
        let columns = &self.columns;
        (&mut self.row_index[self.row_range.start..self.row_range.end])
            .sort_by_key(|&x| OrderedFloat(columns[column][x]))
    }

    pub fn bootstrap_sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Self {
        let row_index = (0..self.rows_len())
            .map(|_| self.row_index[rng.gen_range(self.row_range.start, self.row_range.end)])
            .collect::<Vec<_>>();
        let row_range = Range {
            start: 0,
            end: self.rows_len(),
        };
        Self {
            row_index,
            row_range,
            columns: self.columns.clone(),
        }
    }

    pub fn thresholds(&self, column: usize) -> impl '_ + Iterator<Item = (usize, f64)> {
        // Assumption: `self.columns[column]` has been sorted.
        let column = self.columns[column];
        self.rows()
            .map(move |i| column[i])
            .enumerate()
            .scan(None, |prev, (i, x)| {
                if prev.is_none() {
                    *prev = Some(x);
                    Some(None)
                } else if *prev != Some(x) {
                    let y = prev.expect("never fails");
                    *prev = Some(x);
                    Some(Some((i, (x + y) / 2.0)))
                } else {
                    Some(None)
                }
            })
            .flatten()
    }

    pub fn with_split<F, T>(&mut self, row: usize, mut f: F) -> (T, T)
    where
        F: FnMut(&mut Self) -> T,
    {
        let row = row + self.row_range.start;
        let original = self.row_range.clone();

        self.row_range.end = row;
        let left = f(self);
        self.row_range.end = original.end;

        self.row_range.start = row;
        let right = f(self);
        self.row_range.start = original.start;

        (left, right)
    }
}

#[derive(Debug, Error, Clone)]
pub enum TableError {
    #[error("table must have at least one column and one row")]
    EmptyTable,

    #[error("some of columns have a different row count from others")]
    RowSizeMismatch,

    #[error("target column contains non finite numbers")]
    NonFiniteTarget,
}
