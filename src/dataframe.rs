use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Series {
    name: String,
    index: Vec<usize>,
    data: Vec<f64>,
}

impl Series {
    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn index(&self) -> &[usize] {
        &self.index
    }

    pub fn data(&self) -> &[f64] {
        &self.data
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataFrame {
    columns: Vec<String>,
    index: Vec<usize>,
    data: Vec<Vec<f64>>,
}

impl DataFrame {
    pub fn columns(&self) -> &[String] {
        &self.columns
    }

    pub fn index(&self) -> &[usize] {
        &self.index
    }

    pub fn data(&self) -> &[Vec<f64>] {
        &self.data
    }
}
