pub fn combinations(items: impl IntoIterator<Item = usize>, k: usize) -> Combinations {
    let items = items.into_iter().collect::<Vec<_>>();
    let indices = (0..k).collect();
    Combinations {
        items,
        indices,
        k,
        first: true,
    }
}

pub struct Combinations {
    items: Vec<usize>,
    indices: Vec<usize>,
    k: usize,
    first: bool,
}

impl Iterator for Combinations {
    type Item = Vec<usize>;

    fn next(&mut self) -> Option<Self::Item> {
        let n = self.items.len();
        if self.k > n {
            return None;
        }
        if self.first {
            self.first = false;
            return Some(self.indices.iter().map(|&i| self.items[i]).collect());
        }
        if self.k == 0 {
            return None;
        }
        for i in (0..self.k).rev() {
            if self.indices[i] < n - self.k + i {
                self.indices[i] += 1;
                for j in i + 1..self.k {
                    self.indices[j] = self.indices[j - 1] + 1;
                }
                return Some(self.indices.iter().map(|&idx| self.items[idx]).collect());
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn combinations_works() {
        let collect = |it: Combinations| it.collect::<Vec<Vec<usize>>>();

        assert_eq!(collect(combinations(0..3, 0)), vec![vec![]]);
        assert_eq!(
            collect(combinations(0..3, 1)),
            vec![vec![0], vec![1], vec![2]]
        );
        assert_eq!(
            collect(combinations(0..3, 2)),
            vec![vec![0, 1], vec![0, 2], vec![1, 2]]
        );
        assert_eq!(collect(combinations(0..3, 3)), vec![vec![0, 1, 2]]);
        assert_eq!(collect(combinations(0..3, 4)), Vec::<Vec<usize>>::new());
        assert_eq!(collect(combinations(0..0, 0)), vec![vec![]]);
    }
}
