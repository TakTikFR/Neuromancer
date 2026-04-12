use candle_core::{Result, Tensor};
use rand::seq::SliceRandom;
use rand::thread_rng;

pub struct DataLoader {
    images: Tensor,
    labels: Tensor,
    pub batch_size: usize,
    shuffle: bool,
}

impl DataLoader {
    pub fn new(images: Tensor, labels: Tensor, batch_size: usize, shuffle: bool) -> Self {
        Self {
            images,
            labels,
            batch_size,
            shuffle,
        }
    }

    pub fn len(&self) -> usize {
        let n = self.images.dim(0).unwrap();
        (n + self.batch_size - 1) / self.batch_size
    }

    pub fn iter(&self) -> DataLoaderIter {
        let n = self.images.dim(0).unwrap();
        let mut indices: Vec<usize> = (0..n).collect();
        if self.shuffle {
            indices.shuffle(&mut thread_rng());
        }
        DataLoaderIter {
            images: &self.images,
            labels: &self.labels,
            batch_size: self.batch_size,
            indices,
            current: 0,
        }
    }
}

pub struct DataLoaderIter<'a> {
    images: &'a Tensor,
    labels: &'a Tensor,
    batch_size: usize,
    indices: Vec<usize>,
    current: usize,
}

impl<'a> Iterator for DataLoaderIter<'a> {
    type Item = Result<(Tensor, Tensor)>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current >= self.indices.len() {
            return None;
        }
        let end = (self.current + self.batch_size).min(self.indices.len());
        let batch_indices = &self.indices[self.current..end];
        self.current = end;

        let idx_data: Vec<u32> = batch_indices.iter().map(|&i| i as u32).collect();
        let idx_tensor = match Tensor::new(idx_data.as_slice(), self.images.device()) {
            Ok(t) => t,
            Err(e) => return Some(Err(e)),
        };

        let batch_images = match self.images.index_select(&idx_tensor, 0) {
            Ok(t) => t,
            Err(e) => return Some(Err(e)),
        };
        let batch_labels = match self.labels.index_select(&idx_tensor, 0) {
            Ok(t) => t,
            Err(e) => return Some(Err(e)),
        };

        Some(Ok((batch_images, batch_labels)))
    }
}
