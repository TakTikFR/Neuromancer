use crate::layers::Layer;
use crate::tensor::Tensor;
use candle_core::Result;

pub struct Sequential {
    layers: Vec<Box<dyn Layer>>,
}

impl Sequential {
    pub fn new() -> Self {
        Self { layers: Vec::new() }
    }

    pub fn add(&mut self, layer: impl Layer + 'static) {
        self.layers.push(Box::new(layer));
    }

    pub fn forward(&mut self, input: &Tensor) -> Result<Tensor> {
        let mut x = input.clone();

        for layer in self.layers.iter_mut() {
            x = layer.forward(&x)?;
        }
        Ok(x)
    }

    pub fn backward(&mut self, grad: &Tensor) -> Result<Tensor> {
        let mut y = grad.clone();

        for layer in self.layers.iter_mut().rev() {
            y = layer.backward(&y)?;
        }
        Ok(y)
    }

    pub fn layers_mut(&mut self) -> &mut Vec<Box<dyn Layer>> {
        &mut self.layers
    }
}
