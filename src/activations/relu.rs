use crate::layers::Layer;
use crate::tensor::Tensor;
use candle_core::DType::F32;
use candle_core::Result;

pub struct ReLU {
    input_cache: Option<Tensor>,
}

impl ReLU {
    pub fn new() -> Self {
        Self { input_cache: None }
    }
}

impl Layer for ReLU {
    fn forward(&mut self, input: &Tensor) -> Result<Tensor> {
        self.input_cache = Some(input.clone());
        input.maximum(0f32)
    }

    fn backward(&mut self, grad_output: &Tensor) -> Result<Tensor> {
        let input = self.input_cache.as_ref().unwrap();
        grad_output.mul(&input.gt(0f32)?.to_dtype(F32)?)
    }

    fn params(&mut self) -> Vec<&mut Tensor> {
        vec![]
    }
    fn grads(&self) -> Vec<Tensor> {
        vec![]
    }

    fn zero_grad(&mut self) -> Result<()> {
        Ok(())
    }
}
