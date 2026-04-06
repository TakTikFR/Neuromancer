use candle_core::Result;
use crate::tensor::Tensor;

pub trait Layer {
    fn forward(&mut self, input: &Tensor) -> Result<Tensor>;
    fn backward(&mut self, grad_output: &Tensor) -> Result<Tensor>;
    fn params_and_grads(&mut self) -> Vec<(&Tensor, &Tensor)>;
}
