use crate::layers::Layer;
use crate::tensor::Tensor;
use candle_core::Result;

pub struct Softmax;

impl Softmax {
    pub fn new() -> Self {
        Self
    }
}

impl Layer for Softmax {
    fn forward(&mut self, input: &Tensor) -> Result<Tensor> {
        let max = input.max_keepdim(1)?;
        let input_sub = input.broadcast_sub(&max)?;
        let exp = input_sub.exp()?;
        let exp_sum = exp.sum_keepdim(1)?;
        exp.broadcast_div(&exp_sum)
    }

    fn backward(&mut self, _grad_output: &Tensor) -> Result<Tensor> {
        unimplemented!("Softmax backward handled by the loss")
    }

    fn params_and_grads(&mut self) -> Vec<(&Tensor, &Tensor)> {
        vec![]
    }
}
