use crate::layers::Layer;
use crate::tensor::Tensor;
use candle_core::Result;

pub struct Softmax {
    probs: Option<Tensor>,
}

impl Softmax {
    pub fn new() -> Self {
        Self { probs: None }
    }
}

impl Layer for Softmax {
    fn forward(&mut self, input: &Tensor) -> Result<Tensor> {
        let max = input.max_keepdim(1)?;
        let input_sub = input.broadcast_sub(&max)?;
        let exp = input_sub.exp()?;
        let exp_sum = exp.sum_keepdim(1)?;
        let p = exp.broadcast_div(&exp_sum)?;

        self.probs = Some(p.clone());
        Ok(p)
    }

    fn backward(&mut self, grad_output: &Tensor) -> Result<Tensor> {
        let probs = self.probs.as_ref().unwrap();
        let dot = (grad_output * probs)?.sum_keepdim(1)?;
        let diff = grad_output.broadcast_sub(&dot)?;
        probs * diff
    }

    fn params(&mut self) -> Vec<&mut Tensor> {
        vec![]
    }

    fn grads(&self) -> Vec<&Tensor> {
        vec![]
    }

    fn zero_grad(&mut self) -> Result<()> {
        Ok(())
    }
}
