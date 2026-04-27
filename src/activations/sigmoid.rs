use crate::layers::Layer;
use crate::tensor::Tensor;
use candle_core::Result;

pub struct Sigmoid {
    output_cache: Option<Tensor>,
}

impl Sigmoid {
    pub fn new() -> Self {
        Self { output_cache: None }
    }
}

impl Layer for Sigmoid {
    fn forward(&mut self, input: &Tensor) -> Result<Tensor> {
        let output = (1.0 / (1.0 + input.neg()?.exp()?)?)?;
        self.output_cache = Some(output.clone());
        Ok(output)
    }

    fn backward(&mut self, grad_output: &Tensor) -> Result<Tensor> {
        let output = self.output_cache.as_ref().unwrap();
        let dsigmoid = (output * (output.neg()? + 1.0)?)?;
        grad_output * dsigmoid
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
