use crate::layers::Layer;
use crate::tensor::Tensor;
use candle_core::Result;

pub struct Tanh {
    output_cache: Option<Tensor>,
}

impl Tanh {
    pub fn new() -> Self {
        Self { output_cache: None }
    }
}

impl Layer for Tanh {
    fn forward(&mut self, input: &Tensor) -> Result<Tensor> {
        let output =
            ((input.exp()? - input.neg()?.exp()?)? / (input.exp()? + input.neg()?.exp()?)?)?;
        self.output_cache = Some(output.clone());
        Ok(output)
    }

    fn backward(&mut self, grad_output: &Tensor) -> Result<Tensor> {
        let output = self.output_cache.as_ref().unwrap();
        let dtanh = output.sqr()?.neg()? + 1.0;
        grad_output * dtanh
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
