use crate::layers::Layer;
use crate::tensor::Tensor;
use candle_core::Result;
use std::f64::consts::PI;

pub struct GeLU {
    input_cache: Option<Tensor>,
    tanh_cache: Option<Tensor>,
}

impl GeLU {
    pub fn new() -> Self {
        Self {
            input_cache: None,
            tanh_cache: None,
        }
    }
}

impl Layer for GeLU {
    fn forward(&mut self, input: &Tensor) -> Result<Tensor> {
        self.input_cache = Some(input.clone());

        let x3 = input.powf(3.0)?;
        let inner = (input + (x3 * 0.044715)?)?;
        let tanh_arg = (inner * (2.0 / std::f64::consts::PI).sqrt())?;
        let tanh_val = tanh_arg.tanh()?;

        self.tanh_cache = Some(tanh_val.clone());

        input * ((tanh_val + 1.0)? * 0.5)?
    }

    fn backward(&mut self, grad_output: &Tensor) -> Result<Tensor> {
        let x = self.input_cache.as_ref().unwrap();
        let t = self.tanh_cache.as_ref().unwrap();
        let c = (2.0 / PI).sqrt();

        let term1 = ((t + 1.0)? * 0.5)?;

        let t2 = (t.sqr()?.neg()? + 1.0)?;
        let x2 = ((x.sqr()? * 0.134145)? + 1.0)?;
        let term2 = (x * (t2 * (x2 * c)?)?)? * 0.5;

        let dgelu = (term1 + term2?)?;

        grad_output * dgelu
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
