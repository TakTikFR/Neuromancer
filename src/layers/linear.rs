use crate::layers::Layer;
use crate::tensor::{DType, Device, Tensor};
use candle_core::Result;

pub struct Linear {
    weights: Tensor,
    bias: Tensor,
    grad_weights: Option<Tensor>,
    grad_bias: Option<Tensor>,
    input_cache: Option<Tensor>,
    device: Device,
}

impl Linear {
    pub fn new(input_size: usize, output_size: usize, device: &Device) -> Result<Self> {
        let std_dev = (2.0 / input_size as f64).sqrt();
        let weights = Tensor::randn(0f32, std_dev as f32, (output_size, input_size), device)?;
        let bias = Tensor::zeros((output_size,), DType::F32, device)?;

        Ok(Self {
            weights,
            bias,
            grad_weights: None,
            grad_bias: None,
            input_cache: None,
            device: device.clone(),
        })
    }
}

impl Layer for Linear {
    fn forward(&mut self, input: &Tensor) -> Result<Tensor> {
        self.input_cache = Some(input.clone());
        let out = input.matmul(&self.weights.t()?)?;
        out.broadcast_add(&self.bias)
    }

    fn backward(&mut self, grad_output: &Tensor) -> Result<Tensor> {
        let input = self.input_cache.as_ref().unwrap();

        self.grad_weights = Some(grad_output.t()?.matmul(input)?);
        self.grad_bias = Some(grad_output.sum(0)?);

        grad_output.matmul(&self.weights)
    }

    fn params(&mut self) -> Vec<&mut Tensor> {
        vec![&mut self.weights, &mut self.bias]
    }

    fn grads(&self) -> Vec<Tensor> {
        let mut result = Vec::new();
        if let (Some(gw), Some(gb)) = (&self.grad_weights, &self.grad_bias) {
            result.push(gw.clone());
            result.push(gb.clone());
        }
        result
    }

    fn zero_grad(&mut self) -> Result<()> {
        self.grad_weights = Some(self.weights.zeros_like()?);
        self.grad_bias = Some(self.bias.zeros_like()?);
        Ok(())
    }
}
