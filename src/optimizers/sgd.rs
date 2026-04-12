use crate::layers::Sequential;
use crate::optimizers::Optimizer;
use crate::tensor::Tensor;
use candle_core::Result;

pub struct SGD {
    lr: f64,
    momentum: f64,
    nesterov: bool,
    velocity: Vec<Tensor>,
}

impl SGD {
    pub fn new(learning_rate: f64) -> Self {
        Self {
            lr: learning_rate,
            momentum: 0.0,
            nesterov: false,
            velocity: Vec::new(),
        }
    }

    pub fn momentum(mut self, momentum: f64) -> Self {
        self.momentum = momentum;
        self
    }

    pub fn nesterov(mut self, nesterov: bool) -> Self {
        self.nesterov = nesterov;
        self
    }
}

impl Optimizer for SGD {
    fn step(&mut self, model: &mut Sequential) -> Result<()> {
        let mut global_idx = 0;

        for layer in model.layers_mut() {
            let grads = layer.grads().to_vec();
            let params = layer.params();

            for (idx, (param, grad)) in params.into_iter().zip(grads.iter()).enumerate() {
                if self.velocity.len() <= global_idx {
                    self.velocity.push(Tensor::zeros_like(param)?);
                }

                self.velocity[global_idx] =
                    (self.momentum * (&self.velocity[global_idx] - (self.lr * grad)?)?)?;
                if self.nesterov {
                    *param = (&*param
                        + (self.momentum * (&self.velocity[global_idx] - (self.lr * grad)?)?)?)?;
                } else {
                    *param = (&*param + &self.velocity[global_idx])?;
                }

                global_idx += 1;
            }
        }

        Ok(())
    }

    fn zero_grad(&mut self, model: &mut Sequential) -> Result<()> {
        for layer in model.layers_mut() {
            layer.zero_grad()?;
        }

        Ok(())
    }
}
