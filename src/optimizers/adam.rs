use crate::layers::Sequential;
use crate::optimizers::Optimizer;
use crate::tensor::Tensor;
use candle_core::Result;

pub struct Adam {
    lr: f64,
    beta1: f64,
    beta2: f64,
    epsilon: f64,
    mt_1: Vec<Tensor>,
    vt_1: Vec<Tensor>,
    t: usize,
}

impl Adam {
    pub fn new(learning_rate: f64) -> Self {
        Self {
            lr: learning_rate,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            mt_1: Vec::new(),
            vt_1: Vec::new(),
            t: 0,
        }
    }

    pub fn betas(mut self, betas: (f64, f64)) -> Self {
        self.beta1 = betas.0;
        self.beta2 = betas.1;
        self
    }

    pub fn epsilon(mut self, epsilon: f64) -> Self {
        self.epsilon = epsilon;
        self
    }
}

impl Optimizer for Adam {
    fn step(&mut self, model: &mut Sequential) -> Result<()> {
        for layer in model.layers_mut() {
            let grads = layer.grads().to_vec();
            let params = layer.params();

            for (idx, (param, grad)) in params.into_iter().zip(grads.iter()).enumerate() {
                if self.mt_1.len() <= idx {
                    self.mt_1.push(Tensor::zeros_like(param)?);
                    self.vt_1.push(Tensor::zeros_like(param)?);
                }

                self.t += 1;
                let mt = ((self.beta1 * &self.mt_1[idx])? + ((1.0 - self.beta1) * grad)?)?;
                let vt =
                    ((self.beta2 * &self.vt_1[idx])? + ((1.0 - self.beta2) * grad.powf(2.0)?)?)?;
                let mt_hat = (mt / (1.0 - self.beta1.powi(self.t as i32)))?;
                let vt_hat = (vt / (1.0 - self.beta2.powi(self.t as i32)))?;
                *param = (&*param - ((self.lr * mt_hat)? / (vt_hat.sqrt()? + self.epsilon)?)?)?;
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
