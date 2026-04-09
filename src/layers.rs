pub mod linear;
pub mod sequential;

pub use linear::Linear;
pub use sequential::Sequential;

use crate::tensor::Tensor;
use candle_core::Result;

pub trait Layer {
    fn forward(&mut self, input: &Tensor) -> Result<Tensor>;
    fn backward(&mut self, grad_output: &Tensor) -> Result<Tensor>;
    pub fn params(&mut self) -> Vec<&mut Tensor>;
    pub fn grads(&self) -> Vec<&Tensor>;
}
