pub mod adam;
pub mod sgd;

pub use adam::Adam;
pub use sgd::SGD;

use crate::layers::Sequential;
use candle_core::Result;

pub trait Optimizer {
    fn step(&mut self, model: &mut Sequential) -> Result<()>;
    fn zero_grad(&mut self, model: &mut Sequential) -> Result<()>;
}
