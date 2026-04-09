pub mod adamw;
pub mod sgd;

pub use adamw::adamw;
pub use sgd::sgd;

use sequential::Sequential;

pub trait Optimizer {
    fn step(&self, &mut model: Sequential);
    fn zero_grad(&self, &mut model: Sequential);
}
