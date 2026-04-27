use crate::tensor::Tensor;
use candle_core::Result;

pub fn mse(probs: &Tensor, targets: &Tensor) -> Result<(Tensor, Tensor)> {
    let diff = (probs - targets)?;
    let loss = diff.sqr()?.mean_all()?;

    let n = targets.elem_count() as f64;
    let grad = (diff * (2. / n))?;

    Ok((loss, grad))
}
