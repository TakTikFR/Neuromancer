use crate::tensor::Tensor;
use candle_core::Result;

pub fn l1(probs: &Tensor, targets: &Tensor) -> Result<(Tensor, Tensor)> {
    let diff = (probs - targets)?;
    let loss = diff.abs()?.mean_all()?;

    let n = targets.elem_count() as f64;
    let grad = (diff.sign()? / n)?;

    Ok((loss, grad))
}
