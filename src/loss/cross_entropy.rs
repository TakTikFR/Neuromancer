use crate::tensor::Tensor;
use candle_core::Result;

pub fn cross_entropy(probs: &Tensor, targets: &Tensor) -> Result<(Tensor, Tensor)> {
    let loss = (targets * (probs + 1e-8)?.log()?)?
        .sum_keepdim(1)?
        .mean_all()?
        .neg()?;

    let n = probs.dim(0)? as f64;
    let grad = (targets / (probs * n)?)?.neg()?;

    Ok((loss, grad))
}
