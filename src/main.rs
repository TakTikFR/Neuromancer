use candle_core::{D, DType, Device, Result};

use Neuromancer::activations::{ReLU, Softmax};
use Neuromancer::data::{DataLoader, load, one_hot};
use Neuromancer::layers::{linear::Linear, sequential::Sequential};
use Neuromancer::loss::cross_entropy::cross_entropy;
use Neuromancer::optimizers::Optimizer;
use Neuromancer::optimizers::sgd::SGD;

use indicatif::{ProgressBar, ProgressStyle};

fn main() -> Result<()> {
    let device = Device::Cpu;
    let num_classes = 10usize;
    let epochs = 20;

    // --- Loading MNIST Dataset ---
    println!("MNIST Loading...");
    let mnist = load(&device)?;
    println!(
        "Train: {:?} | Test: {:?}",
        mnist.train_images.shape(),
        mnist.test_images.shape()
    );

    // --- DataLoaders ---
    let train_loader = DataLoader::new(mnist.train_images, mnist.train_labels, 64, true);
    let test_loader = DataLoader::new(mnist.test_images, mnist.test_labels, 64, false);

    // --- Model : 784 → 128 → ReLU → 64 → ReLU → 10 → Softmax ---
    let mut model = Sequential::new();
    model.add(Linear::new(784, 128, &device)?);
    model.add(ReLU::new());
    model.add(Linear::new(128, 64, &device)?);
    model.add(ReLU::new());
    model.add(Linear::new(64, 10, &device)?);
    model.add(Softmax::new());

    // --- Optimizer SGD with momentum and Nesterov ---
    let mut optimizer = SGD::new(0.01).momentum(0.9).nesterov(true);

    // --- Training Loop ---
    for epoch in 0..epochs {
        let mut total_loss = 0f32;
        let mut n_batches = 0usize;

        let batches: Vec<_> = train_loader.iter().collect();
        let pb = ProgressBar::new(batches.len() as u64);
        pb.set_style(
            ProgressStyle::with_template(
                "{prefix} [{bar:40.white/black}] {pos}/{len} | loss: {msg}",
            )
            .unwrap()
            .progress_chars("█░░"),
        );
        pb.set_prefix(format!("Epoch {:2}/{}", epoch + 1, epochs));

        for batch in batches {
            let (images, labels) = batch?;
            let targets = one_hot(&labels, num_classes)?;

            let probs = model.forward(&images)?;
            let (loss, grad) = cross_entropy(&probs, &targets)?;

            model.backward(&grad)?;
            optimizer.step(&mut model)?;
            optimizer.zero_grad(&mut model)?;

            total_loss += loss.to_scalar::<f32>()?;
            n_batches += 1;

            pb.set_message(format!("{:.4}", total_loss / n_batches as f32));
            pb.inc(1);
        }

        // --- Accuracy for the test set ---
        let mut n_correct = 0f32;
        let mut n_total = 0usize;

        for batch in test_loader.iter() {
            let (images, labels) = batch?;
            let probs = model.forward(&images)?;
            let preds = probs.argmax(D::Minus1)?;
            let correct = preds
                .eq(&labels)?
                .to_dtype(DType::F32)?
                .sum_all()?
                .to_scalar::<f32>()?;
            n_correct += correct;
            n_total += images.dim(0)?;
        }

        let acc = 100.0 * n_correct / n_total as f32;
        pb.finish_with_message(format!(
            "{:.4} | Test Accuracy: {:.2}%",
            total_loss / n_batches as f32,
            acc
        ));
    }

    Ok(())
}
