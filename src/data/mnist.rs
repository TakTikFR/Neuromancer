use candle_core::{DType, Device, Result, Tensor};

pub struct MnistDataset {
    pub train_images: Tensor,
    pub train_labels: Tensor,
    pub test_images: Tensor,
    pub test_labels: Tensor,
}

pub fn load(device: &Device) -> Result<MnistDataset> {
    let dataset = candle_datasets::vision::mnist::load().map_err(candle_core::Error::wrap)?;

    Ok(MnistDataset {
        train_images: dataset.train_images.to_device(device)?,
        train_labels: dataset
            .train_labels
            .to_dtype(DType::U32)?
            .to_device(device)?,
        test_images: dataset.test_images.to_device(device)?,
        test_labels: dataset
            .test_labels
            .to_dtype(DType::U32)?
            .to_device(device)?,
    })
}

pub fn one_hot(labels: &Tensor, num_classes: usize) -> Result<Tensor> {
    let device = labels.device();

    let mut eye_data = vec![0f32; num_classes * num_classes];
    for i in 0..num_classes {
        eye_data[i * num_classes + i] = 1.0;
    }
    let eye = Tensor::from_vec(eye_data, (num_classes, num_classes), device)?;

    eye.index_select(&labels.to_dtype(DType::U32)?, 0)
}
