use std::path::Path;

use burn::{
    data::dataset::{Dataset, vision::ImageFolderDataset},
    tensor::{Shape, Tensor, TensorData},
};

use crate::model::{IMAGE_DEPTH, IMAGE_HEIGHT, IMAGE_LENGTH, ModelConfig};

pub trait HandsignDataset {
    fn hs_path(path: impl AsRef<Path>) -> Self;
}

impl HandsignDataset for ImageFolderDataset {
    fn hs_path(path: impl AsRef<Path>) -> Self {
        Self::new_classification(path).unwrap()
    }
}

pub fn mean_std(dataset_dir: &str) {
    type MyBackend = burn::backend::LibTorch;

    let device = Default::default();

    let model = ModelConfig::new(512, 3, 3).init::<MyBackend>(&device);
    println!("{}", model);
    let dataset = ImageFolderDataset::hs_path(dataset_dir);
    println!("{}", dataset.len());

    let ds_outer = dataset
        .iter()
        .map(|val| {
            TensorData::new(
                val.image
                    .iter()
                    .map(|val| -> u8 { val.clone().try_into().unwrap() })
                    .collect::<Vec<_>>(),
                Shape::new([IMAGE_LENGTH, IMAGE_HEIGHT, IMAGE_DEPTH]),
            )
        })
        .map(|data| Tensor::<MyBackend, 3>::from_data(data, &device) / 255)
        .collect::<Vec<_>>();

    let mean = ds_outer
        .iter()
        .cloned()
        .reduce(|val1, val2| val1 + val2)
        .unwrap();

    println!(
        "mean: {:?}",
        mean.clone().mean().to_data().as_slice::<f32>().unwrap()[0]
    );

    let mean_shape = mean.shape();
    let stddev = ds_outer
        .iter()
        .cloned()
        .fold(Tensor::zeros(mean_shape, &device), |acc, val| {
            let thing = val - mean.clone();
            log::error!("STDDEV's STEP IS {}", thing);
            acc + (thing).powi_scalar(2)
        })
        .sqrt();

    let that_mean = stddev.clone().max().into_scalar();
    log::error!("STDDEV: {}", that_mean);
    let stddev = stddev.full_like(that_mean);

    println!(
        "stddev: {:?}",
        stddev.clone().mean().to_data().as_slice::<f32>().unwrap()[0]
    );
}
