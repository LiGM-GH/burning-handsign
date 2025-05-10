#![allow(warnings)]

use burn::{
    backend::{Autodiff},
    data::dataset::{Dataset, vision::ImageFolderDataset},
    optim::AdamConfig,
    tensor::{Shape, Tensor, TensorData},
};

use dataset::HandsignDataset;
use model::{ModelConfig, train};

mod dataset;
mod model;

fn mean_std() {
    type MyBackend = burn::backend::LibTorch;

    let device = Default::default();

    let model = ModelConfig::new(512, 3, 3).init::<MyBackend>(&device);
    println!("{}", model);
    let dataset = ImageFolderDataset::hs_train();
    println!("{}", dataset.len());
    let ds_outer = dataset
        .iter()
        .map(|val| {
            TensorData::new(
                val.image
                    .iter()
                    .map(|val| -> u8 { val.clone().try_into().unwrap() })
                    .collect::<Vec<_>>(),
                Shape::new([820, 200, 1]),
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
    let stddev = ds_outer
        .iter()
        .cloned()
        .fold(Tensor::zeros(mean.shape(), &device), |acc, val| {
            acc + (val - mean.clone()).powi_scalar(2)
        })
        .sqrt();

    println!(
        "stddev: {:?}",
        stddev.clone().mean().to_data().as_slice::<f32>().unwrap()[0]
    );
}

fn learn() {
    type MyBackend = burn::backend::LibTorch;

    let device = Default::default();

    train::<Autodiff<MyBackend>>(
        "artifacts",
        model::TrainingConfig::new(
            ModelConfig::new(3, 1, 16),
            AdamConfig::new(),
        ).with_learning_rate(1.5),
        &device,
    );
}

fn main() {
    let next = std::env::args().nth(1);
    let next = next.as_deref().map(|val| val.trim());
    println!("Input: {:?}", next);

    match next {
        Some("mean_std") => mean_std(),
        Some("learn") => learn(),
        _ => println!("This doens't work that way"),
    }
}
