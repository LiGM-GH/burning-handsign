#![allow(warnings)]

use burn::{
    backend::{Autodiff, Wgpu},
    data::dataset::{Dataset, vision::ImageFolderDataset},
    optim::AdamConfig,
    tensor::{Shape, Tensor, TensorData},
};

use dataset::HandsignDataset;
use model::{ModelConfig, train};

mod dataset;
mod model;

fn main() {
    type MyBackend = Wgpu<f32, i32>;

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
        .reduce(|val1, val2| val1 + val2 - mean.clone())
        .unwrap();

    println!(
        "stddev: {:?}",
        stddev.clone().mean().to_data().as_slice::<f32>().unwrap()[0]
    );

    train::<Autodiff<MyBackend>>(
        "artifacts",
        model::TrainingConfig::new(
            ModelConfig::new(3, 1, 16),
            AdamConfig::new(),
        ),
        &device,
    );
}
