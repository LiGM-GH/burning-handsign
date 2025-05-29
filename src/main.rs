use burn::{
    backend::Autodiff,
    data::dataset::{Dataset, vision::ImageFolderDataset},
    optim::{AdamConfig, RmsPropConfig},
    prelude::Backend,
    tensor::{Shape, Tensor, TensorData},
};

use dataset::{HandsignDataset, IMAGE_DEPTH, IMAGE_HEIGHT, IMAGE_LENGTH};
use model::{ModelConfig, train};

mod dataset;
mod model;

fn mean_std() {
    type MyBackend = burn::backend::LibTorch;

    let device = Default::default();

    let model = ModelConfig::new(512, 3, 3).init::<MyBackend>(&device);
    println!("{}", model);
    let dataset = ImageFolderDataset::new_classification(
        "../handwritten-signatures-ver1/CEDAR_again_1/full_org",
    )
    .expect("Couldn't open the dataset");

    println!("{}", dataset.len());

    let ds_outer = dataset
        .iter()
        .filter(|val| !val.image_path.contains("forge"))
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

fn learn() {
    type MyBackend = burn::backend::LibTorch;

    let device = Default::default();

    train::<Autodiff<MyBackend>>(
        "artifacts",
        model::TrainingConfig::new(
            ModelConfig::new(3, 1, 16).with_dropout(0.5),
            RmsPropConfig::new().with_momentum(0.8).with_weight_decay(Some(
                burn::optim::decay::WeightDecayConfig { penalty: 0.0005 },
            )),
        )
        .with_learning_rate(1e-4)
        .with_num_epochs(20),
        &device,
    );
}

const MEAN_DS: f64 = 99.72897;
const STDDEV_DS: f64 = 16482.139;

fn guess() {
    type MyBackend = burn::backend::LibTorch;

    let device = <MyBackend as Backend>::Device::default();

    model::guess::<MyBackend>(
        "artifacts",
        device,
        "../handwritten-signatures-ver1/next-forge-220x155",
        MEAN_DS,
        STDDEV_DS,
    );
}

fn main() {
    let next = std::env::args().nth(1);
    let next = next.as_deref().map(|val| val.trim());
    println!("Input: {:?}", next);

    match next {
        Some("mean_std") => mean_std(),
        Some("learn") => learn(),
        Some("guess") => guess(),
        _ => println!("This doens't work that way"),
    }
}
