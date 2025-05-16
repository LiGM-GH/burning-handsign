use std::marker::PhantomData;

use burn::{
    data::{
        dataloader::{DataLoader, DataLoaderBuilder, batcher::Batcher},
        dataset::vision::{ImageDatasetItem, ImageFolderDataset},
    },
    nn::{
        Linear, LinearConfig, Relu,
        conv::{Conv2d, Conv2dConfig},
        loss::BinaryCrossEntropyLossConfig,
        pool::{MaxPool2d, MaxPool2dConfig},
    },
    optim::AdamConfig,
    prelude::*,
    record::CompactRecorder,
    tensor::backend::AutodiffBackend,
    train::{
        ClassificationOutput, Learner, LearnerBuilder, TrainOutput, TrainStep,
        ValidStep,
        metric::{AccuracyMetric, LossMetric},
    },
};
use nn::loss::CrossEntropyLossConfig;
use tap::{Pipe, Tap};

use crate::{
    dataset::HandsignDataset,
    model::{IMAGE_DEPTH, IMAGE_HEIGHT, IMAGE_LENGTH, normalizer::Normalizer},
};

#[derive(Debug, Clone)]
pub struct HandsignBatch<B: Backend> {
    pub images: Tensor<B, 4>,
    pub targets: Tensor<B, 1, Int>,
}

#[derive(Clone)]
pub struct HandsignBatcher<B: Backend> {
    device: B::Device,
}

impl<B: Backend> HandsignBatcher<B> {
    pub fn new(device: B::Device) -> Self {
        Self { device }
    }
}

impl<B: Backend> Batcher<ImageDatasetItem, HandsignBatch<B>>
    for HandsignBatcher<B>
{
    fn batch(&self, items: Vec<ImageDatasetItem>) -> HandsignBatch<B> {
        log::error!("This is how much images there is: {:?}", items.len());

        let mut all_alike = true;
        let mut images_iter = items.iter();
        let mut etalon = images_iter.next().unwrap();

        for image in images_iter {
            if image.image != etalon.image
                && image.image_path != etalon.image_path
            {
                log::error!("Not all images are the same!");
                all_alike = false;
                break;
            }
        }

        if all_alike {
            panic!("All are alike!");
        }

        let images: Vec<Tensor<B, 3>> = items
            .iter()
            .map(|val| {
                log::info!("Current image: {}", val.image_path);

                TensorData::new(
                    val.image
                        .iter()
                        .map(|val| -> u8 { val.clone().try_into().unwrap() })
                        .collect::<Vec<_>>(),
                    Shape::new([IMAGE_LENGTH, IMAGE_HEIGHT, IMAGE_DEPTH]),
                )
            })
            .map(|data| Tensor::<B, 3>::from_data(data, &self.device) / 255)
            .map(|tensor| {
                let tensor = tensor.swap_dims(0, 2).swap_dims(1, 2);
                log::error!("These are new dims: {:?}", tensor.dims());
                tensor
            })
            .collect::<Vec<_>>();

        log::error!("Images: {:?}", images);

        let mean = images
            .iter()
            .cloned()
            .reduce(|l, r| l + r)
            .unwrap()
            .div_scalar(items.len() as f64);

        log::error!("Mean: {}", mean);

        let mean_shape = mean.shape();
        let stddev = images
            .iter()
            .cloned()
            .fold(Tensor::zeros(mean_shape, &self.device), |acc, val| {
                let thing = val - mean.clone();
                log::error!("STDDEV's STEP IS {}", thing);
                acc + (thing).powi_scalar(2)
            })
            .sqrt();

        let that_mean = stddev.clone().max().into_scalar();
        log::error!("STDDEV: {}", that_mean);
        let stddev = stddev.full_like(that_mean);

        let norm = Normalizer::new(&self.device, mean.clone(), stddev);

        let targets = items
            .iter()
            .map(|val| {
                Tensor::from_data(
                    [(val.image_path.find("forge").map(|_| 1).unwrap_or(0))
                        .elem::<B::IntElem>()],
                    &self.device,
                )
            })
            .collect::<Vec<_>>();

        let images = images
            .into_iter()
            .map(|img| norm.normalize(img))
            .collect::<Vec<_>>();

        log::info!("images.first(): {}", images.first().unwrap());
        let images: Tensor<B, 4> = Tensor::stack(images, 0).detach();
        log::info!("Images: {}", images);
        let targets = Tensor::cat(targets, 0);
        log::info!("Targets: {}", targets);

        HandsignBatch { images, targets }
    }
}
