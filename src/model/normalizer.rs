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

pub struct Normalizer<B: Backend> {
    pub mean: Tensor<B, 3>,
    pub stddev: Tensor<B, 3>,
}

impl<B: Backend> Normalizer<B> {
    pub fn new(
        device: &Device<B>,
        mean: Tensor<B, 3>,
        stddev: Tensor<B, 3>,
    ) -> Self {
        log::error!("Self's mean is {} and stddev is {}", mean, stddev);
        Self { mean, stddev }
    }

    pub fn normalize(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let result = (input.clone() - self.mean.clone()) / self.stddev.clone();

        log::error!("This {} has been normalized to: {}", input, result);
        assert!(
            !result.clone().abs().greater_elem(1.0).any().into_scalar(),
            "All elements should be less than 1"
        );
        result
    }

    pub fn to_device(&self, device: &B::Device) -> Self {
        Self {
            mean: self.mean.clone().to_device(device),
            stddev: self.stddev.clone().to_device(device),
        }
    }
}
