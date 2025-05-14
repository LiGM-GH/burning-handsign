use std::marker::PhantomData;

use batcher::{HandsignBatch, HandsignBatcher};
use burn::{
    data::{
        dataloader::{DataLoader, DataLoaderBuilder, batcher::Batcher},
        dataset::vision::{ImageDatasetItem, ImageFolderDataset},
    },
    nn::{
        Linear, LinearConfig, Relu, Sigmoid,
        conv::{Conv2d, Conv2dConfig},
        loss::{BinaryCrossEntropyLoss, BinaryCrossEntropyLossConfig},
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

use crate::dataset::HandsignDataset;

mod batcher;
mod normalizer;

const IMAGE_LENGTH: usize = 820;
const IMAGE_HEIGHT: usize = 200;
const IMAGE_DEPTH: usize = 1;

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    activation: Relu,
    conv1: Conv2d<B>,
    pool1: MaxPool2d,
    conv2: Conv2d<B>,
    pool2: MaxPool2d,
    linear1: Linear<B>,
    linear2: Linear<B>,
}

#[derive(Config, Debug)]
pub struct ModelConfig {
    hidden_size: usize,
    conv1_chans: usize,
    conv2_chans: usize,
    #[config(default = "0.5")]
    dropout: f64,
}

impl ModelConfig {
    /// Returns the initialized model.
    pub fn init<B: Backend>(&self, device: &B::Device) -> Model<B> {
        const CONV1: usize = 1;
        const CONV2: usize = 8;
        const CONV3: usize = 16;

        Model {
            activation: Relu::new(),
            conv1: Conv2dConfig::new([CONV1, CONV2], [3, 3])
                .with_padding(nn::PaddingConfig2d::Same)
                .init(device),
            pool1: MaxPool2dConfig::new([3, 3])
                .with_padding(nn::PaddingConfig2d::Same)
                .init(),
            conv2: Conv2dConfig::new([CONV2, CONV3], [3, 3])
                .with_padding(nn::PaddingConfig2d::Same)
                .init(device),
            pool2: MaxPool2dConfig::new([3, 3])
                .with_padding(nn::PaddingConfig2d::Same)
                .init(),
            linear1: LinearConfig::new(
                IMAGE_LENGTH * IMAGE_HEIGHT * IMAGE_DEPTH * CONV3,
                self.hidden_size,
            )
            .init(device),
            linear2: LinearConfig::new(self.hidden_size, 1).init(device),
        }
    }
}

impl<B: Backend> Model<B> {
    fn forward(&self, images: Tensor<B, 4>) -> Tensor<B, 1> {
        let [batch_size, height, width, colors] = images.dims();

        #[rustfmt::skip]
        let result =
            images
            .pipe(|x| { log::error!("FORWARD: {}", x); x})
            .pipe(|x| self.conv1.forward(x))
            .pipe(|x| { log::error!("FORWARD: {}", x); x})
            .pipe(|x| self.activation.forward(x))
            .pipe(|x| { log::error!("FORWARD: {}", x); x})
            .pipe(|x| self.pool1.forward(x))
            .pipe(|x| { log::error!("FORWARD: {}", x); x})
            .pipe(|x| self.conv2.forward(x))
            .pipe(|x| { log::error!("FORWARD: {}", x); x})
            .pipe(|x| self.activation.forward(x))
            .pipe(|x| { log::error!("FORWARD: {}", x); x})
            .pipe(|x| self.pool2.forward(x))
            .pipe(|x| { log::error!("FORWARD: {}", x); x})
            .pipe(|x| x.reshape([0, -1]))
            .pipe(|x| { log::error!("FORWARD: {}", x); x})
            .pipe(|x| self.linear1.forward(x))
            .pipe(|x| { log::error!("FORWARD: {}", x); x})
            .pipe(|x| self.linear2.forward(x))
            .pipe(|x| { log::error!("FORWARD: {}", x); x})
            .pipe(|x| x.squeeze(1))
            .pipe(|x| { log::error!("FORWARD: {}", x); x});

        result
    }

    fn forward_classification(
        &self,
        images: Tensor<B, 4>,
        targets: Tensor<B, 1, Int>,
    ) -> ClassificationOutput<B> {
        let output = self.forward(images);

        log::error!("FORWARD::OUTPUT: {}", output);

        let loss = BinaryCrossEntropyLossConfig::new()
            .with_logits(true)
            .init(&output.device())
            .forward(output.clone(), targets.clone());

        log::error!("FORWARD::LOSS: {}", loss);

        ClassificationOutput::new(loss, output.reshape([0i32, -1]), targets)
    }
}

impl<B: AutodiffBackend> TrainStep<HandsignBatch<B>, ClassificationOutput<B>>
    for Model<B>
{
    fn step(
        &self,
        item: HandsignBatch<B>,
    ) -> TrainOutput<ClassificationOutput<B>> {
        let HandsignBatch { images, targets } = item;
        let item = self.forward_classification(images, targets);
        let grads = item.loss.backward();

        TrainOutput::new(self, grads, item)
    }
}

impl<B: Backend> ValidStep<HandsignBatch<B>, ClassificationOutput<B>>
    for Model<B>
{
    fn step(&self, item: HandsignBatch<B>) -> ClassificationOutput<B> {
        let HandsignBatch { images, targets } = item;
        self.forward_classification(images, targets)
    }
}

#[derive(Config)]
pub struct TrainingConfig {
    pub model: ModelConfig,
    pub optimizer: AdamConfig,
    #[config(default = 10)]
    pub num_epochs: usize,
    #[config(default = 64)]
    pub batch_size: usize,
    #[config(default = 4)]
    pub num_workers: usize,
    #[config(default = 42)]
    pub seed: u64,
    #[config(default = 1.0e-4)]
    pub learning_rate: f64,
}

fn create_artifact_dir(artifact_dir: &str) {
    // Remove existing artifacts before to get an accurate learner summary
    std::fs::remove_dir_all(artifact_dir).ok();
    std::fs::create_dir_all(artifact_dir).ok();
}

pub fn train<B: AutodiffBackend>(
    artifact_dir: &str,
    config: TrainingConfig,
    device: &B::Device,
) {
    create_artifact_dir(artifact_dir);

    config
        .save(format!("{artifact_dir}/config.json"))
        .expect("Couldn't save config");

    B::seed(config.seed);

    let batcher_train = HandsignBatcher::new(device.clone());
    let batcher_valid = HandsignBatcher::new(device.clone());

    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(ImageFolderDataset::hs_train());

    let dataloader_test = DataLoaderBuilder::new(batcher_valid)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(ImageFolderDataset::hs_train());

    let builder: LearnerBuilder<
        B,
        _,
        ClassificationOutput<_>,
        Model<B>,
        burn::optim::adaptor::OptimizerAdaptor<burn::optim::Adam, Model<B>, B>,
        f64,
    > = LearnerBuilder::new(artifact_dir)
        .metric_train_numeric(AccuracyMetric::new())
        .metric_valid_numeric(AccuracyMetric::new())
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .devices(vec![device.clone()])
        .num_epochs(config.num_epochs)
        .summary();

    let learner = builder.build(
        config.model.init::<B>(&device),
        config.optimizer.init(),
        config.learning_rate,
    );

    let model_trained = learner.fit(dataloader_train, dataloader_test);

    model_trained
        .save_file(format!("{artifact_dir}/model"), &CompactRecorder::new())
        .expect("Model should be saved successfully")
}
