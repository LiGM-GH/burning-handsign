use std::marker::PhantomData;

use burn::{
    data::{
        dataloader::{DataLoader, DataLoaderBuilder, batcher::Batcher},
        dataset::vision::{ImageDatasetItem, ImageFolderDataset},
    },
    nn::{
        Linear, LinearConfig,
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

use crate::dataset::HandsignDataset;

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
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
        Model {
            conv1: Conv2dConfig::new([self.conv1_chans, self.conv2_chans], [3, 3])
                .init(device),
            pool1: MaxPool2dConfig::new([3, 3]).init(),
            conv2: Conv2dConfig::new(
                [self.conv2_chans, 2],
                [3, 3],
            )
            .init(device),
            pool2: MaxPool2dConfig::new([3, 3]).init(),
            linear1: LinearConfig::new(2, self.hidden_size).init(device),
            linear2: LinearConfig::new(self.hidden_size, 2).init(device),
        }
    }
}

impl<B: Backend> Model<B> {
    fn forward(&self, images: Tensor<B, 3>) -> Tensor<B, 2> {
        let [batch_size, height, width] = images.dims();

        log::info!("THIS IS HERE");

        images
            .tap(|val| log::info!("\tlineno: {} | {:?}", line!(), val))
            .reshape([batch_size, 1, height, width])
            .tap(|val| log::info!("\tlineno: {} | {:?}", line!(), val))
            .pipe(|val| self.conv1.forward(val))
            .tap(|val| log::info!("\tlineno: {} | {:?}", line!(), val))
            .pipe(|val| self.pool1.forward(val))
            .tap(|val| log::info!("\tlineno: {} | {:?}", line!(), val))
            .pipe(|val| self.conv2.forward(val))
            .tap(|val| log::info!("\tlineno: {} | {:?}", line!(), val))
            .tap(|val| log::info!("\tlineno: {} | {:?}", line!(), val))
            .tap(|val| log::info!("\tlineno: {} | {:?}", line!(), val))
            .pipe(|val| self.pool2.forward(val))
            .tap(|val| log::info!("\tlineno: {} | {:?}", line!(), val))
            .pipe(|val| val.reshape([0, -1i32]))
            .tap(|val| log::info!("\tlineno: {} | {:?}", line!(), val))
            .pipe(|val| self.linear1.forward(val))
            .tap(|val| log::info!("\tlineno: {} | {:?}", line!(), val))
            .pipe(|val| self.linear2.forward(val))
            .tap(|val| log::info!("\tlineno: {} | {:?}", line!(), val))
    }

    fn forward_classification(
        &self,
        images: Tensor<B, 3>,
        targets: Tensor<B, 1, Int>,
    ) -> ClassificationOutput<B> {
        let output = self.forward(images);
        let loss = CrossEntropyLossConfig::new()
            .init(&output.device())
            .forward(output.clone(), targets.clone());

        ClassificationOutput::new(loss, output, targets)
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

        TrainOutput::new(self, item.loss.backward(), item)
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
        .build(ImageFolderDataset::hs_test());

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

#[derive(Clone)]
struct HandsignBatcher<B: Backend> {
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
        let images = items
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
            .map(|data| Tensor::<B, 3>::from_data(data, &self.device) / 255)
            .collect::<Vec<_>>();

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

        let images = Tensor::cat(images, 0);
        log::info!("Images: {:?}", images);
        let targets = Tensor::cat(targets, 0);
        log::info!("Targets: {:?}", targets);

        HandsignBatch { images, targets }
    }
}

#[derive(Debug, Clone)]
struct HandsignBatch<B: Backend> {
    pub images: Tensor<B, 3>,
    pub targets: Tensor<B, 1, Int>,
}

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
        Self { mean, stddev }
    }

    pub fn normalize(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        (input - self.mean.clone()) / self.stddev.clone()
    }

    pub fn to_device(&self, device: &B::Device) -> Self {
        Self {
            mean: self.mean.clone().to_device(device),
            stddev: self.stddev.clone().to_device(device),
        }
    }
}
