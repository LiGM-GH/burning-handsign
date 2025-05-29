use std::path::Path;

use batcher::{CedarBatch, HandsignBatch, HandsignBatcher};
use burn::{
    data::{
        dataloader::DataLoaderBuilder,
        dataset::{vision::ImageFolderDataset, Dataset},
    },
    lr_scheduler::{exponential::{ExponentialLrScheduler, ExponentialLrSchedulerConfig}, LrScheduler},
    optim::{adaptor::OptimizerAdaptor, AdamConfig, RmsPropConfig},
    prelude::*,
    record::{CompactRecorder, Recorder},
    tensor::{backend::AutodiffBackend, cast::ToElement},
    train::{
        metric::LossMetric, LearnerBuilder, TrainOutput, TrainStep, ValidStep
    },
};
use metric::{
    BinaryClassificationOutput, TimesGuessedMetric, tensor_to_guesses,
};

use normalizer::Normalizer;
use single_twin_model::SingleTwinModel;

use crate::dataset::{
    CedarDataset, HandsignDataset, IMAGE_DEPTH, IMAGE_HEIGHT, IMAGE_LENGTH,
};

mod batcher;
mod metric;
mod normalizer;
mod single_twin_model;

#[derive(Config, Debug)]
pub struct ModelConfig {
    hidden_size: usize,
    conv1_chans: usize,
    conv2_chans: usize,
    #[config(default = "0.5")]
    dropout: f64,
}

#[derive(Module, Debug)]
pub struct TwinModel<B: Backend> {
    inner_model: SingleTwinModel<B>,
}

impl<B: Backend> TwinModel<B> {
    pub fn forward(
        &self,
        left: Tensor<B, 4>,
        right: Tensor<B, 4>,
    ) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let left_result = self.inner_model.forward(left);
        let right_result = self.inner_model.forward(right);
        (left_result, right_result)
    }

    fn forward_classification(
        &self,
        left: Tensor<B, 4>,
        right: Tensor<B, 4>,
        targets: Tensor<B, 1, Int>,
    ) -> BinaryClassificationOutput<B> {
        const ALPHA: f64 = 0.5;
        const BETA: f64 = 0.5;
        const M: f64 = 1.0;

        let (left_result, right_result) = self.forward(left, right);
        let diff = left_result - right_result;
        let dw_square = diff.powi_scalar(2).sum_dim(1).squeeze(1);
        let dw = dw_square.clone().sqrt();
        let output = dw.clone();

        // dw² * α * (1 - y) - β * y max { 0, (-1 * (dw - m)) }²
        let loss = dw_square
            .mul_scalar(ALPHA)
            .mul(targets.clone().sub_scalar(1).neg().float())
            .sub(
                dw.sub_scalar(M)
                    .neg()
                    .clamp_min(0)
                    .powi_scalar(2)
                    .mul(targets.clone().float())
                    .mul_scalar(BETA),
            )
            .mean();

        log::error!("LOSS: {:?}", loss);

        BinaryClassificationOutput {
            loss,
            output,
            targets,
        }
    }
}

impl ModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> TwinModel<B> {
        let inner_model = single_twin_model::ModelConfig::new(
            self.hidden_size,
            self.conv1_chans,
            self.conv2_chans,
        )
        .with_dropout(self.dropout)
        .init(device);

        TwinModel { inner_model }
    }
}

impl<B: AutodiffBackend> TrainStep<CedarBatch<B>, BinaryClassificationOutput<B>>
    for TwinModel<B>
{
    fn step(
        &self,
        item: CedarBatch<B>,
    ) -> TrainOutput<BinaryClassificationOutput<B>> {
        let CedarBatch {
            left,
            right,
            targets,
        } = item;

        let item = self.forward_classification(left, right, targets);
        let grads = item.loss.backward();

        TrainOutput::new(self, grads, item)
    }
}

impl<B: Backend> ValidStep<CedarBatch<B>, BinaryClassificationOutput<B>>
    for TwinModel<B>
{
    fn step(&self, item: CedarBatch<B>) -> BinaryClassificationOutput<B> {
        let CedarBatch {
            left,
            right,
            targets,
        } = item;

        self.forward_classification(left, right, targets)
    }
}

#[derive(Config)]
pub struct TrainingConfig {
    pub model: ModelConfig,
    pub optimizer: RmsPropConfig,
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
        .build(CedarDataset::hs_test());

    let dataloader_test = DataLoaderBuilder::new(batcher_valid)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(CedarDataset::hs_test());

    type C1<B> = BinaryClassificationOutput<B>;
    type C2<B> =
        BinaryClassificationOutput<<B as AutodiffBackend>::InnerBackend>;

    type Opt<B> = OptimizerAdaptor<burn::optim::RmsProp, TwinModel<B>, B>;
    type Model<B> = TwinModel<B>;

    type Builder<B> = LearnerBuilder<B, C1<B>, C2<B>, Model<B>, Opt<B>, ExponentialLrScheduler>;

    let builder: Builder<B> = LearnerBuilder::new(artifact_dir)
        .metric_train_numeric(TimesGuessedMetric::new())
        .metric_valid_numeric(TimesGuessedMetric::new())
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .devices(vec![device.clone()])
        .num_epochs(config.num_epochs)
        .summary();

    let learner = builder.build(
        config.model.init::<B>(device),
        config.optimizer.init(),
        ExponentialLrSchedulerConfig::new(config.learning_rate, 0.1)
            .init()
            .expect("Couldn't initialize learning rate scheduler"),
    );

    let model_trained = learner.fit(dataloader_train, dataloader_test);

    model_trained
        .save_file(format!("{artifact_dir}/model"), &CompactRecorder::new())
        .expect("Model should be saved successfully")
}

pub fn guess<B: Backend>(
    artifact_dir: &str,
    device: B::Device,
    images_path: impl AsRef<Path>,
    mean: f64,
    stddev: f64,
) {
    let right = ImageFolderDataset::new_classification(&images_path)
        .unwrap_or_else(|_| {
            panic!(
                "Dataset should exist  in path {}",
                images_path.as_ref().to_string_lossy()
            )
        })
        .iter()
        .map(|val| {
            println!("Current image: {}", val.image_path);

            TensorData::new(
                val.image
                    .iter()
                    .map(|val| -> u8 { val.clone().try_into().unwrap() })
                    .collect::<Vec<_>>(),
                Shape::new([IMAGE_LENGTH, IMAGE_HEIGHT, IMAGE_DEPTH]),
            )
        })
        .map(|data| Tensor::<B, 3>::from_data(data, &device) / 255)
        .map(|tensor| {
            let tensor = tensor.swap_dims(0, 2).swap_dims(1, 2);
            log::error!("These are new dims: {:?}", tensor.dims());
            tensor
        })
        .collect::<Vec<_>>();

    let mean = right
        .first()
        .expect("Not a single item in the directory!")
        .full_like(mean);

    let stddev = mean.full_like(stddev);

    let norm = Normalizer::new(&device, mean, stddev);

    let right = right
        .into_iter()
        .map(|val| norm.normalize(val))
        .collect::<Vec<_>>();

    let config = TrainingConfig::load(format!("{artifact_dir}/config.json"))
        .expect("Couldn't load config");

    let record = CompactRecorder::new()
        .load(format!("{artifact_dir}/model").into(), &device)
        .expect("Trained model should exist; run train first");

    let model = config.model.init::<B>(&device).load_record(record);

    let len = right.len();

    let targets = Tensor::from_data(
        TensorData::new(
            std::iter::repeat_n(1, len).collect::<Vec<_>>(),
            Shape::new([right.len()]),
        ),
        &device,
    );

    let left = ImageFolderDataset::new_classification(
        "../handwritten-signatures-ver1/png-220x155/real",
    )
    .unwrap_or_else(|_| {
        panic!(
            "Dataset should exist  in path {}",
            images_path.as_ref().to_string_lossy()
        )
    })
    .iter()
    .map(|val| {
        println!("Current image LEFT: {}", val.image_path);

        TensorData::new(
            val.image
                .iter()
                .map(|val| -> u8 { val.clone().try_into().unwrap() })
                .collect::<Vec<_>>(),
            Shape::new([IMAGE_LENGTH, IMAGE_HEIGHT, IMAGE_DEPTH]),
        )
    })
    .map(|data| Tensor::<B, 3>::from_data(data, &device) / 255)
    .map(|tensor| {
        let tensor = tensor.swap_dims(0, 2).swap_dims(1, 2);
        log::error!("These are new dims: {:?}", tensor.dims());
        tensor
    })
    .collect::<Vec<_>>();

    dbg!(&left);

    let left = left
        .into_iter()
        .cycle()
        .take(len)
        .map(|val| norm.normalize(val))
        .collect::<Vec<_>>();

    dbg!(&right);

    let batch = CedarBatch {
        left: Tensor::stack(left, 0),
        right: Tensor::stack(right, 0),
        targets: targets.clone(),
    };

    println!("batch: {:?}", batch);

    let (left, right) = model.forward(batch.left, batch.right);
    let diff = left - right;

    let dw_square: Tensor<B, 1> = diff.powi_scalar(2).sum_dim(1).squeeze(1);
    let dw = dw_square.clone().sqrt();
    let guesses = dw.clone();

    println!("Guesses: {}", dw);

    const ALPHA: f64 = 0.5;
    const BETA: f64 = 0.5;
    const M: f64 = 1.0;

    let loss = dw_square
        .mul_scalar(ALPHA)
        .mul(targets.clone().sub_scalar(1).neg().float())
        .add(
            dw.sub_scalar(M)
                .neg()
                .clamp_min(0)
                .powi_scalar(2)
                .mul(targets.clone().float())
                .mul_scalar(BETA),
        );

    let guessed_part = guesses.round().int().clamp(0, 1).sum().into_scalar();

    println!(
        "Guessed {} times out of {}: {:.2}%",
        guessed_part,
        len,
        guessed_part.to_i8() as f64 / len as f64 * 100.0,
    );

    println!("Loss: {:?}", loss.to_data().to_vec::<f32>())
}
