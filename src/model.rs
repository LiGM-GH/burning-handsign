use std::path::Path;

use batcher::{HandsignBatch, HandsignBatcher};
use burn::{
    backend::Autodiff,
    data::{
        dataloader::DataLoaderBuilder,
        dataset::{Dataset, vision::ImageFolderDataset},
    },
    nn::{
        BatchNorm, BatchNormConfig, Dropout, DropoutConfig, Linear,
        LinearConfig, Relu, Sigmoid,
        conv::{Conv2d, Conv2dConfig},
        loss::BinaryCrossEntropyLossConfig,
        pool::{MaxPool2d, MaxPool2dConfig},
    },
    optim::{AdamConfig, adaptor::OptimizerAdaptor},
    prelude::*,
    record::{CompactRecorder, Recorder},
    tensor::{backend::AutodiffBackend, cast::ToElement},
    train::{
        LearnerBuilder, TrainOutput, TrainStep, ValidStep, metric::LossMetric,
    },
};
use metric::{
    BinaryClassificationOutput, TimesGuessedMetric, tensor_to_guesses,
};
use normalizer::Normalizer;
use tap::Pipe;

use crate::{MEAN_DS, STDDEV_DS, dataset::HandsignDataset};

mod batcher;
mod metric;
mod normalizer;

type MyBackend = burn::backend::LibTorch;

pub const IMAGE_LENGTH: usize = 600;
pub const IMAGE_HEIGHT: usize = 600;
pub const IMAGE_DEPTH: usize = 1;

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    activation: Relu,
    act2: Sigmoid,

    conv1: Conv2d<B>,
    norm2: BatchNorm<B, 2>,
    pool3: MaxPool2d,

    conv4: Conv2d<B>,
    norm5: BatchNorm<B, 2>,

    pool6: MaxPool2d,
    drop7: Dropout,

    conv8: Conv2d<B>,
    conv9: Conv2d<B>,
    pool10: MaxPool2d,
    drop11: Dropout,

    linear12: Linear<B>,
    drop13: Dropout,

    linear14: Linear<B>,
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
    #[allow(unused)]
    pub fn init<B: Backend>(&self, device: &B::Device) -> Model<B> {
        const CONV1: usize = 1;
        const CONV1_KERSIZE: usize = 11;
        const POOL3_STRIDE: usize = 2;
        const POOL3_KERSIZE: usize = 3;
        const CONV4: usize = 8;
        const CONV4_KERSIZE: usize = 5;
        const CONV4_PADDING: usize = 2;
        const POOL6_KERSIZE: usize = 3;
        const POOL6_STRIDE: usize = 2;
        const CONV8: usize = 16;
        const CONV8_KERSIZE: usize = 3;
        const CONV8_PADDING: usize = 1;
        const CONV8_STRIDE: usize = 1;
        const CONV9: usize = 16;
        const CONV9_PADDING: usize = 1;
        const CONV9_STRIDE: usize = 1;
        const CONV9_KERSIZE: usize = 3;
        const POOL10_KERSIZE: usize = 3;
        const POOL10_STRIDE: usize = 2;

        const SIZE0: [usize; 3] = [IMAGE_HEIGHT, IMAGE_LENGTH, IMAGE_DEPTH];
        const SIZE1: [usize; 3] = [
            SIZE0[0] - (CONV1_KERSIZE - 1),
            SIZE0[1] - (CONV1_KERSIZE - 1),
            SIZE0[2] * CONV4 / CONV1,
        ];
        const SIZE2: [usize; 3] = [SIZE1[0], SIZE1[1], SIZE1[2]];
        const SIZE3: [usize; 3] = [
            SIZE2[0] / POOL3_STRIDE - POOL3_KERSIZE / 2,
            SIZE2[1] / POOL3_STRIDE - POOL3_KERSIZE / 2,
            SIZE2[2],
        ];
        const SIZE4: [usize; 3] = [
            SIZE3[0] - CONV4_KERSIZE / 2 + CONV4_PADDING,
            SIZE3[1] - CONV4_KERSIZE / 2 + CONV4_PADDING,
            SIZE3[2] * CONV8 / CONV4,
        ];
        const SIZE5: [usize; 3] = [SIZE4[0], SIZE4[1], SIZE4[2]];
        const SIZE6: [usize; 3] = [
            SIZE5[0] / POOL6_STRIDE - POOL6_KERSIZE / 2,
            SIZE5[1] / POOL6_STRIDE - POOL6_KERSIZE / 2,
            SIZE5[2],
        ];
        const SIZE7: [usize; 3] = [SIZE6[0], SIZE6[1], SIZE6[2]];
        const SIZE8: [usize; 3] = [
            SIZE7[0] / CONV8_STRIDE - CONV8_KERSIZE / 2 + CONV8_PADDING,
            SIZE7[1] / CONV8_STRIDE - CONV8_KERSIZE / 2 + CONV8_PADDING,
            SIZE7[2] * CONV9 / CONV8,
        ];
        const SIZE9: [usize; 3] = [
            SIZE8[0] / CONV9_STRIDE - CONV9_KERSIZE / 2 + CONV9_PADDING,
            SIZE8[1] / CONV9_STRIDE - CONV9_KERSIZE / 2 + CONV9_PADDING,
            SIZE8[2],
        ];
        const SIZE10: [usize; 3] = [
            SIZE9[0] / POOL10_STRIDE - POOL10_KERSIZE / 2,
            SIZE9[1] / POOL10_STRIDE - POOL10_KERSIZE / 2,
            SIZE9[2],
        ];
        const SIZE11: [usize; 3] = [SIZE10[0], SIZE10[1], SIZE10[2]];
        const SIZE12: usize = SIZE11[0] * SIZE11[1] * SIZE11[2];
        const SIZE13: usize = 16;
        const SIZE14: usize = 1;
        const FINAL_SIZE: usize = SIZE14;

        const LINEAR12: usize = SIZE12;
        const LINEAR14: usize = SIZE13;

        let thing = Model {
            activation: Relu::new(),
            act2: Sigmoid::new(),
            conv1: Conv2dConfig::new(
                [CONV1, CONV4],
                [CONV1_KERSIZE, CONV1_KERSIZE],
            )
            .with_stride([1, 1])
            .init(device),
            norm2: BatchNormConfig::new(CONV4).init(device),
            pool3: MaxPool2dConfig::new([3, 3])
                .with_strides([POOL3_STRIDE, POOL3_STRIDE])
                .init(),
            conv4: Conv2dConfig::new(
                [CONV4, CONV8],
                [CONV4_KERSIZE, CONV4_KERSIZE],
            )
            .with_stride([1, 1])
            .with_padding(nn::PaddingConfig2d::Explicit(
                CONV4_PADDING,
                CONV4_PADDING,
            ))
            .init(device),
            norm5: BatchNormConfig::new(CONV8).init(device),
            pool6: MaxPool2dConfig::new([POOL6_KERSIZE, POOL6_KERSIZE])
                .with_strides([POOL6_STRIDE, POOL6_STRIDE])
                .init(),
            drop7: DropoutConfig::new(0.3).init(),
            conv8: Conv2dConfig::new([CONV8, CONV8], [3, 3])
                .with_stride([CONV8_STRIDE, CONV8_STRIDE])
                .with_padding(nn::PaddingConfig2d::Explicit(
                    CONV8_PADDING,
                    CONV8_PADDING,
                ))
                .init(device),
            conv9: Conv2dConfig::new([CONV8, CONV8], [3, 3])
                .with_stride([CONV9_STRIDE, CONV9_STRIDE])
                .with_padding(nn::PaddingConfig2d::Explicit(
                    CONV9_PADDING,
                    CONV9_PADDING,
                ))
                .init(device),
            pool10: MaxPool2dConfig::new([POOL10_KERSIZE, POOL10_KERSIZE])
                .with_strides([POOL10_STRIDE, POOL10_STRIDE])
                .init(),
            drop11: DropoutConfig::new(0.3).init(),
            linear12: LinearConfig::new(SIZE12, SIZE13).init(device),
            drop13: DropoutConfig::new(0.5).init(),
            linear14: LinearConfig::new(SIZE13, SIZE14).init(device),
        };

        println!("thing: {:?}", thing);

        thing
    }
}

impl<B: Backend> Model<B> {
    fn forward(&self, images: Tensor<B, 4>) -> Tensor<B, 1> {
        macro_rules! dprint {
            ($x:ident) => {{
                log::error!("line {} | FORWARD: {:?}", line!(), $x.dims());
                $x
            }};
        }

        #[rustfmt::skip]
        let result = images
            .pipe(|x| dprint!(x))
            .pipe(|x| self.conv1.forward(x))
            .pipe(|x| dprint!(x))
            .pipe(|x| self.norm2.forward(x))
            .pipe(|x| dprint!(x))
            .pipe(|x| self.pool3.forward(x))
            .pipe(|x| dprint!(x))
            .pipe(|x| self.conv4.forward(x))
            .pipe(|x| dprint!(x))
            .pipe(|x| self.norm5.forward(x))
            .pipe(|x| dprint!(x))
            .pipe(|x| self.pool6.forward(x))
            .pipe(|x| dprint!(x))
            .pipe(|x| self.drop7.forward(x))
            .pipe(|x| dprint!(x))
            .pipe(|x| self.conv8.forward(x))
            .pipe(|x| dprint!(x))
            .pipe(|x| self.conv9.forward(x))
            .pipe(|x| dprint!(x))
            .pipe(|x| self.pool10.forward(x))
            .pipe(|x| dprint!(x))
            .pipe(|x| self.drop11.forward(x))
            .pipe(|x| dprint!(x))
            .pipe(|x| x.reshape([0, -1]))
            .pipe(|x| dprint!(x))
            .pipe(|x| self.linear12.forward(x))
            .pipe(|x| dprint!(x))
            .pipe(|x| self.drop13.forward(x))
            .pipe(|x| dprint!(x))
            .pipe(|x| self.linear14.forward(x))
            .pipe(|x| dprint!(x))
            .pipe(|x| self.act2.forward(x))
            .pipe(|x| dprint!(x))
            .pipe(|x| x.squeeze::<1>(1))
            .pipe(|x| dprint!(x));

        result
    }

    fn forward_classification(
        &self,
        images: Tensor<B, 4>,
        targets: Tensor<B, 1, Int>,
    ) -> BinaryClassificationOutput<B> {
        let output = self.forward(images);

        log::error!("FORWARD::OUTPUT: {}", output);

        let loss = BinaryCrossEntropyLossConfig::new()
            .with_logits(false)
            .init(&output.device())
            .forward(output.clone(), targets.clone());

        log::error!("FORWARD::LOSS: {}", loss);

        BinaryClassificationOutput::new(loss, output, targets)
    }
}

impl<B: AutodiffBackend>
    TrainStep<HandsignBatch<B>, BinaryClassificationOutput<B>> for Model<B>
{
    fn step(
        &self,
        item: HandsignBatch<B>,
    ) -> TrainOutput<BinaryClassificationOutput<B>> {
        let HandsignBatch { images, targets } = item;
        let item = self.forward_classification(images, targets);
        let grads = item.loss.backward();

        TrainOutput::new(self, grads, item)
    }
}

impl<B: Backend> ValidStep<HandsignBatch<B>, BinaryClassificationOutput<B>>
    for Model<B>
{
    fn step(&self, item: HandsignBatch<B>) -> BinaryClassificationOutput<B> {
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
    dataset_dir: &str,
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
        .build(ImageFolderDataset::hs_path(dataset_dir));

    let dataloader_test = DataLoaderBuilder::new(batcher_valid)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(ImageFolderDataset::hs_path(dataset_dir));

    type C1<B> = BinaryClassificationOutput<B>;
    type C2<B> =
        BinaryClassificationOutput<<B as AutodiffBackend>::InnerBackend>;

    type Opt<B> = OptimizerAdaptor<burn::optim::Adam, Model<B>, B>;

    type Builder<B> = LearnerBuilder<B, C1<B>, C2<B>, Model<B>, Opt<B>, f64>;

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
        config.learning_rate,
    );

    let model_trained = learner.fit(dataloader_train, dataloader_test);

    model_trained
        .save_file(format!("{artifact_dir}/model"), &CompactRecorder::new())
        .expect("Model should be saved successfully");
}

pub fn guess_inner<B: Backend>(
    config_json_path: &str,
    model_path: &str,
    device: B::Device,
    images_path: impl AsRef<Path>,
    mean: f64,
    stddev: f64,
) {
    let items = ImageFolderDataset::new_classification(&images_path)
        .unwrap_or_else(|_| {
            panic!(
                "Dataset should exist in path {}",
                images_path.as_ref().to_string_lossy()
            )
        });

    let images = items
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

    let mean = images
        .first()
        .expect("Not a single item in the directory!")
        .full_like(mean);

    let stddev = mean.full_like(stddev);

    let norm = Normalizer::new(&device, mean, stddev);

    let images = images
        .into_iter()
        .map(|val| norm.normalize(val))
        .collect::<Vec<_>>();

    let config =
        TrainingConfig::load(config_json_path).expect("Couldn't load config");

    let record = CompactRecorder::new()
        .load(model_path.into(), &device)
        .expect("Trained model should exist; run train first");

    let model = config.model.init::<B>(&device).load_record(record);

    // let len = images.len();

    let targets = items
        .iter()
        .map(|val| {
            let is_forge = val.image_path.find("forge").map(|_| 1).unwrap_or(0);
            Tensor::from_data([is_forge.elem::<B::IntElem>()], &device)
        })
        .collect::<Vec<_>>();

    let batch = HandsignBatch {
        images: Tensor::stack(images, 0),
        targets: Tensor::cat(targets, 0),
    };

    println!("batch: {:?}", batch);

    let output = model.forward(batch.images);

    println!("\noutput: {:?}\n", output.to_data().to_vec::<f32>());

    let guesses = tensor_to_guesses(output);

    println!("\nguesses: {:?}\n", guesses.to_data().to_vec::<i64>());

    {
        let guessed_part = guesses
            .clone()
            .sub_scalar(1)
            .powi_scalar(2)
            .mul(batch.targets.clone())
            .sum()
            .into_scalar();

        let len = batch.targets.clone().sum().into_scalar().to_i32();

        println!(
            "FAR: {} times out of {}: {:.2}%",
            guessed_part,
            len,
            guessed_part.to_i8() as f64 / len as f64 * 100.0,
        );
    }

    {
        let nontargets = batch.targets.clone().sub_scalar(1).powi_scalar(2);

        let guessed_part =
            guesses.clone().mul(nontargets.clone()).sum().into_scalar();

        let len = nontargets.clone().sum().into_scalar().to_i32();

        println!(
            "FFR: {} times out of {}: {:.2}%",
            guessed_part,
            len,
            guessed_part.to_i8() as f64 / len as f64 * 100.0
        );
    }
}

pub fn learn(dataset_dir: &str, artifacts_dir: &str) {
    let device = Default::default();

    train::<Autodiff<MyBackend>>(
        dataset_dir,
        artifacts_dir,
        TrainingConfig::new(
            ModelConfig::new(3, 1, 16).with_dropout(0.5),
            AdamConfig::new(),
        )
        .with_num_epochs(40),
        &device,
    )
}

pub fn guess(path: &str, model_path: &str) {
    let device = <MyBackend as Backend>::Device::default();

    guess_inner::<MyBackend>(
        "artifacts/config.json",
        model_path,
        device,
        path,
        MEAN_DS,
        STDDEV_DS,
    );
}
