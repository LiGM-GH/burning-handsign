use burn::{
    nn::{
        BatchNorm, BatchNormConfig, Dropout, DropoutConfig, Linear,
        LinearConfig, Relu,
        conv::{Conv2d, Conv2dConfig},
        pool::{AvgPool2d, AvgPool2dConfig, MaxPool2d, MaxPool2dConfig},
    },
    prelude::*,
};
use tap::Pipe;

use crate::dataset::{IMAGE_DEPTH, IMAGE_HEIGHT, IMAGE_LENGTH};

#[derive(Module, Debug)]
pub struct SingleTwinModel<B: Backend> {
    activation: Relu,

    conv1: Conv2d<B>,
    norm1: BatchNorm<B, 2>,

    conv2: Conv2d<B>,
    norm2: BatchNorm<B, 2>,

    drop1: Dropout,

    conv3: Conv2d<B>,
    norm3: BatchNorm<B, 2>,

    conv4: Conv2d<B>,
    norm4: BatchNorm<B, 2>,

    pool1: AvgPool2d,
    drop2: Dropout,

    linear1: Linear<B>,
}

#[derive(Config, Debug)]
pub struct ModelConfig {
    hidden_size: usize,
    conv1_chans: usize,
    conv2_chans: usize,
    #[config(default = "0.5")]
    dropout: f64,
    #[config(default = 128)]
    last_layer: usize,
}

impl ModelConfig {
    /// Returns the initialized model.
    #[allow(unused)]
    pub fn init<B: Backend>(&self, device: &B::Device) -> SingleTwinModel<B> {
        const CONV1: usize = 1;
        const CONV1_KERSIZE: usize = 11;

        const POOL3_STRIDE: usize = 2;
        const POOL3_KERSIZE: usize = 3;

        const CONV2: usize = 64;
        const CONV2_KERSIZE: usize = 5;
        const CONV2_PADDING: usize = 2;

        const POOL6_KERSIZE: usize = 3;
        const POOL6_STRIDE: usize = 2;

        const CONV3: usize = 128;
        const CONV3_KERSIZE: usize = 3;
        const CONV3_PADDING: usize = 1;
        const CONV3_STRIDE: usize = 1;

        const CONV4: usize = 256;
        const CONV4_KERSIZE: usize = 3;
        const CONV4_PADDING: usize = 1;
        const CONV4_STRIDE: usize = 1;
        const CONV4_OUT: usize = 512;
        const POOL1_KERSIZE: usize = 3;
        const POOL1_STRIDE: usize = 2;

        const SIZE0: [usize; 3] = [IMAGE_HEIGHT, IMAGE_LENGTH, IMAGE_DEPTH];
        const SIZE1: [usize; 3] = [
            SIZE0[0] - (CONV1_KERSIZE - 1),
            SIZE0[1] - (CONV1_KERSIZE - 1),
            SIZE0[2] * CONV2 / CONV1,
        ];
        const SIZE2: [usize; 3] = [SIZE1[0], SIZE1[1], SIZE1[2]];
        const SIZE3: [usize; 3] = [SIZE2[0], SIZE2[1], SIZE2[2]];
        const SIZE4: [usize; 3] = [
            SIZE3[0] - CONV2_KERSIZE / 2 + CONV2_PADDING,
            SIZE3[1] - CONV2_KERSIZE / 2 + CONV2_PADDING,
            SIZE3[2] * CONV3 / CONV2,
        ];
        const SIZE5: [usize; 3] = [SIZE4[0], SIZE4[1], SIZE4[2]];
        const SIZE6: [usize; 3] = [SIZE5[0], SIZE5[1], SIZE5[2]];
        const SIZE7: [usize; 3] = [SIZE6[0], SIZE6[1], SIZE6[2]];
        const SIZE8: [usize; 3] = [
            SIZE7[0] / CONV3_STRIDE - CONV3_KERSIZE / 2 + CONV3_PADDING,
            SIZE7[1] / CONV3_STRIDE - CONV3_KERSIZE / 2 + CONV3_PADDING,
            SIZE7[2] * CONV4 / CONV3,
        ];
        const SIZE9: [usize; 3] = [
            SIZE8[0] / CONV4_STRIDE - CONV4_KERSIZE / 2 + CONV4_PADDING,
            SIZE8[1] / CONV4_STRIDE - CONV4_KERSIZE / 2 + CONV4_PADDING,
            SIZE8[2],
        ];
        const SIZE10: [usize; 3] = [
            (SIZE9[0] - POOL1_KERSIZE / 2) / POOL1_STRIDE,
            (SIZE9[1] - POOL1_KERSIZE / 2) / POOL1_STRIDE,
            CONV4_OUT,
        ];
        const SIZE11: [usize; 3] = [SIZE10[0], SIZE10[1], SIZE10[2]];
        const SIZE12: usize = SIZE11[0] * SIZE11[1] * SIZE11[2];
        const SIZE13: usize = 256;

        let size14: usize = self.last_layer;
        let final_size: usize = size14;

        const LINEAR12: usize = SIZE12;
        const LINEAR14: usize = SIZE13;

        let thing = SingleTwinModel {
            activation: Relu::new(),
            conv1: Conv2dConfig::new(
                [CONV1, CONV2],
                [CONV1_KERSIZE, CONV1_KERSIZE],
            )
            .with_stride([1, 1])
            .init(device),

            norm1: BatchNormConfig::new(CONV2).init(device),

            conv2: Conv2dConfig::new(
                [CONV2, CONV3],
                [CONV2_KERSIZE, CONV2_KERSIZE],
            )
            .with_stride([1, 1])
            .with_padding(nn::PaddingConfig2d::Explicit(
                CONV2_PADDING,
                CONV2_PADDING,
            ))
            .init(device),

            norm2: BatchNormConfig::new(CONV3).init(device),

            drop1: DropoutConfig::new(0.3).init(),

            conv3: Conv2dConfig::new(
                [CONV3, CONV4],
                [CONV3_KERSIZE, CONV3_KERSIZE],
            )
            .with_stride([CONV3_STRIDE, CONV3_STRIDE])
            .with_padding(nn::PaddingConfig2d::Explicit(
                CONV3_PADDING,
                CONV3_PADDING,
            ))
            .init(device),

            norm3: BatchNormConfig::new(CONV4).init(device),

            conv4: Conv2dConfig::new(
                [CONV4, CONV4_OUT],
                [CONV4_KERSIZE, CONV4_KERSIZE],
            )
            .with_stride([CONV4_STRIDE, CONV4_STRIDE])
            .with_padding(nn::PaddingConfig2d::Explicit(
                CONV4_PADDING,
                CONV4_PADDING,
            ))
            .init(device),

            norm4: BatchNormConfig::new(CONV4_OUT).init(device),

            pool1: AvgPool2dConfig::new([POOL1_KERSIZE, POOL1_KERSIZE])
                .with_strides([POOL1_STRIDE, POOL1_STRIDE])
                .init(),

            drop2: DropoutConfig::new(0.3).init(),

            linear1: LinearConfig::new(SIZE12, SIZE13).init(device),
        };

        println!("thing: {:?}", thing);

        thing
    }
}

impl<B: Backend> SingleTwinModel<B> {
    pub fn forward(&self, images: Tensor<B, 4>) -> Tensor<B, 2> {
        let [_batch_size, _height, _width, _colors] = images.dims();

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
            .pipe(|x| self.activation.forward(x))
            .pipe(|x| dprint!(x))
            .pipe(|x| self.norm1.forward(x))
            .pipe(|x| dprint!(x))
            .pipe(|x| self.conv2.forward(x))
            .pipe(|x| dprint!(x))
            .pipe(|x| self.activation.forward(x))
            .pipe(|x| dprint!(x))
            .pipe(|x| self.norm2.forward(x))
            .pipe(|x| dprint!(x))
            .pipe(|x| self.drop1.forward(x))
            .pipe(|x| dprint!(x))
            .pipe(|x| self.conv3.forward(x))
            .pipe(|x| dprint!(x))
            .pipe(|x| self.activation.forward(x))
            // .pipe(|x| dprint!(x))
            // .pipe(|x| self.norm3.forward(x))
            .pipe(|x| dprint!(x))
            .pipe(|x| self.conv4.forward(x))
            .pipe(|x| dprint!(x))
            .pipe(|x| self.activation.forward(x))
            // .pipe(|x| dprint!(x))
            // .pipe(|x| self.norm4.forward(x))
            .pipe(|x| dprint!(x))
            .pipe(|x| self.pool1.forward(x))
            .pipe(|x| dprint!(x))
            .pipe(|x| self.drop2.forward(x))
            .pipe(|x| dprint!(x))
            .pipe(|x| x.reshape([0, -1]))
            .pipe(|x| dprint!(x))
            .pipe(|x| self.linear1.forward(x))
            .pipe(|x| dprint!(x))
            .pipe(|x| self.activation.forward(x))
            .pipe(|x| dprint!(x));

        result
    }
}
