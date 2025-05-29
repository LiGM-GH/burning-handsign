use burn::{
    nn::{
        BatchNorm, BatchNormConfig, Dropout, DropoutConfig, Linear,
        LinearConfig, Relu,
        conv::{Conv2d, Conv2dConfig},
        pool::{MaxPool2d, MaxPool2dConfig},
    },
    prelude::*,
};
use tap::Pipe;

use crate::dataset::{IMAGE_DEPTH, IMAGE_HEIGHT, IMAGE_LENGTH};

#[derive(Module, Debug)]
pub struct SingleTwinModel<B: Backend> {
    activation: Relu,

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
    pub fn init<B: Backend>(&self, device: &B::Device) -> SingleTwinModel<B> {
        const CONV1: usize = 1;
        const CONV1_KERSIZE: usize = 11;
        const POOL3_STRIDE: usize = 2;
        const POOL3_KERSIZE: usize = 3;
        const CONV4: usize = 16;
        const CONV4_KERSIZE: usize = 5;
        const CONV4_PADDING: usize = 2;
        const POOL6_KERSIZE: usize = 3;
        const POOL6_STRIDE: usize = 2;
        const CONV8: usize = 32;
        const CONV8_KERSIZE: usize = 3;
        const CONV8_PADDING: usize = 1;
        const CONV8_STRIDE: usize = 1;
        const CONV9: usize = 64;
        const CONV9_PADDING: usize = 1;
        const CONV9_STRIDE: usize = 1;
        const CONV9_KERSIZE: usize = 3;
        const CONV9_OUT: usize = 128;
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
            (SIZE2[0] - POOL3_KERSIZE / 2) / POOL3_STRIDE,
            (SIZE2[1] - POOL3_KERSIZE / 2) / POOL3_STRIDE,
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
            (SIZE9[0] - POOL10_KERSIZE / 2) / POOL10_STRIDE,
            (SIZE9[1] - POOL10_KERSIZE / 2) / POOL10_STRIDE,
            CONV9_OUT,
        ];
        const SIZE11: [usize; 3] = [SIZE10[0], SIZE10[1], SIZE10[2]];
        const SIZE12: usize = SIZE11[0] * SIZE11[1] * SIZE11[2];
        const SIZE13: usize = 256;
        const SIZE14: usize = 128;
        const FINAL_SIZE: usize = SIZE14;

        const LINEAR12: usize = SIZE12;
        const LINEAR14: usize = SIZE13;

        let thing = SingleTwinModel {
            activation: Relu::new(),
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

            conv8: Conv2dConfig::new(
                [CONV8, CONV9],
                [CONV8_KERSIZE, CONV8_KERSIZE],
            )
            .with_stride([CONV8_STRIDE, CONV8_STRIDE])
            .with_padding(nn::PaddingConfig2d::Explicit(
                CONV8_PADDING,
                CONV8_PADDING,
            ))
            .init(device),

            conv9: Conv2dConfig::new(
                [CONV9, CONV9_OUT],
                [CONV9_KERSIZE, CONV9_KERSIZE],
            )
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
            .pipe(|x| self.norm2.forward(x))
            .pipe(|x| dprint!(x))
            .pipe(|x| self.pool3.forward(x))
            .pipe(|x| dprint!(x))
            .pipe(|x| self.conv4.forward(x))
            .pipe(|x| dprint!(x))
            .pipe(|x| self.activation.forward(x))
            .pipe(|x| dprint!(x))
            .pipe(|x| self.norm5.forward(x))
            .pipe(|x| dprint!(x))
            .pipe(|x| self.pool6.forward(x))
            .pipe(|x| dprint!(x))
            .pipe(|x| self.drop7.forward(x))
            .pipe(|x| dprint!(x))
            .pipe(|x| self.conv8.forward(x))
            .pipe(|x| dprint!(x))
            .pipe(|x| self.activation.forward(x))
            .pipe(|x| dprint!(x))
            .pipe(|x| self.conv9.forward(x))
            .pipe(|x| dprint!(x))
            .pipe(|x| self.activation.forward(x))
            .pipe(|x| dprint!(x))
            .pipe(|x| self.pool10.forward(x))
            .pipe(|x| dprint!(x))
            .pipe(|x| self.drop11.forward(x))
            .pipe(|x| dprint!(x))
            .pipe(|x| x.reshape([0, -1]))
            .pipe(|x| dprint!(x))
            .pipe(|x| self.linear12.forward(x))
            .pipe(|x| dprint!(x))
            .pipe(|x| self.activation.forward(x))
            .pipe(|x| dprint!(x))
            .pipe(|x| self.drop13.forward(x))
            .pipe(|x| dprint!(x))
            .pipe(|x| self.linear14.forward(x))
            .pipe(|x| dprint!(x))
            .pipe(|x| self.activation.forward(x))
            .pipe(|x| dprint!(x));

        result
    }
}
