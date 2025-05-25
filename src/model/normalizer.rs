use burn::prelude::*;

pub struct Normalizer<B: Backend> {
    pub mean: Tensor<B, 3>,
    pub stddev: Tensor<B, 3>,
}

#[allow(unused)]
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

        result
    }

    pub fn to_device(&self, device: &B::Device) -> Self {
        Self {
            mean: self.mean.clone().to_device(device),
            stddev: self.stddev.clone().to_device(device),
        }
    }
}
