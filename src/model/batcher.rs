use burn::{
    data::{dataloader::batcher::Batcher, dataset::vision::ImageDatasetItem},
    prelude::*,
};

use crate::model::{
    IMAGE_DEPTH, IMAGE_HEIGHT, IMAGE_LENGTH, normalizer::Normalizer,
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
        let mut all_alike = true;
        let mut images_iter = items.iter();
        let etalon = images_iter.next().unwrap();

        for image in images_iter {
            if image.image != etalon.image
                && image.image_path != etalon.image_path
            {
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
                tensor
            })
            .collect::<Vec<_>>();

        use crate::{MEAN_DS, STDDEV_DS};

        let targets = items
            .iter()
            .map(|val| {
                let is_forge =
                    val.image_path.find("forge").map(|_| 1).unwrap_or(0);

                // log::info!("Image {} is forged: {}", val.image_path, is_forge);

                Tensor::from_data([is_forge.elem::<B::IntElem>()], &self.device)
            })
            .collect::<Vec<_>>();

        let mean = images
            .first()
            .expect("Not a single item in the dir")
            .full_like(MEAN_DS);

        let stddev = mean.full_like(STDDEV_DS);

        let norm = Normalizer::new(&self.device, mean.clone(), stddev);

        let images = images
            .into_iter()
            .map(|img| norm.normalize(img))
            .collect::<Vec<_>>();

        let images: Tensor<B, 4> = Tensor::stack(images, 0).detach();
        let targets = Tensor::cat(targets, 0);

        HandsignBatch { images, targets }
    }
}
