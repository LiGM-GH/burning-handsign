use burn::{
    data::{dataloader::batcher::Batcher, dataset::vision::ImageDatasetItem},
    prelude::*,
};

use crate::{
    MEAN_DS, STDDEV_DS,
    dataset::CedarItem,
    model::{IMAGE_DEPTH, IMAGE_HEIGHT, IMAGE_LENGTH, normalizer::Normalizer},
};

#[derive(Debug, Clone)]
pub struct CedarBatch<B: Backend> {
    pub left: Tensor<B, 4>,
    pub right: Tensor<B, 4>,
    pub targets: Tensor<B, 1, Int>,
}

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
        let etalon = images_iter.next().unwrap();

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

        let targets = items
            .iter()
            .map(|val| {
                let is_forge =
                    val.image_path.find("forge").map(|_| 1).unwrap_or(0);

                log::info!("Image {} is forged: {}", val.image_path, is_forge);

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

        log::info!("images.first(): {}", images.first().unwrap());
        let images: Tensor<B, 4> = Tensor::stack(images, 0).detach();
        log::info!("Images: {}", images);
        let targets = Tensor::cat(targets, 0);
        log::info!("Targets: {}", targets);

        HandsignBatch { images, targets }
    }
}

impl<B: Backend> Batcher<CedarItem, CedarBatch<B>> for HandsignBatcher<B> {
    fn batch(&self, items: Vec<CedarItem>) -> CedarBatch<B> {
        let left = items
            .iter()
            .map(|val| {
                TensorData::new(
                    val.first
                        .iter()
                        .map(|val| -> u8 { val.clone().try_into().unwrap() })
                        .collect::<Vec<_>>(),
                    Shape::new([IMAGE_LENGTH, IMAGE_HEIGHT, IMAGE_DEPTH]),
                )
            })
            .map(|val| {
                let thing: Tensor<B, 3> = Tensor::from_data(val, &self.device);
                thing
            })
            .map(|tensor| {
                let tensor = tensor.swap_dims(0, 2).swap_dims(1, 2);
                log::error!("These are new dims: {:?}", tensor.dims());
                tensor
            })
            .collect::<Vec<_>>();

        let right = items
            .iter()
            .map(|val| {
                TensorData::new(
                    val.second
                        .iter()
                        .map(|val| -> u8 { val.clone().try_into().unwrap() })
                        .collect::<Vec<_>>(),
                    Shape::new([IMAGE_LENGTH, IMAGE_HEIGHT, IMAGE_DEPTH]),
                )
            })
            .map(|val| {
                let thing: Tensor<B, 3> = Tensor::from_data(val, &self.device);
                thing
            })
            .map(|tensor| {
                let tensor = tensor.swap_dims(0, 2).swap_dims(1, 2);
                log::error!("These are new dims: {:?}", tensor.dims());
                tensor
            })
            .collect::<Vec<_>>();

        let targets = items
            .iter()
            .map(|val| {
                let is_forge = val.is_ok;

                log::info!("Image is forged: {}", is_forge);

                Tensor::from_data([is_forge.elem::<B::IntElem>()], &self.device)
            })
            .collect::<Vec<_>>();

        let mean = right
            .first()
            .expect("Not a single item in the dir")
            .full_like(MEAN_DS);

        let stddev = mean.full_like(STDDEV_DS);

        let norm = Normalizer::new(&self.device, mean.clone(), stddev);

        let left = left
            .into_iter()
            .map(|img| norm.normalize(img))
            .collect::<Vec<_>>();

        let right = right
            .into_iter()
            .map(|img| norm.normalize(img))
            .collect::<Vec<_>>();

        let left: Tensor<B, 4> = Tensor::stack(left, 0);
        let right: Tensor<B, 4> = Tensor::stack(right, 0);
        let targets = Tensor::cat(targets, 0);

        CedarBatch {
            left,
            right,
            targets,
        }
    }
}
