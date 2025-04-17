use std::{collections::HashSet, ops::ControlFlow};

use burn::{
    backend::Wgpu,
    data::dataset::{
        Dataset,
        vision::{Annotation, ImageFolderDataset},
    },
    tensor::{Float, Shape, Tensor, TensorData},
};

use dataset::HandsignDataset;
use model::ModelConfig;

mod dataset;
mod model;

trait IntoDebug {
    fn into_debug(&self) -> &dyn std::fmt::Debug;
}

impl IntoDebug for Annotation {
    fn into_debug(&self) -> &dyn std::fmt::Debug {
        match self {
            Self::Label(label) => label,
            Self::MultiLabel(labels) => labels,
            Self::BoundingBoxes(vals) => vals,
            Self::SegmentationMask(mask) => mask,
        }
    }
}

fn main() {
    type MyBackend = Wgpu<f32, i32>;

    let device = Default::default();
    let model = ModelConfig::new(10, 512).init::<MyBackend>(&device);

    println!("{}", model);

    let dataset = ImageFolderDataset::hs_test();
    println!("{}", dataset.len());

    let mean = dataset
        .iter()
        .map(|val| {
            TensorData::new(
                val.image
                    .iter()
                    .map(|val| -> u8 { val.clone().try_into().unwrap() })
                    .collect::<Vec<_>>(),
                Shape::new([820, 200, 3]),
            )
        })
        .map(|data| Tensor::<MyBackend, 3>::from_data(data, &device) / 255)
        .reduce(|val1, val2| val1 + val2);

    println!("mean: {:?}", mean);
}
