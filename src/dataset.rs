use burn::data::dataset::{
    Dataset,
    vision::{ImageFolderDataset, PixelDepth},
};
use itertools::Itertools;
use rand::{Rng, distr::Bernoulli, seq::SliceRandom};

const TEST_PATH: &str = "../handwritten-signatures-ver1/png/";
const TRAIN_PATH: &str = "../handwritten-signatures-ver1/png/";

pub const IMAGE_LENGTH: usize = 220;
pub const IMAGE_HEIGHT: usize = 155;
pub const IMAGE_DEPTH: usize = 1;

pub trait HandsignDataset {
    fn hs_test() -> Self;
    fn hs_train() -> Self;
}

impl HandsignDataset for ImageFolderDataset {
    fn hs_test() -> Self {
        Self::new_classification(TEST_PATH).unwrap()
    }

    fn hs_train() -> Self {
        Self::new_classification(TRAIN_PATH).unwrap()
    }
}

#[derive(Clone)]
pub struct CedarItem {
    pub left: Vec<PixelDepth>,
    pub right: Vec<PixelDepth>,
    pub is_ok: bool,
}

impl std::fmt::Debug for CedarItem {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CedarItem")
            .field("first", &"Something")
            .field("second", &"Something")
            .field("is_ok", &self.is_ok)
            .finish()
    }
}

pub struct CedarDataset {
    pub pictures: Vec<CedarItem>,
}

impl Dataset<CedarItem> for CedarDataset {
    fn get(&self, index: usize) -> Option<CedarItem> {
        self.pictures.get(index).cloned()
    }

    fn len(&self) -> usize {
        self.pictures.len()
    }
}

impl HandsignDataset for CedarDataset {
    fn hs_test() -> Self {
        let mut rng = rand::rng();
        let orig_path = "../handwritten-signatures-ver1/CEDAR_again_1/full_org";

        let left_seq = std::fs::read_dir(orig_path)
            .expect("Couldn't read dataset dir")
            .flatten()
            .sorted_by_key(|val| val.file_name())
            .take(960)
            .chunks(24)
            .into_iter()
            .flat_map(|val| {
                let mut val = val
                    .map(|entry| {
                        Ok::<_, std::io::Error>((
                            entry.file_type()?,
                            entry.path(),
                        ))
                    })
                    .map(|val| val.expect("Couldn't get file type"))
                    .map(|(ty, path)| ty.is_file().then_some(path))
                    .map(|val| dbg!(val.expect("val isn't file")))
                    .filter(|thing| {
                        thing.extension().and_then(|val| val.to_str())
                            == Some("png")
                    })
                    .collect::<Vec<_>>();

                val.shuffle(&mut rng);

                println!("Images in a single batch: {:#?}", val);

                val
            })
            .map(image::open)
            .map(|val| val.expect("Image couldn't open this"))
            .map(|image| {
                image
                    .into_luma8()
                    .iter()
                    .map(|px| PixelDepth::U8(*px))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        let orig_chunks = std::fs::read_dir(orig_path)
            .expect("Couldn't read dataset dir")
            .flatten()
            .sorted_by_key(|val| val.file_name())
            .take(960)
            .chunks(24);

        let orig = orig_chunks
            .into_iter()
            .flat_map(|val| {
                let val = val
                    .map(|entry| {
                        Ok::<_, std::io::Error>((
                            entry.file_type()?,
                            entry.path(),
                        ))
                    })
                    .map(|val| val.expect("Couldn't get file type"))
                    .map(|(ty, path)| ty.is_file().then_some(path))
                    .map(|val| dbg!(val.expect("val isn't file")))
                    .filter(|thing| {
                        thing.extension().and_then(|val| val.to_str())
                            == Some("png")
                    })
                    .collect::<Vec<_>>();

                val
            })
            .collect::<Vec<_>>();

        let forge_path = "../handwritten-signatures-ver1/CEDAR_again_1/full_forg";

        let seq = rng.sample_iter(
            Bernoulli::new(0.5).expect("Couldn't create Bernoulli"),
        );

        let forge = std::fs::read_dir(forge_path)
            .expect("Couldn't read dataset dir")
            .flatten()
            .sorted_by_key(|val| val.file_name())
            .take(960)
            .map(|entry| {
                Ok::<_, std::io::Error>((entry.file_type()?, entry.path()))
            })
            .map(|val| val.expect("Couldn't get file type"))
            .map(|(ty, path)| ty.is_file().then_some(path))
            .map(|val| dbg!(val.expect("val isn't file")))
            .filter(|thing| {
                thing.extension().and_then(|val| val.to_str()) == Some("png")
            });

        let result = orig.into_iter().zip(forge).zip(seq).map(
            |((first, second), get_original)| {
                println!("first: {:?}, second: {:?}", first, second);

                if get_original {
                    (first, true)
                } else {
                    (second, false)
                }
            },
        );

        let right_seq = result
            .map(|(image, is_ok)| (image::open(image), is_ok))
            .map(|(val, is_ok)| (val.expect("Image couldn't open this"), is_ok))
            .map(|(image, is_ok)| {
                (
                    image
                        .into_luma8()
                        .iter()
                        .map(|px| PixelDepth::U8(*px))
                        .collect::<Vec<_>>(),
                    is_ok,
                )
            })
            .collect::<Vec<_>>();

        let pictures = left_seq
            .into_iter()
            .zip(right_seq)
            .map(|(left, (right, is_ok))| CedarItem { left, right, is_ok })
            .collect();

        Self { pictures }
    }

    fn hs_train() -> Self {
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use super::{CedarDataset, HandsignDataset};

    #[test]
    fn test_cedar() {
        let ds = CedarDataset::hs_test();
    }
}
