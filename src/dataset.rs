use burn::data::dataset::{
    Dataset,
    vision::{ImageFolderDataset, PixelDepth},
};
use rand::{Rng, distr::Bernoulli};

const TEST_PATH: &str = "../handwritten-signatures-ver1/png/";
const TRAIN_PATH: &str = "../handwritten-signatures-ver1/png/";

pub const IMAGE_LENGTH: usize = 600;
pub const IMAGE_HEIGHT: usize = 600;
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
    pub first: Vec<PixelDepth>,
    pub second: Vec<PixelDepth>,
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
        let rng = rand::rng();
        let seq = rng.sample_iter(
            Bernoulli::new(0.5).expect("Couldn't create Bernoulli"),
        );

        let orig_path = "/home/gregory/Documents/mpei/khorev-handsign-login/handwritten-signatures-ver1/CEDAR_again/full_org";

        let orig = std::fs::read_dir(orig_path)
            .expect("Couldn't read dataset dir")
            .flatten()
            .take(100)
            .map(|entry| {
                Ok::<_, std::io::Error>((entry.file_type()?, entry.path()))
            })
            .map(|val| val.expect("Couldn't get file type"))
            .map(|(ty, path)| ty.is_file().then_some(path))
            .map(|val| dbg!(val.expect("val isn't file")))
            .filter(|thing| {
                thing.extension().and_then(|val| val.to_str()) == Some("png")
            });

        let forge_path = "/home/gregory/Documents/mpei/khorev-handsign-login/handwritten-signatures-ver1/CEDAR_again/full_forg";

        let forge = std::fs::read_dir(forge_path)
            .expect("Couldn't read dataset dir")
            .flatten()
            .take(100)
            .map(|entry| {
                Ok::<_, std::io::Error>((entry.file_type()?, entry.path()))
            })
            .map(|val| val.expect("Couldn't get file type"))
            .map(|(ty, path)| ty.is_file().then_some(path))
            .map(|val| dbg!(val.expect("val isn't file")))
            .filter(|thing| {
                thing.extension().and_then(|val| val.to_str()) == Some("png")
            });

        let result =
            orig.zip(forge)
                .zip(seq)
                .map(|((first, second), get_original)| {
                    if get_original {
                        (first, true)
                    } else {
                        (second, false)
                    }
                });

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

        let left_seq = std::fs::read_dir(orig_path)
            .expect("Couldn't read dataset dir")
            .flatten()
            .take(100)
            .map(|entry| {
                Ok::<_, std::io::Error>((entry.file_type()?, entry.path()))
            })
            .map(|val| val.expect("Couldn't get file type"))
            .map(|(ty, path)| ty.is_file().then_some(path))
            .map(|val| dbg!(val.expect("val isn't file")))
            .filter(|thing| {
                thing.extension().and_then(|val| val.to_str()) == Some("png")
            })
            .map(image::open)
            .map(|val| val.expect("Image couldn't open this"))
            .map(|image| {
                image
                    .into_luma8()
                    .iter()
                    .map(|px| PixelDepth::U8(*px))
                    .collect::<Vec<_>>()
            });

        let pictures = left_seq
            .zip(right_seq)
            .map(|(first, (second, is_ok))| CedarItem {
                first,
                second,
                is_ok,
            })
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
