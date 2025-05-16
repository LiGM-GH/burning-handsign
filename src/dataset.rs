use burn::data::dataset::vision::ImageFolderDataset;

const TEST_PATH: &str  = "../handwritten-signatures-ver1/png/";
const TRAIN_PATH: &str = "../handwritten-signatures-ver1/png/";

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
