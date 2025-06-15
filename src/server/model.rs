use std::{
    hash::{DefaultHasher, Hash, Hasher},
    panic::catch_unwind,
    path::Path,
};

use axum::{
    http::{Request, StatusCode},
    response::IntoResponse,
};
use tap::Tap;
use tokio::{
    fs::{DirEntry, File, ReadDir},
    io::AsyncWriteExt,
    process::Command,
    task::spawn_blocking,
};
use tower_http::services::ServeFile;

use crate::model::learn;

const NUMBER_OF_SAMPLES: usize = 5;

trait SpawnEach {
    async fn spawn_each(
        self,
        cmd: impl FnMut(DirEntry) -> Command,
    ) -> Result<(), StatusCode>;
}

impl SpawnEach for ReadDir {
    async fn spawn_each(
        mut self,
        mut cmd: impl FnMut(DirEntry) -> Command,
    ) -> Result<(), StatusCode> {
        log::trace!("Enter");
        let mut results = Vec::with_capacity(NUMBER_OF_SAMPLES * 2);

        let mut i = 0;

        while let Ok(Some(entry)) = self.next_entry().await {
            let mut command = cmd(entry);

            log::trace!("Command: {:?}", command);

            let result = command.spawn().map_err(|err| {
                log::error!(
                    "Couldn't spawn process for image preparation: {:?}",
                    err
                );

                StatusCode::INTERNAL_SERVER_ERROR
            })?;

            results.push(result);
            i += 1;
        }

        log::trace!("Middle: i = {i}");

        for mut child in results {
            let Ok(_) = child.wait().await else {
                log::error!("Couldn't process some image");

                return Err(StatusCode::INTERNAL_SERVER_ERROR);
            };
        }

        log::trace!("End");

        Ok(())
    }
}

pub async fn model(
    mut images: axum::extract::Multipart,
) -> Result<impl IntoResponse, StatusCode> {
    let mut hasher = DefaultHasher::new();

    std::time::SystemTime::now().hash(&mut hasher);

    let dirname = format!("user_files/{:x}", hasher.finish());

    let Ok(_) = std::fs::create_dir(&dirname) else {
        log::error!("Create dir failed");

        return Err(StatusCode::INTERNAL_SERVER_ERROR);
    };

    {
        let mut i = 0;

        while let Ok(Some(image)) = images.next_field().await {
            log::trace!("Image {}: {:?}", i, image);

            let Some("image/png" | "image/jpg" | "image/jpeg") =
                image.content_type()
            else {
                return Err(StatusCode::UNSUPPORTED_MEDIA_TYPE);
            };

            let mut hasher = DefaultHasher::new();

            std::time::SystemTime::now().hash(&mut hasher);

            let name = image.file_name().unwrap_or("");

            let name = match name.strip_suffix(".jpg") {
                Some(val) => val,
                None => name,
            };

            let name = match name.strip_suffix(".png") {
                Some(val) => val,
                None => name,
            };

            let mut hasher = DefaultHasher::new();
            name.hash(&mut hasher);

            std::time::SystemTime::now().hash(&mut hasher);

            let full_fname = match image.name() {
                Some("real") => format!("{dirname}/{:x}.png", hasher.finish()),
                Some("forge") => {
                    format!("{dirname}/forge_{:x}.png", hasher.finish())
                }
                _ => return Err(StatusCode::BAD_REQUEST),
            };

            log::trace!("Current filename: {}", full_fname);

            let mut file = File::options()
                .write(true)
                .create_new(true)
                .open(full_fname)
                .await
                .map_err(|err| {
                    log::error!("Error while creating file: {:?}", err);
                    StatusCode::INTERNAL_SERVER_ERROR
                })?;

            let bytes = image.bytes().await.map_err(|err| {
                log::error!("Error while getting file: {:?}", err);
                StatusCode::INTERNAL_SERVER_ERROR
            })?;

            file.write_all(&bytes).await.map_err(|err| {
                log::error!("Error while saving file: {:?}", err);
                StatusCode::INTERNAL_SERVER_ERROR
            })?;

            i += 1;
        }

        if i < NUMBER_OF_SAMPLES {
            return Err(StatusCode::UNPROCESSABLE_ENTITY);
        }
    }

    let files = tokio::fs::read_dir(&dirname).await.map_err(|err| {
        log::error!("Couldn't read dir {}: {}", dirname, err);
        StatusCode::INTERNAL_SERVER_ERROR
    })?;

    files
        .spawn_each(|entry| {
            let fname =
                format!("{dirname}/{}", entry.file_name().to_str().unwrap());

            Command::new("scripts/prep.sh").tap_mut(|command| {
                command.args([&fname, &fname]);
            })
        })
        .await?;

    log::error!("THIS IS STILL EXECUTED 2");

    let artifacts_dir = format!("artifacts_{}", dirname);
    let artifacts_dir_clone = artifacts_dir.clone();

    let learning =
        move || catch_unwind(|| learn(&dirname, &artifacts_dir_clone));

    let model_path = spawn_blocking(learning)
        .await
        .map_err(|err| {
            log::error!("Couldn't Tokio::join the learning: {:?}", err);
            StatusCode::INTERNAL_SERVER_ERROR
        })?
        .map_err(|err| {
            log::error!("Couldn't complete the learning: {:?}", err);
            StatusCode::INTERNAL_SERVER_ERROR
        })?;

    let onnx_path = {
        let artifacts_dir =
            Path::new(&artifacts_dir).canonicalize().map_err(|err| {
                log::error!(
                    "artifacts_dir path coulnd't be canonicalized: {:?}",
                    err
                );
                StatusCode::INTERNAL_SERVER_ERROR
            })?;

        let artifacts_dir = artifacts_dir.to_str().ok_or_else(|| {
            log::error!("artifacts_dir path coulnd't be canonicalized",);
            StatusCode::INTERNAL_SERVER_ERROR
        })?;

        let onnx_path = format!("{artifacts_dir}/model.onnx");

        let model_path =
            Path::new(&model_path).canonicalize().map_err(|err| {
                log::error!(
                    "model_path path coulnd't be canonicalized: {:?}",
                    err
                );
                StatusCode::INTERNAL_SERVER_ERROR
            })?;

        let model_path = model_path
            .to_str()
            .ok_or_else(|| {
                log::error!("model_path path coulnd't be canonicalized",);
                StatusCode::INTERNAL_SERVER_ERROR
            })?
            .to_string();

        println!("ARTIFACTS_DIR: {}", artifacts_dir);
        println!("ONNX_PATH: {}", onnx_path);
        println!("MODEL_PATH: {}", model_path);

        let thing: tokio::process::Child =
            Command::new("scripts/convert_to_onnx.sh")
                .args([model_path, onnx_path.clone()])
                .spawn()
                .map_err(|err| {
                    log::error!("Couldn't spawn process: {:?}", err);
                    StatusCode::INTERNAL_SERVER_ERROR
                })?;

        thing.wait_with_output().await.map_err(|err| {
            log::error!("Coudln't wait for child process: {:?}", err);
            StatusCode::INTERNAL_SERVER_ERROR
        })?;

        onnx_path
    };

    ServeFile::new(onnx_path)
        .try_call(Request::new(""))
        .await
        .map_err(|err| {
            log::error!("Couldn't get model file: {}", err);
            StatusCode::INTERNAL_SERVER_ERROR
        })
}
