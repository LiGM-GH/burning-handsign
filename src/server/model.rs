use std::{
    hash::{DefaultHasher, Hash, Hasher},
    panic::catch_unwind,
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
};
use tower_http::services::ServeFile;

use crate::model::learn;

const NUMBER_OF_SAMPLES: usize = 10;
const FORGE_METHOD_NUMBER: usize = 5;

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
        let mut results = Vec::with_capacity(NUMBER_OF_SAMPLES);

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
            let full_fname = format!("{dirname}/{:x}.png", hasher.finish());

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

            if i >= NUMBER_OF_SAMPLES {
                break;
            }
        }

        if i < NUMBER_OF_SAMPLES {
            return Err(StatusCode::UNPROCESSABLE_ENTITY);
        }
    }

    let files = tokio::fs::read_dir(&dirname).await.map_err(|err| {
        log::error!("Couldn't read dir {}: {}", dirname, err);
        StatusCode::INTERNAL_SERVER_ERROR
    })?;

    {
        let mut i = 0;
        files
            .spawn_each(|entry| {
                let forge_name = format!("scripts/forge{i}.sh");

                i += 1;

                if i >= FORGE_METHOD_NUMBER {
                    i = 0;
                }

                let fname = format!(
                    "{dirname}/{}",
                    entry.file_name().to_str().unwrap()
                );
                let result_fname = format!(
                    "{dirname}/forge_{}",
                    entry.file_name().to_str().unwrap()
                );

                log::trace!("Result fname: {result_fname}");

                Command::new(&forge_name).tap_mut(|command| {
                    command.args([&fname, &result_fname]);
                })
            })
            .await?;
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

    tokio::task::spawn_blocking(move || {
        catch_unwind(|| learn(&dirname, &artifacts_dir_clone))
    })
    .await
    .map_err(|err| {
        log::error!("Couldn't Tokio::join the learning: {:?}", err);

        StatusCode::INTERNAL_SERVER_ERROR
    })?
    .map_err(|err| {
        log::error!("Couldn't complete the learning: {:?}", err);

        StatusCode::INTERNAL_SERVER_ERROR
    })?;

    ServeFile::new(format!("{artifacts_dir}/model.mpk"))
        .try_call(Request::new(""))
        .await
        .map_err(|err| {
            log::error!("Couldn't get model file: {}", err);
            StatusCode::INTERNAL_SERVER_ERROR
        })
}
