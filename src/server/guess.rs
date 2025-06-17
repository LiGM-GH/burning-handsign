use std::{
    hash::{DefaultHasher, Hash, Hasher},
    panic::catch_unwind,
};

use axum::{Json, http::StatusCode, response::IntoResponse};
use tokio::{fs::File, io::AsyncWriteExt, process::Command};

pub async fn guess(
    mut input: axum::extract::Multipart,
) -> Result<impl IntoResponse, StatusCode> {
    log::info!("Got guess request");
    let mut hasher = DefaultHasher::new();

    std::time::SystemTime::now().hash(&mut hasher);

    let dirname = format!("user_files/{:x}", hasher.finish());

    let Ok(_) = std::fs::create_dir(&dirname) else {
        log::error!("Create dir failed");

        return Err(StatusCode::INTERNAL_SERVER_ERROR);
    };

    let mut model = None;
    let mut image = None;
    let mut x_bound = None;
    let mut y_bound = None;

    'this_for: for _ in 0..4 {
        let input = match input.next_field().await {
            Ok(Some(input)) => input,
            Err(err) => {
                log::error!("Error happened during getting files: {:?}", err);

                return Err(StatusCode::BAD_REQUEST);
            }
            Ok(None) => {
                break 'this_for
            }
        };

        match input.name() {
            Some("image")
                if image.is_none()
                    && (input.content_type() == Some("image/jpeg")
                        || input.content_type() == Some("image/jpg")
                        || input.content_type() == Some("image/png")) =>
            {
                let path = std::path::Path::new(&dirname).join("image.png");

                let mut file = File::create(&path).await.map_err(|err| {
                    log::error!("Couldn't open the image: {:?}", err);
                    StatusCode::INTERNAL_SERVER_ERROR
                })?;

                file.write_all(
                    input
                        .bytes()
                        .await
                        .map_err(|err| {
                            log::error!(
                                "Couldn't get input image bytes: {:?}",
                                err
                            );

                            StatusCode::INTERNAL_SERVER_ERROR
                        })?
                        .into_iter()
                        .collect::<Vec<_>>()
                        .as_slice(),
                )
                .await
                .map_err(|err| {
                    log::error!("Error while writing to the file: {:?}", err);

                    StatusCode::INTERNAL_SERVER_ERROR
                })?;

                image = Some(path);
            }
            Some("model") => {
                let path = std::path::Path::new(&dirname).join("model.mpk");

                let mut file = File::create(&path).await.map_err(|err| {
                    log::error!("Couldn't open the model: {:?}", err);
                    StatusCode::INTERNAL_SERVER_ERROR
                })?;

                file.write_all(
                    input
                        .bytes()
                        .await
                        .map_err(|err| {
                            log::error!(
                                "Couldn't get input model bytes: {:?}",
                                err
                            );

                            StatusCode::INTERNAL_SERVER_ERROR
                        })?
                        .into_iter()
                        .collect::<Vec<_>>()
                        .as_slice(),
                )
                .await
                .map_err(|err| {
                    log::error!("Error while writing to the file: {:?}", err);

                    StatusCode::INTERNAL_SERVER_ERROR
                })?;

                model = Some(path);
            }
            Some("x_bound") => {
                x_bound = Some(
                    input
                        .text()
                        .await
                        .map_err(|err| {
                            log::error!("Couldn't read x_bound: {:?}", err);
                            StatusCode::INTERNAL_SERVER_ERROR
                        })?
                        .parse::<u32>()
                        .map_err(|err| {
                            log::error!("Couldn't read x_bound: {:?}", err);
                            StatusCode::INTERNAL_SERVER_ERROR
                        })?,
                );
            }
            Some("y_bound") => {
                y_bound = Some(
                    input
                        .text()
                        .await
                        .map_err(|err| {
                            log::error!("Couldn't read y_bound: {:?}", err);
                            StatusCode::INTERNAL_SERVER_ERROR
                        })?
                        .parse::<u32>()
                        .map_err(|err| {
                            log::error!("Couldn't read y_bound: {:?}", err);
                            StatusCode::INTERNAL_SERVER_ERROR
                        })?,
                );
            }
            _ => return Err(StatusCode::UNPROCESSABLE_ENTITY),
        }
    }

    log::trace!("model: {:?}", model);
    let Some(model) = model else {
        log::error!("model nonexistent");
        return Err(StatusCode::BAD_REQUEST);
    };

    let Some(model) = model.to_str() else {
        log::error!("model had non-unicode chars in path");
        return Err(StatusCode::BAD_REQUEST);
    };

    let Some(model) = model.strip_suffix(".mpk") else {
        log::error!("model didn't have .mpk suffix");
        return Err(StatusCode::BAD_REQUEST);
    };

    log::trace!("image: {:?}", image);
    let Some(image) = image else {
        log::error!("image nonexistent");
        return Err(StatusCode::BAD_REQUEST);
    };

    log::trace!("x_bound: {:?}", x_bound);
    let Some(x_bound) = x_bound else {
        log::error!("x_bound nonexistent");
        return Err(StatusCode::BAD_REQUEST);
    };

    log::trace!("y_bound: {:?}", y_bound);
    let Some(y_bound) = y_bound else {
        log::error!("y_bound nonexistent");
        return Err(StatusCode::BAD_REQUEST);
    };

    let mut child = Command::new("scripts/prep_for_guess.sh")
        .arg(&image)
        .arg(&image)
        .arg(x_bound.to_string())
        .arg(y_bound.to_string())
        .spawn()
        .map_err(|err| {
            log::error!("Couldn't spawn child process: {:?}", err);
            StatusCode::INTERNAL_SERVER_ERROR
        })?;

    child.wait().await.map_err(|err| {
        log::error!("Couldn't execute preparation of the image: {:?}", err);
        StatusCode::INTERNAL_SERVER_ERROR
    })?;

    Ok(Json(
        catch_unwind(|| crate::model::guess_single(image, model)).map_err(
            |err| {
                log::error!("Panic happened during guessing: {:?}", err);

                StatusCode::INTERNAL_SERVER_ERROR
            },
        )??,
    ))
}
