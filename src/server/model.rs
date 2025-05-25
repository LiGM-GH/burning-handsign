use std::hash::{DefaultHasher, Hash, Hasher};

use axum::{
    http::StatusCode,
    response::{IntoResponse, Redirect},
};
use tokio::{fs::File, io::AsyncWriteExt};

pub async fn model(
    mut images: axum::extract::Multipart,
) -> Result<impl IntoResponse, StatusCode> {
    let mut i = 0;

    while let Ok(Some(image)) = images.next_field().await {
        log::trace!("Image {}: {:?}", i, image);

        if image.content_type() != Some("image/png") {
            return Err(StatusCode::UNPROCESSABLE_ENTITY);
        }

        let mut hasher = DefaultHasher::new();

        std::time::SystemTime::now().hash(&mut hasher);

        let name = image.file_name().unwrap_or("");

        let name = match name.strip_suffix(".png") {
            Some(val) => val,
            None => name,
        };

        let full_fname =
            format!("user_files/{:x}_{}.png", hasher.finish(), name);

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

    Ok(Redirect::to("/"))
}
