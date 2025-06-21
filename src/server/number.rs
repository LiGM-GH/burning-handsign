use std::hash::{DefaultHasher, Hash, Hasher};

use axum::{Json, http::StatusCode, response::IntoResponse};
use serde::{Deserialize, Serialize};
use tap::Tap;
use tokio::{fs::File, io::AsyncWriteExt, process::Command};

pub async fn number(
    mut image: axum::extract::Multipart,
) -> Result<impl IntoResponse, StatusCode> {
    log::trace!("Entered number");

    let mut hasher = DefaultHasher::new();

    std::time::SystemTime::now().hash(&mut hasher);

    let dirname = format!("user_files/{:x}", hasher.finish());

    let Ok(_) = std::fs::create_dir(&dirname) else {
        log::error!("Create dir failed");
        return Err(StatusCode::INTERNAL_SERVER_ERROR);
    };

    let input = match image.next_field().await {
        Ok(Some(input)) => input,
        Err(err) => {
            log::error!("Error happened during getting files: {:?}", err);

            return Err(StatusCode::BAD_REQUEST);
        }
        Ok(None) => {
            log::error!(
                "Error happened during getting files. Unexpected end of input"
            );
            return Err(StatusCode::UNPROCESSABLE_ENTITY);
        }
    };

    let dirpath = std::path::Path::new(&dirname);

    match input.name() {
        Some("image")
            if (input.content_type() == Some("image/jpeg")
                || input.content_type() == Some("image/jpg")
                || input.content_type() == Some("image/png")) =>
        {
            let path = dirpath.join("image.png");

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

            path
        }
        _ => {
            log::error!("Unexpected value");
            return Err(StatusCode::UNPROCESSABLE_ENTITY);
        }
    };

    let fname = "image.png";

    let mut cmd = Command::new(
        std::path::absolute("scripts/numbers.sh").map_err(|err| {
            log::error!("Error while making path absolute: {:?}", err);
            StatusCode::INTERNAL_SERVER_ERROR
        })?,
    );

    let val = cmd
        .args([&fname])
        .current_dir(dirpath)
        .spawn()
        .map_err(|err| {
            log::error!("Error while executing the commands: {:?}", err);
            StatusCode::INTERNAL_SERVER_ERROR
        })?
        .wait()
        .await
        .map_err(|err| {
            log::error!("Error while executing the commands: {:?}", err);
            StatusCode::INTERNAL_SERVER_ERROR
        })?;

    if !val.success() {
        log::error!("Couldn't execute the script: {:?}", val.code());
        return Err(StatusCode::INTERNAL_SERVER_ERROR);
    }

    let path = dirpath.join("result.txt");

    let content = tokio::fs::read_to_string(path).await.map_err(|err| {
        log::error!("Error while reading the file: {:?}", err);

        StatusCode::INTERNAL_SERVER_ERROR
    })?;

    let result = content.parse::<u64>().map_err(|err| {
        log::error!("Error while parsing value: {:?}", err);

        StatusCode::INTERNAL_SERVER_ERROR
    })?;

    #[derive(Serialize, Deserialize)]
    struct UintAndBounds {
        uint: u64,
        x_bound: u32,
        y_bound: u32,
    }

    let file =
        std::fs::read_to_string(dirpath.join("sorted.txt")).map_err(|err| {
            log::error!("Error while reading the file: {:?}", err);
            StatusCode::INTERNAL_SERVER_ERROR
        })?;

    let mut x_values = Vec::new();
    let mut y_values = Vec::new();

    for line in file.lines() {
        let [_classname, x, y, w, h] = line.split(" ").collect::<Vec<_>>()[..]
        else {
            log::error!("Error while reading the file");
            return Err(StatusCode::INTERNAL_SERVER_ERROR);
        };

        fn append(
            values: &mut Vec<f32>,
            x: &str,
            width: &str,
        ) -> Result<(), std::num::ParseFloatError> {
            values.push(x.parse::<f32>()? - width.parse::<f32>()? / 2.0);

            Ok(())
        }

        append(&mut x_values, x, w).map_err(|err| {
            log::error!("Error while reading the file: {:?}", err);

            StatusCode::INTERNAL_SERVER_ERROR
        })?;

        append(&mut y_values, y, h).map_err(|err| {
            log::error!("Error while reading the file: {:?}", err);

            StatusCode::INTERNAL_SERVER_ERROR
        })?;
    }

    let mut x_bound = x_values
        .iter()
        .min_by(|l, r| l.total_cmp(r))
        .ok_or_else(|| {
            log::error!("Error while reading the file: bound x not found");

            StatusCode::INTERNAL_SERVER_ERROR
        })?;

    let mut y_bound = y_values
        .iter()
        .min_by(|l, r| l.total_cmp(r))
        .ok_or_else(|| {
            log::error!("Error while reading the file: bound y not found");

            StatusCode::INTERNAL_SERVER_ERROR
        })?;

    if std::fs::exists(dirpath.join("rotated.txt")).is_ok_and(|val| val) {
        std::mem::swap(&mut x_bound, &mut y_bound);
    }

    let image_width = std::fs::read_to_string(dirpath.join("sizex.txt"))
        .map_err(|err| {
            log::error!("Error while reading the file SIZEX: {:?}", err);
            StatusCode::INTERNAL_SERVER_ERROR
        })?
        .tap(|image_width| {
            dbg!(image_width);
        })
        .trim()
        .parse::<u32>()
        .map_err(|err| {
            log::error!("Error while parsing the file SIZEX: {:?}", err);
            StatusCode::INTERNAL_SERVER_ERROR
        })?;

    let image_height = std::fs::read_to_string(dirpath.join("sizey.txt"))
        .map_err(|err| {
            log::error!("Error while reading the file SIZEY: {:?}", err);
            StatusCode::INTERNAL_SERVER_ERROR
        })?
        .tap(|image_height| {
            dbg!(image_height);
        })
        .trim()
        .parse::<u32>()
        .map_err(|err| {
            log::error!("Error while parsing the file SIZEY: {:?}", err);
            StatusCode::INTERNAL_SERVER_ERROR
        })?;

    let x_bound = (x_bound * image_width as f32).round() as u32;
    let y_bound = (y_bound * image_height as f32).round() as u32;

    Ok(Json(UintAndBounds {
        uint: result,
        x_bound,
        y_bound,
    }))
}
