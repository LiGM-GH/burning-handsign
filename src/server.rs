use axum::{
    Router,
    extract::DefaultBodyLimit,
    http::{Request, StatusCode},
    response::{Html, IntoResponse},
    routing::{get, post},
};
use axum_server::tls_rustls::RustlsConfig;
use tera::Context;

use model::model;
use std::net::SocketAddr;

mod model;

#[tokio::main]
pub async fn serve() {
    let app = Router::new()
        .route("/", get(get_view))
        .route("/static/main.js", get(get_js))
        .route("/static/main.css", get(get_css))
        .route("/model", post(model))
        .layer(DefaultBodyLimit::max(
            const {
                1024 * 1024 * 12 // 12 MB
            },
        ));

    let addr = SocketAddr::from(([0, 0, 0, 0], 3000));

    let config = RustlsConfig::from_pem_file("keys/cert.pem", "keys/key_pkcs8.pem")
        .await
        .unwrap();

    axum_server::bind_rustls(addr, config)
        .serve(app.into_make_service())
        .await
        .unwrap();
}

async fn get_view() -> Result<impl IntoResponse, StatusCode> {
    let view = get_view_inner().await;

    log::trace!("View: {:?}", view.as_ref().map(|_| ()));

    view
}

async fn get_view_inner()
-> Result<impl IntoResponse + std::fmt::Debug, StatusCode> {
    log::trace!("View");

    let tera = tera::Tera::new("templates/**/*.html")
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    let ctx = Context::new();

    let thing = tera.render("view.html", &ctx).map_err(|err| {
        log::error!("Error while rendering: {}", err);

        StatusCode::IM_A_TEAPOT
    })?;

    Ok(Html::from(thing))
}

async fn get_js() -> Result<impl IntoResponse, StatusCode> {
    log::trace!("GET::JS");

    let mut file = tower_http::services::ServeFile::new("static/main.js");

    file.try_call(Request::new(""))
        .await
        .map_err(|_| StatusCode::BAD_REQUEST)
}

async fn get_css() -> Result<impl IntoResponse, StatusCode> {
    log::trace!("GET::CSS");

    let mut file = tower_http::services::ServeFile::new("static/main.css");

    file.try_call(Request::new(""))
        .await
        .map_err(|_| StatusCode::BAD_REQUEST)
}
