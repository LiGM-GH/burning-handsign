use clap::Parser;
use claps::CliCommands;
use dataset::mean_std;
use log4rs::config::Deserializers;
use model::{guess, learn};
use server::serve;

mod claps;
mod dataset;
mod model;
mod server;

const MEAN_DS: f64 = 8.853009;
const STDDEV_DS: f64 = 24.0;

fn main() {
    log4rs::init_file("log4rs.yaml", Deserializers::new()).ok();

    let args = CliCommands::parse();

    match args {
        CliCommands::Mean { dataset_path } => mean_std(&dataset_path),
        CliCommands::Learn { dataset_path } => learn(&dataset_path, "artifacts"),
        CliCommands::Guess {
            guess_path,
            model_path,
        } => guess(&guess_path, &model_path),
        CliCommands::Serve => serve(),
    }
}
