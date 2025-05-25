use clap::Parser;
use claps::Things;
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

    let args = Things::parse();

    match args {
        Things::Mean => mean_std(),
        Things::Learn => learn(),
        Things::Guess => guess(),
        Things::Serve => serve(),
    }
}
