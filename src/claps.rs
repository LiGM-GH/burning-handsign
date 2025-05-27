use clap::{Args, Parser};

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
pub enum Things {
    Serve,
    Learn {
        dataset_path: String,
    },
    Guess {
        guess_path: String,
        #[arg(default_value = "artifacts/model.mpk")]
        model_path: String,
    },
    Mean {
        dataset_path: String,
    },
}

#[derive(Args, Debug)]
pub struct Inner {
    nothing: i32,
}
