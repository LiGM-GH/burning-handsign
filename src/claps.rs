use clap::{Args, Parser};

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
pub enum Things {
    Serve,
    Learn,
    Guess,
    Mean,
}

#[derive(Args, Debug)]
pub struct Inner {
    nothing: i32,
}
