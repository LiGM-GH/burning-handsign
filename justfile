# List all commands
default:
  just --list

# Model generation
learn: check
  cargo run --release learn
  less ./artifacts/experiment.log

# Mean and std generation
mean: check
  cargo run --release mean_std

# Guess first picture in other dataset
guess: check
  cargo run --release guess

# Learn and then guess
reguess:
  cargo run --release learn
  cargo run --release guess

# Run Rust pre-compilation check
check:
  cargo check

# Look at the result of the previous run
peek:
  less ./artifacts/experiment.log
