# List all commands
default:
  just --list

# Model generation
learn: check
  cargo run learn
  less ./artifacts/experiment.log

# Mean and std generation
mean: check
  cargo run mean_std

# Run Rust pre-compilation check
check:
  cargo check

# Look at the result of the previous run
peek:
  less ./artifacts/experiment.log
