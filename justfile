# List all commands
default:
  just --list

serve: check
  cargo run serve

# Model generation
learn: check
  cargo run --release learn
  less ./artifacts/experiment.log

# Mean and std generation
mean: check
  cargo run mean_std

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
  gum confirm "Proceed?"

# Look at the result of the previous run
peek:
  less ./log/main.log
