# List all commands
default:
  just --list

# Start the server
serve: pre-run
  cargo run serve

# Learn and look at the logs
fun: learn peek

# Model generation
learn: pre-run
  cargo run --release learn

# Mean and std generation
mean: pre-run
  cargo run mean_std

# Guess first picture in other dataset
guess: pre-run
  cargo run --release guess

# Learn and then guess
reguess:
  cargo run --release learn
  cargo run --release guess

# Run Rust pre-compilation check
check:
  cargo check

[private]
pre-run: check
  gum confirm "Proceed?"

# Look at the result of the previous run
peek:
  less -I ./log/main.log
