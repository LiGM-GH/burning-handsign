# List all commands
default:
    just --list

# Model generation
rust-learn:
    (cargo run learn || (cat ./artifacts/valid/epoch-10/Accuracy.log | nview)) && (cat ./artifacts/experiment.log | nview)

# Mean and std generation
rust-mean:
    cargo run mean_std

check:
    cargo check

# # Python conversion to ONNX format (not operating currently)
# py:
#     uv --directory scripts run to_onnx.py

# # Python conversion to ONNX format
# pyalt:
#     uv --directory scripts run alt_convert.py

# # Run sequentially first creation of the model, then its conversion to ONNX
# all: rust pyalt
