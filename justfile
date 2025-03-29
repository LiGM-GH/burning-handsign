# List all commands
default:
    just --list

# Model generation
rust:
    cargo run

# Python conversion to ONNX format (not operating currently)
py:
    uv --directory scripts run to_onnx.py

# Python conversion to ONNX format
pyalt:
    uv --directory scripts run alt_convert.py

# Run sequentially first creation of the model, then its conversion to ONNX
all: rust pyalt
