# What is this?
This is an AI model training repo.
With this I'll train AI model to check static handwritten signatures for stability.

# How to run?
Just
```sh
just
```
will list all available options.
To run rust part,
```sh
just rust
```
To run python part,
```sh
just pyalt
```
To run all at once,
```sh
just all
```

### Why is Python here?
Unfortunately, Rust's Burn library doesn't currently support direct ONNX export which is needed by frontend, so python's here to make it easier
