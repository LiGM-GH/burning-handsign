from typing import Any, cast
import keras
from tensorflow import TensorSpec
import torch
import msgpack
import tf2onnx
import sys

CONV1_KERSIZE = 11
POOL3_STRIDE = 2
POOL3_KERSIZE = 3
CONV4_KERSIZE = 5
CONV4_PADDING = 2
POOL6_KERSIZE = 3
POOL6_STRIDE = 2
CONV8_KERSIZE = 3
CONV8_PADDING = 1
CONV8_STRIDE = 1
CONV9_PADDING = 1
CONV9_STRIDE = 1
POOL10_KERSIZE = 3
POOL10_STRIDE = 2

IMAGE_LENGTH = 600
IMAGE_HEIGHT = 600
IMAGE_DEPTH = 1

import numpy as np


def decode_bytes(param_dict):
    # Extract bytes, dtype, shape
    byte_data = param_dict["bytes"]
    dtype_str = param_dict["dtype"]
    shape = param_dict["shape"]

    # Map dtype string to numpy dtype
    dtype_map = {
        "F16": np.float16,
    }
    dtype = dtype_map[dtype_str]

    # Convert bytes to numpy array
    array = np.frombuffer(byte_data, dtype=dtype)
    array = array.reshape(shape)
    return array


def main():
    infile = sys.argv[1]
    outfile = sys.argv[2]

    model = keras.Sequential(
        [
            keras.layers.Conv2D(
                filters=8, kernel_size=11, strides=1, input_shape=(600, 600, 1)
            ),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPooling2D(pool_size=3, strides=2),
            keras.layers.ZeroPadding2D(padding=2),
            keras.layers.Conv2D(filters=16, kernel_size=5, strides=1),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPooling2D(pool_size=3, strides=2),
            keras.layers.Dropout(0.3),
            keras.layers.ZeroPadding2D(padding=1),
            keras.layers.Conv2D(filters=16, kernel_size=3, strides=1),
            keras.layers.ZeroPadding2D(padding=CONV9_PADDING),
            keras.layers.Conv2D(filters=16, kernel_size=3, strides=1),
            keras.layers.MaxPooling2D(pool_size=POOL10_KERSIZE, strides=POOL10_STRIDE),
            keras.layers.Dropout(0.3),
            keras.layers.Flatten(),
            keras.layers.Dense(16),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(1),
        ]
    )

    with open(infile, "rb") as f:
        data: dict[str, Any] = cast(dict[str, Any], msgpack.load(f))

        for layer in data["item"].keys():
            print(layer)

        for layer in model.layers:
            print(layer.name)

        layer_name_map = {
            "conv1": "conv2d",
            "norm2": "batch_normalization",
            "conv4": "conv2d_1",
            "norm5": "batch_normalization_1",
            "conv8": "conv2d_2",
            "conv9": "conv2d_3",
            "linear12": "dense",
            "linear14": "dense_1",
        }

        for layer_name, keras_layer_name in layer_name_map.items():
            print(f"{layer_name = }")
            print(f"{keras_layer_name = }")

            layer = model.get_layer(name=keras_layer_name)
            layer_data = cast(dict[str, Any], data["item"][layer_name])
            weights = []

            if "weight" in layer_data:
                weight_array = decode_bytes(layer_data["weight"]["param"])
                print(f"{weight_array.shape = }")
                if len(weight_array.shape) == 4:
                    weight_array = weight_array.transpose(2, 3, 1, 0)
                weights.append(weight_array)

            if "bias" in layer_data and layer_data["bias"] is not None:
                bias_array = decode_bytes(layer_data["bias"]["param"])
                weights.append(bias_array)

            if weights:
                layer.set_weights(weights)

    model.output_names = ["output"]

    tf2onnx.convert.from_keras(
        model,
        input_signature=[
            TensorSpec(model.inputs[0].shape, model.inputs[0].dtype, name="digit")
        ],
        opset=13,
        output_path=outfile,
    )


if __name__ == "__main__":
    main()
