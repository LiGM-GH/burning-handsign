from sys import stderr
from typing import Any, Callable, Literal, cast
import msgpack  # pyright: ignore[reportMissingTypeStubs]
import re
import numpy
import torch
from torch import nn

type MpkModelBytes = bytes
type MpkModelParamNum = dict[
    Literal["bytes"] | Literal["shape"] | Literal["dtype"], MpkModelBytes | str
]
type MpkModelParam = dict[Literal["id"] | Literal["param"], MpkModelParamNum | str]
type MpkModelLayer = dict[
    Literal["weight"]
    | Literal["bias"]
    | Literal["stride"]
    | Literal["kernel_size"]
    | Literal["dilation"]
    | Literal["groups"]
    | Literal["padding"],
    MpkModelParam | str | list[None] | None,
]
type MpkModelItem = dict[str, MpkModelLayer | None]
type MpkModelMetadata = dict[str, str]
type MpkModel_B = None
type MpkModel = dict[
    Literal["metadata"] | Literal["item"] | Literal["_b"],
    MpkModelMetadata | MpkModelItem | MpkModel_B,
]


class Net(nn.Module):
    conv1: nn.Conv2d | None
    conv2: nn.Conv2d | None
    pool: nn.AdaptiveAvgPool2d | None
    activation: nn.ReLU | None
    linear1: nn.Linear | None
    linear2: nn.Linear | None
    dropout: nn.Dropout | None

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = None
        self.conv2 = None
        self.pool = None
        self.activation = None
        self.linear1 = None
        self.linear2 = None
        self.dropout = None


import numpy as np
import torch
import msgpack


def deserialize_mpk_model(mpk_model: MpkModel):
    """
    Deserialize the MpkModel structure into a format suitable for PyTorch.

    Args:
        mpk_model: The deserialized Model structure.

    Returns:
        model_params: A dictionary where keys are layer names and values are learned parameters (weights, biases).
    """

    model_params = {}

    # Accessing metadata (if needed, can be printed or logged)
    metadata = mpk_model["metadata"]
    print(f"Model Metadata: {metadata}")

    # Accessing items (the layers and their parameters)
    items = cast(MpkModelItem, mpk_model["item"])

    for layer_name, layer in items.items():
        if layer is None:
            continue

        layer_params = {}

        for param_name in ["weight", "bias"]:
            param = layer.get(param_name)
            if param is not None:
                # Extract id and param
                param_id = param["id"]
                param_data = param["param"]

                # Extract bytes, shape, and dtype
                param_bytes = param_data["bytes"]
                param_shape = tuple(param_data["shape"])

                # Deserialize the bytes into a numpy array
                data_array = np.frombuffer(param_bytes, dtype=param_data["dtype"])

                # Reshape the data array to the appropriate shape
                data_array = data_array.reshape(param_shape)

                # Convert numpy array to a PyTorch tensor
                layer_params[param_name] = torch.tensor(data_array)

        model_params[layer_name] = layer_params

        # Extract other layer configurations if needed
        for config in ["stride", "kernel_size", "dilation", "groups", "padding"]:
            if config in layer:
                layer_params[config] = layer[config]

    return model_params


def ok[T](operation: Callable[[], T]) -> T | None:
    try:
        return operation()
    except:
        return None


def keys_recursive(thing: dict[str, Any] | Any, indent_level: int = 0) -> None: # pyright: ignore[reportExplicitAny] # fmt:skip
    if not isinstance(thing, dict):
        return None

    for key in thing.keys(): # pyright: ignore[reportUnknownVariableType] # fmt:skip
        print(" " * indent_level + key)  # pyright: ignore[reportUnknownArgumentType]
        keys_recursive(thing[key], indent_level=indent_level + 4) # pyright: ignore[reportUnknownArgumentType] # fmt:skip


def extract(fname: str) -> torch.nn.Module | None:
    model = Net().to(torch.device("cuda"))

    with open(fname, "rb") as file:
        full_model = cast(MpkModel, msgpack.unpackb(file.read()))

        # keys_recursive(full_model)
        if full_model["metadata"] is None:
            return None

        if "item" not in full_model:
            return None

        item = cast(MpkModelItem, full_model["item"])

        for key in item:
            print(key)
            value = item[key]

            if value is None:
                continue

            print(f"{list(value.keys()) = }")
            weights = value["weight"]

            if not isinstance(weights, dict):
                print("ERROR: WEIGHTS NOT FOUND", file=stderr)
                continue

            params = weights["param"]
            if not isinstance(params, dict):
                print("ERROR: PARAMS NOT DICT", file=stderr)
                continue
            shape = params["shape"]

            if not isinstance(shape, list):
                print("ERROR: SHAPE NOT LIST", file=stderr)
                continue

            if isinstance(shape, str):
                print("ERROR: SIZE  NOT INT ", file=stderr)
                continue

            size = shape[:2]

            kernel_size: list[int] = list(shape[-2:])
            model.conv1 = nn.Conv2d(size[0], size[1], (kernel_size[0], kernel_size[1]))
            model.conv1.weight.data = torch.tensor(
                numpy.array(params["bytes"], dtype="float32"), dtype=torch.float32
            )

            print(f"{size = }")
            print(f"{kernel_size = }")

            if not re.compile(f"conv.*").match(key):
                print("WARNING: KEY NOT Conv2d")
                continue


def main():
    model = extract("../models/model_conv2x2.mpk")


if __name__ == "__main__":
    main()
