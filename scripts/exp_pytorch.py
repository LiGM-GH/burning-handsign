import io
import msgpack
import torch
import array
from PIL import Image
from torchvision import transforms

# import pickle

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

# def decode_bytes(param_dict):
#     # Extract bytes, dtype, shape
#     byte_data = param_dict["bytes"]
#     dtype_str = param_dict["dtype"]
#     shape = param_dict["shape"]

#     # Map dtype string to numpy dtype
#     dtype_map = {
#         "F16": np.float16,
#     }
#     dtype = dtype_map[dtype_str]

#     # Convert bytes to numpy array
#     array = np.frombuffer(byte_data, dtype=dtype)
#     array = array.reshape(shape)
#     return array


class TheModelClass(torch.nn.Module):
    def __init__(self):
        super(TheModelClass, self).__init__()
        self.conv1: torch.nn.Conv2d = torch.nn.Conv2d(
            in_channels=1, out_channels=8, kernel_size=11, stride=1
        )
        self.norm2: torch.nn.BatchNorm2d = torch.nn.BatchNorm2d(8)
        self.pool3: torch.nn.MaxPool2d = torch.nn.MaxPool2d(3, stride=2)
        self.conv4: torch.nn.Conv2d = torch.nn.Conv2d(
            in_channels=8, out_channels=16, kernel_size=5, stride=1, padding=2
        )
        self.norm5: torch.nn.BatchNorm2d = torch.nn.BatchNorm2d(16)
        self.pool6: torch.nn.MaxPool2d = torch.nn.MaxPool2d(3, stride=2)
        self.drop7: torch.nn.Dropout = torch.nn.Dropout(0.3)
        self.conv8: torch.nn.Conv2d = torch.nn.Conv2d(
            in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1
        )
        self.conv9: torch.nn.Conv2d = torch.nn.Conv2d(
            in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1
        )
        self.pool10: torch.nn.MaxPool2d = torch.nn.MaxPool2d(3, stride=2)
        self.drop11: torch.nn.Dropout = torch.nn.Dropout(0.3)
        self.linear12: torch.nn.Linear = torch.nn.Linear(82944, 16)
        self.drop13: torch.nn.Dropout = torch.nn.Dropout(0.5)
        self.linear14: torch.nn.Linear = torch.nn.Linear(16, 1)

    def forward(self, x):
        x = self.conv1.forward(x)
        x = self.norm2.forward(x)
        x = self.pool3.forward(x)
        x = self.conv4.forward(x)
        x = self.norm5.forward(x)
        x = self.pool6.forward(x)
        x = self.drop7.forward(x)
        x = self.conv8.forward(x)
        x = self.conv9.forward(x)
        x = self.pool10.forward(x)
        x = self.drop11.forward(x)
        x = x.view(1, 82944)
        x = self.linear12.forward(x)
        x = self.drop13.forward(x)
        x = self.linear14.forward(x)

        return x


def load_state_dict(path: str):
    exceptions = {
        "conv1.weight": (8, 1, 11, 11),
        "norm2.weight": (8),
        "norm2.bias": (8),
        "conv4.weight": (16, 8, 5, 5),
        "norm5.weight": (16),
        "norm5.bias": (16),
        "conv8.weight": (16, 16, 3, 3),
        "conv9.weight": (16, 16, 3, 3),
        "linear12.weight": (16, 82944),
        "linear14.weight": (1, 16),
    }
    exceptional_names = {
        "norm2.gamma": "norm2.weight",
        "norm2.beta": "norm2.bias",
        "norm5.gamma": "norm5.weight",
        "norm5.beta": "norm5.bias",
    }
    structure = {
        "conv1": [
            "weight",
            "bias",
        ],
        "norm2": [
            "gamma",
            "beta",
            "running_mean",
            "running_var",
        ],
        "conv4": ["weight", "bias"],
        "norm5": [
            "gamma",
            "beta",
            "running_mean",
            "running_var",
        ],
        "conv8": ["weight", "bias"],
        "conv9": ["weight", "bias"],
        "linear12": ["weight", "bias"],
        "linear14": ["weight", "bias"],
    }

    thedict = {}

    with open(path, "rb") as f:
        inner_dict = msgpack.unpack(f)

        print(
            f"{torch.frombuffer(inner_dict["item"]["conv1"]["bias"]["param"]["bytes"], dtype=torch.float16) = }"
        )

        inner_dict = inner_dict["item"]

        for key in structure.keys():
            inner_structure = structure[key]
            for inner_key in inner_structure:
                value = inner_dict[key][inner_key]["param"]["bytes"]

                name = key + "." + inner_key

                print(f"{name = }")

                thing = torch.frombuffer(value, dtype=torch.float16)

                if exceptions.get(name) is not None:
                    thing = thing.reshape(exceptions.get(name))

                actual_name = name
                if name in exceptional_names:
                    actual_name = exceptional_names[name]

                print(f"{actual_name = }")

                thedict[actual_name] = thing

    return thedict


def main():
    model = TheModelClass()
    path = "model_2.mpk"
    new_state_dict = load_state_dict(path)

    model.load_state_dict(new_state_dict)

    model.eval()

    input_size = (1, 1, 600, 600)
    args = torch.randn(input_size)

    preprocess = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=8.853009, std=24.0),
        ]
    )

    image = Image.open(
        "/home/gregory/Documents/mpei/khorev-handsign-login/burning-handsign/user_files/d5c20e7af15c6b67/forge_33b158ee71c5b0c4.png"
    )
    x = preprocess(image)
    x = x.unsqueeze(0)

    print(model.forward(x))

    torch.onnx.export(model, args, "model3.onnx", opset_version=13)


if __name__ == "__main__":
    main()
