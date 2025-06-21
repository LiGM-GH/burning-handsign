import torch
import msgpack

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


def main():
    # infile = sys.argv[1]
    # outfile = sys.argv[2]

    model = TheModelClass()

    messagepack_data = None
    with open("model.mpk", "rb") as f:
        messagepack_data = msgpack.unpack(f)

    real_data = messagepack_data["item"]
    for elem in list(real_data.keys()):
        print(elem)
        if real_data[elem] is not None:
            for val in list(real_data[elem].keys()):
                print(f"\t{val}")

    # with open('model_state_dict.txt', 'wb') as f:
    #     pickle.dump(model.state_dict(), f)
            
if __name__ == "__main__":
    main()
