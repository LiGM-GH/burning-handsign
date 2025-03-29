import numpy as np
import torch
import torch.nn as nn
import msgpack

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3)  # Adjust channels as needed
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.5)
        self.linear1 = nn.Linear(32 * 6 * 6, 128)  # Example dimensions, adjust based on your input
        self.linear2 = nn.Linear(128, 10)  # Output classes

    def forward(self, x):
        x = self.pool(self.conv1(x))
        x = self.pool(self.conv2(x))
        x = x.view(-1, 32 * 6 * 6)  # Flatten the tensor
        x = self.dropout(x)
        x = self.linear1(x)
        x = self.linear2(x)
        return x

def deserialize_mpk_model(mpk_model):
    """
    Deserialize the MpkModel structure into model parameters for PyTorch.
    
    Returns:
        model: An instance of MyModel with loaded weights and biases.
    """
    
    model = MyModel()
    
    items = mpk_model['item']
    
    for layer_name, layer in items.items():
        if layer is None:
            continue
        
        for param_name in ['weight', 'bias']:
            param = layer.get(param_name)

            if param is None:
                continue

            param_data = param['param']
            param_bytes = param_data['bytes']
            param_shape = tuple(param_data['shape'])

            data_array = np.frombuffer(param_bytes, dtype="float32")
            data_array = data_array.reshape(param_shape)

            tensor_param = torch.tensor(data_array)

            if layer_name == 'conv1':
                if param_name == 'weight':
                    model.conv1.weight.data = tensor_param
                elif param_name == 'bias':
                    model.conv1.bias.data = tensor_param
            elif layer_name == 'conv2':
                if param_name == 'weight':
                    model.conv2.weight.data = tensor_param
                elif param_name == 'bias':
                    model.conv2.bias.data = tensor_param
            elif layer_name == 'linear1':
                if param_name == 'weight':
                    model.linear1.weight.data = tensor_param
                elif param_name == 'bias':
                    model.linear1.bias.data = tensor_param
            elif layer_name == 'linear2':
                if param_name == 'weight':
                    model.linear2.weight.data = tensor_param
                elif param_name == 'bias':
                    model.linear2.bias.data = tensor_param

    return model

def main() -> None:
    with open("../models/model_conv2x2.mpk", "rb") as file:
        deserialized_model_data = msgpack.unpackb(file.read())
        model = deserialize_mpk_model(deserialized_model_data)
        # torch.onnx.export(model, f="../models/model_conv2x2.onnx")

if __name__ == "__main__":
    main()
