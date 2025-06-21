import onnxruntime as ort
from torch import Tensor
from torchvision import transforms
from PIL import Image


def do_guess(model, value):
    preprocess = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=8.853009, std=24.0),
        ]
    )

    image = Image.open(value)
    input_tensor: Tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)
    outputs = model.run(None, {"input.1": input_batch.numpy()})

    return outputs


model_path = "model3.onnx"
session = ort.InferenceSession(model_path)

val = do_guess(
    session,
    "/home/gregory/Documents/mpei/khorev-handsign-login/burning-handsign/user_files/d5c20e7af15c6b67/forge_86cd71d5e73ff230.png",
)

print(val)
