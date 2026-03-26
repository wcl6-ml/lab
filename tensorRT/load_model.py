import torch
import torch_tensorrt
import onnx

# Ensure torch_tensorrt is imported before loading to register the necessary components

# Load the model
# The torch_tensorrt.load() function handles both 'exported_program' and 'torchscript' formats
loaded_model = torch_tensorrt.load("rf_classifier.ts")

# The loaded model is a callable PyTorch module
# You can now run inference with it
inputs = [torch.randn((1, 1, 1024)).cuda() ]


output = loaded_model(*inputs)

print(f'type of the ouptuts: {type(output)}')
print(f'dimension of outputs: {output.shape}')
print("Model loaded and inference successful.")