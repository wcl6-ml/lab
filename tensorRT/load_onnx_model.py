# try ONNX format
import torch.onnx
import onnxruntime as ort
import numpy as np
# 1. Correct Export
dummy_input = torch.randn((1, 1, 1024)).cuda() # Example RF signal shape

# 2. Setup ONNX Runtime for Benchmarking
providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
session = ort.InferenceSession("rf_classifier.onnx", providers=providers)
input_name = session.get_inputs()[0].name
# 3. Benchmark ONNX
# Convert torch tensor to numpy for ONNX Runtime
numpy_input = dummy_input.cpu().numpy()


output = session.run(None, {input_name: numpy_input})

print(f'type of the ouptuts: {type(output[0])}')
print(f'dimension of outputs: {output[0].shape}')
print("Model loaded and inference successful.")