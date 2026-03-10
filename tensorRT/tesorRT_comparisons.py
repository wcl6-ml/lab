import torch
import torch_tensorrt


import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN1D_4Layer(nn.Module):
    def __init__(self, num_classes=4):
        super(CNN1D_4Layer, self).__init__()

        # Conv Layer 1
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=7, padding=3)
        self.pool1 = nn.MaxPool1d(kernel_size=2)

        # Conv Layer 2
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool1d(2)

        # Conv Layer 3
        self.conv3 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.pool3 = nn.MaxPool1d(2)

        # Conv Layer 4
        self.conv4 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool1d(2)

        # After 4 poolings: 1024 → 64
        self.fc = nn.Linear(128 * 64, num_classes)

    def forward(self, x):

        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = self.pool4(F.relu(self.conv4(x)))

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x




model = CNN1D_4Layer(num_classes=4).eval().cuda()

#  Compile with TensorRT backend
optimized_model = torch.compile(model, backend="torch_tensorrt")

# First run will be slow (compilation), second run will be fast!
dummy_input = torch.randn((1, 1, 1024)).cuda() # Example RF signal shape
output = optimized_model(dummy_input)


# Prepare your inputs (TensorRT needs to know the exact shape/type to optimize)
inputs = [torch_tensorrt.Input((1, 1, 1024), dtype=torch.float32)]

# Compile to a TensorRT Graph Module
trt_model = torch_tensorrt.compile(model, ir="dynamo", inputs=inputs)

# Save for deployment (This is what goes to your k3s edge container)
torch.export.save(trt_model, "rf_classifier.ts")

import time
start = time.time()
for _ in range(100): _ = trt_model(dummy_input)
print(f"TensorRT Avg Latency: {(time.time() - start)/100:.4f}s")

start = time.time()
for _ in range(100): _ = model(dummy_input)
print(f"NoramlModel Avg Latency: {(time.time() - start)/100:.4f}s")