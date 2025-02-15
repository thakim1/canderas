"""Script to test container dependencies and libs 
Disclaimer: This script was AI GENERATED and slightly modified"""
import torch
import torch.nn as nn
import torch.onnx
import onnx
import tensorrt as trt
import numpy as np
import time

# Step 1: Create a simple CNN model in PyTorch
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 28 * 28, 128)
        self.fc2 = nn.Linear(128, 10)  # 10 classes for MNIST

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Step 2: Instantiate and load the model, move it to GPU
device = torch.device('cpu')

if torch.cuda.is_available():
        device = torch.device('cuda')
        print("Using GPU...")


model = SimpleCNN().to(device)
model.eval()

# Generate some random input data (batch_size=1, channels=1, height=28, width=28)
dummy_input = torch.randn(1, 1, 28, 28).to(device)

# Step 3: Convert the model to ONNX format
onnx_path = 'simple_cnn.onnx'
torch.onnx.export(model, dummy_input, onnx_path, input_names=["input"], output_names=["output"], opset_version=11)

print(f"Model saved as {onnx_path}")

# Step 4: Load and verify the ONNX model
onnx_model = onnx.load(onnx_path)
onnx.checker.check_model(onnx_model)
print("ONNX model is valid!")

# Step 5: Use TensorRT to optimize the ONNX model
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(TRT_LOGGER)
network = builder.create_network(flags=1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser = trt.OnnxParser(network, TRT_LOGGER)

# Parse the ONNX model to TensorRT
with open(onnx_path, 'rb') as f:
    if not parser.parse(f.read()):
        print('Failed to parse the ONNX file.')
        for error in range(parser.num_errors):
            print(parser.get_error(error))
        exit(1)

print("ONNX model successfully parsed to TensorRT.")

# Step 6: Build the TensorRT engine
builder.max_batch_size = 1  # Set maximum batch size


# Create a builder config
config = builder.create_builder_config()

# You can adjust additional configuration options if needed, for example:
# config.set_flag(trt.BuilderFlag.FP16)  # Enable FP16 for faster inference if supported

# Now build the engine using the builder and config
engine = builder.build_engine(network, config)
if engine is None:
    print("Failed to build the TensorRT engine!")
    exit(1)

print("TensorRT engine successfully created.")

# Step 7: Run inference with TensorRT engine

# Allocate memory for input and output buffers
input_data = np.random.random((1, 1, 28, 28)).astype(np.float32)  # Example input

# Allocate memory on the GPU (using torch.cuda)
d_input = torch.tensor(input_data).to(device)  # Input on GPU
d_output = torch.empty((1, 10), dtype=torch.float32).to(device)  # Output tensor on GPU

# Run inference using TensorRT (via the execution context)
start_time = time.time()
# In TensorRT, execute_v2 expects bindings to be pointers, but with PyTorch we'll need to move the data
context = engine.create_execution_context()
context.execute_v2(bindings=[d_input.data_ptr(), d_output.data_ptr()])
end_time = time.time()

# Copy output back to CPU
output_data = d_output.cpu().numpy()

print("Inference completed in {:.6f} seconds".format(end_time - start_time))

# Print the output (class probabilities or raw logits)
print("Output:", output_data)

