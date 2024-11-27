import tensorrt as trt
import torch
import torchvision.models as models
import torch.nn as nn
import torch.onnx

# This is a first test to see if I can load models and correctly transfer to onnx and tensorrt.

def convert_onnx_to_tensorrt(onnx_model_path, trt_model_path):
    # Create the logger and builder
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    
    # Parse the ONNX model
    with open(onnx_model_path, 'rb') as f:
        if not parser.parse(f.read()):
            print("ERROR: Failed to parse ONNX model")
            return
    
    # Build the TensorRT engine with TensorRT 8.X appropriate `build_engine()` 
    engine = builder.build_engine(network, builder.create_builder_config())
    
    # Save the TensorRT engine to a file
    with open(trt_model_path, 'wb') as f:
        f.write(engine.serialize())
    
    print(f"TensorRT model saved to {trt_model_path}")


# Load DenseNet model and set classifier 
model = models.densenet121(pretrained=False)
model.classifier = nn.Sequential(
    nn.Linear(in_features=model.classifier.in_features, out_features=1), 
    nn.Sigmoid()
) # From Fabian's eval script 'evaluate_model.py'
model.load_state_dict(torch.load('../Training/Models/model_DenseNet_epochs_10_lr_0.0003_gammaStepLR_0'))

# Use CUDA if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Dummy in for export (adjust batch size and shape if necessary)
dummy_input = torch.randn(1, 3, 224, 224).to(device)

# Export ONNX
onnx_output_path = "test_model.onnx"
torch.onnx.export(
    model, 
    dummy_input, 
    onnx_output_path, 
    verbose=True, 
    input_names=["input"], 
    output_names=["output"]
)
print(f"Model exported to ONNX at {onnx_output_path}")

# Convert ONNX to TensorRT and save TensorRT model 
trt_model_path = "test_model.trt"
convert_onnx_to_tensorrt(onnx_output_path, trt_model_path)
