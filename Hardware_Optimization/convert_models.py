""" Test script converst all .pth models to the .ONNX and .TRT format:
(1) Loading trained model from ../Training/Models/...
(2) Converting model to ONNX and save as .onnx
(3) Converting .onnx to tensorRT model and save as .trt
(4) Testing .trt model inference on image data from Images/...
"""

import os
import torch
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import torchvision.models as models
import torch.onnx
import torch.nn as nn
import torch.utils
from helper_functions import load_test_images, load_trt_engine
from helper_functions import allocate_buffers, perform_inference

def convert_onnx_to_tensorrt(onnx_model_path, trt_model_path):
    """
    Docstring content...

    Args: 
        onnx_model_path: ...
        trt_model_path: ...
    
    Returns: 
        

    
    """
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    
    # Parse the ONNX model
    with open(onnx_model_path, 'rb') as onnx_model:
        if not parser.parse(onnx_model.read()):
            print("ERROR: Failed to parse ONNX model")
            return
    
    engine = builder.build_engine(network, builder.create_builder_config())
    
    # Save the TensorRT engine to .trt file
    with open(trt_model_path, 'wb') as trt_model:
        trt_model.write(engine.serialize())
    
    print(f"TensorRT model saved to {trt_model_path}")


def get_model(model_name):
    """
    Docstring content...

    Args: 
        model_name: ...

    Returns: 
        model: ...

    """  

    # Load the model and change classification layer
    if model_name == 'DenseNet':
        model = models.densenet121(pretrained=False)
        model.classifier = nn.Sequential(
            nn.Linear(in_features=model.classifier.in_features, out_features=1), 
            nn.Sigmoid()
        ) # From Fabian's eval script 'evaluate_model.py'
        model.load_state_dict(torch.load('../Training/Models/DenseNet_KR_Dataset.pth'))

    if model_name == 'MobileNetV2':
        model = models.mobilenet_v2(pretrained=False)
        model.classifier[1] = nn.Sequential(nn.Linear(in_features=1280, out_features=1), nn.Sigmoid())
        
        model.load_state_dict(torch.load('../Training/Models/MobileNetV2_KR_Dataset.pth'))
    
    if model_name == 'MobileNetV3':
        model = models.mobilenet_v3_small(pretrained=False)
        model.classifier[3] = nn.Sequential(nn.Linear(in_features=1024, out_features=1), nn.Sigmoid())
        model.load_state_dict(torch.load('../Training/Models/MobileNetV3_KR_Dataset.pth'))    
    
    if model_name == 'MobileNetV3_Large':
        model = models.mobilenet_v3_large(pretrained=False)
        model.classifier[3] = nn.Sequential(nn.Linear(in_features=1280, out_features=1), nn.Sigmoid())        
        model.load_state_dict(torch.load('../Training/Models/MobileNetV3_big_KR_Dataset.pth'))
    
    if model_name == 'ResNet18':
        model = models.resnet18(pretrained=False)
        model.fc = nn.Sequential(nn.Linear(model.fc.in_features, 1), nn.Sigmoid())
        model.load_state_dict(torch.load('../Training/Models/ResNet18_KR_Dataset.pth'))        

    return model



if __name__ == '__main__':

    img_folder = "../Images"
    trained_models = ['DenseNet', 'MobileNetV2', 'MobileNetV3', 'MobileNetV3_Large', 'ResNet18']

    for model_name in trained_models:
        # Load Model
        print(f"-------------{model_name}-------------")
        model = get_model(model_name)

        #### To CUDA section #### 
        device = "cpu"  # Initially set to CPU
        if torch.cuda.is_available():
            device = "cuda"  # Use CUDA if available
            print("Using CUDA.")
        model.to(device)
        model.eval()
        #### To CUDA section end ####

        #### TensorRT section ####

        # Export to ONNX format
        input_format = torch.randn(1,3,224,224).to(device)
        onnx_output_path = model_name +".onnx"
        torch.onnx.export(
            model, 
            input_format, 
            onnx_output_path, 
            verbose=False, 
            input_names=["input"], 
            output_names=["output"]
        )
        print(f"Model exported to ONNX at {onnx_output_path}")

        # Export to tensorRT format
        trt_model_path = model_name + ".trt"
        convert_onnx_to_tensorrt(onnx_output_path, trt_model_path)

        # Load TensoRT engine
        print("Loading TensorRT engine.")
        trt_logger = trt.Logger(trt.Logger.WARNING)
        trt_engine = load_trt_engine(trt_logger, trt_model_path)

        #### TensorRT section end ####

        # Load Images
        print("Loading test images.")
        images = load_test_images(img_folder)

        # Allocate memory w/ CUDA      
        print("Allocating GPU memory.")
        d_input, d_output, bindings = allocate_buffers(trt_engine)

        # Evaluate Images
        for i, image in enumerate(images):
            anom = image["anomaly"]
            img = image["image"]
            filename = image["file-name"]

            # Move img to GPU and do inference:
            image_in = img.unsqueeze(0).to(device)
            print("Inference for ", filename, ":", i ,"/",len(images)-1, " Groundtruth: anomaly ", anom)
            inference = perform_inference(trt_engine, d_input, d_output, bindings, image_in)

            # Print results
            print(f"Result: Found anomaly with {inference*100}% confidence") # Maybe use softmax?    
