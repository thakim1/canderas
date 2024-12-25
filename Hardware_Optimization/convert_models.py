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
import numpy as np
import torchvision.transforms as transforms
import cv2


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


def get_img_transform(img_size=224, normalize=([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])):
    """
    Docstring content...

    Args: 
        img_size: ...
        normalize: ...

    Returns: 
        transform: ...

    
    """
    
    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((img_size, img_size)), 
            transforms.Normalize(mean=normalize[0], std=normalize[1]) # Normalize with ImageNet stats
            ])
    return transform


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

def load_test_images(img_folder):
    """
    Docstring content...

    Args: 
        img_folder: ...

    Returns: 
        image_loader: ...
    
    
    """

    image_loader = []

    for image in os.listdir(img_folder):
        if image.endswith(".png") or image.endswith(".jpg"):
            img_np = cv2.imread(os.path.join(img_folder, image))
            # Check if anomaly is in Image
            anom = 1 if image.startswith("composite") else 0
            #anomaly = torch.tensor([True]) if image.startswith("composite") else torch.tensor([False])
            #anom = anomaly.to(torch.float32).to(device).view((-1,1))

            # TODO: Get additional Anomaly Data here

            # Crop and Transform Image
            img_crop = img_np[:img_np.shape[1], : , :]
            T = get_img_transform(img_size=224)
            img_torch = T(img_crop)
            image_loader.append({
                                 "image":img_torch,
                                 "anomaly": anom,
                                 "file-name": image
                                 })
    return image_loader

def load_trt_engine(trt_logger, engine_file):
    """
    Docstring content...

    Args: 
        trt_logger: ...
        engine_file: ...

    Returns: 
        engine: ...
    
    """

    with open(engine_file, 'rb') as f:
        engine_data = f.read()
    runtime = trt.Runtime(trt_logger)
    engine = runtime.deserialize_cuda_engine(engine_data)
    return engine

def allocate_buffers(engine):
    """
    Docstring content...

    Args: 
        engine: ...
    
    Returns: 
        d_input: ....
        d_output: ...
        bindings: ...
    
    
    """

    bindings = []
    input_binding_idx = engine.get_binding_index('input')
    output_binding_idx = engine.get_binding_index('output')

    input_size = trt.volume(engine.get_binding_shape(input_binding_idx)) * engine.max_batch_size
    output_size = trt.volume(engine.get_binding_shape(output_binding_idx)) * engine.max_batch_size
    dtype = trt.nptype(engine.get_binding_dtype(input_binding_idx))

    d_input = cuda.mem_alloc(input_size * np.dtype(dtype).itemsize)
    d_output = cuda.mem_alloc(output_size * np.dtype(dtype).itemsize)
    bindings.append(int(d_input))
    bindings.append(int(d_output))

    return d_input, d_output, bindings


# Inference function
def infer(engine, d_input, d_output, bindings, image_tensor):
    """
    Docstring content...

    Args: 
        engine: ...
        d_input: ....
        d_output: ...
        bindings: ...
        image_tensor: ...

    Returns: 
        output_data: ...    
    
    
    """

    context = engine.create_execution_context()
    # Transfer image to device and do inference
    cuda.memcpy_htod(d_input, image_tensor.cpu().numpy())
    context.execute_v2(bindings=bindings)

    # Copy output back to host
    output_data = np.empty(trt.volume(engine.get_binding_shape(engine.get_binding_index('output'))), dtype=np.float32)
    cuda.memcpy_dtoh(output_data, d_output)
    
    return output_data


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
            inference = infer(trt_engine, d_input, d_output, bindings, image_in)

            # Print results
            print(f"Result: Found anomaly with {inference*100}% confidence") # Maybe use softmax?    
