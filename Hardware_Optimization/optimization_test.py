"""This script...

"""


import os
import tensorrt as trt
import pycuda.driver as cuda
import numpy as np
import torch
from helper_functions import load_trt_engine, load_test_images
from helper_functions import allocate_buffers, perform_inference


if __name__ == '__main__':

    img_folder = "../Images"
    trained_models = ['DenseNet', 'MobileNetV2', 'MobileNetV3', 'MobileNetV3_Large', 'ResNet18']

    # Initialize the CUDA context manually
    cuda.init()
    
    device = torch.device("cpu") 
    if torch.cuda.is_available():
        device = torch.device("cuda")  
        print("Using CUDA.")

    if torch.cuda.is_available():
        device_id = 0  # Change the device ID based on system if necessary
        gpu_device = cuda.Device(device_id)
        context = gpu_device.make_context()
    else:
        context = None  # No context for CPU 

    for model_name in trained_models:
        print(f"-------------{model_name}-------------")
        

        # Load model as TensoRT engine
        trt_model_path = model_name + ".trt"
        print("Loading TensorRT engine.")
        trt_logger = trt.Logger(trt.Logger.WARNING)
        trt_engine = load_trt_engine(trt_logger, trt_model_path)

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

    if context:
        context.pop()