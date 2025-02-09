"""This script...

"""

import os
import tensorrt as trt
import pycuda.driver as cuda
import numpy as np
import torch
from helper_functions import load_trt_engine, load_test_images
from helper_functions import allocate_buffers, perform_inference
import time
import argparse


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Perform inference on 10 test images on specified model")

    # Add arguments
    parser.add_argument('--model', type=str, required=True, help="The model name to benchmark (e.g., DenseNet_P40_FP16.trt, ResNet18_P70.trt).")

    # Parse the arguments
    args = parser.parse_args()
    print(f"Running benchmark for model: {args.model}")

    img_folder = "../Images" # test images

    # Initialize the CUDA context manually
    cuda.init()
    
    device = torch.device("cpu") 
    if torch.cuda.is_available():
        device = torch.device("cuda")  
        print("Using CUDA.")

    if torch.cuda.is_available():
        device_id = 0  # Change the device ID based on system if necessary
        gpu_device = cuda.Device(device_id)
        context = gpu_device.retain_primary_context()
    else:
        context = None  # No context for CPU 

    # Standard TensoRT model, default config
    
    print(f"-------------{args.model}-------------")

    # Load model as TensoRT engine
    trt_model_path = args.model
    print(f"Check if file {trt_model_path} exists...")
    if not os.path.exists(trt_model_path):
        print(f"Model not found... trying next precision.")
        exit(0)

    # Load TensoRT engine
    print(f"Found... loading TensorRT engine {trt_model_path}")
    trt_logger = trt.Logger(trt.Logger.WARNING)
    trt_engine = load_trt_engine(trt_logger, trt_model_path)

    # Load Images
    print("Loading test images...")
    images = load_test_images(img_folder)
    
    # Allocate memory w/ CUDA      
    print("Allocating GPU memory...")
    d_input, d_output, bindings = allocate_buffers(trt_engine)

    total_inf_time = 0
    iterations = 5
    for _ in range(iterations):
        for i, image in enumerate(images):
            anom = image["anomaly"]
            img = image["image"]
            filename = image["file-name"]
            print(f"img-{i}: {filename}")

            # Move img to GPU and do inference:
            image_in = img.unsqueeze(0).to(device)
            t_start = time.time()
            inference = perform_inference(trt_engine, d_input, d_output, bindings, image_in)
            t_stop = time.time()
            total_inf_time = total_inf_time + (t_stop-t_start)

    avg_inf_t = total_inf_time/(iterations*len(images))
    print(f"Avg. inf. {trt_model_path}: {1e3*avg_inf_t:.6f}ms => {1/avg_inf_t} FPS.")
            
    if context:
        context.pop()    
