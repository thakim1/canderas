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

if __name__ == '__main__':

    img_folder = "../Images"

    # First network here twice because of bug where first network has memory alloc error, no idea why atm
    clear_bug = True

    trained_models = ['DenseNet', 'DenseNet', 'MobileNetV2', 'MobileNetV3', 'MobileNetV3_Large', 'ResNet18']
    precision = ['', 'INT8', 'FP16']

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

    # Standard TensoRT model, default config
    for model_name in trained_models:
        for p in precision:
            print(f"-------------{model_name + ' ' + p}-------------")

            # Load model as TensoRT engine
            trt_model_path = model_name + p + ".trt"
            print(f"Check if file {trt_model_path} exists...")
            if not os.path.exists(trt_model_path):
                print(f"Model not found... trying next precision.")
                continue

            # Load TensoRT engine
            print(f"Found... loading TensorRT engine {model_name + p + '.trt'}")
            trt_logger = trt.Logger(trt.Logger.WARNING)
            trt_engine = load_trt_engine(trt_logger, trt_model_path)

            # Load Images
            print("Loading test images...")
            images = load_test_images(img_folder)

            # Allocate memory w/ CUDA      
            print("Allocating GPU memory...")
            d_input, d_output, bindings = allocate_buffers(trt_engine)

            inference_time = 0

            # Evaluate Images
            for i, image in enumerate(images):
                anom = image["anomaly"]
                img = image["image"]
                filename = image["file-name"]

                # Move img to GPU and do inference:
                image_in = img.unsqueeze(0).to(device)
                t_start = time.time()
                inference = perform_inference(trt_engine, d_input, d_output, bindings, image_in)
                t_stop = time.time()
                inference_time = inference_time + (t_stop-t_start)
                # print(f"Found anomaly with {inference[0]*100}%")  

            avg_inf_t = inference_time/len(images)
            print(f"Avg. inference speed of {model_name+' '+p}: {1e3*avg_inf_t:.6f}ms -> {1/avg_inf_t} FPS.")
            inference_time = 0
        
        if clear_bug:
            os.system('clear')
            clear_bug = False

    if context:
        context.pop()    
