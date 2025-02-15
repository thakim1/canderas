"""This script performs inference latency and memory usage of the TRT IExecution Context 
for all model/pruning/quantization/acceleration combinations and saves it to specified log_file.
Consult README.md for more information. 
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


    # First network here twice because of bug where first network has memory alloc error, no idea why atm
    trained_models = ['DenseNet', 'DenseNet', 'MobileNetV2', 'MobileNetV3', 'MobileNetV3_Large', 'ResNet18']
    clear_bug = True

    # Other stuff 
    img_folder = "../Images"
    log_file = 'data/bench_dump.dat'
    prune = ['','_P40', '_P70']
    precision = ['', '_FP16']
    acceleration = ['', '_dla0']

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

    # Create logger instance for memory information 
    trt_logger = trt.Logger(trt.Logger.VERBOSE)

    # Standard TensoRT model, default config
    iter = 0
    for model_name in trained_models:
        for pru in prune:
            for prec in precision:
                for acc in acceleration:    
                    # print(f"Benchmarking ---{model_name + pru  + prec + acc}--- fps | {iter} of {len(trained_models)*len(prune)*len(precision)}", end='\r')

                    # Load model as TensoRT engine
                    trt_model_path = model_name + pru + prec + acc + '.trt'
                    if not os.path.exists(trt_model_path):
                        print(f"Model not found... trying next precision.")
                        continue


                    with open(log_file, 'a') as lf:
                        lf.write(f"######{model_name + pru  + prec + acc}######\n")

                    trt_engine = load_trt_engine(trt_logger, trt_model_path)
                    images = load_test_images(img_folder)
                    d_input, d_output, bindings = allocate_buffers(trt_engine)

                    total_inf_time = 0
                    iterations = 5
                    for _ in range(iterations):
                        for i, image in enumerate(images):
                            anom = image["anomaly"]
                            img = image["image"]
                            filename = image["file-name"]
                            # print(f"img-{i}: {filename}")

                            image_in = img.unsqueeze(0).to(device)
                            t_start = time.time()
                            inference = perform_inference(trt_engine, d_input, d_output, bindings, image_in)
                            t_stop = time.time()
                            total_inf_time = total_inf_time + (t_stop-t_start)

                    avg_inf_t = total_inf_time/(iterations*len(images))
                    with open(log_file, 'a') as lf:
                        lf.write(f"{model_name+pru+prec+acc} : {1e3*avg_inf_t:.6f}ms => {1/avg_inf_t}FPS \n")
                    iter+=1

        if clear_bug:
            os.system('clear')
            clear_bug = False

    if context:
        context.pop()    

