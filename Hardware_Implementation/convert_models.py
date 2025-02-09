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
import subprocess

def convert_onnx_to_tensorrt(onnx_model_path, trt_model_path):
    """
    Docstring content...

    Args: 
        onnx_model_path: ...
        trt_model_path: ...
    
    Returns: 
        

    
    """
    # Print only warnings, keep output clean 
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    NetworkDefinitionCreationFlags = trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH # ... more options possible?
    network = builder.create_network(1 << int(NetworkDefinitionCreationFlags))

    # Parse ONNX model
    parser = trt.OnnxParser(network, logger)
    with open(onnx_model_path, 'rb') as onnx_model:
        if not parser.parse(onnx_model.read()):
            print("ERROR: Failed to parse ONNX model")
            return
    
    # Default: 
    # Save the TensorRT engine to .trt file
    engine = builder.build_engine(network, builder.create_builder_config())
    with open(trt_model_path + ".trt", 'wb') as trt_model:
        print("Creating default prec (FP32)  model...")
        trt_model.write(engine.serialize())
    print(f"TensorRT model saved to {trt_model_path}.trt")

    # FP16:
    config_fp16 = builder.create_builder_config()
    config_fp16.set_flag(trt.BuilderFlag.FP16)
    engine_fp16 = builder.build_engine(network, config_fp16)
    if engine_fp16:
        print("Creating FP16 prec model...")
        with open(trt_model_path+"_FP16.trt", 'wb') as trt_model:
            trt_model.write(engine_fp16.serialize())
    print(f"TensorRT model saved to {trt_model_path}_FP16.trt")
    
    
    # Enable INT8 calibration if needed
    # INT8 commented because whole dataset is 
    # needed for calibration and not available on jetson

    #config_int8 = builder.create_builder_config()
    #config_int8.set_flag(trt.BuilderFlag.INT8)
    #config_int8.int8_calibrator = my_calibrator # Implement a calibrator if needed
    #engine_int8 = builder.build_engine(network, config_int8)
    #if engine_int8: # Needs Calibrator to work, missing atm
    #    print("Creating INT8 prec model...")
    #    with open(trt_model_path+"INT8.trt", 'wb') as trt_model:
    #        trt_model.write(engine_int8.serialize())
    #    print(f"TensorRT model saved to {trt_model_path}INT8.trt")


def get_model(model_name, prune=None):
    """
    Docstring content...

    Args: 
        model_name: ...

    Returns: 
        model: ...

    """  

    #####################
    #                   #
    #  Unpruned models  #
    #                   #
    #####################

    if prune is None:
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

    #####################
    #                   #
    #   Pruned models   #
    #                   #
    #####################

    else:
        if model_name == 'DenseNet':
            model = models.densenet121(pretrained=False)
            model.classifier = nn.Sequential(
                nn.Linear(in_features=model.classifier.in_features, out_features=1), 
                nn.Sigmoid()
            ) # From Fabian's eval script 'evaluate_model.py'
            model.load_state_dict(torch.load('../Training/Models/DenseNet_KR_Dataset'+prune+'.pth'))

        if model_name == 'MobileNetV2':
            model = models.mobilenet_v2(pretrained=False)
            model.classifier[1] = nn.Sequential(nn.Linear(in_features=1280, out_features=1), nn.Sigmoid())
            
            model.load_state_dict(torch.load('../Training/Models/MobileNetV2_KR_Dataset'+prune+'.pth'))
        
        if model_name == 'MobileNetV3':
            model = models.mobilenet_v3_small(pretrained=False)
            model.classifier[3] = nn.Sequential(nn.Linear(in_features=1024, out_features=1), nn.Sigmoid())
            model.load_state_dict(torch.load('../Training/Models/MobileNetV3_KR_Dataset'+prune+'.pth'))    
        
        if model_name == 'MobileNetV3_Large':
            model = models.mobilenet_v3_large(pretrained=False)
            model.classifier[3] = nn.Sequential(nn.Linear(in_features=1280, out_features=1), nn.Sigmoid())        
            model.load_state_dict(torch.load('../Training/Models/MobileNetV3_big_KR_Dataset'+prune+'.pth'))
        
        if model_name == 'ResNet18':
            model = models.resnet18(pretrained=False)
            model.fc = nn.Sequential(nn.Linear(model.fc.in_features, 1), nn.Sigmoid())
            model.load_state_dict(torch.load('../Training/Models/ResNet18_KR_Dataset'+prune+'.pth'))        

    return model



if __name__ == '__main__':

    img_folder = "../Images"
    trained_models = ['DenseNet', 'MobileNetV2', 'MobileNetV3', 'MobileNetV3_Large', 'ResNet18']
    prune = ['','_P40', '_P70']
    precision = ['', '_FP16'] #, 'INT8']

    for model_name in trained_models:
        # Load Model
        for pru in prune:
            print(f"-------------{model_name + pru}-------------")
            model = get_model(model_name, pru)

            device = "cpu"  # Initially set to CPU
            if torch.cuda.is_available():
                device = "cuda"  # Use CUDA if available
                print("Using CUDA.")
            model.to(device)
            model.eval()

            # Export to ONNX format
            input_format = torch.randn(1,3,224,224).to(device)
            onnx_output_path = model_name+pru+".onnx"
            torch.onnx.export(
                model, 
                input_format, 
                onnx_output_path, 
                verbose=False, 
                input_names=["input"], 
                output_names=["output"]
            )
            print(f"Model exported to ONNX at {onnx_output_path}")

            # Convert to .trt:
            # Creates default (FP32) and FP16 precision TRT models
            convert_onnx_to_tensorrt(onnx_output_path, model_name+pru)

            # Uncomment to save space, .onnx models necesarry for 
            # DLA acceleration script 'make_dla_models.sh' 
            # print(f"Deleting {onnx_output_path} to save space...")
            # try:
            #    subprocess.run("rm -f *.onnx", shell=True, check=True)
            # except subprocess.CalledProcessError as e:
            #    print(f"Error occurred while deleting .onnx files: {e}")



    ####### Test inference on converted models #######
    for model_name in trained_models:
        for pru in prune:
            for prec in precision:
                print(f"-------------{model_name + pru +  prec}-------------")

                # Export to tensorRT format
                trt_model_path = model_name + pru + prec + '.trt'
                print(f"Check if file {trt_model_path} exists...")
                if not os.path.exists(trt_model_path):
                    print(f"Model not found... trying next.")
                    continue

                # Load TensoRT engine
                print(f"Found... loading TensorRT engine {model_name + pru + prec + '.trt'}")
                trt_logger = trt.Logger(trt.Logger.WARNING)
                trt_engine = load_trt_engine(trt_logger, trt_model_path)

                # Load Images
                print("Loading test images...")
                images = load_test_images(img_folder)

                # Allocate memory w/ CUDA      
                print("Allocating GPU memory...")
                d_input, d_output, bindings = allocate_buffers(trt_engine)

                # Evaluate Images
                for i, image in enumerate(images):
                    anom = image["anomaly"]
                    img = image["image"]
                    filename = image["file-name"]

                    # Move img to GPU and do inference:
                    image_in = img.unsqueeze(0).to(device)
                    print("Inference ", filename, ":", i ,"/",len(images)-1, ", has anomaly y/n : 1/0 ", anom)
                    inference = perform_inference(trt_engine, d_input, d_output, bindings, image_in)

                    # Print results
                    print(f"Found anomaly with {inference[0]*100:.3f}%")  

