"""This module contains functions frequently called in the conversion and 
benchmarks scripts such as memory allocation, image loading, loading of TRT 
engine etc.  
"""


import os
import torchvision.transforms as transforms
import pycuda.driver as cuda
import tensorrt as trt
import numpy as np
import cv2


def load_trt_engine(trt_logger, engine_file):
    """
    Loads .trt model/engine from specified file and connect to logger.

    Args: 
        trt_logger: trt.logger, Instance of TRT logger to display inference information (memory, latency, etc.).
        engine_file: String, .trt model to be loaded.

    Returns: 
        engine: Instance of .trt model.
    """
    # From NVIDIA DEV DOCUMENTATION: 
    # When you have a previously serialized optimized model and want to
    # perform inference, you must first create an instance of the Runtime 
    # interface. Like the builder, the runtime requires an instance of the 
    # logger.
    # 
    # Used here: 'In-memory Deserialization' 
    # This method is straightforward and suitable for smaller models or when 
    # memory isn't a constraint. Read the plan file into a memory buffer.
    # 
    # When choosing a deserialization method, consider your specific requirements:
    # - For small models or simple use cases, in-memory deserialization is often 
    #   sufficient.
    # - For large models or when memory efficiency is crucial, consider using 
    #   trt.IStreamReaderV2.
    # - If you need custom file handling or streaming capabilities, trt.IStreamReaderV2 
    #   provides the necessary flexibility

    with open(engine_file, 'rb') as f:
        engine_data = f.read()
    runtime = trt.Runtime(trt_logger)
    engine = runtime.deserialize_cuda_engine(engine_data)
    return engine


def allocate_buffers(engine):
    """
    Allocate memory on GPU w/ CUDA.

    Args: 
        engine: Instantiated .trt engine set to perform inference.
    
    Returns: 
        d_input: Device memory pointers for the input data.
        d_output: Device memory pointers for the output data.
        bindings: Bindings (pointers) to connect the buffers 
                  to the TRT engine during inference.
    """

    #bindings = []
    input_binding_idx = engine.get_binding_index('input')
    output_binding_idx = engine.get_binding_index('output')

    input_size = trt.volume(engine.get_binding_shape(input_binding_idx)) * engine.max_batch_size
    output_size = trt.volume(engine.get_binding_shape(output_binding_idx)) * engine.max_batch_size
    dtype = trt.nptype(engine.get_binding_dtype(input_binding_idx))

    d_input = cuda.mem_alloc(input_size * np.dtype(dtype).itemsize)
    d_output = cuda.mem_alloc(output_size * np.dtype(dtype).itemsize)
    #bindings.append(int(d_input))
    #bindings.append(int(d_output))
    bindings = [int(d_input), int(d_output)]

    return d_input, d_output, bindings 

def perform_inference(engine, d_input, d_output, bindings, image_tensor):
    """
    Docstring content...

    Args: 
        engine: Instantiated .trt engine set to perform inference.
        d_input: Device memory pointers for the input data.
        d_output: Device memory pointers for the output data.
        bindings: Bindings (pointers) to connect the buffers 
                  to the TRT engine during inference.
        image_tensor: Tensor of inference candidate image.

    Returns: 
        output_data: inference result.    
    
    
    """
    # From NVIDIA DEV DOCUMENTATION: 
    # Engine holds optimized model but inference requires additional state for 
    # intermediate activations. This is done via the execution_context interface. 
    # An engine can have multiple execution contexts, allowing one set 
    # of weights to be used for multiple overlapping inference tasks.
    context = engine.create_execution_context()  

    # Transfer image to device and do inference
    cuda.memcpy_htod(d_input, image_tensor.cpu().numpy())
    # Other option: context.set_tensor_address(name, ptr)
    # This approach allows more direct control over how and where the memory is 
    # allocated and where TensorRT looks for the input and output tensors.

    # v2: Memory is in use until execute_v2() returns 
    context.execute_v2(bindings=bindings)
    # Other option: context.execute_async_v3(stream_handle: int) -> bool  
    # stream_handle – The cuda stream on which the inference kernels will 
    # be enqueued. Using default stream may lead to performance issues due 
    # to additional cudaDeviceSynchronize() calls by TensorRT to ensure correct 
    # synchronizations. Please use non-default stream instead.
    # v3: Memory is in use until network is done executing 
    # -> likely better for smaller/faster models -> more throughput

    # Copy output back to host
    output_data = np.empty(trt.volume(engine.get_binding_shape(engine.get_binding_index('output'))), dtype=np.float32)
    cuda.memcpy_dtoh(output_data, d_output)
    
    return output_data


def get_img_transform(img_size=224, normalize=([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])):
    """
    Crop and normalize image with ImageNet normalization parameters. 

    Args: 
        img_size: int, Size of image.
        normalize: list(int), ImageNet normalization parameters.

    Returns: 
        transform: Image tensor.
    """
    
    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((img_size, img_size)), 
            transforms.Normalize(mean=normalize[0], std=normalize[1]) # Normalize with ImageNet stats
            ])
    return transform

def load_test_images(img_folder):
    """
    Loads images to perform inference and measure latency + memory usage. 

    Args: 
        img_folder: String, Path to image source folder.

    Returns: 
        image_loader: list(image tensor), List of dictionary with: loaded images, anomaly (y/n), file name of the image.
    """

    image_loader = []

    for image in os.listdir(img_folder):
        if image.endswith(".png") or image.endswith(".jpg"):
            img_np = cv2.imread(os.path.join(img_folder, image))
            
            # Check if anomaly is in Image
            anom = 1 if image.startswith("composite") else 0
            #anomaly = torch.tensor([True]) if image.startswith("composite") else torch.tensor([False])
            #anom = anomaly.to(torch.float32).to(device).view((-1,1))

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
