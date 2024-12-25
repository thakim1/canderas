"""
Helper functions...
"""


import os
import torchvision.transforms as transforms
import pycuda.driver as cuda
import tensorrt as trt
import numpy as np
import cv2


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

def perform_inference(engine, d_input, d_output, bindings, image_tensor):
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