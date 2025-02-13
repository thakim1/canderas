# CANDERAS -**C**NN-based **An**omaly **D**etection on **Ra**il Track **S**ystems on the Edge - Hardware Implementation


### How-To: Reproduce inference time and memory allocation results

Assuming you have a working setup with the correct docker image and the listed dependencies the following steps will show you how to generate the same data we used in our design exploration.

#### 0) Start container
Clone this repository and start the docker container with the `run_docker.sh` 'script'.

#### 1) Navigate to the `Hardware_Implementation` directory

After you started the docker container navigate to the directory that has been mirrored to the container i.e., this repository. Your current working directory should now be just `/home/canderas` as specified in the `run_docker.sh` file, which just contains this somewhat lengthy command:

+ `docker run -it --rm --network host --runtime nvidia -v "$src_dir":"$trg_dir" "$image_name"` 
  
In case of the unmodified image the `$image_name` will just be the base image `nvcr.io/nvidia/l4t-ml:r32.7.1-py3r`. If you want to modify the image with a Dockerfile you need to change `$image_name` to the new image you created from the base NVIDIA image. Now navigate to `Hardware_Implementation` with cd.

#### 2) Building the models

Before running the benchmark we need to build the `.trt` models from the PyTorch state dictionary. To do just that, simply run the conversion script `python3 convert_models.py`. This script will take all the different, pruned and unpruned models from `Training/Models`, first convert them to the `.onnx` format, and then from the `.onnx` format create the TensorRT engines for both `FP16` and default precision (`FP32`). The models will be stored in this directory `/<some_previous_dir>/canderas/Hardware_Implementation`. Building all the different models may very well take 30+ minutes. 


#### 3) Running the benchmark

Having created all the needed models we are ready to run the benchmark script. This script will run a series of 10 test-images with 5 iterations and evaluate the average inference time needed, and the `IExecution context` memory allocation for each TensorRT engine i.e., each model. 

If you want to save the logger output, which contains memory and allocation information, to a file you can use the simple command below, otherwise only inference time will be captured and saved to `bench_dump.dat`. 

+ `python3 inference_benchmark.py >> data/bench_dump.dat`

This somewhat inconvenient handling is due to TensoRT version 4.2.1 not allowing us to redirect the TRT logger's output stream.


#### 4) Results 

The results should now be available in `data/bench_dump.dat`. To get a bar graph of memory alloaction and average inference time per model run `process_dump.py`.