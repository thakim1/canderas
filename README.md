# CANDERAS -**C**NN-based **An**omaly **D**etection on **Ra**il Track **S**ystems on the Edge

![Thomas_v2](https://github.com/user-attachments/assets/fdea44fa-1c8f-416b-88ed-c23c1b41674c)


is maintained by Fabian Seiler, Karel Rusy, Hakim Tayari

## General Info

Hardware: Jetson Xavier, Platform: 4.9.253-tegra
Docker base image: nvcr.io/nvidia/l4t-ml:r32.7.1-py3
Dependencies: 
 - Python: 3.6.9
 - CUDA: 10.2.300
 - cuDNN: 8.2.1.32
 - TensorRT: 8.2.1.8
 - Jetpack 4.6.2
 - Ubuntu: 18.04 Bionic Beaver

Python3.6 compatible versions of used libs: 
 - einops==?  einops doesn't say to run with Python3.6
 - torch-summary==1.4.5 
 - pillow==8.4.0  
 - timm==0.6.12 
 - pandas==1.1.5 
 - scikit-learn==0.24.2
 - matplotlib==3.3.4 
 - tqdm==4.64.1 
 - opencv-python-headless  opencv-python-headless works with all python versions >= 3.6

# Table of Contents

1. [Folder Structure](#folder-structure)
2. [Important Resources](#important-resources) 
3. [Contributing](#contributions)
4. [License](#license)
5. [Acknowledgments](#acknowledgments)

---

## Folder Structure

- `Evaluation`: Contains scripts to evaluate models on the data in Images
- `Training`: Contains training script & original dataloader to the harmony server, Subfolder with pretrained Models
- `Pruning_Quantization`: Holds the scripts/code for pruning and quantization
- `Images`: Holds Example Images to evaluate/showcase
- `Hardware_Optimization`: Contains scripts/code to optimize the pretrained models for the Jetson

- `run_docker`: Starts container with lengthy command detailed in [docker section](#some-commands-to-get-along-with-docker-easily).
- `test_l4tml_container.py`: Basic test for base image dependencies/libraries
- showcase.py ... TODO

## Important Resources 

### Some commands to get along with docker easily 
- Start docker container
`docker run -it --rm --network host --runtime nvidia -v /home/cdleml/canderas/:/home/canderas <name>`
In case of the unmodified image the name will just be the base image nvcr.io/nvidia/l4t-ml:r32.7.1-py3r

- Show running containers
`docker ps`

- Enter new shell of currently running container
`docker exec -it <name of container> sh`
If the container has not been given a distinct name in docker run command it will show as a some identifier

- Build new image from existing base image using Dockerfile in working directory
`docker build -t <image_name> .`

### Dev Links
- DustyNV - Jetson Intefernce Tutorial: https://github.com/dusty-nv/jetson-inference/tree/master
- Blog Post on TensorRT: https://vilsonrodrigues.medium.com/a-friendly-introduction-to-tensorrt-building-engines-de8ae0b74038
- General TensorRT Repo by NVIDIA: https://github.com/NVIDIA/TensorRT
- Cuda Toolkit Download Site: https://developer.nvidia.com/cuda-10.2-download-archive?target_os=Linux&target_arch=ppc64le&target_distro=Ubuntu&target_version=1804&target_type=debnetwork

- TensorRT Quickstart: https://docs.nvidia.com/deeplearning/tensorrt/quick-start-guide/index.html
- TensorRT Install Guide: https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html#installing-tar
- TensorRT Containers Release Notes: https://docs.nvidia.com/deeplearning/tensorrt/container-release-notes/index.html#rel_20-11
- TensorRT Developer Guide: https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#object-lifetimes
- TensorRT Python API Documentation: https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/index.html

### Docker Container Resources:
- Container Catalog: https://catalog.ngc.nvidia.com/containers?filters=architecture%7CMulti+Arch%7Ccontainers_multiarch&orderBy=weightPopularDESC&query=&page=&pageSize=

- x86 pytorch CUDA Container: https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch
- Jetson pytorch CUDA Container: https://catalog.ngc.nvidia.com/orgs/nvidia/containers/l4t-pytorch
- Jetson Pytorch Container Ref on Github: https://github.com/dusty-nv/jetson-containers/tree/master/packages/l4t/l4t-pytorch
- Jetson Zoo: https://elinux.org/Jetson_Zoo#PyTorch_.28Caffe2.29

## Contributions

Contributions to the project...

## License

Detail the license under which your project is distributed.

## Acknowledgments

List any acknowledgments or credits to others.

