# CANDERAS -**C**NN-based **An**omaly **D**etection on **Ra**il Track **S**ystems on the Edge

is maintained by Fabian Seiler, Karel Rusy, Hakim Tayari

Hardware: Jetson Xavier
Docker image: nvcr.io/nvidia/l4t-ml:r32.7.1-py3
Dependencies: 
 - Python: 3.6.9
 - CUDA: 10.2.300
 - cuDNN: 8.2.1.32
 - TensorRT: 8.2.1.8
 - Jetpack 4.6.2
 - Ubuntu: 18.04 Bionic Beaver
 - Platform: 4.9.253-tegra
 - 


## Folder Structure


- Evaluation: Contains scripts to evaluate models on the data in Images
- Training: Contains training script & original dataloader to the harmony server, Subfolder with pretrained Models
- Pruning_Quantization: Holds the scripts/code for pruning and quantization
- Images: Holds Example Images to evaluate/showcase
- Hardware_Optimization: Contains scripts/code to optimize the pretrained models for the Jetson

- run_docker: runs the docker container
- test_l4tml_container.py: Tests the container
- showcase.py ... TODO

- ## Important Resources 

- DustyNV - jetson intefernce tutorial
https://github.com/dusty-nv/jetson-inference/tree/master
- Basic Medium Blog Post
https://vilsonrodrigues.medium.com/a-friendly-introduction-to-tensorrt-building-engines-de8ae0b74038

- General TensorRT Repo by NVIDIA
https://github.com/NVIDIA/TensorRT

- Cuda Toolkit download site
https://developer.nvidia.com/cuda-10.2-download-archive?target_os=Linux&target_arch=ppc64le&target_distro=Ubuntu&target_version=1804&target_type=debnetwork

- TensorRT Quickstart
https://docs.nvidia.com/deeplearning/tensorrt/quick-start-guide/index.html
- TensorRT Install Guide
https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html#installing-tar
- TensorRT containers release notes 
https://docs.nvidia.com/deeplearning/tensorrt/container-release-notes/index.html#rel_20-11
- TensorRT Developer Guide 
https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#object-lifetimes
- TensorRT Python API Documentation
https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/index.html


- Container Catalog 
https://catalog.ngc.nvidia.com/containers?filters=architecture%7CMulti+Arch%7Ccontainers_multiarch&orderBy=weightPopularDESC&query=&page=&pageSize=

- x86 pytorch cuda container
https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch

- Jetson pytorch cuda container
https://catalog.ngc.nvidia.com/orgs/nvidia/containers/l4t-pytorch

- Jetson Zoo
https://elinux.org/Jetson_Zoo#PyTorch_.28Caffe2.29

- Pytorch Container
https://github.com/dusty-nv/jetson-containers/tree/master/packages/l4t/l4t-pytorch
