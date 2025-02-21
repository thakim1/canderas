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


# Table of Contents

1. [Folder Structure](#folder-structure)
2. [Important Resources](#important-resources) 
3. [Contributing](#contributions)
4. [License](#license)

---

## Folder Structure

- `Training`: Contains training script & original dataloader to the harmony server, Subfolder with pretrained Models
- `Pruning_Quantization`: Contains scripts/code to optimize and evaluate the models on the software side
- `Hardware_Implementation`: Contains scripts/code on how to map the networks on the Jetson
- `Images`: Holds Example Images to evaluate/showcase
- `run_docker`: Starts container with lengthy command detailed in [docker section](#some-commands-to-get-along-with-docker-easily).
- `test_l4tml_container.py`: Basic test for base image dependencies/libraries
- `test_model.py`: Run an script that evaluates an inference of the networks on example images

## Important Resources 

### Some commands to get along with docker easily 
- Start docker container
`docker run -it --rm --network host --runtime nvidia -v "$src_dir":"$trg_dir" "$image_name"`
In case of the unmodified image the name will just be the base image nvcr.io/nvidia/l4t-ml:r32.7.1-py3r

- Show running containers
`docker ps`

- Enter new shell of currently running container
`docker exec -it <name of container> sh`
If the container has not been given a distinct name in docker run command it will show as a some identifier

- Build new image from existing base image using Dockerfile in working directory
`docker build -t <image_name> .`

### Dev Links
- HOW TO DLA: [https://github.com/NVIDIA-AI-IOT/jetson_dla_tutorial?tab=readme-ov-file#step-5](https://github.com/NVIDIA-AI-IOT/jetson_dla_tutorial)
- HOW TO DLA SW: https://github.com/NVIDIA/Deep-Learning-Accelerator-SW
- DustyNV - Jetson Intefernce Tutorial: https://github.com/dusty-nv/jetson-inference/tree/master
- TRT Blog Post: https://vilsonrodrigues.medium.com/a-friendly-introduction-to-tensorrt-building-engines-de8ae0b74038
- NVIDIA TRT Repo: https://github.com/NVIDIA/TensorRT
- Cuda Toolkit Download: https://developer.nvidia.com/cuda-10.2-download-archive?target_os=Linux&target_arch=ppc64le&target_distro=Ubuntu&target_version=1804&target_type=debnetwork

- TRT Quickstart: https://docs.nvidia.com/deeplearning/tensorrt/quick-start-guide/index.html
- TRT Install Guide: https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html#installing-tar
- TRT Release Notes: https://docs.nvidia.com/deeplearning/tensorrt/container-release-notes/index.html#rel_20-11
- TRT Dev Guide: https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#object-lifetimes
- TRT Python API Documentation: https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/index.html

### Docker Container Resources:
- Container Catalog: https://catalog.ngc.nvidia.com/containers?filters=architecture%7CMulti+Arch%7Ccontainers_multiarch&orderBy=weightPopularDESC&query=&page=&pageSize=
- x86 pytorch CUDA Container: https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch
- Jetson pytorch CUDA Container: https://catalog.ngc.nvidia.com/orgs/nvidia/containers/l4t-pytorch
- Jetson Pytorch Container Ref on Github: https://github.com/dusty-nv/jetson-containers/tree/master/packages/l4t/l4t-pytorch
- Jetson Zoo: https://elinux.org/Jetson_Zoo#PyTorch_.28Caffe2.29

## Contributors

Hakim Tayari, Fabian Seiler, Karel Rusy

hakim.tayari@tuwien.ac.at, fabian.seiler@tuwien.ac.at, karel.rusy@tuwien.ac.at

## License

The training and optimization part of this project is licensed under the MIT Lincese.
The actual training images are part of an ongoing research project and can therefore not be published.
