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