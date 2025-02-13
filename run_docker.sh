#!/bin/bash
# Define your source and target directories
src_dir=$(dirname "$(realpath "$0")")
trg_dir="/home/canderas"  
image_name="nvcr.io/nvidia/l4t-ml:r32.7.1-py3"

# Run the Docker container with the appropriate volume mount
docker run -it --rm --network host --runtime nvidia -v "$src_dir":"$trg_dir" "$image_name"