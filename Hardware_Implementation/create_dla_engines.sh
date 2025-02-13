#!/bin/bash 
# Define models, pruning options, and precision options
models=('DenseNet' 'MobileNetV2' 'MobileNetV3' 'MobileNetV3_Large' 'ResNet18')
pruning=('' '_P40' '_P70')
# This short script creates .trt models that can leverage the DLA (Deep Learning Acceleration) 
# cores. Requires .onnx format of all listed models. Execute convert_models.py first to get
# .onnx and .trt models without DLA acceleration. 
# There are two DLA cores available 0, 1 
# It does not matter which core is being used, be aware that two models using the same
# DLA core cannot run simultaneously. 
# trtexec might not be available from every working directory  
# => adjust path for trtexec executable accordingly 
# Loop through all combinations of model, pruning, and precision


for model in "${models[@]}"; do
    # FP32 Quantization
    for prune in "${pruning[@]}"; do

            model_name="${model}${prune}"
            full_model_name="${model}${prune}_dla0"
            /usr/src/tensorrt/bin/trtexec --onnx="$model_name".onnx --useDLACore=0 --allowGPUFallback --saveEngine="$full_model_name".trt --verbose=0
            
            full_model_name_qu="${model}${prune}_FP16_dla0"
            /usr/src/tensorrt/bin/trtexec --onnx="$model_name".onnx --fp16 --useDLACore=0 --allowGPUFallback --saveEngine="$full_model_name_qu".trt --verbose=0
            rm -f "${model_name}.onnx" # save space
    done

done
echo "Generating DLA-core accelerated TRT-models completed."
echo "Exiting..."
