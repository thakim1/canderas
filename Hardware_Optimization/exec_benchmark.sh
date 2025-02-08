#!/bin/bash 

# Define models, prune options, and precision options
models=('DenseNet' 'DenseNet' 'MobileNetV2' 'MobileNetV3' 'MobileNetV3_Large' 'ResNet18')
prune=('' '_P40' '_P70')
precision=('' '_FP16')  # You can also add '_INT8' if required

# Create or clear the benchmark_capture.dat file
> benchmark_capture.dat

# Loop through all combinations of model, prune, and precision
for model in "${models[@]}"; do
    for pr in "${prune[@]}"; do
        for prec in "${precision[@]}"; do
            # Construct the full model name
            full_model="${model}${pr}${prec}.trt"
            
            # Print which combination is being run (for clarity in the log)
            echo "Running inference for model: $full_model"
            
            # Run the inference_speed.py script and append the output to benchmark_capture.dat
            python3 model_inference.py --model "$full_model" >> benchmark_capture.dat
        done
    done
done

echo "Benchmarking complete. Results saved in benchmark_capture.dat."
