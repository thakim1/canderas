"""
AI GENERATED!!!"""
import re
import matplotlib.pyplot as plt

# Function to parse the log file and extract memory usage, inference time, and FPS for each model
def parse_log_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    model_data = {}
    current_model = None
    mem_usage_pattern = re.compile(r'IExecutionContext creation:.*?GPU (\d+)\s\(MiB\)')  # Regex to capture GPU memory
    inference_pattern = re.compile(r'(\S+)\s*:\s*([\d\.]+)\s*ms\s*=>\s*([\d\.]+)\s*FPS')  # Regex to capture floating-point inference time and FPS

    for line in lines:
        # Check if the line is a model name
        if line.startswith('######'):
            current_model = line.strip('######').strip()  # Extract model name and clean it
            current_model = current_model.strip('#')
            model_data[current_model] = {'memory_usage': [], 'inference_time': [], 'fps': []}
        
        # If the line contains memory usage info, extract the GPU memory value
        elif 'GPU' in line:
            match = mem_usage_pattern.search(line)
            if match and current_model:
                gpu_memory = int(match.group(1))  # Extract GPU memory in MiB
                model_data[current_model]['memory_usage'].append(gpu_memory)
        
        # If the line contains inference time and FPS, extract these values
        elif 'ms' in line and 'FPS' in line:
            match = inference_pattern.search(line)
            if match and current_model:
                inference_time = float(match.group(2))  # Extract inference time in ms
                fps = float(match.group(3))  # Extract FPS
                model_data[current_model]['inference_time'].append(inference_time)
                model_data[current_model]['fps'].append(fps)

    # Calculate the average memory usage, inference time, and FPS for each model
    average_data = {
        model: {
            'avg_memory_usage': sum(data['memory_usage']) / len(data['memory_usage']) if data['memory_usage'] else 0,
            'avg_inference_time': sum(data['inference_time']) / len(data['inference_time']) if data['inference_time'] else 0,
            'avg_fps': sum(data['fps']) / len(data['fps']) if data['fps'] else 0
        }
        for model, data in model_data.items()
    }

    return average_data

# Function to create bar charts for average memory usage, inference time, and FPS
def plot_data(average_data):
    # Sorting models by average memory usage
    sorted_models = sorted(average_data.items(), key=lambda x: x[1]['avg_memory_usage'], reverse=True)

    models = [model for model, _ in sorted_models]
    memory_usage = [data['avg_memory_usage'] for _, data in sorted_models]
    inference_time = [data['avg_inference_time'] for _, data in sorted_models]
    fps = [data['avg_fps'] for _, data in sorted_models]

    # Create the figure and subplots
    fig, axs = plt.subplots(3, 1, figsize=(10, 18))

    # Memory usage plot
    bars_mem = axs[0].bar(models, memory_usage, color='skyblue')
    axs[0].set_xlabel('Model')
    axs[0].set_ylabel('Average GPU Memory Usage (MiB)')
    axs[0].set_title('Average GPU Memory Usage per Model')
    axs[0].tick_params(axis='x', rotation=90)
    
    # Add annotations to memory usage plot
    for bar in bars_mem:
        yval = bar.get_height()
        axs[0].text(bar.get_x() + bar.get_width() / 2, yval + 10, round(yval, 2), ha='center', va='bottom', fontsize=10)

    # Inference time plot
    bars_infer = axs[1].bar(models, inference_time, color='lightgreen')
    axs[1].set_xlabel('Model')
    axs[1].set_ylabel('Average Inference Time (ms)')
    axs[1].set_title('Average Inference Time per Model')
    axs[1].tick_params(axis='x', rotation=90)

    # Add annotations to inference time plot
    for bar in bars_infer:
        yval = bar.get_height()
        axs[1].text(bar.get_x() + bar.get_width() / 2, yval + 10, round(yval, 2), ha='center', va='bottom', fontsize=10)

    # FPS plot
    bars_fps = axs[2].bar(models, fps, color='salmon')
    axs[2].set_xlabel('Model')
    axs[2].set_ylabel('Average FPS')
    axs[2].set_title('Average FPS per Model')
    axs[2].tick_params(axis='x', rotation=90)

    # Add annotations to FPS plot
    for bar in bars_fps:
        yval = bar.get_height()
        axs[2].text(bar.get_x() + bar.get_width() / 2, yval + 0.1, round(yval, 2), ha='center', va='bottom', fontsize=10)

    # Display the plots
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Define the path to your log file
    file_path = 'bench_dump.dat'  # Change this to the path of your log file

    # Parse the log file and get the average data for each model
    average_data = parse_log_file(file_path)

    # Create the plots
    plot_data(average_data)
