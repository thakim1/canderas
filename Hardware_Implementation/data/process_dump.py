"""

AI GENERATED !!! 

"""
import re
import matplotlib.pyplot as plt

# Function to parse the log file and extract memory usage for each model
def parse_log_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    model_memory_usage = {}
    current_model = None
    mem_usage_pattern = re.compile(r'IExecutionContext creation:.*?GPU (\d+)\s\(MiB\)')  # Regex to capture GPU memory after "IExecutionContext creation"

    for line in lines:
        # Check if the line is a model name
        if line.startswith('######'):
            current_model = line.strip('######').strip()  # Extract model name and clean it
            current_model = current_model.strip('#')
            model_memory_usage[current_model] = []
        
        # If the line contains memory usage info, extract the GPU memory value
        elif 'GPU' in line:
            match = mem_usage_pattern.search(line)
            if match and current_model:
                gpu_memory = int(match.group(1))  # Extract GPU memory in MiB
                model_memory_usage[current_model].append(gpu_memory)

    # Calculate the average GPU memory usage for each model
    average_memory_usage = {model: sum(usage) / len(usage) for model, usage in model_memory_usage.items()}

    return average_memory_usage

# Function to create a bar chart of the average memory usage
def plot_average_memory_usage(average_memory_usage):
    # Sort models by average memory usage (ascending)
    sorted_models = sorted(average_memory_usage.items(), key=lambda x: x[1], reverse=True)

    models = [model for model, _ in sorted_models]
    memory_usage = [usage for _, usage in sorted_models]

    # Create the bar plot
    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, memory_usage, color='skyblue')

    # Add labels and title
    plt.xlabel('Model')
    plt.ylabel('Average GPU Memory Usage (MiB)')
    plt.title('Average GPU Memory Usage per Model')

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=90)

    # Annotate the bars with the actual numerical values at the top
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height + 0.2, f'{height:.2f}', ha='center', va='bottom', fontsize=7)

    # Display the plot
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Define the path to your log file
    file_path = 'bench_dump.dat'  # Change this to the path of your log file

    # Parse the log file and get the average memory usage for each model
    average_memory_usage = parse_log_file(file_path)

    # Create the bar chart
    plot_average_memory_usage(average_memory_usage)
