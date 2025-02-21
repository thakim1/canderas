import os
import argparse
import torch
import torchvision
import torch.nn as nn

def get_model(model_name, weights='IMAGENET1K_V1'):

    if model_name == 'DenseNet':
        model = torchvision.models.densenet121(weights=weights)
        model.classifier = nn.Sequential(
            nn.Linear(in_features=model.classifier.in_features, out_features=1),
            nn.Sigmoid()
        )
    elif model_name == 'MobileNetV2':
        model = torchvision.models.mobilenet_v2(weights=weights)
        model.classifier[1] = nn.Sequential(
            nn.Linear(in_features=1280, out_features=1),
            nn.Sigmoid()
        )
    elif model_name == 'MobileNetV3':
        model = torchvision.models.mobilenet_v3_small(weights=weights)
        model.classifier[3] = nn.Sequential(
            nn.Linear(in_features=1024, out_features=1),
            nn.Sigmoid()
        )
    elif model_name == 'ResNet18':
        model = torchvision.models.resnet18(weights=weights)
        model.fc = nn.Sequential(
            nn.Linear(in_features=model.fc.in_features, out_features=1),
            nn.Sigmoid()
        )
    elif model_name == 'EfficientNet':
        model = torchvision.models.efficientnet_v2_s(weights=weights)
        model.classifier[1] = nn.Sequential(
            nn.Linear(in_features=1280, out_features=1),
            nn.Sigmoid()
        )
    elif model_name == 'MobileNetV3_big':
        model = torchvision.models.mobilenet_v3_large(weights=weights)
        model.classifier[3] = nn.Sequential(
            nn.Linear(in_features=1280, out_features=1),
            nn.Sigmoid()
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    return model

def infer_model_name(file_name):

    if "DenseNet" in file_name:
        return "DenseNet"
    elif "MobileNetV2" in file_name:
        return "MobileNetV2"
    elif "MobileNetV3_big" in file_name:
        return "MobileNetV3_big"
    elif "MobileNetV3" in file_name:
        return "MobileNetV3"
    elif "ResNet18" in file_name:
        return "ResNet18"
    elif "EfficientNet" in file_name:
        return "EfficientNet"
    else:
        return None
    
def main(args):
    src_dir = args.src_dir
    dst_dir = args.dst_dir

    if not os.path.exists(src_dir):
        raise FileNotFoundError(f"Source directory {src_dir} does not exist!")
    os.makedirs(dst_dir, exist_ok=True)

    # Loop over all files in the source directory.
    for file_name in os.listdir(src_dir):
        file_path = os.path.join(src_dir, file_name)
        if not os.path.isfile(file_path):
            continue

        model_arch = infer_model_name(file_name)
        if model_arch is None:
            print(f"Skipping {file_name}")
            continue

        print(f"Processing file '{file_name}' as architecture '{model_arch}'")

        model = get_model(model_arch)
        try:
            state_dict = torch.load(file_path, map_location=torch.device('cpu'))
        except Exception as e:
            print(f"Failed to load {file_name}: {e}")
            continue
        try:
            model.load_state_dict(state_dict)
        except Exception as e:
            print(f"Error loading state dict for {file_name}: {e}")
            continue

        model.half()

        model.eval()

        base_name, ext = os.path.splitext(file_name)
        dst_file_name = f"{base_name}_fp16{ext}"
        dst_file_path = os.path.join(dst_dir, dst_file_name)
        torch.save(model.state_dict(), dst_file_path)
        print(f"Saved FP16 model to {dst_file_path}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert trained models to FP16 quantized versions.")
    parser.add_argument('--src_dir', type=str, required=True,
                        help="Source directory containing trained model state dicts")
    parser.add_argument('--dst_dir', type=str, required=True,
                        help="Destination directory where FP16 quantized models will be saved")
    args = parser.parse_args()

    main(args)

