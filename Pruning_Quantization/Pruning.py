import torch
from torch import nn
import torch.nn.utils.prune as prune
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import copy
import os

from KRD_dataloader import KRD

def get_transform(img_size=224, normalize=([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])):
    """Defines the transformation pipeline."""
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)), 
        transforms.Normalize(mean=normalize[0], std=normalize[1])  # ImageNet stats
    ])
    return transform

def evaluate_model(model, data_loader, device):
    """Evaluates the model on the given data loader and returns metrics."""
    model.eval()
    TP = FP = TN = FN = 0
    T = get_transform()

    with torch.no_grad():
        for sample in tqdm(data_loader, desc="Evaluating"):
            anomaly, img = sample
            img_crop = T(img.view(img.shape[0], 3, img.shape[2], img.shape[3]))
            target_tensor = anomaly.to(torch.float32).to(device).view((-1,1))


            result = model((img_crop).to(device))


            predictions = (result >= 0.5).float()  # Binary threshold
            TP += ((predictions == 1) & (target_tensor == 1)).sum().item()
            TN += ((predictions == 0) & (target_tensor == 0)).sum().item()
            FP += ((predictions == 1) & (target_tensor == 0)).sum().item()
            FN += ((predictions == 0) & (target_tensor == 1)).sum().item()

    # Calculate metrics
    total = TP + TN + FP + FN
    accuracy = (TP + TN) / total if total > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    false_positive_rate = FP / (FP + TN) if (FP + TN) > 0 else 0

    return accuracy, precision, recall, false_positive_rate

def print_pruning_stats(model):
    """Prints the pruning statistics for each pruned layer."""
    for name, module in model.named_modules():
        if hasattr(module, 'weight') and module.weight is not None:
            w = module.weight.detach()
            if len(w.shape) == 4:
                # Conv layer
                filter_magnitudes = w.view(w.size(0), -1).abs().sum(dim=1)
                pruned_filters = torch.sum(filter_magnitudes == 0).item()
                total_filters = w.size(0)
                print(f"Conv Layer: {name}, Pruned filters: {pruned_filters}/{total_filters} ({(pruned_filters/total_filters)*100:.2f}%)")
            elif len(w.shape) == 2:
                # Linear layer
                unit_magnitudes = w.abs().sum(dim=1)
                pruned_units = torch.sum(unit_magnitudes == 0).item()
                total_units = w.size(0)
                print(f"Linear Layer: {name}, Pruned units: {pruned_units}/{total_units} ({(pruned_units/total_units)*100:.2f}%)")

def remove_pruning(model):
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            try:
                prune.remove(module, 'weight')
            except ValueError:
                pass
    return model

def create_model(model_name, weights='IMAGENET1K_V1'):

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
            nn.Linear(model.fc.in_features, 1),
            nn.Sigmoid()
        )
    elif model_name == 'MobileNetV3_big':
        model = torchvision.models.mobilenet_v3_large(weights=weights)
        model.classifier[3] = nn.Sequential(
            nn.Linear(in_features=1280, out_features=1),
            nn.Sigmoid()
        )
    else:
        raise ValueError(f"Model name {model_name} is not supported.")
    return model

def load_weights(model, weight_path, device):

    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"Weight path {weight_path} does not exist.")
    
    state_dict = torch.load(weight_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    return model

def apply_pruning(model, pruning_method, amount=0.3):

    for name, module in model.named_modules():
        # Only prune Conv2d or Linear
        if not (isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear)):
            continue

        # 1) Unstructured magnitude-based pruning
        if pruning_method == "L1-Unstructured":
            prune.l1_unstructured(module, name='weight', amount=amount)
        elif pruning_method == "Random-Unstructured":
            prune.random_unstructured(module, name='weight', amount=amount)
        elif pruning_method == "L2-Structured":
            prune.ln_structured(module, name='weight', amount=amount, n=2, dim=0)
        elif pruning_method == "Filter-Pruning":
            if isinstance(module, nn.Conv2d):
                prune.ln_structured(module, name='weight', amount=amount, n=1, dim=0)
        elif pruning_method == "Channel-Pruning":
            if isinstance(module, nn.Conv2d):
                if module.kernel_size == (1, 1):
                    prune.ln_structured(module, name='weight', amount=amount, n=1, dim=1)
        else:
            raise ValueError(f"Pruning method '{pruning_method}' is not recognized.")

    return model

if __name__ == "__main__":
    # Setup
    batch_size = 32
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load Dataset
    dataset = KRD(batch_size=batch_size)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    data_loader_test = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Define models and recommended pruning methods
    model_names = [
        "DenseNet",
        "MobileNetV2",
        "MobileNetV3",
        "ResNet18",
        "MobileNetV3_big"
    ]

    # For each model, we have a list of recommended pruning methods
    pruning_methods = {
        "DenseNet":         ["L1-Unstructured", "Random-Unstructured", "L2-Structured", "Filter-Pruning", "Channel-Pruning"],
        "MobileNetV2":      ["L1-Unstructured", "Random-Unstructured", "L2-Structured", "Filter-Pruning", "Channel-Pruning"],
        "MobileNetV3":      ["L1-Unstructured", "Random-Unstructured", "L2-Structured", "Filter-Pruning", "Channel-Pruning"],
        "ResNet18":         ["L1-Unstructured", "Random-Unstructured", "L2-Structured", "Filter-Pruning", "Channel-Pruning"],
        "MobileNetV3_big":  ["L1-Unstructured", "Random-Unstructured", "L2-Structured", "Filter-Pruning", "Channel-Pruning"]
    }

    # Directory to trained (unpruned) model weights
    trained_models_dir = "/home/krusy/Pruning/trained_networks"  
    # Directory to save pruned models
    pruned_models_dir = "/home/krusy/Pruning/pruned_models"
    os.makedirs(pruned_models_dir, exist_ok=True)

    # Dictionary to store results
    results = {}

    # Iterate over each model
    prune_amounts = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    for prune_amount in prune_amounts:
        for model_name in model_names:
            print(f"\n Evaluating {model_name} with various pruning methods with amount: {prune_amount}")

            # Construct the path to the trained model’s weights
            model_path = os.path.join(trained_models_dir, f"{model_name}_KR_Dataset.pth")
            if not os.path.isfile(model_path):
                print(f"Trained weights not found at {model_path}. Skipping.")
                continue

            # Get the list of pruning methods for the model
            pruning_methods = pruning_methods.get(model_name, [])
            
            for method in pruning_methods:
                print(f"\n--- Pruning Method: {method} ---")

                # Create and load the original model
                model = create_model(model_name)
                model = load_weights(model, model_path, device)

                # Apply pruning
                pruned_model = apply_pruning(model, pruning_method=method, amount=prune_amount)

                # Permanently remove pruning re-param
                pruned_model = remove_pruning(pruned_model)

                # Evaluate the pruned model
                accuracy, precision, recall, fpr = evaluate_model(pruned_model, data_loader_test, device)
                results_key = f"{model_name}_{method}"
                results[results_key] = (accuracy, precision, recall, fpr)

                print(f"Results => Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, "
                    f"Recall: {recall:.4f}, FPR: {fpr:.4f}")

                # Save the pruned model
                save_path = os.path.join(pruned_models_dir, f"pruned_{prune_amount}_{method}_{model_name}_KR_Dataset.pth")
                torch.save(pruned_model.state_dict(), save_path)
                print(f"Pruned model saved to: {save_path}")

        # Print Summary of All Results
        print("\n============= Pruning Summary =============")
        for key, (acc, prec, rec, fpr) in results.items():
            print(f"{key} => Accuracy: {acc:.4f}, Precision: {prec:.4f}, "
                f"Recall: {rec:.4f}, FPR: {fpr:.4f}")

