import torch
from torch import nn
import torch.utils
import torchvision.transforms as transforms
import torchvision
import cv2
import os



def get_transform(img_size=224, normalize=([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])):
    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((img_size, img_size)), 
            transforms.Normalize(mean=normalize[0], std=normalize[1]) # Normalize with ImageNet stats
            ])
    return transform

def get_model(model_name):
     # Load the model and change classification layer
    if model_name == 'DenseNet':
        model = torchvision.models.densenet121(weights='DEFAULT')
        model.classifier = nn.Sequential(nn.Linear(in_features=model.classifier.in_features, out_features=1), nn.Sigmoid())

    elif model_name == 'MobileNetV2': 
        model = torchvision.models.mobilenet_v2(weights='DEFAULT')
        model.classifier[1] = nn.Sequential(nn.Linear(in_features=1280, out_features=1), nn.Sigmoid())

    elif model_name == 'MobileNetV3':
        model = torchvision.models.mobilenet_v3_small(weights='DEFAULT')
        model.classifier[3] = nn.Sequential(nn.Linear(in_features=1024, out_features=1), nn.Sigmoid())

    elif model_name == 'ResNet18':
        model = torchvision.models.resnet18(weights='DEFAULT')
        model.fc = nn.Sequential(nn.Linear(model.fc.in_features, 1), nn.Sigmoid())

    elif model_name == 'EfficientNet':
        model = torchvision.models.efficientnet_v2_s(weights='DEFAULT')
        model.classifier[1] = nn.Sequential(nn.Linear(1280, 1), nn.Sigmoid())

    elif model_name == 'MobileNetV3_big':
        model = torchvision.models.mobilenet_v3_large(weights='DEFAULT')
        model.classifier[3] = nn.Sequential(nn.Linear(in_features=1280, out_features=1), nn.Sigmoid())
    else:
        return Exception("Model Not Found")
    
    return model

def load_test_images(img_folder):
    image_loader = []

    for image in os.listdir(img_folder):
        if image.endswith(".png") or image.endswith(".jpg"):
            img_np = cv2.imread(os.path.join(img_folder, image))
            # Check if anomaly is in Image
            anom = 1 if image.startswith("composite") else 0
            #anomaly = torch.tensor([True]) if image.startswith("composite") else torch.tensor([False])
            #anom = anomaly.to(torch.float32).to(device).view((-1,1))

            # TODO: Get additional Anomaly Data here

            # Crop and Transform Image
            img_crop = img_np[:img_np.shape[1], : , :]
            T = get_transform(img_size=224)
            img_torch = T(img_crop)
            image_loader.append({
                                 "image":img_torch,
                                 "anomaly": anom,
                                 })
    return image_loader

if __name__ == '__main__':

    img_folder = "./Images"
    weights = "./Models/MobileNetV2_weights"
    model_name = ['DenseNet', 'MobileNetV2', 'MobileNetV3', 'ResNet18', 'EfficientNet', 'MobileNetV3_big'][1]

    # Load Model
    print(f"-------------{model_name}-------------")
    model = get_model(model_name)
    model.load_state_dict(torch.load(weights, weights_only=True))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    # Load Images
    images = load_test_images(img_folder)
    
    # Evaluate Images
    for image in images:
        anom = image["anomaly"]
        img = image["image"]

        result = model(img.unsqueeze(0).to(device))
        print(f"Estimate={result.detach().cpu().numpy()[0][0]:.2f} | Exact={anom}")

    