"""
Created by Fabian Seiler @ 18.10.2024
"""

import torch
from torch import nn
import torch.utils
from Dataloader import Dataloader
from KRD_dataloader import KRD
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import DataLoader, random_split


def get_transform(img_size=224, normalize=([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])):
    transform = transforms.Compose([
            transforms.Resize((img_size, img_size)), 
            transforms.Normalize(mean=normalize[0], std=normalize[1]) # Normalize with ImageNet stats
            ])
    return transform


if __name__ == '__main__':

    # Parameters:
    lr = 3e-4
    max_epoch = 20
    batch_size = 32
    gamma=0.1
    weights = 'IMAGENET1K_V1'
    
    # Load Dataset and Create Train/Test Split:

    # ------- Either Load generic Data or KRD Dataset

    #dataset = KRD(batch_size=batch_size, root="/DATASET_PATH", augment=False)
    dataset = Dataloader(batch_size=batch_size, root="/DATASET_PATH", augment=False)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    data_loader_train = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    data_loader_test = DataLoader(test_dataset, batch_size=batch_size, shuffle=False) 

    results = []

    model_names = ['DenseNet', 'MobileNetV2', 'MobileNetV3', 'ResNet18', 'EfficientNet', 'MobileNetV3_big']

    for model_name in model_names:
        print(f"-------------{model_name}-------------")
    
        # Load the model and change classification layer
        if model_name == 'DenseNet':
            model = torchvision.models.densenet121(weights=weights)
            model.classifier = nn.Sequential(nn.Linear(in_features=model.classifier.in_features, out_features=1), nn.Sigmoid())

        elif model_name == 'MobileNetV2': 
            model = torchvision.models.mobilenet_v2(weights=weights)
            model.classifier[1] = nn.Sequential(nn.Linear(in_features=1280, out_features=1), nn.Sigmoid())

        elif model_name == 'MobileNetV3':
            model = torchvision.models.mobilenet_v3_small(weights=weights)
            model.classifier[3] = nn.Sequential(nn.Linear(in_features=1024, out_features=1), nn.Sigmoid())

        elif model_name == 'ResNet18':
            model = torchvision.models.resnet18(weights=weights)
            model.fc = nn.Sequential(nn.Linear(model.fc.in_features, 1), nn.Sigmoid())

        elif model_name == 'EfficientNet':
            model = torchvision.models.efficientnet_v2_s(weights=weights)
            model.classifier[1] = nn.Sequential(nn.Linear(1280, 1), nn.Sigmoid())

        elif model_name == 'MobileNetV3_big':
            model = torchvision.models.mobilenet_v3_large(weights=weights)
            model.classifier[3] = nn.Sequential(nn.Linear(in_features=1280, out_features=1), nn.Sigmoid())
        

        # Move Model to GPU and show Layer
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(device)
        model.to(device)
        #print(model)

        # Setup for Loss/Optimizer/Scheduler
        T = get_transform()
        criterion = nn.BCELoss() #nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # Adam optimizer
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=gamma)

        # Start Training:
        print("Training Started!\n")
        model.train()
        for epoch in range(max_epoch):
            
            TP = FP = TN = FN = 0

            for sample in tqdm(data_loader_train):
                # Load Image and Label + Preprocessing
                anomaly, img = sample
                img_crop = T(img.view(img.shape[0], 3, img.shape[2], img.shape[3]))
                target_tensor = anomaly.to(torch.float32).to(device).view((-1,1))

                # Forward Pass + Backprop
                result = model((img_crop).to(device))
                loss = criterion(result, target_tensor)
                #optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                # Evaluate the Confusion Matrix
                predictions = (result >= 0.5).float()  # Binary prediction threshold
                TP += ((predictions == 1) & (target_tensor == 1)).sum().item()
                TN += ((predictions == 0) & (target_tensor == 0)).sum().item()
                FP += ((predictions == 1) & (target_tensor == 0)).sum().item()
                FN += ((predictions == 0) & (target_tensor == 1)).sum().item()

            # Calculate Metrics after each Epoch
            train_accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
            train_precision = TP / (TP + FP) if (TP + FP) > 0 else 0
            train_recall = TP / (TP + FN) if (TP + FN) > 0 else 0
            train_false_positive_rate = FP / (FP + TN) if (FP + TN) > 0 else 0

            #---------------------------------------------------------------------------------
            # Evaluate on Test Dataset
            TP = FP = TN = FN = 0

            with torch.no_grad():
                for sample in tqdm(data_loader_test):
                    #Load Image and Label + Preprocessing
                    anomaly, img = sample
                    img_crop = T(img.view(img.shape[0], 3, img.shape[2], img.shape[3]))
                    target_tensor = anomaly.to(torch.float32).to(device).view((-1,1))

                    # Forward Pass 
                    result = model((img_crop).to(device))
                    loss_t = criterion(result, target_tensor)

                    # Evaluate the Confusion Matrix
                    predictions = (result >= 0.5).float()  # Binary prediction threshold
                    TP += ((predictions == 1) & (target_tensor == 1)).sum().item()
                    TN += ((predictions == 0) & (target_tensor == 0)).sum().item()
                    FP += ((predictions == 1) & (target_tensor == 0)).sum().item()
                    FN += ((predictions == 0) & (target_tensor == 1)).sum().item()

            # Calculate Metrics after each Epoch
            test_accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
            test_precision = TP / (TP + FP) if (TP + FP) > 0 else 0
            test_recall = TP / (TP + FN) if (TP + FN) > 0 else 0
            test_false_positive_rate = FP / (FP + TN) if (FP + TN) > 0 else 0      

            print(f"Epoch {epoch+1}, TRAIN || Accuracy: {train_accuracy:.4f}, Precision: {train_precision:.4f}, Recall: {train_recall:.4f}, False Positive Rate: {train_false_positive_rate:.4f}")
            print(f"Epoch {epoch+1}, TEST  || Accuracy: {test_accuracy:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}, False Positive Rate: {test_false_positive_rate:.4f}")
            
        s_tr = f"TRAIN || A={train_accuracy:.4f} | P={train_precision:.4f} | R={train_recall:.4f} | FPR={train_false_positive_rate:.4f}\n"
        s_te = f"TEST || A={test_accuracy:.4f} | P={test_precision:.4f} | R={test_recall:.4f} | FPR={test_false_positive_rate:.4f}"
        results.append([s_tr + s_te])

        modelpath = "./SoC_Labor/Models/model_{}_epochs_{}_lr_{}_gammaStepLR_{}".format(model_name, max_epoch, lr, gamma)
        torch.save(model.state_dict(), modelpath)

    print([res for res in results])