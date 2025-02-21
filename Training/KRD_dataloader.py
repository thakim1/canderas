import os
from torchvision import transforms
import torchvision.transforms.functional as F
import torch
import cv2
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt


class KRD():

    def __init__(self, batch_size, root="PATH_TO_DATA", augment: bool = False):

        self.root = root
        self.batch_size = batch_size

        self.T = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
        ])

        self.data = self.load_data(augment)

    def __getitem__(self, index: int):
        return self.data[index]
    
    def __len__(self):
        return len(self.data)


    def crop_img(self, img, h_line, v_delta):
        #img = cv2.cvtColor(img, cv2.COLOR_BRG2RGB)
        if len(img.shape) == 3:
            h, w, c = img.shape
        else:
            h, w = img.shape
        left = (w - 2*v_delta) // 2
        top = h - h_line
        img = self.T(img)
        return F.crop(img, top, left, h_line, 2*v_delta)

    def mirror_imgs(self, img):
        return torch.flip(img, [2])
    
    def change_lighting(self, img):

        transform = transforms.ColorJitter(
            brightness=0.2,  # Change brightness by up to 20%
            saturation=0.2  # Change saturation by up to 20%
        )
        # Apply the transform only to the image
        img_aug = transform(img)
        return img_aug
 
    def load_data(self, augmentation: bool = False):
        """ Load all training images"""
        train_imgs = []
        train_anomaly = []
        anomaly_path = self.root + "/anomalous_images/output_images/"
        normal_path = self.root + "/images/video_2_images/"
        
        for i, image in enumerate(tqdm(os.listdir(normal_path))):
            if image.endswith('.png'):

                if i%2 == 0 and os.path.exists(os.path.join(anomaly_path, "composite_" + image)):
                    cropped_img = self.T(cv2.imread(os.path.join(anomaly_path, "composite_" + image)))
                    train_anomaly.append(1)

                else:
                    cropped_img = self.T(cv2.imread(os.path.join(normal_path, image)))
                    train_anomaly.append(0)

                img_crop = cropped_img[:cropped_img.shape[1], : , :]
                train_imgs.append(img_crop)
        
        # Stack the images + ground truths
        train_normal_images = torch.stack(train_imgs)
        return list(zip(train_anomaly, train_normal_images))
    

if __name__ == '__main__':

    dataset = KRD(batch_size=10)
    for data in dataset:
            anom, img = data
            #print(img.shape)
            print(anom)
            print(img.shape)

