"""
author: Fabian Seiler @ 20.06.24
"""
import os
from torchvision import transforms
import torchvision.transforms.functional as F
import torch
import cv2
from tqdm import tqdm


# Loads Test/Training Data into one Data folder

class Dataloader():

    def __init__(self, batch_size, root="/DATASET_PATH", augment: bool = False):

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
        #train_path = os.path.join(self.root, "train/")
        train_path = os.path.join(self.root, "train/")
        
        # Load all training data into one container
        for folder in os.listdir(train_path):
            if folder != 'summer_test': continue
            for subfolder in os.listdir(train_path + folder):
                print(f"folder={folder}_{subfolder[-1]}")
                #if subfolder != 'section1': continue
                for i, image in enumerate(tqdm(os.listdir(train_path + folder + "/" + subfolder))):
                    if image.endswith('.png'):
                        # Crop and append input image
                        #cropped_img = self.crop_img(cv2.imread(train_path + folder + "/" + subfolder + "/" + image), h_line=512, v_delta=480)
                        cropped_img = self.T(cv2.imread(train_path + folder + "/" + subfolder + "/" + image))
                        train_imgs.append(cropped_img)
                        anom = image.split('.png')[0].startswith('composite')

                        if anom:
                            train_anomaly.append(True)
                        else:
                            train_anomaly.append(False)

                        
                        if augmentation:
                            # Flip around x-axis (left - right)
                            img_flip = self.mirror_imgs(cropped_img)
                            train_imgs.append(img_flip)
    
                            if anom:
                                train_anomaly.append(True)
                            else:
                                train_anomaly.append(False)

                            # Lighting effects
                            for _ in range(3):
                                img_l = self.change_lighting(cropped_img)
                                train_imgs.append(img_l)
                                if anom:
                                    train_anomaly.append(True)
                                else:
                                    train_anomaly.append(False)
                        
        # Stack the images + ground truths
        train_normal_images = torch.stack(train_imgs)
        return list(zip(train_anomaly, train_normal_images))
