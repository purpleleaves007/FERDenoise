import torch.utils.data as data
from PIL import Image
import os
import torchvision.transforms as transforms
import pandas as pd
import numpy as np
import cv2
import argparse,random

class dataprtrraf(data.Dataset):
     def __init__(self, data_list, transform=False):
        self.transform = transform
        self.img_paths = []
        self.label = []
        self.labeltr = []
        file_names = []

        count = 0
        j0 = 0
        j1 = 0
        j2 = 0
        j3 = 0
        j4 = 0
        j5 = 0
        j6 = 0
        n = 0
        img_path = data_list
        total_imgs = len(os.listdir(img_path))
        count = count+total_imgs
        self.n_data = count

        self.raf_path = data_list
        self.file_paths = []
        list_patition_label = pd.read_csv(r'/root/rafdb/0.3noise_train1.txt', header=None, delim_whitespace=True)
        list_patition_label = np.array(list_patition_label)
        list_patition_labeltr = pd.read_csv(r'/root/rafdb/0.0noise_train1.txt', header=None, delim_whitespace=True)
        list_patition_labeltr = np.array(list_patition_labeltr)
        for index in range(list_patition_label.shape[0]):
            if list_patition_label[index,0][:5] == "train":
                file_names.append(list_patition_label[index,0])
                self.label.append(list_patition_label[index,1]-1)
                self.labeltr.append(list_patition_labeltr[index,1]-1)
                if list_patition_label[index,1] == 1:
                    j0 = j0 + 1
                elif list_patition_label[index,1] == 2:
                    j1 = j1 + 1
                elif list_patition_label[index,1] == 3:
                    j2 = j2 + 1
                elif list_patition_label[index,1] == 4:
                    j3 = j3 + 1
                elif list_patition_label[index,1] == 5:
                    j4 = j4 + 1
                elif list_patition_label[index,1] == 6:
                    j5 = j5 + 1
                elif list_patition_label[index,1] == 7:
                    j6 = j6 + 1
        print(j0)
        print(j1)
        print(j2)
        print(j3)
        print(j4)
        print(j5)
        print(j6)
        for f in file_names:
            f = f.split(".")[0]
            f = f + "_aligned.jpg"
            path = os.path.join(self.raf_path, 'aligned', f)
            self.file_paths.append(path)   

     def __len__(self):
        return len(self.file_paths)

     def __getitem__(self, idx):
        path = self.file_paths[idx]
        image = Image.open(path).convert('RGB')
        label = self.label[idx]
        labeltr = self.labeltr[idx]
        # augmentation
        if self.transform is not None:
            image1 = self.transform(image)
            #image2 = self.transforms1(image1)
        return labeltr, image1, label

class dataprteraf(data.Dataset):
     def __init__(self, data_list, transform=False):
        self.transform = transform
        self.img_paths = []
        self.label = []
        file_names = []

        count = 0
        img_path = data_list
        total_imgs = len(os.listdir(img_path))
        count = count+total_imgs
        self.n_data = count

        self.raf_path = data_list
        self.file_paths = []
        list_patition_label = pd.read_csv(r'/root/rafdb/0.0noise_train1.txt', header=None, delim_whitespace=True)
        list_patition_label = np.array(list_patition_label)
        for index in range(list_patition_label.shape[0]):
            if list_patition_label[index,0][:4] == "test":
                file_names.append(list_patition_label[index,0])
                self.label.append(list_patition_label[index,1]-1)

        for f in file_names:
            f = f.split(".")[0]
            f = f +"_aligned.jpg"
            path = os.path.join(self.raf_path, 'aligned', f)
            self.file_paths.append(path)  

     def __len__(self):
        return len(self.file_paths)

     def __getitem__(self, idx):
        path = self.file_paths[idx]
        image = Image.open(path).convert('RGB')
        #image = image[:, :, ::-1] # BGR to RGB
        label = self.label[idx]
        # augmentation

        if self.transform is not None:
            image = self.transform(image)
        
        return image, label

class dataprterafou(data.Dataset):
     def __init__(self, data_list, transform=False):
        self.transform = transform
        self.img_paths = []
        self.label = []
        file_names = []

        count = 0
        img_path = data_list
        total_imgs = len(os.listdir(img_path))
        count = count+total_imgs
        self.n_data = count

        self.raf_path = data_list
        self.file_paths = []
        list_patition_label = pd.read_csv(r'/root/rafdb/rafdb_occlusion_list_45.txt', header=None, delim_whitespace=True)
        list_patition_label = np.array(list_patition_label)
        for index in range(list_patition_label.shape[0]):
            #if list_patition_label[index,0][:4] == "test":
            file_names.append(list_patition_label[index,0])
            self.label.append(list_patition_label[index,1])

        for f in file_names:
            #f = f.split(".")[0]
            f = f +".jpg"
            path = os.path.join(self.raf_path, 'aligned', f)
            self.file_paths.append(path)  

     def __len__(self):
        return len(self.file_paths)

     def __getitem__(self, idx):
        path = self.file_paths[idx]
        image = Image.open(path).convert('RGB')
        #image = image[:, :, ::-1] # BGR to RGB
        label = self.label[idx]
        # augmentation

        if self.transform is not None:
            image = self.transform(image)
        
        return image, label

class dataprterafpo(data.Dataset):
     def __init__(self, data_list, transform=False):
        self.transform = transform
        self.img_paths = []
        self.label = []
        file_names = []

        count = 0
        img_path = data_list
        total_imgs = len(os.listdir(img_path))
        count = count+total_imgs
        self.n_data = count

        self.raf_path = data_list
        self.file_paths = []
        list_patition_label = pd.read_csv(r'/root/rafdb/val_raf_db_list.txt', header=None, delim_whitespace=True)
        list_patition_label = np.array(list_patition_label)
        for index in range(list_patition_label.shape[0]):
            #if list_patition_label[index,0][:4] == "test":
            file_names.append(list_patition_label[index,0][2:-1])
            self.label.append(list_patition_label[index,0][0])

        for f in file_names:
            f = f.split(".")[0]
            f = f +".jpg"
            path = os.path.join(self.raf_path, 'aligned', f)
            self.file_paths.append(path)  

     def __len__(self):
        return len(self.file_paths)

     def __getitem__(self, idx):
        path = self.file_paths[idx]
        image = Image.open(path).convert('RGB')
        #image = image[:, :, ::-1] # BGR to RGB
        label = int(self.label[idx])

        if self.transform is not None:
            image = self.transform(image)
        
        return image, label