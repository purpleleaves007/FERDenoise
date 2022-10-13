import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from utils import *
from data_loader import  dataprtrraf, dataprteraf
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import datasets
import torchvision.models as models
from model import *
import pandas as pd
from PIL import Image
import torch.utils.data as data

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = MBSNETV4().to(device)
params = net.parameters()
optimizer = optim.Adam(net.parameters(), lr=0.0002)
milestones = [10,20]
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1, last_epoch=-1)

# define dataset
data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
                transforms.RandomCrop(224, padding=32),
            ], p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(scale=(0.02,0.25))
        ])

data_transforms_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])   

train_list = r'/root/rafdb/'
#train_list = r'/root/autodl-tmp/ferplus/Trainingplus1'
#train_list = r'/media/a808/G/ZXDATA/FED/affectnet/train' 
dataset_source = dataprtrraf(
    data_list=train_list,
    transform=data_transforms
)

trainloader = data.DataLoader(
    dataset=dataset_source,
    batch_size=96,
    shuffle=True,
    num_workers=8,
    pin_memory = True)

lengthtr = len(trainloader)

test_list = r'/root/rafdb/'
#test_list = r'/root/autodl-tmp/ferplus/Valid3'
#test_list = r'/root/autodl-tmp/affectnet/gentest' 
dataset_target = dataprteraf(
    data_list=test_list,
    transform=data_transforms_val
)

testloader = data.DataLoader(
    dataset=dataset_target,
    batch_size=32,
    shuffle=False,
    num_workers=10,
    pin_memory = True)

lengthte = len(testloader)
print('Train set size:', dataset_source.__len__())
print('Validation set size:', dataset_target.__len__())

# Train and evaluate multi-task network
if __name__ == '__main__':
    cross_denoising_trainer(trainloader,
                   testloader,
                   net,
                   device,
                   optimizer,
                   scheduler,
                   lengthtr,
                   lengthte,
                   30)