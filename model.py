import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *
import torch.optim as optim
import torchvision.transforms as transforms
import pickle
import math
import torchvision.models as models
from torch.autograd.function import Function

class MBSNETV4(nn.Module):
    def __init__(self):
        super(MBSNETV4, self).__init__()
        backbone = models.resnet18(True) 
        backbone1 = models.resnet18(True)      
        pretrained = torch.load('/root/FERDenoise-BMM-Cotaching/resnet18_msceleb.pth')
        pretrained_state_dict = pretrained['state_dict']
        model_state_dict = backbone.state_dict()
        loaded_keys = 0
        total_keys = 0
        for key in pretrained_state_dict:
            if  ((key=='module.fc.weight')|(key=='module.fc.bias')):
                pass
            else:    
                model_state_dict[key] = pretrained_state_dict[key]
                total_keys+=1
                if key in model_state_dict:
                    loaded_keys+=1
        print("Loaded params num:", loaded_keys)
        print("Total params num:", total_keys)
        backbone.load_state_dict(model_state_dict, strict = False)
        #backbone1.load_state_dict(model_state_dict, strict = False)
        self.tasks = ['br1', 'br2', 'br3', 'br4']
        num_classes = 7
        self.feature = nn.Sequential(*list(backbone.children())[0:-2])
        self.feature1 = nn.Sequential(*list(backbone1.children())[0:-2])
        self.avp = nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.shared_layer4_t = nn.Sequential(*list(backbone.children())[-2:-1])
        self.shared_layer4_t1 = nn.Sequential(*list(backbone1.children())[-2:-1])
        self.fc = nn.Linear(512, num_classes)
        self.fc1 = nn.Linear(512, num_classes)
       
    def forward(self, x):
        xa = self.feature(x)
        x1a = self.feature1(x)
        u_4_ta = self.shared_layer4_t(xa)
        u_4_t1a = self.shared_layer4_t1(x1a)
        fea = [0 for _ in self.tasks]
        fea[0] = u_4_ta = u_4_ta.view(u_4_ta.size(0), -1)
        fea[1] = u_4_t1a = u_4_t1a.view(u_4_t1a.size(0), -1)
        out = [0 for _ in self.tasks]
        out[0] = self.fc(u_4_ta)
        out[1] = self.fc1(u_4_t1a)
        return fea,out