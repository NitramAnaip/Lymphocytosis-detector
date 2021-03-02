import numpy as np
import itertools

import torch
import torch.nn as nn
from torch.nn.modules.module import Module




class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        
        self.conv_layer1 = self._conv_layer_set(3, 8, 3)
        self.conv_layer2 = self._conv_layer_set(8, 16, 3)
        self.conv_layer3 = self._conv_layer_set(16, 16, 2)
        self.fc1 = nn.Linear(156800, 2048)
        self.fc2 = nn.Linear(2048, 128)
        self.fc3 = nn.Linear(128, 16)
        self.fc4 = nn.Linear(16, 1)
        self.relu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()
        self.batch=nn.BatchNorm1d(128)
        self.drop=nn.Dropout(p=0.15)        
        
    def _conv_layer_set(self, in_c, out_c, maxpool_size):
        conv_layer = nn.Sequential(
        nn.Conv3d(in_c, out_c, kernel_size=(1, maxpool_size, maxpool_size), padding=0),
        nn.LeakyReLU(),
        nn.MaxPool3d((1, 3, 3)),
        )
        return conv_layer


    def forward(self, x):
        # Set 1
        out = self.conv_layer1(x)
        out = self.conv_layer2(out)
        out = self.conv_layer3(out)
        #print(out.shape)
        out = out.view(out.size(0), -1)
        #print(out.shape)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.relu(out)
        #out = self.batch(out)
        out = self.drop(out)
        out = self.fc3(out)
        out = self.relu(out)
        #out = self.batch(out)
        out = self.drop(out)
        out = self.fc4(out)
        
        return self.sigmoid(out)

"""
def tryout(nn.Module):
    def __init__(self):
        super(tryout, self).__init__()
        
        self.conv_layer1 = nn.Conv3d(in_c, out_c, kernel_size=(1, 3, 3), padding=0),
"""
#[32, 3, 1, 3, 3], but got 4-dimensional input of size [200, 224, 224, 3]