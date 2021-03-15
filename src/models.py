import numpy as np
import itertools

import torch
import torch.nn as nn
from torch.nn.modules.module import Module

import segmentation_models_pytorch as smp


class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()

        self.conv_layer1 = self._conv_layer_set(3, 4, 3)
        self.conv_layer2 = self._conv_layer_set(4, 8, 3)
        self.conv_layer3 = self._conv_layer_set(8, 8, 3)
        self.fc1 = nn.Linear(78400, 2048)
        self.fc = nn.Linear(2048*5, 2048)
        self.fc2 = nn.Linear(2048, 128)
        self.fc3 = nn.Linear(128, 16)
        self.fc4 = nn.Linear(19, 1)
        self.relu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()
        self.batch = nn.BatchNorm1d(128)
        self.drop = nn.Dropout(p=0.2)

    def _conv_layer_set(self, in_c, out_c, maxpool_size):
        conv_layer = nn.Sequential(
            nn.Conv3d(
                in_c, out_c, kernel_size=(1, maxpool_size, maxpool_size), padding=0
            ),
            nn.LeakyReLU(),
            nn.MaxPool3d((1, 3, 3)),
        )
        return conv_layer

    def forward(self, bag, annotation):
        # Set 1
        out = self.conv_layer1(bag)
        out = self.conv_layer2(out)
        out = self.conv_layer3(out)
        #print(out.shape)
        out = out.view(out.size(0), -1)
        #print(out.shape)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.drop(out)
        out = self.fc3(out)
        out = self.relu(out)
        gender = annotation["GENDER"].unsqueeze(1).float()
        age = annotation["AGE"].unsqueeze(1).float()
        lymph_count = annotation["LYMPH_COUNT"].unsqueeze(1).float()
        output = torch.cat((out, gender, age, lymph_count), 1)
        # out = self.batch(out)
        # out = self.drop(out)
        out = self.fc4(output)

        return self.sigmoid(out)


class TransferUNet(nn.Module):
    def __init__(self):
        super(TransferUNet, self).__init__()
        self.UNet = smp.Unet(
            encoder_name="resnet18",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=3,  # model output channels (number of classes in your dataset)
        )
        self.encoder = self.UNet.encoder
        self.fc = nn.Sequential(
            nn.MaxPool3d(5),
            nn.Flatten(),
            nn.Linear(20_400, 2048),
            nn.ReLU(),
            nn.Linear(2048, 128),
            nn.ReLU(),
            nn.Linear(128, 16),
            nn.ReLU(),
        )
        self.combine_image_and_annotation = nn.Sequential(
            nn.Linear(19, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, bag, annotation):
        if bag.dim() == 5:
            bag_tuple = torch.unbind(bag)
            encoded_bag_tuple = list(
                self.encoder(bag_unique)[-1] for bag_unique in bag_tuple
            )
            encoded_bag = torch.stack(encoded_bag_tuple)
        else:
            raise Exception("Works only with batch of data")
        output_image = self.fc(encoded_bag)
        gender = annotation["GENDER"].unsqueeze(1).float()
        age = annotation["AGE"].unsqueeze(1).float()
        lymph_count = annotation["LYMPH_COUNT"].unsqueeze(1).float()
        output = torch.cat((output_image, gender, age, lymph_count), 1)
        output = self.combine_image_and_annotation(output)
        return output



# Inspired by https://arxiv.org/pdf/1802.04712.pdf





class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()

        self.conv_layer1 = self._conv_layer_set(3, 4, 3)
        self.conv_layer2 = self._conv_layer_set(4, 8, 3)
        self.conv_layer3 = self._conv_layer_set(8, 8, 3) 


        self.attention = nn.Sequential(
            nn.Linear(49, 8),
            nn.Tanh(),
            nn.Linear(8, 1)
        )

        self.fc = nn.Sequential(
          nn.Linear(1600, 512),
          nn.Dropout(0.3),
          nn.LeakyReLU(),
          nn.Linear(512, 64),
          nn.Dropout(0.3),
          nn.LeakyReLU(),
          nn.Linear(64, 16),
          nn.Dropout(0.3),
          nn.LeakyReLU()
        )

        self.combine_image_and_annotation = nn.Sequential(
            nn.Linear(19, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )
        

    def _conv_layer_set(self, in_c, out_c, maxpool_size):
        conv_layer = nn.Sequential(
            nn.Conv3d(
                in_c, out_c, kernel_size=(1, maxpool_size, maxpool_size), padding=0
            ),
            nn.LeakyReLU(),
            nn.MaxPool3d((1, 3, 3)),
        )


        return conv_layer

    def forward(self, bag, annotation):
        # Set 1
        out = self.conv_layer1(bag)
        out = self.conv_layer2(out)
        out = self.conv_layer3(out)
        #print(out.shape)

        out = out.view(out.size(0), out.size(1), out.size(2), 7*7)
        #print(out.shape)
        out = self.attention(out)
        #print(out.shape)


        out = out.view(out.size(0), -1)

        out = self.fc(out)
        #print(out.shape)

        gender = annotation["GENDER"].unsqueeze(1).float()
        age = annotation["AGE"].unsqueeze(1).float()
        lymph_count = annotation["LYMPH_COUNT"].unsqueeze(1).float()
        output = torch.cat((out, gender, age, lymph_count), 1)

        out = self.combine_image_and_annotation(output)

        return out
"""
def tryout(nn.Module):
    def __init__(self):
        super(tryout, self).__init__()
        
        self.conv_layer1 = nn.Conv3d(in_c, out_c, kernel_size=(1, 3, 3), padding=0),
"""
# [32, 3, 1, 3, 3], but got 4-dimensional input of size [200, 224, 224, 3]
