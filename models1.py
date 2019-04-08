## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        
        # input image : 224x224 -> conv1 -> (224-3)/1+1 = 222 -> maxpool -> 110
        self.conv1 = nn.Conv2d(1, 32, 5) 
        #I.xavier_uniform_(self.conv1.weight)
        self.conv1_bn = nn.BatchNorm2d(32)
        
        # dim: conv2/pool -> (110-3)/1+1 = 108 -> maxpool -> 54
        self.conv2 = nn.Conv2d(32, 64, 3)
        #I.xavier_uniform_(self.conv2.weight)
        self.conv2_bn = nn.BatchNorm2d(64)
        
        # dim: conv3/pool -> (53-5)/1+1 = 53 -> maxpool -> 26
        self.conv3 = nn.Conv2d(64, 128, 3)
        #I.xavier_uniform_(self.conv3.weight)
        self.conv3_bn = nn.BatchNorm2d(128)
        
        # dim: conv4 -> (26-3)/1+1 = 24 -> maxpool -> 12
        self.conv4 = nn.Conv2d(128, 256, 2)
        #I.xavier_uniform_(self.conv4.weight)
        self.conv4_bn = nn.BatchNorm2d(256)
        
        self.pool = nn.MaxPool2d(2, 2)
        
        self.fc1 = nn.Linear(12*12*256, 512)
        #I.xavier_uniform_(self.fc1.weight)
        self.dropout = nn.Dropout(0.2)
        
        self.fc2 = nn.Linear(512, 136)
        #I.xavier_uniform_(self.fc2.weight)
        
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        
        x = self.pool(F.relu(self.conv1_bn(self.conv1(x))))
        x = self.pool(F.relu(self.conv2_bn(self.conv2(x))))
        x = self.pool(F.relu(self.conv3_bn(self.conv3(x))))
        x = self.pool(F.relu(self.conv4_bn(self.conv4(x))))
        
        x = x.view(x.size(0),-1)
        
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x