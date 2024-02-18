r"""
this package is created by Dennis Guo for dealing with the Kannada computer vision project
"""

import os,sys
import numpy as np
import pandas as pd
import zipfile
import matplotlib.pyplot as plt
import seaborn as sns
from random import seed, shuffle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from torchsummary import summary
device = "cuda" if torch.cuda.is_available() else "cpu"
import numpy as np
import pandas as pd
from torchvision import models, transforms, datasets
from torchvision.models.resnet import ResNet50_Weights, ResNet101_Weights
from torchviz import make_dot
from tqdm import *
import torch.functional as F


__all__ = ["kannada_data", "testdata","KannaResNet50", "train_batch", "accuracy", "random_seed"]


class kannada_data(Dataset):
    def __init__(self, data : pd.DataFrame, test_size : float = 0.2, datatype = "train") -> None:
        super().__init__()
        train, val = train_test_split(data, test_size = test_size, shuffle= True, random_state=42)
        if datatype not in ["train", "val"]:
            raise ValueError("only train and val allowed")
        elif datatype == "train":
            self.dataframe = train
        elif datatype == "val":
            self.dataframe = val
            
        self.train = np.array(self.dataframe.drop(["label"], axis = 1))
        self.target = np.array(self.dataframe.label)
        
    def __len__(self):
        return len(self.train)
    
    def __getitem__(self, index):
        return torch.Tensor(self.train[index].reshape(-1,28,28)/255).to(device=device), torch.Tensor(self.target[index].reshape(1)).long().to(device=device)
    
class testdata(Dataset):
    def __init__(self, data : pd.DataFrame) -> None:
        super().__init__()
        self.dataframe = data
    
        self.train = np.array(self.dataframe.drop(["id"], axis = 1))
        self.id = np.array(self.dataframe.id)
        
    def __len__(self):
        return len(self.train)
    
    def __getitem__(self, index):
        return torch.Tensor(self.train[index].reshape(-1,28,28)/255).to(device=device)    
    
class KannaResNet50(nn.Module):
    def __init__(self, num_classes):
        super(KannaResNet50, self).__init__()
        # Load a pre-trained ResNet18 model
        self.resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        for parameter in self.resnet.parameters():
            parameter.requires_grad = False        
        self.resnet.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7, stride=2, padding=3)

        # Replace the final fully connected layer to match the number of classes
        num_features = self.resnet.fc.in_features  # Get the number of input features of the original fc layer

        self.resnet.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_features, 128),     #here is 2048 for the resnet50
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
            nn.Sigmoid(),
        )        

    def forward(self, x):
        return self.resnet(x)
    
    
class KannaResNet101(nn.Module):
    def __init__(self, num_classes):
        super(KannaResNet50, self).__init__()
        # Load a pre-trained ResNet18 model
        self.resnet = models.resnet101(weights=ResNet101_Weights.IMAGENET1K_V1)
        for parameter in self.resnet.parameters():
            parameter.requires_grad = False        
        self.resnet.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7, stride=2, padding=3)

        # Replace the final fully connected layer to match the number of classes
        num_features = self.resnet.fc.in_features  # Get the number of input features of the original fc layer

        self.resnet.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_features, 128),     #here is 2048 for the resnet50
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
            nn.Sigmoid(),
        )        

    def forward(self, x):
        return self.resnet(x)    
    
    
def train_batch(x, y, model, loss_fn, optimizer):
    prediction = model(x)
    loss_fn = loss_fn
    optimizer = optimizer
    batch_loss = loss_fn(prediction, y)
    batch_loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return batch_loss.item()

@torch.no_grad()
def accuracy(x, y, model):
    model.eval()
    prediction = model(x)
    _, argmaxes = prediction.max(-1) #I only need the index here
    is_correct = (argmaxes == y)
    return is_correct.detach().cpu().tolist()

def random_seed(seed_value, use_cuda):
    np.random.seed(seed_value) # cpu vars
    torch.manual_seed(seed_value) # cpu  vars
    seed(seed_value) # Python
    if use_cuda: 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value) # gpu vars
        torch.backends.cudnn.deterministic = True  #needed
        torch.backends.cudnn.benchmark = False
        
        
def testdata() -> None:
    pass
            
# class shit(nn.Module):
#     def __init__(self, num_classes):
#         super(KannaResNet50, self).__init__()
#         # Load a pre-trained ResNet18 model
#         self.resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
#         for parameter in self.resnet.parameters():
#             parameter.requires_grad = False        
#         self.resnet.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7, stride=2, padding=3)

#         # Replace the final fully connected layer to match the number of classes
#         num_features = self.resnet.fc.in_features  # Get the number of input features of the original fc layer

#         self.resnet.fc = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(num_features, 128),     #here is 2048 for the resnet50
#             nn.ReLU(),
#             nn.Dropout(0.2),
#             nn.Linear(128, num_classes),
#             nn.Sigmoid(),
#         )        

#     def forward(self, x):
#         # Manually apply each layer up to avgpool
#         x = self.resnet.conv1(x)
#         x = self.resnet.bn1(x)
#         x = self.resnet.relu(x)
#         x = self.resnet.maxpool(x)

#         x = self.resnet.layer1(x)
#         x = self.resnet.layer2(x)
#         x = self.resnet.layer3(x)
#         x = self.resnet.layer4(x)
#         # print(f"x shape is {x.shape} before avgpool")
#         # Apply AdaptiveAvgPool2d
#         x = self.resnet.avgpool(x)
        
#         # Apply the custom fully connected layer
#         x = self.resnet.fc(x)
#         # print(x.shape)
    
#         return x