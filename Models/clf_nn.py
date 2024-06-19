import torch.nn as nn
import torch
from torch.nn import functional as F


class SFCN(nn.Module):
    def __init__(self):
        super(SFCN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding="same")
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding="same")
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding="same")
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding="same")
        self.bn4 = nn.BatchNorm2d(256)
        self.conv5 = nn.Conv2d(256, 64, kernel_size=1, padding="same")
        self.bn5 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(512, 1)

    def forward(self, x):
        x = torch.swapaxes(x, 2, 3)
        x = F.max_pool2d(F.leaky_relu(self.bn1(self.conv1(x))),2)
        x = F.max_pool2d(F.leaky_relu(self.bn2(self.conv2(x))),2)
        x = F.max_pool2d(F.leaky_relu(self.bn3(self.conv3(x))),2)
        x = F.max_pool2d(F.leaky_relu(self.bn4(self.conv4(x))),2)
        x = F.avg_pool2d(F.leaky_relu(self.bn5(self.conv5(x))),2,padding=(1,0))
        emb = torch.flatten(x,start_dim=1, end_dim=-1)  # flatten
        print(emb.shape)
        x = F.dropout(emb, p=0.5)
        print(x.shape)
        x = torch.sigmoid(self.fc1(x))
        return x, emb
    
class SFCN_mega(nn.Module):
    def __init__(self):
        super(SFCN_mega, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding="same")
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding="same")
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=5, padding="same")
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=5, padding="same")
        self.bn4 = nn.BatchNorm2d(256)
        self.conv5 = nn.Conv2d(256, 64, kernel_size=1, padding="same")
        self.bn5 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(512, 1)

    def forward(self, x):
        # x = torch.swapaxes(x, 2, 3)
        x = F.max_pool2d(F.leaky_relu(self.bn1(self.conv1(x))),2)
        x = F.max_pool2d(F.leaky_relu(self.bn2(self.conv2(x))),2)
        x = F.max_pool2d(F.leaky_relu(self.bn3(self.conv3(x))),2)
        x = F.max_pool2d(F.leaky_relu(self.bn4(self.conv4(x))),2)
        x = F.avg_pool2d(F.leaky_relu(self.bn5(self.conv5(x))),2,padding=(1,0))
        emb = torch.flatten(x,start_dim=1, end_dim=-1)  # flatten
        x = F.dropout(emb, p=0.5)
        x = self.fc1(x)
        x = torch.sigmoid(x)
        return x #, emb
    
class CNNClf(nn.Module):
    def __init__(self):
        super(CNNClf, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=1)
        self.pool = nn.MaxPool2d((2, 2))  # Pooling only on the y-axis
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=1)
        self.fc1 = nn.Linear(64 * 67 * 6, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 1)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 67 * 6)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))
        return x
    
class SampleModel(nn.Module):
    def __init__(self, num_channel_in=5, class_num=2):
        super(SampleModel, self).__init__()
        self.c1 = nn.Conv2d(num_channel_in, 128, kernel_size=3, stride=1, padding=1)
        self.c2 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.c3 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = torch.nn.Linear(512, class_num)
        self.softmax = torch.nn.Softmax(dim = 1)

    def forward(self, x):
        x = self.c1(x)
        x = self.c2(x)
        x = self.c3(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.linear(x)
        x = self.softmax(x)
        return x
    
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(5, 8, kernel_size=3, stride = 1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=5, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=5, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.linear = nn.Linear(32*3*33, 2)
        self.sig = nn.Sigmoid()


    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        x = x.reshape(x.shape[0], -1)
        x = self.linear(x)
        x = self.sig(x)
        
        return x
    

    
