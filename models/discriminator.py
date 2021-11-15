import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16,
                               kernel_size=3, stride=2, padding=1)  # out: 64 x 64 x 16
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3,
                               stride=2, padding=1, bias=False)  # out: 32 x 32 x 32
        self.b2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3,
                               stride=1, padding=1, bias=False)  # out: 16 x 16 x 32
        self.b3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3,
                               stride=1, padding=1, bias=False)  # out: 8 x 8 x 32
        self.b4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3,
                               stride=2, padding=1, bias=False)  # out: 8 x 8 x 32
        self.b5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3,
                               stride=2, padding=1, bias=False)  # out: 8 x 8 x 32
        self.b6 = nn.BatchNorm2d(128)                
        self.conv7 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3,
                               stride=2, padding=1, bias=False)  # out: 4 x 4 x 32                               
        self.b7 = nn.BatchNorm2d(256)                                               
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=4*4*256, out_features=2048, bias=False)
        self.bfc1 = nn.BatchNorm1d(2048)
        self.dropout = nn.Dropout(0.5)        
        self.fc2 = nn.Linear(in_features=2048, out_features=1, bias=False)

        nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.kaiming_normal_(self.conv2.weight)
        nn.init.kaiming_normal_(self.conv3.weight)
        nn.init.kaiming_normal_(self.conv4.weight)
        nn.init.kaiming_normal_(self.conv5.weight)
        nn.init.kaiming_normal_(self.conv6.weight)
        nn.init.kaiming_normal_(self.conv7.weight)        
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)        


    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2(x), 0.2)
        x = self.b2(x)
        x = F.leaky_relu(self.conv3(x), 0.2)
        x = self.b3(x)
        x = F.leaky_relu(self.conv4(x), 0.2)
        x = self.b4(x)
        x = F.leaky_relu(self.conv5(x), 0.2)
        x = self.b5(x)
        x = F.leaky_relu(self.conv6(x), 0.2)
        x = self.b6(x)                
        x = F.leaky_relu(self.conv7(x), 0.2)        
        x = self.flatten(x)
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = self.bfc1(x)
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))

        return x
