import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3,
                               stride=2, padding=1, bias=False)  # out: 64 x 64 x 32
        self.b1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3,
                               stride=2, padding=1, bias=False)  # out: 32 x 32 x 64
        self.b2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3,
                               stride=2, padding=1, bias=False)  # out: 16 x 16 x 128
        self.b3 = nn.BatchNorm2d(128)        
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3,
                               stride=1, padding=1, bias=False)  # out: 16 x 16 x 256
        self.b4 = nn.BatchNorm2d(256)                
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=512,
                               kernel_size=3, stride=2, padding=1, bias=False)  # out: 8 x 8 x 512
        self.b5 = nn.BatchNorm2d(512)
        self.conv6 = nn.Conv2d(in_channels=512, out_channels=1024,
                               kernel_size=3, stride=2, padding=1, bias=False)  # out: 4 x 4 x 1024
        self.b6 = nn.BatchNorm2d(1024)

        nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.kaiming_normal_(self.conv2.weight)
        nn.init.kaiming_normal_(self.conv3.weight)
        nn.init.kaiming_normal_(self.conv4.weight)
        nn.init.kaiming_normal_(self.conv5.weight)
        nn.init.kaiming_normal_(self.conv6.weight)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x),0.2)
        x = self.b1(x)
        x = F.leaky_relu(self.conv2(x),0.2)
        x = self.b2(x)
        x = F.leaky_relu(self.conv3(x),0.2)
        x = self.b3(x)       
        x = F.leaky_relu(self.conv4(x),0.2)
        x = self.b4(x)                
        x = F.leaky_relu(self.conv5(x),0.2)
        x = self.b5(x)
        x = F.leaky_relu(self.conv6(x),0.2)
        x = self.b6(x)

        return x


class Eshared(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super(Eshared, self).__init__()
        self.conv7 = nn.Conv2d(in_channels=1024, out_channels=1024,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.b7 = nn.BatchNorm2d(1024)        
        self.conv8 = nn.Conv2d(in_channels=1024, out_channels=1024,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.b8 = nn.BatchNorm2d(1024)                
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=4*4*1024,
                             out_features=2048, bias=False)
        self.bfc1 = nn.BatchNorm1d(2048)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(in_features=2048, out_features=2048, bias=False)
        self.bfc2 = nn.BatchNorm1d(2048)

        nn.init.kaiming_normal_(self.conv7.weight)
        nn.init.kaiming_normal_(self.conv8.weight)        
        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.kaiming_normal_(self.fc2.weight)

    def forward(self, x):
        x = F.leaky_relu(self.conv7(x),0.2)
        x = self.b7(x)
        x = F.leaky_relu(self.conv8(x),0.2)
        x = self.b8(x)        
        x = self.flatten(x)
        x = F.leaky_relu(self.fc1(x),0.2)
        x = self.bfc1(x)
        x = self.dropout2(x)
        x = F.leaky_relu(self.fc2(x),0.2)
        x = self.bfc2(x)

        return x
