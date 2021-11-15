import torch
import torch.nn as nn
import torch.nn.functional as F


class Dshared(nn.Module):
    def __init__(self):
        super(Dshared, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(
            in_channels=2048, out_channels=1024, kernel_size=4, stride=2, bias=False)
        self.bd1 = nn.BatchNorm2d(1024)
        self.deconv2 = nn.ConvTranspose2d(
            in_channels=1024, out_channels=1024, kernel_size=1, stride=1, bias=False)
        self.bd2 = nn.BatchNorm2d(1024)

        nn.init.kaiming_normal_(self.deconv1.weight)
        nn.init.kaiming_normal_(self.deconv2.weight)

    def forward(self, x):
        x = x.view(-1, 2048, 1, 1)
        x = F.leaky_relu(self.deconv1(x),0.2)
        x = self.bd1(x)
        x = F.leaky_relu(self.deconv2(x),0.2)
        x = self.bd2(x)

        return x


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.deconv3 = nn.ConvTranspose2d(
            in_channels=1024, out_channels=512, kernel_size=1, stride=1, bias=False)
        self.bd3 = nn.BatchNorm2d(512)
        self.deconv4 = nn.ConvTranspose2d(
            in_channels=512, out_channels=256, kernel_size=2, stride=2, bias=False)
        self.bd4 = nn.BatchNorm2d(256)
        self.deconv5 = nn.ConvTranspose2d(
            in_channels=256, out_channels=128, kernel_size=2, stride=2, bias=False)
        self.bd5 = nn.BatchNorm2d(128)
        self.deconv6 = nn.ConvTranspose2d(
            in_channels=128, out_channels=64, kernel_size=2, stride=2, bias=False)
        self.bd6 = nn.BatchNorm2d(64)
        self.deconv7 = nn.ConvTranspose2d(
            in_channels=64, out_channels=32, kernel_size=2, stride=2, bias=False)
        self.bd7 = nn.BatchNorm2d(32)        
        self.deconv8 = nn.ConvTranspose2d(
            in_channels=32, out_channels=1, kernel_size=2, stride=2)

        nn.init.kaiming_normal_(self.deconv3.weight)
        nn.init.kaiming_normal_(self.deconv4.weight)        
        nn.init.kaiming_normal_(self.deconv5.weight)
        nn.init.kaiming_normal_(self.deconv6.weight)        
        nn.init.kaiming_normal_(self.deconv7.weight)        
        nn.init.xavier_normal_(self.deconv8.weight)

    def forward(self, x):
        x = F.leaky_relu(self.deconv3(x),0.2)
        x = self.bd3(x)
        x = F.leaky_relu(self.deconv4(x),0.2)
        x = self.bd4(x)
        x = F.leaky_relu(self.deconv5(x),0.2)
        x = self.bd5(x)
        x = F.leaky_relu(self.deconv6(x),0.2)
        x = self.bd6(x)
        x = F.leaky_relu(self.deconv7(x),0.2)
        x = self.bd7(x)        
        x = torch.tanh(self.deconv8(x))

        return x