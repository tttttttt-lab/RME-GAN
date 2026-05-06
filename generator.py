import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return self.relu(out)

class RMEGenerator(nn.Module):
    def __init__(self, input_nc=2, output_nc=1, ngf=64, n_blocks=6):
        super().__init__()
        self.conv1 = nn.Conv2d(input_nc, ngf, 7, padding=3)
        self.bn1 = nn.BatchNorm2d(ngf)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(ngf, ngf*2, 3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(ngf*2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(ngf*2, ngf*4, 3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(ngf*4)
        self.relu3 = nn.ReLU(inplace=True)

        self.res_blocks = nn.Sequential(*[
            ResidualBlock(ngf*4) for _ in range(n_blocks)
        ])

        self.upconv1 = nn.ConvTranspose2d(ngf*4, ngf*2, 3, stride=2, padding=1, output_padding=1)
        self.upbn1 = nn.BatchNorm2d(ngf*2)
        self.uprelu1 = nn.ReLU(inplace=True)
        self.upconv2 = nn.ConvTranspose2d(ngf*2, ngf, 3, stride=2, padding=1, output_padding=1)
        self.upbn2 = nn.BatchNorm2d(ngf)
        self.uprelu2 = nn.ReLU(inplace=True)

        self.out_conv = nn.Conv2d(ngf, output_nc, 7, padding=3)
        self.tanh = nn.Tanh()

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.relu3(self.bn3(self.conv3(out)))
        out = self.res_blocks(out)
        out = self.uprelu1(self.upbn1(self.upconv1(out)))
        out = self.uprelu2(self.upbn2(self.upconv2(out)))
        out = self.tanh(self.out_conv(out))
        return out
