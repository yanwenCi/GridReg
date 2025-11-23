import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet3D(nn.Module):
    def __init__(self, in_channels, out_channels, init_features=32):
        super(UNet3D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.encoder1 = DoubleConv(in_channels, init_features)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder2 = DoubleConv(init_features, init_features * 2)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder3 = DoubleConv(init_features * 2, init_features * 4)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder4 = DoubleConv(init_features * 4, init_features * 6)
        self.pool4 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.bottleneck = DoubleConv(init_features * 6, init_features * 8)

        self.upconv1 = nn.ConvTranspose3d(init_features * 8, init_features * 6, kernel_size=2, stride=2)
        self.decoder1 = DoubleConv(init_features * 12, init_features * 6)
        self.upconv2 = nn.ConvTranspose3d(init_features * 6, init_features * 4, kernel_size=2, stride=2)
        self.decoder2 = DoubleConv(init_features * 8, init_features * 4)
        self.upconv3 = nn.ConvTranspose3d(init_features * 4, init_features * 2, kernel_size=2, stride=2)
        self.decoder3 = DoubleConv(init_features * 4, init_features * 2)
        self.upconv4 = nn.ConvTranspose3d(init_features * 2, init_features, kernel_size=2, stride=2)
        self.decoder4 = DoubleConv(init_features * 2, init_features)

        self.out_conv = nn.Conv3d(init_features, out_channels, kernel_size=1)

    def forward(self, x):
        enc1 = self.encoder1(x) 
        enc1_pool = self.pool1(enc1)

        enc2 = self.encoder2(enc1_pool)
        enc2_pool = self.pool2(enc2)

        enc3 = self.encoder3(enc2_pool)
        enc3_pool = self.pool3(enc3)

        enc4 = self.encoder4(enc3_pool)
        enc4_pool = self.pool4(enc4)

        bottleneck = self.bottleneck(enc4_pool)

        dec1 = self.upconv1(bottleneck)
        dec1 = torch.cat((enc4, dec1), dim=1)
        dec1 = self.decoder1(dec1)

        dec2 = self.upconv2(dec1)
        dec2 = torch.cat((enc3, dec2), dim=1)
        dec2 = self.decoder2(dec2)

        dec3 = self.upconv3(dec2)
        dec3 = torch.cat((enc2, dec3), dim=1)
        dec3 = self.decoder3(dec3)

        dec4 = self.upconv4(dec3)
        dec4 = torch.cat((enc1, dec4), dim=1)
        dec4 = self.decoder4(dec4)

        output = self.out_conv(dec4)
        return output
