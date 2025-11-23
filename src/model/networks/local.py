import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
import src.model.layers as layers


class LocalModel(nn.Module):
    def __init__(self, config):
        super(LocalModel, self).__init__()
        self.input_shape = config.input_shape if config.patched == 0 else config.patch_size
        self.ddf_levels = config.ddf_levels

        nc = [config.nc_initial*(2**i) for i in range(5)]
        up_sz = self.calc_upsample_layer_output_size(self.input_shape, 4)
        # print('up_sz', up_sz)

        self.downsample_block0 = layers.DownsampleBlock(inc=config.inc, outc=nc[0])
        self.downsample_block1 = layers.DownsampleBlock(inc=nc[0], outc=nc[1])
        self.downsample_block2 = layers.DownsampleBlock(inc=nc[1], outc=nc[2])
        self.downsample_block3 = layers.DownsampleBlock(inc=nc[2], outc=nc[3])
        self.conv_block = layers.Conv3dBlock(inc=nc[3], outc=nc[4])

        self.upsample_block0 = layers.UpSampleBlock(inc=nc[4], outc=nc[3], output_size=up_sz[0])
        self.upsample_block1 = layers.UpSampleBlock(inc=nc[3], outc=nc[2], output_size=up_sz[1])
        self.upsample_block2 = layers.UpSampleBlock(inc=nc[2], outc=nc[1], output_size=up_sz[2])
        self.upsample_block3 = layers.UpSampleBlock(inc=nc[1], outc=nc[0], output_size=up_sz[3])
        self.ddf_fuse_layers = [layers.DDFFusion(inc=nc[4 - i], out_shape=self.input_shape).cuda() for i in self.ddf_levels]
    

    def calc_upsample_layer_output_size(self, input_shape, num_downsample_layers=4):
        shape = np.array(input_shape)
        tmp = [list(shape//(2**i)) for i in range(num_downsample_layers)]
        tmp.reverse()

        return tmp
        
    def forward(self, x):
        f_down0, f_jc0 = self.downsample_block0(x)
        f_down1, f_jc1 = self.downsample_block1(f_down0)
        f_down2, f_jc2 = self.downsample_block2(f_down1)
        f_down3, f_jc3 = self.downsample_block3(f_down2)
        f_bottleneck = self.conv_block(f_down3)

        # print(f_down0.shape, f_jc0.shape)
        # print(f_down1.shape, f_jc1.shape)
        # print(f_down2.shape, f_jc2.shape)
        # print(f_down3.shape, f_jc3.shape)
        # print(f_bottleneck.shape)

        f_up0 = self.upsample_block0([f_jc3, f_bottleneck])
        f_up1 = self.upsample_block1([f_jc2, f_up0])
        f_up2 = self.upsample_block2([f_jc1, f_up1])
        f_up3 = self.upsample_block3([f_jc0, f_up2])

        # print('-'*20)
        # print(f_up0.shape)
        # print(f_up1.shape)
        # print(f_up2.shape)
        # print(f_up3.shape)

        ddf0 = self.ddf_fuse_layers[0](f_bottleneck)
        ddf1 = self.ddf_fuse_layers[1](f_up0)
        ddf2 = self.ddf_fuse_layers[2](f_up1)
        ddf3 = self.ddf_fuse_layers[3](f_up2)
        ddf4 = self.ddf_fuse_layers[4](f_up3)

        ddf = torch.sum(torch.stack([ddf0, ddf1, ddf2, ddf3, ddf4], axis=5), axis=5)
        return f_bottleneck, ddf


class LocalAffine(nn.Module):
    '''predict an affine matrix'''
    def __init__(self, config):
        super(LocalAffine, self).__init__()
        self.input_shape = config.input_shape
        nc = [config.nc_initial*(2**i) for i in range(5)]

        self.downsample_block0 = layers.DownsampleBlock(inc=2, outc=nc[0])
        self.downsample_block1 = layers.DownsampleBlock(inc=nc[0], outc=nc[1])
        self.downsample_block2 = layers.DownsampleBlock(inc=nc[1], outc=nc[2])
        self.downsample_block3 = layers.DownsampleBlock(inc=nc[2], outc=nc[3])
        self.conv_block = layers.Conv3dBlock(inc=nc[3], outc=nc[4])

    def forward(self, x):
        f_down0, _ = self.downsample_block0(x)
        f_down1, _ = self.downsample_block1(f_down0)
        f_down2, _ = self.downsample_block2(f_down1)
        f_down3, _ = self.downsample_block3(f_down2)
        f_bottleneck = self.conv_block(f_down3)

        return f_bottleneck


class LocalEncoder(nn.Module):
    '''Just an encoder to generate a downsampled DDF'''
    def __init__(self, config):
        super(LocalEncoder, self).__init__()
        self.input_shape = config.input_shape
        nc = [config.nc_initial*(2**i) for i in range(5)]
        self.downsample_block0 = layers.DownsampleBlock(inc=2, outc=nc[0])     # 64
        self.downsample_block1 = layers.DownsampleBlock(inc=nc[0], outc=nc[1]) # 32
        self.downsample_block2 = layers.DownsampleBlock(inc=nc[1], outc=nc[2]) # 16
        self.adpt_pool = nn.AdaptiveAvgPool3d(config.ddf_outshape)

        self.ddf_fuse_layers = layers.DDFFusion(inc=nc[2], out_shape=self.input_shape)

    def forward(self, x):
        f_down0, _ = self.downsample_block0(x)
        f_down1, _ = self.downsample_block1(f_down0)
        f_down2, _ = self.downsample_block2(f_down1)
        
        raw_ddfs = self.adpt_pool(f_down2)
        ddf = self.ddf_fuse_layers(raw_ddfs)

        return raw_ddfs, ddf


class CondiSegUNet(nn.Module):
    '''Unet for conditional segmentation'''
    def __init__(self, config):
        super(CondiSegUNet, self).__init__()
        self.input_shape = config.input_shape if config.patched == 0 else config.patch_size

        nc = [config.nc_initial*(2**i) for i in range(5)]
        up_sz = self.calc_upsample_layer_output_size(self.input_shape, 4)

        self.downsample_block0 = layers.DownsampleBlock(inc=3, outc=nc[0])
        self.downsample_block1 = layers.DownsampleBlock(inc=nc[0], outc=nc[1])
        self.downsample_block2 = layers.DownsampleBlock(inc=nc[1], outc=nc[2])
        self.downsample_block3 = layers.DownsampleBlock(inc=nc[2], outc=nc[3])
        self.conv_block = layers.Conv3dBlock(inc=nc[3], outc=nc[4])

        self.upsample_block0 = layers.UpSampleBlock(inc=nc[4], outc=nc[3], output_size=up_sz[0])
        self.upsample_block1 = layers.UpSampleBlock(inc=nc[3], outc=nc[2], output_size=up_sz[1])
        self.upsample_block2 = layers.UpSampleBlock(inc=nc[2], outc=nc[1], output_size=up_sz[2])
        self.upsample_block3 = layers.UpSampleBlock(inc=nc[1], outc=nc[0], output_size=up_sz[3])
        self.fuse_layer = nn.Conv3d(in_channels=nc[0], out_channels=1, kernel_size=3, stride=1, padding=1)
    
    def calc_upsample_layer_output_size(self, input_shape, num_downsample_layers=4):
        shape = np.array(input_shape)
        tmp = [list(shape//(2**i)) for i in range(num_downsample_layers)]
        tmp.reverse()
        return tmp

    def forward(self, x):
        f_down0, f_jc0 = self.downsample_block0(x)
        f_down1, f_jc1 = self.downsample_block1(f_down0)
        f_down2, f_jc2 = self.downsample_block2(f_down1)
        f_down3, f_jc3 = self.downsample_block3(f_down2)
        f_bottleneck = self.conv_block(f_down3)

        f_up0 = self.upsample_block0([f_jc3, f_bottleneck])
        f_up1 = self.upsample_block1([f_jc2, f_up0])
        f_up2 = self.upsample_block2([f_jc1, f_up1])
        f_up3 = self.upsample_block3([f_jc0, f_up2])

        out = self.fuse_layer(f_up3)
        out = F.interpolate(out, size=self.input_shape, mode='trilinear', align_corners=True)

        return torch.sigmoid(out)



class CondiSegUNet(nn.Module):
    '''Unet for conditional segmentation'''
    def __init__(self, config):
        super(CondiSegUNet, self).__init__()
        self.input_shape = config.input_shape if config.patched == 0 else config.patch_size

        nc = [config.nc_initial*(2**i) for i in range(5)]
        up_sz = self.calc_upsample_layer_output_size(self.input_shape, 4)

        self.downsample_block0 = layers.DownsampleBlock(inc=3, outc=nc[0])
        self.downsample_block1 = layers.DownsampleBlock(inc=nc[0], outc=nc[1])
        self.downsample_block2 = layers.DownsampleBlock(inc=nc[1], outc=nc[2])
        self.downsample_block3 = layers.DownsampleBlock(inc=nc[2], outc=nc[3])
        self.conv_block = layers.Conv3dBlock(inc=nc[3], outc=nc[4])

        self.upsample_block0 = layers.UpSampleBlock(inc=nc[4], outc=nc[3], output_size=up_sz[0])
        self.upsample_block1 = layers.UpSampleBlock(inc=nc[3], outc=nc[2], output_size=up_sz[1])
        self.upsample_block2 = layers.UpSampleBlock(inc=nc[2], outc=nc[1], output_size=up_sz[2])
        self.upsample_block3 = layers.UpSampleBlock(inc=nc[1], outc=nc[0], output_size=up_sz[3])
        self.fuse_layer = nn.Conv3d(in_channels=nc[0], out_channels=1, kernel_size=3, stride=1, padding=1)
    
    def calc_upsample_layer_output_size(self, input_shape, num_downsample_layers=4):
        shape = np.array(input_shape)
        tmp = [list(shape//(2**i)) for i in range(num_downsample_layers)]
        tmp.reverse()
        return tmp

    def forward(self, x):
        f_down0, f_jc0 = self.downsample_block0(x)
        f_down1, f_jc1 = self.downsample_block1(f_down0)
        f_down2, f_jc2 = self.downsample_block2(f_down1)
        f_down3, f_jc3 = self.downsample_block3(f_down2)
        f_bottleneck = self.conv_block(f_down3)

        f_up0 = self.upsample_block0([f_jc3, f_bottleneck])
        f_up1 = self.upsample_block1([f_jc2, f_up0])
        f_up2 = self.upsample_block2([f_jc1, f_up1])
        f_up3 = self.upsample_block3([f_jc0, f_up2])

        out = self.fuse_layer(f_up3)
        out = F.interpolate(out, size=self.input_shape, mode='trilinear', align_corners=True)

        return torch.sigmoid(out)



class UNet(nn.Module):
    '''Unet for general segmentation'''
    def __init__(self, config):
        super(UNet, self).__init__()
        self.input_shape = config.input_shape if config.patched == 0 else config.patch_size

        nc = [config.nc_initial*(2**i) for i in range(5)]
        up_sz = self.calc_upsample_layer_output_size(self.input_shape, 4)

        self.downsample_block0 = layers.DownsampleBlock(inc=config.inc, outc=nc[0])
        self.downsample_block1 = layers.DownsampleBlock(inc=nc[0], outc=nc[1])
        self.downsample_block2 = layers.DownsampleBlock(inc=nc[1], outc=nc[2])
        self.downsample_block3 = layers.DownsampleBlock(inc=nc[2], outc=nc[3])
        self.conv_block = layers.Conv3dBlock(inc=nc[3], outc=nc[4])

        self.upsample_block0 = layers.UpSampleBlock(inc=nc[4], outc=nc[3], output_size=up_sz[0])
        self.upsample_block1 = layers.UpSampleBlock(inc=nc[3], outc=nc[2], output_size=up_sz[1])
        self.upsample_block2 = layers.UpSampleBlock(inc=nc[2], outc=nc[1], output_size=up_sz[2])
        self.upsample_block3 = layers.UpSampleBlock(inc=nc[1], outc=nc[0], output_size=up_sz[3])
        self.fuse_layer = nn.Conv3d(in_channels=nc[0], out_channels=config.outc, kernel_size=3, stride=1, padding=1)
    
    def calc_upsample_layer_output_size(self, input_shape, num_downsample_layers=4):
        shape = np.array(input_shape)
        tmp = [list(shape//(2**i)) for i in range(num_downsample_layers)]
        tmp.reverse()
        return tmp

    def forward(self, x):
        f_down0, f_jc0 = self.downsample_block0(x)
        f_down1, f_jc1 = self.downsample_block1(f_down0)
        f_down2, f_jc2 = self.downsample_block2(f_down1)
        f_down3, f_jc3 = self.downsample_block3(f_down2)
        f_bottleneck = self.conv_block(f_down3)

        f_up0 = self.upsample_block0([f_jc3, f_bottleneck])
        f_up1 = self.upsample_block1([f_jc2, f_up0])
        f_up2 = self.upsample_block2([f_jc1, f_up1])
        f_up3 = self.upsample_block3([f_jc0, f_up2])

        out = self.fuse_layer(f_up3)
        out = F.interpolate(out, size=self.input_shape, mode='trilinear', align_corners=True)

        return torch.sigmoid(out)
    

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)

        # If the input and output dimensions are different, adjust with a 1x1 convolution
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels)
            )

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(residual)  # Residual connection
        out = self.relu(out)

        return out
