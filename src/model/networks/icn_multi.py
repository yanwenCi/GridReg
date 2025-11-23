import torch.nn as nn
import torch
import torch.nn.functional as nnf
import src.model.layers as layers
from src.model.networks.VoxelMorph import Stage
import torchvision.models as models

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions.normal import Normal
import numpy as np



class Segcorrection(nn.Module):
    def __init__(self, config):
        super(Segcorrection, self).__init__()
        self.config = config
        self.segnet = Segmentation()
        self.correction = Segmentation(in_channels=3, out_channels=3, enc_feat=[8, 16, 32], dec_feat=[32, 16, 8], bnorm=True, dropout=True)

    def forward(self, x):
        '''
        x --> [batch, 2, h, w, z]
        '''
        seg = self.segnet(x)
        flow = self.correction(torch.cat([x, seg], dim=1))
        return seg, flow

class Segmentation(nn.Module):
    """
    VoxelMorph network for (unsupervised) nonlinear registration between two images.
    Slightly modified implementation.
    """

    def __init__(
        self, in_channels=1, out_channels=2, enc_feat=[16, 32, 32, 32], dec_feat=[32, 32, 32, 16], bnorm=True, dropout=True,
    ):
        """ 
        Parameters:
            in_channels: channels of the input
            enc_feat: List of encoder filters. e.g. [16, 32, 32, 32]
            dec_feat: List of decoder filters. e.g. [32, 32, 32, 16]
            bnorm: bool. Perform batch-normalization?
            dropout: bool. Perform dropout?
        """
        super().__init__()

        # configure backbone
        self.backbone = Backbone(
            enc_feat,
            dec_feat,
            in_channels=in_channels,
            dropout=dropout,
            bnorm=bnorm,
        )

        # configure flow prediction and integration
        self.flow = FlowPredictor(in_channels=self.backbone.output_channels[-1], out_channels=out_channels)

    def forward(self, x):
        """
        Feed a pair of images through the network, predict a transformation
        
        Parameters:
            source: the moving image
            target: the target image
        
        Return:
            the flow
        """

        # feed through network
        dec_activations = self.backbone(x)
        x = dec_activations[-1]

        # predict flow 
        flow = self.flow(x)

        return flow



class Backbone(nn.Module):
    """ 
    U-net backbone for registration models.
    """

    def __init__(self, enc_feat, dec_feat, in_channels=1, bnorm=False, dropout=True, skip_connections=True):
        """
        Parameters:
            enc_feat: List of encoder features. e.g. [16, 32, 32, 32]
            dec_feat: List of decoder features. e.g. [32, 32, 32, 16]
            in_channels: input channels, eg 1 for a single greyscale image. Default 1.
            bnorm: bool. Perform batch-normalization?
            dropout: bool. Perform dropout?
            skip_connections: bool, Set for U-net like skip cnnections
        """
        super().__init__()

        self.upsample = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)
        self.skip_connections = skip_connections

        # configure encoder (down-sampling path)
        prev_feat = in_channels
        self.encoder = nn.ModuleList()
        for feat in enc_feat:
            self.encoder.append(
                Stage(prev_feat, feat, stride=2, dropout=dropout, bnorm=bnorm)
            )
            prev_feat = feat
            
        if self.skip_connections:
            # pre-calculate decoder sizes and channels
            enc_stages = len(enc_feat)
            dec_stages = len(dec_feat)
            enc_history = list(reversed([in_channels] + enc_feat))
            decoder_out_channels = [
                enc_history[i + 1] + dec_feat[i] for i in range(dec_stages)
            ]
            decoder_in_channels = [enc_history[0]] + decoder_out_channels[:-1]

        else:
            # pre-calculate decoder sizes and channels
            decoder_out_channels = dec_feat
            decoder_in_channels = enc_feat[-1:] + decoder_out_channels[:-1]
            
        # pre-calculate return sizes and channels
        self.output_length = len(dec_feat) + 1
        self.output_channels = [enc_feat[-1]] + decoder_out_channels

        # configure decoder (up-sampling path)
        self.decoder = nn.ModuleList()
        
        for i, feat in enumerate(dec_feat):
            self.decoder.append(
                Stage(
                    decoder_in_channels[i], feat, stride=1, dropout=dropout, bnorm=False
                )
            )

    def forward(self, x):
        """
        Feed x throught the U-Net
        
        Parameters:
            x: the input
        
        Return:
            list of decoder activations, from coarse to fine. Last index is the full resolution output.
        """
        # pass through encoder, save activations
        x_enc = [x]
        for layer in self.encoder:
            x_enc.append(layer(x_enc[-1]))

        # pass through decoder
        x = x_enc.pop()
        x_dec = [x]
        for layer in self.decoder:
            x = layer(x)
            x = self.upsample(x)
            if self.skip_connections:
                x = torch.cat([x, x_enc.pop()], dim=1)
            x_dec.append(x)

        return x_dec


class Stage(nn.Module):
    """
    Specific U-net stage
    """

    def __init__(self, in_channels, out_channels, stride=1, bnorm=True, dropout=True):
        super().__init__()

        if stride == 1:
            ksize = 3
        elif stride == 2:
            ksize = 4
        else:
            raise ValueError("stride must be 1 or 2")

        # build stage
        layers = []
        if bnorm:
            layers.append(nn.BatchNorm3d(in_channels))
        layers.append(nn.Conv3d(in_channels, out_channels, ksize, stride, 1))
        layers.append(nn.LeakyReLU(0.2))
        layers.append(nn.Conv3d(out_channels, out_channels, 3, 1, 1))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout3d())

        self.stage = nn.Sequential(*layers)

    def forward(self, x):
        return self.stage(x)


class FlowPredictor(nn.Module):
    """
    A layer intended for flow prediction. Initialied with small weights for faster training.
    """

    def __init__(self, in_channels, out_channels=3):
        super().__init__()
        """
        instantiates the flow prediction layer.
        
        Parameters:
            in_channels: input channels
        """
        ndims = out_channels#settings.get_ndims()
        # configure cnn
        self.cnn = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv3d(in_channels, ndims, kernel_size=3, padding=1),
        )

        # init final cnn layer with small weights and bias
        self.cnn[-1].weight = nn.Parameter(
            Normal(0, 1e-5).sample(self.cnn[-1].weight.shape)
        )
        self.cnn[-1].bias = nn.Parameter(torch.zeros(self.cnn[-1].bias.shape))

    def forward(self, x):
        """
        predicts the transformation. 
        
        Parameters:
            x: the input
            
        Return:
            pos_flow, neg_flow: the positive and negative flow
        """
        # predict the flow
        return self.cnn(x)
    


class ICNet(nn.Module):
    '''implicit correspondence network'''
    def __init__(self, config):
        super(ICNet, self).__init__()
        self.config = config
        self.img_enc = ImageEncoder(config)
        self.adapter = Adapter(config)
        self.grid_transformer = GridTrasformer(config)

    def forward(self, x, grid):
        '''
        grid --> [batch, 3, h, w, z]
        '''
        enc_feature = self.img_enc(x)
        adapted_feature = self.adapter(enc_feature, grid)
        gdf = self.grid_transformer(adapted_feature)#.transpose(2,1)  # [b, c, N], N=(h*w*z)
        return gdf.reshape(grid.shape)  # Grid displacement field

class ImageEncoder(nn.Module):
    def __init__(self, config):
        super(ImageEncoder, self).__init__()
        self.input_shape = config.input_shape

        nc = [8*(2**i) for i in range(5)]#[16,32,32,32]#
        self.downsample_block0 = layers.DownsampleBlock(inc=2, outc=nc[0])
        self.downsample_block1 = layers.DownsampleBlock(inc=nc[0], outc=nc[1], down=False)
        self.downsample_block2 = layers.DownsampleBlock(inc=nc[1], outc=nc[2], down=False)
        self.downsample_block3 = layers.DownsampleBlock(inc=nc[2], outc=nc[3], down=False)
        self.downsample_block4 = layers.DownsampleBlock(inc=nc[3], outc=nc[4], down=False)
        self.adpt_pool = nn.AdaptiveAvgPool3d((10,10,10))
        #self.fclayer = nn.Sequential(nn.Linear(6144, 2048), nn.Tanh())
       
    def forward(self, x):
        f_down0, _ = self.downsample_block0(x)
        f_down1, _ = self.downsample_block1(f_down0)
        f_down2, _ = self.downsample_block2(f_down1)
        f_down3, _ = self.downsample_block3(f_down2)
        f_down4, _ = self.downsample_block4(f_down3)
        out = self.adpt_pool(f_down4)
        out=out.reshape([out.shape[0], out.shape[1], -1])
        #out = self.fclayer(f_down4.reshape([f_down4.shape[0], -1]))
        return  out # squeeze but preserve the batch dimension.






class Adapter(nn.Module):
    '''a network module to adapte 3d tensors to 1d tensors '''
    def __init__(self, config):
        super(Adapter, self).__init__()
        self.grid_size = config.grid_size
        self.h, self.w, self.z = self.grid_size

    def forward(self, enc_out, grid):
        '''
        enc_out --> [b, L] --> [b, c, L]
        grid    --> [b, 3, h, w, z] --> [b, 3, h*w*z]n--> [b, h*w*z, 3]
        '''
       
        #enc_out = torch.stack([enc_out]*self.h*self.w*self.z, dim=1) #[batch, grid_number, feature_len]
        enc_out = enc_out.permute(0, 2, 1) # [batch, grid_number, feature_len]
        grid = torch.reshape(grid, [grid.shape[0], grid.shape[1], -1])
        grid = torch.transpose(grid, 2, 1) # [batch, grid_number, 3]
        grid_feats = torch.transpose(torch.cat([enc_out, grid], dim=2), 1, 2)
        return grid_feats # [batch, feature_len+3, grid_number]



class GridTrasformer(nn.Module):
    '''transform the grid via image feature'''
    def __init__(self, config):
        super(GridTrasformer, self).__init__()
        #nc = [240*8, 512, 256, 128, 64, 3]
        nc = [128, 64, 32,16, 3]
        self.conv1 = nn.Conv1d(in_channels=nc[0]+3, out_channels=nc[1], kernel_size=1)
        self.actv1 = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.conv2 = nn.Conv1d(in_channels=nc[1], out_channels=nc[2], kernel_size=1)
        self.actv2 = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.conv3 = nn.Conv1d(in_channels=nc[2], out_channels=nc[3], kernel_size=1)
        self.actv3 = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.conv4 = nn.Conv1d(in_channels=nc[3], out_channels=nc[4], kernel_size=1)
        # self.actv4 = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # self.conv5 = nn.Conv1d(in_channels=nc[4], out_channels=nc[5], kernel_size=1)
        self.actv4 = nn.Tanh()

    def forward(self, x):
        '''
        x --> [b, 1027, 1000]
        out --> [b,3,1000] 
        '''
        x = self.actv1(self.conv1(x))
        x = self.actv2(self.conv2(x))
        x = self.actv3(self.conv3(x))
        x = self.actv4(self.conv4(x))
        # x = self.actv5(self.conv5(x))
        return x#torch.transpose(x, 1, 2) 
    


class AffineTransform(nn.Module):
    """
    3-D Affine Transformer
    """

    def __init__(self, mode='bilinear'):
        super().__init__()
        self.mode = mode

    def forward(self, src, affine):

        mat = affine#torch.bmm(shear_mat, torch.bmm(scale_mat, torch.bmm(rot_mat_z, torch.matmul(rot_mat_y, rot_mat_x))))
        inv_mat = mat#torch.inverse(mat)
        grid = nnf.affine_grid(mat, src.size(), align_corners=True)
        #inv_grid = nnf.affine_grid(inv_mat, [src.shape[0], 3, src.shape[2], src.shape[3], src.shape[4]], align_corners=True)
        return nnf.grid_sample(src, grid, align_corners=True, mode=self.mode), mat, inv_mat





