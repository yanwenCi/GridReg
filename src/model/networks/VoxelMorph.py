import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions.normal import Normal
import numpy as np


class Voxelmorph(nn.Module):
    """
    VoxelMorph network for (unsupervised) nonlinear registration between two images.
    Slightly modified implementation.
    """

    def __init__(
        self, in_channels=1, enc_feat=[16, 32, 32, 32], dec_feat=[32, 32, 32, 16], bnorm=True, dropout=True,
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
            in_channels=2 * in_channels,
            dropout=dropout,
            bnorm=bnorm,
        )

        # configure flow prediction and integration
        self.flow1 = FlowPredictor(in_channels=self.backbone.output_channels[-1],)
        # self.flow2 = FlowPredictor(in_channels=self.backbone.output_channels[-2],)
        # self.flow3 = FlowPredictor(in_channels=self.backbone.output_channels[-3],)
        # self.upsamp = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)
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
        x1 = dec_activations[-1]
        # x2 = dec_activations[-2]
        # x3 = dec_activations[-3]
        # x4 = dec_activations[-4]

        # predict flow 
        flow1 = self.flow1(x1)
        # flow2 = self.upsamp(self.flow2(x2))
        # flow3 = self.upsamp(self.upsamp(self.flow3(x3)))

        return flow1 #(flow1+flow2+flow3, flow2+flow3, flow3)




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
        
        # for param in self.parameters():
        #     param.requires_grad = False

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