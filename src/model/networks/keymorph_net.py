import torch.nn as nn
import src.model.networks.keymorph_layers as layers
import src.model.layers as layers1


#h_dims = [32, 64, 128, 256, 512]

class ConvNet(nn.Module):
    def __init__(self, dim, input_ch, out_ch, norm_type, num_layers=8):
        super(ConvNet, self).__init__()
        self.dim = dim
        h_dims = [32, 64, 64, 128, 128, 256, 256, 512]
        h_dims = [input_ch] + h_dims +[out_ch]
        self.model=[]
        assert len(h_dims)>num_layers-1
        for i in range(num_layers):
            if i>4:
                self.model.append(layers.ConvBlock(h_dims[i], h_dims[i+1], 1, norm_type, False, dim))
            else:
                self.model.append(layers.ConvBlock(h_dims[i], h_dims[i+1], 1, norm_type, True, dim))
        # self.model.append(layers.ConvBlock(h_dims[i+1], h_dims[-1], 1, norm_type, True, dim))
        self.model = nn.ModuleList(self.model)
    def forward(self, x):
        out = []
        for layer in self.model:
            x = layer(x)
            out.append(x)
        return out[-1]
    

class ImageEncoder(nn.Module):
    def __init__(self, in_nc,):
        super(ImageEncoder, self).__init__()
        base_nc=16
        nc = [base_nc*(2**i) for i in range(6)]#[16,32,32,32]#
        self.downsample_block0 = layers1.DownsampleBlock(inc=in_nc, outc=nc[0])
        self.downsample_block1 = layers1.DownsampleBlock(inc=nc[0], outc=nc[1])
        self.downsample_block2 = layers1.DownsampleBlock(inc=nc[1], outc=nc[2])
        self.downsample_block3 = layers1.DownsampleBlock(inc=nc[2], outc=nc[3])
        self.downsample_block4 = layers1.DownsampleBlock(inc=nc[3], outc=nc[4])
        self.downsample_block5 = layers1.DownsampleBlock(inc=nc[4], outc=nc[5])
        #self.adpt_pool = nn.AdaptiveAvgPool3d((10,10,10))
        #self.fclayer = nn.Sequential(nn.Linear(6144, 2048), nn.Tanh())
       
    def forward(self, x):
        f_down0, _ = self.downsample_block0(x)
        f_down1, _ = self.downsample_block1(f_down0)
        f_down2, _ = self.downsample_block2(f_down1)
        f_down3, _ = self.downsample_block3(f_down2)
        f_down4, _ = self.downsample_block4(f_down3)
        f_down5, _ = self.downsample_block5(f_down4)
        #out = self.fclayer(f_down4.reshape([f_down4.shape[0], -1]))
        return  f_down5 # squeeze but preserve the batch dimension.

