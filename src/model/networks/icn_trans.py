import torch.nn as nn
import torch
import src.model.layers as layers
from src.model.networks.TransMorph import TransEncoder, CONFIGS


class ICNet(nn.Module):
    '''implicit correspondence network'''
    def __init__(self, config):
        super(ICNet, self).__init__()
        self.config = config
        #self.img_enc = ImageEncoder2(config)
        self.img_enc = TransEncoder(CONFIGS['TransMorph'], pretrained=True)
        self.adapter = Adapter(config)
        self.grid_transformer = GridTrasformer(config)
        #self.grid_transformer = TransformerDecoder(num_layers=4, d_model=512, num_heads=8, d_ff=1024)

    def forward(self, x, grid):
        '''
        grid --> [batch, 3, h, w, z]
        '''
        enc_feature = self.img_enc(x)
        #print('icn21', enc_feature.shape)
        # enc_feature2 = self.img_enc(y)
        # enc_feature = torch.cat([enc_feature1, enc_feature2], dim=1)
        adapted_feature = self.adapter(enc_feature, grid)
        #print(adapted_feature.shape)
        gdf = self.grid_transformer(adapted_feature) # [b, c, N], N=(h*w*z)
        #gdf = self.grid_transformer(adapted_feature.permute(0,2,1)).transpose(2,1)# [b, c, N], N=(h*w*z)
        return gdf.reshape(grid.shape)  # Grid displacement field

class ImageEncoder2(nn.Module):
    def __init__(self, config):
        super(ImageEncoder2, self).__init__()
        self.input_shape = config.input_shape

        nc = [8*(2**i) for i in range(5)]
        self.downsample_block0 = layers.DownsampleBlock(inc=2, outc=nc[0])
        self.downsample_block1 = layers.DownsampleBlock(inc=nc[0], outc=nc[1])
        self.downsample_block2 = layers.DownsampleBlock(inc=nc[1], outc=nc[2])
        self.downsample_block3 = layers.DownsampleBlock(inc=nc[2], outc=nc[3])
        self.downsample_block4 = layers.DownsampleBlock(inc=nc[3], outc=nc[4])
        self.adpt_pool = nn.AdaptiveAvgPool3d((2, 2, 2))

    def forward(self, x):
        f_down0, _ = self.downsample_block0(x)
        f_down1, _ = self.downsample_block1(f_down0)
        f_down2, _ = self.downsample_block2(f_down1)
        f_down3, _ = self.downsample_block3(f_down2)
        f_down4, _ = self.downsample_block4(f_down3)
       
        out = self.adpt_pool(f_down4)
        #print('49', f_down4.shape, out.shape)
        return out.reshape([out.shape[0], -1])  # squeeze but preserve the batch dimension.



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
        enc_out = torch.stack([enc_out]*self.h*self.w*self.z, dim=-1) # b c L
        #print(enc_out.shape)
        grid = torch.reshape(grid, [grid.shape[0], grid.shape[1], -1]) # b 3 L
        #grid = torch.transpose(grid, 2, 1)

        return torch.cat([enc_out, grid], dim=1)#torch.transpose(torch.cat([enc_out, grid], dim=2), 1, 2)  # [batch, feature_len, number]


class GridTrasformer(nn.Module):
    '''transform the grid via image feature'''
    def __init__(self, config):
        super(GridTrasformer, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=6147, out_channels=512, kernel_size=1)
        self.actv1 = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.conv2 = nn.Conv1d(in_channels=512, out_channels=256, kernel_size=1)
        self.actv2 = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.conv3 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=1)
        self.actv3 = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.conv4 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=1)
        self.actv4 = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.conv5 = nn.Conv1d(in_channels=64, out_channels=3, kernel_size=1)
        self.actv5 = nn.Tanh()

    def forward(self, x):
        '''
        x --> [b, 1027, 1000]
        out --> [b, 1000, 3]
        '''
        x = self.actv1(self.conv1(x))
        x = self.actv2(self.conv2(x))
        x = self.actv3(self.conv3(x))
        x = self.actv4(self.conv4(x))
        x = self.actv5(self.conv5(x))

        return x #b,3,1000 #torch.transpose(x, 1, 2)



class TransformerDecoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff):
        super(TransformerDecoder, self).__init__()
        self.num_layers = num_layers
        self.linear0 = nn.Linear(1027,d_model)
        self.actv0 = nn.ReLU()
        self.layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff) for _ in range(num_layers)])
        self.linear = nn.Linear(d_model, 3)
        self.actv5 = nn.Tanh()  
    def forward(self, x):
        x = self.actv0(self.linear0(x))
        for i in range(self.num_layers):
            x = self.layers[i](x)
        x=self.linear(x)
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, num_heads)
        self.feedforward = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        attended = self.multihead_attn(x, x, x)[0]
        x = x + self.dropout(attended)
        x = self.norm1(x)
        
        out = self.feedforward(x)
        x = x + self.dropout(out)
        x = self.norm2(x)
        return x


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
      
    def forward(self, x):
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.linear2(x)
        return x# class TransformerLayer(nn.Module):
#     def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
#         super(TransformerLayer, self).__init__()
#         self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
#         self.linear1 = nn.Linear(d_model, dim_feedforward)
#         self.dropout = nn.Dropout(dropout)
#         self.linear2 = nn.Linear(dim_feedforward, d_model)
#         self.norm1 = nn.LayerNorm(d_model)
#         self.norm2 = nn.LayerNorm(d_model)
#         self.dropout1 = nn.Dropout(dropout)
#         self.dropout2 = nn.Dropout(dropout)
#
#     def forward(self, tgt, memory):
#         tgt2 = self.norm1(tgt + self.dropout1(self.self_attn(tgt, tgt, tgt)[0]))
#         tgt3 = self.norm2(tgt2 + self.dropout2(self.self_attn(tgt2, memory, memory)[0]))
#         ff_output = self.linear2(self.dropout(torch.relu(self.linear1(tgt3))))
#         tgt4 = tgt3 + self.dropout(ff_output)
#
#         return tgt4

