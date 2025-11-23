
import math
import torch
import torch.nn as nn
import torch.nn.functional as nnf
import src.model.layers as layers
from src.model.networks.VoxelMorph import Stage
import torchvision.models as models
from src.model.networks import keymorph_layers as klayers
from src.model.networks.keypoint_aligners import TPS, AffineKeypointAligner
import numpy as np
import src.model.networks.keymorph as keymorph
from src.model.functions import warp3d_v2, apply_rigid_transform_3D
from src.model.networks import transforms, utils

class ICNet(nn.Module):
    '''implicit correspondence network'''
    def __init__(self, config):
        super(ICNet, self).__init__()
        self.config = config
        self.bspl = None
        if hasattr(config, 'ushape'):
            self.ushape = config.ushape
            # self.pre_reg = keymorph.KeyMorph(dim=3, backbone='conv', num_keypoints=64, num_layers=9)
            if config.ushape:
                #self.img_enc = ImageEncoderUshape(config) 
                self.img_enc = ConvNetUshape(3, config.in_nc, 'instance', config.num_layers, config.nc_initial)
                if 'bspl' in config.exp_name:
                    interp_type = 'bspline'
                    print('Using bspline')
                    self.bspl = utils.GridTransform(grid_size=None, interp_type=interp_type, batch_size=config.batch_size,
                                        device='cuda' if torch.cuda.is_available() else 'cpu', volsize=config.input_shape)
                
                elif 'tconv' in config.exp_name:
                    interp_type = 't_conv'
                    print('Using t_conv')
                    self.bspl = transforms.bspline_upsample_tconv
                    # self.bspl = utils.GridTransform(grid_size=grid_size, interp_type=interp_type, batch_size=config.batch_size,
                                        # device='cuda' if torch.cuda.is_available() else 'cpu', volsize=config.input_shape)
                else:
                    self.grid_transformer = GridTrasformerUshape2(config)
                print('Using U shape network')

        

    def forward(self, x, grid):
        '''
        grid --> [batch, 3, h, w, z]
        '''
        enc_feature = self.img_enc(x)#output shape [b, c, N]
        gdf = self.grid_transformer(enc_feature, grid)#.transpose(2,1)  # [b, c, N], N=(h*w*z)
        return gdf


class ConvNetUshape(nn.Module):
    def __init__(self, dim, input_ch, norm_type, num_layers, nc_initial):
        super(ConvNetUshape, self).__init__()
        self.dim = dim
        # h_dims = [input_ch, 32, 64, 64, 128, 128, 256, 256, 512, 512]
        h_dims = [input_ch, nc_initial, nc_initial*2, nc_initial*2, nc_initial*4, nc_initial*4, \
                  nc_initial*8, nc_initial*8, nc_initial*16, nc_initial*16]
        # h_dims = [input_ch] + [nc_initial*(2**i) for i in range(num_layers)] + [nc_initial*(2**num_layers)]
        self.model=[]
        assert len(h_dims)>num_layers
        for i in range(num_layers):
            if i>2:
                self.model.append(layers.ConvBlock(h_dims[i], h_dims[i+1], 1, norm_type, False, dim))
            else:
                self.model.append(layers.ConvBlock(h_dims[i], h_dims[i+1], 1, norm_type, True, dim))
        self.model = nn.ModuleList(self.model)
    def forward(self, x):
        out = []
        for layer in self.model:
            x = layer(x)
            out.append(x)
        
        return out[-1], out[3], out[2], out[1]


 
class ImageEncoderUshape2(nn.Module):
    def __init__(self, config):
        super(ImageEncoderUshape2, self).__init__()
        self.input_shape = config.input_shape
        
        nc = [config.nc_initial*(2**i) for i in range(6)]#[16,32,32,32]#
        nc = [16, 32, 32, 32, 32]
        self.downsample_block0 = layers.DownsampleBlock(inc=config.in_nc, outc=nc[0])
        self.downsample_block1 = layers.DownsampleBlock(inc=nc[0], outc=nc[1])
        self.downsample_block2 = layers.DownsampleBlock(inc=nc[1], outc=nc[2])
        self.downsample_block3 = layers.DownsampleBlock(inc=nc[2], outc=nc[3])
        self.downsample_block4 = layers.DownsampleBlock(inc=nc[3], outc=nc[4])#, down=False)
        # self.downsample_block5 = layers.DownsampleBlock(inc=nc[4], outc=nc[5])
        self.adpt_pool = nn.AdaptiveAvgPool3d((2,2,2))
        #self.fclayer = nn.Sequential(nn.Linear(6144, 2048), nn.Tanh())
       
    def forward(self, x):
        f_down0, _ = self.downsample_block0(x)
        f_down1, _ = self.downsample_block1(f_down0)
        f_down2, _ = self.downsample_block2(f_down1)
        f_down3, _ = self.downsample_block3(f_down2)
        f_down4, _ = self.downsample_block4(f_down3)
        f_down4, f_down3, f_down2, f_down1 = self.adpt_pool(f_down4), \
            self.adpt_pool(f_down3), self.adpt_pool(f_down2), self.adpt_pool(f_down1)
        #print(f_down4.shape, f_down3.shape, f_down2.shape, f_down1.shape)
        return  f_down4, f_down3, f_down2, f_down1 # squeeze but preserve the batch dimension.



class ImageEncoderAffine(nn.Module):
    def __init__(self, config):
        super(ImageEncoderAffine, self).__init__()
        self.input_shape = config.input_shape
        #nc = [16, 32, 32, 64]
        nc = [8, 16, 32, 32]
        pre_nc = 2
        self.encoder = nn.ModuleList()
        for i in range(len(nc)):
            self.encoder.append(Stage(in_channels=pre_nc, out_channels=nc[i], stride=2, dropout=True, bnorm=True))
            pre_nc = nc[i]   
        self.encoder.append(nn.Flatten(),
                            nn.Linear(128*4*4*3, 1024),
                            nn.ReLU(),
                            nn.Linear(1024, 6),
        )

    def forward(self, x):
        for layer in self.encoder:
            x = layer(x)
        return x  # squeeze but preserve the batch dimension.



class Adapter(nn.Module):
    '''a network module to adapte 3d tensors to 1d tensors '''
    def __init__(self, config):
        super(Adapter, self).__init__()
        
    def forward(self, enc_out, grid):
        '''
        enc_out --> [b, L] --> [b, c, L]
        grid    --> [b, 3, h, w, z] --> [b, 3, h*w*z]n--> [b, h*w*z, 3]
        '''
        enc_out=enc_out.reshape([enc_out.shape[0], enc_out.shape[1], -1])
        #enc_out = torch.stack([enc_out]*self.h*self.w*self.z, dim=1) #[batch, grid_number, feature_len]
        enc_out = enc_out.permute(0, 2, 1) # [batch, grid_number, feature_len]
        grid = torch.reshape(grid, [grid.shape[0], grid.shape[1], -1])
        grid = torch.transpose(grid, 2, 1) # [batch, grid_number, 3]
        grid_feats = torch.transpose(torch.cat([enc_out, grid], dim=2), 1, 2)
        return grid_feats # [batch, feature_len, grid_number]

class Adapter2(nn.Module):
    def __init__(self, config):
        super(Adapter2, self).__init__()
        self.h, self.w, self.z = config.grid_size
        self.expand = nn.Linear(1, self.h*self.w*self.z)
        
    def forward(self, enc_out, grid):
        '''
        enc_out --> [b, L] --> [b, c, L]
        grid    --> [b, 3, h, w, z] --> [b, 3, h*w*z]n--> [b, h*w*z, 3]
        '''
        enc_out=enc_out.reshape([enc_out.shape[0], -1])
  
        enc_out = torch.stack([enc_out]*self.h*self.w*self.z, dim=1) #[batch, grid_number, feature_len]
        grid = torch.reshape(grid, [grid.shape[0], grid.shape[1], -1])
        grid = torch.transpose(grid, 2, 1)
        grid_feats = torch.transpose(torch.cat([enc_out, grid], dim=2), 1, 2)
        return grid_feats # [batch, feature_len+3, grid_number]


class GridTrasformer(nn.Module):
    '''transform the grid via image feature'''
    def __init__(self, config):
        super(GridTrasformer, self).__init__()
        nc = [config.nc_initial*(2**4), 512, 256, 128, 64, 3]
        self.conv1 = nn.Conv1d(in_channels=nc[0]+3, out_channels=nc[1], kernel_size=1)
        self.actv1 = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.conv2 = nn.Conv1d(in_channels=nc[1], out_channels=nc[2], kernel_size=1)
        self.actv2 = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.conv3 = nn.Conv1d(in_channels=nc[2], out_channels=nc[3], kernel_size=1)
        self.actv3 = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.conv4 = nn.Conv1d(in_channels=nc[3], out_channels=nc[4], kernel_size=1)
        self.actv4 = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.conv5 = nn.Conv1d(in_channels=nc[4], out_channels=nc[5], kernel_size=1)
        self.actv5 = nn.Tanh()


    def forward(self, x):
        '''
        x --> [b, 1027, 1000]
        out --> [b,3,1000] 
        '''
        x = self.actv1(self.conv1(x))
        x = self.actv2(self.conv2(x))
        x = self.actv3(self.conv3(x))
        x = self.actv4(self.conv4(x))
        x = self.actv5(self.conv5(x))
        return x#torch.transpose(x, 1, 2) 
    
class GaussianProjection3D(nn.Module):
    def __init__(self, in_ch, out_ch, grid_size):
        super().__init__()
        self.g = grid_size
        self.param = nn.Conv3d(in_ch, 6, kernel_size=1) # predicts μ(3), logσ(3) at each voxel
        self.out   = nn.Conv3d(in_ch, out_ch, kernel_size=1)

    def forward(self, feat):                   # [B,C,D,H,W]
        B,C,D,H,W = feat.shape
        pars = self.param(feat)                # [B,6,D,H,W]
        mu = torch.tanh(pars[:,0:3])           # [-1,1] normalized
        sig= torch.exp(pars[:,3:6]) + 1e-3

        # precompute normalized coords for the full map
        xs = torch.linspace(-1,1, D, device=feat.device)[:,None,None]
        ys = torch.linspace(-1,1, H, device=feat.device)[None,:,None]
        zs = torch.linspace(-1,1, W, device=feat.device)[None,None,:]

        # grid cell centers in [-1,1]
        gx = torch.linspace(-1,1,self.g[0], device=feat.device)
        gy = torch.linspace(-1,1,self.g[1], device=feat.device)
        gz = torch.linspace(-1,1,self.g[2], device=feat.device)

        out = self.out(feat)                   # [B,out_ch,D,H,W]
        out = out.view(B, -1, D,H,W)

        pooled = []
        for cx in gx:
            for cy in gy:
                for cz in gz:
                    # Gaussian kernel per voxel: exp(-((x-μx)^2/σx^2+...))
                    kx = torch.exp(-((xs - cx - mu[:,0:1])**2)/(2*sig[:,0:1]**2))
                    ky = torch.exp(-((ys - cy - mu[:,1:2])**2)/(2*sig[:,1:2]**2))
                    kz = torch.exp(-((zs - cz - mu[:,2:3])**2)/(2*sig[:,2:3]**2))
                    w  = (kx*ky*kz)                           # [B,1,D,H,W]
                    w  = w / (w.sum(dim=(2,3,4), keepdim=True) + 1e-6)
                    pooled.append((out * w).sum(dim=(2,3,4))) # [B,out_ch]
        P = torch.stack(pooled, dim=2)        # [B,out_ch,G]
        return P




class AttentionProjection3D(nn.Module):
    """
    Cross-attention from a 3D feature map [B,Cin,D,H,W] to a runtime grid (gw,gh,gd).
    Grid cells are queries (sinusoidal PE → q), feature voxels are keys/values (1x1x1 conv).
    Returns [B, out_channels, G] or [B,out_channels,gw,gh,gd] if reshape_out=True.
    """
    def __init__(self, in_channels, out_channels, *, heads=2, kv_channels=None,
                 pe_channels=32, dropout=0.1, reshape_out=False):
        super().__init__()
        self.h = heads
        self.dk = kv_channels or max(32, out_channels // heads)
        self.pe_channels = pe_channels
        self.reshape_out = reshape_out

        # K/V from feature map
        self.k_proj = nn.Conv3d(in_channels, self.h * self.dk, kernel_size=1, bias=False)
        self.v_proj = nn.Conv3d(in_channels, self.h * self.dk, kernel_size=1, bias=False)

        # Query from positional encodings
        self.q_proj = nn.Linear(self.pe_channels, self.h * self.dk, bias=False)

        # Output projection (concat heads → out_channels)
        self.out = nn.Linear(self.h * self.dk, out_channels, bias=True)

        self.drop = nn.Dropout(dropout)

        # cache PEs by (gw,gh,gd,device,dtype)
        self._pe_cache = {}

    @torch.no_grad()
    def _grid_pe(self, gw, gh, gd, device, dtype):
        """Sinusoidal PE over normalized (x,y,z) in [-1,1], shape [G, pe_channels]."""
        key = (gw, gh, gd, device, dtype)
        if key in self._pe_cache:
            return self._pe_cache[key]

        xs = torch.linspace(-1., 1., gw, device=device, dtype=dtype)
        ys = torch.linspace(-1., 1., gh, device=device, dtype=dtype)
        zs = torch.linspace(-1., 1., gd, device=device, dtype=dtype)
        X, Y, Z = torch.meshgrid(xs, ys, zs, indexing='ij')              # [gw,gh,gd]
        coords = torch.stack([X, Y, Z], dim=-1).reshape(-1, 3)           # [G,3]

        d = max(1, self.pe_channels // 6)                                # per sin/cos per axis
        freqs = torch.logspace(0, math.log10(1000.0), steps=d, device=device, dtype=dtype)
        feats = []
        for a in range(3):
            cx = coords[:, a:a+1] * freqs[None, :]                        # [G,d]
            feats += [torch.sin(cx), torch.cos(cx)]
        pe = torch.cat(feats, dim=1)                                      # [G, 6*d]
        if pe.shape[1] < self.pe_channels:
            pe = nnf.pad(pe, (0, self.pe_channels - pe.shape[1]))
        elif pe.shape[1] > self.pe_channels:
            pe = pe[:, :self.pe_channels]

        self._pe_cache[key] = pe
        return pe

    def forward(self, feat3d, grid_size):
        """
        feat3d:   [B, Cin, D, H, W]
        grid_size: tuple(int) = (gw, gh, gd)
        returns:  [B, out_channels, G] (or [B,out_channels,gw,gh,gd] if reshape_out=True)
        """
        B, Cin, D, H, W = feat3d.shape
        # print(feat3d.shape)
        gw, gh, gd = map(int, grid_size)
        G = gw * gh * gd

        # Queries from runtime grid PE
        N = D * H * W
        K = self.k_proj(feat3d).view(B, self.h, self.dk, N)  # [B,h,dk,N] batch heads per-head-dim N=D*H*W
        V = self.v_proj(feat3d).view(B, self.h, self.dk, N)  # [B,h,dk,N] 

        pe = self._grid_pe(gw, gh, gd, feat3d.device, feat3d.dtype)  # [G, pe] G=gw*gh*gd, pe=pe_channels
        Q = self.q_proj(pe).view(1, G, self.h, self.dk)              # [1,G,h,dk] 
        Q = Q.expand(B, G, self.h, self.dk).permute(0, 2, 1, 3)      # [B,h,G,dk]


        scale = 1.0 / math.sqrt(self.dk)
        attn = torch.einsum('bhgd,bhdn->bhgn', Q, K) * scale         # [B,h,G,N]
        attn = nnf.softmax(attn, dim=-1)
        attn = self.drop(attn)

        ctx = torch.einsum('bhgn,bhdn->bhgd', attn, V)               # [B,h,G,dk]
        ctx = ctx.reshape(B, G, self.h * self.dk)                    # [B,G,h*dk]
        out = self.out(ctx).transpose(1, 2)                          # [B,outc,G]
        if self.reshape_out:
            out = out.view(B, out.shape[1], gw, gh, gd)
        return out
                # [B, out_channels, G]
                            # to grid

class LiteLocalProjection3D(nn.Module):
    def __init__(self, in_ch, out_ch, grid_size, win=7, heads=2, dk=32):
        super().__init__()
        self.gw, self.gh, self.gd = grid_size
        self.h, self.dk, self.win = heads, dk, win
        # cheap pre-downsample
        self.down = nn.Sequential(
            nn.Conv3d(in_ch, in_ch, kernel_size=3, stride=2, padding=1, groups=in_ch), # DW
            nn.Conv3d(in_ch, max(32, in_ch//4), kernel_size=1), nn.ReLU(inplace=True)
        )
        in_ch2 = max(32, in_ch//4)
        self.k_proj = nn.Conv3d(in_ch2, heads*dk, kernel_size=1, bias=False)
        self.v_proj = nn.Conv3d(in_ch2, heads*dk, kernel_size=1, bias=False)
        # queries = fixed sinusoidal PE → no learnable q_proj
        self.out = nn.Linear(heads*dk, out_ch)

    def forward(self, feat):  # feat: [B,C,D,H,W]
        B, _, D, H, W = feat.shape
        f = self.down(feat)                     # [B,C',D',H',W']
        Dp, Hp, Wp = f.shape[-3:]
        K = self.k_proj(f).view(B, self.h, self.dk, Dp, Hp, Wp)
        V = self.v_proj(f).view(B, self.h, self.dk, Dp, Hp, Wp)

        # grid centers mapped to downsampled coords
        xs = torch.linspace(0, Dp-1, steps=self.gw, device=feat.device)
        ys = torch.linspace(0, Hp-1, steps=self.gh, device=feat.device)
        zs = torch.linspace(0, Wp-1, steps=self.gd, device=feat.device)
        X, Y, Z = torch.meshgrid(xs, ys, zs, indexing='ij')
        centers = torch.stack([X,Y,Z], dim=-1).view(-1,3)  # [G,3]
        half = self.win//2

        outs = []
        for c in centers:  # G loops; G≈1000 is fine with tiny windows
            x0,x1 = int(max(0, c[0]-half)), int(min(Dp, c[0]+half+1))
            y0,y1 = int(max(0, c[1]-half)), int(min(Hp, c[1]+half+1))
            z0,z1 = int(max(0, c[2]-half)), int(min(Wp, c[2]+half+1))
            Kc = K[:,:,:,x0:x1,y0:y1,z0:z1].reshape(B, self.h, self.dk, -1)   # [B,h,dk,Nw]
            Vc = V[:,:,:,x0:x1,y0:y1,z0:z1].reshape(B, self.h, self.dk, -1)
            # use a fixed query per head (unit vector) → eliminates q params
            Q = torch.zeros(B, self.h, 1, self.dk, device=feat.device); Q[...,0] = 1
            attn = torch.softmax((Q @ Kc).squeeze(2)/ (self.dk**0.5), dim=-1) # [B,h,Nw]
            ctx = (attn.unsqueeze(2) @ Vc.transpose(2,3)).squeeze(2)          # [B,h,dk]
            outs.append(ctx.flatten(1))                                        # [B,h*dk]
        out = torch.stack(outs, dim=1)                # [B,G,h*dk]
        return self.out(out).transpose(1,2)           # [B, out_ch, G]


class GridTrasformerUshape2(nn.Module):
    '''transform the grid via image feature'''
    def __init__(self, config):
        super(GridTrasformerUshape2, self).__init__()
        self.uncertainty = None#config.uncertainty
        # print('using uncertainty:', self.uncertainty)
        # nc = [512*8, 2048, 1024, 512, 128,  3]
        # nc = [config.nc_initial*(2**4), 512, 256, 128, 64, 3]
        # skip_nc = [config.nc_initial*(2**i) for i in range(5)]
        # nc = [32, 32, 32, 32, 16, 3]
        # skip_nc = [16, 32, 32, 32, 32]
        # skip_nc = [32, 64, 64, 128, 128, 256, 256, 512, 512]
        # skip_nc = [config.nc_initial*(2**i) for i in range(config.num_layers+1)]
        nc_initial = config.nc_initial
        skip_nc = [nc_initial, nc_initial*2, nc_initial*2, nc_initial*4, nc_initial*4, \
                  nc_initial*8, nc_initial*8, nc_initial*16, nc_initial*16]
        if self.uncertainty:
            nc = [skip_nc[config.num_layers-1], 3, 3, 3, 3, 6]
        else:
            nc = [skip_nc[config.num_layers-1], 3, 3, 3, 3, 3]
        if config.num_layers is not None:         
            #skip_nc = skip_nc[:config.num_layers]
            skip_nc = skip_nc[:5]
        proj_nc =128
        # projection_filter = GaussianProjection3D
        projection_filter = AttentionProjection3D
        # projection_filter = LiteLocalProjection3D

        self.projection1 = projection_filter(skip_nc[-2], proj_nc, )
        self.projection2 = projection_filter(skip_nc[-3], proj_nc, )
        self.projection3 = projection_filter(skip_nc[-4], proj_nc, )
        self.projection4 = projection_filter(skip_nc[-4], proj_nc, )
      

        self.conv1 = nn.Conv1d(in_channels=nc[0], out_channels=nc[1], kernel_size=1)
        self.actv1 = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.conv2 = nn.Conv1d(in_channels=nc[1]+proj_nc, out_channels=nc[2], kernel_size=1)
        self.actv2 = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.conv3 = nn.Conv1d(in_channels=nc[2]+proj_nc, out_channels=nc[3], kernel_size=1)
        self.actv3 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv4 = nn.Conv1d(in_channels=nc[3]+proj_nc, out_channels=nc[5], kernel_size=1)
        # self.actv4 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        
        # self.conv5 = nn.Conv1d(in_channels=nc[4], out_channels=nc[5], kernel_size=1)
        self.actv5 = nn.Tanh()
        self.musig = MuSigmaHead(in_ch=nc[3]+proj_nc)
        # self.conv4.weight = nn.Parameter(torch.normal(0, 0.01, self.conv4.weight.shape))
        # self.conv4.bias = nn.Parameter(torch.zeros(self.conv4.bias.shape))
        # self.adpt_pooling = nn.AdaptiveAvgPool3d(grid_size)
        # self.adapter = Adapter(config)
        

    
    def forward(self, xs, grid):
        '''
        x --> [b, 1027, 1000]
        out --> [b,3,1000] 
        '''
        grid_size = grid.shape[2:]
        adpt_pooling = nn.AdaptiveAvgPool3d(grid_size)
        x0 = adpt_pooling(xs[0]).reshape([xs[0].shape[0], xs[0].shape[1], -1])
        # x0 = self.adapter(x0, grid)
        # print('x0', x0.shape)
        x1 = self.projection1(xs[1], grid_size)
        # print('x1', x1.shape)
        x1 = x1.view(x1.shape[0], x1.shape[1], -1)
        x2 = self.projection2(xs[2], grid_size)
        # print('x2', x2.shape)
        x2=x2.view(x2.shape[0], x2.shape[1], -1)
        x3 = self.projection3(xs[3], grid_size)
        x3=x3.view(x3.shape[0], x3.shape[1], -1)
        x0 = self.actv1(self.conv1(x0))
        x1 = torch.cat([x0, x1], dim=1)
        x1 = self.actv2(self.conv2(x1))
        x2 = torch.cat([x1, x2], dim=1)
        x2 = self.actv3(self.conv3(x2))
        x3 = torch.cat([x2, x3], dim=1)
       
        
        if self.uncertainty:
            x_mean, x_std = self.musig(x3)
            # print(x.max(), x.min()) 
            return x_mean.reshape(grid.shape), x_std.reshape(grid.shape)#torch.transpose(x, 1, 2) 
        else:
            x = self.actv5(self.conv4(x3))
            return x.reshape(grid.shape)
    
    def pool_flatten(self, x):
        x = self.adpt_pooling(x).view(x.shape[0], x.shape[1], -1)
        # x = torch.stack([x]*1000, dim=2).reshapw([x.shape[0], x.shape[1], -1])
        return x
    
class MuSigmaHead(nn.Module):
    def __init__(self, in_ch, d_max=1.0,  # in voxels (or mm at CP grid)
                 sigma_min=0.000001, sigma_max=.01,  # same units as d_max
                 sigma_init=0.0):
        super().__init__()
        self.d_max = float(d_max)
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)

        # 1x1x1 convs to 3 channels (x,y,z) at control-point resolution
        self.mu_conv = nn.Conv1d(in_ch, 3, kernel_size=1, bias=True)
        self.lsigma_conv = nn.Conv1d(in_ch, 3, kernel_size=1, bias=True)

        # Init: μ near 0
        nn.init.zeros_(self.mu_conv.weight)
        nn.init.zeros_(self.mu_conv.bias)

        # Init: σ near sigma_init via inverse-softplus
        def inv_softplus(y):
            # numerically stable inverse of softplus
            return torch.log(torch.exp(torch.tensor(y)) - 1.0)
        with torch.no_grad():
            b = inv_softplus(sigma_init).item()
            nn.init.zeros_(self.lsigma_conv.weight)
            self.lsigma_conv.bias.fill_(b)

    def forward(self, x):
        # x: [B, C_in, gw, gh, gz] features at control-point grid
        mu_raw = self.mu_conv(x)                # [B, 3, gw, gh, gz]
        mu = torch.tanh(mu_raw)    # bound μ to [-d_max, d_max]

        lsig = torch.tanh(self.lsigma_conv(x))              # raw logits for σ
        sigma = nnf.softplus(lsig)                # positive
        sigma = torch.clamp(sigma, self.sigma_min, self.sigma_max)
        return mu, sigma
    
class GridTrasformerUshape(nn.Module):
    '''transform the grid via image feature'''
    def __init__(self, config):
        super(GridTrasformerUshape, self).__init__()
        # nc = [512*8, 2048, 1024, 512, 128,  3]
        # nc = [config.nc_initial*(2**4), 512, 256, 128, 64, 3]
        # skip_nc = [config.nc_initial*(2**i) for i in range(5)]
        # nc = [32, 32, 32, 32, 16, 3]
        # skip_nc = [16, 32, 32, 32, 32]
        # skip_nc = [32, 64, 64, 128, 128, 256, 256, 512, 512]
        # skip_nc = [config.nc_initial*(2**i) for i in range(config.num_layers+1)]
        nc_initial = config.nc_initial
        skip_nc = [nc_initial, nc_initial*2, nc_initial*2, nc_initial*4, nc_initial*4, \
                  nc_initial*8, nc_initial*8, nc_initial*16, nc_initial*16]
        nc = [skip_nc[config.num_layers-1], 512, 256, 128, 64, 3]
        if config.num_layers is not None:         
            #skip_nc = skip_nc[:config.num_layers]
            skip_nc = skip_nc[:5]
        self.conv1 = nn.Conv1d(in_channels=nc[0]+3, out_channels=nc[1], kernel_size=1)
        self.actv1 = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.conv2 = nn.Conv1d(in_channels=nc[1]+skip_nc[-2], out_channels=nc[2], kernel_size=1)
        self.actv2 = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.conv3 = nn.Conv1d(in_channels=nc[2]+skip_nc[-3], out_channels=nc[3], kernel_size=1)
        self.actv3 = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.conv4 = nn.Conv1d(in_channels=nc[3]+skip_nc[-4], out_channels=nc[5], kernel_size=1)
        # self.actv4 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        
        # self.conv5 = nn.Conv1d(in_channels=nc[4], out_channels=nc[5], kernel_size=1)
        self.actv5 = nn.Tanh()
        self.conv4.weight = nn.Parameter(torch.normal(0, 0.01, self.conv4.weight.shape))
        self.conv4.bias = nn.Parameter(torch.zeros(self.conv4.bias.shape))
        self.adpt_pooling = nn.AdaptiveAvgPool3d(config.grid_size)
        self.adapter = Adapter(config)
        

    
    def forward(self, xs, grid):
        '''
        x --> [b, 1027, 1000]
        out --> [b,3,1000] 
        '''
        x0 = self.adpt_pooling(xs[0]).reshape([xs[0].shape[0], xs[0].shape[1], -1])
        x0 = self.adapter(x0, grid)
        x1, x2, x3 = self.pool_flatten(xs[1]), self.pool_flatten(xs[2]), self.pool_flatten(xs[3])
        x0 = self.actv1(self.conv1(x0))
        x1 = torch.cat([x0, x1], dim=1)
        x1 = self.actv2(self.conv2(x1))
        x2 = torch.cat([x1, x2], dim=1)
        x2 = self.actv3(self.conv3(x2))
        x3 = torch.cat([x2, x3], dim=1)
        # x3 = self.actv4(self.conv4(x3))
        x = self.actv5(self.conv4(x3))
        # print(x.max(), x.min()) 
        return x#torch.transpose(x, 1, 2) 
    
    def pool_flatten(self, x):
        x = self.adpt_pooling(x).view(x.shape[0], x.shape[1], -1)
        # x = torch.stack([x]*1000, dim=2).reshapw([x.shape[0], x.shape[1], -1])
        return x
    


class COMTrasformerUshape(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.com = klayers.CenterOfMass3d()
        # nc = [config.nc_initial*(2**4), 512, 256, 128, 64, 3]
        # skip_nc = [config.nc_initial*(2**i) for i in range(5)]
        skip_nc = [32, 64, 64, 128, 128, 256, 256, 512, 512]
        nc = [skip_nc[config.num_layers-1], 512, 256, 128, 64, 3]
        if config.num_layers is not None:         
            skip_nc = skip_nc[:config.num_layers+1]
            # skip_nc = skip_nc[:5]
        # self.conv0 = nn.Conv3d(in_channels=nc[0], out_channels=64, kernel_size=1)
        # self.actv0 = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        
        self.conv1 = nn.Conv1d(in_channels=nc[0]*2+3, out_channels=nc[1], kernel_size=1)
        self.actv1 = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.conv2 = nn.Conv1d(in_channels=nc[1]+skip_nc[-2], out_channels=nc[2], kernel_size=1)
        self.actv2 = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.conv3 = nn.Conv1d(in_channels=nc[2]+skip_nc[-3], out_channels=nc[3], kernel_size=1)
        self.actv3 = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.conv4 = nn.Conv1d(in_channels=nc[3]+skip_nc[-4], out_channels=nc[-1], kernel_size=1)
        # self.actv4 = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # self.conv5 = nn.Conv1d(in_channels=nc[4], out_channels=nc[5], kernel_size=1)
        self.actv5 = nn.Tanh()

        self.weighted = nn.Conv1d(in_channels=6, out_channels=3, kernel_size=1)
        self.adpt_pool = nn.AdaptiveAvgPool3d((10, 10, 10))
        self.adapter = Adapter(config)
        self.aligner = AffineKeypointAligner(dim=3)
  
    def forward(self, xs, xs1, ref_grid):
        key0 = self.com(xs[0])
        key1 = self.com(xs1[0])
        ind_key = np.random.randint(0, key0.shape[1], 64)
        key0 = key0[:, ind_key]
        key1 = key1[:, ind_key]
      
        grid = self.aligner.grid_from_points(key0, key1, 
                                                [key0.shape[0], key0.shape[1]]+self.config.grid_size,
                                                 compute_on_subgrids=False if self.training else True,)
        
        x00 = self.pool_flatten(xs[0])
        x01 = self.pool_flatten(xs1[0])
        grid = grid.view(grid.shape[0], 3, -1)
        
        # x0 = self.adapter(x0, grid)
        x0 = torch.cat([x00, x01, grid], dim=1)
        x1, x2, x3 = [self.pool_flatten(xs[i]) for i in range(1, 4)]
        x0 = self.actv1(self.conv1(x0))      
        x1 = torch.cat([x0, x1], dim=1)
        x1 = self.actv2(self.conv2(x1))
        x2 = torch.cat([x1, x2], dim=1)
        x2 = self.actv3(self.conv3(x2))
        x3 = torch.cat([x2, x3], dim=1)
        # x3 = self.actv4(self.conv4(x3))
        x = self.actv5(self.conv4(x3))
        
        x = x+grid-ref_grid.view(ref_grid.shape[0], 3, -1)
        return x, grid

    def pool_flatten(self, x):
        x = self.adpt_pool(x)
        return x.reshape([x.shape[0], x.shape[1], -1])



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

