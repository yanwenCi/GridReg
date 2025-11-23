import torch.nn as nn
import torch
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

class UGridNet(nn.Module):
    '''implicit correspondence network'''
    def __init__(self, config):
        super(UGridNet, self).__init__()
        self.config = config
        if hasattr(config, 'ushape'):
            self.ushape = config.ushape
            # self.pre_reg = keymorph.KeyMorph(dim=3, backbone='conv', num_keypoints=64, num_layers=9)
            if config.ushape:
                #self.img_enc = ImageEncoderUshape(config) 
                self.img_enc = ConvNetUshape(3, config.in_nc, 'instance', config.num_layers, config.nc_initial)
                if 'bspl' in config.exp_name:
                    interp_type = 'bspline'
                    print('Using bspline')
                    self.bspl = utils.GridTransform(grid_size=config.grid_size, interp_type=interp_type, batch_size=config.batch_size,
                                        device='cuda' if torch.cuda.is_available() else 'cpu', volsize=config.input_shape)
                
                elif 'tconv' in config.exp_name:
                    interp_type = 't_conv'
                    print('Using t_conv')
                    self.bspl = utils.GridTransform(grid_size=config.grid_size, interp_type=interp_type, batch_size=config.batch_size,
                                        device='cuda' if torch.cuda.is_available() else 'cpu', volsize=config.input_shape)
                if hasattr(config, 'COM') and config.COM:
                    self.COM = True
                    self.grid_transformer = COMTrasformerUshape(config)
                else:
                    
                    self.grid_transformer = GridTrasformerUshape(config)
                print('Using U shape network')
            else: 
                # self.img_enc = ImageEncoder(config)
                self.img_enc = ConvNet(3, config.in_nc, 'instance', config.num_layers, config.nc_initial)
                self.adapter = Adapter(config)
                self.grid_transformer = GridTrasformer(config)
        else:
            # self.img_enc = ImageEncoder(config)
            self.img_enc = ConvNet(3, config.in_nc, 'instance', config.num_layers, config.nc_initial)
            self.adapter = Adapter(config)
            self.grid_transformer = GridTrasformer(config)
            # self.bspl = transforms.CubicBSplineFFDTransform(ndim=3, img_size=[128,128,102], svf=True)
            
        # 

    def forward(self, x, grid):
        '''
        grid --> [batch, 3, h, w, z]
        '''
        # pre_reg_ddf = self.pre_reg(x[:,0:1,...].clone(), x[:,1:2,...].clone())
        # pre_reg_mov = warp3d_v2(x[:,0:1,...], pre_reg_ddf)
        # xx = torch.cat([x[:,0:1,...], pre_reg_mov], dim=1).clone()
        if hasattr(self, 'ushape') and self.ushape:
            if hasattr(self, 'COM') and self.COM:
                enc_out = self.img_enc(x[:,0:1,...])
                enc_out1 = self.img_enc(x[:,1:2,...])
                gdf, key = self.grid_transformer(enc_out, enc_out1, grid)
                gdf = gdf.reshape(grid.shape)
                return gdf, key
            else:
                enc_feature = self.img_enc(x)
                gdf = self.grid_transformer(enc_feature, grid)#.transpose(2,1)  # [b, c, N], N=(h*w*z)
                return gdf.reshape(grid.shape)#, pre_reg_mov, pre_reg_ddf # Grid displacement field
        else:
            enc_feature = self.img_enc(x)#output shape [b, c, N]
            adapted_feature = self.adapter(enc_feature, grid)
            gdf = self.grid_transformer(adapted_feature)
            gdf = gdf.reshape(grid.shape)  # Grid displacement field
            # gdf = self.bspl.compute_flow(gdf)
            return gdf
    
    # def get_grid(self, ddf, grid=None):
    #     ''' get the grid from the network'''    
    #     new_grid = grid + ddf  # [batch, 3, x, y, z]
    #     new_grid = new_grid.permute(0, 2, 3, 4, 1)
    #     new_grid = new_grid[..., [2, 1, 0]]
    #     return new_grid

class ImageEncoder(nn.Module):
    def __init__(self, config):
        super(ImageEncoder, self).__init__()
        self.input_shape = config.input_shape
        
        nc = [config.nc_initial*(2**i) for i in range(5)]#[16,32,32,32]#
        self.downsample_block0 = layers.DownsampleBlock(inc=2, outc=nc[0])
        self.downsample_block1 = layers.DownsampleBlock(inc=nc[0], outc=nc[1])
        self.downsample_block2 = layers.DownsampleBlock(inc=nc[1], outc=nc[2])
        self.downsample_block3 = layers.DownsampleBlock(inc=nc[2], outc=nc[3])
        self.downsample_block4 = layers.DownsampleBlock(inc=nc[3], outc=nc[4])
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
            if i>3:
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


class ConvNet(nn.Module):
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
            if i>3:
                self.model.append(layers.ConvBlock(h_dims[i], h_dims[i+1], 1, norm_type, False, dim))
            else:
                self.model.append(layers.ConvBlock(h_dims[i], h_dims[i+1], 1, norm_type, True, dim))
        self.model = nn.ModuleList(self.model)
    def forward(self, x):
        out = []
        for layer in self.model:
            x = layer(x)
            out.append(x)
        return out[-1]
     
class ImageEncoderUshape(nn.Module):
    def __init__(self, config):
        super(ImageEncoderUshape, self).__init__()
        self.input_shape = config.input_shape
        
        nc = [config.nc_initial*(2**i) for i in range(6)]#[16,32,32,32]#
        nc = [16, 32, 32, 32, 32]
        self.downsample_block0 = layers.DownsampleBlock(inc=config.in_nc, outc=nc[0])
        self.downsample_block1 = layers.DownsampleBlock(inc=nc[0], outc=nc[1])
        self.downsample_block2 = layers.DownsampleBlock(inc=nc[1], outc=nc[2])
        self.downsample_block3 = layers.DownsampleBlock(inc=nc[2], outc=nc[3])
        self.downsample_block4 = layers.DownsampleBlock(inc=nc[3], outc=nc[4], down=False)
        # self.downsample_block5 = layers.DownsampleBlock(inc=nc[4], outc=nc[5])
        # self.adpt_pool = nn.AdaptiveAvgPool3d((2,2,2))
        #self.fclayer = nn.Sequential(nn.Linear(6144, 2048), nn.Tanh())
       
    def forward(self, x):
        f_down0, _ = self.downsample_block0(x)
        f_down1, _ = self.downsample_block1(f_down0)
        f_down2, _ = self.downsample_block2(f_down1)
        f_down3, _ = self.downsample_block3(f_down2)
        f_down4, _ = self.downsample_block4(f_down3)
        #out = self.fclayer(f_down4.reshape([f_down4.shape[0], -1]))

        return  f_down4, f_down3, f_down2, f_down1 # squeeze but preserve the batch dimension.

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
        self.grid_size = config.grid_size
        
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





