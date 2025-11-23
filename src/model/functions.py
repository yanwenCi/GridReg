import torch
import torch.nn.functional as F
import numpy as np
import pystrum.pynd.ndutils as nd
from torch.autograd import Variable
from math import exp

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def create_window_3D(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t())
    _3D_window = _1D_window.mm(_2D_window.reshape(1, -1)).reshape(window_size, window_size, window_size).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_3D_window.expand(channel, 1, window_size, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)
    
def _ssim_3D(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv3d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv3d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)

    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv3d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv3d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv3d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)
    


class SSIM(torch.nn.Module):
    def __init__(self, window_size = 11, size_average = True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            
            self.window = window
            self.channel = channel


        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)
    
    
class SSIM3D(torch.nn.Module):
    def __init__(self, window_size = 11, size_average = True):
        super(SSIM3D, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window_3D(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window_3D(self.window_size, channel)
            
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            
            self.window = window
            self.channel = channel


        return 1-_ssim_3D(img1, img2, window, self.window_size, channel, self.size_average)

    
def ssim(img1, img2, window_size = 11, size_average = True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    
    return _ssim(img1, img2, window, window_size, channel, size_average)

def ssim3D(img1, img2, window_size = 11, size_average = True):
    (_, channel, _, _, _) = img1.size()
    window = create_window_3D(window_size, channel)
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    
    return _ssim_3D(img1, img2, window, window_size, channel, size_average)

def get_reference_grid3d(img, grid_size=None):
    '''
    return a 5d tensor of the grid, e.g.
    img --> (b, 1, h, w, z)
    out --> (b, 3, h, w, z)

    if grid_size is not None, then return a 3d grid with the size of grid_size
    grid_size --> (gh, gw, gz)
    '''
    if len(img.shape) > 3:
        batch = img.shape[0]
    else: 
        batch = 1
    
    shape = img.shape[-3:]
    
    if grid_size is not None:
        assert len(grid_size) == 3, "maybe not a 3d grid"
        shape = grid_size

    mesh_points = [torch.linspace(-1, 1, dim) for dim in shape]
    grid = torch.stack(torch.meshgrid(*mesh_points, indexing='ij'))  # shape:[3, x, y, z]
    grid = torch.stack([grid]*batch)  # add batch
    grid = grid.type(torch.FloatTensor) # [batch, 3, x, y, z]
    return grid.cuda()

def warp3d(img, ddf, ref_grid=None):
    """
    img: [batch, c, x, y, z]
    new_grid: [batch, x, y, z, 3]
    https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html#torch.nn.functional.grid_sample
    """

    if ref_grid is None:
        assert img.shape[-3:] == ddf.shape[-3:], "Shapes not consistent btw img and ddf."
        grid = get_reference_grid3d(img)
    else:
        grid = ref_grid
  
    new_grid = grid + ddf  # [batch, 3, x, y, z]
    # print(new_grid.max(), new_grid.min(), grid.max(), grid.min(), ddf.max(), ddf.min())
    new_grid = new_grid.permute(0, 2, 3, 4, 1)
    new_grid = new_grid[..., [2, 1, 0]]
    return F.grid_sample(img, new_grid, mode='bilinear', align_corners=False)


def warp3d_v2(img, ddf, ref_grid=None):
    ddf = ddf.permute(0, 2, 3, 4, 1)
    # print(ddf.shape, img.shape)
    return F.grid_sample(img, grid=ddf,mode='bilinear',
                         padding_mode="border",align_corners=False)

def ddf_merge(ddf_A, ddf_B):
    '''merge 2 DDFs to an equal DDF'''
    assert ddf_A.shape == ddf_B.shape, "shape of the 2 ddf must be the same"
    ref_grid = get_reference_grid3d(ddf_A)  # [batch, 3, x, y, z]
    grid_A = ref_grid + ddf_A

    grid_B = ref_grid + ddf_B
    grid_B = grid_B.permute(0, 2, 3, 4, 1)
    grid_B = grid_B[..., [2, 1, 0]]

    warped_grid_A = F.grid_sample(grid_A, grid_B, mode='bilinear', align_corners=False)  # [batch, 3, x, y, z]
    return warped_grid_A - ref_grid

def gen_rand_affine_transform(batch_size, scale, seed=None):
    """
    https://github.com/DeepRegNet/DeepReg/blob/d3edf264b8685b47f1bdd9bb73aca79b1a72790b/deepreg/dataset/preprocess.py
    :param scale: a float number between 0 and 1
    :return: shape = (batch, 4, 3)
    """
    assert 0 <= scale <= 1
    if seed is not None:
        np.random.seed(seed)
    noise = np.random.uniform(1 - scale, 1, [batch_size, 4, 3])  # shape = (batch, 4, 3)

    # old represents four corners of a cube
    # corresponding to the corner C G D A as shown above
    old = np.tile(
        [[[-1, -1, -1, 1], [-1, -1, 1, 1], [-1, 1, -1, 1], [1, -1, -1, 1]]],
        [batch_size, 1, 1],
    )  # shape = (batch, 4, 4)
    new = old[:, :, :3] * noise  # shape = (batch, 4, 3)

    theta = np.array(
        [np.linalg.lstsq(old[k], new[k], rcond=-1)[0] for k in range(batch_size)]
    )  # shape = (batch, 4, 3)

    return theta
    

def rand_affine_grid(img, scale=0.1, random_seed=None):
    grid = get_reference_grid3d(img)  #(batch, 4, 3) (b, i, j)
    theta = gen_rand_affine_transform(img.shape[0], scale, seed=random_seed)  # [batch, 3, x, y, z]  (b, j, x, y, z)
    theta = torch.FloatTensor(theta).cuda()
    padded_grid = torch.cat([grid, torch.ones_like(grid[:, :1, ...])], axis=1)
    warpped_grids = torch.einsum('bixyz,bij->bjxyz', padded_grid, theta)

    warpped_grids = warpped_grids.permute(0, 2, 3, 4, 1)
    warpped_grids = warpped_grids[..., [2, 1, 0]]
    return warpped_grids
    

class SpatialTransformer3d(torch.nn.Module):
    def __init__(self, use_gpu=False):
        self.use_gpu = use_gpu
        super(SpatialTransformer3d, self).__init__()

    def meshgrid(self, height, width):
        x_t = torch.matmul(torch.ones([height, 1]), torch.transpose(torch.unsqueeze(torch.linspace(0.0, width -1.0, width), 1), 1, 0))
        y_t = torch.matmul(torch.unsqueeze(torch.linspace(0.0, height - 1.0, height), 1), torch.ones([1, width]))

        x_t = x_t.expand([height, width])
        y_t = y_t.expand([height, width])
        if self.use_gpu==True:
            x_t = x_t.cuda()
            y_t = y_t.cuda()

        return x_t, y_t

    def repeat(self, x, n_repeats):
        rep = torch.transpose(torch.unsqueeze(torch.ones(n_repeats), 1), 1, 0)
        rep = rep.long()
        x = torch.matmul(torch.reshape(x, (-1, 1)), rep)
        if self.use_gpu:
            x = x.cuda()
        return torch.squeeze(torch.reshape(x, (-1, 1)))


    def interpolate(self, im, x, y):

        im = F.pad(im, (0,0,1,1,1,1,0,0))

        batch_size, height, width, channels = im.shape

        batch_size, out_height, out_width = x.shape

        x = x.reshape(1, -1)
        y = y.reshape(1, -1)

        x = x + 1
        y = y + 1

        max_x = width - 1
        max_y = height - 1

        x0 = torch.floor(x).long()
        x1 = x0 + 1
        y0 = torch.floor(y).long()
        y1 = y0 + 1

        x0 = torch.clamp(x0, 0, max_x)
        x1 = torch.clamp(x1, 0, max_x)
        y0 = torch.clamp(y0, 0, max_y)
        y1 = torch.clamp(y1, 0, max_y)

        dim2 = width
        dim1 = width*height
        base = self.repeat(torch.arange(0, batch_size)*dim1, out_height*out_width)

        base_y0 = base + y0*dim2
        base_y1 = base + y1*dim2

        idx_a = base_y0 + x0
        idx_b = base_y1 + x0
        idx_c = base_y0 + x1
        idx_d = base_y1 + x1

        # use indices to lookup pixels in the flat image and restore
        # channels dim
        im_flat = torch.reshape(im, [-1, channels])
        im_flat = im_flat.float()
        dim, _ = idx_a.transpose(1,0).shape
        Ia = torch.gather(im_flat, 0, idx_a.transpose(1,0).expand(dim, channels))
        Ib = torch.gather(im_flat, 0, idx_b.transpose(1,0).expand(dim, channels))
        Ic = torch.gather(im_flat, 0, idx_c.transpose(1,0).expand(dim, channels))
        Id = torch.gather(im_flat, 0, idx_d.transpose(1,0).expand(dim, channels))

        # and finally calculate interpolated values
        x1_f = x1.float()
        y1_f = y1.float()

        dx = x1_f - x
        dy = y1_f - y

        wa = (dx * dy).transpose(1,0)
        wb = (dx * (1-dy)).transpose(1,0)
        wc = ((1-dx) * dy).transpose(1,0)
        wd = ((1-dx) * (1-dy)).transpose(1,0)

        output = torch.sum(torch.squeeze(torch.stack([wa*Ia, wb*Ib, wc*Ic, wd*Id], dim=1)), 1)
        output = torch.reshape(output, [-1, out_height, out_width, channels])
        return output

    def forward(self, moving_image, deformation_matrix):
        dx = deformation_matrix[:, :, :, 0]
        dy = deformation_matrix[:, :, :, 1]

        batch_size, height, width = dx.shape

        x_mesh, y_mesh = self.meshgrid(height, width)

        x_mesh = x_mesh.expand([batch_size, height, width])
        y_mesh = y_mesh.expand([batch_size, height, width])
        x_new = dx + x_mesh
        y_new = dy + y_mesh

        return self.interpolate(moving_image, x_new, y_new)


def apply_rigid_transform_3D(moving, params):
    """
    Apply 3D rigid transformation to the moving volume.

    Args:
        moving (torch.Tensor): Moving volume tensor of shape (N, C, D, H, W)
        params (torch.Tensor): Transformation parameters of shape (N, 6) -> [θx, θy, θz, tx, ty, tz]

    Returns:
        warped (torch.Tensor): Warped moving volume
    """
    theta_x, theta_y, theta_z = params[:, 0], params[:, 1], params[:, 2]
    tx, ty, tz = params[:, 3], params[:, 4], params[:, 5]
    
    # Rotation matrices around each axis
    Rx = torch.zeros((moving.size(0), 3, 3)).to(moving.device)
    Ry = torch.zeros((moving.size(0), 3, 3)).to(moving.device)
    Rz = torch.zeros((moving.size(0), 3, 3)).to(moving.device)
    
    # Populate Rx
    Rx[:, 0, 0] = 1
    Rx[:, 1, 1] = torch.cos(theta_x)
    Rx[:, 1, 2] = -torch.sin(theta_x)
    Rx[:, 2, 1] = torch.sin(theta_x)
    Rx[:, 2, 2] = torch.cos(theta_x)
    
    # Populate Ry
    Ry[:, 0, 0] = torch.cos(theta_y)
    Ry[:, 0, 2] = torch.sin(theta_y)
    Ry[:, 1, 1] = 1
    Ry[:, 2, 0] = -torch.sin(theta_y)
    Ry[:, 2, 2] = torch.cos(theta_y)
    
    # Populate Rz
    Rz[:, 0, 0] = torch.cos(theta_z)
    Rz[:, 0, 1] = -torch.sin(theta_z)
    Rz[:, 1, 0] = torch.sin(theta_z)
    Rz[:, 1, 1] = torch.cos(theta_z)
    Rz[:, 2, 2] = 1
    
    # Combined rotation matrix: R = Rz * Ry * Rx
    R = torch.bmm(Rz, torch.bmm(Ry, Rx))
    
    # Construct the affine transformation matrix
    affine_matrices = torch.zeros((moving.size(0), 3, 4)).to(moving.device)
    affine_matrices[:, :3, :3] = R
    affine_matrices[:, :, 3] = params[:, 3:]  # tx, ty, tz
    
    # Generate grid
    grid = F.affine_grid(affine_matrices, moving.size(), align_corners=True)
    
    # Sample the moving volume with the grid to get the warped volume
    warped = F.grid_sample(moving, grid, mode='bilinear', padding_mode='border', align_corners=True)
    
    return warped



    




