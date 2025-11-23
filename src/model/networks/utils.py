import torch
from src.model.networks import transforms

class SpatialTransform:
    """
    A class for spatial transformation for 3D image volume (batch,c,z,y,x)
    """

    def __init__(self, volsize, batch_size, device):
        """
        :param volsize: tuple (x,y,z)
        :param batch_size: the batch_size transformations apply on c volumes
        """
        self.volsize = volsize
        self.batch_size = batch_size
        self.device = device
        self.voxel_coords = (
            torch.stack(
                torch.meshgrid(
                    torch.linspace(-1, 1, self.volsize[2]),
                    torch.linspace(-1, 1, self.volsize[1]),
                    torch.linspace(-1, 1, self.volsize[0]),
                    indexing="ij",
                ),
                dim=3,
            )[None, ...]
            .expand(self.batch_size, -1, -1, -1, -1)
            .to(self.device)
        )[
            ..., [2, 1, 0]
        ]  # ijk -> xyz

        return 0

    def warp(self, vol):
        """
        :param vol: 5d (batch,c,z,y,x)
        """
        self.compute_ddf()  # child class function
        return sampler(vol, self.ddf + self.voxel_coords)


# TODO
class GlobalAffine(SpatialTransform):
    def __init__(self):
        super().__init__()


# TODO
class LocalAffine(SpatialTransform):
    def __init__(self):
        super().__init__()


class GridTransform(SpatialTransform):
    def __init__(self, grid_size, interp_type="linear", **kwargs):
        super().__init__(**kwargs)
        """
        :param grid_size: num of control points in (x,y,z) same size between batch_size volumes
        """
        self.interp_type = interp_type
        self.grid_size = grid_size
        self.control_point_coords = (
            torch.stack(
                torch.meshgrid(
                    torch.linspace(-1, 1, self.grid_size[2]),
                    torch.linspace(-1, 1, self.grid_size[1]),
                    torch.linspace(-1, 1, self.grid_size[0]),
                    indexing="ij",
                ),
                dim=3,
            )[None, ...]
            .expand(self.batch_size, -1, -1, -1, -1)
            .to(self.device)
        )[
            ..., [2, 1, 0]
        ]  # ijk -> xyz
        self.grid_dims = [2 / (self.grid_size[i] - 1) for i in [0, 1, 2]]  # (x,y,z)

        self.control_point_displacements = torch.zeros_like(self.control_point_coords)

        # pre-compute for spline kernels
        if self.interp_type == "g-spline":
            num_control_points = (
                self.grid_size[0] * self.grid_size[1] * self.grid_size[2]
            )
            num_voxels = self.volsize[0] * self.volsize[1] * self.volsize[2]
            """ does not work due to inefficient memory use
            d_p2c = self.control_point_coords.reshape(
                self.batch_size,-1,1,3).expand(-1,-1,num_voxels,3)
            - self.voxel_coords.reshape(
                self.batch_size,1,-1,3).expand(-1,num_control_points,-1,3)  # voxel-to-control distances
            self.control_to_voxel_weight = d_p2c.sum(dim=-1)*(-1)/sigma
            # normalise here
            """
            self.control_to_voxel_weights = torch.ones(
                self.batch_size, num_control_points, num_voxels
            ).to(self.device)

        elif self.interp_type == "b-spline":
            self.bspl = transforms.CubicBSplineFFDTransform(ndim=3, img_size=self.volsize, svf=True)

        elif self.interp_type == "t-conv":
            sigma_voxel = [
                10,
                10,
                5,
            ]  #:param sigma_voxel: (x,y,z) Gaussian spline parameter sigma in voxel (the larger sigma the smoother transformation)
            voxdims = [2 / (v - 1) for v in self.volsize]
            grid_dims = [2 / (u - 1) for u in self.grid_size]
            self.strides = [int(grid_dims[d] / voxdims[d]) for d in [0, 1, 2]]
            # make sure tails are odd numbers that can be used for centre-aligning padding
            self.tails = [int(sigma_voxel[d] * 3) for d in [0, 1, 2]]
            gauss_pdf = lambda x, sigma: 2.71828 ** (-0.5 * x**2 / sigma**2)
            self.kernels_1d = [
                torch.tensor(
                    [
                        gauss_pdf(x, sigma_voxel[d])
                        for x in range(-self.tails[d], self.tails[d] + 1)
                    ],
                    device=self.device,
                )
                for d in [0, 1, 2]
            ]
            # N.B normalising by sum does not preserve control point displacements
            # normalising using control point displacement for displacement-preserving alternative in transpose_conv_upsampling
            # self.kernels_1d = [k / k.sum() for k in self.kernels_1d]

    def generate_random_transform(self, rate=0.25, scale=0.1):
        """
        Generate random displacements on control points dcp (uniform distribution)
        :param rate: [0,1] *100% percentage of all control points in use
        :param scale: [0,1] scale of unit grid size the random displacement
        """
        self.control_point_displacements = (
            (
                torch.rand(
                    [
                        self.batch_size,
                        self.grid_size[2],
                        self.grid_size[1],
                        self.grid_size[0],
                        3,
                    ]
                )
                * 2
                - 1
            )
            * torch.tensor([self.grid_dims[i] for i in [2, 1, 0]]).view(1, 1, 1, 1, 3)
            * scale
            * (
                torch.rand(
                    [
                        self.batch_size,
                        self.grid_size[2],
                        self.grid_size[1],
                        self.grid_size[0],
                    ]
                )
                < rate
            )[..., None].expand(-1, -1, -1, -1, 3)
        ).to(self.device)

    def compute_ddf(self, control_point_displacements):
        """
        Compute dense displacement field (ddf), interpolating displacement vectors on all voxels
        N.B. like all volume data, self.ddf is in z-y-x order
        """
        self.control_point_displacements = control_point_displacements
        if self.interp_type=="linear":
            self.ddf = self.linear_interpolation()
        elif self.interp_type == "g-spline_gauss":
                # self.evaluate_gaussian_spline()
            print("Yet implemented.")
        elif self.interp_type == "b-spline":
            self.bspl.compute_flow(self.control_point_displacements)
            
        elif self.interp_type == "t-conv":
                self.ddf = self.transpose_conv_upsampling()
        else:
            raise NotImplementedError

    def linear_interpolation(self):
        """
        input: permute to (batch,c,z,y,x), c=xyz
        grid: (batch,z,y,x,xyz)

        Return ddf: permute back to (batch,z,y,x,xyz)
        """
        return sampler(
            self.control_point_displacements.permute(0, 4, 1, 2, 3), self.voxel_coords
        ).permute(0, 2, 3, 4, 1)

    def transpose_conv_upsampling(self):
        """
        Using transpose convolution to approximate Gaussian spline transformation
        """
        # padding so centres are aligned
        ddf = torch.nn.functional.conv_transpose3d(
            torch.nn.functional.conv_transpose3d(
                torch.nn.functional.conv_transpose3d(
                    input=self.control_point_displacements.permute(0, 4, 1, 2, 3),
                    weight=self.kernels_1d[0]
                    .view(1, 1, 1, 1, -1)
                    .expand(3, 3, -1, -1, -1),
                    stride=(1, 1, self.strides[0]),
                    padding=(0, 0, self.tails[0]),
                ),
                weight=self.kernels_1d[1].view(1, 1, 1, -1, 1).expand(3, 3, -1, -1, -1),
                stride=(1, self.strides[1], 1),
                padding=(0, self.tails[1], 0),
            ),
            weight=self.kernels_1d[2].view(1, 1, -1, 1, 1).expand(3, 3, -1, -1, -1),
            stride=(self.strides[2], 1, 1),
            padding=(self.tails[2], 0, 0),
        )
        ddf = sampler(ddf, self.voxel_coords)
        # normalise to preserve displacement
        control_point_ddf = sampler(ddf, self.control_point_coords).permute(
            0, 2, 3, 4, 1
        )
        ratio = (
            self.control_point_displacements.view(self.batch_size, -1, 3).max(1)[0]
            / control_point_ddf.view(self.batch_size, -1, 3).max(1)[0]
        )
        return ddf.permute(0, 2, 3, 4, 1) * ratio.view(
            self.batch_size, 1, 1, 1, 3
        )  # back to (batch,z,y,x,xyz)

    def evaluate_gaussian_spline(self):
        """
        # compute all voxel-to-control distances
        # compute the weights using gaussian kernel
        # compute ddf
        """
        for d in [0, 1, 2]:
            self.ddf[..., d] = torch.matmul(
                self.control_point_displacements[..., d], self.control_to_voxel_weights
            )


## common functions
def sampler(vol, coords):
    return torch.nn.functional.grid_sample(
        input=vol,
        grid=coords,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )