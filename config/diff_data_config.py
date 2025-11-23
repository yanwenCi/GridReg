from . import configlib
import importlib
from .config_utils import print_config
import sys
from types import SimpleNamespace
import ujson



parser = configlib.add_parser("General config")
# General global options
parser.add_argument('--using_HPC', default=0, type=int, help='using UCL HPC')
parser.add_argument('--exp_name', default=None, type=str, help='experiment name you want to add.')
parser.add_argument('--input_shape', default=[128,128,102], nargs='+', type=int, help='the shape of the images')
parser.add_argument('--voxel_size', default=[1.0, 1.0, 1.0], nargs='+', type=float, help='the size of the voxel')
parser.add_argument('--data_path', default='../AS-morph-interp-ver/0.7-0.7-0.7-64-64-51', type=str, help='the path to the data')
parser.add_argument('--key_file', default='key-train-IFIB-val-IFIB-test-IFIB.pkl', type=str, help='the name of the data')

# Dataloader options / augmentations
parser.add_argument('--affine_scale', default=0.0, type=float, help='affine transformation, scale 0 means not to add.')
parser.add_argument('--affine_seed', default=None, type=int, help='random seed for affine transformation')
parser.add_argument('--patched', default=0, type=float, help='take the cropped image patchs as network input')
parser.add_argument('--patch_size', default=[64,64,64], nargs='+', type=int, help='patch size, only used when --patched is 1.')
parser.add_argument('--inf_patch_stride_factors', default=[4, 4, 4], nargs='+', type=int, help='stride for getting patch in inference, stride=patchsize//this_factor')
parser.add_argument('--cropped', default=0, type=float, help='take the cropped image patchs as network input')
parser.add_argument('--crop_size', default=[128, 128, 96], nargs='+', type=int, help='patch size, only used when --patched is 1.')

# General Training options
parser.add_argument('--lr', default=1e-5, type=float, help='Learning rate.')
parser.add_argument('--batch_size', default=4, type=int, help='The number of batch size.')
parser.add_argument('--gpu', default=0, type=int, help='id of gpu')
parser.add_argument('--num_epochs', default=300, type=int, help='The number of iterations.')
parser.add_argument('--save_frequency', default=10, type=int, help='save frequency')
parser.add_argument('--continue_epoch', default='-1', type=str, help='continue training from a certain ckpt')

config = SimpleNamespace(**configlib.parse())


print_config(config, parser)

# with open('./gen_config.json', 'w') as f:
#     ujson.dump(vars(config), f)
