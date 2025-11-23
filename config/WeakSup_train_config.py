from . import configlib

parser = configlib.add_parser("weakly registration configures")

# Network options
parser.add_argument('--nc_initial', default=16, type=int, help='initial number of the channels in the frist layer of the network')
parser.add_argument('--inc', default=2, type=int, help='input channel number of the network, if 3, mv_seg will be feed into the network')
parser.add_argument('--ddf_levels', default=[0, 1, 2, 3, 4], nargs='+', type=int, help='ddf levels, numbers should be <= 4')
parser.add_argument('--ddf_outshape', default=[10, 10, 10], nargs='+', type=int, help='the out shape of ddf, if only LocalEncoder is used. ')

# Training options
parser.add_argument('--model', default='LocalModel', type=str, help='LocalAffine/LocalEncoder/LocalModel')
parser.add_argument('--cv', default=0, type=int, help='The fold for cross validation')

# sampling options 
parser.add_argument('--two_stage_sampling', default=1, type=int, help='Only in training, random pick up one label for each sample.')

# loss & weights
parser.add_argument('--w_ssd', default=1.0, type=float, help='the weight of ssd loss')
parser.add_argument('--w_bde', default=10.0, type=float, help='the weight of bending energy loss')
parser.add_argument('--w_dce', default=1.0, type=float, help='the weight of dice loss')
parser.add_argument('--w_l2g', default=0.0, type=float, help='the weight of the l2 gradient for the ddf.')





