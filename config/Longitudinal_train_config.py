from . import configlib

parser = configlib.add_parser("Longitudinal registration config")

# Network options
parser.add_argument('--nc_initial', default=16, type=int, help='initial number of the channels in the frist layer of the network')
parser.add_argument('--ddf_levels', default=[0, 1, 2, 3, 4], nargs='+', type=int, help='ddf levels, numbers should be <= 4')
parser.add_argument('--ddf_outshape', default=[10, 10, 10], nargs='+', type=int, help='the out shape of ddf, if only LocalEncoder is used. ')
parser.add_argument('--inc', default=2, type=int, help='input channel number of the network, if 3, mv_seg will be feed into the network')
# Training options
parser.add_argument('--model', default='LocalModel', type=str, help='LocalAffine/LocalEncoder/LocalModel')

# sampling options 
parser.add_argument('--patient_cohort', default='intra', type=str, help='intra-patient type or inter-patient type')
parser.add_argument('--key_file', default='../data/', type=str, help='key file of the image pairs')

# loss & weights
parser.add_argument('--w_ncc', default=1.0, type=float, help='the weight of ncc loss')
parser.add_argument('--w_ssd', default=1.0, type=float, help='the weight of ssd loss')
parser.add_argument('--w_bde', default=10.0, type=float, help='the weight of bending energy loss')
parser.add_argument('--w_mmd', default=1.0, type=float, help='the weight of maximum mean discrepancy loss')
parser.add_argument('--w_dce', default=1.0, type=float, help='the weight of dice loss')
parser.add_argument('--w_l2g', default=0.0, type=float, help='the weight of the l2 gradient for the ddf.')

# parser.add_argument('--sigmas', default=[1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 5.0, 10.0, 15.0,
#                                          20.0, 25.0, 30.0, 35.0, 100.0, 1e3, 1e4, 1e5, 1e6],
#                     nargs='+', type=float, help='sigmas for the gaussian kernels in mmd loss')

parser.add_argument('--sigmas', default=[0.1, 1.0, 10.0, 20.0, 30.0, 100.0, 1e3, 1e4, 1e5, 1e6],
                    nargs='+', type=float, help='sigmas for the gaussian kernels in mmd loss')










