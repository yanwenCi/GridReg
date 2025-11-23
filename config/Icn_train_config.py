from . import configlib

parser = configlib.add_parser("Icn registration config")

parser.add_argument('--nc_initial', default=16, type=int, help='initial number of the channels in the frist layer of the network')
parser.add_argument('--grid_size', default=[10,10,10], nargs='+', type=int, help='size of the down sampled grid')
parser.add_argument('--ushape', action='store_true', help='whether to use the shape information')
parser.add_argument('--num_layers', default=8, type=int, help='number of layers in the network')
# network 
parser.add_argument('--model', default='ICNet', type=str, help='LocalAffine/LocalEncoder/ICNet')

# sampling options 
parser.add_argument('--patient_cohort', default='inter+intra', type=str, help='intra-patient type or inter-patient type')
parser.add_argument('--key_file', default='../data/', type=str, help='key file of the image pairs')

# loss & weights
parser.add_argument('--w_Gssd', default=1.0, type=float, help='the weight of grid sampled image SSD')
parser.add_argument('--w_Gdsc', default=1.0, type=float, help='the weight of grid sampled segmentation dice')
parser.add_argument('--w_Issd', default=1.0, type=float, help='the weight of whole image SSD')
parser.add_argument('--w_Idsc', default=1.0, type=float, help='the weight of whole segmentation dice')
parser.add_argument('--w_bde', default=3000, type=float, help='the weight of bending energy')

# denoise 
parser.add_argument('--noise_std', default=0.1, type=float, help='whether to denoise the image')
parser.add_argument('--noise_var', default=10, type=int, help='whether to denoise the image')

#control points
parser.add_argument('--num_control_points', default=1000, type=int, help='the spacing of the control points')
parser.add_argument('--in_nc', default=1, type=int, help='the number of input channels')
parser.add_argument('--COM', action='store_true', help='whether to use the center of mass as the control points')







