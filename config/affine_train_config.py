from . import configlib

parser = configlib.add_parser("Affine registration config")

parser.add_argument('--nc_initial', default=8, type=int, help='initial number of the channels in the frist layer of the network')

# sampling options 


# loss & weights










