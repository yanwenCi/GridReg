from . import configlib



parser = configlib.add_parser("Conditional segmentation config")

# Network options
parser.add_argument('--nc_initial', default=16, type=int, help='initial number of the channels in the frist layer of the network')

# Training options
parser.add_argument('--model', default='CondiSegUNet', type=str, help='LocalAffine/LocalEncoder/LocalModel/...')
parser.add_argument('--cv', default=0, type=int, help='The fold for cross validation')
# parser.add_argument('--vis_in_val', default=0, type=int, help='generate some visualizations during validation')

# sampling options
parser.add_argument('--two_stage_sampling', default=1, type=int, help='Only in training, random pick up one label for each sample.')
parser.add_argument('--patient_cohort', default='intra', type=str, help='Only in training, input inter or intra pairs')
parser.add_argument('--crop_on_seg_aug', default=0, type=int, help='adding random crop to the segmentaion')
parser.add_argument('--crop_on_seg_rad', default=[10, 20], nargs='+', type=int, help='the minimum and maximum radius for the random crop on seg.')
parser.add_argument('--use_pseudo_label', default=0, type=int, help='using pseudo label in training')


# loss & weights
parser.add_argument('--w_dce', default=1.0, type=float, help='the weight of dice loss')
parser.add_argument('--w_bce', default=0, type=float, help='the weight of weighted binary cross-entropy')
parser.add_argument('--class_weights', default=[0.5, 0.5], nargs='+', type=float, help='the weights for each class of the wbce')


