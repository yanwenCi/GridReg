import os
from src.model import archs
from config.config_utils import print_config
import pickle as pkl
import sys
import time

# if sys.argv[1].endswith('/'):
#     sys.argv[1] = sys.argv[1][:-1]
# exp_name = os.path.basename(sys.argv[1])
# project_name = os.path.basename(os.path.dirname(sys.argv[1]))

# print(f'evaluation... {exp_name}')

# if len(sys.argv) > 2:
#     gpu_idx = sys.argv[2]
# else:
#     gpu_idx = '0'

# if len(sys.argv) > 3:
#     num_epoch = int(sys.argv[3])
# else:
#     num_epoch = 'best'

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = gpu_idx



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_name', type=str, default='Icn')
    parser.add_argument('--exp_name', type=str)
    parser.add_argument('--num_epoch', default='best')
    args = parser.parse_args()

    with open(os.path.join(f'./logs/{args.project_name}/{args.exp_name}/config.pkl'), 'rb') as f:
        config = pkl.load(f)
        print_config(config)

    if not hasattr(config, 'patched'):
        config.patched = 0
    config.exp_name = args.exp_name

    if config.project == 'Longitudinal':
        from src.model.archs.longitudinal import LongiReg
        model = LongiReg(config)
    elif config.project == "Icn":
        from src.model.archs.icReg import icReg
        model = icReg(config)
    elif config.project == "ConditionalSeg":
        from src.model.archs.condiSeg import condiSeg
        model = condiSeg(config)
    elif config.project == "WeakSup":
        from src.model.archs.weakSup import weakSup
        model = weakSup(config)
    elif config.project == "CBCTUnetSeg":
        from src.model.archs.cbctSeg import cbctSeg
        model = cbctSeg(config)
    elif config.project == "mpmrireg":
        from src.model.archs.mpmrireg import mpmrireg
        model = mpmrireg(config)
    else:
        raise NotImplementedError

    model.load_epoch(num_epoch = args.num_epoch)
    # # starting the monitoring
    # tracemalloc.start()
    
    model.inference()
    # # displaying the memory
    # print(tracemalloc.get_traced_memory())

    # # stopping the library
    # tracemalloc.stop()
    
    print('inference done.')
