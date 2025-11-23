import os
from config.global_train_config import config
import torch

if not config.using_HPC:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{config.gpu}"

if __name__ == "__main__":
    # if args.model == "origin":
    #     model = archs.mpMRIRegUnsupervise(args)
    # elif args.model == "weakly":
    #     from src.model.archs.mpMRIRegWeakSupervise import weakSuperVisionMpMRIReg
    #     model = weakSuperVisionMpMRIReg(args)
    # elif args.model == "joint3":
    #     model = archs.joint3(args)

    if config.project == 'Longitudinal':
        from src.model.archs.longitudinal import LongiReg
        model = LongiReg(config)
    elif config.project == "Icn":
        from src.model.archs.icReg import icReg
        model = icReg(config)
    elif config.project == "Ugrid":
        from src.model.archs.gridReg import icReg
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
    elif config.project == "denoise":
        from src.model.archs.denoise import denoise
        model = denoise(config)
    else:
        raise NotImplementedError

    if config.continue_epoch != '-1':
        model.load_epoch(config.continue_epoch)

    model.train()
    print('Optimization done.')
