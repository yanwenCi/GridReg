from src.model.networks.local import UNet
from src.model import loss
import src.model.functions as smfunctions
from src.model.archs.baseArch import BaseArch
from src.data import dataloaders
import torch, os
import torch.optim as optim
from torch.utils.data import DataLoader
import pickle as pkl
import numpy as np
from scipy import stats
import random


class cbctSeg(BaseArch):
    def __init__(self, config):
        super(cbctSeg, self).__init__(config)
        self.config = config
        self.net = self.net_parsing()
        self.set_dataloader()
        self.best_metric = 0

    def net_parsing(self):
        model = self.config.model
        if model == 'UNet':
            net = UNet(self.config)
        else:
            raise NotImplementedError
        return net.cuda()

    def set_dataloader(self):
        self.train_set = dataloaders.CBCTData(config=self.config, phase='train')
        self.train_loader = DataLoader(
            self.train_set, 
            batch_size=self.config.batch_size,  
            num_workers=4, 
            shuffle=True,
            drop_last=True)
        print('>>> Train set ready.')  
        self.val_set = dataloaders.CBCTData(config=self.config, phase='val')
        self.val_loader = DataLoader(self.val_set, batch_size=1, shuffle=False)
        print('>>> Validation set ready.')
        self.test_set = dataloaders.CBCTData(config=self.config, phase='test')
        self.test_loader = DataLoader(self.test_set, batch_size=1, shuffle=False)
        print('>>> Holdout set ready.')

    def get_input(self, input_dict, aug=True):
        fx_img, mv_img = input_dict['fx_img'].cuda(), input_dict['mv_img'].cuda()  # [batch, 1, x, y, z]
        fx_seg, mv_seg = input_dict['fx_seg'].cuda(), input_dict['mv_seg'].cuda()
        fx_seg, mv_seg = fx_seg[:, 0, ...], mv_seg[:, 0, ...]

        if (self.config.affine_scale != 0.0) and aug:
            mv_affine_grid = smfunctions.rand_affine_grid(
                mv_img, 
                scale=self.config.affine_scale, 
                random_seed=self.config.affine_seed
                )
            fx_affine_grid = smfunctions.rand_affine_grid(
                fx_img, 
                scale=self.config.affine_scale,
                random_seed=self.config.affine_seed
                )
            mv_img = torch.nn.functional.grid_sample(mv_img, mv_affine_grid, mode='bilinear', align_corners=True)
            mv_seg = torch.nn.functional.grid_sample(mv_seg, mv_affine_grid, mode='bilinear', align_corners=True)
            fx_img = torch.nn.functional.grid_sample(fx_img, fx_affine_grid, mode='bilinear', align_corners=True)
            fx_seg = torch.nn.functional.grid_sample(fx_seg, fx_affine_grid, mode='bilinear', align_corners=True)
        else:
            pass

        # ct/cbct/oneof/both
        if self.config.input_mode == 'both':
            assert self.config.inc == 2, "input channel needs to be 2"
            return torch.cat([fx_img, mv_img], dim=1), fx_seg  # [cbct, ct], cbct_seg
        elif self.config.input_mode == 'ct':
            assert self.config.inc == 1, "input channel needs to be 1"
            if self.phase == 'train':
                return mv_img, mv_seg
            else:
                return fx_img, fx_seg
        elif self.config.input_mode == 'cbct':
            assert self.config.inc == 1, "input channel needs to be 1"
            return fx_img, fx_seg
        elif self.config.input_mode == 'oneof':
            assert self.config.inc == 1, "input channel needs to be 1"
            if self.phase == 'train':
                tmp = [(fx_img, fx_seg), (mv_img, mv_seg)]
                return random.sample(tmp, 1)[0]
            else:
                return fx_img, fx_seg
        else:
            raise NotImplementedError


    def train(self):
        self.save_configure()
        optimizer = optim.Adam(self.net.parameters(), lr=self.config.lr, weight_decay=1e-6)
        for self.epoch in range(1, self.config.num_epochs + 1):
            self.train_mode()
            print('-' * 10, f'Train epoch_{self.epoch}', '-' * 10)
            for self.step, input_dict in enumerate(self.train_loader):
                input_tensor, gt_seg = self.get_input(input_dict)
                optimizer.zero_grad()

                pred_seg = self.net(input_tensor)

                global_loss = self.loss(pred_seg, gt_seg)
                global_loss.backward()
                optimizer.step()

            if self.epoch % self.config.save_frequency == 0:
                self.save()
            print('-' * 10, 'validation', '-' * 10)
            self.validation()
        
        self.inference()

    def loss(self, pred_seg, gt_seg):
        L_All = 0
        Info = f'step {self.step}'
        
        if self.config.w_dce != 0:
            L_dice = 0
            for label_idx in range(pred_seg.shape[1]):
                pred = pred_seg[:, label_idx:label_idx+1, ...]
                gt = gt_seg[:, label_idx:label_idx+1, ...]
                L_dice += loss.single_scale_dice(pred, gt) * self.config.dice_class_weights[label_idx]
            L_dice *= self.config.w_dce
            
            L_All += L_dice
            Info += f', Loss_dice: {L_dice:.3f}'
            
        if self.config.w_bce != 0:
            L_BCE = loss.wBCE(pred_seg, fx_seg, weights=self.config.class_weights)
            L_All += L_BCE
            Info += f', Loss_wBCE: {L_BCE:.3f}'

        Info += f', Loss_All: {L_All:.3f}'

        print(Info)
        return L_All

    @torch.no_grad()
    def validation(self):
        self.val_mode()
        visualization_path = os.path.join(self.log_dir, f'{self.config.exp_name}-vis-in-val')
        os.makedirs(visualization_path, exist_ok=True)

        res = []
        for idx, input_dict in enumerate(self.val_loader):
            input_tensor, gt_seg = self.get_input(input_dict, aug=False)
            pred_seg = self.net(input_tensor)
            subject = input_dict['subject']

            for label_idx in range(pred_seg.shape[1]):
                binary_dice = loss.binary_dice(pred_seg[:, label_idx, ...], gt_seg[:, label_idx, ...])
                print(f'subject:{subject}', f'label_idx:{label_idx}', f'DICE:{binary_dice:.3f}')
                res.append(binary_dice)

        res = torch.tensor(res)
        mean, std = torch.mean(res), torch.std(res)
        if mean > self.best_metric:
            self.best_metric = mean
            print('better model found.')
            self.save(type='best')
        print('Dice:', mean, std)


    @torch.no_grad()
    def inference(self):
        self.test_mode()
        visualization_path = os.path.join(self.log_dir, f'{self.config.exp_name}-vis-{self.epoch}')
        os.makedirs(visualization_path, exist_ok=True)

        results = {
            'dice': [],
            }

        for idx, input_dict in enumerate(self.test_loader):
            input_tensor, gt_seg = self.get_input(input_dict, aug=False)
            pred_seg = self.net(input_tensor)

            # self.save_img(fx_img, os.path.join(visualization_path, f'{idx+1}-fx_img.nii'))
            # self.save_img(mv_img, os.path.join(visualization_path, f'{idx+1}-mv_img.nii'))
            
            subject = input_dict['subject']
            for label_idx in range(gt_seg.shape[1]):
                aft_dice = loss.binary_dice(pred_seg[:, label_idx, ...], gt_seg[:, label_idx, ...]).cpu().numpy()

                results['dice'].append(aft_dice)
                print(
                    f'subject:{subject}', 
                    f'label_idx:{label_idx}', 
                    f'AFT-DICE:{aft_dice:.3f}'
                    )

                # self.save_img(fx_seg[:, label_idx, ...], os.path.join(visualization_path, f'{idx+1}-fx_img_{label_idx}.nii'))
                # self.save_img(mv_seg[:, label_idx, ...], os.path.join(visualization_path, f'{idx+1}-mv_img_{label_idx}.nii'))
                # self.save_img(pred_seg[0], os.path.join(visualization_path, f'{idx+1}-pred_img_{label_idx}.nii'))

            print('-' * 20)

        for k, v in results.items():
            mean, std = np.mean(v), np.std(v)
            print(k, f'{mean:.3f}, {std:.3f}')

        with open(os.path.join(self.log_dir, 'results.pkl'), 'wb') as f:
            pkl.dump(results, f)