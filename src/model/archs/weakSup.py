from src.model.networks.local import LocalModel
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


class weakSup(BaseArch):
    def __init__(self, config):
        super(weakSup, self).__init__(config)
        self.config = config
        self.net = LocalModel(self.config).cuda()
        self.set_dataloader()
        self.best_metric = 0
        
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
        fx_seg, mv_seg = input_dict['fx_seg'].cuda(), input_dict['mv_seg'].cuda()  # [batch, 1, x, y, z]

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

        return fx_img, mv_img, fx_seg, mv_seg

    def get_sample_name(self, input_dict):
        return input_dict['subject']

    def train(self):
        self.save_configure()
        optimizer = optim.Adam(self.net.parameters(), lr=self.config.lr)
        for self.epoch in range(1, self.config.num_epochs + 1):
            self.net.train()
            for self.step, input_dict in enumerate(self.train_loader):
                fx_img, mv_img, fx_seg, mv_seg = self.get_input(input_dict)
                optimizer.zero_grad()

                if self.config.inc == 3:
                    '''not weaksup, just as a kind of baseline'''
                    _, ddf = self.net(torch.cat([fx_img, mv_img, mv_seg], dim=1))
                else:
                    _, ddf = self.net(torch.cat([fx_img, mv_img], dim=1))

                warpped_mv_seg = smfunctions.warp3d(mv_seg, ddf)
                warpped_mv_img = smfunctions.warp3d(mv_img, ddf)

                global_loss = self.loss(
                    ddf, 
                    fx_img, 
                    warpped_mv_img, 
                    fx_seg, 
                    warpped_mv_seg, 
                    )
                global_loss.backward()
                optimizer.step()

            if self.epoch % self.config.save_frequency == 0:
                self.save()
            print('-'*10, 'validation', '-'*10)
            self.validation()

    def loss(self, ddf, fx_img, wp_mv_img, fx_seg, wp_mv_seg):
        L_ssd = loss.ssd(fx_img, wp_mv_img) * self.config.w_ssd
        L_dice = loss.single_scale_dice(fx_seg, wp_mv_seg) * self.config.w_dce
        L_All = L_ssd + L_dice
        Info = f'step:{self.step}, Loss_ssd: {L_ssd:.3f}, Loss_dice: {L_dice:.3f}'

        if self.config.w_bde != 0:
            L_bending = loss.normalized_bending_energy(
                ddf, 
                self.config.voxel_size, 
                self.config.input_shape) * self.config.w_bde

            L_All += L_bending
            Info += f', Loss_bde: {L_bending:.3f}'
        
        if self.config.w_l2g !=0:
            L_l2g = loss.l2_gradient(ddf) * self.config.w_l2g
            L_All += L_l2g
            Info += f', Loss_l2g: {L_l2g:.3f}'

        Info += f', Loss_all: {L_All:.3f}'

        print(Info)
        return L_All

    @torch.no_grad()
    def validation(self):
        self.net.eval()
        res = []
        for idx, input_dict in enumerate(self.val_loader):
            fx_img, mv_img, fx_seg, mv_seg = self.get_input(input_dict, aug=False)
            subject = self.get_sample_name(input_dict)[0]

            for label_idx in range(fx_seg.shape[2]):
                sub_mv_seg = mv_seg[:, :, label_idx, ...]
                sub_fx_seg = fx_seg[:, :, label_idx, ...]

                if self.config.inc == 3:
                    # print(mv_img.shape, mv_seg.shape)
                    _, ddf = self.net(torch.cat([fx_img, mv_img, sub_mv_seg], dim=1))
                else:
                    _, ddf = self.net(torch.cat([fx_img, mv_img], dim=1))

                warpped_mv_img = smfunctions.warp3d(mv_img, ddf)
                warpped_mv_seg = smfunctions.warp3d(sub_mv_seg, ddf)
                
                aft_dice = loss.binary_dice(warpped_mv_seg, sub_fx_seg)
                bef_dice = loss.binary_dice(sub_fx_seg, sub_mv_seg)

                print(f'subject:{subject}', f'label_idx:{label_idx}', f'BEF-DICE:{bef_dice:.3f}', f'AFT-DICE:{aft_dice:.3f}')
                res.append(aft_dice)
    
        res = torch.tensor(res)
        mean, std = torch.mean(res), torch.std(res)
        if mean > self.best_metric:
            self.best_metric = mean
            print('better model found.')
            self.save(type='best')
        print('Dice:', mean, std)


    @torch.no_grad()
    def inference(self):
        self.net.eval()
        visualization_path = os.path.join(self.log_dir, f'vis-{self.epoch}')
        os.makedirs(visualization_path, exist_ok=True)

        results = {
            'dice': [],
            'dice-wo-reg': [],
            'ssd': [],
            'ssd-wo-reg': [],
            'cd': [],
            'cd-wo-reg': []
            }

        for idx, input_dict in enumerate(self.test_loader):
            fx_img, mv_img, fx_seg, mv_seg = self.get_input(input_dict, aug=False)
            subject = self.get_sample_name(input_dict)[0]

            self.save_img(fx_img, os.path.join(visualization_path, f'{idx+1}-fx_img.nii'))
            self.save_img(mv_img, os.path.join(visualization_path, f'{idx+1}-mv_img.nii'))

            label_num= fx_seg.shape[2]
            for label_idx in range(fx_seg.shape[2]):
                sub_mv_seg = mv_seg[:, :, label_idx, ...]
                sub_fx_seg = fx_seg[:, :, label_idx, ...]

                if self.config.inc == 3:
                    '''not weaksup, just as a kind of baseline'''
                    _, ddf = self.net(torch.cat([fx_img, mv_img, sub_mv_seg], dim=1))
                else:
                    _, ddf = self.net(torch.cat([fx_img, mv_img], dim=1))

                warpped_mv_img = smfunctions.warp3d(mv_img, ddf)
                warpped_mv_seg = smfunctions.warp3d(sub_mv_seg, ddf)

                bef_ssd = loss.ssd(fx_img, mv_img).cpu().numpy()
                aft_ssd = loss.ssd(fx_img, warpped_mv_img).cpu().numpy()
                results['ssd'].append(aft_ssd)
                results['ssd-wo-reg'].append(bef_ssd)

                aft_dice = loss.binary_dice(warpped_mv_seg, sub_fx_seg).cpu().numpy()
                bef_dice = loss.binary_dice(sub_fx_seg, sub_mv_seg).cpu().numpy()
                aft_cd = loss.centroid_distance(sub_fx_seg, warpped_mv_seg).cpu().numpy()
                bef_cd = loss.centroid_distance(sub_fx_seg, sub_mv_seg).cpu().numpy()
                
                results['dice'].append(aft_dice)
                results['dice-wo-reg'].append(bef_dice)
                results['cd'].append(aft_cd)
                results['cd-wo-reg'].append(bef_cd)

                self.save_img(warpped_mv_img, os.path.join(visualization_path, f'{idx+1}-wp_img_{label_idx}.nii'))
                self.save_img(ddf[0, 0, :, :, :], os.path.join(visualization_path, f'{idx+1}-ddf-x_{label_idx}.nii'))
                self.save_img(ddf[0, 1, :, :, :], os.path.join(visualization_path, f'{idx+1}-ddf-y_{label_idx}.nii'))
                self.save_img(ddf[0, 2, :, :, :], os.path.join(visualization_path, f'{idx+1}-ddf-z_{label_idx}.nii'))

                self.save_img(sub_mv_seg, os.path.join(visualization_path, f'{idx+1}-mv_seg_{label_idx}.nii'))
                self.save_img(sub_fx_seg, os.path.join(visualization_path, f'{idx+1}-fx_seg_{label_idx}.nii'))
                self.save_img(warpped_mv_seg, os.path.join(visualization_path, f'{idx+1}-wp_seg_{label_idx}.nii'))

                print(
                    f'subject:{subject}', 
                    f'label_idx:{label_idx}', 
                    f'BEF-SSD:{bef_ssd:.3f}',
                    f'AFT-SSD:{aft_ssd:.3f}',
                    f'BEF-DICE:{bef_dice:.3f}', 
                    f'AFT-DICE:{aft_dice:.3f}',
                    f'AFT-CD:{aft_cd:.3f}',
                    f'BEF-CD:{bef_cd:.3f}',
                    )
            print('-' * 20)

        for k, v in results.items():
            print(f'overall {k}, {np.mean(v):.3f}, {np.std(v):.3f}')
            if 'dice' in k or 'cd' in k:
                for idx in range(label_num):
                    tmp = v[idx::label_num]
                    print(f'results of {k} on label {idx}:, {np.mean(tmp):.3f} +- {np.std(tmp):.3f}')                

        with open(os.path.join(self.log_dir, 'results.pkl'), 'wb') as f:
            pkl.dump(results, f)