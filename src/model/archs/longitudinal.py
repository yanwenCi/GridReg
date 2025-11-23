from src.model.networks.local import LocalModel, LocalEncoder, LocalAffine
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
from tqdm import tqdm


class LongiReg(BaseArch):
    def __init__(self, config):
        super(LongiReg, self).__init__(config)
        self.config = config
        self.net = self.net_parsing()
        self.set_dataloader()
        self.best_metric = 0

    def net_parsing(self):
        model = self.config.model
        if model == 'LocalModel':
            net = LocalModel(self.config)
        elif model == 'LocalEncoder':
            net = LocalEncoder(self.config)
        elif model == 'LocalAffine':
            net = LocalAffine(self.config)
        else:
            raise NotImplementedError
        return net.cuda()

    def set_dataloader(self):
        self.train_set = dataloaders.LongitudinalData(config=self.config, phase='train')
        self.train_loader = DataLoader(
            self.train_set, 
            batch_size=self.config.batch_size, 
            shuffle=False,  
            num_workers=8, 
            drop_last=True)  # no need to shuffle since the shuffling is customized in the dataloader.
        print('>>> Train set ready.')  
        self.val_set = dataloaders.LongitudinalData(config=self.config, phase='val')
        self.val_loader = DataLoader(self.val_set, batch_size=1, shuffle=False)
        print('>>> Validation set ready.')
        self.test_set = dataloaders.LongitudinalData(config=self.config, phase='test')
        self.test_loader = DataLoader(self.test_set, batch_size=1, shuffle=False)
        print('>>> Holdout set ready.')

    def get_input(self, input_dict, aug=True):
        fx_img, mv_img = input_dict['fx_img'].cuda(), input_dict['mv_img'].cuda()  # [batch, 1, x, y, z]
        fx_seg, mv_seg = input_dict['fx_seg'].cuda(), input_dict['mv_seg'].cuda()
        if (self.config.affine_scale != 0.0) and aug:
            # only only affine grid is need here
            affine_grid = smfunctions.rand_affine_grid(
                mv_img, 
                scale=self.config.affine_scale, 
                random_seed=self.config.affine_seed
                )
            mv_img = torch.nn.functional.grid_sample(mv_img, affine_grid, mode='bilinear', align_corners=True)
            mv_seg = torch.nn.functional.grid_sample(mv_seg, affine_grid, mode='bilinear', align_corners=True)
            fx_img = torch.nn.functional.grid_sample(fx_img, affine_grid, mode='bilinear', align_corners=True)
            fx_seg = torch.nn.functional.grid_sample(fx_seg, affine_grid, mode='bilinear', align_corners=True)
        else:
            pass
        return fx_img, fx_seg, mv_img, mv_seg

    def train(self):
        self.save_configure()
        optimizer = optim.Adam(self.net.parameters(), lr=self.config.lr, weight_decay=1e-6)
        for self.epoch in range(1, self.config.num_epochs + 1):
            self.net.train()
            print('-' * 10, f'Train epoch_{self.epoch}', '-' * 10)
            for self.step, input_dict in enumerate(self.train_loader):
                fx_img, fx_seg, mv_img, mv_seg = self.get_input(input_dict)
                optimizer.zero_grad()
                f_bottleneck, ddf = self.net(torch.cat([fx_img, mv_img], dim=1))

                warpped_mv_img = smfunctions.warp3d(mv_img, ddf)
                warpped_mv_seg = smfunctions.warp3d(mv_seg, ddf)

                global_loss = self.loss(
                    ddf, 
                    fx_img, 
                    warpped_mv_img, 
                    fx_seg, 
                    warpped_mv_seg, 
                    f_bottleneck
                    )
                global_loss.backward()
                optimizer.step()

            if self.epoch % self.config.save_frequency == 0:
                self.save()
            print('-' * 10, 'validation', '-' * 10)
            self.validation()
        
        self.inference()

    def loss(self, ddf, fx_img, wp_mv_img, fx_seg, wp_mv_seg, f_bottleneck):
        L_ssd = loss.ssd(fx_img, wp_mv_img) * self.config.w_ssd
        L_dice = loss.single_scale_dice(fx_seg, wp_mv_seg) * self.config.w_dce
        L_All = L_ssd + L_dice
        Info = f'step:{self.step}, Loss_ssd: {L_ssd:.3f}, Loss_dice: {L_dice:.3f}'

        if self.config.w_bde != 0:
            L_bending = loss.normalized_bending_energy(
                ddf, 
                self.config.voxel_size, 
                self.config.input_shape) * self.config.w_bde
            # L_bending = loss.bending_energy(ddf) * self.config.w_bde
            L_All += L_bending
            Info += f', Loss_bde: {L_bending:.3f}'
        
        if self.config.w_l2g !=0:
            L_l2g = loss.l2_gradient(ddf) * self.config.w_l2g
            L_All += L_l2g
            Info += f', Loss_l2g: {L_l2g:.3f}'

        if self.config.w_mmd != 0:
            x1 = f_bottleneck[[i for i in range(self.config.batch_size) if (i % 2) == 0], ...]
            x2 = f_bottleneck[[i for i in range(self.config.batch_size) if (i % 2) != 0], ...]
            L_mmd = loss.mmd(x1, x2, sigmas=torch.cuda.FloatTensor(self.config.sigmas)) * self.config.w_mmd
            L_All += L_mmd
            Info += f', Loss_mmd: {L_mmd:.3f}'

        Info += f', Loss_all: {L_All:.3f}'

        print(Info)
        return L_All

    @torch.no_grad()
    def validation(self):
        self.net.eval()
        res = []
        for idx, input_dict in enumerate(self.val_loader):
            fx_img, fx_seg, mv_img, mv_seg = self.get_input(input_dict, aug=False)
            fx_key, mv_key = input_dict['fx_key'][0], input_dict['mv_key'][0]

            if not self.config.patched: 
                f_bottleneck, ddf = self.net(torch.cat([fx_img, mv_img], dim=1))
            else:
                patch_coords = self.get_patch_cords_from_ref_image(mv_img)
                fused_ddf = torch.cat([torch.zeros(mv_img.shape)]*3, dim=1).cuda()
                count_arr = torch.zeros(mv_img.shape[-3:]).cuda()
                
                for (x1, x2, y1, y2, z1, z2) in patch_coords:
                    patch_mv = mv_img[..., x1:x2, y1:y2, z1:z2]
                    patch_fx = fx_img[..., x1:x2, y1:y2, z1:z2]
                    _, patch_ddf = self.net(torch.cat([patch_fx, patch_mv], dim=1))
                    fused_ddf[..., x1:x2, y1:y2, z1:z2] += patch_ddf
                    count_arr[..., x1:x2, y1:y2, z1:z2] += 1

                ddf = fused_ddf / count_arr
           
            warpped_mv_img = smfunctions.warp3d(mv_img, ddf)
            warpped_mv_seg = smfunctions.warp3d(mv_seg, ddf)

            aft_dsc = loss.binary_dice(fx_seg, warpped_mv_seg)
            bef_dsc = loss.binary_dice(fx_seg, mv_seg)
            print(idx, f'mv:{mv_key}', f'fx:{fx_key}', f'BEF-DICE:{bef_dsc:.3f}', f'AFT-DICE:{aft_dsc:.3f}')
            res.append(aft_dsc)
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
        visualization_path = os.path.join(self.log_dir, f'{self.config.exp_name}-vis-{self.epoch}')
        os.makedirs(visualization_path, exist_ok=True)

        results = {
            'dice': [],
            'dice-wo-reg': [],
            'ssd': [],
            'ssd-wo-reg': [],
            'ldmk': [],
            'ldmk-wo-reg': [], 
            'cd': [],
            'cd-wo-reg': []
            }

        for idx, input_dict in enumerate(self.test_loader):
            fx_img, fx_seg, mv_img, mv_seg = self.get_input(input_dict, aug=False)
            fx_key, mv_key = input_dict['fx_key'], input_dict['mv_key']
            mv_ldmk_arrs = torch.stack([i.cuda() for i in input_dict['mv_ldmks']], dim=1)
            mv_ldmk_paths = input_dict['mv_ldmk_paths']
            fx_ldmk_arrs = torch.stack([i.cuda() for i in input_dict['fx_ldmks']], dim=1)
            fx_ldmk_paths = input_dict['fx_ldmk_paths']

            if not self.config.patched: 
                _, ddf = self.net(torch.cat([fx_img, mv_img], dim=1))
            else:
                print('DDF fusion from patches...')
                patch_coords = self.get_patch_cords_from_ref_image(mv_img)
                fused_ddf = torch.cat([torch.zeros(mv_img.shape)]*3, dim=1).cuda()
                count_arr = torch.zeros(mv_img.shape[-3:]).cuda()
                
                for (x1, x2, y1, y2, z1, z2) in patch_coords:
                    patch_mv = mv_img[..., x1:x2, y1:y2, z1:z2]
                    patch_fx = fx_img[..., x1:x2, y1:y2, z1:z2]
                    _, patch_ddf = self.net(torch.cat([patch_fx, patch_mv], dim=1))
                    fused_ddf[..., x1:x2, y1:y2, z1:z2] += patch_ddf
                    count_arr[..., x1:x2, y1:y2, z1:z2] += 1

                ddf = fused_ddf / count_arr
            
            warpped_mv_img = smfunctions.warp3d(mv_img, ddf)  # [batch=1, 1, w, h, z]
            warpped_mv_seg = smfunctions.warp3d(mv_seg, ddf)
            
            results['dice'].append(loss.binary_dice(fx_seg, warpped_mv_seg).cpu().numpy())
            results['dice-wo-reg'].append(loss.binary_dice(fx_seg, mv_seg).cpu().numpy())
            results['ssd'].append(loss.ssd(fx_img, warpped_mv_img).cpu().numpy())
            results['ssd-wo-reg'].append(loss.ssd(fx_img, mv_img).cpu().numpy())
            results['cd'].append(loss.centroid_distance(fx_seg, warpped_mv_seg).cpu().numpy())
            results['cd-wo-reg'].append(loss.centroid_distance(fx_seg, mv_seg).cpu().numpy())

            for i in range(mv_ldmk_arrs.shape[1]):
                mv_ldmk = mv_ldmk_arrs[:, i:i+1, :, :, :]
                fx_ldmk = fx_ldmk_arrs[:, i:i+1, :, :, :]

                wp_ldmk = smfunctions.warp3d(mv_ldmk, ddf)

                TRE = loss.centroid_distance(fx_ldmk, wp_ldmk).cpu().numpy()
                TRE_wo_reg = loss.centroid_distance(fx_ldmk, mv_ldmk).cpu().numpy()
                
                if not np.isnan(TRE):
                    results['ldmk'].append(TRE)
                    results['ldmk-wo-reg'].append(TRE_wo_reg)
                    

                    print(
                        f'{idx+1}-{i+1}',
                        (input_dict['fx_key'][0], input_dict['mv_key'][0]),
                        # os.path.basename(mv_ldmk_paths[i][0]), 
                        # os.path.basename(fx_ldmk_paths[i][0]),
                        'woreg:', np.around(TRE_wo_reg, decimals=3),
                        'after-reg:', np.around(TRE, decimals=3),
                        'ipmt:', np.around(TRE_wo_reg - TRE, decimals=3)
                    )
                else:
                    print(i + 1, 'warning: nan exists.')

            print('-' * 20)
            self.save_img(fx_img, os.path.join(visualization_path, f'{idx+1}-fx_img.nii'))
            self.save_img(mv_img, os.path.join(visualization_path, f'{idx+1}-mv_img.nii'))

            self.save_img(mv_seg, os.path.join(visualization_path, f'{idx+1}-mv_seg.nii'))
            self.save_img(fx_seg, os.path.join(visualization_path, f'{idx+1}-fx_seg.nii'))

            self.save_img(warpped_mv_img, os.path.join(visualization_path, f'{idx+1}-wp_img.nii'))
            self.save_img(warpped_mv_seg, os.path.join(visualization_path, f'{idx+1}-wp_seg.nii'))

            self.save_img(ddf[0, 0, :, :, :], os.path.join(visualization_path, f'{idx+1}-ddf-x.nii'))
            self.save_img(ddf[0, 1, :, :, :], os.path.join(visualization_path, f'{idx+1}-ddf-y.nii'))
            self.save_img(ddf[0, 2, :, :, :], os.path.join(visualization_path, f'{idx+1}-ddf-z.nii'))

            self.save_img(count_arr, os.path.join(visualization_path, f'{idx+1}-count.nii'))

        for k, v in results.items():
            print(k, np.mean(v), np.std(v))

        with open(os.path.join(self.log_dir, 'results.pkl'), 'wb') as f:
            pkl.dump(results, f)