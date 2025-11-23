import src.model.networks.icn as icn_models
import src.model.networks.icn_trans as icn_models_trans
from src.model.networks import  Unet
import matplotlib.pyplot as plt
import src.model.networks.local as local_models 
from src.model import loss
import src.model.functions as smfunctions
from src.model.archs.baseArch import BaseArch
from src.model.networks import transforms
from src.data import dataloaders
import torch, os
import torch.optim as optim
from torch.utils.data import DataLoader
import pickle as pkl
import numpy as np
from scipy import stats
import torch.nn.functional as F
from scipy.ndimage import zoom
import time 
from skimage.metrics import peak_signal_noise_ratio as psnr
from tqdm import tqdm
from src.model.functions import ssim3D, SSIM3D

class denoise(BaseArch):
    def __init__(self, config):
        super(denoise, self).__init__(config)
        self.config = config
        self.net = self.net_parsing()
        self.last_layer = torch.nn.Conv3d(17, 1, kernel_size=3, padding=1)
        self.set_dataloader()
        self.best_metric = 0
        self.loss = torch.nn.MSELoss()
        self.ssim_loss = SSIM3D(window_size=11)
       
        
    def net_parsing(self):
        self.model = self.config.model
        self.exp_name = self.config.exp_name
       
        net = Unet.UNet3D(in_channels=1, out_channels=1)
        
        return net.cuda()

        
    def set_dataloader(self):
        self.train_set = dataloaders.LongitudinalData(config=self.config, phase='train')
        self.train_loader = DataLoader(
            self.train_set, 
            batch_size=self.config.batch_size, 
            shuffle=False,  
            num_workers=4, 
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
            fx_seg = torch.nn.functional.grid_sample(fx_seg, mv_affine_grid, mode='bilinear', align_corners=True)
        else:
            pass
        #return fx_img, fx_seg, mv_img, mv_seg
        return torch.cat([fx_img, mv_img], dim=0), torch.cat([fx_seg, mv_seg], dim=0)
    
    

    def train(self):
        self.losses=0
        self.save_configure()
        total_trainable_params = sum(p.numel() for p in self.net.parameters() if p.requires_grad)
        print(f"Total Trainable Parameters: {total_trainable_params}")
        optimizer = optim.Adam(self.net.parameters(), lr=self.config.lr)
        
        for self.epoch in range(1, self.config.num_epochs + 1):
            self.net.train()
            print('-' * 10, f'Train epoch_{self.epoch}', '-' * 10)
            
            with tqdm(self.train_loader, desc=f'Epoch {self.epoch}') as pbar:
                for self.step, input_dict in enumerate(pbar):
                    fx_img, fx_seg = self.get_input(input_dict)
                    optimizer.zero_grad()       
                
               
                    # Calculate gird-level images
                    noise = torch.randn(fx_img.shape).cuda() * self.config.noise_std * np.sqrt(self.config.noise_var)
                    denoise_img = self.net(fx_img+noise)
                
                    mse_loss = self.loss(denoise_img, fx_img) 
                    ssim_loss = self.ssim_loss(denoise_img, fx_img)
                    loss = mse_loss + ssim_loss
                    loss.backward()

                    optimizer.step()
                    self.losses += loss.item()

                    pbar.set_postfix({'MSE Loss:': mse_loss.item(), 'SSIM Loss:': ssim_loss.item()})
            loss_mean = self.losses / len(self.train_loader)
            # print(f'Loss:{loss_mean:.3f}')
            pbar.set_postfix({'Mean Loss:': loss_mean})
            if self.epoch % self.config.save_frequency == 0:
                self.save()
            print('-'*10, 'validation', '-'*10)
            self.validation()


    # def loss(self, ddf, fx_img, wp_mv_img, fx_seg, wp_mv_seg):
    #     L_bending = loss.normalized_bending_energy(
    #         ddf, 
    #         self.config.input_shape, 
    #         self.config.voxel_size) * self.config.w_bde
    #     L_ssd = loss.ssd(fx_img, wp_mv_img) * self.config.w_ssd
    #     L_dice = loss.single_scale_dice(fx_seg, wp_mv_seg) * self.config.w_dce
    #     L_All = L_bending + L_ssd + L_dice
    #     Info = f'L_All:{L_All:.3f}, Loss_Dreg: {L_bending:.6f}, Loss_ssd: {L_ssd:.3f}, Loss_dice: {L_dice:.3f}'
    #     print(Info)

    #     return L_All

    @torch.no_grad()
    def validation(self):
        self.net.eval()
        self.best_metric = 1
        ssim_values = 0
        start_time = time.time()

        with tqdm(self.val_loader, desc=f'Epoch {self.epoch}') as pbar:
            for idx, input_dict in enumerate(pbar):
                fx_img, fx_seg = self.get_input(input_dict, aug=False)
                #fx_key, mv_key = input_dict['fx_key'], input_dict['mv_key']

                noise = torch.randn(fx_img.shape).cuda() * self.config.noise_std * np.sqrt(self.config.noise_var)
                denoise_img = self.net(fx_img+noise)

                loss = self.loss(denoise_img, fx_img) 
                ssim_value = ssim3D(denoise_img, fx_img)
                ssim_values += ssim_value
                self.best_metric += loss
                pbar.set_postfix({'loss:': loss, 'ssim:': ssim_value})
            self.best_metric = self.best_metric / len(self.val_loader)  
            end_time = time.time()
            escape_time = end_time - start_time
            print('mean loss:', self.best_metric, 'mean ssim:', ssim_values/len(self.val_loader), 'Escape Time:', escape_time)

        if loss < self.best_metric:
            self.best_metric = loss
            print('better model found.')
            self.save(type='best')
            plt.figure()
            plt.subplot(1,2,1)
            plt.imshow(denoise_img.cpu().detach().numpy()[0,0,:,:,45])
            plt.subplot(1,2,2)
            plt.imshow(fx_img.cpu().detach().numpy()[0,0,:,:,45])
            plt.savefig(os.path.join(self.log_dir, f'{self.config.exp_name}-vis-{self.epoch}.png'))

        

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
            if self.model == 'TransMorph' or self.model == 'VoxelMorph':
                # Calculate gird-level images
                gdf = self.net(torch.cat([fx_img, mv_img], dim=1))
                grid = grid = smfunctions.get_reference_grid3d(img=fx_img, grid_size=gdf.shape[-3:])     
            else:
                grid = smfunctions.get_reference_grid3d(img=fx_img, grid_size=self.config.grid_size)
                gdf = self.net(torch.cat([fx_img, mv_img], dim=1), grid)
            
            if 'bspl' in self.exp_name.lower():
                flow, ddf = self.transform(gdf)
                ddf = F.interpolate(ddf, size=self.config.input_shape, mode='trilinear', align_corners=True)
                warpped_mv_img = transforms.warp(mv_img, ddf)
                warpped_mv_seg = transforms.warp(mv_seg, ddf)
            else:
                # Calculate volumn-level images
                ddf = F.interpolate(gdf, size=self.config.input_shape, mode='trilinear', align_corners=True)
                warpped_mv_img = smfunctions.warp3d(mv_img, ddf)
                warpped_mv_seg = smfunctions.warp3d(mv_seg, ddf)
            # Calculate volumn-level images
            #ddf = F.interpolate(gdf, size=self.config.input_shape, mode='trilinear', align_corners=True)
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

        for k, v in results.items():
            print(k, np.mean(v), np.std(v))

        with open(os.path.join(self.log_dir, 'results.pkl'), 'wb') as f:
            pkl.dump(results, f)
