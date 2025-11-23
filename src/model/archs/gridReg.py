import src.model.networks.icn as icn_models
from src.model.networks import icn_control 
import src.model.networks.icn_trans as icn_models_trans
from src.model.networks import TransMorph, VoxelMorph, keymorph
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
import psutil
from src.model.functions import apply_rigid_transform_3D
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

torch.autograd.set_detect_anomaly(True)

class icReg(BaseArch):
    def __init__(self, config):
        super(icReg, self).__init__(config)
        self.config = config
        self.net = self.net_parsing()
        self.set_dataloader()
        self.best_metric = 0
      
        
    def net_parsing(self):
        self.model = self.config.model
        self.exp_name = self.config.exp_name
        if self.model == 'ICNet':
            net = icn_models.ICNet(self.config)
        elif self.model =='ICNet_trans':
            net = icn_models_trans.ICNet(self.config)
        elif self.model =='ICNet_control':
            net = icn_control.ICNet(self.config)
        elif self.model == 'LocalEncoder':
            net = local_models.LocalEncoder(self.config)
        elif self.model == 'LocalAffine':
            net = local_models.LocalAffine(self.config)
        elif self.model == 'TransMorph':
            net = TransMorph.TransMorphTrainer(TransMorph.CONFIGS)
        elif self.model == 'VoxelMorph':
            net = VoxelMorph.Voxelmorph()
        elif self.model == 'KeyMorph':
            net = keymorph.KeyMorph(backbone='conv', num_keypoints=self.config.num_control_points, dim=3,
                                     num_layers=self.config.num_layers)    
        
        else:
            raise NotImplementedError

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
        return fx_img, fx_seg, mv_img, mv_seg
    

    def train(self):
        self.save_configure()
        print(self.net)
        total_trainable_params = sum(p.numel() for p in self.net.parameters() if p.requires_grad)
        print(f"Total Trainable Parameters: {total_trainable_params}")
        optimizer = optim.Adam(self.net.parameters(), lr=self.config.lr)

        for self.epoch in range(1, self.config.num_epochs + 1):
            self.net.train()
            print('-' * 10, f'Train epoch_{self.epoch}', '-' * 10)
            pre_reg_weight = (1-self.epoch / self.config.num_epochs)
            for self.step, input_dict in enumerate(self.train_loader):
                fx_img, fx_seg, mv_img, mv_seg = self.get_input(input_dict)
                optimizer.zero_grad()       
                
                if self.model == 'TransMorph' or self.model == 'VoxelMorph':
                    # Calculate gird-level images
                    warping_func = smfunctions.warp3d
                    gdf = self.net(torch.cat([fx_img, mv_img], dim=1))
                    grid = smfunctions.get_reference_grid3d(img=fx_img, grid_size=gdf.shape[-3:]) 
                elif self.model == 'KeyMorph':
                    warping_func = smfunctions.warp3d_v2
                    gdf = self.net(fx_img, mv_img)
                    grid = smfunctions.get_reference_grid3d(img=fx_img, grid_size=gdf.shape[-3:])
                elif self.model == 'ICNet_control':
                    warping_func = smfunctions.warp3d
                    gdf, grid = self.net(torch.cat([fx_img, mv_img], dim=1))  
                else: 
                    if 'com' in self.exp_name.lower():             
                        warping_func = smfunctions.warp3d
                        grid = smfunctions.get_reference_grid3d(img=fx_img, grid_size=self.config.grid_size)
                        gdf, grid_key = self.net(torch.cat([fx_img, mv_img], dim=1), grid)
                    if 'affine' in self.exp_name.lower():
                        warping_func = smfunctions.warp3d
                        grid = smfunctions.get_reference_grid3d(img=fx_img, grid_size=self.config.grid_size)
                        gdf, mov_affine, ddf_affine = self.net(torch.cat([fx_img, mv_img], dim=1), grid)
                        mov_affine_seg = smfunctions.warp3d_v2(mv_seg, ddf_affine)
                        pre_reg_loss = loss.ssd(mov_affine, fx_img) + loss.single_scale_dice(mov_affine_seg, fx_seg)
                    else:
                        warping_func = smfunctions.warp3d
                        grid = smfunctions.get_reference_grid3d(img=fx_img, grid_size=self.config.grid_size)
                        gdf = self.net(torch.cat([fx_img, mv_img], dim=1), grid)

                Gsample_fx_img = warping_func(fx_img, ddf=torch.zeros(grid.shape).cuda(), ref_grid=grid) 
                Gsample_fx_seg = warping_func(fx_seg, ddf=torch.zeros(grid.shape).cuda(), ref_grid=grid)

                Gsample_warpped_mv_img = warping_func(mv_img, ddf=gdf, ref_grid=grid)
                Gsample_warpped_mv_seg = warping_func(mv_seg, ddf=gdf, ref_grid=grid)
                
            
                if 'bspl' in self.exp_name.lower() or 'tconv' in self.exp_name.lower():
                    # ddf = self.transform.compute_flow(gdf)
                    ddf = F.interpolate(gdf, size=self.config.input_shape, mode='trilinear', align_corners=True)
                    warpped_mv_img = warping_func(mv_img, ddf)
                    warpped_mv_seg = warping_func(mv_seg, ddf)
                else:
                    # Calculate volumn-level images
                    ddf = F.interpolate(gdf, size=mv_img.shape[2:], mode='trilinear', align_corners=True)
                    if 'com' in self.exp_name.lower():
                        warpped_mv_img = warping_func(mv_img, ddf)
                        warpped_mv_seg = warping_func(mv_seg, ddf)
                    else:
                        warpped_mv_img = warping_func(mv_img, ddf)
                        warpped_mv_seg = warping_func(mv_seg, ddf)

                # loss functions 
                Gsample_ssd = loss.ssd(Gsample_fx_img, Gsample_warpped_mv_img) * self.config.w_Gssd
                WholeIm_ssd = loss.ssd(fx_img, warpped_mv_img) * self.config.w_Issd

                # if 'com' in self.exp_name.lower():
                #     warpped_mv_img2 = warping_func1(mv_img, grid_key)
                #     WholeIm_ssd += loss.ssd(fx_img, warpped_mv_img2) * self.config.w_Issd
                
                Gsample_dsc = loss.single_scale_dice(Gsample_fx_seg, Gsample_warpped_mv_seg) * self.config.w_Gdsc
                WholeIm_dsc = loss.single_scale_dice(fx_seg, warpped_mv_seg) * self.config.w_Idsc

                bending = loss.bending_energy(ddf) * self.config.w_bde  ##### might need change
                
                global_loss = Gsample_ssd + Gsample_dsc + WholeIm_ssd + WholeIm_dsc + bending 
                # global_loss = global_loss*(1-pre_reg_loss)+pre_reg_loss * pre_reg_weight
                # global_loss = pre_reg_loss
                global_loss.backward()

                optimizer.step()
                print(f'L_All:{global_loss:.3f}, BDE: {bending:.6f}, Gssd: {Gsample_ssd:.3f}, Gdsc: {Gsample_dsc:.3f}, Issd: {WholeIm_ssd:.3f}, Idsc: {WholeIm_dsc:.3f}')
                with open(os.path.join(self.log_dir, 'train.log'), 'a') as f:
                    f.writelines(f'L_All:{global_loss:.3f}, BDE: {bending:.6f}, Gssd: {Gsample_ssd:.3f}, Gdsc: {Gsample_dsc:.3f}, Issd: {WholeIm_ssd:.3f}, Idsc: {WholeIm_dsc:.3f}\n')
                

            if self.epoch % self.config.save_frequency == 0:
                self.save()
            print('-'*10, 'validation', '-'*10)
            with open(os.path.join(self.log_dir, 'train.log'), 'a') as f:
                f.writelines(f'-*{10}, validation, -*{10}\n')

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
        res = []
        start_time = time.time()
        for idx, input_dict in enumerate(self.val_loader):
            fx_img, fx_seg, mv_img, mv_seg = self.get_input(input_dict, aug=False)
            fx_key, mv_key = input_dict['fx_key'], input_dict['mv_key']

            if self.model == 'TransMorph' or self.model == 'VoxelMorph':
                # Calculate gird-level images
                gdf = self.net(torch.cat([fx_img, mv_img], dim=1))
                warping_func = smfunctions.warp3d
                #grid  = smfunctions.get_reference_grid3d(img=fx_img, grid_size=gdf.shape[-3:]) 
            elif self.model == 'KeyMorph':
                gdf = self.net(fx_img, mv_img)
                warping_func = smfunctions.warp3d_v2
                #grid = smfunctions.get_reference_grid3d(img=fx_img, grid_size=gdf.shape[-3:])
            elif self.model == 'ICNet_control':
                warping_func = smfunctions.warp3d
                gdf, grid = self.net(torch.cat([fx_img, mv_img], dim=1))    
           
            else: 
                if 'com' in self.exp_name.lower():
                    warping_func = smfunctions.warp3d
                    grid = smfunctions.get_reference_grid3d(img=fx_img, grid_size=self.config.grid_size)
                    gdf, grid_key = self.net(torch.cat([fx_img, mv_img], dim=1), grid)
                elif 'affine' in self.exp_name.lower():
                    warping_func = smfunctions.warp3d
                    grid = smfunctions.get_reference_grid3d(img=fx_img, grid_size=self.config.grid_size)
                    gdf, mov_affine, ddf_affine = self.net(torch.cat([fx_img, mv_img], dim=1), grid)
                    mov_affine_seg = smfunctions.warp3d_v2(mv_seg, ddf_affine)
                else:
                    warping_func = smfunctions.warp3d
                    grid = smfunctions.get_reference_grid3d(img=fx_img, grid_size=self.config.grid_size)
                    gdf = self.net(torch.cat([fx_img, mv_img], dim=1), grid)



            # Calculate volumn-level images
            if 'bspl' in self.exp_name.lower() or 'tconv' in self.exp_name.lower():
                # flow, ddf = self.transform(gdf)
                ddf = F.interpolate(gdf, size=self.config.input_shape, mode='trilinear', align_corners=True)
                warpped_mv_img = warping_func(mv_img, ddf)
                warpped_mv_seg = warping_func(mv_seg, ddf)
            else:
                # Calculate volumn-level 
                ddf = F.interpolate(gdf, size=mv_img.shape[2:], mode='trilinear', align_corners=True)
                if 'com' in self.exp_name.lower():
                    warpped_mv_img = warping_func(mv_img, ddf)
                    warpped_mv_seg = warping_func(mv_seg, ddf)
                else:
                    warpped_mv_img = warping_func(mv_img, ddf)
                    warpped_mv_seg = warping_func(mv_seg, ddf)

            aft_dsc = loss.binary_dice(fx_seg, warpped_mv_seg)
            # aft_dsc = loss.binary_dice(fx_seg, mov_affine_seg)
            bef_dsc = loss.binary_dice(fx_seg, mv_seg)

            print(idx, f'mv:{mv_key}', f'fx:{fx_key}', f'Before-DICE:{bef_dsc:.3f}', f'After-DICE:{aft_dsc:.3f}')
            res.append(aft_dsc)
     
        res = torch.tensor(res)
        mean, std = torch.mean(res), torch.std(res)
        if mean > self.best_metric:
            self.best_metric = mean
            print('better model found.')
            self.save(type='best')
        end_time = time.time()
        escape_time = end_time - start_time
        print('Dice:', mean, std, 'Best Dice:', self.best_metric, 'Escape Time:', escape_time)
        with open(os.path.join(self.log_dir, 'train.log'), 'a') as f:
            f.writelines(f'Dice:, {mean}, {std}, Best Dice:, {self.best_metric}, Escape Time:, {escape_time}\n')
    

    def monitor_memory_usage(self):
        process = psutil.Process()

        memory_info = process.memory_info()
        memory_usage = memory_info.rss / (1024.0 ** 2)  # Convert to MB
        print(f"Memory usage: {memory_usage:.2f} MB")
        time.sleep(1)  # Update every 1 second
    
    @torch.no_grad()
    def inference(self):
        time_start = time.time()
        self.net.eval()
        visualization_path = os.path.join(self.log_dir, f'{self.config.exp_name}-vis-{self.epoch}')
        os.makedirs(visualization_path, exist_ok=True)
        print(f'length of test loader {len(self.test_loader)}')
        results = {
            'dice': [],
            'dice-wo-reg': [],
            'ssd': [],
            'ssd-wo-reg': [],
            'ldmk': [],
            'ldmk-wo-reg': [], 
            'cd': [],
            'cd-wo-reg': [],
            'time': [],
            'memory': []
            }

        for idx, input_dict in enumerate(self.test_loader):
            # if idx<16:
            #     continue
            fx_img, fx_seg, mv_img, mv_seg = self.get_input(input_dict, aug=False)
            fx_key, mv_key = input_dict['fx_key'], input_dict['mv_key']
            mv_ldmk_arrs = torch.stack([i.cuda() for i in input_dict['mv_ldmks']], dim=1)
            mv_ldmk_paths = input_dict['mv_ldmk_paths']
            fx_ldmk_arrs = torch.stack([i.cuda() for i in input_dict['fx_ldmks']], dim=1)
            fx_ldmk_paths = input_dict['fx_ldmk_paths']
            if self.model == 'TransMorph' or self.model == 'VoxelMorph':
                # Calculate gird-level images
                gdf = self.net(torch.cat([fx_img, mv_img], dim=1))
                warping_func = smfunctions.warp3d
                
            elif self.model == 'KeyMorph':
                gdf = self.net(fx_img, mv_img)
                warping_func = smfunctions.warp3d_v2
                
            elif self.model == 'ICNet_control':
                warping_func = smfunctions.warp3d
                gdf, grid = self.net(torch.cat([fx_img, mv_img], dim=1))    
            else:
                if 'com' in self.exp_name.lower():
                    warping_func = smfunctions.warp3d
                    grid = smfunctions.get_reference_grid3d(img=fx_img, grid_size=self.config.grid_size)
                    gdf, grid_key = self.net(torch.cat([fx_img, mv_img], dim=1), grid)
                elif 'affine' in self.exp_name.lower():
                    warping_func = smfunctions.warp3d
                    grid = smfunctions.get_reference_grid3d(img=fx_img, grid_size=self.config.grid_size)
                    gdf, mov_affine, ddf_affine = self.net(torch.cat([fx_img, mv_img], dim=1), grid)       
                    
                else:
                    warping_func = smfunctions.warp3d
                    grid = smfunctions.get_reference_grid3d(img=fx_img, grid_size=self.config.grid_size)
                    gdf = self.net(torch.cat([fx_img, mv_img], dim=1), grid)

            # Calculate volumn-level images
            if 'bspl' in self.exp_name.lower() or 'tconv' in self.exp_name.lower():
                # flow, ddf = self.transform(gdf)
                ddf = F.interpolate(gdf, size=self.config.input_shape, mode='trilinear', align_corners=True)
                warpped_mv_img = warping_func(mv_img, ddf)
                warpped_mv_seg = warping_func(mv_seg, ddf)
            else:
                # Calculate volumn-level images
                ddf = F.interpolate(gdf, size=mv_img.shape[2:], mode='trilinear', align_corners=True)
                if 'com' in self.exp_name.lower():
                    warpped_mv_img = warping_func(mv_img, ddf)
                    warpped_mv_seg = warping_func(mv_seg, ddf)
                # print(ddf.shape, gdf.shape, mv_img.shape)
                else:
                    warpped_mv_img = warping_func(mv_img, ddf)
                    warpped_mv_seg = warping_func(mv_seg, ddf)
            os.makedirs(os.path.join(self.log_dir, 'visualization'), exist_ok=True)
            if idx == len(self.test_loader) - 1:
                memory_allocated = torch.cuda.memory_allocated() / (1024 ** 2)  # Convert to MB
                print(f"Memory allocated after forward pass: {memory_allocated:.2f} MB")
                results['memory'].append(memory_allocated)

            # warpped_mv_seg = smfunctions.warp3d_v2(mv_seg, ddf_affine)
            time_end = time.time()
            results['dice'].append(loss.binary_dice(fx_seg, warpped_mv_seg).cpu().numpy())
            results['dice-wo-reg'].append(loss.binary_dice(fx_seg, mv_seg).cpu().numpy())
            results['ssd'].append(loss.ssd(fx_img, warpped_mv_img).cpu().numpy())
            results['ssd-wo-reg'].append(loss.ssd(fx_img, mv_img).cpu().numpy())
            results['cd'].append(loss.centroid_distance(fx_seg, warpped_mv_seg).cpu().numpy())
            results['cd-wo-reg'].append(loss.centroid_distance(fx_seg, mv_seg).cpu().numpy())
            results['time'].append((time_end - time_start)/len(self.test_loader))
            
            len_ldmk = mv_ldmk_arrs.shape[1]
            for i in range(len_ldmk):
                mv_ldmk = mv_ldmk_arrs[:, i:i+1, :, :, :]
                fx_ldmk = fx_ldmk_arrs[:, i:i+1, :, :, :]
                if ddf.shape != mv_ldmk.shape[2:]:
                    ddf = F.interpolate(gdf, size=mv_ldmk.shape[2:], mode='trilinear', align_corners=True)
                wp_ldmk = warping_func(mv_ldmk, ddf)
                # plot landmarks
                
                self.plot_landmarks(fx_img, mv_img, warpped_mv_img, fx_ldmk, mv_ldmk, wp_ldmk,\
                                    fx_seg, mv_seg, warpped_mv_seg, idx, i)

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

            self.save_img(wp_ldmk, os.path.join(visualization_path, f'{idx+1}-wp_ldmk.nii'))
            self.save_img(mv_ldmk, os.path.join(visualization_path, f'{idx+1}-mv_ldmk.nii'))
            self.save_img(fx_ldmk, os.path.join(visualization_path, f'{idx+1}-fx_ldmk.nii'))

            self.save_img(ddf[0, 0, :, :, :], os.path.join(visualization_path, f'{idx+1}-ddf-x.nii'))
            self.save_img(ddf[0, 1, :, :, :], os.path.join(visualization_path, f'{idx+1}-ddf-y.nii'))
            self.save_img(ddf[0, 2, :, :, :], os.path.join(visualization_path, f'{idx+1}-ddf-z.nii'))

        
        
        with open(os.path.join(self.log_dir, 'test.log'), 'w') as f:
            for k, v in results.items():
                print(k, np.mean(v), np.std(v))
                f.writelines(f'{k}, {np.mean(v)}, {np.std(v)}\n')

        with open(os.path.join(self.log_dir, 'results.pkl'), 'wb') as f:
            pkl.dump(results, f)
    
    def plot_landmarks(self, fx_img, mv_img, wp_img, fx_ldmk, mv_ldmk, wp_ldmk, fx_seg, mv_seg, wp_seg,idx,  slice=0):
        fx_img, mv_img, wp_img, fx_ldmk, mv_ldmk, wp_ldmk = fx_img.cpu().numpy(), \
        mv_img.cpu().numpy(), wp_img.cpu().numpy(), fx_ldmk.cpu().numpy(), mv_ldmk.cpu().numpy(),\
        wp_ldmk.cpu().numpy()
        mv_seg, fx_seg, wp_seg = mv_seg.cpu().numpy(), fx_seg.cpu().numpy(), wp_seg.cpu().numpy()
        #fx_ldmk, mv_ldmk, wp_ldmk = fx_ldmk.transpose(0, 1, 3, 2, 4), mv_ldmk.transpose(0, 1, 3, 2, 4), wp_ldmk.transpose(0, 1, 3, 2, 4)
        fig, axs = plt.subplots(3, 1, figsize=(5, 10))
        non_zero_fx = np.argwhere(fx_ldmk[0, 0, :, :, :])
        non_zero_mv = np.argwhere(mv_ldmk[0, 0, :, :, :])
        non_zero_wp = np.argwhere(wp_ldmk[0, 0, :, :, :])
        # Plot the landmarks
        mid = non_zero_fx.shape[-1]//2
        x,y,z = fx_img.shape[2:]
        axs[0].imshow(np.rot90(fx_img[0, 0, :, :, non_zero_fx[mid, 2]]), cmap='gray')
        axs[0].scatter(non_zero_fx[mid, 0], y-non_zero_fx[mid, 1], c='r', label='Fixed', linewidths=2)
        axs[0].contour(np.rot90(fx_seg[0, 0, :, :, non_zero_fx[mid, 2]]), cmap='Reds', alpha=0.5, linewidths=2)
        
        # axs[0].invert_yaxis()
        mid = non_zero_mv.shape[-1]//2
        axs[1].imshow(np.rot90(mv_img[0, 0, :, :, non_zero_mv[mid, 2]]), cmap='gray')
        axs[1].scatter(non_zero_mv[mid, 0], y-non_zero_mv[mid, 1], c='b', label='Moving', linewidths=2)
        axs[1].contour(np.rot90(mv_seg[0, 0, :, :, non_zero_mv[mid, 2]]), cmap='Blues', alpha=0.5, linewidths=2)
        # axs[1].invert_yaxis()
        mid = non_zero_wp.shape[0]//2
        if mid != 0:   
            axs[2].imshow(np.rot90(wp_img[0, 0, :, :, non_zero_wp[mid, 2]]), cmap='gray')
            axs[2].scatter(non_zero_wp[mid, 0], y-non_zero_wp[mid, 1], c='g', label='Warpped', linewidths=2)
            axs[2].contour(np.rot90(wp_seg[0, 0, :, :, non_zero_wp[mid, 2]]), cmap='Greens', alpha=0.5, linewidths=2)
        # axs[2].invert_yaxis()
        # axs[0].set_title('Fixed Image')
        # axs[1].set_title('Moving Image')
        # axs[2].set_title('Warpped Image')
        axs[0].set_axis_off()
        axs[1].set_axis_off()
        axs[2].set_axis_off()
        path = os.path.join(self.log_dir, 'visualization',  f'{idx}-{slice}-landmarks.png')
        plt.savefig(path)
        plt.close()
