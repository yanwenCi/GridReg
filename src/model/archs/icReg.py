import src.model.networks.icn as icn_models
from src.model.networks import icn_control 
import src.model.networks.icn_trans as icn_models_trans
import src.model.networks.icn_token as icn_token
from src.model.networks import TransMorph, VoxelMorph, keymorph
import src.model.networks.local as local_models 
from src.model import loss
import src.model.functions as smfunctions
from src.model.archs.baseArch import BaseArch
from src.model.networks import transforms
from src.data import dataloader_sam
from src.data import dataloaders
from src.data import dataloaders_brain as dataloaders_brain
from src.data import dataloaders_atlas as dataloaders_atlas
from src.data import dataloaders_pelvis as dataloaders_pelvis
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
import cv2
from scipy.ndimage import label, binary_dilation, generate_binary_structure
torch.autograd.set_detect_anomaly(True)

class icReg(BaseArch):
    def __init__(self, config):
        super(icReg, self).__init__(config)
        self.config = config
        self.net = self.net_parsing()
        # self.set_dataloader()
        if 'sam' in self.config.exp_name.lower():
            self.set_dataloader_multi()
        elif 'brain' in self.config.exp_name.lower():
            self.set_dataloader(dataloaders_brain)
        elif 'atlas' in self.config.exp_name.lower():
            self.set_dataloader(dataloaders_atlas)
        elif 'pelvis' in self.config.exp_name.lower():
            self.set_dataloader(dataloaders_pelvis)
        else: 
            self.set_dataloader(dataloaders)
        self.best_metric = 0
      
        
    def net_parsing(self):
        self.model = self.config.model
        self.exp_name = self.config.exp_name
        if self.model == 'ICNet':
            net = icn_models.ICNet(self.config)
        elif self.model == 'ICNet_auto':
            from src.model.networks import icn_auto
            net = icn_auto.ICNet(self.config)
        elif self.model =='trans':
            net = icn_models_trans.ICNet(self.config)
        elif self.model == 'token':
            net = icn_token.ICNet(self.config)
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

    def data_parallel(self, model, input, grid, gpu_ids):
        if len(gpu_ids) == 1:
            return model(input, grid)
        else:
            return torch.nn.DataParallel(model)(input)

    def set_dataloader(self, dataloaders=dataloaders):
        self.train_set = dataloaders.LongitudinalData(config=self.config, phase='train')
        self.train_loader = DataLoader(
            self.train_set, 
            batch_size=self.config.batch_size, 
            shuffle=False,  
            num_workers=4, 
            drop_last=True)  # no need to shuffle since the shuffling is customized in the dataloader.
        print('>>> Train set ready. length:', len(self.train_loader)*self.config.batch_size)  
        self.val_set = dataloaders.LongitudinalData(config=self.config, phase='val')
        self.val_loader = DataLoader(self.val_set, batch_size=1, shuffle=False)
        print('>>> Validation set ready. length:', len(self.val_loader))
        self.test_set = dataloaders.LongitudinalData(config=self.config, phase='test')
        self.test_loader = DataLoader(self.test_set, batch_size=1, shuffle=False)
        print('>>> Holdout set ready. length:', len(self.test_loader))

    def set_dataloader_multi(self):
        self.train_set = dataloader_sam.dataset_loaders(self.config.data_path,
                         'train', batch_size=self.config.batch_size, crop_size=self.config.crop_size)
        self.train_loader = DataLoader(self.train_set, batch_size=self.config.batch_size, 
                                       shuffle=False, drop_last=True)
        print('>>> Train set ready. length:', len(self.train_loader))
        self.val_set = dataloader_sam.dataset_loaders(self.config.data_path,
                       'valid', batch_size=self.config.batch_size, crop_size=self.config.crop_size)
        self.val_loader = DataLoader(self.val_set, batch_size=1, shuffle=False)
        print('>>> Validation set ready. length:', len(self.val_loader))
        self.test_set = dataloader_sam.dataset_loaders(self.config.data_path,
                        'test', batch_size=self.config.batch_size, crop_size=self.config.crop_size)
        self.test_loader = DataLoader(self.test_set, batch_size=1, shuffle=False)
        print('>>> Holdout set ready. length:', len(self.test_loader))

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
    
    def get_model_size(self):
        param_size = 0
        for param in self.net.parameters():
            param_size += param.nelement() * param.element_size()  # num_elements * bytes_per_element
        buffer_size = 0
        for buffer in self.net.buffers():  # includes non-trainable parameters like BatchNorm stats
            buffer_size += buffer.nelement() * buffer.element_size()
        total_size = param_size + buffer_size
        return total_size  # Convert to MB


    def train(self):
        self.save_configure()
        # print(self.net)
        print('model size: ', self.get_model_size())
        total_trainable_params = sum(p.numel() for p in self.net.parameters() if p.requires_grad)
        print(f"Total Trainable Parameters: {total_trainable_params}")
        optimizer = optim.Adam(self.net.parameters(), lr=self.config.lr)
        sim_loss = loss.NCC()
        # sim_loss = loss.ssd
        # if 'bspl' in self.exp_name.lower():
        #     self.transform = transforms.CubicBSplineTransform(
        #         img_size=self.config.input_shape,
        #         cps_spacing=self.config.cps_spacing
        #     ).cuda()
        # elif 'tconv' in self.exp_name.lower():
        #     self.transform = transforms.bspline_upsample_tconv
        # atlas_ = np.load('/raid/candi/Wen/segment-anything/atlases/atlas0.npz')
        # atlas, atlas_masks = atlas_['atlas'], atlas_['masks']
        # fx_img = atlas[None, None,...].repeat(self.config.batch_size, axis=0)
        # fx_seg = atlas_masks[None, None, ...].repeat(self.config.batch_size,axis=0)
        # fx_img, fx_seg = torch.from_numpy(fx_img).cuda().float(), torch.from_numpy(fx_seg).double().cuda().float()
        # fx_seg[fx_seg>0] = 1
        for self.epoch in range(1, self.config.num_epochs + 1):
            self.net.train()
            print('-' * 10, f'Train epoch_{self.epoch}', '-' * 10)
            pre_reg_weight = (1-self.epoch / self.config.num_epochs)
            for self.step, input_dict in enumerate(self.train_loader):
                # if self.step >1:
                #     continue
                fx_img, fx_seg, mv_img, mv_seg = self.get_input(input_dict)
                # mv_img, mv_seg = input_dict['mv_img'].cuda().float(), input_dict['mv_seg'].cuda().float()
                # mv_seg[mv_seg>0] = 1
                optimizer.zero_grad()       
                # print(fx_img.shape, mv_img.shape)
                # cv2.imwrite('fx_img2.png', fx_img[0, 0, :, 80, :].cpu().numpy()*255)
                # cv2.imwrite('fx_img3.png', fx_img[0, 0, :, :, 80].cpu().numpy()*255)
                # cv2.imwrite('fx_img1.png', fx_img[0, 0, 80, :, :].cpu().numpy()*255)
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
                    # grid_scale = np.random.choice([5, 10, 15, 20])
                    # grid_size = [grid_scale]*3
                    grid = smfunctions.get_reference_grid3d(img=fx_img, grid_size=self.config.grid_size)
                    if 'com' in self.exp_name.lower():             
                        warping_func = smfunctions.warp3d
                        # grid = smfunctions.get_reference_grid3d(img=fx_img, grid_size=self.config.grid_size)
                        gdf, grid_key = self.net(torch.cat([fx_img, mv_img], dim=1), grid)
                    if 'affine' in self.exp_name.lower():
                        warping_func = smfunctions.warp3d
                        # grid = smfunctions.get_reference_grid3d(img=fx_img, grid_size=self.config.grid_size)
                        gdf, mov_affine, ddf_affine = self.net(torch.cat([fx_img, mv_img], dim=1), grid)
                        mov_affine_seg = smfunctions.warp3d_v2(mv_seg, ddf_affine)
                        pre_reg_loss = loss.ssd(mov_affine, fx_img) + loss.single_scale_dice(mov_affine_seg, fx_seg)
                    elif 'tconv' in self.exp_name.lower():
                        warping_func = smfunctions.warp3d
                        gdf = self.net(torch.cat([fx_img, mv_img], dim=1), grid)
                        grid = smfunctions.get_reference_grid3d(img=fx_img, grid_size=gdf.shape[-3:])
                    elif 'auto' in self.exp_name.lower():
                        grid_size = [5, 8, 10, 15]
                        grid_scale = np.random.choice(grid_size)
                        grid = smfunctions.get_reference_grid3d(img=fx_img, grid_size=[grid_scale]*3)
                        warping_func = smfunctions.warp3d
                        gdf = self.net(torch.cat([fx_img, mv_img], dim=1), grid)
                    else:
                        warping_func = smfunctions.warp3d
                        # grid = smfunctions.get_reference_grid3d(img=fx_img, grid_size=self.config.grid_size)
                        gdf = self.net(torch.cat([fx_img, mv_img], dim=1), grid)
                        
                        # gdf = self.data_parallel(self.net, torch.cat([fx_img, mv_img], dim=1), grid, self.config.gpu_ids)
                Gsample_fx_img = warping_func(fx_img, ddf=torch.zeros(grid.shape).cuda(), ref_grid=grid) 
                Gsample_fx_seg = warping_func(fx_seg, ddf=torch.zeros(grid.shape).cuda(), ref_grid=grid)

                Gsample_warpped_mv_img = warping_func(mv_img, ddf=gdf, ref_grid=grid)
                Gsample_warpped_mv_seg = warping_func(mv_seg, ddf=gdf, ref_grid=grid)
                
            
                # if 'bspl' in self.exp_name.lower() or 'tconv' in self.exp_name.lower():
                #     # ddf = self.transform(gdf, out_size_xyz=self.config.input_shape)
                #     ddf = F.interpolate(gdf, size=self.config.input_shape, mode='trilinear', align_corners=True)
                #     warpped_mv_img = warping_func(mv_img, ddf)
                #     warpped_mv_seg = warping_func(mv_seg, ddf)

                # else:
                #     # Calculate volumn-level images
                if isinstance(gdf, tuple):
                    ddf = [None, None]
                    for i in range(len(gdf)):
                        ddf[i] = F.interpolate(gdf[i], size=mv_img.shape[2:], mode='trilinear', align_corners=True)
                    ddf = tuple(ddf)
                else:
                    ddf = F.interpolate(gdf, size=mv_img.shape[2:], mode='trilinear', align_corners=True)
                if 'com' in self.exp_name.lower():
                    warpped_mv_img = warping_func(mv_img, ddf)
                    warpped_mv_seg = warping_func(mv_seg, ddf)
                else:
                    warpped_mv_img = warping_func(mv_img, ddf)
                    warpped_mv_seg = warping_func(mv_seg, ddf)

                # if 'com' in self.exp_name.lower():
                #     warpped_mv_img2 = warping_func1(mv_img, grid_key)
                #     WholeIm_ssd += loss.ssd(fx_img, warpped_mv_img2) * self.config.w_Issd
                
                Gsample_dsc = 0#loss.dice_loss(Gsample_fx_seg, Gsample_warpped_mv_seg) * self.config.w_Gdsc
                # print(fx_seg.max(), warpped_mv_seg.max())
                WholeIm_dsc = loss.reg_dice_loss(fx_seg, warpped_mv_seg) * self.config.w_Idsc
                # Gsample_dsc = 0
                # WholeIm_dsc = 0
                
                if self.config.uncertainty:
                    Gsample_ssd = loss.ssd(Gsample_fx_img, Gsample_warpped_mv_img, gdf[0], gdf[1]) * self.config.w_Gssd
                    WholeIm_ssd = loss.ssd(fx_img, warpped_mv_img, ddf[0], ddf[1]) * self.config.w_Issd
                    bending = loss.bending_energy(ddf[0]) * self.config.w_bde  ##### might need change
                    
                else:
                    Gsample_ssd = 0#sim_loss(Gsample_fx_img, Gsample_warpped_mv_img) * self.config.w_Gssd
                    WholeIm_ssd = sim_loss.loss(fx_img, warpped_mv_img) * self.config.w_Issd
                    bending = loss.bending_energy(ddf) * self.config.w_bde  ##### might need change
                
                global_loss = Gsample_ssd + Gsample_dsc + WholeIm_ssd + WholeIm_dsc + bending 
                    
                # global_loss = global_loss*(1-pre_reg_loss)+pre_reg_loss * pre_reg_weight
                # global_loss = pre_reg_loss
                # global_loss = WholeIm_ssd + bending + Gsample_ssd
                global_loss.backward()

                optimizer.step()
                if self.step%100==0:
                    print(f'L_All:{global_loss:.3f}, BDE: {bending:.6f}, Gssd: {Gsample_ssd:.3f}, Gdsc: {Gsample_dsc:.3f}, Issd: {WholeIm_ssd:.3f}, Idsc: {WholeIm_dsc:.3f}')
                # with open(os.path.join(self.log_dir, 'train.log'), 'a') as f:
                #     f.writelines(f'L_All:{global_loss:.3f}, BDE: {bending:.6f}, Gssd: {Gsample_ssd:.3f}, Gdsc: {Gsample_dsc:.3f}, Issd: {WholeIm_ssd:.3f}, Idsc: {WholeIm_dsc:.3f} \n')
                

            if self.epoch % self.config.save_frequency == 0:
                self.save()
            print('-'*10, 'validation', '-'*10)
            with open(os.path.join(self.log_dir, 'train.log'), 'a') as f:
                f.writelines(f'-*{10}, validation, -*{10}\n')

            self.validation()




    @torch.no_grad()
    def validation(self):
        self.net.eval()
        # print(f"model size: {self.get_model_size() }")
        res = []
        start_time = time.time()
        # atlas_ = np.load('/raid/candi/Wen/segment-anything/atlases/atlas0.npz')
        # atlas, atlas_masks = atlas_['atlas'], atlas_['masks']
        # fx_img = atlas[None, None,...]
        # fx_seg = atlas_masks[None, None, ...]
        # fx_img, fx_seg = torch.from_numpy(fx_img).cuda().float(), torch.from_numpy(fx_seg).double().cuda().float()
        # fx_seg[fx_seg>0] = 1
        # fx_key = 'atlas'
        before_dice_sum = 0
        after_dice_sum = 0
        for idx, input_dict in enumerate(self.val_loader):
            if idx>50:
                break
            fx_img, fx_seg, mv_img, mv_seg = self.get_input(input_dict, aug=False)
            fx_key, mv_key = input_dict['fx_key'], input_dict['mv_key']
            mv_img, mv_seg = input_dict['mv_img'].cuda().float(), input_dict['mv_seg'].cuda().float()
            # mv_seg[mv_seg>0] = 1
            # fx_seg[fx_seg>0] = 1
            mv_key = input_dict['mv_key']
            
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
                grid = smfunctions.get_reference_grid3d(img=fx_img, grid_size=self.config.grid_size)
                if 'com' in self.exp_name.lower():
                    warping_func = smfunctions.warp3d
                    # grid = smfunctions.get_reference_grid3d(img=fx_img, grid_size=self.config.grid_size)
                    gdf, grid_key = self.net(torch.cat([fx_img, mv_img], dim=1), grid)
                elif 'affine' in self.exp_name.lower():
                    warping_func = smfunctions.warp3d
                    # grid = smfunctions.get_reference_grid3d(img=fx_img, grid_size=self.config.grid_size)
                    gdf, mov_affine, ddf_affine = self.net(torch.cat([fx_img, mv_img], dim=1), grid)
                    mov_affine_seg = smfunctions.warp3d_v2(mv_seg, ddf_affine)
                elif 't_conv' in self.exp_name.lower():
                    warping_func = smfunctions.warp3d
                    gdf = self.net(torch.cat([fx_img, mv_img], dim=1), grid)
                    grid = smfunctions.get_reference_grid3d(img=fx_img, grid_size=gdf.shape[-3:])
                elif 'auto' in self.exp_name.lower():
                    grid_size = 5
                    grid = smfunctions.get_reference_grid3d(img=fx_img, grid_size=[grid_size]*3)
                    warping_func = smfunctions.warp3d
                    gdf = self.net(torch.cat([fx_img, mv_img], dim=1), grid)
                else:
                    warping_func = smfunctions.warp3d
                    grid = smfunctions.get_reference_grid3d(img=fx_img, grid_size=self.config.grid_size)
                    if self.config.uncertainty:
                        gdf = smfunctions.sample_deformation_field(self.net, torch.cat([fx_img, mv_img], dim=1), grid, 
                                                                   self.config.num_samples, idx=idx, plot=False)
                    else: # original grid network  
                        gdf = self.net(torch.cat([fx_img, mv_img], dim=1), grid)



            # Calculate volumn-level images
            # if 'bspl' in self.exp_name.lower() or 'tconv' in self.exp_name.lower():
            #     flow, ddf = self.transform(gdf)
            #     # ddf = F.interpolate(gdf, size=self.config.input_shape, mode='trilinear', align_corners=True)
            #     warpped_mv_img = warping_func(mv_img, ddf)
            #     warpped_mv_seg = warping_func(mv_seg, ddf)
            # else:
                # Calculate volumn-level 
            if isinstance(gdf, tuple) or self.config.uncertainty:
                ddf = [None, None]
                for i in range(len(gdf)):
                    ddf[i] = F.interpolate(gdf[i], size=mv_img.shape[2:], mode='trilinear', align_corners=True)
                    ddf = tuple(ddf)
            else:
                ddf = F.interpolate(gdf, size=mv_img.shape[2:], mode='trilinear', align_corners=True)
            if 'com' in self.exp_name.lower():
                warpped_mv_img = warping_func(mv_img, ddf)
                warpped_mv_seg = warping_func(mv_seg, ddf)
            else:
                warpped_mv_img = warping_func(mv_img, ddf)
                warpped_mv_seg = warping_func(mv_seg, ddf)

            aft_dsc = 1-loss.reg_dice_loss(fx_seg , warpped_mv_seg )
            # aft_dsc = loss.binary_dice(fx_seg, mov_affine_seg)
            bef_dsc = 1-loss.reg_dice_loss(fx_seg , mv_seg )
            before_dice_sum += bef_dsc
            
            # print(idx, f'mv:{mv_key}', f'fx:{fx_key}', f'Before-DICE:{bef_dsc:.3f}', f'After-DICE:{aft_dsc:.3f}')
            res.append(aft_dsc)
     
        res = torch.tensor(res)
        mean, std = torch.mean(res), torch.std(res)
        if mean > self.best_metric:
            self.best_metric = mean
            print('better model found.')
            self.save(type='best')
        end_time = time.time()
        escape_time = end_time - start_time
        print('Before-Dice:', before_dice_sum/len(self.val_loader), 'After-Dice:', mean, std, 'Best Dice:', self.best_metric, 'Escape Time:', escape_time)
        with open(os.path.join(self.log_dir, 'train.log'), 'a') as f:
            f.writelines(f'Dice:, {mean}, {std}, Best Dice:, {self.best_metric}, Escape Time:, {escape_time}\n')
    
    def monitor_memory_usage(self,  device='cuda'):
        self.net.to(device)  # Move model to GPU
        param_mem = sum(p.nelement() * p.element_size() for p in self.net.parameters())
        buffer_mem = sum(b.nelement() * b.element_size() for b in self.net.buffers())
        memo=param_mem + buffer_mem  # Bytes
        return memo/ (1024**2)


    
    @torch.no_grad()
    def inference(self):
        time_start = time.time()
        # atlas_ = np.load('/raid/candi/Wen/segment-anything/atlases/atlas0.npz')
        # atlas, atlas_masks = atlas_['atlas'], atlas_['masks']
        # fx_img = atlas[None, None,...]
        # fx_seg = atlas_masks[None, None, ...]
        # fx_img, fx_seg = torch.from_numpy(fx_img).cuda().float(), torch.from_numpy(fx_seg).double().cuda().float()
        # fx_seg[fx_seg>0] = 1
        self.net.eval()
        print('model size: ', self.get_model_size())
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
            'memory': [],
            'jac_det': []
            }

        for idx, input_dict in enumerate(self.test_loader):
            if idx>50:
                break
            # mv_img, mv_seg = input_dict['mv_img'].cuda().float(), input_dict['mv_seg'].cuda().float()
            # mv_seg[mv_seg>0] = 1
            mv_key = input_dict['mv_key']
            # fx_key = 'atlas'
            fx_img, fx_seg, mv_img, mv_seg = self.get_input(input_dict, aug=False)
            # fx_seg[fx_seg>0] = 1
            # mv_seg[mv_seg>0] = 1
            fx_key, mv_key = input_dict['fx_key'], input_dict['mv_key']

            if input_dict.get('mv_ldmks') is not None:
                mv_ldmk_arrs = torch.stack([i.cuda() for i in input_dict['mv_ldmks']], dim=1)
                mv_ldmk_paths = input_dict['mv_ldmk_paths']
                fx_ldmk_arrs = torch.stack([i.cuda() for i in input_dict['fx_ldmks']], dim=1)
                fx_ldmk_paths = input_dict['fx_ldmk_paths']
                len_ldmk = mv_ldmk_arrs.shape[1]

            memory_allocated0 = torch.cuda.memory_allocated() / (1024 ** 2)  # Convert to MB
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
                elif 't_conv' in self.exp_name.lower():
                    warping_func = smfunctions.warp3d
                    gdf = self.net(torch.cat([fx_img, mv_img], dim=1), grid)
                    grid = smfunctions.get_reference_grid3d(img=fx_img, grid_size=gdf.shape[-3:])   
                elif 'auto' in self.exp_name.lower():
                    grid_size = 5
                    print(f'auto grid size: {grid_size}')
                    grid = smfunctions.get_reference_grid3d(img=fx_img, grid_size=[grid_size]*3)
                    warping_func = smfunctions.warp3d
                    gdf = self.net(torch.cat([fx_img, mv_img], dim=1), grid) 
                else:
                    warping_func = smfunctions.warp3d
                    grid = smfunctions.get_reference_grid3d(img=fx_img, grid_size=self.config.grid_size)
                    # gdf = self.net(torch.cat([fx_img, mv_img], dim=1), grid)
                    if self.config.uncertainty:
                        gdf, sigma = smfunctions.sample_deformation_field(self.net, torch.cat([fx_img, mv_img], dim=1), 
                                                                          grid, self.config.num_samples,  idx=idx)
                    else: # original grid network
                        gdf = self.net(torch.cat([fx_img, mv_img], dim=1), grid)

            # Calculate volumn-level images
            jac_det = smfunctions.jacobian_det_3d(gdf, spacing=[0.7, .7,.7],) 
            # if 'bspl' in self.exp_name.lower() or 'tconv' in self.exp_name.lower():
            #     # flow, ddf = self.transform(gdf)
            #     ddf = F.interpolate(gdf, size=self.config.input_shape, mode='trilinear', align_corners=True)
            #     warpped_mv_img = warping_func(mv_img, ddf)
            #     warpped_mv_seg = warping_func(mv_seg, ddf)
            # else:
            if isinstance(gdf, tuple) :
                ddf = [None, None]
                for i in range(len(gdf)):
                    ddf[i] = F.interpolate(gdf[i], size=mv_img.shape[2:], mode='trilinear', align_corners=True)
                ddf = tuple(ddf)
            else:
                ddf = F.interpolate(gdf, size=mv_img.shape[2:], mode='trilinear', align_corners=True)
                

            warpped_mv_img = warping_func(mv_img, ddf)
            warpped_mv_seg = warping_func(mv_seg, ddf)
            
            os.makedirs(os.path.join(self.log_dir, 'visualization'), exist_ok=True)
            if idx == 1:
                memory_allocated = torch.cuda.memory_allocated() / (1024 ** 2)  # Convert to MB
                print(f"Memory allocated after forward pass: {memory_allocated:.2f} MB")
                results['memory'].append(memory_allocated)

            # warpped_mv_seg = smfunctions.warp3d_v2(mv_seg, ddf_affine)
            time_end = time.time()
            results['time'].append((time_end - time_start)/len(self.test_loader))
            # print(fx_seg.shape, warpped_mv_seg.shape, fx_seg.max(), warpped_mv_seg.max())
            results['dice'].append(1-loss.reg_dice_loss(fx_seg, warpped_mv_seg).cpu().numpy())
            results['dice-wo-reg'].append(1-loss.reg_dice_loss(fx_seg, mv_seg).cpu().numpy())
            results['ssd'].append(loss.ssd(fx_img, warpped_mv_img).cpu().numpy())
            results['ssd-wo-reg'].append(loss.ssd(fx_img, mv_img).cpu().numpy())
            results['cd'].append(loss.centroid_distance(fx_seg, warpped_mv_seg).cpu().numpy())
            results['cd-wo-reg'].append(loss.centroid_distance(fx_seg, mv_seg).cpu().numpy())
            results['ldmk'].append(loss.NCC().loss(fx_img, warpped_mv_img).cpu().numpy())
            results['ldmk-wo-reg'].append(loss.NCC().loss(fx_img, mv_img).cpu().numpy())
            results['jac_det'].append(smfunctions.jacobian_smaller_than_threshold(jac_det.cpu().numpy()))
            

            for i in range(len_ldmk):
                mv_ldmk = mv_ldmk_arrs[:, i:i+1, :, :, :]
                fx_ldmk = fx_ldmk_arrs[:, i:i+1, :, :, :]
                if isinstance(gdf, tuple):
                    ddf = ddf[0]
                    gdf = gdf[0]
                    # print(ddf.shape, gdf.shape, mv_ldmk.shape)
                if ddf.shape[1:] != mv_ldmk.shape[2:]:
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
                        'ipmt:', np.around(TRE_wo_reg - TRE, decimals=3),
                        'jac_det:', np.around(jac_det.cpu().numpy().mean(), decimals=3)
                    )
                else:
                    print(i + 1, 'warning: nan exists.')

            print('-' * 20)
         
            # self.save_img(fx_img, os.path.join(visualization_path, f'{idx+1}-fx_img.nii'))
            # self.save_img(mv_img, os.path.join(visualization_path, f'{idx+1}-mv_img.nii'))

            # self.save_img(mv_seg, os.path.join(visualization_path, f'{idx+1}-mv_seg.nii'))
            # self.save_img(fx_seg, os.path.join(visualization_path, f'{idx+1}-fx_seg.nii'))

            # self.save_img(warpped_mv_img, os.path.join(visualization_path, f'{idx+1}-wp_img.nii'))
            # self.save_img(warpped_mv_seg, os.path.join(visualization_path, f'{idx+1}-wp_seg.nii'))

            # self.save_img(wp_ldmk, os.path.join(visualization_path, f'{idx+1}-wp_ldmk.nii'))
            # self.save_img(mv_ldmk, os.path.join(visualization_path, f'{idx+1}-mv_ldmk.nii'))
            # self.save_img(fx_ldmk, os.path.join(visualization_path, f'{idx+1}-fx_ldmk.nii'))

            # self.save_img(ddf[0, 0, :, :, :], os.path.join(visualization_path, f'{idx+1}-ddf-x.nii'))
            # self.save_img(ddf[0, 1, :, :, :], os.path.join(visualization_path, f'{idx+1}-ddf-y.nii'))
            # self.save_img(ddf[0, 2, :, :, :], os.path.join(visualization_path, f'{idx+1}-ddf-z.nii'))

        
        
        with open(os.path.join(self.log_dir, 'test.log'), 'w') as f:
            for k, v in results.items():
                print(k, np.mean(v, axis=0), np.std(v, axis=0))
                f.writelines(f'{k}, {np.mean(v, axis=0)}, {np.std(v, axis=0)}\n')

        with open(os.path.join(self.log_dir, 'results.pkl'), 'wb') as f:
            pkl.dump(results, f)


    def remove_small_roi(self, mask, size_threshold=2):
        mask = np.round(mask)
        labeled_mask, num_features = label(mask)
        # Set a size threshold (e.g., remove components smaller than 50 voxels)
        sizes = np.bincount(labeled_mask.ravel())
        # print('515', np.unique(mask), sizes)
        mask_filtered = np.isin(labeled_mask, np.where(sizes >= size_threshold)[0]).astype(np.uint8)
        return mask_filtered

    def plot_landmarks(self, fx_img, mv_img, wp_img, fx_ldmk, mv_ldmk, wp_ldmk, fx_seg, mv_seg, wp_seg,idx,  slice=0):
        fx_img, mv_img, wp_img, fx_ldmk, mv_ldmk, wp_ldmk = fx_img.cpu().numpy(), \
        mv_img.cpu().numpy(), wp_img.cpu().numpy(), fx_ldmk.cpu().numpy(), mv_ldmk.cpu().numpy(),\
        wp_ldmk.cpu().numpy()
        mv_seg, fx_seg, wp_seg = mv_seg.cpu().numpy(),fx_seg.cpu().numpy(),wp_seg.cpu().numpy()
        # mv_seg, fx_seg, wp_seg = self.remove_small_roi(mv_seg.cpu().numpy()), self.remove_small_roi(fx_seg.cpu().numpy()),\
        # self.remove_small_roi(wp_seg.cpu().numpy())
        #fx_ldmk, mv_ldmk, wp_ldmk = fx_ldmk.transpose(0, 1, 3, 2, 4), mv_ldmk.transpose(0, 1, 3, 2, 4), wp_ldmk.transpose(0, 1, 3, 2, 4)
        fig, axs = plt.subplots(3, 1, figsize=(5, 10))
        non_zero_fx = np.argwhere(fx_ldmk[0, 0, :, :, :])
        non_zero_mv = np.argwhere(mv_ldmk[0, 0, :, :, :])
        non_zero_wp = np.argwhere(wp_ldmk[0, 0, :, :, :])
        # Plot the landmarks
        mid = non_zero_fx.shape[0]//2
        x,y,z = fx_img.shape[2:]
        # print(non_zero_fx.shape, mid, fx_img.shape)
        axs[0].imshow(np.rot90(fx_img[0, 0, :, :, non_zero_fx[mid, 2]]), cmap='gray')
        axs[0].contour(np.rot90(fx_seg[0, 0, :, :, non_zero_fx[mid, 2]]), cmap='Reds', alpha=0.5, linewidths=1)
        axs[0].scatter(non_zero_fx[mid, 0], y-non_zero_fx[mid, 1], c='r', label='Fixed', linewidths=5)
        
        # axs[0].invert_yaxis()
        mid = non_zero_mv.shape[0]//2
        axs[1].imshow(np.rot90(mv_img[0, 0, :, :, non_zero_mv[mid, 2]]), cmap='gray')
        axs[1].contour(np.rot90(mv_seg[0, 0, :, :, non_zero_mv[mid, 2]]), cmap='Blues', alpha=0.5, linewidths=1)
        axs[1].scatter(non_zero_mv[mid, 0], y-non_zero_mv[mid, 1], c='b', label='Moving', linewidths=5)
        # axs[1].invert_yaxis()
        mid = non_zero_wp.shape[0]//2

        if mid != 0:   
            axs[2].imshow(np.rot90(wp_img[0, 0, :, :, non_zero_wp[mid, 2]]), cmap='gray')
            axs[2].contour(np.rot90(wp_seg[0, 0, :, :, non_zero_wp[mid, 2]]), cmap='Set1', alpha=0.5, linewidths=1)
            axs[2].scatter(non_zero_wp[mid, 0], y-non_zero_wp[mid, 1], c='#F39C12', label='Warpped', linewidths=5)
        # axs[2].invert_yaxis()
        # axs[0].set_title('Fixed Image')
        # axs[1].set_title('Moving Image')
        # axs[2].set_title('Warpped Image')
        axs[0].set_axis_off()
        axs[1].set_axis_off()
        axs[2].set_axis_off()
        plt.tight_layout()
        path = os.path.join(self.log_dir, 'visualization',  f'{idx}-{slice}-landmarks.png')
        plt.savefig(path)
        plt.close()
