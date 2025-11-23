from src.model.networks.local import CondiSegUNet
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
import cv2, random


class condiSeg(BaseArch):
    def __init__(self, config):
        super(condiSeg, self).__init__(config)
        self.config = config
        self.net = self.net_parsing()
        self.set_dataloader()
        self.best_metric = 0

    def net_parsing(self):
        model = self.config.model
        if model == 'CondiSegUNet':
            net = CondiSegUNet(self.config)
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
        return fx_img, fx_seg, mv_img, mv_seg

    def gen_pseudo_data(self):
        
        pseudo_data = []
        for i in range(self.config.batch_size):
            lx, ly, lz = self.config.input_shape
            cx = random.sample(list(range(lx)), 1)[0] 
            cy = random.sample(list(range(ly)), 1)[0]
            cz = random.sample(list(range(lz)), 1)[0]
            rad = 16

            Lx, Rx = max(0, cx-rad), min(lx, cx+rad)  # Left & Right x
            Ly, Ry = max(0, cy-rad), min(ly, cy+rad)  # Left & Right y
            Lz, Rz = max(0, cz-rad), min(lz, cz+rad)  # Left & Right z
            
            seg_arr = torch.zeros(self.config.input_shape)
            seg_arr = seg_arr[None, ...]  # add channel dim
            seg_arr[:, Lx:Rx, Ly:Ry, Lz:Rz] = 1.0
            pseudo_data.append(seg_arr)

        pseudo_data = torch.stack(pseudo_data, dim=0)
        return pseudo_data.cuda()
    
    @torch.no_grad()
    def forward_pseudo_data(self, pseudo_input):
        self.net.eval()
        pseudo_out = self.net(torch.cat(pseudo_input, dim=1))
        pseudo_out = (pseudo_out >= 0.5) * 1.0
        return pseudo_out

    def train(self):
        self.save_configure()
        optimizer = optim.Adam(self.net.parameters(), lr=self.config.lr, weight_decay=1e-6)
        for self.epoch in range(1, self.config.num_epochs + 1):
            self.net.train()

            print('-' * 10, f'Train epoch_{self.epoch}', '-' * 10)
            for self.step, input_dict in enumerate(self.train_loader):
                fx_img, fx_seg, mv_img, mv_seg = self.get_input(input_dict)

                optimizer.zero_grad()
                pred_seg = self.net(torch.cat([fx_img, mv_img, mv_seg], dim=1))

                global_loss = self.loss(pred_seg, fx_seg)
                global_loss.backward()
                optimizer.step()

                if self.config.use_pseudo_label:
                    print("in pseudo training....")
                    # generate pseudo data
                    pseudo_input = self.gen_pseudo_data()
                    pseudo_out = self.forward_pseudo_data([fx_img, mv_img, pseudo_input])
                    pseudo_label = pseudo_out.detach()

                    # use pseudo data to train another round
                    self.net.train()
                    optimizer.zero_grad()
                    pred_seg = self.net(torch.cat([fx_img, mv_img, pseudo_input], dim=1))
                    global_loss = self.loss(pred_seg, pseudo_label)
                    global_loss.backward()
                    optimizer.step()

            if self.epoch % self.config.save_frequency == 0:
                self.save()
            print('-' * 10, 'validation', '-' * 10)
            self.validation()
        
        self.inference()

    def loss(self, pred_seg, fx_seg):
        L_All = 0
        Info = f'step {self.step}'
        
        if self.config.w_dce != 0: 
            L_dice = loss.single_scale_dice(fx_seg, pred_seg) * self.config.w_dce
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
        self.net.eval()
        # visualization_path = os.path.join(self.log_dir, f'{self.config.exp_name}-vis-in-val')
        # os.makedirs(visualization_path, exist_ok=True)

        res = []
        for idx, input_dict in enumerate(self.val_loader):
            fx_img, fx_seg, mv_img, mv_seg = self.get_input(input_dict, aug=False)
            # fx Size([1, 1, 2, 152, 269, 121]) mv Size([1, 1, 2, 152, 269, 121])

            # self.save_img(fx_img, os.path.join(visualization_path, f'{idx+1}-fx_img.nii'))
            # self.save_img(mv_img, os.path.join(visualization_path, f'{idx+1}-mv_img.nii'))

            for label_idx in range(fx_seg.shape[2]):
                pred_seg = self.net(torch.cat([fx_img, mv_img, mv_seg[:, :, label_idx, ...]], dim=1))
                binary_dice = loss.binary_dice(pred_seg, fx_seg[:, :, label_idx, ...])

                subject = input_dict['subject']

                print(f'subject:{subject}', f'label_idx:{label_idx}', f'DICE:{binary_dice:.3f}')
                res.append(binary_dice)

                # self.save_img(fx_seg[:, :, label_idx, ...], os.path.join(visualization_path, f'{idx+1}-fx_img_{label_idx}.nii'))
                # self.save_img(mv_seg[:, :, label_idx, ...], os.path.join(visualization_path, f'{idx+1}-mv_img_{label_idx}.nii'))
                # self.save_img(pred_seg[0], os.path.join(visualization_path, f'{idx+1}-pred_img_{label_idx}.nii'))

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
            }

        for idx, input_dict in enumerate(self.test_loader):
            fx_img, fx_seg, mv_img, mv_seg = self.get_input(input_dict, aug=False)
            self.save_img(fx_img, os.path.join(visualization_path, f'{idx+1}-fx_img.nii'))
            self.save_img(mv_img, os.path.join(visualization_path, f'{idx+1}-mv_img.nii'))

            label_num = fx_seg.shape[2]
            for label_idx in range(fx_seg.shape[2]):
                pred_seg = self.net(torch.cat([fx_img, mv_img, mv_seg[:, :, label_idx, ...]], dim=1))

                aft_dice = loss.binary_dice(pred_seg, fx_seg[:, :, label_idx, ...]).cpu().numpy()
                bef_dice = loss.binary_dice(fx_seg[:, :, label_idx, ...], mv_seg[:, :, label_idx, ...]).cpu().numpy()

                subject = input_dict['subject']
                results['dice'].append(aft_dice)
                results['dice-wo-reg'].append(bef_dice)
                print(f'subject:{subject}', f'label_idx:{label_idx}', f'BEF-DICE:{bef_dice:.3f}', f'AFT-DICE:{aft_dice:.3f}')

                self.save_img(fx_seg[:, :, label_idx, ...], os.path.join(visualization_path, f'{idx+1}-fx_img_{label_idx}.nii'))
                self.save_img(mv_seg[:, :, label_idx, ...], os.path.join(visualization_path, f'{idx+1}-mv_img_{label_idx}.nii'))
                self.save_img(pred_seg[0], os.path.join(visualization_path, f'{idx+1}-pred_img_{label_idx}.nii'))

                anatomical_list = ['bladder', 'rectum']
                self.vis_with_contour(
                    fx_img=fx_img[0, 0].cpu().numpy(), 
                    fx_seg=fx_seg[:, :, label_idx, ...][0, 0].cpu().numpy(), 
                    mv_img=mv_img[0, 0].cpu().numpy(), 
                    mv_seg=mv_seg[:, :, label_idx, ...][0, 0].cpu().numpy(), 
                    pred_seg=pred_seg[0, 0].cpu().numpy(), 
                    save_folder=os.path.join(visualization_path, 'vis_png', subject[0]),
                    color=(255, 0, 0), 
                    prefix=f'DSC_{anatomical_list[label_idx]}_bef_{bef_dice:.3f}_after_{aft_dice:.3f}',
                    suffix=''
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


    @staticmethod
    def vis_with_contour(fx_img, fx_seg, mv_img, mv_seg, pred_seg, save_folder, color=(255, 255, 0), prefix='', suffix=''):
        """fx/mv_img/seg -> 3d volume"""
        def normalize0255(arr):
            return (arr - arr.min())*255.0 / (arr.max() - arr.min())

        def add_contours(t2, label, color):
            if len(t2.shape) != 3:
                _t2 = np.tile(t2, (3,1,1)).transpose(1, 2, 0)
            else:
                _t2 = t2
            
            _t2 = normalize0255(_t2).astype('uint8')
            _label = label.astype('uint8')
            blank = np.zeros(_t2.shape)
            contours, hierarchy = cv2.findContours(_label.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE, offset=(0, 0))
            tmp = _t2.copy()  # ?????
            cv2.drawContours(tmp, contours, -1, color, 1)
            return tmp

        img_set = np.concatenate([mv_img, fx_img, fx_img], axis=0)
        img_set = normalize0255(img_set)
        seg_set = np.concatenate([mv_seg, fx_seg, pred_seg], axis=0)
        
        for z in range(fx_img.shape[-1]):
            img_slice = img_set[..., z]
            seg_slice = seg_set[..., z]
            contoured_slice = add_contours(img_slice, seg_slice, color=color)
            os.makedirs(save_folder, exist_ok=True)

            dst_img = np.transpose(contoured_slice, (1,0,2))[::-1, ...]
            # print(np.array(dst_img.shape[:2])*3)
            cv2.imwrite(
                os.path.join(save_folder, f"{prefix}_{z}_{suffix}.png"), 
                # dst_img
                cv2.resize(dst_img, (dst_img.shape[1]*3, dst_img.shape[0]*3))
                )