import src.model.networks.icn as icn_models
import src.model.networks.icn_trans as icn_models_trans
from src.model.networks import TransMorph
from src.model.networks import VoxelMorph
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
import time 
from src.data import dataloaders
import math
from pathlib import Path
from random import random
from functools import partial
from collections import namedtuple
from multiprocessing import cpu_count
from torch import nn, einsum
from torch.cuda.amp import autocast
import torch.nn.functional as F

from torch.optim import Adam

from torchvision import transforms as T, utils

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange

from PIL import Image
from tqdm.auto import tqdm
from ema_pytorch import EMA

from accelerate import Accelerator

from src.model.denoising_diffusion_pytorch.attend import Attend
from src.model.denoising_diffusion_pytorch.fid_evaluation import FIDEvaluation
from src.model.denoising_diffusion_pytorch.version import __version__
        


class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        config,
        *,
        train_batch_size = 16,
        gradient_accumulate_every = 1,
        augment_horizontal_flip = True,
        train_lr = 1e-4,
        train_num_steps = 700000,
        ema_update_every = 10,
        ema_decay = 0.995,
        adam_betas = (0.9, 0.99),
        save_and_sample_every = 1000,
        num_samples = 25,
        results_folder = './results',
        amp = False,
        mixed_precision_type = 'fp16',
        split_batches = True,
        convert_image_to = None,
        calculate_fid = True,
        inception_block_idx = 2048,
        max_grad_norm = 1.,
        num_fid_samples = 500,
        save_best_and_latest_only = False
        ):
        super().__init__()

        # accelerator

        self.accelerator = Accelerator(
            split_batches = split_batches,
            mixed_precision = mixed_precision_type if amp else 'no'
        )

        # model

        self.model = diffusion_model
        print('total trainable params : ', sum(p.numel() for p in self.model.parameters() if p.requires_grad))
        self.channels = diffusion_model.channels
        is_ddim_sampling = diffusion_model.is_ddim_sampling
        self.data_config = config
        # default convert_image_to depending on channels

        # if not os.path.exists(convert_image_to):
        #     convert_image_to = {1: 'L', 3: 'RGB', 4: 'RGBA'}.get(self.channels)

        # sampling and training hyperparameters

        # assert has_int_squareroot(num_samples), 'number of samples must have an integer square root'
        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every
        assert (train_batch_size * gradient_accumulate_every) >= 16, f'your effective batch size (train_batch_size x gradient_accumulate_every) should be at least 16 or above'

        self.train_num_steps = train_num_steps
        self.image_size = diffusion_model.image_size

        self.max_grad_norm = max_grad_norm

        # dataset and dataloader
        self.set_dataloader()
        dl = self.accelerator.prepare(self.train_loader)
        self.train_batch = self.permute_batch(dl)
        dt = self.accelerator.prepare(self.test_loader)
        self.test_batch = self.permute_batch(dt)
 
        #assert len(self.ds) >= 100, 'you should have at least 100 images in your folder. at least 10k images recommended'

        # optimizer

        self.opt = Adam(diffusion_model.parameters(), lr = train_lr, betas = adam_betas)

        # for logging results in a folder periodically
        print('self.accelarator.device', self.accelerator.device)
        if self.accelerator.is_main_process:
            self.ema = EMA(diffusion_model, beta = ema_decay, update_every = ema_update_every)
            self.ema.to(self.device)

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok = True)

        # step counter state

        self.step = 0

        # prepare model, dataloader, optimizer with accelerator

        self.model, self.opt = self.accelerator.prepare(self.model, self.opt)

        # FID-score computation

        self.calculate_fid = calculate_fid and self.accelerator.is_main_process

        if self.calculate_fid:
            if not is_ddim_sampling:
                self.accelerator.print(
                    "WARNING: Robust FID computation requires a lot of generated samples and can therefore be very time consuming."\
                    "Consider using DDIM sampling to save time."
                )
            self.fid_scorer = FIDEvaluation(
                batch_size=self.batch_size,
                dl=self.test_batch,
                sampler=self.ema.ema_model,
                channels=self.channels,
                accelerator=self.accelerator,
                stats_dir=results_folder,
                device=self.device,
                num_fid_samples=num_fid_samples,
                inception_block_idx=inception_block_idx
            )

        if save_best_and_latest_only:
            assert calculate_fid, "`calculate_fid` must be True to provide a means for model evaluation for `save_best_and_latest_only`."
            self.best_fid = 1e10 # infinite

        self.save_best_and_latest_only = save_best_and_latest_only


    def set_dataloader(self):
        self.train_set = dataloaders.DiffusionData(config=self.data_config, phase='train')

        self.train_loader = DataLoader(
            self.train_set, 
            batch_size=1, 
            shuffle=False,  
            num_workers=4, 
            drop_last=True)  # no need to shuffle since the shuffling is customized in the dataloader.

        print(f'>>> Train set ready. Length is:, {len(self.train_set)})')  
        self.val_set = dataloaders.DiffusionData(config=self.data_config, phase='val')
        self.val_loader = DataLoader(self.val_set, batch_size=1, shuffle=False)
        print('>>> Validation set ready.')
        self.test_set = dataloaders.DiffusionData(config=self.data_config, phase='test')
        self.test_loader = DataLoader(self.test_set, batch_size=1, shuffle=False)
        print('>>> Holdout set ready.')


    
   


    def num_to_groups(self, num, divisor):
        # print('num', num, 'divisor', divisor)
        groups = num // divisor
        remainder = num % divisor
        arr = [divisor] * groups
        if remainder > 0:
            arr.append(remainder)
        return arr
    
    def divisible_by(self, numer, denom):
        return (numer % denom) == 0
    
    def exists(self, x):
        return x is not None

    @property
    def device(self):
        return self.accelerator.device

    def save(self, milestone):
        if not self.accelerator.is_local_main_process:
            return

        data = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.model),
            'opt': self.opt.state_dict(),
            'ema': self.ema.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if self.exists(self.accelerator.scaler) else None,
            'version': __version__
        }

        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))

    def permute_batch(self, dl):
        while True:
            for data in dl:
                #print(data.shape, '212')
                data = data.permute(3, 0, 1, 2)
                data = data.repeat(1, 3, 1, 1)
                index = torch.randperm(data.shape[0])[:self.batch_size]
                data = data[index,...]
                #print(data.shape, '214', index)
                yield data.float()

    def load(self, milestone):
        accelerator = self.accelerator
        device = accelerator.device

        data = torch.load(str(self.results_folder / f'model-{milestone}.pt'), map_location=device)

        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data['model'])

        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        if self.accelerator.is_main_process:
            self.ema.load_state_dict(data["ema"])

        if 'version' in data:
            print(f"loading from version {data['version']}")

        if self.exists(self.accelerator.scaler) and self.exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])

    def train(self):
        accelerator = self.accelerator
        device = accelerator.device
        
        with tqdm(initial = self.step, total = self.train_num_steps, disable = not accelerator.is_main_process) as pbar:

            while self.step < self.train_num_steps:

                total_loss = 0.

                for _ in range(self.gradient_accumulate_every):
                    data = next(self.train_batch).to(device)  
                    #print('270 ', data.shape, 'data.shape')              
                    with self.accelerator.autocast():
                       
                        loss = self.model(data)
                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss.item()

                    self.accelerator.backward(loss)

                pbar.set_description(f'loss: {total_loss:.4f}')

                accelerator.wait_for_everyone()
                accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                self.opt.step()
                self.opt.zero_grad()

                accelerator.wait_for_everyone()

                self.step += 1
    
                if accelerator.is_main_process:
                    self.ema.update()

                    if self.step != 0 and self.divisible_by(self.step, self.save_and_sample_every):
                        self.ema.ema_model.eval()

                        with torch.inference_mode():
                            milestone = self.step // self.save_and_sample_every                            
                            batches = self.num_to_groups(self.num_samples, self.batch_size)# [num_samples // batch] * batch + [num_samples % batch]
                            all_images_list = list(map(lambda n: self.ema.ema_model.sample(batch_size=n), batches))

                        all_images = torch.cat(all_images_list, dim = 0)

                        utils.save_image(all_images, str(self.results_folder / f'sample-{milestone}.png'), nrow = int(math.sqrt(self.num_samples)))

                        # whether to calculate fid

                        if self.calculate_fid:
                            fid_score = self.fid_scorer.fid_score()
                            accelerator.print(f'fid_score: {fid_score}')
                        if self.save_best_and_latest_only:
                            if self.best_fid > fid_score:
                                self.best_fid = fid_score
                                self.save("best")
                            self.save("latest")
                        else:
                            self.save(milestone)

                pbar.update(1)

        accelerator.print('training complete')
