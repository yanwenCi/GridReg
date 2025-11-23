'''
To change the dumped numpy files into nii files,
in order to use medical image browser for labeling the landmarks.
'''

import nibabel as nib
import numpy as np
import os
import argparse
from glob import glob
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--pixdim', default=[1.0, 1.0, 1.0], nargs='+', type=float, help='Output pixel size in mm.')
parser.add_argument('--data_path', default='../../../data/mpMriReg/FinalProcessed', type=str, help='data dump path')
parser.add_argument('--dataset', default='52-52-46', type=str, help='choose which dataset')
args = parser.parse_args()


def dump2nii(npy_path, save_path):
    arr = np.load(npy_path)
    nib_img = nib.Nifti1Image(arr, affine=np.eye(4))
    nib_img.header['pixdim'][1:4] = np.array(args.pixdim)
    nib.save(img=nib_img, filename=save_path)

raw_npy_list = glob(os.path.join(args.data_path, '**', '*.npy'), recursive=True)
npy_list = [i for i in raw_npy_list if 'train' not in i]

for npy_path in tqdm(npy_list):
    basename = os.path.basename(npy_path).replace('.npy', '.nii')
    save_folder = os.path.dirname(npy_path.replace(args.dataset, f'{args.dataset}-nii'))
    os.makedirs(save_folder, exist_ok=True)
    save_path = os.path.join(save_folder, basename)
    dump2nii(npy_path, save_path)
    