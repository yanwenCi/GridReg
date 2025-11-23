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
parser.add_argument('--data_path', default='../../../data/mpMriReg/FinalProcessed-v2', type=str, help='data dump path')
parser.add_argument('--source_dataset', default='52-52-46-nii', type=str, help='choose the nii path with annotated landmarks.')
parser.add_argument('--target_dataset', default='52-52-46-ldmk', type=str, help='choose a path which going to store the data.')
args = parser.parse_args()


def dump2npy(nii_path, save_path):
    arr = nib.load(nii_path).get_fdata()
    np.save(save_path, arr)

image_list = glob(os.path.join(args.data_path, args.source_dataset, '**', '*.nii.gz'), recursive=True)
print(image_list)

for im in image_list:
    target_path = im.replace(args.source_dataset, args.target_dataset).replace('.nii.gz', '.npy').replace('.png', '')
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    dump2npy(im, target_path)


    