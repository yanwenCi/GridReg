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
parser.add_argument('--data_path', default='../Data/mpMriReg/FinalProcessed', type=str, help='data dump path')
parser.add_argument('--source_dataset', default='52-52-46-nii', type=str, help='choose the nii path with annotated landmarks.')
parser.add_argument('--target_dataset', default='52-52-46-ldmk', type=str, help='choose a path which going to store the data.')
args = parser.parse_args()


def dump2npy(nii_path, save_path):
    arr = nib.load(nii_path).get_fdata()
    np.save(save_path, arr)

def check_ldmk_list(ldmk_nii_list):
    dwi_ldmk = [i for i in ldmk_nii_list if 'dwi' in i]
    t2_ldmk = [i for i in ldmk_nii_list if 't2' in i]
    return len(dwi_ldmk) == len(t2_ldmk)

ldmk_nii_list = glob(os.path.join(args.data_path, args.source_dataset, '**', '*_ldmk_*.nii*'), recursive=True)
assert check_ldmk_list(ldmk_nii_list), "landmarks numbers not equal"

'''only convert patient images which include landmarks'''
patient_folders = [os.path.dirname(i) for i in ldmk_nii_list]
patient_folders = list(set(patient_folders))
# print(patient_folders, len(patient_folders))

for pf in tqdm(patient_folders):
    nii_files = glob(os.path.join(pf, '*'))
    for nf in nii_files:
        save_path = nf.replace(args.source_dataset, args.target_dataset).replace('.nii', '.npy')
        if os.path.exists(save_path):
            continue
        else:
            save_path = save_path.replace('.gz', '')
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            dump2npy(nf, save_path)


    