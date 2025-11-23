import os
import pickle as pkl
import src.model.loss as loss
from collections import Counter
from glob import glob
import nibabel as nib
import torch
import numpy as np
from tqdm import tqdm


data_path = '../data/mpMriReg/FinalProcessed/rand_affine'
target_folder = '/media/yipeng/data/data/mpMriReg/FinalProcessed/52-52-46/train'
folders = ['dgx1', 'laptop', 'local', 'pt5']

def get_mi_by_folder(folder):
    dwi_img = nib.load(os.path.join(folder, 'dwi.nii.gz')).get_fdata()
    dwi_aff = nib.load(os.path.join(folder, 'dwi_b0_afTrans.nii.gz')).get_fdata()
    dwi_img = torch.FloatTensor(dwi_img)[None, None, ...].cuda()
    dwi_aff = torch.FloatTensor(dwi_aff)[None, None, ...].cuda()
    mi = loss.global_mutual_information(dwi_aff, dwi_img)
    return mi.cpu().numpy()
    
def dump2npy(nii_path, save_path):
    arr = nib.load(nii_path).get_fdata()
    np.save(save_path, arr)

res = []
for f in folders:
    res.extend(glob(os.path.join(data_path, f, '*')))

collections = {}
for r in res:
    key = os.path.basename(r)
    collections[key] = []

for r in res:
    key = os.path.basename(r)
    if r not in collections[key]:
        collections[key].append(r)

final_fused_paths = []
for k,v in collections.items():
    if len(v) == 1:
        final_fused_paths.append(v[0])
    else:
        MIs = [get_mi_by_folder(p) for p in v]
        final_fused_paths.append(v[np.argmax(MIs)])

for pat_path in tqdm(final_fused_paths):
    src_path = os.path.join(pat_path, 'dwi_b0_afTrans.nii.gz')
    target_path = os.path.join(target_folder, os.path.basename(pat_path), 'dwi_b0_afTrans.npy')
    dump2npy(src_path, target_path)




