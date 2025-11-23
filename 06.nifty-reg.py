import os
import numpy as np
import sys
import nibabel as nib
import pickle as pkl
import shutil
from time import time
from glob import glob
if 'ipykernel' in sys.argv[0]:
    from tqdm import tqdm_notebook as tqdm
else:
    import tqdm as tqdm
import torch
from src.model.loss import centroid_distance, global_mutual_information
from scipy import stats
from time import time

# data_path = '../data/mpMriReg/FinalProcessed-v2/52-52-46-nii/test'
# data_path = '../data/mpMriReg/external/CIA-external-nii'
data_path = '../data/mpMriReg/external/Miami-external-nii'

assert os.path.exists(data_path), "data path not exists"
# save_folder = './Others/niftyreg-test'
# save_folder = './Others/niftyreg-test-cia'
save_folder = './Others/niftyreg-test-miami'

# results_pkl_name = 'results.pkl'
# results_pkl_name = 'results_cia.pkl'
results_pkl_name = 'results_miami.pkl'

os.makedirs(save_folder, exist_ok=True)

patient_folders = glob(os.path.join(data_path, '*'))
patient_folders = [i for i in patient_folders if 't2_ldmk_1.nii.gz' in os.listdir(i)]
patient_folders.sort()


def get_input_file_paths(patient_folder):
    dwi_ldmks = glob(os.path.join(patient_folder, 'dwi_ldmk_*'))
    dwi_ldmks.sort()
    t2_ldmks = [i.replace('dwi_ldmk', 't2_ldmk') for i in dwi_ldmks]
    for idx in range(len(t2_ldmks)):
        if not os.path.exists(t2_ldmks[idx]):
            t2_ldmks[idx] = t2_ldmks[idx].replace('.nii.gz', '.nii')
            assert os.path.exists(t2_ldmks[idx]), f"ldmks not match, in {t2_ldmks[idx]}"
        else: pass
    return {
        't2': os.path.join(patient_folder, 't2.nii.gz'),
        'dwi_b0': os.path.join(patient_folder, 'dwi_b0.nii.gz'),
        'dwi': os.path.join(patient_folder, 'dwi.nii.gz'),
        'dwi_ldmks': dwi_ldmks,
        't2_ldmks': t2_ldmks
    }

def get_torch_tensor(nii_path):
    arr = torch.FloatTensor(nib.load(nii_path).get_fdata()).cuda()
    return arr[None, None, ...]

def calc_centriod_distance(ref_ldmk, flo_ldmk):
    ref_arr = get_torch_tensor(ref_ldmk)
    flo_arr = get_torch_tensor(flo_ldmk)
    return centroid_distance(ref_arr, flo_arr).cpu().numpy()

def normalize(arr):
    return (arr - arr.min())/(arr.max() - arr.min())
    
def calc_mutual_info(ref, flo):
    ref_arr = get_torch_tensor(ref)
    ref_arr = normalize(ref_arr)
    flo_arr = get_torch_tensor(flo)
    flo_arr = normalize(flo_arr)
    return global_mutual_information(ref_arr, flo_arr).cpu().numpy()
    

reg_command_tpl = 'reg_f3d -ref {} -flo {} -be 0.005 -cpp {} -res {}'
resample_command_tpl = 'reg_resample -ref {} -flo {} -cpp {} -res {}'
results = {'mi':[], 'ldmk':[], 'mi-wo-reg':[], 'ldmk-wo-reg':[]}

total_time = 0

ldmk_improvement_topX = []
ldmk_num_each_sample = []
ldmk_footprint = []

for idx, pf in enumerate(patient_folders):
    print(pf)
    pid = os.path.basename(pf)
    data = get_input_file_paths(pf)
    
    cpp_name = os.path.join(save_folder, f'{pid}-ref-t2-flo-dwi-cpp.nii')
    result_name = os.path.join(save_folder, f'{pid}-ref-t2-flo-dwi.nii')
    ref, flo = data['t2'], data['dwi']
    
    '''registering'''
    reg_command = reg_command_tpl.format(ref, flo, cpp_name, result_name)
    print('reg_command', reg_command)
    if os.path.exists(cpp_name) and os.path.exists(result_name):
        print(f'cpp file found, using exists one {cpp_name}')
    else:
        start = time()
        os.system(reg_command)
        end = time()
        total_time += (end - start)
    
    '''resample ldmks'''
    for (ref_lm, flo_lm) in zip(data['t2_ldmks'], data['dwi_ldmks']):
        print(ref_lm, flo_lm)
        ldmk_footprint.append(idx)
        ldmk_id = os.path.basename(ref_lm).split('_')[2].replace('.nii.gz', '')
        resample_ldmk_name = os.path.join(save_folder, f'{pid}-ref-t2-flo-dwi-ldmk-{ldmk_id}.nii')
        
        resample_command = resample_command_tpl.format(ref, flo_lm, cpp_name, resample_ldmk_name)
        os.system(resample_command)

        results['ldmk'].append(calc_centriod_distance(ref_lm, resample_ldmk_name))
        results['ldmk-wo-reg'].append(calc_centriod_distance(ref_lm, flo_lm))
        ldmk_improvement_topX.append(results['ldmk-wo-reg'][-1] - results['ldmk'][-1])
    results['mi'].append(calc_mutual_info(ref, result_name))
    results['mi-wo-reg'].append(calc_mutual_info(ref, flo))


"""save results and calculate p-value of paired t-test"""   
tmp_dict = {}
for k, v in results.items():
    mean, std = np.around(np.mean(v), decimals=3), np.around(np.std(v), decimals=3)
    print(k, mean, std)
    tmp_dict[f'{k}_stat'] = mean, std
results.update(tmp_dict)

results['mi-p-value'] = stats.ttest_rel(results['mi'], results['mi-wo-reg'])[1]
results['ldmk-p-value'] = stats.ttest_rel(results['ldmk'], results['ldmk-wo-reg'])[1]
print('mi-p-value:', results['mi-p-value'])
print('ldmk-p-value:', results['ldmk-p-value'])
print(f'total time used: {total_time}s, average: {total_time/len(patient_folders)}s')

with open(os.path.join(save_folder, results_pkl_name), 'wb') as f:
    pkl.dump(results, f)


"""check the improvement on the worst cases"""
case_num = 7
wo_reg_topX, after_reg_topX = [], []
dist_copy = results['ldmk-wo-reg'].copy()

misalignment_footprint = [i for _, i in sorted(zip(dist_copy, ldmk_footprint), reverse=True)]
dist_copy.sort(reverse=True)

showed_up, bef_mis, aft_mis = [], [], []
for idx in misalignment_footprint[:case_num]:
    if idx not in showed_up:
        showed_up.append(idx)
        bef_mis.append(results['mi-wo-reg'][idx])
        aft_mis.append(results['mi'][idx])
    else:
        continue

ldmk_improvement_bef = [i for _, i in sorted(zip(ldmk_improvement_topX, results['ldmk-wo-reg'].copy()), reverse=True)]
ldmk_improvement_aft = [i for _, i in sorted(zip(ldmk_improvement_topX, results['ldmk'].copy()), reverse=True)]
improvement_footprint = [i for _, i in sorted(zip(ldmk_improvement_topX, ldmk_footprint), reverse=True)]
ldmk_improvement_topX.sort(reverse=True)

standard = dist_copy[case_num]
for (wr_dist, ar_dist) in zip(results['ldmk-wo-reg'], results['ldmk']):
    if wr_dist > standard:
        wo_reg_topX.append(wr_dist)
        after_reg_topX.append(ar_dist)

print(f'wo_reg_top_{case_num}_tre:', np.around(np.mean(wo_reg_topX), decimals=3), np.around(np.std(wo_reg_topX), decimals=3))
print(f'after_reg_top_{case_num}_tre:', np.around(np.mean(after_reg_topX), decimals=3), np.around(np.std(after_reg_topX), decimals=3))
print(f'wo_reg_top_{case_num}_mi:', np.around(np.mean(bef_mis), decimals=3), np.around(np.std(bef_mis), decimals=3))
print(f'after_reg_top_{case_num}_mi:', np.around(np.mean(aft_mis), decimals=3), np.around(np.std(aft_mis), decimals=3))
print('p-value-tre:', stats.ttest_rel(after_reg_topX, wo_reg_topX)[1])
print('p-value-mi:', stats.ttest_rel(aft_mis, bef_mis)[1])

print(f'improvement_top_{case_num}:', np.around(np.mean(ldmk_improvement_topX[:case_num]), decimals=3), np.around(np.std(ldmk_improvement_topX[:case_num]), decimals=3))
print(f'improvement_top{case_num}_bef-tre:', np.around(np.mean(ldmk_improvement_bef[:case_num]), decimals=3), np.around(np.std(ldmk_improvement_bef[:case_num]), decimals=3))
print(f'improvement_top{case_num}_aft-tre:', np.around(np.mean(ldmk_improvement_aft[:case_num]), decimals=3), np.around(np.std(ldmk_improvement_aft[:case_num]), decimals=3))
top_pids = list(set(improvement_footprint[:case_num]))
bef_mis = [results['mi-wo-reg'][i] for i in top_pids]
aft_mis = [results['mi'][i] for i in top_pids]
print(f'improvement_top{case_num}_bef-mi:', np.around(np.mean(bef_mis), decimals=3), np.around(np.std(bef_mis), decimals=3))
print(f'improvement_top{case_num}_aft-mi:', np.around(np.mean(aft_mis), decimals=3), np.around(np.std(aft_mis), decimals=3))
print('p-value:', stats.ttest_rel(ldmk_improvement_bef[:case_num], ldmk_improvement_aft[:case_num])[1])
print('p-value-mi:', stats.ttest_rel(aft_mis, bef_mis)[1])

tmp_dict = {}
for k, v in results.items():
    mean, std = np.around(np.mean(v), decimals=3), np.around(np.std(v), decimals=3)
    print(k, mean, std)
    tmp_dict[f'{k}_stat'] = mean, std
results.update(tmp_dict)

results['mi-p-value'] = stats.ttest_rel(results['mi'], results['mi-wo-reg'])[1]
results['ldmk-p-value'] = stats.ttest_rel(results['ldmk'], results['ldmk-wo-reg'])[1]
print('mi-p-value:', results['mi-p-value'])
print('ldmk-p-value:', results['ldmk-p-value'])