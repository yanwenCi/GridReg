
from cmicMPMRILib import *
from glob import glob
import os

# 1. generate b=0 images, with t2/adc/label, and count the images
# 2. get the image pixel size
# 3. resample and crop the images, save to numpy, and visualization

data_root = '../../../data/MR_Prostate_201910/Anon/'
data_dump_path = '../../../data/mpMriReg/1stRoundProcessedx-v2'

def is_target_study(study):
    req = True
    for v in study.series.values():
        req = req and (0 < len(v))
    return req

pat_folders = glob(os.path.join(data_root, '*'))
pat_folders.sort()

for num, pat_folder in enumerate(pat_folders):
    if os.path.isdir(pat_folder):
        print(f'processing {num+1}/{len(pat_folders)}', pat_folder, '...')
        patient_obj = mpMRIPatient(pat_folder)
        for idx, study in enumerate(patient_obj.studies):
            if is_target_study(study):
                target_path = os.path.join(data_dump_path, f'{patient_obj.ID}_Study_{idx}')
                os.makedirs(target_path)
                study.contour_match()
                study.dump2file(target_path)
            else:
                continue




