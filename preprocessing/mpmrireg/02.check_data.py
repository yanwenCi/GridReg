import nibabel as nib
import os
from glob import glob
from precrocess import cmicMPMRIStudy


# check if all the image have prostate gland masks and lesion masks
# if the t2 equal to the label shape & pixdim
# if the dwi qual to dwi_b0 shape & pixdim

data_dump_path = '../../../data/mpMriReg/1stRoundProcessedx'
abandon_cases = []  # because lack of lesion mask or prostate gland mask

for t2_folders in glob(os.path.join(data_dump_path, '**', 't2-0')):
    files = os.listdir(t2_folders)
    if 'mask_prostate.nii.gz' in files and ('mask_lesion_1.nii.gz' in files or 'mask_lesion_one.nii.gz' in files):
        pass
    else:
        abandon_cases.append(os.path.dirname(t2_folders))

print('the following studies have no complete masks:')
[print(i) for i in abandon_cases]


# check if the image shape, and if the image shapes are consistent
for study_path in glob(os.path.join(data_dump_path, '*')):
    if study_path in abandon_cases:
        continue
    else:
        study = cmicMPMRIStudy(study_path)
        reject = study.get_all_info()
        if reject:
            abandon_cases.append(study_path)
        print('------------------------')


selected_cases = [i for i in glob(os.path.join(data_dump_path, '*')) if i not in abandon_cases]
with open('./selected_cases.txt', 'w') as f:
    f.writelines("\n".join(selected_cases))

with open('./selected_cases.txt', 'r') as f:
    cases = f.readlines()

print(cases)
print('data cleaning done.')
