'''
1. resample to isotropic -> [1, 1, 1]
2. center crop to the same size -> need to check the gland label is not out of the boundray
3. add countours of the gland/lesion to the t2 
4. generate 3d->2d images, with t2/adc/dwi/dwi_b0 (800*4)*800
5. dump the numpy files
'''

import nibabel as nib
import numpy as np
import argparse
from preprocess import *
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--out_pixdim', default=[1.0, 1.0, 1.0], nargs='+', type=float, help='Output pixel size in mm.')
parser.add_argument('--rad', default=[52, 52, 46], nargs='+', type=int, help='Output size on cropped patch (half).')
parser.add_argument('--vis', action='store_true', help='Visualize the generated data.')
parser.add_argument('--vis_nii', action='store_true', help='generate nii files for visualization, if set --vis.')
parser.add_argument('--vis_save_path', default='./vis-52-52-46', type=str, help='output folder for storing the visualizaions.')
parser.add_argument('--dump', action='store_true', help='Dump the data')
parser.add_argument('--dump_path', default='../../../data/mpMriReg/FinalProcessedx/52-52-46--nouse', type=str, help='data dump path')
args = parser.parse_args()

assert args.vis or args.dump, "neither generate visulizations or dump the data?"
if args.vis:
    assert not os.path.exists(args.vis_save_path), "visulazation folder already exists, consider change the name."
if args.dump:
    assert not os.path.exists(args.dump_path), "dump folder already exists, consider change the name."

color_board = [
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0), 
    (205, 133, 63),
    (255, 0, 255),
    (255, 165, 0),
    (255, 0, 0)
]

with open('./selected_cases.txt', 'r') as f:
    study_list = f.readlines()
    study_list = [i.strip() for i in study_list]

def composed_processing(arr, in_pixdim, out_pixdim, order, radius, label_completion_check=False):
    arr = resample(arr, in_pixdim=in_pixdim, out_pixdim=out_pixdim, order=order)
    arr2 = center_crop(arr, radius=radius, allow_padding=True)
    if label_completion_check:
        assert np.sum(arr) == np.sum(arr2), 'label might exceed the cropped boundary'
    return arr2

def gen_vis(target_dir, processed_data):
    assert len(processed_data['lesions_mask']) <= len(color_board)-1, 'color board not enough'

    t2_flat = plot_3d22d(processed_data['t2'])
    dwi_flat = np.tile(plot_3d22d(processed_data['dwi']), (3,1,1)).transpose(1,2,0).astype('uint8')
    dwi_b0_flat = np.tile(plot_3d22d(processed_data['dwi_b0']), (3,1,1)).transpose(1,2,0).astype('uint8')
    adc_flat = np.tile(plot_3d22d(processed_data['adc']), (3,1,1)).transpose(1,2,0).astype('uint8')
    gland_mask_flat = plot_3d22d(processed_data['gland_mask'])

    t2_flat = add_contours(t2_flat, gland_mask_flat, color=color_board[-1])
    # plt.imsave(target_dir.replace('png', '_t2_gland_mask.png'), t2_flat)
    
    for idx, lesion_mask in enumerate(processed_data['lesions_mask']):
        lesion_mask_flat = plot_3d22d(lesion_mask)
        t2_flat = add_contours(t2_flat, lesion_mask_flat, color=color_board[idx])

    assert t2_flat.shape == dwi_b0_flat.shape == dwi_b0_flat.shape == adc_flat.shape, 'shape not equal, unable to concat.'
    combined_img = np.concatenate([t2_flat, adc_flat, dwi_flat, dwi_b0_flat], axis=1)
    plt.imsave(target_dir, combined_img)

    # plt.imsave(target_dir.replace('png', '_t2.png'), t2_flat)
    # plt.imsave(target_dir.replace('png', '_dwi.png'), dwi_flat)
    # plt.imsave(target_dir.replace('png', '_dwi_b0.png'), dwi_b0_flat)
    # plt.imsave(target_dir.replace('png', '_adc.png'), adc_flat)

def calibration(arr):
    """fix the orient of the images"""
    return arr[:,::-1,::-1]

def save_nifty(arr, save_path, pixdim=[1.0, 1.0, 1.0]):
    save_folder = os.path.dirname(save_path)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder, exist_ok=True)

    nib_img = nib.Nifti1Image(arr, affine=np.eye(4))
    nib_img.header['pixdim'][1:4] = np.array(pixdim)
    nib.save(img=nib_img, filename=save_path)

def gen_nii(target_dir, processed_data):
    t2 = calibration(processed_data['t2'])
    dwi = calibration(processed_data['dwi'])
    dwi_b0 = calibration(processed_data['dwi_b0'])
    
    save_nifty(t2, os.path.join(target_dir, 't2.nii.gz'))
    save_nifty(dwi, os.path.join(target_dir, 'dwi.nii.gz'))
    save_nifty(dwi_b0, os.path.join(target_dir, 'dwi_b0.nii.gz'))

def dump_data(target_dir, processed_data):
    os.makedirs(target_dir, exist_ok=True)
    for k, v in processed_data.items():
        if k != 'lesions_mask':
            save_path = os.path.join(target_dir, f'{k}.npy')
            np.save(save_path, v)
        else:
            for idx, lm in enumerate(v):
                save_path = os.path.join(target_dir, f'{k}_{idx+1}.npy')
                np.save(save_path, lm)

with open('./lowq.txt', 'r') as f: 
    lowq_list = f.readlines()
    lowq_list = [i.strip() for i in lowq_list]

### comment if not neccessary ###
# print('low quality list found.')
# [print(i) for i in lowq_list]
###

for idx, study_path in enumerate(study_list):
    # if os.path.basename(study_path) in lowq_list:
    #     pass
    # else:
    #     continue
        
    print(f'{idx+1}/{len(study_list)}, processing {study_path}...')
    study = cmicMPMRIStudy(study_path)
    processed_data = {}
    for mod in ['t2', 'dwi', 'dwi_b0', 'adc', 'gland_mask']:
        arr, in_pixdim = study.get_img(mod)
        if mod in ['t2', 'dwi']:
            print('pixdim:', in_pixdim)
        order = 0 if mod=='gland_mask' else 3
        label_completion_check = True if mod=='gland_mask' else False
        processed_data[mod] = composed_processing(
            arr=arr,
            in_pixdim=in_pixdim,
            out_pixdim=args.out_pixdim,
            order=order,
            radius=args.rad,
            label_completion_check=label_completion_check)

    assert len(study.img_paths['lesions_mask']) != 0, f'no lesions found in {study_path}.'  # double check
    lesion_arr_list, in_pixdim = study.get_img('lesions_mask')
    processed_data['lesions_mask'] = [composed_processing(
            arr = i,
            in_pixdim=in_pixdim,
            out_pixdim=args.out_pixdim,
            order=0,
            radius=args.rad) for i in lesion_arr_list]

    if args.vis:
        print('generate visualizations......')
        if not args.vis_nii:
            os.makedirs(args.vis_save_path, exist_ok=True)
            target_dir = os.path.join(args.vis_save_path, f'{os.path.basename(study_path)}.png')
            gen_vis(target_dir, processed_data)
        else:
            target_dir = os.path.join(args.vis_save_path, f'{os.path.basename(study_path)}.png')
            print(target_dir)
            gen_nii(target_dir, processed_data)

    if args.dump:
        print('generate processed data......')
        target_dir = os.path.join(args.dump_path, os.path.basename(study_path))
        dump_data(target_dir, processed_data)



        
    
