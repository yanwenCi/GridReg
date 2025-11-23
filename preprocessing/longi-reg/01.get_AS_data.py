#!/usr/bin/env python
# coding: utf-8

import os
import re
import itk
import sys
import nibabel as nib
import numpy as np
from glob import glob
from scipy import ndimage
import itertools
import pickle as pkl
import cv2
import argparse
import matplotlib.pyplot as plt
if 'ipykernel' in sys.argv[0]:
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--o_dim', default=[0.7, 0.7, 0.7], nargs='+', type=float, help='Output pixel size in mm.')
parser.add_argument('--rad', default=[64, 64, 51], nargs='+', type=int, help='Output size on cropped patch (half).')
parser.add_argument('--vis', action='store_true', help='Visualize the generated data.')
parser.add_argument('--data_root', default='../ActivateSurveillance/AS-Data', type=str, help='the data root of the AS data')
parser.add_argument('--output_folder', default='../data/AS-morph-interp-ver-ldmk', type=str, help='the output of the processed AS data ')
args = parser.parse_args()
print(args)


def plot_pat(arr, return_img=False):
    length = arr.shape[0]
    edge_num = int(np.sqrt(arr.shape[2])) + 1
    img = np.zeros([length*edge_num, length*edge_num])
    for z in range(arr.shape[2]):
        x_num = int(z / edge_num)
        y_num = int(z % edge_num)
        img[x_num*length:(x_num+1)*length, y_num*length:(y_num+1)*length] = arr[:, :, z].T
    if return_img:
        return img
    else:
        plt.figure(figsize=(16, 16))
        plt.imshow(img, cmap='gray')


def add_contours(t2, label):
    _t2 = np.tile(t2, (3,1,1)).transpose(1, 2, 0)
    _t2 = (_t2*255).astype('uint8')
    _label = label.astype('uint8')
    blank = np.zeros(_t2.shape)
    contours, hierarchy = cv2.findContours(_label.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE, offset=(0, 0))
    tmp = _t2.copy()  # ?????
    cv2.drawContours(tmp, contours, -1, (255, 0, 0), 3)
    print(len(contours))
    return tmp

def vis_image_and_label(t2_arr, label_arr, save_path):
    flat_t2 = plot_pat(t2_arr, return_img=True)
    flat_label = plot_pat(label_arr, return_img=True)
    tmp = add_contours(flat_t2, flat_label)
    plt.imsave(save_path, tmp)


def get_patient_number(data_root, abandon_cases):
    nums = []
    content = os.listdir(data_root)
    for i in content:
        no = i.split('Patient')[1]
        nums.append(int(no))
    return [i for i in nums if i not in abandon_cases]


abandon_cases = list(range(128, 137)) + [52]
patient_nums = get_patient_number(args.data_root, abandon_cases)
patient_nums.sort()


class ASData(object):
    def __init__(self, patient_id):
        self.data_root = args.data_root
        self.patient_id = patient_id
        self.patient_folder = os.path.join(self.data_root, 'Patient{}'.format(patient_id))
        self.visit_folders = self.__get_visit_folders__()
        self.visit_times = len(self.visit_folders)
        self.t2_collections, self.label_collections, self.landmark_collections = self.__check_t2_and_label_exits__()

    def __get_visit_folders__(self):
        """
        Get sorted visit folder paths to avoid potential bugs.
        """
        vfolders = glob(os.path.join(self.patient_folder, '*'))
        return sorted(vfolders, key=lambda x: self.__sort_keys__(x))

    def __sort_keys__(self, folder_path):
        """
        e.g. input a fold_path like .../Visit2, then return int 2.
        """
        folder = os.path.basename(folder_path)
        key = int(re.findall('Visit(\d+)', folder)[0])
        return key

    def __check_t2_and_label_exits__(self):
        t2_collections, label_collections, landmarks_collections = [], [], []
        for t in range(self.visit_times):
            t2_folder = os.path.join(self.visit_folders[t], 'T2')
            if os.path.exists(t2_folder):
                folder_content = os.listdir(t2_folder)
            else:
                folder_content = []
            nii_files = [i for i in folder_content if (i.endswith('.nii') or i.endswith('.nii.gz'))]
            if 'ProstateBoundingBox.nii' in nii_files:
                label_dir = os.path.join(t2_folder, 'ProstateBoundingBox.nii')
            else:
                label_dir = None

            if 'landmarks.nii.gz' in nii_files:
                landmarks_dir = os.path.join(t2_folder, 'landmarks.nii.gz')
            else:
                landmarks_dir = None

            label_collections.append(label_dir)
            landmarks_collections.append(landmarks_dir)

            selected_images = [i for i in nii_files if 't2' in os.path.basename(i).lower()]
            selected_images = [i for i in selected_images if 'cor' not in i.lower()]
            selected_images = [i for i in selected_images if 'cornal' not in i.lower()]
            selected_images = [i for i in selected_images if 'sag' not in i.lower()]
            t2_collections.append(selected_images)
        return t2_collections, label_collections, landmarks_collections

    def check_selection(self):
        t2_cont, label_cont, landmarks_cont = False, False, False

        t2_idxs = [idx for idx, i in enumerate(self.t2_collections) if i != []]
        if [] in self.t2_collections:
            print(self.patient_id, 't2 not continuous', t2_idxs, 'max-vis-time:', self.visit_times)
        else:
            print(self.patient_id, 't2 continuous', t2_idxs, 'max-vis-time:', self.visit_times)
            t2_cont = True
        
        label_idxs = [idx for idx, i in enumerate(self.label_collections) if i is not None]
        if None in self.label_collections:
            print(self.patient_id, 'label not continuous', label_idxs, 'max-vis-time:', self.visit_times)
        else:
            print(self.patient_id, 'label continuous', label_idxs, 'max-vis-time:', self.visit_times)
            label_cont = True

        landmarks_idxs = [idx for idx, i in enumerate(self.landmark_collections) if i is not None]
        if None in self.landmark_collections:
            print(self.patient_id, 'landmarks not continuous', landmarks_idxs, 'max-vis-time:', self.visit_times)
        else:
            print(self.patient_id, 'landmarks continuous', landmarks_idxs, 'max-vis-time:', self.visit_times)
            landmarks_cont = True

        return t2_cont, t2_idxs, label_cont, label_idxs, landmarks_cont, landmarks_idxs


    def __normalize__(self, arr, method='simple', rm_outliers=False):
        if rm_outliers:
            p = np.percentile(arr, 95)
            arr = np.clip(arr, 0, p)
        if method == 'simple':
            return arr / (np.max(arr) - np.min(arr))
        elif method == 'std':
            return (arr - np.mean(arr)) / np.std(arr)
        else:
            print('normalize method wrong')
            raise NotImplementedError

    def __crop__(self, arr, radius):
        """if 2*radius exceeded the boundaries, zero-padding will be added."""
        len_x, len_y, len_z = arr.shape
        rx, ry, rz = radius[0], radius[1], radius[2]
        cx, cy, cz = int(len_x / 2), int(len_y / 2), int(len_z / 2)
        if len_z < 2 * rz:
            upd, dpd = int(np.ceil((2*rz - len_z)/2)), int(np.floor((2*rz - len_z)/2))
            arr = np.pad(arr, ((0, 0), (0, 0), (upd, dpd)), 'constant', constant_values=0)
        return arr[cx-rx:cx+rx, cy-ry:cy+ry, cz-rz:cz+rz]

    def __resample__(self, arr, method='pix', out_size=(256, 256, 32), in_pixdim=[1, 1, 1], out_pixdim=[0.4, 0.4, 1.5], order=2):
        if method =='pix':
            factor = np.array(out_size) / np.array(arr.shape)
            img = ndimage.zoom(arr, zoom=factor, order=order)
        elif method == 'mm':
            factor = np.array(in_pixdim) / np.array(out_pixdim)
            img = ndimage.zoom(arr, zoom=factor, order=order)
        else:
            print('method wrong')
            raise NotImplementedError
        return img

    def __morph_resample__(self, arr, method='pix', out_size=(256, 256, 32), in_pixdim=[1, 1, 1], out_pixdim=[0.4, 0.4, 1.5]):
        if method =='pix':
            factor = np.array(out_size) / np.array(arr.shape)
        elif method == 'mm':
            factor = np.array(in_pixdim) / np.array(out_pixdim)
        else:
            print('method wrong')
            raise NotImplementedError

        ori_shape = np.array(arr.shape)
        new_z = (ori_shape[2] - 1)*6 + 1

        vessel = np.zeros([ori_shape[0], ori_shape[1], new_z])
        for z in range(ori_shape[2]):
            vessel[:, :, z*6] = arr[:, :, z]
        vessel = vessel.astype('uint8')
        vessel = itk.GetImageFromArray(vessel)

        filled = itk.morphological_contour_interpolator(vessel)
        filled = itk.GetArrayFromImage(filled)

        interp_factor = ori_shape / np.array(filled.shape)
        img = ndimage.zoom(filled, zoom=factor*interp_factor, order=0)

        return img

    def __center_crop__(self, arr):
        x, y, z = arr.shape
        return arr[int(x / 4):int(x * 3 / 4), int(y / 4):int(y * 3 / 4), :]

    def get_image_data_4AS(self, visit_number, modality, request_data=True, normalize=False, resample=None):
        """
        if request_data is True, return image data and path,
        or only the path will be returned.
        modality is T2 or DWI
        if use resample, set this param....
        """
        image_folder = os.path.join(self.patient_folder, 'Visit{}'.format(visit_number), modality)
        if modality == 'T2':
            image_path = glob(os.path.join(image_folder, '*SFOV_TE*'))
        elif modality == 'DWI':
            image_path = glob(os.path.join(image_folder, '*diff_new_16_measipat_ADC.nii'))
        else:
            print('modality wrong and does not exist.')

        if len(image_path) == 0:  # indicates no corresponding image
            return None

        assert len(image_path) <= 1, '{} image is not unique'.format(modality)
        image_path = image_path[0]

        if request_data == False:
            return image_path
        else:
            nibfile = nib.load(image_path)
            data = nibfile.get_data()
            data = self.__center_crop__(data)
            data = data if resample is None else self.__resample__(data, resample)
            data = self.__normalize__(data) if normalize else data
            return image_path, nibfile, data

    def get_selected_t2_img(self, visit_number, o_dim=[0.4, 0.4, 1.5], rad=[128, 128, 16]):
        nib_file = nib.load(os.path.join(self.visit_folders[visit_number], 'T2', self.t2_collections[visit_number][0]))
        image_arr = nib_file.get_fdata()
        in_pixdim = nib_file.header['pixdim'][1:4]
        print('img_arr_shape', image_arr.shape)
        processed_data = self.__resample__(image_arr, method='mm', in_pixdim=in_pixdim, out_pixdim=o_dim)
        print('resample_shape', processed_data.shape)
        processed_data = self.__normalize__(processed_data, method='simple', rm_outliers=True)
        processed_data = self.__crop__(processed_data, radius=rad)
        key = 'Patient{}-Visit{}'.format(self.patient_id, visit_number)

        if self.label_collections[visit_number] is None:
            label = None
        else:
            label_nib_file = nib.load(self.label_collections[visit_number])
            # label_arr = label_nib_file.get_data()

            label_itk = itk.imread(self.label_collections[visit_number], itk.UC)
            label_arr_itk = itk.GetArrayFromImage(label_itk)
            label_arr = np.transpose(label_arr_itk, [2, 1, 0])

            # print('####', image_arr.shape, label_arr.shape, label_arr_itk.shape)#####################
            in_pixdim = label_nib_file.header['pixdim'][1:4]
            processed_label = self.__morph_resample__(label_arr, method='mm', in_pixdim=in_pixdim, out_pixdim=o_dim)
            label = self.__crop__(processed_label, radius=rad)

        if self.landmark_collections[visit_number] is None:
            landmarks = None
        else:
            landmarks_nibfile = nib.load(self.landmark_collections[visit_number])
            landmarks_arr = landmarks_nibfile.get_fdata()
            in_pixdim = landmarks_nibfile.header['pixdim'][1:4]
            landmarks = []
            for lb in range(1, np.unique(landmarks_arr).shape[0]):
                sub_ldmk_arr = ((landmarks_arr == lb)*1).astype('int')
                processed_landmarks = self.__resample__(sub_ldmk_arr, method='mm', in_pixdim=in_pixdim, out_pixdim=o_dim, order=1)
                processed_landmarks = self.__crop__(processed_landmarks, radius=rad)
                landmarks.append(processed_landmarks)


        return key, processed_data, label, landmarks


# save_data - npy
dataset_folder = f'{args.o_dim[0]}-{args.o_dim[1]}-{args.o_dim[2]}-{args.rad[0]}-{args.rad[1]}-{args.rad[2]}'
img_save_path = os.path.join(args.output_folder, dataset_folder, 'images')
os.makedirs(img_save_path, exist_ok=True)

if args.vis:
    vis_folder = os.path.join(args.output_folder, dataset_folder, 'visualization')
    os.makedirs(vis_folder, exist_ok=True)



for idx, pat_num in enumerate(patient_nums):

    pat = ASData(pat_num)
    t2_cont, t2_idxs, label_cont, label_idxs, landmarks_cont, landmarks_idxs = pat.check_selection()

    for vis_num in t2_idxs:
        data_key, t2_arr, label_arr, landmarks_arr = pat.get_selected_t2_img(vis_num, o_dim=args.o_dim, rad=args.rad)

        if label_arr is not None:
            image_and_label_arr = np.stack([t2_arr, label_arr])
            np.save(os.path.join(img_save_path, data_key+'-T2.npy'), image_and_label_arr)

            if args.vis:
                vis_save_path = os.path.join(vis_folder, f'{data_key}.png')
                vis_image_and_label(t2_arr, label_arr, vis_save_path)

        if landmarks_arr is not None:
            for idx, ldmk_arr in enumerate(landmarks_arr):
                ldmk_name=f'{data_key}-T2-ldmark-{idx+1}.npy'
                np.save(os.path.join(img_save_path, ldmk_name), ldmk_arr)
                print(f'{data_key}-ldmark-{idx}', 'was created.')
        print(data_key, 'was created.')
        

# generate key files 
labeled_patients = []
for i in patient_nums:
    pat = ASData(i)
    t2_cont, t2_idxs, label_cont, label_idxs, landmarks_cont, landmarks_idxs = pat.check_selection()
    if len(label_idxs)>1 and (set(label_idxs) <= set(t2_idxs)):
        labeled_patients.append(i)


key_dict = {'train': [], 'test':[], 'val':[]}
for i in labeled_patients[:70]:
    pat = ASData(i)
    t2_cont, t2_idxs, label_cont, label_idxs, landmarks_cont, landmarks_idxs = pat.check_selection()
    print(i, pat.t2_collections, pat.label_collections)
    for comb in itertools.combinations(label_idxs, 2):
        key1 = f'Patient{i}-Visit{comb[0]}'
        key2 = f'Patient{i}-Visit{comb[1]}'
        key_dict['train'].append((key1, key2))
        key_dict['train'].append((key2, key1))

for i in labeled_patients[70:76]:
    pat = ASData(i)
    t2_cont, t2_idxs, label_cont, label_idxs, landmarks_cont, landmarks_idxs = pat.check_selection()
    print(i, pat.t2_collections, pat.label_collections)
    for comb in itertools.combinations(label_idxs, 2):
        key1 = f'Patient{i}-Visit{comb[0]}'
        key2 = f'Patient{i}-Visit{comb[1]}'
        key_dict['val'].append((key1, key2))
        key_dict['val'].append((key2, key1))

for i in labeled_patients[76:]:
    pat = ASData(i)
    t2_cont, t2_idxs, label_cont, label_idxs, landmarks_cont, landmarks_idxs = pat.check_selection()
    print(i, pat.t2_collections, pat.label_collections)
    for comb in itertools.combinations(label_idxs, 2):
        key1 = f'Patient{i}-Visit{comb[0]}'
        key2 = f'Patient{i}-Visit{comb[1]}'
        key_dict['test'].append((key1, key2))
        key_dict['test'].append((key2, key1))

with open(os.path.join(args.output_folder, dataset_folder, 'key-train-IFIB-val-IFIB-test-IFIB.pkl'), 'wb') as f:
    pkl.dump(key_dict, f)


# #
key_dict['test'], key_dict['val'] = [], []
for i in labeled_patients[70:76]:
    pat = ASData(i)
    t2_cont, t2_idxs, label_cont, label_idxs, landmarks_cont, landmarks_idxs = pat.check_selection()
    print(i, pat.t2_collections, pat.label_collections)
    for comb in itertools.combinations(label_idxs, 2):
        key1 = f'Patient{i}-Visit{comb[0]}'
        key2 = f'Patient{i}-Visit{comb[1]}'
        key_dict['val'].append((key1, key2))

for i in labeled_patients[76:]:
    pat = ASData(i)
    t2_cont, t2_idxs, label_cont, label_idxs, landmarks_cont, landmarks_idxs = pat.check_selection()
    print(i, pat.t2_collections, pat.label_collections)
    for comb in itertools.combinations(label_idxs, 2):
        key1 = f'Patient{i}-Visit{comb[0]}'
        key2 = f'Patient{i}-Visit{comb[1]}'
        key_dict['test'].append((key1, key2))

with open(os.path.join(args.output_folder, dataset_folder, 'key-train-IFIB-val-IF-test-IF.pkl'), 'wb') as f:
    pkl.dump(key_dict, f)


# #
key_dict['train'] = []
for i in labeled_patients[:70]:
    pat = ASData(i)
    t2_cont, t2_idxs, label_cont, label_idxs, landmarks_cont, landmarks_idxs = pat.check_selection()
    print(i, pat.t2_collections, pat.label_collections)
    for comb in itertools.combinations(label_idxs, 2):
        key1 = f'Patient{i}-Visit{comb[0]}'
        key2 = f'Patient{i}-Visit{comb[1]}'
        key_dict['train'].append((key1, key2))

with open(os.path.join(args.output_folder, dataset_folder, 'key-train-IF-val-IF-test-IF.pkl'), 'wb') as f:
    pkl.dump(key_dict, f)


