import os
from glob import glob
import nibabel as nib
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
import itk
import pickle as pkl


CMIC_DATA_PATH = '/home/QianyeYang/project/MulticentreProstateSegmentation/MRI'
NPY_SAVE_PATH = '../data/AS-morph-interp-ver/0.7-0.7-0.7-64-64-51/images'
key_file = '../data/AS-morph-interp-ver/0.7-0.7-0.7-64-64-51/key-train-IFIB-val-IFIB-test-IFIB.pkl'

def plot_pat(arr, return_img=False):
    length = arr.shape[0]
    edge_num = int(np.ceil(np.sqrt(arr.shape[2])))
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


class Patient(object):
    def __init__(self, path):
        self.root = CMIC_DATA_PATH
        self.pid = os.path.basename(path)
        self.path = path
        self.sub_folder_path = os.path.join(self.path, '1')
        self.T2_path = self.__get_t2_path__()
        self.prostate_mask_path = self.__get_prostate_mask_path__()
    
    def __get_t2_path__(self):
        path = os.path.join(self.sub_folder_path, 'Prostate_T2W_AX_1.nii')
        if not os.path.exists(path):
            path = None
        return path

    def __get_prostate_mask_path__(self):
        path = os.path.join(self.sub_folder_path, 'Prostate_T2W_AX_1_ProstateMask.nii')
        if not os.path.exists(path):
            path = None
        return path

    def __normalize__(self, arr, method='simple', rm_outliers=False):
        if rm_outliers:
            p = np.percentile(arr, 96)
            arr = np.clip(arr, 0, p)
        if method == 'simple':
            return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))
        elif method == 'std':
            return (arr - np.mean(arr)) / np.std(arr)
        else:
            print('normalize method wrong')
            raise NotImplementedError

    def __center_crop__(self, arr, radius, allow_padding=True):
        """if 2*radius exceeded the boundaries, zero-padding will be added."""
        len_x, len_y, len_z = arr.shape
        rx, ry, rz = radius[0], radius[1], radius[2]
        if len_z < 2 * rz:
            assert allow_padding is True, "boundary exceeded, consider setting allow_padding=True"
            upd, dpd = int(np.ceil((2 * rz - len_z) / 2)), int(np.floor((2 * rz - len_z) / 2))
            arr = np.pad(arr, ((0, 0), (0, 0), (upd, dpd)), 'constant', constant_values=0)
        if len_x < 2 * rx:
            assert allow_padding is True, "boundary exceeded, consider setting allow_padding=True"
            upd, dpd = int(np.ceil((2 * rx - len_x) / 2)), int(np.floor((2 * rx - len_x) / 2))
            arr = np.pad(arr, ((upd, dpd), (0, 0), (0, 0)), 'constant', constant_values=0)
        if len_y < 2 * ry:
            assert allow_padding is True, "boundary exceeded, consider setting allow_padding=True"
            upd, dpd = int(np.ceil((2 * ry - len_y) / 2)), int(np.floor((2 * ry - len_y) / 2))
            arr = np.pad(arr, ((0, 0), (upd, dpd), (0, 0)), 'constant', constant_values=0)

        len_x, len_y, len_z = arr.shape
        cx, cy, cz = int(len_x / 2), int(len_y / 2), int(len_z / 2)
        return arr[cx - rx:cx + rx, cy - ry:cy + ry, cz - rz:cz + rz]

    def __resample__(self, arr, method='pix', out_size=(256, 256, 32), in_pixdim=[1, 1, 1], out_pixdim=[0.4, 0.4, 1.5], order=3):
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
        
    def gen_t2_and_gland_mask(self):
        assert self.T2_path is not None, f'{self.pid} has no t2 images.'
        assert self.prostate_mask_path is not None, f'{self.pid} has no prostate gland mask.'
        t2_nib = nib.load(self.T2_path)
        t2_arr = t2_nib.get_fdata()
        t2_arr = self.__resample__(t2_arr, method='mm', in_pixdim=t2_nib.header['pixdim'][1:4], out_pixdim=[0.7, 0.7, 0.7], order=3)
        t2_arr = self.__center_crop__(t2_arr, radius=[64, 64, 51])
        t2_arr = self.__normalize__(t2_arr)
        
        mask_nib = nib.load(self.prostate_mask_path)
        mask_arr = mask_nib.get_fdata()
        mask_arr = self.__morph_resample__(mask_arr, method='mm', in_pixdim=mask_nib.header['pixdim'][1:4], out_pixdim=[0.7, 0.7, 0.7])
        mask_arr = self.__center_crop__(mask_arr, radius=[64, 64, 51])
        mask_arr = self.__normalize__(mask_arr)

        return np.stack([t2_arr, mask_arr])
        
        
if __name__=="__main__":

    print('checking patient with both t2 and gland mask images...')
    collections = []
    abandon_list = ['CMICProstate004009']
    for i in glob(os.path.join(CMIC_DATA_PATH, '*'), recursive=False):
        if os.path.isdir(i):
            pat = Patient(i)
            if pat.pid in abandon_list:
                continue
            if (pat.T2_path is not None) and (pat.prostate_mask_path is not None):
                collections.append(pat)
    print(f'{len(collections)} found.')

    vis_path, save_path = './vis', NPY_SAVE_PATH
    # os.makedirs(vis_path, exist_ok=True)
    
    extra_list = []
    for idx, pat in enumerate(collections):
        # if idx<25:
        #     continue
        print(pat.pid)
        t2, mask = pat.gen_t2_and_gland_mask()
        print(f'{idx+1}/{len(collections)}, {t2.shape}, {mask.shape}')
        np.save(os.path.join(save_path, f'Patient{idx+201}-Visit0-T2.npy'), np.stack([t2, mask]))
        extra_list.append(f'Patient{idx+201}-Visit0')
        
        # plt.imsave(os.path.join(vis_path, f'Patient{idx+201}-Visit0-T2.png'), plot_pat(t2, return_img=True), cmap='gray')
        # plt.imsave(os.path.join(vis_path, f'Patient{idx+201}-Visit0-mask.png'), plot_pat(mask, return_img=True), cmap='gray')
    
    # modify the key files    
    with open(key_file, 'rb') as f:
        data_dict = pkl.load(f)

    data_dict['extra'] = extra_list
    with open(key_file, 'wb') as f:
        pkl.dump(data_dict, f)