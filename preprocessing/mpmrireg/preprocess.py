import nibabel as nib
import os
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import cv2


class cmicMPMRIStudy(object):
    def __init__(self, data_path):
        self.data_path = data_path
        self.img_paths = self.get_image_paths()

    def get_image_paths(self):
        img_paths = {}
        for mod in ['t2', 'dwi', 'dwi_b0', 'adc']:
            img_paths[mod] = os.path.join(self.data_path, f'{mod}-0', 'image.nii.gz')
        img_paths['gland_mask'] = os.path.join(self.data_path, 't2-0', 'mask_prostate.nii.gz')
        img_paths['lesions_mask'] = glob(os.path.join(self.data_path, 't2-0', 'mask_lesion_*.nii.gz'))
        return img_paths

    def get_all_info(self):
        reject = False

        for mod in ['t2', 'dwi', 'dwi_b0', 'adc', 'gland_mask']:
            nib_img = nib.load(self.img_paths[mod])
            shape = nib_img.shape
            pixdim = nib_img.header['pixdim'][1:4]
            # print(f'{mod}: {shape}, {pixdim}')
            if mod == 't2':
                t2_shape = shape
                t2_pixdim = pixdim
            elif mod =='dwi':
                dwi_shape = shape
                dwi_pixdim = pixdim
            elif mod =='dwi_b0':
                dwi_b0_shape = shape
                dwi_b0_pixdim = pixdim
            else: pass

        if dwi_shape != dwi_b0_shape:
            # print('dwi has different shape with dwi_b0')
            # print(dwi_shape, dwi_b0_shape)
            print('******')

        if 1 in dwi_shape:
            print('dwi not complete')
            reject = True

        if 1 in dwi_b0_shape:
            print('dwi_b0 not complete')
            reject = True

        if 1 in t2_shape:
            print('t2 not complete')
            reject = True

        if len(self.img_paths['lesions_mask']) == 0:
            print('no lesions found.')
            reject = True

        for lesion_path in self.img_paths['lesions_mask']:
            nib_img = nib.load(lesion_path)
            pixdim = nib_img.header['pixdim'][1:4]
            # print(f'{os.path.basename(lesion_path)}: {nib_img.shape}, {pixdim}')
            if (pixdim != t2_pixdim).any():
                reject = True
                print(f'{lesion_path} has different pixdim with corresponding t2 image' )

        return reject
    
    def get_img(self, mod):
        if 'lesion' not in mod:
            nib_img = nib.load(self.img_paths[mod])
            return nib_img.get_fdata(), nib_img.header['pixdim'][1:4]
        else:
            nib_img_collection = [nib.load(i) for i in self.img_paths[mod]]
            img_arr_collection = [i.get_fdata() for i in nib_img_collection]
            return img_arr_collection, nib_img_collection[0].header['pixdim'][1:4]
        
def center_crop(arr, radius, allow_padding=True):
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


def resample(arr, in_pixdim, out_pixdim, order=3):
    """care about pixel physical size"""
    factor = np.array(in_pixdim) / np.array(out_pixdim)
    return ndimage.zoom(arr, zoom=factor, order=order)

def normalization(arr):
    arr = 255 * (arr - arr.min())/(arr.max() - arr.min())
    return arr.astype('uint8') 

def plot_3d22d(arr, return_img=True):
    arr = normalization(arr)
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

def add_contours(t2, label, color):
    if len(t2.shape) == 2:
        _t2 = np.tile(t2, (3,1,1)).transpose(1, 2, 0)
    else:
        _t2 = t2
    
    if _t2.max() != 255 or _t2.min() != 0:
        _t2 = 255 * (_t2 - _t2.min())/(_t2.max() - _t2.min()) 
        
    _t2 = _t2.astype('uint8')

    _label = label.astype('uint8')
    contours, hierarchy = cv2.findContours(_label.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE, offset=(0, 0))
    tmp = _t2.copy()  # ?????
    cv2.drawContours(tmp, contours, -1, color, 3)
    # print(len(contours))
    return tmp