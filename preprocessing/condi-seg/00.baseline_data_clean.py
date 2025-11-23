import nibabel as nib
import os
import numpy as np
from scipy import ndimage
import nibabel as nib
import matplotlib.pyplot as plt
import cv2


def plot_pat(arr, return_img=False):
    length_x, length_y = arr.shape[0], arr.shape[1]
    edge_num = int(np.sqrt(arr.shape[2])) + 1
    img = np.zeros([length_x*edge_num, length_y*edge_num])
    for z in range(arr.shape[2]):
        x_num = int(z / edge_num)
        y_num = int(z % edge_num)
        img[x_num*length_x:(x_num+1)*length_x, y_num*length_y:(y_num+1)*length_y] = arr[:, :, z]
    if return_img:
        return img
    else:
        plt.figure(figsize=(16, 16))
        plt.imshow(img, cmap='gray')

def normalize0255(arr):
    return (arr - arr.min())*255.0 / (arr.max() - arr.min())

def add_contours(t2, label, color=(255, 0, 0)):
    if len(t2.shape) != 3:
        _t2 = np.tile(t2, (3,1,1)).transpose(1, 2, 0)
    else:
        _t2 = t2
    
    _t2 = normalize0255(_t2).astype('uint8')
    _label = label.astype('uint8')
    blank = np.zeros(_t2.shape)
    contours, hierarchy = cv2.findContours(_label.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE, offset=(0, 0))
    tmp = _t2.copy()  # ?????
    cv2.drawContours(tmp, contours, -1, color, 1)
    print(len(contours))
    return tmp

def center_crop(arr, x, y, z):
    cx, cy, cz = np.array(arr.shape)//2
    assert x<=cx and y<=cy and z<=cz, "rad exceeded the boundary"
    return arr[cx-x:cx+x, cy-y:cy+y, cz-z:cz+z]

def center_crop2(arr, x, y, z):
    cx, cy, cz = np.array(arr.shape)//2
    assert x<=cx and y<=cy and z<=cz, "rad exceeded the boundary"
    return arr[x:-x, y:-y, z:-z]
    

def save_img(tensor_arr, save_path, pixdim=[1.0, 1.0, 1.0]):
    save_folder = os.path.dirname(save_path)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
        
    arr = np.squeeze(tensor_arr)
    assert len(arr.shape)==3, "not a 3 dimentional volume, need to check."

    nib_img = nib.Nifti1Image(arr, affine=np.eye(4))
    nib_img.header['pixdim'][1:4] = np.array(pixdim)
    nib.save(img=nib_img, filename=save_path)
    
    

save_root = '/media/yipeng/data/data/CBCT/fullResCropIntensityClip_resampled'
for k in ['moving_images', 'moving_labels', 'fixed_images', 'fixed_labels']:
    os.makedirs(os.path.join(save_root, k), exist_ok=True)

class CBCTCropped(object):
    def __init__(self, pid):
        self.dataroot = '/media/yipeng/data/data/CBCT/fullResCropIntensityClip'
        self.pid = pid
        
        self.path={}
        self.path['mv_img'] = os.path.join(self.dataroot, 'moving_images', f'{pid}.nii.gz')
        self.path['mv_seg'] = os.path.join(self.dataroot, 'moving_labels', f'{pid}.nii.gz')
        self.path['fx_img'] = os.path.join(self.dataroot, 'fixed_images', f'{pid}.nii.gz')
        self.path['fx_seg'] = os.path.join(self.dataroot, 'fixed_labels', f'{pid}.nii.gz')

    def get_arr(self, mod):
        f = nib.load(self.path[mod])
        pixdim = np.array(f.header['pixdim'][1:4])
        return pixdim, f.get_fdata()


for pid in os.listdir('/media/yipeng/data/data/CBCT/fullResCropIntensityClip/fixed_images'):
    if pid == 'others' or pid == '.DS_Store':
        continue
    else:
        sub = CBCTCropped(pid.replace('.nii.gz', ''))
        print(f'------------{pid}------------')

        mv_pixdim, mv_img = sub.get_arr('mv_img')
        fx_pixdim, fx_img = sub.get_arr('fx_img')
        _, mv_seg = sub.get_arr('mv_seg')
        _, fx_seg = sub.get_arr('fx_seg')
        
        mv_shape = np.array(mv_img.shape)
        fx_shape = np.array(fx_img.shape)

        fx_img = ndimage.zoom(fx_img, fx_pixdim/np.array([2.0 ,2.0, 2.0]), order=2)
        mv_img = ndimage.zoom(mv_img, mv_pixdim/np.array([2.0 ,2.0, 2.0]), order=2)
        mv_img = center_crop2(mv_img, 5, 15, 30)
        

        print(fx_img.shape, mv_img.shape)
        
        tmp = []
        for i in range(fx_seg.shape[-1]):
            resampled_fx = ndimage.zoom(fx_seg[..., i], fx_pixdim/np.array([2.0, 2.0, 2.0]), order=0)
            tmp.append(resampled_fx)
        fx_seg = np.stack(tmp, axis=-1)

        tmp = []
        for i in range(mv_seg.shape[-1]):
            resampled_mv = ndimage.zoom(mv_seg[..., i], mv_pixdim/np.array([2.0, 2.0, 2.0]), order=0)
            resampled_mv = center_crop2(resampled_mv, 5, 15, 30)
            tmp.append(resampled_mv)
        mv_seg = np.stack(tmp, axis=-1)


        # keep the same shape as mv_img, will cause litte differences on the voxels sizes, but sligtly.
        fx_img = ndimage.zoom(fx_img, np.array(mv_img.shape)/np.array(fx_img.shape), order=2)  
        tmp = []
        for i in range(fx_seg.shape[-1]):
            resampled_fx = ndimage.zoom(fx_seg[..., i], np.array(mv_seg[...,i].shape)/np.array(fx_seg[...,i].shape), order=0)
            tmp.append(resampled_fx)
        fx_seg = np.stack(tmp)

        mv_seg = np.transpose(mv_seg, (3, 0, 1, 2))


        # handle some of the image shapes:
        if fx_img.shape != (64, 101, 91):
            fx_img = ndimage.zoom(fx_img, np.array([64, 101, 91])/np.array(fx_img.shape), order=2)

        if mv_img.shape != (64, 101, 91):
            mv_img = ndimage.zoom(mv_img, np.array([64, 101, 91])/np.array(mv_img.shape), order=2)

        tmp= []
        if fx_seg.shape[:-3] != (64, 101, 91):
            for i in range(fx_seg.shape[0]):
                resampled_fx = ndimage.zoom(fx_seg[i], np.array([64, 101, 91])/np.array(fx_seg[i].shape), order=0)
                tmp.append(resampled_fx)
            fx_seg = np.stack(tmp)

        tmp = []
        if mv_seg.shape[:-3] != (64, 101, 91):
            for i in range(mv_seg.shape[0]):
                resampled_mv = ndimage.zoom(mv_seg[i], np.array([64, 101, 91])/np.array(mv_seg[i].shape), order=0)
                tmp.append(resampled_mv)
            mv_seg = np.stack(tmp)


        print(pid, fx_img.shape, fx_seg.shape, mv_img.shape, mv_seg.shape)
        def get_path(sub, mod):
            return sub.path[mod].replace('fullResCropIntensityClip', 'fullResCropIntensityClip_resampled').replace('.nii.gz', '.npy')
        
        np.save(get_path(sub, 'mv_img'), mv_img)
        np.save(get_path(sub, 'fx_img'), fx_img)
        np.save(get_path(sub, 'mv_seg'), mv_seg)
        np.save(get_path(sub, 'fx_seg'), fx_seg)


        #  -----------UNCOMMENT THESE CODE FOR VISULIAZATION ---------------
        # save_img(mv_img, get_path(sub, 'mv_img').replace('.npy', '.nii.gz'), pixdim=[2.0, 2.0, 2.0])
        # save_img(fx_img, get_path(sub, 'fx_img').replace('.npy', '.nii.gz'), pixdim=[2.0, 2.0, 2.0])

        # save_img(mv_seg[0], get_path(sub, 'mv_seg').replace('.npy', '_0.nii.gz'), pixdim=[2.0, 2.0, 2.0])
        # save_img(mv_seg[1], get_path(sub, 'mv_seg').replace('.npy', '_1.nii.gz'), pixdim=[2.0, 2.0, 2.0])
        # save_img(fx_seg[0], get_path(sub, 'fx_seg').replace('.npy', '_0.nii.gz'), pixdim=[2.0, 2.0, 2.0])
        # save_img(fx_seg[1], get_path(sub, 'fx_seg').replace('.npy', '_1.nii.gz'), pixdim=[2.0, 2.0, 2.0])

        # fix2d = plot_pat(fx_img, return_img=True)
        # fxseg2d_0 = plot_pat(fx_seg[0], return_img=True)
        # fxseg2d_1 = plot_pat(fx_seg[1], return_img=True)
        # fix2d = add_contours(fix2d, fxseg2d_0, color=(255, 0, 0))
        # fix2d = add_contours(fix2d, fxseg2d_1, color=(255, 255, 0))
        
        # mv2d = plot_pat(mv_img, return_img=True)
        # mvseg2d_0 = plot_pat(mv_seg[0], return_img=True)
        # mvseg2d_1 = plot_pat(mv_seg[1], return_img=True)
        # mv2d = add_contours(mv2d, mvseg2d_0, color=(255, 0, 0))
        # mv2d = add_contours(mv2d, mvseg2d_1, color=(255, 255, 0))

        # plt.imsave(get_path(sub, 'fx_seg').replace('.npy', '.png'), fix2d, cmap='gray')
        # plt.imsave(get_path(sub, 'mv_seg').replace('.npy', '.png'), mv2d, cmap='gray')
        #  -----------UNCOMMENT THESE CODE FOR VISULIAZATION ---------------
        

        
                
