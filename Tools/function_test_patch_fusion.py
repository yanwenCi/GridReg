import numpy as np
import nibabel as nib
import os

def save_img(tensor_arr, save_path, pixdim=[1.0, 1.0, 1.0]):
    save_folder = os.path.dirname(save_path)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
        
    arr = np.squeeze(tensor_arr)
    assert len(arr.shape)==3, "not a 3 dimentional volume, need to check."

    nib_img = nib.Nifti1Image(arr, affine=np.eye(4))
    nib_img.header['pixdim'][1:4] = np.array(pixdim)
    nib.save(img=nib_img, filename=save_path)


def get_patch_cords_from_image(ps, ipsf, ref_img):
    '''get patch coordinates from a ref img'''

    patch_size = ps  # [48 48 48]
    inf_patch_stride_factors = ipsf  # [4, 4, 4] 

    if len(ref_img.shape) > 3:
        shape = ref_img.shape[-3:]
    else: shape = np.array(ref_img.shape)

    patch_size = np.array(patch_size)
    stride = patch_size // np.array(inf_patch_stride_factors)

    iters = (shape - patch_size) // stride + 1
    coords = [np.array([x, y, z])*stride for x in range(iters[0]) for y in range(iters[1]) for z in range(iters[2])]  # left top points
    coords = [list(i) for i in coords]

    z_slice = [np.array([x, y, shape[2]-patch_size[2]])*np.array([stride[0], stride[1], 1]) for x in range(iters[0]) for y in range(iters[1])]
    z_slice = [list(i) for i in z_slice]
    x_slice = [np.array([shape[0]-patch_size[0], y, z])*np.array([1, stride[1], stride[2]]) for y in range(iters[1]) for z in range(iters[2])]
    x_slice = [list(i) for i in x_slice]
    y_slice = [np.array([x, shape[1]-patch_size[1], z])*np.array([stride[0], 1, stride[2]]) for x in range(iters[0]) for z in range(iters[2])]
    y_slice = [list(i) for i in y_slice]

    zb = [np.array([shape[0]-patch_size[0], shape[1]-patch_size[1], z])*np.array([1, 1, stride[2]]) for z in range(iters[2])]  # z bound
    zb = [list(i) for i in zb]
    xb = [np.array([x, shape[1]-patch_size[1], shape[2]-patch_size[2]])*np.array([stride[0], 1, 1]) for x in range(iters[0])]  # x bound
    xb = [list(i) for i in xb]
    yb = [np.array([shape[0]-patch_size[0], y, shape[2]-patch_size[2]])*np.array([1, stride[1], 1]) for y in range(iters[1])]  # y bound
    yb = [list(i) for i in yb]
    br = [[shape[0]-patch_size[0], shape[1]-patch_size[1], shape[2]-patch_size[2]]]

    # print(len(coords), len(xb), len(yb), len(zb))

    for ex in [zb, xb, yb, br, z_slice, x_slice, y_slice]:
        for p in ex:
            if p not in coords:
                coords.append(p)
    
    return [[x, x+patch_size[0], y, y+patch_size[1], z, z+patch_size[2]] for (x, y, z) in coords]

coords = get_patch_cords_from_image(
    [48, 48, 48],
    [4, 4, 4],
    np.ones([1, 1, 128, 128, 102])
    )
print(len(coords))
[print(i) for i in coords]

count_arr = np.zeros([1, 1, 128, 128, 102])
for (x1, x2, y1, y2, z1, z2) in coords:
    count_arr[:, :, x1:x2, y1:y2, z1:z2] += 1

error_arr = (count_arr==0)*1.0
save_img(count_arr, './patch_fusion_count_arr.nii')
save_img(error_arr, './patch_fusion_error_arr.nii')
    

    


