import os
from glob import glob
import nibabel as nib 
import pickle as pkl
import numpy as np
import argparse
import matplotlib.pyplot as plt
import cv2
import pingouin as pg
from scipy import stats
from glob import glob


parser = argparse.ArgumentParser()
parser.add_argument('--out_pixdim', default=[1.0, 1.0, 1.0], nargs='+', type=float, help='Output pixel size in mm.')
parser.add_argument('--vis', action='store_true', help='Visualize the generated data.')
parser.add_argument('--update_res', action='store_true', help='update the results')

args = parser.parse_args()

usp_dir = './logs/mpmrireg/01-2.unsupervised_gmi0.15_l2n1000'
pr5_dir = './logs/mpmrireg/05-6.pri_gmi0.15_l2n1000_sample5'
pr10_dir = './logs/mpmrireg/05-7.pri_gmi0.15_l2n1000_sample10'
mix_dir = './logs/mpmrireg/02-2.mixed_gmi0.15_l2n1000'
jnt_dir = './logs/mpmrireg/04-1.joint_gmi0.15_l2n1000'
pb0_dir = './logs/mpmrireg/03.b0_gmi0.15_l2n1000'




landmark_colors = ['#8795e2', '#98ea8a', '#fb90a4']  # light blue/yellow/red

contour_color = (230, 0, 0)
bbox_color = (225, 225, 0)

case_dict = {
    'Patient093057400_Study_0':{'slice': 60, 'roi': [(45, 30), (70, 55)], 'ldmk_idx': 1},
    'Patient095506196_Study_0':{'slice': 52, 'roi': [(25, 35), (75, 85)], 'ldmk_idx': 2},
    'Patient559654976_Study_0':{'slice': 61, 'roi': [(40, 40), (52, 52)], 'ldmk_idx': 2},
    'Patient563139412_Study_0':{'slice': 54, 'roi': [(72, 50), (90, 68)], 'ldmk_idx': 2},
    'Patient645116766_Study_0':{'slice': 55, 'roi': [(35, 25), (70, 60)], 'ldmk_idx': 2},
    # 'Patient567565733_Study_0':{'slice': 52, 'roi': [(20, 20), (75, 75)], 'ldmk_idx': []},
}
figure_params = {
    'col_interval': 1,
    'row_interval': 20,
    'col_gap': 10,
    'zoom_col_interval':104,
    'zoom_row_interval':104,
    'zoom_col_gap': 10,
}

case_dict_2 = {
    'Patient452778775_Study_0':{'slice': 50, 'roi': [(23, 38), (55, 70)], 'ldmk_idx': 1},
    'Patient598444984_Study_0':{'slice': 40, 'roi': [(42, 27), (73, 58)], 'ldmk_idx': 2},
    'Patient020061593_Study_0':{'slice': 48, 'roi': [(25, 45), (53, 73)], 'ldmk_idx': 1},
    'Patient714941836_Study_0':{'slice': 46, 'roi': [(20, 20), (85, 85)], 'ldmk_idx': 3},
    'Patient204018851_Study_1':{'slice': 46, 'roi': [(60, 70), (90, 100)], 'ldmk_idx': 2},
    }

case_dict_3 = {
    'Patient452778775_Study_0':{'slice': 50, 'roi': [(23, 38), (55, 70)], 'ldmk_idx': 1},
    'Patient095506196_Study_0':{'slice': 52, 'roi': [(25, 35), (75, 85)], 'ldmk_idx': 2},
    'Patient559654976_Study_0':{'slice': 61, 'roi': [(40, 40), (52, 52)], 'ldmk_idx': 2},
    'Patient563139412_Study_0':{'slice': 54, 'roi': [(72, 50), (90, 68)], 'ldmk_idx': 2},
    }


figure2_params = {
    'interval': 15,
    'zoom_col_interval':15,
    'zoom_row_interval':15,
    'zoom_resize': (60, 60)
    }

figure3_params = {
    'interval': 1,
    }

def get_spec_tre():
    pass

def read_pkl(pkl_path):
    with open(pkl_path, 'rb') as f:
        data = pkl.load(f)
    return data 

def normalize(image):
    image = (image - image.min()) * 255 / (image.max() - image.min())
    image = image.astype('uint8')

    if len(image.shape)==3 and image.shape[-1]==3:
        pass
    else:
        image = np.stack([image]*3, axis=2)

    return image

def get_results(exp_dir):
    with open(os.path.join(exp_dir, 'results.pkl'), 'rf') as f:
        data = pkl.load(f)
    return data

def get_vis(exp_dir):
    vis_dir = glob(os.path.join(exp_dir, 'vis-*'))
    vis_dir = [i for i in vis_dir if '_' not in os.path.basename(i)]
    assert len(vis_dir) == 1, f"multiple or no vis folders found in {exp_dir}"
    return vis_dir[0]

def nib_load(nii_path):
    return nib.load(nii_path).get_fdata().transpose(1, 0, 2)[::-1, ::-1, :]

def update_results():
    os.system(f'python test.py {usp_dir}')
    os.system(f'python test.py {pr5_dir}')
    os.system(f'python test.py {pr10_dir}')
    os.system(f'python test.py {mix_dir}')
    os.system(f'python test.py {jnt_dir}')
    os.system(f'python test.py {pb0_dir}')

def add_interval(image, interval, axis, alpha=False):
    
    new_shape = list(image.shape)
    new_shape[axis] += interval

    if len(new_shape)==3 and new_shape[2]==3:
        new_image = np.ones(new_shape)*255
        new_image[:image.shape[0], :image.shape[1], ...] = image
        
        if alpha:
            alpha_channel = np.zeros([new_shape[0], new_shape[1], 1])
            alpha_channel[:image.shape[0], :image.shape[1], ...] = 255
            new_image = np.concatenate([new_image, alpha_channel], axis=2)
        
    elif len(image.shape)==3 and image.shape[2]==4:
        if alpha:
            new_image = np.zeros(new_shape)
        else:
            new_image = np.ones(new_shape) * 255 
        new_image[:image.shape[0], :image.shape[1], ...] = image

    else:
        print(f"image shape {image.shape} can not be processed.")
        raise NotImplementedError

    return new_image

def concatenate_with_interval(img_list, interval, axis, alpha=False):
    tmp = []
    for img in img_list:
        new_img = add_interval(img, interval, axis, alpha=alpha)
        tmp.append(new_img)
    cat_image = np.concatenate(tmp, axis=axis)
    cat_image = cat_image.astype('uint8')
    
    return cat_image 

def get_zoom_roi(slice, roi):
    topLeft, bottomRight = roi
    x1, y1 = topLeft
    x2, y2 = bottomRight
    crop_img = slice[y1:y2, x1:x2, ...]
    crop_img = cv2.resize(crop_img, (75, 75), interpolation=cv2.INTER_LINEAR)

    return crop_img

def draw_roi(slice, roi):
    topLeft, bottomRight = roi
    slice = cv2.rectangle(slice, topLeft, bottomRight, bbox_color, 0)

    return slice

def draw_edge(slice):
    x, y, _ = slice.shape
    topLeft, bottomRight = (0, 0), (x-1, y-1)
    slice = cv2.rectangle(slice, topLeft, bottomRight, bbox_color, 0)
    return slice

def get_ref_column(case_dict, figure_params, alpha=False):
    '''get fix mv and pri images'''

    case_collection = []
    zoom_collection = []
    tres = read_pkl('./logs/mpmrireg/01-2.unsupervised_gmi0.15_l2n1000/tre_dic.pkl')
    tres_collection = []

    for pid, info in case_dict.items():
        slice, roi, ldmk_idx = info['slice']-1, info['roi'], info['ldmk_idx']
        image_folder = os.path.join(
            get_vis('./logs/mpmrireg/01-2.unsupervised_gmi0.15_l2n1000'), 
            pid
        )

        bef_tre, _ = tres[pid][0]

        # get image and ldmk
        fx_img = normalize(nib_load(os.path.join(image_folder, 'fx_img.nii'))[:, :, slice])
        mv_img = normalize(nib_load(os.path.join(image_folder, 'mv_img.nii'))[:, :, slice])
        pr_img = normalize(nib_load(os.path.join(image_folder, 'pr_img.nii'))[:, :, slice])

        # put text of the tre on the mv image or print it.
        # cv2.putText(mv_img, f"{bef_tre:.2f}", (30, 10), cv2.FONT_HERSHEY_PLAIN, 0.75, (255,255,0), 0)
        tres_collection.append(bef_tre)

        # draw contours on fx and mv image
        if ldmk_idx != []:
            fx_ldmk = normalize(nib_load(os.path.join(image_folder, f't2_ldmk_{ldmk_idx}_fx.nii'))[:, :, slice])
            mv_ldmk = normalize(nib_load(os.path.join(image_folder, f'dwi_ldmk_{ldmk_idx}_mv.nii'))[:, :, slice])

            fx_img = add_contours(fx_img, fx_ldmk, color=contour_color)
            mv_img = add_contours(mv_img, mv_ldmk, color=contour_color)
        
        fx_zoom = draw_edge(get_zoom_roi(fx_img, roi))
        mv_zoom = draw_edge(get_zoom_roi(mv_img, roi))
        pr_zoom = draw_edge(get_zoom_roi(pr_img, roi))
        
        fx_img = draw_roi(fx_img, roi)
        mv_img = draw_roi(mv_img, roi)
        pr_img = draw_roi(pr_img, roi)
        
        single_case = concatenate_with_interval(
            [pr_img, fx_img, mv_img], 
            figure_params['col_interval'],
            axis=1,
            alpha=alpha
            )
        single_zoom = concatenate_with_interval(
            [pr_zoom, fx_zoom, mv_zoom],
            figure_params['zoom_col_interval'],
            axis=1,
            alpha=alpha
        )
        case_collection.append(single_case)
        zoom_collection.append(single_zoom)

    figs = concatenate_with_interval(case_collection, figure_params['row_interval'], axis=0, alpha=alpha)
    zoom = concatenate_with_interval(
        zoom_collection, figure_params['zoom_row_interval']+figure_params['row_interval']+15, axis=0, alpha=alpha)

    print('before landmarks for each case:')
    [print(f"{i:.2f}") for i in tres_collection]
    print("-----------------------")

    return figs, zoom

def get_method_column(method_path, case_dict, figure_params, alpha=False):
    case_collection = []
    zoom_collection = []

    print(method_path)

    for pid, info in case_dict.items():
        slice, roi, ldmk_idx = info['slice']-1, info['roi'], info['ldmk_idx']
        image_folder = os.path.join(
            get_vis(method_path), 
            pid
        )
        wp_img = normalize(nib_load(os.path.join(image_folder, 'wp_mv_img.nii'))[:, :, slice])
        wppr_img = normalize(nib_load(os.path.join(image_folder, 'wp_pr_img.nii'))[:, :, slice])

        if ldmk_idx != []:
            wp_ldmk = normalize(nib_load(os.path.join(image_folder, f'dwi_ldmk_{ldmk_idx}_wp.nii'))[:, :, slice])            
            wp_img = add_contours(wp_img, wp_ldmk, color=contour_color)


        wp_zoom = draw_edge(get_zoom_roi(wp_img, roi))
        wppr_zoom = draw_edge(get_zoom_roi(wppr_img, roi))

        wp_img = draw_roi(wp_img, roi)
        wppr_img = draw_roi(wppr_img, roi)

        single_case = concatenate_with_interval(
            [wp_img, wppr_img], 
            figure_params['col_interval'],
            axis=1,
            alpha=alpha
            )
        single_zoom = concatenate_with_interval(
            [wp_zoom, wppr_zoom],
            figure_params['zoom_col_interval'],
            axis=1,
            alpha=alpha
        )
        case_collection.append(single_case)
        zoom_collection.append(single_zoom)

        tre_dict = read_pkl(os.path.join(
            os.path.dirname(os.path.dirname(image_folder)),
            'tre_dic.pkl' 
            ))

        _, aft_reg = tre_dict[pid][0]
        print(f"{aft_reg:.2f}")

    figs = concatenate_with_interval(case_collection, figure_params['row_interval'], axis=0, alpha=alpha)
    zoom = concatenate_with_interval(
        zoom_collection, figure_params['zoom_row_interval']+figure_params['row_interval']+15, axis=0, alpha=alpha)
    return figs, zoom


def gen_figure(case_dict, figure_params):
    ref_col, ref_zoom = get_ref_column(case_dict, figure_params, alpha=True)
    usp_col, usp_zoom = get_method_column(usp_dir, case_dict, figure_params, alpha=True)
    jnt_col, jnt_zoom = get_method_column(jnt_dir, case_dict, figure_params, alpha=True)
    mix_col, mix_zoom = get_method_column(mix_dir, case_dict, figure_params, alpha=True)
    pri_col, pri_zoom = get_method_column(pr5_dir, case_dict, figure_params, alpha=True)
    
    base_figure = concatenate_with_interval(
        [ref_col, pri_col, jnt_col, mix_col, usp_col],
        figure_params['col_gap'], 
        axis=1,
        alpha=True
    )
    
    float_zoom = concatenate_with_interval(
        [ref_zoom, pri_zoom, jnt_zoom, mix_zoom, usp_zoom],
        figure_params['zoom_col_gap']+figure_params['col_gap'], 
        axis=1,
        alpha=True
    )

    return base_figure, float_zoom

def gen_figure2(case_dict_2, figure2_params):

    print('figure2 ldmk data')

    def padding(img, interval):
        s = np.array(img.shape)
        tmp = np.ones(s+np.array([interval*2, interval*2, 0])) * 255
        tmp[interval:s[0]+interval, interval:s[1]+interval, :] = img
        return tmp

    def padding_zero(img, interval):
        s = np.array(img.shape)
        tmp = np.zeros(s+np.array([interval*2, interval*2, 0]))
        tmp[interval:s[0]+interval, interval:s[1]+interval, :] = img
        return tmp.astype('uint8')


    def process_single_zoom(slice, roi, params=figure2_params):
        (x1, y1), (x2, y2) = roi
        zoom = slice[y1:y2, x1:x2, :]
        zoom = cv2.resize(zoom, params['zoom_resize'], interpolation=cv2.INTER_LINEAR)
        zoom = draw_edge(zoom)
        return zoom

    def flip_lr(img):
        return img[:, ::-1, ...]
    
    interval = figure2_params['interval']

    cols = []
    zooms = []
    for pid, v in case_dict_2.items():
        image_folder = os.path.join(
            get_vis('./logs/mpmrireg/05-6.pri_gmi0.15_l2n1000_sample5'), 
            pid
        )

        slice = v['slice'] - 1
        (x1, y1), (x2, y2) = v['roi']
        roi = [(x1+interval, y1+interval), (x2+interval, y2+interval)]

        ldmk_idx = v['ldmk_idx']

        
        fx_img = padding(normalize(nib_load(os.path.join(image_folder, 'fx_img.nii'))[:, :, slice]), interval)
        mv_img = padding(normalize(nib_load(os.path.join(image_folder, 'mv_img.nii'))[:, :, slice]), interval)
        pr_img = padding(normalize(nib_load(os.path.join(image_folder, 'pr_img.nii'))[:, :, slice]), interval)
        wp_img = padding(normalize(nib_load(os.path.join(image_folder, 'wp_mv_img.nii'))[:, :, slice]), interval)
        wppr_img = padding(normalize(nib_load(os.path.join(image_folder, 'wp_pr_img.nii'))[:, :, slice]), interval)

        
        # draw contours on fx and mv image
        if ldmk_idx != []:
            fx_ldmk = padding_zero(normalize(nib_load(os.path.join(image_folder, f't2_ldmk_{ldmk_idx}_fx.nii'))[:, :, slice]), interval)
            mv_ldmk = padding_zero(normalize(nib_load(os.path.join(image_folder, f'dwi_ldmk_{ldmk_idx}_mv.nii'))[:, :, slice]), interval)
            wp_ldmk = padding_zero(normalize(nib_load(os.path.join(image_folder, f'dwi_ldmk_{ldmk_idx}_wp.nii'))[:, :, slice]), interval)     

            fx_img = add_contours(fx_img, fx_ldmk, color=contour_color)
            mv_img = add_contours(mv_img, mv_ldmk, color=contour_color)
            wp_img = add_contours(wp_img, wp_ldmk, color=contour_color)
        

        fx_img = draw_roi(fx_img, roi)
        mv_img = draw_roi(mv_img, roi)
        pr_img = draw_roi(pr_img, roi)    
        wp_img = draw_roi(wp_img, roi)
        wppr_img = draw_roi(wppr_img, roi)

        fx_zoom = draw_edge(process_single_zoom(fx_img, roi))
        mv_zoom = draw_edge(process_single_zoom(mv_img, roi))
        pr_zoom = draw_edge(process_single_zoom(pr_img, roi))
        wp_zoom = draw_edge(process_single_zoom(wp_img, roi))
        wppr_zoom = draw_edge(process_single_zoom(wppr_img, roi))

        if pid == 'Patient204018851_Study_1':
            fx_img, mv_img, pr_img, wp_img, wppr_img = flip_lr(fx_img), flip_lr(mv_img), flip_lr(pr_img), flip_lr(wp_img), flip_lr(wppr_img)
            fx_zoom, mv_zoom, pr_zoom, wp_zoom, wppr_zoom = flip_lr(fx_zoom), flip_lr(mv_zoom), flip_lr(pr_zoom), flip_lr(wp_zoom), flip_lr(wppr_zoom)

        tre_dict = read_pkl(os.path.join(
            os.path.dirname(os.path.dirname(image_folder)),
            'tre_dic.pkl' 
            ))

        bef_reg, aft_reg = tre_dict[pid][0]
        print(f"{bef_reg:.2f}, {aft_reg:.2f}")

        row = np.concatenate([fx_img, mv_img, wp_img, pr_img, wppr_img], axis=1)
        cols.append(row)
        zooms.append([fx_zoom, mv_zoom, wp_zoom, pr_zoom, wppr_zoom])

    fig2 = np.concatenate(cols, axis=0)
    fig2 = padding(fig2, interval=25)

    zoom_start_point = (110, 110)

    image_x, image_y = 104, 104
    zoom_x, zoom_y = figure2_params['zoom_resize']
    for row, zoom_sub_list in enumerate(zooms):
        for col, zoom in enumerate(zoom_sub_list):
            # cor_x = zoom_start_point[0] + row*(figure2_params['interval']+image_x+figure2_params['zoom_col_interval'])
            # cor_y = zoom_start_point[1] + col*(figure2_params['interval']+image_y+figure2_params['zoom_row_interval'])
            cor_x = zoom_start_point[0] + row*(figure2_params['interval']*2+image_x)
            cor_y = zoom_start_point[1] + col*(figure2_params['interval']*2+image_y)
            fig2[cor_x:cor_x+zoom_x, cor_y:cor_y+zoom_y, :] = zoom

    return fig2


def gen_figure3(case_dict_3, figure3_params):

    def padding(img, interval):
        s = np.array(img.shape)
        tmp = np.ones(s+np.array([interval*2, interval*2, 0])) * 255
        tmp[interval:s[0]+interval, interval:s[1]+interval, :] = img
        return tmp
    
    interval = figure3_params['interval']

    cols = []
    for pid, v in case_dict_3.items():
        image_folder = os.path.join(
            get_vis('./logs/mpmrireg/05-6.pri_gmi0.15_l2n1000_sample5'), 
            pid
        )
        slice = v['slice'] - 1
        ldmk_idx = v['ldmk_idx']
        
        fx_img = padding(normalize(nib_load(os.path.join(image_folder, 'fx_img.nii'))[:, :, slice]), interval)
        mv_img = padding(normalize(nib_load(os.path.join(image_folder, 'mv_img.nii'))[:, :, slice]), interval)
        pr_img = padding(normalize(nib_load(os.path.join(image_folder, 'pr_img.nii'))[:, :, slice]), interval)

        # draw contours on fx and mv image
        if ldmk_idx != []:
            fx_ldmk = normalize(nib_load(os.path.join(image_folder, f't2_ldmk_{ldmk_idx}_fx.nii'))[:, :, slice])
            mv_ldmk = normalize(nib_load(os.path.join(image_folder, f'dwi_ldmk_{ldmk_idx}_mv.nii'))[:, :, slice]) 

            fx_img_with_ldmk = add_contours(fx_img, fx_ldmk, color=(220, 220, 0))
            mv_img_with_ldmk = add_contours(mv_img, mv_ldmk, color=(220, 220, 0))

        row = np.concatenate([fx_img, fx_img_with_ldmk, mv_img, mv_img_with_ldmk, pr_img], axis=1)
        cols.append(row)

    fig3 = np.concatenate(cols, axis=0)

    return fig3

def get_results():
    usp_res = os.path.join(usp_dir, 'results.pkl')
    pr5_res = os.path.join(pr5_dir, 'results.pkl')
    pr10_res = os.path.join(pr10_dir, 'results.pkl')
    mix_res = os.path.join(mix_dir, 'results.pkl')
    jnt_res = os.path.join(jnt_dir, 'results.pkl')
    pb0_res = os.path.join(pb0_dir, 'results.pkl')
    nft_res = './Others/niftyreg-test/results.pkl'

    res_dict = {
        'usp': read_pkl(usp_res),
        'pr5': read_pkl(pr5_res),
        'pr10': read_pkl(pr10_res),
        'mix': read_pkl(mix_res),
        'jnt': read_pkl(jnt_res),
        'nft': read_pkl(nft_res),
        'pb0': read_pkl(pb0_res)
    }
    return res_dict


def get_external_results():
    usp_res = os.path.join(usp_dir, 'results_miami.pkl')
    pr5_res = os.path.join(pr5_dir, 'results_miami.pkl')
    mix_res = os.path.join(mix_dir, 'results_miami.pkl')
    jnt_res = os.path.join(jnt_dir, 'results_miami.pkl')
    nft_res = './Others/niftyreg-test-miami/results_miami.pkl'

    miami_dict = {
        'usp': read_pkl(usp_res),
        'pr5': read_pkl(pr5_res),
        'mix': read_pkl(mix_res),
        'jnt': read_pkl(jnt_res),
        'nft': read_pkl(nft_res),
    }

    usp_res = os.path.join(usp_dir, 'results_cia.pkl')
    pr5_res = os.path.join(pr5_dir, 'results_cia.pkl')
    mix_res = os.path.join(mix_dir, 'results_cia.pkl')
    jnt_res = os.path.join(jnt_dir, 'results_cia.pkl')
    nft_res = './Others/niftyreg-test-cia/results_cia.pkl'

    cia_dict = {
        'usp': read_pkl(usp_res),
        'pr5': read_pkl(pr5_res),
        'mix': read_pkl(mix_res),
        'jnt': read_pkl(jnt_res),
        'nft': read_pkl(nft_res),
    }

    res_dict = {
        'miami': miami_dict,
        'cia': cia_dict
    }

    
    return res_dict


def get_statistics(l):
    mean = np.around(np.mean(l), decimals=3)  # np.around not works in f expression
    std = np.around(np.std(l), decimals=3)
    median = np.around(np.median(l), decimals=3)
    pctl = np.around(np.percentile(l, 90), decimals=3)
    
    return [f"{mean:.3f}", f"{std:.3f}", f"{median:.3f}", f"{pctl:.3f}"]


def get_statistics_tab2(l):
    mean = np.around(np.mean(l), decimals=3)
    std = np.around(np.std(l), decimals=3)

    return [f"{mean:.2f}$\pm${std:.2f}"]
    

def update_table_1():
    res_dict = get_results()
    
    wo = get_statistics(res_dict['usp']['mi-wo-reg']) + get_statistics(res_dict['usp']['tre-wo-reg'])
    nf = get_statistics(res_dict['nft']['mi']) + get_statistics(res_dict['nft']['ldmk'])
    us = get_statistics(res_dict['usp']['mi']) + get_statistics(res_dict['usp']['tre'])
    jn = get_statistics(res_dict['jnt']['mi']) + get_statistics(res_dict['jnt']['tre'])
    mx = get_statistics(res_dict['mix']['mi']) + get_statistics(res_dict['mix']['tre'])
    pr = get_statistics(res_dict['pr5']['mi']) + get_statistics(res_dict['pr5']['tre'])

    table_content = \
        "\\begin{table*}[tp] \n" + \
        "\\centering \n" + \
        "\\caption{Holdout set performance for registering T2w-DWI$_{b=2000}$} \n" + \
        "\\resizebox{1.0\\textwidth}{!}{ \n" + \
        "\\begin{tabular}{c|c|c|c} \n" + \
        "\\toprule \n" + \
        "Methods            & Training input         & MI: Mean, Median, 90$^{th}$ Pctl. & TRE: Mean, Median, 90$^{th}$ Pctl. (mm) \\\\ \n" + \
        "\\midrule \n" + \
        "w/o registration   &         -              " + f"& {wo[0]}$\pm${wo[1]}, {wo[2]}, {wo[3]} & {wo[4]}$\pm${wo[5]}, {wo[6]}, {wo[7]} \\\\ \n" + \
        "NiftyReg           & non-learning method    " + f"& {nf[0]}$\pm${nf[1]}, {nf[2]}, {nf[3]} & {nf[4]}$\pm${nf[5]}, {nf[6]}, {nf[7]} \\\\ \n" + \
        "Direct             & T2w-DWI$_{b=2000}$     " + f"& {us[0]}$\pm${us[1]}, {us[2]}, {us[3]} & {us[4]}$\pm${us[5]}, {us[6]}, {us[7]} \\\\ \n" + \
        "Mixed              & T2w-DWI$_{b=0,b=2000}$ " + f"& {mx[0]}$\pm${mx[1]}, {mx[2]}, {mx[3]} & {mx[4]}$\pm${mx[5]}, {mx[6]}, {mx[7]} \\\\ \n" + \
        "Joint              & T2w-DWI$_{b=0,b=2000}$ " + f"& {jn[0]}$\pm${jn[1]}, {jn[2]}, {jn[3]} & {jn[4]}$\pm${jn[5]}, {jn[6]}, {jn[7]} \\\\ \n" + \
        "Privileged Sup.    & T2w-DWI$_{b=0,b=2000}$ " + f"& {pr[0]}$\pm${pr[1]}, {pr[2]}, {pr[3]} & {pr[4]}$\pm${pr[5]}, {pr[6]}, {pr[7]} \\\\ \n" + \
        "\\bottomrule \n" + \
        "\\end{tabular}} \n" + \
        "\\label{tab:results} \n" + \
        "\\end{table*} \n"
    print(table_content)


def update_table_2():

    def merge_dict(b, a):
        '''merge b 2 a'''
        for k in a.keys():
            a[k].extend(b[k])
        return a

    def dict_like(ref_dict):
        tmp = {}
        for k in res_dict.keys():
            tmp[k] = []
        return tmp

    res_dict = get_results()
    results = dict_like(res_dict)

    def calc_topX_improvement(res_dict, portion):
        tmp = dict_like(res_dict)

        for metric in ['mi', 'tre']:
            num_cases = int(np.ceil(portion*len(res_dict['usp'][metric])))
            for exp, res in res_dict.items():
                if metric == 'tre' and exp=='nft':
                    met = 'ldmk'
                else:
                    met = metric
                bef_reg = np.array(res[f'{met}-wo-reg'])
                aft_reg = np.array(res[f'{met}'])

                if metric == 'tre':
                    improvements = aft_reg - bef_reg
                else:
                    improvements = bef_reg - aft_reg
                idxs = np.argpartition(improvements, num_cases)[:num_cases]
                
                bef_topX_res = bef_reg[idxs]
                aft_topX_res = aft_reg[idxs]
                tmp[exp].extend(get_statistics_tab2(bef_topX_res))
                tmp[exp].extend(get_statistics_tab2(aft_topX_res))

        return tmp

    def calc_topX_misalignment(res_dict, portion):
        tmp = dict_like(res_dict)

        for metric in ['mi', 'tre']:
            num_cases = int(np.ceil(portion*len(res_dict['usp'][metric])))

            for i, (exp, res) in enumerate(res_dict.items()):
                if metric == 'tre' and exp=='nft':
                    met = 'ldmk'
                else:
                    met = metric

                if metric == 'mi':
                    idxs = np.argpartition(np.array(res[f'{met}-wo-reg']), num_cases)[:num_cases]
                else:
                    idxs = np.argpartition(-np.array(res[f'{met}-wo-reg']), num_cases)[:num_cases]

                bef_reg = np.array(res[f'{met}-wo-reg'])
                aft_reg = np.array(res[f'{met}'])
                
                bef_topX_res = bef_reg[idxs]
                aft_topX_res = aft_reg[idxs]

                tmp[exp].extend(get_statistics_tab2(aft_topX_res))
                
                if i==0:
                    print('----------------')
                    print(metric, portion, f"mean:{np.mean(bef_topX_res):.3f}, std:{np.std(bef_topX_res):.3f}")
                    print('----------------')


        return tmp

        

    for portion in [0.1, 0.2]:
        tmp_res = calc_topX_misalignment(res_dict, portion)
        merge_dict(tmp_res, results)

    for portion in [0.1, 0.2]:
        tmp_res = calc_topX_improvement(res_dict, portion)
        merge_dict(tmp_res, results)
        
    nf = results['nft']
    us = results['usp']
    jn = results['jnt']
    mx = results['mix']
    pr = results['pr5']

    table_content = \
        "\\begin{table*} \n" + \
        "\\centering \n" + \
        "\\caption{Holdout set performance in 10\\% and 20\\% samples with the largest initial misalignment (pre-registration independent stratification) and those with the most improvement (selective results for reference).} \n" + \
        "\\resizebox{0.95\\textwidth}{!}{ \n" + \
        "\\begin{tabular}{c|cc|cc|cccc|cccc} \n" + \
        "\\hline \n" + \
        "& \multicolumn{2}{c|}{\\begin{tabular}[c]{@{}c@{}}10\% with largest\\\\ initial misalignment\end{tabular}} & \multicolumn{2}{c|}{\\begin{tabular}[c]{@{}c@{}}20\% with largest\\\\ initial misalignment\end{tabular}} & \multicolumn{4}{c|}{\\begin{tabular}[c]{@{}c@{}}10\% with most \\\\ improvement (selective)\end{tabular}} & \multicolumn{4}{c}{\\begin{tabular}[c]{@{}c@{}}20\% with most \\\\ improvement (selective)\end{tabular}} \\\\ \cline{6-13} \n" + \
        "&                                                           &                                           &                                                           &                                           & \multicolumn{2}{c|}{Before}                                   & \multicolumn{2}{c|}{After}             & \multicolumn{2}{c|}{Before}                                  & \multicolumn{2}{c}{After}              \\\\ \cline{2-13}  \n" + \
        "& \multicolumn{1}{c|}{MI}                                   & TREs(mm)                                  & \multicolumn{1}{c|}{MI}                                   & TREs(mm)                                  & \multicolumn{1}{c|}{MI}    & \multicolumn{1}{c|}{TREs(mm)}    & \multicolumn{1}{c|}{MI}   & TREs(mm)   & \multicolumn{1}{c|}{MI}    & \multicolumn{1}{c|}{TREs(mm)}   & \multicolumn{1}{c|}{MI}   & TREs(mm)   \\\\ \\hline \n" + \
        f"NiftyReg   & \multicolumn{{1}}{{c|}}{{{nf[0]}}} & {nf[1]} & \multicolumn{{1}}{{c|}}{{{nf[2]}}}  & {nf[3]} & \multicolumn{{1}}{{c|}}{{{nf[4]}}} & \multicolumn{{1}}{{c|}}{{{nf[6]}}} & \multicolumn{{1}}{{c|}}{{{nf[5]}}} & {nf[7]} & \multicolumn{{1}}{{c|}}{{{nf[8]}}} & \multicolumn{{1}}{{c|}}{{{nf[10]}}} & \multicolumn{{1}}{{c|}}{{{nf[9]}}} & {nf[11]}  \\\\ \n" + \
        f"Direct     & \multicolumn{{1}}{{c|}}{{{us[0]}}} & {us[1]} & \multicolumn{{1}}{{c|}}{{{us[2]}}}  & {us[3]} & \multicolumn{{1}}{{c|}}{{{us[4]}}} & \multicolumn{{1}}{{c|}}{{{us[6]}}} & \multicolumn{{1}}{{c|}}{{{us[5]}}} & {us[7]} & \multicolumn{{1}}{{c|}}{{{us[8]}}} & \multicolumn{{1}}{{c|}}{{{us[10]}}} & \multicolumn{{1}}{{c|}}{{{us[9]}}} & {us[11]}  \\\\ \n" + \
        f"Mixed      & \multicolumn{{1}}{{c|}}{{{mx[0]}}} & {mx[1]} & \multicolumn{{1}}{{c|}}{{{mx[2]}}}  & {mx[3]} & \multicolumn{{1}}{{c|}}{{{mx[4]}}} & \multicolumn{{1}}{{c|}}{{{mx[6]}}} & \multicolumn{{1}}{{c|}}{{{mx[5]}}} & {mx[7]} & \multicolumn{{1}}{{c|}}{{{mx[8]}}} & \multicolumn{{1}}{{c|}}{{{mx[10]}}} & \multicolumn{{1}}{{c|}}{{{mx[9]}}} & {mx[11]}  \\\\ \n" + \
        f"Joint      & \multicolumn{{1}}{{c|}}{{{jn[0]}}} & {jn[1]} & \multicolumn{{1}}{{c|}}{{{jn[2]}}}  & {jn[3]} & \multicolumn{{1}}{{c|}}{{{jn[4]}}} & \multicolumn{{1}}{{c|}}{{{jn[6]}}} & \multicolumn{{1}}{{c|}}{{{jn[5]}}} & {jn[7]} & \multicolumn{{1}}{{c|}}{{{jn[8]}}} & \multicolumn{{1}}{{c|}}{{{jn[10]}}} & \multicolumn{{1}}{{c|}}{{{jn[9]}}} & {jn[11]}  \\\\ \n" + \
        f"Privileged & \multicolumn{{1}}{{c|}}{{{pr[0]}}} & {pr[1]} & \multicolumn{{1}}{{c|}}{{{pr[2]}}}  & {pr[3]} & \multicolumn{{1}}{{c|}}{{{pr[4]}}} & \multicolumn{{1}}{{c|}}{{{pr[6]}}} & \multicolumn{{1}}{{c|}}{{{pr[5]}}} & {pr[7]} & \multicolumn{{1}}{{c|}}{{{pr[8]}}} & \multicolumn{{1}}{{c|}}{{{pr[10]}}} & \multicolumn{{1}}{{c|}}{{{pr[9]}}} & {pr[11]}  \\\\ \\hline \n" + \
        "\\end{tabular}} \n" + \
        "\\label{tab:results.top} \n" + \
        "\\end{table*} \n"

    print(table_content)

def update_table_3():
    resdict = get_external_results()
    
    wo = get_statistics_tab2(resdict['miami']['usp']['mi-wo-reg']) + get_statistics_tab2(resdict['miami']['usp']['tre-wo-reg'])
    nf = get_statistics_tab2(resdict['miami']['nft']['mi']) + get_statistics_tab2(resdict['miami']['nft']['ldmk'])
    us = get_statistics_tab2(resdict['miami']['usp']['mi']) + get_statistics_tab2(resdict['miami']['usp']['tre'])
    jn = get_statistics_tab2(resdict['miami']['jnt']['mi']) + get_statistics_tab2(resdict['miami']['jnt']['tre'])
    mx = get_statistics_tab2(resdict['miami']['mix']['mi']) + get_statistics_tab2(resdict['miami']['mix']['tre'])
    pr = get_statistics_tab2(resdict['miami']['pr5']['mi']) + get_statistics_tab2(resdict['miami']['pr5']['tre'])

    wo += get_statistics_tab2(resdict['cia']['usp']['mi-wo-reg']) + get_statistics_tab2(resdict['cia']['usp']['tre-wo-reg'])
    nf += get_statistics_tab2(resdict['cia']['nft']['mi']) + get_statistics_tab2(resdict['cia']['nft']['ldmk'])
    us += get_statistics_tab2(resdict['cia']['usp']['mi']) + get_statistics_tab2(resdict['cia']['usp']['tre'])
    jn += get_statistics_tab2(resdict['cia']['jnt']['mi']) + get_statistics_tab2(resdict['cia']['jnt']['tre'])
    mx += get_statistics_tab2(resdict['cia']['mix']['mi']) + get_statistics_tab2(resdict['cia']['mix']['tre'])
    pr += get_statistics_tab2(resdict['cia']['pr5']['mi']) + get_statistics_tab2(resdict['cia']['pr5']['tre'])

    table_content = \
        "\\begin{table}[tp] \n" + \
        "\\centering \n" + \
        "\\caption{T2w-DWI$_{high-b}$ registration performance on external validation data sets.} \n" + \
        "\\label{tab:external} \n" + \
        "\\resizebox{\\columnwidth}{!}{% \n" + \
        "\\begin{tabular}{c|cc|cc} \n" + \
        "\\hline \n" + \
        "& \multicolumn{2}{c|}{Data set A} & \multicolumn{2}{c}{Data set B} \\\\ \\cline{2-5} \n" + \
        "& \multicolumn{1}{c|}{MI}  & TREs(mm) & \multicolumn{1}{c|}{MI} & TREs(mm) \\\\ \\hline \n" + \
        f"w/o registration & \multicolumn{{1}}{{c|}}{{{wo[0]}}}    & {wo[1]}     & \multicolumn{{1}}{{c|}}{{{wo[2]}}}   & {wo[3]}     \\\\ \n" + \
        f"NiftyReg         & \multicolumn{{1}}{{c|}}{{{nf[0]}}}    & {nf[1]}     & \multicolumn{{1}}{{c|}}{{{nf[2]}}}   & {nf[3]}     \\\\ \n" + \
        f"Direct           & \multicolumn{{1}}{{c|}}{{{us[0]}}}    & {us[1]}     & \multicolumn{{1}}{{c|}}{{{us[2]}}}   & {us[3]}     \\\\ \n" + \
        f"Mixed            & \multicolumn{{1}}{{c|}}{{{mx[0]}}}    & {mx[1]}     & \multicolumn{{1}}{{c|}}{{{mx[2]}}}   & {mx[3]}    \\\\ \n" + \
        f"Joint            & \multicolumn{{1}}{{c|}}{{{jn[0]}}}    & {jn[1]}     & \multicolumn{{1}}{{c|}}{{{jn[2]}}}   & {jn[3]}     \\\\ \n" + \
        f"Privileged       & \multicolumn{{1}}{{c|}}{{{pr[0]}}}    & {pr[1]}     & \multicolumn{{1}}{{c|}}{{{pr[2]}}}   & {pr[3]}     \\\\ \\hline \n" + \
        "\\end{tabular}} \n" + \
        "\\end{table} \n"
    print(table_content)



def BA_plot():
    import matplotlib
    matplotlib.rc('xtick', labelsize=9) 
    matplotlib.rc('ytick', labelsize=9) 
    font = {'size': 9}
    matplotlib.rc('font', **font)

    res_dict = get_results()

    fig, axs = plt.subplots(2, 4)
    fig.set_size_inches(16, 6)

    # TREs on the first row
    
    exp = 'pr5'
    tre = res_dict[exp]['tre']
    tre_wo_reg = res_dict[exp]['tre-wo-reg']
    point_color = [landmark_colors[i] for i in res_dict[exp]['ldmk-type']]
    pg.plot_blandaltman(tre, tre_wo_reg, scatter_kws={'s':25,'alpha':0.8, 'c':point_color}, ax=axs[0, 3], xaxis="y")
    axs[0, 3].set_xlabel('Method: Privileged - TREs')
    axs[0, 3].set_ylabel('')
    axs[0, 3].axhline(0, color='r', linestyle=':', lw=1)

    exp = 'jnt'
    tre = res_dict[exp]['tre']
    tre_wo_reg = res_dict[exp]['tre-wo-reg']
    point_color = [landmark_colors[i] for i in res_dict[exp]['ldmk-type']]
    pg.plot_blandaltman(tre, tre_wo_reg, scatter_kws={'s':25,'alpha':0.8,'c':point_color}, ax=axs[0, 2], xaxis="y")
    axs[0, 2].set_xlabel('Method: Joint - TREs')
    axs[0, 2].set_ylabel('')
    axs[0, 2].axhline(0, color='r', linestyle=':', lw=1)

    exp = 'mix'
    tre = res_dict[exp]['tre']
    tre_wo_reg = res_dict[exp]['tre-wo-reg']
    point_color = [landmark_colors[i] for i in res_dict[exp]['ldmk-type']]
    pg.plot_blandaltman(tre, tre_wo_reg, scatter_kws={'s':25,'alpha':0.8,'c':point_color}, ax=axs[0, 1], xaxis="y")
    axs[0, 1].set_xlabel('Method: Mixed - TREs')
    axs[0, 1].set_ylabel('')
    axs[0, 1].axhline(0, color='r', linestyle=':', lw=1)

    exp = 'usp'
    tre = res_dict[exp]['tre']
    tre_wo_reg = res_dict[exp]['tre-wo-reg']
    pg.plot_blandaltman(tre, tre_wo_reg, scatter_kws={'s':25,'alpha':0.8,'c':point_color}, ax=axs[0, 0], xaxis="y")
    axs[0, 0].set_xlabel('Method: Direct - TREs')
    axs[0, 0].set_ylabel('Difference in TREs(mm)')
    axs[0, 0].axhline(0, color='r', linestyle=':', lw=1)

    # mutual information on the second row 
    pg.plot_blandaltman(res_dict['pr5']['mi'], res_dict['pr5']['mi-wo-reg'], scatter_kws={'s':25,'alpha':0.8,'color':'tab:blue'}, ax=axs[1, 3], xaxis="y")
    axs[1, 3].set_xlabel('Method: Privileged - MIs')
    axs[1, 3].set_ylabel('')
    axs[1, 3].axhline(0, color='r', linestyle=':', lw=1)

    pg.plot_blandaltman(res_dict['mix']['mi'], res_dict['mix']['mi-wo-reg'], scatter_kws={'s':25,'alpha':0.8,'color':'tab:blue'}, ax=axs[1, 1], xaxis="y")
    axs[1, 1].set_xlabel('Method: Mixed - MIs')
    axs[1, 1].set_ylabel('')
    axs[1, 1].axhline(0, color='r', linestyle=':', lw=1)

    pg.plot_blandaltman(res_dict['jnt']['mi'], res_dict['jnt']['mi-wo-reg'], scatter_kws={'s':25,'alpha':0.8,'color':'tab:blue'}, ax=axs[1, 2], xaxis="y")
    axs[1, 2].set_xlabel('Method: Joint - MIs')
    axs[1, 2].set_ylabel('')
    axs[1, 2].axhline(0, color='r', linestyle=':', lw=1)

    pg.plot_blandaltman(res_dict['usp']['mi'], res_dict['usp']['mi-wo-reg'], scatter_kws={'s':25,'alpha':0.8,'color':'tab:blue'}, ax=axs[1, 0], xaxis="y")
    axs[1, 0].set_xlabel('Method: Direct - MIs')
    axs[1, 0].set_ylabel('Difference in MI')
    axs[1, 0].axhline(0, color='r', linestyle=':', lw=1)

    plt.subplots_adjust(hspace=0.32)
    plt.savefig(fname='./BAPlot.png', dpi=500, bbox_inches='tight')


def gen_fig1():
    figure, zoom = gen_figure(case_dict, figure_params)
    figure = figure.astype('uint8')
    zoom = zoom.astype('uint8')

    plt.imsave('./comparison.png', figure)
    plt.imsave('./comparison_zoom.png', zoom)

def gen_fig2():
    figure = gen_figure2(case_dict_2, figure2_params)
    figure = figure.astype('uint8')
    plt.imsave('./vis.png', figure)

def gen_fig3():
    figure = gen_figure3(case_dict_3, figure3_params)
    figure = figure.astype('uint8')
    plt.imsave('./landmarks.png', figure)
    
def calc_pvalues(a, b):
    return stats.ttest_rel(a, b)[1]

def pvalues_summary():
    res_dict = get_results()
    external_dict = get_external_results()

    print('------ p-value b0 vs pr5 ------')
    p = calc_pvalues(res_dict['pr5']['tre'], res_dict['pb0']['tre'])
    print(f'p-value={p:.3f}')

    print('------ p-value b0 vs mix ------')
    p = calc_pvalues(res_dict['mix']['tre'], res_dict['pb0']['tre'])
    print(f'p-value={p:.3f}')

    print('------ p-value b0 vs joint ------')
    p = calc_pvalues(res_dict['jnt']['tre'], res_dict['pb0']['tre'])
    print(f'p-value={p:.3f}')

    print('------ p-value b0 vs direct ------')
    p = calc_pvalues(res_dict['usp']['tre'], res_dict['pb0']['tre'])
    print(f'p-value={p:.3f}')

    print('------ p-value pr5 vs pr10 ------')
    p = calc_pvalues(res_dict['pr5']['tre'], res_dict['pr10']['tre'])
    print(f'p-value={p:.3f}')
    print('stat of pr10:', get_statistics(res_dict['pr10']['tre']))
    print('stat of pr5:', get_statistics(res_dict['pr5']['tre']))

    print('------ p-value pr5 vs unsp on MI ------')
    p = calc_pvalues(res_dict['pr5']['mi'], res_dict['usp']['mi'])
    print(f'p-value={p:.3f}')

    print('------ p-value usp vs mix on MI ------')
    p = calc_pvalues(res_dict['mix']['mi'], res_dict['usp']['mi'])
    print(f'p-value={p:.3f}')

    print('------ p-value usp vs joint on MI ------')
    p = calc_pvalues(res_dict['jnt']['mi'], res_dict['usp']['mi'])
    print(f'p-value={p:.3f}')

    print('------ p-value b0 vs pr10 ------')
    p = calc_pvalues(res_dict['pr10']['tre'], res_dict['pb0']['tre'])
    print(f'p-value={p:.3f}')

    print('------ p-value pr5 vs joint ------')
    p = calc_pvalues(res_dict['pr5']['tre'], res_dict['jnt']['tre'])
    print(f'p-value={p:.3f}')

    print('------ p-value pr5 before and after ------')
    p = calc_pvalues(res_dict['pr5']['tre'], res_dict['pr5']['tre-wo-reg'])
    print(f'tre, p-value={p:.3f}')
    p = calc_pvalues(res_dict['pr5']['mi'], res_dict['pr5']['mi-wo-reg'])
    print(f'mi, p-value={p:.3f}')

    print('------ p-value direct before and after ------')
    p = calc_pvalues(res_dict['usp']['tre'], res_dict['usp']['tre-wo-reg'])
    print(f'tre, p-value={p:.3f}')
    p = calc_pvalues(res_dict['usp']['mi'], res_dict['usp']['mi-wo-reg'])
    print(f'mi, p-value={p:.3f}')

    print('------ p-value nifty before and after ** ------')
    p = calc_pvalues(res_dict['nft']['ldmk'], res_dict['nft']['ldmk-wo-reg'])
    print(f'tre, p-value={p:.3f}')
    p = calc_pvalues(res_dict['nft']['mi'], res_dict['nft']['mi-wo-reg'])
    print(f'mi, p-value={p:.3f}')

    print('------ p-value usp before and after ** ------')
    p = calc_pvalues(res_dict['usp']['tre'], res_dict['usp']['tre-wo-reg'])
    print(f'tre, p-value={p:.3f}')

    print('------ p-value jnt before and after ** ------')
    p = calc_pvalues(res_dict['jnt']['tre'], res_dict['jnt']['tre-wo-reg'])
    print(f'tre, p-value={p:.3f}')

    print('------ p-value mix before and after ** ------')
    p = calc_pvalues(res_dict['mix']['tre'], res_dict['mix']['tre-wo-reg'])
    print(f'tre, p-value={p:.3f}')


    print('------ p-value external direct and niftyreg on cia ------')
    p = calc_pvalues(external_dict['cia']['nft']['ldmk'], external_dict['cia']['usp']['tre'])
    print(f'tre, p-value={p:.3f}')
    print('------ p-value external joint and niftyreg on cia ------')
    p = calc_pvalues(external_dict['cia']['nft']['ldmk'], external_dict['cia']['jnt']['tre'])
    print(f'tre, p-value={p:.3f}')
    print('------ p-value external direct and niftyreg on miami------')
    p = calc_pvalues(external_dict['miami']['nft']['ldmk'], external_dict['miami']['usp']['tre'])
    print(f'tre, p-value={p:.3f}')
    print('------ p-value external joint and niftyreg on miami ------')
    p = calc_pvalues(external_dict['miami']['nft']['ldmk'], external_dict['miami']['jnt']['tre'])
    print(f'tre, p-value={p:.3f}')

    print('------ p-value external pr5 and mixed on miami ------')
    p = calc_pvalues(external_dict['miami']['pr5']['tre'], external_dict['miami']['mix']['tre'])
    print(f'tre, p-value={p:.3f}')

    print('------ p-value external pr5 and mixed on cia ------')
    p = calc_pvalues(external_dict['cia']['pr5']['tre'], external_dict['cia']['mix']['tre'])
    print(f'tre, p-value={p:.3f}')
    
def add_contours(t2, label, color):
    _t2 = t2

    if len(label.shape)==3 and label.shape[-1]==3:
        _label = label[:, :, 0]

    contours, hierarchy = cv2.findContours(_label, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE, offset=(0, 0))
    tmp = _t2.copy()
    cv2.drawContours(tmp, contours, -1, color, 1)

    return tmp
    

if __name__ == "__main__":

    if args.update_res:
        update_results()
    
    gen_fig1()
    # update_table_1()
    # BA_plot()
    # update_table_2()
    # update_table_3()
    # gen_fig2()
    # gen_fig3()
    # pvalues_summary()

    
    

    
    
