import pickle as pkl
import os
from glob import glob
import numpy as np

def read_pkl(pkl_file):
    with open(pkl_file, 'rb') as f:
        data = pkl.load(f)
    return data

def mean(x):
    return np.mean(x)

def std(x):
    return np.std(x)

def get_stats(data, method):
    overall_dice = data['dice']
    bladder_dice_aft = overall_dice[0::2]
    rectum_dice_atf = overall_dice[1::2]

    results = {
    'case': [],
    'organ': [],
    'dice':[],
    'method':[]
    }

    for organ_id, dice in enumerate([bladder_dice_aft, rectum_dice_atf]):
        if organ_id == 0:
            organ = 'bladder'
        else:
            organ = 'rectum'
        for case_id, d in enumerate(dice):
            results['case'].append(case_id)
            results['organ'].append(organ)
            results['dice'].append(float(d))
            results['method'].append(method)
    return results
            
        
    

def get_numbers_v1(data):
    overall_dice = data['dice']
    wo_dice = data['dice-wo-reg']

    bladder_dice_aft = overall_dice[0::2]
    rectum_dice_atf = overall_dice[1::2]

    bladder_dice_bef = wo_dice[0::2]
    rectum_dice_bef = wo_dice[1::2]

    return f'{mean(overall_dice):.3f} +- {std(overall_dice):.3f},\
    {mean(wo_dice):.3f} +- {std(wo_dice):.3f},\
    {mean(bladder_dice_aft):.3f} +- {std(bladder_dice_aft):.3f},\
    {mean(bladder_dice_bef):.3f} +- {std(bladder_dice_bef):.3f},\
    {mean(rectum_dice_atf):.3f} +- {std(rectum_dice_atf):.3f},\
    {mean(rectum_dice_bef):.3f} +- {std(rectum_dice_bef):.3f}'

def get_numbers_v2(data):
    overall_dice = data['dice']

    bladder_dice_aft = overall_dice[0::2]
    rectum_dice_atf = overall_dice[1::2]

    return f'{mean(overall_dice):.3f} +- {std(overall_dice):.3f},\
    {mean(bladder_dice_aft):.3f} +- {std(bladder_dice_aft):.3f},\
    {mean(rectum_dice_atf):.3f} +- {std(rectum_dice_atf):.3f}'  

exp_dir_list = [
    './logs/ConditionalSeg/hpc.01-*',
    './logs/CBCTUnetSeg/hpc.04-*',
    './logs/CBCTUnetSeg/hpc.05-*',
    './logs/CBCTUnetSeg/hpc.06-*',
    './logs/CBCTUnetSeg/hpc.07-*',
    './logs/WeakSup/hpc.02-*',
    './logs/WeakSup/hpc.03-*',
    './logs/ConditionalSeg/hpc.09-*',
    './logs/ConditionalSeg/hpc.08-*',
]

entire_results = {
    'case': [],
    'organ': [],
    'dice':[],
    'method':[]
}


for exp_d in exp_dir_list:
    exp_resfiles = glob(os.path.join(exp_d, 'results.pkl'))
    exp_resfiles.sort()

    [print(i) for i in exp_resfiles]
    data_collection = list(map(read_pkl, exp_resfiles))

    data_all = {}
    for k in data_collection[0].keys():
        data_all[k] = []
    if 'dice-wo-reg' in data_collection[0].keys():
        get_numbers = get_numbers_v1
    else:
        get_numbers = get_numbers_v2

    for data in data_collection:
        print(get_numbers(data))
        for k, v in data.items():
            data_all[k].extend(v)
    
    print(get_numbers(data_all))

    results = get_stats(data_all, method=os.path.basename(exp_d)[4:6])
    for k, v in results.items():
        entire_results[k].extend(v)
    

    
import seaborn as sns
import matplotlib
import pandas as pd

sns.set_theme(style="whitegrid")
plot_data = pd.DataFrame(entire_results)
print(plot_data)
ax = sns.boxplot(x="method", y="dice", hue="organ", data=plot_data, palette="Set3", fliersize=4)

fig = ax.get_figure()
fig.savefig("./preprocessing/condi-seg/compare.png", dpi=500)
plot_data.to_csv("./preprocessing/condi-seg/plot_data.csv")