import re
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

def read_file(path):
    with open(path, 'r') as file:
        return file.read()
    
def plot_dice_values(text, path, legend, length=None, memory=False):
    # Extract Dice values using regex
   
    dice_values_dict = {}
    dice_std_dict = {}
    for i in range(0, len(text)):
        
        dice_values = re.findall(r'Dice:,\s(\d+\.\d+),\s(\d+\.\d+)', text[i])        
        dice_std = [float(value[1]) for value in dice_values]
        if i == 4:
            dice_values = [float(value[0])-0 for value in dice_values]
        else:
            dice_values = [float(value[0]) for value in dice_values]
        dice_values_dict[i] = dice_values[0:length]
        dice_std_dict[i] = dice_std[0:length]
        


        
   
    # Plotting the Dice values
    plt.figure(figsize=(10, 8))
    markers = ['o', 's', '^', 'D', 'v', 'p', '*']
    # dice_df = pd.DataFrame.from_dict(dice_values_dict, orient='index')
    # #  # Rename the columns if needed
    # print(dice_df)
    # sns.lineplot(data=dice_df)
    for i in range(0, len(dice_values_dict)):
        x=np.arange(0, len(dice_values_dict[i]))
        y=np.array(dice_values_dict[i])
        std = np.array(dice_std_dict[i])      
    #     sns.lineplot(data=dice_df, x=dice_df.index, y=dice_df[i], marker=np.random.choice(markers), linestyle='-', markersize=2)
        plt.fill_between(x, y-std, y+std, alpha=0.1)
        plt.plot(x,y, marker=np.random.choice(markers), linestyle='-', linewidth=2, markersize=4, label=legend[i], )


    plt.legend( loc='lower right', fontsize=15)
    plt.title('Dice Values Across Epochs', fontsize=20)
    plt.xlabel('Epoch', fontsize=15)
    plt.ylabel('Dice Value', fontsize=15)
    plt.grid(True)
    plt.savefig(path)


if __name__ == '__main__':
    import argparse
    import os
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, help='Path to the log file')
    parser.add_argument('--path2', type=str, required=False, help='Path to the log file')
    parser.add_argument('--path3', type=str, required=False, help='Path to save the plot')
    parser.add_argument('--path4', type=str, required=False, help='Path to save the plot')
    parser.add_argument('--path5', type=str, required=False, help='Path to save the plot')
    parser.add_argument('--path6', type=str, required=False, help='Path to save the plot')
    parser.add_argument('--path7', type=str, required=False, help='Path to save the plot')
    parser.add_argument('--path8', type=str, required=False, help='Path to save the plot')
    args = parser.parse_args()

    path = os.path.join('./logs/Icn', args.path, 'train.log')
    text = read_file(path)
    if args.path2:
        path2 = os.path.join('./logs/Icn', args.path2, 'train.log')
        text2 = read_file(path2)
        text_list = [text, text2]
        legend = ['Voxelmorph', 'U-GRID']
        if args.path6:
            path6 = os.path.join('./logs/Icn', args.path6, 'train.log')
            text6 = read_file(path6)
            text_list.append(text6)
            legend.append('U-GRID-small')
        if args.path7:
            path7 = os.path.join('./logs/Icn', args.path7, 'train.log')
            text7 = read_file(path7)
            text_list.append(text7)
            legend.append('U-GRID-large')
        if args.path3:
            path3 = os.path.join('./logs/Icn', args.path3, 'train.log')
            text3 = read_file(path3)
            text_list.append(text3)
            legend.append('GRID')
        if args.path4:
            path4 = os.path.join('./logs/Icn', args.path4, 'train.log')
            text4 = read_file(path4)
            text_list.append(text4)
            legend.append('Transmorph')
        if args.path5:
            path5 = os.path.join('./logs/Icn', args.path5, 'train.log')
            text5 = read_file(path5)
            text_list.append(text5)
            legend.append('Keymorph')
        if args.path8:
            path8 = os.path.join('./logs/Icn', args.path8, 'train.log')
            text8 = read_file(path8)
            text_list.append(text8)
            legend.append('KeyMporph-small')

    
    plot_dice_values(text_list, path.replace('train.log', 'train_new.png'), legend, length=100, memory=True)
    plot_dice_values(text_list, path.replace('train.log', 'train_detail.png'), legend, length=20)