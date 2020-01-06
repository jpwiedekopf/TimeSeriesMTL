import numpy as np
import json
from os import path
import os
import sys
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib.gridspec as gridspec
import json

def labelTableForRange(min_inclusive: int, max_inclusive: int):
    return {f'{k}': v for (k, v) in [(0, 0)] + list(zip(range(min_inclusive, max_inclusive + 1), range(1, max_inclusive - min_inclusive + 2)))}

# Gives the association between the original labels and the labels used for the training of the network
label_map = [
    labelTableForRange(1, 5),  # Locomotion 1-5
    labelTableForRange(101, 105),  # HL Activity
    labelTableForRange(201, 213),  # LL Left
    labelTableForRange(301, 323),  # LL Left Object
    labelTableForRange(401, 413),  # LL Right
    labelTableForRange(501, 523),  # LL Right Object
    {'0': 0, '406516': 1, '406517': 2, '404516': 3, '404517': 4, '406520': 5, '404520': 6,
     '406505': 7, '404505': 8, '406519': 9, '404519': 10, '406511': 11, '404511': 12, '406508': 13,
     '404508': 14, '408512': 15, '407521': 16, '405506': 17}  # ML Both Arms
]

label_names = ['LL_Locomotion', 'HL_Activity', 'LL_Left', 'LL_LeftObject', 'LL_Right', 'LL_RightObject', 'ML_Both']

nbSensorsByVar = 107
if nbSensorsByVar == 107:
    dataFolder = f"/data/opportunity/clean-mtl"
else:
    dataFolder = f"/data/opportunity/clean-mtl/" + \
        str(nbSensorsByVar)+'_highest_var_sensors/'

plot_channels = [
    list(range(0, 3))# +
    #list(range(16, 19))
]
plot_channel_labels  = [
    'ACC RKN^ accXLOC', 'ACC RKN^ accYLOC', 'ACC RKN^ accZLOC', 
    'ACC HIP accXLOC', 'ACC HIP accYLOC', 'ACC HIP accZLOC', 
    'ACC LUA^ accXLOC', 'ACC LUA^ accYLOC', 'ACC LUA^ accZLOC', 
    'ACC RUA_ accXLOC', 'ACC RUA_ accYLOC', 'ACC RUA_ accZLOC', 
    'ACC BACK accXLOC', 'ACC BACK accYLOC', 'ACC BACK accZLOC', 
    'ACC RKN_ accXLOC', 'ACC RKN_ accYLOC', 'ACC RKN_ accZLOC', 
    'ACC RWR accXLOC', 'ACC RWR accYLOC', 'ACC RWR accZLOC', 
    'ACC RUA^ accXLOC', 'ACC RUA^ accYLOC', 'ACC RUA^ accZLOC', 
    'ACC LUA_ accXLOC', 'ACC LUA_ accYLOC', 'ACC LUA_ accZLOC', 
    'ACC LWR accXLOC', 'ACC LWR accYLOC', 'ACC LWR accZLOC', 
    'IMU BACK accXLOC', 'IMU BACK accYLOC', 'IMU BACK accZLOC', 
    'IMU BACK gyroXLOC', 'IMU BACK gyroYLOC', 'IMU BACK gyroZLOC', 
    'IMU BACK magneticXLOC', 'IMU BACK magneticYLOC', 'IMU BACK magneticZLOC', 
    'IMU RUA accXLOC', 'IMU RUA accYLOC', 'IMU RUA accZLOC', 
    'IMU RUA gyroXLOC', 'IMU RUA gyroYLOC', 'IMU RUA gyroZLOC', 
    'IMU RUA magneticXLOC', 'IMU RUA magneticYLOC', 'IMU RUA magneticZLOC', 
    'IMU RLA accXLOC', 'IMU RLA accYLOC', 'IMU RLA accZLOC', 
    'IMU RLA gyroXLOC', 'IMU RLA gyroYLOC', 'IMU RLA gyroZLOC', 
    'IMU RLA magneticXLOC', 'IMU RLA magneticYLOC', 'IMU RLA magneticZLOC', 
    'IMU LUA accXLOC', 'IMU LUA accYLOC', 'IMU LUA accZLOC', 
    'IMU LUA gyroXLOC', 'IMU LUA gyroYLOC', 'IMU LUA gyroZLOC', 
    'IMU LUA magneticXLOC', 'IMU LUA magneticYLOC', 'IMU LUA magneticZLOC', 
    'IMU LLA accXLOC', 'IMU LLA accYLOC', 'IMU LLA accZLOC', 
    'IMU LLA gyroXLOC', 'IMU LLA gyroYLOC', 'IMU LLA gyroZLOC', 
    'IMU LLA magneticXLOC', 'IMU LLA magneticYLOC', 'IMU LLA magneticZLOC', 
    'IMU L-SHOE EuXLOC', 'IMU L-SHOE EuYLOC', 'IMU L-SHOE EuZLOC', 
    'IMU L-SHOE Nav_AxLOC', 'IMU L-SHOE Nav_AyLOC', 'IMU L-SHOE Nav_AzLOC', 
    'IMU L-SHOE Body_AxLOC', 'IMU L-SHOE Body_AyLOC', 'IMU L-SHOE Body_AzLOC', 
    'IMU L-SHOE AngVelBodyFrameXLOC', 'IMU L-SHOE AngVelBodyFrameYLOC', 'IMU L-SHOE AngVelBodyFrameZLOC', 
    'IMU L-SHOE AngVelNavFrameXLOC', 'IMU L-SHOE AngVelNavFrameYLOC', 'IMU L-SHOE AngVelNavFrameZLOC', 
    'IMU L-SHOE CompassLOC', 'IMU R-SHOE EuXLOC', 'IMU R-SHOE EuYLOC', 
    'IMU R-SHOE EuZLOC', 'IMU R-SHOE Nav_AxLOC', 'IMU R-SHOE Nav_AyLOC', 
    'IMU R-SHOE Nav_AzLOC', 'IMU R-SHOE Body_AxLOC', 'IMU R-SHOE Body_AyLOC', 
    'IMU R-SHOE Body_AzLOC', 'IMU R-SHOE AngVelBodyFrameXLOC', 'IMU R-SHOE AngVelBodyFrameYLOC', 
    'IMU R-SHOE AngVelBodyFrameZLOC', 'IMU R-SHOE AngVelNavFrameXLOC', 'IMU R-SHOE AngVelNavFrameYLOC', 
    'IMU R-SHOE AngVelNavFrameZLOC', 'IMU R-SHOE CompassLOC'
]

dataFileList = os.listdir(dataFolder)

for fileID, inputFile in enumerate(sorted(dataFileList)):
    if not ('ADL' in inputFile or 'Drill' in inputFile):
        continue

    print('Processing file %s (%d/%d)...' %
          (inputFile, fileID+1, len(dataFileList)))

    # Get file contents as string
    fh = open(dataFolder+'/'+inputFile, 'r')
    contents = fh.readlines()
    fh.close()

    nbLabelCols = 7
    nbTimestamps = len(contents)
    nbSensors = len([np.float32(e) for e in contents[0].split()]) - nbLabelCols
    data = np.zeros((nbTimestamps, nbSensors))

    labels = np.zeros((nbTimestamps, nbLabelCols), dtype=int)

    for idx in range(nbTimestamps):
        dataLineTmp = [np.float32(e) for e in contents[idx].strip().split()]
        dataLine = dataLineTmp[:-nbLabelCols] #remove labels
        labels[idx] = np.asarray(dataLineTmp[-nbLabelCols:], dtype=int)
        data[idx] = dataLine

    for idx, m in enumerate(label_map):
        col = labels[:, idx]
        for k, r in m.items():
            col[col == int(k)] = r
        labels[:, idx] = col

    #np.savetxt(inputFile + '_labels.csv', labels, delimiter=',', fmt='%d')

    print("Shape: ", data.shape)

    for group in plot_channels:
        plot_data = [data[:, i] for i in group]
        channel_names = [plot_channel_labels[i] for i in group]
        x = list(range(len(plot_data[0])))
        file_tag = "+".join([str(i) for i in group])
        fig = plt.figure(figsize=(20, 2 * (len(group) + 2)), constrained_layout=True)
        spec = gridspec.GridSpec(ncols=1, nrows=(len(group) + 2), figure=fig)        
        #fig = plt.figure(figsize=(20, 2 * (len(group) + 2)), constrained_layout=True)
        #spec = gridspec.GridSpec(ncols=2, nrows=(len(group) + 2), figure=fig)
        for i, y in enumerate(plot_data):
            ax = fig.add_subplot(spec[i, 0])
            # highlight values that are outliers by drawing the k-sigma tunnel across the mean
            k = 10
            mu = np.mean(y)
            sigma = np.std(y)
            mapmin = mu - k * sigma
            mapmax = mu + k * sigma
            ax.axhline(mapmin, linewidth=1, color='r')
            ax.axhline(mapmax, linewidth=1, color='r')
            ax.axhline(mu, linewidth=1, color='k')
            ax.plot(x, y)
            ax.grid(True)
            ax.set_title(channel_names[i])

        #label_data = label_data[label_data.y != 0]
        for sub, i in enumerate([0, 6]):
            label_channel = labels[:, i]
            max_label_val = len(label_map[i])
            label_name = label_names[i]
            label_data = pd.DataFrame(dict(x=x, y=label_channel, labels=label_channel))
            label_ax = fig.add_subplot(spec[-(sub + 1), 0])
            sc = sns.scatterplot(data=label_data, x="x", y="y", hue="labels", 
                ax=label_ax, size=0.5, legend=False, 
                palette=sns.color_palette(palette="dark", n_colors=len(np.unique(label_channel)))
                )
            plt.yticks(list(range(max_label_val)))
            label_ax.set_title(f"Labels ({label_name})")
        #fig.tight_layout()
        filename = path.join(os.getcwd(), "sensor-plots-mtl", f"{os.path.splitext(inputFile)[0]}_{file_tag}.png")
        plt.savefig(filename)
        print(filename)
        plt.close()