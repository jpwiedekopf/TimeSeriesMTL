#-------------------------------------------------------------------------------------------------------
# Script to extract the time windows of data + labels from the clean OPPORTUNITY data files
#
# Inputs: 
#   - [string] dataFolder: path to the folder containing the cleaned .txt data files
#   - [string] resultFolder: path to the folder to save the results
#   - [int] timeWindow: size of the time window
#   - [int] slidingStride: stride of the time window
#
# This script saves the time windows in a .npy file containing a 3D array of size 
# nb_windows x time_window_size x nb_sensors
# The labels are also saved in a .npy file containing a vector with the corresponding labels
# Note: one data and label .npy files are created for each .txt file in the data folder
#-------------------------------------------------------------------------------------------------------


import numpy as np
import os
import sys
from math import floor


def labelTableForRange(min_inclusive: int, max_inclusive: int):
    return {f'{k}': v for (k, v) in [(0, 0)] + list(zip(range(min_inclusive, max_inclusive + 1), range(1, max_inclusive - min_inclusive + 2)))}
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

#-------------------------------------------------------------------------------------------------------
# Hyper-parameters -> TODO: change and check before use
#-------------------------------------------------------------------------------------------------------

# Time window parameters
timeWindow = 64
slidingStride = 3

# Number of sensors ranked by variance to use
nbSensorsByVar = 107

# dataFolder: path to the folder containing the clean OPPORTUNITY data
# resultFolder: path to the folder where to save the data frames
if nbSensorsByVar == 107:
    dataFolder = '/data/opportunity/clean-mtl/'
    resultFolder = '/data/opportunity/window_labels-mtl'
else:
    dataFolder = '/hdd/datasets/OPPORTUNITY/clean/'+str(nbSensorsByVar)+'_highest_var_sensors/'
    resultFolder = '/hdd/datasets/OPPORTUNITY/windows_labels/'+str(nbSensorsByVar)+'_highest_var_sensors/'



#-------------------------------------------------------------------------------------------------------
# Function extractTimeWindowsAndLabels: load the OPPORTUNITY data, and extract and save the time windows 
# and labels as .npy files
#-------------------------------------------------------------------------------------------------------

def extractTimeWindowsAndLabels(pathToDataFolder=dataFolder,resultFolder=resultFolder,timeWindow=timeWindow,slidingStride=slidingStride): 

    if not os.path.exists(resultFolder):
        os.makedirs(resultFolder)

    # List files in the data folder
    print('Input data file folder: %s' % (pathToDataFolder))
    dataFileList = os.listdir(pathToDataFolder)

    # Extraction of the time windows + labels for each .txt data file
    for fileid, fileName in enumerate(sorted(dataFileList)):

        if 'ADL' in fileName or 'Drill' in fileName:

            print('Processing file {} ({}/{})...'.format(fileName, fileid+1, len(dataFileList)))

            # Get file contents as string
            fh = open(pathToDataFolder+'/'+fileName,'r')
            contents = fh.readlines()
            fh.close()

            # Convert to a matrix of floats Note: the gesture label is in the
            # last column. 
            # for mtl we need multiple labels, so all label columns are extracted
            numLabels = 7
            nbTimestamps = len(contents)
            nbSensors = len([np.float32(e) for e in contents[0].split()])-numLabels
            data = np.zeros((nbTimestamps,nbSensors))
            labels = -1*np.ones((nbTimestamps, numLabels),dtype=int) 

            for idx in range(nbTimestamps):
                dataLineTmp = [np.float32(e) for e in contents[idx].split()]
                dataLine = dataLineTmp[:-numLabels] 
                data[idx] = dataLine
                labels[idx,:] = np.asarray(dataLineTmp[-numLabels:],int)

            for idx, m in enumerate(label_map):
                col = labels[:, idx]
                for k, r in m.items():
                    col[col == int(k)] = r
                labels[:, idx] = col
            
            # Determination of the total number of time windows, and pre-allocation of result arrays
            nbTimeWindows = int(floor((nbTimestamps-timeWindow)/slidingStride))+1
            timeWindowArray = np.empty((nbTimeWindows,timeWindow,nbSensors),dtype=np.float32)
            labelsVector = -1*np.ones((nbTimeWindows,numLabels),dtype=int)

            # Iteration on the data file to build the examples of size timeWindow x nbOfSensors
            idx = 0
            timeWindowCounter = 0
            while idx < nbTimestamps - timeWindow + 1:
                windowData = data[idx:idx+timeWindow]
                windowLabels = labels[idx:idx+timeWindow,:]
                # Determine the majoritary label among those of the timeWindow examples considered | TODO: uncomment for the majority label approach
                tslabels = -1 * np.ones(numLabels,dtype=int)
                for labelcolid in range(0,numLabels):
                    labelcol = windowLabels[:,labelcolid]
                    (values,counts) = np.unique(labelcol,return_counts=True)
                    majoritaryLabel = values[np.argmax(counts)]
                    tslabels[labelcolid] = majoritaryLabel
                #print(tslabels)
                #majoritaryLabel = windowLabels[-1] # TODO: uncomment for the last label solution
                # Store the data and labels
                timeWindowArray[timeWindowCounter] = windowData
                #labelsVector[timeWindowCounter] = majoritaryLabel
                labelsVector[timeWindowCounter,:] = tslabels
                # Iterate
                timeWindowCounter += 1
                idx += slidingStride 

            k = 10
            for sensor_channel_idx in range(nbSensors):
                y = data[:, sensor_channel_idx]
                mu = np.mean(y)
                sigma = np.std(y)
                mapmin = round(mu - k * sigma, 1)
                mapmax = round(mu + k * sigma, 1)
                prevmin = np.min(y)
                prevmax = np.max(y)
                print(f"  Mapping channel {sensor_channel_idx} to range [{mapmin}, {mapmax}] (from [{prevmin}, {prevmax}]).")
                clip = np.where(y < mapmin, mapmin, y)
                clip = np.where(clip > mapmax, mapmax, clip)
                data[:, sensor_channel_idx] = clip
            

            # Project labels in the range (0,L-1) with L number of different labels
            #unique = set(windowLabels)
            #cleanLabels = [list(unique).index(l) for l in windowLabels]

            # Save data and labels in the result folder as .npy files 
            resultname = fileName.replace('.txt','')
            dataname = os.path.join(resultFolder, f"{resultname}_data.npy")
            labelname = os.path.join(resultFolder, f"{resultname}_labels.npy")
            np.save(dataname,timeWindowArray)
            np.save(labelname,labelsVector) 
            #np.savetxt(resultFolder+'/'+resultname+'_labels.csv', labelsVector,
            #           delimiter=',', fmt='%d')
            print("Data shape: ", timeWindowArray.shape)
            print("Lbls shape: ", labelsVector.shape)
    print('Results saved in the folder %s' % (resultFolder))


#-------------------------------------------------------------------------------------------------------
# Main
#-------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    extractTimeWindowsAndLabels()
