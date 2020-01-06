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
from matplotlib import pyplot as plt
import json
import pandas as pd
from scipy import stats

#-------------------------------------------------------------------------------------------------------
# Hyper-parameters -> TODO: change and check before use
#-------------------------------------------------------------------------------------------------------

# Time window parameters

# Number of sensors ranked by variance to use
nbSensorsByVar = 107

#mtl = "-mtl" #set to empty string if stl data is analyzed
mtl = ""

# dataFolder: path to the folder containing the clean OPPORTUNITY data
# resultFolder: path to the folder where to save the data frames
if nbSensorsByVar == 107:
    dataFolder = f"/data/opportunity/clean{mtl}/"
    resultFolder = f"/data/opportunity/window_labels{mtl}"
else:
    dataFolder = f"/data/opportunity/clean{mtl}/" +  str(nbSensorsByVar)+'_highest_var_sensors/'
    resultFolder = f"/data/opportunity/clean{mtl}/" + str(nbSensorsByVar)+'_highest_var_sensors/'

plot = True
generate_stat_json = False

allDataOutputFile = dataFolder + "allData.npy"

#-------------------------------------------------------------------------------------------------------
# Function extractTimeWindowsAndLabels: load the OPPORTUNITY data, and extract and save the time windows 
# and labels as .npy files
#-------------------------------------------------------------------------------------------------------

def generateStatistics(pathToDataFolder=dataFolder,resultFolder=resultFolder): 

    if (os.path.isfile(allDataOutputFile)):
        allData = np.load(allDataOutputFile)
        print(f"Loaded data from {allDataOutputFile}")
    else:
        print(f"All data file was not found at {allDataOutputFile}") 
        # List files in the data folder
        print('Input data file folder: %s' % (pathToDataFolder))
        dataFileList = os.listdir(pathToDataFolder)

        allData = None

        # Extraction of the time windows + labels for each .txt data file
        for fileID, fileName in enumerate(dataFileList):

            if 'ADL' in fileName or 'Drill' in fileName:

                print('Processing file %s (%d/%d)...' % (fileName, fileID+1,
                                                         len(dataFileList)))

                # Get file contents as string
                fh = open(pathToDataFolder+'/'+fileName,'r')
                contents = fh.readlines()
                fh.close()

                # Convert to a matrix of floats
                # Note: the gesture label is in the last column. 
                nbTimestamps = len(contents)
                nbSensors = len([np.float32(e) for e in contents[0].split()])-1
                data = np.zeros((nbTimestamps,nbSensors))

                for idx in range(nbTimestamps):
                    dataLineTmp = [np.float32(e) for e in contents[idx].split()]
                    dataLine = dataLineTmp[:-1] 
                    data[idx] = dataLine

                print("Shape: ", data.shape)

                if allData is None:
                    allData = data
                else:
                    allData = np.append(allData, data, 0)
                print("All Data Shape: ", allData.shape)

            #if fileID > 1:
            #    break
    
        with open(allDataOutputFile, "wb") as f:
            np.save(f, allData)
            print(f"Saved all data to {allDataOutputFile}")
        
    allDataVectors = [allData[:, i] for i in range(allData.shape[1])]
    
    
    
    if plot:
        #labels = range(len(allDataVectors))
        #fig, axes = plt.subplots(nrows=4, ncols=1, sharey=True)
        #for pltrow in range(4):
        #    beginpart = (pltrow * 27)
        #    endpart = (pltrow + 1) * 27
        #    print(f"{pltrow}: {beginpart} -> {endpart}")
        #    dataSlice = allDataVectors[beginpart:endpart]
        #    axes[pltrow].violinplot(dataSlice)
        #    axes[pltrow].set_title(f"Sensors {beginpart} to {endpart}")
        #plt.show()
        for i, v in enumerate(allDataVectors):
            fig = plt.figure(figsize=(15, 7))
            plt.boxplot(v, notch=True)
            fn = f"violin-{i+1}.png"
            fig.savefig(fn, dpi=320)
            print(fn)
            plt.close()

        
    all_stat = { "channels" : {}}
    if generate_stat_json:
        iqrrange = [90, 10]
        percentiles = [.25, .5, .75, .90]
        for colidx, col in enumerate(allDataVectors):
            pv = pd.DataFrame(col)
            stat = pv.describe(percentiles = percentiles).to_dict()[0]
            stat["iqr"] = stats.iqr(col, rng=iqrrange)

            mapmin = max(stat["min"], stat["50%"] - 1.5 * stat["iqr"]) 
            mapmax = min(stat["max"], stat["50%"] + 1.5 * stat["iqr"]) 

            stat["mapmin"] = mapmin
            stat["mapmax"] = mapmax

            all_stat['channels'][colidx] = stat

        all_stat["iqrrange"] = iqrrange
        all_stat["percentiles"] = percentiles
        with open("statistics.json", "w") as f:
          json.dump(all_stat, f, indent=2)

#-------------------------------------------------------------------------------------------------------
# Main
#-------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
   generateStatistics()
