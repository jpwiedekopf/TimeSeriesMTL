#-------------------------------------------------------------------------------------------------------
# Script to process the OPPORTUNITY data by removing the NaN values according to the following rules:
#    - All sensors with only NaN values are removed
#    - All NaN values at the end of the file are removed
#    - All remaning residual NaN values are replaced by the previous non-NaN value for each sensor channel
#
# Inputs: 
#   - [string] dataFolder: path to the folder containing the .dat data files. Note: must be the full data files
#   - [string] resultFolder: path to the folder to save the result files
#
# This script writes new .txt data files containing continuous data records from the files in the 
# input data folder purged of NaN values. The format of the output files is nb_timetamps x nb_clean_sensors
#-------------------------------------------------------------------------------------------------------

import numpy as np
import os
import sys

#-------------------------------------------------------------------------------------------------------
# Hyper-parameters -> TODO: change and check before use
#-------------------------------------------------------------------------------------------------------

# Path to the folder containing the OPPORTUNITY data
dataFolder = '/data/opportunity/raw'

# Path to the folder to save the new clean data files
resultFolder = '/data/opportunity/clean-mtl'

allSensors=True

#-------------------------------------------------------------------------------------------------------
# Function to clean the OPPORTUNITY data:
#
# Inputs: 
#   - [string] dataFolder: path to the folder containing the .dat data files
#   - [string] resultFolder: path to the folder to save the result files
#   - [boolean] allSensors: if true, all sensors are used. Otherwise, only those on the right lower arm
#
# This script writes new .txt data files containing continuous data records from the files in the 
# input data folder purged of NaN values. The format of the output files is nb_timetamps x nb_clean_sensors
#-------------------------------------------------------------------------------------------------------

def cleanOpportunityData(dataFolder, resultFolder, allSensors=True):

    if not os.path.exists(resultFolder):
        os.makedirs(resultFolder)
    
    print('--------------------------------------------------------------------')
    print('Processing the data files in '+dataFolder)


    # Sensors indices of the OPPORTUNITY data
    # Note: fully corrupted sensors are found by observing the data only
    # Note2: the label information is contained in the last column
    if allSensors:
        sensorId = list(range(1,13)) + list(range(16,34)) + list(range(37,46)) + list(range(50,59)) + list(range(63,72)) + list(range(76,85)) + list(range(89,98)) + list(range(102,134)) + list(range(243,250))
    else:
        sensorId = [63,64,65,66,67,68] + list(range(243,250))
    nbSensors = len(sensorId)

    # For all .dat files in the target folder
    for file in os.listdir(dataFolder):

        if file.endswith('.dat'):

            print('Processing file ' + file + ' ...')

            # File name
            fileName = file.replace('.dat','')

            # Open and read the file
            fhr = open(dataFolder+'/'+file,'r')
            contents = fhr.readlines() # Contents of the file as string
            fhr.close()

            # Convert to a matrix of float and extract the labels
            nbTimestamps = len(contents)
            data = np.empty((nbTimestamps,nbSensors), dtype=float)
            
            # Keep only the relevant content
            # Note: also removes the NaN values at the end of the data file
            # Removing criteria (arbitrary): if above 85% sensors have NaN values at a particular timestamp, then the following values are considered to be NaN
            stoppingIdx = 0 # Index after which all values are considered to be NaN
            for idx in range(nbTimestamps):
                dataLineTmp = [float(e) for e in contents[idx].split()]
                data[idx] = [e for i,e in enumerate(dataLineTmp) if i in sensorId]
                nbNaNDataLine = np.isnan(dataLineTmp).sum()
                if nbNaNDataLine >= 0.85*nbSensors:
                    stoppingIdx = idx 

            data = data[:stoppingIdx]
            newNbTimestamps = len(data)
            
            # Replace all remaining NaN values with the previous non-NaN one, for each sensor channel
            for sensorIdx in range(nbSensors):

                # Check if the sensor column contains any NaN
                detectedNaN = np.isnan(data[:,sensorIdx]).any()

                # If at least a NaN value is detected, replace any of them by the previous non-NaN one
                if detectedNaN:
                    sensorColumn = data[:,sensorIdx]

                    # Find the first non-NaN value of the column
                    previousNonNaN = sensorColumn[0]
                    firstNonNanElementIdx = 1
                    while np.isnan(previousNonNaN) and firstNonNanElementIdx < newNbTimestamps:
                        previousNonNaN = sensorColumn[firstNonNanElementIdx]
                        firstNonNanElementIdx += 1

                    if np.isnan(previousNonNaN):
                        print('ERROR: all sensor readings for one channel are NaN!')
                        sys.exit()
                    
                    # Replace NaN values
                    for timeIdx in range(newNbTimestamps):
                        if np.isnan(sensorColumn[timeIdx]):
                            data[timeIdx,sensorIdx] = previousNonNaN
                        else:
                            previousNonNaN = sensorColumn[timeIdx]

            # Save the data and labels in a new file
            if not allSensors:
                outputFileName = fileName + '_RLA.txt'
            else:
                outputFileName = fileName + '.txt'

            #pdb.set_trace()
            outname = os.path.join(resultFolder, outputFileName)
            np.savetxt(outname, data, delimiter=' ', fmt='%d')

    print('New files created in '+resultFolder)
    print('--------------------------------------------------------------------')




#-------------------------------------------------------------------------------------------------------
# Main
#-------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    cleanOpportunityData(dataFolder,resultFolder,allSensors)
