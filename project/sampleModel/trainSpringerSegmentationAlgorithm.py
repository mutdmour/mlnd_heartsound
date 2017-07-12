# function [logistic_regression_B_matrix, pi_vector, total_obs_distribution] = trainSpringerSegmentationAlgorithm(PCGCellArray, annotationsArray, Fs, figures)
#
# Training the Springer HMM segmentation algorithm. Developed for use in
# the paper:
# D. Springer et al., "Logistic Regression-HSMM-based Heart Sound
# Segmentation," IEEE Trans. Biomed. Eng., In Press, 2015.
#
## Inputs:
# PCGCellArray: A 1XN cell array of the N audio signals. For evaluation
# purposes, these signals should be from a distinct training set of
# recordings, while the algorithm should be evaluated on a separate test
# set of recordings, which are recorded from a completely different set of
# patients (for example, if there are numerous recordings from each
# patient).
# annotationsArray: a Nx2 cell array: position (n,1) = the positions of the
# R-peaks and postion (n,2) = the positions of the end-T-waves
# (both in SAMPLES)
# Fs: The sampling frequency of the PCG signals
# figures (optional): boolean variable dictating the disaplay of figures.
#
## Outputs:
# logistic_regression_B_matrix:
# pi_vector:
# total_obs_distribution:
# As Springer et al's algorithm is a duration dependant HMM, there is no
# need to calculate the A_matrix, as the transition between states is only
# dependant on the state durations.
#
## Copyright (C) 2016  David Springer
# dave.springer@gmail.com
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY;without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.


#function [logistic_regression_B_matrix, pi_vector, total_obs_distribution] = 
import numpy as np
import getSpringerPCGFeatures as gSPCGF
import labelPCGStates as labelPCG

#options instead of Fs
def trainSpringerSegmentationAlgorithm(PCGCellArray, annotationsArray, springer_options, figures=False):

    numberOfStates = 4
    numPCGs = len(PCGCellArray)

    Fs = springer_options['audio_Fs']

    # A matrix of the values from each state in each of the PCG recordings:
    state_observation_values = np.empty((numPCGs,numberOfStates), dtype=object)

    for PCGi in range(0,len(PCGCellArray)):
        PCG_audio = PCGCellArray[PCGi]

        S1_locations = annotationsArray[PCGi][0]
        S2_locations = annotationsArray[PCGi][1]
        
        [PCG_Features, featuresFs] = gSPCGF.getSpringerPCGFeatures(PCG_audio, springer_options)
        
        PCG_states = labelPCG.labelPCGStates(PCG_Features[:][0],S1_locations, S2_locations, featuresFs) #XXX
        
        ## XXX
        ## Plotting assigned states:
        # if(figures)
        #     figure('Name','Assigned states to PCG')
            
        #     t1 = (1:length(PCG_audio))./Fs
        #     t2 = (1:length(PCG_Features))./featuresFs
            
        #     plot(t1, PCG_audio, 'k-')
        #     hold on
        #     plot(t2, PCG_Features, 'b-')
        #     plot(t2, PCG_states, 'r-')
            
        #     legend('Audio','Features','States')
        #     pause()
        
        # Group together all observations from the same state in the PCG recordings:
        for state_i in range(0,numberOfStates):
            x = np.multiply(PCG_Features[state_i],np.int16(PCG_states == state_i+1)) #xxx variable name
            x = PCG_Features[state_i][np.nonzero(x)]
            # print np.shape(x)#, x
            state_observation_values[PCGi][state_i] = x
        # print np.shape(state_observation_values[0][3]), state_observation_values

    # Save the state observation values to the main workspace of Matlab for
    # later investigation if needed:
    # assignin('base', 'state_observation_values', state_observation_values) #XXX

    # ## Train the B and pi matrices after all the PCG recordings have been labelled:
    logistic_regression_B_matrix, pi_vector, total_obs_distribution = trainBandPiMatricesSpringer(state_observation_values) #XXX

    # return [logistic_regression_B_matrix, pi_vector, total_obs_distribution]

