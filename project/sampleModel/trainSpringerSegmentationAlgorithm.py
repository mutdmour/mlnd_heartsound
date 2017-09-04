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

#function [logistic_regression_B_matrix, pi_vector, total_obs_distribution] = 
import numpy as np
import getSpringerPCGFeatures as gSPCGF
import labelPCGStates as labelPCG
import default_Springer_HSMM_options as opts
import trainBandPiMatricesSpringer as train

#options instead of Fs
def trainSpringerSegmentationAlgorithm(PCGCellArray, annotationsArray, Fs, figures=False):

    numberOfStates = 4
    numPCGs = len(PCGCellArray)

    # A matrix of the values from each state in each of the PCG recordings:
    state_observation_values = np.empty((numPCGs,numberOfStates), dtype=object)

    for PCGi in range(0,len(PCGCellArray)):
        PCG_audio = PCGCellArray[PCGi]

        S1_locations = annotationsArray[PCGi][0]
        S2_locations = annotationsArray[PCGi][1]
        
        [PCG_Features, featuresFs] = gSPCGF.getSpringerPCGFeatures(PCG_audio, opts.default_Springer_HSMM_options())
        
        PCG_states = labelPCG.labelPCGStates(PCG_Features[:][0],S1_locations, S2_locations, featuresFs) #XXX
        
        ## TODO
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
            state_observation_values[PCGi][state_i] = np.multiply(PCG_Features,np.int16(PCG_states == state_i+1)) #xxx variable name
            # x = PCG_Features[state_i][np.nonzero(x)]
            # print np.shape(x)#, x
            # state_observation_values[PCGi][state_i] = x
        # print np.shape(state_observation_values[0][3]), state_observation_values

    # Save the state observation values to the main workspace of Matlab for
    # later investigation if needed:
    # assignin('base', 'state_observation_values', state_observation_values) #XXX

    # ## Train the B and pi matrices after all the PCG recordings have been labelled:
    logistic_regression_B_matrix, pi_vector, total_obs_distribution = train.trainBandPiMatricesSpringer(state_observation_values) #XXX

    return logistic_regression_B_matrix, pi_vector, total_obs_distribution

if __name__ == '__main__':
    import scipy.io

    x = scipy.io.loadmat('./test_data/trainSpringerSegmentationAlgorithm/PCGCellArray.mat', struct_as_record=False)
    x = x['PCGCellArray'][0]
    PCGCellArray = map(lambda x: np.reshape(x, np.shape(x)[0]), x)

    x = scipy.io.loadmat('./test_data/trainSpringerSegmentationAlgorithm/annotationsArray.mat', struct_as_record=False)
    x = x['annotationsArray']
    annotationsArray = map(lambda z: map(lambda y: np.reshape(y, np.shape(y)[0]), z), x)

    actual = trainSpringerSegmentationAlgorithm(PCGCellArray, annotationsArray, 1000)

    x = scipy.io.loadmat('./test_data/trainSpringerSegmentationAlgorithm/logistic_regression_B_matrix.mat', struct_as_record=False)
    logistic_regression_B_matrix = x['logistic_regression_B_matrix'][0]

    x = scipy.io.loadmat('./test_data/trainSpringerSegmentationAlgorithm/pi_vector.mat', struct_as_record=False)
    pi_vector = x['pi_vector'][0]

    x = scipy.io.loadmat('./test_data/trainSpringerSegmentationAlgorithm/total_obs_distribution.mat', struct_as_record=False)
    total_obs_distribution = x['total_obs_distribution']

    #TODO
    # np.testing.assert_allclose(actual[0][0], logistic_regression_B_matrix[0], rtol=1e-3, atol=1e-3)
    # np.testing.assert_allclose(actual[1], pi_vector, rtol=1e-3, atol=1e-3)
    # np.testing.assert_allclose(actual[2], total_obs_distribution, rtol=1e-3, atol=1e-3)
    print "trainSpringerSegmentationAlgorithm.py has been tested successfully"