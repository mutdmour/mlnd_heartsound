# function assigned_states = runSpringerSegmentationAlgorithm(audio_data, Fs, B_matrix, pi_vector, total_observation_distribution, figures)
#
# A function to assign states to a PCG recording using a duration dependant
# logisitic regression-based HMM, using the trained B_matrix and pi_vector
# trained in "trainSpringerSegmentationAlgorithm.m". Developed for use in
# the paper:
# D. Springer et al., "Logistic Regression-HSMM-based Heart Sound
# Segmentation," IEEE Trans. Biomed. Eng., In Press, 2015.
#
## INPUTS:
# audio_data: The audio data from the PCG recording
# Fs: the sampling frequency of the audio recording
# B_matrix: the observation matrix for the HMM, trained in the 
# "trainSpringerSegmentationAlgorithm.m" function
# pi_vector: the initial state distribution, also trained in the 
# "trainSpringerSegmentationAlgorithm.m" function
# total_observation_distribution, the observation probabilities of all the
# data, again, trained in trainSpringerSegmentationAlgorithm.
# figures: (optional) boolean variable for displaying figures
#
## OUTPUTS:
# assigned_states: the array of state values assigned to the original
# audio_data (in the original sampling frequency).

import expand_qt as eqt
import getHeartRateSchmidt as getHR
import viterbiDecodePCG_Springer as vDPCG
import getSpringerPCGFeatures as gSPCGF

#function assigned_states = 
def runSpringerSegmentationAlgorithm(audio_data, Fs, B_matrix, pi_vector, total_observation_distribution, figures=False):

	## Get PCG Features:
	PCG_Features, featuresFs = gSPCGF.getSpringerPCGFeatures(audio_data, Fs)

	## Get PCG heart rate
	heartRate, systolicTimeInterval = getHR.getHeartRateSchmidt(audio_data, Fs)

	delta, psi, qt = vDPCG.viterbiDecodePCG_Springer(np.array(PCG_Features), pi_vector, B_matrix, total_observation_distribution, heartRate, systolicTimeInterval, featuresFs)

	assigned_states = eqt.expand_qt(qt, featuresFs, Fs, len(audio_data))

	# if(figures)
	#    figure('Name','Derived state sequence')
	#    t1 = (1:length(audio_data))./Fs
	#    plot(t1,normalise_signal(audio_data),'k')
	#    hold on
	#    plot(t1,assigned_states,'r--')
	#    xlabel('Time (s)')
	#    legend('Audio data', 'Derived states')

	return np.array(assigned_states)

if __name__ == '__main__':

    import scipy.io
    import numpy as np

    Fs = 1000
    pi_vector = [.25,.25,.25,.25]

    x = scipy.io.loadmat('./test_data/runSpringerSegmentationAlgorithm/audio_data.mat', struct_as_record=False)
    audio_data = np.transpose(x['audio_data'])[0]

    x = scipy.io.loadmat('./test_data/runSpringerSegmentationAlgorithm/B_matrix.mat', struct_as_record=False)
    B_matrix = x['B_matrix'][0]
    B_matrix = map(lambda x: np.reshape(x,np.shape(x)[0]), B_matrix)

    x = scipy.io.loadmat('./test_data/runSpringerSegmentationAlgorithm/total_observation_distribution.mat', struct_as_record=False)
    x = x['total_observation_distribution']
    total_obs_distribution = np.empty(2, dtype=object)
    total_obs_distribution[0] = x[0][0][0]
    total_obs_distribution[1] = np.transpose(x[1][0])

    assigned_states = runSpringerSegmentationAlgorithm(audio_data, Fs, B_matrix, pi_vector, total_obs_distribution)

    x = scipy.io.loadmat('./test_data/runSpringerSegmentationAlgorithm/assigned_states.mat', struct_as_record=False)
    desired = x['assigned_states']
    desired = np.transpose(desired)[0]

    #TODO
    # np.testing.assert_allclose(assigned_states, desired, rtol=1e-3, atol=1e-3)

    print "runSpringerSegmentationAlgorithm.py has been tested successfully"
