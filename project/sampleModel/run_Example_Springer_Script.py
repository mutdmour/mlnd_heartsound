## Example Springer script
# A script to demonstrate the use of the Springer segmentation algorithm

import default_Springer_HSMM_options
import trainSpringerSegmentationAlgorithm
import scipy.io
import numpy as np
from runSpringerSegmentationAlgorithm import runSpringerSegmentationAlgorithm

def runExampleScript():
	## Load the default options:
	# These options control options such as the original sampling frequency of
	# the data, the sampling frequency for the derived features and whether the
	# mex code should be used for the Viterbi decoding:
	springer_options = default_Springer_HSMM_options.default_Springer_HSMM_options()

	# print springer_options
	## Load the audio data and the annotations:
	# These are 6 example PCG recordings, downsampled to 1000 Hz, with
	# annotations of the R-peak and end-T-wave positions.
	example_data = scipy.io.loadmat('example_data.mat',struct_as_record=False) ## numpy.ndarray
	example_data = example_data['example_data'][0][0]

	## extract variables
	example_audio_data = example_data.example_audio_data[0]
	example_audio_data = map(lambda x: np.reshape(x, np.shape(x)[0]), example_audio_data)
	example_audio_data = np.array(example_audio_data)

	example_annotations = example_data.example_annotations
	example_annotations = map(lambda x: map(lambda y: np.reshape(y, np.shape(y)[0]), x), example_annotations)
	# annotations = np.empty(1, dtype=object)
	# annotations[0] = example_annotations
	# train_recordings = np.transpose(np.array([example_audio_data[0][i] for i in training_indices]))
	# there's also patient_number and binary_diagnosis

	## Split the data into train and test sets:
	# Select the 5 recordings for training and a sixth for testing:
	# training_indices = [0, 46, 360, 401, 571]
	training_indices = range(0,100)
	train_recordings = example_audio_data[training_indices]
	train_annotations = np.array([example_annotations[i] for i in training_indices])
	# train_annotations = annotations[training_indices]

	test_index = [663]
	test_recordings = example_audio_data[test_index]
	# test_annotations = np.array(example_annotations[test_index])

	## Train the HMM:
	#xxx B_matrix, pi_vector, total_obs_distribution = trainSpringerSegmentationAlgorithm.trainSpringerSegmentationAlgorithm(train_recordings,train_annotations,springer_options['audio_Fs'], False)
	[B_matrix, pi_vector, total_obs_distribution] = trainSpringerSegmentationAlgorithm.trainSpringerSegmentationAlgorithm(train_recordings,train_annotations,springer_options['audio_Fs'], False)

	## Run the HMM on an unseen test recording:
	# And display the resulting segmentation
	numPCGs = len(test_recordings)

	for PCGi in range(0,numPCGs):
	    assigned_states = runSpringerSegmentationAlgorithm(test_recordings[PCGi], springer_options['audio_Fs'], B_matrix, pi_vector, total_obs_distribution, False)
		#todo calculate F1 score
	return assigned_states

if __name__ == '__main__':
	assigned_states = runExampleScript()

	x = scipy.io.loadmat('./test_data/runSpringerSegmentationAlgorithm/assigned_states.mat', struct_as_record=False)
	desired = x['assigned_states']
	desired = np.transpose(desired)[0]

	mismatch = np.sum((desired - assigned_states) != 0) / float(np.shape(desired)[0]) * 100
	print mismatch
	assert (mismatch < 2)  # assert that mismatch is less than 2 %

	print "runSpringerSegmentationAlgorithm.py has been tested successfully"
